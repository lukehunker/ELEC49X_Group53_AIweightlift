"""
Unified RPE Prediction Pipeline

This module provides a complete pipeline for predicting RPE from workout videos:
1. Extract features from video (bar speed, facial, posture)
2. Combine features in the format expected by LGBM model
3. Load trained model and make prediction

Usage:
    from rpe_pipeline import predict_rpe
    
    result = predict_rpe('squat_video.mp4', movement='Squat')
    print(f"Predicted RPE: {result['predicted_rpe']}")
"""

import os
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import joblib
import json
from pathlib import Path
from typing import Dict, Optional, Any

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import feature extractors
from Bar_Tracking.extract_features import extract_bar_speed
from OpenFace.extract_features import extract_facial_features, flatten_features
from MMPose.extract_features import extract_posture_features


class RPEPredictor:
    """
    RPE Predictor that combines all feature extraction pipelines
    and uses a trained ensemble (LGBM + XGBoost) to predict RPE.
    """
    
    def __init__(self, model_path=None, verbose=True):
        """
        Initialize the RPE predictor.
        
        Args:
            model_path: Path to model directory containing ensemble models
                       If None, uses default location in Train_Outputs/
            verbose: Print progress messages
        """
        self.verbose = verbose
        self.lgbm_model = None
        self.xgb_model = None
        self.scaler = None
        self.metadata = None
        self.feature_names = None
        self.ensemble_weights = {'lgbm': 0.6, 'xgb': 0.4}  # Default
        
        # Determine model directory
        if model_path is None:
            script_dir = Path(__file__).parent
            model_dir = script_dir / "Train_Outputs"
        else:
            model_dir = Path(model_path)
        
        self.model_dir = model_dir
        
        if self.model_dir.exists():
            self._load_models()
        elif self.verbose:
            print(f"Model directory not found: {self.model_dir}")
            print("   Run training first (LGBMMod2_Ensemble.py) or provide model path")
    
    def _load_models(self):
        """Load the trained ensemble models (LGBM + XGBoost) and metadata."""
        try:
            # Load metadata first to get feature names
            metadata_path = self.model_dir / "ensemble_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                
                self.feature_names = self.metadata.get('feature_columns', [])
                self.ensemble_weights = self.metadata.get('ensemble_weights', 
                                                          {'lgbm': 0.6, 'xgb': 0.4})
                
                if self.verbose:
                    cv_perf = self.metadata.get('cv_performance', {})
                    print(f"Ensemble Configuration:")
                    print(f"   LGBM weight: {self.ensemble_weights['lgbm']:.1%}")
                    print(f"   XGBoost weight: {self.ensemble_weights['xgb']:.1%}")
                    if cv_perf:
                        print(f"   CV MAE: {cv_perf.get('ensemble_mae', 0):.4f}")
                        print(f"   CV R²: {cv_perf.get('ensemble_r2', 0):.4f}")
            
            # Load LightGBM model
            lgbm_path = self.model_dir / "lgbm_ensemble_model.txt"
            if lgbm_path.exists():
                self.lgbm_model = lgb.Booster(model_file=str(lgbm_path))
                if self.verbose:
                    print(f"Loaded LightGBM: {lgbm_path.name}")
            else:
                print(f"LightGBM model not found: {lgbm_path}")
            
            # Load XGBoost model
            xgb_path = self.model_dir / "xgb_ensemble_model.json"
            if xgb_path.exists():
                self.xgb_model = xgb.XGBRegressor()
                self.xgb_model.load_model(str(xgb_path))
                if self.verbose:
                    print(f"Loaded XGBoost: {xgb_path.name}")
            else:
                print(f"XGBoost model not found: {xgb_path}")
            
            # Load scaler
            scaler_path = self.model_dir / "ensemble_scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                if self.verbose:
                    print(f"Loaded scaler: {scaler_path.name}")
            else:
                print(f"Scaler not found: {scaler_path}")
            
            if self.verbose and self.feature_names:
                print(f"   Expected features: {len(self.feature_names)}")
        
        except Exception as e:
            if self.verbose:
                print(f"Error loading models: {e}")
            self.lgbm_model = None
            self.xgb_model = None
            self.scaler = None
    
    def extract_all_features(self, video_path: str, movement: str, 
                            output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract features from all three pipelines.
        
        Args:
            video_path: Path to video file
            movement: Movement type ('Squat', 'Bench Press', 'Deadlift')
            output_dir: Optional directory for intermediate outputs
        
        Returns:
            dict: Combined features from all extractors with keys:
                - bar_speed: Dict of bar speed features
                - facial: Dict of facial features
                - posture: Dict of posture features
                - combined: Dict of combined features ready for model
                - success: Boolean indicating if all extractions succeeded
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Extracting features from: {os.path.basename(video_path)}")
            print(f"Movement: {movement}")
            print(f"{'='*70}")
        
        results = {
            'video_name': os.path.basename(video_path),
            'movement': movement,
            'bar_speed': None,
            'facial': None,
            'posture': None,
            'combined': {},
            'success': False,
            'warnings': []
        }
        
        # 1. Bar Speed Features
        if self.verbose:
            print("\n[1/3] Extracting bar speed features...")
        try:
            bar_features = extract_bar_speed(video_path, movement=movement, output_dir=output_dir)
            results['bar_speed'] = bar_features
            
            if bar_features:
                if self.verbose:
                    print(f"  ✓ {bar_features['rep_count']} reps, fatigue: {bar_features['fatigue_s']:.2f}s")
            else:
                results['warnings'].append("Bar speed extraction returned None")
        
        except Exception as e:
            results['warnings'].append(f"Bar speed failed: {e}")
            if self.verbose:
                print(f"  ✗ Failed: {e}")
        
        # 2. Facial Features
        if self.verbose:
            print("\n[2/3] Extracting facial expression features...")
        try:
            facial_features = extract_facial_features(
                video_path,
                verbose=False,
                use_pose_guidance=True,
                visualize=False
            )
            results['facial'] = flatten_features(facial_features)
            
            if facial_features:
                detection_rate = facial_features.get('detection_rate', 0)
                if self.verbose:
                    print(f"  ✓ Detection rate: {detection_rate:.1%}")
                
                if detection_rate < 0.5:
                    results['warnings'].append(
                        f"Low face detection ({detection_rate:.1%}). Ensure face is visible."
                    )
            else:
                results['warnings'].append("Facial extraction returned None")
        
        except Exception as e:
            results['warnings'].append(f"Facial extraction failed: {e}")
            if self.verbose:
                print(f"  ✗ Failed: {e}")
        
        # 3. Posture Features
        if self.verbose:
            print("\n[3/3] Extracting body posture features...")
        try:
            posture_features = extract_posture_features(
                video_path,
                movement=movement,
                output_dir=output_dir
            )
            results['posture'] = posture_features
            
            if posture_features:
                if self.verbose:
                    print(f"  ✓ D-metric: {posture_features['d_value']:.3f}")
            else:
                # Provide context about why posture failed
                rep_count = results['bar_speed'].get('rep_count', 0) if results['bar_speed'] else 0
                if rep_count < 2:
                    results['warnings'].append(f"Posture extraction skipped (only {rep_count} rep detected, need 2+ for D-metric)")
                else:
                    results['warnings'].append("Posture extraction returned None")
        
        except Exception as e:
            results['warnings'].append(f"Posture extraction failed: {e}")
            if self.verbose:
                print(f"  ✗ Failed: {e}")
        
        # Combine features for model input
        results['combined'] = self._combine_features(
            results['bar_speed'],
            results['facial'],
            results['posture'],
            movement
        )
        
        # Check success - require bar_speed and facial (posture is optional for low-rep sets)
        results['success'] = all([
            results['bar_speed'] is not None,
            results['facial'] is not None
        ])
        
        if self.verbose:
            print(f"\n{'='*70}")
            if results['success']:
                print("Core features extracted successfully!")
            else:
                print("Some core features failed to extract")
            print(f"{'='*70}")
        
        return results
    
    def _combine_features(self, bar_speed: Optional[Dict], facial: Optional[Dict],
                         posture: Optional[Dict], movement: str) -> Dict[str, float]:
        """
        Combine features from all extractors into the format expected by ensemble model.
        
        Includes feature engineering to match training pipeline:
        - AU aggregations (sum, mean, max, std)
        - AU interactions (AU04*AU07, AU06*AU12, etc.)
        - Quality scores (detection_rate * confidence)
        - Body-face strain (d_value * AU features)
        - Tempo interactions
        """
        combined = {}
        
        # Bar speed features (use rep_change_s as the speed metric)
        if bar_speed:
            fatigue_s = bar_speed.get('fatigue_s', 0)
            combined['rep_change_s'] = fatigue_s  # Map to expected name
            combined['rep_count'] = bar_speed.get('rep_count', 0)
        else:
            combined['rep_change_s'] = 0
            combined['rep_count'] = 0
        
        # Collect all AU max values for aggregations
        au_values = []
        
        # Facial features - extract AU max values
        if facial:
            for key, value in facial.items():
                if 'AU' in key and '_max' in key:
                    # Already in correct format (AU04_max, etc.)
                    combined[key] = value
                    au_values.append(value)
                elif 'detection_rate' in key:
                    combined['detection_rate'] = value
                elif 'confidence' in key and 'mean' in key:
                    combined['confidence_mean'] = value
                elif 'confidence' in key and 'std' in key:
                    combined['confidence_std'] = value
        
        # If no facial features, set defaults
        if not au_values:
            combined['detection_rate'] = 0.5
            combined['confidence_mean'] = 0.5
            combined['confidence_std'] = 0.1
        
        # AU Aggregations (feature engineering)
        if au_values:
            combined['AU_sum'] = sum(au_values)
            combined['AU_mean'] = np.mean(au_values)
            combined['AU_max'] = max(au_values)
            combined['AU_std'] = np.std(au_values) if len(au_values) > 1 else 0
        else:
            combined['AU_sum'] = 0
            combined['AU_mean'] = 0
            combined['AU_max'] = 0
            combined['AU_std'] = 0
        
        # AU Interactions (key feature engineering from training)
        if 'AU04_max' in combined and 'AU07_max' in combined:
            combined['AU04_07_interaction'] = combined['AU04_max'] * combined['AU07_max']
        if 'AU06_max' in combined and 'AU12_max' in combined:
            combined['AU06_12_interaction'] = combined['AU06_max'] * combined['AU12_max']
        if 'AU09_max' in combined and 'AU10_max' in combined:
            combined['AU09_10_interaction'] = combined['AU09_max'] * combined['AU10_max']
        if 'AU25_max' in combined and 'AU26_max' in combined:
            combined['AU25_26_interaction'] = combined['AU25_max'] * combined['AU26_max']
        if 'AU04_max' in combined and 'AU06_max' in combined:
            combined['AU04_06_interaction'] = combined['AU04_max'] * combined['AU06_max']
        
        # Quality score
        if 'detection_rate' in combined and 'confidence_mean' in combined:
            combined['quality_score'] = combined['detection_rate'] * combined['confidence_mean']
        
        # Confidence coefficient of variation
        if 'confidence_mean' in combined and 'confidence_std' in combined:
            combined['confidence_cv'] = combined['confidence_std'] / (combined['confidence_mean'] + 1e-6)
        
        # Posture features
        if posture:
            d_value = posture.get('d_value', 0)
            combined['d_value'] = d_value
            
            # Body-face strain interactions
            if 'AU_sum' in combined:
                combined['body_face_strain'] = d_value * combined['AU_sum']
            if 'AU_mean' in combined:
                combined['d_value_x_AU_mean'] = d_value * combined['AU_mean']
            
            # Also keep rep_count from posture as backup
            if 'rep_count' not in combined or combined['rep_count'] == 0:
                combined['rep_count'] = posture.get('rep_count', 0)
        else:
            combined['d_value'] = 0
        
        # Tempo-related features
        if 'rep_change_s' in combined and 'rep_count' in combined and combined['rep_count'] > 0:
            combined['avg_rep_duration'] = combined['rep_change_s'] / (combined['rep_count'] + 1)
            
            # Tempo interactions
            if 'AU_sum' in combined:
                combined['tempo_x_facial_strain'] = combined['avg_rep_duration'] * combined['AU_sum']
            if 'd_value' in combined:
                combined['tempo_x_body_movement'] = combined['avg_rep_duration'] * combined['d_value']
        
        # Rep count transformations
        if 'rep_count' in combined and combined['rep_count'] > 0:
            combined['rep_count_sq'] = combined['rep_count'] ** 2
            
            if 'AU_sum' in combined:
                combined['AU_sum_per_rep'] = combined['AU_sum'] / (combined['rep_count'] + 1)
            if 'd_value' in combined:
                combined['d_value_per_rep'] = combined['d_value'] / (combined['rep_count'] + 1)
        
        # Lift type encoding (0=Bench, 1=Squat, 2=Deadlift)
        movement_encoding = {
            'Bench Press': 0,
            'Squat': 1,
            'Deadlift': 2
        }
        lift_code = movement_encoding.get(movement, 1)
        combined['Lift_Type_Code'] = lift_code
        
        # Lift type interactions
        if 'rep_count' in combined:
            combined['lift_type_x_reps'] = lift_code * combined['rep_count']
        
        return combined
    
    def predict(self, video_path: str, movement: Optional[str] = None,
                output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Complete pipeline: extract features and predict RPE.
        
        Args:
            video_path: Path to video file
            movement: Movement type ('Squat', 'Bench Press', 'Deadlift')
                     If None, attempts to auto-detect from filename
            output_dir: Optional directory for intermediate outputs
        
        Returns:
            dict: Prediction results with keys:
                - predicted_rpe: Predicted RPE value (1-10 scale)
                - confidence: Confidence score (0-1)
                - video_name: Name of the video
                - movement: Movement type
                - features: Dictionary of extracted features
                - warnings: List of any warnings
                - success: Boolean indicating success
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        if self.lgbm_model is None or self.xgb_model is None:
            raise RuntimeError(
                "Ensemble models not loaded. Train models first using LGBMMod2_Ensemble.py"
            )
        
        if self.scaler is None:
            raise RuntimeError("Scaler not loaded. Ensure ensemble_scaler.pkl exists.")
        
        # Auto-detect movement if not provided
        if movement is None:
            video_lower = os.path.basename(video_path).lower()
            if "bench" in video_lower:
                movement = "Bench Press"
            elif "squat" in video_lower:
                movement = "Squat"
            elif "deadlift" in video_lower:
                movement = "Deadlift"
            else:
                raise ValueError(
                    "Could not detect movement type. "
                    "Please specify movement='Squat', 'Bench Press', or 'Deadlift'"
                )
        
        # Extract all features
        extraction_results = self.extract_all_features(video_path, movement, output_dir)
        combined_features = extraction_results['combined']
        
        # Prepare feature vector for model
        if self.feature_names:
            # Use model's expected feature order
            feature_dict = {}
            missing_features = []
            
            for feature_name in self.feature_names:
                if feature_name in combined_features:
                    value = combined_features[feature_name]
                    # Handle Lift_Type encoding
                    if feature_name == 'Lift_Type_Code' or feature_name == 'Lift_Type':
                        if isinstance(value, str):
                            # Encode movement type (0=Bench, 1=Squat, 2=Deadlift)
                            movement_encoding = {
                                'Bench Press': 0,
                                'Squat': 1,
                                'Deadlift': 2
                            }
                            value = movement_encoding.get(value, 1)
                        feature_dict[feature_name] = float(value)
                    else:
                        feature_dict[feature_name] = float(value)
                else:
                    feature_dict[feature_name] = 0.0  # Default for missing features
                    missing_features.append(feature_name)
            
            if missing_features and self.verbose:
                print(f"\nMissing {len(missing_features)} features, using defaults")
                if len(missing_features) <= 10:
                    print(f"   Missing: {', '.join(missing_features[:10])}")
        else:
            # Fallback: create dict from combined features
            feature_dict = {k: float(v) for k, v in combined_features.items() 
                           if isinstance(v, (int, float))}
        
        # Convert to DataFrame for scaler (maintains column order)
        feature_df = pd.DataFrame([feature_dict])
        
        # Ensure feature order matches training
        if self.feature_names:
            feature_df = feature_df[self.feature_names]
        
        # Scale features
        feature_scaled = self.scaler.transform(feature_df)
        
        # Make predictions with both models
        if self.verbose:
            print(f"\nMaking ensemble RPE prediction...")
        
        # LGBM prediction
        lgbm_pred = self.lgbm_model.predict(feature_scaled)[0]
        
        # XGBoost prediction
        xgb_pred = self.xgb_model.predict(feature_scaled)[0]
        
        # Ensemble: weighted average
        lgbm_weight = self.ensemble_weights['lgbm']
        xgb_weight = self.ensemble_weights['xgb']
        ensemble_pred = lgbm_weight * lgbm_pred + xgb_weight * xgb_pred
        
        # Clip to valid RPE range
        ensemble_pred = np.clip(ensemble_pred, 1, 10)
        
        if self.verbose:
            print(f"   LGBM prediction: {lgbm_pred:.2f}")
            print(f"   XGBoost prediction: {xgb_pred:.2f}")
            print(f"   Ensemble ({lgbm_weight:.0%}/{xgb_weight:.0%}): {ensemble_pred:.2f}")
        
        # Calculate confidence based on detection quality and prediction agreement
        detection_rate = combined_features.get('detection_rate', 0.8)
        has_all_features = extraction_results['success']
        
        # Lower confidence if models disagree significantly
        prediction_diff = abs(lgbm_pred - xgb_pred)
        agreement_factor = max(0.5, 1.0 - (prediction_diff / 10.0))
        
        confidence = detection_rate * agreement_factor * (1.0 if has_all_features else 0.7)
        
        rpe_rounded = round(float(ensemble_pred) * 2) / 2
        
        result = {
            'predicted_rpe': rpe_rounded,
            'confidence': round(float(confidence), 2),
            'video_name': os.path.basename(video_path),
            'movement': movement,
            'model_details': {
                'lgbm_prediction': round(float(lgbm_pred), 2),
                'xgb_prediction': round(float(xgb_pred), 2),
                'lgbm_weight': lgbm_weight,
                'xgb_weight': xgb_weight
            },
            'features': extraction_results,
            'warnings': extraction_results['warnings'],
            'success': True
        }
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"PREDICTED RPE: {result['predicted_rpe']}")
            print(f"   Confidence: {result['confidence']:.1%}")
            print(f"{'='*70}\n")
        
        return result


def predict_rpe(video_path: str, movement: Optional[str] = None,
                model_path: Optional[str] = None, verbose: bool = True) -> Dict[str, Any]:
    """
    Convenience function to predict RPE from a video.
    
    Args:
        video_path: Path to video file
        movement: Movement type ('Squat', 'Bench Press', 'Deadlift') or None for auto-detect
        model_path: Path to trained model (uses default if None)
        verbose: Print progress messages
    
    Returns:
        dict: Prediction results including predicted_rpe, confidence, features, etc.
    
    Example:
        >>> result = predict_rpe('squat_video.mp4')
        >>> print(f"RPE: {result['predicted_rpe']}")
    """
    predictor = RPEPredictor(model_path=model_path, verbose=verbose)
    return predictor.predict(video_path, movement=movement)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict RPE from workout video")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("--movement", choices=['Squat', 'Bench Press', 'Deadlift'],
                       help="Movement type (optional, auto-detected from filename)")
    parser.add_argument("--model", help="Path to trained model file (optional)")
    parser.add_argument("--output-dir", help="Directory for intermediate outputs (optional)")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress messages")
    
    args = parser.parse_args()
    
    try:
        result = predict_rpe(
            args.video_path,
            movement=args.movement,
            model_path=args.model,
            verbose=not args.quiet
        )
        
        print("\n" + "="*70)
        print("RPE PREDICTION RESULT")
        print("="*70)
        print(f"Video: {result['video_name']}")
        print(f"Movement: {result['movement']}")
        print(f"Predicted RPE: {result['predicted_rpe']}")
        print(f"Confidence: {result['confidence']:.1%}")
        
        if result['warnings']:
            print(f"\nWarnings:")
            for warning in result['warnings']:
                print(f"  {warning}")
        
        print("="*70)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
