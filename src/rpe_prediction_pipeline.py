#!/usr/bin/env python3
"""
End-to-End RPE Prediction Pipeline

Complete pipeline from raw video → RPE prediction:

STEP-BY-STEP FLOW:
==================

1. FEATURE EXTRACTION (using OpenFace)
   - Input: Raw video file(s)
   - Process:
     a) MediaPipe detects body pose → crops video to face region (pose_guided_face_detection.py)
     b) OpenFace processes cropped video → extracts facial landmarks + AUs (openface_feature_extractor.py)
     c) Load only 10 key AU columns (AU04, AU06, AU07, AU09, AU10, AU12, AU17, AU20, AU25, AU26)
     d) Sample every 3rd frame (30 FPS → 10 FPS)
     e) Extract max values: AU_max, AU_q95, AU_peak_ratio
   - Output: DataFrame with ~64 features per video
   - Possible failures: "Insufficient valid frames (0 detected)" → creates 'error' column

2. IMPUTATION (using impute_missing_features.py module)
   - Input: Features DataFrame (may have 'error' column for failed videos)
   - Process:
     a) identify_missing_videos() → finds which videos have 'error' column
     b) load_rpe_labels() → loads RPE values for each video
     c) impute_by_rpe_average() → for each failed video:
        - Find videos with same exercise + same RPE
        - Average their features
        - Fill in the missing video with these averages
   - Output: Complete DataFrame (no 'error' column, has 'imputed' flag)
   
3. PREDICTION (using trained LGBM model)
   - Input: Complete features DataFrame
   - Process:
     a) Select only features the model expects
     b) Scale features using StandardScaler
     c) LGBM model predicts RPE (1-10)
   - Output: DataFrame with 'predicted_rpe' column

Usage:
    # Predict RPE for a single video
    python rpe_prediction_pipeline.py --video path/to/video.mp4 --model model.pkl
    
    # Predict RPE for all videos in a directory
    python rpe_prediction_pipeline.py --video-dir path/to/videos/ --model model.pkl
    
    # With RPE labels for better imputation
    python rpe_prediction_pipeline.py --video-dir ./videos --model model.pkl --rpe-labels labels.csv
"""

import os
import sys
import argparse
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import glob

# Add parent directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from OpenFace.openface_feature_extractor import OpenFaceExtractor
from OpenFace.impute_missing_features import impute_by_rpe_average, identify_missing_videos


class RPEPredictionPipeline:
    """
    End-to-end pipeline: Video → Features → RPE Prediction
    """
    
    def __init__(self, model_path=None, verbose=True):
        """
        Initialize RPE prediction pipeline.
        
        Args:
            model_path: Path to trained LGBM model (.pkl file)
            verbose: Print progress messages
        """
        self.verbose = verbose
        self.model = None
        self.scaler = None
        self.feature_columns = None
        
        # Load model if provided
        if model_path:
            self.load_model(model_path)
    
    def _log(self, message):
        """Print if verbose."""
        if self.verbose:
            print(message)
    
    # =========================================
    # FEATURE EXTRACTION
    # =========================================
    
    def extract_features_single_video(self, video_path, visualize=False):
        """
        Extract features from a single video using OpenFace.
        
        Args:
            video_path: Path to video file
            visualize: Show video with landmarks during processing
            
        Returns:
            dict: Feature dictionary ready for prediction
        """
        self._log(f"\n{'='*80}")
        self._log(f"EXTRACTING FEATURES: {os.path.basename(video_path)}")
        self._log(f"{'='*80}")
        
        # Initialize OpenFace extractor with all preprocessing
        extractor = OpenFaceExtractor(
            use_pose_guidance=True,    
            max_only=True,              
            load_minimal_columns=True,  
            sample_fps=10,              
            verbose=self.verbose,
            visualize=visualize
        )
        
        try:
            features = extractor.extract_from_video(video_path)
            
            # Format for prediction (exclude metadata)
            result = {'video_name': os.path.basename(video_path)}
            for key, value in features.items():
                if key != 'metadata':
                    result[key] = value
            
            self._log(f"Feature extraction successful")
            return result
            
        except Exception as e:
            self._log(f"ERROR: {e}")
            return {
                'video_name': os.path.basename(video_path),
                'error': str(e)
            }
    
    def extract_features_batch(self, video_dir, max_videos=None, visualize=False):
        """
        Extract features from all videos in a directory.
        
        Args:
            video_dir: Directory containing videos
            max_videos: Limit number of videos to process
            visualize: Show videos with landmarks during processing
            
        Returns:
            pd.DataFrame: Features for all videos
        """
        self._log(f"\n{'='*80}")
        self._log(f"BATCH FEATURE EXTRACTION")
        self._log(f"{'='*80}")
        
        # Find all video files
        patterns = ['*.mp4', '*.mov', '*.MOV', '*.MP4']
        video_files = []
        for pat in patterns:
            video_files.extend(glob.glob(os.path.join(video_dir, '**', pat), recursive=True))
        
        video_files = sorted(list(set(video_files)))
        
        if max_videos:
            video_files = video_files[:max_videos]
        
        if len(video_files) == 0:
            raise ValueError(f"No videos found in {video_dir}")
        
        self._log(f"\nFound {len(video_files)} videos")
        
        # Extract features from each video
        results = []
        for idx, video_path in enumerate(video_files, 1):
            self._log(f"\n[{idx}/{len(video_files)}] Processing: {os.path.basename(video_path)}")
            result = self.extract_features_single_video(video_path, visualize=visualize)
            results.append(result)
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Check for errors
        has_errors = 'error' in df.columns and df['error'].notna().any()
        successful = len(df) if not has_errors else len(df[df['error'].isna()])
        failed = 0 if not has_errors else len(df[df['error'].notna()])
        
        self._log(f"\n{'='*80}")
        self._log(f"EXTRACTION COMPLETE")
        self._log(f"{'='*80}")
        self._log(f"Total: {len(results)} | Successful: {successful} | Failed: {failed}")
        
        return df
    
    def impute_missing_features(self, features_df, rpe_labels_path=None):
        """
        Impute missing features for failed videos using RPE-based averaging.
        
        This is a WRAPPER around the existing impute_missing_features.py module.
        It simply calls the functions from that module.
        
        Args:
            features_df: DataFrame with extracted features (may have 'error' column)
            rpe_labels_path: Path to RPE labels CSV (for RPE-based imputation)
            
        Returns:
            pd.DataFrame: Features with imputed values (no 'error' column, has 'imputed' flag)
        """
        # Check if there are any failed videos
        has_errors = 'error' in features_df.columns and features_df['error'].notna().any()
        
        if not has_errors:
            self._log("No failed videos, no imputation needed")
            features_df['imputed'] = False
            return features_df
        
        self._log(f"\n{'='*80}")
        self._log(f"IMPUTING MISSING FEATURES")
        self._log(f"{'='*80}")
        
        # Use the existing imputation module
        # Step 1: Identify which videos failed (from impute_missing_features.py)
        failed_videos = identify_missing_videos(features_df)
        
        if rpe_labels_path and os.path.exists(rpe_labels_path):
            # Step 2: Load RPE labels (from impute_missing_features.py)
            from OpenFace.impute_missing_features import load_rpe_labels
            
            self._log(f"Loading RPE labels from: {rpe_labels_path}")
            rpe_df = load_rpe_labels(rpe_labels_path)
            
            # Step 3: Impute using RPE averaging (from impute_missing_features.py)
            features_imputed = impute_by_rpe_average(features_df, rpe_df, failed_videos)
            self._log(f"Imputed using RPE-based averaging")
            
        else:
            # Fallback: Simple mean imputation if no RPE labels
            self._log("No RPE labels provided, using simple mean imputation")
            features_imputed = features_df.copy()
            
            # Get successful videos for averaging
            successful_mask = features_df['error'].isna()
            successful_features = features_df[successful_mask]
            
            # Calculate means for numeric columns
            numeric_cols = successful_features.select_dtypes(include=[np.number]).columns
            means = successful_features[numeric_cols].mean()
            
            # Impute failed videos
            for idx in features_df[~successful_mask].index:
                for col in numeric_cols:
                    if col not in ['video_name']:
                        features_imputed.at[idx, col] = means[col]
                features_imputed.at[idx, 'imputed'] = True
            
            # Mark successful videos as not imputed
            features_imputed.loc[successful_mask, 'imputed'] = False
            
            # Remove error column
            if 'error' in features_imputed.columns:
                features_imputed = features_imputed.drop(columns=['error'])
            
            self._log(f"Imputed {len(failed_videos)} videos using mean values")
        
        return features_imputed
    
    # =========================================
    # MODEL OPERATIONS
    # =========================================
    
    def load_model(self, model_path):
        """
        Load trained LGBM model.
        
        Args:
            model_path: Path to .pkl file containing model, scaler, and feature columns
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self._log(f"Loading model from: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data.get('scaler', None)
        self.feature_columns = model_data.get('feature_columns', None)
        
        self._log(f"Model loaded successfully")
        if self.feature_columns:
            self._log(f"  Expected features: {len(self.feature_columns)}")
    
    def predict_rpe(self, features_df):
        """
        Predict RPE from extracted features.
        
        Args:
            features_df: DataFrame with extracted features
            
        Returns:
            pd.DataFrame: Original data with 'predicted_rpe' column added
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Use load_model() first.")
        
        self._log(f"\n{'='*80}")
        self._log(f"PREDICTING RPE")
        self._log(f"{'='*80}")
        
        # Prepare features for prediction
        df = features_df.copy()
        
        # Select only the features the model expects
        if self.feature_columns:
            # Check which features are available
            missing_features = [f for f in self.feature_columns if f not in df.columns]
            if missing_features:
                self._log(f"WARNING: {len(missing_features)} features missing from extracted data")
                self._log(f"  Missing: {missing_features[:5]}...")
                # Fill missing features with 0
                for feat in missing_features:
                    df[feat] = 0
            
            X = df[self.feature_columns]
        else:
            # Use all numeric columns except video_name and imputed
            exclude_cols = ['video_name', 'imputed', 'error', 'predicted_rpe']
            feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in [np.float64, np.int64]]
            X = df[feature_cols]
        
        # Scale features if scaler is available
        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        # Clip predictions to valid RPE range (1-10)
        predictions = np.clip(predictions, 1, 10)
        
        df['predicted_rpe'] = predictions
        
        self._log(f"Predicted RPE for {len(df)} videos")
        self._log(f"  RPE range: {predictions.min():.1f} - {predictions.max():.1f}")
        self._log(f"  Mean RPE: {predictions.mean():.1f}")
        
        return df
    
    # =========================================
    # FULL PIPELINE
    # =========================================
    
    def predict_from_video(self, video_path, visualize=False):
        """
        Complete pipeline: single video → RPE prediction.
        
        Args:
            video_path: Path to video file
            visualize: Show video with landmarks during processing
            
        Returns:
            float: Predicted RPE value
        """
        # Extract features
        features = self.extract_features_single_video(video_path, visualize=visualize)
        features_df = pd.DataFrame([features])
        
        # Check for errors
        if 'error' in features and pd.notna(features['error']):
            self._log(f"WARNING: Feature extraction failed, cannot predict RPE")
            return None
    
        # Predict
        results_df = self.predict_rpe(features_df)
        predicted_rpe = results_df['predicted_rpe'].iloc[0]
        
        return predicted_rpe
    
    def predict_from_directory(self, video_dir, max_videos=None, rpe_labels_path=None, 
                              output_csv=None, visualize=False):
        """
        Complete pipeline: video directory → RPE predictions for all videos.
        
        Args:
            video_dir: Directory containing videos
            max_videos: Limit number of videos to process
            rpe_labels_path: Path to RPE labels CSV (for imputation)
            output_csv: Path to save results CSV
            visualize: Show videos with landmarks during processing
            
        Returns:
            pd.DataFrame: Results with predicted RPE for each video
        """
        # Extract features
        features_df = self.extract_features_batch(video_dir, max_videos=max_videos, visualize=visualize)
        
        # Impute missing features if needed
        features_df = self.impute_missing_features(features_df, rpe_labels_path=rpe_labels_path)
        
        # Predict RPE
        results_df = self.predict_rpe(features_df)
        
        # Save results if requested
        if output_csv:
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            results_df.to_csv(output_csv, index=False)
            self._log(f"\nResults saved to: {output_csv}")
        
        return results_df


def main():
    parser = argparse.ArgumentParser(
        description='RPE Prediction Pipeline - Video to RPE',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict RPE for a single video
  python rpe_prediction_pipeline.py --video video.mp4 --model model.pkl
  
  # Predict RPE for all videos in a directory
  python rpe_prediction_pipeline.py --video-dir ./videos/ --model model.pkl
  
  # With RPE labels for imputation
  python rpe_prediction_pipeline.py --video-dir ./videos/ --model model.pkl --rpe-labels labels.csv
  
  # Visualize processing
  python rpe_prediction_pipeline.py --video video.mp4 --model model.pkl --visualize
        """
    )
    
    # Input
    parser.add_argument('--video', help='Single video file to process')
    parser.add_argument('--video-dir', help='Directory containing videos')
    parser.add_argument('--max-videos', type=int, help='Maximum number of videos to process')
    
    # Model
    parser.add_argument('--model', required=True, help='Path to trained LGBM model (.pkl)')
    
    # Optional
    parser.add_argument('--rpe-labels', help='Path to RPE labels CSV (for imputation)')
    parser.add_argument('--output', help='Path to save results CSV')
    parser.add_argument('--visualize', '-v', action='store_true', 
                       help='Show video with landmarks during processing')
    
    args = parser.parse_args()
    
    # Validate input
    if not args.video and not args.video_dir:
        parser.error("Must provide either --video or --video-dir")
    
    # Initialize pipeline
    pipeline = RPEPredictionPipeline(model_path=args.model, verbose=True)
    
    print(f"\n{'='*80}")
    print(f"RPE PREDICTION PIPELINE")
    print(f"{'='*80}")
    print(f"\nSteps:")
    print(f"  1. MediaPipe pose-guided face cropping")
    print(f"  2. OpenFace feature extraction (max values from key AUs)")
    print(f"  3. Feature preprocessing (select columns, impute missing)")
    print(f"  4. LGBM model prediction")
    print(f"{'='*80}\n")
    
    # Run pipeline
    if args.video:
        # Single video
        if not os.path.exists(args.video):
            print(f"ERROR: Video not found: {args.video}")
            return
        
        predicted_rpe = pipeline.predict_from_video(args.video, visualize=args.visualize)
        
        if predicted_rpe is not None:
            print(f"\n{'='*80}")
            print(f"PREDICTION RESULT")
            print(f"{'='*80}")
            print(f"Video: {os.path.basename(args.video)}")
            print(f"Predicted RPE: {predicted_rpe:.1f}")
            print(f"{'='*80}")
        
    else:
        # Directory
        if not os.path.exists(args.video_dir):
            print(f"ERROR: Directory not found: {args.video_dir}")
            return
        
        results_df = pipeline.predict_from_directory(
            video_dir=args.video_dir,
            max_videos=args.max_videos,
            rpe_labels_path=args.rpe_labels,
            output_csv=args.output,
            visualize=args.visualize
        )
        
        # Display summary
        print(f"\n{'='*80}")
        print(f"PREDICTION SUMMARY")
        print(f"{'='*80}")
        print(f"Total videos: {len(results_df)}")
        print(f"RPE range: {results_df['predicted_rpe'].min():.1f} - {results_df['predicted_rpe'].max():.1f}")
        print(f"Mean RPE: {results_df['predicted_rpe'].mean():.1f} ± {results_df['predicted_rpe'].std():.1f}")
        
        # Show sample predictions
        print(f"\nSample predictions:")
        sample = results_df[['video_name', 'predicted_rpe']].head(10)
        for _, row in sample.iterrows():
            print(f"  {row['video_name']}: {row['predicted_rpe']:.1f}")
        
        if args.output:
            print(f"\nFull results saved to: {args.output}")
        
        print(f"{'='*80}")


if __name__ == '__main__':
    main()
