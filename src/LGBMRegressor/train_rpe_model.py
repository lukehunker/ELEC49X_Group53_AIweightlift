"""
LGBM RPE Prediction Model - Training Script

Trains LGBMRegressor on features extracted from:
- OpenFace (facial strain)
- MMPose (body pose & ROM)
- Bar tracking (speed & movement)

Usage:
    1. Extract features from training videos:
       python train_rpe_model.py --extract --videos ./training_videos/ --labels ./rpe_labels.csv
    
    2. Train model on extracted features:
       python train_rpe_model.py --train --features ./output/training_features.csv
    
    3. Or do both in one step:
       python train_rpe_model.py --extract --train --videos ./training_videos/ --labels ./rpe_labels.csv
"""

import os
import sys
import argparse
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt

# Add parent directory to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from main_pipeline import RPEPipeline


class RPEModelTrainer:
    """
    Train LGBM model to predict RPE from video features.
    """
    
    def __init__(self, output_dir='output/lgbm_models'):
        """
        Initialize trainer.
        
        Args:
            output_dir: Directory to save trained models and results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.model = None
        self.scaler = None
        self.feature_columns = None
    
    # =========================================
    # FEATURE EXTRACTION
    # =========================================
    
    def extract_features_from_videos(self, video_dir, rpe_labels_csv, output_csv=None):
        """
        Extract features from training videos using main pipeline.
        
        Args:
            video_dir: Directory containing training videos
            rpe_labels_csv: CSV with columns: video_name, rpe, (optional: expected_reps)
            output_csv: Where to save extracted features
            
        Returns:
            pd.DataFrame: Features merged with RPE labels
        """
        print(f"\n{'='*80}")
        print(f"EXTRACTING FEATURES FROM TRAINING VIDEOS")
        print(f"{'='*80}\n")
        
        # Load RPE labels
        labels_df = pd.read_csv(rpe_labels_csv)
        print(f"Loaded {len(labels_df)} RPE labels from {rpe_labels_csv}")
        print(f"Columns: {list(labels_df.columns)}")
        
        # Check if expected_reps column exists
        expected_reps = None
        if 'expected_reps' in labels_df.columns:
            # Create dict mapping video_name -> expected_reps
            expected_reps = dict(zip(labels_df['video_name'], labels_df['expected_reps']))
            print(f"Using expected_reps from labels file")
        
        # Initialize pipeline
        pipeline = RPEPipeline()
        
        # Extract features
        features_df = pipeline.extract_features_batch(
            video_dir=video_dir,
            output_csv=output_csv,
            expected_reps=expected_reps
        )
        
        # Merge with RPE labels
        print(f"\nMerging features with RPE labels...")
        df = features_df.merge(labels_df[['video_name', 'rpe']], on='video_name', how='inner')
        
        if len(df) == 0:
            raise ValueError("No matching videos found between features and labels!")
        
        print(f"✓ Successfully merged {len(df)} videos with RPE labels")
        
        # Save merged dataset
        if output_csv:
            merged_csv = output_csv.replace('.csv', '_with_rpe.csv')
            df.to_csv(merged_csv, index=False)
            print(f"✓ Saved merged dataset to: {merged_csv}")
        
        return df
    
    # =========================================
    # FEATURE ENGINEERING
    # =========================================
    
    def compute_composite_features(self, df):
        """
        Compute composite features from raw extracted features.
        Uses MAX AU intensities for peak RPE prediction.
        
        This maps the 200+ raw features into meaningful composite scores.
        """
        print(f"\nComputing composite features...")
        
        # 1. Facial Strain Score (weighted average of MAX exertion AUs)
        # Using MAX values to capture peak exertion moment
        au_weights = {
            'face_AU04_max': 0.25,  # Brow lowerer (concentration)
            'face_AU06_max': 0.20,  # Cheek raiser (strain)
            'face_AU07_max': 0.15,  # Lid tightener (strain)
            'face_AU09_max': 0.10,  # Nose wrinkler (strain)
            'face_AU25_max': 0.15,  # Lips part (breathing)
            'face_AU26_max': 0.15,  # Jaw drop (breathing)
        }
        
        facial_strain = 0
        for au, weight in au_weights.items():
            if au in df.columns:
                facial_strain += df[au] * weight
        df['facial_strain_composite'] = facial_strain
        
        # 2. ROM Change (average knee ROM - key for squats/deadlifts)
        if 'pose_rom_left_knee' in df.columns and 'pose_rom_right_knee' in df.columns:
            df['rom_composite'] = (df['pose_rom_left_knee'] + df['pose_rom_right_knee']) / 2
        elif 'pose_movement_hip_range' in df.columns:
            df['rom_composite'] = df['pose_movement_hip_range']
        else:
            df['rom_composite'] = 0
        
        # 3. Bar Speed Change (already computed, just rename if needed)
        if 'bar_speed_change' in df.columns:
            df['bar_speed_composite'] = df['bar_speed_change']
        else:
            df['bar_speed_composite'] = 0
        
        print(f"  ✓ Computed 3 composite features")
        
        return df
    
    def select_features(self, df, remove_low_variance=True, remove_correlated=True, 
                        variance_threshold=0.01, correlation_threshold=0.95):
        """
        Select which features to use for training with intelligent filtering.
        
        Options:
        1. Use only composite features (3 features)
        2. Use all extracted features (200+ features)
        3. Use selected important features (10-50 features) - with optimization
        
        Args:
            remove_low_variance: Remove features with very low variance (constant/near-constant)
            remove_correlated: Remove highly correlated features (redundant)
            variance_threshold: Min variance to keep a feature
            correlation_threshold: Max correlation before removing one feature
        """
        # Option 1: Just composite features (simple, interpretable)
        composite_features = [
            'facial_strain_composite',
            'rom_composite',
            'bar_speed_composite'
        ]
        
        # Option 2: Add detection quality features
        quality_features = [
            'face_detection_rate',
            'face_confidence_mean',
        ]
        
        # Option 3: Add key AU MAX features (peak exertion indicators)
        key_au_features = [col for col in df.columns if col.startswith('face_AU') and '_max' in col]
        
        # Option 4: Add AU peak ratios (how much AU spiked)
        key_au_ratios = [col for col in df.columns if col.startswith('face_AU') and '_peak_ratio' in col]
        
        # Option 5: Add pose features
        key_pose_features = [col for col in df.columns if col.startswith('pose_rom_') or col.startswith('pose_angle_')]
        
        # Option 6: Add bar features
        key_bar_features = [col for col in df.columns if col.startswith('bar_')]
        
        # Start with composite + quality + top AUs (max values)
        selected = composite_features + quality_features + key_au_features[:10] + key_au_ratios[:5]
        
        # Filter to only columns that exist
        selected = [col for col in selected if col in df.columns]
        
        print(f"\nInitial feature selection: {len(selected)} features")
        
        # OPTIMIZATION 1: Remove low-variance features (near-constant values)
        if remove_low_variance and len(selected) > 0:
            removed_low_var = []
            filtered = []
            for col in selected:
                if col in df.columns:
                    variance = df[col].var()
                    if variance > variance_threshold:
                        filtered.append(col)
                    else:
                        removed_low_var.append(col)
            
            if removed_low_var:
                print(f"  ✓ Removed {len(removed_low_var)} low-variance features (var < {variance_threshold})")
                if len(removed_low_var) <= 5:
                    print(f"    {removed_low_var}")
            selected = filtered
        
        # OPTIMIZATION 2: Remove highly correlated features (redundant)
        if remove_correlated and len(selected) > 1:
            from scipy.stats import pearsonr
            
            removed_corr = []
            final_features = []
            
            for i, col in enumerate(selected):
                if col not in df.columns:
                    continue
                    
                # Check correlation with already selected features
                is_redundant = False
                for other_col in final_features:
                    try:
                        corr, _ = pearsonr(df[col].fillna(0), df[other_col].fillna(0))
                        if abs(corr) > correlation_threshold:
                            is_redundant = True
                            removed_corr.append((col, other_col, corr))
                            break
                    except:
                        pass
                
                if not is_redundant:
                    final_features.append(col)
            
            if removed_corr:
                print(f"  ✓ Removed {len(removed_corr)} highly correlated features (|r| > {correlation_threshold})")
                for col, other, corr in removed_corr[:3]:  # Show first 3
                    print(f"    {col} correlated with {other} (r={corr:.3f})")
            selected = final_features
        
        print(f"\nFinal selected features: {len(selected)}")
        for feat in selected:
            print(f"  - {feat}")
        
        return selected
    
    # =========================================
    # MODEL TRAINING
    # =========================================
    
    def remove_outliers(self, X, y, video_names=None, method='iqr', threshold=3.0):
        """
        Remove outlier samples that could corrupt the model.
        
        Args:
            X: Feature matrix
            y: Target values (RPE)
            video_names: Video names for logging
            method: 'iqr' (Interquartile Range) or 'zscore' (Z-score)
            threshold: Threshold for outlier detection
            
        Returns:
            X_clean, y_clean, mask (boolean array of kept samples)
        """
        print(f"\nDetecting outliers using {method} method...")
        
        if method == 'iqr':
            # Detect outliers in feature space using IQR
            Q1 = np.percentile(X, 25, axis=0)
            Q3 = np.percentile(X, 75, axis=0)
            IQR = Q3 - Q1
            
            # Define outliers as points outside [Q1 - threshold*IQR, Q3 + threshold*IQR]
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Check if any feature is an outlier
            outlier_mask = np.any((X < lower_bound) | (X > upper_bound), axis=1)
            
        elif method == 'zscore':
            # Detect outliers using Z-score
            z_scores = np.abs((X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8))
            outlier_mask = np.any(z_scores > threshold, axis=1)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Keep non-outliers
        keep_mask = ~outlier_mask
        X_clean = X[keep_mask]
        y_clean = y[keep_mask]
        
        n_outliers = np.sum(outlier_mask)
        print(f"  Found {n_outliers} outliers ({n_outliers/len(X)*100:.1f}% of data)")
        
        if n_outliers > 0 and video_names is not None:
            outlier_videos = np.array(video_names)[outlier_mask]
            print(f"  Outlier videos: {list(outlier_videos[:5])}")
        
        return X_clean, y_clean, keep_mask
    
    def train(self, features_csv, test_size=0.2, val_size=0.15, feature_selection='auto',
              remove_outliers=True, outlier_method='iqr'):
        """
        Train LGBM model on extracted features.
        
        Args:
            features_csv: CSV file with features and 'rpe' column
            test_size: Fraction of data for testing
            val_size: Fraction of training data for validation
            feature_selection: 'auto', 'composite', or 'all'
            
        Returns:
            dict: Training results and metrics
        """
        print(f"\n{'='*80}")
        print(f"TRAINING LGBM RPE PREDICTION MODEL")
        print(f"{'='*80}\n")
        
        # Load features
        df = pd.read_csv(features_csv)
        print(f"Loaded {len(df)} samples from {features_csv}")
        
        if 'rpe' not in df.columns:
            raise ValueError("CSV must contain 'rpe' column with ground truth RPE values")
        
        # Compute composite features
        df = self.compute_composite_features(df)
        
        # Select features
        if feature_selection == 'auto':
            self.feature_columns = self.select_features(df)
        elif feature_selection == 'composite':
            self.feature_columns = ['facial_strain_composite', 'rom_composite', 'bar_speed_composite']
        elif feature_selection == 'all':
            # Use all numeric columns except rpe and video_name
            self.feature_columns = [col for col in df.columns 
                                   if col not in ['rpe', 'video_name'] 
                                   and df[col].dtype in ['float64', 'int64']]
        
        # Prepare data
        X = df[self.feature_columns].values
        y = df['rpe'].values
        video_names = df['video_name'].values if 'video_name' in df.columns else None
        
        print(f"\nFeature matrix shape: {X.shape}")
        print(f"RPE range: {y.min():.1f} - {y.max():.1f}")
        
        # OPTIMIZATION 3: Remove outliers
        if remove_outliers:
            X, y, keep_mask = self.remove_outliers(X, y, video_names, method=outlier_method)
            if video_names is not None:
                video_names = video_names[keep_mask]
            print(f"  ✓ Cleaned data shape: {X.shape}")
        
        # Split data with stratification by RPE bins
        bins = pd.cut(y, bins=[2.5, 4.5, 6.5, 8.5, 10.5], labels=False)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=bins
        )
        
        # Further split train into train/val
        bins_train = pd.cut(y_train, bins=[2.5, 4.5, 6.5, 8.5, 10.5], labels=False)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=42, stratify=bins_train
        )
        
        print(f"\nData split:")
        print(f"  Training:   {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples")
        print(f"  Test:       {len(X_test)} samples")
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train LGBM with optimized hyperparameters
        print(f"\nTraining LGBMRegressor with {len(self.feature_columns)} features...")
        self.model = LGBMRegressor(
            objective='regression',
            metric='mae',  # Mean Absolute Error - more robust to outliers
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=15,              # Small tree = less overfitting
            max_depth=5,                # Limit depth (was -1)
            min_child_samples=30,       # Increased from 20 for regularization
            min_child_weight=0.001,     # Minimum sum of weights in a leaf
            reg_lambda=2.0,             # L2 regularization (increased from 1.0)
            reg_alpha=1.0,              # L1 regularization (increased from 0.5)
            feature_fraction=0.8,       # Use 80% of features per tree (was 0.9)
            bagging_fraction=0.8,       # Use 80% of data per iteration (was 0.9)
            bagging_freq=5,             # Bagging every 5 iterations (was 1)
            min_split_gain=0.01,        # Minimum gain to split (prevents tiny splits)
            path_smooth=0.1,            # Smoothing for leaf values
            random_state=42,
            verbose=-1,
            force_col_wise=True         # Faster for many features
        )
        
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            eval_metric='l1',
        )
        
        # Evaluate
        results = self._evaluate(X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test)
        
        # OPTIMIZATION 4: Prune low-importance features and retrain (optional)
        self._prune_low_importance_features(
            X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test,
            importance_threshold=0.005  # Remove features with <0.5% importance
        )
        
        # Save model
        model_path = os.path.join(self.output_dir, 'rpe_model.pkl')
        self.save_model(model_path)
        
        return results
    
    def _prune_low_importance_features(self, X_train, y_train, X_val, y_val, X_test, y_test, 
                                       importance_threshold=0.005):
        """
        Remove features with very low importance and retrain.
        This reduces model complexity and potential overfitting.
        """
        importances = self.model.feature_importances_
        total_importance = importances.sum()
        normalized_importance = importances / total_importance if total_importance > 0 else importances
        
        # Find low-importance features
        low_importance_mask = normalized_importance < importance_threshold
        n_low_importance = low_importance_mask.sum()
        
        if n_low_importance == 0:
            print(f"\n✓ All features have sufficient importance (>{importance_threshold*100:.1f}%)")
            return
        
        print(f"\n{'='*80}")
        print(f"FEATURE PRUNING: Found {n_low_importance} low-importance features")
        print(f"{'='*80}")
        
        # Show what's being removed
        removed_features = [self.feature_columns[i] for i, low in enumerate(low_importance_mask) if low]
        print(f"\nRemoving features with <{importance_threshold*100:.1f}% importance:")
        for feat in removed_features[:5]:  # Show first 5
            idx = self.feature_columns.index(feat)
            print(f"  - {feat}: {normalized_importance[idx]*100:.2f}%")
        if len(removed_features) > 5:
            print(f"  ... and {len(removed_features)-5} more")
        
        # Keep only important features
        keep_mask = ~low_importance_mask
        new_feature_columns = [feat for feat, keep in zip(self.feature_columns, keep_mask) if keep]
        
        # Retrain with pruned features
        print(f"\nRetraining with {len(new_feature_columns)} important features...")
        
        X_train_pruned = X_train[:, keep_mask]
        X_val_pruned = X_val[:, keep_mask]
        X_test_pruned = X_test[:, keep_mask]
        
        # Create new model with same hyperparameters
        pruned_model = LGBMRegressor(
            objective='regression',
            metric='mae',
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=5,
            min_child_samples=30,
            min_child_weight=0.001,
            reg_lambda=2.0,
            reg_alpha=1.0,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            min_split_gain=0.01,
            path_smooth=0.1,
            random_state=42,
            verbose=-1,
            force_col_wise=True
        )
        
        pruned_model.fit(
            X_train_pruned, y_train,
            eval_set=[(X_val_pruned, y_val)],
            eval_metric='l1',
        )
        
        # Evaluate pruned model
        y_test_pred_pruned = np.clip(pruned_model.predict(X_test_pruned), 3, 10)
        y_test_pred_original = np.clip(self.model.predict(X_test), 3, 10)
        
        mae_pruned = mean_absolute_error(y_test, y_test_pred_pruned)
        mae_original = mean_absolute_error(y_test, y_test_pred_original)
        
        print(f"\nPruned model performance:")
        print(f"  Original ({len(self.feature_columns)} features): MAE = {mae_original:.3f}")
        print(f"  Pruned ({len(new_feature_columns)} features):   MAE = {mae_pruned:.3f}")
        
        # Use pruned model if performance is similar (within 2%)
        if mae_pruned <= mae_original * 1.02:
            print(f"  ✓ Using pruned model (MAE within 2% and {n_low_importance} fewer features)")
            self.model = pruned_model
            self.feature_columns = new_feature_columns
        else:
            print(f"  ✗ Keeping original model (pruning degraded performance too much)")
    
    def _evaluate(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Evaluate model performance on all splits."""
        results = {}
        
        for split_name, X, y_true in [
            ('train', X_train, y_train),
            ('val', X_val, y_val),
            ('test', X_test, y_test)
        ]:
            y_pred = self.model.predict(X)
            
            # Clip predictions to valid RPE range
            y_pred = np.clip(y_pred, 3, 10)
            
            results[split_name] = {
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred),
                'bias': np.mean(y_pred - y_true),
                'y_true': y_true,
                'y_pred': y_pred
            }
        
        # Print results
        print(f"\n{'='*80}")
        print(f"MODEL PERFORMANCE")
        print(f"{'='*80}\n")
        
        for split_name in ['train', 'val', 'test']:
            r = results[split_name]
            print(f"{split_name.upper()} SET:")
            print(f"  RMSE: {r['rmse']:.3f}")
            print(f"  MAE:  {r['mae']:.3f}")
            print(f"  R²:   {r['r2']:.3f}")
            print(f"  Bias: {r['bias']:+.3f}")
            print()
        
        # Feature importance
        print(f"TOP 10 FEATURE IMPORTANCES:")
        importances = self.model.feature_importances_
        feature_importance = sorted(zip(self.feature_columns, importances), 
                                    key=lambda x: x[1], reverse=True)
        for feat, imp in feature_importance[:10]:
            print(f"  {feat:40s}: {imp:.4f}")
        
        results['feature_importance'] = feature_importance
        
        # Plot results
        self._plot_results(results)
        
        return results
    
    def _plot_results(self, results):
        """Create visualization plots."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Predictions vs True (test set)
        ax = axes[0, 0]
        y_true = results['test']['y_true']
        y_pred = results['test']['y_pred']
        ax.scatter(y_true, y_pred, alpha=0.6)
        ax.plot([3, 10], [3, 10], 'r--', label='Perfect prediction')
        ax.set_xlabel('True RPE')
        ax.set_ylabel('Predicted RPE')
        ax.set_title(f"Test Set (MAE: {results['test']['mae']:.3f})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Residuals
        ax = axes[0, 1]
        residuals = y_pred - y_true
        ax.scatter(y_true, residuals, alpha=0.6)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('True RPE')
        ax.set_ylabel('Residual (Pred - True)')
        ax.set_title('Residual Plot')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Feature importance
        ax = axes[1, 0]
        feature_importance = results['feature_importance'][:15]
        features, importances = zip(*feature_importance)
        y_pos = np.arange(len(features))
        ax.barh(y_pos, importances)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f[:30] for f in features])
        ax.set_xlabel('Importance')
        ax.set_title('Top 15 Feature Importances')
        ax.invert_yaxis()
        
        # Plot 4: Performance comparison
        ax = axes[1, 1]
        splits = ['train', 'val', 'test']
        mae_values = [results[s]['mae'] for s in splits]
        rmse_values = [results[s]['rmse'] for s in splits]
        x = np.arange(len(splits))
        width = 0.35
        ax.bar(x - width/2, mae_values, width, label='MAE')
        ax.bar(x + width/2, rmse_values, width, label='RMSE')
        ax.set_xticks(x)
        ax.set_xticklabels([s.upper() for s in splits])
        ax.set_ylabel('Error')
        ax.set_title('Performance Across Splits')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, 'training_results.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved results plot to: {plot_path}")
        plt.show()
    
    # =========================================
    # MODEL PERSISTENCE
    # =========================================
    
    def save_model(self, model_path):
        """Save trained model and scaler."""
        if self.model is None:
            raise ValueError("No model to save. Train first.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n✓ Saved model to: {model_path}")
    
    def load_model(self, model_path):
        """Load trained model and scaler."""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        
        print(f"✓ Loaded model from: {model_path}")
        print(f"  Features: {len(self.feature_columns)}")


# =========================================
# COMMAND LINE INTERFACE
# =========================================

def main():
    parser = argparse.ArgumentParser(description='Train LGBM RPE prediction model')
    
    # Actions
    parser.add_argument('--extract', action='store_true', help='Extract features from videos')
    parser.add_argument('--train', action='store_true', help='Train model on features')
    
    # Paths
    parser.add_argument('--videos', type=str, help='Directory containing training videos')
    parser.add_argument('--labels', type=str, help='CSV with RPE labels (columns: video_name, rpe)')
    parser.add_argument('--features', type=str, help='CSV with extracted features (if already extracted)')
    parser.add_argument('--output', type=str, default='output/lgbm_models', help='Output directory')
    
    # Training options
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set fraction')
    parser.add_argument('--val-size', type=float, default=0.15, help='Validation set fraction')
    parser.add_argument('--feature-selection', type=str, default='auto', 
                       choices=['auto', 'composite', 'all'], help='Feature selection strategy')
    
    args = parser.parse_args()
    
    trainer = RPEModelTrainer(output_dir=args.output)
    
    # Extract features if requested
    if args.extract:
        if not args.videos or not args.labels:
            parser.error("--extract requires --videos and --labels")
        
        features_csv = os.path.join(args.output, 'training_features.csv')
        df = trainer.extract_features_from_videos(
            video_dir=args.videos,
            rpe_labels_csv=args.labels,
            output_csv=features_csv
        )
        args.features = features_csv.replace('.csv', '_with_rpe.csv')
    
    # Train model if requested
    if args.train:
        if not args.features:
            parser.error("--train requires --features (or --extract)")
        
        results = trainer.train(
            features_csv=args.features,
            test_size=args.test_size,
            val_size=args.val_size,
            feature_selection=args.feature_selection
        )
    
    if not args.extract and not args.train:
        parser.print_help()


if __name__ == '__main__':
    main()
