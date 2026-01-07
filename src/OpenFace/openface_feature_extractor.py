"""
OpenFace Feature Extractor for Exercise Exertion Prediction

Production module for extracting facial expression features from exercise videos.
Outputs formatted features ready for LGBM regression model.

Usage:
    from openface_feature_extractor import OpenFaceExtractor
    
    extractor = OpenFaceExtractor()
    features = extractor.extract_from_video('workout_video.mp4')
    # features is a dict ready for LGBM input
"""

import os
import cv2
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

import openface_utils as ofu


class OpenFaceExtractor:
    
    # Key Action Units for physical exertion
    EXERTION_AUS = [
        'AU04_r',  # Brow Lowerer (concentration/strain)
        'AU06_r',  # Cheek Raiser (squinting/strain)
        'AU07_r',  # Lid Tightener (strain)
        'AU09_r',  # Nose Wrinkler (strain)
        'AU10_r',  # Upper Lip Raiser (strain)
        'AU12_r',  # Lip Corner Puller (grimace)
        'AU17_r',  # Chin Raiser (strain)
        'AU20_r',  # Lip Stretcher (strain)
        'AU25_r',  # Lips Part (breathing/exertion)
        'AU26_r',  # Jaw Drop (breathing/exertion)
    ]
    
    # All AU intensity columns
    ALL_AU_INTENSITY = [
        'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r',
        'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r',
        'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r'
    ]
    
    # AU classification columns (presence/absence)
    ALL_AU_CLASSIFICATION = [
        'AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c',
        'AU09_c', 'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c',
        'AU20_c', 'AU23_c', 'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c'
    ]
    
    def __init__(self, verbose=True, use_pose_guidance=True, sample_fps=10, load_minimal_columns=True, max_only=True):
        """
        Initialize OpenFace feature extractor.
        
        Args:
            verbose: Print progress messages
            use_pose_guidance: Use MediaPipe to crop to face (improves accuracy)
            sample_fps: Target FPS for frame sampling (default: 10 = analyze 10 frames/sec)
                       Set to None to use all frames (slower but more data)
            load_minimal_columns: If True, only load columns needed for RPE (faster, less memory)
            max_only: If True, focus on maximum AU values for peak RPE prediction (default: True)
        """
        self.verbose = verbose
        self.use_pose_guidance = use_pose_guidance
        self.sample_fps = sample_fps
        self.load_minimal_columns = load_minimal_columns
        self.max_only = max_only
        self._check_openface()
    
    def _check_openface(self):
        """Verify OpenFace binary is available."""
        if not ofu.check_openface_binary():
            raise RuntimeError("OpenFace binary not found. Cannot extract features.")
    
    def _log(self, message):
        """Print message if verbose mode enabled."""
        if self.verbose:
            print(message)
    
    # =========================================
    # CORE EXTRACTION
    # =========================================
    
    def extract_from_video(self, video_path, expected_reps=None):
        """
        Extract facial expression features from a video.
        
        Args:
            video_path: Path to video file
            expected_reps: Expected number of repetitions (optional, for validation)
        
        Returns:
            dict: Feature dictionary ready for LGBM model with keys:
                - 'detection_rate': Face detection success rate
                - 'au_*': AU-based features (mean, max, std, range, etc.)
                - 'rep_*': Repetition-based features
                - 'landmark_*': Landmark stability features
                - 'metadata': Video metadata
        """
        video_name = os.path.basename(video_path).split('.')[0]
        self._log(f"\n{'='*60}\nExtracting features from: {video_name}\n{'='*60}")
        
        # Get video metadata
        metadata = ofu.get_video_metadata(video_path)
        if not metadata:
            raise ValueError(f"Could not open video: {video_path}")
        
        self._log(f"Resolution: {metadata['resolution']}, FPS: {metadata['fps']:.1f}, Frames: {metadata['total_frames']}")
        
        # Run OpenFace with pose guidance if enabled
        csv_path = ofu.run_openface(video_path, use_pose_guidance=self.use_pose_guidance)
        
        # Load data with optimizations
        columns_filter = 'rpe_minimal' if self.load_minimal_columns else None
        df_all = ofu.load_landmark_data(csv_path, success_only=False, 
                                       sample_fps=self.sample_fps,
                                       columns_only=columns_filter)
        df = ofu.load_landmark_data(csv_path, success_only=True,
                                   sample_fps=self.sample_fps, 
                                   columns_only=columns_filter)
        
        if len(df) < 10:
            raise ValueError(f"Insufficient valid frames ({len(df)} detected). Check video quality.")
        
        self._log(f"Detected faces in {len(df)}/{len(df_all)} frames ({len(df)/len(df_all)*100:.1f}%)")
        
        # Extract feature groups
        features = {}
        
        # 1. Detection quality
        features.update(self._extract_detection_features(df_all, metadata))
        
        # 2. AU intensity features
        features.update(self._extract_au_features(df))
        
        # 3. Repetition features
        rep_features = self._extract_repetition_features(df, expected_reps)
        features.update(rep_features)
        
        # 4. Landmark stability features
        features.update(self._extract_landmark_features(df))
        
        # 5. Temporal dynamics
        features.update(self._extract_temporal_features(df))
        
        # 6. Metadata
        features['metadata'] = {
            'video_name': video_name,
            'width': metadata['width'],
            'height': metadata['height'],
            'fps': metadata['fps'],
            'total_frames': metadata['total_frames'],
            'valid_frames': len(df)
        }
        
        self._log(f"\n✓ Extracted {len([k for k in features.keys() if k != 'metadata'])} features")
        
        return features
    
    # =========================================
    # FEATURE EXTRACTION METHODS
    # =========================================
    
    def _extract_detection_features(self, df_all, metadata):
        """Extract face detection quality features."""
        detected = (df_all['success'] == 1).sum()
        total = len(df_all)
        
        # Confidence statistics (for detected frames only)
        df_detected = df_all[df_all['success'] == 1]
        conf_mean = df_detected['confidence'].mean() if 'confidence' in df_detected.columns and len(df_detected) > 0 else 0
        conf_std = df_detected['confidence'].std() if 'confidence' in df_detected.columns and len(df_detected) > 0 else 0
        conf_min = df_detected['confidence'].min() if 'confidence' in df_detected.columns and len(df_detected) > 0 else 0
        
        return {
            'detection_rate': detected / total if total > 0 else 0,
            'detection_count': detected,
            'total_frames': total,
            'confidence_mean': conf_mean,
            'confidence_std': conf_std,
            'confidence_min': conf_min,
        }
    
    def _extract_au_features(self, df):
        """Extract Action Unit features (max-focused for peak RPE prediction)."""
        features = {}
        
        if self.max_only:
            # MAX-ONLY MODE: Focus on peak AU intensities for max RPE prediction
            # Only extract maximum values - the point of highest exertion
            
            for au in self.EXERTION_AUS:
                if au not in df.columns:
                    continue
                
                au_data = df[au].values
                prefix = au.replace('_r', '')
                
                # Maximum intensity (primary feature for peak exertion)
                features[f'{prefix}_max'] = np.max(au_data)
                
                # 95th percentile (captures sustained peak)
                features[f'{prefix}_q95'] = np.percentile(au_data, 95)
                
                # Peak-to-baseline ratio (how much AU spiked)
                baseline = np.percentile(au_data, 10)
                peak = np.max(au_data)
                features[f'{prefix}_peak_ratio'] = peak / baseline if baseline > 0.01 else peak
            
            # Overall maximum AU activity
            all_au_data = df[self.EXERTION_AUS].values
            features['au_overall_max'] = np.max(all_au_data)
            
            # Count of AUs with high peak intensity (max > 2.0)
            features['au_high_peak_count'] = sum([
                1 for au in self.EXERTION_AUS 
                if au in df.columns and df[au].max() > 2.0
            ])
            
        else:
            # FULL STATISTICS MODE: Extract comprehensive features (legacy)
            for au in self.EXERTION_AUS:
                if au not in df.columns:
                    continue
                
                au_data = df[au].values
                prefix = au.replace('_r', '')
                
                # Basic statistics
                features[f'{prefix}_mean'] = np.mean(au_data)
                features[f'{prefix}_std'] = np.std(au_data)
                features[f'{prefix}_max'] = np.max(au_data)
                features[f'{prefix}_min'] = np.min(au_data)
                features[f'{prefix}_range'] = np.max(au_data) - np.min(au_data)
                features[f'{prefix}_median'] = np.median(au_data)
                
                # Percentiles
                features[f'{prefix}_q25'] = np.percentile(au_data, 25)
                features[f'{prefix}_q75'] = np.percentile(au_data, 75)
                features[f'{prefix}_q90'] = np.percentile(au_data, 90)
                
                # Shape features
                features[f'{prefix}_skew'] = skew(au_data)
                features[f'{prefix}_kurtosis'] = kurtosis(au_data)
                
                # Activation ratio (% of time AU is active, >0.5 intensity)
                features[f'{prefix}_activation_ratio'] = np.mean(au_data > 0.5)
                
                # Peak-to-baseline ratio
                baseline = np.percentile(au_data, 10)
                peak = np.percentile(au_data, 90)
                features[f'{prefix}_peak_baseline_ratio'] = peak / baseline if baseline > 0.01 else 0
            
            # Overall AU activity
            all_au_data = df[self.EXERTION_AUS].values
            features['au_overall_mean'] = np.mean(all_au_data)
            features['au_overall_max'] = np.max(all_au_data)
            features['au_overall_std'] = np.std(all_au_data)
            
            # Count of highly active AUs (mean > 1.0)
            features['au_highly_active_count'] = sum([
                1 for au in self.EXERTION_AUS 
                if au in df.columns and df[au].mean() > 1.0
            ])
        
        return features
    
    def _extract_repetition_features(self, df, expected_reps=None):
        """Extract repetition-based features."""
        features = {}
        
        # Use most responsive AU for rep detection
        au_ranges = {au: df[au].max() - df[au].min() 
                     for au in self.EXERTION_AUS if au in df.columns}
        
        if not au_ranges:
            return {
                'detected_reps': 0,
                'rep_consistency': 0,
                'rep_avg_intensity': 0,
            }
        
        best_au = max(au_ranges.items(), key=lambda x: x[1])[0]
        au_data = df[best_au].values
        
        # Detect peaks (repetitions)
        peaks, smoothed = self._detect_peaks(au_data)
        
        features['detected_reps'] = len(peaks)
        
        if len(peaks) >= 2:
            peak_values = au_data[peaks]
            
            # Repetition consistency
            peak_mean = np.mean(peak_values)
            peak_std = np.std(peak_values)
            features['rep_consistency'] = 1 - (peak_std / peak_mean) if peak_mean > 0 else 0
            features['rep_avg_intensity'] = peak_mean
            features['rep_intensity_std'] = peak_std
            features['rep_intensity_range'] = np.max(peak_values) - np.min(peak_values)
            
            # Rep timing (frames between peaks)
            if len(peaks) > 1:
                rep_intervals = np.diff(peaks)
                features['rep_avg_interval'] = np.mean(rep_intervals)
                features['rep_interval_std'] = np.std(rep_intervals)
                features['rep_tempo_consistency'] = 1 - (np.std(rep_intervals) / np.mean(rep_intervals)) if np.mean(rep_intervals) > 0 else 0
            else:
                features['rep_avg_interval'] = 0
                features['rep_interval_std'] = 0
                features['rep_tempo_consistency'] = 0
        else:
            features['rep_consistency'] = 0
            features['rep_avg_intensity'] = 0
            features['rep_intensity_std'] = 0
            features['rep_intensity_range'] = 0
            features['rep_avg_interval'] = 0
            features['rep_interval_std'] = 0
            features['rep_tempo_consistency'] = 0
        
        # Rep detection accuracy (if expected reps provided)
        if expected_reps:
            features['rep_detection_accuracy'] = min(1.0, features['detected_reps'] / expected_reps)
        else:
            features['rep_detection_accuracy'] = 1.0  # Unknown, assume perfect
        
        return features
    
    def _detect_peaks(self, signal, min_distance=20):
        """Detect peaks in AU signal for repetition counting."""
        # Smooth signal
        if len(signal) > 5:
            window_len = min(11, len(signal) if len(signal) % 2 == 1 else len(signal) - 1)
            smoothed = savgol_filter(signal, window_len, 3)
        else:
            smoothed = signal
        
        # Find peaks with prominence
        peaks, _ = find_peaks(smoothed, distance=min_distance, prominence=0.3)
        
        return peaks, smoothed
    
    def _extract_landmark_features(self, df):
        """Extract facial landmark stability features."""
        features = {}
        
        # Calculate landmark movement/jitter
        x_cols = [f'x_{i}' for i in range(68)]
        y_cols = [f'y_{i}' for i in range(68)]
        
        # Average landmark stability (lower = more stable)
        x_stds = [df[col].std() for col in x_cols if col in df.columns]
        y_stds = [df[col].std() for col in y_cols if col in df.columns]
        
        features['landmark_stability_x'] = np.mean(x_stds) if x_stds else 0
        features['landmark_stability_y'] = np.mean(y_stds) if y_stds else 0
        features['landmark_stability_overall'] = np.mean(x_stds + y_stds) if (x_stds and y_stds) else 0
        
        # Head movement (based on face center movement)
        if 'x_30' in df.columns and 'y_30' in df.columns:  # Nose tip
            nose_x = df['x_30'].values
            nose_y = df['y_30'].values
            
            # Movement magnitude
            dx = np.diff(nose_x)
            dy = np.diff(nose_y)
            movement = np.sqrt(dx**2 + dy**2)
            
            features['head_movement_mean'] = np.mean(movement)
            features['head_movement_max'] = np.max(movement)
            features['head_movement_std'] = np.std(movement)
        else:
            features['head_movement_mean'] = 0
            features['head_movement_max'] = 0
            features['head_movement_std'] = 0
        
        return features
    
    def _extract_temporal_features(self, df):
        """Extract temporal dynamics of facial expressions."""
        features = {}
        
        # Rate of change for key AUs (velocity)
        for au in self.EXERTION_AUS[:3]:  # Top 3 AUs to avoid too many features
            if au not in df.columns:
                continue
            
            au_data = df[au].values
            velocity = np.abs(np.diff(au_data))
            
            prefix = au.replace('_r', '')
            features[f'{prefix}_velocity_mean'] = np.mean(velocity)
            features[f'{prefix}_velocity_max'] = np.max(velocity)
            features[f'{prefix}_velocity_std'] = np.std(velocity)
        
        # Overall expression change rate
        all_au_data = df[self.EXERTION_AUS].values
        overall_velocity = np.mean(np.abs(np.diff(all_au_data, axis=0)), axis=1)
        
        features['expression_change_mean'] = np.mean(overall_velocity)
        features['expression_change_max'] = np.max(overall_velocity)
        features['expression_change_std'] = np.std(overall_velocity)
        
        return features
    
    # =========================================
    # BATCH PROCESSING
    # =========================================
    
    def extract_from_directory(self, video_dir, output_csv=None, expected_reps=None):
        """
        Extract features from all videos in a directory.
        
        Args:
            video_dir: Directory containing video files
            output_csv: Path to save feature CSV (optional)
            expected_reps: Expected reps per video (can be dict {video_name: reps})
        
        Returns:
            pd.DataFrame: Features for all videos
        """
        video_files = ofu.find_videos(['*.mp4', '*.avi', '*.mov', '*.mkv'], search_dir=video_dir)
        
        if not video_files:
            raise ValueError(f"No videos found in {video_dir}")
        
        self._log(f"\nProcessing {len(video_files)} videos from {video_dir}\n{'='*60}")
        
        all_features = []
        
        for idx, video_path in enumerate(video_files, 1):
            video_name = os.path.basename(video_path).split('.')[0]
            
            # Get expected reps for this video
            if isinstance(expected_reps, dict):
                reps = expected_reps.get(video_name, None)
            else:
                reps = expected_reps
            
            try:
                self._log(f"\n[{idx}/{len(video_files)}] Processing: {video_name}")
                features = self.extract_from_video(video_path, expected_reps=reps)
                
                # Flatten features (exclude metadata dict)
                flat_features = {'video_name': video_name}
                for key, value in features.items():
                    if key == 'metadata':
                        for mk, mv in value.items():
                            flat_features[f'meta_{mk}'] = mv
                    else:
                        flat_features[key] = value
                
                all_features.append(flat_features)
                
            except Exception as e:
                self._log(f"ERROR processing {video_name}: {e}")
                continue
        
        # Create DataFrame
        df = pd.DataFrame(all_features)
        
        # Save if requested
        if output_csv:
            df.to_csv(output_csv, index=False)
            self._log(f"\n✓ Saved features to: {output_csv}")
        
        self._log(f"\n{'='*60}\n✓ Completed: {len(all_features)}/{len(video_files)} videos processed")
        self._log(f"✓ Total features per video: {len(df.columns) - 1}")  # Exclude video_name
        
        return df
    
    # =========================================
    # FEATURE SUMMARY
    # =========================================
    
    def get_feature_names(self):
        """Get list of all feature names that will be extracted."""
        # Run on a dummy DataFrame to get feature names
        dummy_df = pd.DataFrame({
            'frame': [1, 2, 3],
            'success': [1, 1, 1],
            'confidence': [0.9, 0.9, 0.9],
            **{au: [0.5, 1.0, 0.5] for au in self.EXERTION_AUS},
            **{f'x_{i}': [100.0, 101.0, 100.0] for i in range(68)},
            **{f'y_{i}': [100.0, 101.0, 100.0] for i in range(68)},
        })
        
        features = {}
        features.update(self._extract_au_features(dummy_df))
        features.update(self._extract_repetition_features(dummy_df, expected_reps=3))
        features.update(self._extract_landmark_features(dummy_df))
        features.update(self._extract_temporal_features(dummy_df))
        features.update(self._extract_detection_features(
            pd.DataFrame({'success': [1, 1, 1], 'confidence': [0.9, 0.9, 0.9]}),
            {'width': 1920, 'height': 1080, 'fps': 30, 'total_frames': 3}
        ))
        
        return sorted(features.keys())
    
    def print_feature_summary(self):
        """Print a summary of all features that will be extracted."""
        feature_names = self.get_feature_names()
        
        print(f"\n{'='*60}")
        print(f"OpenFace Feature Extractor - Feature Summary")
        print(f"{'='*60}")
        print(f"Total Features: {len(feature_names)}\n")
        
        # Group by category
        categories = {
            'Detection': [f for f in feature_names if f.startswith('detection') or f.startswith('confidence')],
            'AU Statistics': [f for f in feature_names if any(au.replace('_r', '') in f for au in self.EXERTION_AUS) and not ('velocity' in f)],
            'Repetition': [f for f in feature_names if f.startswith('rep_')],
            'Landmark': [f for f in feature_names if f.startswith('landmark_') or f.startswith('head_')],
            'Temporal': [f for f in feature_names if 'velocity' in f or 'change' in f],
            'Overall AU': [f for f in feature_names if f.startswith('au_overall') or f.startswith('au_highly')],
        }
        
        for category, features in categories.items():
            if features:
                print(f"{category} ({len(features)} features):")
                for f in sorted(features)[:5]:  # Show first 5
                    print(f"  - {f}")
                if len(features) > 5:
                    print(f"  ... and {len(features) - 5} more")
                print()
        
        print(f"{'='*60}\n")


# =========================================
# CONVENIENCE FUNCTIONS
# =========================================

def extract_features(video_path, expected_reps=None, verbose=True, use_pose_guidance=True):
    """
    Quick function to extract features from a single video.
    
    Args:
        video_path: Path to video file
        expected_reps: Expected number of repetitions (optional)
        verbose: Print progress
        use_pose_guidance: Use MediaPipe pose to focus on main person's face (default: True)
    
    Returns:
        dict: Feature dictionary
    """
    extractor = OpenFaceExtractor(verbose=verbose, use_pose_guidance=use_pose_guidance)
    return extractor.extract_from_video(video_path, expected_reps=expected_reps)


def extract_features_batch(video_dir, output_csv=None, expected_reps=None, verbose=True, use_pose_guidance=True):
    """
    Quick function to extract features from all videos in a directory.
    
    Args:
        video_dir: Directory containing videos
        output_csv: Path to save CSV (optional)
        expected_reps: Expected reps (int or dict)
        verbose: Print progress
        use_pose_guidance: Use MediaPipe pose to focus on main person's face (default: True)
    
    Returns:
        pd.DataFrame: Features for all videos
    """
    extractor = OpenFaceExtractor(verbose=verbose, use_pose_guidance=use_pose_guidance)
    return extractor.extract_from_directory(video_dir, output_csv=output_csv, expected_reps=expected_reps)


# =========================================
# MAIN (FOR TESTING)
# =========================================

if __name__ == "__main__":
    import sys
    
    print(f"\n{'='*60}")
    print(f"OpenFace Feature Extractor - Production Module")
    print(f"{'='*60}\n")
    
    # Show feature summary
    extractor = OpenFaceExtractor(verbose=True)
    extractor.print_feature_summary()
    
    # Check if video path provided
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        
        if os.path.isfile(video_path):
            # Single video
            print(f"Extracting features from: {video_path}\n")
            features = extractor.extract_from_video(video_path)
            
            print(f"\n{'='*60}")
            print(f"Sample Features (first 10):")
            print(f"{'='*60}")
            for key, value in list(features.items())[:10]:
                if key != 'metadata':
                    print(f"{key}: {value}")
            
        elif os.path.isdir(video_path):
            # Directory of videos
            output_csv = os.path.join(ofu.OUTPUT_DIR, f"features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            df = extractor.extract_from_directory(video_path, output_csv=output_csv)
            print(f"\n{df.head()}")
        else:
            print(f"ERROR: {video_path} is not a valid file or directory")
    else:
        print("Usage:")
        print("  Single video:  python openface_feature_extractor.py path/to/video.mp4")
        print("  Directory:     python openface_feature_extractor.py path/to/videos/")
        print("\nNo video provided, showing feature summary only.")
