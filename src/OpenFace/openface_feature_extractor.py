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

from . import openface_utils as ofu


class OpenFaceExtractor:
    """
    Extract facial expression features from exercise videos for exertion prediction.
    """
    
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
    
    def __init__(self, verbose=True, use_pose_guidance=True, sample_fps=10, load_minimal_columns=True, max_only=True, visualize=False):
        """
        Initialize the OpenFace feature extractor.
        
        Args:
            verbose: Print progress messages
            use_pose_guidance: Use MediaPipe to crop to face (improves accuracy)
            sample_fps: Target FPS for frame sampling (default: 10)
                       Lower = faster processing, less data
                       Higher = more data, slower processing
                       Set to None to use all frames
            load_minimal_columns: Only load RPE-relevant columns (faster)
            max_only: If True, extract only max/peak features (for RPE prediction)
            visualize: Show video with landmarks drawn in real-time
        """
        self.verbose = verbose
        self.use_pose_guidance = use_pose_guidance
        self.sample_fps = sample_fps
        self.load_minimal_columns = load_minimal_columns
        self.max_only = max_only
        self.visualize = visualize
        self._check_openface()
    
    def _check_openface(self):
        """Verify OpenFace binary is available."""
        if not ofu.check_openface_binary():
            raise RuntimeError("OpenFace binary not found. Cannot extract features.")
    
    def _log(self, message):
        """Print message if verbose mode enabled."""
        if self.verbose:
            print(message)
    
    def _visualize_landmarks(self, video_path, csv_path):
        """
        Display video with OpenFace landmarks drawn on frames.
        
        Args:
            video_path: Path to the video file (pose-guided if use_pose_guidance=True)
            csv_path: Path to the OpenFace CSV output
        """
        # Check if display is available (important for WSL/SSH)
        try:
            import os as os_check
            if 'DISPLAY' not in os_check.environ and not os_check.path.exists('/mnt/wslg'):
                self._log("WARNING: No display detected (DISPLAY not set, WSLg not found)")
                self._log("  For WSL2: Ensure Windows 11 with WSLg, or use VcXsrv/X11")
                self._log("  For SSH: Use 'ssh -X' for X11 forwarding")
                self._log("  Skipping visualization...")
                return
        except:
            pass
        
        # Load landmark data
        df = pd.read_csv(csv_path)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self._log(f"Warning: Could not open video for visualization: {video_path}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self._log(f"\nVISUALIZATION ACTIVE")
        self._log(f"  Video: {os.path.basename(video_path)}")
        self._log(f"  Frames: {total_frames}, FPS: {fps:.1f}")
        self._log(f"  Controls: Press 'q' to quit, 's' to skip")
        
        window_name = f"OpenFace Landmarks - {os.path.basename(video_path)}"
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        except Exception as e:
            self._log(f"ERROR: Could not create window (display issue): {e}")
            self._log("  Check DISPLAY variable or WSLg/X11 setup")
            cap.release()
            return
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Find corresponding row in CSV (match by frame number)
            if frame_idx < len(df):
                row = df.iloc[frame_idx]
                success = row.get('success', 0) == 1
                confidence = row.get('confidence', 0)
                
                # Draw detection status and confidence
                status_text = f"Frame {frame_idx}/{total_frames}"
                conf_text = f"Detection: {'SUCCESS' if success else 'FAILED'} (conf: {confidence:.2f})"
                
                cv2.putText(frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, conf_text, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                           (0, 255, 0) if success else (0, 0, 255), 2)
                
                if success:
                    # Draw 2D facial landmarks (x_0 to x_67, y_0 to y_67)
                    for i in range(68):
                        x_col = f'x_{i}'
                        y_col = f'y_{i}'
                        if x_col in row and y_col in row:
                            x = int(row[x_col])
                            y = int(row[y_col])
                            # Draw landmark point
                            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                    
                    # Draw AU intensities for key AUs
                    y_offset = 90
                    for au in ['AU04_r', 'AU06_r', 'AU12_r']:
                        if au in row:
                            au_val = row[au]
                            au_text = f"{au}: {au_val:.2f}"
                            cv2.putText(frame, au_text, (10, y_offset),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                            y_offset += 25
            
            # Display frame
            cv2.imshow(window_name, frame)
            
            # Control playback speed (approximate original FPS)
            key = cv2.waitKey(int(1000/fps)) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Skip to end
                break
            
            frame_idx += 1
        
        cap.release()
        cv2.destroyAllWindows()
        self._log("Visualization complete")
    
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
        
        # Visualize if requested (before processing features)
        if self.visualize:
            # The CSV contains coordinates relative to the CROPPED video
            # So we MUST visualize the pose-guided version, not the original
            vis_video_path = video_path
            if self.use_pose_guidance:
                # Pose-guided video is saved in output/openface/
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                pose_guided_path = os.path.join(repo_root, "output", "openface", f"{video_name}_pose_guided.mp4")
                if os.path.exists(pose_guided_path):
                    vis_video_path = pose_guided_path
                    self._log(f"Visualizing cropped video: {os.path.basename(pose_guided_path)}")
                else:
                    self._log(f"WARNING: Pose-guided video not found: {pose_guided_path}")
                    self._log("  Landmarks will not align correctly with original video!")
            
            self._visualize_landmarks(vis_video_path, csv_path)
        
        # Load data with minimal columns if enabled
        columns_to_load = 'rpe_minimal' if self.load_minimal_columns else None
        df_all = ofu.load_landmark_data(csv_path, success_only=False, 
                                        sample_fps=self.sample_fps, 
                                        columns_only=columns_to_load)
        df = ofu.load_landmark_data(csv_path, success_only=True,
                                    sample_fps=self.sample_fps,
                                    columns_only=columns_to_load)
        
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
        
        self._log(f"\nExtracted {len([k for k in features.keys() if k != 'metadata'])} features")
        
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
        """Extract Action Unit statistical features."""
        features = {}
        
        # For each exertion AU
        for au in self.EXERTION_AUS:
            if au not in df.columns:
                continue
            
            au_data = df[au].values
            prefix = au.replace('_r', '')
            
            if self.max_only:
                # Extract only max/peak features for RPE prediction
                features[f'{prefix}_max'] = np.max(au_data)
                features[f'{prefix}_q95'] = np.percentile(au_data, 95)
                # Peak ratio: max / mean (indicates how much stronger peak is vs average)
                mean_val = np.mean(au_data)
                features[f'{prefix}_peak_ratio'] = np.max(au_data) / mean_val if mean_val > 0.01 else 0
            else:
                # Full statistical features
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
        
        if not self.max_only:
            # Overall AU activity (only in full mode)
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
