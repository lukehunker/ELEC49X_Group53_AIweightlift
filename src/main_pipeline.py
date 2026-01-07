"""
Unified RPE Prediction Pipeline
"""

import os
import cv2
import pandas as pd
import numpy as np
from pathlib import Path

# Import your modules
from OpenFace.openface_feature_extractor import OpenFaceExtractor
from MMPose.pose_feature_extractor import PoseFeatureExtractor  
from MMDetection.bar_feature_extractor import BarFeatureExtractor 

class RPEPipeline:
    def __init__(self, lgbm_model_path=None, sample_fps=10, max_only=True):
        """
        Initialize pipeline with all feature extractors.
        
        Args:
            lgbm_model_path: Path to trained LGBM model (optional, for prediction)
            sample_fps: Target FPS for OpenFace frame sampling (default: 10)
                       Lower = faster processing, less data
                       Higher = more data, slower processing
                       Set to None to use all frames
            max_only: If True, focus on maximum AU intensities for peak RPE (default: True)
        """
        self.openface_extractor = OpenFaceExtractor(
            use_pose_guidance=True,
            sample_fps=sample_fps,
            load_minimal_columns=True,  # Only load RPE-relevant columns
            max_only=max_only  # Focus on peak exertion features
        )
        self.pose_extractor = PoseFeatureExtractor()
        self.bar_extractor = BarFeatureExtractor()
        
        # Load LGBM model if provided
        self.model = None
        self.scaler = None
        self.feature_columns = None
        
        if lgbm_model_path:
            self.load_model(lgbm_model_path)
    
    def preprocess_video(self, video_path, output_path=None):
        """
        Standardize video format before feature extraction.
        """
        if output_path is None:
            video_name = Path(video_path).stem
            output_path = f"output/preprocessed/{video_name}_preprocessed.mp4"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        
        # Get original properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Target properties
        TARGET_FPS = 60
        TARGET_WIDTH = 1920
        TARGET_HEIGHT = 1080
        
        # Setup writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, TARGET_FPS, 
                             (TARGET_WIDTH, TARGET_HEIGHT))
        
        frame_interval = int(original_fps / TARGET_FPS)
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames to achieve target FPS
            if frame_idx % frame_interval == 0:
                # Resize to target resolution
                frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
                out.write(frame)
            
            frame_idx += 1
        
        cap.release()
        out.release()
        
        return output_path
    
    def extract_features(self, video_path, expected_reps=None, preprocess=True):
        
        # Extract features from all modules for a single video
    
        # Preprocess if requested
        if preprocess:
            print(f"Preprocessing video: {video_path}")
            processed_video = self.preprocess_video(video_path)
        else:
            processed_video = video_path
        
        print(f"\nExtracting features from: {os.path.basename(processed_video)}")
        
    
        features = {}
        
        # 1. OpenFace - Facial Action Units
        print("OpenFace (facial strain)...")
        openface_features = self.openface_extractor.extract_from_video(
            processed_video, expected_reps=expected_reps
        )
        for key, value in openface_features.items():
            if key != 'metadata':
                features[f'face_{key}'] = value
        
        # 2. MMPose - Body dynamics and ROM
        print("MMPose (body pose & ROM)...")
        pose_features = self.pose_extractor.extract_from_video(
            processed_video, expected_reps=expected_reps
        )
        for key, value in pose_features.items():
            features[f'pose_{key}'] = value
        
        # 3. Bar Tracking - Movement velocity
        print("Bar Tracking (speed)...")
        bar_features = self.bar_extractor.extract_from_video(
            processed_video, expected_reps=expected_reps
        )
        for key, value in bar_features.items():
            features[f'bar_{key}'] = value
        
        return features
    
    def extract_features_batch(self, video_dir, output_csv=None, expected_reps=None):
        """
        Extract features from all videos in a directory.
        
        Args:
            video_dir: Directory containing videos
            output_csv: Where to save feature CSV
            expected_reps: Expected reps (int or dict {video_name: reps})
            
        Returns:
            pd.DataFrame: Features for all videos
        """
        video_files = sorted(Path(video_dir).glob('*.mp4'))
        
        all_features = []
        
        for idx, video_path in enumerate(video_files, 1):
            video_name = video_path.stem
            
            # Get expected reps for this video
            if isinstance(expected_reps, dict):
                reps = expected_reps.get(video_name, None)
            else:
                reps = expected_reps
            
            try:
                print(f"\n[{idx}/{len(video_files)}] Processing: {video_name}")
                features = self.extract_features(str(video_path), expected_reps=reps)
                features['video_name'] = video_name
                all_features.append(features)
                
            except Exception as e:
                print(f"ERROR processing {video_name}: {e}")
                continue
        
        # Create DataFrame
        df = pd.DataFrame(all_features)
        
        # Save if requested
        if output_csv:
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            df.to_csv(output_csv, index=False)
            print(f"\n✓ Saved features to: {output_csv}")
        
        return df
    
    def predict_rpe(self, video_path, expected_reps=None):
        """
        Predict RPE for a single video.
        
        Args:
            video_path: Path to video file
            expected_reps: Expected number of reps
            
        Returns:
            float: Predicted RPE score
        """
        if self.model is None:
            raise ValueError("No LGBM model loaded. Provide lgbm_model_path in __init__")
        
        # Extract features
        features = self.extract_features(video_path, expected_reps)
        
        # Compute composite features (same as training)
        features = self._compute_composite_features(features)
        
        # Select only the features the model was trained on
        X = np.array([[features.get(col, 0) for col in self.feature_columns]])
        
        # Scale features
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # Predict
        rpe = self.model.predict(X)[0]
        
        # Clip to valid range
        rpe = np.clip(rpe, 3, 10)
        
        return rpe
    
    def _compute_composite_features(self, features):
        """
        Compute composite features from raw extracted features.
        Uses MAX values for peak RPE prediction.
        This must match the training script's computation.
        """
        # Facial strain composite - using MAX intensities for peak exertion
        au_weights = {
            'face_AU04_max': 0.25,  # Brow Lowerer (concentration/strain)
            'face_AU06_max': 0.20,  # Cheek Raiser (squinting)
            'face_AU07_max': 0.15,  # Lid Tightener
            'face_AU09_max': 0.10,  # Nose Wrinkler
            'face_AU25_max': 0.15,  # Lips Part (breathing)
            'face_AU26_max': 0.15,  # Jaw Drop (exertion)
        }
        
        facial_strain = sum(features.get(au, 0) * weight for au, weight in au_weights.items())
        features['facial_strain_composite'] = facial_strain
        
        # ROM composite
        if 'pose_rom_left_knee' in features and 'pose_rom_right_knee' in features:
            features['rom_composite'] = (features['pose_rom_left_knee'] + features['pose_rom_right_knee']) / 2
        elif 'pose_movement_hip_range' in features:
            features['rom_composite'] = features['pose_movement_hip_range']
        else:
            features['rom_composite'] = 0
        
        # Bar speed composite
        features['bar_speed_composite'] = features.get('bar_speed_change', 0)
        
        return features
    
    def load_model(self, model_path):
        """Load trained LGBM model with scaler and feature columns."""
        import pickle
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data.get('scaler', None)
        self.feature_columns = model_data.get('feature_columns', [])
        
        print(f"✓ Loaded LGBM model from: {model_path}")
        print(f"  Features: {len(self.feature_columns)}")