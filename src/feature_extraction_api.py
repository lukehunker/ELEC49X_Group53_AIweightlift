"""
Wrapper functions for feature extraction to be called from API server.
These wrap the existing extraction scripts into simple Python functions.
"""
import os
import sys
import json
import subprocess
import pandas as pd
import numpy as np

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def extract_mmpose_features(video_path, lift_type="Squat"):
    """
    Extract body pose features from video using MMPose.
    
    Args:
        video_path: Path to video file
        lift_type: Type of lift (Bench Press, Squat, Deadlift)
        
    Returns:
        dict: Pose features (angles, velocities, etc.)
    """
    # Determine which extraction script to use
    script_map = {
        "Squat": "feature_extraction_squat.py",
        "Deadlift": "feature_extraction_deadlift.py",
        "Bench Press": "feature_extraction_bench_v2.py"
    }
    
    script_name = script_map.get(lift_type, "feature_extraction_squat.py")
    script_path = os.path.join(os.path.dirname(__file__), "MMPose", script_name)
    
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"MMPose script not found: {script_path}")
    
    # Create temp output file for keypoints
    temp_json = video_path.replace(".mp4", "_keypoints.json")
    
    try:
        # Run MMPose extraction
        cmd = [
            sys.executable,  # python
            script_path,
            video_path,
            "--out", temp_json,
            "--device", "cpu"  # Use CPU for server (or cuda:0 if GPU available)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            raise RuntimeError(f"MMPose extraction failed: {result.stderr}")
        
        # Load extracted keypoints
        if not os.path.exists(temp_json):
            raise FileNotFoundError(f"MMPose output not generated: {temp_json}")
        
        with open(temp_json, 'r') as f:
            keypoints_data = json.load(f)
        
        # Calculate aggregate features from keypoints
        features = _calculate_pose_features(keypoints_data, lift_type)
        
        return features
        
    finally:
        # Cleanup temp file
        if os.path.exists(temp_json):
            os.remove(temp_json)


def _calculate_pose_features(keypoints_data, lift_type):
    """
    Calculate aggregate features from raw keypoints.
    
    This is a simplified version - you should match the features 
    used during training in dmetrics_to_excel.py
    """
    frames = keypoints_data.get('frames', [])
    
    if not frames:
        return {
            'pose_mean_velocity': 0,
            'pose_max_velocity': 0,
            'pose_range_of_motion': 0,
            'pose_smoothness': 0
        }
    
    # Extract key joint positions over time
    # This is simplified - actual implementation depends on your training features
    velocities = []
    positions = []
    
    for frame in frames:
        if 'keypoints' in frame and len(frame['keypoints']) > 0:
            kpts = np.array(frame['keypoints'][0])  # First person
            positions.append(kpts[:, :2])  # x, y coordinates
    
    if len(positions) > 1:
        positions = np.array(positions)
        # Calculate velocities between frames
        diffs = np.diff(positions, axis=0)
        velocities = np.linalg.norm(diffs, axis=2).mean(axis=1)
        
        features = {
            'pose_mean_velocity': float(np.mean(velocities)),
            'pose_max_velocity': float(np.max(velocities)),
            'pose_range_of_motion': float(np.ptp(positions)),
            'pose_smoothness': float(np.std(velocities))
        }
    else:
        features = {
            'pose_mean_velocity': 0,
            'pose_max_velocity': 0,
            'pose_range_of_motion': 0,
            'pose_smoothness': 0
        }
    
    return features


def extract_bar_speed_features(video_path, lift_type="Squat"):
    """
    Extract bar velocity features from video.
    
    Args:
        video_path: Path to video file
        lift_type: Type of lift
        
    Returns:
        dict: Bar speed features (avg velocity, peak velocity, etc.)
    """
    # Import bar tracking module
    from Bar_Tracking import barspeed_to_excel as bst
    
    # Create temp output directory
    temp_output_dir = os.path.join(os.path.dirname(video_path), "temp_bar_tracking")
    os.makedirs(temp_output_dir, exist_ok=True)
    
    try:
        # Process video with bar tracker
        bst.process_video(
            video_path=video_path,
            movement_name=lift_type,
            output_dir=temp_output_dir
        )
        
        # Load results from Excel/CSV
        output_file = os.path.join(temp_output_dir, "barSpeed.xlsx")
        
        if os.path.exists(output_file):
            df = pd.read_excel(output_file)
            
            # Extract aggregate features
            features = {
                'bar_speed_mean': float(df['avg_velocity'].mean()) if 'avg_velocity' in df else 0,
                'bar_speed_max': float(df['peak_velocity'].max()) if 'peak_velocity' in df else 0,
                'bar_speed_min': float(df['min_velocity'].min()) if 'min_velocity' in df else 0,
                'bar_speed_range': float(df['peak_velocity'].max() - df['min_velocity'].min()) if 'peak_velocity' in df and 'min_velocity' in df else 0
            }
        else:
            # Fallback if tracking failed
            features = {
                'bar_speed_mean': 0,
                'bar_speed_max': 0,
                'bar_speed_min': 0,
                'bar_speed_range': 0
            }
        
        return features
        
    except Exception as e:
        print(f"Warning: Bar tracking failed: {e}")
        return {
            'bar_speed_mean': 0,
            'bar_speed_max': 0,
            'bar_speed_min': 0,
            'bar_speed_range': 0
        }
    
    finally:
        # Cleanup temp directory
        if os.path.exists(temp_output_dir):
            import shutil
            shutil.rmtree(temp_output_dir)


# Simple test
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        video = sys.argv[1]
        print("Testing MMPose extraction...")
        mmpose_features = extract_mmpose_features(video, "Squat")
        print(f"MMPose features: {mmpose_features}")
        
        print("\nTesting Bar Speed extraction...")
        bar_features = extract_bar_speed_features(video, "Squat")
        print(f"Bar Speed features: {bar_features}")
