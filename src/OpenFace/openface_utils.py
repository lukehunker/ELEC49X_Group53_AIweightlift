"""
Shared utilities for OpenFace testing modules.
Provides common functions for video processing, OpenFace execution, and visualization.

Optimizations for Accuracy:
- High-quality OpenFace settings (-wild, -3Dfp, -gaze, -pdmparams)
- Proper error handling and validation
- Efficient CSV caching to avoid redundant processing
- Multi-parameter extraction for comprehensive feature analysis
"""

import os
import cv2
import subprocess
import pandas as pd
import numpy as np
import glob
import random
import sys

# Import pose-guided face detection (optional - only if MediaPipe available)
try:
    from pose_guided_face_detection import PoseGuidedFaceDetector
    POSE_GUIDANCE_AVAILABLE = True
except ImportError:
    POSE_GUIDANCE_AVAILABLE = False
    print("Note: Pose guidance not available (MediaPipe not installed)")

# =========================================
# CONFIGURATION
# =========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Now in src/OpenFace/, need to go up 2 levels to reach repo root
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# Paths
OPENFACE_BIN = os.path.join(REPO_ROOT, "OpenFace", "build", "bin", "FeatureExtraction")
VIDEOS_DIR = os.path.join(REPO_ROOT, "lifting_videos")
OUTPUT_DIR = os.path.join(REPO_ROOT, "output", "openface")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# OpenFace landmark indices
LANDMARK_GROUPS = {
    'left_eyebrow': list(range(17, 22)),
    'right_eyebrow': list(range(22, 27)),
    'nose_bridge': list(range(27, 31)),
    'nose_tip': [30, 31, 32, 33, 34, 35],
    'left_eye': list(range(36, 42)),
    'right_eye': list(range(42, 48)),
    'mouth_outer': list(range(48, 60)),
    'mouth_inner': list(range(60, 68)),
    'jawline': list(range(0, 17))
}


# =========================================
# VIDEO PROCESSING
# =========================================
def get_video_metadata(video_path):
    """Extract video metadata including resolution, FPS, and frame count."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    metadata = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'resolution': f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
    }
    cap.release()
    return metadata


# =========================================
# OPENFACE PROCESSING
# =========================================
def run_openface(video_path, force_rerun=False, high_quality=True, use_pose_guidance=False):
    """
    Run OpenFace on a video and return the path to the output CSV.
    
    Args:
        video_path: Path to the video file
        force_rerun: If True, rerun even if CSV exists
        high_quality: If True, use higher quality settings (slower but more accurate)
        use_pose_guidance: If True, use MediaPipe Pose to crop video to main person's face first
                          (helps with multi-person videos and background faces)
        
    Returns:
        Path to the CSV file containing OpenFace results
    """
    # Apply pose guidance if requested
    original_video_path = video_path
    if use_pose_guidance and POSE_GUIDANCE_AVAILABLE:
        print("Using pose-guided face detection (cropping to main person's face)...")
        video_path = preprocess_video_with_pose_guidance(video_path)
        if video_path is None:
            print("Warning: Pose guidance failed, using original video")
            video_path = original_video_path
    elif use_pose_guidance and not POSE_GUIDANCE_AVAILABLE:
        print("Warning: Pose guidance requested but MediaPipe not available, using original video")
    
    video_name = os.path.splitext(os.path.basename(original_video_path))[0]  # Use original name for CSV
    
    # Check if CSV already exists
    csv_pattern = os.path.join(OUTPUT_DIR, f"{video_name}*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if csv_files and not force_rerun:
        print(f"Using existing OpenFace output: {os.path.basename(csv_files[0])}")
        return csv_files[0]
    
    # Run OpenFace with optimized parameters
    print(f"Running OpenFace on {video_name}...")
    if not os.path.isfile(OPENFACE_BIN):
        raise FileNotFoundError(f"OpenFace binary NOT found at: {OPENFACE_BIN}")
    
    # Build command with accuracy optimizations
    cmd = [
        OPENFACE_BIN,
        "-f", video_path,
        "-out_dir", OUTPUT_DIR,
        "-2Dfp",  # 2D facial landmarks
        "-3Dfp",  # 3D facial landmarks (better for head pose)
        "-pdmparams",  # PDM parameters (shape variations)
        "-aus",  # Action Units
        "-gaze",  # Gaze direction (useful for strain detection)
    ]
    
    if high_quality:
        # High accuracy settings
        cmd.extend([
            "-wild",  # Trained for in-the-wild conditions (better robustness)
            "-q",  # Quiet mode (less console spam)
        ])
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Warning: OpenFace returned code {result.returncode}")
    except subprocess.CalledProcessError as e:
        print(f"Error output: {e.stderr if e.stderr else 'No error message'}")
        raise RuntimeError(f"OpenFace processing failed: {e}")
    
    csv_files = glob.glob(csv_pattern)
    if not csv_files:
        raise RuntimeError("OpenFace finished but did not generate a CSV file.")
    
    print(f"✓ OpenFace processing complete: {os.path.basename(csv_files[0])}")
    return csv_files[0]


def load_landmark_data(csv_path, success_only=True, sample_fps=None, columns_only=None):
    """
    Load and parse landmark data from OpenFace CSV with optimization options.
    
    Args:
        csv_path: Path to OpenFace CSV file
        success_only: If True, return only successfully detected frames
        sample_fps: If provided, downsample to this FPS (e.g., 10 instead of 30)
        columns_only: List of specific columns to load (None = all columns)
                     Use 'rpe_minimal' for RPE prediction essentials only
        
    Returns:
        DataFrame with landmark data
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Define minimal columns needed for RPE prediction
    RPE_MINIMAL_COLUMNS = [
        'frame', 'timestamp', 'success', 'confidence',
        # Key exertion Action Units (intensity)
        'AU04_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r',
        'AU12_r', 'AU17_r', 'AU20_r', 'AU25_r', 'AU26_r',
        # Key landmarks for stability (just a few, not all 68x2=136)
        'x_28', 'y_28',  # Nose tip
        'x_36', 'y_36',  # Left eye corner
        'x_45', 'y_45',  # Right eye corner
        # Head pose (useful for movement analysis)
        'pose_Rx', 'pose_Ry', 'pose_Rz',
    ]
    
    # Determine which columns to load
    if columns_only == 'rpe_minimal':
        usecols = RPE_MINIMAL_COLUMNS
    elif isinstance(columns_only, list):
        usecols = columns_only
    else:
        usecols = None  # Load all columns
    
    try:
        # Load CSV with optional column filtering
        if usecols:
            # First read just header to check which columns exist
            df_check = pd.read_csv(csv_path, nrows=0)
            available_cols = [col.strip() for col in df_check.columns]
            # Filter to only columns that actually exist
            usecols_filtered = [col for col in usecols if col in available_cols]
            if len(usecols_filtered) < len(usecols):
                missing = set(usecols) - set(usecols_filtered)
                print(f"Warning: {len(missing)} requested columns not found: {list(missing)[:5]}...")
            df = pd.read_csv(csv_path, usecols=usecols_filtered if usecols_filtered else None)
        else:
            df = pd.read_csv(csv_path)
        
        df.columns = [col.strip() for col in df.columns]
    except Exception as e:
        raise ValueError(f"Failed to read CSV: {e}")
    
    if 'success' not in df.columns:
        print("Warning: 'success' column not found, assuming all frames valid")
        return df
    
    if success_only:
        df = df[df['success'] == 1].copy()
        if len(df) == 0:
            print("Warning: No successful face detections in video")
            return df
    
    # Downsample frames if requested
    if sample_fps is not None and 'timestamp' in df.columns and len(df) > 0:
        original_fps = 1.0 / df['timestamp'].diff().median() if len(df) > 1 else 30
        if original_fps > sample_fps:
            # Keep every Nth frame
            sample_interval = int(original_fps / sample_fps)
            df = df.iloc[::sample_interval].reset_index(drop=True)
            print(f"  Downsampled from ~{original_fps:.1f} FPS to ~{sample_fps} FPS ({len(df)} frames)")
    
    return df


def check_openface_binary():
    """Check if OpenFace binary exists and is accessible."""
    if not os.path.isfile(OPENFACE_BIN):
        print(f"\n{'!'*80}")
        print(f"ERROR: OpenFace binary NOT found at:")
        print(f"{OPENFACE_BIN}")
        print(f"Please ensure OpenFace is built and the path is correct.")
        print(f"{'!'*80}\n")
        return False
    return True


def preprocess_video_with_pose_guidance(video_path, cache_dir=None):
    """
    Preprocess video using pose-guided cropping to focus on main person's face.
    
    This solves the problem of OpenFace detecting background faces instead of
    the exercising person by using MediaPipe Pose body keypoints to locate the correct face.
    
    Args:
        video_path: Path to input video
        cache_dir: Directory to save cropped videos (default: OUTPUT_DIR)
    
    Returns:
        Path to cropped video, or None if failed
    """
    if not POSE_GUIDANCE_AVAILABLE:
        return None
    
    if cache_dir is None:
        cache_dir = OUTPUT_DIR
    
    # Generate output path
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(cache_dir, f"{video_name}_pose_guided.mp4")
    
    # Check if already processed
    if os.path.exists(output_path):
        print(f"Using cached pose-guided video: {os.path.basename(output_path)}")
        return output_path
    
    # Create cropped video
    try:
        detector = PoseGuidedFaceDetector(verbose=False)
        result_path = detector.create_cropped_video(
            video_path, 
            output_path,
            fallback_to_full=True  # Use full frame if pose detection fails
        )
        return result_path
    except Exception as e:
        print(f"Warning: Pose-guided preprocessing failed: {e}")
        return None
