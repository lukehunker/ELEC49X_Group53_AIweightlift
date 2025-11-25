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


def classify_resolution(width, height):
    """Classify video resolution quality."""
    pixels = width * height
    if pixels >= 1920 * 1080:
        return "High (1080p+)"
    elif pixels >= 1280 * 720:
        return "Medium (720p)"
    elif pixels >= 640 * 480:
        return "Low (480p)"
    else:
        return "Very Low (<480p)"


def find_videos(pattern_list, max_per_exercise=2):
    """
    Find a small subset of videos for testing, with some from each exercise.
    
    Args:
        pattern_list: List of file patterns to search for
        max_per_exercise: Maximum videos to return per exercise type (default: 2)
    
    Returns:
        List of video file paths
    """
    video_files = []
    # Search in Augmented subdirectories for each exercise type
    target_dirs = [
        os.path.join('Augmented', 'Deadlift'),
        os.path.join('Augmented', 'Squat'),
        os.path.join('Augmented', 'Bench Press')
    ]
    
    # Collect videos per exercise, limiting to max_per_exercise each
    for subdir in target_dirs:
        exercise_videos = []
        for pattern in pattern_list:
            found = glob.glob(os.path.join(VIDEOS_DIR, subdir, pattern))
            exercise_videos.extend(found)
        
        # Remove duplicates and limit
        exercise_videos = sorted(set(exercise_videos))
        video_files.extend(exercise_videos[:max_per_exercise])
    
    return video_files


# =========================================
# OPENFACE PROCESSING
# =========================================
def run_openface(video_path, force_rerun=False, high_quality=True):
    """
    Run OpenFace on a video and return the path to the output CSV.
    
    Args:
        video_path: Path to the video file
        force_rerun: If True, rerun even if CSV exists
        high_quality: If True, use higher quality settings (slower but more accurate)
        
    Returns:
        Path to the CSV file containing OpenFace results
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
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


def load_landmark_data(csv_path, success_only=True):
    """
    Load and parse landmark data from OpenFace CSV.
    
    Args:
        csv_path: Path to OpenFace CSV file
        success_only: If True, return only successfully detected frames
        
    Returns:
        DataFrame with landmark data
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    try:
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


# =========================================
# VISUALIZATION HELPERS
# =========================================
def draw_landmarks(frame, xs, ys, show_points=True, show_box=True, color_by_group=False):
    """
    Draw facial landmarks on a frame.
    
    Args:
        frame: The video frame to draw on
        xs: X coordinates of landmarks (array of 68 values)
        ys: Y coordinates of landmarks (array of 68 values)
        show_points: Whether to draw landmark points
        show_box: Whether to draw bounding box
        color_by_group: Whether to color landmarks by facial region
    """
    if show_box:
        # Calculate bounding box with padding
        x_min, x_max = int(np.min(xs)), int(np.max(xs))
        y_min, y_max = int(np.min(ys)), int(np.max(ys))
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(frame.shape[1], x_max + padding)
        y_max = min(frame.shape[0], y_max + padding)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
    
    if show_points:
        if color_by_group:
            colors = {
                'left_eyebrow': (255, 200, 0),
                'right_eyebrow': (255, 200, 0),
                'nose_bridge': (0, 255, 0),
                'nose_tip': (0, 255, 0),
                'left_eye': (255, 0, 255),
                'right_eye': (255, 0, 255),
                'mouth_outer': (0, 255, 255),
                'mouth_inner': (0, 200, 200),
                'jawline': (255, 100, 100)
            }
            for group_name, indices in LANDMARK_GROUPS.items():
                color = colors.get(group_name, (255, 255, 255))
                for idx in indices:
                    if idx < len(xs):
                        x, y = int(xs[idx]), int(ys[idx])
                        cv2.circle(frame, (x, y), 3, color, -1)
                        cv2.circle(frame, (x, y), 4, (255, 255, 255), 1)
        else:
            # Simple yellow dots
            for i in range(min(68, len(xs))):
                x, y = int(xs[i]), int(ys[i])
                cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)


def add_text_overlay(frame, lines, bg_color=(0, 0, 0), bg_alpha=0.6):
    """
    Add text overlay with semi-transparent background.
    
    Args:
        frame: The video frame
        lines: List of (text, y_position) tuples
        bg_color: Background color (BGR)
        bg_alpha: Background transparency (0-1)
    """
    if not lines:
        return
    
    # Calculate background size
    max_y = max(y for _, y in lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 20), (450, max_y + 15), bg_color, -1)
    cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)
    
    # Draw text
    for text, y_pos in lines:
        cv2.putText(frame, text, (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


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
