"""
Bar Speed Feature Extraction - Single Video API

This module provides a simple function to extract bar speed features from a single video.
Returns a dictionary with rep count, duration, and fatigue metrics.

Usage:
    from Bar_Tracking.extract_features import extract_bar_speed
    
    features = extract_bar_speed('video.mp4', movement='Squat')
    print(features)
    # {'video_name': 'video.mp4', 'rep_count': 5, 'first_rep_duration_s': 2.1, ...}
"""

import os
import tempfile
import shutil
from pathlib import Path
from . import barspeed_to_excel as bst


def extract_bar_speed(video_path, movement=None, output_dir=None):
    """
    Extract bar speed features from a single video.
    
    Args:
        video_path (str): Path to the video file
        movement (str, optional): Movement type ('Squat', 'Bench Press', 'Deadlift')
                                 If None, will attempt to detect from filename
        output_dir (str, optional): Directory for intermediate outputs. 
                                   If None, uses a temporary directory
    
    Returns:
        dict: Feature dictionary with keys:
            - video_name: Name of the video file
            - rep_count: Number of repetitions detected
            - first_rep_duration_s: Duration of first rep in seconds
            - last_rep_duration_s: Duration of last rep in seconds
            - fatigue_s: Fatigue metric (last rep - first rep duration)
            
        Returns None if processing fails.
    
    Example:
        >>> features = extract_bar_speed('squat_video.mp4')
        >>> print(f"Detected {features['rep_count']} reps")
        >>> print(f"Fatigue: {features['fatigue_s']:.2f}s")
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Auto-detect movement type from filename if not provided
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
                "Could not detect movement type from filename. "
                "Please specify movement='Squat', 'Bench Press', or 'Deadlift'"
            )
    
    # Normalize movement name
    movement = movement.strip().title()
    if movement not in ["Squat", "Bench Press", "Deadlift"]:
        raise ValueError(
            f"Invalid movement '{movement}'. "
            "Must be 'Squat', 'Bench Press', or 'Deadlift'"
        )
    
    # Use temp directory if no output dir specified
    cleanup_temp = False
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="barspeed_")
        cleanup_temp = True
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    # Determine visualization output directory
    script_dir = Path(__file__).parent.parent.parent  # Project root
    vis_output_dir = script_dir / "output" / "bar_tracking" / "visualized"
    vis_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Process video using existing function
        features = bst.process_video(video_path, movement, output_dir)
        
        # Move visualization video to centralized output directory
        if features:
            video_basename = os.path.splitext(os.path.basename(video_path))[0]
            vis_filename = f"{movement}__{video_basename}__wrist_vis.mp4"
            vis_src = os.path.join(output_dir, vis_filename)
            vis_dest = vis_output_dir / vis_filename
            
            if os.path.exists(vis_src):
                shutil.move(vis_src, vis_dest)
                features['visualization_path'] = str(vis_dest)
        
        return features
    except Exception as e:
        print(f"Error extracting bar speed features: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Clean up temp directory if we created it
        if cleanup_temp:
            try:
                shutil.rmtree(output_dir, ignore_errors=True)
            except:
                pass
