"""
MMPose Body Posture Feature Extraction - Single Video API

This module provides a simple function to extract body posture features from a single video.
Uses MMPose to track body keypoints and compute the D-metric (postural deviation).

The D-metric measures the average deviation in body posture between the first and last rep,
indicating fatigue-related form breakdown.

Usage:
    from MMPose.extract_features import extract_posture_features
    
    features = extract_posture_features('squat_video.mp4', movement='Squat')
    print(features['d_value'])  # Postural deviation metric
    print(features['rep_count'])  # Number of reps detected
"""

import os
import sys
import json
import tempfile
import subprocess
import numpy as np
from pathlib import Path


def extract_posture_features(video_path, movement=None, output_dir=None, device="cuda:0"):
    """
    Extract body posture features from a single video.
    
    This performs two steps:
    1. Extract keypoints from video using MMPose
    2. Compute D-metric (postural deviation) from keypoints
    
    Args:
        video_path (str): Path to the video file
        movement (str, optional): Movement type ('Squat', 'Bench Press', 'Deadlift')
                                 If None, will attempt to detect from filename
        output_dir (str, optional): Directory for intermediate outputs. 
                                   If None, uses a temporary directory
        device (str): Device to run inference on (default: 'cuda:0', can use 'cpu')
    
    Returns:
        dict: Feature dictionary with keys:
            - video_name: Name of the video file
            - movement: Movement type
            - d_value: D-metric (postural deviation between first and last rep)
                      Higher values indicate more form breakdown
            - rep_count: Number of valid repetitions detected
            - first_rep_frames: Tuple (start, end) frame indices for first rep
            - last_rep_frames: Tuple (start, end) frame indices for last rep
            - keypoints_file: Path to the keypoints JSON file (if output_dir provided)
            
        Returns None if processing fails or fewer than 2 reps detected.
    
    Example:
        >>> features = extract_posture_features('squat_video.mp4')
        >>> print(f"D-metric: {features['d_value']:.3f}")
        >>> print(f"Detected {features['rep_count']} reps")
    
    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If movement type is invalid or cannot be detected
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
    movement_normalized = movement.strip().title()
    if movement_normalized not in ["Squat", "Bench Press", "Deadlift"]:
        raise ValueError(
            f"Invalid movement '{movement}'. "
            "Must be 'Squat', 'Bench Press', or 'Deadlift'"
        )
    
    # Map to internal names
    movement_map = {
        "Squat": "squat",
        "Bench Press": "bench",
        "Deadlift": "deadlift"
    }
    movement_internal = movement_map[movement_normalized]
    
    # Use temp directory if no output dir specified
    cleanup_temp = False
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="mmpose_")
        cleanup_temp = True
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    keypoints_file = os.path.join(output_dir, f"{video_name}_keypoints.json")
    
    try:
        # Get the directory where this script lives (src/MMPose/)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Step 1: Extract keypoints using MMPose
        print(f"[1/2] Extracting keypoints from video...")
        extraction_script = os.path.join(script_dir, f"feature_extraction_{movement_internal}.py")
        
        if not os.path.exists(extraction_script):
            raise FileNotFoundError(
                f"Extraction script not found: {extraction_script}"
            )
        
        # Run extraction script
        # Use absolute paths to handle spaces in filenames
        extraction_cmd = [
            sys.executable,
            os.path.abspath(extraction_script),
            os.path.abspath(video_path),
            "--out", os.path.abspath(keypoints_file),
            "--device", device
        ]
        
        result = subprocess.run(
            extraction_cmd,
            cwd=script_dir,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode != 0:
            print(f"Extraction failed: {result.stderr}")
            return None
        
        if not os.path.exists(keypoints_file):
            print("Extraction completed but keypoints file not found")
            return None
        
        print(f"  ✓ Keypoints saved to: {os.path.basename(keypoints_file)}")
        
        # Step 2: Compute D-metric from keypoints
        print(f"[2/2] Computing D-metric (postural deviation)...")
        preprocess_script = os.path.join(script_dir, f"preprocess_{movement_internal}.py")
        
        if not os.path.exists(preprocess_script):
            raise FileNotFoundError(
                f"Preprocessing script not found: {preprocess_script}"
            )
        
        # Run preprocessing script
        preprocess_cmd = [
            sys.executable,
            preprocess_script,
            keypoints_file
        ]
        
        result = subprocess.run(
            preprocess_cmd,
            cwd=script_dir,
            capture_output=True,
            text=True,
            timeout=60  # 1 minute timeout
        )
        
        # Parse output to extract D-metric
        output_text = result.stdout
        
        # Look for results in output
        d_value = None
        first_rep = None
        last_rep = None
        rep_count = None
        
        lines = output_text.split('\n')
        for i, line in enumerate(lines):
            if "First rep:" in line:
                # Parse "First rep: (123, 456)"
                import re
                match = re.search(r'First rep:\s*\((\d+),\s*(\d+)\)', line)
                if match:
                    first_rep = (int(match.group(1)), int(match.group(2)))
            
            elif "Last rep :" in line or "Last rep:" in line:
                # Parse "Last rep : (789, 890)"
                import re
                match = re.search(r'Last rep\s*:\s*\((\d+),\s*(\d+)\)', line)
                if match:
                    last_rep = (int(match.group(1)), int(match.group(2)))
            
            elif "D value" in line:
                # Parse "D value  : 1.234567"
                import re
                match = re.search(r'D value\s*:\s*([\d.]+)', line)
                if match:
                    d_value = float(match.group(1))
            
            elif "KEEP Rep" in line:
                # Count valid reps
                if rep_count is None:
                    rep_count = 1
                else:
                    rep_count += 1
        
        # Check if we got insufficient reps
        if "SKIP:" in output_text and "valid rep" in output_text:
            print(f"  ⚠ Insufficient reps detected (need at least 2)")
            return None
        
        if d_value is None or first_rep is None or last_rep is None:
            print(f"  ⚠ Could not parse D-metric from output")
            if result.returncode != 0:
                print(f"  Error: {result.stderr}")
            return None
        
        print(f"  ✓ D-metric computed: {d_value:.4f}")
        
        # Build result dictionary
        features = {
            'video_name': os.path.basename(video_path),
            'movement': movement_normalized,
            'd_value': d_value,
            'rep_count': rep_count or 2,  # At least 2 if we got here
            'first_rep_frames': first_rep,
            'last_rep_frames': last_rep,
        }
        
        # Keep keypoints file path if not using temp dir
        if not cleanup_temp:
            features['keypoints_file'] = keypoints_file
        
        return features
        
    except subprocess.TimeoutExpired:
        print("Processing timed out")
        return None
    except Exception as e:
        print(f"Error extracting posture features: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Clean up temp directory if we created it
        if cleanup_temp:
            try:
                import shutil
                shutil.rmtree(output_dir, ignore_errors=True)
            except:
                pass


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract body posture features from a video")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("--movement", choices=['Squat', 'Bench Press', 'Deadlift'],
                       help="Movement type (optional, auto-detected from filename)")
    parser.add_argument("--output-dir", help="Directory for output files (optional)")
    parser.add_argument("--device", default="cuda:0",
                       help="Device to run inference on (default: cuda:0)")
    
    args = parser.parse_args()
    
    print(f"\nExtracting body posture features from: {args.video_path}")
    if args.movement:
        print(f"Movement type: {args.movement}")
    else:
        print("Auto-detecting movement type from filename...")
    
    features = extract_posture_features(
        args.video_path,
        movement=args.movement,
        output_dir=args.output_dir,
        device=args.device
    )
    
    if features:
        print("\n" + "="*60)
        print("BODY POSTURE FEATURES")
        print("="*60)
        for key, value in features.items():
            if key == 'keypoints_file':
                print(f"  {key}: {os.path.basename(value)}")
            else:
                print(f"  {key}: {value}")
        
        print("\n" + "="*60)
        print("D-METRIC INTERPRETATION")
        print("="*60)
        d_val = features['d_value']
        if d_val < 0.5:
            level = "EXCELLENT - Minimal form breakdown"
        elif d_val < 1.0:
            level = "GOOD - Minor form changes"
        elif d_val < 1.5:
            level = "MODERATE - Noticeable form deviation"
        else:
            level = "HIGH - Significant form breakdown"
        print(f"  D = {d_val:.3f}: {level}")
        
    else:
        print("\nFailed to extract features")
        sys.exit(1)
