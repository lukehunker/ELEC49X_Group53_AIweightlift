"""
Standalone script to test MediaPipe body cropping.
Tests the preprocessing step without running the full pipeline.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("ERROR: MediaPipe not available. Install with: pip install mediapipe")
    exit(1)


def crop_video_to_body(video_path, output_path=None, padding_percent=0.25, show_preview=False):
    """
    Crop video to focus on the person's body using MediaPipe Pose.
    Removes background, other people, and gym equipment.
    
    Args:
        video_path: Path to input video
        output_path: Path to save cropped video (optional)
        padding_percent: Extra space around body bbox (0.25 = 25% padding)
        show_preview: If True, display side-by-side comparison
    
    Returns:
        Path to cropped video
    """
    video_name = Path(video_path).stem
    if output_path is None:
        # Save to project-level output/body_cropped/
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        output_dir = project_root / 'output' / 'body_cropped'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"{video_name}_body_cropped.mp4")
    
    # Check if already processed
    if os.path.exists(output_path) and not show_preview:
        print(f"✓ Using cached body-cropped video: {os.path.basename(output_path)}")
        return output_path
    
    print(f"\n{'='*60}")
    print(f"Cropping video to body region: {os.path.basename(video_path)}")
    print(f"{'='*60}")
    
    # Try different backends for problematic codecs (AV1, HEVC, etc.)
    backends_to_try = [
        (cv2.CAP_FFMPEG, "FFmpeg"),
        (cv2.CAP_ANY, "Auto"),
        (cv2.CAP_GSTREAMER, "GStreamer") if hasattr(cv2, 'CAP_GSTREAMER') else None
    ]
    backends_to_try = [b for b in backends_to_try if b is not None]
    
    cap = None
    backend_used = None
    for backend_id, backend_name in backends_to_try:
        test_cap = cv2.VideoCapture(video_path, backend_id)
        if test_cap.isOpened():
            # Test if we can actually read frames
            ret, test_frame = test_cap.read()
            if ret and test_frame is not None:
                test_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
                cap = test_cap
                backend_used = backend_name
                break
            test_cap.release()
    
    if cap is None or not cap.isOpened():
        print(f"\n⚠️  ERROR: Cannot read video with OpenCV (likely AV1 codec issue)")
        print(f"   Video: {os.path.basename(video_path)}")
        print(f"\n   Solution: Re-encode to H.264 format:")
        print(f"   ffmpeg -i \"{os.path.basename(video_path)}\" -c:v libx264 -crf 23 \"{video_name}_h264.mp4\"")
        print(f"\n   Or batch process all videos:")
        print(f"   for f in *.mp4; do ffmpeg -i \"$f\" -c:v libx264 -crf 23 \"${{f%.mp4}}_h264.mp4\"; done")
        return video_path
    
    print(f"  Using backend: {backend_used}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info:")
    print(f"  Resolution: {original_width}x{original_height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.2f}s")
    
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Pass 1: Detect body bbox in all frames
    print(f"\nPass 1: Detecting body in {total_frames} frames...")
    all_bboxes = []
    frame_idx = 0
    detected_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Get all landmark coordinates
            landmarks = results.pose_landmarks.landmark
            h, w = frame.shape[:2]
            
            # Extract x, y coordinates of all visible landmarks
            xs = [lm.x * w for lm in landmarks if lm.visibility > 0.5]
            ys = [lm.y * h for lm in landmarks if lm.visibility > 0.5]
            
            if len(xs) > 0 and len(ys) > 0:
                # Calculate bbox with padding
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                
                width = x_max - x_min
                height = y_max - y_min
                
                # Add padding (extra at top for face)
                pad_x = width * padding_percent
                pad_y = height * padding_percent
                pad_y_top = height * (padding_percent + 0.15)  # Extra 15% at top for face
                
                x1 = max(0, int(x_min - pad_x))
                y1 = max(0, int(y_min - pad_y_top))  # Extra top padding
                x2 = min(w, int(x_max + pad_x))
                y2 = min(h, int(y_max + pad_y))
                
                all_bboxes.append((x1, y1, x2, y2))
                detected_count += 1
            else:
                all_bboxes.append(None)
        else:
            all_bboxes.append(None)
        
        frame_idx += 1
        if frame_idx % 50 == 0:
            detection_rate = detected_count / frame_idx * 100
            print(f"  Processed {frame_idx}/{total_frames} frames... "
                  f"({detection_rate:.1f}% detected)")
    
    cap.release()
    mp_pose.close()
    
    # Check if any frames were processed
    if not all_bboxes:
        print("\n⚠️  ERROR: No frames were read from video!")
        print("   Possible causes:")
        print("   - Unsupported video codec (try re-encoding with H.264)")
        print("   - Corrupted video file")
        print("   - OpenCV can't decode this format")
        print("\n   Try re-encoding with: ffmpeg -i input.mp4 -c:v libx264 output.mp4")
        return video_path
    
    # Get valid bboxes and compute stable crop region
    valid_bboxes = [b for b in all_bboxes if b is not None]
    detection_rate = len(valid_bboxes) / len(all_bboxes) * 100
    
    print(f"\nDetection Results:")
    print(f"  Total frames: {len(all_bboxes)}")
    print(f"  Detected frames: {len(valid_bboxes)}")
    print(f"  Detection rate: {detection_rate:.1f}%")
    
    if not valid_bboxes:
        print("\n⚠️  WARNING: No body detected in any frame!")
        print("   Using original video without cropping.")
        return video_path
    
    if detection_rate < 50:
        print("\n⚠️  WARNING: Low detection rate (< 50%)")
        print("   Consider:")
        print("   - Using better lighting")
        print("   - Ensuring person is fully visible")
        print("   - Reducing camera shake")
    
    # Use median bbox to get stable crop region (reduces jitter)
    x1s, y1s, x2s, y2s = zip(*valid_bboxes)
    crop_x1 = int(np.median(x1s))
    crop_y1 = int(np.median(y1s))
    crop_x2 = int(np.median(x2s))
    crop_y2 = int(np.median(y2s))
    
    crop_width = crop_x2 - crop_x1
    crop_height = crop_y2 - crop_y1
    
    width_reduction = (1 - crop_width/original_width) * 100
    height_reduction = (1 - crop_height/original_height) * 100
    
    print(f"\nCrop Region:")
    print(f"  Original: {original_width}x{original_height}")
    print(f"  Cropped:  {crop_width}x{crop_height}")
    print(f"  Reduction: {width_reduction:.1f}% width, {height_reduction:.1f}% height")
    print(f"  Position: ({crop_x1}, {crop_y1}) to ({crop_x2}, {crop_y2})")
    
    # Pass 2: Crop all frames to stable region
    if not show_preview:
        print(f"\nPass 2: Cropping {total_frames} frames...")
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (crop_width, crop_height))
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Crop to stable region
            cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            out.write(cropped)
            
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  Cropped {frame_idx}/{total_frames} frames...")
        
        cap.release()
        out.release()
        
        print(f"\n{'='*60}")
        print(f"✓ Body-cropped video saved:")
        print(f"  {output_path}")
        print(f"{'='*60}\n")
        
        return output_path
    
    else:
        # Preview mode: show side-by-side comparison
        print(f"\n{'='*60}")
        print("PREVIEW MODE - Showing side-by-side comparison")
        print("Press 'q' to quit, 'space' to pause/resume")
        print(f"{'='*60}\n")
        
        cap = cv2.VideoCapture(video_path)
        paused = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
            
            # Crop frame
            cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # Resize for display (fit side by side)
            display_height = 480
            display_width_orig = int(original_width * display_height / original_height)
            display_width_crop = int(crop_width * display_height / crop_height)
            
            frame_display = cv2.resize(frame, (display_width_orig, display_height))
            cropped_display = cv2.resize(cropped, (display_width_crop, display_height))
            
            # Add labels
            cv2.putText(frame_display, "ORIGINAL", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(cropped_display, "BODY CROP", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw crop region on original
            scale_x = display_width_orig / original_width
            scale_y = display_height / original_height
            pt1 = (int(crop_x1 * scale_x), int(crop_y1 * scale_y))
            pt2 = (int(crop_x2 * scale_x), int(crop_y2 * scale_y))
            cv2.rectangle(frame_display, pt1, pt2, (0, 255, 0), 2)
            
            # Combine side by side
            combined = np.hstack([frame_display, cropped_display])
            
            cv2.imshow('Body Cropping Preview', combined)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
        
        cap.release()
        cv2.destroyAllWindows()
        
        return None


def test_single_video(video_path, preview=False):
    """Test body cropping on a single video."""
    if not os.path.exists(video_path):
        print(f"ERROR: Video not found: {video_path}")
        return
    
    output = crop_video_to_body(video_path, show_preview=preview)
    
    if output and not preview:
        # Check output file size
        input_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
        output_size = os.path.getsize(output) / (1024 * 1024)  # MB
        
        print(f"File sizes:")
        print(f"  Input:  {input_size:.2f} MB")
        print(f"  Output: {output_size:.2f} MB")
        print(f"  Ratio:  {output_size/input_size*100:.1f}%")


def test_directory(video_dir, recursive=True, preview=False):
    """Test body cropping on all videos in a directory."""
    video_extensions = ['.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV']
    
    if recursive:
        video_files = []
        for ext in video_extensions:
            video_files.extend(Path(video_dir).rglob(f'*{ext}'))
    else:
        video_files = []
        for ext in video_extensions:
            video_files.extend(Path(video_dir).glob(f'*{ext}'))
    
    video_files = sorted(video_files)
    
    if not video_files:
        print(f"No video files found in {video_dir}")
        return
    
    print(f"\nFound {len(video_files)} videos")
    print(f"{'='*60}\n")
    
    success_count = 0
    for i, video_path in enumerate(video_files, 1):
        print(f"[{i}/{len(video_files)}] Processing: {video_path.name}")
        
        try:
            output = crop_video_to_body(str(video_path), show_preview=preview)
            if output:
                success_count += 1
        except Exception as e:
            print(f"ERROR processing {video_path.name}: {e}\n")
        
        if preview and i < len(video_files):
            response = input("\nContinue to next video? (y/n): ")
            if response.lower() != 'y':
                break
    
    print(f"\n{'='*60}")
    print(f"Completed: {success_count}/{len(video_files)} videos successfully cropped")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MediaPipe body cropping")
    parser.add_argument('input', help='Video file or directory path')
    parser.add_argument('--recursive', '-r', action='store_true',
                       help='Search subdirectories for videos')
    parser.add_argument('--preview', '-p', action='store_true',
                       help='Show side-by-side preview (no output file)')
    parser.add_argument('--padding', type=float, default=0.15,
                       help='Padding around body (default: 0.15 = 15%%)')
    
    args = parser.parse_args()
    
    if not MEDIAPIPE_AVAILABLE:
        print("\nERROR: MediaPipe not installed")
        print("Install with: pip install mediapipe")
        exit(1)
    
    # Test single video or directory
    if os.path.isfile(args.input):
        test_single_video(args.input, preview=args.preview)
    elif os.path.isdir(args.input):
        test_directory(args.input, recursive=args.recursive, preview=args.preview)
    else:
        print(f"ERROR: Path not found: {args.input}")
