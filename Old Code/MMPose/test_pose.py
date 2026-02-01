"""
MMPose Body Pose Estimation Test

Processes lifting videos (deadlift and squat) from lifting_videos/Raw/
and creates annotated output videos showing detected body keypoints.

Features:
- Real-time visualization window showing keypoints detected by MMPose on person
- Interactive playback controls (pause, speed up, slow down)
- Automatic video looping - videos repeat until you press Q
- Saves annotated videos with skeleton overlay
- 17 COCO keypoints: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles

Playback Controls:
- SPACE: Pause/Resume
- + or ]: Speed up (decrease delay)
- - or [: Slow down (increase delay)
- R: Restart/Loop video from beginning
- Q: Skip to next video
- ESC: Exit completely

Uses MMPose 0.29.0 with mmcv-full 1.7.0 for compatibility.
"""

import os
import cv2
import glob
import numpy as np
from pathlib import Path

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
VIDEOS_DIR = os.path.join(REPO_ROOT, "lifting_videos", "Raw")
OUTPUT_DIR = os.path.join(REPO_ROOT, "output", "mmpose_test")

# Test parameters - adjust these as needed
TEST_FRAME_INTERVAL = 30  # Process every Nth frame (30 = process 1 frame per second at 30fps)
MAX_FRAMES = 150          # Maximum frames to process per video
SHOW_VISUALIZATION = True # Display video window while processing


def check_dependencies():
    """Check if required packages are installed."""
    print("\nChecking dependencies...")
    
    try:
        import mmpose
        print(f"✓ MMPose version: {mmpose.__version__}")
    except ImportError:
        print("❌ MMPose not installed")
        return False
    
    try:
        import mmcv
        print(f"✓ MMCV version: {mmcv.__version__}")
    except ImportError:
        print("❌ MMCV not installed")
        return False
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    return True


def find_test_videos():
    """Find first video from each exercise type."""
    videos = []
    
    for exercise in ['Deadlift', 'Squat']:
        exercise_dir = os.path.join(VIDEOS_DIR, exercise)
        if os.path.exists(exercise_dir):
            # Find first video file
            for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
                found = glob.glob(os.path.join(exercise_dir, ext))
                if found:
                    videos.append(found[0])
                    print(f"Found: {exercise}/{os.path.basename(found[0])}")
                    break
    
    return videos


def process_video(video_path, output_path):
    """
    Process video with MMPose body pose estimation.
    
    Args:
        video_path: Path to input video file
        output_path: Path to save annotated output video
        
    Returns:
        bool: True if successful, False otherwise
    """
    from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                             vis_pose_result)
    
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(video_path)}")
    print(f"{'='*60}")
    
    # Initialize pose estimation model
    print("Loading MMPose HRNet model...")
    
    # Use local model files (downloaded by install.sh)
    pose_config = os.path.join(SCRIPT_DIR, 'topdown_heatmap_hrnet_w48_coco_256x192.py')
    pose_checkpoint = os.path.join(SCRIPT_DIR, 'hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth')
    
    # Check if model files exist
    if not os.path.exists(pose_config):
        print(f"❌ Model config not found: {pose_config}")
        print("   Run: mim download mmpose --config topdown_heatmap_hrnet_w48_coco_256x192 --dest .")
        return False
    if not os.path.exists(pose_checkpoint):
        print(f"❌ Model checkpoint not found: {pose_checkpoint}")
        print("   Run: mim download mmpose --config topdown_heatmap_hrnet_w48_coco_256x192 --dest .")
        return False
    
    device = 'cuda:0' if __import__('torch').cuda.is_available() else 'cpu'
    
    try:
        pose_model = init_pose_model(pose_config, pose_checkpoint, device=device)
        print(f"✓ Model loaded on {device}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # Get dataset info from config
    dataset = 'TopDownCocoDataset'
    dataset_info = None  # Will use default from model
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
    print(f"Processing every {TEST_FRAME_INTERVAL} frames (max {MAX_FRAMES})...")
    
    # Prepare output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps / TEST_FRAME_INTERVAL, (width, height))
    
    # Store all processed frames for looping
    processed_frames = []
    
    frame_idx = 0
    processed_count = 0
    playback_delay = 500  # milliseconds between frames (500ms = 0.5 seconds)
    
    # Assume person is in full frame (no separate person detection)
    # In production, you'd use MMDetection to detect person bboxes first
    person_bbox = [0, 0, width, height]
    person_results = [{'bbox': np.array(person_bbox)}]
    
    print("\nPlayback Controls:")
    print("  SPACE - Pause/Resume")
    print("  + or ] - Speed up (decrease delay)")
    print("  - or [ - Slow down (increase delay)")
    print("  R - Restart/Loop video")
    print("  Q - Skip to next video")
    print("  ESC - Exit completely")
    
    paused = False
    
    # First pass: Process and save frames
    print("\nProcessing frames...")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every Nth frame
            if frame_idx % TEST_FRAME_INTERVAL == 0 and processed_count < MAX_FRAMES:
                # Run pose estimation (MMPose detects keypoints on person)
                pose_results, returned_outputs = inference_top_down_pose_model(
                    pose_model,
                    frame,
                    person_results,
                    bbox_thr=None,
                    format='xyxy',
                    dataset=dataset,
                    dataset_info=dataset_info,
                    return_heatmap=False,
                    outputs=None
                )
                
                # Visualize pose on frame (MMPose draws skeleton on person)
                vis_frame = vis_pose_result(
                    pose_model,
                    frame,
                    pose_results,
                    dataset=dataset,
                    dataset_info=dataset_info,
                    kpt_score_thr=0.3,  # Only show keypoints with >30% confidence
                    radius=8,           # Larger keypoint dots
                    thickness=4,        # Thicker skeleton lines
                    show=False
                )
                
                # Add frame counter and keypoint info
                num_keypoints = len(pose_results[0]['keypoints']) if pose_results else 0
                cv2.putText(
                    vis_frame,
                    f"Frame: {frame_idx}/{total_frames} | Keypoints: {num_keypoints}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
                
                # Store frame for playback loop
                processed_frames.append(vis_frame.copy())
                
                # Write to output video
                out.write(vis_frame)
                
                processed_count += 1
                if processed_count % 10 == 0:
                    print(f"  Processed {processed_count} frames...")
            
            frame_idx += 1
    finally:
        cap.release()
        out.release()
    
    print(f"\n✓ Output saved: {output_path}")
    print(f"  Processed {processed_count} frames")
    print(f"  Detection: Body keypoints detected by MMPose on person")
    
    if not processed_frames:
        print("  No frames to display")
        return True
    
    # Second pass: Loop playback with controls
    print("\nStarting playback loop (press Q to continue to next video)...")
    
    try:
        while True:
            for idx, vis_frame in enumerate(processed_frames):
                # Add control info overlay
                status = "PAUSED" if paused else f"Delay: {playback_delay}ms"
                frame_with_controls = vis_frame.copy()
                cv2.putText(
                    frame_with_controls,
                    f"{status} | SPACE=Pause +/-=Speed R=Restart Q=Next ESC=Exit",
                    (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
                
                # Display the frame
                cv2.imshow('MMPose - Body Keypoints Detection', frame_with_controls)
                
                # Wait for key press with adjustable delay
                while True:
                    key = cv2.waitKey(playback_delay if not paused else 0) & 0xFF
                    
                    if key == ord(' '):  # Space - toggle pause
                        paused = not paused
                        print(f"  {'Paused' if paused else 'Resumed'}")
                    elif key == ord('r') or key == ord('R'):  # R - restart
                        print("  Restarting video...")
                        break
                    elif key == ord('q') or key == ord('Q'):  # Q - skip video
                        print("\n  Continuing to next video...")
                        cv2.destroyAllWindows()
                        return True
                    elif key == 27:  # ESC - exit completely
                        print("\n  Exiting...")
                        cv2.destroyAllWindows()
                        return False
                    elif key == ord('+') or key == ord('=') or key == ord(']'):  # Speed up
                        playback_delay = max(10, playback_delay - 100)
                        print(f"  Speed increased - delay: {playback_delay}ms")
                    elif key == ord('-') or key == ord('_') or key == ord('['):  # Slow down
                        playback_delay = min(2000, playback_delay + 100)
                        print(f"  Speed decreased - delay: {playback_delay}ms")
                    
                    if not paused or key != 255:  # 255 = no key pressed
                        break
                
                # Check if restart was pressed
                if key == ord('r') or key == ord('R'):
                    break
            
            # Loop continues unless user presses Q or ESC
    
    finally:
        cv2.destroyAllWindows()
    
    return True


def main():
    """Main function - processes videos and creates annotated outputs."""
    print("="*60)
    print("MMPose Body Pose Estimation Test")
    print("="*60)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Missing dependencies. Please run: bash install.sh")
        return
    
    # Find test videos
    print(f"\nSearching for videos in: {VIDEOS_DIR}")
    videos = find_test_videos()
    
    if not videos:
        print(f"\n❌ No videos found in {VIDEOS_DIR}")
        print("   Expected structure: lifting_videos/Raw/Deadlift/*.mp4")
        print("   Expected structure: lifting_videos/Raw/Squat/*.mp4")
        return
    
    print(f"\nFound {len(videos)} test video(s)")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process each video
    success_count = 0
    for video_path in videos:
        video_name = Path(video_path).stem
        output_path = os.path.join(OUTPUT_DIR, f"{video_name}_pose.mp4")
        
        if process_video(video_path, output_path):
            success_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"✓ Test Complete!")
    print(f"  Successfully processed: {success_count}/{len(videos)} videos")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
