"""
MMDetection + MMPose Hybrid Barbell Tracking

Combines body pose estimation with barbell detection for better accuracy.
Uses MMPose to detect the person, then looks for the barbell near their hands.

Features:
- Real-time visualization showing both body keypoints and barbell
- Body-guided barbell detection (looks near hands/shoulders)
- Bar path tracking across frames
- Interactive playback controls
- Automatic video looping
- Saves annotated videos with both pose and barbell tracking

Playback Controls:
- SPACE: Pause/Resume
- + or ]: Speed up
- - or [: Slow down
- R: Restart video
- Q: Skip to next video
- ESC: Exit completely

Uses MMPose for body keypoints + OpenCV for barbell detection.
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
OUTPUT_DIR = os.path.join(REPO_ROOT, "output", "mmdet_test")

# Test parameters
TEST_FRAME_INTERVAL = 5   # Process every 5 frames
MAX_FRAMES = 300          # Process more frames for tracking
SHOW_VISUALIZATION = True


def check_dependencies():
    """Check if required packages are installed."""
    print("\nChecking dependencies...")
    
    try:
        import mmpose
        print(f"✓ MMPose version: {mmpose.__version__}")
    except ImportError:
        print("❌ MMPose not installed. Run: mim install mmpose")
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
    
    for exercise in ['Deadlift', 'Squat', 'Bench Press']:
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


def detect_barbell(frame, pose_result=None, prev_bar_center=None, exercise_type=None):
    """
    Detect barbell using body pose guidance and exercise-specific logic.
    
    Different exercises have different bar positions:
    - Squat: Bar on shoulders/back, moves with torso
    - Deadlift: Bar at ground level initially, then hip/thigh level
    - Bench: Bar starts high, comes to chest level
    
    Args:
        frame: Input video frame
        pose_result: MMPose result with keypoint positions
        prev_bar_center: Previous bar center for tracking continuity
        exercise_type: 'squat', 'deadlift', or 'bench' (optional)
    
    Returns:
        Detection dict with bbox, score, center, etc. or None
    """
    frame_height, frame_width = frame.shape[:2]
    
    # Get search region based on pose
    search_region = None
    expected_bar_y = None
    
    if pose_result is not None and len(pose_result) > 0:
        # Get keypoints from the first detected person
        keypoints = pose_result[0]['keypoints']
        
        # COCO keypoint indices:
        # 0: Nose, 5: Left shoulder, 6: Right shoulder
        # 7: Left elbow, 8: Right elbow  
        # 9: Left wrist, 10: Right wrist
        # 11: Left hip, 12: Right hip
        # 13: Left knee, 14: Right knee
        # 15: Left ankle, 16: Right ankle
        
        # Get key body positions
        nose = keypoints[0]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        left_knee = keypoints[13]
        right_knee = keypoints[14]
        
        # Define search regions based on exercise
        valid_points = []
        
        # For squat: bar is on shoulders/upper back
        # For deadlift: bar is at hands level (waist to floor)
        # For bench: bar is above chest, moves to chest level
        
        # Collect all relevant points with good confidence
        reference_points = []
        if left_wrist[2] > 0.3:
            reference_points.append(('wrist', left_wrist[:2]))
        if right_wrist[2] > 0.3:
            reference_points.append(('wrist', right_wrist[:2]))
        if left_shoulder[2] > 0.3:
            reference_points.append(('shoulder', left_shoulder[:2]))
        if right_shoulder[2] > 0.3:
            reference_points.append(('shoulder', right_shoulder[:2]))
        if left_hip[2] > 0.3:
            reference_points.append(('hip', left_hip[:2]))
        if right_hip[2] > 0.3:
            reference_points.append(('hip', right_hip[:2]))
        if left_knee[2] > 0.3:
            reference_points.append(('knee', left_knee[:2]))
        if right_knee[2] > 0.3:
            reference_points.append(('knee', right_knee[:2]))
        
        if reference_points:
            # Get all Y positions to determine search area
            all_points = np.array([p[1] for p in reference_points])
            all_x = all_points[:, 0]
            all_y = all_points[:, 1]
            
            # Search region: full body range + margins
            min_x = max(0, int(all_x.min() - 150))
            max_x = min(frame_width, int(all_x.max() + 150))
            min_y = max(0, int(all_y.min() - 100))
            max_y = min(frame_height, int(all_y.max() + 100))
            
            search_region = (min_x, min_y, max_x, max_y)
            
            # Expected bar position varies by exercise
            # Use hand positions as primary indicator
            hand_points = [p[1] for p in reference_points if p[0] == 'wrist']
            shoulder_points = [p[1] for p in reference_points if p[0] == 'shoulder']
            
            if hand_points:
                hands_y = np.array(hand_points)[:, 1].mean()
                expected_bar_y = hands_y
            elif shoulder_points:
                shoulders_y = np.array(shoulder_points)[:, 1].mean()
                expected_bar_y = shoulders_y
    
    # Apply edge detection to find horizontal bars
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Use region of interest if available
    if search_region:
        min_x, min_y, max_x, max_y = search_region
        roi = gray[min_y:max_y, min_x:max_x]
    else:
        roi = gray
        min_x, min_y = 0, 0
    
    # Enhance edges - barbells have strong horizontal edges
    blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    
    # Detect lines using Hough transform
    # More sensitive parameters to catch the bar in various positions
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40, 
                           minLineLength=60, maxLineGap=30)
    
    detections = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Convert back to full frame coordinates
            x1, x2 = x1 + min_x, x2 + min_x
            y1, y2 = y1 + min_y, y2 + min_y
            
            # Calculate line properties
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # Filter for horizontal lines (barbell is horizontal)
            # Angle close to 0 or 180 degrees
            if length > 60 and (angle < 20 or angle > 160):
                # Create bounding box around the line
                bbox_height = 30  # Approximate bar thickness
                center_y = (y1 + y2) / 2
                center_x = (x1 + x2) / 2
                
                bbox = [
                    min(x1, x2) - 10,
                    center_y - bbox_height/2,
                    max(x1, x2) + 10,
                    center_y + bbox_height/2
                ]
                
                # Score based on length and horizontal alignment
                score = min(0.85, length / 250.0)
                
                # Boost score if near expected bar position
                if expected_bar_y is not None:
                    distance_to_expected = abs(center_y - expected_bar_y)
                    if distance_to_expected < 150:
                        proximity_score = 0.4 * (1.0 - distance_to_expected / 150.0)
                        score += proximity_score
                        score = min(0.99, score)
                
                # Additional boost for very horizontal lines (more parallel)
                if angle < 5 or angle > 175:
                    score += 0.1
                    score = min(0.99, score)
                
                detections.append({
                    'bbox': bbox,
                    'score': score,
                    'class_id': 0,
                    'aspect_ratio': length / bbox_height,
                    'area': length * bbox_height,
                    'center': [center_x, center_y],
                    'length': length,
                    'angle': angle
                })
    
    # If we have previous bar center, prefer detection closest to it (temporal consistency)
    if prev_bar_center and detections:
        def distance_to_prev(det):
            dx = det['center'][0] - prev_bar_center[0]
            dy = det['center'][1] - prev_bar_center[1]
            # Weight Y distance more heavily (bar shouldn't jump vertically much between frames)
            return dx*dx + 4*dy*dy
        
        detections.sort(key=distance_to_prev)
        # But still check score threshold
        if detections[0]['score'] < 0.3:
            # If closest is too weak, try highest score
            detections.sort(key=lambda x: x['score'], reverse=True)
    else:
        # Sort by score and length
        detections.sort(key=lambda x: (x['score'] * x['length']), reverse=True)
    
    return detections[0] if detections else None


def process_video(video_path, output_path):
    """
    Process video with MMPose + barbell tracking.
    """
    from mmpose.apis import init_pose_model, inference_top_down_pose_model, vis_pose_result
    
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(video_path)}")
    print(f"{'='*60}")
    
    # Detect exercise type from path
    exercise_type = None
    video_path_lower = video_path.lower()
    if 'squat' in video_path_lower:
        exercise_type = 'squat'
        print(f"Detected exercise: SQUAT (bar on shoulders)")
    elif 'deadlift' in video_path_lower:
        exercise_type = 'deadlift'
        print(f"Detected exercise: DEADLIFT (bar at ground/hip level)")
    elif 'bench' in video_path_lower:
        exercise_type = 'bench'
        print(f"Detected exercise: BENCH PRESS (bar at chest/rack level)")
    else:
        print(f"Exercise type: UNKNOWN (using general detection)")
    
    # Initialize MMPose model
    print("Loading MMPose model for body keypoints...")
    
    pose_config = os.path.join(REPO_ROOT, 'src', 'MMPose', 'topdown_heatmap_hrnet_w48_coco_256x192.py')
    pose_checkpoint = os.path.join(REPO_ROOT, 'src', 'MMPose', 'hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth')
    
    if not os.path.exists(pose_config) or not os.path.exists(pose_checkpoint):
        print(f"❌ MMPose model files not found")
        print(f"   Run: bash src/MMPose/install.sh")
        return False
    
    device = 'cuda:0' if __import__('torch').cuda.is_available() else 'cpu'
    
    try:
        pose_model = init_pose_model(pose_config, pose_checkpoint, device=device)
        print(f"✓ MMPose model loaded on {device}")
    except Exception as e:
        print(f"❌ Failed to load MMPose model: {e}")
        return False
    
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
    
    # Store processed frames and bar path
    processed_frames = []
    bar_path = []  # Track bar center points
    
    frame_idx = 0
    processed_count = 0
    prev_bar_center = None
    
    print("\nPlayback Controls:")
    print("  SPACE - Pause/Resume")
    print("  + or ] - Speed up")
    print("  - or [ - Slow down")
    print("  R - Restart video")
    print("  Q - Skip to next video")
    print("  ESC - Exit completely")
    
    # First pass: Process and detect barbell
    print("\nProcessing frames and tracking barbell...")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every Nth frame
            if frame_idx % TEST_FRAME_INTERVAL == 0 and processed_count < MAX_FRAMES:
                # Get pose estimation first
                # Use simple person bounding box (assume person fills most of frame)
                person_results = [{'bbox': np.array([0, 0, width, height, 1.0])}]
                pose_result = inference_top_down_pose_model(
                    pose_model, 
                    frame, 
                    person_results, 
                    format='xyxy',
                    dataset=pose_model.cfg.data.test.type
                )[0]
                
                # Detect barbell using pose guidance
                bar_detection = detect_barbell(frame, pose_result, prev_bar_center, exercise_type)
                
                # Create visualization
                vis_frame = frame.copy()
                
                # Draw pose skeleton
                vis_frame = vis_pose_result(
                    pose_model,
                    vis_frame,
                    pose_result,
                    radius=4,
                    thickness=2,
                    kpt_score_thr=0.3
                )
                
                if bar_detection:
                    # Draw barbell bounding box
                    x1, y1, x2, y2 = map(int, bar_detection['bbox'])
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    
                    # Draw center point
                    center_x, center_y = map(int, bar_detection['center'])
                    cv2.circle(vis_frame, (center_x, center_y), 5, (0, 0, 255), -1)
                    
                    # Add label
                    label = f"Bar: {bar_detection['score']:.2f}"
                    cv2.putText(vis_frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Track bar path
                    bar_path.append(bar_detection['center'])
                    prev_bar_center = bar_detection['center']
                
                # Draw bar path (trail)
                if len(bar_path) > 1:
                    points = np.array(bar_path, dtype=np.int32)
                    cv2.polylines(vis_frame, [points], False, (255, 0, 0), 2)
                
                # Add frame info
                status_text = f"Frame: {frame_idx}/{total_frames} | Bar detected: {'Yes' if bar_detection else 'No'}"
                cv2.putText(vis_frame, status_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # Store frame
                processed_frames.append(vis_frame.copy())
                
                # Write to output
                out.write(vis_frame)
                
                processed_count += 1
                if processed_count % 20 == 0:
                    print(f"  Processed {processed_count} frames...")
            
            frame_idx += 1
    
    finally:
        cap.release()
        out.release()
    
    print(f"\n✓ Output saved: {output_path}")
    print(f"  Processed {processed_count} frames")
    print(f"  Barbell detected in {len(bar_path)} frames")
    print(f"  Bar path tracking: {len(bar_path)} points")
    
    if not processed_frames:
        print("  No frames to display")
        return True
    
    # Second pass: Loop playback with controls
    print("\nStarting playback loop (press Q to continue to next video)...")
    
    playback_delay = 200  # Faster default for tracking visualization
    paused = False
    
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
                cv2.imshow('MMPose + Barbell Tracking', frame_with_controls)
                
                # Wait for key press
                while True:
                    key = cv2.waitKey(playback_delay if not paused else 0) & 0xFF
                    
                    if key == ord(' '):
                        paused = not paused
                        print(f"  {'Paused' if paused else 'Resumed'}")
                    elif key == ord('r') or key == ord('R'):
                        print("  Restarting video...")
                        break
                    elif key == ord('q') or key == ord('Q'):
                        print("\n  Continuing to next video...")
                        cv2.destroyAllWindows()
                        return True
                    elif key == 27:  # ESC
                        print("\n  Exiting...")
                        cv2.destroyAllWindows()
                        return False
                    elif key == ord('+') or key == ord('=') or key == ord(']'):
                        playback_delay = max(10, playback_delay - 50)
                        print(f"  Speed increased - delay: {playback_delay}ms")
                    elif key == ord('-') or key == ord('_') or key == ord('['):
                        playback_delay = min(2000, playback_delay + 50)
                        print(f"  Speed decreased - delay: {playback_delay}ms")
                    
                    if not paused or key != 255:
                        break
                
                if key == ord('r') or key == ord('R'):
                    break
    
    finally:
        cv2.destroyAllWindows()
    
    return True


def main():
    """Main function - processes videos with pose + barbell tracking."""
    print("="*60)
    print("MMPose + Barbell Tracking Test")
    print("Body-guided barbell detection")
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
        print("   Expected structure: lifting_videos/Raw/Bench Press/*.mp4")
        return
    
    print(f"\nFound {len(videos)} test video(s)")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process each video
    success_count = 0
    for video_path in videos:
        video_name = Path(video_path).stem
        output_path = os.path.join(OUTPUT_DIR, f"{video_name}_barbell_tracking.mp4")
        
        result = process_video(video_path, output_path)
        if result:
            success_count += 1
        elif result is False:  # User pressed ESC
            break
    
    # Summary
    print(f"\n{'='*60}")
    print(f"✓ Test Complete!")
    print(f"  Successfully processed: {success_count}/{len(videos)} videos")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
