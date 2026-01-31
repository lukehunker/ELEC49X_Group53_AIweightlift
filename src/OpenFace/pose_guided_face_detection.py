"""
Pose-Guided Face Detection for OpenFace

Uses MediaPipe Pose body keypoints to guide OpenFace to the correct face in multi-person scenarios.
This solves the problem of OpenFace locking onto background faces instead of the exercising person.

Strategy:
1. Detect body pose with MediaPipe (finds the main exercising person)
2. Calculate head bounding box from body keypoints (nose, eyes, ears, shoulders)
3. Crop video to head region before passing to OpenFace
4. OpenFace processes only the region with the correct person's face

COCO Keypoint Indices (converted from MediaPipe's 33 landmarks):
0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
"""

import os
import cv2
import numpy as np
import sys
from pathlib import Path


class PoseGuidedFaceDetector:
    """
    Detects the exercising person's face using body pose guidance.
    """
    
    # COCO keypoint indices for head region
    HEAD_KEYPOINTS = {
        'nose': 0,
        'left_eye': 1,
        'right_eye': 2,
        'left_ear': 3,
        'right_ear': 4,
        'left_shoulder': 5,
        'right_shoulder': 6
    }
    
    def __init__(self, verbose=True):
        """Initialize pose-guided face detector with MediaPipe."""
        self.verbose = verbose
        self._load_mediapipe()
    
    def _load_mediapipe(self):
        
        try:
            import mediapipe as mp
            self._mp_pose = mp.solutions.pose.Pose(
                static_image_mode=False,  
                model_complexity=1,
                enable_segmentation=False,
                smooth_landmarks=True,  
                min_detection_confidence=0.3,  # Lowered from 0.5 for challenging frames
                min_tracking_confidence=0.3  # Lowered from 0.5 for better continuity
            )
            if self.verbose:
                print("✓ Using MediaPipe Pose for face region detection")
                print("  (Video mode with temporal smoothing enabled)")
        except ImportError:
            raise ImportError("MediaPipe required. Install with: pip install mediapipe")
    
    def get_head_bbox_from_pose(self, keypoints, img_shape, confidence_threshold=0.2):
        """
        Calculate head bounding box from body keypoints.
        
        Args:
            keypoints: Array of shape (17, 3) with [x, y, confidence] for each keypoint
            img_shape: (height, width) of image
            confidence_threshold: Minimum confidence for keypoint to be valid
        
        Returns:
            (x1, y1, x2, y2) bounding box or None if insufficient keypoints
        """
        height, width = img_shape[:2]
        
        # Extract head keypoints with confidence check
        head_points = []
        for name, idx in self.HEAD_KEYPOINTS.items():
            x, y, conf = keypoints[idx]
            if conf > confidence_threshold:
                head_points.append((x, y))
        
        if len(head_points) < 3:  # Need at least 3 points for reliable bbox
            return None
        
        # Calculate bounding box
        xs = [p[0] for p in head_points]
        ys = [p[1] for p in head_points]
        
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        
        # Expand bbox to include full head
        # Head is typically 1.5x wider than eye-to-eye distance
        # and extends above eyes and below chin
        
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        # INCREASED EXPANSION for better OpenFace detection
        # Larger crop gives OpenFace more context and handles motion better
        expansion_x = bbox_width * 1.0  # 100% wider (was 60%)
        x1 = max(0, x1 - expansion_x)
        x2 = min(width, x2 + expansion_x)
        
        # Expand vertically (more space = better detection during head movement)
        expansion_top = bbox_height * 1.5  # More forehead (was 1.0)
        expansion_bottom = bbox_height * 2.0  # More chin/neck (was 1.5)
        y1 = max(0, y1 - expansion_top)
        y2 = min(height, y2 + expansion_bottom)
        
        return int(x1), int(y1), int(x2), int(y2)
    
    def detect_main_person(self, frame):
        """
        Detect the main person in frame using MediaPipe.
        Prioritizes the largest/closest person (the lifter, not background people).
        
        Args:
            frame: Input video frame (BGR format)
        
        Returns:
            keypoints array (17, 3) with [x, y, confidence] or None if no person detected
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._mp_pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return None
        
        # Convert MediaPipe landmarks to COCO format (17 keypoints)
        h, w = frame.shape[:2]
        
        # COCO keypoint mapping from MediaPipe (33 landmarks -> 17 COCO keypoints)
        coco_from_mp = {
            0: 0,   # nose
            1: 2,   # left_eye (inner)
            2: 5,   # right_eye (inner)
            3: 7,   # left_ear
            4: 8,   # right_ear
            5: 11,  # left_shoulder
            6: 12,  # right_shoulder
            7: 13,  # left_elbow
            8: 14,  # right_elbow
            9: 15,  # left_wrist
            10: 16, # right_wrist
            11: 23, # left_hip
            12: 24, # right_hip
            13: 25, # left_knee
            14: 26, # right_knee
            15: 27, # left_ankle
            16: 28  # right_ankle
        }
        
        keypoints = np.zeros((17, 3))
        for coco_idx, mp_idx in coco_from_mp.items():
            lm = results.pose_landmarks.landmark[mp_idx]
            keypoints[coco_idx] = [lm.x * w, lm.y * h, lm.visibility]
        
        # Calculate person size (torso height) to prioritize largest person
        # This ensures we track the lifter, not background people
        visible_kpts = keypoints[keypoints[:, 2] > 0.3]
        if len(visible_kpts) > 0:
            # Store person size for consistency checking
            y_coords = visible_kpts[:, 1]
            person_height = np.max(y_coords) - np.min(y_coords)
            keypoints_with_size = (keypoints, person_height)
            return keypoints_with_size
        
        return keypoints
    

    
    def create_cropped_video(self, input_video_path, output_video_path, 
                            track_smoothing=5, fallback_to_full=True):
        """
        Create a cropped video focused on the main person's head region.
        
        Args:
            input_video_path: Path to input video
            output_video_path: Path to save cropped video
            track_smoothing: Number of frames to smooth bounding box (reduces jitter)
            fallback_to_full: If True, use full frame when pose not detected
        
        Returns:
            Path to cropped video, or None if failed
        """
        # Try different backends for problematic codecs (AV1, HEVC, etc.)
        backends_to_try = [
            (cv2.CAP_FFMPEG, "FFmpeg"),
            (cv2.CAP_ANY, "Auto")
        ]
        
        cap = None
        for backend_id, backend_name in backends_to_try:
            test_cap = cv2.VideoCapture(input_video_path, backend_id)
            if test_cap.isOpened():
                # Test if we can actually read frames
                ret, test_frame = test_cap.read()
                if ret and test_frame is not None:
                    test_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
                    cap = test_cap
                    if self.verbose:
                        print(f"  Using OpenCV backend: {backend_name}")
                    break
                test_cap.release()
        
        if cap is None or not cap.isOpened():
            raise ValueError(f"Cannot open video with any backend: {input_video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if self.verbose:
            print(f"\nCreating pose-guided cropped video...")
            print(f"  Input: {os.path.basename(input_video_path)}")
            print(f"  Resolution: {original_width}x{original_height}")
            print(f"  Frames: {total_frames}")
        
        # First pass: detect head bbox for EVERY frame
        if self.verbose:
            print(f"  Pass 1: Detecting head regions in every frame...")
            print(f"  Strategy: Track largest person (lifter, not background)")
        
        all_bboxes = []  # One bbox per frame
        all_person_sizes = []  # Track person size for consistency
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect pose in THIS frame
            detection = self.detect_main_person(frame)
            
            if detection is not None:
                # Unpack keypoints and person size
                if isinstance(detection, tuple):
                    keypoints, person_size = detection
                    all_person_sizes.append(person_size)
                else:
                    keypoints = detection
                    all_person_sizes.append(0)
                
                bbox = self.get_head_bbox_from_pose(keypoints, frame.shape)
                all_bboxes.append(bbox if bbox is not None else None)
            else:
                all_bboxes.append(None)
            
            frame_idx += 1
            
            if self.verbose and frame_idx % 100 == 0:
                print(f"    Processed {frame_idx}/{total_frames} frames...")
        
        cap.release()
        
        # Validate consistent person tracking (filter outliers)
        if all_person_sizes:
            median_size = np.median([s for s in all_person_sizes if s > 0])
            if self.verbose:
                print(f"  Median person size: {median_size:.0f}px (filtering outliers)")
            
            # Filter out detections that are too small (likely background people)
            size_threshold = median_size * 0.6  # Must be at least 60% of median size
            for i, (bbox, size) in enumerate(zip(all_bboxes, all_person_sizes)):
                if bbox is not None and size > 0 and size < size_threshold:
                    if self.verbose and i % 100 == 0:
                        print(f"    Frame {i}: Filtered small person (size={size:.0f} < {size_threshold:.0f})")
                    all_bboxes[i] = None  # Remove background person detection
        
        # Fill in missing detections with nearest valid bbox
        valid_bboxes = [b for b in all_bboxes if b is not None]
        if not valid_bboxes:
            if self.verbose:
                print("  WARNING: No head region detected, using full frame")
            if fallback_to_full:
                import shutil
                shutil.copy(input_video_path, output_video_path)
                return output_video_path
            return None
        
        # Interpolate missing bboxes
        for i in range(len(all_bboxes)):
            if all_bboxes[i] is None:
                # Find nearest valid bbox
                if i > 0 and all_bboxes[i-1] is not None:
                    all_bboxes[i] = all_bboxes[i-1]  # Use previous
                elif i < len(all_bboxes) - 1:
                    # Look forward for next valid
                    for j in range(i+1, len(all_bboxes)):
                        if all_bboxes[j] is not None:
                            all_bboxes[i] = all_bboxes[j]
                            break
        
        # Use FIXED crop region for entire video (eliminates all jitter)
        # Calculate median bbox position and max dimensions
        valid_bboxes = [b for b in all_bboxes if b is not None]
        if not valid_bboxes:
            if self.verbose:
                print(f"  Warning: No valid bboxes detected")
            return None
        
        # Get median center and max dimensions
        x1_median = int(np.median([b[0] for b in valid_bboxes]))
        y1_median = int(np.median([b[1] for b in valid_bboxes]))
        x2_median = int(np.median([b[2] for b in valid_bboxes]))
        y2_median = int(np.median([b[3] for b in valid_bboxes]))
        
        # Use max dimensions to ensure face fits in all frames
        max_width = max(b[2] - b[0] for b in valid_bboxes)
        max_height = max(b[3] - b[1] for b in valid_bboxes)
        
        # Calculate fixed crop region centered on median position
        center_x = (x1_median + x2_median) // 2
        center_y = (y1_median + y2_median) // 2
        
        crop_width = int(max_width * 1.2)  # 20% extra padding
        crop_height = int(max_height * 1.2)
        
        # Fixed bbox for ALL frames (completely stable)
        fixed_x1 = max(0, center_x - crop_width // 2)
        fixed_y1 = max(0, center_y - crop_height // 2)
        fixed_x2 = min(original_width, fixed_x1 + crop_width)
        fixed_y2 = min(original_height, fixed_y1 + crop_height)
        
        # Adjust if we hit boundaries
        if fixed_x2 - fixed_x1 < crop_width:
            fixed_x1 = max(0, fixed_x2 - crop_width)
        if fixed_y2 - fixed_y1 < crop_height:
            fixed_y1 = max(0, fixed_y2 - crop_height)
        
        crop_width = fixed_x2 - fixed_x1
        crop_height = fixed_y2 - fixed_y1
        
        if self.verbose:
            print(f"  Fixed crop region: {crop_width}x{crop_height} (no jitter)")
            print(f"  Position: ({fixed_x1}, {fixed_y1}) to ({fixed_x2}, {fixed_y2})")
        
        # Second pass: crop each frame individually with its bbox
        if self.verbose:
            print(f"  Pass 2: Cropping each frame to tracked face...")
        
        # Try different backends for problematic codecs
        backends_to_try = [
            (cv2.CAP_FFMPEG, "FFmpeg"),
            (cv2.CAP_ANY, "Auto")
        ]
        
        cap = None
        for backend_id, backend_name in backends_to_try:
            test_cap = cv2.VideoCapture(input_video_path, backend_id)
            if test_cap.isOpened():
                ret, test_frame = test_cap.read()
                if ret and test_frame is not None:
                    test_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    cap = test_cap
                    break
                test_cap.release()
        
        if cap is None or not cap.isOpened():
            raise ValueError(f"Cannot reopen video: {input_video_path}")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (crop_width, crop_height))
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Use FIXED crop region for all frames (completely stable)
            cropped = frame[fixed_y1:fixed_y2, fixed_x1:fixed_x2]
            
            # Ensure consistent output size
            if cropped.shape[1] != crop_width or cropped.shape[0] != crop_height:
                cropped = cv2.resize(cropped, (crop_width, crop_height))
            
            out.write(cropped)
            frame_idx += 1
        
        cap.release()
        out.release()
        
        if self.verbose:
            print(f"  ✓ Cropped video saved: {output_video_path}")
            print(f"    Reduction: {original_width}x{original_height} → {crop_width}x{crop_height}")
            print(f"    Crop %: {(crop_width*crop_height)/(original_width*original_height)*100:.1f}%")
            print(f"    Mode: Frame-by-frame face tracking (follows head movement)")
        
        return output_video_path
    
    def _smooth_bboxes(self, bboxes, window=15):
        """Apply moving average smoothing to reduce bbox jitter."""
        try:
            from scipy.ndimage import uniform_filter1d
        except ImportError:
            # Fallback: simple moving average
            return bboxes
        
        valid_indices = [i for i, b in enumerate(bboxes) if b is not None]
        if not valid_indices:
            return bboxes
        
        # Extract coordinates
        x1s = np.array([bboxes[i][0] if b is not None else 0 for i, b in enumerate(bboxes)])
        y1s = np.array([bboxes[i][1] if b is not None else 0 for i, b in enumerate(bboxes)])
        x2s = np.array([bboxes[i][2] if b is not None else 0 for i, b in enumerate(bboxes)])
        y2s = np.array([bboxes[i][3] if b is not None else 0 for i, b in enumerate(bboxes)])
        
        # Smooth
        x1s_smooth = uniform_filter1d(x1s, size=window, mode='nearest')
        y1s_smooth = uniform_filter1d(y1s, size=window, mode='nearest')
        x2s_smooth = uniform_filter1d(x2s, size=window, mode='nearest')
        y2s_smooth = uniform_filter1d(y2s, size=window, mode='nearest')
        
        # Reconstruct
        smoothed = []
        for i in range(len(bboxes)):
            if bboxes[i] is not None:
                smoothed.append((
                    int(x1s_smooth[i]),
                    int(y1s_smooth[i]),
                    int(x2s_smooth[i]),
                    int(y2s_smooth[i])
                ))
            else:
                smoothed.append(None)
        
        return smoothed


def create_pose_guided_video_for_openface(video_path, output_dir=None, detector=None):
    """
    Convenience function to create a pose-guided cropped video for OpenFace processing.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save cropped video (default: same as input)
        detector: Pre-initialized PoseGuidedFaceDetector (optional)
    
    Returns:
        Path to cropped video
    """
    if output_dir is None:
        output_dir = os.path.dirname(video_path)
    
    # Generate output filename
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"{video_name}_pose_guided.mp4")
    
    # Create detector if not provided
    if detector is None:
        detector = PoseGuidedFaceDetector(verbose=True)
    
    # Create cropped video
    result_path = detector.create_cropped_video(video_path, output_path)
    
    return result_path


# =========================================
# TESTING
# =========================================

if __name__ == "__main__":
    import sys
    import glob
    
    print("=" * 70)
    print("Pose-Guided Face Detection for OpenFace")
    print("=" * 70)
    
    if len(sys.argv) < 2:
        print("\nUsage: python pose_guided_face_detection.py <video_path_or_directory>")
        print("\nExamples:")
        print("  Single video:  python pose_guided_face_detection.py video.mp4")
        print("  Directory:     python pose_guided_face_detection.py ./lifting_videos/Augmented/Deadlift/")
        print("\nThis will create cropped videos focused on the main person's face,")
        print("which can then be processed by OpenFace for more accurate facial analysis.")
        sys.exit(0)
    
    input_path = sys.argv[1]
    
    # Resolve path: try current dir first, then relative to repo root
    if not os.path.exists(input_path):
        # Try relative to repo root (2 levels up from this script)
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        alt_path = os.path.join(repo_root, input_path.lstrip('./'))
        
        if os.path.exists(alt_path):
            input_path = alt_path
        else:
            print(f"Error: Path not found: {sys.argv[1]}")
            print(f"\nTried:")
            print(f"  1. {os.path.abspath(sys.argv[1])}")
            print(f"  2. {alt_path}")
            print(f"\nTip: Use absolute path or run from repository root")
            sys.exit(1)
    
    # Handle directory vs file
    video_files = []
    if os.path.isdir(input_path):
        # Process all videos in directory
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
            video_files.extend(glob.glob(os.path.join(input_path, ext)))
        
        if not video_files:
            print(f"Error: No video files found in {input_path}")
            print(f"\nSupported formats: .mp4, .avi, .mov, .mkv")
            sys.exit(1)
        
        print(f"\nFound {len(video_files)} videos to process\n")
    elif os.path.isfile(input_path):
        video_files = [input_path]
    else:
        print(f"Error: {input_path} is not a valid file or directory")
        sys.exit(1)
    
    # Create pose-guided videos
    detector = PoseGuidedFaceDetector(verbose=True)
    success_count = 0
    
    for idx, video_path in enumerate(video_files, 1):
        print(f"\n[{idx}/{len(video_files)}] Processing: {os.path.basename(video_path)}")
        print("=" * 70)
        
        output_path = video_path.replace('.mp4', '_pose_guided.mp4')
        if not output_path.endswith('_pose_guided.mp4'):
            # Handle non-.mp4 extensions
            base, ext = os.path.splitext(video_path)
            output_path = f"{base}_pose_guided{ext}"
        
        try:
            cropped_video = detector.create_cropped_video(video_path, output_path)
            
            if cropped_video:
                print(f"\n✓ Success! Cropped video saved to:")
                print(f"  {cropped_video}")
                success_count += 1
            else:
                print(f"\n✗ Failed to create cropped video for {os.path.basename(video_path)}")
        except Exception as e:
            print(f"\n✗ Error processing {os.path.basename(video_path)}: {e}")
    
    print("\n" + "=" * 70)
    print(f"Completed: {success_count}/{len(video_files)} videos processed successfully")
    print("=" * 70)
    
    if success_count > 0:
        print("\nNext step: Process the cropped videos with OpenFace for better face detection.")
