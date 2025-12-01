import os
import cv2
import numpy as np
from mmpose.apis import MMPoseInferencer

# ==========================================
# USER SETTINGS
# ==========================================
input_folder = r"C:\Users\ajmal\Documents\ELEC49X_Group53_AIweightlift\lifting_videos\Augmented\Squat"
output_folder = os.path.join(input_folder, "Final_Stabilized_Crops")
os.makedirs(output_folder, exist_ok=True)

crop_size = (256, 256)
SMOOTHING_FACTOR = 0.15  # Lower = Smoother. 0.15 is very steady.

# ==========================================
# 1. SETUP MODEL
# ==========================================
print("Setting up MMPose (RTMPose)...")
inferencer = MMPoseInferencer(pose2d='face', device='cpu')


# ==========================================
# 2. HELPER: ROBUST DEEP SEARCH
# ==========================================
def find_keypoints_and_scores(data):
    """
    Recursively hunts for 'keypoints' and 'keypoint_scores'
    inside the nested data structure.
    """
    # If we found a dict containing both keys, return them
    if isinstance(data, dict):
        if 'keypoints' in data and 'keypoint_scores' in data:
            return data['keypoints'], data['keypoint_scores']

        # Otherwise keep searching deeper in values
        for key in data:
            result = find_keypoints_and_scores(data[key])
            if result: return result

    # If it's a list, search every item
    elif isinstance(data, (list, tuple)):
        for item in data:
            result = find_keypoints_and_scores(item)
            if result: return result

    return None


def get_centroid(result, min_score=0.2):
    # 1. Deep Search for data
    found = find_keypoints_and_scores(result)
    if not found:
        return None

    kpts, scores = found

    # Convert to numpy
    kpts = np.array(kpts)
    scores = np.array(scores)

    # Flatten if needed (sometimes 1xNx2)
    if len(kpts.shape) == 3: kpts = kpts[0]
    if len(scores.shape) == 2: scores = scores[0]

    # 2. Filter by Score (Remove noise)
    valid_mask = scores > min_score
    valid_kpts = kpts[valid_mask]

    # We need at least 3 points to trust it's a face
    if len(valid_kpts) < 3:
        return None

    # 3. Calculate Mean
    cx = np.mean(valid_kpts[:, 0])
    cy = np.mean(valid_kpts[:, 1])

    return (cx, cy)


def get_crop_coords(center_x, center_y, img_w, img_h, size):
    half_w = size[0] // 2
    half_h = size[1] // 2

    x1 = int(center_x - half_w)
    y1 = int(center_y - half_h)
    x2 = x1 + size[0]
    y2 = y1 + size[1]

    # Clamp edges (Shift box back if it goes OOB)
    if x1 < 0: x2 -= x1; x1 = 0
    if y1 < 0: y2 -= y1; y1 = 0
    if x2 > img_w: x1 -= (x2 - img_w); x2 = img_w
    if y2 > img_h: y1 -= (y2 - img_h); y2 = img_h

    return x1, y1, x2, y2


# ==========================================
# 3. PROCESSING LOOP
# ==========================================
import re

video_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
video_files.sort(key=lambda f: int(re.search(r'\d+', f).group()) if re.search(r'\d+', f) else -1)

print(f"Found {len(video_files)} videos.")

for video_file in video_files:
    video_path = os.path.join(input_folder, video_file)
    save_path = os.path.join(output_folder, f"Locked_{video_file}")

    print(f"Processing: {video_file}...")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    out_writer = None
    result_generator = inferencer(video_path, return_vis=False)

    # TRACKING MEMORY
    prev_cx, prev_cy = None, None
    frame_idx = 0
    faces_detected = 0

    try:
        for result in result_generator:
            ret, frame = cap.read()
            if not ret: break

            h_img, w_img = frame.shape[:2]

            # --- 1. Find Centroid ---
            target = get_centroid(result, min_score=0.2)

            final_cx, final_cy = prev_cx, prev_cy

            if target:
                faces_detected += 1
                curr_cx, curr_cy = target

                # --- 2. Smooth Movement ---
                if prev_cx is None:
                    final_cx, final_cy = curr_cx, curr_cy
                else:
                    final_cx = (curr_cx * SMOOTHING_FACTOR) + (prev_cx * (1 - SMOOTHING_FACTOR))
                    final_cy = (curr_cy * SMOOTHING_FACTOR) + (prev_cy * (1 - SMOOTHING_FACTOR))

                # Update memory
                prev_cx, prev_cy = final_cx, final_cy

            # --- 3. Crop & Write ---
            if final_cx is not None:
                x1, y1, x2, y2 = get_crop_coords(final_cx, final_cy, w_img, h_img, crop_size)

                crop = frame[y1:y2, x1:x2]

                # Safety resize
                if crop.shape[0] != crop_size[1] or crop.shape[1] != crop_size[0]:
                    try:
                        crop = cv2.resize(crop, crop_size)
                    except:
                        pass  # Skip if crop is 0 size

                if out_writer is None:
                    print(f"  > Initialized writer at Frame {frame_idx}")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out_writer = cv2.VideoWriter(save_path, fourcc, fps, crop_size)

                if crop.size > 0:
                    out_writer.write(crop)

            # Debug first few frames
            if frame_idx < 5:
                status = "FACE FOUND" if target else "NO FACE"
                print(f"  [Frame {frame_idx}] {status} | Memory: {'Yes' if prev_cx else 'No'}")

            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"  > {frame_idx} frames processed...")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        if out_writer:
            out_writer.release()
            print(f"Saved: {save_path} (Faces detected in {faces_detected} frames)")
        else:
            print(f"WARNING: No video saved for {video_file} (No faces found)")

print("All done!")