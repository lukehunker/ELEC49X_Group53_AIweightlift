import os
import re
import cv2
import numpy as np
import traceback
from datetime import datetime
from mmpose.apis import MMPoseInferencer
import pandas as pd

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
VIDEOS_ROOT = "../../lifting_videos/Demo_Videos/"
MOVEMENT_FOLDERS = ["Bench Press", "Squat", "Deadlift"]

# Outputs
OUTPUT_ROOT = "../Train_Outputs"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# *** THE MASTER EXCEL FILE ***
MASTER_EXCEL_PATH = os.path.join(OUTPUT_ROOT, "barSpeed.xlsx")

print(f"[INFO] Videos root: {VIDEOS_ROOT}")
print(f"[INFO] Output root: {OUTPUT_ROOT}")
print(f"[INFO] Master Excel: {MASTER_EXCEL_PATH}")

# Note: Pixels per CM is less critical for time, but kept for depth calcs
PIXELS_PER_CM = 10.0

# ---------------------------------------------------------
# MMPose inferencer
# ---------------------------------------------------------
inferencer = MMPoseInferencer(
    pose2d="body",
    device="cpu"
)

LEFT_WRIST_IDX = 9
RIGHT_WRIST_IDX = 10
CONF_THR = 0.35

# Wrist tracking knobs
POSE_INIT_FRAMES = 12
POSE_RESEED_EVERY = 12
MIN_GOOD_POINTS = 3
MAX_RESEED_DIST_PX = 120
CLUSTER_OFFSETS = [(-8, 0), (8, 0), (0, -8), (0, 8), (0, 0), (-6, -6), (6, 6)]


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def extract_number(filename: str) -> int:
    m = re.search(r"\d+", filename)
    return int(m.group()) if m else 10 ** 9


def moving_average(x, window=7):
    x = np.asarray(x, dtype=float)
    if len(x) == 0: return x
    idx = np.arange(len(x))
    mask = np.isfinite(x)
    if mask.sum() < 2: return x
    x_interp = np.interp(idx, idx[mask], x[mask])
    kernel = np.ones(window) / window
    return np.convolve(x_interp, kernel, mode="same")


def _flatten_bbox_candidate(b):
    if b is None: return None
    if isinstance(b, dict):
        for k in ("bbox", "bboxes", "box"):
            if k in b:
                b = b[k]
                break
    for _ in range(6):
        if isinstance(b, (list, tuple)):
            if len(b) == 0: return None
            if len(b) == 1 and isinstance(b[0], (list, tuple)):
                b = b[0]
                continue
            if len(b) == 2 and isinstance(b[0], (list, tuple)) and not isinstance(b[1], (list, tuple, dict)):
                b = b[0]
                continue
        break
    if not isinstance(b, (list, tuple)) or len(b) < 4: return None
    out = []
    for i in range(4):
        v = b[i]
        if isinstance(v, (list, tuple, dict)): return None
        try:
            out.append(float(v))
        except Exception:
            return None
    return out


def _bbox_to_xyxy(b):
    flat = _flatten_bbox_candidate(b)
    if flat is None: return None
    x1, y1, a, d = flat
    if a > x1 and d > y1: return [x1, y1, a, d]
    return [x1, y1, x1 + a, y1 + d]


def iou_xyxy(b1, b2):
    if b1 is None or b2 is None: return 0.0
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    denom = a1 + a2 - inter
    return float(inter / denom) if denom > 0 else 0.0


def seed_cluster(cx, cy):
    pts = [[cx + dx, cy + dy] for dx, dy in CLUSTER_OFFSETS]
    return np.array(pts, dtype=np.float32).reshape(-1, 1, 2)


def run_pose_on_frame(frame, inferencer, conf_thr):
    gen = inferencer(frame, return_vis=False, kpt_thr=conf_thr)
    try:
        result = next(gen)
    except StopIteration:
        return []
    preds_all = result.get("predictions", None)
    if not preds_all or not preds_all[0]: return []
    return preds_all[0]


def score_person_for_lifter(person, frame_w, frame_h):
    ksc = np.array(person.get("keypoint_scores", []), dtype=float)
    bbox = _bbox_to_xyxy(person.get("bbox", None))
    if bbox is not None:
        area = max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        cx = 0.5 * (bbox[0] + bbox[2])
        cy = 0.5 * (bbox[1] + bbox[3])
    else:
        area = 1.0
        cx = frame_w / 2.0
        cy = frame_h / 2.0
    nx = (cx - frame_w / 2.0) / (frame_w / 2.0)
    ny = (cy - frame_h / 2.0) / (frame_h / 2.0)
    center_penalty = (nx * nx + ny * ny)
    wrist_bonus = 0.0
    if len(ksc) > RIGHT_WRIST_IDX:
        wrist_bonus = float(ksc[LEFT_WRIST_IDX] + ksc[RIGHT_WRIST_IDX])
    return area * (1.5 + wrist_bonus) * (1.0 / (1.0 + 2.0 * center_penalty))


def seed_from_plate(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2.0)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=80,
        param1=100, param2=30, minRadius=25, maxRadius=350,
    )
    if circles is None: return None
    circles = np.round(circles[0, :]).astype(int)
    circles = sorted(circles, key=lambda c: c[2], reverse=True)
    x, y, r = circles[0]
    return ("plate", (float(x), float(y)), None)


def choose_lifter_and_wrist_seed(pose_frames, frame_w, frame_h, inferencer, conf_thr):
    chosen_bbox = None
    left_pts, right_pts = [], []
    left_scores, right_scores = [], []
    for frame in pose_frames:
        persons = run_pose_on_frame(frame, inferencer, conf_thr)
        if not persons: continue
        best = None
        best_score = -1e18
        for p in persons:
            s = score_person_for_lifter(p, frame_w, frame_h)
            if chosen_bbox is not None:
                s += 2.0 * iou_xyxy(chosen_bbox, _bbox_to_xyxy(p.get("bbox", None)))
            if s > best_score:
                best_score = s
                best = p
        if best is None: continue
        bbox_xyxy = _bbox_to_xyxy(best.get("bbox", None))
        if bbox_xyxy is not None: chosen_bbox = bbox_xyxy
        kpts = np.array(best.get("keypoints", []), dtype=float)
        ksc = np.array(best.get("keypoint_scores", []), dtype=float)
        if len(kpts) <= RIGHT_WRIST_IDX or len(ksc) <= RIGHT_WRIST_IDX: continue
        if ksc[LEFT_WRIST_IDX] >= conf_thr:
            left_pts.append(kpts[LEFT_WRIST_IDX])
            left_scores.append(ksc[LEFT_WRIST_IDX])
        if ksc[RIGHT_WRIST_IDX] >= conf_thr:
            right_pts.append(kpts[RIGHT_WRIST_IDX])
            right_scores.append(ksc[RIGHT_WRIST_IDX])
    if not left_pts and not right_pts:
        plate = seed_from_plate(pose_frames[0])
        if plate is None: return None
        return plate

    def quality(pts, scores):
        if not pts: return 0.0, None
        arr = np.stack(pts, axis=0)
        return len(pts) * float(np.mean(scores)), arr

    lq, la = quality(left_pts, left_scores)
    rq, ra = quality(right_pts, right_scores)
    if rq > lq:
        wrist_side = "right"
        arr = ra
    else:
        wrist_side = "left"
        arr = la if la is not None else ra
    center = arr.mean(axis=0)
    return (wrist_side, (float(center[0]), float(center[1])), chosen_bbox)


def pick_person_by_bbox(persons, prev_bbox):
    if not persons: return None, None
    if prev_bbox is None:
        best, best_area, best_bbox = None, -1.0, None
        for p in persons:
            b = _bbox_to_xyxy(p.get("bbox", None))
            if b is None: continue
            area = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
            if area > best_area:
                best_area = area
                best = p
                best_bbox = b
        return best, best_bbox
    best, best_iou, best_bbox = None, -1.0, None
    for p in persons:
        b = _bbox_to_xyxy(p.get("bbox", None))
        i = iou_xyxy(prev_bbox, b)
        if i > best_iou:
            best_iou = i
            best = p
            best_bbox = b
    return best, best_bbox


def get_pose_wrist(person, prefer_side, conf_thr):
    if person is None: return None
    kpts = np.array(person.get("keypoints", []), dtype=float)
    ksc = np.array(person.get("keypoint_scores", []), dtype=float)
    if len(kpts) <= RIGHT_WRIST_IDX or len(ksc) <= RIGHT_WRIST_IDX: return None
    if prefer_side == "left":
        cand = [LEFT_WRIST_IDX, RIGHT_WRIST_IDX]
    elif prefer_side == "right":
        cand = [RIGHT_WRIST_IDX, LEFT_WRIST_IDX]
    else:
        cand = [LEFT_WRIST_IDX, RIGHT_WRIST_IDX]
    for idx in cand:
        if ksc[idx] >= conf_thr:
            return (float(kpts[idx][0]), float(kpts[idx][1]), float(ksc[idx]))
    return None


# ---------------------------------------------------------
# LOGIC: State Machine & Speed
# ---------------------------------------------------------
def detect_reps_state_machine_auto(
        y_smooth,
        fps,
        top_margin=0.40,
        bottom_margin=0.40,
        min_rep_seconds=0.4,
        force_pattern=None,
):
    reps = []
    y = np.asarray(y_smooth, dtype=float)

    # 1. Determine Movement Pattern
    valid = y[np.isfinite(y)]
    top_y = np.percentile(valid, 5)
    bottom_y = np.percentile(valid, 95)
    rom = bottom_y - top_y

    if force_pattern in ("TBT", "BTB"):
        pattern = force_pattern
    else:
        pattern = "BTB" if abs(y[0] - bottom_y) < abs(y[0] - top_y) else "TBT"

    # 2. Find Candidate Bottoms (Local Peaks)
    from scipy.signal import find_peaks
    # We look for peaks (bottoms) that are at least 60% into the total ROM
    height_thresh = top_y + 0.60 * rom if pattern == "TBT" else None
    if pattern == "BTB":  # Deadlift peaks are "highs" (min Y)
        peaks, _ = find_peaks(-y, height=-(bottom_y - 0.60 * rom), distance=int(fps * min_rep_seconds))
    else:
        peaks, _ = find_peaks(y, height=height_thresh, distance=int(fps * min_rep_seconds))

    # 3. For each peak, find the true start and true end
    for p in peaks:
        # --- FIND START (Look Back) ---
        # Look back up to 4 seconds. Find the last point where the bar was "Stationary"
        # at the top before it started moving towards the peak.
        search_start = max(0, p - int(4.0 * fps))
        pre_snippet = y[search_start:p]

        if pattern == "TBT":
            # Start is the frame where Y was at its minimum (highest point)
            # closest to the descent.
            start_idx = search_start + np.argmin(pre_snippet)
        else:
            # Deadlift: Start is the frame where Y was maximum (on floor)
            start_idx = search_start + np.argmax(pre_snippet)

        # --- FIND END (Look Forward) ---
        # Look forward up to 4 seconds. Find the first point where the bar returns
        # to a stationary "Top" position.
        search_end = min(len(y), p + int(4.0 * fps))
        post_snippet = y[p:search_end]

        if pattern == "TBT":
            # End is the frame where Y is at its minimum (lockout)
            end_idx = p + np.argmin(post_snippet)
        else:
            # Deadlift: End is the frame where Y is at its maximum (floor)
            end_idx = p + np.argmax(post_snippet)

        # --- VALIDATION ---
        # Ensure the rep has enough movement and duration
        duration = (end_idx - start_idx) / fps
        if duration >= min_rep_seconds:
            # Avoid duplicate or overlapping reps
            if not reps or start_idx > reps[-1][2]:
                reps.append((start_idx, p, end_idx))

    safe_thr = {"top_y": float(top_y), "bottom_y": float(bottom_y), "rom": float(rom)}
    return reps, pattern, safe_thr


# --- UPDATED FUNCTION FOR TIME DURATION ---
def compute_rep_durations(y_smooth, fps, reps, pattern):
    """
    Calculates the duration (seconds) of the Concentric (lifting) phase.
    Returns a list of tuples: (duration_seconds, depth_pixels)
    """
    y = np.asarray(y_smooth, dtype=float)
    stats = []

    for triple in reps:
        if pattern == "TBT":  # Squat/Bench (Top -> Bottom -> Top)
            t1, b, t2 = triple

            # DURATION CALCULATION:
            # Concentric Only (Pushing up): (t2 - b) / fps
            # Full Rep (Down + Up): (t2 - t1) / fps

            # Using Concentric as it is best for fatigue tracking
            duration = (t2 - b) / fps

            depth_px = y[b] - min(y[t1], y[t2])
        else:  # Deadlift (Bottom -> Top -> Bottom)
            b1, t, b2 = triple

            # DURATION CALCULATION:
            # Concentric Only (Pulling up): (t - b1) / fps

            duration = (t - b1) / fps

            depth_px = max(y[b1], y[b2]) - y[t]

        stats.append((float(duration), float(depth_px)))

    return stats


def find_movement_folder(root, desired_name):
    folder_path = os.path.join(root, desired_name)
    if os.path.isdir(folder_path): return folder_path, desired_name
    return None, None


def process_video(video_path, movement_name, out_dir):
    cap_pose = cv2.VideoCapture(video_path)
    if not cap_pose.isOpened():
        print(f"[ERROR] Could not open: {video_path}")
        return None
    fps = cap_pose.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap_pose.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_pose.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pose_frames = []
    for _ in range(POSE_INIT_FRAMES):
        ret, frame = cap_pose.read()
        if not ret: break
        pose_frames.append(frame)
    cap_pose.release()
    if not pose_frames: return None

    seed = choose_lifter_and_wrist_seed(pose_frames, width, height, inferencer, CONF_THR)
    if seed is None:
        print(f"[ERROR] No seed for {os.path.basename(video_path)}")
        return None
    wrist_side, (cx, cy), lifter_bbox = seed
    p0 = seed_cluster(cx, cy)

    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret: return None
    old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    vis_path = os.path.join(out_dir, f"{movement_name}__{base_name}__wrist_vis.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(vis_path, fourcc, fps, (width, height))

    ys = [float(cy)]
    xs = [float(cx)]
    frame_idx = 1
    lost_counter = 0
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Print dots to show it is alive
        if frame_idx % 30 == 0:
            print(".", end="", flush=True)

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        good = None
        if p1 is not None and st is not None: good = p1[st == 1]

        if good is None or len(good) < MIN_GOOD_POINTS:
            ys.append(np.nan)
            lost_counter += 1
        else:
            mx, my = np.median(good, axis=0)
            ys.append(float(my))
            p0 = good.reshape(-1, 1, 2)
            lost_counter = 0
            for gx, gy in good: cv2.circle(frame, (int(gx), int(gy)), 4, (0, 255, 255), -1)
            cv2.circle(frame, (int(mx), int(my)), 6, (0, 0, 255), -1)

        do_reseed = (frame_idx % POSE_RESEED_EVERY == 0) or (lost_counter >= 2)
        if do_reseed and wrist_side != "plate":
            persons = run_pose_on_frame(frame, inferencer, CONF_THR)
            chosen_person, chosen_bbox = pick_person_by_bbox(persons, lifter_bbox)
            if chosen_bbox is not None: lifter_bbox = chosen_bbox
            pw = get_pose_wrist(chosen_person, wrist_side, CONF_THR)
            if pw:
                px, py, pconf = pw
                if np.isfinite(ys[-1]) and np.hypot(px - xs[-1], py - ys[-1]) > MAX_RESEED_DIST_PX and lost_counter < 2:
                    pass
                else:
                    p0 = seed_cluster(px, py)
                    ys[-1] = float(py)
                    xs.append(float(px))

        old_gray = frame_gray
        writer.write(frame)
        frame_idx += 1
    cap.release()
    writer.release()

    ys = np.array(ys, dtype=float)
    y_smooth = moving_average(ys, window=7)
    force = "BTB" if movement_name.lower() == "deadlift" else None

    reps, pattern, thr = detect_reps_state_machine_auto(y_smooth, fps, force_pattern=force)

    # --- CHANGED TO DURATION CALCULATION ---
    # stats is a list of tuples: (duration_seconds, depth_pixels)
    stats = compute_rep_durations(y_smooth, fps, reps, pattern)

    if len(stats) < len(reps): reps = reps[:len(stats)]
    if len(reps) < 1:
        print(f"[WARN] No valid reps for {base_name}")
        return None

    # Extract just the durations
    durations = [s[0] for s in stats]

    first_rep_duration = durations[0]
    last_rep_duration = durations[-1]

    # FATIGUE: Last Rep Time - First Rep Time
    # Positive result means you got SLOWER (Last rep took longer)
    fatigue_seconds = last_rep_duration - first_rep_duration

    txt_path = os.path.join(out_dir, f"{movement_name}__{base_name}__wrist_barspeed.txt")
    with open(txt_path, "w") as f:
        f.write(f"Video: {base_name}\nMovement: {movement_name}\nRep pattern: {pattern}\n")
        f.write(f"Detected reps: {len(reps)}\n\n")
        for i, dur in enumerate(durations, 1):
            f.write(f"Rep {i}: {dur:.2f} seconds\n")
        f.write(f"\nFatigue (Last - First): {fatigue_seconds:.2f}s\n")

    np.save(os.path.join(out_dir, f"{movement_name}__{base_name}__wrist_y.npy"), ys)
    print(
        f"\n[OK] {movement_name} | {base_name}: reps={len(reps)}, first={first_rep_duration:.2f}s, last={last_rep_duration:.2f}s, fatigue={fatigue_seconds:.2f}s")

    return {
        "video_name": os.path.basename(video_path),
        "rep_count": len(reps),
        "first_rep_duration_s": first_rep_duration,
        "last_rep_duration_s": last_rep_duration,
        "fatigue_s": fatigue_seconds,
    }


def runManual():
    rows = []
    print("\n==========================================")
    print("    SINGLE VIDEO PROCESSING MODE")
    print("==========================================\n")

    # 1. Ask for Movement
    print("Select Movement Folder:")
    for i, m in enumerate(MOVEMENT_FOLDERS):
        print(f"  {i + 1}. {m}")

    try:
        idx_str = input("\nEnter number (1-3): ")
        idx = int(idx_str) - 1
        if idx < 0 or idx >= len(MOVEMENT_FOLDERS):
            print("Invalid selection.")
            exit()
        folder_name = MOVEMENT_FOLDERS[idx]
    except ValueError:
        print("Invalid input.")
        exit()

    # 2. Ask for Video Filename
    print(f"\nTarget Folder: {folder_name}")
    target_video = input(f"Enter exact video filename (e.g. '{folder_name} 11.mp4'): ").strip()

    # Check if file exists
    folder_path, _ = find_movement_folder(VIDEOS_ROOT, folder_name)
    video_path = os.path.join(folder_path, target_video)

    if not os.path.exists(video_path):
        print(f"\n[ERROR] File not found: {video_path}")
        print("Please check the name and try again.")
        exit()

    # 3. Process
    print(f"\n[INFO] Processing: {target_video} ...")

    # Use output folder for that movement (so text files land in right place)
    movement_out = os.path.join(OUTPUT_ROOT, folder_name.replace(" ", "_"))  # Fixed Path
    os.makedirs(movement_out, exist_ok=True)

    try:
        feats = process_video(video_path, folder_name, movement_out)

        if feats:
            # --- APPEND TO MASTER EXCEL ---
            # Load existing master if it exists
            if os.path.exists(MASTER_EXCEL_PATH):
                try:
                    master_df = pd.read_excel(MASTER_EXCEL_PATH)
                    # Remove any previous entry for this exact video to prevent duplicates
                    master_df = master_df[master_df["video_name"] != feats["video_name"]]
                except Exception:
                    master_df = pd.DataFrame()
            else:
                master_df = pd.DataFrame()

            # Create DataFrame for current row
            new_row_df = pd.DataFrame([feats])

            # Concatenate
            final_df = pd.concat([master_df, new_row_df], ignore_index=True)

            # Save back to Master
            try:
                final_df.to_excel(MASTER_EXCEL_PATH, index=False)
                print(f"\n[DONE] Added result to MASTER file: {MASTER_EXCEL_PATH}")
                print(new_row_df)
            except PermissionError:
                print(f"\n[ERROR] Could not save to {MASTER_EXCEL_PATH}. Is it open?")
                print("Result data (Save manually):", feats)

        else:
            print("\n[FAILED] Processing completed but no valid data returned.")

    except Exception as e:
        print(f"\n[CRASH] The video caused a crash: {e}")
        traceback.print_exc()


def run():
    print("\n==========================================")
    print("      BATCH VIDEO PROCESSING MODE")
    print("==========================================\n")

    # 1. Loop through each movement folder defined in CONFIG
    for movement_name in MOVEMENT_FOLDERS:
        folder_path = os.path.join(VIDEOS_ROOT, movement_name)

        if not os.path.exists(folder_path):
            print(f"[WARN] Folder not found: {folder_path} - Skipping...")
            continue

        # 2. Find all video files in that folder
        all_files = os.listdir(folder_path)
        # Filter for video extensions to avoid trying to process .txt or .DS_Store files
        video_files = [f for f in all_files if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))]

        # Sort them naturally (1, 2, ... 10) instead of alphabetically (1, 10, 2)
        video_files.sort(key=extract_number)

        print(f"\n------------------------------------------------")
        print(f" Folder: {movement_name} | Found {len(video_files)} videos")
        print(f"------------------------------------------------")

        # Prepare the output directory for this movement
        movement_out = os.path.join(OUTPUT_ROOT, movement_name.replace(" ", "_"))
        os.makedirs(movement_out, exist_ok=True)

        # 3. Process each video in the list
        for i, video_file in enumerate(video_files):
            video_path = os.path.join(folder_path, video_file)
            print(f"\n[{i + 1}/{len(video_files)}] Processing: {video_file} ...")

            try:
                # Run the processing logic
                feats = process_video(video_path, movement_name, movement_out)

                if feats:
                    # --- SAFE EXCEL UPDATE ---
                    # We load and save the Excel file every time.
                    # This is slightly slower but ensures that if the script crashes
                    # on video #50, you don't lose the data for the first 49.
                    if os.path.exists(MASTER_EXCEL_PATH):
                        try:
                            master_df = pd.read_excel(MASTER_EXCEL_PATH)
                            # Remove any previous entry for this exact video to avoid duplicates
                            master_df = master_df[master_df["video_name"] != feats["video_name"]]
                        except Exception:
                            master_df = pd.DataFrame()
                    else:
                        master_df = pd.DataFrame()

                    # Add the new result
                    new_row_df = pd.DataFrame([feats])
                    final_df = pd.concat([master_df, new_row_df], ignore_index=True)

                    # Save back to disk
                    try:
                        final_df.to_excel(MASTER_EXCEL_PATH, index=False)
                        print(f"[SAVED] Master Excel updated successfully.")
                    except PermissionError:
                        print(f"[ERROR] Could not save to {MASTER_EXCEL_PATH}.")
                        print("        Please close the Excel file if it is open!")
                else:
                    print(f"[SKIP] No valid data found for {video_file}")

            except Exception as e:
                print(f"[CRASH] Critical error processing {video_file}: {e}")
                traceback.print_exc()

    print("\n==========================================")
    print("           BATCH PROCESSING COMPLETE      ")
    print("==========================================")


# ---------------------------------------------------------
# Main (Interactive Single Mode)
# ---------------------------------------------------------
if __name__ == "__main__":
    runManual()
    #run()