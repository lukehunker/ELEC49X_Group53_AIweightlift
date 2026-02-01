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
VIDEOS_ROOT = r"C:\Users\avedi\mmpose\videos"
MOVEMENT_FOLDERS = ["Bench Press", "Squat", "Deadlift"]

# Outputs
OUTPUT_ROOT = os.path.join(VIDEOS_ROOT, "Wrist_BarSpeed_Results")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# *** THE MASTER EXCEL FILE ***
MASTER_EXCEL_PATH = os.path.join(OUTPUT_ROOT, "Master_Results.xlsx")

print(f"[INFO] Videos root: {VIDEOS_ROOT}")
print(f"[INFO] Output root: {OUTPUT_ROOT}")
print(f"[INFO] Master Excel: {MASTER_EXCEL_PATH}")

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
    return int(m.group()) if m else 10**9

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
        try: out.append(float(v))
        except Exception: return None
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
    if prefer_side == "left": cand = [LEFT_WRIST_IDX, RIGHT_WRIST_IDX]
    elif prefer_side == "right": cand = [RIGHT_WRIST_IDX, LEFT_WRIST_IDX]
    else: cand = [LEFT_WRIST_IDX, RIGHT_WRIST_IDX]
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
    safe_reps = []
    safe_thr = {"top_y": 0, "bottom_y": 0, "rom": 0, "thr_top": 0, "thr_bottom": 0}

    try:
        y = np.asarray(y_smooth, dtype=float)
        valid = y[np.isfinite(y)]
        if valid.size < 5: return safe_reps, "UNKNOWN", safe_thr

        top_y = np.percentile(valid, 5)
        bottom_y = np.percentile(valid, 95)
        rom = bottom_y - top_y
        safe_thr.update({"top_y": float(top_y), "bottom_y": float(bottom_y), "rom": float(rom)})

        if rom < 5.0: return safe_reps, "STATIONARY", safe_thr

        if force_pattern in ("TBT", "BTB"):
            pattern = force_pattern
        else:
            first_idx = int(np.where(np.isfinite(y))[0][0])
            y0 = y[first_idx]
            start_at_bottom = abs(y0 - bottom_y) < abs(y0 - top_y)
            pattern = "BTB" if start_at_bottom else "TBT"

        if pattern == "BTB":
            # DEADLIFT: Top > 35%, Bottom > 50% (halfway point)
            thr_top = top_y + 0.35 * rom      
            thr_bottom = bottom_y - 0.50 * rom 
        else:
            # SQUAT/BENCH
            thr_top = top_y + top_margin * rom
            thr_bottom = bottom_y - bottom_margin * rom
            
        safe_thr.update({"thr_top": float(thr_top), "thr_bottom": float(thr_bottom)})
        min_rep_frames = int(min_rep_seconds * fps)

        reps = []
        if pattern == "TBT": # Squat/Bench
            state = "search_top"
            last_top_idx = None
            bottom_idx = None
            for i, yi in enumerate(y):
                if not np.isfinite(yi): continue
                if state == "search_top":
                    if yi <= thr_top:
                        last_top_idx = i
                        state = "going_down"
                elif state == "going_down":
                    if yi < y[last_top_idx]: last_top_idx = i
                    if yi >= thr_bottom:
                        bottom_idx = i
                        state = "going_up"
                elif state == "going_up":
                    if yi > y[bottom_idx]: bottom_idx = i
                    if yi <= thr_top:
                        end_top_idx = i
                        if (end_top_idx - last_top_idx) >= min_rep_frames:
                            reps.append((last_top_idx, bottom_idx, end_top_idx))
                        last_top_idx = end_top_idx
                        bottom_idx = None
                        state = "going_down"
        else: # Deadlift (BTB)
            state = "search_bottom"
            last_bottom_idx = None
            top_idx = None
            for i, yi in enumerate(y):
                if not np.isfinite(yi): continue
                if state == "search_bottom":
                    if yi >= thr_bottom:
                        last_bottom_idx = i
                        state = "going_up"
                elif state == "going_up":
                    if yi > y[last_bottom_idx]: last_bottom_idx = i
                    if yi <= thr_top:
                        top_idx = i
                        state = "going_down"
                elif state == "going_down":
                    if yi < y[top_idx]: top_idx = i
                    if yi >= thr_bottom:
                        end_bottom_idx = i
                        if (end_bottom_idx - last_bottom_idx) >= min_rep_frames:
                            reps.append((last_bottom_idx, top_idx, end_bottom_idx))
                        last_bottom_idx = end_bottom_idx
                        top_idx = None
                        state = "going_up"

        return reps, pattern, safe_thr
    except Exception as e:
        print(f"[ERROR] Crash in detect_reps: {e}")
        return safe_reps, "ERROR", safe_thr

def compute_rep_speeds(y_smooth, fps, reps, pattern):
    y = np.asarray(y_smooth, dtype=float)
    speeds = []
    for triple in reps:
        if pattern == "TBT":
            t1, b, t2 = triple
            depth_px = y[b] - min(y[t1], y[t2])
            ascent_time = (t2 - b) / fps
        else: 
            b1, t, b2 = triple
            depth_px = max(y[b1], y[b2]) - y[t]
            ascent_time = (t - b1) / fps

        if ascent_time <= 0: continue
        avg_px_s = depth_px / ascent_time
        avg_m_s = (avg_px_s / PIXELS_PER_CM) / 100.0

        if avg_m_s > 3.0: # Filter glitches
            print(f"[WARN] Ignored glitch rep with speed {avg_m_s:.2f} m/s")
            continue
        speeds.append((float(avg_px_s), float(avg_px_s/PIXELS_PER_CM), float(avg_m_s)))
    return speeds

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
                if np.isfinite(ys[-1]) and np.hypot(px-xs[-1], py-ys[-1]) > MAX_RESEED_DIST_PX and lost_counter < 2:
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
    speeds = compute_rep_speeds(y_smooth, fps, reps, pattern)
    
    if len(speeds) < len(reps): reps = reps[:len(speeds)]
    if len(reps) < 1:
        print(f"[WARN] No valid reps for {base_name}")
        return None

    first_m_s = speeds[0][2]
    last_m_s = speeds[-1][2]
    delta_m_s = float(last_m_s - first_m_s)

    txt_path = os.path.join(out_dir, f"{movement_name}__{base_name}__wrist_barspeed.txt")
    with open(txt_path, "w") as f:
        f.write(f"Video: {base_name}\nMovement: {movement_name}\nRep pattern: {pattern}\n")
        f.write(f"Detected reps: {len(reps)}\n\n")
        for i, (triple, (px_s, cm_s, m_s)) in enumerate(zip(reps, speeds), 1):
            f.write(f"Rep {i}: Speed {m_s:.4f} m/s\n")
        f.write(f"\nDelta speed: {delta_m_s:.6f}\n")
    
    np.save(os.path.join(out_dir, f"{movement_name}__{base_name}__wrist_y.npy"), ys)
    print(f"\n[OK] {movement_name} | {base_name}: reps={len(reps)}, first={first_m_s:.3f}, last={last_m_s:.3f}, Δv={delta_m_s:.4f}")

    return {
        "video_name": os.path.basename(video_path),
        "rep_count": len(reps),
        "first_rep_speed_m_s": first_m_s,
        "last_rep_speed_m_s": last_m_s,
        "delta_speed_m_s": delta_m_s,
    }

def run():
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

# ---------------------------------------------------------
# Main (Interactive Single Mode)
# ---------------------------------------------------------
if __name__ == "__main__":
    rows = []
    print("\n==========================================")
    print("    SINGLE VIDEO PROCESSING MODE")
    print("==========================================\n")

    # 1. Ask for Movement
    print("Select Movement Folder:")
    for i, m in enumerate(MOVEMENT_FOLDERS):
        print(f"  {i+1}. {m}")
    
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
    movement_out = os.path.join(OUTPUT_ROOT, folder_name.replace(" ", "_")) # Fixed Path
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