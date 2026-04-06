import os
import re
import cv2
import numpy as np
import traceback
import pandas as pd
import gc
from datetime import datetime
from mmpose.apis import MMPoseInferencer

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
VIDEOS_ROOT = "../lifting_videos/Augmented/"
MOVEMENT_FOLDERS = ["Broom"]

# Outputs
OUTPUT_ROOT = "../Train_Outputs"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# *** THE MASTER EXCEL FILE ***
MASTER_EXCEL_PATH = os.path.join(OUTPUT_ROOT, "barSpeed_broom.xlsx")

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

# Wrist tracking knobs (Optimized for pre-trimmed videos)
POSE_INIT_FRAMES = 1
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
    if len(valid) == 0: return reps, "UNKNOWN", {"top_y": 0, "bottom_y": 0, "rom": 0}

    top_y = np.percentile(valid, 5)
    bottom_y = np.percentile(valid, 95)
    rom = bottom_y - top_y

    if force_pattern in ("TBT", "BTB"):
        pattern = force_pattern
    else:
        pattern = "BTB" if abs(y[0] - bottom_y) < abs(y[0] - top_y) else "TBT"

    # 2. Find Candidate Bottoms (Local Peaks)
    from scipy.signal import find_peaks

    # PROMINENCE: Ignores wobbles during grinds.
    peak_prominence = 0.15 * rom

    if pattern == "BTB":
        peaks, _ = find_peaks(-y, height=-(bottom_y - 0.60 * rom), distance=int(fps * min_rep_seconds),
                              prominence=peak_prominence)
    else:
        peaks, _ = find_peaks(y, height=top_y + 0.60 * rom, distance=int(fps * min_rep_seconds),
                              prominence=peak_prominence)

    # 3. For each peak, find the true start and true end
    for i, p in enumerate(peaks):
        # Fences to prevent bleed-over
        search_start_limit = max(0, p - int(15.0 * fps)) if i == 0 else peaks[i - 1]
        search_end_limit = min(len(y), p + int(15.0 * fps)) if i == len(peaks) - 1 else peaks[i + 1]

        pre_snippet = y[search_start_limit:p]
        post_snippet = y[p:search_end_limit]

        if len(pre_snippet) == 0 or len(post_snippet) == 0: continue

        if pattern == "TBT":
            start_idx = search_start_limit + np.argmin(pre_snippet)

            # PLATEAU CATCHER: Cuts out hold times by finding first frame in lockout zone
            abs_min = np.min(post_snippet)
            lockout_thresh = abs_min + 0.03 * rom
            end_idx = p + np.argmax(post_snippet <= lockout_thresh)

        else:
            start_idx = search_start_limit + np.argmax(pre_snippet)

            abs_max = np.max(post_snippet)
            floor_thresh = abs_max - 0.03 * rom
            end_idx = p + np.argmax(post_snippet >= floor_thresh)

        # --- VALIDATION ---
        duration = (end_idx - p) / fps if pattern == "TBT" else (p - start_idx) / fps

        if duration >= min_rep_seconds / 2.0:
            if not reps or start_idx > reps[-1][2]:
                reps.append((start_idx, p, end_idx))

    safe_thr = {"top_y": float(top_y), "bottom_y": float(bottom_y), "rom": float(rom)}
    return reps, pattern, safe_thr


def compute_rep_durations(y_smooth, fps, reps, pattern):
    """
    Calculates the duration (seconds) of the Concentric phase.
    """
    y = np.asarray(y_smooth, dtype=float)
    stats = []

    for triple in reps:
        if pattern == "TBT":
            t1, b, t2 = triple
            duration = (t2 - b) / fps
            depth_px = y[b] - min(y[t1], y[t2])
        else:
            b1, t, b2 = triple
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

        # Draw tracking overlay for visualization
        vis_frame = frame.copy()
        if good is not None and len(good) > 0:
            for gx, gy in good:
                cv2.circle(vis_frame, (int(gx), int(gy)), 4, (0, 255, 255), -1)
            cv2.circle(vis_frame, (int(mx), int(my)), 6, (0, 0, 255), -1)
        writer.write(vis_frame)

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
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"[SAVED] Bar speed overlay: {vis_path}")

    ys = np.array(ys, dtype=float)
    y_smooth = moving_average(ys, window=7)
    force = "BTB" if movement_name.lower() == "deadlift" else None

    reps, pattern, thr = detect_reps_state_machine_auto(y_smooth, fps, force_pattern=force)
    stats = compute_rep_durations(y_smooth, fps, reps, pattern)

    if len(stats) < len(reps): reps = reps[:len(stats)]
    if len(reps) < 1:
        print(f"\n[WARN] No valid reps for {base_name}")
        return None

    durations = [s[0] for s in stats]

    first_rep_duration = durations[0]
    last_rep_duration = durations[-1]

    # FLAG 1-REP MAX
    is_one_rep_max = (len(durations) == 1)

    # REP CHANGE CALCULATION
    if is_one_rep_max:
        rep_change_seconds = first_rep_duration
    else:
        rep_change_seconds = last_rep_duration - first_rep_duration

    # txt_path = os.path.join(out_dir, f"{movement_name}__{base_name}__wrist_barspeed.txt")
    # with open(txt_path, "w") as f:
    #     f.write(f"Video: {base_name}\nMovement: {movement_name}\nRep pattern: {pattern}\n")
    #     f.write(f"Detected reps: {len(reps)}\n\n")
    #     for i, dur in enumerate(durations, 1):
    #         f.write(f"Rep {i}: {dur:.2f} seconds\n")
    #     f.write(f"\nRep Change (Last - First): {rep_change_seconds:.2f}s\n")

    # np.save(os.path.join(out_dir, f"{movement_name}__{base_name}__wrist_y.npy"), ys)

    print(
        f"\n[OK] {movement_name} | {base_name}: reps={len(reps)}, first={first_rep_duration:.2f}s, last={last_rep_duration:.2f}s, rep_change={rep_change_seconds:.2f}s, 1RM={is_one_rep_max}")

    return {
        "video_name": os.path.basename(video_path),
        "rep_count": len(reps),
        "first_rep_duration_s": first_rep_duration,
        "last_rep_duration_s": last_rep_duration,
        "rep_change_s": rep_change_seconds,
        "1_rep_max": is_one_rep_max,
    }


def runManual():
    print("\n==========================================")
    print("    SINGLE VIDEO PROCESSING MODE")
    print("==========================================\n")

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

    print(f"\nTarget Folder: {folder_name}")
    target_video = input(f"Enter exact video filename (e.g. '{folder_name} 11.mp4'): ").strip()

    folder_path, _ = find_movement_folder(VIDEOS_ROOT, folder_name)
    video_path = os.path.join(folder_path, target_video)

    if not os.path.exists(video_path):
        print(f"\n[ERROR] File not found: {video_path}")
        print("Please check the name and try again.")
        exit()

    print(f"\n[INFO] Processing: {target_video} ...")

    movement_out = os.path.join(OUTPUT_ROOT, folder_name.replace(" ", "_"))
    # os.makedirs(movement_out, exist_ok=True)

    try:
        feats = process_video(video_path, folder_name, movement_out)

        if feats:
            if os.path.exists(MASTER_EXCEL_PATH):
                try:
                    master_df = pd.read_excel(MASTER_EXCEL_PATH)
                    master_df = master_df[master_df["video_name"] != feats["video_name"]]
                except Exception:
                    master_df = pd.DataFrame()
            else:
                master_df = pd.DataFrame()

            new_row_df = pd.DataFrame([feats])
            final_df = pd.concat([master_df, new_row_df], ignore_index=True)

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

    for movement_name in MOVEMENT_FOLDERS:
        folder_path = os.path.join(VIDEOS_ROOT, movement_name)

        if not os.path.exists(folder_path):
            print(f"[WARN] Folder not found: {folder_path} - Skipping...")
            continue

        all_files = os.listdir(folder_path)
        video_files = [f for f in all_files if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))]
        video_files.sort(key=extract_number)

        print(f"\n------------------------------------------------")
        print(f" Folder: {movement_name} | Found {len(video_files)} videos")
        print(f"------------------------------------------------")

        movement_out = os.path.join(OUTPUT_ROOT, movement_name.replace(" ", "_"))
        # os.makedirs(movement_out, exist_ok=True)

        for i, video_file in enumerate(video_files):
            video_path = os.path.join(folder_path, video_file)
            print(f"\n[{i + 1}/{len(video_files)}] Processing: {video_file} ...")

            try:
                feats = process_video(video_path, movement_name, movement_out)

                if feats:
                    if os.path.exists(MASTER_EXCEL_PATH):
                        try:
                            master_df = pd.read_excel(MASTER_EXCEL_PATH)
                            master_df = master_df[master_df["video_name"] != feats["video_name"]]
                        except Exception:
                            master_df = pd.DataFrame()
                    else:
                        master_df = pd.DataFrame()

                    new_row_df = pd.DataFrame([feats])
                    final_df = pd.concat([master_df, new_row_df], ignore_index=True)

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

            # FORCE CLEAR RAM AFTER EVERY VIDEO TO PREVENT CRASHES
            gc.collect()

    print("\n==========================================")
    print("           BATCH PROCESSING COMPLETE      ")
    print("==========================================")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    # runManual()
    run()