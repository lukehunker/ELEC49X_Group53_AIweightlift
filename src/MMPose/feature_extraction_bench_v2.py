import json
import cv2
import argparse
import numpy as np
from collections import deque

from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules

# -----------------------------
# Args
# -----------------------------
parser = argparse.ArgumentParser(description="Extract pose keypoints from video using MMPose (bench press - OPTIMIZED tracking)")
parser.add_argument("video_in", help="Path to input video (mp4)")
parser.add_argument("--out", default="keypoints.json", help="Output JSON file")
parser.add_argument("--device", default="cuda:0", help="Device to run inference on (e.g. cuda:0 or cpu)")
args = parser.parse_args()

VIDEO_IN = args.video_in
JSON_OUT = args.out
DEVICE = args.device

# -----------------------------
# MMPose init
# -----------------------------
register_all_modules()
config_file = "td-hm_hrnet-w48_8xb32-210e_coco-256x192.py"
checkpoint_file = "td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth"
model = init_model(config_file, checkpoint_file, device=DEVICE)

# -----------------------------
# Indices (head removed => 16 keypoints)
# -----------------------------
L_SHOULDER, R_SHOULDER = 4, 5
L_ELBOW, R_ELBOW = 6, 7
L_WRIST, R_WRIST = 8, 9
L_HIP, R_HIP = 10, 11

# -----------------------------
# OPTIMIZED Tracking Parameters (V3 - Best Balance)
# Keep improvements from v2, but fix over-aggressive stuck detection
# -----------------------------
BOOT_RANGE = 20          
MIN_COMPLETENESS = 0.48  # Balanced
ROI_PAD = 2.8            # Slightly tighter than original
MAX_MISSING = 15         # Allow more frames
EMA_ALPHA = 0.28         # Moderate smoothing
MAX_JUMP_REL = 1.0       # Reasonable for bench
KPT_SCORE_THR = 0.20     

# arm separation constraint (KEEP - prevents L/R collapse)
MIN_ARM_SEP_REL = 0.35  
MIN_ARM_SEP_PX = 18.0   

# Stuck detection - DISABLED FOR WRISTS (they legitimately move slowly in bench)
# Only apply to torso/legs which should be stationary
STUCK_WIN = 12          
STUCK_MOVE_REL = 0.025   # Less sensitive
STUCK_HOLD_MAX = 50      

# Temporal consistency - more lenient for slow bench press movement
VELOCITY_HISTORY = 5     
MAX_VELOCITY_REL = 0.9   # Very lenient - bench is slow
MAX_ACCEL_REL = 0.4      # Very lenient

# Multi-person filtering
MIN_PERSON_SIZE_REL = 0.25  # Slightly more permissive

# Side-profile guard
SIDEVIEW_ELBOWSEP_OVER_SW_THR = 0.45
SIDEVIEW_SW_PX_THR = 20.0

# Periodic global re-detect
FULL_REDETECT_EVERY = 12  # Slightly less frequent
REDETECT_COMP_THR = 0.38  
REDETECT_MISSING_THR = 4

# NEW: Define which joints to apply stuck detection to
# Wrists/elbows move during bench - DON'T mark as stuck
# Torso/legs should be still - OK to mark as stuck
JOINTS_TO_CHECK_STUCK = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]  # Only check torso

# -----------------------------
# Helpers
# -----------------------------
def to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def remove_head(k17):
    return k17[1:, :].astype(np.float32)

def completeness(k16, w, h):
    ok = np.isfinite(k16[:, 0]) & np.isfinite(k16[:, 1])
    if ok.sum() == 0:
        return 0.0
    inb = (k16[:, 0] >= -0.05*w) & (k16[:, 0] <= 1.05*w) & (k16[:, 1] >= -0.05*h) & (k16[:, 1] <= 1.05*h)
    good = ok & inb
    return float(good.sum()) / float(k16.shape[0])

def shoulder_width(k16):
    if np.all(np.isfinite(k16[L_SHOULDER])) and np.all(np.isfinite(k16[R_SHOULDER])):
        d = k16[L_SHOULDER] - k16[R_SHOULDER]
        return float(np.sqrt(d[0]*d[0] + d[1]*d[1]))
    return 0.0

def torso_center(k16):
    pts = []
    for j in [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]:
        if np.all(np.isfinite(k16[j])):
            pts.append(k16[j])
    if len(pts) >= 2:
        return np.mean(np.stack(pts, axis=0), axis=0)
    ok = np.isfinite(k16[:, 0]) & np.isfinite(k16[:, 1])
    if ok.sum() == 0:
        return np.array([np.nan, np.nan], dtype=np.float32)
    return np.mean(k16[ok], axis=0)

def kpt_bbox(k16, w, h, pad_mult=ROI_PAD):
    ok = np.isfinite(k16[:, 0]) & np.isfinite(k16[:, 1])
    if ok.sum() < 4:
        return None
    xs = k16[ok, 0]
    ys = k16[ok, 1]
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())
    cx = 0.5*(x1 + x2)
    cy = 0.5*(y1 + y2)
    bw = (x2 - x1) * pad_mult
    bh = (y2 - y1) * pad_mult
    x1 = max(0.0, cx - 0.5*bw)
    y1 = max(0.0, cy - 0.5*bh)
    x2 = min(float(w - 1), cx + 0.5*bw)
    y2 = min(float(h - 1), cy + 0.5*bh)
    if (x2 - x1) < 5 or (y2 - y1) < 5:
        return None
    return np.array([[x1, y1, x2, y2]], dtype=np.float32)

def person_size(k16, w, h):
    """Estimate person size as fraction of frame area"""
    ok = np.isfinite(k16[:, 0]) & np.isfinite(k16[:, 1])
    if ok.sum() < 4:
        return 0.0
    xs, ys = k16[ok, 0], k16[ok, 1]
    bbox_area = (xs.max() - xs.min()) * (ys.max() - ys.min())
    frame_area = w * h
    return float(bbox_area / frame_area)

def ema(prev, cur, alpha=EMA_ALPHA):
    if prev is None:
        return cur
    out = cur.copy()
    for j in range(cur.shape[0]):
        cx, cy = cur[j]
        px, py = prev[j]
        c_ok = np.isfinite(cx) and np.isfinite(cy)
        p_ok = np.isfinite(px) and np.isfinite(py)
        if c_ok and p_ok:
            out[j, 0] = alpha*cx + (1-alpha)*px
            out[j, 1] = alpha*cy + (1-alpha)*py
        elif (not c_ok) and p_ok:
            out[j] = prev[j]
    return out

def clamp_jumps(prev, cur):
    """Reject crazy jumps by comparing to previous and scaling by shoulder width."""
    if prev is None:
        return cur
    sw = shoulder_width(prev)
    if sw <= 1e-3:
        return cur
    max_jump = MAX_JUMP_REL * sw
    out = cur.copy()
    for j in range(cur.shape[0]):
        if not (np.all(np.isfinite(prev[j])) and np.all(np.isfinite(cur[j]))):
            continue
        d = cur[j] - prev[j]
        if float(d[0]*d[0] + d[1]*d[1]) > max_jump*max_jump:
            out[j] = prev[j]
    return out

def enforce_velocity_consistency(velocity_hist, prev, cur):
    """Reject keypoints with impossible velocities or accelerations (lenient for slow bench movements)"""
    if prev is None or len(velocity_hist) == 0:
        return cur
    
    sw = shoulder_width(prev)
    if sw <= 1e-3:
        return cur
    
    max_vel = MAX_VELOCITY_REL * sw
    max_accel = MAX_ACCEL_REL * sw
    
    out = cur.copy()
    for j in range(cur.shape[0]):
        if not (np.all(np.isfinite(prev[j])) and np.all(np.isfinite(cur[j]))):
            continue
        
        # Current velocity
        vel_cur = cur[j] - prev[j]
        speed_cur = float(np.hypot(vel_cur[0], vel_cur[1]))
        
        # Check velocity magnitude
        if speed_cur > max_vel:
            out[j] = prev[j]
            continue
        
        # Check acceleration
        if len(velocity_hist[j]) > 0:
            vel_prev = velocity_hist[j][-1]
            accel = vel_cur - vel_prev
            accel_mag = float(np.hypot(accel[0], accel[1]))
            
            if accel_mag > max_accel:
                out[j] = prev[j]
    
    return out

def enforce_arm_separation(prev, cur):
    """Prevent L/R arm collapse"""
    if prev is None:
        return cur

    sw = shoulder_width(prev)
    thr = max(MIN_ARM_SEP_PX, MIN_ARM_SEP_REL * sw) if sw > 1e-3 else MIN_ARM_SEP_PX
    thr2 = thr * thr

    out = cur.copy()
    pairs = [(L_ELBOW, R_ELBOW), (L_WRIST, R_WRIST)]

    for a, b in pairs:
        if not (np.all(np.isfinite(out[a])) and np.all(np.isfinite(out[b]))):
            continue

        d = out[a] - out[b]
        if float(d[0]*d[0] + d[1]*d[1]) >= thr2:
            continue

        if np.all(np.isfinite(prev[a])) and np.all(np.isfinite(prev[b])):
            da = out[a] - prev[a]
            db = out[b] - prev[b]
            ma = float(da[0]*da[0] + da[1]*da[1])
            mb = float(db[0]*db[0] + db[1]*db[1])
            if ma > mb:
                out[a] = prev[a]
            else:
                out[b] = prev[b]
        else:
            if np.all(np.isfinite(prev[a])):
                out[a] = prev[a]
            if np.all(np.isfinite(prev[b])):
                out[b] = prev[b]

    return out

def apply_stuck_gate_selective(prev, cur, motion_hist, stuck_count):
    """FIXED: Only check torso joints for stuckness - arms SHOULD move in bench press!"""
    if prev is None:
        return cur

    sw = shoulder_width(prev)
    if sw <= 1e-3:
        return cur

    thr = STUCK_MOVE_REL * sw
    out = cur.copy()

    # Only check specific joints (NOT wrists/elbows)
    for j in JOINTS_TO_CHECK_STUCK:
        if np.all(np.isfinite(prev[j])) and np.all(np.isfinite(out[j])):
            d = out[j] - prev[j]
            step = float(np.sqrt(d[0]*d[0] + d[1]*d[1]))
        else:
            step = np.nan

        hist = motion_hist[j]
        hist.append(step)
        if len(hist) > STUCK_WIN:
            hist.pop(0)

        vals = [v for v in hist if np.isfinite(v)]
        if len(vals) >= max(5, STUCK_WIN // 3):
            avg_step = float(np.mean(vals))
            if avg_step < thr:
                stuck_count[j] += 1
                out[j] = prev[j]
                if stuck_count[j] >= STUCK_HOLD_MAX:
                    out[j] = np.array([np.nan, np.nan], dtype=np.float32)
            else:
                stuck_count[j] = 0
        else:
            stuck_count[j] = 0

    return out

def likely_side_view(k16):
    """Heuristic: if L/R elbows are close compared to shoulder width, the view is likely side-on."""
    sw = shoulder_width(k16)
    if sw <= 1e-3:
        return True
    if sw < SIDEVIEW_SW_PX_THR:
        return True

    elbow_sep = np.nan
    if np.all(np.isfinite(k16[L_ELBOW])) and np.all(np.isfinite(k16[R_ELBOW])):
        d = k16[L_ELBOW] - k16[R_ELBOW]
        elbow_sep = float(np.hypot(d[0], d[1]))

    if np.isfinite(elbow_sep):
        return (elbow_sep / sw) < SIDEVIEW_ELBOWSEP_OVER_SW_THR

    return False

def collect_candidates(results, w, h):
    cands = []
    for ds in results:
        inst = getattr(ds, "pred_instances", None)
        if inst is None:
            continue

        kpts = to_numpy(inst.keypoints)
        if kpts.ndim == 2 and kpts.shape[0] == 17:
            kpts = kpts[None, ...]

        scores = None
        if hasattr(inst, "keypoint_scores"):
            scores = to_numpy(inst.keypoint_scores)
            if scores.ndim == 1:
                scores = scores[None, ...]

        for p in range(kpts.shape[0]):
            k17 = kpts[p]
            k16 = remove_head(k17)

            comp = completeness(k16, w, h)
            sw = shoulder_width(k16)
            ctr = torso_center(k16)
            size = person_size(k16, w, h)

            mean_score = None
            if scores is not None and p < scores.shape[0]:
                sc16 = scores[p][1:]
                mean_score = float(np.nanmean(sc16))
                low = sc16 < KPT_SCORE_THR
                k16 = k16.copy()
                k16[low] = np.nan

                comp = completeness(k16, w, h)
                sw = shoulder_width(k16)
                ctr = torso_center(k16)

            size_term = min(sw / 60.0, 2.5)
            score = comp + 0.25 * size_term
            if mean_score is not None:
                score += 0.15 * mean_score
            
            # Boost score for larger people (prefer lifter over spotter)
            score += 0.3 * size

            cands.append({
                "k16": k16,
                "score": float(score),
                "comp": float(comp),
                "sw": float(sw),
                "center": ctr,
                "size": float(size),
                "mean_kpt_score": mean_score
            })
    return cands

def pick_best(cands, prev_center=None, w=None, h=None):
    if not cands:
        return None
    
    # Filter out small people (likely spotters or noise)
    if w is not None and h is not None:
        cands = [c for c in cands if c["size"] >= MIN_PERSON_SIZE_REL]
    
    if not cands:
        return None

    if prev_center is not None and np.all(np.isfinite(prev_center)):
        def dist2(c):
            d = c["center"] - prev_center
            return float(d[0]*d[0] + d[1]*d[1])
        cands = sorted(cands, key=lambda c: (dist2(c), -c["score"]))
        return cands[0]["k16"]

    cands = sorted(cands, key=lambda c: c["score"], reverse=True)
    return cands[0]["k16"]

# -----------------------------
# Video open
# -----------------------------
cap = cv2.VideoCapture(VIDEO_IN)
if not cap.isOpened():
    raise IOError(f"Cannot open video {VIDEO_IN}")

n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if n_frames <= 0:
    n_frames = 0

# -----------------------------
# Bootstrap from mid-video
# -----------------------------
prev_k16 = None
prev_center = None

if n_frames > 0:
    mid = n_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, mid - BOOT_RANGE))
    best = None
    best_s = -1e9

    for _ in range(2 * BOOT_RANGE + 1):
        ret, frame = cap.read()
        if not ret:
            break
        results = inference_topdown(model, frame)
        cands = collect_candidates(results, w, h)
        for c in cands:
            if c["comp"] < MIN_COMPLETENESS:
                continue
            if c["score"] > best_s:
                best_s = c["score"]
                best = c["k16"]

    if best is not None:
        prev_k16 = best
        prev_center = torso_center(prev_k16)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# per-joint motion tracking
motion_hist = [[] for _ in range(16)]
stuck_count = np.zeros(16, dtype=np.int32)

# velocity tracking for temporal consistency
velocity_hist = [deque(maxlen=VELOCITY_HISTORY) for _ in range(16)]

# -----------------------------
# Main loop
# -----------------------------
output = []
frame_idx = 0
missing = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    force_global = (frame_idx % FULL_REDETECT_EVERY == 0) or (missing >= REDETECT_MISSING_THR)

    bboxes = None
    if (not force_global) and (prev_k16 is not None):
        bboxes = kpt_bbox(prev_k16, w, h, pad_mult=ROI_PAD)

    if bboxes is not None:
        results = inference_topdown(model, frame, bboxes=bboxes)
    else:
        results = inference_topdown(model, frame)

    cands = collect_candidates(results, w, h)
    chosen = pick_best(cands, prev_center=prev_center, w=w, h=h)

    if chosen is not None and (not force_global):
        comp_now = completeness(chosen, w, h)
        if comp_now < REDETECT_COMP_THR:
            results2 = inference_topdown(model, frame)
            cands2 = collect_candidates(results2, w, h)
            chosen2 = pick_best(cands2, prev_center=prev_center, w=w, h=h)
            if chosen2 is not None:
                comp2 = completeness(chosen2, w, h)
                if comp2 > comp_now:
                    chosen = chosen2

    if chosen is None:
        missing += 1
        if prev_k16 is not None and missing <= MAX_MISSING:
            chosen = prev_k16.copy()
        else:
            chosen = np.full((16, 2), np.nan, dtype=np.float32)
    else:
        missing = 0
        
        # Apply filtering pipeline
        chosen = clamp_jumps(prev_k16, chosen)
        
        # Temporal consistency check
        chosen = enforce_velocity_consistency(velocity_hist, prev_k16, chosen)
        
        # Update velocity history
        if prev_k16 is not None:
            for j in range(16):
                if np.all(np.isfinite(prev_k16[j])) and np.all(np.isfinite(chosen[j])):
                    vel = chosen[j] - prev_k16[j]
                    velocity_hist[j].append(vel)
        
        # Enforce L/R arm separation (only when NOT side-view)
        if not likely_side_view(chosen):
            chosen = enforce_arm_separation(prev_k16, chosen)
        
        # FIXED: Selective stuck detection - only torso
        chosen = apply_stuck_gate_selective(prev_k16, chosen, motion_hist, stuck_count)
        
        # Finally apply EMA smoothing
        chosen = ema(prev_k16, chosen, alpha=EMA_ALPHA)

    prev_k16 = chosen
    prev_center = torso_center(chosen)

    output.append({
        "frame_index": frame_idx,
        "people": [{"keypoints": chosen.tolist()}]
    })
    frame_idx += 1

cap.release()

with open(JSON_OUT, "w") as f:
    json.dump(output, f, indent=2)

print(f"[OPTIMIZED v3] Wrote pose results for {frame_idx} frames to {JSON_OUT}")
