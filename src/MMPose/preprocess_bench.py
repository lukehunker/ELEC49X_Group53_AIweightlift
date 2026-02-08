import json
import numpy as np
import argparse

# =============================
# Args
# =============================
parser = argparse.ArgumentParser(description="Compute D metric from keypoints JSON (bench press)")
parser.add_argument("json_in", help="Input keypoints JSON file (e.g., bench1_keypoints.json)")
args = parser.parse_args()

JSON_IN = args.json_in

# =============================
# Config (Bench Press Specific)
# =============================
# Head removed => COCO indices shifted by -1
L_SHOULDER, R_SHOULDER = 4, 5
L_ELBOW, R_ELBOW = 6, 7
L_WRIST, R_WRIST = 8, 9
L_HIP, R_HIP = 10, 11

# Smoothing (for rep signal)
MED_WIN = 5
EMA_ALPHA = 0.35

# Rep detection constraints (frames)
MIN_REP_FRAMES = 25  # Bench press reps tend to be faster than squats
PROM_FRAC = 0.15     # slightly lower prominence for bench

# Rep validity filtering (RELATIVE units)
MIN_DROP_REL = 0.20        # minimum wrist drop (in normalized body units)
DROP_FRAC_OF_MAX = 0.5     # keep reps with drop >= this fraction of max drop

# Time normalization
N_SAMPLES = 100

# D metric joint selection - for bench press, focus on upper body
# Keep shoulders, elbows, wrists (exclude head, hips, legs)
KEEP = [4, 5, 6, 7, 8, 9]  # shoulders, elbows, wrists (head removed 16-joint layout)

# =============================
# Helpers
# =============================
def interp_nans(x):
    idx = np.arange(len(x))
    good = np.isfinite(x)
    if good.sum() == 0:
        return np.zeros_like(x)
    return np.interp(idx, idx[good], x[good])

def median_filter_1d(x, win):
    if win < 3:
        return x.copy()
    if win % 2 == 0:
        win += 1
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.array([np.median(xp[i:i+win]) for i in range(len(x))], dtype=np.float32)

def ema_1d(x, alpha):
    out = np.empty_like(x, dtype=np.float32)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i-1]
    return out

def find_local_minima(y, min_dist, prom):
    """Find local minima (for bench press, bar lowest = wrists lowest)"""
    cand = []
    for i in range(1, len(y) - 1):
        if y[i] < y[i - 1] and y[i] <= y[i + 1]:
            cand.append(i)

    peaks = []
    half = max(3, min_dist // 2)
    for i in cand:
        lo = max(0, i - half)
        hi = min(len(y), i + half + 1)
        if (np.max(y[lo:hi]) - y[i]) >= prom:
            peaks.append(i)

    peaks = sorted(peaks, key=lambda i: y[i])
    chosen = []
    for i in peaks:
        if all(abs(i - j) >= min_dist for j in chosen):
            chosen.append(i)

    return sorted(chosen)

def resample_sequence(seq, n):
    T = seq.shape[0]
    if T == 1:
        return np.repeat(seq, n, axis=0)
    x_old = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, n)
    out = np.zeros((n, seq.shape[1], 2), dtype=np.float32)
    for j in range(seq.shape[1]):
        for d in range(2):
            out[:, j, d] = np.interp(x_new, x_old, seq[:, j, d])
    return out

def normalize_pose(kpts):
    """
    Normalize pose into body-relative coordinates:
      - origin = shoulder center (bench press is upper body focused)
      - scale = shoulder width
    """
    k = kpts.astype(np.float32).copy()

    shoulder_center = 0.5 * (k[:, L_SHOULDER] + k[:, R_SHOULDER])
    k -= shoulder_center[:, None, :]

    shoulder_w = np.linalg.norm(kpts[:, L_SHOULDER] - kpts[:, R_SHOULDER], axis=1)
    
    scale = shoulder_w.copy()
    bad = ~np.isfinite(scale) | (scale < 1e-3)
    
    # Fallback: use hip width if shoulder width is bad
    if bad.any():
        hip_w = np.linalg.norm(kpts[:, L_HIP] - kpts[:, R_HIP], axis=1)
        scale[bad] = hip_w[bad]
    
    scale = np.clip(scale, 1e-3, None)

    k /= scale[:, None, None]
    return k

# =============================
# Load JSON -> (T,16,2)
# =============================
with open(JSON_IN, "r") as f:
    frames = json.load(f)

frames = sorted(frames, key=lambda fr: fr["frame_index"])

kpts = []
for fr in frames:
    people = fr.get("people", [])
    if not people:
        kpts.append(np.full((16, 2), np.nan, dtype=np.float32))
        continue

    kp = np.array(people[0].get("keypoints", []), dtype=np.float32)
    if kp.shape != (16, 2):
        kp = np.full((16, 2), np.nan, dtype=np.float32)

    kpts.append(kp)

kpts = np.stack(kpts, axis=0)  # (T,16,2)

# =============================
# Normalize FIRST
# =============================
kpts_norm = normalize_pose(kpts)

# =============================
# Wrist rep signal (for bench press)
# =============================
# Use average wrist height as rep signal (lower = bottom of rep)
wrist_y = 0.5 * (kpts_norm[:, L_WRIST, 1] + kpts_norm[:, R_WRIST, 1])
wrist_y = interp_nans(wrist_y)

wrist_y_smooth = median_filter_1d(wrist_y, MED_WIN)
wrist_y_smooth = ema_1d(wrist_y_smooth, EMA_ALPHA)

yr = float(np.max(wrist_y_smooth) - np.min(wrist_y_smooth))
prom = PROM_FRAC * yr

# Find local minima (bottom of bench press reps)
bottoms = find_local_minima(wrist_y_smooth, MIN_REP_FRAMES, prom)
rep_intervals = [(bottoms[i], bottoms[i + 1]) for i in range(len(bottoms) - 1)]

# =============================
# Filter shallow non-reps (RELATIVE)
# =============================
rep_info = []
for i, (a, b) in enumerate(rep_intervals):
    bottom_y = float(wrist_y_smooth[a])
    top_y = float(np.max(wrist_y_smooth[a:b + 1]))
    drop = top_y - bottom_y  # For bench: higher - lower (arms extend up)
    rep_info.append((i, a, b, top_y, bottom_y, drop))

max_drop = max((r[-1] for r in rep_info), default=0.0)
drop_thr = max(MIN_DROP_REL, DROP_FRAC_OF_MAX * max_drop)

valid = [
    (i, a, b, top_y, bottom_y, drop)
    for (i, a, b, top_y, bottom_y, drop) in rep_info
    if drop >= drop_thr
]

# =============================
# Debug output
# =============================
print("\n--- DEBUG: Detected reps (wrist height for bench press) ---")
print("JSON:", JSON_IN)
print("Bottom frames:", bottoms)
for (i, a, b, top_y, bottom_y, drop) in rep_info:
    tag = "KEEP" if any(i == v[0] for v in valid) else "DROP"
    print(
        f"{tag} Rep {i}: frames {a}->{b} ({b-a} frames), "
        f"top={top_y:.3f}, bottom={bottom_y:.3f}, drop={drop:.3f}"
    )
print(f"Max drop={max_drop:.3f} | threshold={drop_thr:.3f}")
print("-----------------------------------------------\n")

rep_intervals = [(a, b) for (_, a, b, *_rest) in valid]

# =============================
# Handle short videos safely
# =============================
if len(rep_intervals) < 2:
    print(f"SKIP: only {len(rep_intervals)} valid rep(s) detected.")
    raise SystemExit(0)

first_rep = rep_intervals[0]
last_rep = rep_intervals[-1]

if first_rep == last_rep:
    print("SKIP: first and last rep are identical.")
    raise SystemExit(0)

# =============================
# Compute D (already normalized)
# =============================
E = resample_sequence(kpts_norm[first_rep[0]:first_rep[1] + 1], N_SAMPLES)
L = resample_sequence(kpts_norm[last_rep[0]:last_rep[1] + 1], N_SAMPLES)

# Focus on upper body for bench press
E = E[:, KEEP, :]
L = L[:, KEEP, :]

diff = E - L
D = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=2))))

print("RESULT")
print("First rep:", first_rep)
print("Last rep :", last_rep)
print("D value  :", D)
