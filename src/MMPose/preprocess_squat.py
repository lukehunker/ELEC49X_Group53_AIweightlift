import json
import numpy as np
import argparse

# =============================
# Args
# =============================
parser = argparse.ArgumentParser(description="Compute D metric from keypoints JSON")
parser.add_argument("json_in", help="Input keypoints JSON file (e.g., s2_keypoints.json)")
args = parser.parse_args()

JSON_IN = args.json_in

# =============================
# Config
# =============================
# Head removed => COCO indices shifted by -1
L_SHOULDER, R_SHOULDER = 4, 5
L_HIP, R_HIP = 10, 11

# Smoothing (for rep signal)
MED_WIN = 5
EMA_ALPHA = 0.35

# Rep detection constraints (frames)
MIN_REP_FRAMES = 35
PROM_FRAC = 0.2   # relative prominence

# Rep validity filtering (RELATIVE units)
MIN_DROP_REL = 0.25        # minimum shoulder drop (in normalized body units)
DROP_FRAC_OF_MAX = 0.5     # keep reps with drop >= this fraction of max drop

# Time normalization
N_SAMPLES = 100

# D metric joint selection (OPTION 1: keep shoulders/arms in JSON, ignore them in D)
# Keep lower body only to avoid bar/rack corrupting upper body
KEEP = [10, 11, 12, 13, 14, 15]  # hips, knees, ankles, feet (head removed 16-joint layout)

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

def find_local_maxima(y, min_dist, prom):
    cand = []
    for i in range(1, len(y) - 1):
        if y[i] > y[i - 1] and y[i] >= y[i + 1]:
            cand.append(i)

    peaks = []
    half = max(3, min_dist // 2)
    for i in cand:
        lo = max(0, i - half)
        hi = min(len(y), i + half + 1)
        if (y[i] - np.min(y[lo:hi])) >= prom:
            peaks.append(i)

    peaks = sorted(peaks, key=lambda i: y[i], reverse=True)
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
      - origin = hip center
      - scale = shoulder width (fallback hip width)
    """
    k = kpts.astype(np.float32).copy()

    hip_center = 0.5 * (k[:, L_HIP] + k[:, R_HIP])
    k -= hip_center[:, None, :]

    shoulder_w = np.linalg.norm(kpts[:, L_SHOULDER] - kpts[:, R_SHOULDER], axis=1)
    hip_w = np.linalg.norm(kpts[:, L_HIP] - kpts[:, R_HIP], axis=1)

    scale = shoulder_w.copy()
    bad = ~np.isfinite(scale) | (scale < 1e-3)
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
# Relative shoulder rep signal
# =============================
shoulder_y = 0.5 * (kpts_norm[:, L_SHOULDER, 1] + kpts_norm[:, R_SHOULDER, 1])
shoulder_y = interp_nans(shoulder_y)

shoulder_y_smooth = median_filter_1d(shoulder_y, MED_WIN)
shoulder_y_smooth = ema_1d(shoulder_y_smooth, EMA_ALPHA)

yr = float(np.max(shoulder_y_smooth) - np.min(shoulder_y_smooth))
prom = PROM_FRAC * yr

bottoms = find_local_maxima(shoulder_y_smooth, MIN_REP_FRAMES, prom)
rep_intervals = [(bottoms[i], bottoms[i + 1]) for i in range(len(bottoms) - 1)]

# =============================
# Filter shallow non-reps (RELATIVE)
# =============================
rep_info = []
for i, (a, b) in enumerate(rep_intervals):
    bottom_y = float(shoulder_y_smooth[a])
    top_y = float(np.min(shoulder_y_smooth[a:b + 1]))
    drop = bottom_y - top_y
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
print("\n--- DEBUG: Detected reps (relative shoulder height) ---")
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

# >>> OPTION 1 CHANGE: ignore shoulders/arms for D <<<
E = E[:, KEEP, :]
L = L[:, KEEP, :]

diff = E - L
D = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=2))))

print("RESULT")
print("First rep:", first_rep)
print("Last rep :", last_rep)
print("D value  :", D)
