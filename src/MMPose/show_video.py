import json
import cv2
import argparse
import numpy as np

# =============================
# Args
# =============================
parser = argparse.ArgumentParser(description="Overlay pose keypoints on video")
parser.add_argument("video_in", help="Path to input video")
parser.add_argument("--json", required=True, help="Path to keypoints JSON file")
parser.add_argument("--out", default="overlay.mp4", help="Output video path")
args = parser.parse_args()

VIDEO_IN = args.video_in
JSON_IN = args.json
VIDEO_OUT = args.out

# =============================
# Load keypoints
# =============================
with open(JSON_IN, "r") as f:
    frames = json.load(f)

frames = sorted(frames, key=lambda fr: fr["frame_index"])

# =============================
# Video I/O
# =============================
cap = cv2.VideoCapture(VIDEO_IN)
if not cap.isOpened():
    raise IOError(f"Cannot open video {VIDEO_IN}")

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (w, h))

# =============================
# Skeleton connections (16 joints, head removed)
# =============================
# Indices for 16-joint layout (COCO with nose removed)
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # eyes, ears
    (4, 5),                          # shoulders
    (4, 6), (6, 8),                  # left arm
    (5, 7), (7, 9),                  # right arm
    (4, 10), (5, 11),                # shoulder to hip
    (10, 11),                        # hips
    (10, 12), (12, 14),              # left leg
    (11, 13), (13, 15),              # right leg
]

# =============================
# Draw overlays
# =============================
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get keypoints for current frame
    if frame_idx < len(frames):
        people = frames[frame_idx].get("people", [])
        if people:
            kpts = np.array(people[0].get("keypoints", []), dtype=np.float32)
            if kpts.shape == (16, 2):
                # Draw skeleton
                for (j1, j2) in SKELETON:
                    if np.all(np.isfinite(kpts[j1])) and np.all(np.isfinite(kpts[j2])):
                        pt1 = tuple(kpts[j1].astype(int))
                        pt2 = tuple(kpts[j2].astype(int))
                        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

                # Draw keypoints
                for j in range(16):
                    if np.all(np.isfinite(kpts[j])):
                        pt = tuple(kpts[j].astype(int))
                        cv2.circle(frame, pt, 4, (0, 0, 255), -1)

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()

print(f"Wrote overlay video to {VIDEO_OUT}")
