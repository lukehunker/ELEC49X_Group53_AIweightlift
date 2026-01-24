import json
import cv2
import numpy as np
import argparse

# -----------------------------
# Command-line args
# -----------------------------
parser = argparse.ArgumentParser(description="Overlay pose keypoints on video")
parser.add_argument("video_in", help="Path to input video (mp4)")
parser.add_argument("--json", default="keypoints.json", help="Path to keypoints JSON")
parser.add_argument("--out", default="output.mp4", help="Path to output video")
args = parser.parse_args()

VIDEO_IN = args.video_in
JSON_IN = args.json
VIDEO_OUT = args.out

# -----------------------------
# Settings
# -----------------------------
KP_RADIUS = 3
SK_THICKNESS = 2

# COCO skeleton WITHOUT head
# Original COCO indices minus 1 (since nose was removed)
SKELETON = [
    (4, 5), (4, 6), (6, 8), (5, 7), (7, 9),     # arms
    (4, 10), (5, 11), (10, 11),                 # torso
    (10, 12), (11, 13), (12, 14), (13, 15)      # legs
]

KP_COLOR = (0, 255, 0)      # green
SK_COLOR = (0, 128, 255)    # orange

# -----------------------------
# Load JSON
# -----------------------------
with open(JSON_IN, "r") as f:
    frames_data = json.load(f)

by_frame = {int(fr["frame_index"]): fr for fr in frames_data}

# -----------------------------
# Open video
# -----------------------------
cap = cv2.VideoCapture(VIDEO_IN)
if not cap.isOpened():
    raise IOError(f"Cannot open video {VIDEO_IN}")

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (w, h))
if not writer.isOpened():
    raise IOError(f"Cannot open video writer {VIDEO_OUT}")

frame_idx = 0

# -----------------------------
# Main loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    fr = by_frame.get(frame_idx)
    if fr:
        for person in fr.get("people", []):
            kpts = np.array(person.get("keypoints", []), dtype=np.float32)

            if kpts.ndim != 2 or kpts.shape[1] != 2:
                continue

            # Draw keypoints
            for x, y in kpts:
                cv2.circle(frame, (int(x), int(y)), KP_RADIUS, KP_COLOR, -1)

            # Draw skeleton
            for a, b in SKELETON:
                if a < len(kpts) and b < len(kpts):
                    ax, ay = kpts[a]
                    bx, by = kpts[b]
                    cv2.line(
                        frame,
                        (int(ax), int(ay)),
                        (int(bx), int(by)),
                        SK_COLOR,
                        SK_THICKNESS
                    )

    cv2.putText(
        frame,
        f"Frame: {frame_idx}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    writer.write(frame)
    frame_idx += 1

cap.release()
writer.release()

print(f"Saved overlay video to {VIDEO_OUT}")
