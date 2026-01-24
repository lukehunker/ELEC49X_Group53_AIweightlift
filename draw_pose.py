import json
import cv2
import numpy as np

# -----------------------------
# COCO skeleton + keypoint list
# -----------------------------
# (All MMPose COCO models output these 17 keypoints)
SKELETON = [
    (0, 1), (1, 3), (0, 2), (2, 4),      # face
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (5, 11), (6, 12), (11, 12),          # torso
    (11, 13), (12, 14), (13, 15), (14, 16)  # legs
]

# -----------------------------
# Load keypoints JSON
# -----------------------------
with open("keypoints.json", "r") as f:
    data = json.load(f)

# Load original image
img = cv2.imread("demo.jpg")
if img is None:
    raise FileNotFoundError("demo.jpg not found")

# Colors
kp_color = (0, 255, 0)     # green keypoints
sk_color = (0, 128, 255)   # orange skeleton

# Thickness
kp_radius = 3
sk_thickness = 2

# -----------------------------
# Process each detected person
# -----------------------------
for person in data:
    keypoints = np.array(person["keypoints"])       # shape (N,17,2) or (17,2) if only one
    scores = np.array(person["keypoint_scores"])

    # If this result contains multiple people, loop over them
    if keypoints.ndim == 3:
        people = range(keypoints.shape[0])
    else:
        keypoints = keypoints[np.newaxis, ...]
        scores = scores[np.newaxis, ...]
        people = [0]

    for i in people:
        kpts = keypoints[i]
        scrs = scores[i]

        # Draw keypoints
        for j, (x, y) in enumerate(kpts):
            if scrs[j] > 0.3:  # ignore low-confidence points
                cv2.circle(img, (int(x), int(y)), kp_radius, kp_color, -1)

        # Draw skeleton
        for a, b in SKELETON:
            if scrs[a] > 0.3 and scrs[b] > 0.3:
                ax, ay = kpts[a]
                bx, by = kpts[b]
                cv2.line(img, (int(ax), int(ay)), (int(bx), int(by)),
                         sk_color, sk_thickness)

# Save visualization
cv2.imwrite("output.jpg", img)
print("Saved pose visualization to output.jpg")
