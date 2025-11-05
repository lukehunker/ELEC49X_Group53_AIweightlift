import os
import cv2
import mediapipe as mp
import subprocess
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import glob

# --------------------------
# MediaPipe setup
# --------------------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False,
                             max_num_faces=1,
                             refine_landmarks=True,
                             min_detection_confidence=0.5)

# --------------------------
# Paths
# --------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VIDEO_PATH = os.path.join(BASE_DIR, "videos", "test2.mp4")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OPENFACE_BIN = os.path.join(BASE_DIR, "OpenFace", "build", "bin", "FeatureExtraction")

video_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]

# --------------------------
# Ensure OpenFace CSV exists for this video
# --------------------------
csv_files = glob.glob(os.path.join(OUTPUT_DIR, f"{video_name}*.csv"))

if not csv_files:
    if not os.path.isfile(OPENFACE_BIN):
        raise FileNotFoundError(f"OpenFace binary not found at {OPENFACE_BIN}")

    print(f"No CSV found for {video_name}. Running OpenFace FeatureExtraction...")
    result = subprocess.run([
        OPENFACE_BIN,
        "-f", VIDEO_PATH,
        "-out_dir", OUTPUT_DIR,
        "-2Dfp",
        "-aus"
    ], capture_output=True, text=True)

    print(result.stdout)
    print(result.stderr)

    csv_files = glob.glob(os.path.join(OUTPUT_DIR, f"{video_name}*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"OpenFace did not generate a CSV for {VIDEO_PATH}")

csv_output = csv_files[0]
print(f"Using OpenFace CSV: {csv_output}")
df = pd.read_csv(csv_output)
print("OpenFace CSV columns:", df.columns)

# --------------------------
# Define relevant AUs for exertion
# --------------------------
AU_WEIGHTS = {
    "AU4_r": 0.25,   # brow lowerer
    "AU6_r": 0.2,    # cheek raiser
    "AU7_r": 0.15,   # lid tightener
    "AU10_r": 0.2,   # upper lip raiser
    "AU26_r": 0.2,   # jaw drop
}

AU_COLUMNS = [au for au in AU_WEIGHTS.keys() if au in df.columns]
if not AU_COLUMNS:
    raise ValueError("None of the selected AUs are present in the OpenFace CSV")

# --------------------------
# Smooth AU values
# --------------------------
WINDOW_SIZE = 5
smoothed_aus = {}
for au in AU_COLUMNS:
    smoothed = np.convolve(df[au].values, np.ones(WINDOW_SIZE)/WINDOW_SIZE, mode='same')
    smoothed_aus[au] = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min() + 1e-6)

# --------------------------
# Combine into RPE per frame
# --------------------------
rpe_values = np.zeros(len(df))
for au, weight in AU_WEIGHTS.items():
    if au in smoothed_aus:
        rpe_values += smoothed_aus[au] * weight

# Normalize RPE to 1-10
rpe_values = 1 + 9 * (rpe_values - rpe_values.min()) / (rpe_values.max() - rpe_values.min() + 1e-6)

# --------------------------
# Align AU frames with video frames
# --------------------------
cap_temp = cv2.VideoCapture(VIDEO_PATH)
video_frame_count = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
cap_temp.release()

if len(rpe_values) != video_frame_count:
    f_interp = interp1d(np.linspace(0, 1, len(rpe_values)), rpe_values, kind='linear')
    rpe_values = f_interp(np.linspace(0, 1, video_frame_count))

# --------------------------
# Display video with RPE overlay
# --------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
frame_idx = 0
playback_speed = 100  # ms per frame

print("Press '+' or '-' to adjust speed, 'r' to restart, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    h, w, _ = frame.shape

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        xs = [int(lm.x * w) for lm in face_landmarks.landmark]
        ys = [int(lm.y * h) for lm in face_landmarks.landmark]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(frame, f"Face detected (Frame {frame_idx})", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        rpe_score = rpe_values[frame_idx] if frame_idx < len(rpe_values) else 0
        cv2.putText(frame, f"AU-based RPE: {rpe_score:.1f}", (x_min, y_max + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Face + RPE", frame)
    key = cv2.waitKey(playback_speed) & 0xFF
    if key == ord('+'):
        playback_speed = max(10, playback_speed - 10)
    elif key == ord('-'):
        playback_speed += 10
    elif key == ord('q'):
        break
    elif key == ord('r'):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        continue

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
