import os
import cv2
import mediapipe as mp
import subprocess
import pandas as pd
import tempfile

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

# --------------------------
# Run OpenFace FeatureExtraction on video
# --------------------------
import glob
csv_files = glob.glob(os.path.join(OUTPUT_DIR, "*.csv"))
if not csv_files:
    raise FileNotFoundError("No CSV output found from OpenFace.")
csv_output = csv_files[0]
df = pd.read_csv(csv_output)


subprocess.run([
    OPENFACE_BIN,
    "-f", VIDEO_PATH,
    "-out_dir", OUTPUT_DIR,
    "-2Dfp",  # output 2D landmarks
    "-aus"    # output Action Units
])

# Load OpenFace CSV
df = pd.read_csv(csv_output)
print("OpenFace CSV columns:", df.columns)

# --------------------------
# Display video with MediaPipe bounding boxes
# --------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = frame.shape

        # Bounding box from landmarks
        xs = [int(lm.x * w) for lm in face_landmarks.landmark]
        ys = [int(lm.y * h) for lm in face_landmarks.landmark]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # Draw bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(frame, "Face detected", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Optional: get "strain" from OpenFace embeddings if available
        if "fc6_1" in df.columns:  # example embedding column from FeatureExtraction
            rep = df.loc[frame_idx, [c for c in df.columns if "fc6_" in c]].values
            strain_score = sum(abs(v) for v in rep)
            cv2.putText(frame, f"Strain: {strain_score:.2f}", (x_min, y_max + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Face + AU Preview", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
