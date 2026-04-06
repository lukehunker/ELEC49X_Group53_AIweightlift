#!/bin/bash
# Unified batch feature extraction for Bar Tracking, MMPose, and OpenFace
# Input folder: lifting_videos/Augmented/Broom
# Output: CSVs for each model in output/batch_broom/

set -e

INPUT_DIR="../lifting_videos/Augmented/Broom"
OUTPUT_DIR="../output/batch_broom"
mkdir -p "$OUTPUT_DIR"

# 1. Bar Tracking
echo "[Bar Tracking] Extracting bar speed features..."
for vid in "$INPUT_DIR"/*.mp4; do
    echo "  Processing $vid"
    python3 ../src/Bar_Tracking/barspeed_to_excel.py "$vid" Squat "$OUTPUT_DIR"
done

echo "[Bar Tracking] Done."

# 2. MMPose (Squat)
echo "[MMPose] Extracting pose features..."
for vid in "$INPUT_DIR"/*.mp4; do
    base=$(basename "$vid" .mp4)
    python3 ../src/MMPose/feature_extraction_squat.py "$vid" --out "$OUTPUT_DIR/${base}_keypoints.json"
done

echo "[MMPose] Done."
done
echo "[OpenFace] Done."

# 3. OpenFace (Facial features, full pipeline)
echo "[OpenFace] Extracting facial features using full OpenFace pipeline..."
cd ../src/OpenFace
./run_batch.sh
cd -
# Copy the resulting openface_features_all.csv to the output folder for convenience
cp ../../Train_Outputs/openface_features_all.csv "$OUTPUT_DIR/openface_features_all.csv"
echo "[OpenFace] Done."

echo "All feature extraction complete. Results in $OUTPUT_DIR/"
