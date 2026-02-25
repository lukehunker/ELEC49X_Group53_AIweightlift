#!/usr/bin/env bash

SQUAT_DIR="../../lifting_videos/Demo_Videos"
OUTPUT_DIR="../../Train_Outputs/Demo"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

rm -f "$OUTPUT_DIR/demo_outputs.txt"

for vid in "$SQUAT_DIR"/*.mp4 "$SQUAT_DIR"/*.mov; do
  [ -e "$vid" ] || continue

  base="$(basename "$vid")"
  base_noext="${base%.*}"
  kp_file="$OUTPUT_DIR/${base_noext}_keypoints.json"

  echo "Processing $base_noext"

  # only run feature extraction if keypoints don't already exist
  if [ ! -f "$kp_file" ]; then
    python3 feature_extraction_squat.py \
      "$vid" \
      --out "$kp_file"
  else
    echo "  skipping feature extraction (found $kp_file)"
  fi

  python3 show_video.py \
    "$vid" \
    --json "$kp_file" \
    --out "$OUTPUT_DIR/${base_noext}_overlay.mp4"

  python3 preprocess_squat.py "$kp_file" | sed "s/^/${base_noext} /" >> "$OUTPUT_DIR/demo_outputs.txt"


done
