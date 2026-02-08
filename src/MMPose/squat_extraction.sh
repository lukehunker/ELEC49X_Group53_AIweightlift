#!/usr/bin/env bash

SQUAT_DIR="/home/hys/ELEC49X_Group53_AIweightlift/lifting_videos/Augmented/Squat"

rm -f squat_outputs.txt

for vid in "$SQUAT_DIR"/*.mp4 "$SQUAT_DIR"/*.mov; do
  [ -e "$vid" ] || continue

  base="$(basename "$vid")"
  base_noext="${base%.*}"
  kp_file="${base_noext}_keypoints.json"

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
    --out "${base_noext}_overlay.mp4"

  python3 preprocess_squat.py "${base_noext}_keypoints.json" | sed "s/^/${base_noext} /" >> squat_outputs.txt


done
