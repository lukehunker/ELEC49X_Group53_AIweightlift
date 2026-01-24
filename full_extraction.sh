#!/usr/bin/env bash

SQUAT_DIR="/home/hys/ELEC49X_Group53_AIweightlift/lifting_videos/Augmented/Squat"

rm -f outputs.txt

for vid in "$SQUAT_DIR"/s*.mp4; do
  [ -e "$vid" ] || continue

  base="$(basename "$vid" .mp4)"
  kp_file="${base}_keypoints.json"

  echo "Processing $base"

  # only run feature extraction if keypoints don't already exist
  if [ ! -f "$kp_file" ]; then
    python3 feature_extraction.py \
      "$vid" \
      --out "$kp_file"
  else
    echo "  skipping feature extraction (found $kp_file)"
  fi

  python3 show_video.py \
    "$vid" \
    --json "$kp_file" \
    --out "${base}_overlay.mp4"

  python3 preprocess.py "${base}_keypoints.json" | sed "s/^/${base} /" >> outputs.txt


done
