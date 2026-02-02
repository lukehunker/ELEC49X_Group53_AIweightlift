#!/usr/bin/env bash

DEADLIFT_DIR="/home/hys/ELEC49X_Group53_AIweightlift/lifting_videos/Augmented/Deadlift"

rm -f deadlift_outputs.txt

for vid in "$DEADLIFT_DIR"/*.mp4 "$DEADLIFT_DIR"/*.mov; do
  [ -e "$vid" ] || continue

  base="$(basename "$vid")"
  base_noext="${base%.*}"
  kp_file="${base_noext}_keypoints.json"

  echo "Processing $base_noext"

  # only run feature extraction if keypoints don't already exist
  if [ ! -f "$kp_file" ]; then
    python3 feature_extraction_deadlift.py \
      "$vid" \
      --out "$kp_file"
  else
    echo "  skipping feature extraction (found $kp_file)"
  fi

  python3 show_video.py \
    "$vid" \
    --json "$kp_file" \
    --out "${base_noext}_overlay.mp4"

  python3 preprocess_deadlift.py "${base_noext}_keypoints.json" | sed "s/^/${base_noext} /" >> deadlift_outputs.txt


done
