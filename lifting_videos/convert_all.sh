#!/bin/bash

OUTPUT_DIR="h264_converted"
mkdir -p "$OUTPUT_DIR"

for f in *.mp4 *.mov; do
    [ -e "$f" ] || continue

    if [ -f "$OUTPUT_DIR/${f%.*}.mp4" ]; then
        echo "$f already processed, skipping"
        continue
    fi

    codec=$(ffprobe -v error -select_streams v:0 \
        -show_entries stream=codec_name \
        -of default=noprint_wrappers=1:nokey=1 "$f")

    if [ "$codec" != "h264" ]; then
        echo "Converting $f ..."

        ffmpeg -probesize 50M -analyzeduration 100M -i "$f" \
            -c:v h264_nvenc -preset p5 -rc vbr -cq 21 -b:v 0 \
            -pix_fmt yuv420p \
            -color_primaries bt709 -color_trc bt709 -colorspace bt709 \
            -movflags +faststart \
            -c:a copy \
            "$OUTPUT_DIR/${f%.*}.mp4"

        if [ $? -eq 0 ]; then
            echo "Finished $f"
        else
            echo "Error converting $f"
        fi
    else
        echo "$f already H264, copying to output folder"
        cp "$f" "$OUTPUT_DIR/${f%.*}.mp4"
    fi
done

echo "All done! Converted files are in $OUTPUT_DIR/"
