#!/bin/bash
# Re-encode problematic videos (AV1, HEVC) to H.264 codec for maximum compatibility

INPUT_DIR="${1:-lifting_videos/Augmented}"
OUTPUT_DIR="${2:-${INPUT_DIR}_h264}"

echo "======================================"
echo "Video Re-encoding Script"
echo "======================================"
echo "Input directory:  $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Check if ffmpeg is available
if ! command -v ffmpeg &> /dev/null; then
    echo "ERROR: ffmpeg not found. Install with:"
    echo "  sudo apt install ffmpeg   (Ubuntu/Debian)"
    echo "  brew install ffmpeg       (macOS)"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Counter
total=0
success=0
failed=0
skipped=0

# Process all video files
shopt -s nullglob  # Don't fail if no files match
for video in "$INPUT_DIR"/*.{mp4,mov,avi,MP4,MOV,AVI}; do
    # Check if file exists (glob may not match anything)
    [ -f "$video" ] || continue
    
    filename=$(basename "$video")
    name="${filename%.*}"
    output="$OUTPUT_DIR/${name}.mp4"
    
    # Skip if already processed
    if [ -f "$output" ]; then
        echo "⏭️  Skipping (already exists): $filename"
        skipped=$((skipped + 1))
        continue
    fi
    
    total=$((total + 1))
    echo ""
    echo "[$total] Processing: $filename"
    
    # Re-encode with H.264 codec
    # -c:v libx264: Use H.264 video codec
    # -crf 23: Quality (18-28, lower = better quality, 23 = good balance)
    # -preset medium: Encoding speed/compression tradeoff
    # -c:a aac: AAC audio codec (universal)
    # -movflags +faststart: Enable streaming (good for web playback)
    if ffmpeg -i "$video" \
        -c:v libx264 \
        -crf 23 \
        -preset medium \
        -c:a aac \
        -b:a 128k \
        -movflags +faststart \
        "$output" \
        -y \
        -loglevel error \
        -stats 2>&1; then
        
        success=$((success + 1))
        
        # Get file sizes
        input_size=$(du -h "$video" | cut -f1)
        output_size=$(du -h "$output" | cut -f1)
        
        echo "   ✓ Success: $input_size → $output_size"
    else
        failed=$((failed + 1))
        echo "   ✗ Failed to encode $filename"
        # Remove partial output file
        rm -f "$output"
    fi
done

echo ""
echo "======================================"
echo "Re-encoding Complete"
echo "======================================"
echo "Total processed: $total"
echo "  Success: $success"
echo "  Failed:  $failed"
echo "  Skipped: $skipped"
echo "======================================"

if [ $success -gt 0 ]; then
    echo ""
    echo "✓ Re-encoded videos saved to: $OUTPUT_DIR"
    echo ""
    echo "Next steps:"
    echo "  1. Test a re-encoded video:"
    echo "     cd src"
    echo "     python test_body_cropping.py ../$OUTPUT_DIR/your_video.mp4"
    echo ""
    echo "  2. Use re-encoded videos in pipeline:"
    echo "     Update video paths to use $OUTPUT_DIR"
fi
