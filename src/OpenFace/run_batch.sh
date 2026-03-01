#!/bin/bash
# Auto-runner for batch processing - keeps running until all videos are processed

cd "$(dirname "$0")"

echo "=========================================="
echo "OpenFace Batch Processing (WSL Memory Safe)"
echo "=========================================="
echo ""
echo "Processing 30 videos at a time, fully exiting Python between batches"
echo "to release WSL memory back to Windows."
echo ""

BATCH=0
MAX_BATCHES=30  # 30 batches x 30 videos = 900 videos max

while [ $BATCH -lt $MAX_BATCHES ]; do
    echo ""
    echo "=== Starting Batch #$((BATCH + 1)) ==="
    
    # Always use batch index 0 since we filter out processed videos each time
    python test_openface_batch.py 0
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ ALL VIDEOS PROCESSED SUCCESSFULLY!"
        break
    else
        echo "Batch complete. Continuing..."
        BATCH=$((BATCH + 1))
        sleep 1
    fi
done

if [ $BATCH -eq $MAX_BATCHES ]; then
    echo "Reached maximum batches. Check for issues."
fi
