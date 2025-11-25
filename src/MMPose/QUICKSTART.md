# MMPose Quick Reference

## Files in src/MMPose/

```
├── README.md                                      # Full documentation
├── install.sh                                     # Installation script
├── test_pose.py                                   # Main test script
├── hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth  # Model weights (243MB)
└── topdown_heatmap_hrnet_w48_coco_256x192.py     # Model config
```

## Usage

### Install (first time only)
```bash
cd src/MMPose
bash install.sh
```

### Run
```bash
python test_pose.py
```

**What You'll See:**
- Real-time window displaying body keypoints on the person
- Skeleton overlay with 17 keypoints (nose, eyes, shoulders, elbows, wrists, hips, knees, ankles)
- Frame counter showing progress
- Press 'Q' to skip to next video

### Output
- Location: `../../output/mmpose_test/`
- Format: Annotated MP4 videos with skeleton overlay
- Keypoints: 17 body points (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles)

## Customize Processing

Edit `test_pose.py` lines 21-22:
```python
TEST_FRAME_INTERVAL = 30  # Process every Nth frame (30 = 1fps at 30fps video)
MAX_FRAMES = 150          # Maximum frames to process per video
```

## What It Does

1. Finds videos in `lifting_videos/Raw/Deadlift/` and `lifting_videos/Raw/Squat/`
2. Loads HRNet-W48 pose estimation model
3. Processes frames and detects 17 body keypoints
4. Draws skeleton overlay on video
5. Saves annotated output

## Dependencies Installed

- mmpose 0.29.0
- mmcv-full 1.7.0  
- numpy <2.0
- opencv-python <4.10
- openmim

## System Requirements

- Python 3.7+
- PyTorch with CUDA 11.8 (GPU highly recommended)
- 2GB+ GPU memory
- ~500MB disk space (for model + outputs)
