# MMPose Body Pose Estimation

This module uses MMPose to detect body keypoints (17 COCO format keypoints) in weightlifting videos.

## Quick Start

### 1. Installation

```bash
cd src/MMPose
bash install.sh
```

This will install:
- MMPose 0.29.0 (compatible version)
- mmcv-full 1.7.0
- numpy <2.0, opencv-python <4.10
- Download HRNet-W48 model files (~243MB)

### 2. Run Test

```bash
python test_pose.py
```

**Interactive Controls:**
- **SPACE**: Pause/Resume playback
- **+ or ]**: Speed up (decrease delay between frames)
- **- or [**: Slow down (increase delay between frames)
- **R**: Restart/Loop video from beginning
- **Q**: Skip to next video
- **ESC**: Exit completely

**Features:**
- Real-time visualization window showing keypoints detected by MMPose on person
- Automatic video looping - videos repeat until you press Q
- Adjustable playback speed for detailed inspection
- Saves annotated videos to `../../output/mmpose_test/`

## What It Does

- **Detects 17 body keypoints** per frame (COCO format) using MMPose HRNet model
- **Real-time display** shows skeleton overlay while processing with interactive controls
- **Visualizes skeleton** on saved output videos
- **Processes every 30th frame** (configurable in test_pose.py)
- **GPU accelerated** (if CUDA available)

**How MMPose Works:**
MMPose uses the HRNet-W48 neural network to analyze each video frame and detect 17 body keypoints on the person (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles). These keypoints are then connected to form a skeleton overlay.

## Output

**Live Display:**
- Real-time window shows body keypoints being detected
- Larger keypoint dots (radius 8) and thicker skeleton lines (thickness 4)
- Frame counter and keypoint count overlay

**Saved Videos:**
Annotated MP4 files showing:
- Green skeleton lines connecting keypoints
- Red dots for each detected keypoint
- Frame counter and keypoint information

Example output location: `output/mmpose_test/Deadlift_12_pose.mp4`

## Body Keypoints Detected

```
0: Nose          6: Right Shoulder   12: Right Hip
1: Left Eye      7: Right Elbow      13: Right Knee
2: Right Eye     8: Right Wrist      14: Right Ankle
3: Left Ear      9: Left Hip         15: Left Knee
4: Right Ear    10: Left Knee        16: Left Ankle
5: Left Shoulder 11: Left Hip
```

## Model

**HRNet-W48** (High-Resolution Net)
- Input: 256x192 pixels
- Trained on: COCO dataset
- Files: `hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth` (243MB)

## Configuration

Edit `test_pose.py` to adjust:

```python
TEST_FRAME_INTERVAL = 30   # Process every Nth frame
MAX_FRAMES = 150           # Max frames per video
SHOW_VISUALIZATION = True  # Show real-time display window
```

**Visualization Settings (lines 173-175):**
```python
radius=8,      # Keypoint dot size
thickness=4,   # Skeleton line thickness
```

## Requirements

- Python 3.7+
- PyTorch with CUDA 11.8 (recommended)
- 2GB+ GPU memory (for processing)
- Model files downloaded by install.sh

## Troubleshooting

**Model files not found:**
```bash
mim download mmpose --config topdown_heatmap_hrnet_w48_coco_256x192 --dest .
```

**No videos found:**
- Check that videos exist in `lifting_videos/Raw/Deadlift/` and `lifting_videos/Raw/Squat/`

**Import errors:**
- Re-run `bash install.sh`
- Ensure virtual environment is activated

## Notes

- Uses older MMPose (0.29.0) for numpy 1.x compatibility
- Assumes person occupies full frame (no person detection)
- For production: add MMDetection for person bounding box detection first
