#!/bin/bash

# MMDetection Installation Script (Unified with MMPose)
# Use OpenMMLab's mim tool for proper dependency management

set -e  # Exit on error

echo "============================================================"
echo "MMDetection Installation Script"
echo "Using OpenMMLab mim for unified installation"
echo "============================================================"
echo ""

# Install OpenMMLab mim tool
echo "Installing openmim..."
pip install openmim

echo ""
echo "Installing mmcv-full, mmpose, and mmdet with mim..."
# mim handles all dependency resolution automatically
mim install mmcv-full==1.7.0
mim install mmpose==0.29.0  # Install mmpose too for unified environment
mim install "mmdet<3.0"  # Use 2.x for compatibility with mmcv-full 1.7.0

echo ""
echo "Downloading Faster R-CNN model files for MMDet 2.x..."
# Download Faster R-CNN model for object detection
# Note: MMDet 2.x uses different model repository than 3.x
mim download mmdet --config faster_rcnn_r50_fpn_1x_coco --dest . || {
    echo "Downloading model files manually..."
    wget -q https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth -O faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
    wget -q https://raw.githubusercontent.com/open-mmlab/mmdetection/v2.28.2/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py -O faster_rcnn_r50_fpn_1x_coco.py
}

echo ""
echo "============================================================"
echo "✓ Installation Complete!"
echo "============================================================"
echo ""
echo "To test barbell tracking, run:"
echo "  python test_barbell_tracking.py"
echo ""
echo "This will track the barbell in deadlift, squat, and bench press videos"
echo "and save annotated videos to ../../output/mmdet_test/"
echo ""
