#!/bin/bash

# MMPose Installation Script (Unified with MMDetection)
# Use OpenMMLab's mim tool for proper dependency management

set -e  # Exit on error

echo "============================================================"
echo "MMPose Installation Script"
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
mim install mmpose==0.29.0
mim install "mmdet<3.0"  # Install mmdet too for unified environment

echo ""
echo "Downloading HRNet model files..."
# Download model config and checkpoint
mim download mmpose --config topdown_heatmap_hrnet_w48_coco_256x192 --dest .

echo ""
echo "============================================================"
echo "✓ Installation Complete!"
echo "============================================================"
echo ""
echo "To test the installation, run:"
echo "  python test_pose.py"
echo ""
echo "This will process sample deadlift and squat videos and"
echo "save annotated videos to ../../output/mmpose_test/"
echo ""
