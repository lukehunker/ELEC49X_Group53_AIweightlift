#!/bin/bash

# Modern Facial Strain Analysis Pipeline Installation
# Uses MediaPipe FaceDetection + FaceMesh + ByteTrack
# GPU-accelerated via OpenGL ES (no CUDA conflicts)

set -e

echo "============================================================"
echo "Facial Strain Analysis Pipeline Installation"
echo "============================================================"
echo ""

echo "Installing dependencies..."

# Core packages
pip install mediapipe

# For tracking
pip install filterpy  # Kalman filter for ByteTrack
pip install lap  # Linear assignment problem solver

echo ""
echo "============================================================"
echo "✓ Installation Complete!"
echo "============================================================"
echo ""
echo "Pipeline components:"
echo "  1. MediaPipe FaceDetection - Fast GPU face detection"
echo "  2. ByteTrack - Face tracking across frames"
echo "  3. MediaPipe FaceMesh - Facial landmarks (468 points)"
echo "  4. Custom strain metrics - RPE correlation"
echo ""
echo "GPU Acceleration: OpenGL ES (no CUDA version conflicts)"
echo ""
echo "To test the pipeline, run:"
echo "  python test_facial_strain.py"
echo ""
