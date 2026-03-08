#!/bin/bash
# Stop script on first error
set -e

echo "Upgrading pip and installing setuptools < 70 to keep pkg_resources for MMCV..."
pip install --upgrade pip
pip install "setuptools<70.0.0" wheel

echo "Installing PyTorch CPU (required by mim to find the right MMCV wheel)..."
pip install torch==2.1.0+cpu torchvision==0.16.0+cpu --index-url https://download.pytorch.org/whl/cpu

echo "Installing mmcv via explicit pre-compiled wheel link..."
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1.0/index.html

echo "Bypassing build isolation for chumpy (mmpose dependency) to avoid pip import error..."
pip install chumpy --no-build-isolation

echo "Installing remaining requirements..."
pip install -r requirements.txt
