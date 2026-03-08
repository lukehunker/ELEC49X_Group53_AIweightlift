#!/bin/bash
# Stop script on first error
set -e

echo "Upgrading pip and installing setuptools < 70 to keep pkg_resources for MMCV..."
pip install --upgrade pip
pip install "setuptools<70.0.0" wheel

echo "Installing PyTorch CPU (required by mim to find the right MMCV wheel)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo "Installing openmim..."
pip install openmim

echo "Installing mmcv via mim to get pre-built wheels..."
# mim install automatically picks up the correct pre-built wheel based on the torch and python versions
mim install mmcv>=2.1.0

echo "Installing remaining requirements..."
pip install -r requirements.txt
