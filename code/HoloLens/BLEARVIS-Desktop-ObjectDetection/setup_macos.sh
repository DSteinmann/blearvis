#!/bin/bash

# BLEARVIS Environment Setup Script for macOS ARM64
# This script sets up the conda environment for YOLO v7/v11 object detection

echo "🔧 Setting up BLEARVIS environment for macOS ARM64..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed. Please install Miniconda or Anaconda first."
    echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create the environment
echo "📦 Creating conda environment..."
conda env create -f blearvis-env.yaml

# Activate the environment
echo "✅ Environment created successfully!"
echo "🔄 To activate the environment, run:"
echo "    conda activate blearvis-env"
echo ""
echo "🧪 To test YOLO v11, run:"
echo "    cd modules/YoloModule/app"
echo "    python test_yolo11.py"
echo ""
echo "🚀 To run the application:"
echo "    python main.py"