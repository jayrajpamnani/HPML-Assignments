#!/bin/bash
# Simple script to build and run the convolution code
# Run this directly: bash run.sh

echo "=========================================="
echo "Convolution Benchmark - Part C"
echo "Date: $(date)"
echo "=========================================="
echo ""

# Auto-detect CUDA path if not set
if [ -z "$CUDA_PATH" ]; then
    if command -v nvcc &> /dev/null; then
        CUDA_PATH=$(dirname $(dirname $(which nvcc)))
        echo "Auto-detected CUDA_PATH: $CUDA_PATH"
    else
        CUDA_PATH="/usr/local/cuda"
        echo "Using default CUDA_PATH: $CUDA_PATH"
    fi
    export CUDA_PATH
fi

# Set GPU architecture if not set (default to sm_70)
if [ -z "$GPU_ARCH" ]; then
    GPU_ARCH="sm_70"
    echo "Using default GPU_ARCH: $GPU_ARCH"
    echo "To override, set GPU_ARCH environment variable (e.g., export GPU_ARCH=sm_80)"
fi
export GPU_ARCH

echo ""
echo "Configuration:"
echo "  CUDA_PATH: $CUDA_PATH"
echo "  GPU_ARCH: $GPU_ARCH"
echo ""

# Build the executable
echo "Building executable..."
make clean
make

# Check if build was successful
if [ ! -f "./convl" ]; then
    echo "ERROR: Build failed! Executable 'convl' not found."
    exit 1
fi

echo ""
echo "Build successful!"
echo ""

# Run the convolution benchmark
echo "Running convolution benchmark..."
echo "----------------------------------------"
./convl

echo ""
echo "=========================================="
echo "Completed at $(date)"
echo "=========================================="

