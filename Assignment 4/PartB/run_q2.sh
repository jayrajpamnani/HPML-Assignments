#!/bin/bash
# Script to run Q2 experiments for GPU vector addition
# Three scenarios: 
#   1. 1 block, 1 thread
#   2. 1 block, 256 threads
#   3. Multiple blocks with 256 threads/block (total threads = array size)

echo "=========================================="
echo "GPU Vector Addition - Q2 Results"
echo "Date: $(date)"
echo "=========================================="
echo ""

# Test values for K
K_VALUES=(1 5 10 50 100)

# Scenario 1: 1 block with 1 thread
echo "=========================================="
echo "SCENARIO 1: 1 block, 1 thread"
echo "=========================================="
for K in "${K_VALUES[@]}"
do
    echo "K = $K million"
    ./vecaddgpu00 1 1 $K
    echo ""
done

# Scenario 2: 1 block with 256 threads
echo "=========================================="
echo "SCENARIO 2: 1 block, 256 threads"
echo "=========================================="
for K in "${K_VALUES[@]}"
do
    echo "K = $K million"
    ./vecaddgpu00 1 256 $K
    echo ""
done

# Scenario 3: Multiple blocks with 256 threads per block
# Total threads = array size (n) as closely as possible
# So: grid_size * 256 = n = K * 1000000
# grid_size = ceil(K * 1000000 / 256)
echo "=========================================="
echo "SCENARIO 3: Multiple blocks, 256 threads/block"
echo "Total threads = array size (or as close as possible)"
echo "=========================================="
for K in "${K_VALUES[@]}"
do
    # Calculate grid size: ceil(n / 256) = ceil(K * 1000000 / 256)
    # Using integer arithmetic: (n + 255) / 256 for ceiling division
    N=$((K * 1000000))
    GRID_SIZE=$(((N + 255) / 256))
    TOTAL_THREADS=$((GRID_SIZE * 256))
    echo "K = $K million (n=$N, Grid size: $GRID_SIZE, Block size: 256, Total threads: $TOTAL_THREADS)"
    ./vecaddgpu00 $GRID_SIZE 256 $K
    echo ""
done

echo "=========================================="
echo "All tests completed at $(date)"
echo "=========================================="
