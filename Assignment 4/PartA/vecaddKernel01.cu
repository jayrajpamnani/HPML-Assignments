// vecAddKernel01.cu
// For ECE-GY 9143 - High Performance Computing for Machine Learning
// Instructor: Zehra Sura and Robert Kingan
// Based on code from the CUDA Programming Guide

// This Kernel adds two Vectors A and B in C on GPU
// using coalesced memory access.

__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    // Calculate the base index for this thread
    // All threads in a block work on consecutive elements first
    int blockStartIndex = blockIdx.x * blockDim.x * N;
    int threadBaseIndex = blockStartIndex + threadIdx.x;
    
    // Each thread processes N elements with stride blockDim.x
    // This ensures that consecutive threads access consecutive memory locations
    // in each iteration, enabling memory coalescing
    for(int i = 0; i < N; ++i) {
        int index = threadBaseIndex + i * blockDim.x;
        C[index] = A[index] + B[index];
    }
}

