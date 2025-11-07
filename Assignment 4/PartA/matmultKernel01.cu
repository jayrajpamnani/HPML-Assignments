// matmultKernel01.cu
// For ECE-GY 9143 - High Performance Computing for Machine Learning
// Instructor: Zehra Sura and Robert Kingan
// Based on code from the CUDA Programming Guide
//
// OPTIMIZED VERSION:
// This kernel computes a larger footprint (32x32) per thread block compared to
// the base version (16x16). Each thread in a 16x16 block computes a 2x2 sub-matrix
// of the output, improving memory bandwidth utilization and reducing the number
// of thread blocks needed. This optimization is particularly effective for larger
// matrices as it reduces grid overhead and improves cache locality.
//
// Key optimizations:
// 1. Larger footprint (32x32) reduces grid size and overhead
// 2. Each thread computes 4 elements (2x2), improving register utilization
// 3. Better memory access patterns with increased work per thread
// 4. Loop unrolling for better performance

#include "matmultKernel.h"

// Each thread block computes a FOOTPRINT_SIZE x FOOTPRINT_SIZE block
// This is defined at compile time via -DFOOTPRINT_SIZE=32
#ifndef FOOTPRINT_SIZE
#define FOOTPRINT_SIZE 32
#endif

// Define a gpu kernel to perform optimized matrix multiplication
// A x B = C, where each thread computes multiple output elements
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C){

  // matrix blocks
  float *Asub, *Bsub, *Csub;
  // Putting these into registers speeds access.
  int thread_row = threadIdx.y;
  int thread_col = threadIdx.x;
  int block_row = blockIdx.y;
  int block_col = blockIdx.x;

  // Each thread computes a 2x2 sub-matrix of the output
  // Since FOOTPRINT_SIZE = 32 and BLOCK_SIZE = 16, each thread handles 4 elements
  int sub_tile_size = FOOTPRINT_SIZE / BLOCK_SIZE;  // = 2
  int output_row = thread_row * sub_tile_size;
  int output_col = thread_col * sub_tile_size;

  // Each THREAD BLOCK computes one sub matrix Csub of C (32x32)
  // EACH THREAD creates its own matrix descriptor Csub
  Csub = &C.elements[C.stride * FOOTPRINT_SIZE * block_row + FOOTPRINT_SIZE * block_col];

  // Each thread computes 4 elements (2x2 sub-matrix) of Csub
  float Cvalue[2][2];
  Cvalue[0][0] = 0.0f;
  Cvalue[0][1] = 0.0f;
  Cvalue[1][0] = 0.0f;
  Cvalue[1][1] = 0.0f;

  // Shared memory for tile data
  // We use 32x32 shared memory to store the larger tiles needed for 32x32 output
  // Padding to avoid bank conflicts: 32x33 instead of 32x32
  __shared__ float shared_A[FOOTPRINT_SIZE][FOOTPRINT_SIZE + 1];
  __shared__ float shared_B[FOOTPRINT_SIZE][FOOTPRINT_SIZE + 1];

  // Loop over all sub matrices in block_row of A and block_col of B
  // required to compute Csub. Block multiply each pair of sub matrices
  // and accumulate results
  for (int m = 0;  m < (A.width / BLOCK_SIZE); ++m){
    // Load 32x16 tile from A (32 rows, 16 cols from the m-th column block)
    // Each thread loads 2 elements from A (one per output row)
    Asub = &A.elements[A.stride * FOOTPRINT_SIZE * block_row + BLOCK_SIZE * m];
    shared_A[output_row][thread_col] = Asub[output_row * A.stride + thread_col];
    shared_A[output_row + 1][thread_col] = Asub[(output_row + 1) * A.stride + thread_col];
    
    // Load 16x32 tile from B (16 rows, 32 cols from the m-th row block)
    // We need to load all 32 columns, so we use a different strategy:
    // Each thread loads 2 elements from B to cover the 32 columns
    // Since we have 16x16 threads = 256 threads, and we need 16*32 = 512 elements,
    // each thread loads 2 elements
    Bsub = &B.elements[B.stride * BLOCK_SIZE * m + FOOTPRINT_SIZE * block_col];
    
    // Each thread loads 2 elements from B to help load all 32 columns
    // Thread at (thread_row, thread_col) loads columns at thread_col*2 and thread_col*2+1
    int b_col0 = thread_col * sub_tile_size;
    int b_col1 = b_col0 + 1;
    if (b_col0 < FOOTPRINT_SIZE) {
      shared_B[thread_row][b_col0] = Bsub[thread_row * B.stride + b_col0];
    }
    if (b_col1 < FOOTPRINT_SIZE) {
      shared_B[thread_row][b_col1] = Bsub[thread_row * B.stride + b_col1];
    }

    // Synchronize to ensure all elements are read
    __syncthreads();

    // Compute dot product: each thread computes 2x2 = 4 elements
    // For each output element, we multiply a row from A with a column from B
    float a_val0, a_val1, b_val0, b_val1;

    // Compute Cvalue[0][0] and Cvalue[0][1] using first row of A
    // We iterate over the 16 columns of the loaded tile (m-th tile)
    #pragma unroll
    for(int e=0; e<BLOCK_SIZE; ++e) {
      a_val0 = shared_A[output_row][e];
      b_val0 = shared_B[e][output_col];
      b_val1 = shared_B[e][output_col + 1];
      Cvalue[0][0] += a_val0 * b_val0;
      Cvalue[0][1] += a_val0 * b_val1;
    }

    // Compute Cvalue[1][0] and Cvalue[1][1] using second row of A
    #pragma unroll
    for(int e=0; e<BLOCK_SIZE; ++e) {
      a_val1 = shared_A[output_row + 1][e];
      b_val0 = shared_B[e][output_col];
      b_val1 = shared_B[e][output_col + 1];
      Cvalue[1][0] += a_val1 * b_val0;
      Cvalue[1][1] += a_val1 * b_val1;
    }

    // Synchronize to ensure all Cvalues have been incremented
    // before reading in the next shared_A AND shared_B BLOCKS
    __syncthreads();
  }

  // Write the 2x2 sub-matrix to GLOBAL memory
  // Each thread writes 4 elements
  Csub[output_row * C.stride + output_col] = Cvalue[0][0];
  Csub[output_row * C.stride + output_col + 1] = Cvalue[0][1];
  Csub[(output_row + 1) * C.stride + output_col] = Cvalue[1][0];
  Csub[(output_row + 1) * C.stride + output_col + 1] = Cvalue[1][1];
}
