#include <iostream>
#include <cmath>
#include <time.h>

// Variables for vectors
float *x, *y, *z;

// Utility Functions
void Cleanup(bool);
void checkCUDAError(const char *msg);

// Kernel function to add the elements of two vectors 
__global__ void addVec(int n, const float *x, const float *y, float *z){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = index; i < n; i += stride)
        z[i] = x[i] + y[i];
}

int main(int argc, char** argv) {
    int millions = 1;
    int n = 1000000;
    int grid_size = 1;
    int block_size = 1;

    if (argc == 4) {
        sscanf(argv[1], "%d", &grid_size);
        sscanf(argv[2], "%d", &block_size);
		sscanf(argv[3], "%d", &millions);
    }

    std::cout << "K: " << millions << " million, Grid size: " << grid_size << ", Block size: " << block_size << std::endl;
    
    struct timespec start, end;
    double unified_memory_allocation_time, time, total_time;
	
    n = millions * n;
    size_t size = n * sizeof(float);
    
    // Allocate Unified Memory - accessible from CPU or GPU
    clock_gettime(CLOCK_MONOTONIC, &start);
    cudaMallocManaged(&x, size);
    cudaMallocManaged(&y, size);
    cudaMallocManaged(&z, size);
    cudaThreadSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    unified_memory_allocation_time = ((double)(end.tv_sec - start.tv_sec)*1000000) + ((double)(end.tv_nsec - start.tv_nsec)/1000);

    // Initialize vectors on the host
    for (int i = 0; i < n; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
        z[i] = 0.0f;
    }

    dim3 dimGrid(grid_size);
    dim3 dimBlock(block_size);
    
    cudaError_t error;

    // Warm up
    addVec<<<dimGrid, dimBlock>>>(n, x, y, z);
    error = cudaGetLastError();
    if (error != cudaSuccess) Cleanup(false);
    cudaThreadSynchronize();


    // Invoke kernel
    clock_gettime(CLOCK_MONOTONIC, &start);
    addVec<<<dimGrid, dimBlock>>>(n, x, y, z);
    error = cudaGetLastError();
    if (error != cudaSuccess) Cleanup(false);
    cudaThreadSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &end);

    time = ((double)(end.tv_sec - start.tv_sec)*1000000) + ((double)(end.tv_nsec - start.tv_nsec)/1000);
    total_time = unified_memory_allocation_time + time;
    
    std::cout << "Unified Memory Allocation Time : " << unified_memory_allocation_time/1000 << " millisec" << std::endl;
    std::cout << "Execution Time                 : " << time/1000 << " millisec" << std::endl;
    std::cout << "Total Time                     : " << total_time/1000 << " millisec" << std::endl;

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < n; i++) {
        maxError = fmax(maxError, std::fabs(z[i] - 3.0f));
    }
    
    if (maxError > 0)
        std::cout << "Max error: " << maxError << std::endl;
   
    // Clean up and exit.
    Cleanup(true);
    
    return 0;
}

void Cleanup(bool noError) {  // simplified version from CUDA SDK
    cudaError_t error;

    // Free device vectors
    if (x)
        cudaFree(x);
    if (y)
        cudaFree(y);
    if (z)
        cudaFree(z);

    error = cudaThreadExit();

    if (!noError || error != cudaSuccess)
        printf("cuda malloc or cuda thread exit failed \n");

    fflush( stdout);
    fflush( stderr);

    exit(0);
}

void checkCUDAError(const char *msg)
{
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err)
    {
      fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err) );
      exit(-1);
    }
}
