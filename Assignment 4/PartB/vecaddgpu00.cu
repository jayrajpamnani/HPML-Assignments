#include <iostream>
#include <cmath>
#include <time.h>

// Variables for host and device vectors
float *device_x, *device_y, *device_z;
float *host_x, *host_y, *host_z;

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
    double cpu_memory_allocation_time, gpu_memory_allocation_time, memcopy_time, time;
	
    n = millions * n;
    size_t size = n * sizeof(float);
    
	// Allocate input vectors in host memory
    clock_gettime(CLOCK_MONOTONIC, &start);
    host_x = (float*)malloc(size);
    if (host_x == 0) Cleanup(false);
    host_y = (float*)malloc(size);
    if (host_y == 0) Cleanup(false);
    host_z = (float*)malloc(size);
    if (host_z == 0) Cleanup(false);
    clock_gettime(CLOCK_MONOTONIC, &end);

    cpu_memory_allocation_time = ((double)(end.tv_sec - start.tv_sec)*1000000) + ((double)(end.tv_nsec - start.tv_nsec)/1000);

    // Initialize vectors on the host
    for (int i = 0; i < n; i++) {
        host_x[i] = 1.0f;
        host_y[i] = 2.0f;
        host_z[i] = 0.0f;
    }

    // Allocate vectors in device memory
    cudaError_t error;
    clock_gettime(CLOCK_MONOTONIC, &start);
    error = cudaMalloc((void**)&device_x, size);
    if (error != cudaSuccess) Cleanup(false);
    error = cudaMalloc((void**)&device_y, size);
    if (error != cudaSuccess) Cleanup(false);
    error = cudaMalloc((void**)&device_z, size);
    if (error != cudaSuccess) Cleanup(false);
    cudaThreadSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    gpu_memory_allocation_time = ((double)(end.tv_sec - start.tv_sec)*1000000) + ((double)(end.tv_nsec - start.tv_nsec)/1000);
	
    // Copy vectors from host memory to device global memory
    clock_gettime(CLOCK_MONOTONIC, &start);
    error = cudaMemcpy(device_x, host_x, size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) Cleanup(false);
    error = cudaMemcpy(device_y, host_y, size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) Cleanup(false);
    error = cudaMemcpy(device_z, host_z, size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) Cleanup(false);
    cudaThreadSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &end);

    memcopy_time = ((double)(end.tv_sec - start.tv_sec)*1000000) + ((double)(end.tv_nsec - start.tv_nsec)/1000);

    dim3 dimGrid(grid_size);
    dim3 dimBlock(block_size);

    // Warm up
    addVec<<<dimGrid, dimBlock>>>(n, device_x, device_y, device_z);
    error = cudaGetLastError();
    if (error != cudaSuccess) Cleanup(false);
    cudaThreadSynchronize();


    // Invoke kernel
    clock_gettime(CLOCK_MONOTONIC, &start);
    addVec<<<dimGrid, dimBlock>>>(n, device_x, device_y, device_z);
    error = cudaGetLastError();
    if (error != cudaSuccess) Cleanup(false);
    cudaThreadSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &end);

    time = ((double)(end.tv_sec - start.tv_sec)*1000000) + ((double)(end.tv_nsec - start.tv_nsec)/1000);
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    error = cudaMemcpy(host_z, device_z, size, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) Cleanup(false);
    cudaThreadSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    memcopy_time += ((double)(end.tv_sec - start.tv_sec)*1000000) + ((double)(end.tv_nsec - start.tv_nsec)/1000);
    
    double total_time = cpu_memory_allocation_time + gpu_memory_allocation_time + memcopy_time + time;

    std::cout << "CPU Memory Allocation Time : " << cpu_memory_allocation_time/1000 << " millisec" << std::endl;
    std::cout << "GPU Memory Allocation Time : " << gpu_memory_allocation_time/1000 << " millisec" << std::endl;
    std::cout << "Mem Copy Time              : " << memcopy_time/1000 << " millisec" << std::endl;
    std::cout << "Execution Time             : " << time/1000 << " millisec" << std::endl;
    std::cout << "Total Time                 : " << total_time/1000 << " millisec" << std::endl;


    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < n; i++) {
        maxError = fmax(maxError, std::fabs(host_z[i] - 3.0f));
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
    if (device_x)
        cudaFree(device_x);
    if (device_y)
        cudaFree(device_y);
    if (device_z)
        cudaFree(device_z);

    // Free host memory
    if (host_x)
        free(host_x);
    if (host_y)
        free(host_y);
    if (host_z)
        free(host_z);

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
