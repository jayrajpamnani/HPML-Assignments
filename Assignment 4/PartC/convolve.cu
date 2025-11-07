#include <stdio.h>
#include <stdlib.h>
#include <cudnn.h>
#include <cuda_runtime.h>

#define H 1024
#define W 1024
#define C 3
#define FW 3
#define FH 3
#define K 64
#define BLOCK_SIZE 18
#define SHARED_I_SIZE (C * BLOCK_SIZE * BLOCK_SIZE)
#define SHARED_F_SIZE (FW * FH * C)
#define P 1

typedef struct
{
    int width;
    int height;
    int channel;
    int stride;
    double *elements;
} Matrix;

typedef struct
{
    int width;
    int height;
    int channel;
    int numKernels;
    double *elements;
} Filter;

// Kernel to perform basic convolution without using shared memory
__global__ void basicConvolution(Matrix I, Filter F, Matrix O)
{
    // Determine the unique indices for each thread to operate on
    int kernel_idx = blockIdx.x; // Index of the current output kernel
    int c = threadIdx.x;        // Column index within the output matrix
    int r = blockIdx.y;         // Row index within the output matrix

    int inputGrid = I.width * I.height;                      // Total size of one input channel
    int filterGrid = F.width * F.height;                     // Total size of one filter
    int kernelOffset = kernel_idx * F.channel * filterGrid; // Offset to access correct filter for the kernel
    int outputGrid = O.width * O.height;                     // Total size of one output channel

    double sum = 0.0; // Accumulator for the convolution result
    // Perform the convolution for each channel
    for (int c = 0; c < F.channel; c++)
    {
        for (int j = 0; j < F.height; j++)
        {
            for (int i = 0; i < F.width; i++)
            {
                sum += I.elements[c * inputGrid + (r + j) * I.width + (c + i)] *
                       F.elements[kernelOffset + c * filterGrid + (F.height - j - 1) * F.width + (F.width - i - 1)];
            }
        }
    }
    O.elements[kernel_idx * outputGrid + r * O.width + c] = sum;
}

// Kernel to perform tiled convolution using shared memory for better performance
__global__ void tiledConvolutionSharedMemory(Matrix I, Filter F, Matrix O)
{
    int kernel_idx = blockIdx.z; 
    int thread_row = threadIdx.y; 
    int thread_col = threadIdx.x; 
    int block_row = blockIdx.y;   
    int block_col = blockIdx.x;   

    int inputGrid = I.width * I.height;  
    int filterGrid = F.width * F.height;    
    int filterSize = filterGrid * F.channel;
    int isubGrid = BLOCK_SIZE * BLOCK_SIZE; 
    int outputGrid = O.width * O.height;     

    // Calculate starting indices for the input sub-matrix in global memory
    int y_0_in_I = block_row * (BLOCK_SIZE - 2 * P);
    int x_0_in_I = block_col * (BLOCK_SIZE - 2 * P);
    double *subTensorStart = &I.elements[y_0_in_I * I.width + x_0_in_I];

    // Define and load the input sub-matrix to shared memory
    __shared__ double sharedInput[SHARED_I_SIZE];
    for (int c = 0; c < I.channel; c++)
    {
        sharedInput[c * isubGrid + thread_row * BLOCK_SIZE + thread_col] = subTensorStart[c * inputGrid + thread_row * I.width + thread_col];
    }

    // Define and load the filter to shared memory
    __shared__ double sharedFilters[SHARED_F_SIZE];
    int thread_idx_in_block = thread_row * BLOCK_SIZE + thread_col;
    if (thread_idx_in_block < filterSize)
    {
        sharedFilters[thread_idx_in_block] = F.elements[kernel_idx * filterSize + thread_idx_in_block];
    }

    __syncthreads(); // Synchronize threads to ensure all data is loaded to shared memory

    // Perform convolution using the shared memory data
    if (thread_row < BLOCK_SIZE - 2 * P && thread_col < BLOCK_SIZE - 2 * P)
    {
        double sum = 0.0;
        for (int c = 0; c < F.channel; c++)
        {
            for (int j = 0; j < F.height; j++)
            {
                for (int i = 0; i < F.width; i++)
                {
                    sum += sharedInput[c * isubGrid + (thread_row + j) * BLOCK_SIZE + (thread_col + i)] *
                           sharedFilters[c * filterGrid + (F.height - j - 1) * F.width + (F.width - i - 1)];
                }
            }
        }
        int r = block_row * (BLOCK_SIZE - 2 * P) + thread_row;
        int c = block_col * (BLOCK_SIZE - 2 * P) + thread_col;
        O.elements[kernel_idx * outputGrid + r * O.width + c] = sum;
    }
}

// Function to allocate and initialize a matrix structure in host memory.
Matrix hostMatrix(int width, int height, int channel)
{
    Matrix newHostMatrix;
    newHostMatrix.width = width;
    newHostMatrix.height = height;
    newHostMatrix.channel = channel;
    // Calculate the total size needed for the matrix elements.
    size_t size = newHostMatrix.width * newHostMatrix.height * newHostMatrix.channel * sizeof(double);
    // Allocate memory for the matrix.
    newHostMatrix.elements = (double *)malloc(size);
    return newHostMatrix;
}

// Function to allocate and initialize a filter structure in host memory.
Filter hostFilter(int width, int height, int channel, int numKernels)
{
    Filter newHostFilter;
    newHostFilter.width = width;
    newHostFilter.height = height;
    newHostFilter.channel = channel;
    newHostFilter.numKernels = numKernels;
    // Calculate the total size needed for the filter elements.
    size_t size = newHostFilter.width * newHostFilter.height * newHostFilter.channel *
                  newHostFilter.numKernels * sizeof(double);
    newHostFilter.elements = (double *)malloc(size);
    return newHostFilter;
}

// Function to allocate and initialize a matrix structure in device memory.
Matrix deviceMatrix(Matrix M, bool copy)
{
    Matrix newDeviceMatrix;
    newDeviceMatrix.width = M.width;
    newDeviceMatrix.height = M.height;
    newDeviceMatrix.channel = M.channel;
    newDeviceMatrix.stride = M.width; // Stride set to width for simple r-major layout.
    // Allocate memory on the GPU.
    cudaMalloc((void **)&newDeviceMatrix.elements, M.width * M.height * M.channel * sizeof(double));
    // Optionally copy data from host to device.
    if (copy)
    {
        cudaMemcpy(newDeviceMatrix.elements, M.elements, M.width * M.height * M.channel * sizeof(double), cudaMemcpyHostToDevice);
    }
    return newDeviceMatrix;
}

// Function to allocate and initialize a filter structure in device memory.
Filter deviceFilter(Filter F, bool copy)
{
    Filter newDeviceFilter;
    newDeviceFilter.width = F.width;
    newDeviceFilter.height = F.height;
    newDeviceFilter.channel = F.channel;
    newDeviceFilter.numKernels = F.numKernels;
    // Allocate memory on the GPU.
    cudaMalloc((void **)&newDeviceFilter.elements, F.width * F.height * F.channel * F.numKernels * sizeof(double));
    // Optionally copy data from host to device.
    if (copy)
    {
        cudaMemcpy(newDeviceFilter.elements, F.elements, F.width * F.height * F.channel * F.numKernels * sizeof(double), cudaMemcpyHostToDevice);
    }
    return newDeviceFilter;
}

// Function to add padding around an existing matrix to facilitate convolution operations.
Matrix padHostInput(Matrix I, int padding)
{
    Matrix I_0;
    I_0.width = I.width + 2 * padding;
    I_0.height = I.height + 2 * padding;
    I_0.channel = I.channel;
    // Calculate the total size of the padded matrix.
    size_t size = I_0.width * I_0.height * I_0.channel * sizeof(double);
    I_0.elements = (double *)malloc(size);
    // Initialize padded matrix elements.
    for (int c = 0; c < I_0.channel; c++)
    {
        for (int y = 0; y < I_0.height; y++)
        {
            for (int x = 0; x < I_0.width; x++)
            {
                // Check if the current element is within the padding boundary.
                if (y < padding || y >= I.height + padding || x < padding || x >= I.width + padding)
                {
                    I_0.elements[x + y * I_0.width + c * (I_0.width * I_0.height)] = 0;
                }
                else
                {
                    // Map the internal non-padding area to the original matrix elements.
                    I_0.elements[x + y * I_0.width + c * (I_0.width * I_0.height)] =
                        I.elements[(x - padding) + (y - padding) * I.width + c * (I.width * I.height)];
                }
            }
        }
    }
    return I_0;
}

// Function to create and initialize a matrix with values based on channel and position.
Matrix getHostInput(int width, int height, int channel)
{
    Matrix M = hostMatrix(width, height, channel);
    for (int c = 0; c < M.channel; c++)
    {
        for (int y = 0; y < M.height; y++)
        {
            for (int x = 0; x < M.width; x++)
            {
                // Initialize each element with a function of its coordinates and channel.
                M.elements[x + y * M.width + c * (M.width * M.height)] = c * (x + y);
            }
        }
    }
    return M;
}

// Function to create and initialize a filter with values based on channel, kernel index, and position.
Filter getHostFilter(int width, int height, int channel, int numKernels)
{
    Filter F = hostFilter(width, height, channel, numKernels);
    for (int k = 0; k < F.numKernels; k++)
    {
        for (int c = 0; c < F.channel; c++)
        {
            for (int j = 0; j < F.height; j++)
            {
                for (int i = 0; i < F.width; i++)
                {
                    // Initialize each filter element with a function of its coordinates, channel, and kernel index.
                    F.elements[i + j * F.width + c * (F.width * F.height) + k * (F.width * F.height * F.channel)] =
                        (c + k) * (i + j);
                }
            }
        }
    }
    return F;
}

// Function to calculate the checksum of a matrix to verify the results.
double getCheckSum(Matrix M)
{
    double sum = 0.0;
    int grid = M.width * M.height; // Total number of elements in one channel.
    // Sum up all elements across all channels.
    for (int c = 0; c < M.channel; c++)
    {
        for (int y = 0; y < M.height; y++)
        {
            for (int x = 0; x < M.width; x++)
            {
                sum += M.elements[x + y * M.width + c * grid];
            }
        }
    }
    return sum;
}

// Function to perform the basic convolution (C1) using CUDA.
void convolutionC1(const Matrix input, const Filter filter, Matrix output, float &executionTime)
{
    // Allocate and initialize device memory for input and output matrices and the filter.
    Matrix device_input = deviceMatrix(input, true);
    Matrix device_output = deviceMatrix(output, false);
    Filter device_filter = deviceFilter(filter, true);

    // Define the grid and block dimensions for the kernel execution.
    dim3 dimBlock(output.width);                 // One thread per output element in the width.
    dim3 dimGrid(output.channel, output.height); // Each block handles a channel and a r.

    // Execute the kernel once to warm up (ensure GPU is ready for timing).
    basicConvolution<<<dimGrid, dimBlock>>>(device_input, device_filter, device_output);

    // Wait for GPU to finish before starting the timed execution.
    cudaThreadSynchronize();

    // Setup timing events to measure execution time.
    cudaEvent_t start_time, end_time;
    cudaEventCreate(&start_time);
    cudaEventCreate(&end_time);

    // Start timing.
    cudaEventRecord(start_time);

    // Execute the convolution kernel.
    basicConvolution<<<dimGrid, dimBlock>>>(device_input, device_filter, device_output);

    // Wait for GPU to finish the real execution.
    cudaThreadSynchronize();

    // Stop timing.
    cudaEventRecord(end_time);
    cudaEventSynchronize(end_time);

    // Calculate the elapsed time in milliseconds.
    cudaEventElapsedTime(&executionTime, start_time, end_time);

    // Copy the result from device to host memory.
    size_t size = output.width * output.height * output.channel * sizeof(double);
    cudaMemcpy(output.elements, device_output.elements, size, cudaMemcpyDeviceToHost);

    // Free the device memory.
    cudaFree(device_input.elements);
    cudaFree(device_filter.elements);
    cudaFree(device_output.elements);

    // Destroy the events.
    cudaEventDestroy(start_time);
    cudaEventDestroy(end_time);
}

// Function to perform the tiled convolution (C2) using shared memory and CUDA.
void convolutionC2(const Matrix input, const Filter filter, Matrix output, float &executionTime)
{
    // Allocate and initialize device memory for input and output matrices and the filter.
    Matrix device_input = deviceMatrix(input, true);
    Filter device_filter = deviceFilter(filter, true);
    Matrix device_output = deviceMatrix(output, false);

    // Calculate the number of blocks needed per r and column considering the padding.
    int block_per_row = (input.width - 2 * P) / (BLOCK_SIZE - 2 * P);
    int block_per_col = (input.height - 2 * P) / (BLOCK_SIZE - 2 * P);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);         // Define the size of each block.
    dim3 dimGrid(block_per_row, block_per_col, K); // Grid dimensions, including depth for kernels.

    // Execute the kernel once to warm up the GPU.
    tiledConvolutionSharedMemory<<<dimGrid, dimBlock>>>(device_input, device_filter, device_output);

    // Wait for GPU to finish the warm up.
    cudaThreadSynchronize();

    // Setup timing events.
    cudaEvent_t start_time, end_time;
    cudaEventCreate(&start_time);
    cudaEventCreate(&end_time);

    // Start timing.
    cudaEventRecord(start_time);

    // Execute the tiled convolution kernel.
    tiledConvolutionSharedMemory<<<dimGrid, dimBlock>>>(device_input, device_filter, device_output);

    // Wait for GPU to finish the real execution.
    cudaThreadSynchronize();

    // Stop timing.
    cudaEventRecord(end_time);
    cudaEventSynchronize(end_time);

    // Calculate the elapsed time in milliseconds.
    cudaEventElapsedTime(&executionTime, start_time, end_time);

    // Copy the result from device to host memory.
    size_t size = output.width * output.height * output.channel * sizeof(double);
    cudaMemcpy(output.elements, device_output.elements, size, cudaMemcpyDeviceToHost);

    // Free the device memory.
    cudaFree(device_input.elements);
    cudaFree(device_filter.elements);
    cudaFree(device_output.elements);

    // Destroy the events.
    cudaEventDestroy(start_time);
    cudaEventDestroy(end_time);
}

// Function to perform convolution using the cuDNN library, optimized for CUDA.
void convolutionC3(const Matrix input, const Filter filter, Matrix output, float &executionTime)
{
    // Allocate and initialize device memory for the input matrix, filter, and output matrix.
    Matrix device_input = deviceMatrix(input, true);
    Filter device_filter = deviceFilter(filter, true);
    Matrix device_output = deviceMatrix(output, false);

    // Initialize the cuDNN library handle.
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    // Set up the tensor descriptor for input data.
    cudnnTensorDescriptor_t inputDesc;
    cudnnCreateTensorDescriptor(&inputDesc);
    cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, input.channel, input.height, input.width);

    // Set up the tensor descriptor for output data.
    cudnnTensorDescriptor_t outputDesc;
    cudnnCreateTensorDescriptor(&outputDesc);
    cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, output.channel, output.height, output.width);

    // Set up the filter descriptor for convolution.
    cudnnFilterDescriptor_t filterDesc;
    cudnnCreateFilterDescriptor(&filterDesc);
    cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, filter.numKernels, filter.channel, filter.height, filter.width);

    // Create the convolution descriptor and configure it.
    cudnnConvolutionDescriptor_t convDesc;
    cudnnCreateConvolutionDescriptor(&convDesc);
    cudnnSetConvolution2dDescriptor(convDesc, P, P, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_DOUBLE);

    // Perform an automatic tuning to find the optimal convolution algorithm.
    cudnnConvolutionFwdAlgoPerf_t perfResults;
    cudnnConvolutionFwdAlgo_t selectedAlgo;
    int returnedCount;
    cudnnFindConvolutionForwardAlgorithm(cudnn, inputDesc, filterDesc, convDesc, outputDesc, 1, &returnedCount, &perfResults);
    selectedAlgo = perfResults.algo;

    // Query for the amount of workspace memory required for the selected algorithm.
    size_t workspaceSize;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn, inputDesc, filterDesc, convDesc, outputDesc, selectedAlgo, &workspaceSize);

    // Allocate the workspace required for convolution operation.
    void *workspace;
    cudaMalloc(&workspace, workspaceSize);

    // Start measuring execution time using CUDA events.
    double alpha = 1.0, beta = 0.0;
    cudaEvent_t start_time, end_time;
    cudaEventCreate(&start_time);
    cudaEventCreate(&end_time);
    cudaEventRecord(start_time);

    // Perform the convolution operation.
    cudnnConvolutionForward(cudnn, &alpha, inputDesc, device_input.elements, filterDesc, device_filter.elements,
                            convDesc, selectedAlgo, workspace, workspaceSize, &beta,
                            outputDesc, device_output.elements);

    // Wait for the convolution operation to complete.
    cudaThreadSynchronize();

    // Stop the timer and calculate the elapsed time.
    cudaEventRecord(end_time);
    cudaEventSynchronize(end_time);
    cudaEventElapsedTime(&executionTime, start_time, end_time);

    // Copy the convolution results from device to host.
    size_t size = output.width * output.height * output.channel * sizeof(double);
    cudaMemcpy(output.elements, device_output.elements, size, cudaMemcpyDeviceToHost);

    // Clean up resources.
    cudaFree(workspace);
    cudaFree(device_input.elements);
    cudaFree(device_filter.elements);
    cudaFree(device_output.elements);
    cudaEventDestroy(start_time);
    cudaEventDestroy(end_time);
    cudnnDestroyTensorDescriptor(inputDesc);
    cudnnDestroyTensorDescriptor(outputDesc);
    cudnnDestroyFilterDescriptor(filterDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroy(cudnn);
}

int main()
{
    float executionTimeC1, executionTimeC2, executionTimeC3 = 0.0;
    double checksumC1, checksumC2, checksumC3  = 0.0;

    // Initialize host matrices for input, padded input, and filters.
    Matrix inputMatrix = getHostInput(W, H, C);               // Original input matrix.
    Matrix paddedInputMatrix = padHostInput(inputMatrix, P); // Input matrix with padding.
    Filter convolutionFilter = getHostFilter(FW, FH, C, K);   // Convolution filters.
    Matrix outputMatrix1 = hostMatrix(W, H, K);               // Output matrix for convolution method C1.
    Matrix outputMatrix2 = hostMatrix(W, H, K);               // Output matrix for convolution method C2.
    Matrix outputMatrix3 = hostMatrix(W, H, K);               // Output matrix for convolution method C3.

    // Perform the first convolution using basic method.
    convolutionC1(paddedInputMatrix, convolutionFilter, outputMatrix1, executionTimeC1);
    checksumC1 = getCheckSum(outputMatrix1);

    // Perform the second convolution using tiled shared memory method.
    convolutionC2(paddedInputMatrix, convolutionFilter, outputMatrix2, executionTimeC2);
    checksumC2 = getCheckSum(outputMatrix2);

    // Perform the third convolution using cuDNN optimized method.
    convolutionC3(inputMatrix, convolutionFilter, outputMatrix3, executionTimeC3);
    checksumC3 = getCheckSum(outputMatrix3);

    // Print the checksums and execution times for each convolution method.
    printf("%f, %.3f ms\n", checksumC1, executionTimeC1);
    printf("%f, %.3f ms\n", checksumC2, executionTimeC2);
    printf("%f, %.3f ms\n", checksumC3, executionTimeC3);

    // Free memory allocated for matrices and filters.
    free(inputMatrix.elements);
    free(paddedInputMatrix.elements);
    free(convolutionFilter.elements);
    free(outputMatrix1.elements);
    free(outputMatrix2.elements);
    free(outputMatrix3.elements);

    return 0;
}
