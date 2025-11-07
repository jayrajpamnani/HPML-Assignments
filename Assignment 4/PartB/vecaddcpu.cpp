#include <iostream>
#include <time.h>
#include <cmath>

// Function to add arrays
void vecAdd(const float *vec1, const float *vec2, float *vec3, int n) {
    for (int i = 0; i < n; i++) {
        vec3[i] = vec1[i] + vec2[i];
    }
}

int main(int argc, char** argv) {
    int millions = 1;
    int n = 1000000;

    if (argc == 2) {
        sscanf(argv[1], "%d", &millions);
    }

    n = millions * n;
    
    struct timespec start, end;
    double memory_allocation_time, execution_time, total_time;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    float *vecA = (float*)malloc(n * sizeof(float));
    float *vecB = (float*)malloc(n * sizeof(float));
    float *vecC = (float*)malloc(n * sizeof(float));
    clock_gettime(CLOCK_MONOTONIC, &end);

    memory_allocation_time = ((double)(end.tv_sec - start.tv_sec)*1000000) + ((double)(end.tv_nsec - start.tv_nsec)/1000);
    
    // Initialize arrays
    for (int i = 0; i < n; ++i) {
        vecA[i] = 1.0f;
        vecB[i] = 2.0f;
    }

    // Warmup Run
    vecAdd(vecA, vecB, vecC, n);
    
    
    // Actual Run 
    clock_gettime(CLOCK_MONOTONIC, &start);
    vecAdd(vecA, vecB, vecC, n);
    clock_gettime(CLOCK_MONOTONIC, &end);

    execution_time = ((double)(end.tv_sec - start.tv_sec)*1000000) + ((double)(end.tv_nsec - start.tv_nsec)/1000);

    total_time = memory_allocation_time + execution_time;

    std::cout << "K: " << millions << " million" << std::endl; 
    std::cout << "Memory Allocation Time : " << memory_allocation_time/1000 << " millisec" << std::endl;
    std::cout << "Execution Time         : " << execution_time/1000 << " millisec" << std::endl;
    std::cout << "Total Time             : " << total_time/1000 << " millisec" << std::endl;
    
    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < n; i++) {
        maxError = std::fmax(maxError, std::fabs(vecC[i] - 3.0f));
    }
    
    if (maxError > 0) {
        std::cout << "TEST FAILED" << std::endl;
        std::cout << "Max error: " << maxError << std::endl;
    }

    // Free memory
    free(vecA);
    free(vecB);
    free(vecC);

    return 0;
}
