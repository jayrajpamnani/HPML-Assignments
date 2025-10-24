// dp2.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float dpunroll(long N, float *pA, float *pB) {
    float R = 0.0f;
    long j;
    for (j = 0; j + 3 < N; j += 4) {
        R += pA[j]*pB[j] + pA[j+1]*pB[j+1] +
             pA[j+2]*pB[j+2] + pA[j+3]*pB[j+3];
    }
    // cleanup for remainder
    for (; j < N; ++j) {
        R += pA[j] * pB[j];
    }
    return R;
}

double timespec_diff_sec(const struct timespec *start, const struct timespec *end) {
    double s = (double)(end->tv_sec - start->tv_sec);
    double ns = (double)(end->tv_nsec - start->tv_nsec);
    return s + ns * 1e-9;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <N> <repetitions>\n", argv[0]);
        return 1;
    }

    long N = atol(argv[1]);
    long repetitions = atol(argv[2]);

    float *A = malloc(N * sizeof(float));
    float *B = malloc(N * sizeof(float));
    if (!A || !B) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

for (long i = 0; i < N; i++) { A[i] = 1.0f; B[i] = 1.0f; }

    struct timespec t0, t1;
    double sum_sec = 0.0;
    long start_index = repetitions / 2;
    float result = 0.0f;

    for (long it = 0; it < repetitions; it++) {
        clock_gettime(CLOCK_MONOTONIC, &t0);
        result = dpunroll(N, A, B);
        clock_gettime(CLOCK_MONOTONIC, &t1);

        double elapsed = timespec_diff_sec(&t0, &t1);
        if (it >= start_index) sum_sec += elapsed;
    }

    double avg_sec = sum_sec / (repetitions - start_index);

    // Bandwidth: 2 floats per element = 8 bytes * N
    double bandwidth_GBps = (2.0 * sizeof(float) * N / avg_sec) / 1e9;
    // FLOPs: 2 per element
    double gflops = (2.0 * N / avg_sec) / 1e9;

    printf("N: %ld <T>: %.8f sec B: %.5f GB/sec F: %.5f GFLOPS R: %.3f\n",
           N, avg_sec, bandwidth_GBps, gflops, result);

    free(A);
    free(B);
    return 0;
}