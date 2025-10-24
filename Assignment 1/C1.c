#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <errno.h>
#include <string.h>

float dp(long N, float *pA, float *pB) {
    float R = 0.0f;
    for (long j = 0; j < N; ++j) {
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
    errno = 0;
    char *endptr;
    long N = strtol(argv[1], &endptr, 10);
    if (errno || *endptr != '\0' || N <= 0) {
        fprintf(stderr, "Invalid N: '%s'\n", argv[1]);
        return 1;
    }
    long repetitions = strtol(argv[2], &endptr, 10);
    if (errno || *endptr != '\0' || repetitions <= 0) {
        fprintf(stderr, "Invalid repetitions: '%s'\n", argv[2]);
        return 1;
    }
    // allocate aligned memory (helps vectorized code)
    float *A = NULL;
    float *B = NULL;
    if (posix_memalign((void**)&A, 64, (size_t)N * sizeof(float)) != 0 ||
        posix_memalign((void**)&B, 64, (size_t)N * sizeof(float)) != 0) {
        fprintf(stderr, "posix_memalign failed\n");
        free(A); free(B);
        return 1;
    }
    // initialize
    for (long i = 0; i < N; ++i) { A[i] = 1.0f; B[i] = 1.0f; }
    struct timespec t0, t1;
    double sum_sec = 0.0;
    long start_index = repetitions / 2; // second half
    float result = 0.0f;
    for (long it = 0; it < repetitions; ++it) {
        clock_gettime(CLOCK_MONOTONIC, &t0);
        result = dp(N, A, B);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double elapsed = timespec_diff_sec(&t0, &t1);
        if (it >= start_index) sum_sec += elapsed;
    }
    long used_reps = repetitions - start_index;
    double avg_sec = (used_reps > 0) ? (sum_sec / (double)used_reps) : 0.0;

    // Memory traffic: reading A and B -> 2 * sizeof(float) * N bytes
    double bytes = 2.0 * sizeof(float) * (double)N;
    double bandwidth_GBps = (bytes / avg_sec) / 1.0e9; // GB/s

    // FLOPs: one multiply + one add per element = 2 * N floating ops
    double gflops = (2.0 * (double)N / avg_sec) / 1.0e9; // GFLOPS

    // Print: N, average time, bandwidth, gflops, and dot-product result (sanity)
    printf("N: %ld <T>: %.8f sec B: %.5f GB/sec F: %.5f GFLOPS R: %.3f\n",
           N, avg_sec, bandwidth_GBps, gflops, (double)result);

    free(A);
    free(B);
    return 0;
}