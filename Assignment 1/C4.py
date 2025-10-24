#!/usr/bin/env python3
import sys
import time
import numpy as np

def dp(N, A, B):
    R = 0.0  # Python float (double)
    for j in range(N):
        R += A[j] * B[j]
    return R

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <N> <repetitions>")
        sys.exit(1)

    N = int(sys.argv[1])
    repetitions = int(sys.argv[2])
    if N <= 0 or repetitions <= 0:
        raise SystemExit("N and repetitions must be positive integers")

    # prepare data
    A = np.ones(N, dtype=np.float32)
    B = np.ones(N, dtype=np.float32)

    start_index = repetitions // 2  # use second half for averaging
    total_time = 0.0
    result = 0.0

    for k in range(repetitions):
        t0 = time.perf_counter()
        result = dp(N, A, B)
        t1 = time.perf_counter()
        if k >= start_index:
            total_time += (t1 - t0)

    used_reps = repetitions - start_index
    if used_reps <= 0:
        raise SystemExit("Not enough repetitions to average (need >=2)")

    avg_sec = total_time / used_reps

    # bandwidth: 2 floats read per element -> 2 * 4 bytes * N
    bytes_moved = 2 * np.dtype(np.float32).itemsize * N
    bandwidth_GBps = (bytes_moved / avg_sec) / 1e9

    # FLOPs: 2 ops per element (1 mul + 1 add)
    gflops = (2.0 * N / avg_sec) / 1e9

    print(f"N: {N} <T>: {avg_sec:.8f} sec B: {bandwidth_GBps:.5f} GB/sec F: {gflops:.5f} GFLOPS R: {result}")

if __name__ == "__main__":
    main()
