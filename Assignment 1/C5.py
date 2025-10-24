#!/usr/bin/env python3
import os
import sys
import time
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <N> <repetitions>")
        sys.exit(1)

    N = int(sys.argv[1])
    repetitions = int(sys.argv[2])
    if N <= 0 or repetitions <= 0:
        raise SystemExit("N and repetitions must be positive integers")

    # Import numpy AFTER setting env vars to help ensure single-thread BLAS
    import numpy as np

    A = np.ones(N, dtype=np.float32)
    B = np.ones(N, dtype=np.float32)

    # warm-up
    _ = np.dot(A, B)

    start_index = repetitions // 2
    total_time = 0.0
    result = 0.0

    for k in range(repetitions):
        t0 = time.perf_counter()
        result = np.dot(A, B)
        t1 = time.perf_counter()
        if k >= start_index:
            total_time += (t1 - t0)

    used_reps = repetitions - start_index
    if used_reps <= 0:
        raise SystemExit("Need at least two repetitions to average (use repetitions>=2)")

    avg_sec = total_time / used_reps

    # bytes moved = 2 floats read per element
    bytes_moved = 2 * np.dtype(np.float32).itemsize * N  # 2 * 4 * N
    bandwidth_GBps = (bytes_moved / avg_sec) / 1e9

    # FLOPs = 2 per element (1 mul + 1 add)
    gflops = (2.0 * N / avg_sec) / 1e9

    print(f"N: {N} <T>: {avg_sec:.8f} sec B: {bandwidth_GBps:.5f} GB/sec F: {gflops:.5f} GFLOPS R: {result:.3f}")

if __name__ == "__main__":
    main()
