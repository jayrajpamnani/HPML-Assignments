#!/usr/bin/env python3
"""
Q4: Plotting script for CPU vs GPU comparison
- Chart 1: Step 2 (without Unified Memory) - vecaddgpu00
- Chart 2: Step 3 (with Unified Memory) - vecaddgpu01

Each chart shows:
- CPU total time (from vecaddcpu) - includes memory allocation + execution
- GPU total time for 3 scenarios - includes memory operations + execution
  - For non-unified memory: Total = CPU alloc + GPU alloc + memcpy + execution
  - For unified memory: Total = unified alloc + execution

Note: Using Total Time (not just Execution Time) to show the full overhead
difference between unified and non-unified memory approaches.
"""

import matplotlib.pyplot as plt
import numpy as np
import re
import sys

def parse_cpu_output(output_file):
    """Parse CPU output file and extract execution times for each K"""
    cpu_times = {}
    with open(output_file, 'r') as f:
        lines = f.readlines()
    
    current_K = None
    for line in lines:
        # Detect K value
        match = re.search(r'K: (\d+) million', line)
        if match:
            current_K = int(match.group(1))
        
        # Extract Total Time (for consistency, use total time for CPU too)
        # For CPU: Total Time = Memory Allocation + Execution Time
        match = re.search(r'Total Time\s+:\s+([\d.]+)\s+millisec', line, re.IGNORECASE)
        if not match:
            # Fallback to execution time if total time not found
            match = re.search(r'Execution Time\s+:\s+([\d.]+)\s+millisec', line, re.IGNORECASE)
        if match and current_K:
            cpu_times[current_K] = float(match.group(1))
            current_K = None  # Reset after extraction
    
    return cpu_times

def parse_gpu_output(output_file, has_unified_memory=False):
    """Parse GPU output file and extract execution times for each K and scenario"""
    gpu_times = {
        'scenario1': {},  # 1 block, 1 thread
        'scenario2': {},  # 1 block, 256 threads
        'scenario3': {}   # multiple blocks, 256 threads/block
    }
    
    with open(output_file, 'r') as f:
        lines = f.readlines()
    
    current_scenario = None
    current_K = None
    
    for i, line in enumerate(lines):
        # Detect scenario - check for both uppercase and case variations
        line_upper = line.upper()
        if 'SCENARIO 1' in line_upper or ('1 BLOCK' in line_upper and '1 THREAD' in line_upper):
            current_scenario = 'scenario1'
        elif 'SCENARIO 2' in line_upper or ('1 BLOCK' in line_upper and '256 THREADS' in line_upper):
            current_scenario = 'scenario2'
        elif 'SCENARIO 3' in line_upper or 'MULTIPLE BLOCKS' in line_upper:
            current_scenario = 'scenario3'
        
        # Detect K value - try multiple patterns
        match = re.search(r'K\s*=\s*(\d+)\s*million', line, re.IGNORECASE)
        if not match:
            match = re.search(r'K:\s*(\d+)\s*million', line, re.IGNORECASE)
        if match:
            current_K = int(match.group(1))
        
        # Extract Total Time (includes memory transfers for non-unified memory)
        # For non-unified memory: Total Time = CPU alloc + GPU alloc + memcpy + execution
        # For unified memory: Total Time = unified alloc + execution
        match = re.search(r'Total Time\s*:\s*([\d.]+)\s*millisec', line, re.IGNORECASE)
        if match and current_scenario and current_K:
            time_val = float(match.group(1))
            if time_val > 0:  # Only store non-zero values
                gpu_times[current_scenario][current_K] = time_val
            current_K = None  # Reset after extraction
    
    return gpu_times

def create_plot(cpu_times, gpu_times, title, output_file):
    """Create a log-log plot comparing CPU and GPU execution times"""
    K_values = [1, 5, 10, 50, 100]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Prepare data - filter out zeros for loglog plots
    cpu_y = [cpu_times.get(K, None) for K in K_values]
    gpu_s1_y = [gpu_times['scenario1'].get(K, None) for K in K_values]
    gpu_s2_y = [gpu_times['scenario2'].get(K, None) for K in K_values]
    gpu_s3_y = [gpu_times['scenario3'].get(K, None) for K in K_values]
    
    # Debug: print what we're plotting
    print(f"  CPU times: {dict(zip(K_values, cpu_y))}")
    print(f"  GPU S1 times: {dict(zip(K_values, gpu_s1_y))}")
    print(f"  GPU S2 times: {dict(zip(K_values, gpu_s2_y))}")
    print(f"  GPU S3 times: {dict(zip(K_values, gpu_s3_y))}")
    
    # Filter out None values and prepare for plotting
    def filter_and_plot(x_vals, y_vals, style, label, color):
        """Filter None values and plot"""
        filtered_x = [x for x, y in zip(x_vals, y_vals) if y is not None and y > 0]
        filtered_y = [y for y in y_vals if y is not None and y > 0]
        if filtered_x and filtered_y:
            ax.loglog(filtered_x, filtered_y, style, linewidth=2, markersize=8, label=label, color=color)
    
    # Plot lines
    filter_and_plot(K_values, cpu_y, 'o-', 'CPU', 'black')
    filter_and_plot(K_values, gpu_s1_y, 's-', 'GPU: 1 block, 1 thread', 'blue')
    filter_and_plot(K_values, gpu_s2_y, '^-', 'GPU: 1 block, 256 threads', 'green')
    filter_and_plot(K_values, gpu_s3_y, 'd-', 'GPU: Multiple blocks, 256 threads/block', 'red')
    
    # Formatting
    ax.set_xlabel('K (millions)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Execution Time (milliseconds)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(loc='best', fontsize=10)
    
    # Set x-axis ticks
    ax.set_xticks(K_values)
    ax.set_xticklabels([str(k) for k in K_values])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_file}")
    plt.close()

def main():
    if len(sys.argv) != 4:
        print("Usage: python3 plot_q4.py <cpu_output.txt> <gpu00_output.txt> <gpu01_output.txt>")
        print("  cpu_output.txt: Output from vecaddcpu (run_q1.sh)")
        print("  gpu00_output.txt: Output from vecaddgpu00 (run_q2.sh)")
        print("  gpu01_output.txt: Output from vecaddgpu01 (run_q3.sh)")
        sys.exit(1)
    
    cpu_file = sys.argv[1]
    gpu00_file = sys.argv[2]  # Without unified memory
    gpu01_file = sys.argv[3]  # With unified memory
    
    # Parse outputs
    print("Parsing CPU output...")
    cpu_times = parse_cpu_output(cpu_file)
    print(f"Found CPU times: {cpu_times}")
    
    print("Parsing GPU output (without unified memory)...")
    gpu00_times = parse_gpu_output(gpu00_file, has_unified_memory=False)
    print(f"Found GPU times: {gpu00_times}")
    
    print("Parsing GPU output (with unified memory)...")
    gpu01_times = parse_gpu_output(gpu01_file, has_unified_memory=True)
    print(f"Found GPU times: {gpu01_times}")
    
    # Create plots
    print("\nCreating Chart 1: Step 2 (without Unified Memory)...")
    create_plot(cpu_times, gpu00_times, 
                'Vector Addition: CPU vs GPU (Without Unified Memory)', 
                'q4_chart1_without_unified_memory.png')
    
    print("Creating Chart 2: Step 3 (with Unified Memory)...")
    create_plot(cpu_times, gpu01_times, 
                'Vector Addition: CPU vs GPU (With Unified Memory)', 
                'q4_chart2_with_unified_memory.png')
    
    print("\nDone! Check the generated PNG files.")

if __name__ == '__main__':
    main()

