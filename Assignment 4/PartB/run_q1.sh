#!/bin/bash
#SBATCH --job-name=vecadd_cpu_q1
#SBATCH --output=vecadd_cpu_q1_%j.out
#SBATCH --error=vecadd_cpu_q1_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

# Load necessary modules (adjust based on your HPC system)
# Uncomment and modify as needed:
# module load gcc
# module load cuda

cd $SLURM_SUBMIT_DIR

echo "=========================================="
echo "CPU Vector Addition - Q1 Results"
echo "Date: $(date)"
echo "=========================================="
echo ""

# Run for K = 1, 5, 10, 50, 100
for K in 1 5 10 50 100
do
    echo "----------------------------------------"
    echo "Running for K = $K million"
    echo "----------------------------------------"
    ./vecaddcpu $K
    echo ""
done

echo "=========================================="
echo "All tests completed at $(date)"
echo "=========================================="

