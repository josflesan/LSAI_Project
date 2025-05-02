#!/bin/bash

#SBATCH --account=a-large-sc
#SBATCH --time=00:14:59
#SBATCH --job-name=lsai
#SBATCH --output=/iopsstor/scratch/cscs/%u/assignment-2/logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --environment=/users/aoudrhiri/scratch/ngc_pt_jan.toml     # Vanilla 25.01 PyTorch NGC Image 
#SBATCH --no-requeue	# Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs

echo "START TIME: $(date)"

# Set up ENV
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ASSIGNMENT_DIR="/iopsstor/scratch/cscs/$USER/assignment-2"



TRAINING_CMD="nsys profile -s none -w true \
--trace='nvtx,cudnn,cublas,cuda' \
--output=/iopsstor/scratch/cscs/$USER/assignment-2/nsys-trace.nsys-rep \
--force-overwrite true \
--capture-range=cudaProfilerApi \
--capture-range-end=stop -x true numactl --membind=0-3 python3 train.py --profile"

srun bash -c "$TRAINING_CMD"

echo "END TIME: $(date)"