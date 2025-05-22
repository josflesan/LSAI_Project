#!/bin/bash

#SBATCH --account=a-large-sc
#SBATCH --time=00:14:59
#SBATCH --job-name=lsai
#SBATCH --output=/iopsstor/scratch/cscs/%u/LSAI_Project/logs/runs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --environment=/users/aoudrhiri/scratch/ngc_pt_jan.toml     # Vanilla 25.01 PyTorch NGC Image 
#SBATCH --no-requeue	# Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs

# Stop the script if a command fails or if an undefined variable is used
set -eo pipefail

echo "START TIME: $(date)"

# Print SLURM variables so you see how your resources are allocated
echo "Job Name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated Node(s): $SLURM_NODELIST"
echo "Number of Tasks: $SLURM_NTASKS"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "GPUs per Node: $SLURM_GPUS_PER_NODE"
echo "Number of Nodes: : $SLURM_NNODES"
echo "Number of Tasks per Node: $SLURM_NTASKS_PER_NODE"
echo "Current path: $(pwd)"s
echo "Current user: $(whoami)"

# Set up ENV
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ASSIGNMENT_DIR="/iopsstor/scratch/cscs/$USER/LSAI_Project/src"

# The defined environment vars will be shared with the other compute nodes.
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=12345  # Choose an unused port
export FOOBAR=666
export WORLD_SIZE=$(( SLURM_NNODES * SLURM_GPUS_PER_NODE ))  #TODO: maybe we should change this back to SLURM_NTASKS_PER_NODE

echo "execute command on compute nodes"

# The command that will run on each process
CMD="
# print current environment variables
echo \"[srun] rank=\$SLURM_PROCID host=\$(hostname) noderank=\$SLURM_NODEID localrank=\$SLURM_LOCALID\"

torchrun \
  --nnodes="${SLURM_NNODES}" \
  --node_rank=\$SLURM_NODEID \
  --nproc_per_node=1 \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  /iopsstor/scratch/cscs/$USER/LSAI_Project/src/train.py \
  --sequence-length 4096 \
  --batch-size 1 \
  --learning-rate 5e-5 \
  --lr-warmup-steps 10 \
  --training-steps 1000 \
  --logging-frequency 5 \
  --experiment baseline \
"

srun bash -c "$CMD"

echo "END TIME: $(date)"
