#!/bin/bash
#SBATCH --job-name=saf_regression
#SBATCH --output=saf_regression_%j.out
#SBATCH --error=saf_regression_%j.err
#SBATCH --time=00:15:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Load necessary modules (adjust based on your cluster's module system)
module purge
module load python/3.9
module load cuda/11.8

# Option 1: Install packages using pip (recommended for this project)
echo "Installing Python packages..."
# pip install -e .

# Option 2: Activate existing virtual environment (uncomment if you have one)
# source /path/to/your/venv/bin/activate

# Option 3: Create and activate a new virtual environment (uncomment if needed)
# python -m venv venv
# source venv/bin/activate
pip install -e .

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Run the Python script
python saf_regression_slurm.py

# Print completion time
echo "End time: $(date)"
echo "Job completed"
