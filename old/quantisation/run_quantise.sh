#!/bin/bash
#export SLURM_CONF=/cluster/adm/slurm-amdgpu/slurm/etc/slurm.conf
#SBATCH --job-name=saf_regression
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G

module load stack/2024-06 python/3.12.8 eth_proxy
source /cluster/home/lusaur/its/rocm641_python/bin/activate

# Define env variables
export HF_HOME=$SCRATCH/huggingface
export TOKENIZERS_PARALLELISM=true

# login to hugging face
hf auth login --token $(head -n 1 /cluster/home/lusaur/.secrets/hf_token) | cat

python /cluster/home/lusaur/its/quantisation/quantise.py

echo "Job completed: $(date)"