#!/bin/bash -l
#SBATCH --job-name=train_ces
#SBATCH --time=24:00:00
#SBATCH --mem=50G
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=gpu-h100-80g
#SBATCH --output=out/%x_%j.txt

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"  # Submit from the repository root.
mkdir -p out
source .venv/bin/activate
export PYTHONUNBUFFERED=1

python -u run_ces.py \
    task=ces task.prob_good_trajectory=0.7 task.prior_injection=False \
    use_trajectories=True n_summary_tokens=2 gating=True \
    T=10 max_epoch=250000 burning_epoch=30000 \
    eval_interval=1000 \
    L_final=10000000 eval_batch_size_final=20 T_final=10 \
    wandb.use_wandb=False
