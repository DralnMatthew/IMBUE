#!/bin/bash -l
#SBATCH --job-name=train_location
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

python -u run_bed.py \
    task=location_finding task.n_target_theta=4 task.K=2 \
    task.prior_injection=True task.theta_dist=ace task.prob_good_prior=0.7 \
    gating=True \
    lr=1e-3 T=20 max_epoch=70000 burning_epoch=15000 \
    eval_interval=1000 eval_batch_size=1000 \
    L_final=1000000 eval_batch_size_final=200 T_final=30 \
    wandb.use_wandb=True
