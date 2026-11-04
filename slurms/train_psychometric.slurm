#!/bin/bash -l
#SBATCH --job-name=train_psychometric
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

python -u run_psychometric.py \
    task=psychometric task.prob_good_trajectory=0.7 \
    use_trajectories=True n_summary_tokens=4 gating=True \
    T=30 max_epoch=120000 burning_epoch=15000 \
    eval_interval=1000 wandb.use_wandb=False
