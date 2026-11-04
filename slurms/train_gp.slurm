#!/bin/bash -l
#SBATCH --job-name=train_gp
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

python -u run_gp.py \
    task=gp task.dim_x=1 task.n_target_theta=2 \
    task.lengthscale_lower=0.1 task.lengthscale_upper=2.0 task.design_scale=5 \
    task.prob_good_trajectory=0.7 \
    use_trajectories=True n_summary_tokens=4 gating=True \
    T=15 gamma=1 max_epoch=250000 burning_epoch=30000 \
    eval_interval=1000 wandb.use_wandb=False
