#!/bin/bash -l
#SBATCH --job-name=eval_location
#SBATCH --time=1:00:00
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

RUN_DIR=${RUN_DIR:?set RUN_DIR to a Location Finding training run directory}
PTH="$RUN_DIR/model/aae_Location_d2_1.pth"
CFG="$RUN_DIR/.hydra/config.yaml"
SEED=0

python -u scripts/eval/location.py \
    --pth_path "$PTH" --cfg_yaml "$CFG" \
    --out_dir results/location --name_prefix imbue \
    --pool_type uniform --n_query 2000 \
    --conditions strong weak miscalibrated uniform \
    --seed "$SEED"

python -u scripts/eval/location_ood.py \
    --pth_path "$PTH" --cfg_yaml "$CFG" \
    --out_dir results/location --name_prefix lf_ood_prior \
    --n_query 2000 --conditions truncated boundary \
    --seed "$SEED"
