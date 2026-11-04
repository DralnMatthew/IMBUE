#!/bin/bash -l
#SBATCH --job-name=eval_ces_ood
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

RUN_DIR=${RUN_DIR:?set RUN_DIR to a CES training run directory}
PTH="$RUN_DIR/model/aae_CES_d6_1.pth"
CFG="$RUN_DIR/.hydra/config.yaml"
SEED=0

python -u scripts/eval/ces_ood_bias.py \
    --pth_path "$PTH" --cfg_yaml "$CFG" \
    --out_dir results/ces --name_prefix ood_bias \
    --kinds no_aux relevant biased irrelevant \
    --K 5 --T 10 --source_T 10 --bias_scale 0.3 \
    --seed "$SEED"

python -u scripts/eval/ces_ood_noise.py \
    --pth_path "$PTH" --cfg_yaml "$CFG" \
    --out_dir results/ces --name_prefix ood_noise \
    --K 5 --T 10 --source_T 10 --df 3 --scale_mult 3 \
    --seed "$SEED"

python -u scripts/eval/ces_gating.py \
    --pth_path "$PTH" --cfg_yaml "$CFG" \
    --out_dir results/ces --kinds relevant studentt biased irrelevant \
    --K 5 --T 10 --M 2000 \
    --seed "$SEED"
