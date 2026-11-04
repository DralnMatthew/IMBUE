#!/bin/bash -l
#SBATCH --job-name=eval_psychometric
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

RUN_DIR=${RUN_DIR:?set RUN_DIR to a psychometric training run directory}
PTH="$RUN_DIR/model/aae_Psychometric_d1_1.pth"
CFG="$RUN_DIR/.hydra/config.yaml"
SEED=0

python -u scripts/eval/psychometric.py \
    --pth_path "$PTH" --cfg_yaml "$CFG" \
    --out_dir results/psychometric --name_prefix imbue \
    --M 500 --T 30 --T_aux 30 \
    --batch_size 64 --seed "$SEED"
