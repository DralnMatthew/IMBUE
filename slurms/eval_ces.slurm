#!/bin/bash -l
#SBATCH --job-name=eval_ces
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

python -u scripts/eval/ces.py \
    --pth_path "$PTH" --cfg_yaml "$CFG" \
    --out_dir results/ces --name_prefix imbue \
    --pool_type uniform --n_query 2000 \
    --T 10 --source_T 10 \
    --seed "$SEED"
