#!/bin/bash
#SBATCH --job-name=lipo_v2_A_clip_popnyul
#SBATCH --partition=long
#SBATCH --time=4-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -o logs/slurm_%x_%j.out
#SBATCH -e logs/slurm_%x_%j.err

# ============================================================
# Lipo MRI-EDAIN v2  --  PATH A  (clip + population_nyul + learnable)
# fold 0, full 1000-epoch run.
#
# Rationale:
#   Outlier voxels in raw Lipo MR inflate per-volume z-score std by ~4x,
#   compressing the post-z-score body distribution. The population_nyul
#   anchor then has to develop ~30x slopes in dense bins (1% -> 10%) to
#   match a Gaussian-percentile target -- amplifying noise, hurting Dice.
#
#   Fix: clip foreground voxels to [0.5%, 99.5%] percentile range BEFORE
#   per-volume z-score (--outlier_clip percentile). std becomes robust,
#   post-z-score body distribution becomes near-Gaussian, and the
#   population_nyul mapping naturally develops slopes near 1.
#
# This is the run that PRESERVES the v2 paper's Nyul framing while
# fixing the slope-explosion bug. If it reaches ~0.75+, the v2 Nyul
# story stands.
# ============================================================

set -euo pipefail
module purge
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source ../myenv/bin/activate

[[ -f ./lipo_split.json ]] || { echo "FATAL: lipo_split.json missing"; exit 1; }
[[ -d ../dataset/lipo  ]] || { echo "FATAL: ../dataset/lipo missing"; exit 1; }

STALE_ARTIFACT="./outputs/lipo_v2_A_clip_popnyul/artifacts/fold_0.pt"
if [[ -f "$STALE_ARTIFACT" ]]; then
    echo "[init] removing stale precompute artifact $STALE_ARTIFACT"
    rm -f "$STALE_ARTIFACT"
fi

export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1

echo "Fold: 0 PATH A | Start: $(date)"

python code/lipo_mri_edain_v2.py \
    --fold 0 \
    --gpu 0 \
    --data_dir ../dataset/lipo \
    --split_json ./lipo_split.json \
    --out_dir ./outputs/lipo_v2_A_clip_popnyul \
    --epochs 1000 \
    --seed 2025 \
    --batch_size 1 \
    --num_patches 2 \
    --patch_size 256,224,32 \
    --max_channels 320 \
    --hypernet_lr_factor 0.1 \
    --outlier_clip percentile \
    --anchor_type population_nyul

echo "End: $(date)"
