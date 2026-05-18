#!/bin/bash
#SBATCH --job-name=lipo_baseline_zscore_f0
#SBATCH --partition=long
#SBATCH --time=4-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -o logs/slurm_%x_%j.out
#SBATCH -e logs/slurm_%x_%j.err

# ============================================================
# Lipo BASELINE  ~=  vanilla per-volume z-score + DynUNet
# (fold 0, full 1000-epoch run)
#
# This is the REPRODUCTION TARGET for the user's previous Lipo result
# (~0.79 Val Dice). It runs the EXACT same code path as the v2 main
# run, with two flags that collectively turn the spline into a no-op:
#
#     --frozen_hypernet           : hypernet stays frozen for all 1000 ep,
#                                   so theta = theta_0 forever.
#     --anchor_type identity      : theta_0 is fit to the identity function
#                                   (target(x) = x), so f_{theta_0}(x) ~ x
#                                   for all x in the support.
#
# Net effect inside MRIEDAINLayer:
#     gamma_raw -> ignored (hypernet frozen at zero output)
#     theta = theta_0 (frozen, identity-ish)
#     spline = near-identity, voxel-wise applied
#     X_tilde ~ X (no meaningful change)
#
# So the backbone trains on the upstream-z-scored MONAI pipeline output
# directly. This MUST match a vanilla z-score-only baseline on Lipo to
# within ~0.01 Val Dice (the spline's tiny residual deviation from
# identity at the boundary).
#
# Why we need this baseline:
# ---------------------------------------------------------------
# The v2 main run on Lipo fold 0 plateaued at Val Dice ~0.61 by ep 174.
# The user's prior Lipo baseline reached ~0.79. The 0.18 gap is way
# larger than δ_min, and the diagnostic numbers (r_med dropping from
# 0.30 to 0.07; large local slopes in the population Nyul mapping)
# pointed at the *anchor design* as the cause, not at the rest of
# the pipeline (augmentation, optimizer, etc.).
#
# If this baseline reaches ~0.79 -> anchor design IS the culprit.
# If this baseline also stalls at ~0.6 -> something else in our pipeline
#   (augmentation, transform pipeline, deep supervision weights, etc.)
#   is the problem and v2 anchor is a red herring.
#
# Run in parallel with the main v2 run; each takes one GPU.
# ============================================================

set -euo pipefail
module purge
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source ../myenv/bin/activate

[[ -f ./lipo_split.json ]] || { echo "FATAL: lipo_split.json missing in $(pwd)"; exit 1; }
[[ -d ../dataset/lipo  ]] || { echo "FATAL: ../dataset/lipo missing";              exit 1; }

STALE_ARTIFACT="./outputs/lipo_baseline_zscore/artifacts/fold_0.pt"
if [[ -f "$STALE_ARTIFACT" ]]; then
    echo "[init] removing stale precompute artifact $STALE_ARTIFACT"
    rm -f "$STALE_ARTIFACT"
fi

export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1

echo "Fold: 0 (vanilla z-score baseline via identity anchor + frozen hypernet) | Start: $(date)"

python code/lipo_mri_edain_v2.py \
    --fold 0 \
    --gpu 0 \
    --data_dir ../dataset/lipo \
    --split_json ./lipo_split.json \
    --out_dir ./outputs/lipo_baseline_zscore \
    --epochs 1000 \
    --seed 2025 \
    --batch_size 1 \
    --num_patches 2 \
    --patch_size 256,224,32 \
    --max_channels 320 \
    --frozen_hypernet \
    --anchor_type identity \
    --outlier_clip percentile

echo "End: $(date)"
