#!/bin/bash
#SBATCH --job-name=lipo_mri_edain_v2_f0
#SBATCH --partition=long
#SBATCH --time=4-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -o logs/slurm_%x_%j.out
#SBATCH -e logs/slurm_%x_%j.err

# ============================================================
# Lipo MRI-EDAIN v2 -- fold 0, full 1000-epoch run.
#
# Pipeline: Load -> Orientation(RAS) -> Spacing
#           -> CropForeground(Otsu) -> NormalizeIntensity(zscore on fg)
#           -> RandCropByPosNegLabel patches -> nnU-Net augmentations
#           -> MRIEDAINLayer (gamma -> hypernet -> theta = theta_0 + delta
#               -> RQ-spline f_theta voxel-wise) -> DynUNet
#           -> DiceCE + lambda_anc * function-space anchor + lambda_KL * KL
#
# Precompute (auto on first invocation, ~3-5 min on 100+ cases):
#   gamma per case -> fit CoordinateStandardizer (mu, sigma per percentile slot)
#   -> fit theta_0 via L-BFGS so the spline approximates the population Nyul
#      piecewise-linear mapping.  Cached at outputs/.../artifacts/fold_0.pt.
#
# 3-phase lambda schedule (blueprint section 5.1):
#   Phase 0 (0-1%):   hypernet frozen, train backbone on f_{theta_0}(X).
#   Phase 1 (1-10%):  hypernet unfrozen, lambda_anc = 1e-2, lambda_KL = 0.
#   Phase 2 (10-100%):lambda_anc cosine-decay to 1e-4,
#                     lambda_KL ramp 0 -> 1e-4 over first 5% of phase 2.
#
# EMA shadow of hypernet (decay 0.99) swapped in at every validation.
# Phase-I diagnostic Metric 1 (non-affineness r_i median + p90) logged per val.
# ============================================================

set -euo pipefail
module purge
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source ../myenv/bin/activate

[[ -f ./lipo_split.json ]] || { echo "FATAL: lipo_split.json missing in $(pwd)"; exit 1; }
[[ -d ../dataset/lipo  ]] || { echo "FATAL: ../dataset/lipo missing";              exit 1; }

export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
# Help CUDA allocator handle fragmentation on smaller GPUs (e.g., 2080Ti 11GB).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Make `from mri_edain_v2 import ...` work from anywhere.
export PYTHONPATH="$SLURM_SUBMIT_DIR"

echo "Fold: 0 | Start: $(date)"

# NOTE: num_patches reduced from 4 -> 2 for 2080Ti compatibility, matching v11.
# Comment out --num_patches (or set 4) on A40 / similar to use default.
python code/lipo_mri_edain_v2.py \
    --fold 0 \
    --gpu 0 \
    --data_dir ../dataset/lipo \
    --split_json ./lipo_split.json \
    --out_dir ./outputs/lipo_mri_edain_v2 \
    --epochs 1000 \
    --num_patches 2 \
    --seed 2025

echo "End: $(date)"
