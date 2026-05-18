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
# Lipo MRI-EDAIN v2 -- fold 0, full 1000-epoch run, 2080Ti-sized.
#
# Memory budget (RTX 2080Ti, 11 GB):
#   patch_size_max = 128  -> auto patch X/Y capped at 128 (vs 192 default).
#                            Z falls out at 40 = floor(median_47 / stride_8) * 8
#                            (was 48 = ceil pre-fix, which overshot median 47
#                            and forced zero-padding for short-Z samples,
#                            erasing tumor context).
#   max_channels   = 256  -> DynUNet caps filter count per level at 256
#                            (nnU-Net's reference plan uses 320; default 512).
#   batch_size     = 1
#   num_patches    = 2    -> 2 patches / forward step (vs 4 default).
#
# Reference nnU-Net plan for this dataset:
#   patch=[32,224,256] (ZYX) channels=[32,64,128,256,320,320].
#   We trade some patch coverage on Y/X for memory; Z=40 still gives ~85%
#   coverage of the median image while staying divisible by total stride 8.
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

# Force-reset the precompute artifact if it was produced by an earlier version
# that pre-dates the percentile-subsample fix (some Lipo cases would fail with
# `quantile() input tensor is too large` and the standardizer would be fit on
# the surviving subset only). With the fix in place we want a clean recompute.
STALE_ARTIFACT="./outputs/lipo_mri_edain_v2/artifacts/fold_0.pt"
if [[ -f "$STALE_ARTIFACT" ]]; then
    echo "[init] removing stale precompute artifact $STALE_ARTIFACT"
    rm -f "$STALE_ARTIFACT"
fi

export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
# Help CUDA allocator handle fragmentation on smaller GPUs (e.g., 2080Ti 11GB).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Make `from mri_edain_v2 import ...` work from anywhere.
export PYTHONPATH="$SLURM_SUBMIT_DIR"

echo "Fold: 0 | Start: $(date)"

python code/lipo_mri_edain_v2.py \
    --fold 0 \
    --gpu 0 \
    --data_dir ../dataset/lipo \
    --split_json ./lipo_split.json \
    --out_dir ./outputs/lipo_mri_edain_v2 \
    --epochs 1000 \
    --seed 2025 \
    --batch_size 1 \
    --num_patches 2 \
    --patch_size_max 128 \
    --max_channels 256

echo "End: $(date)"
