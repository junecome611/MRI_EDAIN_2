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
# Configuration mirrors nnU-Net's plan for this Lipo dataset exactly
# (Dataset500_Lipo, 3d_fullres):
#
#     patch  [Z=32, Y=224, X=256]   in our XYZ axis order: (256, 224, 32)
#     features_per_stage [32, 64, 128, 256, 320, 320]    (6 stages)
#     strides ZYX [[1,1,1],[1,2,2],[1,2,2],[2,2,2],[2,2,2],[2,2,2]]
#     -> auto-derived in our code from patch + anisotropy axis
#     batch_size 2  (nnU-Net effective: 2 samples per step)
#         in MONAI: --batch_size 1 --num_patches 2 -> 2 patches/step
#
# Rationale for not uniformly capping patch (vs old --patch_size_max 128):
#   - Z is naturally small (axial through-plane spacing 3.84 mm, median 47
#     slices) so 32 covers ~70% of median image, divisible by total_stride 8.
#   - X / Y at 256 / 224 give large in-plane context, important for soft-
#     tissue tumour localization where surrounding organ anatomy is the
#     primary landmark cue. A uniform 128 cap throws away that context.
#
# Memory budget on 2080Ti 11GB (estimated with AMP autocast):
#     params + grads + SGD-momentum  ~250 MB
#     encoder activations (FP16)     ~420 MB / 2 patches
#     decoder activations            ~420 MB
#     CUDNN workspace               ~800 MB - 1 GB
#     ----
#     Total                         ~2.7-3.0 GB peak       (well under 11)
#
# Pipeline: Load -> Orientation(RAS) -> Spacing
#           -> CropForeground(Otsu) -> NormalizeIntensity(zscore on fg)
#           -> RandCropByPosNegLabel patches -> nnU-Net augmentations
#           -> MRIEDAINLayer (gamma -> hypernet -> theta = theta_0 + delta
#               -> RQ-spline f_theta voxel-wise) -> DynUNet
#           -> DiceCE + lambda_anc * function-space anchor + lambda_KL * KL
#
# 3-phase lambda schedule (blueprint section 5.1):
#   Phase 0 (0-1%):    hypernet frozen, train backbone on f_{theta_0}(X).
#   Phase 1 (1-10%):   hypernet unfrozen, lambda_anc = 1e-2, lambda_KL = 0.
#   Phase 2 (10-100%): lambda_anc cosine-decay to 1e-4,
#                      lambda_KL ramp 0 -> 1e-4 over first 5% of phase 2.
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

# Force-reset the precompute artifact when relevant code changes happen
# (patch_size, normalization, foreground method). The standardizer mu/sigma
# and theta_0 are tied to the gamma distribution which is sensitive to those.
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
# Unbuffered stdout so per-epoch progress shows up in slurm-*.out immediately.
export PYTHONUNBUFFERED=1

echo "Fold: 0 | Start: $(date)"

# --patch_size is XYZ in this code's axis order (matches MONAI orientation).
# nnU-Net's plan reports ZYX = [32, 224, 256]; transposed to XYZ that's
# 256,224,32. --max_channels 320 matches the plan's features_per_stage cap.
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
    --patch_size 256,224,32 \
    --max_channels 320

echo "End: $(date)"
