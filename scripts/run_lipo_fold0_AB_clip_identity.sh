#!/bin/bash
#SBATCH --job-name=lipo_v2_AB_clip_identity
#SBATCH --partition=long
#SBATCH --time=4-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -o logs/slurm_%x_%j.out
#SBATCH -e logs/slurm_%x_%j.err

# ============================================================
# Lipo MRI-EDAIN v2  --  PATH A+B  (clip + identity + learnable)
# fold 0, full 1000-epoch run.
#
# Rationale:
#   Belt-and-suspenders combination. Both:
#     (A) Foreground percentile clip before z-score -> robust std,
#         near-Gaussian post-z-score distribution.
#     (B) Identity anchor -> spline starts as a no-op, hypernet learns
#         pure task-driven deviations.
#
#   Should reach AT LEAST whichever of A or B alone is best, and
#   probably slightly more. This is the "safest" v2 run scientifically.
#
#   The trade-off: like Path B, this gives up the v2 Nyul framing
#   (anchor is identity, not Nyul). Plan B language in the paper.
# ============================================================

set -euo pipefail
module purge
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source ../myenv/bin/activate

[[ -f ./lipo_split.json ]] || { echo "FATAL: lipo_split.json missing"; exit 1; }
[[ -d ../dataset/lipo  ]] || { echo "FATAL: ../dataset/lipo missing"; exit 1; }

STALE_ARTIFACT="./outputs/lipo_v2_AB_clip_identity/artifacts/fold_0.pt"
if [[ -f "$STALE_ARTIFACT" ]]; then
    echo "[init] removing stale precompute artifact $STALE_ARTIFACT"
    rm -f "$STALE_ARTIFACT"
fi

export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1

echo "Fold: 0 PATH A+B | Start: $(date)"

python code/lipo_mri_edain_v2.py \
    --fold 0 \
    --gpu 0 \
    --data_dir ../dataset/lipo \
    --split_json ./lipo_split.json \
    --out_dir ./outputs/lipo_v2_AB_clip_identity \
    --epochs 1000 \
    --seed 2025 \
    --batch_size 1 \
    --num_patches 2 \
    --patch_size 256,224,32 \
    --max_channels 320 \
    --hypernet_lr_factor 0.1 \
    --outlier_clip percentile \
    --anchor_type identity

echo "End: $(date)"
