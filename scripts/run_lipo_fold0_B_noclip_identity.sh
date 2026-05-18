#!/bin/bash
#SBATCH --job-name=lipo_v2_B_noclip_identity
#SBATCH --partition=long
#SBATCH --time=4-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -o logs/slurm_%x_%j.out
#SBATCH -e logs/slurm_%x_%j.err

# ============================================================
# Lipo MRI-EDAIN v2  --  PATH B  (no clip + identity + learnable)
# fold 0, full 1000-epoch run.
#
# Rationale:
#   Sidestep the population_nyul slope problem by changing the anchor.
#   With anchor_type=identity, theta_0 fits f(x)=x; the spline starts as
#   a no-op and the backbone trains on the raw upstream-z-scored input
#   (= the user's prior 0.79 baseline pipeline). The hypernet then
#   learns task-driven deviations from identity.
#
#   No outlier clipping: this tests whether identity anchor ALONE is
#   sufficient even on the raw heavy-tailed Lipo z-score distribution.
#
# Compared to PATH A: identity sidesteps the anchor design issue
# completely (no slope problem regardless of source distribution shape),
# but gives up the v2 Nyul story.
# ============================================================

set -euo pipefail
module purge
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source ../myenv/bin/activate

[[ -f ./lipo_split.json ]] || { echo "FATAL: lipo_split.json missing"; exit 1; }
[[ -d ../dataset/lipo  ]] || { echo "FATAL: ../dataset/lipo missing"; exit 1; }

STALE_ARTIFACT="./outputs/lipo_v2_B_noclip_identity/artifacts/fold_0.pt"
if [[ -f "$STALE_ARTIFACT" ]]; then
    echo "[init] removing stale precompute artifact $STALE_ARTIFACT"
    rm -f "$STALE_ARTIFACT"
fi

export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1

echo "Fold: 0 PATH B | Start: $(date)"

python code/lipo_mri_edain_v2.py \
    --fold 0 \
    --gpu 0 \
    --data_dir ../dataset/lipo \
    --split_json ./lipo_split.json \
    --out_dir ./outputs/lipo_v2_B_noclip_identity \
    --epochs 1000 \
    --seed 2025 \
    --batch_size 1 \
    --num_patches 2 \
    --patch_size 256,224,32 \
    --max_channels 320 \
    --hypernet_lr_factor 0.1 \
    --outlier_clip none \
    --anchor_type identity

echo "End: $(date)"
