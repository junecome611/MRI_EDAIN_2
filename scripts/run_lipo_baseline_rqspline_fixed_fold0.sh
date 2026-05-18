#!/bin/bash
#SBATCH --job-name=lipo_baseline_rqs_f0
#SBATCH --partition=long
#SBATCH --time=4-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -o logs/slurm_%x_%j.out
#SBATCH -e logs/slurm_%x_%j.err

# ============================================================
# Lipo BASELINE #5 (RQSplineFixed) -- fold 0, full 1000-epoch run.
#
# This is the CONTROL for the v2 main method. It runs the EXACT same
# pipeline as scripts/run_lipo_fold0.sh, with --frozen_hypernet:
#
#   * Spline parameters are FROZEN at theta_0 (population Nyul) for all
#     1000 epochs. The hypernet never updates.
#   * anc_loss and KL_loss stay at 0 throughout (because Delta theta = 0).
#   * Only the DynUNet backbone trains.
#
# Result: "DynUNet trained on f_{theta_0}(X)" = population-Nyul-fixed
# normalization with the v2 RQ-spline parameterization.
#
# Why we need this baseline:
# ---------------------------------------------------------------
# Without this run, any Dice improvement during phase 1 of the v2 main
# run could be attributed either to (a) the hypernet learning a useful
# task-driven spline deviation, or to (b) the backbone simply continuing
# to train for more epochs. Only by comparing v2 (main) versus
# v2-with-frozen-hypernet (this) at the SAME training budget can we
# isolate the spline-learning contribution.
#
# Pairwise comparison (blueprint section 6.3):
#   #5 (this run) vs #8 (v2 main, run_lipo_fold0.sh) at convergence:
#     -> answers "does input-conditional spline learning beat fixed Nyul?"
#
# Run both in parallel: each one sbatch, each one GPU.
# ============================================================

set -euo pipefail
module purge
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source ../myenv/bin/activate

[[ -f ./lipo_split.json ]] || { echo "FATAL: lipo_split.json missing in $(pwd)"; exit 1; }
[[ -d ../dataset/lipo  ]] || { echo "FATAL: ../dataset/lipo missing";              exit 1; }

# Note: precompute artifact is per-out_dir, so baseline gets its own copy.
# Same fingerprint, same gamma, same theta_0 by construction.
STALE_ARTIFACT="./outputs/lipo_baseline_rqsplinefixed/artifacts/fold_0.pt"
if [[ -f "$STALE_ARTIFACT" ]]; then
    echo "[init] removing stale precompute artifact $STALE_ARTIFACT"
    rm -f "$STALE_ARTIFACT"
fi

export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1

echo "Fold: 0 (baseline #5 RQSplineFixed) | Start: $(date)"

python code/lipo_mri_edain_v2.py \
    --fold 0 \
    --gpu 0 \
    --data_dir ../dataset/lipo \
    --split_json ./lipo_split.json \
    --out_dir ./outputs/lipo_baseline_rqsplinefixed \
    --epochs 1000 \
    --seed 2025 \
    --batch_size 1 \
    --num_patches 2 \
    --patch_size 256,224,32 \
    --max_channels 320 \
    --frozen_hypernet

echo "End: $(date)"
