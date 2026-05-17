# MRI-EDAIN v2

Input-conditional rational-quadratic monotone spline normalization for MRI
segmentation, implementing the spec in
[`MRI-EDAIN-v2-Complete-Blueprint.md`](MRI-EDAIN-v2-Complete-Blueprint.md).

The method replaces nnU-Net's per-volume z-score with a learnable nonlinear
mapping conditioned on a per-volume percentile summary, anchored at the
population Nyul mapping and regularised in function space.

## Layout

```
.
├── MRI-EDAIN-v2-Complete-Blueprint.md   # spec (single source of truth)
├── mri_edain_v2/                        # the implementation
│   ├── modules/                         # core: foreground, gamma, standardiser,
│   │                                    #       RQ-spline, hypernet, Nyul init,
│   │                                    #       MRIEDAINLayer
│   ├── losses/                          # function-space anchor, KL, combiner
│   ├── baselines/                       # affine-hypernet kill-switch (#6)
│   ├── training/                        # precompute, 3-phase scheduler, EMA
│   └── tests/                           # 26 unit tests (pytest)
└── code/
    ├── brats_mri_edain_v2_t1n.py        # BraTS T1n entry (5-fold CV)
    └── lipo_mri_edain_v2.py             # Lipo MR entry (5-fold CV)
```

## Expected directory layout on the cluster

```
<parent>/
├── MRI_EDAIN_2/         # this repo (git clone target)
└── dataset/
    ├── brats/
    │   ├── brats_t1n/   # <case>-t1n.nii.gz + <case>-seg.nii.gz
    │   └── split.json
    └── lipo/            # Lipo-XXX_MR_N_image.nii.gz + Lipo-XXX_MR_N_segmentation.nii.gz
    └── lipo_split.json  # (can also live inside dataset/lipo/)
```

Both training scripts auto-detect this default; override with
`--data_dir <path>` if your layout differs.

## Installation

```bash
# CUDA 12.4 (PyTorch wheel; adjust if your cluster uses a different CUDA)
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Quick verification

The full unit-test suite covers the blueprint section 12.5 checklist
(monotonicity, identity behaviour, zero-init, anchor invariance, batched
support, EMA, etc.).

```bash
cd MRI_EDAIN_2
PYTHONPATH=. pytest mri_edain_v2/tests/test_modules.py -v
# 26 passed
```

A smoke test runs precompute + 2 mini-epochs on 5 train / 2 val cases:

```bash
# BraTS
PYTHONPATH=. python code/brats_mri_edain_v2_t1n.py --smoke

# Lipo
PYTHONPATH=. python code/lipo_mri_edain_v2.py --smoke
```

Each smoke test completes in about 20 s on a single GPU and verifies the
full forward + backward + validation + EMA + checkpoint pathway.

## Running a single fold

```bash
# BraTS T1n, fold 0
PYTHONPATH=. python code/brats_mri_edain_v2_t1n.py --fold 0 --gpu 0

# Lipo, fold 0 (using lipo_split.json fold ids 0..4)
PYTHONPATH=. python code/lipo_mri_edain_v2.py --fold 0 --gpu 0
```

Outputs land in:

```
./outputs/{brats,lipo}_mri_edain_v2/fold_<id>/best_model.pt
./logs/{brats,lipo}_mri_edain_v2/fold_<id>/training_log.csv
./artifacts/{brats,lipo}_mri_edain_v2/fold_<id>.pt   # standardiser + theta_0
```

The precompute artifact is reused across seeds within a fold; delete it to
re-fit the population-Nyul anchor.

## Cluster launch examples (SLURM)

```bash
# 5 folds of BraTS in parallel
for i in 0 1 2 3 4; do
    sbatch --gres=gpu:1 --time=72:00:00 \
        --wrap="PYTHONPATH=$PWD python code/brats_mri_edain_v2_t1n.py --fold $i --gpu 0"
done

# Lipo
for i in 0 1 2 3 4; do
    sbatch --gres=gpu:1 --time=48:00:00 \
        --wrap="PYTHONPATH=$PWD python code/lipo_mri_edain_v2.py --fold $i --gpu 0"
done
```

## Key CLI flags

| Flag | Purpose |
|---|---|
| `--fold N` | Train a single fold (parallel sbatch). |
| `--gpu N` | CUDA device id. |
| `--smoke` | 5/2 mini-train mode for verification (~20 s). |
| `--epochs N` | Override `MAX_EPOCHS` (default 1000). |
| `--seed N` | Determinism seed (default 2025). |
| `--data_dir PATH` | Override dataset location. |
| `--split_json PATH` | (Lipo only) override split file location. |
| `--out_dir PATH` | Re-root outputs/logs/artifacts. |

## Training behaviour (per blueprint section 5)

| Step | What happens |
|---|---|
| Precompute (once per fold) | Computes `gamma` on every training case, fits the `CoordinateStandardizer` (per-percentile mu, sigma), fits `theta_0` via L-BFGS so the RQ-spline approximates the population-Nyul mapping. Cached to `./artifacts/`. |
| Phase 0 (steps 0 to 1% of total) | Hypernet FROZEN. U-Net trains on `f_{theta_0}(X)`. |
| Phase 1 (1% to 10%) | Hypernet unfrozen. `lambda_anc = 1e-2` (strong anchor). `lambda_KL = 0`. |
| Phase 2 (10% to 100%) | `lambda_anc` cosine-decays to 1e-4. `lambda_KL` ramps 0 to 1e-4 over the first 5% of Phase 2. |
| Every step in Phase 1/2 | EMA shadow of hypernet updated (decay 0.99). |
| Every validation | EMA hypernet swapped in. Phase-I diagnostic `r_i` (non-affineness ratio) logged per case. |

## What is NOT yet wired (blueprint section 8-10)

These are scaffolded in the package but not yet hooked into the trainer:
- The other 4 Phase-I diagnostic metrics: `CV(f')`, `eta_i` (post-IN survival),
  effective rank, `kappa_i` (tumor contrast preservation).
- Phase II / Phase III diagnostic scripts (offline; run post-training).
- Baselines #1, #3, #4, #5, #7 (NoNorm, PercClip, NyulFixed, RQSplineFixed,
  WhiteStripe). Only baseline #6 (Affine-Hypernet kill-switch) is implemented.

These are next on the roadmap; PRs welcome.

## Reference

Implementation of the spec in
[`MRI-EDAIN-v2-Complete-Blueprint.md`](MRI-EDAIN-v2-Complete-Blueprint.md);
that document is the authoritative source for design decisions, error
history, and the abandonment / pivot criteria.
