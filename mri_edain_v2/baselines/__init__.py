"""Baseline normalization layers (blueprint section 6).

The list:
    #1 NoNorm           pass-through (sanity floor)
    #2 ZScore           per-volume foreground z-score (nnU-Net default; the bar)
    #3 PercentileClip   [0.5, 99.5] clip + min-max
    #4 NyulFixed        Shah-2011 piecewise-linear (trained landmarks frozen)
    #5 RQSplineFixed    main method with hypernet frozen at theta_0 (no per-image)
    #6 AffineHypernet   THE KILL-SWITCH (P2)
    #7 WhiteStripe      BraTS only (Shinohara 2014)

Only the kill-switch baseline is implemented in this phase. The simpler
baselines (#1-#5, #7) are added in the next phase together with the trainer.
"""

from mri_edain_v2.baselines.affine_hypernet import AffineHypernetLayer

__all__ = ["AffineHypernetLayer"]
