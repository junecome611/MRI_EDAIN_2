"""Core modules for MRI-EDAIN v2 (blueprint section 3.2)."""

from mri_edain_v2.modules.foreground import ForegroundExtractor
from mri_edain_v2.modules.percentile import (
    PERCENTILES,
    PercentileSummary,
    percentile_summary,
)
from mri_edain_v2.modules.standardizer import CoordinateStandardizer
from mri_edain_v2.modules.rq_spline import (
    SplineParams,
    RQSplineParameterizer,
    rq_spline_parameterize,
    RQSplineApply,
    rq_spline_apply,
)
from mri_edain_v2.modules.hypernetwork import Hypernetwork
from mri_edain_v2.modules.nyul_init import (
    PopulationNyulInitializer,
    fit_population_nyul_theta_0,
    compute_non_affineness,
)
from mri_edain_v2.modules.edain_layer import MRIEDAINLayer

__all__ = [
    "ForegroundExtractor",
    "PERCENTILES",
    "PercentileSummary",
    "percentile_summary",
    "CoordinateStandardizer",
    "SplineParams",
    "RQSplineParameterizer",
    "rq_spline_parameterize",
    "RQSplineApply",
    "rq_spline_apply",
    "Hypernetwork",
    "PopulationNyulInitializer",
    "fit_population_nyul_theta_0",
    "compute_non_affineness",
    "MRIEDAINLayer",
]
