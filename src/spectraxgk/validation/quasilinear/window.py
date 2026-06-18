"""Nonlinear late-window convergence statistics for quasilinear calibration.

The helpers in this module are intentionally data-only: they operate on time
traces or compact diagnostics artifacts and do not launch nonlinear solves.
They provide the metadata required before a nonlinear holdout can support an
absolute-flux quasilinear promotion.
"""

from __future__ import annotations

from spectraxgk.validation.quasilinear.window_config import (
    NonlinearWindowConvergenceConfig,
    NonlinearWindowEnsembleConfig,
    NonlinearWindowEnsembleManifestConfig,
)
from spectraxgk.validation.quasilinear.window_ensemble import (
    nonlinear_window_ensemble_artifact_manifest,
    nonlinear_window_ensemble_report,
)
from spectraxgk.validation.quasilinear.window_io import (
    nonlinear_window_convergence_from_csv,
    nonlinear_window_convergence_from_summary,
)
from spectraxgk.validation.quasilinear.window_promotion import (
    nonlinear_window_stats_promotion_ready,
)
from spectraxgk.validation.quasilinear.window_statistics import (
    nonlinear_window_convergence_report,
)

__all__ = [
    "NonlinearWindowConvergenceConfig",
    "NonlinearWindowEnsembleConfig",
    "NonlinearWindowEnsembleManifestConfig",
    "nonlinear_window_ensemble_artifact_manifest",
    "nonlinear_window_ensemble_report",
    "nonlinear_window_convergence_from_csv",
    "nonlinear_window_convergence_from_summary",
    "nonlinear_window_convergence_report",
    "nonlinear_window_stats_promotion_ready",
]
