"""Configuration contracts for nonlinear transport-window gates."""

from __future__ import annotations

from dataclasses import dataclass
import math

@dataclass(frozen=True)
class NonlinearWindowConvergenceConfig:
    """Gate settings for a nonlinear post-transient transport window."""

    tmin: float | None = None
    tmax: float | None = None
    transient_fraction: float = 0.5
    min_samples: int = 24
    min_blocks: int = 4
    block_size: int | None = None
    bootstrap_samples: int = 256
    bootstrap_seed: int = 0
    max_running_mean_rel_drift: float = 0.15
    terminal_fraction: float = 0.25
    min_terminal_samples: int = 8
    max_terminal_mean_rel_delta: float = 0.10
    max_sem_rel: float = 0.25
    value_floor: float = 1.0e-12
    require_all_finite: bool = True


@dataclass(frozen=True)
class NonlinearWindowEnsembleConfig:
    """Gate settings for replicated nonlinear transport-window summaries."""

    min_reports: int = 2
    max_mean_rel_spread: float = 0.15
    max_combined_sem_rel: float = 0.25
    value_floor: float = 1.0e-12
    require_individual_passed: bool = True


@dataclass(frozen=True)
class NonlinearWindowEnsembleManifestConfig:
    """Artifact requirements before a replicated nonlinear ensemble can run."""

    min_replicates_per_case: int = 2
    required_variant_axes: tuple[str, ...] = ("seed", "timestep")
    require_observed_windows_ready: bool = True

def _validate_config(config: NonlinearWindowConvergenceConfig) -> None:
    if config.tmin is not None and not math.isfinite(float(config.tmin)):
        raise ValueError("tmin must be finite when supplied")
    if config.tmax is not None and not math.isfinite(float(config.tmax)):
        raise ValueError("tmax must be finite when supplied")
    if config.tmin is not None and config.tmax is not None:
        if float(config.tmin) >= float(config.tmax):
            raise ValueError("tmin must be less than tmax")
    if not 0.0 <= float(config.transient_fraction) < 1.0:
        raise ValueError("transient_fraction must be in [0, 1)")
    if int(config.min_samples) < 2:
        raise ValueError("min_samples must be at least 2")
    if int(config.min_blocks) < 2:
        raise ValueError("min_blocks must be at least 2")
    if config.block_size is not None and int(config.block_size) < 1:
        raise ValueError("block_size must be positive when supplied")
    if int(config.bootstrap_samples) < 0:
        raise ValueError("bootstrap_samples must be non-negative")
    if float(config.max_running_mean_rel_drift) < 0.0:
        raise ValueError("max_running_mean_rel_drift must be non-negative")
    if not 0.0 < float(config.terminal_fraction) <= 1.0:
        raise ValueError("terminal_fraction must be in (0, 1]")
    if int(config.min_terminal_samples) < 1:
        raise ValueError("min_terminal_samples must be positive")
    if float(config.max_terminal_mean_rel_delta) < 0.0:
        raise ValueError("max_terminal_mean_rel_delta must be non-negative")
    if float(config.max_sem_rel) < 0.0:
        raise ValueError("max_sem_rel must be non-negative")
    if float(config.value_floor) <= 0.0:
        raise ValueError("value_floor must be positive")


def _validate_ensemble_config(config: NonlinearWindowEnsembleConfig) -> None:
    if int(config.min_reports) < 2:
        raise ValueError("min_reports must be at least 2")
    if float(config.max_mean_rel_spread) < 0.0:
        raise ValueError("max_mean_rel_spread must be non-negative")
    if float(config.max_combined_sem_rel) < 0.0:
        raise ValueError("max_combined_sem_rel must be non-negative")
    if float(config.value_floor) <= 0.0:
        raise ValueError("value_floor must be positive")


def _validate_ensemble_manifest_config(
    config: NonlinearWindowEnsembleManifestConfig,
) -> None:
    if int(config.min_replicates_per_case) < 2:
        raise ValueError("min_replicates_per_case must be at least 2")
    axes = tuple(str(axis).strip() for axis in config.required_variant_axes)
    if not axes or any(not axis for axis in axes):
        raise ValueError("required_variant_axes must contain non-empty names")


__all__ = [
    "NonlinearWindowConvergenceConfig",
    "NonlinearWindowEnsembleConfig",
    "NonlinearWindowEnsembleManifestConfig",
]
