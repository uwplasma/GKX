"""Configuration and optional-backend path policy for VMEC transport objectives."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np

from spectraxgk.objectives.stellarator import StellaratorITGSampleSet


VMECJAXTransportObjectiveKind = Literal[
    "growth",
    "quasilinear_flux",
    "nonlinear_window_heat_flux",
]
VMECJAXTransportObjectiveTransform = Literal["raw", "scaled", "log1p"]


def _module_search_root(module_name: str) -> Path | None:
    """Return the import root for an already importable optional backend."""

    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    raw_file = getattr(module, "__file__", None)
    if raw_file is not None:
        return Path(str(raw_file)).resolve(strict=False).parent.parent
    raw_paths = getattr(module, "__path__", None)
    if raw_paths is None:
        return None
    for raw in raw_paths:
        path = Path(str(raw)).resolve(strict=False)
        if path.exists():
            return path
    return None


def _pin_current_optional_backend_paths() -> None:
    """Keep geometry discovery on the same optional backends VMEC-JAX imported.

    The differentiable-geometry bridge intentionally prefers explicit local
    checkouts over globally installed packages.  When examples run from a fresh
    temporary clone while another VMEC-JAX checkout exists in ``$HOME``, that
    preference can otherwise evict the VMEC-JAX module that owns the traced
    optimization state.  Pinning the currently importable backend paths makes
    the VMEC-JAX/SPECTRAX-GK objective reproducible without requiring users to
    hand-set environment variables.
    """

    if not (
        os.environ.get("SPECTRAX_VMEC_JAX_PATH") or os.environ.get("VMEC_JAX_PATH")
    ):
        root = _module_search_root("vmec_jax")
        if root is not None:
            os.environ.setdefault("SPECTRAX_VMEC_JAX_PATH", str(root))
    if not (
        os.environ.get("SPECTRAX_BOOZ_XFORM_JAX_PATH")
        or os.environ.get("BOOZ_XFORM_JAX_PATH")
    ):
        root = _module_search_root("booz_xform_jax")
        if root is not None:
            os.environ.setdefault("SPECTRAX_BOOZ_XFORM_JAX_PATH", str(root))


@dataclass(frozen=True)
class VMECJAXTransportObjectiveConfig:
    """Configuration for VMEC-JAX to SPECTRAX-GK objective evaluation."""

    kind: VMECJAXTransportObjectiveKind = "nonlinear_window_heat_flux"
    sample_set: StellaratorITGSampleSet = field(default_factory=StellaratorITGSampleSet)
    objective_weights: tuple[float, ...] | None = None
    ntheta: int = 24
    mboz: int = 21
    nboz: int = 21
    n_laguerre: int = 2
    n_hermite: int = 3
    nx: int = 1
    ny: int = 4
    nonlinear_csat: float = 0.85
    nonlinear_saturation_floor: float = 1.0e-10
    reference_length: float | None = None
    reference_b: float | None = None
    objective_transform: VMECJAXTransportObjectiveTransform = "raw"
    objective_scale: float = 1.0
    surface_chunk_size: int = 0
    validate_finite: bool = True

    @property
    def gradient_scope(self) -> str:
        """Return the differentiated part of this objective."""

        if self.kind == "growth":
            return "eigenvalue_growth_ad"
        return "eigenvalue_growth_ad_with_geometry_transport_weights"

    def __post_init__(self) -> None:
        if self.kind not in (
            "growth",
            "quasilinear_flux",
            "nonlinear_window_heat_flux",
        ):
            raise ValueError(f"unknown VMEC-JAX transport objective kind {self.kind!r}")
        if int(self.ntheta) < 4:
            raise ValueError("ntheta must be >= 4")
        if int(self.mboz) < 21 or int(self.nboz) < 21:
            raise ValueError(
                "mboz and nboz must be at least 21 for paper-facing QA optimization"
            )
        if int(self.n_laguerre) < 1 or int(self.n_hermite) < 1:
            raise ValueError("n_laguerre and n_hermite must be positive")
        if int(self.nx) < 1 or int(self.ny) < 3:
            raise ValueError("nx must be positive and ny must be at least 3")
        if float(self.nonlinear_csat) <= 0.0:
            raise ValueError("nonlinear_csat must be positive")
        if self.objective_transform not in ("raw", "scaled", "log1p"):
            raise ValueError(
                f"unknown VMEC-JAX transport objective transform {self.objective_transform!r}"
            )
        if float(self.objective_scale) <= 0.0:
            raise ValueError("objective_scale must be positive")
        if int(self.surface_chunk_size) < 0:
            raise ValueError("surface_chunk_size must be non-negative")
        if int(self.surface_chunk_size) > 0 and self.sample_set.reduction not in (
            "weighted_mean",
            "mean",
        ):
            raise ValueError(
                "surface_chunk_size currently supports only mean or weighted_mean reductions"
            )

    def objective_options(self) -> dict[str, Any]:
        """Return SPECTRAX-GK solver options for this objective."""

        options: dict[str, Any] = {
            "ntheta": int(self.ntheta),
            "mboz": int(self.mboz),
            "nboz": int(self.nboz),
            "n_laguerre": int(self.n_laguerre),
            "n_hermite": int(self.n_hermite),
            "nx": int(self.nx),
            "ny": int(self.ny),
            "reference_length": self.reference_length,
            "reference_b": self.reference_b,
            "validate_finite": bool(self.validate_finite),
        }
        return {key: value for key, value in options.items() if value is not None}


def _reference_wout_from_context(ctx: Any) -> Any:
    """Return the minimal WOUT metadata needed by the VMEC/Boozer bridge."""

    cfg = getattr(getattr(ctx, "static", None), "cfg", None)
    nfp = int(getattr(cfg, "nfp", 1))
    return SimpleNamespace(
        signgs=int(getattr(ctx, "signgs", 1)),
        nfp=nfp,
        Aminor_p=1.0,
        phi=np.asarray([0.0, -np.pi], dtype=float),
    )


__all__ = [
    "VMECJAXTransportObjectiveConfig",
    "VMECJAXTransportObjectiveKind",
    "VMECJAXTransportObjectiveTransform",
    "_module_search_root",
    "_pin_current_optional_backend_paths",
    "_reference_wout_from_context",
]
