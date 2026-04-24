"""Runtime result containers and small assembly helpers."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from spectraxgk.analysis import ModeSelection
from spectraxgk.diagnostics import SimulationDiagnostics
from spectraxgk.terms.config import FieldState


@dataclass(frozen=True)
class RuntimeLinearResult:
    """Result container for runtime linear runs."""

    ky: float
    gamma: float
    omega: float
    selection: ModeSelection
    t: np.ndarray | None = None
    signal: np.ndarray | None = None
    state: np.ndarray | None = None
    z: np.ndarray | None = None
    eigenfunction: np.ndarray | None = None
    fit_window_tmin: float | None = None
    fit_window_tmax: float | None = None
    fit_signal_used: str | None = None


@dataclass(frozen=True)
class RuntimeLinearScanResult:
    """Result container for runtime linear ky scans."""

    ky: np.ndarray
    gamma: np.ndarray
    omega: np.ndarray


@dataclass(frozen=True)
class RuntimeNonlinearResult:
    """Result container for runtime nonlinear runs."""

    t: np.ndarray
    diagnostics: SimulationDiagnostics | None
    phi2: np.ndarray | None = None
    fields: FieldState | None = None
    state: np.ndarray | None = None
    ky_selected: float | None = None
    kx_selected: float | None = None


def nonlinear_field_phi2(fields: FieldState) -> np.ndarray:
    """Return the mean electrostatic energy density from final fields."""

    return np.asarray(jnp.mean(jnp.abs(fields.phi) ** 2))


def build_runtime_nonlinear_result(
    *,
    t: np.ndarray,
    diagnostics: SimulationDiagnostics | None,
    fields: FieldState | None,
    state: np.ndarray | None,
    ky_selected: float | None,
    kx_selected: float | None,
    summarize_fields: bool,
) -> RuntimeNonlinearResult:
    """Build a runtime nonlinear result with optional final-field summary."""

    phi2 = None
    t_out = np.asarray(t)
    diag_out = diagnostics
    if summarize_fields:
        if fields is None:
            raise RuntimeError("final fields are required when summarize_fields=True")
        phi2 = nonlinear_field_phi2(fields)
        t_out = np.asarray([])
        diag_out = None
    return RuntimeNonlinearResult(
        t=t_out,
        diagnostics=diag_out,
        phi2=phi2,
        fields=fields,
        state=state,
        ky_selected=ky_selected,
        kx_selected=kx_selected,
    )
