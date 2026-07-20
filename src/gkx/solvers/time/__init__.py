"""Time-integration policies and config-driven solver runners."""

from __future__ import annotations

from typing import Any

from gkx.solvers.time.diffrax_linear import integrate_linear_diffrax
from gkx.solvers.time.diffrax_nonlinear import integrate_nonlinear_diffrax
from gkx.solvers.time.diffrax_streaming import integrate_linear_diffrax_streaming
from gkx.solvers.time.explicit import (
    ExplicitTimeConfig,
    integrate_linear_explicit,
    integrate_linear_explicit_diagnostics,
)

_RUNNER_EXPORTS = {
    "integrate_linear_from_config",
    "integrate_nonlinear_from_config",
}


def __getattr__(name: str) -> Any:
    """Load config-driven runners only when requested to avoid nonlinear cycles."""

    if name in _RUNNER_EXPORTS:
        from gkx.solvers.time import runners

        value = getattr(runners, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'gkx.solvers.time' has no attribute {name!r}")


__all__ = [
    "ExplicitTimeConfig",
    "integrate_linear_diffrax",
    "integrate_linear_diffrax_streaming",
    "integrate_linear_explicit",
    "integrate_linear_explicit_diagnostics",
    "integrate_linear_from_config",
    "integrate_nonlinear_diffrax",
    "integrate_nonlinear_from_config",
]
