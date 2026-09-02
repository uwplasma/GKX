"""Time-integration policies and config-driven solver runners."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - static visibility for the lazy exports
    # These are provided at run time by ``__getattr__`` below, which defers the
    # import to avoid a nonlinear cycle. Declaring them here keeps ``__all__``
    # statically checkable now that this is a module rather than a package
    # ``__init__``, where the checker resolved them through the submodule.
    from gkx.solvers_time_runners import (
        integrate_linear_from_config,
        integrate_nonlinear_from_config,
    )

from gkx.solvers_time_explicit import (
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
        import gkx.solvers_time_runners as runners

        value = getattr(runners, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'gkx.solvers_time' has no attribute {name!r}")


__all__ = [
    "ExplicitTimeConfig",
    "integrate_linear_explicit",
    "integrate_linear_explicit_diagnostics",
    "integrate_linear_from_config",
    "integrate_nonlinear_from_config",
]
