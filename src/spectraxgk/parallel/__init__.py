"""Parallel execution, decomposition, and sharding helpers."""

from __future__ import annotations

from typing import Any

from spectraxgk.parallel.batch import *  # noqa: F403
from spectraxgk.parallel.batch import __all__ as _batch_all
from spectraxgk.parallel.decomposition import *  # noqa: F403
from spectraxgk.parallel.decomposition import __all__ as _decomposition_all
from spectraxgk.parallel.identity import *  # noqa: F403
from spectraxgk.parallel.identity import __all__ as _identity_all
from spectraxgk.parallel.independent import *  # noqa: F403
from spectraxgk.parallel.independent import __all__ as _independent_all
from spectraxgk.parallel.state import *  # noqa: F403
from spectraxgk.parallel.state import __all__ as _state_all
from spectraxgk.parallel.velocity import *  # noqa: F403
from spectraxgk.parallel.velocity import __all__ as _velocity_all

_INTEGRATOR_EXPORTS = {
    "integrate_linear_sharded",
    "integrate_nonlinear_sharded",
}


def __getattr__(name: str) -> Any:
    """Load sharded integrators only when requested to avoid import cycles."""

    if name in _INTEGRATOR_EXPORTS:
        from spectraxgk.parallel import integrators

        value = getattr(integrators, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'spectraxgk.parallel' has no attribute {name!r}")


__all__ = list(
    dict.fromkeys(
        [
            *_identity_all,
            *_batch_all,
            *_independent_all,
            *_decomposition_all,
            *_state_all,
            *_velocity_all,
            *_INTEGRATOR_EXPORTS,
        ]
    )
)
