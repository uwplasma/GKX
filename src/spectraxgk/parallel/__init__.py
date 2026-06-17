"""Parallel execution, decomposition, and sharding helpers."""

from __future__ import annotations

from spectraxgk.parallel.core import *  # noqa: F403
from spectraxgk.parallel.core import __all__ as _core_all
from spectraxgk.parallel.decomposition import *  # noqa: F403
from spectraxgk.parallel.decomposition import __all__ as _decomposition_all
from spectraxgk.parallel.integrators import *  # noqa: F403
from spectraxgk.parallel.integrators import __all__ as _integrators_all
from spectraxgk.parallel.state import *  # noqa: F403
from spectraxgk.parallel.state import __all__ as _state_all
from spectraxgk.parallel.velocity import *  # noqa: F403
from spectraxgk.parallel.velocity import __all__ as _velocity_all

__all__ = list(
    dict.fromkeys(
        [
            *_core_all,
            *_decomposition_all,
            *_integrators_all,
            *_state_all,
            *_velocity_all,
        ]
    )
)
