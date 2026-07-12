"""Reference data and policies for documented code-comparison workflows.

Simulation execution belongs to :mod:`spectraxgk.runtime`.  This compact
facade keeps reviewed reference tables, normalization contracts, and branch
selection policies together without maintaining a second solver stack.
"""

from spectraxgk.benchmarking.shared import *  # noqa: F403
from spectraxgk.benchmarking.shared import __all__ as _SHARED_EXPORTS
from spectraxgk.config import CycloneBaseCase, KBMBaseCase
from spectraxgk.diagnostics.modes import ModeSelection
from spectraxgk.solvers.linear.krylov import KrylovConfig
from spectraxgk.solvers.time.explicit import ExplicitTimeConfig

__all__ = [
    *_SHARED_EXPORTS,
    "CycloneBaseCase",
    "ExplicitTimeConfig",
    "KBMBaseCase",
    "KrylovConfig",
    "ModeSelection",
]
