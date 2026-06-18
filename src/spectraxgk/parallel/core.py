"""Stable facade for production parallelization helpers."""

from __future__ import annotations

from spectraxgk.parallel.batch import *  # noqa: F403
from spectraxgk.parallel.batch import __all__ as _batch_all
from spectraxgk.parallel.identity import *  # noqa: F403
from spectraxgk.parallel.identity import __all__ as _identity_all
from spectraxgk.parallel.independent import *  # noqa: F403
from spectraxgk.parallel.independent import __all__ as _independent_all

__all__ = list(dict.fromkeys([*_identity_all, *_batch_all, *_independent_all]))
