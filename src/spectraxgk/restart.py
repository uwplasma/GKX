"""Restart-state IO helpers.

SPECTRAX-GK reuses GX's flat complex64 restart layout so that:
- runtime `init_file` can consume restart files directly
- users can roundtrip state between GX and SPECTRAX for audits

The file format is a raw `complex64` buffer with no header. Consumers must
already know the target shape from (nspecies, Nl, Nm, Ny, Nx, Nz).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from jax.typing import ArrayLike


def write_gx_restart_state(path: str | Path, state: ArrayLike) -> Path:
    """Write a restart state in GX-compatible flat complex64 layout."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(state, dtype=np.complex64).tofile(out)
    return out

