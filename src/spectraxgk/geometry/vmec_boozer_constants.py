"""Boozer-transform constants and cache prewarm helpers for vmex bridges."""

from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Any

import numpy as np

_VMEC_BOOZER_PARITY_MIN_MODE_COUNT = 21


def _grid_mode_limits(ntheta1: int, nzeta: int) -> tuple[int, int]:
    """Return the ``(m_max, n_max)`` grid-representable Fourier cutoffs.

    ``vmex.core.boozer_tables.boozer_input_tables`` projects every surface on
    the grid-representable modes ``m = 0..ntheta1//2 - 1`` and
    ``n = -(nzeta//2 - 1)..nzeta//2 - 1`` (single mode set, wout ordering).
    The cached constants below must reproduce exactly that mode table.
    """

    return max(int(ntheta1) // 2 - 1, 0), max(int(nzeta) // 2 - 1, 0)


@lru_cache(maxsize=32)
def _cached_booz_xform_constants(
    *,
    nfp: int,
    ntheta1: int,
    nzeta: int,
    mboz: int,
    nboz: int,
    asym: bool,
) -> tuple[Any, Any]:
    """Prepare Boozer constants outside traced vmex residual callbacks.

    The mode table matches the single grid-representable mode set emitted by
    ``vmex.core.boozer_tables.boozer_input_tables`` (``xn`` carries ``nfp``;
    ``xm_nyq``/``xn_nyq`` equal ``xm``/``xn`` because the traceable tables
    project |B| and R/Z/lambda on one mode set).
    """

    fourier_mod = importlib.import_module("vmex.core.fourier")
    bx = importlib.import_module("booz_xform_jax.jax_api")
    m_max, n_max = _grid_mode_limits(int(ntheta1), int(nzeta))
    modes = fourier_mod.mode_table(m_max + 1, n_max)
    xm = np.asarray(modes.m, dtype=np.int32)
    xn = np.asarray(modes.n * int(nfp), dtype=np.int32)
    return bx.prepare_booz_xform_constants(
        nfp=int(nfp),
        mboz=int(mboz),
        nboz=int(nboz),
        asym=bool(asym),
        xm=xm,
        xn=xn,
        xm_nyq=xm,
        xn_nyq=xn,
    )


def prewarm_vmec_boozer_equal_arc_cache(
    runtime: Any,
    wout: Any,
    *,
    mboz: int = _VMEC_BOOZER_PARITY_MIN_MODE_COUNT,
    nboz: int = _VMEC_BOOZER_PARITY_MIN_MODE_COUNT,
    asym: bool | None = None,
) -> None:  # pragma: no cover - exercised by optional vmex optimizer smoke tests.
    """Precompute Boozer constants before vmex jits residual callbacks."""

    resolution = runtime.resolution
    nfp_raw = getattr(wout, "nfp", None)
    if nfp_raw is None:
        nfp_raw = getattr(resolution, "nfp", 1)
    nfp_int = 1 if nfp_raw is None else int(nfp_raw)
    _cached_booz_xform_constants(
        nfp=nfp_int,
        ntheta1=int(resolution.ntheta1),
        nzeta=int(resolution.nzeta),
        mboz=int(mboz),
        nboz=int(nboz),
        asym=bool(getattr(resolution, "lasym", False) if asym is None else asym),
    )


__all__ = [
    "_VMEC_BOOZER_PARITY_MIN_MODE_COUNT",
    "_cached_booz_xform_constants",
    "prewarm_vmec_boozer_equal_arc_cache",
]
