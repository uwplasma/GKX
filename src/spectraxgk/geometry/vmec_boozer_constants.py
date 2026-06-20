"""Boozer-transform constants and cache prewarm helpers for VMEC-JAX bridges."""

from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Any

import numpy as np

_VMEC_BOOZER_PARITY_MIN_MODE_COUNT = 21


@lru_cache(maxsize=32)
def _cached_booz_xform_constants(
    *,
    nfp: int,
    mpol: int,
    ntor: int,
    ntheta: int,
    nzeta: int,
    mboz: int,
    nboz: int,
    asym: bool,
) -> tuple[Any, Any]:
    """Prepare Boozer constants outside traced VMEC-JAX residual callbacks."""

    modes_mod = importlib.import_module("vmec_jax.modes")
    bx = importlib.import_module("booz_xform_jax.jax_api")
    main_modes = modes_mod.vmec_mode_table(int(mpol), int(ntor))
    nyq_modes = modes_mod.nyquist_mode_table_from_grid(
        mpol=int(mpol),
        ntor=int(ntor),
        ntheta=int(ntheta),
        nzeta=int(nzeta),
    )
    return bx.prepare_booz_xform_constants(
        nfp=int(nfp),
        mboz=int(mboz),
        nboz=int(nboz),
        asym=bool(asym),
        xm=np.asarray(main_modes.m, dtype=np.int32),
        xn=np.asarray(main_modes.n * int(nfp), dtype=np.int32),
        xm_nyq=np.asarray(nyq_modes.m, dtype=np.int32),
        xn_nyq=np.asarray(nyq_modes.n * int(nfp), dtype=np.int32),
    )


def prewarm_vmec_boozer_equal_arc_cache(
    static: Any,
    wout: Any,
    *,
    mboz: int = _VMEC_BOOZER_PARITY_MIN_MODE_COUNT,
    nboz: int = _VMEC_BOOZER_PARITY_MIN_MODE_COUNT,
    asym: bool | None = None,
) -> None:  # pragma: no cover - exercised by optional VMEC-JAX optimizer smoke tests.
    """Precompute Boozer constants before VMEC-JAX jits residual callbacks."""

    cfg = static.cfg
    nfp_raw = getattr(wout, "nfp", None)
    if nfp_raw is None:
        nfp_raw = getattr(cfg, "nfp", 1)
    nfp_int = 1 if nfp_raw is None else int(nfp_raw)
    _cached_booz_xform_constants(
        nfp=nfp_int,
        mpol=int(cfg.mpol),
        ntor=int(cfg.ntor),
        ntheta=int(cfg.ntheta),
        nzeta=int(cfg.nzeta),
        mboz=int(mboz),
        nboz=int(nboz),
        asym=bool(getattr(cfg, "lasym", False) if asym is None else asym),
    )


__all__ = [
    "_VMEC_BOOZER_PARITY_MIN_MODE_COUNT",
    "_cached_booz_xform_constants",
    "prewarm_vmec_boozer_equal_arc_cache",
]
