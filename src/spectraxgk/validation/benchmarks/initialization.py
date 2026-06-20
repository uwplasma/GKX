"""Benchmark initial-condition builders.

These helpers are pure setup utilities shared by linear benchmark families.  They
are separated from runner orchestration so initialization conventions can be
unit-tested and audited without importing the full benchmark facade.
"""

from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp
import numpy as np

from spectraxgk.config import InitializationConfig, KineticElectronBaseCase
from spectraxgk.geometry import FluxTubeGeometryLike
from spectraxgk.core.grid import SpectralGrid


__all__ = [
    "_build_gaussian_profile",
    "_build_initial_condition",
    "_kinetic_reference_init_cfg",
]


def _build_gaussian_profile(
    z: np.ndarray,
    *,
    kx: float,
    ky: float,
    s_hat: float,
    init_cfg: InitializationConfig,
) -> np.ndarray:
    if ky == 0.0:
        return np.zeros_like(z)
    theta0 = kx / (s_hat * ky)
    envelope = (
        init_cfg.gaussian_envelope_constant
        + init_cfg.gaussian_envelope_sine * np.sin(z - theta0)
    )
    width = init_cfg.gaussian_width
    if width <= 0.0:
        raise ValueError("gaussian_width must be > 0")
    return envelope * np.exp(-(((z - theta0) / width) ** 2))


def _build_initial_condition(
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    *,
    ky_index: int | Sequence[int] | np.ndarray,
    kx_index: int,
    Nl: int,
    Nm: int,
    init_cfg: InitializationConfig,
) -> jnp.ndarray:
    init_field = init_cfg.init_field.lower()
    field_map = {
        "density": (0, 0),
        "upar": (0, 1),
        "tpar": (0, 2),
        "tperp": (1, 0),
        "qpar": (0, 3),
        "qperp": (1, 1),
    }
    # Moment-normalized initializer amplitudes for init_field="all".
    all_scales = {
        "density": 1.0,
        "upar": 1.0,
        "tpar": 1.0 / np.sqrt(2.0),
        "tperp": 1.0,
        "qpar": 1.0 / np.sqrt(6.0),
        "qperp": 1.0,
    }
    if init_field != "all" and init_field not in field_map:
        raise ValueError(
            "init_field must be one of {'density','upar','tpar','tperp','qpar','qperp','all'}"
        )

    G0 = np.zeros((Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64)
    amp = float(init_cfg.init_amp)
    ky_idx = np.atleast_1d(np.asarray(ky_index, dtype=int))
    for ky_i in ky_idx:
        if init_cfg.gaussian_init:
            profile = _build_gaussian_profile(
                np.asarray(grid.z),
                kx=float(grid.kx[kx_index]),
                ky=float(grid.ky[ky_i]),
                s_hat=geom.s_hat,
                init_cfg=init_cfg,
            )
            init_vals = amp * profile * (1.0 + 1.0j)
        else:
            init_vals = amp * (1.0 + 1.0j) * np.ones_like(grid.z)
        if grid.ky[ky_i] != 0.0:
            if init_field == "all":
                for field_name, (l_idx, m_idx) in field_map.items():
                    if l_idx < Nl and m_idx < Nm:
                        scale = all_scales.get(field_name, 1.0)
                        G0[l_idx, m_idx, ky_i, kx_index, :] = init_vals * scale
            else:
                l_idx, m_idx = field_map[init_field]
                if l_idx >= Nl or m_idx >= Nm:
                    raise ValueError("init_field moment exceeds (Nl, Nm) resolution")
                G0[l_idx, m_idx, ky_i, kx_index, :] = init_vals
    return jnp.asarray(G0)


def _kinetic_reference_init_cfg(
    init_cfg: InitializationConfig,
    *,
    reference_aligned: bool | None = None,
) -> InitializationConfig:
    """Use the reference-aligned kinetic benchmark seed when requested.

    Reference-aligned kinetic runs seed a constant electron-density moment.
    Explicit user overrides are preserved by replacing only the exact current
    kinetic default initialization.
    """

    if not bool(True if reference_aligned is None else reference_aligned):
        return init_cfg
    kinetic_default_init = KineticElectronBaseCase().init
    if init_cfg != kinetic_default_init:
        return init_cfg
    return InitializationConfig(
        init_field="density",
        init_amp=1.0e-3,
        init_single=True,
        random_seed=kinetic_default_init.random_seed,
        gaussian_init=False,
        gaussian_width=kinetic_default_init.gaussian_width,
        gaussian_envelope_constant=kinetic_default_init.gaussian_envelope_constant,
        gaussian_envelope_sine=kinetic_default_init.gaussian_envelope_sine,
        kpar_init=kinetic_default_init.kpar_init,
        init_file=kinetic_default_init.init_file,
        init_file_scale=kinetic_default_init.init_file_scale,
        init_file_mode=kinetic_default_init.init_file_mode,
        init_electrons_only=kinetic_default_init.init_electrons_only,
    )
