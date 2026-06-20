"""Krylov seed-selection policies for Cyclone ky scans."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np


def seed_from_explicit_trace(
    G0: Any,
    grid: Any,
    cache: Any,
    params: Any,
    geom: Any,
    terms: Any,
    cfg_use: Any,
    hooks: Any,
    *,
    show_progress: bool,
) -> tuple[bool, bool, float, float]:
    """Estimate the starting branch from a short explicit trace."""

    gamma_seed = 0.0
    omega_seed = 0.0
    try:
        t_seed = min(150.0, float(cfg_use.power_dt) * 15000.0)
        explicit_time_cfg = hooks.explicit_time_config(
            dt=float(cfg_use.power_dt),
            t_max=t_seed,
            sample_stride=1,
            fixed_dt=True,
        )
        t_short, phi_seed, _g_t, _o_t = hooks.integrate_linear_explicit(
            jnp.array(G0),
            grid,
            cache,
            params,
            geom,
            explicit_time_cfg,
            terms=terms,
            mode_method="z_index",
            show_progress=show_progress,
        )
        sel = hooks.mode_selection(
            ky_index=0,
            kx_index=0,
            z_index=hooks.midplane_index(grid),
        )
        gamma_seed, omega_seed, _g, _o, _t_mid = (
            hooks.instantaneous_growth_rate_from_phi(
                phi_seed,
                t_short,
                sel,
                navg_fraction=0.5,
                mode_method="z_index",
            )
        )
        omega_ok = np.isfinite(omega_seed) and abs(omega_seed) > 1.0e-8
        seed_ok = omega_ok and np.isfinite(gamma_seed) and gamma_seed > 0.0
    except Exception:
        seed_ok = False
        omega_ok = False
    return seed_ok, omega_ok, float(gamma_seed), float(omega_seed)


def reduced_seed_from_explicit_trace(
    grid: Any,
    geom: Any,
    params: Any,
    terms: Any,
    cfg_use: Any,
    hooks: Any,
    *,
    init_cfg: Any,
    n_laguerre: int,
    n_hermite: int,
    show_progress: bool,
) -> tuple[bool, bool, float, float]:
    """Build a smaller velocity-space seed trace when the full seed fails."""

    n_laguerre_seed = min(n_laguerre, 16)
    n_hermite_seed = min(n_hermite, 12)
    try:
        cache_seed = hooks.build_linear_cache(
            grid,
            geom,
            params,
            n_laguerre_seed,
            n_hermite_seed,
        )
        G0_seed = hooks.build_initial_condition(
            grid,
            geom,
            ky_index=0,
            kx_index=0,
            Nl=n_laguerre_seed,
            Nm=n_hermite_seed,
            init_cfg=init_cfg,
        )
    except Exception:
        return False, False, 0.0, 0.0
    return seed_from_explicit_trace(
        G0_seed,
        grid,
        cache_seed,
        params,
        geom,
        terms,
        cfg_use,
        hooks,
        show_progress=show_progress,
    )


def seed_shift(
    prev_eig: complex | None,
    *,
    omega_ok: bool,
    seed_ok: bool,
    gamma_seed: float,
    omega_seed: float,
) -> complex | None:
    """Choose the Krylov shift implied by continuation or a trace seed."""

    if prev_eig is not None and np.isfinite(prev_eig):
        return prev_eig
    if omega_ok:
        return complex(float(gamma_seed) if seed_ok else 0.0, float(-omega_seed))
    return None


def use_explicit_seed(
    gamma: float,
    omega: float,
    *,
    seed_ok: bool,
    gamma_seed: float,
    omega_seed: float,
) -> bool:
    """Return whether the explicit trace should override the eigen-solver result."""

    if not seed_ok:
        return False
    seed_strong = (gamma_seed > 0.0) and (abs(omega_seed) > 1.0e-6)
    if not seed_strong:
        return False
    omega_tol = 0.15 * max(abs(omega_seed), 1.0e-6)
    gamma_tol = 0.15 * max(abs(gamma_seed), 1.0e-6)
    return (
        not np.isfinite(gamma)
        or not np.isfinite(omega)
        or (gamma_seed > 0.0 and gamma < 0.0)
        or abs(omega - omega_seed) > omega_tol
        or abs(gamma - gamma_seed) > gamma_tol
    )


__all__ = [
    "reduced_seed_from_explicit_trace",
    "seed_from_explicit_trace",
    "seed_shift",
    "use_explicit_seed",
]
