"""Physics gate: parallel-domain end damping on a linked flux tube.

The fixed rate 50 preserves legacy strength .1 at the pinned ``dt=0.002``.
Using 0.1 as a rate instead weakens damping 500-fold and
allows an unphysical boundary mode to dominate (uwplasma/GKX#192).
References below are measured compatibility regressions, not independent
literature validation. Float64 is deliberate: this small float32 case did not
expose the unstable boundary mode. Issue #194 owns the cross-route rate migration;
the analytic scalar RK stage-map and tangent tests live in test_linear.py.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gkx.config import GeometryConfig, GridConfig
from gkx.core_grid import build_spectral_grid, select_ky_grid
from gkx.geometry import SAlphaGeometry
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.params import LinearTerms, linear_params_for_geometry
from gkx.solvers_linear_integrator_diagnostics import integrate_linear_diagnostics

# One Cyclone-like linked flux tube at the tokamak parity decks' step size. The
# ky is the top of the s-alpha deck's scan, which is where the campaign traced
# the runaway to; nperiod=2 is what gives the tube the two damped end caps at
# all, and every failing case in the parity matrix has it.
_KY_TARGET = 0.55
_NL, _NM, _NZ = 8, 24, 96
_DT = 0.002
_STEPS = 20000
_SAMPLE_STRIDE = 50
_SEED_AMPLITUDE = 1.0e-6

#: Measured with the per-step contract in place; the spurious mode gives +11.3.
_EXPECTED_GAMMA = -0.0405


def _linked_salpha_flux_tube():
    """Return the single-ky linked s-alpha tube and its geometry."""

    grid_full = build_spectral_grid(
        GridConfig(
            Nx=1,
            Ny=16,
            Nz=_NZ,
            Lx=62.8,
            Ly=62.8,
            boundary="linked",
            y0=10.0,
            ntheta=32,
            nperiod=2,
        )
    )
    ky_index = int(np.argmin(np.abs(np.asarray(grid_full.ky) - _KY_TARGET)))
    grid = select_ky_grid(grid_full, ky_index)
    geom = SAlphaGeometry.from_config(GeometryConfig(s_hat=0.8))
    return grid, geom


def test_end_damping_bounds_the_domain_end_mode_at_a_deck_step_size() -> None:
    """Bound the field/end caps and recover the pinned late-time decay rate."""

    with jax.enable_x64():
        grid, geom = _linked_salpha_flux_tube()
        params = linear_params_for_geometry(
            geom,
            tprim=2.49,
            fprim=0.8,
            damp_ends_amp=50.0,  # Legacy .1/.002; fixed rate during dt refinement.
            damp_ends_widthfrac=0.125,
            nu_hermite=1.0,
            nu_laguerre=2.0,
            hypercollisions_const=0.0,
            hypercollisions_kz=1.0,
        )
        cache = build_linear_cache(grid, geom, params, Nl=_NL, Nm=_NM)
        terms = LinearTerms(
            streaming=1.0,
            mirror=1.0,
            curvature=1.0,
            gradb=1.0,
            diamagnetic=1.0,
            collisions=0.0,
            hypercollisions=1.0,
            hyperdiffusion=0.0,
            end_damping=1.0,
            apar=0.0,
            bpar=0.0,
        )

        # Seed the middle of the tube only. The end caps are populated by
        # parallel streaming, so what the trace measures is whether the damping
        # removes that content faster than the boundary mode amplifies it.
        z = np.arange(_NZ)
        profile = np.exp(-0.5 * ((z - _NZ / 2.0) / (0.1 * _NZ)) ** 2)
        seed = np.zeros((_NL, _NM, 1, 1, _NZ), dtype=complex)
        seed[0, 0, 0, 0, :] = _SEED_AMPLITUDE * profile
        G0 = jnp.asarray(seed, dtype=jnp.complex128)

        _G, phi_t, _density = integrate_linear_diagnostics(
            G0,
            grid,
            geom,
            params,
            _DT,
            _STEPS,
            method="imex2",
            terms=terms,
            sample_stride=_SAMPLE_STRIDE,
            species_index=0,
            cache=cache,
        )

    phi = np.asarray(phi_t)[:, 0, 0, :]
    assert np.all(np.isfinite(phi)), "field history went non-finite"

    # Amplitude as a max over z rather than an L2 norm: under the rate reading
    # the field reaches ~1e163, where squaring it overflows float64 and the
    # runaway would be hidden behind an inf instead of measured.
    amplitude = np.max(np.abs(phi), axis=1)
    assert np.max(amplitude) < 1.0e-4, (
        f"peak |phi| {np.max(amplitude):.3e} ran away from the "
        f"{_SEED_AMPLITUDE:.0e} seed"
    )

    # The end caps must hold a negligible share of the mode. Under the rate
    # reading they hold order 10% of it, and then they are the mode.
    damp_profile = np.asarray(cache.damp_profile, dtype=float).reshape(-1)
    if damp_profile.size != _NZ:
        damp_profile = np.asarray(cache.linked_damp_profile, dtype=float).reshape(-1)
        damp_profile = damp_profile[:_NZ]
    end_caps = damp_profile > 0.5 * damp_profile.max()
    assert end_caps.any()
    end_share = float(np.max(np.abs(phi[-1, end_caps]))) / amplitude[-1]
    assert end_share < 1.0e-2, f"end caps hold {end_share:.3e} of the mode"

    times = _DT * _SAMPLE_STRIDE * np.arange(1, amplitude.size + 1)
    late = slice(int(0.6 * amplitude.size), None)
    gamma = float(np.polyfit(times[late], np.log(amplitude[late]), 1)[0])
    assert gamma == pytest.approx(_EXPECTED_GAMMA, abs=1.0e-2)
