"""Boozer curvature and grad-B drift assembly for the vmex-state bridge.

The drift coefficients are the only flux-tube quantities that depend on the
equilibrium pressure gradient, so they are owned here rather than in
:mod:`gkx.geometry.vmec_boozer_core`.  The conventions match the wout runtime
path in :mod:`gkx.geometry.vmec_field_line_sampling` term for term: the
curvature drift carries ``mu0 dp/ds`` inside the normal curvature, and the
grad-B drift is the curvature drift with that same pressure contribution
removed.  Both paths therefore describe one equilibrium at finite beta, which
is what the A-vs-B parity gate checks.

Still missing relative to the wout path: the Hegna-Nakajima ``beta_b`` Boozer
mode correction to ``kappa_n``.  It was measured rather than assumed small --
zeroing it inside the wout path on the 1.9%-beta QA fixtures moves the drifts
by 4.5e-4 to 8.1e-4 of the drift amplitude, against a bridge-vs-runtime parity
floor of 3e-3 set by ``bmag``/``gds2`` themselves.  Adding it would therefore
change gradients by an amount the parity gate cannot currently resolve, so it
waits until the metric floor comes down.  The companion ``D_HNGC`` local-shear
term is identically zero unless the local-equilibrium overrides
``include_shear_variation``/``include_pressure_variation`` are enabled, which
the bridge does not expose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from gkx.geometry.numerics import _interp_radial, _radial_derivative_profile

_MU_0 = 4.0e-7 * np.pi


@dataclass(frozen=True)
class RawDriftProfiles:
    """Drift coefficients on the Boozer theta grid, before the equal-arc remap."""

    cvdrift: jnp.ndarray
    cvdrift0: jnp.ndarray
    gbdrift: jnp.ndarray
    gbdrift0: jnp.ndarray


def boozer_pressure_gradient(
    wout: Any,
    *,
    s_value: float,
    dtype: Any,
) -> jnp.ndarray:
    """Return ``dp/ds`` at ``s_value`` from the VMEC half-mesh pressure profile.

    The half mesh is rebuilt from the wout ``pres`` length rather than from the
    Boozer output grid, because a stencil-restricted Boozer transform carries
    only a few surfaces while the pressure profile is always full-radius.  This
    mirrors the wout runtime path, which splines ``pres[1:]`` over the same
    half mesh and differentiates the spline.

    A wout without a pressure profile is a vacuum field; the returned zero then
    reproduces the zero-beta drifts exactly.

    The profile stays a JAX array rather than being pulled to host NumPy.  It is
    a constant today -- VMEC pressure is prescribed input, not solved state, so
    no boundary-coefficient gradient flows through it -- but keeping it on the
    tape is what would let a pressure-parameter gradient work later without
    revisiting this.
    """

    raw = getattr(wout, "pres", None)
    if raw is None:
        return jnp.zeros((), dtype=dtype)
    profile = jnp.asarray(raw, dtype=dtype)
    if profile.ndim != 1 or int(profile.shape[0]) < 3:
        raise ValueError(
            "VMEC wout pressure profile must be a radial array of at least three "
            f"half-mesh entries; got shape {tuple(profile.shape)}"
        )
    ns_full = int(profile.shape[0])
    spacing = 1.0 / float(ns_full - 1)
    s_half = jnp.asarray((np.arange(1, ns_full) - 0.5) * spacing, dtype=dtype)
    return _interp_radial(
        _radial_derivative_profile(profile[1:], spacing), s_half, s_value
    )


def _normal_curvature_pressure_term(
    *,
    d_pressure_ds: jnp.ndarray,
    metric_bmag_sq: jnp.ndarray,
    etf_safe: jnp.ndarray,
) -> jnp.ndarray:
    """Return the finite-beta part of the normal curvature ``kappa_n``.

    MHD force balance puts the pressure gradient into the curvature but not
    into ``grad B``: ``kappa = grad_perp B / B + mu0 grad_perp p / B^2``.
    """

    return _MU_0 * d_pressure_ds / (metric_bmag_sq * etf_safe)


def raw_drift_profiles(
    request: Any,
    scales: Any,
    profiles: Any,
    equal_arc: Any,
    state: Any,
) -> RawDriftProfiles:
    """Compute curvature and grad-B drift coefficients before the equal-arc remap.

    ``profiles.d_pressure_ds`` enters twice and with opposite effect: it is the
    finite-beta part of ``kappa_n`` that makes ``cvdrift`` the curvature drift,
    and it is subtracted back out to leave ``gbdrift`` the pure grad-B drift.
    At zero beta the two coincide, which is the aliasing this replaced.
    """

    dtype = request.base_Rcos.dtype
    s_arr = jnp.asarray(request.torflux, dtype=dtype)
    L = jnp.asarray(float(scales.length), dtype=dtype)
    Bref = jnp.asarray(float(scales.magnetic_field), dtype=dtype)
    boozer_current_sum = profiles.boozer_g + profiles.iota_safe * profiles.boozer_i
    d_sqrt_g_booz_d_theta = (
        -2.0
        * boozer_current_sum
        * state.spectral.d_mod_b_d_theta
        / (equal_arc.mod_b_safe**3)
    )
    d_sqrt_g_booz_d_phi = (
        -2.0
        * boozer_current_sum
        * state.spectral.d_mod_b_d_phi
        / (equal_arc.mod_b_safe**3)
    )
    curvature_numerator = (
        profiles.boozer_g * d_sqrt_g_booz_d_theta
        - profiles.boozer_i * d_sqrt_g_booz_d_phi
    )
    curvature_denom = 2.0 * equal_arc.sqrt_g_booz * boozer_current_sum
    curvature_denom_safe = jnp.where(
        jnp.abs(curvature_denom) < state.eps,
        jnp.sign(curvature_denom + state.eps) * state.eps,
        curvature_denom,
    )
    kappa_g = curvature_numerator / curvature_denom_safe
    local_shear_l0 = -(
        state.local_shear_l1 + profiles.d_iota_ds / state.etf_safe * state.shear_phase
    )
    kappa_n = (
        state.spectral.d_mod_b_d_s / (equal_arc.mod_b_safe * state.etf_safe)
        + _normal_curvature_pressure_term(
            d_pressure_ds=profiles.d_pressure_ds,
            metric_bmag_sq=state.metric_bmag_sq,
            etf_safe=state.etf_safe,
        )
        + local_shear_l0 * kappa_g
    )
    b_cross_kappa_dot_grad_alpha = (
        kappa_n + kappa_g * state.local_shear_l1
    ) * state.metric_bmag_sq
    b_cross_kappa_dot_grad_psi = kappa_g * state.metric_bmag_sq
    toroidal_flux_sign = jnp.sign(state.etf)
    sqrt_s = jnp.sqrt(s_arr)
    cvdrift0 = (
        -b_cross_kappa_dot_grad_psi
        * 2.0
        * profiles.s_hat
        / jnp.maximum(state.metric_bmag_sq * sqrt_s, state.eps)
        * toroidal_flux_sign
    )
    cvdrift = (
        -2.0
        * Bref
        * L
        * L
        * sqrt_s
        * b_cross_kappa_dot_grad_alpha
        / state.metric_bmag_sq
        * toroidal_flux_sign
    )
    gbdrift = cvdrift + (
        2.0
        * Bref
        * L
        * L
        * sqrt_s
        * _MU_0
        * profiles.d_pressure_ds
        * toroidal_flux_sign
        / (state.etf_safe * state.metric_bmag_sq)
    )
    # gbdrift0 stays aliased to cvdrift0: both are the grad-psi component of the
    # same curvature drift, and the pressure gradient is parallel to grad psi.
    return RawDriftProfiles(
        cvdrift=cvdrift,
        cvdrift0=cvdrift0,
        gbdrift=gbdrift,
        gbdrift0=cvdrift0,
    )


__all__ = [
    "RawDriftProfiles",
    "boozer_pressure_gradient",
    "raw_drift_profiles",
]
