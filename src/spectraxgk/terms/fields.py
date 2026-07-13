"""Field solves for electromagnetic gyrokinetics."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from spectraxgk.terms.config import FieldState


@dataclass(frozen=True)
class _FieldSolveCoefficients:
    Jl: jnp.ndarray
    JlB: jnp.ndarray
    bmag: jnp.ndarray
    kperp2: jnp.ndarray
    beta: jnp.ndarray
    apar_beta_scale: jnp.ndarray
    ampere_g0_scale: jnp.ndarray
    bpar_beta_scale: jnp.ndarray
    tau_e: jnp.ndarray
    fapar: jnp.ndarray
    w_bpar: jnp.ndarray
    charge: jnp.ndarray
    density: jnp.ndarray
    temp: jnp.ndarray
    mass: jnp.ndarray
    tz: jnp.ndarray
    zt: jnp.ndarray
    vth: jnp.ndarray


@dataclass(frozen=True)
class _FieldMoments:
    Gm0: jnp.ndarray
    nbar: jnp.ndarray
    g0: jnp.ndarray
    qneut: jnp.ndarray
    qphi: jnp.ndarray


def _species_sum(value: jnp.ndarray, axis_name: str | None) -> jnp.ndarray:
    """Sum local species contributions and optionally reduce across devices."""

    local_sum = jnp.sum(value, axis=0)
    return local_sum if axis_name is None else jax.lax.psum(local_sum, axis_name)


def _field_solve_coefficients(
    G: jnp.ndarray,
    cache,
    params,
    *,
    charge: jnp.ndarray,
    density: jnp.ndarray,
    temp: jnp.ndarray,
    mass: jnp.ndarray,
    tz: jnp.ndarray,
    vth: jnp.ndarray,
    fapar: jnp.ndarray,
    w_bpar: jnp.ndarray,
) -> _FieldSolveCoefficients:
    out_dtype = jnp.result_type(G, jnp.complex64)
    real_dtype = jnp.real(jnp.empty((), dtype=out_dtype)).dtype
    tz_arr = jnp.asarray(tz, dtype=real_dtype)
    return _FieldSolveCoefficients(
        Jl=cache.Jl,
        JlB=cache.JlB,
        bmag=cache.bmag,
        kperp2=cache.kperp2,
        beta=jnp.asarray(params.beta, dtype=real_dtype),
        apar_beta_scale=jnp.asarray(params.apar_beta_scale, dtype=real_dtype),
        ampere_g0_scale=jnp.asarray(params.ampere_g0_scale, dtype=real_dtype),
        bpar_beta_scale=jnp.asarray(params.bpar_beta_scale, dtype=real_dtype),
        tau_e=jnp.asarray(params.tau_e, dtype=real_dtype),
        fapar=jnp.asarray(fapar, dtype=real_dtype),
        w_bpar=jnp.asarray(w_bpar, dtype=real_dtype),
        charge=jnp.asarray(charge, dtype=real_dtype),
        density=jnp.asarray(density, dtype=real_dtype),
        temp=jnp.asarray(temp, dtype=real_dtype),
        mass=jnp.asarray(mass, dtype=real_dtype),
        tz=tz_arr,
        zt=jnp.where(tz_arr == 0.0, 0.0, 1.0 / tz_arr),
        vth=jnp.asarray(vth, dtype=real_dtype),
    )


def _field_moments(
    G: jnp.ndarray,
    coeffs: _FieldSolveCoefficients,
    *,
    axis_name: str | None = None,
) -> _FieldMoments:
    Gm0 = G[:, :, 0, ...]
    nbar = _species_sum(
        coeffs.density[:, None, None, None]
        * coeffs.charge[:, None, None, None]
        * jnp.sum(coeffs.Jl * Gm0, axis=1),
        axis_name,
    )
    g0 = jnp.sum(coeffs.Jl * coeffs.Jl, axis=1)
    qneut = _species_sum(
        coeffs.density[:, None, None, None]
        * coeffs.charge[:, None, None, None]
        * coeffs.zt[:, None, None, None]
        * (1.0 - g0),
        axis_name,
    )
    return _FieldMoments(
        Gm0=Gm0,
        nbar=nbar,
        g0=g0,
        qneut=qneut,
        qphi=coeffs.tau_e + qneut,
    )


def _adiabatic_quasineutrality(
    cache, coeffs: _FieldSolveCoefficients, moments: _FieldMoments
) -> jnp.ndarray:
    jacobian = jnp.asarray(cache.jacobian, dtype=coeffs.tau_e.dtype)
    jac = jacobian[None, None, :]
    denom_safe = jnp.where(moments.qphi == 0.0, jnp.inf, moments.qphi)
    phi_avg_num = jnp.where(jac == 0.0, 0.0, moments.nbar / denom_safe * jac)
    phi_avg_num_sum = jnp.sum(phi_avg_num, axis=-1)
    phi_avg_denom = jnp.sum(jacobian * moments.qneut / denom_safe, axis=-1)
    phi_avg_denom_safe = jnp.where(phi_avg_denom == 0.0, jnp.inf, phi_avg_denom)
    ratio = phi_avg_num_sum / phi_avg_denom_safe
    ky0_mask = (cache.ky == 0.0)[:, None]
    kx_mask = (jnp.arange(phi_avg_num_sum.shape[1]) > 0)[None, :]
    phi_avg = jnp.where(ky0_mask & kx_mask, ratio, 0.0)
    return (moments.nbar + coeffs.tau_e * phi_avg[..., None]) / denom_safe


def _electrostatic_phi(
    G: jnp.ndarray,
    cache,
    coeffs: _FieldSolveCoefficients,
    moments: _FieldMoments,
) -> jnp.ndarray:
    denom_safe = jnp.where(moments.qphi == 0.0, jnp.inf, moments.qphi)
    phi_es = jax.lax.cond(
        jnp.any(coeffs.tau_e > 0.0),
        lambda _: _adiabatic_quasineutrality(cache, coeffs, moments),
        lambda _: moments.nbar / denom_safe,
        operand=None,
    )
    return jnp.where(cache.mask0, 0.0, phi_es)


def _solve_phi_bpar(
    cache,
    coeffs: _FieldSolveCoefficients,
    moments: _FieldMoments,
    phi_es: jnp.ndarray,
    *,
    axis_name: str | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    def solve_branch(_) -> tuple[jnp.ndarray, jnp.ndarray]:
        bmag_inv2 = 1.0 / (coeffs.bmag * coeffs.bmag)
        bpar_beta = coeffs.bpar_beta_scale * coeffs.beta
        g01 = jnp.sum(coeffs.Jl * coeffs.JlB, axis=1)
        g11 = jnp.sum(coeffs.JlB * coeffs.JlB, axis=1)
        jperpbar = _species_sum(
            (-bpar_beta)
            * coeffs.density[:, None, None, None]
            * coeffs.temp[:, None, None, None]
            * bmag_inv2[None, None, :]
            * jnp.sum(coeffs.JlB * moments.Gm0, axis=1),
            axis_name,
        )
        qb = -_species_sum(
            coeffs.density[:, None, None, None]
            * coeffs.charge[:, None, None, None]
            * g01,
            axis_name,
        )
        aphi = (
            bpar_beta
            * _species_sum(
                coeffs.density[:, None, None, None]
                * coeffs.charge[:, None, None, None]
                * g01,
                axis_name,
            )
            * bmag_inv2[None, None, :]
        )
        ab = (
            1.0
            + bpar_beta
            * _species_sum(
                coeffs.density[:, None, None, None]
                * coeffs.temp[:, None, None, None]
                * g11,
                axis_name,
            )
            * bmag_inv2[None, None, :]
        )
        denom = moments.qphi * ab - qb * aphi
        denom_safe = jnp.where(denom == 0.0, jnp.inf, denom)
        phi_em = (ab * moments.nbar - qb * jperpbar) / denom_safe
        bpar_em = (-aphi * moments.nbar + moments.qphi * jperpbar) / denom_safe
        return jnp.where(cache.mask0, 0.0, phi_em), jnp.where(
            cache.mask0, 0.0, bpar_em * jnp.sign(coeffs.w_bpar)
        )

    return jax.lax.cond(
        (coeffs.beta > 0.0) & (coeffs.w_bpar != 0.0),
        solve_branch,
        lambda _: (phi_es, jnp.zeros_like(phi_es)),
        operand=None,
    )


def _solve_apar(
    G: jnp.ndarray,
    cache,
    coeffs: _FieldSolveCoefficients,
    moments: _FieldMoments,
    phi: jnp.ndarray,
    *,
    axis_name: str | None = None,
) -> jnp.ndarray:
    def solve_branch(_) -> jnp.ndarray:
        Gm1 = G[:, :, 1, ...]
        jpar = _species_sum(
            coeffs.density[:, None, None, None]
            * coeffs.charge[:, None, None, None]
            * coeffs.vth[:, None, None, None]
            * jnp.sum(coeffs.Jl * Gm1, axis=1),
            axis_name,
        )
        jpar = coeffs.apar_beta_scale * coeffs.beta * jpar
        bmag2 = coeffs.bmag[None, None, :] * coeffs.bmag[None, None, :]
        use_bmag = jnp.asarray(
            getattr(cache, "kperp2_bmag", True), dtype=coeffs.tau_e.dtype
        )
        ampere_kperp2 = coeffs.kperp2 * (use_bmag * bmag2 + (1.0 - use_bmag))
        ampere_denom = (
            ampere_kperp2
            + coeffs.ampere_g0_scale
            * coeffs.beta
            * _species_sum(
                coeffs.density[:, None, None, None]
                * (coeffs.charge * coeffs.charge / coeffs.mass)[:, None, None, None]
                * moments.g0,
                axis_name,
            )
        )
        ampere_safe = jnp.where(ampere_denom == 0.0, jnp.inf, ampere_denom)
        return jnp.where(cache.mask0, 0.0, coeffs.fapar * jpar / ampere_safe)

    if G.shape[2] <= 1:
        return jnp.zeros_like(phi)
    return jax.lax.cond(
        (coeffs.beta > 0.0) & jnp.any(coeffs.fapar != 0.0),
        solve_branch,
        lambda _: jnp.zeros_like(phi),
        operand=None,
    )


def _solve_fields_impl(
    G: jnp.ndarray,
    cache,
    params,
    charge: jnp.ndarray,
    density: jnp.ndarray,
    temp: jnp.ndarray,
    mass: jnp.ndarray,
    tz: jnp.ndarray,
    vth: jnp.ndarray,
    fapar: jnp.ndarray,
    w_bpar: jnp.ndarray,
    *,
    axis_name: str | None = None,
) -> FieldState:
    """Solve for (phi, apar, bpar) given a distribution G and cached geometry."""

    coeffs = _field_solve_coefficients(
        G,
        cache,
        params,
        charge=charge,
        density=density,
        temp=temp,
        mass=mass,
        tz=tz,
        vth=vth,
        fapar=fapar,
        w_bpar=w_bpar,
    )
    moments = _field_moments(G, coeffs, axis_name=axis_name)
    phi_es = _electrostatic_phi(G, cache, coeffs, moments)
    phi, bpar = _solve_phi_bpar(cache, coeffs, moments, phi_es, axis_name=axis_name)
    apar = _solve_apar(G, cache, coeffs, moments, phi, axis_name=axis_name)
    return FieldState(phi=phi, apar=apar, bpar=bpar)


def solve_fields_species_shard(
    G: jnp.ndarray,
    cache,
    params,
    charge: jnp.ndarray,
    density: jnp.ndarray,
    temp: jnp.ndarray,
    mass: jnp.ndarray,
    tz: jnp.ndarray,
    vth: jnp.ndarray,
    fapar: jnp.ndarray,
    w_bpar: jnp.ndarray,
    *,
    axis_name: str = "species",
) -> FieldState:
    """Solve fields from one local species shard using named reductions."""

    return _solve_fields_impl(
        G,
        cache,
        params,
        charge,
        density,
        temp,
        mass,
        tz,
        vth,
        fapar,
        w_bpar,
        axis_name=axis_name,
    )


def solve_electrostatic_phi_species_shard(
    G: jnp.ndarray,
    cache,
    params,
    charge: jnp.ndarray,
    density: jnp.ndarray,
    tz: jnp.ndarray,
    *,
    axis_name: str = "species",
) -> jnp.ndarray:
    """Solve only quasineutrality from one local species shard."""

    zeros = jnp.zeros_like(charge)
    ones = jnp.ones_like(charge)
    coeffs = _field_solve_coefficients(
        G,
        cache,
        params,
        charge=charge,
        density=density,
        temp=ones,
        mass=ones,
        tz=tz,
        vth=ones,
        fapar=zeros,
        w_bpar=zeros,
    )
    moments = _field_moments(G, coeffs, axis_name=axis_name)
    return _electrostatic_phi(G, cache, coeffs, moments)


@jax.custom_vjp
def solve_fields(
    G: jnp.ndarray,
    cache,
    params,
    charge: jnp.ndarray,
    density: jnp.ndarray,
    temp: jnp.ndarray,
    mass: jnp.ndarray,
    tz: jnp.ndarray,
    vth: jnp.ndarray,
    fapar: jnp.ndarray,
    w_bpar: jnp.ndarray,
) -> FieldState:
    """Solve for (phi, apar, bpar) with a custom VJP hook."""

    return _solve_fields_impl(
        G,
        cache,
        params,
        charge=charge,
        density=density,
        temp=temp,
        mass=mass,
        tz=tz,
        vth=vth,
        fapar=fapar,
        w_bpar=w_bpar,
    )


def _solve_fields_fwd(
    G: jnp.ndarray,
    cache,
    params,
    charge: jnp.ndarray,
    density: jnp.ndarray,
    temp: jnp.ndarray,
    mass: jnp.ndarray,
    tz: jnp.ndarray,
    vth: jnp.ndarray,
    fapar: jnp.ndarray,
    w_bpar: jnp.ndarray,
) -> tuple[FieldState, tuple]:
    out = _solve_fields_impl(
        G,
        cache,
        params,
        charge=charge,
        density=density,
        temp=temp,
        mass=mass,
        tz=tz,
        vth=vth,
        fapar=fapar,
        w_bpar=w_bpar,
    )
    res = (G, cache, params, charge, density, temp, mass, tz, vth, fapar, w_bpar)
    return out, res


def _solve_fields_bwd(res, g):
    G, cache, params, charge, density, temp, mass, tz, vth, fapar, w_bpar = res

    def wrapped(
        G_in,
        cache_in,
        params_in,
        charge_in,
        density_in,
        temp_in,
        mass_in,
        tz_in,
        vth_in,
        fapar_in,
        w_bpar_in,
    ):
        return _solve_fields_impl(
            G_in,
            cache_in,
            params_in,
            charge=charge_in,
            density=density_in,
            temp=temp_in,
            mass=mass_in,
            tz=tz_in,
            vth=vth_in,
            fapar=fapar_in,
            w_bpar=w_bpar_in,
        )

    _, pullback = jax.vjp(
        wrapped,
        G,
        cache,
        params,
        charge,
        density,
        temp,
        mass,
        tz,
        vth,
        fapar,
        w_bpar,
    )
    return pullback(g)


solve_fields.defvjp(_solve_fields_fwd, _solve_fields_bwd)
