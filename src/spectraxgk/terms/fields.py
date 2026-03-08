"""Field solves for electromagnetic gyrokinetics."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from spectraxgk.linear import quasineutrality_phi
from spectraxgk.terms.config import FieldState


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
) -> FieldState:
    """Solve for (phi, apar, bpar) given a distribution G and cached geometry."""

    out_dtype = jnp.result_type(G, jnp.complex64)
    real_dtype = jnp.real(jnp.empty((), dtype=out_dtype)).dtype

    Jl = cache.Jl
    JlB = cache.JlB
    bmag = cache.bmag
    kperp2 = cache.kperp2

    beta = jnp.asarray(params.beta, dtype=real_dtype)
    apar_beta_scale = jnp.asarray(params.apar_beta_scale, dtype=real_dtype)
    ampere_g0_scale = jnp.asarray(params.ampere_g0_scale, dtype=real_dtype)
    bpar_beta_scale = jnp.asarray(params.bpar_beta_scale, dtype=real_dtype)
    tau_e = jnp.asarray(params.tau_e, dtype=real_dtype)
    fapar = jnp.asarray(fapar, dtype=real_dtype)
    w_bpar = jnp.asarray(w_bpar, dtype=real_dtype)
    charge = jnp.asarray(charge, dtype=real_dtype)
    density = jnp.asarray(density, dtype=real_dtype)
    temp = jnp.asarray(temp, dtype=real_dtype)
    mass = jnp.asarray(mass, dtype=real_dtype)
    tz = jnp.asarray(tz, dtype=real_dtype)
    zt = jnp.where(tz == 0.0, 0.0, 1.0 / tz)
    vth = jnp.asarray(vth, dtype=real_dtype)

    Gm1 = G[:, :, 1, ...]
    Gm0 = G[:, :, 0, ...]

    nbar = jnp.sum(
        density[:, None, None, None]
        * charge[:, None, None, None]
        * jnp.sum(Jl * Gm0, axis=1),
        axis=0,
    )
    bmag_inv2 = 1.0 / (bmag * bmag)
    bpar_beta = bpar_beta_scale * beta
    jperpbar = jnp.sum(
        (-bpar_beta)
        * density[:, None, None, None]
        * temp[:, None, None, None]
        * bmag_inv2[None, None, :]
        * jnp.sum(JlB * Gm0, axis=1),
        axis=0,
    )
    g0 = jnp.sum(Jl * Jl, axis=1)
    g01 = jnp.sum(Jl * JlB, axis=1)
    g11 = jnp.sum(JlB * JlB, axis=1)
    qneut = jnp.sum(
        density[:, None, None, None]
        * charge[:, None, None, None]
        * zt[:, None, None, None]
        * (1.0 - g0),
        axis=0,
    )
    qphi = tau_e + qneut

    def _gx_quasineutrality_adiabatic() -> jnp.ndarray:
        jacobian = jnp.asarray(cache.jacobian, dtype=real_dtype)
        jac = jacobian[None, None, :]
        denom = tau_e + qneut
        denom_safe = jnp.where(denom == 0.0, jnp.inf, denom)
        phi_avg_num = jnp.where(jac == 0.0, 0.0, nbar / denom_safe * jac)
        phi_avg_num_sum = jnp.sum(phi_avg_num, axis=-1)
        phi_avg_denom = jnp.sum(jacobian * qneut / denom_safe, axis=-1)
        phi_avg_denom_safe = jnp.where(phi_avg_denom == 0.0, jnp.inf, phi_avg_denom)
        ratio = phi_avg_num_sum / phi_avg_denom_safe
        ky0_mask = (cache.ky == 0.0)[:, None]
        kx_mask = (jnp.arange(phi_avg_num_sum.shape[1]) > 0)[None, :]
        phi_avg = jnp.where(ky0_mask & kx_mask, ratio, 0.0)
        phi_val = (nbar + tau_e * phi_avg[..., None]) / denom_safe
        return phi_val

    phi_es = jax.lax.cond(
        jnp.any(tau_e > 0.0),
        lambda _: _gx_quasineutrality_adiabatic(),
        lambda _: quasineutrality_phi(G, Jl, tau_e, charge, density, tz),
        operand=None,
    )
    phi_es = jnp.where(cache.mask0, 0.0, phi_es)
    qb = -jnp.sum(density[:, None, None, None] * charge[:, None, None, None] * g01, axis=0)
    aphi = bpar_beta * jnp.sum(
        density[:, None, None, None] * charge[:, None, None, None] * g01, axis=0
    ) * bmag_inv2[None, None, :]
    ab = 1.0 + bpar_beta * jnp.sum(
        density[:, None, None, None] * temp[:, None, None, None] * g11, axis=0
    ) * bmag_inv2[None, None, :]
    denom = qphi * ab - qb * aphi
    denom_safe = jnp.where(denom == 0.0, jnp.inf, denom)
    phi_em = (ab * nbar - qb * jperpbar) / denom_safe
    bpar_em = (-aphi * nbar + qphi * jperpbar) / denom_safe

    use_bpar = jnp.where((beta > 0.0) & (w_bpar != 0.0), 1.0, 0.0)
    bpar_sign = jnp.sign(w_bpar)
    phi = phi_es * (1.0 - use_bpar) + phi_em * use_bpar
    bpar = bpar_em * use_bpar * bpar_sign
    phi = jnp.where(cache.mask0, 0.0, phi)
    bpar = jnp.where(cache.mask0, 0.0, bpar)

    jpar = jnp.sum(
        density[:, None, None, None]
        * charge[:, None, None, None]
        * vth[:, None, None, None]
        * jnp.sum(Jl * Gm1, axis=1),
        axis=0,
    )
    jpar = apar_beta_scale * beta * jpar
    bmag2 = bmag[None, None, :] * bmag[None, None, :]
    use_bmag = jnp.asarray(getattr(cache, "kperp2_bmag", True), dtype=real_dtype)
    ampere_kperp2 = kperp2 * (use_bmag * bmag2 + (1.0 - use_bmag))
    ampere_denom = ampere_kperp2 + ampere_g0_scale * beta * jnp.sum(
        density[:, None, None, None]
        * (charge * charge / mass)[:, None, None, None]
        * g0,
        axis=0,
    )
    ampere_safe = jnp.where(ampere_denom == 0.0, jnp.inf, ampere_denom)
    apar = fapar * jpar / ampere_safe
    apar = jnp.where(cache.mask0, 0.0, apar)

    return FieldState(phi=phi, apar=apar, bpar=bpar)


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
