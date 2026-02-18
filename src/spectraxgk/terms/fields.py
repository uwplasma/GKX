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
    tau_e = jnp.asarray(params.tau_e, dtype=real_dtype)
    fapar = jnp.asarray(fapar, dtype=real_dtype)
    w_bpar = jnp.asarray(w_bpar, dtype=real_dtype)
    charge = jnp.asarray(charge, dtype=real_dtype)
    density = jnp.asarray(density, dtype=real_dtype)
    temp = jnp.asarray(temp, dtype=real_dtype)
    mass = jnp.asarray(mass, dtype=real_dtype)
    tz = jnp.asarray(tz, dtype=real_dtype)
    vth = jnp.asarray(vth, dtype=real_dtype)

    Gm1 = G[:, :, 1, ...]
    Gm0 = G[:, :, 0, ...]

    phi_es = quasineutrality_phi(G, Jl, tau_e, charge, density, tz)
    phi_es = jnp.where(cache.mask0, 0.0, phi_es)

    nbar = jnp.sum(
        density[:, None, None, None]
        * charge[:, None, None, None]
        * jnp.sum(Jl * Gm0, axis=1),
        axis=0,
    )
    bmag_inv2 = 1.0 / (bmag * bmag)
    jperpbar = jnp.sum(
        (-0.5 * beta)
        * density[:, None, None, None]
        * temp[:, None, None, None]
        * bmag_inv2[None, None, :]
        * jnp.sum(JlB * Gm0, axis=1),
        axis=0,
    )
    g0 = jnp.sum(Jl * Jl, axis=1)
    g01 = jnp.sum(Jl * JlB, axis=1)
    g11 = jnp.sum(JlB * JlB, axis=1)
    qphi = tau_e + jnp.sum(
        density[:, None, None, None]
        * charge[:, None, None, None]
        * tz[:, None, None, None]
        * (1.0 - g0),
        axis=0,
    )
    qb = -jnp.sum(density[:, None, None, None] * charge[:, None, None, None] * g01, axis=0)
    aphi = 0.5 * beta * jnp.sum(
        density[:, None, None, None] * charge[:, None, None, None] * g01, axis=0
    ) * bmag_inv2[None, None, :]
    ab = 1.0 + 0.5 * beta * jnp.sum(
        density[:, None, None, None] * temp[:, None, None, None] * g11, axis=0
    ) * bmag_inv2[None, None, :]
    denom = qphi * ab - qb * aphi
    denom_safe = jnp.where(denom == 0.0, jnp.inf, denom)
    phi_em = (ab * nbar - qb * jperpbar) / denom_safe
    bpar_em = (-aphi * nbar + qphi * jperpbar) / denom_safe

    use_bpar = jnp.where((beta > 0.0) & (w_bpar > 0.0), 1.0, 0.0)
    phi = phi_es * (1.0 - use_bpar) + phi_em * use_bpar
    bpar = bpar_em * use_bpar
    phi = jnp.where(cache.mask0, 0.0, phi)
    bpar = jnp.where(cache.mask0, 0.0, bpar)

    jpar = jnp.sum(
        density[:, None, None, None]
        * charge[:, None, None, None]
        * vth[:, None, None, None]
        * jnp.sum(Jl * Gm1, axis=1),
        axis=0,
    )
    jpar = 0.5 * beta * jpar
    bmag2 = bmag[None, None, :] * bmag[None, None, :]
    ampere_denom = kperp2 * bmag2 + 0.5 * beta * jnp.sum(
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
