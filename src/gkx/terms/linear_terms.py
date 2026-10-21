"""Linear term implementations for gyrokinetic RHS assembly."""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp

from gkx.core_velocity import laguerre_gyroaverage_neighbors

from gkx.operators.nonlinear.collisions import (
    conservative_full_f_dougherty_cross_moments as conservative_full_f_dougherty_cross_moments,
)
from gkx.operators.linear.dissipation import (
    CollisionInvariantRates as CollisionInvariantRates,
    _hermite_mode_drive,
    _hypercollision_kz_source as _hypercollision_kz_source,
    _is_static_zero,
    _zeros_like_result,
    collision_invariant_rates as collision_invariant_rates,
    collision_quadratic_rate as collision_quadratic_rate,
    collisions_contribution as collisions_contribution,
    drift_kinetic_sugama_six_moment_contribution as drift_kinetic_sugama_six_moment_contribution,
    end_damping_contribution as end_damping_contribution,
    hypercollisions_contribution as hypercollisions_contribution,
    hyperdiffusion_contribution as hyperdiffusion_contribution,
    multispecies_collision_invariant_rates as multispecies_collision_invariant_rates,
)
from gkx.operators.linear.cache_model import (
    HermiteWindow,
    hermite_index_of,
    hermite_total_of,
)
from gkx.operators.linear.streaming import (
    grad_z_linked_fft,
    grad_z_periodic,
    shift_axis,
    streaming_ladder_term,
)


def hermite_closure_coefficient(hermite_count: int) -> float:
    r"""Return ``R_{M+1}`` for the reflectionless Hermite closure.

    Closing the hierarchy with ``g_{M+1} = 0`` makes the truncation a
    reflecting wall: the free-energy pulse that streams up in ``m`` bounces
    back and returns as recurrence. The outgoing (absorbing) condition is

    .. math:: g_{M+1} = -i\,\mathrm{sgn}(k_\parallel)\,R_{M+1}\,g_M,
       \qquad R_{M+1}=\frac{M}{\sqrt{2(M+1)}}\frac{\Gamma(M/2)}{\Gamma((M+1)/2)}

    which adds ``-R sqrt(M+1) v_th |k_par| G_M`` to the last moment. Since
    ``R > 0`` that is a strictly positive-definite sink -- it can only remove
    free energy. ``R -> 1 - 1/(4M)``, so the absorption becomes exact as the
    resolution grows, and ``R_3 = 0.921318`` reproduces the Hammett-Perkins
    three-pole coefficient exactly, the ``M = 2`` member of the same family.

    Reference: Kanekar, Schekochihin, Dorland & Loureiro, *J. Plasma Phys.*
    81, 305810104 (2015), equation (4.36).
    """

    order = int(hermite_count) - 1
    if order < 1:
        return 0.0
    return float(
        order
        / math.sqrt(2.0 * (order + 1.0))
        * math.gamma(order / 2.0)
        / math.gamma((order + 1.0) / 2.0)
    )


def streaming_contribution(
    H: jnp.ndarray,
    *,
    kz: jnp.ndarray,
    dz: jnp.ndarray,
    vth: jnp.ndarray,
    sqrt_p: jnp.ndarray,
    sqrt_m: jnp.ndarray,
    kpar_scale: jnp.ndarray,
    weight: jnp.ndarray,
    kx_link_plus: jnp.ndarray | None = None,
    kx_link_minus: jnp.ndarray | None = None,
    kx_mask_plus: jnp.ndarray | None = None,
    kx_mask_minus: jnp.ndarray | None = None,
    linked_indices: tuple[jnp.ndarray, ...] | None = None,
    linked_kz: tuple[jnp.ndarray, ...] | None = None,
    linked_inverse_permutation: jnp.ndarray | None = None,
    linked_full_cover: bool = False,
    linked_gather_map: jnp.ndarray | None = None,
    linked_gather_mask: jnp.ndarray | None = None,
    linked_use_gather: bool = False,
    ny_full: int | None = None,
    ntft_m0: jnp.ndarray | None = None,
    use_twist_shift: bool = False,
) -> jnp.ndarray:
    if _is_static_zero(weight, jnp.real(H).dtype):
        return _zeros_like_result(H, weight)
    vth_s = vth if vth.ndim == 0 else vth[:, None, None, None, None, None]
    return (
        -weight
        * kpar_scale
        * streaming_ladder_term(
            H,
            kz,
            vth_s,
            sqrt_p,
            sqrt_m,
            dz=dz,
            kx_link_plus=kx_link_plus,
            kx_link_minus=kx_link_minus,
            kx_mask_plus=kx_mask_plus,
            kx_mask_minus=kx_mask_minus,
            linked_indices=linked_indices,
            linked_kz=linked_kz,
            linked_inverse_permutation=linked_inverse_permutation,
            linked_full_cover=linked_full_cover,
            linked_gather_map=linked_gather_map,
            linked_gather_mask=linked_gather_mask,
            linked_use_gather=linked_use_gather,
            ny_full=ny_full,
            ntft_m0=ntft_m0,
            use_twist_shift=use_twist_shift,
        )
    )


def _streaming_ladder_rhs(
    G: jnp.ndarray,
    *,
    vth: jnp.ndarray,
    sqrt_p: jnp.ndarray,
    sqrt_m: jnp.ndarray,
) -> jnp.ndarray:
    axis_m = -4
    G_p1 = shift_axis(G, 1, axis=axis_m)
    G_m1 = shift_axis(G, -1, axis=axis_m)
    vth_s = vth[:, None, None, None, None, None]
    return -vth_s * (sqrt_p * G_p1 + sqrt_m * G_m1)


def _field_inverse_temperature(
    tz: jnp.ndarray,
) -> jnp.ndarray:
    tz_arr = tz[:, None, None, None, None, None]
    zt = jnp.where(tz_arr == 0.0, 0.0, 1.0 / tz_arr)
    return zt[:, 0, 0, 0, 0, 0][:, None, None, None, None]


def _streaming_field_drive(
    template: jnp.ndarray,
    *,
    phi: jnp.ndarray,
    apar: jnp.ndarray | None,
    bpar: jnp.ndarray | None,
    Jl: jnp.ndarray,
    JlB: jnp.ndarray,
    tz: jnp.ndarray,
    vth: jnp.ndarray,
    hermite_window: HermiteWindow | None = None,
) -> jnp.ndarray:
    zt5 = _field_inverse_temperature(tz)
    vth5 = vth[:, None, None, None, None]
    phi_s = phi[None, None, ...]
    Nm = hermite_total_of(hermite_window, template.shape[2])
    place = functools.partial(_hermite_mode_drive, hermite_window=hermite_window)
    field_rhs = jnp.zeros_like(template)
    if apar is not None:
        apar_s = apar[None, None, ...]
        drive_m0 = zt5 * (vth5 * vth5) * Jl * apar_s
        field_rhs = field_rhs + place(field_rhs, 0, drive_m0)
    if Nm > 1:
        drive_m1 = -zt5 * vth5 * Jl * phi_s
        if bpar is not None:
            drive_m1 = drive_m1 - vth5 * JlB * bpar[None, None, ...]
        field_rhs = field_rhs + place(field_rhs, 1, drive_m1)
    if Nm > 2 and apar is not None:
        drive_m2 = jnp.sqrt(2.0) * zt5 * (vth5 * vth5) * Jl * apar_s
        field_rhs = field_rhs + place(field_rhs, 2, drive_m2)
    return field_rhs


def _streaming_parallel_derivative(
    rhs: jnp.ndarray,
    *,
    kz: jnp.ndarray,
    dz: jnp.ndarray,
    use_twist_shift: bool,
    linked_indices: tuple[jnp.ndarray, ...] | None,
    linked_kz: tuple[jnp.ndarray, ...] | None,
    linked_inverse_permutation: jnp.ndarray | None,
    linked_full_cover: bool,
    linked_gather_map: jnp.ndarray | None,
    linked_gather_mask: jnp.ndarray | None,
    linked_use_gather: bool,
    ny_full: int | None = None,
    ntft_m0: jnp.ndarray | None = None,
) -> jnp.ndarray:
    if not use_twist_shift:
        return grad_z_periodic(rhs, kz=kz, ny_full=ny_full)
    if linked_indices is None or linked_kz is None:
        raise ValueError(
            "linked_indices and linked_kz must be provided for linked streaming"
        )
    return grad_z_linked_fft(
        rhs,
        dz=dz,
        linked_indices=linked_indices,
        linked_kz=linked_kz,
        linked_inverse_permutation=linked_inverse_permutation,
        linked_full_cover=linked_full_cover,
        linked_gather_map=linked_gather_map,
        linked_gather_mask=linked_gather_mask,
        linked_use_gather=linked_use_gather,
        ny_full=ny_full,
        ntft_m0=ntft_m0,
    )


def linked_streaming_operand(
    G: jnp.ndarray,
    *,
    phi: jnp.ndarray,
    apar: jnp.ndarray | None,
    bpar: jnp.ndarray | None,
    Jl: jnp.ndarray,
    JlB: jnp.ndarray,
    tz: jnp.ndarray,
    vth: jnp.ndarray,
    sqrt_p: jnp.ndarray,
    sqrt_m: jnp.ndarray,
    kpar_scale: jnp.ndarray,
    hermite_window: HermiteWindow | None = None,
) -> jnp.ndarray:
    """Return the Hermite-ladder and field-drive operand of the parallel derivative."""

    ladder_rhs = _streaming_ladder_rhs(G, vth=vth, sqrt_p=sqrt_p, sqrt_m=sqrt_m)
    field_rhs = _streaming_field_drive(
        ladder_rhs,
        phi=phi,
        apar=apar,
        bpar=bpar,
        Jl=Jl,
        JlB=JlB,
        tz=tz,
        vth=vth,
        hermite_window=hermite_window,
    )
    return kpar_scale * (ladder_rhs + field_rhs)


def linked_streaming_contribution(
    G: jnp.ndarray,
    *,
    phi: jnp.ndarray,
    apar: jnp.ndarray | None,
    bpar: jnp.ndarray | None,
    Jl: jnp.ndarray,
    JlB: jnp.ndarray,
    tz: jnp.ndarray,
    vth: jnp.ndarray,
    sqrt_p: jnp.ndarray,
    sqrt_m: jnp.ndarray,
    kpar_scale: jnp.ndarray,
    weight: jnp.ndarray,
    kz: jnp.ndarray,
    dz: jnp.ndarray,
    use_twist_shift: bool = False,
    linked_indices: tuple[jnp.ndarray, ...] | None = None,
    linked_kz: tuple[jnp.ndarray, ...] | None = None,
    linked_inverse_permutation: jnp.ndarray | None = None,
    linked_full_cover: bool = False,
    linked_gather_map: jnp.ndarray | None = None,
    linked_gather_mask: jnp.ndarray | None = None,
    linked_use_gather: bool = False,
    ny_full: int | None = None,
    ntft_m0: jnp.ndarray | None = None,
    hermite_closure: str = "truncation",
    hermite_window: HermiteWindow | None = None,
) -> jnp.ndarray:
    """Hermite-Laguerre streaming: ladder on g, add field terms, then apply parallel derivative.

    ``hermite_closure`` selects how the hierarchy terminates at ``m = M``.
    ``"truncation"`` sets ``g_{M+1} = 0``, which reflects the free-energy pulse
    and produces recurrence. ``"reflectionless"`` applies the outgoing
    condition of :func:`hermite_closure_coefficient`, a strictly dissipative
    sink acting only on the last moment.
    """

    if _is_static_zero(weight, jnp.real(G).dtype):
        return _zeros_like_result(G, weight)

    rhs = linked_streaming_operand(
        G,
        phi=phi,
        apar=apar,
        bpar=bpar,
        Jl=Jl,
        JlB=JlB,
        tz=tz,
        vth=vth,
        sqrt_p=sqrt_p,
        sqrt_m=sqrt_m,
        kpar_scale=kpar_scale,
        hermite_window=hermite_window,
    )
    streamed = weight * _streaming_parallel_derivative(
        rhs,
        kz=kz,
        dz=dz,
        use_twist_shift=use_twist_shift,
        linked_indices=linked_indices,
        linked_kz=linked_kz,
        linked_inverse_permutation=linked_inverse_permutation,
        linked_full_cover=linked_full_cover,
        linked_gather_map=linked_gather_map,
        linked_gather_mask=linked_gather_mask,
        linked_use_gather=linked_use_gather,
        ny_full=ny_full,
        ntft_m0=ntft_m0,
    )
    if str(hermite_closure).strip().lower() != "reflectionless":
        return streamed
    return streamed + weight * _reflectionless_closure(
        G,
        vth=vth,
        kpar_scale=kpar_scale,
        kz=kz,
        dz=dz,
        use_twist_shift=use_twist_shift,
        linked_indices=linked_indices,
        linked_kz=linked_kz,
        linked_inverse_permutation=linked_inverse_permutation,
        linked_full_cover=linked_full_cover,
        linked_gather_map=linked_gather_map,
        linked_gather_mask=linked_gather_mask,
        linked_use_gather=linked_use_gather,
        ny_full=ny_full,
        ntft_m0=ntft_m0,
        hermite_window=hermite_window,
    )


def _reflectionless_closure(
    G: jnp.ndarray,
    *,
    vth: jnp.ndarray,
    kpar_scale: jnp.ndarray,
    kz: jnp.ndarray,
    dz: jnp.ndarray,
    use_twist_shift: bool,
    linked_indices: tuple[jnp.ndarray, ...] | None,
    linked_kz: tuple[jnp.ndarray, ...] | None,
    linked_inverse_permutation: jnp.ndarray | None,
    linked_full_cover: bool,
    linked_gather_map: jnp.ndarray | None,
    linked_gather_mask: jnp.ndarray | None,
    linked_use_gather: bool,
    ny_full: int | None = None,
    ntft_m0: jnp.ndarray | None = None,
    hermite_window: HermiteWindow | None = None,
) -> jnp.ndarray:
    """Return ``-R sqrt(M+1) v_th |k_par| G`` on the last Hermite moment only."""

    from gkx.operators.linear.streaming import abs_z_linked_fft, abs_z_periodic

    axis_m = -4
    local_count = int(G.shape[axis_m])
    hermite_count = hermite_total_of(hermite_window, local_count)
    coefficient = hermite_closure_coefficient(hermite_count)
    if coefficient == 0.0:
        return jnp.zeros_like(G)

    # Isolate the last moment; the closure must not touch any other m. On a
    # Hermite-sharded slab "last" means the last *global* moment, so every
    # shard but the owner contributes zero.
    last = hermite_index_of(hermite_window, local_count) == (hermite_count - 1)
    shape = [1] * G.ndim
    shape[axis_m] = local_count
    mask = last.reshape(shape)
    tail = jnp.where(mask, G, 0.0)

    if use_twist_shift:
        if linked_indices is None or linked_kz is None:
            raise ValueError(
                "linked_indices and linked_kz must be provided for linked streaming"
            )
        abs_tail = abs_z_linked_fft(
            tail,
            linked_indices,
            linked_kz,
            linked_inverse_permutation=linked_inverse_permutation,
            linked_full_cover=linked_full_cover,
            linked_gather_map=linked_gather_map,
            linked_gather_mask=linked_gather_mask,
            linked_use_gather=linked_use_gather,
            ny_full=ny_full,
            ntft_m0=ntft_m0,
        )
    else:
        abs_tail = abs_z_periodic(tail, kz=kz)

    scale = coefficient * math.sqrt(float(hermite_count))
    return -scale * kpar_scale * vth * abs_tail


def _lm_views(H):
    """Return ``view(dl, dm)``: ``H`` shifted by ``(dl, dm)`` in ``(l, m)``, zero-filled.

    Every view slices one padded copy of ``H``. Written as separate pad-and-slice
    shifts, XLA:CPU recomputes the producers of ``H`` and its index logic per
    view; the shared copy is a plain strided read (2x on the drift terms).
    """

    nl, nm = H.shape[-5], H.shape[-4]
    pad = [(0, 0)] * H.ndim
    pad[-5], pad[-4] = (1, 1), (2, 2)
    padded = jnp.pad(H, pad)

    def view(dl, dm):
        part = jax.lax.slice_in_dim(padded, 1 + dl, 1 + dl + nl, axis=H.ndim - 5)
        return jax.lax.slice_in_dim(part, 2 + dm, 2 + dm + nm, axis=H.ndim - 4)

    return view


def mirror_term(H, ell, sqrt_m, sqrt_m_p1):
    """Mirror coupling ``(l, m) <- (l, m+1), (l-1, m+1), (l, m-1), (l+1, m-1)``."""

    view = _lm_views(H)
    return (
        -sqrt_m_p1 * (ell + 1.0) * view(0, 1)
        - sqrt_m_p1 * ell * view(-1, 1)
        + sqrt_m * ell * view(0, -1)
        + sqrt_m * (ell + 1.0) * view(1, -1)
    )


def curvature_term(H, m):
    """Curvature coupling ``m -> m, m +- 2``."""

    # Hermite indices are structural: an instantiated zero tangent through
    # sqrt(m (m - 1)) at m = 0 is 0 * inf = NaN in forward mode.
    m = jax.lax.stop_gradient(m)
    view = _lm_views(H)
    # Materialized: fused, XLA:CPU evaluates the square roots once per state element.
    up, down = jax.lax.optimization_barrier(
        (jnp.sqrt((m + 1.0) * (m + 2.0)), jnp.sqrt(m * (m - 1.0)))
    )
    return up * view(0, 2) + (2.0 * m + 1.0) * H + down * view(0, -2)


def gradb_term(H, ell):
    """Grad-B coupling ``l -> l, l +- 1``."""

    view = _lm_views(H)
    return (ell + 1.0) * view(1, 0) + (2.0 * ell + 1.0) * H + ell * view(-1, 0)


def mirror_contribution(
    H: jnp.ndarray,
    *,
    vth: jnp.ndarray,
    bgrad: jnp.ndarray,
    ell: jnp.ndarray,
    sqrt_m: jnp.ndarray,
    sqrt_m_p1: jnp.ndarray,
    weight: jnp.ndarray,
) -> jnp.ndarray:
    mirror = mirror_term(H, ell, sqrt_m, sqrt_m_p1)
    bgrad_s = bgrad[None, None, None, None, None, :]
    vth_s = vth[:, None, None, None, None, None]
    return -weight * vth_s * bgrad_s * mirror


def curvature_gradb_contribution(
    H: jnp.ndarray,
    *,
    tz: jnp.ndarray,
    omega_d_scale: jnp.ndarray,
    cv_d: jnp.ndarray,
    gb_d: jnp.ndarray,
    ell: jnp.ndarray,
    m: jnp.ndarray,
    imag: jnp.ndarray,
    weight_curv: jnp.ndarray,
    weight_gradb: jnp.ndarray,
) -> jnp.ndarray:
    curv = curvature_term(H, m)
    gradb = gradb_term(H, ell)
    tz_s = tz[:, None, None, None, None, None]
    icv = imag * tz_s * omega_d_scale * cv_d[None, None, None, ...]
    igb = imag * tz_s * omega_d_scale * gb_d[None, None, None, ...]
    return -weight_curv * icv * curv - weight_gradb * igb * gradb


# The drift terms shift ``H`` by up to two Hermite and one Laguerre index. XLA
# lowers each shifted view as its own sweep of the state, and the Laguerre stride
# exceeds the L2 cache, so every view costs a DRAM pass. ``fused_drift`` reads each
# ``(l, m)`` neighbourhood once per tile (Pallas, on real and imaginary planes
# because Triton has no complex dtype) and returns
# ``-f1 M(H) - i f2 C(H) - i f3 B(H)`` with ``M``, ``C``, ``B`` the three terms
# above, ``f`` real fields. The jnp terms stay the reference and the derivatives.
# Stencil: (term, dl, dm, coefficient(l, m)).
_STENCIL = (
    (0, 0, 1, lambda ell, m: -jnp.sqrt(m + 1.0) * (ell + 1.0)),
    (0, -1, 1, lambda ell, m: -jnp.sqrt(m + 1.0) * ell),
    (0, 0, -1, lambda ell, m: jnp.sqrt(m) * ell),
    (0, 1, -1, lambda ell, m: jnp.sqrt(m) * (ell + 1.0)),
    (1, 0, 2, lambda ell, m: jnp.sqrt((m + 1.0) * (m + 2.0))),
    (1, 0, 0, lambda ell, m: 2.0 * m + 1.0),
    (1, 0, -2, lambda ell, m: jnp.sqrt(m * (m - 1.0))),
    (2, 1, 0, lambda ell, m: ell + 1.0),
    (2, 0, 0, lambda ell, m: 2.0 * ell + 1.0),
    (2, -1, 0, lambda ell, m: ell),
)


def _kernel(hr, hi, f1, f2, f3, out_r, out_i, *, nl: int, nm: int):
    """One program: every ``(l, m)`` row of one tile of flattened ``(x, z)`` points."""

    a1, a2, a3 = f1[0, 0, :], f2[0, 0, :], f3[0, 0, :]

    def row(index, _):
        ell = jax.lax.div(index, nm)
        m = index - ell * nm
        fl, fm = ell.astype(a1.dtype), m.astype(a1.dtype)

        def at(ref, dl, dm):  # zero outside the (l, m) box
            inside = (0 <= ell + dl) & (ell + dl < nl) & (0 <= m + dm) & (m + dm < nm)
            at_row = ref[
                0, jnp.clip(ell + dl, 0, nl - 1), jnp.clip(m + dm, 0, nm - 1), 0, :
            ]
            return jnp.where(inside, at_row, 0.0)

        acc = [[0.0, 0.0] for _ in range(3)]  # (re, im) of mirror, curvature, grad-B
        for term, dl, dm, coeff in _STENCIL:
            c = coeff(fl, fm)
            acc[term][0] += c * at(hr, dl, dm)
            acc[term][1] += c * at(hi, dl, dm)
        (mr, mi), (cr, ci), (gr, gi) = acc
        out_r[0, ell, m, 0, :] = -a1 * mr + a2 * ci + a3 * gi
        out_i[0, ell, m, 0, :] = -a1 * mi - a2 * cr - a3 * gr
        return 0

    jax.lax.fori_loop(0, nl * nm, row, 0)


def _pallas_drift(H, f1, f2, f3, *, interpret=False):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu

    ns, nl, nm, ny, nx, nz = H.shape
    n = nx * nz
    # Tuned on an RTX A4000: 64-wide tiles saturate fp32 bandwidth; fp64 is arithmetic-bound.
    block, warps = (64, 4) if H.dtype == jnp.complex64 else (512, 2)
    block = min(block, 1 << (n - 1).bit_length())
    pad = ((0, 0),) * 4 + ((0, -n % block),)
    real = jnp.real(H).dtype
    planes = [
        jnp.pad(x.reshape(ns, nl, nm, ny, n), pad) for x in (jnp.real(H), jnp.imag(H))
    ]
    f1 = jnp.broadcast_to(f1, (ns, 1, nx, nz)).reshape(ns, 1, n)
    f2, f3 = (
        jnp.broadcast_to(f, (ns, ny, nx, nz)).reshape(ns, ny, n) for f in (f2, f3)
    )
    f1, f2, f3 = (jnp.pad(f.astype(real), pad[:3]) for f in (f1, f2, f3))
    npad = n + pad[-1][1]
    state = pl.BlockSpec((1, nl, nm, 1, block), lambda s, y, p: (s, 0, 0, y, p))
    shared = pl.BlockSpec((1, 1, block), lambda s, y, p: (s, 0, p))
    field = pl.BlockSpec((1, 1, block), lambda s, y, p: (s, y, p))
    out_r, out_i = pl.pallas_call(
        functools.partial(_kernel, nl=nl, nm=nm),
        out_shape=[jax.ShapeDtypeStruct((ns, nl, nm, ny, npad), real)] * 2,
        grid=(ns, ny, npad // block),
        in_specs=[state, state, shared, field, field],
        out_specs=[state, state],
        interpret=interpret,
        compiler_params=plgpu.CompilerParams(num_warps=warps),
    )(*planes, f1, f2, f3)
    return jax.lax.complex(out_r, out_i)[..., :n].reshape(H.shape)


def _reference_drift(H, f1, f2, f3):
    nl, nm = H.shape[1:3]
    ell = jnp.arange(nl, dtype=f1.dtype)[:, None, None, None, None]
    m = jnp.arange(nm, dtype=f1.dtype)[:, None, None, None]
    mirror = mirror_term(H, ell, jnp.sqrt(m), jnp.sqrt(m + 1.0))
    return -f1[:, None, None] * mirror - 1j * (
        f2[:, None, None] * curvature_term(H, m)
        + f3[:, None, None] * gradb_term(H, ell)
    )


def drift_contributions(H, cache, *, vth, tz, omega_d_scale, weights, imag, fuse=False):
    """Mirror, curvature and grad-B terms; ``fuse`` returns their sum as the first.

    ``weights`` is ``(mirror, curvature, grad-B)``. The fused sum is the form the
    production RHS adds up; the separate terms serve the per-term diagnostics.
    """

    w_mirror, w_curv, w_gradb = weights
    if fuse and all(jnp.ndim(w) == 0 for w in weights) and drift_kernel_usable(H):
        tz_d = omega_d_scale * tz[:, None, None, None]
        total = fused_drift(
            H,
            w_mirror * vth[:, None, None, None] * cache.bgrad,
            w_curv * tz_d * cache.cv_d,
            w_gradb * tz_d * cache.gb_d,
        )
        return total, jnp.zeros_like(total), jnp.zeros_like(total)
    mirror = mirror_contribution(
        H,
        vth=vth,
        bgrad=cache.bgrad,
        ell=cache.l,
        sqrt_m=cache.sqrt_m,
        sqrt_m_p1=cache.sqrt_m_p1,
        weight=w_mirror,
    )
    zero = jnp.asarray(0.0, dtype=w_curv.dtype)
    kwargs = dict(
        tz=tz,
        omega_d_scale=omega_d_scale,
        cv_d=cache.cv_d,
        gb_d=cache.gb_d,
        ell=cache.l,
        m=cache.m,
        imag=imag,
    )
    curvature = curvature_gradb_contribution(
        H, weight_curv=w_curv, weight_gradb=zero, **kwargs
    )
    gradb = curvature_gradb_contribution(
        H, weight_curv=zero, weight_gradb=w_gradb, **kwargs
    )
    return mirror, curvature, gradb


def drift_kernel_usable(H) -> bool:
    """Whether :func:`fused_drift` applies: an NVIDIA GPU and a state worth tiling."""

    nl, nm, _, nx, nz = H.shape[1:]
    return (
        jax.default_backend() == "gpu"
        and "NVIDIA" in jax.devices()[0].device_kind
        and nx * nz >= 64
        and nl * nm <= 256
    )


@jax.custom_jvp
def fused_drift(H, f1, f2, f3):
    """Return ``-f1 M(H) - i f2 C(H) - i f3 B(H)`` for ``H`` of shape ``(Ns, Nl, Nm, Ny, Nx, Nz)``.

    ``f1`` broadcasts to ``(Ns, 1, Nx, Nz)``; ``f2`` and ``f3`` to ``(Ns, Ny, Nx, Nz)``.
    """

    return _pallas_drift(H, f1, f2, f3)


@fused_drift.defjvp
def _fused_drift_jvp(primals, tangents):
    tangents = tuple(
        jnp.zeros_like(p) if t is None else t for p, t in zip(primals, tangents)
    )
    return fused_drift(*primals), jax.jvp(_reference_drift, primals, tangents)[1]


def _laguerre_gradient_profile(
    J: jnp.ndarray,
    *,
    b: jnp.ndarray,
    l4: jnp.ndarray,
    tprim_s: jnp.ndarray,
    fprim_s: jnp.ndarray,
    thermal_shift: float,
    upper_boundary: jnp.ndarray | None = None,
) -> jnp.ndarray:
    J_m1, J_p1 = laguerre_gyroaverage_neighbors(J, b, axis=1)
    if upper_boundary is not None:
        J_p1 = J_p1.at[:, -1].set(upper_boundary)
    return (
        J_m1 * (l4 * tprim_s)
        + J * (fprim_s + (2.0 * l4 + thermal_shift) * tprim_s)
        + J_p1 * ((l4 + 1.0) * tprim_s)
    )


def _diamagnetic_scalar_factors(
    *,
    tprim: jnp.ndarray,
    fprim: jnp.ndarray,
    tz: jnp.ndarray,
    omega_star_scale: jnp.ndarray,
    ky: jnp.ndarray,
    imag: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    omega_star = imag * omega_star_scale * ky
    tprim_s = tprim[:, None, None, None, None]
    fprim_s = fprim[:, None, None, None, None]
    tz_s = tz[:, None, None, None, None]
    omega_star_s = omega_star[None, None, :, None, None]
    return tprim_s, fprim_s, omega_star_s, omega_star_s * tz_s


def _diamagnetic_m0_drive(
    *,
    phi: jnp.ndarray,
    bpar: jnp.ndarray | None,
    Jl: jnp.ndarray,
    b: jnp.ndarray,
    JlB: jnp.ndarray,
    l4: jnp.ndarray,
    tprim_s: jnp.ndarray,
    fprim_s: jnp.ndarray,
    omega_star_s: jnp.ndarray,
    omega_star_bpar: jnp.ndarray,
) -> jnp.ndarray:
    drive_m0 = (
        omega_star_s
        * phi
        * _laguerre_gradient_profile(
            Jl,
            b=b,
            l4=l4,
            tprim_s=tprim_s,
            fprim_s=fprim_s,
            thermal_shift=0.0,
        )
    )
    if bpar is None:
        return drive_m0
    _, jl_p1 = laguerre_gyroaverage_neighbors(Jl, b, axis=1)
    jlb_upper_boundary = jl_p1[:, -1] + Jl[:, -1]
    return drive_m0 + omega_star_bpar * bpar * _laguerre_gradient_profile(
        JlB,
        b=b,
        l4=l4,
        tprim_s=tprim_s,
        fprim_s=fprim_s,
        thermal_shift=0.0,
        upper_boundary=jlb_upper_boundary,
    )


def _diamagnetic_m2_drive(
    *,
    phi: jnp.ndarray,
    bpar: jnp.ndarray | None,
    Jl: jnp.ndarray,
    JlB: jnp.ndarray,
    tprim_s: jnp.ndarray,
    omega_star_s: jnp.ndarray,
    omega_star_bpar: jnp.ndarray,
) -> jnp.ndarray:
    thermal_factor = tprim_s / jnp.sqrt(2.0)
    drive_m2 = omega_star_s * phi * Jl * thermal_factor
    if bpar is None:
        return drive_m2
    return drive_m2 + omega_star_bpar * bpar * JlB * thermal_factor


def _diamagnetic_apar_profile_drive(
    *,
    apar: jnp.ndarray,
    Jl: jnp.ndarray,
    b: jnp.ndarray,
    l4: jnp.ndarray,
    tprim_s: jnp.ndarray,
    fprim_s: jnp.ndarray,
    vth: jnp.ndarray,
    omega_star_s: jnp.ndarray,
) -> jnp.ndarray:
    vth_s = vth[:, None, None, None, None]
    return (
        -vth_s
        * omega_star_s
        * apar
        * _laguerre_gradient_profile(
            Jl,
            b=b,
            l4=l4,
            tprim_s=tprim_s,
            fprim_s=fprim_s,
            thermal_shift=1.0,
        )
    )


def _diamagnetic_apar_temperature_drive(
    *,
    apar: jnp.ndarray,
    Jl: jnp.ndarray,
    tprim_s: jnp.ndarray,
    vth: jnp.ndarray,
    omega_star_s: jnp.ndarray,
) -> jnp.ndarray:
    vth_s = vth[:, None, None, None, None]
    thermal_factor = tprim_s * jnp.sqrt(3.0 / 2.0)
    return -vth_s * omega_star_s * apar * Jl * thermal_factor


def diamagnetic_contribution(
    dG: jnp.ndarray,
    *,
    phi: jnp.ndarray,
    apar: jnp.ndarray | None,
    bpar: jnp.ndarray | None,
    Jl: jnp.ndarray,
    b: jnp.ndarray,
    JlB: jnp.ndarray,
    l4: jnp.ndarray,
    tprim: jnp.ndarray,
    fprim: jnp.ndarray,
    tz: jnp.ndarray,
    vth: jnp.ndarray,
    omega_star_scale: jnp.ndarray,
    ky: jnp.ndarray,
    imag: jnp.ndarray,
    weight: jnp.ndarray,
    hermite_window: HermiteWindow | None = None,
) -> jnp.ndarray:
    Nm = hermite_total_of(hermite_window, dG.shape[2])
    place = functools.partial(_hermite_mode_drive, hermite_window=hermite_window)
    tprim_s, fprim_s, omega_star_s, omega_star_bpar = _diamagnetic_scalar_factors(
        tprim=tprim,
        fprim=fprim,
        tz=tz,
        omega_star_scale=omega_star_scale,
        ky=ky,
        imag=imag,
    )
    drive_m0 = _diamagnetic_m0_drive(
        phi=phi,
        bpar=bpar,
        Jl=Jl,
        b=b,
        JlB=JlB,
        l4=l4,
        tprim_s=tprim_s,
        fprim_s=fprim_s,
        omega_star_s=omega_star_s,
        omega_star_bpar=omega_star_bpar,
    )
    drive = place(dG, 0, drive_m0)
    if Nm > 2:
        drive_m2 = _diamagnetic_m2_drive(
            phi=phi,
            bpar=bpar,
            Jl=Jl,
            JlB=JlB,
            tprim_s=tprim_s,
            omega_star_s=omega_star_s,
            omega_star_bpar=omega_star_bpar,
        )
        drive = drive + place(dG, 2, drive_m2)
    if Nm > 1 and apar is not None:
        apar_drive = _diamagnetic_apar_profile_drive(
            apar=apar,
            Jl=Jl,
            b=b,
            l4=l4,
            tprim_s=tprim_s,
            fprim_s=fprim_s,
            vth=vth,
            omega_star_s=omega_star_s,
        )
        drive = drive + place(dG, 1, apar_drive)
    if Nm > 3 and apar is not None:
        drive_m3 = _diamagnetic_apar_temperature_drive(
            apar=apar,
            Jl=Jl,
            tprim_s=tprim_s,
            vth=vth,
            omega_star_s=omega_star_s,
        )
        drive = drive + place(dG, 3, drive_m3)
    return dG + weight * drive
