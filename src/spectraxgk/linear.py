"""Linear electrostatic gyrokinetic building blocks (Hermite-Laguerre)."""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.sparse.linalg import gmres

from spectraxgk.geometry import FluxTubeGeometryLike
from spectraxgk.grids import SpectralGrid
from spectraxgk.linear_linked import (
    _build_linked_end_damping_profile,  # noqa: F401 - legacy private helper re-export
    _build_linked_fft_maps,  # noqa: F401 - legacy private helper re-export
    _signed_to_index,  # noqa: F401 - legacy private helper re-export
)
from spectraxgk.linear_moments import (
    apply_hermite_v,  # noqa: F401 - legacy public helper re-export
    apply_hermite_v2,  # noqa: F401 - legacy public helper re-export
    apply_laguerre_x,  # noqa: F401 - legacy public helper re-export
    build_H,  # noqa: F401 - legacy public helper re-export
    compute_b,  # noqa: F401 - legacy public helper re-export
    diamagnetic_drive_coeffs,  # noqa: F401 - legacy public helper re-export
    energy_operator,  # noqa: F401 - legacy public helper re-export
    grad_z_periodic,  # noqa: F401 - legacy public helper re-export
    lenard_bernstein_eigenvalues,  # noqa: F401 - legacy public helper re-export
    quasineutrality_phi,  # noqa: F401 - legacy public helper re-export
    shift_axis,  # noqa: F401 - legacy public helper re-export
    streaming_term,  # noqa: F401 - legacy public helper re-export
)
from spectraxgk.linear_cache import (
    LinearCache,
    _build_end_damping_profile_array,  # noqa: F401 - legacy private helper re-export
    _build_gyroaverage_cache_arrays,  # noqa: F401 - legacy private helper re-export
    _build_low_rank_moment_cache_arrays,  # noqa: F401 - legacy private helper re-export
    _numpy_dtype_for_jax,  # noqa: F401 - legacy private helper re-export
    build_linear_cache,
    collision_damping,
    hypercollision_damping,
)
from spectraxgk.linear_params import (
    LinearParams,
    LinearTerms,
    Preconditioner,  # noqa: F401 - legacy public type alias re-export
    PreconditionerSpec,
    _as_species_array,
    _check_nonnegative,  # noqa: F401 - legacy private helper re-export
    _check_positive,  # noqa: F401 - legacy private helper re-export
    _is_tracer,  # noqa: F401 - legacy private helper re-export
    _resolve_implicit_preconditioner,
    _x64_enabled,
    linear_terms_to_term_config,
    term_config_to_linear_terms,  # noqa: F401 - legacy public helper re-export
)
from spectraxgk.linear_parallel import (
    _FUSED_ELECTROSTATIC_SLICE_KERNEL_CACHE,  # noqa: F401 - legacy private helper re-export
    _electrostatic_streaming_field_rhs,  # noqa: F401 - legacy private helper re-export
    _is_electrostatic_field_terms,
    _is_electrostatic_slice_terms,  # noqa: F401 - legacy private helper re-export
    _is_streaming_only_terms,  # noqa: F401 - legacy private helper re-export
    _linear_rhs_electrostatic_slices_velocity_sharded_fused,  # noqa: F401 - legacy private helper re-export
    _resolve_parallel_devices,  # noqa: F401 - legacy private helper re-export
    _streaming_electrostatic_from_phi_velocity_sharded,  # noqa: F401 - legacy private helper re-export
    linear_rhs_electrostatic_slices_velocity_sharded,  # noqa: F401 - legacy public helper re-export
    linear_rhs_parallel_cached,
    linear_rhs_streaming_electrostatic_velocity_sharded,  # noqa: F401 - legacy public helper re-export
    linear_rhs_streaming_velocity_sharded,  # noqa: F401 - legacy public helper re-export
)


_SSPX3_ADT = float((1.0 / 6.0) ** (1.0 / 3.0))
_SSPX3_WGTFAC = float((9.0 - 2.0 * (6.0 ** (2.0 / 3.0))) ** 0.5)
_SSPX3_W1 = 0.5 * (_SSPX3_WGTFAC - 1.0)
_SSPX3_W2 = 0.5 * ((6.0 ** (2.0 / 3.0)) - 1.0 - _SSPX3_WGTFAC)
_SSPX3_W3 = (1.0 / _SSPX3_ADT) - 1.0 - _SSPX3_W2 * (_SSPX3_W1 + 1.0)


def linear_rhs(
    G: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    terms: LinearTerms | None = None,
    *,
    dt: jnp.ndarray | float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the linear RHS and electrostatic potential.

    Parameters
    ----------
    G : jnp.ndarray
        Laguerre-Hermite moments with shape (Nl, Nm, Ny, Nx, Nz).
    grid : SpectralGrid
        Flux-tube spectral grid.
    geom : SAlphaGeometry
        Analytic s-alpha geometry.
    params : LinearParams
        Physical and normalization parameters.
    """

    if G.ndim == 5:
        Nl, Nm = G.shape[0], G.shape[1]
    elif G.ndim == 6:
        Nl, Nm = G.shape[1], G.shape[2]
    else:
        raise ValueError(
            "G must have shape (Nl, Nm, Ny, Nx, Nz) or (Ns, Nl, Nm, Ny, Nx, Nz)"
        )
    cache = build_linear_cache(grid, geom, params, Nl, Nm)
    return linear_rhs_cached(G, cache, params, terms=terms, dt=dt)


def linear_rhs_cached(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    terms: LinearTerms | None = None,
    *,
    use_jit: bool = True,
    use_custom_vjp: bool = True,
    dt: jnp.ndarray | float | None = None,
    force_electrostatic_fields: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the linear RHS using precomputed geometry arrays."""

    from spectraxgk.terms.assembly import (
        assemble_rhs_cached,
        assemble_rhs_cached_electrostatic_jit,
        assemble_rhs_cached_jit,
    )

    term_cfg = linear_terms_to_term_config(terms)

    if use_jit:
        rhs_fn = (
            assemble_rhs_cached_electrostatic_jit
            if force_electrostatic_fields
            else assemble_rhs_cached_jit
        )
        dG, fields = rhs_fn(G, cache, params, term_cfg, dt)
    else:
        dG, fields = assemble_rhs_cached(
            G,
            cache,
            params,
            terms=term_cfg,
            use_custom_vjp=use_custom_vjp,
            dt=dt,
            force_electrostatic_fields=force_electrostatic_fields,
        )
    return dG, fields.phi


def _integrate_linear_cached_impl(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    dt: float,
    steps: int,
    method: str = "rk4",
    checkpoint: bool = False,
    terms: LinearTerms | None = None,
    sample_stride: int = 1,
    show_progress: bool = False,
    parallel: Any | None = None,
    force_electrostatic_fields: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Time integrate the linear system using cached geometry arrays."""
    if method not in {"euler", "rk2", "rk4", "imex", "imex2", "sspx3"}:
        raise ValueError(
            "method must be one of {'euler', 'rk2', 'rk4', 'imex', 'imex2', 'sspx3'}"
        )
    if terms is None:
        terms = LinearTerms()

    base_dtype = jnp.complex128 if _x64_enabled() else jnp.complex64
    state_dtype = jnp.result_type(G0, base_dtype)
    G0 = jnp.asarray(G0, dtype=state_dtype)
    real_dtype = jnp.real(jnp.empty((), dtype=state_dtype)).dtype
    dt_val = jnp.asarray(dt, dtype=real_dtype)
    hyper_damp = hypercollision_damping(cache, params, real_dtype)
    if G0.ndim == 5 and hyper_damp.ndim == 6:
        hyper_damp = hyper_damp[0]
    damping = (
        collision_damping(cache, params, real_dtype, squeeze_species=(G0.ndim == 5))
        + hyper_damp
    )
    damping = damping.astype(real_dtype)

    parallel_strategy = (
        "serial"
        if parallel is None
        else str(getattr(parallel, "strategy", "serial")).lower().replace("-", "_")
    )

    def rhs(G_in: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        if parallel_strategy == "serial":
            return linear_rhs_cached(
                G_in,
                cache,
                params,
                terms=terms,
                dt=dt_val,
                force_electrostatic_fields=force_electrostatic_fields,
            )
        return linear_rhs_parallel_cached(
            G_in, cache, params, terms=terms, parallel=parallel, dt=dt_val
        )

    def advance(G):
        dG, _phi = rhs(G)
        if method == "imex":
            dG_explicit = dG + damping * G
            return (G + dt_val * dG_explicit) / (1.0 + dt_val * damping)
        if method == "imex2":
            dG_explicit = dG + damping * G
            G_half = (G + 0.5 * dt_val * dG_explicit) / (1.0 + 0.5 * dt_val * damping)
            dG_half, _phi = rhs(G_half)
            dG_half_exp = dG_half + damping * G_half
            return (G + dt_val * dG_half_exp) / (1.0 + dt_val * damping)
        if method == "euler":
            return G + dt_val * dG
        if method == "rk2":
            k1 = dG
            k2, _ = rhs(G + 0.5 * dt_val * k1)
            return G + dt_val * k2
        if method == "sspx3":

            def _euler_step(G_state: jnp.ndarray) -> jnp.ndarray:
                dG_state, _ = rhs(G_state)
                return G_state + (_SSPX3_ADT * dt_val) * dG_state

            G1 = _euler_step(G)
            G2_euler = _euler_step(G1)
            G2 = (1.0 - _SSPX3_W1) * G + (_SSPX3_W1 - 1.0) * G1 + G2_euler
            G3 = _euler_step(G2)
            return (
                (1.0 - _SSPX3_W2 - _SSPX3_W3) * G
                + _SSPX3_W3 * G1
                + (_SSPX3_W2 - 1.0) * G2
                + G3
            )
        k1 = dG
        k2, _ = rhs(G + 0.5 * dt_val * k1)
        k3, _ = rhs(G + 0.5 * dt_val * k2)
        k4, _ = rhs(G + dt_val * k3)
        return G + (dt_val / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def step(G, idx):
        G_new = advance(G)
        _dG_new, phi_new = rhs(G_new)
        if show_progress:
            from spectraxgk.utils.callbacks import print_callback, should_emit_progress

            sim_time = (idx + 1) * dt_val
            sim_total = jnp.asarray(steps, dtype=dt_val.dtype) * dt_val
            phi_max = jnp.max(jnp.abs(phi_new))
            G_new = jax.lax.cond(
                should_emit_progress(idx, steps),
                lambda state: print_callback(
                    state,
                    idx,
                    steps,
                    0.0,
                    0.0,
                    phi_max,
                    0.0,
                    sim_time,
                    sim_total,
                    metric_labels=("|phi|_max", "|n|_max"),
                ),
                lambda state: state,
                G_new,
            )
        return G_new, phi_new

    step_fn = jax.checkpoint(step) if checkpoint else step
    indices = jnp.arange(steps)
    if sample_stride <= 1:
        return jax.lax.scan(step_fn, G0, indices)

    def sample_step(G, idx):
        def inner_step(i, state):
            return advance(state)

        G_out = jax.lax.fori_loop(0, sample_stride, inner_step, G)
        _dG_out, phi_out = rhs(G_out)
        if show_progress:
            from spectraxgk.utils.callbacks import print_callback, should_emit_progress

            completed_idx = jnp.minimum((idx + 1) * sample_stride, steps) - 1
            sim_time = jnp.minimum((idx + 1) * sample_stride, steps) * dt_val
            sim_total = jnp.asarray(steps, dtype=dt_val.dtype) * dt_val
            phi_max = jnp.max(jnp.abs(phi_out))
            G_out = jax.lax.cond(
                should_emit_progress(completed_idx, steps),
                lambda state: print_callback(
                    state,
                    completed_idx,
                    steps,
                    0.0,
                    0.0,
                    phi_max,
                    0.0,
                    sim_time,
                    sim_total,
                    metric_labels=("|phi|_max", "|n|_max"),
                ),
                lambda state: state,
                G_out,
            )
        return G_out, phi_out

    num_samples = steps // sample_stride
    sample_indices = jnp.arange(num_samples)
    return jax.lax.scan(sample_step, G0, sample_indices)


@partial(
    jax.jit,
    static_argnames=(
        "steps",
        "method",
        "checkpoint",
        "sample_stride",
        "show_progress",
        "force_electrostatic_fields",
    ),
)
def _integrate_linear_cached(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    dt: float,
    steps: int,
    method: str = "rk4",
    checkpoint: bool = False,
    terms: LinearTerms | None = None,
    sample_stride: int = 1,
    show_progress: bool = False,
    force_electrostatic_fields: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return _integrate_linear_cached_impl(
        G0,
        cache,
        params,
        dt,
        steps,
        method=method,
        checkpoint=checkpoint,
        terms=terms,
        sample_stride=sample_stride,
        show_progress=show_progress,
        force_electrostatic_fields=force_electrostatic_fields,
    )


@partial(
    jax.jit,
    static_argnames=(
        "steps",
        "method",
        "checkpoint",
        "sample_stride",
        "show_progress",
        "force_electrostatic_fields",
    ),
    donate_argnums=(0,),
)
def _integrate_linear_cached_donate(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    dt: float,
    steps: int,
    method: str = "rk4",
    checkpoint: bool = False,
    terms: LinearTerms | None = None,
    sample_stride: int = 1,
    show_progress: bool = False,
    force_electrostatic_fields: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return _integrate_linear_cached_impl(
        G0,
        cache,
        params,
        dt,
        steps,
        method=method,
        checkpoint=checkpoint,
        terms=terms,
        sample_stride=sample_stride,
        show_progress=show_progress,
        force_electrostatic_fields=force_electrostatic_fields,
    )


def _build_implicit_operator(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    dt: float,
    terms: LinearTerms | None,
    implicit_preconditioner: PreconditionerSpec,
) -> tuple[
    jnp.ndarray,
    tuple[int, ...],
    int,
    jnp.ndarray,
    Callable[[jnp.ndarray], jnp.ndarray],
    Callable[[jnp.ndarray], jnp.ndarray],
    bool,
]:
    if terms is None:
        terms = LinearTerms()
    base_dtype = jnp.complex128 if _x64_enabled() else jnp.complex64
    state_dtype = jnp.result_type(G0, base_dtype)
    G = jnp.asarray(G0, dtype=state_dtype)
    real_dtype = jnp.real(jnp.empty((), dtype=state_dtype)).dtype
    dt_val = jnp.asarray(dt, dtype=real_dtype)

    squeeze_species = False
    if G.ndim == 5:
        G = G[None, ...]
        squeeze_species = True
    shape = G.shape
    size = int(np.prod(np.asarray(shape)))
    ns = shape[0]

    hyper_damp = hypercollision_damping(cache, params, real_dtype)
    damping = (
        collision_damping(cache, params, real_dtype, squeeze_species=False) + hyper_damp
    )
    damping = damping.astype(real_dtype)

    ell = cache.l.astype(real_dtype)
    m = cache.m.astype(real_dtype)
    cv_d = cache.cv_d.astype(real_dtype)
    gb_d = cache.gb_d.astype(real_dtype)
    bgrad = cache.bgrad.astype(real_dtype)
    w_mirror = jnp.asarray(terms.mirror, dtype=real_dtype)
    w_curv = jnp.asarray(terms.curvature, dtype=real_dtype)
    w_gradb = jnp.asarray(terms.gradb, dtype=real_dtype)
    diag = jnp.zeros_like(damping, dtype=state_dtype)
    imag = jnp.asarray(1j, dtype=state_dtype)
    tz = _as_species_array(params.tz, ns, "tz").astype(real_dtype)
    vth = _as_species_array(params.vth, ns, "vth").astype(real_dtype)
    tz_b = tz[:, None, None, None, None, None]
    vth_b = vth[:, None, None, None, None, None]
    omega_d_scale = jnp.asarray(params.omega_d_scale, dtype=real_dtype)
    diag = diag - imag * tz_b * omega_d_scale * (
        w_curv * cv_d[None, None, None, ...] * (2.0 * m + 1.0)
        + w_gradb * gb_d[None, None, None, ...] * (2.0 * ell + 1.0)
    )
    bgrad = bgrad[None, None, None, None, None, :]
    mirror_diag = vth_b * (2.0 * ell + 1.0) * (2.0 * m + 1.0)
    mirror_weight = 0.2
    diag = diag - w_mirror * mirror_weight * bgrad * mirror_diag

    precond_full = 1.0 / (1.0 + dt_val * damping - dt_val * diag)
    precond_full = precond_full.astype(G.dtype)
    precond_damp = (1.0 / (1.0 + dt_val * damping)).astype(G.dtype)
    kpar = params.kpar_scale * cache.kz.astype(real_dtype)
    w_stream = jnp.asarray(terms.streaming, dtype=real_dtype)
    kpar_b = kpar[None, None, None, None, None, :]
    precond_pas = 1.0 / (
        1.0
        + dt_val * damping
        - dt_val * diag
        + imag * dt_val * w_stream * vth_b * kpar_b
    )
    precond_pas = precond_pas.astype(G.dtype)
    resolved_precond = _resolve_implicit_preconditioner(implicit_preconditioner)

    sqrt_m_line = cache.sqrt_m_ladder.reshape(-1).astype(real_dtype)
    sqrt_p_line = cache.sqrt_p.reshape(-1).astype(real_dtype)

    def _solve_hermite_lines_fft(
        x: jnp.ndarray,
        *,
        kz: jnp.ndarray,
    ) -> jnp.ndarray:
        """Invert (I - dt*L_stream) approximately via FFT(z) + tridiagonal(m)."""

        x_hat = jnp.fft.fft(x, axis=-1)
        x_hat_mlast = jnp.moveaxis(x_hat, 2, -1)  # (..., Nz, Nm)
        coeff = (
            (dt_val * w_stream * jnp.asarray(params.kpar_scale, dtype=real_dtype))
            * vth[:, None, None, None, None]
            * (imag * kz)[None, None, None, None, :]
        )
        coeff = coeff[..., None]  # (Ns, 1, 1, 1, Nz, 1)
        dl = coeff * sqrt_m_line
        du = coeff * sqrt_p_line
        du = du.at[..., -1].set(jnp.asarray(0.0, dtype=du.dtype))
        d = jnp.ones_like(du)
        batch_shape = x_hat_mlast.shape
        dl = jnp.broadcast_to(dl, batch_shape)
        d = jnp.broadcast_to(d, batch_shape)
        du = jnp.broadcast_to(du, batch_shape)
        y_hat_mlast = jax.lax.linalg.tridiagonal_solve(
            dl, d, du, x_hat_mlast[..., None]
        )[..., 0]
        y_hat = jnp.moveaxis(y_hat_mlast, -1, 2)
        return jnp.fft.ifft(y_hat, axis=-1)

    def _solve_hermite_lines_linked(x: jnp.ndarray) -> jnp.ndarray:
        """Linked-FFT variant of the Hermite-line streaming preconditioner."""

        if not cache.linked_indices:
            return _solve_hermite_lines_fft(x, kz=cache.kz)

        Ny = x.shape[-3]
        Nx = x.shape[-2]
        Nz = x.shape[-1]
        lead_shape = x.shape[:-3]
        x_flat = x.reshape(*lead_shape, Ny * Nx, Nz)
        y_flat = jnp.zeros_like(x_flat)

        def _scatter_unique(
            target: jnp.ndarray, idx_flat: jnp.ndarray, updates: jnp.ndarray
        ) -> jnp.ndarray:
            idx = jnp.asarray(idx_flat, dtype=jnp.int32)
            target_t = jnp.moveaxis(target, -2, 0)
            updates_t = jnp.moveaxis(updates, -2, 0)
            idx = idx[:, None]
            dnums = jax.lax.ScatterDimensionNumbers(
                update_window_dims=tuple(range(1, updates_t.ndim)),
                inserted_window_dims=(0,),
                scatter_dims_to_operand_dims=(0,),
            )
            out_t = jax.lax.scatter(
                target_t,
                idx,
                updates_t,
                dnums,
                unique_indices=True,
            )
            return jnp.moveaxis(out_t, 0, -2)

        for idx_map, kz_link in zip(cache.linked_indices, cache.linked_kz):
            nChains, nLinks = idx_map.shape
            idx_flat = idx_map.reshape(-1)
            x_link = jnp.take(x_flat, idx_flat, axis=-2)
            x_link = x_link.reshape(*lead_shape, nChains, nLinks * Nz)
            x_hat = jnp.fft.fft(x_link, axis=-1)
            x_hat_mlast = jnp.moveaxis(x_hat, 2, -1)  # (Ns, Nl, nChains, nfreq, Nm)
            coeff = (
                (dt_val * w_stream * jnp.asarray(params.kpar_scale, dtype=real_dtype))
                * vth[:, None, None, None]
                * (imag * kz_link)[None, None, None, :]
            )
            coeff = coeff[..., None]  # (Ns, 1, 1, nfreq, 1)
            dl = coeff * sqrt_m_line
            du = coeff * sqrt_p_line
            du = du.at[..., -1].set(jnp.asarray(0.0, dtype=du.dtype))
            d = jnp.ones_like(du)
            batch_shape = x_hat_mlast.shape
            dl = jnp.broadcast_to(dl, batch_shape)
            d = jnp.broadcast_to(d, batch_shape)
            du = jnp.broadcast_to(du, batch_shape)
            y_hat_mlast = jax.lax.linalg.tridiagonal_solve(
                dl, d, du, x_hat_mlast[..., None]
            )[..., 0]
            y_hat = jnp.moveaxis(y_hat_mlast, -1, 2)
            y_link = jnp.fft.ifft(y_hat, axis=-1)
            y_link = y_link.reshape(*lead_shape, nChains * nLinks, Nz)
            y_flat = _scatter_unique(y_flat, idx_flat, y_link)

        return y_flat.reshape(*lead_shape, Ny, Nx, Nz)

    def apply_precond_full(x_flat: jnp.ndarray) -> jnp.ndarray:
        x = x_flat.reshape(shape)
        return (x * precond_full).reshape(size)

    def apply_precond_damp(x_flat: jnp.ndarray) -> jnp.ndarray:
        x = x_flat.reshape(shape)
        return (x * precond_damp).reshape(size)

    def apply_precond_pas(x_flat: jnp.ndarray) -> jnp.ndarray:
        x = x_flat.reshape(shape)
        return (x * precond_pas).reshape(size)

    def _project_kx_coarse(x: jnp.ndarray) -> jnp.ndarray:
        """Coarse-space projection/prolongation for twist/shift coupling.

        For periodic grids this reduces to the mean over kx. For linked grids we
        average within each linked (ky, kx) chain so the coarse correction does
        not destroy the linked coupling structure.
        """

        if not cache.use_twist_shift or not cache.linked_indices:
            x_mean = jnp.mean(x, axis=4, keepdims=True)
            return jnp.broadcast_to(x_mean, x.shape)

        Ny = x.shape[-3]
        Nx = x.shape[-2]
        Nz = x.shape[-1]
        lead_shape = x.shape[:-3]
        x_flat = x.reshape(*lead_shape, Ny * Nx, Nz)
        y_flat = jnp.zeros_like(x_flat)

        def _scatter_unique(
            target: jnp.ndarray, idx_flat: jnp.ndarray, updates: jnp.ndarray
        ) -> jnp.ndarray:
            idx = jnp.asarray(idx_flat, dtype=jnp.int32)
            target_t = jnp.moveaxis(target, -2, 0)
            updates_t = jnp.moveaxis(updates, -2, 0)
            idx = idx[:, None]
            dnums = jax.lax.ScatterDimensionNumbers(
                update_window_dims=tuple(range(1, updates_t.ndim)),
                inserted_window_dims=(0,),
                scatter_dims_to_operand_dims=(0,),
            )
            out_t = jax.lax.scatter(
                target_t,
                idx,
                updates_t,
                dnums,
                unique_indices=True,
            )
            return jnp.moveaxis(out_t, 0, -2)

        for idx_map in cache.linked_indices:
            nChains, nLinks = idx_map.shape
            idx_flat = idx_map.reshape(-1)
            x_link = jnp.take(x_flat, idx_flat, axis=-2)
            x_link = x_link.reshape(*lead_shape, nChains, nLinks, Nz)
            x_mean = jnp.mean(x_link, axis=-2, keepdims=True)
            x_mean = jnp.broadcast_to(x_mean, x_link.shape)
            x_updates = x_mean.reshape(*lead_shape, nChains * nLinks, Nz)
            y_flat = _scatter_unique(y_flat, idx_flat, x_updates)

        return y_flat.reshape(*lead_shape, Ny, Nx, Nz)

    def apply_precond_pas_coarse(x_flat: jnp.ndarray) -> jnp.ndarray:
        """PAS line + kx-coarse correction (additive Schur-style)."""
        x = x_flat.reshape(shape)
        x_line = x * precond_pas
        x_coarse = _project_kx_coarse(x) * precond_pas
        x_line_coarse = _project_kx_coarse(x_line)
        x_out = x_line + (x_coarse - x_line_coarse)
        return x_out.reshape(size)

    def apply_precond_hermite_line(x_flat: jnp.ndarray) -> jnp.ndarray:
        x = x_flat.reshape(shape)
        x = x * precond_full
        x = (
            _solve_hermite_lines_linked(x)
            if cache.use_twist_shift
            else _solve_hermite_lines_fft(x, kz=cache.kz)
        )
        return x.reshape(size)

    def apply_precond_hermite_line_coarse(x_flat: jnp.ndarray) -> jnp.ndarray:
        x = x_flat.reshape(shape)
        x_line = apply_precond_hermite_line(x.reshape(size)).reshape(shape)
        x_coarse_in = _project_kx_coarse(x)
        x_coarse_full = apply_precond_hermite_line(x_coarse_in.reshape(size)).reshape(
            shape
        )
        x_line_coarse_full = _project_kx_coarse(x_line)
        return (x_line + (x_coarse_full - x_line_coarse_full)).reshape(size)

    def apply_identity(x_flat: jnp.ndarray) -> jnp.ndarray:
        return x_flat

    precond_op: Callable[[jnp.ndarray], jnp.ndarray]
    if callable(resolved_precond):
        precond_op = resolved_precond
    else:
        key = resolved_precond or "auto"
        if key in {"auto", "diag", "diagonal", "physics", "block"}:
            precond_op = apply_precond_full
        elif key in {"damping", "collisional", "hyper"}:
            precond_op = apply_precond_damp
        elif key in {"pas", "pas-line", "pas_line"}:
            precond_op = apply_precond_pas
        elif key in {"pas-coarse", "pas_schur", "block-schur", "schur", "pas-hybrid"}:
            precond_op = apply_precond_pas_coarse
        elif key in {
            "hermite-line",
            "hermite_line",
            "hermite",
            "streaming-line",
            "streaming_line",
        }:
            precond_op = apply_precond_hermite_line
        elif key in {
            "hermite-line-coarse",
            "hermite_line_coarse",
            "hermite_coarse",
            "streaming-line-coarse",
        }:
            precond_op = apply_precond_hermite_line_coarse
        elif key in {"identity", "none", "off"}:
            precond_op = apply_identity
        else:
            raise ValueError(f"Unknown implicit_preconditioner '{resolved_precond}'")

    def matvec(x_flat: jnp.ndarray) -> jnp.ndarray:
        x = x_flat.reshape(shape)
        dG, _phi = linear_rhs_cached(
            x,
            cache,
            params,
            terms=terms,
            use_jit=False,
            use_custom_vjp=False,
            dt=dt_val,
        )
        return (x - dt_val * dG).reshape(size)

    return G, shape, size, dt_val, precond_op, matvec, squeeze_species


def _integrate_linear_implicit_cached(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
    terms: LinearTerms | None = None,
    implicit_tol: float = 1.0e-6,
    implicit_maxiter: int = 200,
    implicit_iters: int = 3,
    implicit_relax: float = 0.7,
    implicit_restart: int = 20,
    implicit_solve_method: str = "batched",
    implicit_preconditioner: PreconditionerSpec = None,
    checkpoint: bool = False,
    sample_stride: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Implicit linear integrator using GMRES with a diagonal preconditioner."""
    if terms is None:
        terms = LinearTerms()
    if sample_stride < 1:
        raise ValueError("sample_stride must be >= 1")
    if steps % sample_stride != 0:
        raise ValueError("steps must be divisible by sample_stride")

    G, shape, size, dt_val, precond_op, matvec, squeeze_species = (
        _build_implicit_operator(G0, cache, params, dt, terms, implicit_preconditioner)
    )

    def fixed_point(G_in: jnp.ndarray, G_rhs: jnp.ndarray) -> jnp.ndarray:
        def body(_i, g):
            dG, _phi = linear_rhs_cached(
                g,
                cache,
                params,
                terms=terms,
                use_jit=False,
                use_custom_vjp=False,
                dt=dt_val,
            )
            g_next = G_rhs + dt_val * dG
            return (1.0 - implicit_relax) * g + implicit_relax * g_next

        return jax.lax.fori_loop(0, max(int(implicit_iters), 0), body, G_in)

    def solve_step(G_in: jnp.ndarray) -> jnp.ndarray:
        G_guess = fixed_point(G_in, G_in)
        sol, _info = gmres(
            matvec,
            G_in.reshape(size),
            x0=G_guess.reshape(size),
            tol=implicit_tol,
            maxiter=implicit_maxiter,
            restart=implicit_restart,
            M=precond_op,
            solve_method=implicit_solve_method,
        )
        return sol.reshape(shape)

    def step(G_in, _):
        G_new = solve_step(G_in)
        _dG_new, phi_new = linear_rhs_cached(
            G_new,
            cache,
            params,
            terms=terms,
            use_jit=False,
            use_custom_vjp=False,
            dt=dt_val,
        )
        return G_new, phi_new

    step_fn = jax.checkpoint(step) if checkpoint else step
    if sample_stride <= 1:
        G_out, phi_t = jax.lax.scan(step_fn, G, None, length=steps)
    else:

        def sample_step(G_in, _):
            def inner_step(_i, g):
                return solve_step(g)

            G_out_local = jax.lax.fori_loop(0, sample_stride, inner_step, G_in)
            _dG_out, phi_out = linear_rhs_cached(
                G_out_local,
                cache,
                params,
                terms=terms,
                use_jit=False,
                use_custom_vjp=False,
                dt=dt_val,
            )
            return G_out_local, phi_out

        num_samples = steps // sample_stride
        G_out, phi_t = jax.lax.scan(sample_step, G, None, length=num_samples)

    G_out = G_out[0] if squeeze_species else G_out
    return G_out, phi_t


def integrate_linear(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    dt: float,
    steps: int,
    method: str = "rk4",
    cache: LinearCache | None = None,
    implicit_tol: float = 1.0e-6,
    implicit_maxiter: int = 200,
    implicit_iters: int = 3,
    implicit_relax: float = 0.7,
    implicit_restart: int = 20,
    implicit_solve_method: str = "batched",
    implicit_preconditioner: PreconditionerSpec = None,
    terms: LinearTerms | None = None,
    checkpoint: bool = False,
    sample_stride: int = 1,
    donate: bool = False,
    show_progress: bool = False,
    parallel: Any | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Time integrate the linear system using a fixed-step scheme."""
    if terms is None:
        terms = LinearTerms()
    if sample_stride < 1:
        raise ValueError("sample_stride must be >= 1")
    if steps % sample_stride != 0:
        raise ValueError("steps must be divisible by sample_stride")
    if cache is None:
        if G0.ndim == 5:
            Nl, Nm = G0.shape[0], G0.shape[1]
        elif G0.ndim == 6:
            Nl, Nm = G0.shape[1], G0.shape[2]
        else:
            raise ValueError(
                "G0 must have shape (Nl, Nm, Ny, Nx, Nz) or (Ns, Nl, Nm, Ny, Nx, Nz)"
            )
        cache = build_linear_cache(grid, geom, params, Nl, Nm)
    if method == "semi-implicit":
        method = "imex"
    parallel_strategy = (
        "serial"
        if parallel is None
        else str(getattr(parallel, "strategy", "serial")).lower().replace("-", "_")
    )
    force_electrostatic_fields = _is_electrostatic_field_terms(terms)
    if method == "implicit":
        if parallel_strategy != "serial":
            raise NotImplementedError(
                "parallel linear integration currently supports only explicit fixed-step methods"
            )
        return _integrate_linear_implicit_cached(
            G0,
            cache,
            params,
            dt=dt,
            steps=steps,
            terms=terms,
            implicit_tol=implicit_tol,
            implicit_maxiter=implicit_maxiter,
            implicit_iters=implicit_iters,
            implicit_relax=implicit_relax,
            implicit_restart=implicit_restart,
            implicit_solve_method=implicit_solve_method,
            implicit_preconditioner=implicit_preconditioner,
            checkpoint=checkpoint,
            sample_stride=sample_stride,
        )
    if parallel_strategy != "serial":
        if donate:
            raise NotImplementedError(
                "parallel linear integration does not currently support donated input buffers"
            )
        return _integrate_linear_cached_impl(
            G0,
            cache,
            params,
            dt,
            steps,
            method=method,
            checkpoint=checkpoint,
            terms=terms,
            sample_stride=sample_stride,
            show_progress=show_progress,
            parallel=parallel,
            force_electrostatic_fields=force_electrostatic_fields,
        )
    integrator = _integrate_linear_cached_donate if donate else _integrate_linear_cached
    return integrator(
        G0,
        cache,
        params,
        dt,
        steps,
        method=method,
        checkpoint=checkpoint,
        terms=terms,
        sample_stride=sample_stride,
        show_progress=show_progress,
        force_electrostatic_fields=force_electrostatic_fields,
    )


def integrate_linear_diagnostics(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
    method: str = "rk4",
    cache: LinearCache | None = None,
    terms: LinearTerms | None = None,
    sample_stride: int = 1,
    species_index: int | None = 0,
    record_hl_energy: bool = False,
    show_progress: bool = False,
) -> (
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    | tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
):
    """Integrate and return (G_out, phi_t, density_t) for diagnostics."""

    if terms is None:
        terms = LinearTerms()
    if sample_stride < 1:
        raise ValueError("sample_stride must be >= 1")
    if steps % sample_stride != 0:
        raise ValueError("steps must be divisible by sample_stride")
    if cache is None:
        if G0.ndim == 5:
            Nl, Nm = G0.shape[0], G0.shape[1]
        elif G0.ndim == 6:
            Nl, Nm = G0.shape[1], G0.shape[2]
        else:
            raise ValueError(
                "G0 must have shape (Nl, Nm, Ny, Nx, Nz) or (Ns, Nl, Nm, Ny, Nx, Nz)"
            )
        cache = build_linear_cache(grid, geom, params, Nl, Nm)

    base_dtype = jnp.complex128 if _x64_enabled() else jnp.complex64
    state_dtype = jnp.result_type(G0, base_dtype)
    G0 = jnp.asarray(G0, dtype=state_dtype)
    real_dtype = jnp.real(jnp.empty((), dtype=state_dtype)).dtype
    dt_val = jnp.asarray(dt, dtype=real_dtype)
    hyper_damp = hypercollision_damping(cache, params, real_dtype)
    if G0.ndim == 5 and hyper_damp.ndim == 6:
        hyper_damp = hyper_damp[0]
    damping = (
        collision_damping(cache, params, real_dtype, squeeze_species=(G0.ndim == 5))
        + hyper_damp
    )
    damping = damping.astype(real_dtype)

    def advance(G_in: jnp.ndarray) -> jnp.ndarray:
        dG, _phi = linear_rhs_cached(
            G_in, cache, params, terms=terms, use_jit=False, dt=dt_val
        )
        if method == "imex":
            dG_explicit = dG + damping * G_in
            return (G_in + dt_val * dG_explicit) / (1.0 + dt_val * damping)
        if method == "imex2":
            dG_explicit = dG + damping * G_in
            G_half = (G_in + 0.5 * dt_val * dG_explicit) / (
                1.0 + 0.5 * dt_val * damping
            )
            dG_half, _phi = linear_rhs_cached(
                G_half, cache, params, terms=terms, use_jit=False, dt=dt_val
            )
            dG_half_exp = dG_half + damping * G_half
            return (G_in + dt_val * dG_half_exp) / (1.0 + dt_val * damping)
        if method == "euler":
            return G_in + dt_val * dG
        if method == "rk2":
            k1 = dG
            k2, _ = linear_rhs_cached(
                G_in + 0.5 * dt_val * k1,
                cache,
                params,
                terms=terms,
                use_jit=False,
                dt=dt_val,
            )
            return G_in + dt_val * k2
        if method == "sspx3":

            def _euler_step(G_state: jnp.ndarray) -> jnp.ndarray:
                dG_state, _phi_state = linear_rhs_cached(
                    G_state,
                    cache,
                    params,
                    terms=terms,
                    use_jit=False,
                    dt=dt_val,
                )
                return G_state + (_SSPX3_ADT * dt_val) * dG_state

            G1 = _euler_step(G_in)
            G2_euler = _euler_step(G1)
            G2 = (1.0 - _SSPX3_W1) * G_in + (_SSPX3_W1 - 1.0) * G1 + G2_euler
            G3 = _euler_step(G2)
            return (
                (1.0 - _SSPX3_W2 - _SSPX3_W3) * G_in
                + _SSPX3_W3 * G1
                + (_SSPX3_W2 - 1.0) * G2
                + G3
            )
        if method == "rk4":
            k1 = dG
            k2, _ = linear_rhs_cached(
                G_in + 0.5 * dt_val * k1,
                cache,
                params,
                terms=terms,
                use_jit=False,
                dt=dt_val,
            )
            k3, _ = linear_rhs_cached(
                G_in + 0.5 * dt_val * k2,
                cache,
                params,
                terms=terms,
                use_jit=False,
                dt=dt_val,
            )
            k4, _ = linear_rhs_cached(
                G_in + dt_val * k3, cache, params, terms=terms, use_jit=False, dt=dt_val
            )
            return G_in + (dt_val / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        raise ValueError(f"Unsupported method '{method}'")

    def density_from_G(G_in: jnp.ndarray) -> jnp.ndarray:
        Jl = cache.Jl
        if G_in.ndim == 5:
            if Jl.ndim == 5:
                Jl_s = Jl[0]
            else:
                Jl_s = Jl
            return jnp.sum(Jl_s * G_in[:, 0, ...], axis=0)
        if Jl.ndim == 5:
            if species_index is None:
                return jnp.sum(jnp.sum(Jl * G_in[:, :, 0, ...], axis=1), axis=0)
            Jl_s = Jl[int(species_index)]
            return jnp.sum(Jl_s * G_in[int(species_index), :, 0, ...], axis=0)
        if species_index is None:
            return jnp.sum(jnp.sum(Jl[None, ...] * G_in[:, :, 0, ...], axis=1), axis=0)
        return jnp.sum(Jl * G_in[int(species_index), :, 0, ...], axis=0)

    def hl_energy_from_G(G_in: jnp.ndarray) -> jnp.ndarray:
        if G_in.ndim == 5:
            return jnp.sum(jnp.abs(G_in) ** 2, axis=(2, 3, 4))
        return jnp.sum(jnp.abs(G_in) ** 2, axis=(0, 3, 4, 5))

    def step(G_in, idx):
        G_out = advance(G_in)
        _dG, phi = linear_rhs_cached(
            G_out, cache, params, terms=terms, use_jit=False, dt=dt_val
        )
        density = density_from_G(G_out)
        if show_progress:
            from spectraxgk.utils.callbacks import print_callback, should_emit_progress

            sim_time = (idx + 1) * dt_val
            sim_total = jnp.asarray(steps, dtype=dt_val.dtype) * dt_val
            phi_max = jnp.max(jnp.abs(phi))
            density_max = jnp.max(jnp.abs(density))
            G_out = jax.lax.cond(
                should_emit_progress(idx, steps),
                lambda state: print_callback(
                    state,
                    idx,
                    steps,
                    0.0,
                    0.0,
                    phi_max,
                    density_max,
                    sim_time,
                    sim_total,
                    metric_labels=("|phi|_max", "|n|_max"),
                ),
                lambda state: state,
                G_out,
            )
        if record_hl_energy:
            hl_energy = hl_energy_from_G(G_out)
            return G_out, (phi, density, hl_energy)
        return G_out, (phi, density)

    if sample_stride <= 1:
        indices = jnp.arange(steps)
        G_out, outputs = jax.lax.scan(step, G0, indices)
    else:

        def sample_step(G_in, idx):
            def inner_step(_i, g):
                return advance(g)

            G_out_local = jax.lax.fori_loop(0, sample_stride, inner_step, G_in)
            _dG, phi_out = linear_rhs_cached(
                G_out_local, cache, params, terms=terms, use_jit=False, dt=dt_val
            )
            density_out = density_from_G(G_out_local)
            if show_progress:
                from spectraxgk.utils.callbacks import (
                    print_callback,
                    should_emit_progress,
                )

                completed_idx = jnp.minimum((idx + 1) * sample_stride, steps) - 1
                sim_time = jnp.minimum((idx + 1) * sample_stride, steps) * dt_val
                sim_total = jnp.asarray(steps, dtype=dt_val.dtype) * dt_val
                phi_max = jnp.max(jnp.abs(phi_out))
                density_max = jnp.max(jnp.abs(density_out))
                G_out_local = jax.lax.cond(
                    should_emit_progress(completed_idx, steps),
                    lambda state: print_callback(
                        state,
                        completed_idx,
                        steps,
                        0.0,
                        0.0,
                        phi_max,
                        density_max,
                        sim_time,
                        sim_total,
                        metric_labels=("|phi|_max", "|n|_max"),
                    ),
                    lambda state: state,
                    G_out_local,
                )
            if record_hl_energy:
                hl_out = hl_energy_from_G(G_out_local)
                return G_out_local, (phi_out, density_out, hl_out)
            return G_out_local, (phi_out, density_out)

        num_samples = steps // sample_stride
        sample_indices = jnp.arange(num_samples)
        G_out, outputs = jax.lax.scan(sample_step, G0, sample_indices)

    if record_hl_energy:
        phi_t, density_t, hl_t = outputs
        return G_out, phi_t, density_t, hl_t
    phi_t, density_t = outputs
    return G_out, phi_t, density_t
