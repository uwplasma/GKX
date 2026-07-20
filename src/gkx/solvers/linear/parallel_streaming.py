"""Velocity-sharded streaming routes for the linear RHS."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from gkx.operators.linear.cache_model import LinearCache
from gkx.operators.linear.params import (
    LinearParams,
    LinearTerms,
    _as_species_array,
    linear_terms_to_term_config,
)
from gkx.solvers.linear.parallel_common import _resolve_parallel_devices


def _species_hermite_mesh_and_state_sharding(
    state: jnp.ndarray,
    *,
    species_chunks: int,
    hermite_chunks: int,
    devices: Any | None,
):
    """Return the validated mixed mesh and state sharding."""

    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    device_list = _resolve_parallel_devices(
        num_devices=species_chunks * hermite_chunks, devices=devices
    )
    mesh = Mesh(
        np.asarray(device_list).reshape((species_chunks, hermite_chunks)),
        ("species", "m"),
    )
    state_spec = PartitionSpec("species", None, "m", None, None, None)
    return mesh, state_spec, NamedSharding(mesh, state_spec)


def prepare_electrostatic_species_hermite_state(
    state: jnp.ndarray,
    *,
    species_chunks: int = 2,
    hermite_chunks: int = 2,
    devices: Any | None = None,
) -> jnp.ndarray:
    """Place a mixed-route state once before entering a time-integration scan."""

    import jax

    arr = jnp.asarray(state)
    if arr.ndim != 6:
        raise ValueError("mixed species-Hermite routing requires a 6D state")
    if int(arr.shape[0]) != int(species_chunks):
        raise ValueError("mixed routing requires one species per mesh row")
    if int(arr.shape[2]) % int(hermite_chunks) != 0:
        raise ValueError("Hermite chunks must divide Nm evenly")
    _mesh, _spec, sharding = _species_hermite_mesh_and_state_sharding(
        arr,
        species_chunks=int(species_chunks),
        hermite_chunks=int(hermite_chunks),
        devices=devices,
    )
    return jax.device_put(arr, sharding)


def linear_rhs_electrostatic_species_hermite_sharded(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    *,
    terms: LinearTerms | None = None,
    dt: jnp.ndarray | float | None = None,
    species_chunks: int = 2,
    hermite_chunks: int = 2,
    devices: Any | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the electrostatic RHS on a species--Hermite mesh."""

    import jax
    from jax.sharding import NamedSharding, PartitionSpec

    from gkx.operators.linear.params import _as_species_array
    from gkx.operators.linear.streaming import (
        abs_z_linked_fft,
        grad_z_linked_fft,
        grad_z_periodic,
        shift_axis,
    )

    arr = jnp.asarray(G)
    if arr.ndim != 6:
        raise ValueError("mixed species-Hermite routing requires a 6D state")
    term_cfg = linear_terms_to_term_config(terms)
    unsupported = {
        "apar": term_cfg.apar,
        "bpar": term_cfg.bpar,
    }
    if any(float(value) != 0.0 for value in unsupported.values()):
        active = ", ".join(name for name, value in unsupported.items() if value)
        raise NotImplementedError(
            "mixed species-Hermite electrostatic routing does not yet support " + active
        )
    if cache.collision_lam.size != 0:
        raise NotImplementedError(
            "mixed routing requires the standard factorized collision operator"
        )
    ns = int(arr.shape[0])
    nm = int(arr.shape[2])
    s_chunks = int(species_chunks)
    m_chunks = int(hermite_chunks)
    if s_chunks != ns:
        raise ValueError("mixed routing currently requires one species per mesh row")
    if m_chunks < 2 or nm % m_chunks != 0:
        raise ValueError("Hermite chunks must be at least two and divide Nm evenly")
    mesh, state_spec, state_sharding = _species_hermite_mesh_and_state_sharding(
        arr,
        species_chunks=s_chunks,
        hermite_chunks=m_chunks,
        devices=devices,
    )
    jl_spec = PartitionSpec("species", None, None, None, None)
    b_spec = PartitionSpec("species", None, None, None)
    species_spec = PartitionSpec("species")
    phi_spec = PartitionSpec(None, None, None)
    jl_sharding = NamedSharding(mesh, jl_spec)
    b_sharding = NamedSharding(mesh, b_spec)
    species_sharding = NamedSharding(mesh, species_spec)
    real_dtype = jnp.real(arr).dtype
    charge = _as_species_array(params.charge_sign, ns, "charge_sign").astype(real_dtype)
    density = _as_species_array(params.density, ns, "density").astype(real_dtype)
    tz = _as_species_array(params.tz, ns, "tz").astype(real_dtype)
    vth = _as_species_array(params.vth, ns, "vth").astype(real_dtype)
    tprim = _as_species_array(params.R_over_LTi, ns, "R_over_LTi").astype(real_dtype)
    fprim = _as_species_array(params.R_over_Ln, ns, "R_over_Ln").astype(real_dtype)
    nu = _as_species_array(params.nu, ns, "nu").astype(real_dtype)
    local_m = nm // m_chunks
    if local_m < 2:
        raise ValueError(
            "mixed drift routing requires at least two modes per Hermite chunk"
        )
    lower_pairs = tuple((index, index + 1) for index in range(m_chunks - 1))
    upper_pairs = tuple((index, index - 1) for index in range(1, m_chunks))

    def mixed_rhs(
        local_state,
        local_jl,
        local_jlb,
        local_b,
        charge_s,
        density_s,
        tz_s,
        vth_s,
        tprim_s,
        fprim_s,
        nu_s,
    ):
        m_start = jax.lax.axis_index("m") * local_m
        global_m = m_start + jnp.arange(local_m, dtype=jnp.int32)
        m_shape = (1, 1, local_m, 1, 1, 1)
        m_real = global_m.astype(real_dtype).reshape(m_shape)
        m0 = (global_m == 0).astype(local_state.dtype).reshape(m_shape)
        m1 = (global_m == 1).astype(local_state.dtype).reshape(m_shape)

        gm0 = jnp.sum(local_state * m0, axis=2)
        weight = density_s[:, None, None, None] * charge_s[:, None, None, None]
        local_nbar = jnp.sum(weight * jnp.sum(local_jl * gm0, axis=1), axis=0)
        nbar = jax.lax.psum(local_nbar, ("species", "m"))
        g0 = jnp.sum(local_jl * local_jl, axis=1)
        zt_s = jnp.where(tz_s == 0.0, 0.0, 1.0 / tz_s)
        local_qneut = jnp.sum(weight * zt_s[:, None, None, None] * (1.0 - g0), axis=0)
        qneut = jax.lax.psum(local_qneut, "species")
        tau_e = jnp.asarray(params.tau_e, dtype=real_dtype)
        denominator = tau_e + qneut
        denominator_safe = jnp.where(denominator == 0.0, jnp.inf, denominator)
        jacobian = jnp.asarray(cache.jacobian, dtype=real_dtype)
        jac = jacobian[None, None, :]
        average_numerator = jnp.sum(
            jnp.where(jac == 0.0, 0.0, nbar / denominator_safe * jac), axis=-1
        )
        average_denominator = jnp.sum(jacobian * qneut / denominator_safe, axis=-1)
        average_denominator_safe = jnp.where(
            average_denominator == 0.0, jnp.inf, average_denominator
        )
        ky0 = (cache.ky == 0.0)[:, None]
        finite_kx = (jnp.arange(average_numerator.shape[1]) > 0)[None, :]
        phi_average = jnp.where(
            ky0 & finite_kx,
            average_numerator / average_denominator_safe,
            0.0,
        )
        phi = jax.lax.cond(
            jnp.any(tau_e > 0.0),
            lambda _: (nbar + tau_e * phi_average[..., None]) / denominator_safe,
            lambda _: nbar / denominator_safe,
            operand=None,
        )
        phi = jnp.where(cache.mask0, 0.0, phi)

        def shift_m(value, offset):
            width = abs(offset)
            if offset > 0:
                boundary = value[:, :, :width, ...]
                received = jax.lax.ppermute(boundary, "m", upper_pairs)
                return jnp.concatenate([value[:, :, width:, ...], received], axis=2)
            boundary = value[:, :, -width:, ...]
            received = jax.lax.ppermute(boundary, "m", lower_pairs)
            return jnp.concatenate([received, value[:, :, :-width, ...]], axis=2)

        lower = shift_m(local_state, -1)
        upper = shift_m(local_state, 1)
        vth6 = vth_s[:, None, None, None, None, None]
        ladder = -vth6 * (jnp.sqrt(m_real + 1.0) * upper + jnp.sqrt(m_real) * lower)
        field_drive = (
            -zt_s[:, None, None, None, None]
            * vth_s[:, None, None, None, None]
            * local_jl
            * phi[None, None, ...]
        )
        pre_derivative = ladder + m1 * field_drive[:, :, None, ...]
        if cache.use_twist_shift:
            parallel_derivative = grad_z_linked_fft(
                pre_derivative,
                dz=cache.dz,
                linked_indices=cache.linked_indices,
                linked_kz=cache.linked_kz,
                linked_inverse_permutation=cache.linked_inverse_permutation,
                linked_full_cover=cache.linked_full_cover,
                linked_gather_map=cache.linked_gather_map,
                linked_gather_mask=cache.linked_gather_mask,
                linked_use_gather=cache.linked_use_gather,
            )
        else:
            parallel_derivative = grad_z_periodic(pre_derivative, kz=cache.kz)
        streaming = (
            jnp.asarray(term_cfg.streaming, dtype=real_dtype)
            * jnp.asarray(params.kpar_scale, dtype=real_dtype)
            * parallel_derivative
        )

        zt6 = zt_s[:, None, None, None, None, None]
        hamiltonian = local_state + m0 * (
            zt6 * local_jl[:, :, None, ...] * phi[None, None, None, ...]
        )
        h_m_p1 = shift_m(hamiltonian, 1)
        h_m_m1 = shift_m(hamiltonian, -1)
        ell = cache.l[None, ...]
        ell_p1 = ell + 1.0
        sqrt_m = jnp.sqrt(m_real)
        sqrt_m_p1 = jnp.sqrt(m_real + 1.0)
        mirror_kernel = (
            -sqrt_m_p1 * ell_p1 * h_m_p1
            - sqrt_m_p1 * ell * shift_axis(h_m_p1, -1, axis=1)
            + sqrt_m * ell * h_m_m1
            + sqrt_m * ell_p1 * shift_axis(h_m_m1, 1, axis=1)
        )
        mirror = (
            -jnp.asarray(term_cfg.mirror, dtype=real_dtype)
            * vth6
            * cache.bgrad[None, None, None, None, None, :]
            * mirror_kernel
        )

        h_m_p2 = shift_m(hamiltonian, 2)
        h_m_m2 = shift_m(hamiltonian, -2)
        curvature_kernel = (
            jnp.sqrt((m_real + 1.0) * (m_real + 2.0)) * h_m_p2
            + (2.0 * m_real + 1.0) * hamiltonian
            + jnp.sqrt(m_real * jnp.maximum(m_real - 1.0, 0.0)) * h_m_m2
        )
        gradb_kernel = (
            ell_p1 * shift_axis(hamiltonian, 1, axis=1)
            + (2.0 * ell + 1.0) * hamiltonian
            + ell * shift_axis(hamiltonian, -1, axis=1)
        )
        drift_scale = (
            jnp.asarray(1j, dtype=local_state.dtype)
            * zt_s[:, None, None, None, None, None]
            * jnp.asarray(params.omega_d_scale, dtype=real_dtype)
        )
        curvature = (
            -jnp.asarray(term_cfg.curvature, dtype=real_dtype)
            * drift_scale
            * cache.cv_d[None, None, None, ...]
            * curvature_kernel
        )
        gradb = (
            -jnp.asarray(term_cfg.gradb, dtype=real_dtype)
            * drift_scale
            * cache.gb_d[None, None, None, ...]
            * gradb_kernel
        )

        jl_m1 = shift_axis(local_jl, -1, axis=1)
        jl_p1 = shift_axis(local_jl, 1, axis=1)
        l4 = cache.l4[None, ...]
        tprim5 = tprim_s[:, None, None, None, None]
        fprim5 = fprim_s[:, None, None, None, None]
        omega_star = (
            jnp.asarray(1j, dtype=local_state.dtype)
            * jnp.asarray(params.omega_star_scale, dtype=real_dtype)
            * cache.ky
        )[None, None, :, None, None]
        drive_m0 = (
            omega_star
            * phi[None, None, ...]
            * (
                jl_m1 * (l4 * tprim5)
                + local_jl * (fprim5 + 2.0 * l4 * tprim5)
                + jl_p1 * ((l4 + 1.0) * tprim5)
            )
        )
        drive_m2 = (
            omega_star
            * phi[None, None, ...]
            * local_jl
            * (tprim5 / jnp.sqrt(jnp.asarray(2.0, dtype=real_dtype)))
        )
        diamagnetic = jnp.asarray(term_cfg.diamagnetic, dtype=real_dtype) * (
            m0 * drive_m0[:, :, None, ...]
            + (global_m == 2).astype(local_state.dtype).reshape(m_shape)
            * drive_m2[:, :, None, ...]
        )

        def m_slice(value):
            return jax.lax.dynamic_slice_in_dim(value, m_start, local_m, axis=1)

        ratio_l = cache.ratio_l[None, ...]
        ratio_m = m_slice(cache.ratio_m)[None, ...]
        ratio_lm = m_slice(cache.ratio_lm)[None, ...]
        hyper_ratio = m_slice(cache.hyper_ratio)[None, ...]
        mask_const = m_slice(cache.mask_const)[None, ...]
        l_norm = jnp.asarray(max(int(arr.shape[1]), 1), dtype=real_dtype)
        m_norm = jnp.asarray(max(nm, 1), dtype=real_dtype)
        constant_rate = -(
            vth6
            * (
                l_norm * jnp.asarray(params.nu_hyper_l, dtype=real_dtype) * ratio_l
                + m_norm * jnp.asarray(params.nu_hyper_m, dtype=real_dtype) * ratio_m
            )
            + jnp.asarray(params.nu_hyper_lm, dtype=real_dtype) * ratio_lm
        )
        hypercollisions = (
            jnp.asarray(term_cfg.hypercollisions, dtype=real_dtype)
            * jnp.asarray(params.hypercollisions_const, dtype=real_dtype)
            * jnp.where(mask_const, constant_rate, 0.0)
            * local_state
            - jnp.asarray(term_cfg.hypercollisions, dtype=real_dtype)
            * jnp.asarray(params.nu_hyper, dtype=real_dtype)
            * hyper_ratio
            * local_state
        )
        kz_source = (
            -jnp.asarray(term_cfg.hypercollisions, dtype=real_dtype)
            * jnp.asarray(params.hypercollisions_kz, dtype=real_dtype)
            * jnp.asarray(params.nu_hyper_m, dtype=real_dtype)
            * cache.m_norm_kz_factor
            * 2.3
            * vth6
            * jnp.abs(jnp.asarray(params.kpar_scale, dtype=real_dtype))
            * jnp.where(
                m_slice(cache.mask_kz)[None, ...],
                m_slice(cache.m_pow)[None, ...],
                0.0,
            )
            * local_state
        )
        if cache.use_twist_shift:
            parallel_hypercollision = abs_z_linked_fft(
                kz_source,
                linked_indices=cache.linked_indices,
                linked_kz=cache.linked_kz,
                linked_inverse_permutation=cache.linked_inverse_permutation,
                linked_full_cover=cache.linked_full_cover,
                linked_gather_map=cache.linked_gather_map,
                linked_gather_mask=cache.linked_gather_mask,
                linked_use_gather=cache.linked_use_gather,
            )
        else:
            parallel_hypercollision = (
                jnp.abs(cache.kz)[None, None, None, None, None, :] * kz_source
            )
        hypercollisions = hypercollisions + parallel_hypercollision

        kperp2 = cache.ky[:, None] ** 2 + cache.kx[None, :] ** 2
        kx_index = max((int(cache.kx.size) - 1) // 3, 0)
        ky_index = max((int(cache.ky.size) - 1) // 3, 0)
        kperp2_max = cache.kx[kx_index] ** 2 + cache.ky[ky_index] ** 2
        kperp2_max = jnp.where(kperp2_max > 0.0, kperp2_max, 1.0)
        hyperdiffusion_rate = jnp.asarray(params.D_hyper, dtype=real_dtype) * (
            kperp2 / kperp2_max
        ) ** jnp.asarray(params.p_hyper_kperp, dtype=real_dtype)
        hyperdiffusion = (
            -jnp.asarray(term_cfg.hyperdiffusion, dtype=real_dtype)
            * hyperdiffusion_rate[None, None, None, :, :, None]
            * cache.dealias_mask[None, None, None, :, :, None]
            * local_state
        )
        damp_amp = jnp.asarray(params.damp_ends_amp, dtype=real_dtype)
        if dt is not None:
            dt_value = jnp.asarray(dt, dtype=real_dtype)
            damp_amp = jnp.where(dt_value != 0.0, damp_amp / dt_value, damp_amp)
        if cache.use_twist_shift and cache.linked_damp_profile.size != 0:
            damping_profile = cache.linked_damp_profile[None, None, None, ...]
        else:
            damping_profile = (cache.ky > 0.0)[
                None, None, None, :, None, None
            ] * cache.damp_profile[None, None, None, None, None, :]
        end_damping = (
            -jnp.asarray(term_cfg.end_damping, dtype=real_dtype)
            * damp_amp
            * damping_profile
            * hamiltonian
        )
        lb_local = jax.lax.dynamic_slice_in_dim(cache.lb_lam, m_start, local_m, axis=1)
        collision_rate = nu_s[:, None, None, None, None, None] * (
            lb_local[None, :, :, None, None, None] + local_b[:, None, None, ...]
        )
        h_m0 = jax.lax.psum(jnp.sum(hamiltonian * m0, axis=2), "m")
        h_m1 = jax.lax.psum(jnp.sum(hamiltonian * m1, axis=2), "m")
        m2 = (global_m == 2).astype(local_state.dtype).reshape(m_shape)
        g_m2 = jax.lax.psum(jnp.sum(local_state * m2, axis=2), "m")
        laguerre_index = jnp.arange(local_jl.shape[1], dtype=real_dtype)[
            None, :, None, None, None
        ]
        coeff_t = (
            laguerre_index * shift_axis(local_jl, -1, axis=1)
            + 2.0 * laguerre_index * local_jl
            + (laguerre_index + 1.0) * shift_axis(local_jl, 1, axis=1)
        )
        if int(local_jl.shape[1]) == 1:
            t_bar = jnp.sqrt(2.0) * jnp.sum(local_jl * g_m2, axis=1)
        else:
            t_bar = (jnp.sqrt(2.0) / 3.0) * jnp.sum(local_jl * g_m2, axis=1) + (
                2.0 / 3.0
            ) * jnp.sum(coeff_t * h_m0, axis=1)
        sqrt_b = jnp.sqrt(jnp.maximum(local_b, 0.0))
        uperp_bar = sqrt_b * jnp.sum(local_jlb * h_m0, axis=1)
        upar_bar = jnp.sum(local_jl * h_m1, axis=1)
        nu5 = nu_s[:, None, None, None, None]
        correction_m0 = (
            nu5 * sqrt_b[:, None, ...] * local_jlb * uperp_bar[:, None, ...]
            + nu5 * 2.0 * coeff_t * t_bar[:, None, ...]
        )
        correction_m1 = nu5 * local_jl * upar_bar[:, None, ...]
        correction_m2 = nu5 * jnp.sqrt(2.0) * local_jl * t_bar[:, None, ...]
        collision_correction = (
            m0 * correction_m0[:, :, None, ...]
            + m1 * correction_m1[:, :, None, ...]
            + m2 * correction_m2[:, :, None, ...]
        )
        collisions = jnp.asarray(term_cfg.collisions, dtype=real_dtype) * (
            -collision_rate * hamiltonian + collision_correction
        )
        rhs = (
            streaming
            + mirror
            + curvature
            + gradb
            + diamagnetic
            + collisions
            + hypercollisions
            + hyperdiffusion
            + end_damping
        )
        return rhs, phi

    mapped = jax.shard_map(
        mixed_rhs,
        mesh=mesh,
        in_specs=(
            state_spec,
            jl_spec,
            jl_spec,
            b_spec,
            species_spec,
            species_spec,
            species_spec,
            species_spec,
            species_spec,
            species_spec,
            species_spec,
        ),
        out_specs=(state_spec, phi_spec),
        axis_names={"species", "m"},
    )
    return mapped(
        jax.device_put(arr, state_sharding),
        jax.device_put(cache.Jl, jl_sharding),
        jax.device_put(cache.JlB, jl_sharding),
        jax.device_put(cache.b, b_sharding),
        jax.device_put(charge, species_sharding),
        jax.device_put(density, species_sharding),
        jax.device_put(tz, species_sharding),
        jax.device_put(vth, species_sharding),
        jax.device_put(tprim, species_sharding),
        jax.device_put(fprim, species_sharding),
        jax.device_put(nu, species_sharding),
    )


def linear_rhs_streaming_velocity_sharded(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    *,
    num_devices: int | None = None,
    devices: Any | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the streaming-only linear RHS with the Hermite shard-map path.

    This diagnostic route is intentionally narrower than
    :func:`linear_rhs_cached`: it covers the velocity-space streaming operator
    only and returns a zero electrostatic potential. It is used to gate the
    future production velocity decomposition before field solves, drifts,
    collisions, and nonlinear terms are exposed through the runtime path.
    """

    from gkx.parallel.velocity import (
        build_velocity_sharding_plan,
        periodic_streaming_shard_map,
    )

    arr = jnp.asarray(G)
    if arr.ndim not in (5, 6):
        raise ValueError(
            "G must have shape (Nl, Nm, Ny, Nx, Nz) or (Ns, Nl, Nm, Ny, Nx, Nz)"
        )

    device_list = _resolve_parallel_devices(num_devices=num_devices, devices=devices)
    plan = build_velocity_sharding_plan(
        arr.shape, num_devices=len(device_list), axes=("hermite",)
    )
    dG = -periodic_streaming_shard_map(
        arr, plan, kz=cache.kz, vth=params.vth, devices=device_list
    )
    phi = jnp.zeros(arr.shape[-3:], dtype=arr.dtype)
    return dG, phi


def _streaming_electrostatic_from_phi_velocity_sharded(
    arr: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    *,
    phi: jnp.ndarray,
    plan: Any,
    devices: Any,
) -> jnp.ndarray:
    """Apply electrostatic streaming with a precomputed electrostatic field."""

    from gkx.parallel.velocity import periodic_streaming_shard_map
    from gkx.operators.linear.streaming import grad_z_periodic as operator_grad_z_periodic

    particle_streaming = -periodic_streaming_shard_map(
        arr, plan, kz=cache.kz, vth=params.vth, devices=devices
    )
    real_dtype = jnp.real(arr).dtype
    G6 = arr[None, ...]
    tz = _as_species_array(params.tz, 1, "tz").astype(real_dtype)
    vth = _as_species_array(params.vth, 1, "vth").astype(real_dtype)
    field_rhs = _electrostatic_streaming_field_rhs(
        G6, phi=phi, Jl=cache.Jl, tz=tz, vth=vth
    )
    field_streaming = jnp.asarray(
        params.kpar_scale, dtype=real_dtype
    ) * operator_grad_z_periodic(field_rhs, kz=cache.kz)
    return particle_streaming + field_streaming[0]


def _electrostatic_streaming_field_rhs(
    G6: jnp.ndarray,
    *,
    phi: jnp.ndarray,
    Jl: jnp.ndarray,
    tz: jnp.ndarray,
    vth: jnp.ndarray,
) -> jnp.ndarray:
    """Build the pre-derivative electrostatic streaming field term."""

    Nm = G6.shape[2]
    m_idx = jnp.arange(Nm, dtype=jnp.int32)[None, None, :, None, None, None]
    zt = jnp.where(tz == 0.0, 0.0, 1.0 / tz)
    zt5 = zt[:, None, None, None, None]
    vth5 = vth[:, None, None, None, None]
    phi_s = phi[None, None, ...]
    drive_m1 = -zt5 * vth5 * Jl * phi_s
    return (m_idx == 1).astype(G6.dtype) * drive_m1[:, :, None, ...]


def linear_rhs_streaming_electrostatic_velocity_sharded(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    *,
    num_devices: int | None = None,
    devices: Any | None = None,
    use_custom_vjp: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute electrostatic streaming RHS with Hermite-sharded particle streaming.

    This route solves ``phi`` with the production electrostatic field solve,
    applies the Hermite velocity-sharded particle-streaming operator, and adds
    the benchmark-compatible electrostatic streaming field term. It is limited to periodic
    field-line grids and excludes electromagnetic fields by construction.
    """

    from gkx.parallel.velocity import (
        build_velocity_sharding_plan,
        electrostatic_phi_shard_map,
    )

    del use_custom_vjp
    arr = jnp.asarray(G)
    if arr.ndim != 5:
        raise NotImplementedError(
            "velocity-sharded electrostatic streaming currently supports single-species 5D states"
        )
    if bool(getattr(cache, "use_twist_shift", False)):
        raise NotImplementedError(
            "velocity-sharded electrostatic streaming currently requires a periodic z grid"
        )

    device_list = _resolve_parallel_devices(num_devices=num_devices, devices=devices)
    plan = build_velocity_sharding_plan(
        arr.shape, num_devices=len(device_list), axes=("hermite",)
    )
    phi = electrostatic_phi_shard_map(
        arr,
        plan,
        Jl=cache.Jl,
        tau_e=params.tau_e,
        charge=params.charge_sign,
        density=params.density,
        tz=params.tz,
        mask0=cache.mask0,
        devices=device_list,
    )
    return _streaming_electrostatic_from_phi_velocity_sharded(
        arr, cache, params, phi=phi, plan=plan, devices=device_list
    ), phi


__all__ = [
    "_electrostatic_streaming_field_rhs",
    "_streaming_electrostatic_from_phi_velocity_sharded",
    "linear_rhs_streaming_electrostatic_velocity_sharded",
    "linear_rhs_electrostatic_species_hermite_sharded",
    "linear_rhs_streaming_velocity_sharded",
    "prepare_electrostatic_species_hermite_state",
]
