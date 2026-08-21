from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from gkx.operators.nonlinear.diagnostic_state import (
    NonlinearDiagnosticKernels,
    compute_nonlinear_diagnostic_tuple,
    make_nonlinear_diagnostic_tuple_fn,
)
from gkx.terms.config import FieldState


def _array(value: float) -> jnp.ndarray:
    return jnp.asarray([value], dtype=jnp.float32)


def _unused(*_args, **_kwargs):
    raise AssertionError("kernel should not be used in this branch")


def _growth(*_args, **_kwargs):
    return (
        jnp.asarray([[2.0]], dtype=jnp.float32),
        jnp.asarray([[-3.0]], dtype=jnp.float32),
    )


def _minimal_inputs():
    phi = jnp.ones((1, 1, 1), dtype=jnp.complex64)
    return {
        "G_state": jnp.ones((1, 1, 1, 1, 1, 1), dtype=jnp.complex64),
        "fields_state": FieldState(phi=phi),
        "G_prev_step": jnp.zeros((1, 1, 1, 1, 1, 1), dtype=jnp.complex64),
        "fields_prev_step": FieldState(phi=phi),
        "dt_step": jnp.asarray(0.1, dtype=jnp.float32),
        "grid": SimpleNamespace(),
        "cache": SimpleNamespace(),
        "params": SimpleNamespace(),
        "vol_fac": jnp.ones((1,), dtype=jnp.float32),
        "flux_fac": jnp.asarray(1.0, dtype=jnp.float32),
        "mask": jnp.asarray([[True]]),
        "z_idx": 0,
        "use_dealias": False,
        "real_dtype": jnp.float32,
        "omega_ky_index": None,
        "omega_kx_index": None,
        "flux_scale": 1.0,
        "wphi_scale": 1.0,
    }


def test_compute_nonlinear_diagnostic_tuple_unresolved_uses_scalar_kernels() -> None:
    kernels = NonlinearDiagnosticKernels(
        instantaneous_growth_rate_step=_growth,
        phi2_resolved=_unused,
        zonal_phi_mode_kxt=_unused,
        zonal_phi_line_kxt=_unused,
        distribution_free_energy=lambda *_args, **_kwargs: _array(10),
        distribution_free_energy_resolved=_unused,
        electrostatic_field_energy=lambda *_args, **_kwargs: _array(20),
        electrostatic_field_energy_resolved=_unused,
        magnetic_vector_potential_energy=lambda *_args, **_kwargs: _array(30),
        magnetic_vector_potential_energy_resolved=_unused,
        heat_flux_species=lambda *_args, **_kwargs: jnp.asarray([4.0, 5.0]),
        heat_flux_resolved_species=_unused,
        heat_flux_channel_resolved_species=_unused,
        particle_flux_species=lambda *_args, **_kwargs: jnp.asarray([6.0]),
        particle_flux_resolved_species=_unused,
        particle_flux_channel_resolved_species=_unused,
        turbulent_heating_species=lambda *_args, **_kwargs: jnp.asarray([7.0]),
        turbulent_heating_resolved_species=_unused,
    )

    out = compute_nonlinear_diagnostic_tuple(
        **_minimal_inputs(),
        resolved_diagnostics=False,
        kernels=kernels,
    )

    assert len(out) == 13
    np.testing.assert_allclose(np.asarray(out[0]), 2.0)
    np.testing.assert_allclose(np.asarray(out[1]), -3.0)
    np.testing.assert_allclose(np.asarray(out[2]), [10.0])
    np.testing.assert_allclose(np.asarray(out[5]), 9.0)
    np.testing.assert_allclose(np.asarray(out[6]), 6.0)
    np.testing.assert_allclose(np.asarray(out[7]), 7.0)
    assert out[-1] == ()


def test_make_nonlinear_diagnostic_tuple_fn_preserves_scalar_contract() -> None:
    kernels = NonlinearDiagnosticKernels(
        instantaneous_growth_rate_step=_growth,
        phi2_resolved=_unused,
        zonal_phi_mode_kxt=_unused,
        zonal_phi_line_kxt=_unused,
        distribution_free_energy=lambda *_args, **_kwargs: _array(10),
        distribution_free_energy_resolved=_unused,
        electrostatic_field_energy=lambda *_args, **_kwargs: _array(20),
        electrostatic_field_energy_resolved=_unused,
        magnetic_vector_potential_energy=lambda *_args, **_kwargs: _array(30),
        magnetic_vector_potential_energy_resolved=_unused,
        heat_flux_species=lambda *_args, **_kwargs: jnp.asarray([4.0, 5.0]),
        heat_flux_resolved_species=_unused,
        heat_flux_channel_resolved_species=_unused,
        particle_flux_species=lambda *_args, **_kwargs: jnp.asarray([6.0]),
        particle_flux_resolved_species=_unused,
        particle_flux_channel_resolved_species=_unused,
        turbulent_heating_species=lambda *_args, **_kwargs: jnp.asarray([7.0]),
        turbulent_heating_resolved_species=_unused,
    )
    inputs = _minimal_inputs()
    compute_diag = make_nonlinear_diagnostic_tuple_fn(
        grid=inputs["grid"],
        cache=inputs["cache"],
        params=inputs["params"],
        vol_fac=inputs["vol_fac"],
        flux_fac=inputs["flux_fac"],
        mask=inputs["mask"],
        z_idx=inputs["z_idx"],
        use_dealias=inputs["use_dealias"],
        real_dtype=inputs["real_dtype"],
        omega_ky_index=inputs["omega_ky_index"],
        omega_kx_index=inputs["omega_kx_index"],
        flux_scale=inputs["flux_scale"],
        wphi_scale=inputs["wphi_scale"],
        resolved_diagnostics=False,
        kernels=kernels,
    )

    out = compute_diag(
        inputs["G_state"],
        inputs["fields_state"],
        inputs["G_prev_step"],
        inputs["fields_prev_step"],
        inputs["dt_step"],
    )

    assert len(out) == 13
    np.testing.assert_allclose(np.asarray(out[0]), 2.0)
    np.testing.assert_allclose(np.asarray(out[5]), 9.0)
    assert out[-1] == ()


def test_compute_nonlinear_diagnostic_tuple_resolved_packs_marker_order() -> None:
    def _resolved_tuple(start: int, count: int):
        return tuple(_array(start + idx) for idx in range(count))

    def _channel_tuple(start: int):
        return (
            _resolved_tuple(start, 5),
            _resolved_tuple(start + 5, 5),
            _resolved_tuple(start + 10, 5),
        )

    kernels = NonlinearDiagnosticKernels(
        instantaneous_growth_rate_step=_growth,
        phi2_resolved=lambda *_args, **_kwargs: _resolved_tuple(100, 8),
        zonal_phi_mode_kxt=lambda *_args, **_kwargs: _array(108),
        zonal_phi_line_kxt=lambda *_args, **_kwargs: _array(109),
        distribution_free_energy=lambda *_args, **_kwargs: _unused(),
        distribution_free_energy_resolved=lambda *_args, **_kwargs: _resolved_tuple(
            110, 6
        ),
        electrostatic_field_energy=lambda *_args, **_kwargs: _unused(),
        electrostatic_field_energy_resolved=lambda *_args, **_kwargs: _resolved_tuple(
            116, 5
        ),
        magnetic_vector_potential_energy=lambda *_args, **_kwargs: _unused(),
        magnetic_vector_potential_energy_resolved=lambda *_args,
        **_kwargs: _resolved_tuple(121, 5),
        heat_flux_species=lambda *_args, **_kwargs: _unused(),
        heat_flux_resolved_species=lambda *_args, **_kwargs: _resolved_tuple(126, 5),
        heat_flux_channel_resolved_species=lambda *_args, **_kwargs: _channel_tuple(
            131
        ),
        particle_flux_species=lambda *_args, **_kwargs: _unused(),
        particle_flux_resolved_species=lambda *_args, **_kwargs: _resolved_tuple(
            146, 5
        ),
        particle_flux_channel_resolved_species=lambda *_args, **_kwargs: _channel_tuple(
            151
        ),
        turbulent_heating_species=lambda *_args, **_kwargs: _unused(),
        turbulent_heating_resolved_species=lambda *_args, **_kwargs: _resolved_tuple(
            166, 5
        ),
    )

    out = compute_nonlinear_diagnostic_tuple(
        **_minimal_inputs(),
        resolved_diagnostics=True,
        kernels=kernels,
    )

    resolved = out[-1]
    assert len(resolved) == 58
    np.testing.assert_allclose(np.asarray(out[2]), [110.0])
    np.testing.assert_allclose(np.asarray(out[5]), [126.0])
    np.testing.assert_allclose(np.asarray(resolved[0]), [101.0])
    np.testing.assert_allclose(np.asarray(resolved[7]), [108.0])
    np.testing.assert_allclose(np.asarray(resolved[26]), [132.0])
    np.testing.assert_allclose(np.asarray(resolved[-1]), [170.0])
