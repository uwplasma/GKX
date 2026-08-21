import numpy as np

import jax.numpy as jnp
from dataclasses import fields, replace
import pytest
import gkx.solvers.time.explicit as explicit_time_integrators
import gkx.diagnostics as diagnostics_module
import gkx.diagnostics.moments as diagnostics_moments
import gkx.diagnostics.metadata as diagnostics_metadata
from types import SimpleNamespace

from gkx.config import CycloneBaseCase
from gkx.benchmarking.shared import _build_initial_condition
from gkx.config import InitializationConfig
from gkx.diagnostics import (
    ResolvedDiagnostics,
    SimulationDiagnostics,
    _cached_hermitian_mode_weight,
    _jl_family,
    _transport_mode_weight,
    _heat_flux_channel_contrib_species,
    _particle_flux_channel_contrib_species,
    _turbulent_heating_contrib_species,
    _reduce_scalar_kykxz,
    _reduce_species_kykxz,
    magnetic_vector_potential_energy,
    distribution_free_energy,
    distribution_free_energy_resolved,
    electrostatic_field_energy,
    electrostatic_field_energy_resolved,
    total_energy,
    magnetic_vector_potential_energy_resolved,
    heat_flux_total,
    heat_flux_resolved_species,
    heat_flux_channel_resolved_species,
    heat_flux_channel_species,
    heat_flux_species,
    particle_flux_total,
    particle_flux_resolved_species,
    particle_flux_channel_resolved_species,
    particle_flux_channel_species,
    particle_flux_species,
    phi2_resolved,
    zonal_phi_line_kxt,
    zonal_phi_mode_kxt,
    fieldline_quadrature_weights,
    turbulent_heating_total,
    turbulent_heating_resolved_species,
    turbulent_heating_species,
)
from gkx.core.velocity import gamma0
from gkx.geometry import SAlphaGeometry, sample_flux_tube_geometry
from gkx.core.grid import build_spectral_grid, select_ky_grid
from gkx.diagnostics.analysis import select_ky_index
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.params import LinearParams, LinearTerms, linear_terms_to_term_config
from gkx.solvers.time.explicit import (
    ExplicitTimeConfig,
    _apply_completed_step_state_mask,
    _growth_rate_mode_mask,
    _linear_frequency_bound,
    _instantaneous_growth_rate_step,
    _gradient_ratio_max,
    _geometry_frequency_maxima,
    _cfl_wavenumber_arrays,
    _non_twist_shift_frequency_max,
    _diagnostic_midplane_index,
    _completed_step_state_mask,
    _linear_term_config,
    _parallel_periods_from_grid,
    _rk4_step,
    _rk3_heun_step,
    integrate_linear_explicit_diagnostics,
)
from gkx.workflows.runtime.diagnostic_arrays import (
    validate_finite_runtime_diagnostics,
)
from gkx.operators.linear.params import Species, build_linear_params
from gkx.terms.assembly import assemble_rhs_cached
from gkx.terms.config import FieldState


def test_diagnostics_refactor_preserves_runtime_import_identities() -> None:
    assert (
        diagnostics_module.ResolvedDiagnostics
        is diagnostics_metadata.ResolvedDiagnostics
    )
    assert (
        diagnostics_module.SimulationDiagnostics
        is diagnostics_metadata.SimulationDiagnostics
    )
    assert (
        diagnostics_module.SimulationDiagnostics
        is diagnostics_metadata.SimulationDiagnostics
    )
    assert (
        diagnostics_module.ResolvedDiagnostics
        is diagnostics_metadata.ResolvedDiagnostics
    )
    assert (
        diagnostics_module.fieldline_quadrature_weights
        is diagnostics_moments.fieldline_quadrature_weights
    )
    assert (
        diagnostics_module._hermitian_mode_weight
        is diagnostics_moments._hermitian_mode_weight
    )
    assert (
        diagnostics_module._cached_hermitian_mode_weight
        is diagnostics_moments._cached_hermitian_mode_weight
    )
    assert (
        diagnostics_module._transport_mode_weight
        is diagnostics_moments._transport_mode_weight
    )
    assert diagnostics_module._jl_family is diagnostics_moments._jl_family
    assert (
        diagnostics_module._heat_flux_channel_contrib_species
        is diagnostics_moments._heat_flux_channel_contrib_species
    )
    assert (
        diagnostics_module._particle_flux_channel_contrib_species
        is diagnostics_moments._particle_flux_channel_contrib_species
    )
    assert (
        diagnostics_module._turbulent_heating_contrib_species
        is diagnostics_moments._turbulent_heating_contrib_species
    )


def test_jl_family_accepts_four_dimensional_arrays_and_rejects_bad_ranks() -> None:
    cache_4d = SimpleNamespace(
        Jl=jnp.ones((2, 2, 1, 3), dtype=jnp.float32),
        JlB=2.0 * jnp.ones((2, 2, 1, 3), dtype=jnp.float32),
    )

    jl, jlb, jfac = _jl_family(cache_4d)

    assert jl.shape == (1, 2, 2, 1, 3)
    assert jlb.shape == (1, 2, 2, 1, 3)
    assert jfac.shape == jl.shape

    with pytest.raises(ValueError, match="unexpected Jl rank"):
        _jl_family(SimpleNamespace(Jl=jnp.ones((2, 3, 4)), JlB=cache_4d.JlB))
    with pytest.raises(ValueError, match="unexpected JlB rank"):
        _jl_family(SimpleNamespace(Jl=cache_4d.Jl, JlB=jnp.ones((2, 3, 4))))


def test_flux_fac_nonzero_matches_positive_ky_convention() -> None:
    cfg = CycloneBaseCase()
    grid = build_spectral_grid(replace(cfg.grid, Ny=8, Nx=4))
    fac = np.asarray(_transport_mode_weight(grid, use_dealias=False))
    ky = np.asarray(grid.ky, dtype=float)
    pos = ky > 0.0
    assert np.allclose(fac[pos], 1.0)
    assert np.allclose(fac[~pos], 0.0)


def test_state_mask_and_apply_mask_remove_dealiased_and_zonal00_modes() -> None:
    cache = SimpleNamespace(
        ky=jnp.asarray([0.0, 0.25], dtype=jnp.float32),
        kx=jnp.asarray([0.0, 0.5], dtype=jnp.float32),
        dealias_mask=jnp.asarray([[1, 1], [0, 1]], dtype=bool),
    )
    state = jnp.asarray(
        np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3) + 1.0j,
        dtype=jnp.complex64,
    )

    mask = np.asarray(_completed_step_state_mask(cache))
    expected_mask = np.asarray([[False, True], [False, True]])
    np.testing.assert_array_equal(mask, expected_mask)

    masked = np.asarray(_apply_completed_step_state_mask(state, cache))
    np.testing.assert_allclose(masked[0, 0], 0.0)
    np.testing.assert_allclose(masked[1, 0], 0.0)
    np.testing.assert_allclose(masked[0, 1], np.asarray(state[0, 1]))
    np.testing.assert_allclose(masked[1, 1], np.asarray(state[1, 1]))


def test_validate_finite_runtime_diagnostics_covers_optional_and_resolved_schema() -> (
    None
):
    n = 3
    t = np.asarray([0.0, 1.0, 2.0])
    resolved_payload = {
        field.name: np.full((n, 2), 1.0 + idx, dtype=np.float64)
        for idx, field in enumerate(fields(ResolvedDiagnostics))
    }
    resolved = ResolvedDiagnostics(**resolved_payload)
    diag = SimulationDiagnostics(
        t=t,
        dt_t=np.full(n, 0.1),
        dt_mean=np.asarray(0.1),
        gamma_t=np.linspace(0.0, 0.2, n),
        omega_t=np.linspace(0.3, 0.5, n),
        Wg_t=np.linspace(1.0, 1.2, n),
        Wphi_t=np.linspace(0.5, 0.7, n),
        Wapar_t=np.zeros(n),
        heat_flux_t=np.linspace(0.0, 0.2, n),
        particle_flux_t=np.linspace(0.1, 0.3, n),
        energy_t=np.linspace(1.5, 1.9, n),
        heat_flux_species_t=np.ones((n, 2)),
        particle_flux_species_t=np.ones((n, 2)),
        turbulent_heating_t=np.ones(n),
        turbulent_heating_species_t=np.ones((n, 2)),
        phi_mode_t=np.asarray([1.0 + 0.0j, 0.5 + 0.25j, 0.25 + 0.5j]),
        resolved=resolved,
    )

    validate_finite_runtime_diagnostics(diag, label="bounded")

    expected_resolved = {
        "Phi_zonal_mode_kxt",
        "Phi_zonal_line_kxt",
        "HeatFluxES_kxst",
        "ParticleFluxBpar_zst",
        "TurbulentHeating_zst",
    }
    assert expected_resolved.issubset(
        {field.name for field in fields(ResolvedDiagnostics)}
    )

    bad_scalar = replace(diag, heat_flux_t=np.asarray([0.0, np.inf, 0.2]))
    with pytest.raises(RuntimeError, match="heat_flux_t at sample 1 at t=1"):
        validate_finite_runtime_diagnostics(bad_scalar, label="bounded")

    bad_resolved_payload = {
        field.name: np.asarray(getattr(resolved, field.name)).copy()
        for field in fields(ResolvedDiagnostics)
    }
    bad_resolved_payload["Phi_zonal_line_kxt"][2, 0] = np.nan
    bad_resolved = replace(diag, resolved=ResolvedDiagnostics(**bad_resolved_payload))
    with pytest.raises(
        RuntimeError, match=r"resolved\.Phi_zonal_line_kxt at sample 2 at t=2"
    ):
        validate_finite_runtime_diagnostics(bad_resolved, label="bounded")


def test_cached_hermitian_mode_weight_matches_full_and_one_sided_conventions() -> None:
    dealias = jnp.asarray([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=jnp.float32)
    full_cache = SimpleNamespace(
        ky=jnp.asarray([-0.25, 0.0, 0.25], dtype=jnp.float32),
        kx=jnp.asarray([-0.5, 0.5], dtype=jnp.float32),
        dealias_mask=dealias,
    )
    one_sided_cache = SimpleNamespace(
        ky=jnp.asarray([0.0, 0.25, 0.5], dtype=jnp.float32),
        kx=jnp.asarray([-0.5, 0.5], dtype=jnp.float32),
        dealias_mask=dealias,
    )

    np.testing.assert_allclose(
        np.asarray(_cached_hermitian_mode_weight(full_cache, use_dealias=True)),
        np.asarray(dealias),
    )
    np.testing.assert_allclose(
        np.asarray(_cached_hermitian_mode_weight(one_sided_cache, use_dealias=False)),
        np.asarray([[1.0, 1.0], [2.0, 2.0], [2.0, 2.0]], dtype=np.float32),
    )


def test_grid_helper_contracts_preserve_selected_ky_and_fft_ordering() -> None:
    grid = SimpleNamespace(
        kx=jnp.asarray([-1.0, 0.0, 1.0], dtype=jnp.float32),
        ky=jnp.asarray([-0.3], dtype=jnp.float32),
        z=jnp.asarray([-np.pi, -0.5 * np.pi, 0.0, 0.5 * np.pi], dtype=jnp.float32),
        ky_mode=np.asarray([2], dtype=np.int32),
    )

    assert _parallel_periods_from_grid(grid) == pytest.approx(1.0)
    kx, ky, kz = _cfl_wavenumber_arrays(grid)
    np.testing.assert_allclose(kx, [-1.0, 0.0, 1.0])
    np.testing.assert_allclose(ky, [0.3])
    np.testing.assert_allclose(kz, [0.0, 1.0, 2.0, -1.0])

    assert _parallel_periods_from_grid(
        SimpleNamespace(z=jnp.asarray([0.0], dtype=jnp.float32))
    ) == pytest.approx(1.0)


def test_eta_geometry_and_ntft_helpers_match_manual_limits() -> None:
    assert _gradient_ratio_max(
        np.asarray([2.0, 4.0]), np.asarray([1.0, 0.0])
    ) == pytest.approx(1.0e6)
    assert _diagnostic_midplane_index(1) == 0
    assert _diagnostic_midplane_index(4) == 3

    cfg = CycloneBaseCase()
    grid = build_spectral_grid(
        replace(cfg.grid, Nx=4, Ny=8, Nz=8, ntheta=None, nperiod=None, non_twist=True)
    )
    geom = SAlphaGeometry.from_config(cfg.geometry)
    theta = np.asarray(grid.z, dtype=float)
    cv_j, gb_j, cv0_j, gb0_j = geom.drift_coeffs(jnp.asarray(theta))
    bmag_j = geom.bmag(jnp.asarray(theta))

    maxima = _geometry_frequency_maxima(geom, theta)
    np.testing.assert_allclose(
        np.asarray(maxima),
        np.asarray(
            [
                np.max(np.abs(np.asarray(bmag_j))),
                np.max(np.abs(np.asarray(cv_j))),
                np.max(np.abs(np.asarray(gb_j))),
                np.max(np.abs(np.asarray(cv0_j))),
                np.max(np.abs(np.asarray(gb0_j))),
                float(geom.gradpar()),
            ]
        ),
    )

    m0_max, cv0_max, gb0_max = _non_twist_shift_frequency_max(
        geom, grid, ky_max=0.0, vpar_max=2.0, muB_max=1.5
    )
    assert m0_max == pytest.approx(0.0)
    assert cv0_max == pytest.approx(float(np.max(np.abs(np.asarray(cv0_j)))))
    assert gb0_max == pytest.approx(float(np.max(np.abs(np.asarray(gb0_j)))))


def _small_setup():
    cfg = CycloneBaseCase()
    grid_full = build_spectral_grid(cfg.grid)
    ky_index = select_ky_index(np.asarray(grid_full.ky), 0.2)
    grid = select_ky_grid(grid_full, ky_index)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(
        fprim=cfg.model.fprim,
        tprim=cfg.model.tprim_i,
        tprim_e=cfg.model.tprim_e,
        kpar_scale=float(geom.gradpar()),
        nu=cfg.model.nu_i,
    )
    cache = build_linear_cache(grid, geom, params, 4, 4)
    return cfg, grid, geom, params, cache


def _multispecies_setup(*, Nl: int = 3, Nm: int = 4):
    cfg = CycloneBaseCase()
    grid = build_spectral_grid(
        replace(cfg.grid, Nx=4, Ny=8, Nz=8, ntheta=None, nperiod=None)
    )
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = build_linear_params(
        [
            Species(
                charge=1.0, mass=1.0, density=1.0, temperature=1.0, tprim=1.0, fprim=1.0
            ),
            Species(
                charge=-1.0,
                mass=0.00027,
                density=1.0,
                temperature=1.0,
                tprim=1.0,
                fprim=1.0,
            ),
        ],
        kpar_scale=float(geom.gradpar()),
    )
    cache = build_linear_cache(grid, geom, params, Nl, Nm)
    vol_fac, flux_fac = fieldline_quadrature_weights(geom, grid)
    return cfg, grid, geom, params, cache, vol_fac, flux_fac


def test_volume_and_flux_weights_are_finite_positive_and_normalized() -> None:
    _cfg, grid, geom, _params, _cache = _small_setup()

    vol_fac, flux_fac = fieldline_quadrature_weights(geom, grid)

    assert np.all(np.isfinite(np.asarray(vol_fac)))
    assert np.all(np.isfinite(np.asarray(flux_fac)))
    assert np.all(np.asarray(vol_fac) > 0.0)
    assert np.all(np.asarray(flux_fac) > 0.0)
    np.testing.assert_allclose(np.asarray(vol_fac).sum(), 1.0, rtol=1.0e-7, atol=5.0e-7)
    np.testing.assert_allclose(
        np.asarray(flux_fac).sum(), 1.0, rtol=1.0e-7, atol=5.0e-7
    )


def test_resolved_energy_reductions_sum_to_scalar_totals() -> None:
    _cfg, grid, _geom, params, cache, vol_fac, _flux_fac = _multispecies_setup(
        Nl=3, Nm=4
    )
    shape = (2, 3, 4, grid.ky.size, grid.kx.size, grid.z.size)
    base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1.0
    G = jnp.asarray(base + 1.0j * (0.25 * base + 1.0), dtype=jnp.complex64)
    field_base = np.arange(
        grid.ky.size * grid.kx.size * grid.z.size, dtype=np.float32
    ).reshape(grid.ky.size, grid.kx.size, grid.z.size)
    phi = jnp.asarray(field_base + 1.0j * (field_base + 0.5), dtype=jnp.complex64)
    apar = (0.2 - 0.1j) * phi

    Wg = distribution_free_energy(G, grid, params, vol_fac, use_dealias=False)
    Wg_st, Wg_kxst, Wg_kyst, Wg_kxkyst, Wg_zst, Wg_lmst = (
        distribution_free_energy_resolved(G, grid, params, vol_fac, use_dealias=False)
    )
    np.testing.assert_allclose(
        np.asarray(Wg_st).sum(), np.asarray(Wg), rtol=1.0e-5, atol=1.0e-5
    )
    for spectrum, axes in (
        (Wg_kxst, 1),
        (Wg_kyst, 1),
        (Wg_kxkyst, (1, 2)),
        (Wg_zst, 1),
        (Wg_lmst, (1, 2)),
    ):
        np.testing.assert_allclose(
            np.asarray(spectrum).sum(axis=axes),
            np.asarray(Wg_st),
            rtol=1.0e-5,
            atol=1.0e-5,
        )

    Wphi = electrostatic_field_energy(phi, cache, params, vol_fac, use_dealias=False)
    Wphi_st, Wphi_kxst, Wphi_kyst, Wphi_kxkyst, Wphi_zst = (
        electrostatic_field_energy_resolved(
            phi, cache, params, vol_fac, use_dealias=False
        )
    )
    np.testing.assert_allclose(
        np.asarray(Wphi_st).sum(), np.asarray(Wphi), rtol=1.0e-6, atol=1.0e-6
    )
    for spectrum, axes in (
        (Wphi_kxst, 1),
        (Wphi_kyst, 1),
        (Wphi_kxkyst, (1, 2)),
        (Wphi_zst, 1),
    ):
        np.testing.assert_allclose(
            np.asarray(spectrum).sum(axis=axes),
            np.asarray(Wphi_st),
            rtol=1.0e-6,
            atol=1.0e-6,
        )

    Wapar = magnetic_vector_potential_energy(apar, cache, vol_fac, use_dealias=False)
    Wapar_st, Wapar_kxst, Wapar_kyst, Wapar_kxkyst, Wapar_zst = (
        magnetic_vector_potential_energy_resolved(
            apar, cache, vol_fac, nspecies=2, use_dealias=False
        )
    )
    np.testing.assert_allclose(
        np.asarray(Wapar_st).sum(), np.asarray(Wapar), rtol=1.0e-6, atol=1.0e-6
    )
    for spectrum, axes in (
        (Wapar_kxst, 1),
        (Wapar_kyst, 1),
        (Wapar_kxkyst, (1, 2)),
        (Wapar_zst, 1),
    ):
        np.testing.assert_allclose(
            np.asarray(spectrum).sum(axis=axes),
            np.asarray(Wapar_st),
            rtol=1.0e-6,
            atol=1.0e-6,
        )

    phi2_t, phi2_kxt, phi2_kyt, phi2_kxkyt, phi2_zt, *_zonal = phi2_resolved(
        phi, grid, vol_fac, use_dealias=False
    )
    for spectrum, axes in (
        (phi2_kxt, 0),
        (phi2_kyt, 0),
        (phi2_kxkyt, (0, 1)),
        (phi2_zt, 0),
    ):
        np.testing.assert_allclose(
            np.asarray(spectrum).sum(axis=axes),
            np.asarray(phi2_t),
            rtol=1.0e-6,
            atol=1.0e-6,
        )


def test_diagnostics_mask_dealiased_nonfinite_modes_before_reduction() -> None:
    _cfg, grid, _geom, params, cache, vol_fac, flux_fac = _multispecies_setup(
        Nl=2, Nm=2
    )
    shape = (2, 2, 2, grid.ky.size, grid.kx.size, grid.z.size)
    base = np.ones(shape, dtype=np.complex64)
    field_shape = (grid.ky.size, grid.kx.size, grid.z.size)
    phi_base = np.ones(field_shape, dtype=np.complex64) * (1.0 + 0.25j)
    apar_base = np.ones(field_shape, dtype=np.complex64) * (0.1 - 0.05j)
    bpar_base = np.ones(field_shape, dtype=np.complex64) * (0.02 + 0.03j)
    masked_indices = np.argwhere(~np.asarray(grid.dealias_mask, dtype=bool))
    assert masked_indices.size > 0
    ky_idx, kx_idx = masked_indices[0]
    contaminated = base.copy()
    contaminated[:, :, :, ky_idx, kx_idx, :] = np.inf + 0.0j
    clean = base.copy()
    clean[:, :, :, ky_idx, kx_idx, :] = 0.0
    phi_contaminated = phi_base.copy()
    apar_contaminated = apar_base.copy()
    bpar_contaminated = bpar_base.copy()
    phi_contaminated[ky_idx, kx_idx, :] = np.inf + 0.0j
    apar_contaminated[ky_idx, kx_idx, :] = np.inf + 0.0j
    bpar_contaminated[ky_idx, kx_idx, :] = np.inf + 0.0j
    phi_clean = phi_base.copy()
    apar_clean = apar_base.copy()
    bpar_clean = bpar_base.copy()
    phi_clean[ky_idx, kx_idx, :] = 0.0
    apar_clean[ky_idx, kx_idx, :] = 0.0
    bpar_clean[ky_idx, kx_idx, :] = 0.0

    wg = distribution_free_energy(
        jnp.asarray(contaminated), grid, params, vol_fac, use_dealias=True
    )
    wg_clean = distribution_free_energy(
        jnp.asarray(clean), grid, params, vol_fac, use_dealias=True
    )
    resolved = distribution_free_energy_resolved(
        jnp.asarray(contaminated), grid, params, vol_fac, use_dealias=True
    )
    resolved_clean = distribution_free_energy_resolved(
        jnp.asarray(clean), grid, params, vol_fac, use_dealias=True
    )

    assert np.isfinite(np.asarray(wg))
    np.testing.assert_allclose(np.asarray(wg), np.asarray(wg_clean))
    for got, expected in zip(resolved, resolved_clean, strict=True):
        assert np.all(np.isfinite(np.asarray(got)))
        np.testing.assert_allclose(np.asarray(got), np.asarray(expected))

    energy_pairs = [
        (
            electrostatic_field_energy(
                jnp.asarray(phi_contaminated), cache, params, vol_fac, use_dealias=True
            ),
            electrostatic_field_energy(
                jnp.asarray(phi_clean), cache, params, vol_fac, use_dealias=True
            ),
        ),
        (
            magnetic_vector_potential_energy(
                jnp.asarray(apar_contaminated), cache, vol_fac, use_dealias=True
            ),
            magnetic_vector_potential_energy(
                jnp.asarray(apar_clean), cache, vol_fac, use_dealias=True
            ),
        ),
    ]
    for got, expected in energy_pairs:
        assert np.isfinite(np.asarray(got))
        np.testing.assert_allclose(np.asarray(got), np.asarray(expected))

    resolved_pairs = [
        (
            electrostatic_field_energy_resolved(
                jnp.asarray(phi_contaminated),
                cache,
                params,
                vol_fac,
                use_dealias=True,
            ),
            electrostatic_field_energy_resolved(
                jnp.asarray(phi_clean), cache, params, vol_fac, use_dealias=True
            ),
        ),
        (
            magnetic_vector_potential_energy_resolved(
                jnp.asarray(apar_contaminated),
                cache,
                vol_fac,
                nspecies=2,
                use_dealias=True,
            ),
            magnetic_vector_potential_energy_resolved(
                jnp.asarray(apar_clean),
                cache,
                vol_fac,
                nspecies=2,
                use_dealias=True,
            ),
        ),
        (
            phi2_resolved(jnp.asarray(phi_contaminated), grid, vol_fac),
            phi2_resolved(jnp.asarray(phi_clean), grid, vol_fac),
        ),
        (
            heat_flux_resolved_species(
                jnp.asarray(contaminated),
                jnp.asarray(phi_contaminated),
                jnp.asarray(apar_contaminated),
                jnp.asarray(bpar_contaminated),
                cache,
                grid,
                params,
                flux_fac,
                use_dealias=True,
            ),
            heat_flux_resolved_species(
                jnp.asarray(clean),
                jnp.asarray(phi_clean),
                jnp.asarray(apar_clean),
                jnp.asarray(bpar_clean),
                cache,
                grid,
                params,
                flux_fac,
                use_dealias=True,
            ),
        ),
    ]
    for got_tuple, expected_tuple in resolved_pairs:
        for got, expected in zip(got_tuple, expected_tuple, strict=True):
            assert np.all(np.isfinite(np.asarray(got)))
            np.testing.assert_allclose(np.asarray(got), np.asarray(expected))

    channel_pairs = [
        (
            heat_flux_species(
                jnp.asarray(contaminated),
                jnp.asarray(phi_contaminated),
                jnp.asarray(apar_contaminated),
                jnp.asarray(bpar_contaminated),
                cache,
                grid,
                params,
                flux_fac,
                use_dealias=True,
            ),
            heat_flux_species(
                jnp.asarray(clean),
                jnp.asarray(phi_clean),
                jnp.asarray(apar_clean),
                jnp.asarray(bpar_clean),
                cache,
                grid,
                params,
                flux_fac,
                use_dealias=True,
            ),
        ),
        (
            particle_flux_species(
                jnp.asarray(contaminated),
                jnp.asarray(phi_contaminated),
                jnp.asarray(apar_contaminated),
                jnp.asarray(bpar_contaminated),
                cache,
                grid,
                params,
                flux_fac,
                use_dealias=True,
            ),
            particle_flux_species(
                jnp.asarray(clean),
                jnp.asarray(phi_clean),
                jnp.asarray(apar_clean),
                jnp.asarray(bpar_clean),
                cache,
                grid,
                params,
                flux_fac,
                use_dealias=True,
            ),
        ),
        (
            turbulent_heating_species(
                jnp.asarray(contaminated),
                0.9 * jnp.asarray(contaminated),
                jnp.asarray(phi_contaminated),
                jnp.asarray(apar_contaminated),
                jnp.asarray(bpar_contaminated),
                0.9 * jnp.asarray(phi_contaminated),
                0.9 * jnp.asarray(apar_contaminated),
                0.9 * jnp.asarray(bpar_contaminated),
                cache,
                grid,
                params,
                vol_fac,
                0.1,
                use_dealias=True,
            ),
            turbulent_heating_species(
                jnp.asarray(clean),
                0.9 * jnp.asarray(clean),
                jnp.asarray(phi_clean),
                jnp.asarray(apar_clean),
                jnp.asarray(bpar_clean),
                0.9 * jnp.asarray(phi_clean),
                0.9 * jnp.asarray(apar_clean),
                0.9 * jnp.asarray(bpar_clean),
                cache,
                grid,
                params,
                vol_fac,
                0.1,
                use_dealias=True,
            ),
        ),
    ]
    for got, expected in channel_pairs:
        assert np.all(np.isfinite(np.asarray(got)))
        np.testing.assert_allclose(np.asarray(got), np.asarray(expected))


def test_zero_field_state_has_zero_transport_and_heating() -> None:
    _cfg, grid, _geom, params, cache, vol_fac, flux_fac = _multispecies_setup(
        Nl=3, Nm=4
    )
    G = jnp.zeros(
        (2, 3, 4, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64
    )
    phi = jnp.zeros((grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64)
    apar = jnp.zeros_like(phi)
    bpar = jnp.zeros_like(phi)

    heat_species = heat_flux_species(
        G, phi, apar, bpar, cache, grid, params, flux_fac, use_dealias=False
    )
    heat_split = heat_flux_channel_species(
        G, phi, apar, bpar, cache, grid, params, flux_fac, use_dealias=False
    )
    particle_species = particle_flux_species(
        G, phi, apar, bpar, cache, grid, params, flux_fac, use_dealias=False
    )
    particle_split = particle_flux_channel_species(
        G, phi, apar, bpar, cache, grid, params, flux_fac, use_dealias=False
    )
    turbulent_heating_by_species = turbulent_heating_species(
        G,
        G,
        phi,
        apar,
        bpar,
        phi,
        apar,
        bpar,
        cache,
        grid,
        params,
        vol_fac,
        0.125,
        use_dealias=False,
    )

    np.testing.assert_allclose(np.asarray(heat_species), 0.0)
    np.testing.assert_allclose(
        np.asarray(heat_flux_total(G, phi, apar, bpar, cache, grid, params, flux_fac)),
        0.0,
    )
    for channel in heat_split:
        np.testing.assert_allclose(np.asarray(channel), 0.0)
    np.testing.assert_allclose(np.asarray(particle_species), 0.0)
    np.testing.assert_allclose(
        np.asarray(
            particle_flux_total(G, phi, apar, bpar, cache, grid, params, flux_fac)
        ),
        0.0,
    )
    for channel in particle_split:
        np.testing.assert_allclose(np.asarray(channel), 0.0)
    np.testing.assert_allclose(np.asarray(turbulent_heating_by_species), 0.0)
    np.testing.assert_allclose(
        np.asarray(
            turbulent_heating_total(
                G,
                G,
                phi,
                apar,
                bpar,
                phi,
                apar,
                bpar,
                cache,
                grid,
                params,
                vol_fac,
                0.125,
                use_dealias=False,
            )
        ),
        0.0,
    )


def test_energy_components_finite():
    cfg, grid, geom, params, cache = _small_setup()
    vol_fac, flux_fac = fieldline_quadrature_weights(geom, grid)
    G0 = _build_initial_condition(
        grid, geom, ky_index=0, kx_index=0, Nl=4, Nm=4, init_cfg=cfg.init
    )
    _, fields = cache, None
    # Build dummy fields from RHS for consistent shapes
    from gkx.terms.assembly import assemble_rhs_cached

    _dG, fields = assemble_rhs_cached(G0, cache, params, terms=LinearTerms())
    phi = fields.phi
    apar = fields.apar if fields.apar is not None else jnp.zeros_like(phi)
    bpar = fields.bpar if fields.bpar is not None else jnp.zeros_like(phi)

    Wg = distribution_free_energy(G0, grid, params, vol_fac)
    Wphi = electrostatic_field_energy(phi, cache, params, vol_fac)
    Wapar = magnetic_vector_potential_energy(apar, cache, vol_fac)
    heat = heat_flux_total(G0, phi, apar, bpar, cache, grid, params, flux_fac)
    pflux = particle_flux_total(G0, phi, apar, bpar, cache, grid, params, flux_fac)
    energy = total_energy(Wg, Wphi, Wapar)

    assert np.isfinite(np.asarray(Wg))
    assert np.isfinite(np.asarray(Wphi))
    assert np.isfinite(np.asarray(Wapar))
    assert np.isfinite(np.asarray(heat))
    assert np.isfinite(np.asarray(pflux))
    assert np.isfinite(np.asarray(energy))
    assert energy == Wg + Wphi + Wapar


def test_fieldline_quadrature_weights_accept_sampled_geometry_contract():
    _cfg, grid, geom, _params, _cache = _small_setup()
    sampled = sample_flux_tube_geometry(geom, grid.z)

    vol_ref, flux_ref = fieldline_quadrature_weights(geom, grid)
    vol_s, flux_s = fieldline_quadrature_weights(sampled, grid)

    assert np.allclose(np.asarray(vol_s), np.asarray(vol_ref))
    assert np.allclose(np.asarray(flux_s), np.asarray(flux_ref))


def test_fieldline_quadrature_weights_trim_closed_sampled_geometry_contract():
    _cfg, grid, geom, _params, _cache = _small_setup()
    sampled = sample_flux_tube_geometry(geom, grid.z)
    closed = replace(
        sampled,
        theta=jnp.concatenate([sampled.theta, jnp.asarray([jnp.pi])]),
        bmag_profile=jnp.concatenate([sampled.bmag_profile, sampled.bmag_profile[:1]]),
        bgrad_profile=jnp.concatenate(
            [sampled.bgrad_profile, sampled.bgrad_profile[:1]]
        ),
        gds2_profile=jnp.concatenate([sampled.gds2_profile, sampled.gds2_profile[:1]]),
        gds21_profile=jnp.concatenate(
            [sampled.gds21_profile, sampled.gds21_profile[:1]]
        ),
        gds22_profile=jnp.concatenate(
            [sampled.gds22_profile, sampled.gds22_profile[:1]]
        ),
        cv_profile=jnp.concatenate([sampled.cv_profile, sampled.cv_profile[:1]]),
        gb_profile=jnp.concatenate([sampled.gb_profile, sampled.gb_profile[:1]]),
        cv0_profile=jnp.concatenate([sampled.cv0_profile, sampled.cv0_profile[:1]]),
        gb0_profile=jnp.concatenate([sampled.gb0_profile, sampled.gb0_profile[:1]]),
        jacobian_profile=jnp.concatenate(
            [sampled.jacobian_profile, sampled.jacobian_profile[:1]]
        ),
        grho_profile=jnp.concatenate([sampled.grho_profile, sampled.grho_profile[:1]]),
        theta_closed_interval=True,
    )

    vol_ref, flux_ref = fieldline_quadrature_weights(sampled, grid)
    vol_closed, flux_closed = fieldline_quadrature_weights(closed, grid)

    assert np.allclose(np.asarray(vol_closed), np.asarray(vol_ref))
    assert np.allclose(np.asarray(flux_closed), np.asarray(flux_ref))


def test_zonal_phi_mode_kxt_recovers_signed_zonal_average() -> None:
    cfg = CycloneBaseCase()
    grid = build_spectral_grid(replace(cfg.grid, Ny=8, Nx=4))
    geom = SAlphaGeometry.from_config(cfg.geometry)
    vol_fac, _flux_fac = fieldline_quadrature_weights(geom, grid)
    phi = jnp.zeros((grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64)
    zonal_profile = (0.3 - 0.1j) + (0.2 + 0.05j) * jnp.cos(grid.z)
    phi = phi.at[0, 1, :].set(zonal_profile)

    out = zonal_phi_mode_kxt(phi, grid, vol_fac)
    out_line = zonal_phi_line_kxt(phi, grid)

    expected = jnp.sum(zonal_profile * vol_fac)
    expected_line = jnp.mean(zonal_profile)
    assert np.allclose(np.asarray(out[1]), np.asarray(expected))
    assert np.allclose(np.asarray(out_line[1]), np.asarray(expected_line))
    assert np.allclose(np.asarray(out[0]), 0.0)
    assert np.allclose(np.asarray(out_line[0]), 0.0)


def test_fieldline_quadrature_weights_use_grho_for_flux_weights():
    _cfg, grid, geom, _params, _cache = _small_setup()
    sampled = sample_flux_tube_geometry(geom, grid.z)
    sampled = replace(
        sampled,
        jacobian_profile=jnp.asarray([1.0, 2.0, 3.0, 4.0]),
        grho_profile=jnp.asarray([1.0, 2.0, 1.0, 2.0]),
        theta=jnp.asarray(grid.z[:4]),
    )
    grid_small = replace(grid, z=grid.z[:4])

    vol_fac, flux_fac = fieldline_quadrature_weights(sampled, grid_small)

    jac = np.array([1.0, 2.0, 3.0, 4.0])
    grho = np.array([1.0, 2.0, 1.0, 2.0])
    assert np.allclose(np.asarray(vol_fac), jac / np.sum(jac))
    assert np.allclose(np.asarray(flux_fac), jac / np.sum(jac * grho))


def test_standard_field_energies_match_geometry_weighted_formula():
    _cfg, grid, geom, params, cache = _small_setup()
    vol_fac, _flux_fac = fieldline_quadrature_weights(geom, grid)
    phi = jnp.ones((grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64)
    apar = (1.0 + 2.0j) * jnp.ones_like(phi)

    fac = np.where(np.asarray(grid.ky, dtype=float)[:, None] == 0.0, 1.0, 2.0)
    fac = fac * np.asarray(grid.dealias_mask, dtype=float)
    weight = fac[:, :, None] * np.asarray(vol_fac, dtype=float)[None, None, :]
    phi2 = np.abs(np.asarray(phi)) ** 2
    apar2 = np.abs(np.asarray(apar)) ** 2

    rho = np.atleast_1d(np.asarray(params.rho, dtype=float))
    wphi_expected = 0.0
    for rho_s in rho:
        b = np.asarray(cache.kperp2, dtype=float) * (rho_s * rho_s)
        wphi_expected += 0.5 * np.sum(phi2 * (1.0 - np.asarray(gamma0(b))) * weight)

    bmag2 = (
        np.asarray(cache.bmag, dtype=float)[None, None, :] ** 2
        if cache.kperp2_bmag
        else 1.0
    )
    wapar_expected = 0.5 * np.sum(
        apar2 * np.asarray(cache.kperp2, dtype=float) * bmag2 * weight
    )

    assert np.allclose(
        np.asarray(electrostatic_field_energy(phi, cache, params, vol_fac)),
        wphi_expected,
    )
    assert np.allclose(
        np.asarray(magnetic_vector_potential_energy(apar, cache, vol_fac)),
        wapar_expected,
    )


def test_reduce_scalar_and_species_kykxz_preserve_manual_sums() -> None:
    scalar = jnp.asarray(np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4))
    scalar_reduced = _reduce_scalar_kykxz(scalar)
    np.testing.assert_allclose(
        np.asarray(scalar_reduced[0]), np.asarray(scalar).sum(axis=(0, 2))
    )
    np.testing.assert_allclose(
        np.asarray(scalar_reduced[1]), np.asarray(scalar).sum(axis=(1, 2))
    )
    np.testing.assert_allclose(
        np.asarray(scalar_reduced[2]), np.asarray(scalar).sum(axis=2)
    )
    np.testing.assert_allclose(
        np.asarray(scalar_reduced[3]), np.asarray(scalar).sum(axis=(0, 1))
    )
    np.testing.assert_allclose(np.asarray(scalar_reduced[4]), np.asarray(scalar).sum())

    species = jnp.asarray(
        np.arange(2 * 2 * 3 * 4, dtype=np.float32).reshape(2, 2, 3, 4)
    )
    species_reduced = _reduce_species_kykxz(species)
    np.testing.assert_allclose(
        np.asarray(species_reduced[0]), np.asarray(species).sum(axis=(1, 2, 3))
    )
    np.testing.assert_allclose(
        np.asarray(species_reduced[1]), np.asarray(species).sum(axis=(1, 3))
    )
    np.testing.assert_allclose(
        np.asarray(species_reduced[2]), np.asarray(species).sum(axis=(2, 3))
    )
    np.testing.assert_allclose(
        np.asarray(species_reduced[3]), np.asarray(species).sum(axis=3)
    )
    np.testing.assert_allclose(
        np.asarray(species_reduced[4]), np.asarray(species).sum(axis=(1, 2))
    )


def test_species_flux_sums_to_total():
    cfg, grid, geom, params, cache = _small_setup()
    _vol_fac, flux_fac = fieldline_quadrature_weights(geom, grid)
    G0 = _build_initial_condition(
        grid, geom, ky_index=0, kx_index=0, Nl=4, Nm=4, init_cfg=cfg.init
    )
    from gkx.terms.assembly import assemble_rhs_cached

    _dG, fields = assemble_rhs_cached(G0, cache, params, terms=LinearTerms())
    phi = fields.phi
    apar = fields.apar if fields.apar is not None else jnp.zeros_like(phi)
    bpar = fields.bpar if fields.bpar is not None else jnp.zeros_like(phi)

    heat_s = heat_flux_species(G0, phi, apar, bpar, cache, grid, params, flux_fac)
    pflux_s = particle_flux_species(G0, phi, apar, bpar, cache, grid, params, flux_fac)
    heat = heat_flux_total(G0, phi, apar, bpar, cache, grid, params, flux_fac)
    pflux = particle_flux_total(G0, phi, apar, bpar, cache, grid, params, flux_fac)

    assert heat_s.shape == (1,)
    assert pflux_s.shape == (1,)
    assert np.allclose(np.asarray(jnp.sum(heat_s)), np.asarray(heat))
    assert np.allclose(np.asarray(jnp.sum(pflux_s)), np.asarray(pflux))


def test_heat_flux_total_channel_helper_matches_public_split_reductions() -> None:
    _cfg, grid, _geom, params, cache, _vol_fac, flux_fac = _multispecies_setup(
        Nl=3, Nm=4
    )
    shape = (2, 3, 4, grid.ky.size, grid.kx.size, grid.z.size)
    base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    G = jnp.asarray(base + 1.0j * (base + 1.0), dtype=jnp.complex64)
    field_base = np.arange(
        grid.ky.size * grid.kx.size * grid.z.size, dtype=np.float32
    ).reshape(grid.ky.size, grid.kx.size, grid.z.size)
    phi = jnp.asarray(field_base + 1.0j * (field_base + 1.0), dtype=jnp.complex64)
    apar = 0.3 * phi
    bpar = -0.2 * phi

    es_contrib, apar_contrib, bpar_contrib = _heat_flux_channel_contrib_species(
        G,
        phi,
        apar,
        bpar,
        cache,
        grid,
        params,
        flux_fac,
        use_dealias=False,
        flux_scale=1.0,
    )
    es_resolved, apar_resolved, bpar_resolved = heat_flux_channel_resolved_species(
        G,
        phi,
        apar,
        bpar,
        cache,
        grid,
        params,
        flux_fac,
        use_dealias=False,
    )

    for contrib, reduced in (
        (es_contrib, es_resolved),
        (apar_contrib, apar_resolved),
        (bpar_contrib, bpar_resolved),
    ):
        expected = _reduce_species_kykxz(contrib)
        for got_arr, expected_arr in zip(reduced, expected, strict=True):
            np.testing.assert_allclose(
                np.asarray(got_arr), np.asarray(expected_arr), rtol=1.0e-6, atol=1.0e-6
            )


def test_particle_flux_total_channel_helper_matches_public_split_reductions() -> None:
    _cfg, grid, _geom, params, cache, _vol_fac, flux_fac = _multispecies_setup(
        Nl=3, Nm=3
    )
    shape = (2, 3, 3, grid.ky.size, grid.kx.size, grid.z.size)
    base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    G = jnp.asarray(base + 1.0j * (base + 0.25), dtype=jnp.complex64)
    field_base = np.arange(
        grid.ky.size * grid.kx.size * grid.z.size, dtype=np.float32
    ).reshape(grid.ky.size, grid.kx.size, grid.z.size)
    phi = jnp.asarray(field_base + 1.0j * (field_base + 1.0), dtype=jnp.complex64)
    apar = 0.1 * phi
    bpar = -0.3 * phi

    es_contrib, apar_contrib, bpar_contrib = _particle_flux_channel_contrib_species(
        G,
        phi,
        apar,
        bpar,
        cache,
        grid,
        params,
        flux_fac,
        use_dealias=False,
        flux_scale=1.0,
    )
    es_resolved, apar_resolved, bpar_resolved = particle_flux_channel_resolved_species(
        G,
        phi,
        apar,
        bpar,
        cache,
        grid,
        params,
        flux_fac,
        use_dealias=False,
    )

    for contrib, reduced in (
        (es_contrib, es_resolved),
        (apar_contrib, apar_resolved),
        (bpar_contrib, bpar_resolved),
    ):
        expected = _reduce_species_kykxz(contrib)
        for got_arr, expected_arr in zip(reduced, expected, strict=True):
            np.testing.assert_allclose(
                np.asarray(got_arr), np.asarray(expected_arr), rtol=1.0e-6, atol=1.0e-6
            )


def test_particle_flux_total_channel_helper_single_species_short_circuits_to_zero() -> (
    None
):
    cfg, grid, geom, params, cache = _small_setup()
    _vol_fac, flux_fac = fieldline_quadrature_weights(geom, grid)
    G = _build_initial_condition(
        grid, geom, ky_index=0, kx_index=0, Nl=4, Nm=4, init_cfg=cfg.init
    )
    _dG, fields = assemble_rhs_cached(G, cache, params, terms=LinearTerms())
    phi = fields.phi
    apar = fields.apar if fields.apar is not None else jnp.zeros_like(phi)
    bpar = fields.bpar if fields.bpar is not None else jnp.zeros_like(phi)

    es_contrib, apar_contrib, bpar_contrib = _particle_flux_channel_contrib_species(
        G,
        phi,
        apar,
        bpar,
        cache,
        grid,
        params,
        flux_fac,
        use_dealias=False,
        flux_scale=1.0,
    )

    np.testing.assert_allclose(np.asarray(es_contrib), 0.0)
    np.testing.assert_allclose(np.asarray(apar_contrib), 0.0)
    np.testing.assert_allclose(np.asarray(bpar_contrib), 0.0)


def test_jl_family_preserves_species_axis() -> None:
    cfg = CycloneBaseCase()
    grid = build_spectral_grid(
        replace(cfg.grid, Nx=4, Ny=8, Nz=8, ntheta=None, nperiod=None)
    )
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = build_linear_params(
        [
            Species(
                charge=1.0, mass=1.0, density=1.0, temperature=1.0, tprim=1.0, fprim=1.0
            ),
            Species(
                charge=-1.0,
                mass=0.00027,
                density=1.0,
                temperature=1.0,
                tprim=1.0,
                fprim=1.0,
            ),
        ],
        kpar_scale=float(geom.gradpar()),
    )
    cache = build_linear_cache(grid, geom, params, 3, 3)
    Jl, JlB, Jfac = _jl_family(cache)

    assert np.asarray(Jl).shape == np.asarray(cache.Jl).shape
    assert np.asarray(JlB).shape == np.asarray(cache.JlB).shape
    assert np.asarray(Jfac).shape == np.asarray(cache.Jl).shape
    assert np.allclose(np.asarray(Jl), np.asarray(cache.Jl))
    assert np.allclose(np.asarray(JlB), np.asarray(cache.JlB))


def test_particle_flux_species_matches_manual_multispecies_formula() -> None:
    cfg = CycloneBaseCase()
    grid = build_spectral_grid(
        replace(cfg.grid, Nx=4, Ny=8, Nz=8, ntheta=None, nperiod=None)
    )
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = build_linear_params(
        [
            Species(
                charge=1.0, mass=1.0, density=1.0, temperature=1.0, tprim=1.0, fprim=1.0
            ),
            Species(
                charge=-1.0,
                mass=0.00027,
                density=1.0,
                temperature=1.0,
                tprim=1.0,
                fprim=1.0,
            ),
        ],
        kpar_scale=float(geom.gradpar()),
    )
    cache = build_linear_cache(grid, geom, params, 3, 3)
    _vol_fac, flux_fac = fieldline_quadrature_weights(geom, grid)

    shape = (2, 3, 3, grid.ky.size, grid.kx.size, grid.z.size)
    base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    G = jnp.asarray(base + 1.0j * (base + 1.0), dtype=jnp.complex64)
    field_base = np.arange(
        grid.ky.size * grid.kx.size * grid.z.size, dtype=np.float32
    ).reshape(grid.ky.size, grid.kx.size, grid.z.size)
    phi = jnp.asarray(field_base + 1.0j * (field_base + 1.0), dtype=jnp.complex64)
    apar = 0.3 * phi
    bpar = -0.2 * phi

    got = np.asarray(
        particle_flux_species(
            G,
            phi,
            apar,
            bpar,
            cache,
            grid,
            params,
            flux_fac,
            use_dealias=False,
        )
    )

    fac = np.asarray(_transport_mode_weight(grid, use_dealias=False), dtype=np.float32)[
        :, :, None
    ]
    flx = np.asarray(flux_fac, dtype=np.float32)[None, None, :]
    ky = np.asarray(grid.ky, dtype=np.float32)[:, None, None]
    vphi = 1.0j * ky * np.asarray(phi)
    vapar = 1.0j * ky * np.asarray(apar)
    vbpar = 1.0j * ky * np.asarray(bpar)
    Jl = np.asarray(cache.Jl)
    JlB = np.asarray(cache.JlB)
    dens = np.asarray(params.density, dtype=np.float32)
    vth = np.asarray(params.vth, dtype=np.float32)
    tz = np.asarray(params.tz, dtype=np.float32)
    G_np = np.asarray(G)

    expected = []
    for s in range(2):
        G0 = G_np[s, :, 0, ...]
        G1 = G_np[s, :, 1, ...]
        n_bar = np.sum(Jl[s] * G0, axis=0)
        u_bar = np.sum(Jl[s] * G1, axis=0)
        uB_bar = np.sum(JlB[s] * G0, axis=0)
        fg = (
            np.conj(vphi) * n_bar
            - vth[s] * np.conj(vapar) * u_bar
            + tz[s] * np.conj(vbpar) * uB_bar
        )
        expected.append(np.sum((fg * 2.0 * flx * fac).real) * dens[s])

    assert np.allclose(got, np.asarray(expected), rtol=1.0e-6, atol=1.0e-6)


def test_flux_channel_splits_sum_to_total_multispecies() -> None:
    cfg = CycloneBaseCase()
    grid = build_spectral_grid(
        replace(cfg.grid, Nx=4, Ny=8, Nz=8, ntheta=None, nperiod=None)
    )
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = build_linear_params(
        [
            Species(
                charge=1.0, mass=1.0, density=1.0, temperature=1.0, tprim=1.0, fprim=1.0
            ),
            Species(
                charge=-1.0,
                mass=0.00027,
                density=1.0,
                temperature=1.0,
                tprim=1.0,
                fprim=1.0,
            ),
        ],
        kpar_scale=float(geom.gradpar()),
    )
    cache = build_linear_cache(grid, geom, params, 3, 4)
    _vol_fac, flux_fac = fieldline_quadrature_weights(geom, grid)

    shape = (2, 3, 4, grid.ky.size, grid.kx.size, grid.z.size)
    base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    G = jnp.asarray(base + 1.0j * (base + 1.0), dtype=jnp.complex64)
    field_base = np.arange(
        grid.ky.size * grid.kx.size * grid.z.size, dtype=np.float32
    ).reshape(grid.ky.size, grid.kx.size, grid.z.size)
    phi = jnp.asarray(field_base + 1.0j * (field_base + 1.0), dtype=jnp.complex64)
    apar = 0.3 * phi
    bpar = -0.2 * phi

    heat = np.asarray(
        heat_flux_species(
            G, phi, apar, bpar, cache, grid, params, flux_fac, use_dealias=False
        )
    )
    heat_es, heat_apar, heat_bpar = (
        np.asarray(arr)
        for arr in heat_flux_channel_species(
            G,
            phi,
            apar,
            bpar,
            cache,
            grid,
            params,
            flux_fac,
            use_dealias=False,
        )
    )
    pflux = np.asarray(
        particle_flux_species(
            G, phi, apar, bpar, cache, grid, params, flux_fac, use_dealias=False
        )
    )
    pflux_es, pflux_apar, pflux_bpar = (
        np.asarray(arr)
        for arr in particle_flux_channel_species(
            G,
            phi,
            apar,
            bpar,
            cache,
            grid,
            params,
            flux_fac,
            use_dealias=False,
        )
    )

    np.testing.assert_allclose(
        heat, heat_es + heat_apar + heat_bpar, rtol=1.0e-6, atol=1.0e-6
    )
    np.testing.assert_allclose(
        pflux, pflux_es + pflux_apar + pflux_bpar, rtol=1.0e-6, atol=1.0e-6
    )

    heat_total = heat_flux_resolved_species(
        G, phi, apar, bpar, cache, grid, params, flux_fac, use_dealias=False
    )
    heat_split = heat_flux_channel_resolved_species(
        G, phi, apar, bpar, cache, grid, params, flux_fac, use_dealias=False
    )
    pflux_total = particle_flux_resolved_species(
        G, phi, apar, bpar, cache, grid, params, flux_fac, use_dealias=False
    )
    pflux_split = particle_flux_channel_resolved_species(
        G, phi, apar, bpar, cache, grid, params, flux_fac, use_dealias=False
    )

    for total_arr, split_arrs in ((heat_total, heat_split), (pflux_total, pflux_split)):
        for idx in range(len(total_arr)):
            combined = (
                np.asarray(split_arrs[0][idx])
                + np.asarray(split_arrs[1][idx])
                + np.asarray(split_arrs[2][idx])
            )
            np.testing.assert_allclose(
                np.asarray(total_arr[idx]), combined, rtol=1.0e-6, atol=1.0e-6
            )


def test_turbulent_heating_total_zero_for_steady_state() -> None:
    cfg = CycloneBaseCase()
    grid = build_spectral_grid(
        replace(cfg.grid, Nx=4, Ny=8, Nz=8, ntheta=None, nperiod=None)
    )
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = build_linear_params(
        [
            Species(
                charge=1.0, mass=1.0, density=1.0, temperature=1.0, tprim=1.0, fprim=1.0
            ),
            Species(
                charge=-1.0,
                mass=0.00027,
                density=1.0,
                temperature=1.0,
                tprim=1.0,
                fprim=1.0,
            ),
        ],
        kpar_scale=float(geom.gradpar()),
    )
    cache = build_linear_cache(grid, geom, params, 3, 4)
    vol_fac, _flux_fac = fieldline_quadrature_weights(geom, grid)

    shape = (2, 3, 4, grid.ky.size, grid.kx.size, grid.z.size)
    base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    G = jnp.asarray(base + 1.0j * (base + 0.5), dtype=jnp.complex64)
    field_base = np.arange(
        grid.ky.size * grid.kx.size * grid.z.size, dtype=np.float32
    ).reshape(grid.ky.size, grid.kx.size, grid.z.size)
    phi = jnp.asarray(field_base + 1.0j * (field_base + 1.0), dtype=jnp.complex64)
    apar = 0.2 * phi
    bpar = -0.1 * phi

    heat_species = turbulent_heating_species(
        G,
        G,
        phi,
        apar,
        bpar,
        phi,
        apar,
        bpar,
        cache,
        grid,
        params,
        vol_fac,
        0.05,
        use_dealias=False,
    )
    heat_total = turbulent_heating_total(
        G,
        G,
        phi,
        apar,
        bpar,
        phi,
        apar,
        bpar,
        cache,
        grid,
        params,
        vol_fac,
        0.05,
        use_dealias=False,
    )

    np.testing.assert_allclose(np.asarray(heat_species), 0.0, atol=1.0e-7)
    np.testing.assert_allclose(np.asarray(heat_total), 0.0, atol=1.0e-7)


def test_turbulent_heating_total_resolved_sums_to_species_total() -> None:
    cfg = CycloneBaseCase()
    grid = build_spectral_grid(
        replace(cfg.grid, Nx=4, Ny=8, Nz=8, ntheta=None, nperiod=None)
    )
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = build_linear_params(
        [
            Species(
                charge=1.0, mass=1.0, density=1.0, temperature=1.0, tprim=1.0, fprim=1.0
            ),
            Species(
                charge=-1.0,
                mass=0.00027,
                density=1.0,
                temperature=1.0,
                tprim=1.0,
                fprim=1.0,
            ),
        ],
        kpar_scale=float(geom.gradpar()),
    )
    cache = build_linear_cache(grid, geom, params, 3, 4)
    vol_fac, _flux_fac = fieldline_quadrature_weights(geom, grid)

    shape = (2, 3, 4, grid.ky.size, grid.kx.size, grid.z.size)
    base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    G_old = jnp.asarray(base + 1.0j * (base + 0.5), dtype=jnp.complex64)
    G = 1.03 * G_old + (0.02 - 0.01j)
    field_base = np.arange(
        grid.ky.size * grid.kx.size * grid.z.size, dtype=np.float32
    ).reshape(grid.ky.size, grid.kx.size, grid.z.size)
    phi_old = jnp.asarray(field_base + 1.0j * (field_base + 1.0), dtype=jnp.complex64)
    phi = 1.01 * phi_old + (0.03 + 0.02j)
    apar_old = 0.2 * phi_old
    apar = 0.2 * phi
    bpar_old = -0.1 * phi_old
    bpar = -0.1 * phi + (0.01 - 0.02j)

    heat_species = turbulent_heating_species(
        G,
        G_old,
        phi,
        apar,
        bpar,
        phi_old,
        apar_old,
        bpar_old,
        cache,
        grid,
        params,
        vol_fac,
        0.05,
        use_dealias=False,
    )
    heat_st, heat_kxst, heat_kyst, heat_kxkyst, heat_zst = (
        turbulent_heating_resolved_species(
            G,
            G_old,
            phi,
            apar,
            bpar,
            phi_old,
            apar_old,
            bpar_old,
            cache,
            grid,
            params,
            vol_fac,
            0.05,
            use_dealias=False,
        )
    )

    np.testing.assert_allclose(
        np.asarray(heat_st), np.asarray(heat_species), rtol=1.0e-5, atol=1.0e-6
    )
    np.testing.assert_allclose(
        np.asarray(heat_kxst).sum(axis=1),
        np.asarray(heat_species),
        rtol=1.0e-5,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(heat_kyst).sum(axis=1),
        np.asarray(heat_species),
        rtol=1.0e-5,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(heat_kxkyst).sum(axis=(1, 2)),
        np.asarray(heat_species),
        rtol=1.0e-5,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(heat_zst).sum(axis=1),
        np.asarray(heat_species),
        rtol=1.0e-5,
        atol=1.0e-6,
    )
    assert np.max(np.abs(np.asarray(heat_species))) > 0.0


def test_turbulent_heating_total_helper_zero_dt_guard_returns_zero_for_changed_state() -> (
    None
):
    _cfg, grid, _geom, params, cache, vol_fac, _flux_fac = _multispecies_setup(
        Nl=3, Nm=4
    )
    shape = (2, 3, 4, grid.ky.size, grid.kx.size, grid.z.size)
    base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    G_old = jnp.asarray(base + 1.0j * (base + 0.5), dtype=jnp.complex64)
    G = 1.02 * G_old + (0.03 - 0.01j)
    field_base = np.arange(
        grid.ky.size * grid.kx.size * grid.z.size, dtype=np.float32
    ).reshape(grid.ky.size, grid.kx.size, grid.z.size)
    phi_old = jnp.asarray(field_base + 1.0j * (field_base + 1.0), dtype=jnp.complex64)
    phi = 1.01 * phi_old + (0.02 + 0.01j)
    apar_old = 0.2 * phi_old
    apar = 0.2 * phi
    bpar_old = -0.1 * phi_old
    bpar = -0.1 * phi + (0.01 - 0.02j)

    contrib = _turbulent_heating_contrib_species(
        G,
        G_old,
        phi,
        apar,
        bpar,
        phi_old,
        apar_old,
        bpar_old,
        cache,
        grid,
        params,
        vol_fac,
        0.0,
        use_dealias=False,
    )

    assert np.all(np.isfinite(np.asarray(contrib)))
    np.testing.assert_allclose(np.asarray(contrib), 0.0, atol=1.0e-7)


def test_init_all_scaling_matches_reference():
    cfg = CycloneBaseCase()
    grid_full = build_spectral_grid(cfg.grid)
    ky_index = select_ky_index(np.asarray(grid_full.ky), float(grid_full.ky[1]))
    grid = select_ky_grid(grid_full, ky_index)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    init_cfg = InitializationConfig(
        init_field="all",
        init_amp=1.0,
        gaussian_init=False,
    )
    G0 = _build_initial_condition(
        grid, geom, ky_index=0, kx_index=0, Nl=4, Nm=4, init_cfg=init_cfg
    )
    base = 1.0 + 1.0j
    # density (l=0,m=0) should be unscaled
    assert np.allclose(G0[0, 0, 0, 0, 0], base)
    # tpar (l=0,m=2) scaled by 1/sqrt(2)
    assert np.allclose(G0[0, 2, 0, 0, 0], base / np.sqrt(2.0))
    # qpar (l=0,m=3) scaled by 1/sqrt(6)
    assert np.allclose(G0[0, 3, 0, 0, 0], base / np.sqrt(6.0))


def test_integrate_linear_explicit_diagnostics_shapes():
    cfg, grid, geom, params, cache = _small_setup()
    G0 = _build_initial_condition(
        grid, geom, ky_index=0, kx_index=0, Nl=4, Nm=4, init_cfg=cfg.init
    )
    time_cfg = ExplicitTimeConfig(dt=0.01, t_max=0.1, sample_stride=1, fixed_dt=True)

    t, phi_t, gamma_t, omega_t, diag = integrate_linear_explicit_diagnostics(
        G0,
        grid,
        cache,
        params,
        geom,
        time_cfg,
        terms=LinearTerms(),
        jit=False,
    )
    assert t.shape[0] == phi_t.shape[0] == gamma_t.shape[0] == omega_t.shape[0]
    assert diag.t.shape[0] == t.shape[0]
    assert diag.Wg_t.shape[0] == t.shape[0]
    assert diag.Wphi_t.shape[0] == t.shape[0]
    assert diag.Wapar_t.shape[0] == t.shape[0]
    assert diag.heat_flux_t.shape[0] == t.shape[0]
    assert diag.particle_flux_t.shape[0] == t.shape[0]


def test_integrate_linear_explicit_diagnostics_honors_rk3_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg, grid, geom, params, cache = _small_setup()
    G0 = _build_initial_condition(
        grid, geom, ky_index=0, kx_index=0, Nl=4, Nm=4, init_cfg=cfg.init
    )
    calls: list[str] = []

    def _fake_step(G, cache, params, term_cfg, dt, *, method):
        calls.append(str(method))
        _dG, fields = assemble_rhs_cached(G, cache, params, terms=term_cfg)
        return G, fields

    monkeypatch.setattr(explicit_time_integrators, "_linear_explicit_step", _fake_step)

    time_cfg = ExplicitTimeConfig(
        dt=0.01, t_max=0.01, method="rk3", sample_stride=1, fixed_dt=True
    )
    integrate_linear_explicit_diagnostics(
        G0,
        grid,
        cache,
        params,
        geom,
        time_cfg,
        terms=LinearTerms(),
        jit=False,
    )

    assert calls
    assert set(calls) == {"rk3"}


def test_term_config_and_rk3_wrapper_delegate_to_linear_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terms = LinearTerms(apar=0.0, bpar=1.0, hyperdiffusion=1.0)
    assert _linear_term_config(terms) == linear_terms_to_term_config(terms)

    captured: dict[str, object] = {}

    def _fake_step(G, cache, params, term_cfg, dt, *, method):
        captured["G"] = G
        captured["cache"] = cache
        captured["params"] = params
        captured["term_cfg"] = term_cfg
        captured["dt"] = dt
        captured["method"] = method
        return G, FieldState(phi=G[0, 0])

    monkeypatch.setattr(explicit_time_integrators, "_linear_explicit_step", _fake_step)

    G0 = jnp.ones((1, 1, 1, 1, 1), dtype=jnp.complex64)
    cache = SimpleNamespace()
    params = object()
    term_cfg = linear_terms_to_term_config(terms)

    G_next, fields = _rk3_heun_step(G0, cache, params, term_cfg, 0.125)

    assert captured["G"] is G0
    assert captured["cache"] is cache
    assert captured["params"] is params
    assert captured["term_cfg"] is term_cfg
    assert captured["dt"] == pytest.approx(0.125)
    assert captured["method"] == "rk3"
    assert G_next is G0
    np.testing.assert_allclose(np.asarray(fields.phi), np.asarray(G0[0, 0]))


def test_linear_explicit_step_applies_completed_step_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = SimpleNamespace(
        dealias_mask=jnp.asarray([[True, True], [True, False]]),
        ky=jnp.asarray([0.0, 0.2]),
        kx=jnp.asarray([0.0, 0.1]),
    )
    G0 = jnp.ones((1, 1, 2, 2, 1), dtype=jnp.complex64)
    seen_states: list[np.ndarray] = []

    def _fake_assemble_rhs(state, _cache, _params, *, terms, dt=None):
        seen_states.append(np.asarray(state))
        return jnp.zeros_like(state), FieldState(phi=state[0, 0])

    monkeypatch.setattr(
        explicit_time_integrators, "assemble_rhs_cached", _fake_assemble_rhs
    )

    G_next, fields = explicit_time_integrators._linear_explicit_step(
        G0,
        cache,
        object(),
        linear_terms_to_term_config(LinearTerms()),
        0.1,
        method="euler",
    )

    expected = np.ones((2, 2, 1), dtype=np.complex64)
    expected[0, 0, 0] = 0.0
    expected[1, 1, 0] = 0.0

    assert len(seen_states) == 2
    assert np.allclose(seen_states[0][0, 0], 1.0)
    assert np.allclose(np.asarray(G_next[0, 0]), expected)
    assert np.allclose(np.asarray(seen_states[-1][0, 0]), expected)
    assert np.allclose(np.asarray(fields.phi), expected)


def test_energy_drift_small_no_drive():
    cfg, grid, geom, params, cache = _small_setup()
    params = replace(params, fprim=0.0, tprim=0.0, tprim_e=0.0, nu=0.0)
    G0 = _build_initial_condition(
        grid, geom, ky_index=0, kx_index=0, Nl=4, Nm=4, init_cfg=cfg.init
    )
    time_cfg = ExplicitTimeConfig(dt=0.01, t_max=0.2, sample_stride=1, fixed_dt=True)

    _, _, _, _, diag = integrate_linear_explicit_diagnostics(
        G0,
        grid,
        cache,
        params,
        geom,
        time_cfg,
        terms=LinearTerms(
            streaming=1.0,
            mirror=0.0,
            curvature=0.0,
            gradb=0.0,
            diamagnetic=0.0,
            collisions=0.0,
            hypercollisions=0.0,
            end_damping=0.0,
            apar=0.0,
            bpar=0.0,
        ),
        jit=False,
    )
    energy = np.asarray(diag.energy_t)
    assert np.all(np.isfinite(energy))
    if energy.size > 1:
        rel = np.abs((energy[-1] - energy[0]) / max(abs(energy[0]), 1.0e-12))
        assert rel < 0.05


def test_growth_rate_step_matches_real_imag_validity_mask():
    """Instantaneous growth-rate kernel should require non-zero real and imaginary parts."""

    phi_prev = jnp.asarray([[[1.0 + 1.0j, 1.0 + 1.0j]]], dtype=jnp.complex64)
    phi_now_invalid = jnp.asarray([[[2.0 + 0.0j, 2.0 + 0.0j]]], dtype=jnp.complex64)
    mask = jnp.asarray([[True]])
    gamma_bad, omega_bad = _instantaneous_growth_rate_step(
        phi_now_invalid, phi_prev, 0.1, z_index=0, mask=mask
    )
    assert np.allclose(np.asarray(gamma_bad), 0.0)
    assert np.allclose(np.asarray(omega_bad), 0.0)

    phi_now_valid = jnp.asarray([[[2.0 + 2.0j, 2.0 + 2.0j]]], dtype=jnp.complex64)
    gamma_ok, omega_ok = _instantaneous_growth_rate_step(
        phi_now_valid, phi_prev, 0.1, z_index=0, mask=mask
    )
    assert np.isfinite(np.asarray(gamma_ok)).all()
    assert np.isfinite(np.asarray(omega_ok)).all()
    assert not np.allclose(np.asarray(gamma_ok), 0.0)


def test_growth_rate_step_validity_depends_on_current_phi_only():
    """Instantaneous kernel checks real/imag nonzero on current phi only."""

    phi_prev = jnp.asarray([[[1.0 + 0.0j, 1.0 + 0.0j]]], dtype=jnp.complex64)
    phi_now = jnp.asarray([[[2.0 + 2.0j, 2.0 + 2.0j]]], dtype=jnp.complex64)
    mask = jnp.asarray([[True]])
    gamma, omega = _instantaneous_growth_rate_step(
        phi_now, phi_prev, 0.1, z_index=0, mask=mask
    )
    assert np.isfinite(np.asarray(gamma)).all()
    assert np.isfinite(np.asarray(omega)).all()
    assert not np.allclose(np.asarray(gamma), 0.0)


def test_growth_rate_step_max_uses_per_step_peak():
    """Max-mode diagnostics should follow each step's peak-z sample."""

    phi_prev = jnp.asarray([[[1.0 + 1.0j, 8.0 + 8.0j]]], dtype=jnp.complex64)
    phi_now = jnp.asarray([[[9.0 + 9.0j, 2.0 + 2.0j]]], dtype=jnp.complex64)
    mask = jnp.asarray([[True]])
    gamma, omega = _instantaneous_growth_rate_step(
        phi_now,
        phi_prev,
        0.1,
        z_index=0,
        mask=mask,
        mode_method="max",
    )
    ratio = (9.0 + 9.0j) / (8.0 + 8.0j)
    assert np.allclose(np.asarray(gamma), np.log(np.abs(ratio)) / 0.1)
    assert np.allclose(np.asarray(omega), -np.angle(ratio) / 0.1)


def test_growth_mask_promotes_single_selected_nonzonal_slice() -> None:
    mask = _growth_rate_mode_mask(
        jnp.asarray([-0.01]),
        jnp.asarray([0.0]),
        jnp.asarray([[False]]),
    )
    assert np.asarray(mask).item() is True


def test_growth_mask_keeps_single_selected_zonal_slice_masked() -> None:
    mask = _growth_rate_mode_mask(
        jnp.asarray([0.0]),
        jnp.asarray([0.0]),
        jnp.asarray([[False]]),
    )
    assert np.asarray(mask).item() is False


def test_rk4_step_uses_runtime_scaled_end_damping_once() -> None:
    cfg, grid, geom, params, cache = _small_setup()
    params = replace(params, damp_ends_amp=0.5)
    G0 = _build_initial_condition(
        grid, geom, ky_index=0, kx_index=0, Nl=4, Nm=4, init_cfg=cfg.init
    )
    terms = LinearTerms(
        streaming=0.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=0.0,
        hypercollisions=0.0,
        end_damping=1.0,
        apar=0.0,
        bpar=0.0,
    )
    term_cfg = linear_terms_to_term_config(terms)
    dt = 0.2

    G_step, fields_step = _rk4_step(G0, cache, params, term_cfg, dt)

    def rhs(state: jnp.ndarray) -> jnp.ndarray:
        dG, _fields = assemble_rhs_cached(state, cache, params, terms=term_cfg)
        return dG

    k1 = rhs(G0)
    k2 = rhs(G0 + 0.5 * dt * k1)
    k3 = rhs(G0 + 0.5 * dt * k2)
    k4 = rhs(G0 + dt * k3)
    G_manual = G0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    _dG_manual, fields_manual = assemble_rhs_cached(
        G_manual, cache, params, terms=term_cfg
    )

    assert np.allclose(
        np.asarray(G_step), np.asarray(G_manual), rtol=1.0e-6, atol=1.0e-6
    )
    assert np.allclose(
        np.asarray(fields_step.phi),
        np.asarray(fields_manual.phi),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_linear_explicit_adaptive_default_dt_max_clamps_to_nominal_dt():
    """When dt_max is unset, the adaptive explicit path should clamp to dt."""

    cfg, grid, geom, params, cache = _small_setup()
    G0 = _build_initial_condition(
        grid, geom, ky_index=0, kx_index=0, Nl=4, Nm=4, init_cfg=cfg.init
    )
    time_cfg = ExplicitTimeConfig(
        dt=0.01,
        t_max=0.05,
        sample_stride=1,
        fixed_dt=False,
        dt_max=None,
        cfl=10.0,
    )
    _t, _phi_t, _gamma_t, _omega_t, diag = integrate_linear_explicit_diagnostics(
        G0,
        grid,
        cache,
        params,
        geom,
        time_cfg,
        terms=LinearTerms(),
        jit=False,
    )
    dt_t = np.asarray(diag.dt_t, dtype=float)
    assert dt_t.size > 0
    assert np.nanmax(dt_t) <= float(time_cfg.dt) + 1.0e-12


def test_linear_omega_max_preserves_selected_ky_mode():
    cfg = CycloneBaseCase()
    grid_full = build_spectral_grid(cfg.grid)
    ky_index = select_ky_index(np.asarray(grid_full.ky), 0.2)
    grid = select_ky_grid(grid_full, ky_index)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(
        fprim=cfg.model.fprim,
        tprim=cfg.model.tprim_i,
        tprim_e=cfg.model.tprim_e,
        kpar_scale=float(geom.gradpar()),
        nu=cfg.model.nu_i,
    )

    omega_sel = _linear_frequency_bound(grid, geom, params, 4, 4)

    assert omega_sel[1] > 0.0
