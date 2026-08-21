"""Quasilinear transport diagnostic tests."""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

import gkx
import gkx as public_quasilinear
from gkx.diagnostics import quasilinear_transport
from gkx.geometry import SAlphaGeometry, apply_geometry_grid_defaults
from gkx.core.grid import build_spectral_grid, select_ky_grid
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.params import linear_terms_to_term_config
from gkx.diagnostics.quasilinear_transport import (
    compute_quasilinear_from_linear_state,
    effective_kperp2,
    mixing_length_amplitude2_jax,
    normalize_quasilinear_channels,
    phi_norm2,
    quasilinear_feature_objective,
    saturation_amplitude2,
    saturated_flux_from_linear_weight,
    shape_aware_power_law_objective,
    spectral_phi_weights,
)
from gkx.runtime import (
    build_runtime_linear_params,
    build_runtime_linear_terms,
    run_runtime_linear,
    run_runtime_scan,
)
from gkx.workflows.runtime.config import (
    RuntimeConfig,
    RuntimeNormalizationConfig,
    RuntimeQuasilinearConfig,
    RuntimeSpeciesConfig,
)


def _tiny_runtime_config() -> RuntimeConfig:
    base = RuntimeConfig()
    return replace(
        base,
        grid=replace(
            base.grid,
            Nx=1,
            Ny=4,
            Nz=8,
            Lx=62.8,
            Ly=62.8,
            boundary="periodic",
        ),
        time=replace(
            base.time, t_max=0.04, dt=0.01, use_diffrax=False, sample_stride=1
        ),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(
            contract="cyclone", diagnostic_norm="none"
        ),
    )


def _tiny_linear_objects():
    cfg = _tiny_runtime_config()
    geom = SAlphaGeometry.from_config(cfg.geometry)
    grid_full = build_spectral_grid(apply_geometry_grid_defaults(geom, cfg.grid))
    grid = select_ky_grid(grid_full, 1)
    params = build_runtime_linear_params(cfg, Nm=2, geom=geom)
    terms = build_runtime_linear_terms(cfg)
    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=2)
    rng = np.random.default_rng(2)
    shape = (1, 2, 2, grid.ky.size, grid.kx.size, grid.z.size)
    state = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    state = state.astype(np.complex64)
    return cfg, geom, grid, params, terms, cache, state


def test_top_level_api_exports_promoted_quasilinear_diagnostics() -> None:
    """Promoted diagnostics resolve to their implementation-owner objects."""

    promoted = set(public_quasilinear.__all__) & set(quasilinear_transport.__all__)
    assert promoted
    for name in promoted:
        assert getattr(public_quasilinear, name) is getattr(quasilinear_transport, name)


def test_saturation_amplitude_rules_are_explicit() -> None:
    assert saturation_amplitude2(gamma=0.2, kperp_eff2_value=0.5, rule="none") is None
    assert saturation_amplitude2(
        gamma=0.2,
        kperp_eff2_value=0.5,
        rule="mixing_length",
        csat=2.0,
    ) == pytest.approx(0.8)
    assert saturation_amplitude2(
        gamma=-0.2,
        kperp_eff2_value=0.5,
        rule="mixing_length",
    ) == pytest.approx(0.0)
    assert saturation_amplitude2(
        gamma=-0.2, kperp_eff2_value=0.5, rule="linear_weight"
    ) == pytest.approx(1.0)
    assert saturation_amplitude2(
        gamma=-0.2,
        kperp_eff2_value=0.5,
        rule="absolute_growth_mixing_length",
        csat=2.0,
    ) == pytest.approx(0.8)
    assert saturation_amplitude2(
        gamma=0.2, kperp_eff2_value=0.0, rule="mixing_length"
    ) == pytest.approx(0.0)
    assert saturation_amplitude2(
        gamma=0.2, kperp_eff2_value=np.nan, rule="mixing_length"
    ) == pytest.approx(0.0)
    with pytest.raises(NotImplementedError):
        saturation_amplitude2(
            gamma=0.2, kperp_eff2_value=0.5, rule="calibrated_spectral"
        )


def test_saturation_and_channel_edge_cases_are_fail_closed() -> None:
    assert normalize_quasilinear_channels([" ES ", "es"]) == ("es",)
    assert normalize_quasilinear_channels([]) == ("es",)
    assert saturation_amplitude2(
        gamma=-0.2,
        kperp_eff2_value=0.5,
        rule="mixing_length",
        include_stable_modes=True,
    ) == pytest.approx(-0.4)
    assert saturation_amplitude2(
        gamma=0.03,
        kperp_eff2_value=0.5,
        rule="mixing_length",
        gamma_floor=0.05,
    ) == pytest.approx(0.0)
    assert saturation_amplitude2(
        gamma=0.03,
        kperp_eff2_value=0.5,
        rule="lapillonne_2011",
        gamma_floor=0.05,
    ) == pytest.approx(0.0)


def test_differentiable_saturation_rules_match_scalar_contracts() -> None:
    gamma = jnp.asarray([-0.2, 0.0, 0.3])
    kperp = jnp.asarray([0.5, 0.5, 1.5])
    amp = mixing_length_amplitude2_jax(gamma, kperp, csat=2.0, gamma_floor=0.05)

    np.testing.assert_allclose(
        np.asarray(amp), np.asarray([0.0, 0.0, 2.0 * 0.25 / 1.5]), rtol=1.0e-6
    )
    signed = mixing_length_amplitude2_jax(
        gamma, kperp, csat=1.0, include_stable_modes=True
    )
    np.testing.assert_allclose(
        np.asarray(signed), np.asarray([-0.4, 0.0, 0.2]), rtol=1.0e-6
    )
    flux = saturated_flux_from_linear_weight(
        jnp.asarray([2.0, 3.0]), jnp.asarray([0.2, -0.1]), 0.5, csat=1.5
    )
    np.testing.assert_allclose(np.asarray(flux), np.asarray([1.2, 0.0]), rtol=1.0e-6)

    alias = quasilinear_feature_objective(
        jnp.asarray([-0.2, 0.5, 1.5]),
        rule="abs_growth_mixing_length",
        csat=2.0,
    )
    assert float(alias) == pytest.approx(1.2)


def test_quasilinear_feature_objective_supports_sweep_rules() -> None:
    features = jnp.asarray([-0.2, 0.5, 1.5])

    with pytest.raises(ValueError, match="features"):
        quasilinear_feature_objective(jnp.asarray([0.1, 0.2]))
    assert quasilinear_feature_objective(
        features, rule="linear_weight", csat=2.0
    ) == pytest.approx(3.0)
    assert quasilinear_feature_objective(
        features,
        rule="absolute_growth_mixing_length",
        csat=2.0,
    ) == pytest.approx(1.2)
    with pytest.raises(NotImplementedError):
        quasilinear_feature_objective(features, rule="not_a_rule")


def test_quasilinear_feature_objective_vectorizes_and_applies_stability_floor() -> None:
    features = jnp.asarray([[0.1, 0.5, 2.0], [0.01, 0.25, 4.0], [-0.2, 0.5, 6.0]])
    out = quasilinear_feature_objective(
        features,
        rule="mixing_length",
        csat=1.5,
        gamma_floor=0.05,
    )
    np.testing.assert_allclose(
        np.asarray(out), np.asarray([0.3, 0.0, 0.0]), rtol=1.0e-6
    )
    signed = quasilinear_feature_objective(
        features,
        rule="mixing_length",
        include_stable_modes=True,
    )
    np.testing.assert_allclose(
        np.asarray(signed), np.asarray([0.4, 0.16, -2.4]), rtol=1.0e-6
    )


def test_shape_aware_power_law_objective_uses_geometric_ky_reference() -> None:
    features = jnp.asarray([[0.1, 0.5, 2.0], [0.2, 0.7, 3.0]])
    ky = jnp.asarray([0.1, 0.4])
    out = shape_aware_power_law_objective(features, ky, exponent=0.5, csat=2.0)
    ky_ref = float(np.exp(np.mean(np.log(np.asarray(ky)))))
    expected = 2.0 * np.asarray([2.0, 3.0]) * (np.asarray(ky) / ky_ref) ** 0.5
    np.testing.assert_allclose(np.asarray(out), expected, rtol=1.0e-6)
    explicit_ref = shape_aware_power_law_objective(
        features, ky, exponent=1.0, csat=1.0, ky_ref=0.2
    )
    np.testing.assert_allclose(
        np.asarray(explicit_ref), np.asarray([1.0, 6.0]), rtol=1.0e-6
    )
    with pytest.raises(ValueError, match="features"):
        shape_aware_power_law_objective(jnp.asarray([0.1, 0.2]), ky, exponent=1.0)
    assert gkx.shape_aware_power_law_objective is shape_aware_power_law_objective


def test_shape_aware_power_law_objective_clips_nonpositive_ky_reference() -> None:
    features = jnp.asarray([[0.1, 0.5, 2.0], [0.2, 0.7, 3.0]])
    ky = jnp.asarray([0.0, -0.4])
    out = shape_aware_power_law_objective(
        features,
        ky,
        exponent=0.0,
        csat=0.5,
        ky_ref=-1.0,
    )
    np.testing.assert_allclose(np.asarray(out), np.asarray([1.0, 1.5]), rtol=1.0e-6)


def test_quasilinear_channel_validation_rejects_unvalidated_em_channels() -> None:
    assert normalize_quasilinear_channels("") == ("es",)
    assert normalize_quasilinear_channels("es") == ("es",)
    assert normalize_quasilinear_channels(["es"]) == ("es",)
    with pytest.raises(NotImplementedError):
        normalize_quasilinear_channels(["es", "apar"])


def test_phi_norm_and_kperp_are_phase_and_amplitude_invariant() -> None:
    _cfg, _geom, grid, params, _terms, cache, _state = _tiny_linear_objects()
    phi = jnp.ones((grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64)
    vol_fac = jnp.ones(grid.z.size) / grid.z.size
    kperp = effective_kperp2(phi, cache, vol_fac)
    scaled = 3.0 * jnp.exp(0.7j) * phi
    assert effective_kperp2(scaled, cache, vol_fac) == pytest.approx(float(kperp))
    assert phi_norm2(
        scaled, cache, params, vol_fac, normalization="phi_rms"
    ) == pytest.approx(
        9.0 * float(phi_norm2(phi, cache, params, vol_fac, normalization="phi_rms"))
    )
    assert phi_norm2(
        phi, cache, params, vol_fac, normalization="phi_midplane"
    ) == pytest.approx(1.0)
    assert phi_norm2(phi, cache, params, vol_fac, normalization="field_energy") > 0.0
    with pytest.raises(ValueError, match="normalization"):
        phi_norm2(phi, cache, params, vol_fac, normalization="not_a_norm")


def test_spectral_phi_weights_account_for_half_plane_and_dealiasing() -> None:
    _cfg, _geom, grid, _params, _terms, cache, _state = _tiny_linear_objects()
    phi = jnp.ones((grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64)
    vol_fac = jnp.ones(grid.z.size) / grid.z.size

    weights = spectral_phi_weights(phi, cache, vol_fac, use_dealias=False)
    assert weights.shape == phi.shape
    # The selected ky grid is a positive half-plane mode, so the spectral weight
    # doubles non-zonal contributions to preserve the real-field convention.
    assert float(jnp.sum(weights)) == pytest.approx(2.0 * grid.kx.size)

    masked = spectral_phi_weights(phi, cache, vol_fac, use_dealias=True)
    assert float(jnp.sum(masked)) <= float(jnp.sum(weights))


def test_quasilinear_weights_are_phase_and_amplitude_invariant() -> None:
    cfg, geom, grid, params, terms, cache, state = _tiny_linear_objects()
    ql = compute_quasilinear_from_linear_state(
        state,
        cache=cache,
        grid=grid,
        geom=geom,
        params=params,
        ky=float(grid.ky[0]),
        gamma=0.1,
        omega=-0.2,
        terms=linear_terms_to_term_config(terms),
        species_names=[cfg.species[0].name],
    )
    ql_scaled = compute_quasilinear_from_linear_state(
        4.0 * np.exp(0.4j) * state,
        cache=cache,
        grid=grid,
        geom=geom,
        params=params,
        ky=float(grid.ky[0]),
        gamma=0.1,
        omega=-0.2,
        terms=linear_terms_to_term_config(terms),
        species_names=[cfg.species[0].name],
    )
    np.testing.assert_allclose(
        ql_scaled.heat_flux_weight_species, ql.heat_flux_weight_species, rtol=1e-5
    )
    np.testing.assert_allclose(
        ql_scaled.particle_flux_weight_species,
        ql.particle_flux_weight_species,
        rtol=1e-5,
    )
    assert ql.to_dict()["metadata"]["claim_level"] == "linear_weights"

    saturated = compute_quasilinear_from_linear_state(
        state,
        cache=cache,
        grid=grid,
        geom=geom,
        params=params,
        ky=float(grid.ky[0]),
        gamma=0.1,
        omega=-0.2,
        terms=linear_terms_to_term_config(terms),
        mode="saturated",
        saturation_rule="mixing_length",
        species_names=["wrong", "length"],
    )
    assert saturated.saturated_heat_flux_species is not None
    assert saturated.species == ("s0",)
    with pytest.raises(ValueError, match="mode"):
        compute_quasilinear_from_linear_state(
            state,
            cache=cache,
            grid=grid,
            geom=geom,
            params=params,
            ky=float(grid.ky[0]),
            gamma=0.1,
            omega=-0.2,
            terms=linear_terms_to_term_config(terms),
            mode="invalid",
        )
    with pytest.raises(NotImplementedError, match="kperp"):
        compute_quasilinear_from_linear_state(
            state,
            cache=cache,
            grid=grid,
            geom=geom,
            params=params,
            ky=float(grid.ky[0]),
            gamma=0.1,
            omega=-0.2,
            terms=linear_terms_to_term_config(terms),
            kperp_average="arithmetic",
        )


def test_runtime_linear_quasilinear_krylov_smoke() -> None:
    cfg = replace(
        _tiny_runtime_config(),
        quasilinear=RuntimeQuasilinearConfig(
            enabled=True,
            mode="saturated",
            saturation_rule="mixing_length",
            csat=0.3,
        ),
    )
    out = run_runtime_linear(cfg, ky_target=0.2, Nl=2, Nm=2, solver="krylov")
    assert out.quasilinear is not None
    assert out.state is None
    assert out.quasilinear["mode"] == "saturated"
    assert out.quasilinear["saturation_rule"] == "mixing_length"
    assert out.quasilinear["kperp_eff2"] >= 0.0


def test_runtime_scan_collects_quasilinear_payloads_and_rejects_batch() -> None:
    cfg = replace(
        _tiny_runtime_config(),
        quasilinear=RuntimeQuasilinearConfig(enabled=True),
    )
    out = run_runtime_scan(cfg, ky_values=[0.2, 0.3], Nl=2, Nm=2, solver="krylov")
    assert out.quasilinear is not None
    assert len(out.quasilinear) == 2
    with pytest.raises(NotImplementedError):
        run_runtime_scan(
            cfg, ky_values=[0.2, 0.3], Nl=2, Nm=2, solver="time", batch_ky=True
        )
