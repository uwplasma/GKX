"""Quasilinear transport diagnostic tests."""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

import gkx as public_quasilinear
from gkx.diagnostics import quasilinear_transport
from gkx.geometry import SAlphaGeometry, apply_geometry_grid_defaults
from gkx.core_grid import build_spectral_grid, select_ky_grid
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
    spectral_phi_weights,
)
from gkx.runtime import (
    build_runtime_linear_params,
    build_runtime_linear_terms,
    run_runtime_linear,
    run_runtime_scan,
)
from gkx.config import (
    RuntimeConfig,
    RuntimeNormalizationConfig,
    RuntimeQuasilinearConfig,
    RuntimeSpeciesConfig,
)
import json
from pathlib import Path
from support.paths import load_artifact_tool
import gkx
import gkx.diagnostics.quasilinear_calibration as qlc
from gkx.diagnostics.quasilinear_calibration import (
    QuasilinearCalibrationPoint,
    apply_heat_flux_scale,
    calibration_point_from_nonlinear_window_summary,
    calibration_point_from_spectrum_and_nonlinear_window,
    fit_train_heat_flux_scale,
    integrated_quasilinear_flux_from_spectrum,
    quasilinear_calibration_report,
    write_quasilinear_calibration_report,
)
from gkx.diagnostics.transport_windows import (
    NonlinearWindowConvergenceConfig,
    nonlinear_window_convergence_report,
)
import os
from support.paths import REPO_ROOT, load_release_tool
import subprocess
import sys
from support.paths import load_tool_script
from gkx.diagnostics.saturation import (
    SaturationStopConfig,
    _sokal_window_mean_sem,
    saturation_stop_decision,
    sokal_autocorrelation_time,
)
from gkx.diagnostics.transport_windows import (
    NonlinearWindowEnsembleConfig,
    nonlinear_window_convergence_from_csv,
    nonlinear_window_convergence_from_summary,
    nonlinear_window_ensemble_report,
    nonlinear_window_stats_promotion_ready,
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
        time=replace(base.time, t_max=0.04, dt=0.01, sample_stride=1),
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


def test_top_level_api_preserves_named_quasilinear_diagnostics() -> None:
    """Legacy named imports remain available outside the advertised facade."""

    domain_exports = set(quasilinear_transport.__all__)
    assert domain_exports.isdisjoint(public_quasilinear.__all__)
    legacy_exports = {
        "QuasilinearTransportResult",
        "compute_quasilinear_from_linear_state",
        "effective_kperp2",
        "mixing_length_amplitude2_jax",
        "phi_norm2",
        "quasilinear_feature_objective",
        "saturation_amplitude2",
        "saturated_flux_from_linear_weight",
    }
    assert legacy_exports <= domain_exports
    for name in legacy_exports:
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


# Nm=4, not Nm=2, in the two runtime smoke tests below. With two Hermite
# moments this deck carries no resolvable mode at all: the adaptive eigensolve
# returns gamma = 4.3e-08 with an original-operator residual of 5.4e-05, which
# is not a converged eigenpair -- it only passed because
# certifiable_residual_tolerance floors the float32 gate at 1e3 * eps, i.e.
# 1.19e-04, looser than that noise. The float32 lane still runs the same
# arithmetic, so this was invisible until the runtime seed started honouring
# JAX_ENABLE_X64 and the float64 lane began applying the 1e-09 gate it had
# asked for, at which point the honest residual is 1.06665 and the solve fails
# closed, as Q12 intends. Nm=4 gives a mode both precisions certify and agree
# on: gamma = -1.676437e-03 at residual 1.69e-06 against the 1.19e-04 float32
# gate, and -1.676444e-03 at 3.58e-15 against the 1e-09 float64 one, agreeing
# to 4e-06 relative. These are plumbing smoke tests, so the resolution only has
# to be high enough that the eigenpair they exercise exists.
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
    out = run_runtime_linear(cfg, ky_target=0.1, Nl=2, Nm=4, solver="krylov")
    assert out.quasilinear is not None
    assert out.state is None
    assert out.quasilinear["mode"] == "saturated"
    assert out.quasilinear["saturation_rule"] == "mixing_length"
    assert out.quasilinear["kperp_eff2"] >= 0.0


def test_runtime_scan_collects_quasilinear_payloads_and_rejects_batch() -> None:
    base = _tiny_runtime_config()
    # Ny=12 puts ky 0.2 and 0.3 on the grid; the tiny Ny=4 grid stops at 0.1.
    cfg = replace(
        base,
        grid=replace(base.grid, Ny=12),
        quasilinear=RuntimeQuasilinearConfig(enabled=True),
    )
    out = run_runtime_scan(cfg, ky_values=[0.2, 0.3], Nl=2, Nm=4, solver="krylov")
    assert out.quasilinear is not None
    assert len(out.quasilinear) == 2
    with pytest.raises(NotImplementedError):
        run_runtime_scan(
            cfg, ky_values=[0.2, 0.3], Nl=2, Nm=2, solver="time", batch_ky=True
        )


# ---- from test_quasilinear_calibration.py ----
# Tests for quasilinear calibration artifact helpers.


def _load_build_tool_module():
    return load_artifact_tool("plot_quasilinear_calibration")


def _valid_window_stats(case: str = "holdout") -> dict:
    t = np.linspace(0.0, 120.0, 121)
    heat = 2.0 + 0.02 * np.sin(2.0 * np.pi * t / 12.0)
    return nonlinear_window_convergence_report(
        t,
        heat,
        case=case,
        source_artifact=f"{case}.csv",
        config=NonlinearWindowConvergenceConfig(
            transient_fraction=0.5,
            min_samples=40,
            max_running_mean_rel_drift=0.03,
            max_sem_rel=0.03,
        ),
    )


def test_quasilinear_calibration_report_tracks_train_holdout_claim_level(
    tmp_path: Path,
) -> None:
    assert gkx.QuasilinearCalibrationPoint is QuasilinearCalibrationPoint
    points = [
        QuasilinearCalibrationPoint(
            case="cyclone_ky0p2",
            split="train",
            predicted_heat_flux=1.0,
            observed_heat_flux=1.1,
            saturation_rule="mixing_length",
            geometry="cyclone",
            electron_model="adiabatic",
            nonlinear_window_stats=_valid_window_stats("train"),
        ),
        {
            "case": "cyclone_ky0p3",
            "split": "holdout",
            "predicted_heat_flux": 0.9,
            "observed_heat_flux": 1.0,
            "saturation_rule": "mixing_length",
            "geometry": "cyclone",
            "electron_model": "adiabatic",
            "nonlinear_window_stats": _valid_window_stats("holdout"),
        },
    ]

    report = quasilinear_calibration_report(
        points,
        saturation_rule="mixing_length",
        holdout_mean_rel_gate=0.2,
        metadata={"calibration_policy": "one_constant_train_holdout"},
    )

    assert report["passed"] is True
    assert report["claim_level"] == "calibrated_absolute_flux"
    assert report["by_split"]["train"]["n"] == 1
    assert report["by_split"]["holdout"]["mean_abs_relative_error"] == pytest.approx(
        0.1
    )
    out = write_quasilinear_calibration_report(tmp_path / "ql_calibration.json", report)
    assert json.loads(out.read_text(encoding="utf-8"))["passed"] is True


def test_quasilinear_calibration_report_demotes_missing_holdout_or_failed_gate() -> (
    None
):
    train_only = quasilinear_calibration_report(
        [
            QuasilinearCalibrationPoint(
                case="cyclone_train",
                split="train",
                predicted_heat_flux=1.0,
                observed_heat_flux=1.0,
                saturation_rule="mixing_length",
            )
        ],
        saturation_rule="mixing_length",
    )
    assert train_only["passed"] is False
    assert train_only["claim_level"] == "training_or_audit_only"

    failed = quasilinear_calibration_report(
        [
            QuasilinearCalibrationPoint(
                case="train",
                split="train",
                predicted_heat_flux=1.0,
                observed_heat_flux=1.0,
                saturation_rule="mixing_length",
            ),
            QuasilinearCalibrationPoint(
                case="holdout",
                split="holdout",
                predicted_heat_flux=2.0,
                observed_heat_flux=1.0,
                saturation_rule="mixing_length",
            ),
        ],
        saturation_rule="mixing_length",
        holdout_mean_rel_gate=0.2,
    )
    assert failed["passed"] is False
    assert failed["claim_level"] == "calibration_dataset"
    assert failed["by_split"]["holdout"]["max_abs_relative_error"] == pytest.approx(1.0)

    missing_window_stats = quasilinear_calibration_report(
        [
            QuasilinearCalibrationPoint(
                case="train",
                split="train",
                predicted_heat_flux=1.0,
                observed_heat_flux=1.0,
                saturation_rule="mixing_length",
            ),
            QuasilinearCalibrationPoint(
                case="holdout",
                split="holdout",
                predicted_heat_flux=0.95,
                observed_heat_flux=1.0,
                saturation_rule="mixing_length",
            ),
        ],
        saturation_rule="mixing_length",
        holdout_mean_rel_gate=0.2,
    )
    assert missing_window_stats["passed"] is False
    assert missing_window_stats["claim_level"] == "calibration_dataset"
    assert (
        "missing nonlinear_window_stats"
        in missing_window_stats["metadata"]["holdout_window_convergence"]["failures"][0]
    )


def test_quasilinear_calibration_report_can_fit_one_train_scale() -> None:
    points = [
        QuasilinearCalibrationPoint(
            case="train",
            split="train",
            predicted_heat_flux=0.25,
            observed_heat_flux=1.0,
            saturation_rule="mixing_length",
            nonlinear_window_stats=_valid_window_stats("train_scale"),
        ),
        QuasilinearCalibrationPoint(
            case="holdout",
            split="holdout",
            predicted_heat_flux=0.5,
            observed_heat_flux=2.2,
            saturation_rule="mixing_length",
            nonlinear_window_stats=_valid_window_stats("holdout_scale"),
        ),
    ]

    scale_fit = fit_train_heat_flux_scale(points)
    assert gkx.fit_train_heat_flux_scale is fit_train_heat_flux_scale
    assert scale_fit["scale"] == pytest.approx(4.0)
    scaled = apply_heat_flux_scale(points, scale=scale_fit["scale"])
    assert gkx.apply_heat_flux_scale is apply_heat_flux_scale
    assert scaled[0].predicted_heat_flux == pytest.approx(1.0)
    assert scaled[0].raw_predicted_heat_flux == pytest.approx(0.25)
    assert scaled[0].calibration_scale == pytest.approx(4.0)

    report = quasilinear_calibration_report(
        points,
        saturation_rule="mixing_length",
        holdout_mean_rel_gate=0.1,
        fit_train_scale=True,
    )

    assert report["passed"] is True
    assert report["claim_level"] == "calibrated_absolute_flux"
    assert report["metadata"]["heat_flux_scale_fit"]["scale"] == pytest.approx(4.0)
    assert report["points"][1]["predicted_heat_flux"] == pytest.approx(2.0)
    assert report["points"][1]["raw_predicted_heat_flux"] == pytest.approx(0.5)
    assert report["by_split"]["holdout"]["mean_abs_relative_error"] == pytest.approx(
        0.2 / 2.2
    )
    with pytest.raises(ValueError):
        apply_heat_flux_scale(points, scale=-1.0)


def test_quasilinear_calibration_report_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        quasilinear_calibration_report([], saturation_rule="mixing_length")
    with pytest.raises(ValueError):
        quasilinear_calibration_report(
            [
                QuasilinearCalibrationPoint(
                    case="bad",
                    split="validation",
                    predicted_heat_flux=1.0,
                    observed_heat_flux=1.0,
                    saturation_rule="mixing_length",
                )
            ],
            saturation_rule="mixing_length",
        )
    with pytest.raises(ValueError):
        quasilinear_calibration_report(
            [
                QuasilinearCalibrationPoint(
                    case="bad",
                    split="train",
                    predicted_heat_flux=1.0,
                    observed_heat_flux=1.0,
                    saturation_rule="mixing_length",
                )
            ],
            saturation_rule="mixing_length",
            observed_floor=0.0,
        )
    with pytest.raises(ValueError, match="non-finite"):
        quasilinear_calibration_report(
            [
                QuasilinearCalibrationPoint(
                    case="nan_prediction",
                    split="train",
                    predicted_heat_flux=float("nan"),
                    observed_heat_flux=1.0,
                    saturation_rule="mixing_length",
                )
            ],
            saturation_rule="mixing_length",
        )
    with pytest.raises(ValueError, match="negative observed_heat_flux_std"):
        quasilinear_calibration_report(
            [
                QuasilinearCalibrationPoint(
                    case="negative_std",
                    split="audit",
                    predicted_heat_flux=1.0,
                    observed_heat_flux=1.0,
                    observed_heat_flux_std=-0.1,
                    saturation_rule="mixing_length",
                )
            ],
            saturation_rule="mixing_length",
        )
    with pytest.raises(ValueError, match="report saturation_rule"):
        quasilinear_calibration_report(
            [
                QuasilinearCalibrationPoint(
                    case="mixed_rule",
                    split="audit",
                    predicted_heat_flux=1.0,
                    observed_heat_flux=1.0,
                    saturation_rule="linear_weight",
                )
            ],
            saturation_rule="mixing_length",
        )
    metrics = quasilinear_calibration_report(
        [
            QuasilinearCalibrationPoint(
                case="floor",
                split="audit",
                predicted_heat_flux=1.0e-6,
                observed_heat_flux=0.0,
                saturation_rule="mixing_length",
            )
        ],
        saturation_rule="mixing_length",
        observed_floor=1.0e-5,
    )["metrics"]
    assert np.isfinite(float(metrics["mean_abs_relative_error"]))


def test_quasilinear_scale_fit_rejects_unphysical_training_sets() -> None:
    with pytest.raises(ValueError, match="prediction_floor"):
        fit_train_heat_flux_scale([], prediction_floor=-1.0)

    with pytest.raises(ValueError, match="no finite nonzero"):
        fit_train_heat_flux_scale(
            [
                QuasilinearCalibrationPoint(
                    case="zero",
                    split="train",
                    predicted_heat_flux=0.0,
                    observed_heat_flux=1.0,
                    saturation_rule="mixing_length",
                )
            ]
        )

    with pytest.raises(ValueError, match="too small"):
        fit_train_heat_flux_scale(
            [
                QuasilinearCalibrationPoint(
                    case="ill_conditioned",
                    split="train",
                    predicted_heat_flux=1.0e-4,
                    observed_heat_flux=1.0,
                    saturation_rule="mixing_length",
                )
            ],
            prediction_floor=1.0e-5,
        )

    with pytest.raises(ValueError, match="negative"):
        fit_train_heat_flux_scale(
            [
                QuasilinearCalibrationPoint(
                    case="wrong_sign",
                    split="train",
                    predicted_heat_flux=1.0,
                    observed_heat_flux=-1.0,
                    saturation_rule="mixing_length",
                )
            ]
        )

    with pytest.raises(ValueError, match="holdout_mean_rel_gate"):
        quasilinear_calibration_report(
            [
                QuasilinearCalibrationPoint(
                    case="gate",
                    split="audit",
                    predicted_heat_flux=1.0,
                    observed_heat_flux=1.0,
                    saturation_rule="mixing_length",
                )
            ],
            saturation_rule="mixing_length",
            holdout_mean_rel_gate=0.0,
        )


def test_calibration_point_from_nonlinear_window_summary(tmp_path: Path) -> None:
    diag = tmp_path / "diag.csv"
    diag.write_text(
        "t,heat_flux,particle_flux\n0.0,1.0,0.0\n1.0,2.0,0.0\n2.0,4.0,0.0\n",
        encoding="utf-8",
    )
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "case": "cyclone_window",
                "gkx": str(diag),
                "tmin": 0.5,
                "tmax": 2.0,
            }
        ),
        encoding="utf-8",
    )

    point = calibration_point_from_nonlinear_window_summary(
        summary,
        predicted_heat_flux=2.5,
        split="holdout",
        saturation_rule="mixing_length",
        geometry="cyclone",
        electron_model="adiabatic",
    )

    assert point.case == "cyclone_window"
    assert point.observed_heat_flux == pytest.approx(3.0)
    assert point.observed_heat_flux_std == pytest.approx(1.0)
    assert point.nonlinear_artifact == str(diag)


def test_calibration_point_from_replicated_ensemble_gate(tmp_path: Path) -> None:
    summary = tmp_path / "ensemble.json"
    summary.write_text(
        json.dumps(
            {
                "kind": "nonlinear_window_ensemble_report",
                "case": "cth_like_replicated",
                "passed": True,
                "statistics": {
                    "ensemble_mean": 9.5,
                    "combined_sem": 0.4,
                    "combined_sem_rel": 0.042105263157894736,
                    "n_reports": 3,
                },
                "rows": [
                    {
                        "promotion_ready": True,
                        "source_artifact": "replicate_a.csv",
                    },
                    {
                        "promotion_ready": True,
                        "source_artifact": "replicate_b.csv",
                    },
                    {
                        "promotion_ready": True,
                        "source_artifact": "replicate_c.csv",
                    },
                ],
                "gate_report": {"passed": True},
            }
        ),
        encoding="utf-8",
    )

    point = calibration_point_from_nonlinear_window_summary(
        summary,
        predicted_heat_flux=7.5,
        split="holdout",
        saturation_rule="spectral_envelope_ridge",
        geometry="cth_like_external_vmec",
        electron_model="adiabatic",
    )

    assert point.case == "cth_like_replicated"
    assert point.observed_heat_flux == pytest.approx(9.5)
    assert point.observed_heat_flux_std == pytest.approx(0.4)
    assert point.nonlinear_artifact == str(summary)
    assert point.nonlinear_window_stats is not None
    assert point.nonlinear_window_stats["kind"] == "nonlinear_window_ensemble_report"
    assert "nonlinear_source=replicated_ensemble_gate" in str(point.notes)

    report = quasilinear_calibration_report(
        [
            QuasilinearCalibrationPoint(
                case="train",
                split="train",
                predicted_heat_flux=1.0,
                observed_heat_flux=1.0,
                saturation_rule="spectral_envelope_ridge",
                nonlinear_window_stats=_valid_window_stats("train"),
            ),
            point,
        ],
        saturation_rule="spectral_envelope_ridge",
        holdout_mean_rel_gate=0.5,
    )
    assert report["metadata"]["holdout_window_convergence"]["passed"] is True


def test_calibration_point_from_replicated_ensemble_gate_is_fail_closed(
    tmp_path: Path,
) -> None:
    summary = tmp_path / "bad_ensemble.json"
    summary.write_text(
        json.dumps(
            {
                "kind": "nonlinear_window_ensemble_report",
                "case": "bad",
                "passed": True,
                "statistics": {"ensemble_mean": 9.5, "combined_sem": 0.4},
                "rows": [{"promotion_ready": False, "source_artifact": "a.csv"}],
                "gate_report": {"passed": True},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="promotion-ready"):
        calibration_point_from_nonlinear_window_summary(
            summary,
            predicted_heat_flux=1.0,
            split="holdout",
            saturation_rule="linear_weight",
        )

    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["passed"] = "true"
    summary.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="did not pass"):
        calibration_point_from_nonlinear_window_summary(
            summary,
            predicted_heat_flux=1.0,
            split="holdout",
            saturation_rule="linear_weight",
        )


def test_calibration_point_from_nonlinear_netcdf_window_summary(tmp_path: Path) -> None:
    netCDF4 = pytest.importorskip("netCDF4")
    diag = tmp_path / "run.out.nc"
    with netCDF4.Dataset(diag, "w") as root:
        root.createDimension("time", 3)
        root.createDimension("species", 2)
        grids = root.createGroup("Grids")
        diagnostics = root.createGroup("Diagnostics")
        time = grids.createVariable("time", "f8", ("time",))
        heat = diagnostics.createVariable("HeatFlux_st", "f8", ("time", "species"))
        time[:] = np.asarray([0.0, 1.0, 2.0])
        heat[:, :] = np.asarray([[1.0, 10.0], [2.0, 20.0], [4.0, 40.0]])
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "case": "w7x_window",
                "gkx": str(diag),
                "tmin": 0.5,
                "tmax": 2.0,
            }
        ),
        encoding="utf-8",
    )

    summed = calibration_point_from_nonlinear_window_summary(
        summary,
        predicted_heat_flux=1.0,
        split="holdout",
        saturation_rule="mixing_length",
        geometry="w7x",
        electron_model="adiabatic",
    )
    ion = calibration_point_from_nonlinear_window_summary(
        summary,
        predicted_heat_flux=1.0,
        split="holdout",
        saturation_rule="mixing_length",
        geometry="w7x",
        electron_model="adiabatic",
        species_index=0,
    )

    assert summed.observed_heat_flux == pytest.approx(33.0)
    assert summed.observed_heat_flux_std == pytest.approx(11.0)
    assert "nonlinear_variable=Diagnostics/HeatFlux_st" in str(summed.notes)
    assert "nonlinear_window_samples=2" in str(summed.notes)
    assert ion.observed_heat_flux == pytest.approx(3.0)
    assert ion.observed_heat_flux_std == pytest.approx(1.0)


def test_calibration_point_from_nonlinear_netcdf_window_validation(
    tmp_path: Path,
) -> None:
    netCDF4 = pytest.importorskip("netCDF4")
    assert (
        qlc._netcdf_heat_flux_variable("Diagnostics/HeatFlux_st")
        == "Diagnostics/HeatFlux_st"
    )
    assert (
        qlc._netcdf_heat_flux_variable("DiagnosticsHeatFlux_st")
        == "DiagnosticsHeatFlux_st"
    )
    assert (
        qlc._netcdf_heat_flux_variable("HeatFluxES_st") == "Diagnostics/HeatFluxES_st"
    )
    with pytest.raises(ValueError, match="unknown NetCDF heat-flux"):
        qlc._netcdf_heat_flux_variable("not_a_heat_flux")

    diag = tmp_path / "run.out.nc"
    with netCDF4.Dataset(diag, "w") as root:
        root.createDimension("time", 3)
        root.createDimension("species", 2)
        grids = root.createGroup("Grids")
        diagnostics = root.createGroup("Diagnostics")
        grids.createVariable("time", "f8", ("time",))[:] = np.asarray([0.0, 1.0, 2.0])
        diagnostics.createVariable("HeatFluxES_st", "f8", ("time",))[:] = np.asarray(
            [1.0, 2.0, 5.0]
        )
        diagnostics.createVariable("HeatFlux_st", "f8", ("time", "species"))[:, :] = (
            np.asarray([[1.0, 10.0], [2.0, 20.0], [4.0, 40.0]])
        )
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps({"gkx": str(diag), "tmin": 0.5, "tmax": 2.0}), encoding="utf-8"
    )

    es = calibration_point_from_nonlinear_window_summary(
        summary,
        predicted_heat_flux=1.0,
        split="audit",
        saturation_rule="mixing_length",
        heat_flux_column="HeatFluxES_st",
    )
    assert es.observed_heat_flux == pytest.approx(3.5)
    assert "nonlinear_variable=Diagnostics/HeatFluxES_st" in str(es.notes)

    with pytest.raises(ValueError, match="species_index"):
        calibration_point_from_nonlinear_window_summary(
            summary,
            predicted_heat_flux=1.0,
            split="audit",
            saturation_rule="mixing_length",
            species_index=4,
        )

    with netCDF4.Dataset(diag) as root:
        with pytest.raises(ValueError, match="must not be empty"):
            qlc._netcdf_variable(root, "")
        with pytest.raises(KeyError, match="group"):
            qlc._netcdf_variable(root, "Missing/HeatFlux_st")
        with pytest.raises(KeyError, match="variable"):
            qlc._netcdf_variable(root, "Diagnostics/Missing")

    mismatch = tmp_path / "mismatch.out.nc"
    with netCDF4.Dataset(mismatch, "w") as root:
        root.createDimension("time", 3)
        root.createDimension("short_time", 2)
        grids = root.createGroup("Grids")
        diagnostics = root.createGroup("Diagnostics")
        grids.createVariable("time", "f8", ("time",))[:] = np.asarray([0.0, 1.0, 2.0])
        diagnostics.createVariable("HeatFlux_st", "f8", ("short_time",))[:] = (
            np.asarray([1.0, 2.0])
        )
    mismatch_summary = tmp_path / "mismatch_summary.json"
    mismatch_summary.write_text(json.dumps({"gkx": str(mismatch)}), encoding="utf-8")
    with pytest.raises(ValueError, match="first dimension"):
        calibration_point_from_nonlinear_window_summary(
            mismatch_summary,
            predicted_heat_flux=1.0,
            split="audit",
            saturation_rule="mixing_length",
        )

    rank3 = tmp_path / "rank3.out.nc"
    with netCDF4.Dataset(rank3, "w") as root:
        root.createDimension("time", 3)
        root.createDimension("species", 2)
        root.createDimension("field", 2)
        grids = root.createGroup("Grids")
        diagnostics = root.createGroup("Diagnostics")
        grids.createVariable("time", "f8", ("time",))[:] = np.asarray([0.0, 1.0, 2.0])
        diagnostics.createVariable("HeatFlux_st", "f8", ("time", "species", "field"))[
            :, :, :
        ] = np.ones((3, 2, 2))
    rank3_summary = tmp_path / "rank3_summary.json"
    rank3_summary.write_text(json.dumps({"gkx": str(rank3)}), encoding="utf-8")
    with pytest.raises(ValueError, match="must have shape"):
        calibration_point_from_nonlinear_window_summary(
            rank3_summary,
            predicted_heat_flux=1.0,
            split="audit",
            saturation_rule="mixing_length",
        )

    no_window = tmp_path / "no_window.out.nc"
    with netCDF4.Dataset(no_window, "w") as root:
        root.createDimension("time", 2)
        grids = root.createGroup("Grids")
        diagnostics = root.createGroup("Diagnostics")
        grids.createVariable("time", "f8", ("time",))[:] = np.asarray([0.0, 1.0])
        diagnostics.createVariable("HeatFlux_st", "f8", ("time",))[:] = np.asarray(
            [np.nan, np.nan]
        )
    no_window_summary = tmp_path / "no_window_summary.json"
    no_window_summary.write_text(
        json.dumps({"gkx": str(no_window), "tmin": 0.0, "tmax": 1.0}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no finite heat-flux samples"):
        calibration_point_from_nonlinear_window_summary(
            no_window_summary,
            predicted_heat_flux=1.0,
            split="audit",
            saturation_rule="mixing_length",
        )


def test_integrated_quasilinear_flux_from_spectrum_and_window_point(
    tmp_path: Path,
) -> None:
    spectrum = tmp_path / "ql.csv"
    spectrum.write_text(
        "ky,saturated_heat_flux_total,other\n0.1,1.0,0\n0.2,2.0,0\n0.4,4.0,0\n",
        encoding="utf-8",
    )
    summed = integrated_quasilinear_flux_from_spectrum(spectrum)
    assert (
        gkx.integrated_quasilinear_flux_from_spectrum
        is integrated_quasilinear_flux_from_spectrum
    )
    assert summed["estimate"] == pytest.approx(7.0)
    assert summed["n_samples"] == 3
    trapezoid = integrated_quasilinear_flux_from_spectrum(spectrum, method="trapezoid")
    assert trapezoid["estimate"] == pytest.approx(0.75)
    with pytest.raises(ValueError):
        integrated_quasilinear_flux_from_spectrum(spectrum, column="missing")

    diag = tmp_path / "diag.csv"
    diag.write_text("t,heat_flux\n0.0,2.0\n1.0,4.0\n", encoding="utf-8")
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"case": "c", "gkx": str(diag)}), encoding="utf-8")

    point = calibration_point_from_spectrum_and_nonlinear_window(
        spectrum,
        summary,
        split="audit",
        saturation_rule="mixing_length",
        spectrum_method="sum",
        geometry="cyclone",
        electron_model="adiabatic",
    )

    assert point.predicted_heat_flux == pytest.approx(7.0)
    assert point.observed_heat_flux == pytest.approx(3.0)
    assert point.quasilinear_artifact == str(spectrum)
    assert "observed_to_predicted" in str(point.notes)


def test_integrated_quasilinear_flux_from_spectrum_variants_and_validation(
    tmp_path: Path,
) -> None:
    single = tmp_path / "single.csv"
    single.write_text("saturated_heat_flux_total\n2.5\n", encoding="utf-8")
    one_point = integrated_quasilinear_flux_from_spectrum(single, delta_ky=0.2)
    assert one_point["estimate"] == pytest.approx(0.5)
    assert one_point["ky_min"] == pytest.approx(0.0)
    assert one_point["ky_max"] == pytest.approx(0.0)

    spectrum = tmp_path / "spectrum.csv"
    spectrum.write_text(
        "ky,saturated_heat_flux_total\n0.1,1.0\n0.2,nan\n0.4,5.0\n",
        encoding="utf-8",
    )
    averaged = integrated_quasilinear_flux_from_spectrum(spectrum, method="mean")
    assert averaged["estimate"] == pytest.approx(3.0)
    assert averaged["n_samples"] == 2

    unsorted = tmp_path / "unsorted.csv"
    unsorted.write_text(
        "ky,saturated_heat_flux_total\n0.4,4.0\n0.1,1.0\n0.2,2.0\n",
        encoding="utf-8",
    )
    trapezoid = integrated_quasilinear_flux_from_spectrum(unsorted, method="trapezoid")
    assert trapezoid["estimate"] == pytest.approx(0.75)

    no_finite = tmp_path / "no_finite.csv"
    no_finite.write_text("ky,saturated_heat_flux_total\n0.1,nan\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no finite samples"):
        integrated_quasilinear_flux_from_spectrum(no_finite)
    with pytest.raises(ValueError, match="delta_ky"):
        integrated_quasilinear_flux_from_spectrum(single, delta_ky=0.0)
    with pytest.raises(ValueError, match="trapezoid"):
        integrated_quasilinear_flux_from_spectrum(single, method="trapezoid")
    with pytest.raises(ValueError, match="method"):
        integrated_quasilinear_flux_from_spectrum(spectrum, method="simpson")


def test_calibration_summary_ingestion_validates_csv_windows_and_relative_paths(
    tmp_path: Path,
) -> None:
    diag = tmp_path / "diag.csv"
    diag.write_text("t,heat_flux\n0.0,1.0\n1.0,3.0\n", encoding="utf-8")
    summaries = tmp_path / "summaries"
    summaries.mkdir()
    summary = summaries / "summary.json"
    summary.write_text(
        json.dumps({"case": "relative", "gkx": "../diag.csv"}), encoding="utf-8"
    )

    point = calibration_point_from_nonlinear_window_summary(
        summary,
        predicted_heat_flux=2.0,
        split="audit",
        saturation_rule="mixing_length",
    )
    assert point.observed_heat_flux == pytest.approx(2.0)
    assert point.nonlinear_artifact == str(diag.resolve())

    missing_source = tmp_path / "missing_source.json"
    missing_source.write_text(json.dumps({"case": "missing"}), encoding="utf-8")
    with pytest.raises(ValueError, match="diagnostics source"):
        calibration_point_from_nonlinear_window_summary(
            missing_source,
            predicted_heat_flux=1.0,
            split="audit",
            saturation_rule="mixing_length",
        )

    missing_t = tmp_path / "missing_t.csv"
    missing_t.write_text("heat_flux\n1.0\n", encoding="utf-8")
    summary_missing_t = tmp_path / "summary_missing_t.json"
    summary_missing_t.write_text(json.dumps({"gkx": str(missing_t)}), encoding="utf-8")
    with pytest.raises(ValueError, match="'t' column"):
        calibration_point_from_nonlinear_window_summary(
            summary_missing_t,
            predicted_heat_flux=1.0,
            split="audit",
            saturation_rule="mixing_length",
        )

    no_window = tmp_path / "no_window.csv"
    no_window.write_text("t,heat_flux\n0.0,nan\n1.0,nan\n", encoding="utf-8")
    summary_no_window = tmp_path / "summary_no_window.json"
    summary_no_window.write_text(
        json.dumps({"gkx": str(no_window), "tmin": 0.0, "tmax": 1.0}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no finite heat-flux samples"):
        calibration_point_from_nonlinear_window_summary(
            summary_no_window,
            predicted_heat_flux=1.0,
            split="audit",
            saturation_rule="mixing_length",
        )


def test_build_calibration_report_tool_can_generate_point_from_artifacts(
    tmp_path: Path,
) -> None:
    mod = _load_build_tool_module()
    spectrum = tmp_path / "ql.csv"
    spectrum.write_text(
        "ky,saturated_heat_flux_total\n0.1,1.0\n0.2,2.0\n", encoding="utf-8"
    )
    diag = tmp_path / "diag.csv"
    diag.write_text("t,heat_flux\n0.0,4.0\n1.0,6.0\n", encoding="utf-8")
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps({"case": "generated", "gkx": str(diag)}), encoding="utf-8"
    )
    out = tmp_path / "report.json"

    assert (
        mod.main(
            [
                "--spectrum",
                str(spectrum),
                "--nonlinear-summary",
                str(summary),
                "--split",
                "audit",
                "--out",
                str(out),
            ]
        )
        == 0
    )

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["claim_level"] == "training_or_audit_only"
    assert report["points"][0]["predicted_heat_flux"] == pytest.approx(3.0)
    assert report["points"][0]["observed_heat_flux"] == pytest.approx(5.0)


def test_build_calibration_report_tool_can_fit_train_scale(tmp_path: Path) -> None:
    mod = _load_build_tool_module()
    points = tmp_path / "points.json"
    points.write_text(
        json.dumps(
            [
                {
                    "case": "train",
                    "split": "train",
                    "predicted_heat_flux": 0.25,
                    "observed_heat_flux": 1.0,
                    "saturation_rule": "mixing_length",
                    "nonlinear_window_stats": _valid_window_stats("tool_train"),
                },
                {
                    "case": "holdout",
                    "split": "holdout",
                    "predicted_heat_flux": 0.5,
                    "observed_heat_flux": 2.2,
                    "saturation_rule": "mixing_length",
                    "nonlinear_window_stats": _valid_window_stats("tool_holdout"),
                },
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "report.json"

    assert (
        mod.main(
            [
                "--points",
                str(points),
                "--fit-train-scale",
                "--holdout-mean-rel-gate",
                "0.1",
                "--out",
                str(out),
            ]
        )
        == 0
    )

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["metadata"]["heat_flux_scale_fit"]["scale"] == pytest.approx(4.0)


def test_calibration_point_from_nonlinear_window_summary_rejects_unsupported_sources(
    tmp_path: Path,
) -> None:
    diag = tmp_path / "diag.csv"
    diag.write_text("t,heat_flux\n0.0,1.0\n", encoding="utf-8")
    missing_col = tmp_path / "missing.csv"
    missing_col.write_text("t,Wphi\n0.0,1.0\n", encoding="utf-8")

    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps({"case": "c", "gkx": str(missing_col)}), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        calibration_point_from_nonlinear_window_summary(
            summary,
            predicted_heat_flux=1.0,
            split="audit",
            saturation_rule="mixing_length",
        )

    txt_summary = tmp_path / "summary_txt.json"
    txt_summary.write_text(
        json.dumps({"case": "c", "gkx": str(tmp_path / "run.txt")}),
        encoding="utf-8",
    )
    with pytest.raises(NotImplementedError):
        calibration_point_from_nonlinear_window_summary(
            txt_summary,
            predicted_heat_flux=1.0,
            split="audit",
            saturation_rule="mixing_length",
        )


# ---- from test_quasilinear_promotion_guardrails.py ----
# Tests for quasilinear absolute-flux promotion guardrails.


def _load_tool_module():
    return load_release_tool("check_quasilinear_promotion_guardrails")


def _write_doc(path: Path, text: str | None = None) -> None:
    path.write_text(
        text
        or (
            "This diagnostic is not a runtime/TOML absolute-flux predictor. "
            "Absolute-flux prediction not promoted.\n"
        ),
        encoding="utf-8",
    )


def _window_stats(*, passed: bool = True) -> dict:
    return {
        "kind": "nonlinear_window_convergence_report",
        "passed": passed,
        "statistics": {
            "late_mean": 1.0,
            "sem": 0.02,
            "block_bootstrap_sem": 0.02,
            "running_mean_rel_drift": 0.01,
        },
        "window": {
            "input_tmin": None,
            "transient_fraction": 0.5,
            "transient_cutoff": 50.0,
            "late_tmin": 50.0,
            "late_tmax": 100.0,
            "n_finite_late": 64,
        },
        "provenance": {"source_artifact": "tools_out/nonlinear_trace.csv"},
        "gate_report": {"passed": passed},
    }


def _calibration_report(
    *, claim_level: str, passed: bool, holdout_error: float
) -> dict:
    return {
        "kind": "quasilinear_calibration_report",
        "claim_level": claim_level,
        "passed": passed,
        "holdout_mean_rel_gate": 0.35,
        "metadata": {"calibration_policy": "one constant train/holdout"},
        "by_split": {
            "train": {"n": 1, "mean_abs_relative_error": 0.0},
            "holdout": {"n": 1, "mean_abs_relative_error": holdout_error},
        },
        "points": [
            {
                "case": "train",
                "split": "train",
                "geometry": "cyclone",
                "electron_model": "adiabatic",
                "saturation_rule": "mixing_length",
                "nonlinear_artifact": "tools_out/train.csv",
                "quasilinear_artifact": "docs/_static/train_spectrum.csv",
                "predicted_heat_flux": 1.0,
                "raw_predicted_heat_flux": 0.5,
                "calibration_scale": 2.0,
                "observed_heat_flux": 1.0,
                "observed_heat_flux_std": 0.1,
                "nonlinear_window_stats": _window_stats(),
            },
            {
                "case": "holdout",
                "split": "holdout",
                "geometry": "miller",
                "electron_model": "adiabatic",
                "saturation_rule": "mixing_length",
                "nonlinear_artifact": "tools_out/holdout.csv",
                "quasilinear_artifact": "docs/_static/holdout_spectrum.csv",
                "predicted_heat_flux": 1.0,
                "raw_predicted_heat_flux": 0.5,
                "calibration_scale": 2.0,
                "observed_heat_flux": 1.0,
                "observed_heat_flux_std": 0.2,
                "nonlinear_window_stats": _window_stats(),
            },
        ],
    }


def _release_contract(
    *,
    claim_level: str = "scoped_candidate_model_selection_not_runtime_flux_predictor",
    absolute_flux_promoted: bool = False,
    include_guardrail_artifact: bool = True,
) -> dict:
    artifacts = [
        "docs/_static/quasilinear_validated_calibration_inputs.json",
        "docs/_static/quasilinear_candidate_uncertainty.json",
    ]
    if include_guardrail_artifact:
        artifacts.append("docs/_static/quasilinear_promotion_guardrails.json")
    return {
        "kind": "gkx_1_7_frozen_release_contract",
        "release_lanes": [
            {
                "lane": "Quasilinear diagnostics and saturation-model selection",
                "status": "closed",
                "claim_level": claim_level,
                "primary_artifacts": artifacts,
                "key_metrics": {
                    "absolute_flux_promoted": absolute_flux_promoted,
                    "uq_candidate_promotion_passed": False,
                    "dataset_sufficiency_promotion_passed": False,
                    "accepted_uq_candidates": [],
                },
            }
        ],
    }


def _candidate_uncertainty_sidecar() -> dict:
    return {
        "kind": "quasilinear_candidate_uncertainty_report",
        "claim_level": "candidate_model_development_not_runtime_option",
        "passed": False,
        "notes": (
            "Candidate retained only as a scoped rank-screening near miss, "
            "not a runtime/TOML absolute-flux predictor."
        ),
        "null_training_mean_baseline": {"mean_abs_relative_error": 0.79},
        "candidates": {
            "linear_weight": {
                "mean_abs_relative_error": 0.85,
                "promotion_eligible": True,
            },
            "linear_state_ridge": {
                "mean_abs_relative_error": 1.1,
                "promotion_eligible": False,
                "eligibility_failures": ["insufficient_train_to_parameter_ratio"],
            },
            "spectral_envelope_ridge": {
                "mean_abs_relative_error": 0.377,
                "promotion_eligible": True,
            },
        },
        "promotion_gate": {
            "passed": False,
            "accepted_candidates": [],
            "requires_beating_linear_weight_baseline": True,
            "requires_beating_training_mean_null": True,
            "transport_mean_relative_error_gate": 0.35,
        },
    }


def test_promoted_absolute_flux_requires_passed_holdout_gate_and_window_stats(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    report = tmp_path / "report.json"
    payload = _calibration_report(
        claim_level="calibrated_absolute_flux",
        passed=True,
        holdout_error=0.7,
    )
    payload["points"][1]["observed_heat_flux_std"] = float("nan")
    payload["points"][1]["nonlinear_window_stats"] = _window_stats(passed=False)
    report.write_text(json.dumps(payload), encoding="utf-8")
    doc = tmp_path / "doc.rst"
    _write_doc(doc)

    audit = mod.build_guardrail_audit([str(report)], [doc])

    assert audit["passed"] is False
    failed = {
        gate["metric"]: gate["detail"]
        for gate in audit["gate_report"]["gates"]
        if not gate["passed"]
    }
    assert "train_holdout_point_metadata" in failed
    assert "promoted_holdout_gate" in failed
    assert "promoted_holdout_window_convergence" in failed


def test_promoted_absolute_flux_requires_converged_holdout_window_metadata(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    report = tmp_path / "report.json"
    payload = _calibration_report(
        claim_level="calibrated_absolute_flux",
        passed=True,
        holdout_error=0.1,
    )
    del payload["points"][1]["nonlinear_window_stats"]
    report.write_text(json.dumps(payload), encoding="utf-8")
    doc = tmp_path / "doc.rst"
    _write_doc(doc)

    audit = mod.build_guardrail_audit([str(report)], [doc])

    assert audit["passed"] is False
    failed_metrics = {
        gate["metric"] for gate in audit["gate_report"]["gates"] if not gate["passed"]
    }
    assert "promoted_holdout_window_convergence" in failed_metrics


def test_unpromoted_report_with_finite_metadata_passes_synthetic_guardrail(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            _calibration_report(
                claim_level="calibration_dataset",
                passed=False,
                holdout_error=4.0,
            )
        ),
        encoding="utf-8",
    )
    doc = tmp_path / "doc.rst"
    _write_doc(doc)

    audit = mod.build_guardrail_audit([str(report)], [doc])

    assert audit["passed"] is True
    assert audit["calibration_reports"][0]["n_train"] == 1
    assert audit["calibration_reports"][0]["n_holdout"] == 1


def test_docs_without_nonpromotion_marker_fail_scope_check(tmp_path: Path) -> None:
    mod = _load_tool_module()
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            _calibration_report(
                claim_level="calibration_dataset",
                passed=False,
                holdout_error=4.0,
            )
        ),
        encoding="utf-8",
    )
    doc = tmp_path / "doc.rst"
    _write_doc(doc, "This section describes a calibrated absolute-flux predictor.\n")

    audit = mod.build_guardrail_audit([str(report)], [doc])

    assert audit["passed"] is False
    failed_metrics = {
        gate["metric"] for gate in audit["gate_report"]["gates"] if not gate["passed"]
    }
    assert f"doc_scope_marker:{doc}" in failed_metrics
    assert f"doc_no_absolute_flux_overclaim:{doc}" in failed_metrics


def test_wrapped_negative_absolute_flux_phrase_is_not_overclaim(tmp_path: Path) -> None:
    mod = _load_tool_module()
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            _calibration_report(
                claim_level="calibration_dataset",
                passed=False,
                holdout_error=4.0,
            )
        ),
        encoding="utf-8",
    )
    doc = tmp_path / "doc.rst"
    _write_doc(
        doc,
        (
            "This result is scoped model-development evidence, not a\n"
            "runtime/TOML absolute-flux predictor. Absolute-flux prediction not "
            "promoted.\n"
        ),
    )

    audit = mod.build_guardrail_audit([str(report)], [doc])

    assert audit["passed"] is True
    doc_rows = {row["doc"]: row for row in audit["doc_checks"]}
    assert doc_rows[str(doc)]["overclaim_lines"] == []


def test_release_contract_ql_lane_requires_scoped_nonabsolute_candidate(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    report = tmp_path / "release_contract.json"
    payload = _release_contract(
        claim_level="calibrated_absolute_flux",
        absolute_flux_promoted=True,
    )
    payload["release_lanes"][0]["key_metrics"]["uq_candidate_promotion_passed"] = True
    payload["release_lanes"][0]["key_metrics"]["accepted_uq_candidates"] = [
        "spectral_envelope_ridge"
    ]
    report.write_text(json.dumps(payload), encoding="utf-8")
    doc = tmp_path / "doc.rst"
    _write_doc(doc)

    audit = mod.build_guardrail_audit([str(report)], [doc])

    assert audit["passed"] is False
    failed_metrics = {
        gate["metric"] for gate in audit["gate_report"]["gates"] if not gate["passed"]
    }
    assert "release_contract_ql_not_absolute_flux" in failed_metrics
    assert "release_contract_ql_closed_scope_is_non_absolute" in failed_metrics
    assert "release_contract_ql_candidate_scope_not_runtime" in failed_metrics


def test_release_contract_ql_lane_requires_guardrail_artifact(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    report = tmp_path / "release_contract.json"
    report.write_text(
        json.dumps(_release_contract(include_guardrail_artifact=False)),
        encoding="utf-8",
    )
    doc = tmp_path / "doc.rst"
    _write_doc(doc)

    audit = mod.build_guardrail_audit([str(report)], [doc])

    assert audit["passed"] is False
    failed_metrics = {
        gate["metric"] for gate in audit["gate_report"]["gates"] if not gate["passed"]
    }
    assert "release_contract_ql_guardrail_artifact_listed" in failed_metrics


def test_release_contract_ql_lane_passes_when_candidate_is_scoped(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    report = tmp_path / "release_contract.json"
    report.write_text(json.dumps(_release_contract()), encoding="utf-8")
    doc = tmp_path / "doc.rst"
    _write_doc(doc)

    audit = mod.build_guardrail_audit([str(report)], [doc])

    assert audit["passed"] is True
    assert audit["summary"]["n_release_contracts"] == 1
    assert audit["release_contracts"][0]["ql_status"] == "closed"


def test_manuscript_figure_audit_requires_json_sidecar_and_index_entry(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            _calibration_report(
                claim_level="calibration_dataset",
                passed=False,
                holdout_error=4.0,
            )
        ),
        encoding="utf-8",
    )
    doc = tmp_path / "doc.rst"
    _write_doc(doc)
    figure_base = tmp_path / "docs/_static/quasilinear_candidate_uncertainty"
    figure_base.parent.mkdir(parents=True)
    figure_base.with_suffix(".png").write_bytes(b"not a real png for metadata test")
    index = tmp_path / "manuscript_figures.rst"
    index.write_text(
        (
            f"current artifact base: ``{figure_base.with_suffix('.png')}`` "
            "with PDF companion only. "
            "No runtime/TOML absolute-flux predictor; absolute-flux runtime "
            "promotion remains blocked.\n"
        ),
        encoding="utf-8",
    )

    audit = mod.build_guardrail_audit(
        [str(report)],
        [doc],
        [figure_base],
        index,
    )

    assert audit["passed"] is False
    failed_metrics = {
        gate["metric"] for gate in audit["gate_report"]["gates"] if not gate["passed"]
    }
    assert f"ql_figure_json_sidecar_exists:{figure_base}" in failed_metrics
    assert f"ql_figure_index_mentions_json_sidecar:{figure_base}" in failed_metrics


def test_manuscript_figure_audit_requires_explicit_failed_baselines(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            _calibration_report(
                claim_level="calibration_dataset",
                passed=False,
                holdout_error=4.0,
            )
        ),
        encoding="utf-8",
    )
    doc = tmp_path / "doc.rst"
    _write_doc(doc)
    figure_base = tmp_path / "docs/_static/quasilinear_candidate_uncertainty"
    figure_base.parent.mkdir(parents=True)
    figure_base.with_suffix(".png").write_bytes(b"not a real png for metadata test")
    sidecar = _candidate_uncertainty_sidecar()
    sidecar["promotion_gate"]["accepted_candidates"] = [
        "spectral_envelope_ridge",
        "linear_weight",
    ]
    figure_base.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")
    index = tmp_path / "manuscript_figures.rst"
    index.write_text(
        (
            f"current artifact base: ``{figure_base.with_suffix('.png')}`` "
            f"with JSON companion ``{figure_base.with_suffix('.json')}``. "
            "No runtime/TOML absolute-flux predictor; absolute-flux runtime "
            "promotion remains blocked.\n"
        ),
        encoding="utf-8",
    )

    audit = mod.build_guardrail_audit(
        [str(report)],
        [doc],
        [figure_base],
        index,
    )

    assert audit["passed"] is False
    failed_metrics = {
        gate["metric"] for gate in audit["gate_report"]["gates"] if not gate["passed"]
    }
    assert f"ql_figure_failed_baselines_explicit:{figure_base}" in failed_metrics


def test_manuscript_figure_audit_accepts_scoped_candidate_sidecar(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            _calibration_report(
                claim_level="calibration_dataset",
                passed=False,
                holdout_error=4.0,
            )
        ),
        encoding="utf-8",
    )
    doc = tmp_path / "doc.rst"
    _write_doc(doc)
    figure_base = tmp_path / "docs/_static/quasilinear_candidate_uncertainty"
    figure_base.parent.mkdir(parents=True)
    figure_base.with_suffix(".json").write_text(
        json.dumps(_candidate_uncertainty_sidecar()),
        encoding="utf-8",
    )
    index = tmp_path / "manuscript_figures.rst"
    index.write_text(
        (
            f"current artifact base: ``{figure_base.with_suffix('.png')}`` "
            f"with JSON companion ``{figure_base.with_suffix('.json')}``. "
            "No runtime/TOML absolute-flux predictor; absolute-flux runtime "
            "promotion remains blocked.\n"
        ),
        encoding="utf-8",
    )

    audit = mod.build_guardrail_audit(
        [str(report)],
        [doc],
        [figure_base],
        index,
    )

    assert audit["passed"] is True
    assert audit["summary"]["n_manuscript_figure_checks"] == 1
    assert audit["manuscript_figure_provenance"][0]["png_exists"] is False
    assert audit["manuscript_figure_provenance"][0]["claim_scoped"] is True
    assert audit["manuscript_figure_provenance"][0]["failed_baselines_explicit"] is True


def test_dataset_sufficiency_failure_can_be_downstream_skill_not_data_volume(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    doc = tmp_path / "doc.rst"
    _write_doc(doc)
    figure_base = tmp_path / "docs/_static/quasilinear_dataset_sufficiency"
    figure_base.parent.mkdir(parents=True)
    figure_base.with_suffix(".png").write_bytes(b"not a real png for metadata test")
    figure_base.with_suffix(".json").write_text(
        json.dumps(
            {
                "kind": "quasilinear_dataset_sufficiency",
                "claim_level": "scoped_low_parameter_candidate_promotion_not_runtime_option",
                "notes": "Dataset guard, not a runtime/TOML absolute-flux predictor.",
                "candidate_requirements": [
                    {
                        "candidate": "linear_state_ridge",
                        "data_volume_passed": True,
                    }
                ],
                "downstream_gates": {
                    "saturation_rule_sweep": {
                        "passed": False,
                        "accepted": [],
                    }
                },
                "promotion_gate": {
                    "passed": False,
                    "blockers": ["downstream_candidate_skill_gates_not_passed"],
                    "requires_downstream_candidate_skill_gates": True,
                },
                "input_validation": {
                    "passed": True,
                    "cases": [{"case": "holdout", "required": True, "passed": True}],
                },
            }
        ),
        encoding="utf-8",
    )
    index = tmp_path / "manuscript_figures.rst"
    index.write_text(
        (
            f"current artifact base: ``{figure_base.with_suffix('.png')}`` "
            f"with JSON companion ``{figure_base.with_suffix('.json')}``. "
            "No runtime/TOML absolute-flux predictor; absolute-flux runtime "
            "promotion remains blocked.\n"
        ),
        encoding="utf-8",
    )

    audit = mod.build_guardrail_audit(
        [str(figure_base.with_suffix(".json"))],
        [doc],
        [figure_base],
        index,
    )

    failed_metrics = {
        gate["metric"] for gate in audit["gate_report"]["gates"] if not gate["passed"]
    }
    assert f"ql_figure_failed_baselines_explicit:{figure_base}" not in failed_metrics


def test_tracked_quasilinear_promotion_guardrails_pass() -> None:
    mod = _load_tool_module()

    audit = mod.build_guardrail_audit(
        list(mod.DEFAULT_REPORT_PATTERNS),
        [str(path) for path in mod.DEFAULT_DOCS],
    )

    assert audit["passed"] is True
    assert audit["summary"]["n_calibration_reports"] == 4
    assert audit["summary"]["n_input_validation_reports"] >= 4
    assert audit["summary"]["n_promotion_gate_reports"] >= 4
    assert audit["summary"]["n_release_contracts"] == 1
    assert audit["summary"]["n_doc_checks"] == 4
    assert audit["summary"]["n_manuscript_figure_checks"] == len(
        mod.DEFAULT_MANUSCRIPT_FIGURE_BASES
    )


def test_guardrail_script_runs_before_editable_install(tmp_path: Path) -> None:
    root = REPO_ROOT
    out = tmp_path / "guardrails.json"
    env = {**os.environ, "PYTHONPATH": ""}

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/check.py",
            "quasilinear",
            "--out-json",
            str(out),
        ],
        cwd=root,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "quasilinear_promotion_guardrails_passed=True" in completed.stdout
    assert json.loads(out.read_text(encoding="utf-8"))["passed"] is True


# ---- from test_quasilinear_window.py ----


def _load_tool_module_quasilinear_window():
    return load_release_tool("check_nonlinear_transport_gates")


def test_autocorrelation_campaign_uses_runtime_effective_sample_count(
    tmp_path: Path,
) -> None:
    campaign = load_tool_script("campaigns", "heat_flux_autocorrelation")
    rng = np.random.default_rng(31)
    time = np.arange(400, dtype=float) * 0.2
    flux = np.empty(time.size)
    flux[0] = 4.0
    for index in range(1, flux.size):
        flux[index] = 4.0 + 0.9 * (flux[index - 1] - 4.0) + rng.normal(0.0, 0.1)
    trace = tmp_path / "trace.csv"
    np.savetxt(
        trace,
        np.column_stack((time, flux)),
        delimiter=",",
        header="t,heat_flux",
        comments="",
    )

    report = campaign.analyse(trace)

    assert report is not None
    n = report["samples_in_window"]
    expected = min(
        float(n),
        n * report["output_dt"] / (2.0 * report["tau_ac"]),
    )
    assert report["independent_samples"] == pytest.approx(expected)


def _saturated_trace() -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 200.0, 201)
    heat = 4.0 + 0.04 * np.sin(2.0 * np.pi * t / 10.0)
    return t, heat


def test_converged_saturated_transport_window_passes_with_finite_uncertainty() -> None:
    t, heat = _saturated_trace()

    report = nonlinear_window_convergence_report(
        t,
        heat,
        case="synthetic_saturated_itg",
        source_artifact="synthetic.csv",
        config=NonlinearWindowConvergenceConfig(
            transient_fraction=0.5,
            min_samples=64,
            min_blocks=4,
            max_running_mean_rel_drift=0.02,
            max_sem_rel=0.02,
        ),
    )

    assert report["passed"] is True
    assert report["statistics"]["late_mean"] == pytest.approx(4.0, abs=5.0e-3)
    assert np.isfinite(report["statistics"]["block_bootstrap_sem"])
    assert np.isfinite(report["statistics"]["sem"])
    assert report["statistics"]["terminal_mean_rel_delta"] < 0.02
    assert report["window"]["transient_cutoff"] == pytest.approx(100.0)
    ready, failures = nonlinear_window_stats_promotion_ready(report)
    assert ready is True
    assert failures == []


def test_window_report_supports_explicit_bounds_and_deterministic_blocks() -> None:
    t, heat = _saturated_trace()

    report = nonlinear_window_convergence_report(
        t,
        heat,
        case="bounded_saturated_itg",
        source_artifact="bounded.csv",
        config=NonlinearWindowConvergenceConfig(
            tmin=50.0,
            tmax=150.0,
            transient_fraction=0.25,
            block_size=8,
            bootstrap_samples=0,
            min_samples=40,
            min_blocks=4,
            max_running_mean_rel_drift=0.03,
            max_sem_rel=0.03,
        ),
    )

    assert report["passed"] is False
    assert report["window"]["selected_tmin"] == pytest.approx(50.0)
    assert report["window"]["selected_tmax"] == pytest.approx(150.0)
    assert report["statistics"]["block_size"] == 8
    assert report["statistics"]["block_bootstrap_sem"] is None
    failed = {gate["metric"] for gate in report["gates"] if not gate["passed"]}
    assert failed == {"block_bootstrap_sem"}
    # With no bootstrap samples, the diagnostic SEM falls back to sample/block SEM,
    # but promotion still fails closed because bootstrap uncertainty was requested.
    assert np.isfinite(report["statistics"]["sem"])


def test_transient_only_trace_fails_running_mean_gate() -> None:
    t = np.linspace(0.0, 120.0, 121)
    heat = 0.05 * t

    report = nonlinear_window_convergence_report(
        t,
        heat,
        case="ramping_transient",
        source_artifact="ramp.csv",
        config=NonlinearWindowConvergenceConfig(
            transient_fraction=0.5,
            min_samples=32,
            max_running_mean_rel_drift=0.05,
            max_sem_rel=1.0,
        ),
    )

    failed = {gate["metric"] for gate in report["gates"] if not gate["passed"]}
    assert report["passed"] is False
    assert "running_mean_drift" in failed


def test_terminal_subwindow_gate_blocks_cancelled_running_mean_drift() -> None:
    t = np.arange(96.0)
    heat = np.ones_like(t)
    heat[48:] = np.concatenate(
        [
            np.ones(24),
            np.zeros(12),
            2.0 * np.ones(12),
        ]
    )

    report = nonlinear_window_convergence_report(
        t,
        heat,
        case="terminal_drift_hidden_by_half_means",
        source_artifact="terminal.csv",
        config=NonlinearWindowConvergenceConfig(
            transient_fraction=0.5,
            min_samples=48,
            min_blocks=4,
            max_running_mean_rel_drift=0.01,
            terminal_fraction=0.25,
            min_terminal_samples=8,
            max_terminal_mean_rel_delta=0.20,
            max_sem_rel=10.0,
        ),
    )

    failed = {gate["metric"] for gate in report["gates"] if not gate["passed"]}
    assert report["passed"] is False
    assert "terminal_mean_agreement" in failed
    assert "running_mean_drift" not in failed
    assert report["statistics"]["terminal_n_samples"] == 12
    assert report["statistics"]["terminal_mean_rel_delta"] == pytest.approx(1.0)


def test_nonlinear_window_ensemble_gate_accepts_seed_replicates() -> None:
    import gkx as sgk

    assert sgk.NonlinearWindowEnsembleConfig is NonlinearWindowEnsembleConfig
    assert sgk.nonlinear_window_ensemble_report is nonlinear_window_ensemble_report

    t, heat = _saturated_trace()
    reports = [
        nonlinear_window_convergence_report(
            t,
            heat + offset,
            case=f"seed_{idx}",
            source_artifact=f"seed_{idx}.csv",
            config=NonlinearWindowConvergenceConfig(
                transient_fraction=0.5,
                min_samples=64,
                min_blocks=4,
                max_running_mean_rel_drift=0.02,
                max_sem_rel=0.02,
            ),
        )
        for idx, offset in enumerate((-0.02, 0.0, 0.02))
    ]

    report = nonlinear_window_ensemble_report(
        reports,
        case="seed_uncertainty_gate",
        comparison="random_seed_replicates",
        config=NonlinearWindowEnsembleConfig(
            min_reports=3,
            max_mean_rel_spread=0.02,
            max_combined_sem_rel=0.02,
        ),
    )

    assert report["passed"] is True
    assert report["statistics"]["n_reports"] == 3
    assert report["statistics"]["mean_rel_spread"] == pytest.approx(0.01)
    assert report["rows"][0]["source_artifact"] == "seed_0.csv"
    assert {gate["metric"] for gate in report["gates"] if not gate["passed"]} == set()


def test_nonlinear_window_ensemble_gate_blocks_spread_and_failed_inputs() -> None:
    t, heat = _saturated_trace()
    good = nonlinear_window_convergence_report(
        t,
        heat,
        case="good_seed",
        source_artifact="good.csv",
        config=NonlinearWindowConvergenceConfig(
            transient_fraction=0.5,
            min_samples=64,
            max_running_mean_rel_drift=0.02,
            max_sem_rel=0.02,
        ),
    )
    drifted = nonlinear_window_convergence_report(
        t,
        heat + 2.0,
        case="drifted_seed",
        source_artifact="drifted.csv",
        config=NonlinearWindowConvergenceConfig(
            transient_fraction=0.5,
            min_samples=64,
            max_running_mean_rel_drift=0.02,
            max_sem_rel=0.02,
        ),
    )
    failed = dict(good)
    failed["case"] = "failed_seed"
    failed["passed"] = False
    failed["gate_report"] = {"passed": False}

    report = nonlinear_window_ensemble_report(
        [good, drifted, failed],
        config=NonlinearWindowEnsembleConfig(
            min_reports=3,
            max_mean_rel_spread=0.05,
            max_combined_sem_rel=0.02,
        ),
    )

    failed_metrics = {gate["metric"] for gate in report["gates"] if not gate["passed"]}
    assert report["passed"] is False
    assert "individual_windows_passed" in failed_metrics
    assert "mean_relative_spread" in failed_metrics
    assert report["rows"][2]["promotion_ready"] is False


def test_small_window_and_nan_late_window_fail() -> None:
    t = np.linspace(0.0, 10.0, 11)
    heat = np.ones_like(t)
    small = nonlinear_window_convergence_report(
        t,
        heat,
        case="small_window",
        source_artifact="small.csv",
        config=NonlinearWindowConvergenceConfig(
            transient_fraction=0.5,
            min_samples=16,
            min_blocks=4,
        ),
    )
    small_failed = {gate["metric"] for gate in small["gates"] if not gate["passed"]}
    assert "finite_sample_count" in small_failed

    t2, heat2 = _saturated_trace()
    heat2[-3] = np.nan
    nan_report = nonlinear_window_convergence_report(
        t2,
        heat2,
        case="nan_late_window",
        source_artifact="nan.csv",
        config=NonlinearWindowConvergenceConfig(
            transient_fraction=0.5,
            min_samples=64,
        ),
    )
    nan_failed = {gate["metric"] for gate in nan_report["gates"] if not gate["passed"]}
    assert nan_report["passed"] is False
    assert "finite_late_window" in nan_failed


def test_nonlinear_window_convergence_subcommand_writes_json(tmp_path: Path) -> None:
    mod = _load_tool_module_quasilinear_window()
    t, heat = _saturated_trace()
    csv = tmp_path / "trace.csv"
    csv.write_text(
        "t,heat_flux\n"
        + "\n".join(f"{ti:.8g},{qi:.12g}" for ti, qi in zip(t, heat, strict=True))
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "window.json"

    assert (
        mod.main(
            [
                "convergence",
                "--csv",
                str(csv),
                "--out-json",
                str(out),
                "--min-samples",
                "64",
                "--max-sem-rel",
                "0.02",
                "--max-running-mean-rel-drift",
                "0.02",
            ]
        )
        == 0
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["provenance"]["source_artifact"] == str(csv)


def test_nonlinear_window_script_imports_before_editable_install() -> None:
    root = REPO_ROOT
    env = {**os.environ, "PYTHONPATH": ""}

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/check.py",
            "nonlinear-transport",
            "convergence",
            "--help",
        ],
        cwd=root,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Check nonlinear late-window convergence metadata" in completed.stdout


def test_nonlinear_window_config_and_input_validation_are_fail_closed() -> None:
    t = np.linspace(0.0, 10.0, 11)
    heat = np.ones_like(t)

    bad_configs = [
        (NonlinearWindowConvergenceConfig(tmin=np.nan), "tmin must be finite"),
        (NonlinearWindowConvergenceConfig(tmax=np.inf), "tmax must be finite"),
        (NonlinearWindowConvergenceConfig(tmin=5.0, tmax=5.0), "tmin must be less"),
        (
            NonlinearWindowConvergenceConfig(transient_fraction=1.0),
            "transient_fraction",
        ),
        (NonlinearWindowConvergenceConfig(min_samples=1), "min_samples"),
        (NonlinearWindowConvergenceConfig(min_blocks=1), "min_blocks"),
        (NonlinearWindowConvergenceConfig(block_size=0), "block_size"),
        (NonlinearWindowConvergenceConfig(bootstrap_samples=-1), "bootstrap_samples"),
        (
            NonlinearWindowConvergenceConfig(max_running_mean_rel_drift=-1.0),
            "running_mean",
        ),
        (NonlinearWindowConvergenceConfig(terminal_fraction=0.0), "terminal_fraction"),
        (NonlinearWindowConvergenceConfig(min_terminal_samples=0), "min_terminal"),
        (
            NonlinearWindowConvergenceConfig(max_terminal_mean_rel_delta=-1.0),
            "terminal_mean",
        ),
        (NonlinearWindowConvergenceConfig(max_sem_rel=-1.0), "max_sem_rel"),
        (NonlinearWindowConvergenceConfig(value_floor=0.0), "value_floor"),
    ]
    for config, message in bad_configs:
        with pytest.raises(ValueError, match=message):
            nonlinear_window_convergence_report(t, heat, config=config)

    with pytest.raises(ValueError, match="same length"):
        nonlinear_window_convergence_report([0.0, 1.0], [1.0])
    with pytest.raises(ValueError, match="must not be empty"):
        nonlinear_window_convergence_report([], [])
    with pytest.raises(ValueError, match="time contains non-finite"):
        nonlinear_window_convergence_report([0.0, np.nan], [1.0, 1.0])
    with pytest.raises(ValueError, match="selected nonlinear window is empty"):
        nonlinear_window_convergence_report([1.0], [1.0])

    bad_ensemble_configs = [
        (NonlinearWindowEnsembleConfig(min_reports=1), "min_reports"),
        (NonlinearWindowEnsembleConfig(max_mean_rel_spread=-1.0), "mean_rel_spread"),
        (NonlinearWindowEnsembleConfig(max_combined_sem_rel=-1.0), "combined_sem"),
        (NonlinearWindowEnsembleConfig(value_floor=0.0), "value_floor"),
    ]
    for config, message in bad_ensemble_configs:
        with pytest.raises(ValueError, match=message):
            nonlinear_window_ensemble_report([], config=config)
    with pytest.raises(TypeError, match="report dictionaries"):
        nonlinear_window_ensemble_report(
            [object()],  # type: ignore[list-item]
            config=NonlinearWindowEnsembleConfig(min_reports=2),
        )


def test_nonlinear_window_csv_and_summary_loaders_cover_artifact_contracts(
    tmp_path: Path,
) -> None:
    csv = tmp_path / "diagnostics.csv"
    t, heat = _saturated_trace()
    csv.write_text(
        "time,q_i\n"
        + "\n".join(f"{ti:.8g},{qi:.12g}" for ti, qi in zip(t, heat, strict=True))
        + "\n",
        encoding="utf-8",
    )
    config = NonlinearWindowConvergenceConfig(
        transient_fraction=0.5,
        min_samples=64,
        max_running_mean_rel_drift=0.02,
        max_sem_rel=0.02,
    )

    from_csv = nonlinear_window_convergence_from_csv(
        csv,
        time_column="time",
        value_column="q_i",
        case="csv_case",
        config=config,
        summary_artifact="summary.json",
    )

    assert from_csv["passed"] is True
    assert from_csv["case"] == "csv_case"
    assert from_csv["observable"] == "q_i"
    assert from_csv["provenance"]["summary_artifact"] == "summary.json"

    summary = tmp_path / "window_summary.json"
    summary.write_text(
        json.dumps(
            {
                "case": "summary_case",
                "gkx": "diagnostics.csv",
                "tmin": 50.0,
                "tmax": 200.0,
            }
        ),
        encoding="utf-8",
    )
    from_summary = nonlinear_window_convergence_from_summary(
        summary,
        time_column="time",
        value_column="q_i",
        config=config,
    )

    assert from_summary["case"] == "summary_case"
    assert from_summary["provenance"]["summary_artifact"] == str(summary)
    assert from_summary["provenance"]["source_artifact"].endswith("diagnostics.csv")

    missing_column = tmp_path / "missing_column.csv"
    missing_column.write_text("time,other\n0.0,1.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="observable column"):
        nonlinear_window_convergence_from_csv(
            missing_column,
            time_column="time",
            value_column="q_i",
        )

    bad_summary = tmp_path / "bad_summary.json"
    bad_summary.write_text(json.dumps({"other": "diagnostics.csv"}), encoding="utf-8")
    with pytest.raises(ValueError, match="diagnostics source"):
        nonlinear_window_convergence_from_summary(bad_summary)

    txt_summary = tmp_path / "txt_summary.json"
    txt_summary.write_text(json.dumps({"gkx": "trace.txt"}), encoding="utf-8")
    (tmp_path / "trace.txt").write_text("not,csv\n", encoding="utf-8")
    with pytest.raises(NotImplementedError, match="diagnostics CSV"):
        nonlinear_window_convergence_from_summary(txt_summary)


def test_nonlinear_window_promotion_ready_reports_all_missing_contracts() -> None:
    ready, failures = nonlinear_window_stats_promotion_ready(None)
    assert ready is False
    assert failures == ["missing nonlinear_window_stats object"]

    ready, failures = nonlinear_window_stats_promotion_ready(
        {
            "kind": "wrong",
            "passed": False,
            "provenance": {},
            "statistics": {"late_mean": 1.0},
            "window": {"transient_fraction": 0.0, "n_finite_late": 0},
            "gate_report": {"passed": False},
        }
    )

    assert ready is False
    assert "unexpected nonlinear_window_stats kind" in failures
    assert "nonlinear window convergence report did not pass" in failures
    assert "missing nonlinear source_artifact provenance" in failures
    assert "missing/non-finite statistics.sem" in failures
    assert "missing/non-finite statistics.terminal_mean_rel_delta" in failures
    assert "missing/non-finite window.late_tmin" in failures
    assert "missing declared transient cutoff policy" in failures
    assert "window has no finite late samples" in failures
    assert "missing passed gate_report" in failures


def test_nonlinear_window_promotion_rejects_string_pass_flags() -> None:
    t, heat = _saturated_trace()
    report = nonlinear_window_convergence_report(
        t,
        heat,
        source_artifact="persisted.csv",
        config=NonlinearWindowConvergenceConfig(
            transient_fraction=0.5,
            min_samples=64,
            max_running_mean_rel_drift=0.02,
            max_sem_rel=0.02,
        ),
    )
    report["passed"] = "true"
    report["gate_report"]["passed"] = "true"

    ready, failures = nonlinear_window_stats_promotion_ready(report)

    assert ready is False
    assert "nonlinear window convergence report did not pass" in failures
    assert "missing passed gate_report" in failures


def test_nonlinear_window_ensemble_promotion_rejects_string_pass_flags() -> None:
    t, heat = _saturated_trace()
    windows = [
        nonlinear_window_convergence_report(
            t,
            heat + offset,
            source_artifact=f"seed_{index}.csv",
            config=NonlinearWindowConvergenceConfig(
                transient_fraction=0.5,
                min_samples=64,
                max_running_mean_rel_drift=0.02,
                max_sem_rel=0.02,
            ),
        )
        for index, offset in enumerate((-0.01, 0.01))
    ]
    report = nonlinear_window_ensemble_report(windows)
    report["passed"] = "true"
    report["gate_report"]["passed"] = "true"
    report["rows"][0]["promotion_ready"] = "true"

    ready, failures = nonlinear_window_stats_promotion_ready(report)

    assert ready is False
    assert "nonlinear window ensemble report did not pass" in failures
    assert "missing passed ensemble gate_report" in failures
    assert "not all ensemble rows are promotion-ready" in failures


def _spinup_then_plateau(
    plateau: float = 5.0, *, drift_per_time: float = 0.0, seed: int = 7
) -> tuple[np.ndarray, np.ndarray]:
    """Exponential growth to an overshoot, then a correlated noisy plateau."""

    rng = np.random.default_rng(seed)
    t = np.arange(6000) * 0.05
    noise = np.zeros(t.size)
    for i in range(1, t.size):
        noise[i] = 0.9 * noise[i - 1] + rng.normal(0.0, 0.25)
    growth = np.minimum(1.0e-3 * np.exp(0.35 * t), 2.0 * plateau)
    saturated = plateau + drift_per_time * t + noise
    return t, np.where(t < 30.0, growth, saturated)


def test_saturation_stop_decision_stops_converged_trace_and_excludes_spinup() -> None:
    t, heat = _spinup_then_plateau()

    decision = saturation_stop_decision(
        t, heat, guard=heat**2, config=SaturationStopConfig(rel_sem=0.05)
    )

    assert decision["saturated"] is True
    assert decision["reasons"] == []
    # The exponential spin-up (heat < median until it nears the plateau) must
    # not pollute the window: the mean lands on the plateau, not on the
    # overshoot, and the window starts after the early growth phase.
    assert decision["window_tmin"] > 20.0
    assert decision["mean"] == pytest.approx(5.0, abs=0.5)
    assert decision["sem"] > 0.0
    assert decision["rel_sem"] <= 0.05
    assert decision["tau_ac_resolved"] is True
    assert decision["window_span"] >= decision["min_window"]


def test_saturation_stop_decision_rejects_non_stationary_trace() -> None:
    t, heat = _spinup_then_plateau(drift_per_time=0.02)

    decision = saturation_stop_decision(t, heat)

    assert decision["saturated"] is False
    assert "window_not_stationary" in decision["reasons"]


def test_saturation_stop_decision_rejects_unresolved_or_short_traces() -> None:
    t = np.arange(2000) * 0.05
    growing = 1.0e-3 * np.exp(0.1 * t)

    decision = saturation_stop_decision(t, growing)
    assert decision["saturated"] is False
    assert decision["reasons"]

    short = saturation_stop_decision(t[:128], growing[:128])
    assert short["saturated"] is False
    assert short["reasons"] == ["trace_shorter_than_min_samples"]
    assert short["mean"] is None


def test_saturation_stop_decision_guard_blocks_drifting_field_energy() -> None:
    t, heat = _spinup_then_plateau()
    _, drifting_guard = _spinup_then_plateau(drift_per_time=0.05, seed=11)

    decision = saturation_stop_decision(t, heat, guard=drifting_guard)

    assert decision["saturated"] is False
    assert "guard_not_stationary" in decision["reasons"]
    assert decision["guard_stationary"] is False


def test_saturation_stop_decision_guard_blocks_drifting_free_energy() -> None:
    t, heat = _spinup_then_plateau()
    _, drifting_wg = _spinup_then_plateau(drift_per_time=0.05, seed=11)

    decision = saturation_stop_decision(t, heat, free_energy_guard=drifting_wg)

    assert decision["saturated"] is False
    assert "Wg_guard_not_stationary" in decision["reasons"]
    assert decision["Wg_guard_stationary"] is False


def test_saturation_stop_decision_honors_min_window_override() -> None:
    t, heat = _spinup_then_plateau()

    derived = saturation_stop_decision(
        t, heat, config=SaturationStopConfig(min_window=0.0)
    )
    assert derived["min_window"] == pytest.approx(20.0 * derived["tau_ac"])

    decision = saturation_stop_decision(
        t, heat, config=SaturationStopConfig(rel_sem=0.05, min_window=1.0e6)
    )

    assert decision["saturated"] is False
    assert "window_below_min_window" in decision["reasons"]
    assert decision["min_window"] == pytest.approx(1.0e6)

    for invalid in (float("nan"), float("inf"), -1.0):
        with pytest.raises(ValueError, match="finite and non-negative"):
            saturation_stop_decision(
                t, heat, config=SaturationStopConfig(min_window=invalid)
            )


def test_saturation_stop_decision_applies_mandatory_sample_floor(monkeypatch) -> None:
    time = np.arange(257, dtype=float)
    values = np.concatenate(([0.0], 10.0 + np.cos(np.arange(256) / 10.0)))
    config = SaturationStopConfig(min_samples=1, min_window=0.0)

    short = saturation_stop_decision(time[:-1], values[:-1], config=config)
    boundary = saturation_stop_decision(time, values, config=config)
    strengthened = saturation_stop_decision(
        time, values, config=SaturationStopConfig(min_samples=300)
    )

    assert short["reasons"] == ["post_spinup_window_too_short"]
    assert boundary["n_window"] == 256
    assert boundary["min_samples"] == 256
    assert boundary["min_iat_spans"] == pytest.approx(20.0)
    assert boundary["min_window"] == pytest.approx(20.0 * boundary["tau_ac"])
    assert "window_below_min_window" not in boundary["reasons"]
    assert strengthened["reasons"] == ["trace_shorter_than_min_samples"]
    assert strengthened["min_samples"] == 300

    monkeypatch.setattr(
        "gkx.diagnostics.saturation._sokal_window_mean_sem",
        lambda x, _dt: (float(np.mean(x)), 0.01, 1.0, True),
    )
    constant = np.full(256, 10.0)
    exact = saturation_stop_decision(np.linspace(0.0, 20.0, 256), constant)
    below = saturation_stop_decision(
        np.linspace(0.0, np.nextafter(20.0, 0.0), 256), constant
    )
    assert exact["saturated"] is True
    assert exact["window_span"] == exact["min_window"] == pytest.approx(20.0)
    assert below["reasons"] == ["window_below_min_window"]


def test_saturation_stop_decision_refuses_a_trace_that_never_left_zero() -> None:
    """A flux that is identically zero has no saturated mean to find.

    The relative SEM divides by a floor when the mean is zero, so without a
    signal gate every other criterion passes on a dead trace and the run stops
    in its first chunk. A zonal-response case, whose heat flux is zero by
    construction, was truncated at t=7.7 of a requested 60 that way.
    """

    time = np.linspace(0.0, 60.0, 600)
    decision = saturation_stop_decision(time, np.zeros_like(time))

    assert decision["saturated"] is False
    assert "flux_indistinguishable_from_zero" in decision["reasons"]


def test_saturation_stop_decision_still_stops_a_small_but_real_flux() -> None:
    """The signal gate must not reject a converged run for being faint.

    The fluctuation here is correlated, because that is what separates a faint
    real flux from a flat one. This test originally used white noise about the
    same mean; that trace is now refused, on purpose, for having no resolved
    correlation time -- see the stationary-from-t=0 test below. Faintness is
    still not a reason to refuse: three parts in a billion saturates.
    """

    rng = np.random.default_rng(0)
    time = np.linspace(0.0, 60.0, 600)
    noise = np.zeros(time.size)
    for index in range(1, time.size):
        noise[index] = 0.9 * noise[index - 1] + rng.normal(0.0, 1.0)
    flux = 3.0e-9 * (1.0 + 0.03 * noise / np.std(noise))

    decision = saturation_stop_decision(time, flux)

    assert decision["saturated"] is True
    assert decision["reasons"] == []
    assert decision["mean"] == pytest.approx(3.0e-9, rel=0.05)
    assert decision["tau_ac"] > 0.0


def test_saturation_stop_decision_refuses_a_flux_stationary_from_t_zero() -> None:
    """A flux that never left its starting value is not a saturated flux.

    The zero-flux guard is an absolute threshold, so it only covers traces that
    happen to sit below it. A run whose heat flux is flat from the first sample
    but a few decades above that floor passed every other gate and stopped in
    its first chunk, at any amplitude: the correlation time of such a trace
    crosses zero at lag one, which made ``tau_ac`` exactly zero, which made the
    derived IAT-scaled ``min_window`` zero as well. The window-length
    requirement therefore vanished precisely on the traces carrying the least
    information, and the relative SEM of a long stretch of uncorrelated samples
    is tiny.

    Requiring only ``tau_ac > 0`` does not close it. The lag-one sample
    autocorrelation of uncorrelated noise is positive about half the time, and
    each such realization integrates to a small positive ``tau_ac`` that reads
    as resolved: before the sampling-interval floor, 190 of the 400 draws below
    saturated. So this sweeps realizations rather than checking one, and checks
    two amplitudes because the defect is scale-free while the zero-flux guard
    is not -- scaling a trace cannot change its autocorrelation, so the two
    levels must give identical verdicts.
    """

    time = np.linspace(0.0, 60.0, 600)
    sampling_interval = float(np.median(np.diff(time)))
    saturated_draws = {1.0e-8: 0, 1.0e2: 0}
    for seed in range(400):
        shape = 1.0 + 0.01 * np.random.default_rng(seed).standard_normal(time.size)
        for level in saturated_draws:
            decision = saturation_stop_decision(time, level * shape)
            saturated_draws[level] += bool(decision["saturated"])
            assert decision["tau_ac"] <= sampling_interval, (seed, level)

    assert saturated_draws == {1.0e-8: 0, 1.0e2: 0}, saturated_draws


def test_saturation_stop_decision_still_stops_a_run_that_started_saturated() -> None:
    """A warm-started run has no growth phase and must still be allowed to stop.

    This is the case that rules out the obvious alternative fix. Requiring the
    trace to have risen above its own early-time level before a plateau counts
    would also refuse the flat traces above -- but it would refuse these too,
    and these are the runs the stop policy is worth the most on: a run seeded
    from a converged neighbour begins at its saturated flux by construction and
    never grows. Its late/early ratio is 1.02 against 65 for a run that spun up
    from a small perturbation, so no threshold on growth separates it from a
    dead trace. What separates them is that this one has a correlation time.
    """

    time = np.linspace(0.0, 60.0, 600)
    stopped = 0
    for seed in range(40):
        rng = np.random.default_rng(seed)
        noise = np.zeros(time.size)
        for index in range(1, time.size):
            noise[index] = 0.7 * noise[index - 1] + rng.normal(0.0, 1.0)
        flux = 5.0 * (1.0 + 0.05 * noise / np.std(noise))
        stopped += bool(saturation_stop_decision(time, flux)["saturated"])

    # Not all 40: the half-window stationarity test rejects a minority of
    # realizations on its own, which is its job and predates this gate.
    assert stopped >= 30, stopped


def _stationary_ar1(rho: float, *, seed: int, draws: int = 256) -> np.ndarray:
    """Stationary unit-variance Gaussian draws; no fitted burn-in or scaling."""
    noise = np.random.default_rng(seed).standard_normal((draws, 4096))
    for i in range(1, noise.shape[1]):
        noise[:, i] = rho * noise[:, i - 1] + np.sqrt(1 - rho**2) * noise[:, i]
    return noise


@pytest.mark.parametrize("rho", [0.0, 0.75, 0.95])
def test_correlated_sem_has_fixed_horizon_ar1_coverage(
    rho: float, record_property
) -> None:
    """Parker et al. (2018), arXiv:1807.04779, Eqs. 9–12: covariance sum.

    Predeclared 256 draws, n=4096, seeds 20260912 + 100*rho; 95% interval
    coverage 90–99%, RMS SEM/exact SEM 0.8–1.2. These broad Monte Carlo
    regression gates are not a sequential-stop or turbulent-flux certificate.
    """
    series = _stationary_ar1(rho, seed=20260912 + int(100 * rho))
    n = series.shape[1]
    lags = np.arange(1, n)
    # Exact finite-n variance from Cov(X_i, X_j) = rho**abs(i-j), not
    # the production IAT estimator or its asymptotic effective sample count.
    exact_variance = (n + 2 * np.sum((n - lags) * rho**lags)) / n**2
    stats = np.array([_sokal_window_mean_sem(row, 1.0)[:2] for row in series])
    coverage = np.mean(np.abs(stats[:, 0]) <= 1.96 * stats[:, 1])
    variance_ratio = np.mean(stats[:, 1] ** 2) / exact_variance
    record_property("coverage_95", float(coverage))
    record_property("rms_sem_over_exact", float(np.sqrt(variance_ratio)))
    assert 0.90 <= coverage <= 0.99, (rho, coverage)
    assert 0.8**2 <= variance_ratio <= 1.2**2, (rho, variance_ratio)
    # Mutation control: treating correlated outputs as independent must fail.
    if rho > 0:
        naive = series.std(axis=1, ddof=1) / np.sqrt(n)
        naive_coverage = np.mean(np.abs(stats[:, 0]) <= 1.96 * naive)
        record_property("naive_coverage_95", float(naive_coverage))
        assert naive_coverage < 0.75


@pytest.mark.parametrize(
    "checkpoints",
    [(512, 1024, 2048, 4096), tuple(range(128, 4097, 128))],
    ids=("sparse_reference", "production_cadence"),
)
def test_saturation_causal_prefixes_reject_strong_ar1_drift(
    checkpoints: tuple[int, ...], record_property
) -> None:
    """Bounded negative control, not a proof of sequential interval coverage.

    Flegal–Gong (2015), arXiv:1303.0238, motivates testing the stopping rule,
    not just fixed-window SEM. Predeclared rho=.75, seed=20260913, 128 draws,
    and <=5% ever-stopped rate. Test both the reference checkpoints and the
    production 128-step cadence. The mean rises eight noise standard deviations
    per 512 samples; every prefix is drifting.
    """
    time = np.arange(4096, dtype=float)
    series = 10.0 + time / 64.0 + _stationary_ar1(0.75, seed=20260913, draws=128)
    stopped = sum(
        any(
            saturation_stop_decision(time[:n], row[:n])["saturated"]
            for n in checkpoints
        )
        for row in series
    )
    record_property("checkpoint_schedule", ",".join(map(str, checkpoints)))
    record_property("false_stops", stopped)
    assert stopped / len(series) <= 0.05, stopped


def test_saturation_production_cadence_stops_stationary_ar1(record_property) -> None:
    """The paired stationary control retains >=90% stopping power by 4096."""
    time = np.arange(4096, dtype=float)
    series = 10.0 + _stationary_ar1(0.75, seed=20260913, draws=128)
    stopped = sum(
        any(
            saturation_stop_decision(time[:n], row[:n])["saturated"]
            for n in range(128, 4097, 128)
        )
        for row in series
    )
    record_property("stops", stopped)
    assert stopped / len(series) >= 0.90, stopped


def test_sokal_reports_a_constant_trace_as_unresolved() -> None:
    """A constant signal has no correlation time, so it must not report one.

    Callers read "resolved" as ``cut < rho.size``. The degenerate return placed
    the cut at lag zero, which reads as resolved for any non-empty ``rho`` --
    the opposite of what a zero-variance trace has shown them.
    """

    tau, cut, rho = sokal_autocorrelation_time(np.zeros(64), 0.05)

    assert tau == 0.0
    assert cut >= rho.size
