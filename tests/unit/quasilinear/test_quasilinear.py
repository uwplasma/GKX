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
)
from gkx.runtime import (
    build_runtime_linear_params,
    build_runtime_linear_terms,
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
import scripts.checks._gates.quasilinear_calibration as qlc
from scripts.checks._gates.quasilinear_calibration import (
    QuasilinearCalibrationPoint,
    apply_heat_flux_scale,
    calibration_point_from_nonlinear_window_summary,
    calibration_point_from_spectrum_and_nonlinear_window,
    fit_train_heat_flux_scale,
    integrated_quasilinear_flux_from_spectrum,
    quasilinear_calibration_report,
    write_quasilinear_calibration_report,
)
from scripts.checks._gates.transport_windows import (
    NonlinearWindowConvergenceConfig,
    nonlinear_window_convergence_report,
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
    assert scale_fit["scale"] == pytest.approx(4.0)
    scaled = apply_heat_flux_scale(points, scale=scale_fit["scale"])
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
