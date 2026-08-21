"""Tests for quasilinear calibration artifact helpers."""

from __future__ import annotations

import json
from pathlib import Path

from support.paths import load_artifact_tool

import numpy as np
import pytest

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
    mismatch_summary.write_text(
        json.dumps({"gkx": str(mismatch)}), encoding="utf-8"
    )
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
    summary.write_text(
        json.dumps({"case": "c", "gkx": str(diag)}), encoding="utf-8"
    )

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
    summary_missing_t.write_text(
        json.dumps({"gkx": str(missing_t)}), encoding="utf-8"
    )
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
