"""Reference-comparison maintainer tool contracts."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from functools import partial
import pytest
from support.paths import REPO_ROOT
from gkx.solvers_nonlinear_diagnostics import DIVERGENCE_KNEE_STEPS
from support.paths import load_release_tool
from scripts.checks._gates.portfolio_guard import (
    ReducedPortfolioArtifactGuardConfig,
    reduced_portfolio_artifact_guard_report,
)


# ---- imported-linear growth-dump mode ----

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "comparison"))


# ---- test_compare_gx_imported_linear.py ----


# ---- test_compare_gx_nonlinear_diagnostics.py ----


# ---- test_compare_gx_nonlinear_terms.py ----


# ---- test_compare_gx_rhs_terms.py ----


# ---- test_reference_comparison_contracts.py ----


# ---- from test_nonlinear_gradient_evidence_contracts.py ----
# The nonlinear-autodiff claims must stay attached to their generators.
#
# Commits 612e1311 and a7b41968 removed the tools and the JSON that produced the
# headline adjoint numbers, leaving the docs page and the figure resting on
# literals typed into the plotting script. These tests pin the other direction:
# the tracked measurements exist, the figure reads them, the generators that write


STATIC = REPO_ROOT / "docs" / "_static"
LADDER = STATIC / "nonlinear_heat_flux_gradient_window_rk3.json"
PARITY = STATIC / "nonlinear_window_device_parity.json"
MEMORY = (
    STATIC / "nonlinear_adjoint_checkpointing_cpu32.json",
    STATIC / "nonlinear_adjoint_checkpointing_gpu32.json",
)


def test_tracked_ladder_agrees_with_finite_differences_below_the_knee() -> None:
    ladder = json.loads(LADDER.read_text())
    below = [r for r in ladder["rows"] if r["window"] <= DIVERGENCE_KNEE_STEPS]
    above = [r for r in ladder["rows"] if r["window"] > DIVERGENCE_KNEE_STEPS]
    assert below and above
    assert max(r["ad_fd_relative_error"] for r in below) < 1.0e-8
    assert min(r["ad_fd_relative_error"] for r in above) > 1.0e-6
    # A ladder that never diverges would satisfy the two bounds above only by
    # accident; require the gradient itself to take off past the knee.
    assert above[0]["abs_gradient"] > 10.0 * below[-1]["abs_gradient"]


@pytest.mark.parametrize("path", MEMORY)
def test_checkpoint_memory_profiles_show_a_real_reduction(path: Path) -> None:
    profile = json.loads(path.read_text())
    policies = {row["checkpoint"]: row for row in profile["rows"]}
    assert set(policies) == {"step", "block"}
    assert policies["block"]["temp_bytes"] < policies["step"]["temp_bytes"]
    assert profile["temp_reduction"] > 10.0
    # Rematerialization is the trade; a profile claiming free memory would mean
    # the two policies did not compile to different programs.
    assert profile["runtime_ratio"] > 1.0


def test_device_parity_artifact_compares_one_identical_case() -> None:
    parity = json.loads(PARITY.read_text())
    assert len(parity["runs"]) >= 2
    backends = {run["default_backend"] for run in parity["runs"].values()}
    assert {"cpu", "gpu"} <= backends
    comparisons = {(row["left"], row["right"]): row for row in parity["comparisons"]}
    assert comparisons
    for row in comparisons.values():
        assert row["gradient_relative_difference"] < 1.0e-12
        assert row["value_relative_difference"] < 1.0e-12


# ---- from test_nonlinear_sharding_profile_contracts.py ----


# Sweep-driver contracts for the same nonlinear sharding profiling lane.


# ---- from test_runtime_and_scaling_profile_contracts.py ----


def test_sspx3_stage_profile_preserves_identity_without_speedup_claim() -> None:
    profile = json.loads(
        (REPO_ROOT / "docs/_static/linear_sspx3_stage_profile.json").read_text(
            encoding="utf-8"
        )
    )

    assert profile["identity_gate"]["passed"] is True
    assert profile["before"]["finite"] is True
    assert profile["after"]["finite"] is True
    assert profile["performance_gate"]["measurable_speedup"] is False
    assert profile["performance_gate"]["passed"] is False
    assert abs(profile["performance_gate"]["relative_median_time_change"]) < 0.03


def test_prepared_nonlinear_cpu_gpu_profiles_are_matched_and_clean() -> None:
    cpu = json.loads(
        (
            REPO_ROOT / "docs/_static/prepared_nonlinear_runtime_cpu_profile.json"
        ).read_text(encoding="utf-8")
    )
    gpu = json.loads(
        (
            REPO_ROOT / "docs/_static/prepared_nonlinear_runtime_gpu_profile.json"
        ).read_text(encoding="utf-8")
    )

    for profile in (cpu, gpu):
        assert profile["git_revision"] == cpu["git_revision"]
        assert profile["git_dirty"] is False
        assert profile["reuse_prepared_simulation"] is True
        assert profile["resolved_diagnostics"] is False
        assert profile["steps"] == 200
        assert profile["method"] == "rk3"
        assert profile["fixed_dt"] is False
        assert profile["sample_stride"] == 10
        assert profile["diagnostics_stride"] == 10
        assert profile["software"] == cpu["software"]
        assert "result_summary" in profile
        assert profile["memory_summary"]["host_peak_rss_bytes"] > 0
    assert cpu["backend"] == "cpu"
    assert gpu["backend"] == "gpu"
    assert cpu["run_median_s"] / gpu["run_median_s"] >= 5.0
    expected_shapes = {
        "time": [21],
        "final_state": [1, 4, 8, 64, 64, 24],
        "phi": [64, 64, 24],
        "heat_flux": [21],
        "dt": [21],
    }
    for name, expected_shape in expected_shapes.items():
        assert (
            cpu["result_summary"][name]["shape"] == gpu["result_summary"][name]["shape"]
        )
        assert cpu["result_summary"][name]["shape"] == expected_shape
        assert cpu["result_summary"][name]["finite_fraction"] == 1.0
        assert gpu["result_summary"][name]["finite_fraction"] == 1.0
        # The full nonlinear state is a more sensitive accumulated trajectory
        # than the scalar diagnostics after 200 adaptive steps.  Keep its
        # observed CPU/GPU norm drift explicit instead of letting the old
        # mislabeled time vector provide a falsely exact state gate.
        rtol = 1.0e-3 if name == "final_state" else 1.0e-5
        np.testing.assert_allclose(
            cpu["result_summary"][name]["l2_norm"],
            gpu["result_summary"][name]["l2_norm"],
            rtol=rtol,
            atol=1.0e-12,
        )


def test_resolved_diagnostic_profiles_are_identity_gated_and_bounded() -> None:
    for backend in ("cpu", "gpu"):
        compact = json.loads(
            (
                REPO_ROOT
                / f"docs/_static/prepared_nonlinear_runtime_{backend}_profile.json"
            ).read_text(encoding="utf-8")
        )
        resolved = json.loads(
            (
                REPO_ROOT
                / f"docs/_static/prepared_nonlinear_runtime_{backend}_resolved_profile.json"
            ).read_text(encoding="utf-8")
        )
        assert compact["git_revision"] == resolved["git_revision"]
        assert compact["software"] == resolved["software"]
        assert compact["resolved_diagnostics"] is False
        assert resolved["resolved_diagnostics"] is True
        assert compact["steps"] == resolved["steps"] == 200
        assert resolved["run_median_s"] / compact["run_median_s"] <= 1.25
        assert (
            resolved["memory_summary"]["host_peak_rss_bytes"]
            / compact["memory_summary"]["host_peak_rss_bytes"]
            <= 1.10
        )
        for name in ("time", "final_state", "phi", "heat_flux", "dt"):
            np.testing.assert_allclose(
                compact["result_summary"][name]["l2_norm"],
                resolved["result_summary"][name]["l2_norm"],
                rtol=1.0e-7,
                atol=1.0e-12,
            )
        if backend == "gpu":
            compact_peak = compact["memory_summary"]["device_stats"][
                "peak_bytes_in_use"
            ]
            resolved_peak = resolved["memory_summary"]["device_stats"][
                "peak_bytes_in_use"
            ]
            assert resolved_peak / compact_peak <= 1.10


# Tracked parallel and RHS-term profile artifacts. Their profilers were retired
# in SLIM-SCRIPTS tranche 3 (recovery SHA in plan/research/2026-09-22-slim-tools/MAP.md);
# the committed JSONs stay as fixtures and keep these contracts.


def test_tracked_mixed_species_hermite_profile_is_scoped_and_identity_gated() -> None:
    artifact = (
        REPO_ROOT / "docs" / "_static" / "linear_rhs_species_hermite_profile_cpu.json"
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))

    assert payload["decomposition_axis"] == "species_hermite"
    assert payload["requested_devices"] == 4
    assert payload["actual_devices"] == 4
    assert payload["identity_passed"] is True
    integration = payload["integration"]
    assert integration["identity_passed"] is True
    assert integration["speedup_passed"] is (integration["speedup"] > 1.0)
    assert integration["state_identity"]["max_abs_error"] <= payload["atol"]
    assert integration["field_history_identity"]["max_abs_error"] <= payload["atol"]
    assert (
        "mixed species-Hermite collision-free integration" in integration["claim_scope"]
    )
    assert payload["max_rel_error"] <= payload["rtol"]
    assert payload["max_abs_error"] <= payload["atol"]
    assert payload["max_phi_abs_error"] <= payload["atol"]
    assert payload["speedup"] > 1.0
    assert "not a GPU or general scaling claim" in payload["claim_scope"]
    assert len(payload["git_revision"]) == 40


def test_linear_rhs_terms_tracked_miller_profile_is_active_artifact() -> None:
    path = REPO_ROOT / "docs" / "_static" / "linear_rhs_terms_profile_miller_cpu.json"
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["kind"] == "linear_rhs_terms_profile_summary"
    assert payload["case"] == "runtime_cyclone_nonlinear_miller"
    assert payload["state"] == "z_wave_linear_kick"
    assert payload["full_linear_rhs_seconds"] > 0.0
    assert payload["rows"]["streaming"]["norm"] > 0.0
    assert payload["dominant_nonzero_norm_term"] == "streaming"


# ---- from test_check_vmec_boozer_gates.py ----


holdout_mod = load_release_tool("check_vmec_boozer_gates")


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _training_samples() -> list[dict[str, object]]:
    return [
        {"surface_index": None, "alpha": 0.0, "selected_ky_index": ky, "weight": 0.5}
        for ky in (1, 2)
    ]


def _training_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    return (
        _write_json(tmp_path / "aggregate.json", _aggregate_payload()),
        _write_json(tmp_path / "line_search.json", _line_search_payload()),
    )


def _alpha_holdout(tmp_path: Path) -> Path:
    return _write_json(
        tmp_path / "alpha_holdout.json",
        {
            "promotion_gate": {"passed": True},
            "claim_level": "passed_grid_convergence_candidate_for_transport_holdout",
            "samples": [{"surface_index": None, "alpha": 0.75, "selected_ky_index": 1}],
        },
    )


def _aggregate_payload() -> dict[str, object]:
    return {
        "kind": "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
        "passed": True,
        "claim_scope": "reduced aggregate objective plumbing",
        "samples": _training_samples(),
    }


def _line_search_payload() -> dict[str, object]:
    return {
        "kind": "vmec_boozer_aggregate_scalar_objective_line_search_report",
        "passed": True,
        "samples": _training_samples(),
    }


def _ensemble_payload(*, passed: bool = True) -> dict[str, object]:
    return {
        "kind": "nonlinear_window_ensemble_report",
        "claim_level": "replicated_nonlinear_window_uncertainty_gate_not_simulation_claim",
        "passed": passed,
        "gate_report": {"passed": passed},
    }


def test_aggregate_holdout_gate_blocks_without_surface_or_field_line_holdout(
    tmp_path: Path,
) -> None:
    aggregate, line_search = _training_artifacts(tmp_path)

    report = holdout_mod.check_vmec_boozer_aggregate_holdout_gate(
        aggregate_artifact=aggregate,
        line_search_artifact=line_search,
    )

    assert report["passed"] is False
    assert report["promotion_gate"]["blockers"] == [
        "passed_holdout_surface_or_field_line_artifact",
        "passed_replicated_nonlinear_window_ensemble",
    ]
    assert report["training_sample_summary"]["alphas"] == ["0"]


def test_aggregate_holdout_gate_rejects_ky_only_holdout(tmp_path: Path) -> None:
    aggregate, line_search = _training_artifacts(tmp_path)
    ky_only = _write_json(
        tmp_path / "ky_only.json",
        {
            "passed": True,
            "claim_level": "passed_grid_convergence_candidate_for_transport_holdout",
            "samples": [{"surface_index": None, "alpha": 0.0, "selected_ky_index": 7}],
        },
    )

    report = holdout_mod.check_vmec_boozer_aggregate_holdout_gate(
        aggregate_artifact=aggregate,
        line_search_artifact=line_search,
        holdout_artifacts=(ky_only,),
    )

    assert report["passed"] is False
    assert report["holdout_artifacts"][0]["passed"] is True
    assert report["holdout_artifacts"][0]["heldout_surface_or_field_line"] is False
    assert "k_y-only" in report["promotion_gate"]["requirements"][4]


def test_aggregate_holdout_gate_accepts_passed_field_line_holdout(
    tmp_path: Path,
) -> None:
    aggregate, line_search = _training_artifacts(tmp_path)
    ensemble = _write_json(tmp_path / "ensemble.json", _ensemble_payload())
    holdout = _alpha_holdout(tmp_path)

    report = holdout_mod.check_vmec_boozer_aggregate_holdout_gate(
        aggregate_artifact=aggregate,
        line_search_artifact=line_search,
        holdout_artifacts=(holdout,),
        nonlinear_ensemble_artifacts=(ensemble,),
    )

    assert report["passed"] is True
    assert report["promotion_gate"]["blockers"] == []
    assert report["holdout_artifacts"][0]["qualifies_for_promotion"] is True
    assert (
        report["nonlinear_ensemble_artifacts"][0][
            "qualifies_for_production_nonlinear_promotion"
        ]
        is True
    )
    assert report["gates"][-2]["detail"].endswith("held-out field-line alpha=0.75")


def test_aggregate_holdout_gate_rejects_non_ensemble_nonlinear_artifact(
    tmp_path: Path,
) -> None:
    aggregate, line_search = _training_artifacts(tmp_path)
    holdout = _alpha_holdout(tmp_path)
    single_window = _write_json(
        tmp_path / "single_window.json",
        {
            "kind": "nonlinear_window_convergence_report",
            "passed": True,
            "gate_report": {"passed": True},
        },
    )

    report = holdout_mod.check_vmec_boozer_aggregate_holdout_gate(
        aggregate_artifact=aggregate,
        line_search_artifact=line_search,
        holdout_artifacts=(holdout,),
        nonlinear_ensemble_artifacts=(single_window,),
    )

    assert report["passed"] is False
    assert report["promotion_gate"]["blockers"] == [
        "passed_replicated_nonlinear_window_ensemble"
    ]
    assert (
        report["nonlinear_ensemble_artifacts"][0]["is_nonlinear_window_ensemble"]
        is False
    )


def test_aggregate_holdout_gate_records_readiness_manifest_blockers(
    tmp_path: Path,
) -> None:
    aggregate, line_search = _training_artifacts(tmp_path)
    holdout = _alpha_holdout(tmp_path)
    manifest = _write_json(
        tmp_path / "manifest.json",
        {
            "kind": "nonlinear_window_ensemble_readiness_manifest",
            "passed": False,
            "promotion_gate": {
                "passed": False,
                "blockers": ["seed_and_timestep_replicates_present"],
            },
            "missing_artifacts": [
                {"case": "case_a", "variant_axis": "seed", "missing_count": 2}
            ],
        },
    )

    report = holdout_mod.check_vmec_boozer_aggregate_holdout_gate(
        aggregate_artifact=aggregate,
        line_search_artifact=line_search,
        holdout_artifacts=(holdout,),
        nonlinear_ensemble_artifacts=(manifest,),
    )

    row = report["nonlinear_ensemble_artifacts"][0]
    assert report["passed"] is False
    assert row["is_nonlinear_window_readiness_manifest"] is True
    assert row["readiness_blockers"] == ["seed_and_timestep_replicates_present"]
    assert row["missing_artifacts"][0]["variant_axis"] == "seed"
    assert row["qualifies_for_production_nonlinear_promotion"] is False


def test_aggregate_holdout_gate_rejects_non_promotable_holdout_scope(
    tmp_path: Path,
) -> None:
    aggregate, line_search = _training_artifacts(tmp_path)
    startup_holdout = _write_json(
        tmp_path / "startup_holdout.json",
        {
            "passed": True,
            "claim_level": "startup_transient_nonlinear_plumbing_fd_audit_not_transport_average",
            "transport_average_gate": False,
            "heldout_samples": [
                {"surface_index": 3, "alpha": 0.0, "selected_ky_index": 1},
            ],
        },
    )

    report = holdout_mod.check_vmec_boozer_aggregate_holdout_gate(
        aggregate_artifact=aggregate,
        line_search_artifact=line_search,
        holdout_artifacts=(startup_holdout,),
    )

    assert report["passed"] is False
    assert report["holdout_artifacts"][0]["n_samples"] == 1
    assert report["holdout_artifacts"][0]["heldout_surface_or_field_line"] is True
    assert report["holdout_artifacts"][0]["qualifies_for_promotion"] is False
    assert (
        "transport_average_gate_false"
        in report["holdout_artifacts"][0]["claim_scope_blockers"]
    )
    assert (
        "passed_replicated_nonlinear_window_ensemble"
        in report["promotion_gate"]["blockers"]
    )


def test_aggregate_holdout_gate_main_writes_json(tmp_path: Path) -> None:
    aggregate, line_search = _training_artifacts(tmp_path)
    out = tmp_path / "report.json"

    result = holdout_mod.main_aggregate_holdout(
        f"--aggregate-artifact {aggregate} --line-search-artifact {line_search} --json-out {out}".split()
    )

    assert result == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["kind"] == "vmec_boozer_aggregate_holdout_promotion_gate"
    assert saved["passed"] is False


# VMEC/Boozer reduced portfolio guard assertions
portfolio_mod = holdout_mod


def _row_artifact() -> dict[str, object]:
    samples = [
        {"surface_index": None, "alpha": 0.0, "selected_ky_index": 1, "weight": 0.25},
        {"surface_index": None, "alpha": 0.0, "selected_ky_index": 2, "weight": 0.25},
        {"surface_index": None, "alpha": 0.5, "selected_ky_index": 1, "weight": 0.25},
        {"surface_index": None, "alpha": 0.5, "selected_ky_index": 2, "weight": 0.25},
    ]
    return {
        "kind": "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
        "artifact_kind": "vmec_boozer_multi_point_objective_gate",
        "builder": "scripts/artifacts/build_vmec_boozer_aggregate_objective_gate.py multi-point",
        "passed": True,
        "source_scope": "mode21_vmec_boozer_state_multi_point",
        "claim_scope": "real VMEC/Boozer reduced QL rows; not a nonlinear turbulent transport claim",
        "next_action": "Nonlinear transport optimization still requires separate long-window gates.",
        "objective": "quasilinear_flux",
        "reduction": "mean",
        "input_path": "/tmp/input.nfp4_QH_warm_start",
        "wout_path": "/tmp/wout_nfp4_QH_warm_start.nc",
        "options": {"mboz": 21, "nboz": 21},
        "n_samples": 4,
        "samples": samples,
        "objective_names": [
            "gamma",
            "omega",
            "kperp_eff2",
            "mixing_length_heat_flux_proxy",
        ],
        "minus_sample_values": [0.7, 0.9, 0.8, 1.0],
        "base_sample_values": [0.8, 1.0, 0.9, 1.1],
        "plus_sample_values": [0.9, 1.1, 1.0, 1.2],
        "minus_objective_table": [
            [0.10, -0.2, 0.5, 0.7],
            [0.12, -0.3, 0.6, 0.9],
            [0.11, -0.1, 0.4, 0.8],
            [0.13, -0.4, 0.7, 1.0],
        ],
        "base_objective_table": [
            [0.11, -0.2, 0.5, 0.8],
            [0.13, -0.3, 0.6, 1.0],
            [0.12, -0.1, 0.4, 0.9],
            [0.14, -0.4, 0.7, 1.1],
        ],
        "plus_objective_table": [
            [0.12, -0.2, 0.5, 0.9],
            [0.14, -0.3, 0.6, 1.1],
            [0.13, -0.1, 0.4, 1.0],
            [0.15, -0.4, 0.7, 1.2],
        ],
        "base_value": 0.95,
        "minus_value": 0.85,
        "plus_value": 1.05,
        "central_derivative": 1.0,
        "response_abs": 0.2,
        "curvature_ratio": 0.01,
        "finite_values": True,
        "finite_difference_consistent": True,
        "response_resolved": True,
    }


_ROW_TABLE_KEYS = tuple(
    f"{side}_{kind}"
    for kind in ("sample_values", "objective_table")
    for side in ("base", "minus", "plus")
)


def _write_guard_inputs(tmp_path: Path, row: dict[str, object]) -> tuple[Path, Path]:
    row_path = _write_json(tmp_path / "row.json", row)
    return row_path, _write_json(tmp_path / "gradient.json", _gradient_artifact())


def _gradient_artifact() -> dict[str, object]:
    return {
        "kind": "mode21_vmec_boozer_quasilinear_gradient_gate",
        "passed": True,
        "objective_gates": [
            {
                "objective": "gamma",
                "parameter": "Rcos_mid_surface_m1",
                "passed": True,
                "implicit": 10.0,
                "finite_difference": 10.01,
                "abs_error": 0.01,
                "rel_error": 0.001,
            },
            {
                "objective": "mixing_length_heat_flux_proxy",
                "parameter": "Rcos_mid_surface_m1",
                "passed": True,
                "implicit": 20.0,
                "finite_difference": 20.02,
                "abs_error": 0.02,
                "rel_error": 0.001,
            },
        ],
    }


def test_reduced_portfolio_guard_passes_real_metadata_contract() -> None:
    report = reduced_portfolio_artifact_guard_report(
        _row_artifact(),
        gradient_artifacts=[_gradient_artifact()],
    )

    assert report["passed"] is True
    assert report["provenance_gate"]["passed"] is True
    assert report["coverage_gate"]["n_alphas"] == 2
    assert report["coverage_gate"]["n_ky"] == 2
    assert report["portfolio_reducer_gate"]["contract"]["row_shape"] == [1, 2, 2, 1]
    assert report["ad_fd_gradient_gate"]["has_growth_ad_fd_gate"] is True
    assert report["ad_fd_gradient_gate"]["has_quasilinear_ad_fd_gate"] is True
    assert report["claim_scope_gate"]["passed"] is True


@pytest.mark.parametrize(
    ("reduction", "base_value", "expected_shape"),
    [("weighted_mean", 0.95, [1, 2, 2, 1]), ("max", 1.0, [1, 2, 2, 1])],
)
def test_reduced_portfolio_guard_accepts_declared_reducer_semantics(
    reduction: str,
    base_value: float,
    expected_shape: list[int],
) -> None:
    artifact = _row_artifact()
    artifact["reduction"] = reduction
    if reduction == "max":
        artifact["base_sample_values"] = [0.5, 0.75, 0.875, 1.0]
    artifact["base_value"] = base_value

    report = reduced_portfolio_artifact_guard_report(
        artifact,
        gradient_artifacts=[_gradient_artifact()],
    )

    assert report["passed"] is True
    assert report["portfolio_reducer_gate"]["reduction"] == reduction
    assert report["portfolio_reducer_gate"]["contract"]["row_shape"] == expected_shape


def test_reduced_portfolio_guard_distinguishes_physical_torflux_surfaces() -> None:
    artifact = _row_artifact()
    artifact["samples"] = [
        {
            "surface_index": None,
            "torflux": torflux,
            "surface": torflux,
            "alpha": 0.0,
            "ky": ky,
            "selected_ky_index": index,
            "weight": 0.25,
        }
        for torflux in (0.5, 0.7)
        for index, ky in ((1, 0.1), (2, 0.2))
    ]
    report = reduced_portfolio_artifact_guard_report(
        artifact,
        gradient_artifacts=[_gradient_artifact()],
        config=ReducedPortfolioArtifactGuardConfig(min_alphas=1),
    )

    assert report["passed"] is True
    assert report["coverage_gate"]["n_surfaces"] == 2
    assert report["coverage_gate"]["n_alphas"] == 1
    assert report["coverage_gate"]["n_ky"] == 2
    assert report["portfolio_reducer_gate"]["contract"]["row_shape"] == [2, 1, 2, 1]


def test_reduced_portfolio_guard_rejects_duplicate_or_incomplete_sample_grids() -> None:
    duplicate = _row_artifact()
    duplicate["samples"] = [
        {"surface_index": None, "alpha": 0.0, "selected_ky_index": 1, "weight": 0.25},
        {"surface_index": None, "alpha": 0.0, "selected_ky_index": 1, "weight": 0.25},
        {"surface_index": None, "alpha": 0.5, "selected_ky_index": 1, "weight": 0.25},
        {"surface_index": None, "alpha": 0.5, "selected_ky_index": 2, "weight": 0.25},
    ]
    with pytest.raises(ValueError, match="duplicate"):
        reduced_portfolio_artifact_guard_report(
            duplicate,
            gradient_artifacts=[_gradient_artifact()],
        )

    incomplete = _row_artifact()
    for key in ("samples", *_ROW_TABLE_KEYS):
        incomplete[key] = incomplete[key][:-1]  # type: ignore[index]
    with pytest.raises(ValueError, match="complete rectangular"):
        reduced_portfolio_artifact_guard_report(
            incomplete,
            gradient_artifacts=[_gradient_artifact()],
        )


def test_reduced_portfolio_guard_marks_bad_gradient_gate_without_crashing() -> None:
    bad_gradient = {
        "kind": "mode21_vmec_boozer_quasilinear_gradient_gate",
        "passed": False,
        "objective_gates": [
            {
                "objective": "gamma",
                "passed": True,
                "implicit": 1.0,
                "finite_difference": float("nan"),
                "abs_error": 0.0,
                "rel_error": 0.0,
            },
            "not-a-dict",
        ],
    }

    report = reduced_portfolio_artifact_guard_report(
        _row_artifact(),
        gradient_artifacts=[bad_gradient],  # type: ignore[list-item]
    )

    assert report["passed"] is False
    assert report["ad_fd_gradient_gate"]["passed"] is False
    assert report["ad_fd_gradient_gate"]["finite_ad_fd_values"] is False


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda artifact: artifact.update({"reduction": "median"}),
            "artifact reduction",
        ),
        (
            lambda artifact: artifact.update({"base_sample_values": [0.8, 1.0]}),
            "base_sample_values",
        ),
        (
            lambda artifact: artifact.update(
                {"base_objective_table": [0.8, 1.0, 0.9, 1.1]}
            ),
            "two-dimensional",
        ),
        (
            lambda artifact: artifact.update(
                {"samples": [*artifact["samples"][:-1], "bad-sample"]}
            ),
            "all samples",
        ),
    ],
)
def test_reduced_portfolio_guard_rejects_malformed_artifact_shapes(
    mutator,
    message: str,
) -> None:
    artifact = _row_artifact()
    mutator(artifact)

    with pytest.raises(ValueError, match=message):
        reduced_portfolio_artifact_guard_report(
            artifact,
            gradient_artifacts=[_gradient_artifact()],
        )


def test_reduced_portfolio_guard_reports_objective_and_provenance_blockers() -> None:
    artifact = _row_artifact()
    artifact["objective_names"] = ["omega", "kperp_eff2"]
    artifact["options"] = {"mboz": 8, "nboz": 8}
    artifact["input_path"] = ""
    artifact["wout_path"] = ""

    report = reduced_portfolio_artifact_guard_report(
        artifact,
        gradient_artifacts=[_gradient_artifact()],
    )

    assert report["passed"] is False
    assert report["objective_name_gate"]["passed"] is False
    assert report["objective_name_gate"]["has_growth_objective"] is False
    assert report["objective_name_gate"]["has_quasilinear_objective"] is False
    assert report["provenance_gate"]["passed"] is False
    assert report["provenance_gate"]["mboz"] == 8
    assert report["provenance_gate"]["has_input_and_wout_paths"] is False


def test_reduced_portfolio_guard_reports_unresolved_finite_difference_diagnostics() -> (
    None
):
    artifact = _row_artifact()
    artifact["response_resolved"] = False
    artifact["finite_difference_consistent"] = False
    artifact["plus_value"] = float("inf")

    report = reduced_portfolio_artifact_guard_report(
        artifact,
        gradient_artifacts=[_gradient_artifact()],
    )

    assert report["passed"] is False
    assert report["finite_difference_gate"]["passed"] is False
    assert report["finite_difference_gate"]["response_resolved"] is False
    assert report["finite_difference_gate"]["finite_difference_consistent"] is False
    assert (
        report["finite_difference_gate"]["finite_scalar_fields"]["plus_value"] is False
    )


def test_reduced_portfolio_guard_fails_single_alpha_or_missing_gradient_gate() -> None:
    artifact = _row_artifact()
    artifact["samples"] = _training_samples()
    artifact["base_sample_values"] = [0.8, 1.0]
    artifact["minus_sample_values"] = [0.7, 0.9]
    artifact["plus_sample_values"] = [0.9, 1.1]
    artifact["base_objective_table"] = [[0.11, -0.2, 0.5, 0.8], [0.13, -0.3, 0.6, 1.0]]
    artifact["minus_objective_table"] = [[0.10, -0.2, 0.5, 0.7], [0.12, -0.3, 0.6, 0.9]]
    artifact["plus_objective_table"] = [[0.12, -0.2, 0.5, 0.9], [0.14, -0.3, 0.6, 1.1]]
    artifact["base_value"] = 0.9
    artifact["minus_value"] = 0.8
    artifact["plus_value"] = 1.0

    report = reduced_portfolio_artifact_guard_report(artifact, gradient_artifacts=[])

    assert report["passed"] is False
    assert report["coverage_gate"]["passed"] is False
    assert report["ad_fd_gradient_gate"]["passed"] is False


def test_reduced_portfolio_guard_fails_production_nonlinear_claim() -> None:
    artifact = _row_artifact()
    artifact["objective"] = "nonlinear_heat_flux"
    artifact["claim_scope"] = (
        "production nonlinear turbulent transport optimization claim"
    )

    report = reduced_portfolio_artifact_guard_report(
        artifact,
        gradient_artifacts=[_gradient_artifact()],
    )

    assert report["passed"] is False
    assert report["claim_scope_gate"]["passed"] is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_alphas": 0}, "min_alphas"),
        ({"min_ky": 0}, "min_ky"),
        ({"min_objectives": 0}, "min_objectives"),
        ({"min_boozer_mode": 0}, "min_boozer_mode"),
        ({"value_rtol": -1.0e-8}, "tolerances"),
        ({"value_atol": -1.0e-8}, "tolerances"),
    ],
)
def test_reduced_portfolio_guard_validates_config(
    kwargs: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ReducedPortfolioArtifactGuardConfig(**kwargs)


def test_tool_writes_guard_artifact(tmp_path: Path) -> None:
    row_path, gradient_path = _write_guard_inputs(tmp_path, _row_artifact())
    out_path = tmp_path / "guard.json"

    payload = portfolio_mod.build_vmec_boozer_reduced_portfolio_guard_payload(
        row_artifact=row_path,
        gradient_artifacts=[gradient_path],
    )
    written = portfolio_mod.write_vmec_boozer_reduced_portfolio_guard_artifact(
        payload, out=out_path
    )

    assert Path(written) == out_path
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["passed"] is True
    assert data["row_artifact"] == str(row_path)


def test_tool_exposes_reducer_value_tolerances(tmp_path: Path) -> None:
    row = _row_artifact()
    row["base_value"] = 0.95000004
    row_path, gradient_path = _write_guard_inputs(tmp_path, row)

    build = partial(
        portfolio_mod.build_vmec_boozer_reduced_portfolio_guard_payload,
        row_artifact=row_path,
        gradient_artifacts=[gradient_path],
    )
    strict_payload = build()
    loose_payload = build(value_rtol=1.0e-6, value_atol=1.0e-6)

    assert strict_payload["portfolio_reducer_gate"]["passed"] is False
    assert strict_payload["passed"] is False
    assert loose_payload["portfolio_reducer_gate"]["passed"] is True
    assert loose_payload["passed"] is True


def test_tool_main_returns_nonzero_for_failed_guard(tmp_path: Path) -> None:
    row = _row_artifact()
    row["options"] = {"mboz": 8, "nboz": 8}
    row_path, gradient_path = _write_guard_inputs(tmp_path, row)

    result = portfolio_mod.main_reduced_portfolio_guard(
        [
            "--row-artifact",
            str(row_path),
            "--gradient-artifact",
            str(gradient_path),
            "--out",
            str(tmp_path / "guard.json"),
        ]
    )

    assert result == 1
