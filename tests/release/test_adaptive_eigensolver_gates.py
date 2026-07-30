"""Release gates for the adaptive linear eigensolver evidence."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).resolve().parents[2]
STATIC = REPOSITORY / "docs" / "_static"
STELLARATOR_CASES = ("qa", "qh", "qi")


def _load(name: str) -> dict[str, object]:
    with (STATIC / name).open(encoding="utf-8") as stream:
        value = json.load(stream)
    assert isinstance(value, dict)
    return value


@pytest.mark.parametrize("case", STELLARATOR_CASES)
def test_adaptive_propagator_dense_oracle_accuracy_gate(case: str) -> None:
    """The certified cold solver must retain its dense eigenpair agreement."""

    report = _load(f"adaptive_propagator_cold_{case}_rung4_validation.json")
    assert report["schema_version"] == 5
    assert report["passed"] is True
    assert report["accuracy_verified"] is True
    assert report["accuracy_passed"] is True
    assert report["residual_passed"] is True
    assert report["reference_identity_verified"] is True
    assert report["claim_scope"] == "dense-oracle accuracy and cold/warm performance"

    rows = report["rows"]
    assert isinstance(rows, list) and len(rows) == 1
    row = rows[0]
    assert row["n"] == 4480
    assert row["converged"] is True
    assert row["stability_passed"] is True
    assert row["relative_error"] <= 5.0e-10
    assert row["residual"] <= 1.0e-8
    assert row["eigenvector_overlap"] >= 1.0 - 1.0e-10

    provenance = report["provenance"]
    assert provenance["jax_x64"] is True
    assert provenance["adaptive_candidates"] >= 2
    assert provenance["restart_krylov_dim"] < provenance["krylov_dim"]
    assert len(provenance["gkx_commit"]) == 40
    assert len(provenance["solvax_commit"]) == 40


@pytest.mark.parametrize("case", STELLARATOR_CASES)
def test_adaptive_propagator_true_cold_speed_gate(case: str) -> None:
    """Fresh processes must beat dense formation and solve, not only warm runs."""

    report = _load(f"adaptive_propagator_true_cold_{case}_rung4_validation.json")
    assert report["schema_version"] == 5
    assert report["passed"] is True
    assert report["residual_passed"] is True
    assert report["reference_identity_verified"] is True
    assert report["accuracy_verified"] is False
    assert report["accuracy_passed"] is None
    assert "continuous residual only" in report["claim_scope"]

    rows = report["rows"]
    assert isinstance(rows, list) and len(rows) == 1
    row = rows[0]
    adaptive_seconds = float(row["cold_total_seconds"])
    dense_seconds = float(row["reference_dense_cold_total_seconds"])
    assert math.isfinite(adaptive_seconds) and adaptive_seconds > 0.0
    assert math.isfinite(dense_seconds) and dense_seconds > 0.0
    assert dense_seconds / adaptive_seconds >= 1.5
    assert row["reference_verified"] is True
    assert row["converged"] is True
    assert row["stability_passed"] is True
    assert row["residual"] <= 1.0e-8


def test_adaptive_objective_cold_reverse_ad_gate() -> None:
    """The production-sized cold value-and-gradient path must beat dense AD."""

    report = _load("adaptive_objective_gradient_cold_n6144_validation.json")
    assert report["schema_version"] == 1
    assert report["passed"] is True
    assert report["accuracy_passed"] is True
    assert report["cold_speed_required"] is True
    assert report["cold_speed_passed"] is True
    assert report["resolution"]["operator_size"] >= 6144
    assert report["cold_speedup_over_dense"] >= 1.1
    assert report["value_relative_error"] <= 1.0e-8
    assert report["gradient_relative_error"] <= 1.0e-7

    adaptive = report["adaptive"]
    dense = report["dense"]
    assert adaptive["finite"] is True
    assert dense["finite"] is True
    assert (
        adaptive["cold_value_and_gradient_seconds"]
        < dense["cold_value_and_gradient_seconds"]
    )
    assert adaptive["adaptive_config"]["candidate_count"] >= 2
    assert adaptive["adaptive_config"]["sensitivity_solver"] == "propagator"


def test_adaptive_objective_physics_and_ad_matrix_gate() -> None:
    """Branch, field, geometry, species, and layout coverage must stay complete."""

    report = _load("adaptive_objective_physics_matrix.json")
    assert report["schema_version"] == 1
    assert report["passed"] is True
    assert "not velocity-space convergence" in report["scope"]
    rows = report["rows"]
    assert isinstance(rows, list) and len(rows) == 7
    assert {row["case"] for row in rows} == {
        "itg-circular",
        "etg-linked",
        "tem-electromagnetic-linked",
        "kbm-electromagnetic-linked",
        "itg-miller",
        "itg-qhs",
        "itg-qi",
    }
    assert {row["branch"] for row in rows} == {"ITG", "ETG", "TEM", "KBM"}
    assert {row["field_model"] for row in rows} == {
        "electrostatic",
        "electromagnetic",
    }
    assert {row["geometry_model"] for row in rows} == {"s-alpha", "miller", "vmec"}
    assert {row["spectral_layout"] for row in rows} == {"periodic", "linked"}
    assert {row["species"] for row in rows} == {1, 2}
    for row in rows:
        assert row["passed"] is True
        assert row["failure"] is None
        assert row["dense"]["finite"] is True
        assert row["adaptive"]["finite"] is True
        assert row["value_relative_error"] <= 1.0e-8
        assert row["gradient_relative_error"] <= 1.0e-7


def test_adaptive_real_branch_crossing_continuation_gate() -> None:
    """A prior left/right pair must retain one real mode after its rank changes."""

    report = _load("adaptive_propagator_branch_crossing_validation.json")
    assert report["schema_version"] == 1
    assert report["passed"] is True
    assert "real Cyclone linear-operator growth-order exchange" in report["scope"]
    assert all(report["checks"].values())

    exchange = report["exchange"]
    assert exchange["parameter"] == "R_over_LTi"
    assert exchange["first_subdominant_parameter"] == pytest.approx(13.25)
    assert exchange["subdominant_row_count"] >= 3
    assert exchange["adaptive_subdominant_selection_count"] >= 3

    thresholds = report["thresholds"]
    rows = report["rows"]
    assert isinstance(rows, list) and len(rows) >= 6
    assert {row["dense_tracked_growth_rank"] for row in rows} >= {0, 1, 2}
    for index, row in enumerate(rows):
        assert row["adaptive_converged"] is True
        assert row["adaptive_stable"] is True
        assert row["adaptive_residual"] <= thresholds["residual"]
        assert (
            row["adaptive_relative_eigenvalue_error"]
            <= thresholds["relative_eigenvalue_error"]
        )
        assert (
            row["adaptive_dense_right_overlap"]
            >= thresholds["dense_eigenvector_overlap"]
        )
        assert (
            row["adaptive_selected_spectral_gap"]
            >= thresholds["selected_complex_spectral_gap"]
        )
        if index:
            assert row["adaptive_continued"] is True
            assert row["adaptive_continuation_passed"] is True
            assert (
                row["adaptive_continuation_overlap"]
                >= thresholds["continuation_overlap"]
            )
        if row["dense_tracked_growth_rank"] > 0:
            assert row["adaptive_selected_candidate_index"] > 0


def test_qi_frequency_near_miss_remains_explicit_negative_evidence() -> None:
    """Direct frequency convergence must not hide a full-mode near miss."""

    report = _load("adaptive_propagator_convergence_qi_frequency_validation.json")
    assert report["schema_version"] == 1
    assert report["passed"] is False
    assert report["required_observable"] == "frequency"
    assert report["certified"] is True
    assert report["all_growth_converged"] is True
    assert report["all_frequency_converged"] is True
    assert report["all_eigenvalue_converged"] is False

    provenance = report["provenance"]
    tolerance = provenance["convergence_tolerance"]
    assert tolerance == pytest.approx(0.05)
    assert provenance["ntheta"] == 64
    assert provenance["warm_repeats"] == 0

    devices = report["devices"]
    assert isinstance(devices, list) and len(devices) == 1
    qi = devices[0]
    assert qi["device"] == "qi"
    assert qi["certified"] is True
    assert qi["growth_converged"] is True
    assert qi["frequency_converged"] is True
    assert qi["eigenvalue_converged"] is False
    assert all(change < tolerance for change in qi["growth_relative_changes"][-2:])
    assert all(change < tolerance for change in qi["frequency_normalized_changes"][-2:])
    assert all(change >= tolerance for change in qi["eigenvalue_relative_changes"][-2:])


def test_qi_resolution_branch_switch_remains_explicit_negative_evidence() -> None:
    """A certified rightmost-mode exchange must not be called convergence."""

    report = _load("adaptive_propagator_convergence_qi_branch_switch_checkpoint.json")
    assert report["schema_version"] == 1
    assert report["complete"] is False
    assert "not a convergence certificate" in report["scope"]
    assert report["device"] == "qi"

    rows = report["rows"]
    assert isinstance(rows, list) and len(rows) == 3
    assert rows[1]["continuation_overlap"] >= 0.99
    assert rows[2]["continuation_overlap"] < 0.5
    assert all(row["converged"] is True for row in rows)
    assert all(row["stability_passed"] is True for row in rows)
    assert all(row["residual"] <= 1.0e-9 for row in rows)

    before = complex(*rows[1]["eigenvalue"])
    after = complex(*rows[2]["eigenvalue"])
    assert before.imag > 0.0
    assert after.imag < -0.2
    assert abs(after - before) / abs(after) >= 0.9


def test_qi_full_frequency_cold_convergence_gate() -> None:
    """The newly rightmost QI branch must pass the unchanged full-mode gate."""

    report = _load(
        "adaptive_propagator_convergence_qi_frequency_extension_validation.json"
    )
    assert report["schema_version"] == 1
    assert report["passed"] is True
    assert report["required_observable"] == "frequency"
    assert report["certified"] is True
    assert report["all_growth_converged"] is True
    assert report["all_eigenvalue_converged"] is True
    assert report["all_frequency_converged"] is True

    provenance = report["provenance"]
    tolerance = provenance["convergence_tolerance"]
    assert tolerance == pytest.approx(0.05)
    assert provenance["ntheta"] == 64
    assert provenance["ladder"] == [[72, 80], [76, 84], [80, 88], [84, 92]]
    assert provenance["krylov_dim"] == 24
    assert provenance["restart_krylov_dim"] == 12
    assert provenance["adaptive_candidates"] == 2
    assert provenance["warm_repeats"] == 0

    devices = report["devices"]
    assert isinstance(devices, list) and len(devices) == 1
    qi = devices[0]
    assert qi["device"] == "qi"
    assert qi["certified"] is True
    assert qi["growth_converged_resolution"] == [80, 88]
    assert qi["eigenvalue_converged_resolution"] == [80, 88]
    assert qi["frequency_converged_resolution"] == [80, 88]
    assert all(change < tolerance for change in qi["growth_relative_changes"])
    assert all(change < tolerance for change in qi["eigenvalue_relative_changes"])
    assert all(change < tolerance for change in qi["frequency_normalized_changes"])

    rows = qi["rows"]
    assert isinstance(rows, list) and len(rows) == 4
    assert rows[-1]["n"] == 494592
    for index, row in enumerate(rows):
        assert row["converged"] is True
        assert row["stability_passed"] is True
        assert row["residual"] <= provenance["residual_tolerance"]
        assert row["restarts"] == 1
        assert math.isfinite(row["cold_seconds"]) and row["cold_seconds"] > 0.0
        assert row["warm_seconds"] is None
        if index:
            assert row["continuation_overlap"] >= 0.99
