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
