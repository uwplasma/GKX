"""Release gates for the four retained adaptive-eigensolver records."""

from __future__ import annotations

import json
import math
from pathlib import Path

STATIC = Path(__file__).resolve().parents[2] / "docs" / "_static"


def _load(name: str) -> dict[str, object]:
    with (STATIC / name).open(encoding="utf-8") as stream:
        report = json.load(stream)
    assert isinstance(report, dict)
    return report


def test_cold_reverse_ad_gate() -> None:
    """A production-sized objective and reverse derivative must match dense AD."""

    report = _load("adaptive_objective_gradient_cold_n6144_validation.json")
    assert report["schema_version"] == 1
    assert report["passed"] is True
    assert report["resolution"]["operator_size"] >= 6144
    assert report["value_relative_error"] <= 1.0e-8
    assert report["gradient_relative_error"] <= 1.0e-7
    assert report["cold_speedup_over_dense"] >= 1.1
    assert report["adaptive"]["adaptive_config"]["candidate_count"] >= 2
    assert report["adaptive"]["adaptive_config"]["sensitivity_solver"] == "propagator"


def test_physics_and_ad_matrix_gate() -> None:
    """Branch, field, geometry, species, and spectral-layout coverage is pinned."""

    report = _load("adaptive_objective_physics_matrix.json")
    assert report["schema_version"] == 1
    assert report["passed"] is True
    assert "not velocity-space convergence" in report["scope"]
    rows = report["rows"]
    assert len(rows) == 7
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
        assert row["value_relative_error"] <= 1.0e-8
        assert row["gradient_relative_error"] <= 1.0e-7


def test_real_branch_crossing_gate() -> None:
    """Biorthogonal continuation must keep a mode after its growth rank changes."""

    report = _load("adaptive_propagator_branch_crossing_validation.json")
    assert report["schema_version"] == 1
    assert report["passed"] is True
    assert all(report["checks"].values())
    assert report["exchange"]["subdominant_row_count"] >= 3
    thresholds = report["thresholds"]
    rows = report["rows"]
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
            row["adaptive_selected_spectral_gap"]
            >= thresholds["selected_complex_spectral_gap"]
        )
        if index:
            assert row["adaptive_continuation_passed"] is True
            assert (
                row["adaptive_continuation_overlap"]
                >= thresholds["continuation_overlap"]
            )
        if row["dense_tracked_growth_rank"] > 0:
            assert row["adaptive_selected_candidate_index"] > 0


def test_qi_full_frequency_convergence_and_cold_cost_gate() -> None:
    """QI frequency convergence is real, and its unresolved cold cost stays visible."""

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
    assert provenance["ladder"] == [[72, 80], [76, 84], [80, 88], [84, 92]]
    assert provenance["warm_repeats"] == 0
    qi = report["devices"][0]
    assert qi["device"] == "qi"
    assert qi["frequency_converged_resolution"] == [80, 88]
    rows = qi["rows"]
    assert rows[-1]["n"] == 494592
    for index, row in enumerate(rows):
        assert row["converged"] is True
        assert row["stability_passed"] is True
        assert row["residual"] <= provenance["residual_tolerance"]
        assert row["restarts"] == 1
        assert math.isfinite(row["cold_seconds"]) and row["cold_seconds"] > 0.0
        assert row["selected_propagator_steps"] > 0
        assert row["original_operator_evaluations"] > 0
        if index:
            assert row["continuation_overlap"] >= 0.99


def test_readme_does_not_overclaim_cold_performance() -> None:
    readme = (STATIC.parents[1] / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(readme.split())
    assert "not yet an acceptable production cold time" in normalized
    assert (
        "does **not** mean the current cold solve is universally faster" in normalized
    )
