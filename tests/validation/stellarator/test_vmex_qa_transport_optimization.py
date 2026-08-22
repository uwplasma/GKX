from __future__ import annotations

import csv
import json

from support.paths import REPO_ROOT
import py_compile
import re
from types import SimpleNamespace

import numpy as np

import gkx
from gkx.objectives.vmec_candidate_admission import (
    build_solved_vmec_candidate_gate,
)


ROOT = REPO_ROOT
EXAMPLES = ROOT / "examples" / "optimization"
QA_SCRIPT = EXAMPLES / "QA_optimization.py"
TRANSPORT_SUMMARY = ROOT / "docs" / "_static" / "qa_transport_summary.csv"
TRANSPORT_TRACES = ROOT / "docs" / "_static" / "qa_transport_traces.csv"
TRANSPORT_TIMESERIES = (
    ROOT / "docs" / "_static" / "qa_transport_nominal_timeseries.csv"
)


def _transport_summary() -> dict[str, dict[str, float]]:
    with TRANSPORT_SUMMARY.open(encoding="utf-8", newline="") as stream:
        return {
            row["case"]: {
                key: float(value) for key, value in row.items() if key != "case"
            }
            for row in csv.DictReader(stream)
        }


def test_vmex_style_qa_script_appends_physical_autodiff_transport() -> None:
    text = QA_SCRIPT.read_text(encoding="utf-8")

    py_compile.compile(str(QA_SCRIPT), doraise=True)
    assert "argparse" not in text
    assert "MAX_MODES, MAX_NFEV = [1, 2, 3, 4, 5]" in text
    assert "SEED_PERTURBATION = 0.01" in text
    assert "ASPECT_TARGET, IOTA_TARGET = 6.0, 0.42" in text
    assert "am=np.zeros_like(inp.am), pres_scale=0.0" in text
    assert "A_OVER_LT, A_OVER_LN = 3.0, 1.0" in text
    assert "kpar_scale=local_geometry.gradpar_value" in text
    assert "p_hyper_m=float(min(20, max(NM // 2, 1)))" in text
    assert "gkx.integrate_nonlinear(" in text
    assert "gkx.nonlinear_heat_flux_window(" in text
    assert "implicit_jacobian_method=\"auto\"" in text
    assert "objective_function_terms = [" in text
    assert "(qs, 0.0, QA_PRIORITY)," in text
    assert "(opt.aspect_ratio, ASPECT_TARGET, ASPECT_PRIORITY)," in text
    assert "(opt.mean_iota, IOTA_TARGET, IOTA_PRIORITY)," in text
    assert "(turbulent_transport, 0.0, transport_weight)," in text
    assert "result = least_squares(" in text
    assert "def report(label, local_equilibrium):" in text


def test_docs_name_the_single_qa_autodiff_script() -> None:
    docs = [
        ROOT / "README.md",
        ROOT / "docs" / "stellarator_optimization.rst",
        EXAMPLES / "README.md",
    ]
    for path in docs:
        text = path.read_text(encoding="utf-8")
        assert "QA_optimization.py" in text, path

    examples_readme = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    assert "exact discrete differentiation" in re.sub(r"\s+", " ", examples_readme)


def test_docs_scope_vmex_transport_optimizer_claims() -> None:
    docs = [
        ROOT / "README.md",
        ROOT / "docs" / "stellarator_optimization.rst",
        EXAMPLES / "README.md",
    ]
    for path in docs:
        text = path.read_text(encoding="utf-8")
        normalized = re.sub(r"\s+", " ", text)
        assert "transport" in text, path
        assert "nonlinear" in text, path
        assert "post-saturation" in normalized, path


def test_readme_qa_figures_and_reproduction_inputs_are_checked_in() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    docs = (ROOT / "docs" / "stellarator_optimization.rst").read_text(
        encoding="utf-8"
    )
    figures = (
        "nonlinear_autodiff_validation.png",
        "qa_transport_equilibria.png",
        "qa_transport_reduction.svg",
    )
    for filename in figures:
        path = ROOT / "docs" / "_static" / filename
        assert f"docs/_static/{filename}" in readme
        assert path.stat().st_size > 0
    equilibria = ROOT / "docs" / "_static" / "qa_transport_equilibria.png"
    assert equilibria.stat().st_size < 100_000
    assert equilibria.read_bytes()[25] == 3  # indexed-color PNG

    for filename in (
        "input.qa_transport_baseline",
        "input.qa_transport_candidate",
    ):
        assert (EXAMPLES / filename).is_file()
        assert filename in docs
    for script in (
        ROOT / "tools" / "campaigns" / "qa_transport_validation.py",
        ROOT / "tools" / "artifacts" / "build_qa_transport_figures.py",
    ):
        py_compile.compile(str(script), doraise=True)
        assert script.name in docs

    with TRANSPORT_TIMESERIES.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    time = np.asarray([float(row["time"]) for row in rows])
    values = np.asarray(
        [
            [
                float(row["baseline_mean"]),
                float(row["baseline_sem"]),
                float(row["candidate_mean"]),
                float(row["candidate_sem"]),
            ]
            for row in rows
        ]
    )
    assert len(rows) == 301
    assert np.all(np.diff(time) > 0.0)
    assert time[0] < 1.0 and time[-1] > 1499.0
    assert np.all(np.isfinite(values))
    assert np.all(values[:, (1, 3)] >= 0.0)


def test_optimization_examples_document_user_customization_knobs() -> None:
    examples_readme = (EXAMPLES / "README.md").read_text(encoding="utf-8")

    assert "QA nonlinear transport" in examples_readme
    assert "A_OVER_LT" in examples_readme
    assert "A_OVER_LN" in examples_readme
    assert "SATURATION_STEPS" in examples_readme
    assert "WINDOW_STEPS" in examples_readme
    assert "objective_function_terms" in examples_readme
    assert "least_squares" in examples_readme


def test_readme_uses_solved_vmec_qa_geometry_not_reduced_surface_panel() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    docs = (ROOT / "docs" / "stellarator_optimization.rst").read_text(encoding="utf-8")
    manuscript = (ROOT / "docs" / "manuscript_figures.rst").read_text(encoding="utf-8")
    normalized_readme = re.sub(r"\s+", " ", readme)

    # The README presents the production method, not a reduced proxy panel.
    assert "docs/_static/qa_itg_optimization_summary_panel.png" not in readme
    assert "docs/_static/vmex_qa_solved_boundary_boozer_panel.png" not in readme
    assert "docs/_static/stellarator_itg_optimization_comparison.png" not in readme
    assert "docs/_static/stellarator_itg_optimization_uq.png" not in readme
    assert "independent matched runs validate" in normalized_readme.lower()
    assert "12.26% reduction" in normalized_readme
    assert "replicated" in normalized_readme.lower()
    assert "post-saturation" in normalized_readme.lower()

    assert "QA_optimization.py" in docs
    assert "nonlinear_heat_flux_window" not in docs or "physical heat-flux window" in docs
    assert ".. figure:: _static/stellarator_itg_optimization_comparison.png" not in docs
    assert "screening diagnostics" in docs
    assert (
        "current artifact bases: ``docs/_static/stellarator_itg_optimization_comparison.png``"
        not in manuscript
    )
    assert "is not a solved-geometry optimization figure" in manuscript
    assert (
        "production QA optimization examples are the VMEC-JAX-style scripts"
        in manuscript
    )


def test_reduced_surface_comparison_is_not_current_primary_optimization_figure() -> (
    None
):
    release_contract = (
        ROOT / "benchmarks" / "references" / "gkx_1_7_release_contract.json"
    ).read_text(encoding="utf-8")
    examples_readme = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    docs = (ROOT / "docs" / "stellarator_optimization.rst").read_text(encoding="utf-8")

    reduced_png = '"docs/_static/stellarator_itg_optimization_comparison.png"'
    assert reduced_png not in release_contract
    assert "stellarator_itg_growth_optimization.py" not in examples_readme
    assert "reduced_stellarator_itg" not in examples_readme
    assert "screening diagnostics" in re.sub(r"\s+", " ", docs)


def test_matched_qa_transport_is_statistically_resolved_and_converged() -> None:
    """Freeze the Sokal-estimated matched-run validation of Kim et al.'s task."""
    rows = _transport_summary()
    assert set(rows) == {
        "nominal",
        "dt04",
        "dt025",
        "perp12",
        "perp20",
        "perp24",
        "perp24long",
        "z16",
        "z32",
        "v36",
        "v612",
    }
    assert all(np.isfinite(value) for row in rows.values() for value in row.values())
    with TRANSPORT_TRACES.open(encoding="utf-8", newline="") as stream:
        traces = list(csv.DictReader(stream))
    for row in rows.values():
        assert row["p_hyper_m"] == min(20, max(int(row["nm"]) // 2, 1))
    for case, row in rows.items():
        case_traces = [trace for trace in traces if trace["case"] == case]
        pairs = int(row["pairs"])
        assert len(case_traces) == 2 * pairs
        for design in ("baseline", "candidate"):
            seeds = {int(trace["seed"]) for trace in case_traces if trace["design"] == design}
            assert seeds == set(range(pairs))
        assert all(float(trace["tau"]) > 0.0 for trace in case_traces)
        assert all(float(trace["neff"]) > 0.0 for trace in case_traces)

    nominal = rows["nominal"]
    assert nominal["pairs"] == 24
    assert nominal["positive_pairs"] == nominal["pairs"]
    assert nominal["ci95_low_percent"] > 0.0
    assert nominal["minimum_window_in_tau"] > 10.0
    assert abs(nominal["baseline_half_shift_percent"]) < 2.0 * nominal[
        "baseline_half_shift_sem_percent"
    ]
    assert abs(nominal["candidate_half_shift_percent"]) < 2.0 * nominal[
        "candidate_half_shift_sem_percent"
    ]

    for case in ("dt04", "dt025"):
        assert rows[case]["ci95_low_percent"] > 0.0
        assert (
            abs(rows[case]["reduction_percent"] - nominal["reduction_percent"])
            < 5.0
        )

    perp20 = rows["perp20"]
    short_perp24 = rows["perp24"]
    long_perp24 = rows["perp24long"]
    assert perp20["pairs"] == 16
    assert perp20["ci95_low_percent"] > 0.0
    assert short_perp24["ci95_low_percent"] < 0.0
    assert short_perp24["baseline_half_shift_percent"] > 2.0 * short_perp24[
        "baseline_half_shift_sem_percent"
    ]
    assert long_perp24["pairs"] == 16
    assert long_perp24["positive_pairs"] == long_perp24["pairs"]
    assert long_perp24["ci95_low_percent"] > 0.0
    assert long_perp24["minimum_window_in_tau"] > 10.0
    assert abs(long_perp24["baseline_half_shift_percent"]) < 2.0 * long_perp24[
        "baseline_half_shift_sem_percent"
    ]
    assert abs(long_perp24["candidate_half_shift_percent"]) < 2.0 * long_perp24[
        "candidate_half_shift_sem_percent"
    ]
    assert max(
        perp20["ci95_low_percent"], long_perp24["ci95_low_percent"]
    ) <= min(perp20["ci95_high_percent"], long_perp24["ci95_high_percent"])
    for key in ("baseline_mean", "candidate_mean"):
        relative_difference = abs(perp20[key] - long_perp24[key]) / max(
            perp20[key], long_perp24[key]
        )
        assert relative_difference < 5.5e-2

    for case in ("z16", "z32"):
        assert rows[case]["ci95_low_percent"] > 0.0
        assert rows[case]["ci95_low_percent"] < nominal["ci95_high_percent"]
        assert rows[case]["ci95_high_percent"] > nominal["ci95_low_percent"]

    coarse_velocity = rows["v36"]
    refined_velocity = rows["v612"]
    assert any(
        abs(coarse_velocity[key] - nominal[key]) / nominal[key] > 0.15
        for key in ("baseline_mean", "candidate_mean")
    )
    assert refined_velocity["pairs"] == 16
    assert refined_velocity["ci95_low_percent"] > 0.0
    assert refined_velocity["ci95_low_percent"] < nominal["ci95_high_percent"]
    assert refined_velocity["ci95_high_percent"] > nominal["ci95_low_percent"]
    for key in ("baseline_mean", "candidate_mean"):
        assert abs(refined_velocity[key] - nominal[key]) / nominal[key] < 5.0e-2


def test_solved_wout_candidate_gate_passes_valid_qa_branch() -> None:
    assert (
        gkx.build_solved_vmec_candidate_gate is build_solved_vmec_candidate_gate
    )
    result = SimpleNamespace(
        history={
            "aspect_final": 5.999233,
            "iota_final": 0.427011,
            "qs_final": 2.604013e-2,
        },
    )

    report = build_solved_vmec_candidate_gate(
        result,
        target_aspect=6.0,
        aspect_atol=5.0e-2,
        min_abs_mean_iota=0.41,
        qs_residual_max=5.0e-2,
        iota_profile_floor=0.41,
        iota_profiles=(
            np.asarray([0.0, 0.410131, 0.414]),
            np.asarray([0.410706, 0.414]),
        ),
    )

    assert report["passed"] is True
    assert report["checks"]["aspect"]["passed"] is True
    assert report["checks"]["mean_iota"]["passed"] is True
    assert report["checks"]["quasisymmetry"]["passed"] is True
    assert report["checks"]["iota_profile"]["passed"] is True
    json.dumps(report, allow_nan=False)


def test_solved_wout_candidate_gate_rejects_transport_branch_that_breaks_constraints() -> (
    None
):
    result = SimpleNamespace(
        history={
            "aspect_final": 5.996817,
            "iota_final": 0.425028,
            "qs_final": 1.091236e-1,
        },
    )

    report = build_solved_vmec_candidate_gate(
        result,
        target_aspect=6.0,
        aspect_atol=5.0e-2,
        min_abs_mean_iota=0.41,
        qs_residual_max=5.0e-2,
        iota_profile_floor=0.41,
        iota_profiles=(
            np.asarray([0.0, 0.402043, 0.414]),
            np.asarray([0.402493, 0.414]),
        ),
    )

    assert report["passed"] is False
    assert report["checks"]["aspect"]["passed"] is True
    assert report["checks"]["mean_iota"]["passed"] is True
    assert report["checks"]["quasisymmetry"]["passed"] is False
    assert report["checks"]["iota_profile"]["passed"] is False
    assert "do not promote" in report["next_action"]
    json.dumps(report, allow_nan=False)
