"""The nonlinear-autodiff claims must stay attached to their generators.

Commits 612e1311 and a7b41968 removed the tools and the JSON that produced the
headline adjoint numbers, leaving the docs page and the figure resting on
literals typed into the plotting script. These tests pin the other direction:
the tracked measurements exist, the figure reads them, the generators that write
them are still in the tree, and the divergence knee the runtime guard uses is
the one the tracked ladder actually shows.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from support.paths import REPO_ROOT, load_artifact_tool, load_tool_script

from gkx.solvers.nonlinear.state_integration import DIVERGENCE_KNEE_STEPS

STATIC = REPO_ROOT / "docs" / "_static"
LADDER = STATIC / "nonlinear_heat_flux_gradient_window_rk3.json"
PARITY = STATIC / "nonlinear_window_device_parity.json"
MEMORY = (
    STATIC / "nonlinear_adjoint_checkpointing_cpu32.json",
    STATIC / "nonlinear_adjoint_checkpointing_gpu32.json",
)
GENERATORS = (
    "tools/campaigns/nonlinear_gradient_window.py",
    "tools/profiling/profile_nonlinear_adjoint_checkpointing.py",
    "tools/profiling/profile_nonlinear_window_device_parity.py",
)


def _ladder_tool():
    return load_tool_script("campaigns", "nonlinear_gradient_window")


def test_gradient_ladder_requires_compatible_clean_state_source(
    monkeypatch,
) -> None:
    tool = _ladder_tool()
    provenance = {
        "repository_root": str(REPO_ROOT),
        "git_commit": "current",
        "git_dirty": False,
    }

    assert (
        tool._require_compatible_state_source(
            {
                "gkx_git_commit": np.asarray("current"),
                "gkx_git_dirty": np.asarray(0),
            },
            provenance,
        )
        == "current"
    )
    with pytest.raises(SystemExit, match="no GKX source provenance"):
        tool._require_compatible_state_source({}, provenance)
    monkeypatch.setattr(tool, "_gkx_source_tree_matches", lambda *_args: False)
    with pytest.raises(SystemExit, match="differs from current source"):
        tool._require_compatible_state_source(
            {
                "gkx_git_commit": np.asarray("old"),
                "gkx_git_dirty": np.asarray(0),
            },
            provenance,
        )


@pytest.mark.parametrize("relative", GENERATORS)
def test_generator_scripts_are_present(relative: str) -> None:
    assert (REPO_ROOT / relative).is_file()


def test_docs_page_names_every_generator() -> None:
    page = (REPO_ROOT / "docs" / "nonlinear_autodiff.rst").read_text()
    for relative in GENERATORS:
        assert relative in page, f"{relative} is not documented as regenerable"


def test_figure_builder_reads_measurements_rather_than_literals() -> None:
    module = load_artifact_tool("build_nonlinear_autodiff_figure")
    assert module.LADDER == LADDER
    assert {path for _label, path in module.MEMORY_PROFILES} == set(MEMORY)
    for path in (module.LADDER, *(p for _l, p in module.MEMORY_PROFILES)):
        assert path.is_file(), f"missing tracked measurement {path}"
    # The knee shading on the figure and the knee the ladder reports have to be
    # the same threshold, or the picture and the number disagree.
    assert module.TOLERANCE == _ladder_defaults()["tolerance"]


def _ladder_defaults() -> dict:
    parser = _ladder_tool().build_parser()
    return {action.dest: action.default for action in parser._actions}


def test_tracked_ladder_shows_the_knee_the_runtime_guard_uses() -> None:
    ladder = json.loads(LADDER.read_text())
    tolerance = _ladder_defaults()["tolerance"]
    rows = [
        dict(row, agrees=row["ad_fd_relative_error"] <= tolerance)
        for row in ladder["rows"]
    ]
    knee = _ladder_tool().locate_knee(rows)
    assert knee["divergence_knee_steps"] == DIVERGENCE_KNEE_STEPS
    assert knee["knee_bracket"] == [DIVERGENCE_KNEE_STEPS, 2 * DIVERGENCE_KNEE_STEPS]


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
