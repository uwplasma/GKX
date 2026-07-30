"""Numerical policy tests for adaptive eigensolver qualification campaigns."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from support.paths import load_repo_script


convergence = load_repo_script(
    "tools/campaigns/validate_adaptive_propagator_convergence.py",
    module_name="test_adaptive_propagator_convergence_campaign",
)
continuation = load_repo_script(
    "tools/campaigns/validate_adaptive_branch_continuation.py",
    module_name="test_adaptive_branch_continuation_campaign",
)


def test_frequency_gate_measures_imaginary_change_on_full_mode_scale() -> None:
    values = [1.0 + 0.10j, 1.0 + 0.06j, 1.0 + 0.03j]
    changes = convergence._frequency_changes(values)

    assert changes == [
        abs(0.06 - 0.10) / abs(values[1]),
        abs(0.03 - 0.06) / abs(values[2]),
    ]


def test_two_change_plateau_reports_the_finest_certified_rung() -> None:
    ladder = ((40, 48), (44, 52), (48, 56), (52, 60))

    assert convergence._plateau([0.04, 0.03, 0.02], ladder, 0.05) == (
        48,
        56,
    )
    assert convergence._plateau([0.06, 0.04, 0.03], ladder, 0.05) == (
        52,
        60,
    )
    assert convergence._plateau([0.06, 0.05, 0.04], ladder, 0.05) is None


def test_relative_biorthogonal_score_is_one_for_the_anchored_mode() -> None:
    left = np.asarray([1.0, 0.0], dtype=complex)
    anchor = np.asarray([2.0j, 0.0], dtype=complex)
    candidates = np.asarray(
        [
            [3.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=complex,
    ).T

    np.testing.assert_allclose(
        continuation._relative_biorthogonal_scores(left, anchor, candidates),
        np.asarray([1.0, 0.0]),
    )


def test_partial_checkpoint_uses_a_portable_input_label(tmp_path: Path) -> None:
    repository = Path(convergence.__file__).resolve().parents[2]
    input_path = repository / "examples/vmec/input.nfp3_QI_fixed_resolution_final"

    checkpoint = convergence._write_checkpoint(
        tmp_path / "partial.json",
        device="qi",
        input_path=input_path,
        ladder=((60, 68), (64, 72), (68, 76)),
        rows=[],
    )

    report = json.loads(checkpoint.read_text())
    assert report["complete"] is False
    assert report["input"] == "examples/vmec/input.nfp3_QI_fixed_resolution_final"
