"""Numerics: does the linear-threshold estimator recover a planted root?

The critical gradient sets the window for the nonlinear Dimits scan, so an
estimator that returns a confident wrong number costs GPU time on an
uninterpretable run. Three earlier versions of ``tools/campaigns/dimits_shift.py``
reduced over ``k_y`` before fitting, which mixes branches by construction: the
argmax moves between modes as the drive changes, so the fitted curve is a
different mode at each end.

These cases plant curves with known roots and check the two properties that
failure depended on -- that the reported threshold is the *earliest* mode's
rather than the most unstable one's, and that a fitted root outside its own
sign-change bracket is refused rather than returned.

The planted curves cross zero linearly, because that is what the tracked branch
measurably does: a simple eigenvalue crossing the imaginary axis moves
analytically in the drive.
"""

from __future__ import annotations

import numpy as np
import pytest
from support.paths import load_tool_script


def _mode(x: float, critical: float, slope: float) -> dict[str, float]:
    """One point on ``gamma = slope (x - x_crit)``, damped below the root."""

    return {"gamma": float(slope * (x - critical)), "omega": -0.25}


def _rows(multipliers, modes: dict[str, tuple[float, float]]) -> list[dict]:
    return [
        {
            "multiplier": float(x),
            "by_ky": {key: _mode(x, *params) for key, params in modes.items()},
        }
        for x in multipliers
    ]


def test_minimum_over_ky_reports_the_earliest_mode_not_the_strongest():
    module = load_tool_script("campaigns", "dimits_shift")
    # ky#3 grows faster wherever both are unstable, but ky#5 goes unstable first,
    # and the case's critical gradient belongs to whichever mode is first.
    rows = _rows(np.linspace(0.20, 0.80, 31), {"3": (0.55, 0.40), "5": (0.35, 0.10)})

    result = module.threshold_from_scan(rows)

    assert result["resolved"]
    assert result["critical_ky_index"] == 5
    assert result["critical_multiplier"] == pytest.approx(0.35, abs=0.01)


def test_a_root_outside_its_sign_change_bracket_is_refused():
    module = load_tool_script("campaigns", "dimits_shift")
    rows = _rows(np.linspace(0.20, 0.80, 31), {"4": (0.45, 0.25)})
    # One stray unstable point far below the root, which is what a continuation
    # that jumped to a neighbouring branch produces. The curve now changes sign
    # at 0.30 while the fit window still straddles 0.45, so the two disagree --
    # and that disagreement, not the fit residual, is what has to stop the
    # number.
    for row in rows:
        if row["multiplier"] == pytest.approx(0.30):
            row["by_ky"]["4"] = {"gamma": 0.09, "omega": -0.25}

    result = module.threshold_from_scan(rows)

    assert not result["resolved"]
    assert result["per_ky"]["4"]["in_bracket"] is False
    assert result["per_ky"]["4"]["sign_change_bracket"] == pytest.approx([0.28, 0.30])
