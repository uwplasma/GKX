"""Combine symmetric continuation samples into a stationary heat-flux FD."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("kind") != "nonlinear_stationary_heat_flux_response_sample":
        raise ValueError(f"{path} is not a stationary-response sample")
    return payload


def _sample(payload: dict[str, Any]) -> tuple[float, float, float, bool]:
    statistics = payload["window_convergence"]["statistics"]
    sem = statistics.get("sem")
    if sem is None or not math.isfinite(float(sem)):
        raise ValueError("stationary-response sample has no finite block SEM")
    return (
        float(payload["drive_scale"]),
        float(payload["late_time_weighted_mean_heat_flux"]),
        float(sem),
        bool(payload["window_convergence"]["passed"]),
    )


def centered_response(
    minus: dict[str, Any], plus: dict[str, Any]
) -> dict[str, float | bool]:
    """Return a centered derivative and independent-window uncertainty."""

    minus_scale, minus_mean, minus_sem, minus_passed = _sample(minus)
    plus_scale, plus_mean, plus_sem, plus_passed = _sample(plus)
    midpoint = 0.5 * (minus_scale + plus_scale)
    half_step = 0.5 * (plus_scale - minus_scale)
    if half_step <= 0.0:
        raise ValueError("plus drive scale must exceed minus drive scale")
    if not math.isclose(midpoint, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("drive scales must be symmetric about one")
    if minus.get("initial_state") != plus.get("initial_state"):
        raise ValueError("symmetric samples must start from the same state")
    gradient = (plus_mean - minus_mean) / (2.0 * half_step)
    gradient_sem = math.hypot(plus_sem, minus_sem) / (2.0 * half_step)
    return {
        "midpoint": midpoint,
        "half_step": half_step,
        "minus_mean": minus_mean,
        "plus_mean": plus_mean,
        "gradient": gradient,
        "gradient_sem": gradient_sem,
        "z_score": abs(gradient) / gradient_sem,
        "both_windows_passed": minus_passed and plus_passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minus", type=Path, required=True)
    parser.add_argument("--plus", type=Path, required=True)
    parser.add_argument("--window-artifact", type=Path, default=None)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    minus = _load(args.minus)
    plus = _load(args.plus)
    response = centered_response(minus, plus)
    comparison = None
    if args.window_artifact is not None:
        window_payload = json.loads(args.window_artifact.read_text())
        rows = list(window_payload.get("rows", []))
        if not rows:
            raise ValueError("window artifact contains no gradient rows")
        requested = args.window
        candidates = (
            rows
            if requested is None
            else [row for row in rows if int(row["window"]) == int(requested)]
        )
        if not candidates:
            raise ValueError(f"window artifact has no N={requested} row")
        row = candidates[-1]
        tangent = float(row["gradient"])
        stationary = float(response["gradient"])
        comparison = {
            "window_artifact": str(args.window_artifact),
            "window": int(row["window"]),
            "window_time": int(row["window"]) * float(window_payload["dt"]),
            "window_time_in_tau_ac": int(row["window"])
            * float(window_payload["dt"])
            / float(window_payload["tau_ac_from_state"]),
            "finite_window_gradient": tangent,
            "stationary_fd_gradient": stationary,
            "same_sign": bool(tangent * stationary > 0.0),
            "window_to_stationary_magnitude_ratio": abs(tangent)
            / max(abs(stationary), 1.0e-300),
        }

    payload = {
        "kind": "nonlinear_stationary_heat_flux_centered_fd",
        "claim_level": "single_anchor_production_long_window_response_not_promoted_gradient",
        "parameter": "uniform_tprim_scale",
        "minus_artifact": str(args.minus),
        "plus_artifact": str(args.plus),
        "initial_state": minus["initial_state"],
        "response": response,
        "finite_window_comparison": comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        "d<Q>/d(scale)={gradient:+.6e} +/- {sem:.3e} (z={z:.2f})".format(
            gradient=response["gradient"],
            sem=response["gradient_sem"],
            z=response["z_score"],
        )
    )
    print(f"written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
