"""Assess repeatability of stationary heat-flux finite differences across anchors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def combine_responses(
    payloads: list[dict[str, Any]], *, min_pairs: int = 3, min_z_score: float = 2.0
) -> dict[str, Any]:
    """Combine equally weighted anchor responses with conservative uncertainty."""

    if not payloads:
        raise ValueError("at least one centered response is required")
    gradients = np.asarray(
        [float(payload["response"]["gradient"]) for payload in payloads]
    )
    within_sems = np.asarray(
        [float(payload["response"]["gradient_sem"]) for payload in payloads]
    )
    if not np.all(np.isfinite(gradients)) or not np.all(np.isfinite(within_sems)):
        raise ValueError("gradients and uncertainties must be finite")
    count = int(gradients.size)
    mean = float(np.mean(gradients))
    within_sem = float(np.sqrt(np.sum(within_sems * within_sems)) / count)
    between_sem = (
        float(np.std(gradients, ddof=1) / np.sqrt(count)) if count >= 2 else None
    )
    conservative_sem = max(within_sem, 0.0 if between_sem is None else between_sem)
    z_score = abs(mean) / max(conservative_sem, 1.0e-300)
    signs = np.sign(gradients[np.abs(gradients) > 0.0])
    consistent_sign = bool(signs.size > 0 and np.all(signs == signs[0]))
    all_windows_passed = all(
        bool(payload["response"]["both_windows_passed"]) for payload in payloads
    )
    distinct_anchors = len({str(payload["initial_state"]) for payload in payloads})
    gates = {
        "minimum_anchor_pairs": count >= int(min_pairs),
        "distinct_anchors": distinct_anchors == count,
        "all_component_windows_passed": all_windows_passed,
        "consistent_gradient_sign": consistent_sign,
        "ensemble_gradient_resolved": z_score >= float(min_z_score),
    }
    return {
        "anchor_pairs": count,
        "distinct_anchors": distinct_anchors,
        "gradients": gradients.tolist(),
        "individual_sems": within_sems.tolist(),
        "ensemble_mean_gradient": mean,
        "propagated_within_anchor_sem": within_sem,
        "between_anchor_sem": between_sem,
        "conservative_sem": conservative_sem,
        "z_score": z_score,
        "gates": gates,
        "passed": all(gates.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--min-pairs", type=int, default=3)
    parser.add_argument("--min-z-score", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payloads = [json.loads(path.read_text()) for path in args.inputs]
    for path, payload in zip(args.inputs, payloads, strict=True):
        if payload.get("kind") != "nonlinear_stationary_heat_flux_centered_fd":
            raise ValueError(f"{path} is not a centered stationary response")
    statistics = combine_responses(
        payloads, min_pairs=args.min_pairs, min_z_score=args.min_z_score
    )
    payload = {
        "kind": "nonlinear_stationary_heat_flux_fd_ensemble",
        "claim_level": "production_long_window_response_ensemble",
        "parameter": "uniform_tprim_scale",
        "inputs": [str(path) for path in args.inputs],
        "statistics": statistics,
        "limitations": [
            "block uncertainty treats the late adaptive-step samples as approximately uniform",
            "finite-difference spacing must still be checked with a second step size",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        "ensemble gradient={mean:+.6e} +/- {sem:.3e} (z={z:.2f}) {status}".format(
            mean=statistics["ensemble_mean_gradient"],
            sem=statistics["conservative_sem"],
            z=statistics["z_score"],
            status="PASS" if statistics["passed"] else "NOT RESOLVED",
        )
    )
    print(f"written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
