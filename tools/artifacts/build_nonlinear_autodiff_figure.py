"""Plot the measured nonlinear-adjoint memory and AD/FD validation ladder.

Every number on this figure is read from the JSON the generators write, not
typed in here. The generators are::

    tools/profiling/profile_nonlinear_adjoint_checkpointing.py   # left panel
    tools/campaigns/nonlinear_gradient_window.py                 # right panel

Both are documented in their own module docstrings, including the exact
commands and their cost. Re-run them with ``--output`` pointing at the files
below and this script redraws the figure from the new measurements.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "docs" / "_static"
OUT = STATIC / "nonlinear_autodiff_validation.png"
MEMORY_PROFILES = (
    ("CPU", STATIC / "nonlinear_adjoint_checkpointing_cpu32.json"),
    ("RTX A4000", STATIC / "nonlinear_adjoint_checkpointing_gpu32.json"),
)
LADDER = STATIC / "nonlinear_heat_flux_gradient_window_rk3.json"
#: AD/FD disagreement above which a rung is no longer a gradient. Matches the
#: default in the ladder generator and the knee behind
#: ``gkx.solvers.nonlinear.state_integration.DIVERGENCE_KNEE_STEPS``.
TOLERANCE = 1.0e-6


def _load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(
            f"missing measurement {path.relative_to(ROOT)}; regenerate it with "
            "the generator named in this module's docstring rather than editing "
            "the numbers into the figure"
        )
    return json.loads(path.read_text())


def _memory_megabytes(profile: dict) -> tuple[float, float]:
    by_policy = {
        row["checkpoint"]: row["temp_bytes"] / 1.0e6 for row in profile["rows"]
    }
    return by_policy["step"], by_policy["block"]


def main() -> int:
    memory = np.asarray(
        [_memory_megabytes(_load(path)) for _label, path in MEMORY_PROFILES]
    )
    labels = [label for label, _path in MEMORY_PROFILES]

    ladder = _load(LADDER)
    rows = ladder["rows"]
    steps = np.asarray([row["window"] for row in rows])
    adjoint = np.asarray([abs(row["gradient"]) for row in rows])
    finite_difference = np.asarray([abs(row["centered_fd_gradient"]) for row in rows])
    relative = np.asarray([row["ad_fd_relative_error"] for row in rows])
    diverged = np.flatnonzero(relative > TOLERANCE)
    knee = int(steps[diverged[0] - 1]) if diverged.size else None

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), constrained_layout=True)
    x = np.arange(len(labels))
    axes[0].bar(x - 0.18, memory[:, 0], 0.36, label="step checkpoint")
    axes[0].bar(x + 0.18, memory[:, 1], 0.36, label="block checkpoint")
    axes[0].set(
        xticks=x,
        xticklabels=labels,
        ylabel="XLA temporary memory [MB]",
        yscale="log",
    )
    axes[0].legend(frameon=False)

    axes[1].plot(steps, adjoint, "o-", label="discrete adjoint")
    axes[1].plot(steps, finite_difference, "x--", label="centered FD")
    if knee is not None:
        axes[1].axvspan(knee, steps[diverged[0]], color="0.9", label="divergence knee")
    axes[1].set(
        xlabel="window steps",
        ylabel=r"$|d\langle Q\rangle/dp|$",
        xscale="log",
        yscale="log",
    )
    axes[1].legend(frameon=False)

    fig.savefig(OUT, dpi=180)
    print(OUT)
    print(f"measured divergence knee: {knee} steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
