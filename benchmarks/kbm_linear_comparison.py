#!/usr/bin/env python3
"""Plot the reviewed fixed-beta KBM linear comparison table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gkx.artifacts.plotting import scan_comparison_figure

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TABLE = ROOT / "docs/_static/comparison/kbm_reference_candidates.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--output", type=Path, default=Path("kbm_linear_comparison.png"))
    args = parser.parse_args()

    table = pd.read_csv(args.table).sort_values("ky")
    if "selected" in table:
        selected = table["selected"].map(
            lambda value: str(value).strip().lower() in {"true", "1", "yes"}
        )
        table = table[selected]
    fig, _axes = scan_comparison_figure(
        table["ky"].to_numpy(),
        table["gamma"].to_numpy(),
        table["omega"].to_numpy(),
        x_label=r"$k_y\rho_i$",
        title=r"KBM linear scan ($\beta=0.015$)",
        x_ref=table["ky"].to_numpy(),
        gamma_ref=table["gamma_gx"].to_numpy(),
        omega_ref=table["omega_gx"].to_numpy(),
        ref_label="Reference",
    )
    fig.savefig(args.output, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
