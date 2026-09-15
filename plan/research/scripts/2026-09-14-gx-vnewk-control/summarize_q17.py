#!/usr/bin/env python3
"""Tabulate the Q17 GX vnewk=1e-2 fits against GKX's species nu=1e-2 runs.

Run from the repository root:
    python plan/research/scripts/2026-09-14-gx-vnewk-control/summarize_q17.py

GX values come from results/<key>.json (gx_fit.py: phi2 fit on [0.7T, T],
half-time probe [0.35T, 0.5T], omega = second-half mean of omega_kxkyt).
GKX values come from the committed Q8 summary.csv (Q3 rows reused at Nl 24/32,
Q8 row at Nl48): same T=150, fit window and half-time probe. The relative
difference is (GX - GKX) / GKX. Q8's P4 criterion is |difference| <= 2%.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
Q8_SUMMARY = HERE.parent / "2026-09-13-collisional-convergence" / "summary.csv"
GX_KEYS = {24: "gx-nu1e-2-nl24", 32: "gx-nu1e-2-nl32", 48: "gx-nu1e-2-nl48"}
TOLERANCE = 0.02


def _gkx_rows() -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    with Q8_SUMMARY.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row["code"] == "GKX"
                and float(row["nu"]) == 0.01
                and float(row["t_end"]) == 150.0
            ):
                rows[int(row["Nl"])] = {
                    "gamma": float(row["gamma"]),
                    "omega": float(row["omega"]),
                    "shift": float(row["gamma_half_time_shift"]),
                }
    return rows


def main() -> None:
    gkx = _gkx_rows()
    lines = [
        "Q17: GX vnewk=1e-2 vs GKX species nu=1e-2, Cyclone s-alpha ITG ky=.55, Nm96, Nz96,",
        "rk4 dt .002, absorber 50, T=150, fit [105,150]; rel = (GX-GKX)/GKX; P4 pass if |rel| <= 2%",
        "",
        "| Nl | GKX gamma | GX gamma | rel gamma | GKX omega | GX omega | rel omega | GX half shift | GX settled | P4 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    gx_gamma: dict[int, float] = {}
    for nl, key in GX_KEYS.items():
        path = HERE / "results" / f"{key}.json"
        ref = gkx[nl]
        if not path.exists():
            lines.append(
                f"| {nl} | {ref['gamma']:.7f} | not run | | {ref['omega']:.6f} | not run | | | | not run |"
            )
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        assert rec["nlaguerre"] == nl and rec["collisions"] == 1
        assert rec["nonfinite_phi2"] == 0 and rec["nonfinite_omega"] == 0
        gamma = float(rec["gamma_phi2_late"])
        omega = float(rec["omega_second_half_mean"])
        gx_gamma[nl] = gamma
        rel_g = (gamma - ref["gamma"]) / ref["gamma"]
        rel_w = (omega - ref["omega"]) / ref["omega"]
        verdict = (
            "pass" if abs(rel_g) <= TOLERANCE and abs(rel_w) <= TOLERANCE else "fail"
        )
        lines.append(
            f"| {nl} | {ref['gamma']:.7f} | {gamma:.7f} | {rel_g:+.2e} | {ref['omega']:.6f} | "
            f"{omega:.6f} | {rel_w:+.2e} | {rec['gamma_half_time_shift']:+.2e} | "
            f"{'yes' if rec['settled'] else 'no'} | {verdict} |"
        )
    lines.append("")
    if 24 in gx_gamma and 32 in gx_gamma:
        d_gx = (gx_gamma[32] - gx_gamma[24]) / gx_gamma[24]
        d_gkx = (gkx[32]["gamma"] - gkx[24]["gamma"]) / gkx[24]["gamma"]
        lines.append(f"Nl24->32 gamma change: GX {d_gx:+.4%}, GKX {d_gkx:+.4%}")
    d_gkx48 = (gkx[48]["gamma"] - gkx[32]["gamma"]) / gkx[32]["gamma"]
    lines.append(f"GKX Nl32->48 gamma change: {d_gkx48:+.4%}")
    text = "\n".join(lines) + "\n"
    (HERE / "summary.txt").write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
