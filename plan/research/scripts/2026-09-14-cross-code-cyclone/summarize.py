#!/usr/bin/env python3
"""Q20 final tables from results/: GS2/stella ladders with convergence labels, GKX eigenpairs,
gyaradax and the GX shipped goldens, all in GX units (Lref = a, vt = sqrt(T/m)).

    python summarize.py results/ > results/final_tables.txt

Convergence rule (manifest.toml): converged at R_k+1 when |gamma(R_k+1) - gamma(R_k)| / gamma(R_k+1) < 2%
and both rungs are settled (|drift| <= 1%); the reported value is gamma(R_k+1) of the highest such pair.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

KY = ["0.15", "0.30", "0.40", "0.50", "0.55"]
# GX shipped goldens (benchmarks/linear/ITG_cyclone/*_correct.out.nc, upstream 3865a537; Nl16 Nm48,
# back-half mean of omega_kxkyt as the upstream check.py computes it). (gamma, omega)
GX = {
    "S": {"0.15": (0.05497, 0.12685), "0.30": (0.09303, 0.28199), "0.40": (0.08091, 0.37494),
          "0.50": (0.05406, 0.45591), "0.55": (0.03460, 0.49835)},
    "M": {"0.15": (0.05841, 0.09182), "0.30": (0.12586, 0.21547), "0.40": (0.14312, 0.30669),
          "0.50": (0.13642, 0.39400), "0.55": (0.12594, 0.43364)},
}


def load_fits(root: Path) -> dict:
    rows = {}
    with open(root / "fit_gs2_stella.txt") as fh:
        for r in csv.DictReader(fh):
            # skip failed fits and cases without a DONE marker (killed or still running)
            if r.get("gamma_gx") and r.get("rc") != "running":
                rows[r["case"]] = r
    return rows


def ladder(rows: dict, code: str, geom: str, ky: str) -> str:
    vals = []
    for rung in ("r1", "r2", "r3", "r4"):
        r = rows.get(f"{code}_{geom}_ky{ky}_{rung}")
        if r is not None and r["settled"] in ("yes", "no"):
            vals.append((rung, float(r["gamma_gx"]), float(r["omega_gx"]), r["settled"] == "yes"))
    if not vals:
        return "not run"
    parts, verdict = [], None
    for i, (rung, g, o, ok) in enumerate(vals):
        s = f"{rung} {g:.5f}{'' if ok else ' (unsettled)'}"
        if i:
            rel = (g - vals[i - 1][1]) / g
            s += f" [{rel:+.1%}]"
            if abs(rel) < 0.02 and ok and vals[i - 1][3]:
                verdict = f"converged {g:.5f} (omega {o:.4f}) at {rung}"
        parts.append(s)
    last = vals[-1]
    if verdict is None or not verdict.endswith(last[0]):
        if verdict is None:
            verdict = "unconverged"
        elif abs((last[1] - vals[-2][1]) / last[1]) >= 0.02:
            verdict = f"unconverged (last step {(last[1] - vals[-2][1]) / last[1]:+.1%})"
    return "; ".join(parts) + f" -> {verdict}"


def gkx(root: Path, geom: str, ky: str) -> str:
    out = []
    for nl in (8, 16, 24):
        f = root / f"gkx_{geom}_ky{ky}_nl{nl}_nm48.txt"
        if not f.exists():
            continue
        lines = [ln for ln in f.read_text().splitlines() if ln.startswith("RESULT ")]
        if not lines:
            out.append(f"Nl{nl}: no pair (stopped)")
            continue
        r = json.loads(lines[-1][7:])
        if r.get("error"):
            out.append(f"Nl{nl}: rejected ({r['error'][:60]})")
        else:
            out.append(f"Nl{nl} {r['gamma']:.6f} (omega {r['omega']:.5f}, res {r['residual']:.1e}, "
                       f"{r['route_s']:.0f} s)")
    return "; ".join(out) if out else "not run"


def gyaradax(root: Path, ky: str) -> str:
    out = []
    for rung in ("g1", "g2", "g3"):
        f = root / f"gyaradax_S_ky{ky}_{rung}.txt"
        if not f.exists():
            continue
        lines = [ln for ln in f.read_text().splitlines() if ln.startswith("RESULT ")]
        if not lines:
            out.append(f"{rung}: no result")
            continue
        r = json.loads(lines[-1][7:])
        out.append(f"{rung} {r['gamma_gx']:.5f} (drift {r['drift']:+.1e}, {r['wall_s']:.0f} s)")
    return "; ".join(out) if out else "not run"


def main() -> None:
    root = Path(sys.argv[1])
    rows = load_fits(root)
    for geom, codes in (("S", ["gs2"]), ("M", ["gs2", "stella"])):
        print(f"## geometry {geom}")
        for ky in KY:
            print(f"ky {ky}: GX golden gamma {GX[geom][ky][0]:.5f} omega {GX[geom][ky][1]:.5f}")
            for code in codes:
                print(f"  {code}: {ladder(rows, code, geom, ky)}")
            print(f"  gkx: {gkx(root, geom, ky)}")
            if geom == "S":
                print(f"  gyaradax: {gyaradax(root, ky)}")
        print()


if __name__ == "__main__":
    main()
