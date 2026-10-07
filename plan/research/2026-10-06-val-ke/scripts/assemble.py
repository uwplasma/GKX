#!/usr/bin/env python3
"""Write docs/_static/cross_code_linear_points.csv from the recorded runs.

    python plan/research/2026-10-06-val-ke/scripts/assemble.py

Inputs (all tracked):
- results/gkx_results.txt: RESULT lines of gkx_eig.py (this record);
- results/gs2_tem_fits.txt: tem_gs2.py fit output (this record);
- plan/research/2026-09-27-xcode/results/gx_repaired_96a53403.txt: GX goldens
  rerun with the dampEnds_linked grid-stride fix;
- plan/research/2026-09-27-xcode/results/gkx_cert.txt and the Q20 record
  plan/research/scripts/2026-09-14-cross-code-cyclone/results/final_tables.txt
  (GS2 and gyaradax values below are transcribed from those files).
Every number is in GX/GKX units (Lref as in each deck, vref = sqrt(T/m)).
"""

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
XCODE = ROOT / "plan/research/2026-09-27-xcode/results"
Q20 = "plan/research/scripts/2026-09-14-cross-code-cyclone/results/final_tables.txt"
OUT = ROOT / "docs/_static/cross_code_linear_points.csv"

# (case, ky, code, resolution, gamma, omega, source) transcribed from tracked records.
TRANSCRIBED = [
    (
        "cyclone_salpha",
        0.30,
        "gs2",
        "r4 ntheta64 nperiod4 negrid24",
        0.09219,
        0.2840,
        Q20,
    ),
    ("cyclone_salpha", 0.30, "gyaradax", "g2 ns144 nvpar64 nmu16", 0.09209, "", Q20),
    (
        "cyclone_miller",
        0.15,
        "gs2",
        "r3 ntheta48 nperiod3 negrid16",
        0.05783,
        0.0916,
        Q20,
    ),
    (
        "cyclone_miller",
        0.30,
        "gs2",
        "r3 ntheta48 nperiod3 negrid16",
        0.12546,
        0.2157,
        Q20,
    ),
    (
        "cyclone_miller",
        0.40,
        "gs2",
        "r3 ntheta48 nperiod3 negrid16",
        0.14297,
        0.3078,
        Q20,
    ),
    (
        "cyclone_miller",
        0.55,
        "gs2",
        "r4 ntheta64 nperiod4 negrid24",
        0.12516,
        0.4399,
        Q20,
    ),
    (
        "miller_ke",
        0.30,
        "gs2",
        "e3 ntheta48 nperiod3 negrid16",
        0.23484,
        0.2210,
        "plan/research/2026-09-27-xcode/results/fit_grid_codes.csv",
    ),
    (
        "miller_ke",
        0.50,
        "gs2",
        "e3 ntheta48 nperiod3 negrid16",
        0.25354,
        0.4571,
        "plan/research/2026-09-27-xcode/results/fit_grid_codes.csv",
    ),
]
GX_CASE = {
    "salpha": "cyclone_salpha",
    "miller": "cyclone_miller",
    "miller_ke": "miller_ke",
}
GKX_DECK = {
    "case_full.toml": "cyclone_salpha",
    "cyclone_miller_linear.toml": "cyclone_miller",
    "cyclone_miller_kinetic_electrons.toml": "miller_ke",
    "tem_dannert_jenko_2005.toml": "tem_dj2005",
}


def rows():
    out = [
        dict(zip(("case", "ky", "code", "resolution", "gamma", "omega", "source"), r))
        for r in TRANSCRIBED
    ]
    case = None
    for line in (XCODE / "gx_repaired_96a53403.txt").read_text().splitlines():
        if line.startswith("#"):
            case = GX_CASE.get(line.split()[1].split("/")[0])
        elif case:
            ky, g, w = (float(v) for v in line.split(","))
            out.append(
                dict(
                    case=case,
                    ky=ky,
                    code="gx-repaired",
                    resolution="Nl16 Nm48",
                    gamma=g,
                    omega=w,
                    source="plan/research/2026-09-27-xcode/results/gx_repaired_96a53403.txt",
                )
            )
    for path in (XCODE / "gkx_cert.txt", HERE / "results/gkx_results.txt"):
        for line in path.read_text().splitlines():
            if not line.startswith("RESULT"):
                continue
            r = json.loads(line.split(" ", 1)[1])
            res = f"Nl{r['Nl']} Nm{r['Nm']}" + "".join(
                f" {k}={v}" for k, v in r.get("over", {}).items()
            )
            ky = r["ky"] * (3**0.5 if GKX_DECK[r["deck"]] == "tem_dj2005" else 1.0)
            out.append(
                dict(
                    case=GKX_DECK[r["deck"]],
                    ky=round(ky, 6),
                    code="gkx",
                    resolution=res,
                    gamma=r["gamma"],
                    omega=r["omega"],
                    source=str(path.relative_to(ROOT)),
                )
            )
    for r in csv.DictReader((HERE / "results/gs2_tem_fits.txt").open()):
        if r.get("case", "").startswith("TEM_"):
            out.append(
                dict(
                    case="tem_dj2005",
                    ky=float(r["ky_rhos"]),
                    code="gs2",
                    resolution=r["case"].split("_", 2)[2],
                    gamma=float(r["gamma_gkx"]),
                    omega=float(r["omega_gkx"]),
                    source="plan/research/2026-10-06-val-ke/results/gs2_tem_fits.txt",
                )
            )
    return out


if __name__ == "__main__":
    data = sorted(
        rows(), key=lambda r: (r["case"], float(r["ky"]), r["code"], r["resolution"])
    )
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)
    print(f"{OUT.relative_to(ROOT)}: {len(data)} rows")
