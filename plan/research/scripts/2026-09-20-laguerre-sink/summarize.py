"""Q28: tables and verdict for the declared Laguerre sink (plan §0.5 (iii)-(iv)).

Reads this row's sink records and, unchanged, #252's twelve certified
collisionless / species-nu records, and answers the three questions §0.5 hands
forward:

* does a declared sink remove the cutoff pile-up in the Laguerre free-energy
  spectrum of the certified eigenvector (#252's diagnostic: strict interior
  local maxima, whose last one sat at 0.79-0.85 of Nl at every unregularized
  rung);
* does it make gamma Nl-independent, and at what strength does the answer stop
  moving;
* do the two regularizations -- an ell^p hypercollision sink and the conserving
  collision term -- extrapolate to the same collisionless limit.

Usage: summarize.py <sink_results_dir> <q16_results_dir> [--json out.json]
Output: a fixed-width report on stdout and, with --json, the same numbers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SINK_STRENGTHS = (0.001, 0.01, 0.1, 0.5)
RUNGS = (24, 32, 48, 64)


def load(directory: Path) -> list[dict]:
    records = []
    for path in sorted(directory.glob("*.txt")):
        if path.name.endswith(".time.txt"):
            continue
        text = path.read_text()
        if "RESULT " not in text:
            records.append({"key": path.stem, "error": "no RESULT line"})
            continue
        records.append(json.loads(text.split("RESULT ", 1)[1]))
    return records


def interior_maxima(spectrum: list[float]) -> dict:
    arr = np.asarray(spectrum, dtype=float)
    idx = [
        int(i)
        for i in range(1, arr.size - 1)
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]
    ]
    last = idx[-1] if idx else None
    return {
        "count": len(idx),
        "last_index": last,
        "last_fraction_of_Nl": None if last is None else float(last) / float(arr.size),
        "monotone_decaying": not idx,
    }


def phi(record: dict) -> np.ndarray | None:
    if record.get("phi_z_real") is None:
        return None
    return np.asarray(record["phi_z_real"]) + 1j * np.asarray(record["phi_z_imag"])


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    return float(abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b)))


def relative_change(new: float, old: float) -> float:
    return 100.0 * (new - old) / abs(old)


def zero_intercept(xs: list[float], ys: list[float]) -> float | None:
    """Least-squares straight line through (x, y), evaluated at x = 0."""

    if len(xs) < 2:
        return None
    coef = np.polyfit(np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), 1)
    return float(np.polyval(coef, 0.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sink_dir", type=Path)
    parser.add_argument("q16_dir", type=Path)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    sink = {r["key"]: r for r in load(args.sink_dir)}
    q16 = {r["key"]: r for r in load(args.q16_dir)}

    report: dict = {"sink": {}, "reused": {}, "verdict": {}}
    lines: list[str] = []

    def emit(text: str = "") -> None:
        lines.append(text)

    # ---------------------------------------------------------------- sink table
    emit("Q28 (iii) declared Laguerre sink: certified eigenpairs, Cyclone ky=.55")
    emit("  deck Nm96 Nz96 nkx1 f64, species nu = 0, |k_z| Hermite branch ON,")
    emit("  const branch = Laguerre only (nu_hyper_m_const = 0), p_hyper_l = 6")
    emit("")
    header = (
        f"{'nu_hyper_l':>10} {'Nl':>3} {'gamma':>11} {'omega':>10} "
        f"{'residual':>10} {'imax':>4} {'last/Nl':>8} {'upperQ':>10} "
        f"{'decades':>8} {'spectrum':>10} {'s':>6}"
    )
    emit(header)
    emit("-" * len(header))
    tag_of = {0.001: "hl1e-3", 0.01: "hl1e-2", 0.1: "hl1e-1", 0.5: "hl5e-1"}
    for strength in SINK_STRENGTHS:
        for nl in RUNGS:
            key = f"{tag_of[strength]}-nl{nl}"
            record = sink.get(key)
            if record is None:
                continue
            if record.get("error"):
                emit(f"{strength:>10g} {nl:>3} {'FAILED':>11}  {record['error'][:48]}")
                report["sink"][key] = {"error": record["error"]}
                continue
            tail = record["laguerre_tail"]
            maxima = record.get("laguerre_interior_maxima") or interior_maxima(
                record["laguerre_spectrum"]
            )
            row = {
                "nu_hyper_l": strength,
                "Nl": nl,
                "gamma": record["gamma"],
                "omega": record["omega"],
                "residual": record["recertified_residual"],
                "interior_maxima": maxima["count"],
                "last_max_fraction": maxima["last_fraction_of_Nl"],
                "upper_quarter_fraction": tail["upper_quarter_fraction"],
                "decades_peak_to_cutoff": tail["decades_peak_to_cutoff"],
                "monotone_decaying": maxima["monotone_decaying"],
                "route_s": record["route_s"],
                "top_laguerre_rate": (record.get("operator_params") or {}).get(
                    "top_laguerre_rate"
                ),
            }
            report["sink"][key] = row
            frac = maxima["last_fraction_of_Nl"]
            emit(
                f"{strength:>10g} {nl:>3} {record['gamma']:>11.6f} "
                f"{record['omega']:>10.5f} {record['recertified_residual']:>10.2e} "
                f"{maxima['count']:>4} "
                f"{('-' if frac is None else f'{frac:.3f}'):>8} "
                f"{tail['upper_quarter_fraction']:>10.3e} "
                f"{tail['decades_peak_to_cutoff']:>8.2f} "
                f"{('monotone' if maxima['monotone_decaying'] else 'pile-up'):>10} "
                f"{record['route_s']:>6.0f}"
            )

    # ------------------------------------------------- reused #252 ladders (iv)
    emit("")
    emit("Q28 (iv) reused from #252 (plan/research/scripts/2026-09-18-...): the same")
    emit("  deck and route with GKX's conserving collision term instead of a sink")
    emit("")
    emit(header)
    emit("-" * len(header))
    nu_tag = {0.0: "nu0", 1e-3: "nu1e-3", 3e-3: "nu3e-3", 1e-2: "nu1e-2"}
    for nu in (0.0, 1e-3, 3e-3, 1e-2):
        for nl in RUNGS:
            key = f"{nu_tag[nu]}-nl{nl}"
            record = q16.get(key)
            if record is None or record.get("error") or "gamma" not in record:
                continue
            tail = record["laguerre_tail"]
            maxima = interior_maxima(record["laguerre_spectrum"])
            row = {
                "species_nu": nu,
                "Nl": nl,
                "gamma": record["gamma"],
                "omega": record["omega"],
                "residual": record["recertified_residual"],
                "interior_maxima": maxima["count"],
                "last_max_fraction": maxima["last_fraction_of_Nl"],
                "upper_quarter_fraction": tail["upper_quarter_fraction"],
                "decades_peak_to_cutoff": tail["decades_peak_to_cutoff"],
                "monotone_decaying": maxima["monotone_decaying"],
            }
            report["reused"][key] = row
            frac = maxima["last_fraction_of_Nl"]
            emit(
                f"{nu:>10g} {nl:>3} {record['gamma']:>11.6f} "
                f"{record['omega']:>10.5f} {record['recertified_residual']:>10.2e} "
                f"{maxima['count']:>4} "
                f"{('-' if frac is None else f'{frac:.3f}'):>8} "
                f"{tail['upper_quarter_fraction']:>10.3e} "
                f"{tail['decades_peak_to_cutoff']:>8.2f} "
                f"{('monotone' if maxima['monotone_decaying'] else 'pile-up'):>10} "
                f"{record['route_s']:>6.0f}"
            )

    # -------------------------------------------------------- Nl convergence
    emit("")
    emit("Nl convergence of the certified gamma (percent change per rung)")
    emit("")
    emit(f"{'arm':>22} {'24->32':>10} {'32->48':>10} {'converged?':>12}")
    emit("-" * 58)
    convergence: dict = {}
    for label, table, tags in (
        ("sink nu_hyper_l", sink, [(s, tag_of[s]) for s in SINK_STRENGTHS]),
        ("collisions nu", q16, [(v, nu_tag[v]) for v in (0.0, 1e-3, 3e-3, 1e-2)]),
    ):
        for value, tag in tags:
            gammas = {}
            for nl in RUNGS:
                record = table.get(f"{tag}-nl{nl}")
                if record and not record.get("error") and "gamma" in record:
                    gammas[nl] = record["gamma"]
            if 32 not in gammas or 48 not in gammas:
                continue
            first = (
                relative_change(gammas[32], gammas[24])
                if 24 in gammas
                else float("nan")
            )
            second = relative_change(gammas[48], gammas[32])
            converged = abs(second) < 2.0
            name = f"{label} {value:g}"
            convergence[name] = {
                "d24_32_pct": first,
                "d32_48_pct": second,
                "converged_2pct": converged,
                "gamma_Nl48": gammas[48],
            }
            emit(
                f"{name:>22} {first:>9.2f}% {second:>9.2f}% "
                f"{('YES' if converged else 'no'):>12}"
            )
    report["convergence"] = convergence

    # ------------------------------------------------------- zero extrapolation
    emit("")
    emit("Extrapolation to zero regularization, at each FIXED Nl")
    emit("  (a straight line through the available strengths, evaluated at 0;")
    emit("   an Nl-independent intercept would be a collisionless limit)")
    emit("")
    emit(f"{'arm':>22} {'Nl24':>11} {'Nl32':>11} {'Nl48':>11} {'spread':>9}")
    emit("-" * 68)
    extrapolation: dict = {}
    for label, table, pairs in (
        ("sink nu_hyper_l->0", sink, [(s, tag_of[s]) for s in SINK_STRENGTHS]),
        ("collisions nu->0", q16, [(v, nu_tag[v]) for v in (1e-3, 3e-3, 1e-2)]),
    ):
        intercepts = {}
        for nl in (24, 32, 48):
            xs, ys = [], []
            for value, tag in pairs:
                record = table.get(f"{tag}-nl{nl}")
                if record and not record.get("error") and "gamma" in record:
                    xs.append(value)
                    ys.append(record["gamma"])
            if len(xs) >= 2:
                intercepts[nl] = zero_intercept(xs, ys)
        if not intercepts:
            continue
        values = [v for v in intercepts.values() if v is not None]
        spread = (
            100.0 * (max(values) - min(values)) / abs(np.mean(values))
            if len(values) > 1
            else float("nan")
        )
        extrapolation[label] = {"per_Nl": intercepts, "spread_pct": spread}
        cells = " ".join(
            f"{intercepts.get(nl, float('nan')):>11.6f}" for nl in (24, 32, 48)
        )
        emit(f"{label:>22} {cells} {spread:>8.1f}%")
    report["extrapolation"] = extrapolation

    # -------------------------------------------------------------- phi overlaps
    emit("")
    emit("phi(z) overlap of each sink eigenvector against #252's collisionless")
    emit("  eigenvector at the same Nl (is the sink acting on the same branch?)")
    emit("")
    overlaps: dict = {}
    for strength in SINK_STRENGTHS:
        row = []
        for nl in RUNGS:
            a = sink.get(f"{tag_of[strength]}-nl{nl}")
            b = q16.get(f"nu0-nl{nl}")
            if not a or not b or a.get("error") or b.get("error"):
                continue
            pa, pb = phi(a), phi(b)
            if pa is None or pb is None:
                continue
            value = overlap(pa, pb)
            overlaps[f"{tag_of[strength]}-nl{nl}"] = value
            row.append(f"Nl{nl}={value:.4f}")
        if row:
            emit(f"  nu_hyper_l={strength:<7g} " + "  ".join(row))
    report["phi_overlaps_vs_collisionless"] = overlaps

    text = "\n".join(lines)
    print(text)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
