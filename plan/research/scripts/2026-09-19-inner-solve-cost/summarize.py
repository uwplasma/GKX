#!/usr/bin/env python3
"""Q21: per-arm table from the ``RESULT {json}`` lines written by ``q21.py``.

Usage: ``summarize.py DIR [DIR ...]`` -- prints one row per ``*.txt`` that
carries a RESULT line: matvec-equivalents to the 1e-6 and the target gate, wall
time, certified residual and the eigenvalue, plus the per-step inner-iteration
profile of the shift-invert arms.
"""

import json
import sys
from pathlib import Path


def load(path: Path) -> dict | None:
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    return None


def mvq(rec: dict, key: str) -> float | None:
    table = rec.get("matvec_equivalents") or {}
    entry = table.get(key)
    if entry is None:
        return table.get("total") if key == "total" else None
    return entry.get("total") if isinstance(entry, dict) else entry


def main() -> None:
    rows = []
    for d in sys.argv[1:]:
        for path in sorted(Path(d).glob("*.txt")):
            rec = load(path)
            if rec is None:
                continue
            gates = rec.get("gates") or {}
            target = [k for k in gates if k != "1e-06"]
            rows.append(
                {
                    "file": f"{Path(d).name}/{path.stem}",
                    "arm": rec["arm"],
                    "case": rec["case"],
                    "n": rec.get("n"),
                    "wall_s": rec.get("wall_s"),
                    "route_s": rec.get("route_s") or rec.get("outer_s"),
                    "mvq_1e6": mvq(rec, "1e-06"),
                    "mvq_tgt": mvq(rec, target[0]) if target else mvq(rec, "total"),
                    "t_1e6": (gates.get("1e-06") or {}).get("since_process_start_s"),
                    "t_tgt": (gates.get(target[0]) or {}).get("since_process_start_s")
                    if target
                    else None,
                    "residual": rec.get("certified_residual"),
                    "certified": rec.get("certified"),
                    "lambda": rec.get("lambda"),
                    "rss": rec.get("peak_rss_gib"),
                    "t_mv_ms": (rec.get("cost") or {}).get("t_matvec_s", 0) * 1e3,
                    "c_p": rec.get("c_precond"),
                    "inner_total": sum(s["inner_its"] for s in rec.get("steps", []))
                    or rec.get("matvecs"),
                    "steps": len(rec.get("steps", []) or rec.get("rounds", [])),
                }
            )
    head = (
        f"{'arm':28} {'case':5} {'n':>7} {'mvq(1e-6)':>11} {'mvq(target)':>12} "
        f"{'wall_s':>8} {'t(1e-6)':>8} {'t(tgt)':>8} {'inner/mv':>9} {'steps':>5} "
        f"{'residual':>10} {'cert':>5} {'t_mv_ms':>8} {'c_P':>6} {'RSS':>5}"
    )
    print(head)
    print("-" * len(head))
    for r in rows:

        def fmt(v, spec=".0f"):
            return "-" if v is None else format(v, spec)

        print(
            f"{r['file']:28} {r['case']:5} {r['n'] or 0:7d} "
            f"{fmt(r['mvq_1e6']):>11} {fmt(r['mvq_tgt']):>12} "
            f"{fmt(r['wall_s'], '.1f'):>8} {fmt(r['t_1e6'], '.1f'):>8} "
            f"{fmt(r['t_tgt'], '.1f'):>8} {fmt(r['inner_total']):>9} "
            f"{r['steps']:5d} {fmt(r['residual'], '.2e'):>10} "
            f"{str(r['certified']):>5} {fmt(r['t_mv_ms'], '.3f'):>8} "
            f"{fmt(r['c_p'], '.1f'):>6} {fmt(r['rss'], '.2f'):>5}"
        )
    print()
    for d in sys.argv[1:]:
        for path in sorted(Path(d).glob("*.txt")):
            rec = load(path)
            if rec is None or not rec.get("steps"):
                continue
            prof = " ".join(
                f"{s['inner_its']}@{s['inner_tol']:.0e}" for s in rec["steps"]
            )
            res = " ".join(f"{s['residual']:.1e}" for s in rec["steps"])
            print(f"{Path(d).name}/{path.stem} inner: {prof}")
            print(f"{Path(d).name}/{path.stem} resid: {res}")
    for d in sys.argv[1:]:
        for path in sorted(Path(d).glob("*.txt")):
            rec = load(path)
            if rec is None or not rec.get("rounds"):
                continue
            res = " ".join(f"{r['residual']:.2e}" for r in rec["rounds"])
            print(f"{Path(d).name}/{path.stem} harmonic residuals: {res}")


if __name__ == "__main__":
    main()
