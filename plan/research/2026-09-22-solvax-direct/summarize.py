"""Print the SOLVAX-DIRECT tables from the committed records.

Usage: python plan/research/2026-09-22-solvax-direct/summarize.py [records_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(path: Path) -> dict | None:
    if path.suffix == ".json":
        return json.loads(path.read_text())
    for line in path.read_text().splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    return None


def fmt(x, spec=".3g"):
    return "-" if x is None else format(x, spec)


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "records"
    benches = sorted(root.glob("*/*.json"))
    print("## Assembly and structure (A, full operator)")
    print("| record | n | products | nnz/row | RCM bw | assembly s | verify |")
    print("|---|---|---|---|---|---|---|")
    for p in benches:
        r = load(p)
        if not r or "assembly" not in r:
            continue
        a, s = r["assembly"], r["structure"]
        print(
            f"| {p.parent.name}/{p.stem} | {r['n']} | {a['products']} | "
            f"{fmt(s['nnz_row_mean'], '.0f')} | {s['bandwidth_rcm']} | "
            f"{fmt(a['wall_s'], '.1f')} | {fmt(a['verify_rel'], '.1e')} |"
        )
    print("\n## Factorizations of B = A - sigma I (single thread)")
    print(
        "| record | n | route | analysis s | factor s | fill | MB | "
        "solve N s | 16-RHS s | max backward err |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for p in benches:
        r = load(p)
        if not r or "direct" not in r:
            continue
        for name, d in r["direct"].items():
            if not d.get("factored", True) or "factor_s" not in d:
                print(
                    f"| {p.stem} | {r['n']} | {name} | {fmt(d.get('analysis_s'))} | "
                    f"refused (est. {d.get('estimated_mb')} MB) | | | | | |"
                )
                continue
            bwd = max(d[f"res_{t}"]["backward"] for t in "NTH")
            print(
                f"| {p.stem} | {r['n']} | {name} | {fmt(d.get('analysis_s'))} | "
                f"{fmt(d['factor_s'])} | {fmt(d['fill'], '.1f')} | {d['effective_mb']} | "
                f"{fmt(d['solve_N_s'])} | {fmt(d['solve_16rhs_s'])} | {fmt(bwd, '.1e')} |"
            )
    print("\n## pr3-cm FGMRES on the same B")
    print(
        "| record | n | sigma | rtol | iterations | converged | warm s | true relres |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for p in benches:
        r = load(p)
        if not r or "pr3" not in r:
            continue
        sig = complex(*r["sigma"])
        for rtol, d in r["pr3"].items():
            print(
                f"| {p.stem} | {r['n']} | {sig:.3g} | {rtol} | {d['iterations']} | "
                f"{d['converged']} | {fmt(d['warm_s'])} | {fmt(d['true_relres'], '.1e')} |"
            )
    print("\n## Eigenpair by shift-invert on the MUMPS (METIS) factor")
    print(
        "| record | n | lambda | certified residual | right s | left s | solves N/H |"
    )
    print("|---|---|---|---|---|---|---|")
    for p in benches:
        r = load(p)
        if not r or "eigen" not in r:
            continue
        e = r["eigen"]
        lam = complex(*e["lambda"])
        print(
            f"| {p.stem} | {r['n']} | {lam:.10g} | {fmt(e['certified_residual'], '.1e')} | "
            f"{fmt(e['right_s'])} | {fmt(e['left_s'])} | {e['solves_N']}/{e['solves_H']} |"
        )
    print("\n## GKX routes (Q28 measure.py)")
    for p in sorted(root.glob("*/*.txt")):
        r = load(p)
        if not r or "arm" not in r or "matvec_equivalents" not in r:
            continue
        mve = r["matvec_equivalents"]
        print(
            f"- {p.stem}: lambda={r.get('lambda')} certified={r.get('certified')} "
            f"route_s={fmt(r.get('route_s'))} matvec_equivalents={fmt(mve.get('total'))}"
        )
    grads = sorted(root.glob("grad*/*.json"))
    if grads:
        print("\n## Growth-rate value and gradient")
        print(
            "| record | n | gamma | value s (calls) | value+grad s (calls) | peak GiB |"
        )
        print("|---|---|---|---|---|---|")
        for p in grads:
            r = load(p)
            calls = ", ".join(f"{x:.3g}" for x in r["value_and_grad_s"])
            vcalls = ", ".join(f"{x:.3g}" for x in r["value_s"])
            print(
                f"| {p.stem} | {r.get('n')} | {r['gamma']:.15g} | {vcalls} | {calls} | "
                f"{fmt(r.get('peak_rss_gib'), '.2f')} |"
            )
        by_name = {p.stem: load(p) for p in grads}
        dense = next((r for k, r in by_name.items() if k.startswith("dense")), None)
        if dense is not None:
            import numpy as np

            ref = np.concatenate([np.ravel(x) for x in dense["grad"]])
            for k, r in by_name.items():
                if k.startswith("dense") or "grad" not in r:
                    continue
                got = np.concatenate([np.ravel(x) for x in r["grad"]])
                rel = float(np.abs(got - ref).max() / np.abs(ref).max())
                dg = abs(r["gamma"] - dense["gamma"])
                print(
                    f"- {k}: max|grad - dense| / max|dense| = {rel:.2e}; |gamma - dense| = {dg:.1e}"
                )


if __name__ == "__main__":
    main()
