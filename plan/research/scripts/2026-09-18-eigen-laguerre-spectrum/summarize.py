"""Q16: tables from the per-case ``RESULT`` lines.

Prints, in order: the eigenpair table (lambda, both residuals, the certification
gate and route, wall time), the Laguerre tail table, the Hermite tail table, and
the per-index Laguerre spectrum of each case. Reads ``<run>/results/*.txt``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(results_dir: Path) -> list[dict]:
    records = []
    for path in sorted(results_dir.glob("*.txt")):
        if path.name.endswith(".time.txt"):
            continue
        record = None
        for line in path.read_text().splitlines():
            if line.startswith("RESULT "):
                record = json.loads(line[len("RESULT ") :])
        if record is None:
            records.append({"key": path.stem, "error": "no RESULT line"})
        else:
            records.append(record)
    return sorted(records, key=lambda r: (r.get("nu", -1.0), r.get("Nl", 0)))


def fmt(value, spec):
    return "—" if value is None else format(value, spec)


parser = argparse.ArgumentParser()
parser.add_argument("--results", type=Path, required=True)
args = parser.parse_args()
records = load(args.results)

print("### Certified eigenpairs\n")
print(
    "| key | nu | Nl | Nm | n | gamma | omega | route | solver residual | "
    "gate | certified | re-certified residual | wall s | peak RSS MB | note |"
)
print("|" + "---|" * 15)
for r in records:
    status = r.get("eigen_status") or {}
    rss = r.get("process_peak_rss_kib")
    note = (r.get("error") or "").replace("|", "/")[:120]
    print(
        f"| {r.get('key')} | {fmt(r.get('nu'), '.0e') if r.get('nu') else '0'} | "
        f"{r.get('Nl', '—')} | {r.get('Nm', '—')} | {r.get('n', '—')} | "
        f"{fmt(r.get('gamma'), '.7f')} | {fmt(r.get('omega'), '.6f')} | "
        f"{status.get('route', '—')} | {fmt(status.get('residual'), '.3e')} | "
        f"{fmt(status.get('tolerance'), '.1e')} | {status.get('certified', '—')} | "
        f"{fmt(r.get('recertified_residual'), '.3e')} | "
        f"{fmt(r.get('route_s'), '.0f')} | "
        f"{'—' if rss is None else format(rss / 1024, '.0f')} | {note} |"
    )

for axis in ("laguerre", "hermite"):
    print(f"\n### {axis.capitalize()} free-energy spectrum tails\n")
    print(
        "| key | nu | Nl | peak index | peak | upper-quarter fraction | "
        "cutoff fraction | cutoff/peak | decades peak->cutoff | log10 slope/index |"
    )
    print("|" + "---|" * 10)
    for r in records:
        tail = r.get(f"{axis}_tail")
        if tail is None:
            continue
        print(
            f"| {r['key']} | {fmt(r['nu'], '.0e') if r['nu'] else '0'} | {r['Nl']} | "
            f"{tail['peak_index']} | {fmt(tail['peak_value'], '.4e')} | "
            f"{fmt(tail['upper_quarter_fraction'], '.4f')} | "
            f"{fmt(tail['cutoff_fraction'], '.3e')} | "
            f"{fmt(tail['cutoff_over_peak'], '.3e')} | "
            f"{fmt(tail['decades_peak_to_cutoff'], '.2f')} | "
            f"{fmt(tail['log10_slope_per_index_top_half'], '.4f')} |"
        )

print("\n### Laguerre cutoff structure\n")
print(
    "A monotone spectrum has no interior local maximum. A cutoff pile-up shows "
    "one or more, and the last one sits near the cutoff; ``hump ratio`` is that "
    "maximum divided by the local minimum before it.\n"
)
print(
    "| key | nu | Nl | interior maxima | last max index | last max / Nl | "
    "hump ratio | mean fraction per index, upper quarter |"
)
print("|" + "---|" * 8)
for r in records:
    spectrum = r.get("laguerre_spectrum")
    if spectrum is None:
        continue
    n = len(spectrum)
    maxima = [
        i
        for i in range(1, n - 1)
        if spectrum[i] > spectrum[i - 1] and spectrum[i] > spectrum[i + 1]
    ]
    minima = [
        i
        for i in range(1, n - 1)
        if spectrum[i] < spectrum[i - 1] and spectrum[i] < spectrum[i + 1]
    ]
    if maxima:
        top = maxima[-1]
        before = [j for j in minima if j < top] or [0]
        cell = f"{top} | {top / n:.2f} | {spectrum[top] / spectrum[before[-1]]:.1f}"
    else:
        cell = "— | — | —"
    quarter = spectrum[(3 * n) // 4 :]
    print(
        f"| {r['key']} | {fmt(r['nu'], '.0e') if r['nu'] else '0'} | {r['Nl']} | "
        f"{len(maxima)} | {cell} | {sum(quarter) / len(quarter):.4f} |"
    )

print("\n### Laguerre spectrum per index (normalized free energy)\n")
for r in records:
    spectrum = r.get("laguerre_spectrum")
    if spectrum is None:
        continue
    nu = f"{r['nu']:.0e}" if r["nu"] else "0"
    print(f"{r['key']} (nu={nu}, Nl={r['Nl']}):")
    print("  " + " ".join(f"{x:.3e}" for x in spectrum))
