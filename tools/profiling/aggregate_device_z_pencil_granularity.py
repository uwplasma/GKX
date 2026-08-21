#!/usr/bin/env python3
"""Aggregate device-z pencil transport-window profiles into one granularity sweep.

The transport-window profiler writes one artifact per grid. Answering "does the
two-device speedup improve as the workload grows towards production size?"
needs those runs side by side, because the interesting quantity is the trend
rather than any single row.

This tool reads a set of
``nonlinear_device_z_pencil_transport_window_profile`` JSON files and emits a
single combined JSON/CSV/PNG artifact holding, per grid and per window length,
the serial and two-device medians, the speedup, the implied parallel
efficiency, and the identity error that was measured alongside the timing.

It performs no timing of its own; it only reindexes measurements that already
passed their identity gates. It makes no production nonlinear
domain-decomposition speedup claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_PREFIX = (
    REPO_ROOT
    / "docs"
    / "_static"
    / "nonlinear_device_z_pencil_transport_gpu2_granularity_profile"
)
EXPECTED_KIND = "nonlinear_device_z_pencil_transport_window_profile"


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _row_for(payload: dict[str, Any], label: str, source: Path) -> dict[str, Any]:
    if payload.get("kind") != EXPECTED_KIND:
        raise ValueError(f"{source}: kind must be '{EXPECTED_KIND}'")
    shape = [int(item) for item in payload["shape"]]
    one = [row for row in payload["rows"] if int(row["device_count"]) == 1]
    two = [row for row in payload["rows"] if int(row["device_count"]) == 2]
    if not one or not two:
        raise ValueError(f"{source}: needs a one-device and a two-device row")
    one_row, two_row = one[0], two[0]
    if not two_row.get("active"):
        raise ValueError(f"{source}: two-device row is inactive")
    serial_s = float(one_row["median_s"])
    sharded_s = float(two_row["median_s"])
    elements = math.prod(shape)
    return {
        "label": label,
        "source_artifact": source.name,
        "shape": shape,
        "state_elements": int(elements),
        "steps": int(payload.get("steps", 0)),
        "z_chunk_size": (payload.get("fft_batch_pressure_model") or {}).get(
            "effective_z_chunk_size"
        ),
        "serial_median_s": serial_s,
        "two_device_median_s": sharded_s,
        "speedup_vs_serial": serial_s / sharded_s,
        "parallel_efficiency": serial_s / sharded_s / 2.0,
        "serial_ns_per_element_step": (
            serial_s * 1.0e9 / elements / max(int(payload.get("steps", 1)), 1)
        ),
        "two_device_ns_per_element_step": (
            sharded_s * 1.0e9 / elements / max(int(payload.get("steps", 1)), 1)
        ),
        "final_state_max_abs_error": two_row.get("final_state_max_abs_error"),
        "final_state_max_rel_error": two_row.get("final_state_max_rel_error"),
        "identity_passed": bool(two_row.get("identity_passed")),
        "transport_window_identity_passed": bool(
            two_row.get("transport_window_identity_passed")
        ),
        "speedup_gate_passed": bool(two_row.get("speedup_gate_passed")),
        "observable_gate_median_s": two_row.get("observable_gate_median_s"),
        "observable_gate_overhead_vs_compute": two_row.get(
            "observable_gate_overhead_vs_compute"
        ),
    }


def build_granularity_summary(
    entries: list[tuple[str, Path]], *, min_speedup: float
) -> dict[str, Any]:
    rows = [
        _row_for(json.loads(path.read_text(encoding="utf-8")), label, path)
        for label, path in entries
    ]
    rows.sort(key=lambda row: (row["state_elements"], row["steps"]))
    speedups = [float(row["speedup_vs_serial"]) for row in rows]
    identity_all = all(bool(row["identity_passed"]) for row in rows)
    best = max(rows, key=lambda row: float(row["speedup_vs_serial"]))
    largest = max(rows, key=lambda row: row["state_elements"])
    return _json_clean(
        {
            "kind": "nonlinear_device_z_pencil_transport_granularity_profile",
            "claim_scope": (
                "two-device device-z pencil transport-window speedup as a function "
                "of workload granularity on two RTX A4000 GPUs; every row carries "
                "the identity error measured with its timing, and no production "
                "nonlinear domain-decomposition speedup claim is made"
            ),
            "backend": "gpu",
            "device_count": 2,
            "min_speedup": float(min_speedup),
            "rows": rows,
            "summary": {
                "row_count": len(rows),
                "all_identity_passed": bool(identity_all),
                "max_speedup_vs_serial": max(speedups),
                "min_speedup_vs_serial": min(speedups),
                "best_row_label": best["label"],
                "best_row_shape": best["shape"],
                "largest_row_label": largest["label"],
                "largest_row_speedup": float(largest["speedup_vs_serial"]),
                "speedup_improves_with_size": bool(
                    float(largest["speedup_vs_serial"])
                    > float(
                        min(rows, key=lambda row: row["state_elements"])[
                            "speedup_vs_serial"
                        ]
                    )
                ),
                "production_speedup_claim_allowed": bool(
                    identity_all and max(speedups) >= float(min_speedup)
                ),
                "status": (
                    "identity_timed_no_production_speedup"
                    if identity_all
                    else "identity_failed_no_production_speedup"
                ),
            },
        }
    )


def write_artifacts(summary: dict[str, Any], out_prefix: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gkx.artifacts.plotting import set_plot_style  # type: ignore[import-untyped]

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_prefix.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    rows = list(summary["rows"])
    fieldnames = [
        "label",
        "shape",
        "state_elements",
        "steps",
        "serial_median_s",
        "two_device_median_s",
        "speedup_vs_serial",
        "parallel_efficiency",
        "serial_ns_per_element_step",
        "two_device_ns_per_element_step",
        "final_state_max_abs_error",
        "identity_passed",
        "speedup_gate_passed",
    ]
    with out_prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7), constrained_layout=True)
    four_step = [row for row in rows if int(row["steps"]) == 4]
    sizes = [row["state_elements"] / 1.0e6 for row in four_step]
    speeds = [float(row["speedup_vs_serial"]) for row in four_step]
    axes[0].plot(sizes, speeds, "o-", lw=2.0, color="#1b6ca8")
    axes[0].axhline(
        float(summary["min_speedup"]),
        color="0.25",
        ls=":",
        lw=1.2,
        label="promotion gate",
    )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("state size (10$^6$ complex elements)")
    axes[0].set_ylabel("two-GPU speedup vs serial")
    axes[0].set_title("speedup falls as the grid grows")
    axes[0].set_ylim(1.0, 1.6)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(
        sizes,
        [float(row["serial_ns_per_element_step"]) for row in four_step],
        "o-",
        lw=2.0,
        color="#4a4a4a",
        label="serial, 1 GPU",
    )
    axes[1].plot(
        sizes,
        [float(row["two_device_ns_per_element_step"]) for row in four_step],
        "s-",
        lw=2.0,
        color="#b65f23",
        label="shard_map, 2 GPUs",
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("state size (10$^6$ complex elements)")
    axes[1].set_ylabel("ns per element per step")
    axes[1].set_title("sharded route sits at a fixed rate floor")
    axes[1].set_ylim(0.0, None)
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(True, alpha=0.25)
    fig.savefig(out_prefix.with_suffix(".png"), dpi=200)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    parser.add_argument("--min-speedup", type=float, default=1.5)
    parser.add_argument(
        "--entry",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="labelled transport-window profile JSON to include",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    entries: list[tuple[str, Path]] = []
    for raw in args.entry:
        if "=" not in raw:
            raise ValueError(f"--entry must be LABEL=PATH, got {raw!r}")
        label, _, path = str(raw).partition("=")
        entries.append((label, Path(path)))
    summary = build_granularity_summary(entries, min_speedup=float(args.min_speedup))
    write_artifacts(summary, args.out_prefix)
    print(json.dumps(summary["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
