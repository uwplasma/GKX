#!/usr/bin/env python3
"""Decompose the device-z pencil two-device speedup into its two factors.

The transport-window profiler reports one number, ``serial_jit`` wall time
divided by two-device ``shard_map`` wall time. That number mixes two very
different effects:

* the cost of running the pencil ``shard_map`` route at all, relative to the
  serial fused route, on a single device; and
* how well the ``shard_map`` route then scales from one device to two.

A speedup below the promotion gate can come from either factor, and the two
have completely different fixes. This profiler therefore times three routes on
the same state and the same transport window:

``serial_jit``
    the fused serial route, identical to the transport-window profiler baseline;
``shard_map`` on one device
    the sharded route with a single-device mesh, so no work is split;
``shard_map`` on two devices
    the sharded route with the field-line dimension split across devices.

The ratios that matter are ``shard_map_route_overhead`` (single-device sharded
over serial) and ``two_device_parallel_scaling`` (single-device sharded over
two-device sharded). Their quotient is the reported end-to-end speedup, so the
decomposition is exact rather than a model.

This is a diagnostic localization artifact. It makes no production nonlinear
domain-decomposition speedup claim on its own.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_PREFIX = (
    REPO_ROOT
    / "docs"
    / "_static"
    / "nonlinear_device_z_pencil_scaling_decomposition_gpu2_profile"
)


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_clean(value.tolist())
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    return tuple(int(item) for item in str(raw).split(",") if str(item).strip())


def _block_until_ready(value: Any) -> Any:
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return value


def _time_repeated(fn: Any, *, warmups: int, repeats: int) -> tuple[Any, list[float]]:
    for _ in range(int(warmups)):
        _block_until_ready(fn())
    last: Any = None
    times: list[float] = []
    for _ in range(int(repeats)):
        start = time.perf_counter()
        last = _block_until_ready(fn())
        times.append(float(time.perf_counter() - start))
    return last, times


def _stats(times: list[float]) -> dict[str, float]:
    return {
        "min": float(min(times)),
        "median": float(statistics.median(times)),
        "mean": float(statistics.fmean(times)),
        "max": float(max(times)),
        "std": float(statistics.pstdev(times)) if len(times) > 1 else 0.0,
    }


def _max_abs_rel(candidate: Any, reference: Any, *, floor: float) -> tuple[float, float]:
    candidate_arr = np.asarray(candidate)
    reference_arr = np.asarray(reference)
    max_abs = float(np.max(np.abs(candidate_arr - reference_arr)))
    scale = max(float(np.max(np.abs(reference_arr))), float(floor))
    return max_abs, float(max_abs / scale)


def build_decomposition(
    *,
    shape: tuple[int, int, int, int, int],
    device_counts: tuple[int, ...],
    steps: int,
    dt: float,
    warmups: int,
    repeats: int,
    atol: float,
    rtol: float,
    min_speedup: float,
    z_chunk_size: int | None,
) -> dict[str, Any]:
    from gkx.operators.nonlinear.device_z import _device_z_pencil_shard_map_rhs_fn
    from gkx.operators.nonlinear.parallel import (  # type: ignore[import-untyped]
        deterministic_nonlinear_spectral_state,
    )
    from gkx.operators.nonlinear.spectral_core import (
        _host_staged_array_for_sharding,
        _serial_nonlinear_spectral_rhs,
    )

    state = deterministic_nonlinear_spectral_state(shape)
    devices = tuple(jax.devices())
    dt_array = jnp.asarray(float(dt), dtype=jnp.real(state).dtype)

    def serial_route(item: jax.Array) -> jax.Array:
        out = item
        for _ in range(int(steps)):
            _field, _bracket, rhs = _serial_nonlinear_spectral_rhs(out)
            out = out + dt_array * rhs
        return out

    serial_jit = jax.jit(serial_route)
    serial_out, serial_times = _time_repeated(
        lambda: serial_jit(state), warmups=int(warmups), repeats=int(repeats)
    )
    serial_stats = _stats(serial_times)

    rows: list[dict[str, Any]] = [
        {
            "route": "serial_jit",
            "device_count": 1,
            "active": True,
            "identity_passed": True,
            "final_state_max_abs_error": 0.0,
            "final_state_max_rel_error": 0.0,
            "median_s": serial_stats["median"],
            "speedup_vs_serial": 1.0,
            "stats": serial_stats,
        }
    ]

    sharded_medians: dict[int, float] = {}
    for count in device_counts:
        count = int(count)
        if count < 1:
            continue
        if len(devices) < count:
            rows.append(
                {
                    "route": "shard_map",
                    "device_count": count,
                    "active": False,
                    "identity_passed": False,
                    "final_state_max_abs_error": None,
                    "final_state_max_rel_error": None,
                    "median_s": None,
                    "speedup_vs_serial": None,
                    "stats": {},
                    "blocked_reasons": ["not_enough_devices"],
                }
            )
            continue

        mesh = Mesh(np.asarray(devices[:count]), ("z",))
        sharding = NamedSharding(mesh, PartitionSpec(None, None, None, None, "z"))
        with mesh:
            sharded_state = jax.device_put(
                _host_staged_array_for_sharding(state), sharding
            )
            sharded_rhs_fn = _device_z_pencil_shard_map_rhs_fn(
                mesh, axis_name="z", z_chunk_size=z_chunk_size
            )

            def sharded_route(item: jax.Array) -> jax.Array:
                out = item
                for _ in range(int(steps)):
                    out = out + dt_array * sharded_rhs_fn(out)
                return out

            sharded_jit = jax.jit(sharded_route)
            sharded_out, sharded_times = _time_repeated(
                lambda: sharded_jit(sharded_state),
                warmups=int(warmups),
                repeats=int(repeats),
            )

        abs_err, rel_err = _max_abs_rel(sharded_out, serial_out, floor=float(atol))
        stats = _stats(sharded_times)
        sharded_medians[count] = stats["median"]
        rows.append(
            {
                "route": "shard_map",
                "device_count": count,
                "active": True,
                "identity_passed": bool(
                    abs_err <= float(atol) or rel_err <= float(rtol)
                ),
                "final_state_max_abs_error": abs_err,
                "final_state_max_rel_error": rel_err,
                "median_s": stats["median"],
                "speedup_vs_serial": serial_stats["median"] / stats["median"],
                "stats": stats,
            }
        )

    decomposition: dict[str, Any] = {}
    base = sharded_medians.get(1)
    top_count = max(sharded_medians) if sharded_medians else None
    if base is not None and top_count is not None and top_count > 1:
        top = sharded_medians[top_count]
        decomposition = {
            "reference_device_count": int(top_count),
            "shard_map_route_overhead": float(base / serial_stats["median"]),
            "parallel_scaling_vs_one_device": float(base / top),
            "parallel_efficiency_vs_one_device": float(base / top / float(top_count)),
            "net_speedup_vs_serial": float(serial_stats["median"] / top),
            "interpretation": (
                "net_speedup_vs_serial equals parallel_scaling_vs_one_device divided "
                "by shard_map_route_overhead; a route overhead above one means the "
                "sharded route starts behind the serial baseline before any work is "
                "split"
            ),
        }

    identity_rows = [row for row in rows if row["active"] and row["route"] == "shard_map"]
    all_identity = all(bool(row["identity_passed"]) for row in identity_rows)
    return _json_clean(
        {
            "shape": list(shape),
            "steps": int(steps),
            "state_elements": int(math.prod(shape)),
            "z_chunk_size": z_chunk_size,
            "state_dtype": str(np.asarray(state).dtype),
            "serial_stats_s": serial_stats,
            "rows": rows,
            "decomposition": decomposition,
            "all_active_identity_passed": bool(all_identity),
        }
    )


def build_summary(
    *,
    shapes: list[tuple[int, int, int, int, int]],
    device_counts: tuple[int, ...],
    steps: int,
    dt: float,
    warmups: int,
    repeats: int,
    atol: float,
    rtol: float,
    min_speedup: float,
    z_chunk_size: int | None,
) -> dict[str, Any]:
    grids = [
        build_decomposition(
            shape=shape,
            device_counts=device_counts,
            steps=steps,
            dt=dt,
            warmups=warmups,
            repeats=repeats,
            atol=atol,
            rtol=rtol,
            min_speedup=min_speedup,
            z_chunk_size=z_chunk_size,
        )
        for shape in shapes
    ]
    grids.sort(key=lambda grid: grid["state_elements"])
    all_identity = all(bool(grid["all_active_identity_passed"]) for grid in grids)
    nets = [
        float(grid["decomposition"]["net_speedup_vs_serial"])
        for grid in grids
        if grid.get("decomposition")
    ]
    scalings = [
        float(grid["decomposition"]["parallel_scaling_vs_one_device"])
        for grid in grids
        if grid.get("decomposition")
    ]
    overheads = [
        float(grid["decomposition"]["shard_map_route_overhead"])
        for grid in grids
        if grid.get("decomposition")
    ]
    max_net = max(nets, default=None)
    return _json_clean(
        {
            "kind": "nonlinear_device_z_pencil_scaling_decomposition",
            "claim_scope": (
                "diagnostic localization of the device-z pencil transport-window "
                "speedup into single-device shard_map route overhead and "
                "one-to-two-device parallel scaling; this artifact localizes a "
                "bottleneck and makes no production nonlinear domain-decomposition "
                "speedup claim"
            ),
            "backend": str(jax.default_backend()),
            "device_count_available": int(len(jax.devices())),
            "steps": int(steps),
            "dt": float(dt),
            "warmups": int(warmups),
            "repeats": int(repeats),
            "atol": float(atol),
            "rtol": float(rtol),
            "min_speedup": float(min_speedup),
            "grids": grids,
            "summary": {
                "status": (
                    "identity_timed_no_production_speedup"
                    if all_identity
                    else "identity_failed_no_production_speedup"
                ),
                "all_active_identity_passed": bool(all_identity),
                "grid_count": len(grids),
                "max_net_speedup_vs_serial": max_net,
                "min_parallel_scaling_vs_one_device": min(scalings, default=None),
                "max_parallel_scaling_vs_one_device": max(scalings, default=None),
                "min_shard_map_route_overhead": min(overheads, default=None),
                "max_shard_map_route_overhead": max(overheads, default=None),
                "route_overhead_needed_for_gate": (
                    max(scalings) / float(min_speedup) if scalings else None
                ),
                "limiting_factor": (
                    "shard_map_route_overhead"
                    if overheads and min(overheads) > 1.0 / 0.999
                    else "parallel_scaling"
                ),
                "production_speedup_claim_allowed": bool(
                    all_identity and max_net is not None and max_net >= float(min_speedup)
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

    grids = list(summary["grids"])
    fieldnames = [
        "shape",
        "steps",
        "state_elements",
        "serial_median_s",
        "shard_map_one_device_median_s",
        "shard_map_two_device_median_s",
        "shard_map_route_overhead",
        "parallel_scaling_vs_one_device",
        "parallel_efficiency_vs_one_device",
        "net_speedup_vs_serial",
        "final_state_max_abs_error",
        "identity_passed",
    ]
    with out_prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for grid in grids:
            indexed = {
                (row["route"], row["device_count"]): row for row in grid["rows"]
            }
            two = indexed.get(("shard_map", 2), {})
            dec = grid.get("decomposition") or {}
            writer.writerow(
                {
                    "shape": grid["shape"],
                    "steps": grid["steps"],
                    "state_elements": grid["state_elements"],
                    "serial_median_s": indexed[("serial_jit", 1)]["median_s"],
                    "shard_map_one_device_median_s": indexed.get(
                        ("shard_map", 1), {}
                    ).get("median_s"),
                    "shard_map_two_device_median_s": two.get("median_s"),
                    "shard_map_route_overhead": dec.get("shard_map_route_overhead"),
                    "parallel_scaling_vs_one_device": dec.get(
                        "parallel_scaling_vs_one_device"
                    ),
                    "parallel_efficiency_vs_one_device": dec.get(
                        "parallel_efficiency_vs_one_device"
                    ),
                    "net_speedup_vs_serial": dec.get("net_speedup_vs_serial"),
                    "final_state_max_abs_error": two.get("final_state_max_abs_error"),
                    "identity_passed": two.get("identity_passed"),
                }
            )

    set_plot_style()
    gate = float(summary["min_speedup"])

    def _metric(grid: dict[str, Any], key: str) -> float:
        return float((grid.get("decomposition") or {}).get(key) or 0.0)

    families: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for grid in grids:
        perp = (int(grid["shape"][2]), int(grid["shape"][3]))
        families.setdefault(perp, []).append(grid)
    palette = ["#b65f23", "#1b6ca8", "#4a4a4a", "#2e7d32"]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7), constrained_layout=True)
    sizes = [grid["state_elements"] / 1.0e6 for grid in grids]
    axes[0].plot(
        sizes,
        [_metric(grid, "parallel_scaling_vs_one_device") for grid in grids],
        "o-",
        lw=2.0,
        color="#1b6ca8",
        label="parallel scaling, 1→2 GPU",
    )
    axes[0].plot(
        sizes,
        [_metric(grid, "net_speedup_vs_serial") for grid in grids],
        "^--",
        lw=1.6,
        color="#4a4a4a",
        label="net speedup vs serial",
    )
    axes[0].axhline(2.0, color="0.7", ls="--", lw=0.9)
    axes[0].axhline(gate, color="0.25", ls=":", lw=1.2, label=f"{gate:g}x promotion gate")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("state size (10$^6$ complex elements)")
    axes[0].set_ylabel("ratio")
    axes[0].set_ylim(1.0, 2.2)
    axes[0].set_title("two GPUs scale; the net speedup does not")
    axes[0].legend(frameon=False, fontsize=7.5, loc="center left")
    axes[0].grid(True, alpha=0.25)

    for index, (perp, members) in enumerate(sorted(families.items())):
        members = sorted(members, key=lambda grid: grid["state_elements"])
        axes[1].plot(
            [grid["state_elements"] / 1.0e6 for grid in members],
            [_metric(grid, "shard_map_route_overhead") for grid in members],
            "s-",
            lw=2.0,
            color=palette[index % len(palette)],
            label=f"$N_y\\times N_x$ = {perp[0]}$\\times${perp[1]}",
        )
    needed = [
        _metric(grid, "parallel_scaling_vs_one_device") / gate for grid in grids
    ]
    axes[1].plot(
        sizes, needed, "--", lw=1.4, color="0.35", label="overhead needed for gate"
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("state size (10$^6$ complex elements)")
    axes[1].set_ylabel("single-device route overhead")
    axes[1].set_title("the whole deficit, and it tracks $N_y\\times N_x$")
    axes[1].legend(frameon=False, fontsize=7.5)
    axes[1].grid(True, alpha=0.25)
    fig.savefig(out_prefix.with_suffix(".png"), dpi=200)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-prefix", type=Path)
    parser.add_argument(
        "--shape",
        type=_parse_int_tuple,
        action="append",
        help="five comma-separated integers; repeat to sweep several grids",
    )
    parser.add_argument("--device-counts", type=_parse_int_tuple, default=(1, 2))
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--dt", type=float, default=0.0025)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--atol", type=float, default=5.0e-6)
    parser.add_argument("--rtol", type=float, default=1.0e-4)
    parser.add_argument("--min-speedup", type=float, default=1.5)
    parser.add_argument("--z-chunk-size", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw_shapes = args.shape or [(4, 16, 96, 96, 32)]
    shapes: list[tuple[int, int, int, int, int]] = []
    for raw in raw_shapes:
        shape = tuple(int(item) for item in raw)
        if len(shape) != 5:
            raise ValueError("--shape must contain five comma-separated integers")
        shapes.append(shape)  # type: ignore[arg-type]
    summary = build_summary(
        shapes=shapes,
        device_counts=tuple(int(item) for item in args.device_counts),
        steps=int(args.steps),
        dt=float(args.dt),
        warmups=int(args.warmups),
        repeats=int(args.repeats),
        atol=float(args.atol),
        rtol=float(args.rtol),
        min_speedup=float(args.min_speedup),
        z_chunk_size=args.z_chunk_size,
    )
    write_artifacts(summary, args.out_prefix or DEFAULT_OUT_PREFIX)
    print(json.dumps(summary["summary"], indent=2, sort_keys=True))
    for grid in summary["grids"]:
        print(grid["shape"], json.dumps(grid.get("decomposition", {}), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
