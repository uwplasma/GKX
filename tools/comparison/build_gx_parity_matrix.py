#!/usr/bin/env python3
"""Build the linear parity matrix against the reference comparison code.

For every case in the manifest this driver

1. reads the growth rate and frequency spectrum from the reference
   code output, using the reference code's own late-window convention (mean of
   the second half of its diagnostic trace),
2. runs the GKX linear scan over the same ``ky`` values with the same velocity
   resolution, importing the reference run's geometry so the two codes see
   identical geometric coefficients,
3. repeats the scan over half the integration time and reports how far the
   answer moved, so a mode that has not settled is visible rather than hidden,
   and
4. records wall time and peak host memory for the GKX scan.

A case may also declare ``build_reproducibility_floor``: the relative difference
below which its reference number carries no information, because the reference
side moves by that much between two legitimate builds of the same reference
commit. It is carried into the artifact rather than applied as a filter -- rows
below the floor are still reported, and marked -- so that a reader, and any
later gate, sees the resolution of the instrument next to the reading.

The reference outputs are not tracked in this repository. Point
``GX_PARITY_REF_DIR`` at a directory holding them, or pass ``--reference-dir``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any

import tomllib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MANIFEST = REPO_ROOT / "tools" / "gx_parity_matrix_manifest.toml"
DEFAULT_STEM = REPO_ROOT / "docs" / "_static" / "gkx_gx_linear_parity_matrix"


@dataclass(frozen=True)
class ReferenceSpectrum:
    """Reference estimates and optional uniform-sampling temporal probe."""

    ky: np.ndarray
    gamma: np.ndarray
    omega: np.ndarray
    samples: int
    t_end: float
    nonfinite: int
    gamma_half: np.ndarray | None = None
    omega_half: np.ndarray | None = None


def load_reference_spectrum(path: Path) -> ReferenceSpectrum:
    """Average the reference code's late diagnostic window, its own convention."""

    from netCDF4 import Dataset

    with Dataset(path, "r") as root:
        grids = root.groups["Grids"]
        diag = root.groups["Diagnostics"]
        t = np.asarray(grids.variables["time"][:], dtype=float)
        ky = np.asarray(grids.variables["ky"][:], dtype=float)
        series = np.asarray(diag.variables["omega_kxkyt"][:], dtype=float)
    half = int(len(t) / 2)
    omega = np.mean(series[half:, :, 0, 0], axis=0)
    gamma = np.mean(series[half:, :, 0, 1], axis=0)
    positive = ky > 0.0
    # Preserve the historical sample-mean estimator. Do not certify physical
    # time windows from short, invalid or nonuniformly sampled traces. GX may
    # write one final sample before the next regular diagnostic time.
    gamma_half = omega_half = None
    spacing = np.diff(t)
    if (
        len(t) >= 8
        and np.all(np.isfinite(t))
        and np.all(spacing > 0)
        and np.allclose(spacing[:-1], spacing[0], rtol=1e-3, atol=0)
        and spacing[-1] <= spacing[0] * 1.001
    ):
        middle = (t >= t[0] + 0.25 * (t[-1] - t[0])) & (t <= t[half])
        omega_half, gamma_half = np.mean(series[middle, :, 0, :], axis=0).T
        gamma_half, omega_half = gamma_half[positive], omega_half[positive]
    return ReferenceSpectrum(
        ky=ky[positive],
        gamma=gamma[positive],
        omega=omega[positive],
        samples=int(len(t)),
        t_end=float(t[-1]),
        nonfinite=int(np.sum(~np.isfinite(series))),
        gamma_half=gamma_half,
        omega_half=omega_half,
    )


def _peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1.0 if sys.platform == "darwin" else 1024.0
    return float(usage) * scale / (1024.0 * 1024.0)


def _device_peak_mb() -> float | None:
    try:
        import jax

        stats = jax.devices()[0].memory_stats()
    except Exception:  # noqa: BLE001 - device stats are best effort
        return None
    if not stats:
        return None
    peak = stats.get("peak_bytes_in_use")
    if peak is None:
        return None
    return float(peak) / (1024.0 * 1024.0)


def _sample_stride(steps: int, target_samples: int) -> int:
    """Return a stride that divides ``steps`` and keeps the sample count bounded."""

    stride = max(1, int(steps) // max(1, int(target_samples)))
    while stride < steps and steps % stride:
        stride += 1
    return stride if steps % stride == 0 else 1


def _repo_relative(path: Path) -> str:
    """Describe a path relative to the repository when it lives inside it.

    A manifest kept outside the checkout is a normal way to run a variant
    campaign, so this must not raise: losing a completed run while writing its
    own provenance string would be an expensive way to learn that.
    """

    resolved = Path(path).expanduser().resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _relative(value: float, reference: float) -> float:
    if not np.isfinite(value) or not np.isfinite(reference) or reference == 0.0:
        return float("nan")
    return float((value - reference) / abs(reference))


def _settled(*pairs: tuple[float, float]) -> bool:
    return all(
        np.isfinite(full)
        and np.isfinite(half)
        and (full == half or abs(_relative(half, full)) <= 0.05)
        for full, half in pairs
    )


def run_case(
    case: dict[str, Any], *, reference_dir: Path, order: int = 0
) -> dict[str, Any]:
    """Run one parity case and return its serializable record."""

    from gkx import load_runtime_from_toml, run_runtime_scan

    reference_path = reference_dir / str(case["reference_output"])
    spectrum = load_reference_spectrum(reference_path)

    reference_ky = np.asarray(spectrum.ky, dtype=float)
    ky_values = np.asarray(case.get("ky", reference_ky), dtype=float)
    for values in (reference_ky, ky_values):
        if (
            values.ndim != 1
            or not values.size
            or not np.all(np.isfinite(values) & (values > 0))
            or len(np.unique(values)) != len(values)
        ):
            raise ValueError("parity ky must be nonempty, unique, finite and positive")
    matches = np.argmin(abs(reference_ky[:, None] - ky_values), axis=0)
    # Allow decimal rendering of a single-precision reference coordinate, not
    # interpolation or an arbitrary nearest mode. Fail before expensive solves.
    if not np.all(
        np.isclose(
            ky_values, reference_ky[matches], rtol=4 * np.finfo(np.float32).eps, atol=0
        )
    ):
        raise ValueError("requested ky is absent from the reference spectrum")
    if len(np.unique(matches)) != len(matches):
        raise ValueError("requested ky values map to duplicate reference modes")
    ky_values = reference_ky[matches]

    config_path = REPO_ROOT / str(case["config"])
    cfg, _ = load_runtime_from_toml(config_path)
    # Fixed physical rate: the parity reference may use a different historical
    # timestep than a standalone fixture. Never recompute this during refinement.
    if "damp_ends_rate" in case:
        cfg = replace(
            cfg,
            collisions=replace(
                cfg.collisions, damp_ends_amp=float(case["damp_ends_rate"])
            ),
        )

    steps = int(case["steps"])
    dt = float(case["dt"])
    t_end = steps * dt
    late = float(case.get("fit_start_fraction", 0.7))
    samples = int(case.get("target_samples", 2000))

    common = dict(
        Nl=int(case["Nl"]),
        Nm=int(case["Nm"]),
        dt=dt,
        method=str(case.get("method", "imex2")),
        solver="time",
        batch_ky=True,
        fit_signal="phi",
        mode_method=str(case.get("mode_method", "z_index")),
        auto_window=False,
        min_points=int(case.get("min_points", 80)),
        require_positive=True,
    )

    start = time.perf_counter()
    primary = run_runtime_scan(
        cfg,
        ky_values,
        steps=steps,
        sample_stride=_sample_stride(steps, samples),
        tmin=late * t_end,
        tmax=t_end,
        **common,
    )
    elapsed = time.perf_counter() - start
    half_steps = steps // 2
    half_end = half_steps * dt
    secondary = run_runtime_scan(
        cfg,
        ky_values,
        steps=half_steps,
        sample_stride=_sample_stride(half_steps, samples),
        tmin=late * half_end,
        tmax=half_end,
        **common,
    )

    floor = case.get("build_reproducibility_floor")
    floor = None if floor is None else float(floor)

    rows = []
    for index, ky in enumerate(ky_values):
        match = int(np.argmin(np.abs(spectrum.ky - ky)))
        gamma_ref = float(spectrum.gamma[match])
        omega_ref = float(spectrum.omega[match])
        gamma = float(primary.gamma[index])
        omega = float(primary.omega[index])
        gamma_half = float(secondary.gamma[index])
        omega_half = float(secondary.omega[index])
        ref_half = getattr(spectrum, "gamma_half", None)
        gamma_ref_half = None if ref_half is None else float(ref_half[match])
        ref_half = getattr(spectrum, "omega_half", None)
        omega_ref_half = None if ref_half is None else float(ref_half[match])
        reference_settled = (
            None
            if gamma_ref_half is None or omega_ref_half is None
            else _settled((gamma_ref, gamma_ref_half), (omega_ref, omega_ref_half))
        )
        gkx_settled = _settled((gamma, gamma_half), (omega, omega_half))
        rows.append(
            {
                "case": case["key"],
                "ky": float(ky),
                "ky_reference": float(spectrum.ky[match]),
                "gamma_reference": gamma_ref,
                "gamma_gkx": gamma,
                "gamma_relative_difference": _relative(gamma, gamma_ref),
                "omega_reference": omega_ref,
                "omega_gkx": omega,
                "gamma_half_time": gamma_half,
                "omega_half_time": omega_half,
                "omega_relative_difference": _relative(omega, omega_ref),
                "gamma_half_time_shift": _relative(gamma_half, gamma),
                "omega_half_time_shift": _relative(omega_half, omega),
                "converged": gkx_settled,  # Historical field: GKX only.
                "gamma_reference_half_time": gamma_ref_half,
                "omega_reference_half_time": omega_ref_half,
                "reference_settled": reference_settled,
                "both_codes_settled": gkx_settled and reference_settled is True,
                # True when the difference this row reports is inside the
                # spread the reference itself shows between two legitimate
                # builds of the same commit. Such a row is not evidence of
                # agreement or of disagreement; it is below the instrument.
                "within_build_reproducibility_floor": (
                    None
                    if floor is None
                    else bool(
                        abs(_relative(gamma, gamma_ref)) <= floor
                        and abs(_relative(omega, omega_ref)) <= floor
                    )
                ),
            }
        )

    settled = [r for r in rows if r["converged"]]
    finite_errors = [
        abs(r["gamma_relative_difference"])
        for r in settled
        if np.isfinite(r["gamma_relative_difference"])
    ]
    peak_index = int(np.argmax([r["gamma_reference"] for r in rows]))
    return {
        "key": case["key"],
        "order": int(order),
        "label": case["label"],
        "configuration": str(case["config"]),
        "reference_output": str(case["reference_output"]),
        "reference_provenance": case["reference_provenance"],
        "device": case["device"],
        "drive": case["drive"],
        "electrons": case["electrons"],
        "field_model": case["field_model"],
        "geometry": case["geometry"],
        "damp_ends_rate": float(cfg.collisions.damp_ends_amp),
        "resolution": {
            "Nl": int(case["Nl"]),
            "Nm": int(case["Nm"]),
            "Nz": int(cfg.grid.Nz),
            "Ny": int(cfg.grid.Ny),
            "dt": dt,
            "steps": steps,
            "t_end": t_end,
            "fit_window": [late * t_end, t_end],
            "half_time_probe": [late * half_end, half_end],
        },
        "reference_trace": {
            "samples": spectrum.samples,
            "t_end": spectrum.t_end,
            "nonfinite_values": spectrum.nonfinite,
        },
        "cost": {
            "gkx_scan_seconds": float(elapsed),
            "gkx_peak_host_rss_mb": _peak_rss_mb(),
            "gkx_peak_device_mb": _device_peak_mb(),
            "reference_seconds": case.get("reference_seconds"),
            "reference_peak_host_rss_mb": case.get("reference_peak_host_rss_mb"),
            "reference_peak_device_mb": case.get("reference_peak_device_mb"),
        },
        "build_reproducibility_floor": floor,
        "summary": {
            "settled_ky_count": len(settled),
            "both_codes_settled_ky_count": sum(r["both_codes_settled"] for r in rows),
            "finite_relative_error_ky_count": len(finite_errors),
            "total_ky_count": len(rows),
            "max_absolute_gamma_relative_difference_settled": (
                max(finite_errors) if finite_errors else float("nan")
            ),
            "peak_ky": rows[peak_index]["ky"],
            "gamma_relative_difference_at_peak": rows[peak_index][
                "gamma_relative_difference"
            ],
            "omega_relative_difference_at_peak": rows[peak_index][
                "omega_relative_difference"
            ],
        },
        "rows": rows,
    }


def write_csv(records: list[dict[str, Any]], path: Path) -> None:
    """Write one row per (case, ky) with the reported differences."""

    import csv

    records = sorted(records, key=lambda record: record.get("order", 0))

    fields = [
        "case",
        "ky",
        "gamma_reference",
        "gamma_gkx",
        "gamma_relative_difference",
        "omega_reference",
        "omega_gkx",
        "omega_relative_difference",
        "gamma_half_time_shift",
        "omega_half_time_shift",
        "converged",
        "within_build_reproducibility_floor",
        "gamma_reference_half_time",
        "omega_reference_half_time",
        "reference_settled",
        "both_codes_settled",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            for row in record["rows"]:
                writer.writerow({key: row.get(key) for key in fields})


def write_figure(records: list[dict[str, Any]], path: Path) -> None:
    """Draw a growth-rate/frequency panel pair for every case."""

    import matplotlib

    records = sorted(records, key=lambda record: record.get("order", 0))

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = len(records)
    fig, axes = plt.subplots(
        2, ncols, figsize=(3.1 * ncols, 6.0), squeeze=False, sharex="col"
    )
    for column, record in enumerate(records):
        rows = record["rows"]
        ky = [r["ky"] for r in rows]
        top, bottom = axes[0][column], axes[1][column]
        for axis, ref_key, gkx_key, colour in (
            (top, "gamma_reference", "gamma_gkx", "C0"),
            (bottom, "omega_reference", "omega_gkx", "C1"),
        ):
            reference = [r[ref_key] for r in rows]
            axis.plot(
                ky, reference, "o", fillstyle="none", color="0.25", label="reference"
            )
            settled_ky = [r["ky"] for r in rows if r["converged"]]
            settled = [r[gkx_key] for r in rows if r["converged"]]
            loose_ky = [r["ky"] for r in rows if not r["converged"]]
            loose = [r[gkx_key] for r in rows if not r["converged"]]
            axis.plot(settled_ky, settled, "x", color=colour, label="GKX, settled")
            axis.plot(
                loose_ky,
                loose,
                "+",
                color=colour,
                alpha=0.45,
                label="GKX, not settled",
            )
            # Scale on the reference spectrum so that an unsettled point on a
            # different branch runs off the panel instead of flattening it.
            low, high = min(reference), max(reference)
            span = max(high - low, abs(high), 1.0e-12)
            axis.set_ylim(low - 0.35 * span, high + 0.35 * span)
        top.set_title(record["label"], fontsize=8)
        bottom.set_xlabel(r"$k_y \rho_i$", fontsize=8)
        for axis in (top, bottom):
            axis.tick_params(labelsize=7)
            axis.grid(alpha=0.25)
        if column == 0:
            top.set_ylabel(r"$\gamma\, L_{\rm ref}/v_{\rm t}$", fontsize=8)
            bottom.set_ylabel(r"$\omega\, L_{\rm ref}/v_{\rm t}$", fontsize=8)
            top.legend(fontsize=7, frameon=False)
    fig.suptitle(
        "Linear parity against the reference comparison code. Panels are scaled "
        "on the reference spectrum, so unsettled points may run off scale.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--reference-dir", type=Path, default=None)
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--stem", type=Path, default=DEFAULT_STEM)
    parser.add_argument(
        "--merge",
        type=Path,
        default=None,
        help="Existing matrix JSON whose cases are kept when absent here.",
    )
    parser.add_argument(
        "--attach-costs",
        type=Path,
        default=None,
        help=(
            "JSON of reference-code cost rows keyed by case; merged into an "
            "existing matrix without rerunning any physics"
        ),
    )
    return parser


def attach_costs(stem: Path, costs_path: Path) -> None:
    """Merge measured reference-code cost into an existing matrix artifact."""

    json_path = stem.with_suffix(".json")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    costs = json.loads(costs_path.read_text(encoding="utf-8"))
    payload["cases"].sort(key=lambda record: record.get("order", 0))
    for record in payload["cases"]:
        entry = costs.get(record["key"])
        if not entry:
            continue
        cost = record["cost"]
        cost.update(entry)
        reference_t = entry.get("reference_t_end")
        reference_s = entry.get("reference_seconds")
        if reference_t and reference_s:
            cost["reference_seconds_per_unit_time"] = float(reference_s) / float(
                reference_t
            )
        gkx_t = record["resolution"]["t_end"]
        if gkx_t:
            cost["gkx_seconds_per_unit_time"] = float(cost["gkx_scan_seconds"]) / float(
                gkx_t
            )
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_csv(payload["cases"], stem.with_suffix(".csv"))
    write_figure(payload["cases"], stem.with_suffix(".png"))
    print(f"attached reference cost rows to {json_path}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.attach_costs is not None:
        attach_costs(args.stem, args.attach_costs)
        return
    manifest = tomllib.loads(args.manifest.read_text(encoding="utf-8"))
    reference_dir = args.reference_dir
    if reference_dir is None:
        env = os.environ.get("GX_PARITY_REF_DIR")
        if not env:
            raise SystemExit(
                "set GX_PARITY_REF_DIR or pass --reference-dir to locate the "
                "reference-code outputs"
            )
        reference_dir = Path(env)
    reference_dir = reference_dir.expanduser().resolve()

    cases = manifest["case"]
    if args.cases:
        wanted = set(args.cases)
        cases = [case for case in cases if case["key"] in wanted]
        unknown = wanted.difference({case["key"] for case in cases})
        if unknown:
            raise SystemExit(f"unknown cases: {sorted(unknown)}")

    order = {case["key"]: index for index, case in enumerate(manifest["case"])}
    records = [
        run_case(case, reference_dir=reference_dir, order=order[case["key"]])
        for case in cases
    ]
    for record in records:
        summary = record["summary"]
        floor = record.get("build_reproducibility_floor")
        print(
            f"{record['key']:34s} settled {summary['settled_ky_count']}/{summary['total_ky_count']} ky"
            f"  max|d gamma|(settled)={summary['max_absolute_gamma_relative_difference_settled']:.4f}"
            f"  at peak ky={summary['peak_ky']:.3f}: "
            f"d gamma={summary['gamma_relative_difference_at_peak']:+.4f} "
            f"d omega={summary['omega_relative_difference_at_peak']:+.4f}"
            + ("" if floor is None else f"  [build floor {floor:.4f}]"),
            flush=True,
        )

    if args.merge is not None and args.merge.exists():
        previous = json.loads(args.merge.read_text(encoding="utf-8"))
        keys = {record["key"] for record in records}
        records = [
            c for c in previous.get("cases", []) if c["key"] not in keys
        ] + records
        records.sort(key=lambda record: record.get("order", 0))

    stem = args.stem
    payload = {
        "gate_index_include": False,
        "description": (
            "Linear growth rate and frequency parity between GKX and the "
            "reference comparison code, with matched velocity resolution and "
            "imported reference geometry."
        ),
        "manifest": _repo_relative(args.manifest),
        "cases": records,
    }
    json_path = stem.with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_csv(records, stem.with_suffix(".csv"))
    write_figure(records, stem.with_suffix(".png"))
    print(f"saved {json_path}")
    print(f"saved {stem.with_suffix('.csv')}")
    print(f"saved {stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
