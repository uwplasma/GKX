#!/usr/bin/env python3
"""Reduce matched QA runs and render the accepted equilibrium/transport figures."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
import re
import tempfile

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import numpy as np
from PIL import Image
from scipy.stats import t as student_t

from gkx.diagnostics.analysis import integrated_autocorrelation_time


ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "docs" / "_static"
INPUTS = ROOT / "examples" / "optimization"
PATTERN = re.compile(r"(.+)_(baseline|candidate)_seed(\d+)\.npz")
ORDER = (
    "nominal",
    "dt04",
    "dt025",
    "perp12",
    "perp20",
    "perp24",
    "perp24long",
    "z16",
    "z32",
    "v36",
    "v612",
)
METADATA = {
    "nominal": (16, 16, 24, 4, 8, 0.05),
    "dt04": (16, 16, 24, 4, 8, 0.04),
    "dt025": (16, 16, 24, 4, 8, 0.025),
    "perp12": (12, 12, 24, 4, 8, 0.05),
    "perp20": (20, 20, 24, 4, 8, 0.05),
    "perp24": (24, 24, 24, 4, 8, 0.05),
    "perp24long": (24, 24, 24, 4, 8, 0.05),
    "z16": (16, 16, 16, 4, 8, 0.05),
    "z32": (16, 16, 32, 4, 8, 0.05),
    "v36": (16, 16, 24, 3, 6, 0.05),
    "v612": (16, 16, 24, 6, 12, 0.05),
}
LABELS = {
    "nominal": "nominal",
    "dt04": r"$\Delta t=.04$",
    "dt025": r"$\Delta t=.025$",
    "perp12": r"$N_{x,y}=12$",
    "perp20": r"$N_{x,y}=20$",
    "perp24": r"$N_{x,y}=24$ short",
    "perp24long": r"$N_{x,y}=24$ long",
    "z16": r"$N_z=16$",
    "z32": r"$N_z=32$",
    "v36": r"$(N_l,N_m)=(3,6)$",
    "v612": r"$(N_l,N_m)=(6,12)$",
}


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def trace_stats(path: Path) -> dict[str, float | int | str]:
    match = PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(path)
    data = np.load(path)
    time = np.asarray(data["time"], dtype=float)
    flux = np.asarray(data["heat_flux"], dtype=float)
    tmin, tmax = (1900.0, 2500.0) if match[1] == "perp24long" else (1100.0, 1500.0)
    mask = (time >= tmin) & (time <= tmax)
    time, flux = time[mask], flux[mask]
    if not np.all(np.isfinite(flux)):
        raise ValueError(f"nonfinite analysis window: {path}")
    dt = float(np.median(np.diff(time)))
    tau = integrated_autocorrelation_time(flux, dt)
    neff = min(flux.size, flux.size * dt / (2.0 * tau)) if tau > 0.0 else flux.size
    midpoint = 0.5 * (time[0] + time[-1])
    half_shift = 100.0 * (
        flux[time > midpoint].mean() - flux[time <= midpoint].mean()
    ) / flux.mean()
    trend = 100.0 * np.polyfit(time, flux, 1)[0] * (time[-1] - time[0]) / flux.mean()
    return {
        "case": match[1],
        "design": match[2],
        "seed": int(match[3]),
        "mean": float(flux.mean()),
        "std": float(flux.std(ddof=1)),
        "tau": float(tau),
        "neff": float(neff),
        "sem": float(flux.std(ddof=1) / np.sqrt(neff)),
        "window_in_tau": float((time[-1] - time[0]) / tau),
        "half_shift_percent": float(half_shift),
        "trend_percent": float(trend),
        "elapsed_seconds": float(data["elapsed_seconds"]),
    }


def summarize(rows: list[dict]) -> list[dict]:
    output = []
    for case in ORDER:
        base = {int(r["seed"]): r for r in rows if r["case"] == case and r["design"] == "baseline"}
        candidate = {
            int(r["seed"]): r
            for r in rows
            if r["case"] == case and r["design"] == "candidate"
        }
        seeds = sorted(base.keys() & candidate.keys())
        if not seeds:
            continue
        b = np.asarray([float(base[s]["mean"]) for s in seeds])
        c = np.asarray([float(candidate[s]["mean"]) for s in seeds])
        reduction = 100.0 * (b - c) / b
        scatter_sem = reduction.std(ddof=1) / np.sqrt(len(seeds))
        iat_variance = [
            (100.0 * float(candidate[s]["mean"]) * float(base[s]["sem"]) / float(base[s]["mean"]) ** 2) ** 2
            + (100.0 * float(candidate[s]["sem"]) / float(base[s]["mean"])) ** 2
            for s in seeds
        ]
        iat_sem = np.sqrt(np.sum(iat_variance)) / len(seeds)
        sem = max(scatter_sem, iat_sem)
        critical = student_t.ppf(0.975, len(seeds) - 1)
        mean = float(reduction.mean())
        base_half = np.asarray([float(base[s]["half_shift_percent"]) for s in seeds])
        candidate_half = np.asarray(
            [float(candidate[s]["half_shift_percent"]) for s in seeds]
        )
        nx, ny, nz, nl, nm, timestep = METADATA[case]
        output.append(
            {
                "case": case,
                "nx": nx,
                "ny": ny,
                "nz": nz,
                "nl": nl,
                "nm": nm,
                "p_hyper_m": min(20, max(nm // 2, 1)),
                "timestep": timestep,
                "pairs": len(seeds),
                "baseline_mean": float(b.mean()),
                "candidate_mean": float(c.mean()),
                "reduction_percent": mean,
                "minimum_reduction_percent": float(reduction.min()),
                "maximum_reduction_percent": float(reduction.max()),
                "positive_pairs": int(np.count_nonzero(reduction > 0.0)),
                "scatter_sem_percent": float(scatter_sem),
                "iat_sem_percent": float(iat_sem),
                "conservative_sem_percent": float(sem),
                "ci95_low_percent": float(mean - critical * sem),
                "ci95_high_percent": float(mean + critical * sem),
                "median_tau_baseline": float(np.median([float(base[s]["tau"]) for s in seeds])),
                "median_tau_candidate": float(np.median([float(candidate[s]["tau"]) for s in seeds])),
                "median_window_in_tau": float(
                    np.median(
                        [float(base[s]["window_in_tau"]) for s in seeds]
                        + [float(candidate[s]["window_in_tau"]) for s in seeds]
                    )
                ),
                "minimum_window_in_tau": float(
                    min(
                        [float(base[s]["window_in_tau"]) for s in seeds]
                        + [float(candidate[s]["window_in_tau"]) for s in seeds]
                    )
                ),
                "baseline_half_shift_percent": float(base_half.mean()),
                "baseline_half_shift_sem_percent": float(base_half.std(ddof=1) / np.sqrt(len(seeds))),
                "candidate_half_shift_percent": float(candidate_half.mean()),
                "candidate_half_shift_sem_percent": float(candidate_half.std(ddof=1) / np.sqrt(len(seeds))),
                "mean_abs_half_shift_percent": float(
                    np.mean(
                        [abs(float(base[s]["half_shift_percent"])) for s in seeds]
                        + [abs(float(candidate[s]["half_shift_percent"])) for s in seeds]
                    )
                ),
                "mean_abs_trend_percent": float(
                    np.mean(
                        [abs(float(base[s]["trend_percent"])) for s in seeds]
                        + [abs(float(candidate[s]["trend_percent"])) for s in seeds]
                    )
                ),
                "total_gpu_seconds": float(
                    sum(
                        float(base[s]["elapsed_seconds"])
                        + float(candidate[s]["elapsed_seconds"])
                        for s in seeds
                    )
                ),
            }
        )
    return output


def nominal_timeseries(raw_dir: Path) -> list[dict]:
    values = {}
    time = None
    for design in ("baseline", "candidate"):
        traces = []
        for path in sorted(raw_dir.glob(f"nominal_{design}_seed*.npz")):
            data = np.load(path)
            local_time = np.asarray(data["time"], dtype=float)
            if time is None:
                time = local_time
            elif not np.array_equal(time, local_time):
                raise ValueError(f"time grid differs: {path}")
            traces.append(np.asarray(data["heat_flux"], dtype=float))
        if not traces:
            raise ValueError(f"no nominal {design} traces in {raw_dir}")
        stack = np.stack(traces)
        values[design] = (
            stack.mean(axis=0),
            stack.std(axis=0, ddof=1) / np.sqrt(stack.shape[0]),
        )
    assert time is not None
    sample = range(0, time.size, 5)
    return [
        {
            "time": float(time[i]),
            "baseline_mean": float(values["baseline"][0][i]),
            "baseline_sem": float(values["baseline"][1][i]),
            "candidate_mean": float(values["candidate"][0][i]),
            "candidate_sem": float(values["candidate"][1][i]),
        }
        for i in sample
    ]


def reduce_raw(raw_dir: Path, output_dir: Path) -> None:
    rows = [trace_stats(path) for path in sorted(raw_dir.glob("*.npz"))]
    _write_csv(output_dir / "qa_transport_traces.csv", rows)
    _write_csv(output_dir / "qa_transport_summary.csv", summarize(rows))
    _write_csv(
        output_dir / "qa_transport_nominal_timeseries.csv",
        nominal_timeseries(raw_dir),
    )


def plot_transport(output_dir: Path) -> None:
    time_rows = _read_csv(output_dir / "qa_transport_nominal_timeseries.csv")
    trace_rows = _read_csv(output_dir / "qa_transport_traces.csv")
    summary = _read_csv(output_dir / "qa_transport_summary.csv")
    time = np.asarray([float(row["time"]) for row in time_rows])

    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), constrained_layout=True)
    for design, color in (
        ("baseline", "#444444"),
        ("candidate", "#0072B2"),
    ):
        mean = np.asarray([float(row[f"{design}_mean"]) for row in time_rows])
        sem = np.asarray([float(row[f"{design}_sem"]) for row in time_rows])
        sample = slice(None)
        axes[0].plot(time[sample], mean[sample], color=color, label=design)
        axes[0].fill_between(
            time[sample],
            (mean - sem)[sample],
            (mean + sem)[sample],
            color=color,
            alpha=0.18,
        )
    axes[0].axvspan(1100.0, 1500.0, color="#E69F00", alpha=0.12)
    axes[0].set(xlabel=r"$t\,v_{ti}/a$", ylabel=r"$Q_i/Q_{gB}$", xlim=(0.0, 1500.0))
    axes[0].legend(frameon=False, ncols=1)

    labels = [LABELS[row["case"]] for row in summary]
    values = np.asarray([float(row["reduction_percent"]) for row in summary])
    lower = values - np.asarray([float(row["ci95_low_percent"]) for row in summary])
    upper = np.asarray([float(row["ci95_high_percent"]) for row in summary]) - values
    x = np.arange(len(labels), dtype=float)
    for index, summary_row in enumerate(summary):
        rows = [row for row in trace_rows if row["case"] == summary_row["case"]]
        base = {int(row["seed"]): float(row["mean"]) for row in rows if row["design"] == "baseline"}
        candidate = {
            int(row["seed"]): float(row["mean"])
            for row in rows
            if row["design"] == "candidate"
        }
        seeds = sorted(base.keys() & candidate.keys())
        reductions = [100.0 * (base[s] - candidate[s]) / base[s] for s in seeds]
        offsets = np.linspace(-0.12, 0.12, len(seeds)) if len(seeds) > 1 else [0.0]
        axes[1].scatter(
            index + offsets,
            reductions,
            s=9,
            color="#999999",
            alpha=0.45,
            linewidths=0.0,
        )
    failed = np.asarray([row["case"] == "perp24" for row in summary])
    axes[1].errorbar(
        x[~failed], values[~failed], yerr=(lower[~failed], upper[~failed]),
        fmt="o", color="#0072B2", capsize=3,
    )
    axes[1].errorbar(
        x[failed], values[failed], yerr=(lower[failed], upper[failed]),
        fmt="x", color="#D55E00", capsize=3,
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    for boundary in (2.5, 6.5, 8.5):
        axes[1].axvline(boundary, color="#DDDDDD", linewidth=0.7)
    axes[1].set_xticks(x, labels)
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set(ylabel="QA transport reduction [%]")
    figure.savefig(output_dir / "qa_transport_reduction.svg")
    plt.close(figure)


def _solve_inputs(input_dir: Path):
    import vmex as vj
    from vmex import optimize as opt

    equilibria = []
    for filename in ("input.qa_transport_baseline", "input.qa_transport_candidate"):
        inp = vj.VmecInput.from_file(input_dir / filename)
        inp = replace(
            inp,
            am=np.zeros_like(inp.am),
            pres_scale=0.0,
            ns_array=np.asarray([101]),
            ftol_array=np.asarray([1.0e-10]),
            niter_array=np.asarray([15_000]),
        )
        equilibria.append(opt.solve_equilibrium(inp))
    return equilibria


def plot_equilibria(input_dir: Path, output_dir: Path) -> None:
    import vmex as vj
    from vmex.core.boozer import run_booz_xform
    from vmex.core.plotting import (
        boozer_modB_on_surface,
        surface_modB,
        surface_rz,
    )

    equilibria = _solve_inputs(input_dir)
    theta = np.linspace(0.0, 2.0 * np.pi, 64)
    phi = np.linspace(0.0, 2.0 * np.pi, 128)
    surfaces = []
    boozer = []
    with tempfile.TemporaryDirectory() as tmp:
        for index, equilibrium in enumerate(equilibria):
            wout = equilibrium.wout
            radius, vertical = surface_rz(
                wout, s_index=int(wout.ns) - 1, theta=theta, phi=phi
            )
            field = surface_modB(
                wout, s_index=int(wout.ns) - 1, theta=theta, phi=phi
            )
            phi_grid = np.meshgrid(phi, theta)[0]
            surfaces.append(
                (
                    radius * np.cos(phi_grid),
                    radius * np.sin(phi_grid),
                    vertical,
                    field / field.mean(),
                )
            )
            wout_path = vj.write_wout(Path(tmp) / f"wout_{index}.nc", wout)
            transform = run_booz_xform(wout_path, mbooz=28, nbooz=28)
            theta_b, phi_b, field_b = boozer_modB_on_surface(
                transform, s_index=-1, ntheta=161, nphi=161
            )
            boozer.append(
                (
                    phi_b * int(wout.nfp) / (2.0 * np.pi),
                    theta_b / (2.0 * np.pi),
                    field_b / field_b.mean(),
                )
            )

    all_field = np.concatenate(
        [surface[3].ravel() for surface in surfaces]
        + [coordinates[2].ravel() for coordinates in boozer]
    )
    norm = Normalize(float(all_field.min()), float(all_field.max()))
    figure = plt.figure(figsize=(9.2, 7.0), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(1.08, 1.0))
    titles = ("initial", "optimized")
    for column, (title, surface) in enumerate(zip(titles, surfaces)):
        axis = figure.add_subplot(grid[0, column], projection="3d")
        x, y, z, field = surface
        axis.plot_surface(
            x,
            y,
            z,
            facecolors=cm.viridis(norm(field)),
            rstride=1,
            cstride=1,
            linewidth=0.0,
            antialiased=False,
            shade=False,
        )
        scale = 1.02 * max(np.abs(x).max(), np.abs(y).max())
        axis.auto_scale_xyz([-scale, scale], [-scale, scale], [-0.35 * scale, 0.35 * scale])
        axis.set_box_aspect((1.0, 1.0, 0.55), zoom=1.18)
        axis.view_init(elev=27, azim=-55)
        axis.set_axis_off()
        axis.set_title(f"{title} LCFS", pad=4)

    levels = np.linspace(norm.vmin, norm.vmax, 25)
    for column, (title, coordinates) in enumerate(zip(titles, boozer)):
        axis = figure.add_subplot(grid[1, column])
        phi_b, theta_b, field_b = coordinates
        axis.contourf(
            phi_b,
            theta_b,
            field_b,
            levels=levels,
            cmap="viridis",
            norm=norm,
        )
        axis.contour(
            phi_b,
            theta_b,
            field_b,
            levels=levels[::3],
            colors="white",
            linewidths=0.35,
            alpha=0.7,
        )
        axis.set(
            xlabel=r"Boozer toroidal angle $N_{fp}\zeta_B/2\pi$",
            ylabel=r"Boozer poloidal angle $\theta_B/2\pi$" if column == 0 else "",
            title=f"{title} $|B|$ on the LCFS",
        )
        if column:
            axis.set_yticklabels([])
    scalar = cm.ScalarMappable(norm=norm, cmap="viridis")
    colorbar = figure.colorbar(scalar, ax=figure.axes, location="right", shrink=0.72, pad=0.02)
    colorbar.set_label(r"$|B|/\langle |B|\rangle$")
    output = output_dir / "qa_transport_equilibria.png"
    figure.savefig(
        output,
        dpi=170,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(figure)
    with Image.open(output) as rendered:
        palette = rendered.convert("RGB").quantize(
            colors=256, method=Image.Quantize.MEDIANCUT
        )
    palette.save(output, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=STATIC)
    parser.add_argument("--input-dir", type=Path, default=INPUTS)
    parser.add_argument("--skip-equilibria", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["svg.fonttype"] = "none"
    if args.raw_dir is not None:
        reduce_raw(args.raw_dir, args.output_dir)
    plot_transport(args.output_dir)
    if not args.skip_equilibria:
        plot_equilibria(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
