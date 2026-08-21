import argparse
from pathlib import Path

import numpy as np

from gkx import (
    LinearValidationPanel,
    growth_fit_figure,
    linear_validation_figure,
    load_runtime_from_toml,
    normalize_eigenfunction,
    run_runtime_linear,
    run_runtime_scan,
    scan_comparison_figure,
)
from gkx.benchmarking.shared import load_etg_reference


CONFIG = Path("examples/linear/axisymmetric/etg.toml")
N_LAGUERRE = 24
N_HERMITE = 8


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Electron-temperature-gradient benchmark."
    )
    parser.add_argument("--ky", type=float, default=None, help="Run a single ky value.")
    parser.add_argument("--outdir", default=".", help="Output directory.")
    parser.add_argument("--no-fit", action="store_true", help="Skip fit-window plot.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg, _ = load_runtime_from_toml(CONFIG)
    ref = load_etg_reference()
    ky_values = (
        np.array([float(args.ky)]) if args.ky is not None else np.asarray(ref.ky)
    )
    time_options = dict(
        dt=cfg.time.dt,
        steps=int(round(cfg.time.t_max / cfg.time.dt)),
        method=cfg.time.method,
        solver="time",
        auto_window=False,
        tmin=1.0,
        tmax=cfg.time.t_max,
        fit_signal="phi",
        mode_method="z_index",
    )

    scan = run_runtime_scan(
        cfg,
        ky_values,
        Nl=N_LAGUERRE,
        Nm=N_HERMITE,
        batch_ky=True,
        **time_options,
    )
    ky_selected = float(scan.ky[int(np.nanargmax(scan.gamma))])
    mode = run_runtime_linear(
        cfg,
        ky_target=ky_selected,
        Nl=N_LAGUERRE,
        Nm=N_HERMITE,
        **time_options,
    )

    if (
        mode.eigenfunction is None
        or mode.z is None
        or mode.t is None
        or mode.signal is None
    ):
        raise RuntimeError("time-integrated runtime diagnostics are required")
    eig = normalize_eigenfunction(mode.eigenfunction, mode.z)
    panel = LinearValidationPanel(
        name="ETG",
        z=mode.z,
        eigenfunction=eig,
        x=scan.ky,
        gamma=scan.gamma,
        omega=scan.omega,
        x_label=r"$k_y \rho_i$",
        x_ref=ref.ky,
        gamma_ref=ref.gamma,
        omega_ref=ref.omega,
        log_x=True,
    )
    fig, _axes = linear_validation_figure([panel])
    fig.savefig(outdir / "etg_validation.png", dpi=200)

    fig, _axes = scan_comparison_figure(
        scan.ky,
        scan.gamma,
        scan.omega,
        r"$k_y \rho_i$",
        "ETG comparison",
        x_ref=ref.ky,
        gamma_ref=ref.gamma,
        omega_ref=ref.omega,
        log_x=True,
    )
    fig.savefig(outdir / "etg_comparison.png", dpi=200)

    if not args.no_fit:
        fig, _axes = growth_fit_figure(
            mode.t,
            mode.signal,
            tmin=mode.fit_window_tmin,
            tmax=mode.fit_window_tmax,
            title="ETG fit window",
        )
        fig.savefig(outdir / "etg_fit_window.png", dpi=200)


if __name__ == "__main__":
    main()
