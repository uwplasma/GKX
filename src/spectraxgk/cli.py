"""Command line interface for SPECTRAX-GK."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from spectraxgk.analysis import extract_eigenfunction, extract_mode_time_series
from spectraxgk.benchmarking import normalize_eigenfunction, run_linear_scan
from spectraxgk.benchmarks import (
    ETGBaseCase,
    CycloneBaseCase,
    load_cyclone_reference,
    load_etg_reference,
    run_cyclone_linear,
    run_etg_linear,
)
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid
from spectraxgk.io import load_case_from_toml, load_krylov_from_toml, load_linear_terms_from_toml, load_runtime_from_toml
from spectraxgk.plotting import growth_fit_figure, scan_comparison_figure, set_plot_style
from spectraxgk.runtime import run_runtime_linear, run_runtime_scan


def _cmd_cyclone_info(_: argparse.Namespace) -> int:
    cfg = CycloneBaseCase()
    print("Cyclone base case")
    print(cfg.to_dict())
    return 0


def _cmd_cyclone_kperp(args: argparse.Namespace) -> int:
    cfg = CycloneBaseCase()
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    theta = grid.z
    kx0 = jnp.array(args.kx0)
    ky = jnp.array(args.ky)
    kperp2 = geom.k_perp2(kx0, ky, theta)
    print(f"k_perp^2(theta) min={kperp2.min():.6g} max={kperp2.max():.6g}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spectrax-gk")
    sub = parser.add_subparsers(dest="cmd", required=True)

    info = sub.add_parser("cyclone-info", help="Print Cyclone base case defaults")
    info.set_defaults(func=_cmd_cyclone_info)

    kperp = sub.add_parser("cyclone-kperp", help="Compute k_perp^2(theta)")
    kperp.add_argument("--kx0", type=float, default=0.0)
    kperp.add_argument("--ky", type=float, default=0.3)
    kperp.set_defaults(func=_cmd_cyclone_kperp)

    run_linear = sub.add_parser("run-linear", help="Run a single linear case from a TOML config")
    run_linear.add_argument("--config", required=True, help="Path to TOML config")
    run_linear.add_argument("--case", default=None, help="Case name (cyclone, etg, ...)")
    run_linear.add_argument("--ky", type=float, default=None, help="Single ky value")
    run_linear.add_argument("--Nl", type=int, default=None)
    run_linear.add_argument("--Nm", type=int, default=None)
    run_linear.add_argument("--solver", type=str, default=None, help="time or krylov")
    run_linear.add_argument("--method", type=str, default=None, help="time integrator method")
    run_linear.add_argument("--dt", type=float, default=None)
    run_linear.add_argument("--steps", type=int, default=None)
    run_linear.add_argument("--plot", action="store_true", help="Save fit/eigenfunction plots")
    run_linear.add_argument("--outdir", default=".", help="Output directory for plots")
    run_linear.set_defaults(func=_cmd_run_linear)

    scan_linear = sub.add_parser("scan-linear", help="Run a ky scan from a TOML config")
    scan_linear.add_argument("--config", required=True, help="Path to TOML config")
    scan_linear.add_argument("--case", default=None, help="Case name (cyclone, etg, ...)")
    scan_linear.add_argument("--ky-values", type=str, default=None, help="Comma-separated ky list")
    scan_linear.add_argument("--Nl", type=int, default=None)
    scan_linear.add_argument("--Nm", type=int, default=None)
    scan_linear.add_argument("--solver", type=str, default=None, help="time or krylov")
    scan_linear.add_argument("--method", type=str, default=None, help="time integrator method")
    scan_linear.add_argument("--dt", type=float, default=None)
    scan_linear.add_argument("--steps", type=int, default=None)
    scan_linear.add_argument("--plot", action="store_true", help="Save comparison plot if reference exists")
    scan_linear.add_argument("--outdir", default=".", help="Output directory for plots")
    scan_linear.set_defaults(func=_cmd_scan_linear)

    run_runtime = sub.add_parser(
        "run-runtime-linear",
        help="Run one linear point from unified runtime TOML config",
    )
    run_runtime.add_argument("--config", required=True, help="Path to TOML config")
    run_runtime.add_argument("--ky", type=float, default=None, help="Single ky value")
    run_runtime.add_argument("--Nl", type=int, default=None)
    run_runtime.add_argument("--Nm", type=int, default=None)
    run_runtime.add_argument("--solver", type=str, default=None, help="time or krylov")
    run_runtime.add_argument("--method", type=str, default=None, help="time integrator method")
    run_runtime.add_argument("--dt", type=float, default=None)
    run_runtime.add_argument("--steps", type=int, default=None)
    run_runtime.set_defaults(func=_cmd_run_runtime_linear)

    scan_runtime = sub.add_parser(
        "scan-runtime-linear",
        help="Run a ky scan from unified runtime TOML config",
    )
    scan_runtime.add_argument("--config", required=True, help="Path to TOML config")
    scan_runtime.add_argument("--ky-values", type=str, default=None, help="Comma-separated ky list")
    scan_runtime.add_argument("--Nl", type=int, default=None)
    scan_runtime.add_argument("--Nm", type=int, default=None)
    scan_runtime.add_argument("--solver", type=str, default=None, help="time or krylov")
    scan_runtime.add_argument("--method", type=str, default=None, help="time integrator method")
    scan_runtime.add_argument("--dt", type=float, default=None)
    scan_runtime.add_argument("--steps", type=int, default=None)
    scan_runtime.set_defaults(func=_cmd_scan_runtime_linear)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


def _resolve_case(case_name: str):
    name = case_name.lower()
    if name == "cyclone":
        return CycloneBaseCase, run_cyclone_linear
    if name == "etg":
        return ETGBaseCase, run_etg_linear
    raise ValueError(f"Unsupported case '{case_name}' in CLI (supported: cyclone, etg)")


def _load_scan_ky(data: dict) -> np.ndarray:
    scan = data.get("scan", {})
    ky_vals = scan.get("ky")
    if ky_vals is None:
        return np.asarray([])
    return np.asarray(ky_vals, dtype=float)


def _cmd_run_linear(args: argparse.Namespace) -> int:
    case_name, cfg, data = load_case_from_toml(args.config, args.case)
    case_cls, run_fn = _resolve_case(case_name)
    _ = case_cls  # keep type checkers happy
    run_cfg = data.get("run", {})
    fit_cfg = data.get("fit", {})

    ky = args.ky if args.ky is not None else run_cfg.get("ky", 0.3)
    Nl = args.Nl if args.Nl is not None else run_cfg.get("Nl", 24)
    Nm = args.Nm if args.Nm is not None else run_cfg.get("Nm", 12)
    solver = args.solver if args.solver is not None else run_cfg.get("solver", "krylov")
    method = args.method if args.method is not None else run_cfg.get("method", cfg.time.method)
    dt = args.dt if args.dt is not None else run_cfg.get("dt", cfg.time.dt)
    steps = args.steps if args.steps is not None else run_cfg.get("steps", int(round(cfg.time.t_max / cfg.time.dt)))

    terms = load_linear_terms_from_toml(data)
    krylov_cfg = load_krylov_from_toml(data)

    result = run_fn(
        ky_target=float(ky),
        cfg=cfg,
        Nl=int(Nl),
        Nm=int(Nm),
        solver=str(solver),
        method=str(method),
        dt=float(dt),
        steps=int(steps),
        krylov_cfg=krylov_cfg,
        terms=terms,
        **fit_cfg,
    )
    print(f"ky={result.ky:.4f} gamma={result.gamma:.6f} omega={result.omega:.6f}")

    if args.plot:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        grid = build_spectral_grid(cfg.grid)
        if result.t.size > 1:
            signal = extract_mode_time_series(result.phi_t, result.selection, method="project")
            fig, _ax = growth_fit_figure(result.t, signal)
            fig.savefig(outdir / f"{case_name}_ky{result.ky:.3f}_fit.png")
        z_np = np.asarray(grid.z)
        eigen = extract_eigenfunction(result.phi_t, result.t, result.selection, z=z_np)
        eigen = normalize_eigenfunction(eigen, z_np)
        set_plot_style()
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        ax.plot(grid.z, eigen.real, label="Re")
        ax.plot(grid.z, eigen.imag, linestyle="--", label="Im")
        ax.set_xlabel(r"$\\theta$")
        ax.set_ylabel(r"$\\phi$")
        ax.set_title(f"{case_name} ky={result.ky:.3f}")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(outdir / f"{case_name}_ky{result.ky:.3f}_eig.png")
        plt.close(fig)
    return 0


def _cmd_scan_linear(args: argparse.Namespace) -> int:
    case_name, cfg, data = load_case_from_toml(args.config, args.case)
    _case_cls, run_fn = _resolve_case(case_name)
    scan_cfg = data.get("scan", {})
    fit_cfg = data.get("fit", {})

    if args.ky_values is not None:
        ky_values = np.asarray([float(x) for x in args.ky_values.split(",") if x.strip() != ""])
    else:
        ky_values = _load_scan_ky(data)
    if ky_values.size == 0:
        raise ValueError("No ky values provided. Use --ky-values or [scan].ky in TOML.")

    Nl = args.Nl if args.Nl is not None else scan_cfg.get("Nl", 24)
    Nm = args.Nm if args.Nm is not None else scan_cfg.get("Nm", 12)
    solver = args.solver if args.solver is not None else scan_cfg.get("solver", "krylov")
    method = args.method if args.method is not None else scan_cfg.get("method", cfg.time.method)
    dt = args.dt if args.dt is not None else scan_cfg.get("dt", cfg.time.dt)
    steps = args.steps if args.steps is not None else scan_cfg.get("steps", int(round(cfg.time.t_max / cfg.time.dt)))

    terms = load_linear_terms_from_toml(data)
    krylov_cfg = load_krylov_from_toml(data)

    scan = run_linear_scan(
        ky_values=ky_values,
        run_linear_fn=run_fn,
        cfg=cfg,
        Nl=int(Nl),
        Nm=int(Nm),
        dt=float(dt),
        steps=int(steps),
        method=str(method),
        solver=str(solver),
        krylov_cfg=krylov_cfg,
        window_kw=fit_cfg,
        run_kwargs={"terms": terms} if terms is not None else None,
    )

    for ky, g, w in zip(scan.ky, scan.gamma, scan.omega):
        print(f"ky={ky:.4f} gamma={g:.6f} omega={w:.6f}")

    if args.plot:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        ref = None
        if case_name == "cyclone":
            ref = load_cyclone_reference()
        elif case_name == "etg":
            ref = load_etg_reference()
        if ref is not None:
            fig, _ax = scan_comparison_figure(
                scan.ky,
                scan.gamma,
                scan.omega,
                r"$k_y \\rho_i$",
                f"{case_name.upper()} scan",
                x_ref=ref.ky,
                gamma_ref=ref.gamma,
                omega_ref=ref.omega,
                log_x=True,
            )
            fig.savefig(outdir / f"{case_name}_scan_comparison.png")
        else:
            print("No reference available for this case; skipping comparison plot.")
    return 0


_RUNTIME_FIT_KEYS = {
    "auto_window",
    "tmin",
    "tmax",
    "window_fraction",
    "min_points",
    "start_fraction",
    "growth_weight",
    "require_positive",
    "min_amp_fraction",
    "mode_method",
}


def _cmd_run_runtime_linear(args: argparse.Namespace) -> int:
    cfg, data = load_runtime_from_toml(args.config)
    run_cfg = data.get("run", {})
    fit_cfg = {k: v for k, v in data.get("fit", {}).items() if k in _RUNTIME_FIT_KEYS}

    ky = float(args.ky if args.ky is not None else run_cfg.get("ky", 0.3))
    Nl = int(args.Nl if args.Nl is not None else run_cfg.get("Nl", 24))
    Nm = int(args.Nm if args.Nm is not None else run_cfg.get("Nm", 12))
    solver = str(args.solver if args.solver is not None else run_cfg.get("solver", "krylov"))
    method = args.method if args.method is not None else run_cfg.get("method", None)
    dt = args.dt if args.dt is not None else run_cfg.get("dt", None)
    steps = args.steps if args.steps is not None else run_cfg.get("steps", None)

    res = run_runtime_linear(
        cfg,
        ky_target=ky,
        Nl=Nl,
        Nm=Nm,
        solver=solver,
        method=method,
        dt=dt,
        steps=steps,
        **fit_cfg,
    )
    print(f"ky={res.ky:.4f} gamma={res.gamma:.6f} omega={res.omega:.6f}")
    return 0


def _cmd_scan_runtime_linear(args: argparse.Namespace) -> int:
    cfg, data = load_runtime_from_toml(args.config)
    scan_cfg = data.get("scan", {})
    fit_cfg = {k: v for k, v in data.get("fit", {}).items() if k in _RUNTIME_FIT_KEYS}

    if args.ky_values is not None:
        ky_values = np.asarray([float(x) for x in args.ky_values.split(",") if x.strip()], dtype=float)
    else:
        ky_raw = scan_cfg.get("ky", [])
        ky_values = np.asarray(ky_raw, dtype=float)
    if ky_values.size == 0:
        raise ValueError("No ky values provided. Use --ky-values or [scan].ky in TOML.")

    Nl = int(args.Nl if args.Nl is not None else scan_cfg.get("Nl", 24))
    Nm = int(args.Nm if args.Nm is not None else scan_cfg.get("Nm", 12))
    solver = str(args.solver if args.solver is not None else scan_cfg.get("solver", "krylov"))
    method = args.method if args.method is not None else scan_cfg.get("method", None)
    dt = args.dt if args.dt is not None else scan_cfg.get("dt", None)
    steps = args.steps if args.steps is not None else scan_cfg.get("steps", None)

    scan = run_runtime_scan(
        cfg,
        ky_values.tolist(),
        Nl=Nl,
        Nm=Nm,
        solver=solver,
        method=method,
        dt=dt,
        steps=steps,
        **fit_cfg,
    )
    for ky, g, w in zip(scan.ky, scan.gamma, scan.omega):
        print(f"ky={ky:.4f} gamma={g:.6f} omega={w:.6f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
