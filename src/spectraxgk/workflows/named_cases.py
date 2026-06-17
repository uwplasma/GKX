"""Named benchmark-case executable workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class NamedLinearCommandDeps:
    """Patchable dependencies for named linear executable commands."""

    load_case_from_toml: Callable[..., tuple[str, Any, dict[str, Any]]]
    resolve_case: Callable[[str], tuple[type[Any], Callable[..., Any]]]
    load_linear_terms_from_toml: Callable[[dict[str, Any]], Any]
    load_krylov_from_toml: Callable[[dict[str, Any]], Any]
    should_show_progress: Callable[[Any, bool], bool]
    print_linear_run_header: Callable[..., None]
    status_printer: Callable[[str], Callable[[str], None]]
    run_linear_scan: Callable[..., Any]
    build_spectral_grid: Callable[[Any], Any]
    extract_mode_time_series: Callable[..., Any]
    growth_fit_figure: Callable[..., tuple[Any, Any]]
    extract_eigenfunction: Callable[..., Any]
    normalize_eigenfunction: Callable[..., Any]
    set_plot_style: Callable[[], None]
    load_cyclone_reference: Callable[[], Any]
    load_etg_reference: Callable[[], Any]
    scan_comparison_figure: Callable[..., tuple[Any, Any]]


__all__ = [
    "NamedLinearCommandDeps",
    "load_scan_ky",
    "run_named_linear_command",
    "scan_named_linear_command",
]


def load_scan_ky(data: dict[str, Any]) -> np.ndarray:
    """Read optional ``[scan].ky`` values from a named-case TOML payload."""

    scan = data.get("scan", {})
    ky_vals = scan.get("ky")
    if ky_vals is None:
        return np.asarray([])
    return np.asarray(ky_vals, dtype=float)


def run_named_linear_command(args: Any, *, deps: NamedLinearCommandDeps) -> int:
    """Run one named linear case after executable parser dispatch."""

    case_name, cfg, data = deps.load_case_from_toml(args.config, args.case)
    case_cls, run_fn = deps.resolve_case(case_name)
    _ = case_cls
    run_cfg = data.get("run", {})
    fit_cfg = dict(data.get("fit", {}))

    ky = args.ky if args.ky is not None else run_cfg.get("ky", 0.3)
    Nl = args.Nl if args.Nl is not None else run_cfg.get("Nl", 24)
    Nm = args.Nm if args.Nm is not None else run_cfg.get("Nm", 12)
    solver = args.solver if args.solver is not None else run_cfg.get("solver", "auto")
    fit_signal = (
        args.fit_signal
        if args.fit_signal is not None
        else fit_cfg.pop("fit_signal", "auto")
    )
    method = (
        args.method
        if args.method is not None
        else run_cfg.get("method", cfg.time.method)
    )
    dt = args.dt if args.dt is not None else run_cfg.get("dt", cfg.time.dt)
    steps = (
        args.steps
        if args.steps is not None
        else run_cfg.get("steps", int(round(cfg.time.t_max / cfg.time.dt)))
    )
    sample_stride = (
        args.sample_stride
        if args.sample_stride is not None
        else run_cfg.get("sample_stride", getattr(cfg.time, "sample_stride", None))
    )
    show_progress = deps.should_show_progress(
        args, bool(getattr(cfg.time, "progress_bar", False))
    )

    terms = deps.load_linear_terms_from_toml(data)
    krylov_cfg = deps.load_krylov_from_toml(data)
    deps.print_linear_run_header(
        label=f"named linear {case_name} run",
        config_path=str(args.config),
        ky=float(ky),
        Nl=int(Nl),
        Nm=int(Nm),
        solver=str(solver),
        method=str(method),
        dt=float(dt),
        steps=int(steps),
        grid_shape=(int(cfg.grid.Nx), int(cfg.grid.Ny), int(cfg.grid.Nz)),
        show_progress=show_progress,
        extra="detected named case TOML; using run-linear path",
    )

    result = run_fn(
        ky_target=float(ky),
        cfg=cfg,
        Nl=int(Nl),
        Nm=int(Nm),
        solver=str(solver),
        method=str(method),
        dt=float(dt),
        steps=int(steps),
        sample_stride=None if sample_stride is None else int(sample_stride),
        krylov_cfg=krylov_cfg,
        terms=terms,
        fit_signal=str(fit_signal),
        show_progress=show_progress,
        status_callback=deps.status_printer(case_name),
        **fit_cfg,
    )
    print(f"ky={result.ky:.4f} gamma={result.gamma:.6f} omega={result.omega:.6f}")

    if args.plot:
        _write_named_linear_plots(case_name, cfg, result, Path(args.outdir), deps=deps)
    return 0


def _write_named_linear_plots(
    case_name: str,
    cfg: Any,
    result: Any,
    outdir: Path,
    *,
    deps: NamedLinearCommandDeps,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    grid = deps.build_spectral_grid(cfg.grid)
    if result.t.size > 1:
        signal = deps.extract_mode_time_series(
            result.phi_t, result.selection, method="project"
        )
        fig, _ax = deps.growth_fit_figure(result.t, signal)
        fig.savefig(outdir / f"{case_name}_ky{result.ky:.3f}_fit.png")
    z_np = np.asarray(grid.z)
    eigen = deps.extract_eigenfunction(result.phi_t, result.t, result.selection, z=z_np)
    eigen = deps.normalize_eigenfunction(eigen, z_np)
    deps.set_plot_style()
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


def scan_named_linear_command(args: Any, *, deps: NamedLinearCommandDeps) -> int:
    """Run a named-case linear ky scan after executable parser dispatch."""

    case_name, cfg, data = deps.load_case_from_toml(args.config, args.case)
    _case_cls, run_fn = deps.resolve_case(case_name)
    scan_cfg = data.get("scan", {})
    fit_cfg = dict(data.get("fit", {}))

    if args.ky_values is not None:
        ky_values = np.asarray(
            [float(x) for x in args.ky_values.split(",") if x.strip() != ""]
        )
    else:
        ky_values = load_scan_ky(data)
    if ky_values.size == 0:
        raise ValueError("No ky values provided. Use --ky-values or [scan].ky in TOML.")

    Nl = args.Nl if args.Nl is not None else scan_cfg.get("Nl", 24)
    Nm = args.Nm if args.Nm is not None else scan_cfg.get("Nm", 12)
    solver = args.solver if args.solver is not None else scan_cfg.get("solver", "auto")
    fit_signal = (
        args.fit_signal
        if args.fit_signal is not None
        else fit_cfg.pop("fit_signal", "auto")
    )
    auto_window = bool(fit_cfg.pop("auto_window", True))
    method = (
        args.method
        if args.method is not None
        else scan_cfg.get("method", cfg.time.method)
    )
    dt = args.dt if args.dt is not None else scan_cfg.get("dt", cfg.time.dt)
    steps = (
        args.steps
        if args.steps is not None
        else scan_cfg.get("steps", int(round(cfg.time.t_max / cfg.time.dt)))
    )

    terms = deps.load_linear_terms_from_toml(data)
    krylov_cfg = deps.load_krylov_from_toml(data)
    scan = deps.run_linear_scan(
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
        auto_window=auto_window,
        window_kw=fit_cfg,
        run_kwargs={"terms": terms, "fit_signal": str(fit_signal)}
        if terms is not None
        else {"fit_signal": str(fit_signal)},
    )

    for ky, g, w in zip(scan.ky, scan.gamma, scan.omega):
        print(f"ky={ky:.4f} gamma={g:.6f} omega={w:.6f}")

    if args.plot:
        _write_named_scan_plot(case_name, scan, Path(args.outdir), deps=deps)
    return 0


def _write_named_scan_plot(
    case_name: str,
    scan: Any,
    outdir: Path,
    *,
    deps: NamedLinearCommandDeps,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    ref = None
    if case_name == "cyclone":
        ref = deps.load_cyclone_reference()
    elif case_name == "etg":
        ref = deps.load_etg_reference()
    if ref is None:
        print("No reference available for this case; skipping comparison plot.")
        return
    fig, _ax = deps.scan_comparison_figure(
        scan.ky,
        scan.gamma,
        scan.omega,
        r"$k_y \rho_i$",
        f"{case_name.upper()} scan",
        x_ref=ref.ky,
        gamma_ref=ref.gamma,
        omega_ref=ref.omega,
        log_x=True,
    )
    fig.savefig(outdir / f"{case_name}_scan_comparison.png")
