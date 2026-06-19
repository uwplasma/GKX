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


@dataclass(frozen=True)
class NamedLinearRunOptions:
    """Resolved options for one named linear initial-value command."""

    ky: float
    Nl: int
    Nm: int
    solver: str
    fit_signal: str
    method: str
    dt: float
    steps: int
    sample_stride: int | None
    show_progress: bool
    fit_kwargs: dict[str, Any]


@dataclass(frozen=True)
class NamedLinearScanOptions:
    """Resolved options for one named linear scan command."""

    ky_values: np.ndarray
    Nl: int
    Nm: int
    solver: str
    fit_signal: str
    auto_window: bool
    method: str
    dt: float
    steps: int
    fit_kwargs: dict[str, Any]


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


def _parse_named_scan_values(raw: str | None, data: dict[str, Any]) -> np.ndarray:
    """Resolve scan ky values from a comma-separated override or TOML payload."""

    if raw is not None:
        return np.asarray([float(x) for x in raw.split(",") if x.strip()], dtype=float)
    return load_scan_ky(data)


def _arg_or_mapping(args: Any, mapping: dict[str, Any], name: str, default: Any) -> Any:
    """Resolve command-line overrides before named-case TOML/default values."""

    value = getattr(args, name, None)
    if value is not None:
        return value
    return mapping.get(name, default)


def _resolve_named_linear_run_options(
    args: Any,
    cfg: Any,
    data: dict[str, Any],
    *,
    deps: NamedLinearCommandDeps,
) -> NamedLinearRunOptions:
    """Resolve named initial-value command options from CLI, TOML, and defaults."""

    run_cfg = data.get("run", {})
    fit_cfg = dict(data.get("fit", {}))
    method = _arg_or_mapping(args, run_cfg, "method", cfg.time.method)
    dt = _arg_or_mapping(args, run_cfg, "dt", cfg.time.dt)
    steps = _arg_or_mapping(
        args,
        run_cfg,
        "steps",
        int(round(cfg.time.t_max / cfg.time.dt)),
    )
    sample_stride = _arg_or_mapping(
        args,
        run_cfg,
        "sample_stride",
        getattr(cfg.time, "sample_stride", None),
    )
    fit_signal = (
        args.fit_signal
        if args.fit_signal is not None
        else fit_cfg.pop("fit_signal", "auto")
    )
    return NamedLinearRunOptions(
        ky=float(_arg_or_mapping(args, run_cfg, "ky", 0.3)),
        Nl=int(_arg_or_mapping(args, run_cfg, "Nl", 24)),
        Nm=int(_arg_or_mapping(args, run_cfg, "Nm", 12)),
        solver=str(_arg_or_mapping(args, run_cfg, "solver", "auto")),
        fit_signal=str(fit_signal),
        method=str(method),
        dt=float(dt),
        steps=int(steps),
        sample_stride=None if sample_stride is None else int(sample_stride),
        show_progress=deps.should_show_progress(
            args, bool(getattr(cfg.time, "progress_bar", False))
        ),
        fit_kwargs=fit_cfg,
    )


def _resolve_named_linear_scan_options(
    args: Any,
    cfg: Any,
    data: dict[str, Any],
) -> NamedLinearScanOptions:
    """Resolve named scan options from CLI, TOML, and defaults."""

    scan_cfg = data.get("scan", {})
    fit_cfg = dict(data.get("fit", {}))
    ky_values = _parse_named_scan_values(args.ky_values, data)
    if ky_values.size == 0:
        raise ValueError("No ky values provided. Use --ky-values or [scan].ky in TOML.")

    method = _arg_or_mapping(args, scan_cfg, "method", cfg.time.method)
    dt = _arg_or_mapping(args, scan_cfg, "dt", cfg.time.dt)
    steps = _arg_or_mapping(
        args,
        scan_cfg,
        "steps",
        int(round(cfg.time.t_max / cfg.time.dt)),
    )
    fit_signal = (
        args.fit_signal
        if args.fit_signal is not None
        else fit_cfg.pop("fit_signal", "auto")
    )
    auto_window = bool(fit_cfg.pop("auto_window", True))
    return NamedLinearScanOptions(
        ky_values=ky_values,
        Nl=int(_arg_or_mapping(args, scan_cfg, "Nl", 24)),
        Nm=int(_arg_or_mapping(args, scan_cfg, "Nm", 12)),
        solver=str(_arg_or_mapping(args, scan_cfg, "solver", "auto")),
        fit_signal=str(fit_signal),
        auto_window=auto_window,
        method=str(method),
        dt=float(dt),
        steps=int(steps),
        fit_kwargs=fit_cfg,
    )


def run_named_linear_command(args: Any, *, deps: NamedLinearCommandDeps) -> int:
    """Run one named linear case after executable parser dispatch."""

    case_name, cfg, data = deps.load_case_from_toml(args.config, args.case)
    case_cls, run_fn = deps.resolve_case(case_name)
    _ = case_cls
    opts = _resolve_named_linear_run_options(args, cfg, data, deps=deps)

    terms = deps.load_linear_terms_from_toml(data)
    krylov_cfg = deps.load_krylov_from_toml(data)
    deps.print_linear_run_header(
        label=f"named linear {case_name} run",
        config_path=str(args.config),
        ky=opts.ky,
        Nl=opts.Nl,
        Nm=opts.Nm,
        solver=opts.solver,
        method=opts.method,
        dt=opts.dt,
        steps=opts.steps,
        grid_shape=(int(cfg.grid.Nx), int(cfg.grid.Ny), int(cfg.grid.Nz)),
        show_progress=opts.show_progress,
        extra="detected named case TOML; using run-linear path",
    )

    result = run_fn(
        ky_target=opts.ky,
        cfg=cfg,
        Nl=opts.Nl,
        Nm=opts.Nm,
        solver=opts.solver,
        method=opts.method,
        dt=opts.dt,
        steps=opts.steps,
        sample_stride=opts.sample_stride,
        krylov_cfg=krylov_cfg,
        terms=terms,
        fit_signal=opts.fit_signal,
        show_progress=opts.show_progress,
        status_callback=deps.status_printer(case_name),
        **opts.fit_kwargs,
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
    opts = _resolve_named_linear_scan_options(args, cfg, data)

    terms = deps.load_linear_terms_from_toml(data)
    krylov_cfg = deps.load_krylov_from_toml(data)
    scan = deps.run_linear_scan(
        ky_values=opts.ky_values,
        run_linear_fn=run_fn,
        cfg=cfg,
        Nl=opts.Nl,
        Nm=opts.Nm,
        dt=opts.dt,
        steps=opts.steps,
        method=opts.method,
        solver=opts.solver,
        krylov_cfg=krylov_cfg,
        auto_window=opts.auto_window,
        window_kw=opts.fit_kwargs,
        run_kwargs={"terms": terms, "fit_signal": opts.fit_signal}
        if terms is not None
        else {"fit_signal": opts.fit_signal},
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
