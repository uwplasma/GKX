"""Runtime orchestration helpers split from public runtime entry points.

This module owns coordination policy that is not itself a solver kernel:
progress/ETA formatting, combined-ky scan batching, and nonlinear artifact
restart/checkpoint handoff. Callers pass dependency tables so public runtime
and artifact monkeypatch seams remain effective without duplicating orchestration
code.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from spectraxgk.diagnostics.analysis import ModeSelection
from spectraxgk.diagnostics import SimulationDiagnostics
from spectraxgk.workflows.runtime.config import RuntimeConfig
from spectraxgk.workflows.runtime.results import RuntimeLinearScanResult, RuntimeNonlinearResult

__all__ = [
    "NonlinearArtifactPolicy",
    "RuntimeArtifactHandoffDeps",
    "RuntimeProgressSnapshot",
    "RuntimeScanBatchDeps",
    "RuntimeScanDeps",
    "build_runtime_progress_message",
    "format_duration",
    "resolve_nonlinear_artifact_policy",
    "run_runtime_scan_batch",
    "run_runtime_scan_orchestration",
    "run_runtime_nonlinear_artifact_handoff",
]


class RuntimeScanBatchDeps(Protocol):
    """Dependency surface needed by the combined-ky scan batch helper."""

    build_runtime_geometry: Callable[[RuntimeConfig], Any]
    build_runtime_linear_params: Callable[..., Any]
    build_runtime_linear_terms: Callable[[RuntimeConfig], Any]
    build_initial_condition: Callable[..., Any]
    apply_geometry_grid_defaults: Callable[..., Any]
    build_spectral_grid: Callable[[Any], Any]
    select_ky_index: Callable[[Any, float], int]
    midplane_index: Callable[[Any], int]
    integrate_linear_diagnostics: Callable[..., Any]
    extract_mode_time_series: Callable[..., Any]
    fit_growth_rate_auto_with_stats: Callable[..., Any]
    fit_growth_rate_auto: Callable[..., Any]
    fit_growth_rate: Callable[..., Any]
    apply_diagnostic_normalization: Callable[..., tuple[float, float]]


class RuntimeScanDeps(Protocol):
    """Dependency surface for runtime ky-scan orchestration."""

    resolve_runtime_hl_dims: Callable[..., tuple[int, int]]
    normalize_linear_solver_name: Callable[[str], str]
    parallel_requests_combined_ky_scan: Callable[[RuntimeConfig], bool]
    run_runtime_scan_batch: Callable[..., RuntimeLinearScanResult]
    runtime_independent_parallel_plan: Callable[..., Any]
    independent_map: Callable[..., list[Any]]
    run_runtime_scan_ky_task: Callable[[dict[str, Any]], Any]


def run_runtime_scan_orchestration(
    cfg: RuntimeConfig,
    ky_values: Any,
    *,
    Nl: int | None,
    Nm: int | None,
    solver: str,
    method: str | None,
    dt: float | None,
    steps: int | None,
    sample_stride: int | None,
    batch_ky: bool,
    auto_window: bool,
    tmin: float | None,
    tmax: float | None,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
    krylov_cfg: Any,
    mode_method: str,
    fit_signal: str,
    show_progress: bool,
    workers: int,
    parallel_executor: str,
    deps: RuntimeScanDeps,
) -> RuntimeLinearScanResult:
    """Coordinate serial, independent-worker, or combined-ky runtime scans."""

    ky_arr = np.asarray(ky_values, dtype=float)
    Nl_use, Nm_use = deps.resolve_runtime_hl_dims(cfg, Nl=Nl, Nm=Nm)
    solver_key = deps.normalize_linear_solver_name(solver)
    batch_ky = bool(batch_ky or deps.parallel_requests_combined_ky_scan(cfg))
    if batch_ky and solver_key == "krylov":
        raise ValueError("batch_ky is only supported for time integration")
    if batch_ky and bool(getattr(cfg.quasilinear, "enabled", False)):
        raise NotImplementedError(
            "quasilinear scan artifacts currently require serial ky evaluation"
        )
    if batch_ky:
        return deps.run_runtime_scan_batch(
            cfg,
            ky_arr,
            Nl=Nl_use,
            Nm=Nm_use,
            method=method,
            dt=dt,
            steps=steps,
            sample_stride=sample_stride,
            auto_window=auto_window,
            tmin=tmin,
            tmax=tmax,
            window_fraction=window_fraction,
            min_points=min_points,
            start_fraction=start_fraction,
            growth_weight=growth_weight,
            require_positive=require_positive,
            min_amp_fraction=min_amp_fraction,
            mode_method=mode_method,
            fit_signal=fit_signal,
            show_progress=show_progress,
        )

    gamma = np.zeros_like(ky_arr)
    omega = np.zeros_like(ky_arr)
    ql_payloads: list[dict[str, Any]] = []
    tasks = [
        {
            "cfg": cfg,
            "ky": float(ky),
            "Nl": Nl_use,
            "Nm": Nm_use,
            "solver": solver,
            "method": method,
            "dt": dt,
            "steps": steps,
            "sample_stride": sample_stride,
            "auto_window": auto_window,
            "tmin": tmin,
            "tmax": tmax,
            "window_fraction": window_fraction,
            "min_points": min_points,
            "start_fraction": start_fraction,
            "growth_weight": growth_weight,
            "require_positive": require_positive,
            "min_amp_fraction": min_amp_fraction,
            "krylov_cfg": krylov_cfg,
            "mode_method": mode_method,
            "fit_signal": fit_signal,
            "show_progress": show_progress,
        }
        for ky in ky_arr
    ]
    parallel_plan = deps.runtime_independent_parallel_plan(
        cfg, problem_size=int(ky_arr.size), workers=workers, executor=parallel_executor
    )
    results = deps.independent_map(
        deps.run_runtime_scan_ky_task,
        tasks,
        workers=parallel_plan.requested_workers,
        executor=parallel_plan.executor,
    )
    for i, res in enumerate(results):
        gamma[i] = float(res.gamma)
        omega[i] = float(res.omega)
        if res.quasilinear is not None:
            ql_payloads.append(res.quasilinear)
    parallel_payload = parallel_plan.to_dict()
    parallel_payload.update(
        {
            "identity_contract": "independent ky workers must preserve serial ky ordering and values",
            "quasilinear_state_extraction": bool(ql_payloads),
        }
    )
    return RuntimeLinearScanResult(
        ky=ky_arr,
        gamma=gamma,
        omega=omega,
        quasilinear=tuple(ql_payloads) if ql_payloads else None,
        parallel=parallel_payload,
    )


@dataclass(frozen=True)
class RuntimeProgressSnapshot:
    """Computed wall-clock progress fields for a chunked runtime update."""

    progress: float
    eta_seconds: float
    chunk_wall_seconds: float
    elapsed_seconds: float


@dataclass(frozen=True)
class NonlinearArtifactPolicy:
    """Resolved nonlinear artifact/restart policy for a single handoff."""

    out_path: Path | None
    netcdf_output_target: bool
    diagnostics_on: bool
    restart_from: Path | None
    restart_to: Path | None
    resume_requested: bool
    remaining_steps: int | None
    checkpoint_steps: int | None


@dataclass(frozen=True)
class RuntimeArtifactHandoffDeps:
    """Patchable functions used by nonlinear artifact handoff orchestration."""

    is_netcdf_output_target: Callable[[Path], bool]
    resolve_restart_path: Callable[[str | Path, Any], Path]
    resolve_restart_write_path: Callable[[str | Path, Any], Path]
    netcdf_bundle_base: Callable[[Path], Path]
    load_nonlinear_netcdf_diagnostics: Callable[[str | Path], SimulationDiagnostics]
    condense_diagnostics_for_netcdf_output: Callable[
        [SimulationDiagnostics], SimulationDiagnostics
    ]
    concat_runtime_diagnostics: Callable[
        [list[SimulationDiagnostics]], SimulationDiagnostics
    ]
    validate_finite_runtime_result: Callable[[Any], None]
    run_runtime_nonlinear: Callable[..., RuntimeNonlinearResult]
    write_runtime_nonlinear_artifacts: Callable[[str | Path, Any, Any], dict[str, str]]


def format_duration(seconds: float) -> str:
    """Format elapsed seconds as ``MM:SS`` or ``H:MM:SS``."""

    seconds_i = max(int(round(seconds)), 0)
    minutes, secs = divmod(seconds_i, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def build_runtime_progress_message(
    *,
    label: str,
    chunk_index: int,
    t_elapsed: float,
    t_max: float,
    chunk_wall_seconds: float,
    elapsed_seconds: float,
) -> tuple[str, RuntimeProgressSnapshot]:
    """Return the standard adaptive-runtime progress line and policy snapshot."""

    progress = (
        min(max(float(t_elapsed) / float(t_max), 0.0), 1.0)
        if float(t_max) > 0.0
        else 1.0
    )
    eta = (
        float(elapsed_seconds) * (1.0 / progress - 1.0)
        if progress > 1.0e-12
        else float("inf")
    )
    eta_text = format_duration(eta) if np.isfinite(eta) else "--:--"
    snapshot = RuntimeProgressSnapshot(
        progress=float(progress),
        eta_seconds=float(eta),
        chunk_wall_seconds=max(float(chunk_wall_seconds), 0.0),
        elapsed_seconds=max(float(elapsed_seconds), 0.0),
    )
    message = (
        f"completed {label} chunk {int(chunk_index)}: "
        f"t={float(t_elapsed):.6g}/{float(t_max):.6g} "
        f"progress={100.0 * snapshot.progress:5.1f}% "
        f"chunk_wall={format_duration(snapshot.chunk_wall_seconds)} "
        f"elapsed={format_duration(snapshot.elapsed_seconds)} "
        f"eta={eta_text}"
    )
    return message, snapshot


def run_runtime_scan_batch(
    cfg: RuntimeConfig,
    ky_arr: np.ndarray,
    *,
    Nl: int,
    Nm: int,
    method: str | None,
    dt: float | None,
    steps: int | None,
    sample_stride: int | None,
    auto_window: bool,
    tmin: float | None,
    tmax: float | None,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
    mode_method: str,
    fit_signal: str,
    show_progress: bool,
    deps: RuntimeScanBatchDeps,
) -> RuntimeLinearScanResult:
    """Batch a ky scan using one time integration over the full grid."""

    geom = deps.build_runtime_geometry(cfg)
    grid_cfg = deps.apply_geometry_grid_defaults(geom, cfg.grid)
    grid = deps.build_spectral_grid(grid_cfg)
    params = deps.build_runtime_linear_params(cfg, Nm=Nm, geom=geom)
    terms = deps.build_runtime_linear_terms(cfg)

    ky_indices = np.asarray(
        [deps.select_ky_index(np.asarray(grid.ky), ky) for ky in ky_arr], dtype=int
    )
    nspecies = max(len([s for s in cfg.species if s.kinetic]), 1)

    g0 = None
    for ky_idx in ky_indices:
        g0_local = deps.build_initial_condition(
            grid,
            geom,
            cfg,
            ky_index=int(ky_idx),
            kx_index=0,
            Nl=Nl,
            Nm=Nm,
            nspecies=nspecies,
        )
        g0 = g0_local if g0 is None else g0 + g0_local
    if g0 is None:
        raise ValueError("No ky values provided for batch scan")

    tcfg = cfg.time
    if method is not None:
        tcfg = replace(tcfg, method=str(method))
    if dt is not None:
        tcfg = replace(tcfg, dt=float(dt))
    if steps is not None:
        tcfg = replace(tcfg, t_max=float(steps) * float(tcfg.dt))
    if sample_stride is not None:
        tcfg = replace(tcfg, sample_stride=int(sample_stride))

    steps_val = int(round(tcfg.t_max / tcfg.dt))
    diag = deps.integrate_linear_diagnostics(
        g0,
        grid,
        geom,
        params,
        dt=tcfg.dt,
        steps=steps_val,
        method=tcfg.method,
        terms=terms,
        sample_stride=tcfg.sample_stride,
        species_index=0,
        record_hl_energy=False,
        show_progress=show_progress,
    )
    phi_t_np = np.asarray(diag[1])
    dens_t_np = np.asarray(diag[2])
    t_arr = (
        float(tcfg.dt)
        * float(tcfg.sample_stride)
        * (np.arange(phi_t_np.shape[0], dtype=float) + 1.0)
    )

    gamma = np.zeros_like(ky_arr, dtype=float)
    omega = np.zeros_like(ky_arr, dtype=float)
    fit_key = fit_signal.strip().lower()
    if fit_key not in {"phi", "density", "auto"}:
        raise ValueError("fit_signal must be 'phi', 'density', or 'auto'")

    for i, ky_idx in enumerate(ky_indices):
        sel = ModeSelection(
            ky_index=int(ky_idx), kx_index=0, z_index=deps.midplane_index(grid)
        )
        if fit_key == "auto":
            phi_signal = deps.extract_mode_time_series(
                phi_t_np, sel, method=mode_method
            )
            gamma_phi, omega_phi, _, _, r2_phi, r2p_phi = (
                deps.fit_growth_rate_auto_with_stats(
                    t_arr,
                    phi_signal,
                    window_fraction=window_fraction,
                    min_points=min_points,
                    start_fraction=start_fraction,
                    growth_weight=growth_weight,
                    require_positive=require_positive,
                    min_amp_fraction=min_amp_fraction,
                )
            )
            dens_signal = deps.extract_mode_time_series(
                dens_t_np, sel, method=mode_method
            )
            gamma_den, omega_den, _, _, r2_den, r2p_den = (
                deps.fit_growth_rate_auto_with_stats(
                    t_arr,
                    dens_signal,
                    window_fraction=window_fraction,
                    min_points=min_points,
                    start_fraction=start_fraction,
                    growth_weight=growth_weight,
                    require_positive=require_positive,
                    min_amp_fraction=min_amp_fraction,
                )
            )
            score_phi = r2_phi + 0.2 * r2p_phi + growth_weight * gamma_phi
            score_den = r2_den + 0.2 * r2p_den + growth_weight * gamma_den
            g_val, o_val = (
                (gamma_phi, omega_phi)
                if score_phi >= score_den
                else (gamma_den, omega_den)
            )
        else:
            signal = deps.extract_mode_time_series(
                dens_t_np if fit_key == "density" else phi_t_np,
                sel,
                method=mode_method,
            )
            if auto_window:
                g_val, o_val, _tmin, _tmax = deps.fit_growth_rate_auto(
                    t_arr,
                    signal,
                    window_fraction=window_fraction,
                    min_points=min_points,
                    start_fraction=start_fraction,
                    growth_weight=growth_weight,
                    require_positive=require_positive,
                    min_amp_fraction=min_amp_fraction,
                )
            else:
                g_val, o_val = deps.fit_growth_rate(t_arr, signal, tmin=tmin, tmax=tmax)

        g_val, o_val = deps.apply_diagnostic_normalization(
            g_val,
            o_val,
            rho_star=float(np.asarray(params.rho_star)),
            diagnostic_norm=cfg.normalization.diagnostic_norm,
        )
        gamma[i] = float(g_val)
        omega[i] = float(o_val)

    return RuntimeLinearScanResult(ky=ky_arr, gamma=gamma, omega=omega)


def resolve_nonlinear_artifact_policy(
    cfg: Any,
    *,
    out: str | Path | None,
    diagnostics: bool | None,
    steps: int | None,
    dt: float | None,
    deps: RuntimeArtifactHandoffDeps,
) -> NonlinearArtifactPolicy:
    """Resolve nonlinear output, restart, and checkpoint policy."""

    out_path = None if out is None else Path(out)
    netcdf_output_target = out_path is not None and deps.is_netcdf_output_target(
        out_path
    )
    diagnostics_on = bool(cfg.time.diagnostics if diagnostics is None else diagnostics)
    restart_from = None
    restart_to = None
    if netcdf_output_target:
        assert out_path is not None
        restart_from = deps.resolve_restart_path(out_path, cfg)
        restart_to = deps.resolve_restart_write_path(out_path, cfg)
    resume_requested = (
        bool(getattr(cfg.output, "restart", False)) or cfg.init.init_file is not None
    )
    if steps is not None:
        remaining_steps: int | None = int(steps)
    elif bool(cfg.time.fixed_dt):
        remaining_steps = int(
            round(float(cfg.time.t_max) / float(cfg.time.dt if dt is None else dt))
        )
    else:
        remaining_steps = None

    checkpoint_steps: int | None = None
    if (
        netcdf_output_target
        and remaining_steps is not None
        and bool(getattr(cfg.output, "save_for_restart", True))
    ):
        if (
            getattr(cfg.time, "nstep_restart", None) is not None
            and int(cfg.time.nstep_restart) > 0
        ):
            checkpoint_steps = int(cfg.time.nstep_restart)
        elif int(getattr(cfg.output, "nsave", 0)) > 0:
            checkpoint_steps = int(cfg.output.nsave)

    return NonlinearArtifactPolicy(
        out_path=out_path,
        netcdf_output_target=netcdf_output_target,
        diagnostics_on=diagnostics_on,
        restart_from=restart_from,
        restart_to=restart_to,
        resume_requested=resume_requested,
        remaining_steps=remaining_steps,
        checkpoint_steps=checkpoint_steps,
    )


def _restart_init_mode(cfg: Any) -> str:
    return (
        "add" if bool(getattr(cfg.output, "restart_with_perturb", False)) else "replace"
    )


def _apply_restart_input(cfg_run: Any, cfg: Any, restart_from: Path) -> Any:
    return replace(
        cfg_run,
        init=replace(
            cfg_run.init,
            init_file=str(restart_from),
            init_file_scale=float(getattr(cfg.output, "restart_scale", 1.0)),
            init_file_mode=_restart_init_mode(cfg),
        ),
    )


def run_runtime_nonlinear_artifact_handoff(
    cfg: Any,
    *,
    out: str | Path | None,
    ky_target: float,
    kx_target: float | None = None,
    Nl: int | None = None,
    Nm: int | None = None,
    dt: float | None = None,
    steps: int | None = None,
    method: str | None = None,
    sample_stride: int | None = None,
    diagnostics_stride: int | None = None,
    laguerre_mode: str | None = None,
    diagnostics: bool | None = None,
    show_progress: bool = False,
    status_callback: Any = None,
    deps: RuntimeArtifactHandoffDeps,
) -> tuple[RuntimeNonlinearResult, dict[str, str]]:
    """Run nonlinear runtime chunks and hand results to artifact writers."""

    policy = resolve_nonlinear_artifact_policy(
        cfg,
        out=out,
        diagnostics=diagnostics,
        steps=steps,
        dt=dt,
        deps=deps,
    )
    if policy.netcdf_output_target and not policy.diagnostics_on:
        raise ValueError("NetCDF nonlinear output artifacts require diagnostics output")

    cfg_run = cfg
    resume_requested = policy.resume_requested
    if policy.netcdf_output_target and cfg.init.init_file is None:
        if (
            bool(getattr(cfg.output, "restart_if_exists", False))
            and policy.restart_from is not None
            and policy.restart_from.exists()
        ):
            resume_requested = True
            cfg_run = _apply_restart_input(cfg_run, cfg, policy.restart_from)
        elif (
            bool(getattr(cfg.output, "restart", False))
            and policy.restart_from is not None
        ):
            if not policy.restart_from.exists():
                raise FileNotFoundError(
                    f"restart file not found: {policy.restart_from}"
                )
            cfg_run = _apply_restart_input(cfg_run, cfg, policy.restart_from)
    elif cfg.init.init_file is not None and bool(
        getattr(cfg.output, "restart_with_perturb", False)
    ):
        cfg_run = replace(
            cfg_run,
            init=replace(
                cfg_run.init,
                init_file_scale=float(getattr(cfg.output, "restart_scale", 1.0)),
                init_file_mode="add",
            ),
        )

    cumulative_diag: SimulationDiagnostics | None = None
    history_from_file = False
    if (
        policy.netcdf_output_target
        and resume_requested
        and bool(getattr(cfg.output, "append_on_restart", True))
    ):
        assert policy.out_path is not None
        history_path = Path(f"{deps.netcdf_bundle_base(policy.out_path)}.out.nc")
        if history_path.exists():
            cumulative_diag = deps.load_nonlinear_netcdf_diagnostics(history_path)
            history_from_file = True

    remaining_steps = policy.remaining_steps
    checkpoint_steps = policy.checkpoint_steps
    time_offset = 0.0
    if cumulative_diag is not None and np.asarray(cumulative_diag.t).size:
        time_offset = float(np.asarray(cumulative_diag.t)[-1])

    result_final: RuntimeNonlinearResult | None = None
    paths: dict[str, str] = {}
    while True:
        chunk_steps = remaining_steps
        if checkpoint_steps is not None:
            chunk_steps = (
                checkpoint_steps
                if remaining_steps is None
                else min(int(remaining_steps), checkpoint_steps)
            )
        result_chunk = deps.run_runtime_nonlinear(
            cfg_run,
            ky_target=ky_target,
            kx_target=kx_target,
            Nl=Nl,
            Nm=Nm,
            dt=dt,
            steps=chunk_steps,
            method=method,
            sample_stride=sample_stride,
            diagnostics_stride=diagnostics_stride,
            laguerre_mode=laguerre_mode,
            diagnostics=diagnostics,
            return_state=policy.netcdf_output_target,
            show_progress=show_progress,
            status_callback=status_callback,
        )
        deps.validate_finite_runtime_result(result_chunk)
        result_effective = result_chunk
        if result_chunk.diagnostics is not None:
            diag_chunk = result_chunk.diagnostics
            if history_from_file:
                diag_chunk = deps.condense_diagnostics_for_netcdf_output(diag_chunk)
            if time_offset != 0.0:
                diag_chunk = replace(
                    diag_chunk, t=np.asarray(diag_chunk.t) + time_offset
                )
            cumulative_diag = (
                diag_chunk
                if cumulative_diag is None
                else deps.concat_runtime_diagnostics([cumulative_diag, diag_chunk])
            )
            time_offset = (
                float(np.asarray(cumulative_diag.t)[-1])
                if np.asarray(cumulative_diag.t).size
                else time_offset
            )
            result_effective = replace(
                result_chunk,
                diagnostics=cumulative_diag,
                t=np.asarray(cumulative_diag.t),
            )
        result_final = result_effective

        if policy.out_path is not None:
            paths = deps.write_runtime_nonlinear_artifacts(
                policy.out_path, result_effective, cfg
            )

        if checkpoint_steps is None:
            break
        if remaining_steps is not None:
            assert chunk_steps is not None
            remaining_steps -= int(chunk_steps)
            if remaining_steps <= 0:
                break
        elif (
            result_effective.diagnostics is None
            or time_offset >= float(cfg.time.t_max) - 1.0e-12
        ):
            break
        if policy.restart_to is None:
            break
        cfg_run = replace(
            cfg,
            init=replace(
                cfg.init,
                init_file=str(policy.restart_to),
                init_file_scale=1.0,
                init_file_mode="replace",
            ),
        )

    if result_final is None:
        raise RuntimeError("nonlinear runtime produced no result")
    return result_final, paths
