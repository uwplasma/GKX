"""Runtime artifact display, restart, and checkpoint handoff policies."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from gkx.diagnostics import SimulationDiagnostics
from gkx.workflows.runtime.results import RuntimeNonlinearResult


COMMAND_LINEAR_ARTIFACT_DISPLAY_KEYS = (
    "summary",
    "timeseries",
    "eigenfunction",
    "state",
    "quasilinear_summary",
    "quasilinear_species",
)
COMMAND_SCAN_ARTIFACT_DISPLAY_KEYS = ("summary", "scan", "quasilinear_spectrum")
COMMAND_QUASILINEAR_ARTIFACT_DISPLAY_KEYS = (
    "quasilinear_summary",
    "quasilinear_species",
)
COMMAND_NONLINEAR_ARTIFACT_DISPLAY_KEYS = (
    "summary",
    "diagnostics",
    "state",
    "out",
    "big",
    "restart",
)


def print_saved_paths(paths: Mapping[str, str], keys: Sequence[str]) -> None:
    """Print saved artifact paths in command-defined display order."""

    for key in keys:
        if key in paths:
            print(f"saved {paths[key]}")


def write_command_outputs(
    out_path: str | Path | None,
    payload: Any | None,
    *,
    writer: Callable[[str | Path, Any], dict[str, str]],
    display_keys: Sequence[str],
) -> dict[str, str]:
    """Write command artifacts when both destination and payload exist."""

    if out_path is None or payload is None:
        return {}
    paths = writer(out_path, payload)
    print_saved_paths(paths, display_keys)
    return paths


def write_linear_runtime_command_outputs(
    *,
    linear_out_path: str | Path | None,
    quasilinear_out_path: str | Path | None,
    result: Any,
    linear_writer: Callable[[str | Path, Any], dict[str, str]],
    quasilinear_writer: Callable[[str | Path, Any], dict[str, str]],
) -> dict[str, dict[str, str]]:
    """Write all optional artifacts produced by one linear runtime command."""

    linear_paths = write_command_outputs(
        linear_out_path,
        result,
        writer=linear_writer,
        display_keys=COMMAND_LINEAR_ARTIFACT_DISPLAY_KEYS,
    )
    ql_paths = write_command_outputs(
        quasilinear_out_path,
        getattr(result, "quasilinear", None),
        writer=quasilinear_writer,
        display_keys=COMMAND_QUASILINEAR_ARTIFACT_DISPLAY_KEYS,
    )
    return {"linear": linear_paths, "quasilinear": ql_paths}


def write_scan_runtime_command_outputs(
    out_path: str | Path | None,
    scan: Any,
    *,
    writer: Callable[[str | Path, Any], dict[str, str]],
) -> dict[str, str]:
    """Write optional artifacts produced by one linear-scan runtime command."""

    return write_command_outputs(
        out_path,
        scan,
        writer=writer,
        display_keys=COMMAND_SCAN_ARTIFACT_DISPLAY_KEYS,
    )


def print_linear_run_header(
    *,
    label: str,
    config_path: str,
    ky: float,
    Nl: int,
    Nm: int,
    solver: str,
    method: str,
    dt: float,
    steps: int,
    grid_shape: tuple[int, int, int],
    show_progress: bool,
    extra: str | None = None,
) -> None:
    """Print the standard executable header for linear initial-value runs."""

    print(f"starting {label}")
    print(
        f"config={config_path} ky={ky:.4f} Nl={Nl} Nm={Nm} "
        f"solver={solver} method={method} dt={dt:.6g} steps={steps}"
    )
    print(
        f"grid=Nx{grid_shape[0]} Ny{grid_shape[1]} Nz{grid_shape[2]} "
        f"progress={'on' if show_progress else 'off'}"
    )
    if extra is not None:
        print(extra)


def print_nonlinear_run_header(
    *,
    config_path: str,
    ky: float,
    Nl: int,
    Nm: int,
    method: str,
    dt: float,
    steps: int | None,
    grid_shape: tuple[int, int, int],
    diagnostics: bool,
    show_progress: bool,
) -> None:
    """Print the standard executable header for nonlinear initial-value runs."""

    print("starting runtime nonlinear run")
    print(
        f"config={config_path} ky={ky:.4f} Nl={Nl} Nm={Nm} "
        f"method={method} dt={dt:.6g} "
        f"steps={'auto' if steps is None else steps}"
    )
    print(
        f"grid=Nx{grid_shape[0]} Ny{grid_shape[1]} Nz{grid_shape[2]} "
        f"diagnostics={'on' if diagnostics else 'off'} "
        f"progress={'on' if show_progress else 'off'}"
    )


def print_nonlinear_run_summary(result: Any) -> bool:
    """Print final nonlinear diagnostics and return whether diagnostics exist."""

    diag = result.diagnostics
    if diag is None:
        print("nonlinear run completed")
        return False
    t_values = np.asarray(diag.t)
    t_last = float(t_values[-1]) if t_values.size else 0.0
    print(
        "nonlinear: "
        f"t={t_last:.6g} "
        f"ky_sel={result.ky_selected:.6g} "
        f"kx_sel={result.kx_selected:.6g} "
        f"dt_mean={float(diag.dt_mean):.6g} "
        f"Wg={float(diag.Wg_t[-1]):.6g} "
        f"Wphi={float(diag.Wphi_t[-1]):.6g} "
        f"Wapar={float(diag.Wapar_t[-1]):.6g}"
    )
    return True


def print_nonlinear_command_outputs(paths: Mapping[str, str], *, enabled: bool) -> None:
    """Print nonlinear artifact paths after diagnostics confirm a saved run."""

    if enabled:
        print_saved_paths(paths, COMMAND_NONLINEAR_ARTIFACT_DISPLAY_KEYS)


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


@dataclass(frozen=True)
class _NonlinearArtifactRunOptions:
    """Nonlinear runtime options forwarded unchanged to each checkpoint chunk."""

    ky_target: float
    kx_target: float | None
    Nl: int | None
    Nm: int | None
    dt: float | None
    method: str | None
    sample_stride: int | None
    diagnostics_stride: int | None
    laguerre_mode: str | None
    diagnostics: bool | None
    show_progress: bool
    status_callback: Any


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


def _resolve_restart_run_config(
    cfg: Any,
    policy: NonlinearArtifactPolicy,
) -> tuple[Any, bool]:
    """Return the run config after restart/resume policy has been applied."""

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
    return cfg_run, resume_requested


def _load_restart_history(
    cfg: Any,
    policy: NonlinearArtifactPolicy,
    *,
    resume_requested: bool,
    deps: RuntimeArtifactHandoffDeps,
) -> tuple[SimulationDiagnostics | None, bool]:
    """Load existing nonlinear diagnostic history when append-on-restart applies."""

    if not (
        policy.netcdf_output_target
        and resume_requested
        and bool(getattr(cfg.output, "append_on_restart", True))
    ):
        return None, False
    assert policy.out_path is not None
    history_path = Path(f"{deps.netcdf_bundle_base(policy.out_path)}.out.nc")
    if not history_path.exists():
        return None, False
    return deps.load_nonlinear_netcdf_diagnostics(history_path), True


def _next_runtime_chunk_steps(
    *,
    remaining_steps: int | None,
    checkpoint_steps: int | None,
) -> int | None:
    """Return the next nonlinear chunk length from restart/checkpoint policy."""

    if checkpoint_steps is None:
        return remaining_steps
    if remaining_steps is None:
        return checkpoint_steps
    return min(int(remaining_steps), checkpoint_steps)


def _advance_restart_run_config(cfg: Any, restart_to: Path | None) -> Any:
    """Return the config for the next checkpoint chunk."""

    if restart_to is None:
        return cfg
    return replace(
        cfg,
        init=replace(
            cfg.init,
            init_file=str(restart_to),
            init_file_scale=1.0,
            init_file_mode="replace",
        ),
    )


def _merge_chunk_diagnostics(
    result_chunk: RuntimeNonlinearResult,
    *,
    cumulative_diag: SimulationDiagnostics | None,
    time_offset: float,
    history_from_file: bool,
    deps: RuntimeArtifactHandoffDeps,
) -> tuple[RuntimeNonlinearResult, SimulationDiagnostics | None, float]:
    """Merge one chunk's diagnostics with loaded/runtime history."""

    if result_chunk.diagnostics is None:
        return result_chunk, cumulative_diag, time_offset

    diag_chunk = result_chunk.diagnostics
    if history_from_file:
        diag_chunk = deps.condense_diagnostics_for_netcdf_output(diag_chunk)
    if time_offset != 0.0:
        diag_chunk = replace(diag_chunk, t=np.asarray(diag_chunk.t) + time_offset)
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
    return (
        replace(
            result_chunk,
            diagnostics=cumulative_diag,
            t=np.asarray(cumulative_diag.t),
        ),
        cumulative_diag,
        time_offset,
    )


def _checkpoint_loop_done(
    *,
    policy: NonlinearArtifactPolicy,
    result_effective: RuntimeNonlinearResult,
    remaining_steps: int | None,
    time_offset: float,
    cfg: Any,
) -> bool:
    """Return whether checkpointed nonlinear handoff has completed."""

    if policy.checkpoint_steps is None:
        return True
    if remaining_steps is not None and remaining_steps <= 0:
        return True
    return bool(
        remaining_steps is None
        and (
            result_effective.diagnostics is None
            or time_offset >= float(cfg.time.t_max) - 1.0e-12
        )
    )


def _run_runtime_nonlinear_chunk(
    cfg_run: Any,
    *,
    policy: NonlinearArtifactPolicy,
    options: _NonlinearArtifactRunOptions,
    chunk_steps: int | None,
    deps: RuntimeArtifactHandoffDeps,
) -> RuntimeNonlinearResult:
    """Run and validate one nonlinear artifact/checkpoint chunk."""

    result_chunk = deps.run_runtime_nonlinear(
        cfg_run,
        ky_target=options.ky_target,
        kx_target=options.kx_target,
        Nl=options.Nl,
        Nm=options.Nm,
        dt=options.dt,
        steps=chunk_steps,
        method=options.method,
        sample_stride=options.sample_stride,
        diagnostics_stride=options.diagnostics_stride,
        laguerre_mode=options.laguerre_mode,
        diagnostics=options.diagnostics,
        resolved_diagnostics=bool(cfg_run.output.resolved_diagnostics),
        return_state=policy.netcdf_output_target,
        show_progress=options.show_progress,
        status_callback=options.status_callback,
    )
    deps.validate_finite_runtime_result(result_chunk)
    return result_chunk


def _write_nonlinear_artifacts_if_requested(
    policy: NonlinearArtifactPolicy,
    result_effective: RuntimeNonlinearResult,
    cfg: Any,
    deps: RuntimeArtifactHandoffDeps,
) -> dict[str, str]:
    """Write nonlinear artifacts when an output target was requested."""

    if policy.out_path is None:
        return {}
    return deps.write_runtime_nonlinear_artifacts(
        policy.out_path, result_effective, cfg
    )


def _advance_checkpoint_or_stop(
    cfg: Any,
    *,
    policy: NonlinearArtifactPolicy,
    result_effective: RuntimeNonlinearResult,
    remaining_steps: int | None,
    time_offset: float,
) -> tuple[bool, Any]:
    """Return whether checkpointing is complete and the next run config."""

    if policy.checkpoint_steps is None:
        return True, cfg
    if _checkpoint_loop_done(
        policy=policy,
        result_effective=result_effective,
        remaining_steps=remaining_steps,
        time_offset=time_offset,
        cfg=cfg,
    ):
        return True, cfg
    if policy.restart_to is None:
        return True, cfg
    return False, _advance_restart_run_config(cfg, policy.restart_to)


def _run_artifact_checkpoint_loop(
    cfg: Any,
    cfg_run: Any,
    *,
    policy: NonlinearArtifactPolicy,
    options: _NonlinearArtifactRunOptions,
    cumulative_diag: SimulationDiagnostics | None,
    history_from_file: bool,
    deps: RuntimeArtifactHandoffDeps,
) -> tuple[RuntimeNonlinearResult, dict[str, str]]:
    """Run all nonlinear chunks needed by the resolved artifact policy."""

    remaining_steps = policy.remaining_steps
    time_offset = 0.0
    if cumulative_diag is not None and np.asarray(cumulative_diag.t).size:
        time_offset = float(np.asarray(cumulative_diag.t)[-1])

    result_final: RuntimeNonlinearResult | None = None
    paths: dict[str, str] = {}
    while True:
        chunk_steps = _next_runtime_chunk_steps(
            remaining_steps=remaining_steps,
            checkpoint_steps=policy.checkpoint_steps,
        )
        result_chunk = _run_runtime_nonlinear_chunk(
            cfg_run,
            policy=policy,
            options=options,
            chunk_steps=chunk_steps,
            deps=deps,
        )
        result_effective, cumulative_diag, time_offset = _merge_chunk_diagnostics(
            result_chunk,
            cumulative_diag=cumulative_diag,
            time_offset=time_offset,
            history_from_file=history_from_file,
            deps=deps,
        )
        result_final = result_effective
        paths = _write_nonlinear_artifacts_if_requested(
            policy, result_effective, cfg, deps
        )
        if remaining_steps is not None:
            assert chunk_steps is not None
            remaining_steps -= int(chunk_steps)
        stop, cfg_run = _advance_checkpoint_or_stop(
            cfg,
            policy=policy,
            result_effective=result_effective,
            remaining_steps=remaining_steps,
            time_offset=time_offset,
        )
        if stop:
            break

    if result_final is None:
        raise RuntimeError("nonlinear runtime produced no result")
    return result_final, paths


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

    options = _NonlinearArtifactRunOptions(
        ky_target=ky_target,
        kx_target=kx_target,
        Nl=Nl,
        Nm=Nm,
        dt=dt,
        method=method,
        sample_stride=sample_stride,
        diagnostics_stride=diagnostics_stride,
        laguerre_mode=laguerre_mode,
        diagnostics=diagnostics,
        show_progress=show_progress,
        status_callback=status_callback,
    )
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

    cfg_run, resume_requested = _resolve_restart_run_config(cfg, policy)
    cumulative_diag, history_from_file = _load_restart_history(
        cfg,
        policy,
        resume_requested=resume_requested,
        deps=deps,
    )
    return _run_artifact_checkpoint_loop(
        cfg,
        cfg_run,
        policy=policy,
        options=options,
        cumulative_diag=cumulative_diag,
        history_from_file=history_from_file,
        deps=deps,
    )


__all__ = [
    "COMMAND_LINEAR_ARTIFACT_DISPLAY_KEYS",
    "COMMAND_NONLINEAR_ARTIFACT_DISPLAY_KEYS",
    "COMMAND_QUASILINEAR_ARTIFACT_DISPLAY_KEYS",
    "COMMAND_SCAN_ARTIFACT_DISPLAY_KEYS",
    "NonlinearArtifactPolicy",
    "RuntimeArtifactHandoffDeps",
    "print_nonlinear_command_outputs",
    "print_saved_paths",
    "resolve_nonlinear_artifact_policy",
    "run_runtime_nonlinear_artifact_handoff",
    "write_command_outputs",
    "write_linear_runtime_command_outputs",
    "write_scan_runtime_command_outputs",
]
