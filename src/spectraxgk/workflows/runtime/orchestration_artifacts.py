"""Runtime artifact display, restart, and checkpoint handoff policies."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from spectraxgk.diagnostics import SimulationDiagnostics
from spectraxgk.workflows.runtime.results import RuntimeNonlinearResult
from spectraxgk.workflows.runtime.command_artifacts import (
    COMMAND_LINEAR_ARTIFACT_DISPLAY_KEYS,
    COMMAND_NONLINEAR_ARTIFACT_DISPLAY_KEYS,
    COMMAND_QUASILINEAR_ARTIFACT_DISPLAY_KEYS,
    COMMAND_SCAN_ARTIFACT_DISPLAY_KEYS,
    print_nonlinear_command_outputs,
    print_saved_paths,
    write_command_outputs,
    write_linear_runtime_command_outputs,
    write_scan_runtime_command_outputs,
)


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

    cfg_run, resume_requested = _resolve_restart_run_config(cfg, policy)
    cumulative_diag, history_from_file = _load_restart_history(
        cfg,
        policy,
        resume_requested=resume_requested,
        deps=deps,
    )

    remaining_steps = policy.remaining_steps
    checkpoint_steps = policy.checkpoint_steps
    time_offset = 0.0
    if cumulative_diag is not None and np.asarray(cumulative_diag.t).size:
        time_offset = float(np.asarray(cumulative_diag.t)[-1])

    result_final: RuntimeNonlinearResult | None = None
    paths: dict[str, str] = {}
    while True:
        chunk_steps = _next_runtime_chunk_steps(
            remaining_steps=remaining_steps,
            checkpoint_steps=checkpoint_steps,
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
        result_effective, cumulative_diag, time_offset = _merge_chunk_diagnostics(
            result_chunk,
            cumulative_diag=cumulative_diag,
            time_offset=time_offset,
            history_from_file=history_from_file,
            deps=deps,
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
        if _checkpoint_loop_done(
            policy=policy,
            result_effective=result_effective,
            remaining_steps=remaining_steps,
            time_offset=time_offset,
            cfg=cfg,
        ):
            break
        if policy.restart_to is None:
            break
        cfg_run = _advance_restart_run_config(cfg, policy.restart_to)

    if result_final is None:
        raise RuntimeError("nonlinear runtime produced no result")
    return result_final, paths


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
