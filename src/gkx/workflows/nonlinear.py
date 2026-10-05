"""Executable nonlinear runtime workflow."""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from gkx.core_grid import SpectralGrid, build_spectral_grid
from gkx.diagnostics.modes import select_ky_index
from gkx.diagnostics.saturation import (
    SaturationStopConfig,
    saturation_stop_decision,
)
from gkx.geometry import apply_geometry_grid_defaults
from gkx.solvers_time_explicit_cfl import FIXED_DT_CFL_WARN_RATIO
from gkx.config import RuntimeConfig, resolve_cfl_fac
from gkx.solvers_nonlinear_diagnostic_integration import (
    integrate_nonlinear_explicit_diagnostics_state,
    prepare_nonlinear_explicit_diagnostics,
)
from gkx.solvers_time_runners import integrate_nonlinear_from_config
from gkx.workflows.runtime.chunks import run_adaptive_runtime_chunk_loop
from gkx.workflows.runtime.diagnostic_arrays import (
    validate_finite_runtime_diagnostics,
)
from gkx.workflows.runtime.parallel_nonlinear import (
    NonlinearParallelPlan,
    assert_nonlinear_parallel_identity,
    resolve_nonlinear_parallel_plan,
    resolve_species_hermite_mesh,
    shard_nonlinear_state,
)
from gkx.workflows.runtime.results import (
    RuntimeNonlinearResult,
    build_runtime_nonlinear_result,
    checked_solve_summary,
    solve_stats_request,
)
from gkx.workflows.runtime.startup import (
    _build_initial_condition,
    _resolve_runtime_hl_dims,
    _species_to_linear,
    build_runtime_geometry,
    build_runtime_linear_params,
    build_runtime_term_config,
)


def build_runtime_nonlinear_diagnostics_kwargs(
    cfg: RuntimeConfig,
    *,
    dt: float,
    steps: int,
    method: str | None,
    term_config: Any,
    sample_stride: int,
    diagnostics_stride: int,
    laguerre_mode: str,
    ky_index: int,
    kx_index: int,
    fixed_dt: bool,
    fixed_mode_ky_index: int | None,
    fixed_mode_kx_index: int | None,
    external_phi: float | None,
    resolved_diagnostics: bool,
    show_progress: bool,
) -> dict[str, Any]:
    """Build keyword arguments for nonlinear diagnostic integration.

    Runtime drivers use the same physics kwargs for fixed-window and adaptive
    chunked diagnostics. Keeping them in one policy helper prevents drift
    between the two branches while leaving the actual integrator call in the
    public runtime facade.
    """

    method_use = str(method or cfg.time.method)
    kwargs: dict[str, Any] = dict(
        dt=float(dt),
        steps=int(steps),
        method=method_use,
        terms=term_config,
        sample_stride=int(sample_stride),
        diagnostics_stride=int(diagnostics_stride),
        use_dealias_mask=bool(cfg.time.nonlinear_dealias),
        laguerre_mode=str(laguerre_mode),
        omega_ky_index=int(ky_index),
        omega_kx_index=int(kx_index),
        flux_scale=float(cfg.normalization.flux_scale),
        wphi_scale=float(cfg.normalization.wphi_scale),
        fixed_dt=bool(fixed_dt),
        dt_min=float(cfg.time.dt_min),
        dt_max=cfg.time.dt_max,
        cfl=float(cfg.time.cfl),
        cfl_fac=resolve_cfl_fac(method_use, cfg.time.cfl_fac),
        collision_split=bool(cfg.time.collision_split),
        collision_scheme=str(cfg.time.collision_scheme),
        implicit_restart=int(cfg.time.implicit_restart),
        implicit_preconditioner=cfg.time.implicit_preconditioner,
        fixed_mode_ky_index=fixed_mode_ky_index,
        fixed_mode_kx_index=fixed_mode_kx_index,
        external_phi=external_phi,
    )
    if not resolved_diagnostics:
        kwargs["resolved_diagnostics"] = False
    if show_progress:
        kwargs["show_progress"] = True
    return kwargs


def _nearest_index_from_candidates(
    values: np.ndarray,
    target: float,
    candidates: np.ndarray,
) -> int:
    """Return the candidate index nearest to ``target`` in physical coordinates."""

    values_arr = np.asarray(values, dtype=float)
    candidate_arr = np.asarray(candidates, dtype=int)
    if values_arr.size == 0:
        raise ValueError("values must be non-empty")
    if candidate_arr.size == 0:
        raise ValueError("candidate indices must be non-empty")
    return int(
        candidate_arr[int(np.argmin(np.abs(values_arr[candidate_arr] - float(target))))]
    )


def _validate_dealias_mask_shape(
    mask: Any,
    *,
    ky_size: int,
    kx_size: int,
) -> np.ndarray:
    """Return a boolean dealias mask after validating it matches ky/kx axes."""

    mask_arr = np.asarray(mask, dtype=bool)
    expected = (int(ky_size), int(kx_size))
    if mask_arr.shape != expected:
        raise ValueError(
            "dealias_mask shape must match (ky, kx) grid sizes; "
            f"got {mask_arr.shape}, expected {expected}"
        )
    return mask_arr


def _active_ky_indices(mask: np.ndarray, ky_size: int) -> np.ndarray:
    """Return ky rows with at least one retained kx, falling back to all ky."""

    candidates = np.where(np.any(mask, axis=1))[0]
    if candidates.size == 0:
        return np.arange(int(ky_size), dtype=int)
    return candidates


def _active_kx_indices(mask: np.ndarray, ky_index: int, kx_size: int) -> np.ndarray:
    """Return retained kx entries for ``ky_index``, falling back to all kx."""

    candidates = np.where(mask[int(ky_index)])[0]
    if candidates.size == 0:
        return np.arange(int(kx_size), dtype=int)
    return candidates


def _select_nonlinear_mode_indices(
    grid: SpectralGrid,
    *,
    ky_target: float,
    kx_target: float | None,
    use_dealias_mask: bool,
) -> tuple[int, int]:
    ky = np.asarray(grid.ky, dtype=float)
    kx = np.asarray(grid.kx, dtype=float)
    kx_pick_target = 0.0 if kx_target is None else float(kx_target)
    if not use_dealias_mask:
        ky_pick = select_ky_index(ky, ky_target)
        kx_pick = _nearest_index_from_candidates(
            kx, kx_pick_target, np.arange(kx.size, dtype=int)
        )
        return ky_pick, kx_pick

    mask = _validate_dealias_mask_shape(
        grid.dealias_mask,
        ky_size=ky.size,
        kx_size=kx.size,
    )
    ky_pick = _nearest_index_from_candidates(
        ky, ky_target, _active_ky_indices(mask, ky.size)
    )
    kx_pick = _nearest_index_from_candidates(
        kx, kx_pick_target, _active_kx_indices(mask, ky_pick, kx.size)
    )
    return int(ky_pick), int(kx_pick)


def _infer_runtime_nonlinear_steps(
    cfg: RuntimeConfig,
    *,
    dt: float,
    steps: int | None,
) -> int:
    """Infer nonlinear explicit step counts with the same dt ceiling as the integrator."""

    if steps is not None:
        steps_val = int(steps)
    elif bool(cfg.time.fixed_dt):
        steps_val = int(
            np.round(float(cfg.time.t_max) / max(float(cfg.time.dt), 1.0e-12))
        )
    else:
        # Keep runtime inference aligned with adaptive stepping: when
        # dt_max is unset, the nonlinear integrator clamps at dt itself.
        dt_cap = float(cfg.time.dt_max) if cfg.time.dt_max is not None else float(dt)
        steps_val = int(np.ceil(float(cfg.time.t_max) / max(dt_cap, 1.0e-12)))
    if steps_val < 1:
        raise ValueError("steps must be >= 1")
    return steps_val


def _runtime_external_phi(cfg: RuntimeConfig) -> float | None:
    """Return a runtime external-phi source if requested."""

    source = str(cfg.expert.source).strip().lower()
    if source in {"", "default"}:
        return None
    if source != "phiext_full":
        raise ValueError(
            f"unsupported expert.source={cfg.expert.source!r}; expected 'default' or 'phiext_full'"
        )
    return float(cfg.expert.phi_ext)


@dataclass(frozen=True)
class _RunContext:
    geom: Any
    grid: Any
    params: Any
    terms: Any
    G0: Any
    ky_index: int
    kx_index: int
    dt: float
    steps: int
    adaptive_chunked: bool


@dataclass(frozen=True)
class _DiagnosticPolicy:
    diagnostics_on: bool
    sample_stride: int
    diagnostics_stride: int
    laguerre_mode: str
    fixed_mode_on: bool
    fixed_ky_index: int | None
    fixed_kx_index: int | None
    external_phi: Any
    resolved_diagnostics: bool
    return_state: bool
    show_progress: bool

    @property
    def source_on(self) -> bool:
        return self.external_phi is not None

    @property
    def requires_diagnostic_path(self) -> bool:
        return (
            self.diagnostics_on
            or self.fixed_mode_on
            or self.return_state
            or self.source_on
        )


def _status_callback(callback: Callable[[str], None] | None) -> Callable[[str], None]:
    def status(message: str) -> None:
        if callback is not None:
            callback(message)

    return status


def _prepare_context(
    cfg: RuntimeConfig,
    *,
    ky_target: float,
    kx_target: float | None,
    Nl: int,
    Nm: int,
    dt: float | None,
    steps: int | None,
    status: Callable[[str], None],
) -> _RunContext:
    geom = build_runtime_geometry(cfg)
    status("building spectral grid")
    grid = build_spectral_grid(apply_geometry_grid_defaults(geom, cfg.grid))
    status("building runtime nonlinear parameters")
    params = build_runtime_linear_params(cfg, Nm=Nm, geom=geom)
    terms = build_runtime_term_config(cfg)
    ky_index, kx_index = _select_nonlinear_mode_indices(
        grid,
        ky_target=ky_target,
        kx_target=kx_target,
        use_dealias_mask=bool(cfg.time.nonlinear_dealias),
    )
    status(
        f"selected nonlinear mode ky={float(np.asarray(grid.ky[ky_index])):.6g} "
        f"kx={float(np.asarray(grid.kx[kx_index])):.6g}"
    )
    status("building initial condition")
    G0 = _build_initial_condition(
        grid,
        geom,
        cfg,
        ky_index=ky_index,
        kx_index=kx_index,
        Nl=Nl,
        Nm=Nm,
        nspecies=len(_species_to_linear(cfg.species)),
    )
    dt_val = float(cfg.time.dt if dt is None else dt)
    if dt_val <= 0.0:
        raise ValueError("dt must be > 0")
    ctx = _RunContext(
        geom=geom,
        grid=grid,
        params=params,
        terms=terms,
        G0=G0,
        ky_index=int(ky_index),
        kx_index=int(kx_index),
        dt=dt_val,
        steps=int(_infer_runtime_nonlinear_steps(cfg, dt=dt_val, steps=steps)),
        adaptive_chunked=steps is None and not bool(cfg.time.fixed_dt),
    )
    # Before the first step, not after the last: an over-CFL run that is going
    # to overflow should say so while the step is still cheap to change.
    _report_nonlinear_cfl_margin(cfg, ctx, Nl=Nl, Nm=Nm, status=status)
    return ctx


def _diagnostic_policy(
    cfg: RuntimeConfig,
    *,
    diagnostics: bool | None,
    sample_stride: int | None,
    diagnostics_stride: int | None,
    laguerre_mode: str | None,
    resolved_diagnostics: bool,
    return_state: bool,
    show_progress: bool,
) -> _DiagnosticPolicy:
    fixed_mode_on = bool(cfg.expert.fixed_mode)
    fixed_ky: int | None = None
    fixed_kx: int | None = None
    if fixed_mode_on:
        if cfg.expert.iky_fixed is None or cfg.expert.ikx_fixed is None:
            raise ValueError(
                "expert.iky_fixed and expert.ikx_fixed must be set when expert.fixed_mode=true"
            )
        fixed_ky = int(cfg.expert.iky_fixed)
        fixed_kx = int(cfg.expert.ikx_fixed)
    return _DiagnosticPolicy(
        diagnostics_on=cfg.time.diagnostics
        if diagnostics is None
        else bool(diagnostics),
        sample_stride=cfg.time.sample_stride
        if sample_stride is None
        else int(sample_stride),
        diagnostics_stride=(
            cfg.time.diagnostics_stride
            if diagnostics_stride is None
            else int(diagnostics_stride)
        ),
        laguerre_mode=(
            cfg.time.laguerre_nonlinear_mode
            if laguerre_mode is None
            else str(laguerre_mode)
        ),
        fixed_mode_on=fixed_mode_on,
        fixed_ky_index=fixed_ky,
        fixed_kx_index=fixed_kx,
        external_phi=_runtime_external_phi(cfg),
        resolved_diagnostics=resolved_diagnostics,
        return_state=return_state,
        show_progress=show_progress,
    )


def _diagnostic_kwargs(
    cfg: RuntimeConfig,
    ctx: _RunContext,
    policy: _DiagnosticPolicy,
    *,
    steps: int,
    method: str | None,
    sample_stride: int,
    diagnostics_stride: int,
    fixed_dt: bool,
    show_progress: bool,
) -> dict[str, Any]:
    return build_runtime_nonlinear_diagnostics_kwargs(
        cfg,
        dt=ctx.dt,
        steps=steps,
        method=method,
        term_config=ctx.terms,
        sample_stride=int(sample_stride),
        diagnostics_stride=int(diagnostics_stride),
        laguerre_mode=policy.laguerre_mode,
        ky_index=ctx.ky_index,
        kx_index=ctx.kx_index,
        fixed_dt=fixed_dt,
        fixed_mode_ky_index=policy.fixed_ky_index,
        fixed_mode_kx_index=policy.fixed_kx_index,
        external_phi=policy.external_phi,
        resolved_diagnostics=policy.resolved_diagnostics,
        show_progress=show_progress,
    )


_RUN_TO_VALUES = ("saturation", "t_max")


def _saturation_stop_condition(
    cfg: RuntimeConfig, ctx: _RunContext, policy: _DiagnosticPolicy
) -> Callable[[Any, Any, Any, Any], dict[str, Any]] | None:
    """Build the run-to-saturation stop check, or ``None`` to run to t_max.

    Saturation stopping needs the streamed heat-flux trace, so a run with
    diagnostics disabled falls back to the t_max horizon. So does a run whose
    whole step budget is shorter than the minimum sample count: it could never
    reach a stop decision, and routing it through the chunk loop would only
    add machinery around the same integration.
    """

    run_to = str(cfg.time.run_to).strip().lower()
    if run_to not in _RUN_TO_VALUES:
        raise ValueError(f"unknown [time] run_to '{cfg.time.run_to}'")
    if run_to != "saturation" or not policy.diagnostics_on:
        return None
    stop_cfg = SaturationStopConfig(
        rel_sem=float(cfg.time.saturation_rel_sem),
        min_window=(
            None
            if cfg.time.saturation_min_window is None
            else float(cfg.time.saturation_min_window)
        ),
    )
    # The chunked route records one diagnostic sample per step.
    if not ctx.adaptive_chunked and int(ctx.steps) < max(int(stop_cfg.min_samples), 8):
        return None

    def check(t: Any, heat_flux: Any, wphi: Any, wg: Any) -> dict[str, Any]:
        return saturation_stop_decision(
            t, heat_flux, guard=wphi, free_energy_guard=wg, config=stop_cfg
        )

    return check


def _nonlinear_cfl_margin(
    cfg: RuntimeConfig, ctx: _RunContext, *, Nl: int, Nm: int
) -> tuple[float, float, np.ndarray] | None:
    """Return the nonlinear CFL bound and ratio, or ``None`` if unavailable.

    Match the adaptive step's no-diamagnetic-drive convention. This advisory
    calculation must not abort an otherwise valid run when geometry is traced
    or has not yet been sampled onto the grid.
    """

    from gkx.geometry import ensure_flux_tube_geometry_data
    from gkx.solvers_time_explicit_cfl import _linear_frequency_bound

    try:
        geom = ensure_flux_tube_geometry_data(ctx.geom, ctx.grid.z)
        omega = np.asarray(
            _linear_frequency_bound(
                ctx.grid, geom, ctx.params, Nl, Nm, include_diamagnetic_drive=False
            ),
            dtype=float,
        )
    except Exception:
        return None
    wmax = float(np.sum(omega))
    if not np.isfinite(wmax) or wmax <= 0.0:
        return None
    cfl_fac = resolve_cfl_fac(str(cfg.time.method), cfg.time.cfl_fac)
    bound = cfl_fac * float(cfg.time.cfl) / wmax
    return bound, float(ctx.dt) / bound, omega


def _report_nonlinear_cfl_margin(
    cfg: RuntimeConfig,
    ctx: _RunContext,
    *,
    Nl: int,
    Nm: int,
    status: Callable[[str], None],
) -> None:
    """Report the CFL margin because fixed steps bypass adaptive enforcement.

    Keep unsafe deck choices visible in logs and artifact metadata.
    """

    margin = _nonlinear_cfl_margin(cfg, ctx, Nl=Nl, Nm=Nm)
    if margin is None:
        return
    bound, ratio, omega = margin
    total = float(np.sum(omega))
    status(
        f"CFL margin: dt={ctx.dt:.4g} bound={bound:.4g} ratio={ratio:.2f}x "
        f"(streaming {100.0 * float(omega[2]) / total:.0f}% of the bound)"
    )
    if not bool(cfg.time.fixed_dt) or ratio <= FIXED_DT_CFL_WARN_RATIO:
        return
    warnings.warn(
        f"fixed dt={ctx.dt:.4g} is {ratio:.2f}x the explicit CFL bound "
        f"{bound:.4g} for this case (max linear frequency {total:.4g}, of which "
        f"parallel streaming is {float(omega[2]):.4g}). fixed_dt = true means "
        "nothing reduces the step, so the trajectory is expected to overflow "
        f"and the run to fail on non-finite diagnostics. Reduce dt below "
        f"{bound:.3g}, or lower Nm: the streaming bound scales as sqrt(Nm)",
        RuntimeWarning,
        stacklevel=2,
    )


def _run_chunked_diagnostics(
    cfg: RuntimeConfig,
    ctx: _RunContext,
    policy: _DiagnosticPolicy,
    *,
    method: str | None,
    status: Callable[[str], None],
    stop_condition: Callable[[Any, Any, Any, Any], dict[str, Any]] | None,
) -> tuple[Any, Any, Any, Any, dict[str, Any] | None]:
    """Run chunked nonlinear diagnostics for the adaptive and saturation routes.

    The adaptive route keeps its historical chunking; the fixed-step route only
    goes through here for run-to-saturation, chunking on step count so the stop
    check runs a few times per horizon while the total step budget stays the
    hard cap.
    """

    step_capped = not ctx.adaptive_chunked
    chunk_steps = min(ctx.steps, 128)
    G_chunk = ctx.G0
    steps_left = ctx.steps
    # One compiled scan per chunk shape, reused by every chunk of this run.
    compile_cache: dict[Any, Any] = {}

    def run_chunk(chunk_show_progress: bool, remaining_time: float):
        nonlocal G_chunk, steps_left
        steps_now = min(chunk_steps, steps_left) if step_capped else chunk_steps
        kwargs = _diagnostic_kwargs(
            cfg,
            ctx,
            policy,
            steps=steps_now,
            method=method,
            sample_stride=1,
            diagnostics_stride=1,
            fixed_dt=bool(cfg.time.fixed_dt),
            show_progress=chunk_show_progress,
        )
        dt_cap = float(cfg.time.dt_max or ctx.dt)
        if remaining_time <= steps_now * dt_cap:
            kwargs["time_horizon"] = remaining_time
        kwargs["compile_cache"] = compile_cache
        t_chunk, diag_chunk, G_next, fields_next = (
            integrate_nonlinear_explicit_diagnostics_state(
                G_chunk,
                ctx.grid,
                ctx.geom,
                ctx.params,
                **kwargs,
            )
        )
        G_chunk = G_next
        steps_left -= steps_now
        return t_chunk, diag_chunk, G_next, fields_next

    def loop_stop(t: Any, heat_flux: Any, wphi: Any, wg: Any) -> dict[str, Any]:
        assert stop_condition is not None
        decision = stop_condition(t, heat_flux, wphi, wg)
        # "stop" ends the loop; "saturated" is the physics verdict. The step
        # budget running out stops the loop without claiming saturation.
        stop = bool(decision.get("saturated")) or (step_capped and steps_left <= 0)
        return {**decision, "stop": stop}

    chunk_result = run_adaptive_runtime_chunk_loop(
        integrate_chunk=run_chunk,
        t_max=ctx.dt * ctx.steps if step_capped else float(cfg.time.t_max),
        chunk_steps=chunk_steps,
        label="nonlinear",
        show_progress=policy.show_progress,
        status_callback=status,
        diagnostics_stride=max(policy.sample_stride, policy.diagnostics_stride, 1),
        # Opt-in disk spill for runs whose strided diagnostics still outgrow host
        # RAM. Off by default because it trades wall time for peak memory; set
        # GKX_CHUNK_SPILL_DIR to a path to turn it on.
        spill_dir=_chunk_spill_dir(),
        stop_condition=None if stop_condition is None else loop_stop,
    )
    diag = chunk_result.diagnostics
    return (
        jnp.asarray(diag.t),
        diag,
        chunk_result.state,
        chunk_result.fields,
        chunk_result.stop_decision,
    )


def _chunk_spill_dir() -> Path | None:
    """Directory for spilled chunk diagnostics, or ``None`` to keep them in RAM.

    Environment rather than TOML because it is a property of the *machine* the
    run lands on, not of the physics case: the same config is submitted to a
    workstation with 512 GB and to a batch node with 16, and only the second one
    needs to spill.
    """

    raw = os.environ.get("GKX_CHUNK_SPILL_DIR", "").strip()
    return Path(raw) if raw else None


def _run_diagnostics(
    cfg: RuntimeConfig,
    ctx: _RunContext,
    policy: _DiagnosticPolicy,
    *,
    method: str | None,
    status: Callable[[str], None],
) -> tuple[Any, Any, Any, Any, dict[str, Any] | None]:
    status(
        f"sample_stride={policy.sample_stride} "
        f"diagnostics_stride={policy.diagnostics_stride} laguerre_mode={policy.laguerre_mode}"
    )
    stop_condition = _saturation_stop_condition(cfg, ctx, policy)
    if ctx.adaptive_chunked or stop_condition is not None:
        if stop_condition is not None:
            status(
                "run_to=saturation: heat-flux convergence with stationary Wphi/Wg "
                f"(rel_sem<={float(cfg.time.saturation_rel_sem):.6g})"
            )
        return _run_chunked_diagnostics(
            cfg,
            ctx,
            policy,
            method=method,
            status=status,
            stop_condition=stop_condition,
        )
    status(
        f"running nonlinear diagnostics integrator over {ctx.steps} steps with dt={ctx.dt:.6g}"
    )
    kwargs = _diagnostic_kwargs(
        cfg,
        ctx,
        policy,
        steps=ctx.steps,
        method=method,
        sample_stride=policy.sample_stride,
        diagnostics_stride=policy.diagnostics_stride,
        fixed_dt=bool(cfg.time.fixed_dt),
        show_progress=policy.show_progress,
    )
    t, diag, G_final, fields_final = integrate_nonlinear_explicit_diagnostics_state(
        ctx.G0,
        ctx.grid,
        ctx.geom,
        ctx.params,
        **kwargs,
    )
    # The chunked routes above validate every chunk before returning it. This
    # one is a single scan, so nothing has looked at the trace yet: an unstable
    # step runs to the last step and hands back a diagnostics array whose tail
    # is NaN, and every caller that is not the artifact writer -- the library
    # API, and the CLI whenever [output] path is unset -- reports that as a
    # successful run. Refuse the trace instead of returning it.
    validate_finite_runtime_diagnostics(diag, label="fixed-step nonlinear run")
    return t, diag, G_final, fields_final, None


def _result(
    ctx: _RunContext,
    policy: _DiagnosticPolicy,
    *,
    t: Any,
    diagnostics: Any,
    fields: Any,
    state: Any,
    summarize_fields: bool,
    saturation: dict[str, Any] | None = None,
) -> RuntimeNonlinearResult:
    return build_runtime_nonlinear_result(
        t=np.asarray(t),
        diagnostics=diagnostics,
        fields=fields,
        state=np.asarray(state) if policy.return_state else None,
        ky_selected=float(np.asarray(ctx.grid.ky[ctx.ky_index])),
        kx_selected=float(np.asarray(ctx.grid.kx[ctx.kx_index])),
        summarize_fields=summarize_fields,
        saturation=saturation,
    )


def _run_final_state(
    cfg: RuntimeConfig,
    ctx: _RunContext,
    policy: _DiagnosticPolicy,
    *,
    status: Callable[[str], None],
) -> tuple[RuntimeNonlinearResult, Any]:
    status(
        "diagnostics disabled; running final-state nonlinear integrator over "
        f"{ctx.steps} steps with dt={ctx.dt:.6g}"
    )
    time_cfg = replace(cfg.time, dt=ctx.dt, t_max=ctx.dt * ctx.steps)
    request = solve_stats_request(time_cfg.method, kind="nonlinear")
    kwargs = {"terms": ctx.terms, **request}
    if policy.show_progress:
        kwargs["show_progress"] = True
    G_final, fields, *extra = integrate_nonlinear_from_config(
        ctx.G0, ctx.grid, ctx.geom, ctx.params, time_cfg, **kwargs
    )
    solve = checked_solve_summary(extra, request, label="nonlinear IMEX run")
    status("completed nonlinear final-state integration")
    result = _result(
        ctx,
        policy,
        t=np.asarray([]),
        diagnostics=None,
        fields=fields,
        state=G_final,
        summarize_fields=True,
    )
    return (result if solve is None else replace(result, implicit_solve=solve)), G_final


def _diagnostic_run_result(
    ctx: _RunContext,
    policy: _DiagnosticPolicy,
    *,
    t: Any,
    diagnostics: Any,
    fields: Any,
    state: Any,
    saturation: dict[str, Any] | None,
    status: Callable[[str], None],
) -> RuntimeNonlinearResult:
    if policy.diagnostics_on:
        status(f"completed nonlinear run with {int(np.asarray(t).size)} saved samples")
        return _result(
            ctx,
            policy,
            t=t,
            diagnostics=diagnostics,
            fields=fields,
            state=state,
            summarize_fields=False,
            saturation=saturation,
        )
    if fields is None:
        raise RuntimeError("adaptive nonlinear runtime did not produce final fields")
    status("diagnostics disabled; returning final nonlinear field summary")
    return _result(
        ctx,
        policy,
        t=np.asarray([]),
        diagnostics=None,
        fields=fields,
        state=state,
        summarize_fields=True,
    )


def _run_once(
    cfg: RuntimeConfig,
    ctx: _RunContext,
    policy: _DiagnosticPolicy,
    *,
    method: str | None,
    status: Callable[[str], None],
) -> tuple[RuntimeNonlinearResult, Any]:
    """Run one nonlinear trajectory and return its result plus its final state."""

    if not policy.requires_diagnostic_path and not ctx.adaptive_chunked:
        return _run_final_state(cfg, ctx, policy, status=status)

    t, diag, G_final, fields_final, saturation = _run_diagnostics(
        cfg, ctx, policy, method=method, status=status
    )
    return (
        _diagnostic_run_result(
            ctx,
            policy,
            t=t,
            diagnostics=diag,
            fields=fields_final,
            state=G_final,
            saturation=saturation,
            status=status,
        ),
        G_final,
    )


def _run_sharded(
    cfg: RuntimeConfig,
    ctx: _RunContext,
    policy: _DiagnosticPolicy,
    *,
    method: str | None,
    status: Callable[[str], None],
    plan: NonlinearParallelPlan,
) -> RuntimeNonlinearResult:
    """Run the sharded nonlinear route, fail-closed against the serial answer."""

    if plan.axis == "species_hermite":
        plan = resolve_species_hermite_mesh(ctx.G0, plan)
    status(f"routing nonlinear run through {plan.describe()}")
    sharded_ctx = replace(ctx, G0=shard_nonlinear_state(ctx.G0, plan))
    result, sharded_state = _run_once(
        cfg, sharded_ctx, policy, method=method, status=status
    )
    if not plan.strict_identity:
        status("parallel strict_identity=false; skipping the serial identity gate")
        return result
    status("verifying sharded nonlinear identity against the serial route")
    serial_result, serial_state = _run_once(
        cfg, ctx, policy, method=method, status=status
    )
    assert_nonlinear_parallel_identity(
        serial_state=serial_state,
        sharded_state=sharded_state,
        serial_diagnostics=serial_result.diagnostics,
        sharded_diagnostics=result.diagnostics,
        plan=plan,
    )
    status("sharded nonlinear identity gate passed")
    return result


def prepare(case: RuntimeConfig, **options: Any) -> Any:
    """Prepare a reusable compiled nonlinear simulation; see its class docstring."""
    if not case.physics.nonlinear:
        raise ValueError("prepare currently requires nonlinear physics")
    if options.pop("diagnostics", True) is not True:
        raise ValueError("prepare requires diagnostics=True")
    return run_runtime_nonlinear(case, diagnostics=True, prepare_only=True, **options)


def run_runtime_nonlinear(
    cfg: RuntimeConfig,
    *,
    ky_target: float = 0.3,
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
    resolved_diagnostics: bool = True,
    return_state: bool = False,
    show_progress: bool = False,
    status_callback: Callable[[str], None] | None = None,
    prepare_only: bool = False,
) -> Any:
    """Run a nonlinear point using the unified runtime config path.

    ``prepare_only`` returns the compiled prepared simulation instead (the
    :func:`prepare` path).
    """

    Nl, Nm = _resolve_runtime_hl_dims(cfg, Nl=Nl, Nm=Nm)
    if status_callback is not None:
        status_callback("building runtime geometry")

    # Resolved before any geometry or grid work so an unroutable [parallel]
    # request fails immediately instead of after a long serial run.
    plan = resolve_nonlinear_parallel_plan(cfg.parallel)
    if prepare_only and plan is not None:
        raise ValueError(
            "prepared nonlinear execution currently requires serial policy"
        )
    status = _status_callback(status_callback)
    ctx = _prepare_context(
        cfg,
        ky_target=ky_target,
        kx_target=kx_target,
        Nl=Nl,
        Nm=Nm,
        dt=dt,
        steps=steps,
        status=status,
    )
    policy = _diagnostic_policy(
        cfg,
        diagnostics=diagnostics,
        sample_stride=sample_stride,
        diagnostics_stride=diagnostics_stride,
        laguerre_mode=laguerre_mode,
        resolved_diagnostics=resolved_diagnostics,
        return_state=return_state,
        show_progress=show_progress,
    )
    status(
        f"nonlinear diagnostics={'on' if policy.diagnostics_on else 'off'} "
        f"fixed_mode={'on' if policy.fixed_mode_on else 'off'} source={cfg.expert.source}"
    )
    if prepare_only:
        if _saturation_stop_condition(cfg, ctx, policy) is not None:
            # A prepared object compiles one scan of a fixed length, and
            # saturation stopping decides the length while the run is going, so
            # the two cannot both hold. Refusing is right -- quietly dropping
            # the stop condition would hand back a simulation that integrates
            # past where the deck asked it to stop. But the refusal used to end
            # here, and every shipped nonlinear deck sets run_to = "saturation",
            # so gkx.prepare had no working example in the tree. Name the two
            # ways out.
            raise ValueError(
                "prepared execution cannot stop early at saturation: this deck "
                f"sets [time] run_to = {str(cfg.time.run_to).strip().lower()!r}, "
                "and a prepared simulation compiles one scan of a fixed length. "
                "Either pass an explicit length, gkx.prepare(case, steps=N), or "
                'set [time] run_to = "t_max" so the deck itself fixes it. '
                "Both give up saturation stopping for this object; run the case "
                "through gkx.solve or the CLI to keep it."
            )
        kwargs = _diagnostic_kwargs(
            cfg,
            ctx,
            policy,
            steps=ctx.steps,
            method=method,
            sample_stride=policy.sample_stride,
            diagnostics_stride=policy.diagnostics_stride,
            fixed_dt=bool(cfg.time.fixed_dt),
            show_progress=show_progress,
        )
        if not bool(cfg.time.fixed_dt):
            kwargs["time_horizon"] = float(cfg.time.t_max)
        return prepare_nonlinear_explicit_diagnostics(
            ctx.G0, ctx.grid, ctx.geom, ctx.params, **kwargs
        )
    if plan is None:
        return _run_once(cfg, ctx, policy, method=method, status=status)[0]
    return _run_sharded(cfg, ctx, policy, method=method, status=status, plan=plan)


__all__ = [
    "prepare",
    "run_runtime_nonlinear",
]
