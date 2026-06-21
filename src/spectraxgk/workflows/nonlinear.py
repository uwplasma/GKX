"""Executable nonlinear runtime workflow."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np

from spectraxgk.workflows.runtime.config import RuntimeConfig
from spectraxgk.workflows.runtime.results import RuntimeNonlinearResult


@dataclass(frozen=True)
class FullNonlinearRuntimeDeps:
    """Injected dependencies for the full-GK nonlinear runtime workflow."""

    build_runtime_geometry: Callable[[RuntimeConfig], Any]
    apply_geometry_grid_defaults: Callable[..., Any]
    build_spectral_grid: Callable[..., Any]
    build_runtime_linear_params: Callable[..., Any]
    build_runtime_term_config: Callable[..., Any]
    select_nonlinear_mode_indices: Callable[..., tuple[int, int]]
    build_initial_condition: Callable[..., Any]
    species_to_linear: Callable[..., Any]
    infer_runtime_nonlinear_steps: Callable[..., int]
    runtime_external_phi: Callable[..., Any]
    build_runtime_nonlinear_diagnostics_kwargs: Callable[..., dict[str, Any]]
    integrate_nonlinear_explicit_diagnostics_state: Callable[..., Any]
    run_adaptive_runtime_chunk_loop: Callable[..., Any]
    build_runtime_nonlinear_result: Callable[..., RuntimeNonlinearResult]
    integrate_nonlinear_from_config: Callable[..., tuple[Any, Any]]


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
    deps: FullNonlinearRuntimeDeps,
    ky_target: float,
    kx_target: float | None,
    Nl: int,
    Nm: int,
    dt: float | None,
    steps: int | None,
    status: Callable[[str], None],
) -> _RunContext:
    geom = deps.build_runtime_geometry(cfg)
    status("building spectral grid")
    grid = deps.build_spectral_grid(deps.apply_geometry_grid_defaults(geom, cfg.grid))
    status("building runtime nonlinear parameters")
    params = deps.build_runtime_linear_params(cfg, Nm=Nm, geom=geom)
    terms = deps.build_runtime_term_config(cfg)
    ky_index, kx_index = deps.select_nonlinear_mode_indices(
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
    G0 = deps.build_initial_condition(
        grid,
        geom,
        cfg,
        ky_index=ky_index,
        kx_index=kx_index,
        Nl=Nl,
        Nm=Nm,
        nspecies=len(deps.species_to_linear(cfg.species)),
    )
    dt_val = float(cfg.time.dt if dt is None else dt)
    if dt_val <= 0.0:
        raise ValueError("dt must be > 0")
    return _RunContext(
        geom=geom,
        grid=grid,
        params=params,
        terms=terms,
        G0=G0,
        ky_index=int(ky_index),
        kx_index=int(kx_index),
        dt=dt_val,
        steps=int(deps.infer_runtime_nonlinear_steps(cfg, dt=dt_val, steps=steps)),
        adaptive_chunked=steps is None and not bool(cfg.time.fixed_dt),
    )


def _diagnostic_policy(
    cfg: RuntimeConfig,
    *,
    deps: FullNonlinearRuntimeDeps,
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
        diagnostics_on=cfg.time.diagnostics if diagnostics is None else bool(diagnostics),
        sample_stride=cfg.time.sample_stride if sample_stride is None else int(sample_stride),
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
        external_phi=deps.runtime_external_phi(cfg),
        resolved_diagnostics=resolved_diagnostics,
        return_state=return_state,
        show_progress=show_progress,
    )


def _diagnostic_kwargs(
    cfg: RuntimeConfig,
    ctx: _RunContext,
    policy: _DiagnosticPolicy,
    *,
    deps: FullNonlinearRuntimeDeps,
    steps: int,
    method: str | None,
    sample_stride: int,
    diagnostics_stride: int,
    fixed_dt: bool,
    show_progress: bool,
) -> dict[str, Any]:
    return deps.build_runtime_nonlinear_diagnostics_kwargs(
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


def _run_adaptive_diagnostics(
    cfg: RuntimeConfig,
    ctx: _RunContext,
    policy: _DiagnosticPolicy,
    *,
    deps: FullNonlinearRuntimeDeps,
    method: str | None,
    status: Callable[[str], None],
) -> tuple[Any, Any, Any, Any]:
    chunk_steps = min(ctx.steps, 1024)
    G_chunk = ctx.G0

    def run_chunk(chunk_show_progress: bool):
        nonlocal G_chunk
        kwargs = _diagnostic_kwargs(
            cfg,
            ctx,
            policy,
            deps=deps,
            steps=chunk_steps,
            method=method,
            sample_stride=1,
            diagnostics_stride=1,
            fixed_dt=False,
            show_progress=chunk_show_progress,
        )
        t_chunk, diag_chunk, G_next, fields_next = (
            deps.integrate_nonlinear_explicit_diagnostics_state(
                G_chunk,
                ctx.grid,
                ctx.geom,
                ctx.params,
                **kwargs,
            )
        )
        G_chunk = G_next
        return t_chunk, diag_chunk, G_next, fields_next

    chunk_result = deps.run_adaptive_runtime_chunk_loop(
        integrate_chunk=run_chunk,
        t_max=float(cfg.time.t_max),
        chunk_steps=chunk_steps,
        label="nonlinear",
        show_progress=policy.show_progress,
        status_callback=status,
        diagnostics_stride=max(policy.sample_stride, policy.diagnostics_stride, 1),
    )
    diag = chunk_result.diagnostics
    return jnp.asarray(diag.t), diag, chunk_result.state, chunk_result.fields


def _run_diagnostics(
    cfg: RuntimeConfig,
    ctx: _RunContext,
    policy: _DiagnosticPolicy,
    *,
    deps: FullNonlinearRuntimeDeps,
    method: str | None,
    status: Callable[[str], None],
) -> tuple[Any, Any, Any, Any]:
    status(
        f"sample_stride={policy.sample_stride} "
        f"diagnostics_stride={policy.diagnostics_stride} laguerre_mode={policy.laguerre_mode}"
    )
    if ctx.adaptive_chunked:
        return _run_adaptive_diagnostics(
            cfg, ctx, policy, deps=deps, method=method, status=status
        )
    status(
        f"running nonlinear diagnostics integrator over {ctx.steps} steps with dt={ctx.dt:.6g}"
    )
    kwargs = _diagnostic_kwargs(
        cfg,
        ctx,
        policy,
        deps=deps,
        steps=ctx.steps,
        method=method,
        sample_stride=policy.sample_stride,
        diagnostics_stride=policy.diagnostics_stride,
        fixed_dt=bool(cfg.time.fixed_dt),
        show_progress=policy.show_progress,
    )
    return deps.integrate_nonlinear_explicit_diagnostics_state(
        ctx.G0,
        ctx.grid,
        ctx.geom,
        ctx.params,
        **kwargs,
    )


def _result(
    ctx: _RunContext,
    policy: _DiagnosticPolicy,
    *,
    deps: FullNonlinearRuntimeDeps,
    t: Any,
    diagnostics: Any,
    fields: Any,
    state: Any,
    summarize_fields: bool,
) -> RuntimeNonlinearResult:
    return deps.build_runtime_nonlinear_result(
        t=np.asarray(t),
        diagnostics=diagnostics,
        fields=fields,
        state=np.asarray(state) if policy.return_state else None,
        ky_selected=float(np.asarray(ctx.grid.ky[ctx.ky_index])),
        kx_selected=float(np.asarray(ctx.grid.kx[ctx.kx_index])),
        summarize_fields=summarize_fields,
    )


def _run_final_state(
    cfg: RuntimeConfig,
    ctx: _RunContext,
    policy: _DiagnosticPolicy,
    *,
    deps: FullNonlinearRuntimeDeps,
    status: Callable[[str], None],
) -> RuntimeNonlinearResult:
    status(
        "diagnostics disabled; running final-state nonlinear integrator over "
        f"{ctx.steps} steps with dt={ctx.dt:.6g}"
    )
    time_cfg = replace(cfg.time, dt=ctx.dt, t_max=ctx.dt * ctx.steps)
    kwargs = {"terms": ctx.terms}
    if policy.show_progress:
        kwargs["show_progress"] = True
    G_final, fields = deps.integrate_nonlinear_from_config(
        ctx.G0,
        ctx.grid,
        ctx.geom,
        ctx.params,
        time_cfg,
        **kwargs,
    )
    status("completed nonlinear final-state integration")
    return _result(
        ctx,
        policy,
        deps=deps,
        t=np.asarray([]),
        diagnostics=None,
        fields=fields,
        state=G_final,
        summarize_fields=True,
    )


def _diagnostic_run_result(
    ctx: _RunContext,
    policy: _DiagnosticPolicy,
    *,
    deps: FullNonlinearRuntimeDeps,
    t: Any,
    diagnostics: Any,
    fields: Any,
    state: Any,
    status: Callable[[str], None],
) -> RuntimeNonlinearResult:
    if policy.diagnostics_on:
        status(f"completed nonlinear run with {int(np.asarray(t).size)} saved samples")
        return _result(
            ctx,
            policy,
            deps=deps,
            t=t,
            diagnostics=diagnostics,
            fields=fields,
            state=state,
            summarize_fields=False,
        )
    if fields is None:
        raise RuntimeError("adaptive nonlinear runtime did not produce final fields")
    status("diagnostics disabled; returning final nonlinear field summary")
    return _result(
        ctx,
        policy,
        deps=deps,
        t=np.asarray([]),
        diagnostics=None,
        fields=fields,
        state=state,
        summarize_fields=True,
    )


def run_full_nonlinear_runtime(
    cfg: RuntimeConfig,
    *,
    deps: FullNonlinearRuntimeDeps,
    ky_target: float,
    kx_target: float | None,
    Nl: int,
    Nm: int,
    dt: float | None,
    steps: int | None,
    method: str | None,
    sample_stride: int | None,
    diagnostics_stride: int | None,
    laguerre_mode: str | None,
    diagnostics: bool | None,
    resolved_diagnostics: bool,
    return_state: bool,
    show_progress: bool,
    status_callback: Callable[[str], None] | None = None,
) -> RuntimeNonlinearResult:
    """Run one full-GK nonlinear point from a runtime config."""

    status = _status_callback(status_callback)
    ctx = _prepare_context(
        cfg,
        deps=deps,
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
        deps=deps,
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
    if not policy.requires_diagnostic_path and not ctx.adaptive_chunked:
        return _run_final_state(cfg, ctx, policy, deps=deps, status=status)

    t, diag, G_final, fields_final = _run_diagnostics(
        cfg, ctx, policy, deps=deps, method=method, status=status
    )
    return _diagnostic_run_result(
        ctx,
        policy,
        deps=deps,
        t=t,
        diagnostics=diag,
        fields=fields_final,
        state=G_final,
        status=status,
    )


__all__ = ["FullNonlinearRuntimeDeps", "run_full_nonlinear_runtime"]
