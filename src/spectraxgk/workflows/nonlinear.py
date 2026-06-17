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

    def _status(message: str) -> None:
        if status_callback is not None:
            status_callback(message)

    geom = deps.build_runtime_geometry(cfg)
    _status("building spectral grid")
    grid_cfg = deps.apply_geometry_grid_defaults(geom, cfg.grid)
    grid = deps.build_spectral_grid(grid_cfg)
    _status("building runtime nonlinear parameters")
    params = deps.build_runtime_linear_params(cfg, Nm=Nm, geom=geom)
    term_cfg = deps.build_runtime_term_config(cfg)

    ky_index, kx_index = deps.select_nonlinear_mode_indices(
        grid,
        ky_target=ky_target,
        kx_target=kx_target,
        use_dealias_mask=bool(cfg.time.nonlinear_dealias),
    )
    _status(
        f"selected nonlinear mode ky={float(np.asarray(grid.ky[ky_index])):.6g} "
        f"kx={float(np.asarray(grid.kx[kx_index])):.6g}"
    )
    _status("building initial condition")
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
    adaptive_chunked = steps is None and not bool(cfg.time.fixed_dt)
    steps_val = deps.infer_runtime_nonlinear_steps(cfg, dt=dt_val, steps=steps)

    fixed_mode_on = bool(cfg.expert.fixed_mode)
    fixed_ky_index = cfg.expert.iky_fixed
    fixed_kx_index = cfg.expert.ikx_fixed
    external_phi = deps.runtime_external_phi(cfg)
    source_on = external_phi is not None
    fixed_ky_index_use: int | None = None
    fixed_kx_index_use: int | None = None
    if fixed_mode_on:
        if fixed_ky_index is None or fixed_kx_index is None:
            raise ValueError(
                "expert.iky_fixed and expert.ikx_fixed must be set when expert.fixed_mode=true"
            )
        fixed_ky_index_use = int(fixed_ky_index)
        fixed_kx_index_use = int(fixed_kx_index)

    diagnostics_on = cfg.time.diagnostics if diagnostics is None else bool(diagnostics)
    _status(
        f"nonlinear diagnostics={'on' if diagnostics_on else 'off'} "
        f"fixed_mode={'on' if fixed_mode_on else 'off'} source={cfg.expert.source}"
    )
    if diagnostics_on or fixed_mode_on or return_state or adaptive_chunked or source_on:
        sample_stride_use = (
            cfg.time.sample_stride if sample_stride is None else int(sample_stride)
        )
        diag_stride = (
            cfg.time.diagnostics_stride
            if diagnostics_stride is None
            else int(diagnostics_stride)
        )
        laguerre_mode_use = (
            cfg.time.laguerre_nonlinear_mode
            if laguerre_mode is None
            else str(laguerre_mode)
        )
        _status(
            f"sample_stride={int(sample_stride_use)} "
            f"diagnostics_stride={int(diag_stride)} laguerre_mode={laguerre_mode_use}"
        )
        if adaptive_chunked:
            chunk_steps = min(steps_val, 1024)
            G_chunk = G0

            def _run_nonlinear_chunk(chunk_show_progress: bool):
                nonlocal G_chunk
                kwargs = deps.build_runtime_nonlinear_diagnostics_kwargs(
                    cfg,
                    dt=dt_val,
                    steps=chunk_steps,
                    method=method,
                    term_config=term_cfg,
                    sample_stride=1,
                    diagnostics_stride=1,
                    laguerre_mode=laguerre_mode_use,
                    ky_index=int(ky_index),
                    kx_index=int(kx_index),
                    fixed_dt=False,
                    fixed_mode_ky_index=fixed_ky_index_use,
                    fixed_mode_kx_index=fixed_kx_index_use,
                    external_phi=external_phi,
                    resolved_diagnostics=resolved_diagnostics,
                    show_progress=chunk_show_progress,
                )
                t_chunk, diag_chunk, G_next, fields_next = (
                    deps.integrate_nonlinear_explicit_diagnostics_state(
                        G_chunk,
                        grid,
                        geom,
                        params,
                        **kwargs,
                    )
                )
                G_chunk = G_next
                return t_chunk, diag_chunk, G_next, fields_next

            chunk_result = deps.run_adaptive_runtime_chunk_loop(
                integrate_chunk=_run_nonlinear_chunk,
                t_max=float(cfg.time.t_max),
                chunk_steps=chunk_steps,
                label="nonlinear",
                show_progress=show_progress,
                status_callback=_status,
                diagnostics_stride=max(int(sample_stride_use), int(diag_stride), 1),
            )
            diag = chunk_result.diagnostics
            t = jnp.asarray(diag.t)
            G_final = chunk_result.state
            fields_final = chunk_result.fields
        else:
            _status(
                f"running nonlinear diagnostics integrator over {steps_val} steps with dt={dt_val:.6g}"
            )
            diagnostics_call_kwargs = deps.build_runtime_nonlinear_diagnostics_kwargs(
                cfg,
                dt=dt_val,
                steps=steps_val,
                method=method,
                term_config=term_cfg,
                sample_stride=int(sample_stride_use),
                diagnostics_stride=int(diag_stride),
                laguerre_mode=laguerre_mode_use,
                ky_index=int(ky_index),
                kx_index=int(kx_index),
                fixed_dt=bool(cfg.time.fixed_dt),
                fixed_mode_ky_index=fixed_ky_index_use,
                fixed_mode_kx_index=fixed_kx_index_use,
                external_phi=external_phi,
                resolved_diagnostics=resolved_diagnostics,
                show_progress=show_progress,
            )
            t, diag, G_final, fields_final = (
                deps.integrate_nonlinear_explicit_diagnostics_state(
                    G0,
                    grid,
                    geom,
                    params,
                    **diagnostics_call_kwargs,
                )
            )
        if diagnostics_on:
            _status(f"completed nonlinear run with {int(np.asarray(t).size)} saved samples")
            state_out = np.asarray(G_final) if return_state else None
            return deps.build_runtime_nonlinear_result(
                t=np.asarray(t),
                diagnostics=diag,
                fields=fields_final,
                state=state_out,
                ky_selected=float(np.asarray(grid.ky[ky_index])),
                kx_selected=float(np.asarray(grid.kx[kx_index])),
                summarize_fields=False,
            )
        if fields_final is None:
            raise RuntimeError("adaptive nonlinear runtime did not produce final fields")
        _status("diagnostics disabled; returning final nonlinear field summary")
        return deps.build_runtime_nonlinear_result(
            t=np.asarray([]),
            diagnostics=None,
            fields=fields_final,
            state=np.asarray(G_final) if return_state else None,
            ky_selected=float(np.asarray(grid.ky[ky_index])),
            kx_selected=float(np.asarray(grid.kx[kx_index])),
            summarize_fields=True,
        )

    _status(
        f"diagnostics disabled; running final-state nonlinear integrator over {steps_val} steps with dt={dt_val:.6g}"
    )
    t_cfg = replace(cfg.time, dt=dt_val, t_max=dt_val * steps_val)
    if show_progress:
        G_final, fields = deps.integrate_nonlinear_from_config(
            G0,
            grid,
            geom,
            params,
            t_cfg,
            terms=term_cfg,
            show_progress=True,
        )
    else:
        G_final, fields = deps.integrate_nonlinear_from_config(
            G0,
            grid,
            geom,
            params,
            t_cfg,
            terms=term_cfg,
        )
    _status("completed nonlinear final-state integration")
    return deps.build_runtime_nonlinear_result(
        t=np.asarray([]),
        diagnostics=None,
        fields=fields,
        state=np.asarray(G_final) if return_state else None,
        ky_selected=float(np.asarray(grid.ky[ky_index])),
        kx_selected=float(np.asarray(grid.kx[kx_index])),
        summarize_fields=True,
    )
