"""Executable linear runtime workflow."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np

from gkx.benchmarking_shared import LinearRunResult, LinearScanResult, _midplane_index
from gkx.diagnostics.modes import (
    ModeSelection,
)
from gkx.config import RuntimeConfig
from gkx.operators.linear.cache_builder import mask_off_chain_rows
from gkx.core_grid import SpectralGrid, build_spectral_grid, select_ky_grid
from gkx.diagnostics.modes import select_ky_index
from gkx.diagnostics.normalization import apply_diagnostic_normalization
from gkx.geometry import apply_geometry_grid_defaults
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.solvers_linear_integrators import integrate_linear_diagnostics
from gkx.solvers_linear_krylov import KrylovConfig, dominant_eigenpair
from gkx.solvers_time_runners import integrate_linear_from_config
from gkx.workflows.runtime.diagnostics import (
    _fit_signal_key,
    finalize_runtime_linear_quasilinear,
    fit_runtime_linear_diagnostics,
)
from gkx.workflows.runtime.startup import (
    _build_initial_condition,
    _resolve_runtime_hl_dims,
    _runtime_default_krylov_config,
    build_runtime_geometry,
    build_runtime_linear_params,
    build_runtime_linear_terms,
)
from gkx.workflows.runtime.results import (
    RuntimeLinearResult,
    checked_solve_summary,
    solve_stats_request,
)
from gkx.solvers_time_explicit import integrate_linear_explicit_from_config


_StatusCallback = Callable[[str], None] | None


def _normalize_linear_solver_name(solver: str) -> str:
    solver_key = solver.strip().lower().replace("-", "_")
    if solver_key == "explicit_time":
        return "explicit_time"
    return solver_key


def _zero_kx_index(grid: SpectralGrid) -> int:
    kx = np.asarray(grid.kx, dtype=float)
    return int(np.argmin(np.abs(kx)))


@dataclass(frozen=True)
class _LinearRuntimeContext:
    cfg: RuntimeConfig
    geom: Any
    grid: Any
    params: Any
    terms: Any
    selection: ModeSelection
    initial_state: Any
    solver_key: str
    fit_key: str
    ql_enabled: bool
    return_state_requested: bool
    return_state_effective: bool
    n_laguerre: int
    n_hermite: int


@dataclass(frozen=True)
class _LinearFitPolicy:
    mode_method: str
    auto_window: bool
    tmin: float | None
    tmax: float | None
    window_fraction: float
    min_points: int
    start_fraction: float
    growth_weight: float
    require_positive: bool
    min_amp_fraction: float
    window_method: str = "stationary"


@dataclass(frozen=True)
class _LinearTrajectory:
    """Saved state and diagnostics from one linear time integration."""

    g_last: Any | None
    phi_t: Any
    density_t: Any | None
    t: Any | None = None
    implicit_solve: Any | None = None

    def as_numpy(self, time_config: Any) -> _LinearTrajectory:
        """Materialize saved diagnostics and infer fixed-step sample times."""

        phi_t = np.asarray(self.phi_t)
        density_t = None if self.density_t is None else np.asarray(self.density_t)
        times = (
            _linear_saved_sample_times(phi_t, time_config) if self.t is None else self.t
        )
        return _LinearTrajectory(
            g_last=self.g_last,
            phi_t=phi_t,
            density_t=density_t,
            t=np.asarray(times, dtype=float),
            implicit_solve=self.implicit_solve,
        )


def _status(callback: _StatusCallback, message: str) -> None:
    if callback is not None:
        callback(message)


def _prepare_linear_runtime_context(
    cfg: RuntimeConfig,
    *,
    ky_target: float,
    n_laguerre: int,
    n_hermite: int,
    solver: str,
    fit_signal: str,
    return_state: bool,
    initial_state: Any | None,
    status_callback: _StatusCallback,
    kx_target: float | None = None,
) -> _LinearRuntimeContext:
    ql_enabled = bool(getattr(cfg.quasilinear, "enabled", False))
    return_state_requested = bool(return_state)
    return_state_effective = return_state_requested or ql_enabled

    geom = build_runtime_geometry(cfg)
    _status(status_callback, "building spectral grid")
    grid_cfg = apply_geometry_grid_defaults(geom, cfg.grid)
    grid_full = build_spectral_grid(grid_cfg)
    _status(status_callback, "building runtime linear parameters")
    params = build_runtime_linear_params(cfg, Nm=n_hermite, geom=geom)
    terms = build_runtime_linear_terms(cfg)

    ky_index = select_ky_index(np.asarray(grid_full.ky), ky_target)
    grid = select_ky_grid(grid_full, ky_index)
    kx = np.asarray(grid.kx, dtype=float)
    kx_index = 0 if kx_target is None else int(np.argmin(np.abs(kx - kx_target)))
    selection = ModeSelection(
        ky_index=0,
        kx_index=kx_index,
        z_index=_midplane_index(grid),
    )
    _status(
        status_callback,
        f"selected ky index {ky_index} at ky={float(grid.ky[selection.ky_index]):.4f}"
        + ("" if kx_target is None else f", kx={kx[kx_index]:.4f}"),
    )
    kinetic_species = max(len([s for s in cfg.species if s.kinetic]), 1)
    expected_shape = (
        kinetic_species,
        n_laguerre,
        n_hermite,
        int(grid.ky.size),
        int(grid.kx.size),
        int(grid.z.size),
    )
    if initial_state is None:
        _status(status_callback, "building initial condition")
        state = _build_initial_condition(
            grid,
            geom,
            cfg,
            ky_index=selection.ky_index,
            kx_index=selection.kx_index,
            Nl=n_laguerre,
            Nm=n_hermite,
            nspecies=kinetic_species,
        )
    else:
        state = jnp.asarray(initial_state)
        if tuple(state.shape) != expected_shape:
            raise ValueError(
                f"initial_state shape {tuple(state.shape)} does not match "
                f"runtime shape {expected_shape}"
            )
        # State this runtime did not build: on a linked deck it can carry rows
        # the chains never reach, which no growth-rate fit would read but every
        # free-energy and spectrum sum would. Zero them once, at intake.
        state = mask_off_chain_rows(state, grid, geom, params)

    return _LinearRuntimeContext(
        cfg=cfg,
        geom=geom,
        grid=grid,
        params=params,
        terms=terms,
        selection=selection,
        initial_state=state,
        solver_key=_normalize_linear_solver_name(solver),
        fit_key=_fit_signal_key(fit_signal),
        ql_enabled=ql_enabled,
        return_state_requested=return_state_requested,
        return_state_effective=return_state_effective,
        n_laguerre=n_laguerre,
        n_hermite=n_hermite,
    )


def _valid_growth(gamma: float, omega: float, *, require_positive: bool) -> bool:
    return bool(
        np.isfinite(gamma)
        and np.isfinite(omega)
        and (not require_positive or gamma > 0.0)
    )


def _finalize_linear_result(
    result: RuntimeLinearResult,
    *,
    ctx: _LinearRuntimeContext,
    state_for_quasilinear: np.ndarray | None = None,
    status_callback: _StatusCallback,
) -> RuntimeLinearResult:
    return finalize_runtime_linear_quasilinear(
        result,
        enabled=ctx.ql_enabled,
        cfg=ctx.cfg,
        grid=ctx.grid,
        geom=ctx.geom,
        params=ctx.params,
        terms=ctx.terms,
        Nl=ctx.n_laguerre,
        Nm=ctx.n_hermite,
        solver_name=ctx.solver_key,
        species_names=tuple(s.name for s in ctx.cfg.species if s.kinetic),
        return_state_requested=ctx.return_state_requested,
        state_for_quasilinear=state_for_quasilinear,
        status_callback=lambda message: _status(status_callback, message),
    )


def _run_krylov_linear(
    ctx: _LinearRuntimeContext,
    *,
    krylov_cfg: Any | None,
    status_callback: _StatusCallback,
) -> tuple[float, float, np.ndarray, Any]:
    from gkx.solvers_time_explicit import _reject_unsupported_config_collision_operator

    # The eigen solve cannot carry a moment operator; refuse, don't run as LB.
    _reject_unsupported_config_collision_operator(
        ctx.cfg.time, "Krylov eigenvalue", remedy='set solver = "time"'
    )
    _status(status_callback, "starting Krylov solve")
    kcfg = krylov_cfg or _runtime_default_krylov_config(ctx.cfg)
    _status(status_callback, "building linear cache")
    cache = build_linear_cache(
        ctx.grid,
        ctx.geom,
        ctx.params,
        ctx.n_laguerre,
        ctx.n_hermite,
    )
    eig, vec, eigen_status = dominant_eigenpair(
        ctx.initial_state,
        cache,
        ctx.params,
        terms=ctx.terms,
        krylov_dim=kcfg.krylov_dim,
        restarts=kcfg.restarts,
        omega_min_factor=kcfg.omega_min_factor,
        omega_target_factor=kcfg.omega_target_factor,
        omega_cap_factor=kcfg.omega_cap_factor,
        omega_sign=kcfg.omega_sign,
        method=kcfg.method,
        power_iters=kcfg.power_iters,
        power_dt=kcfg.power_dt,
        shift=kcfg.shift,
        shift_source=kcfg.shift_source,
        shift_tol=kcfg.shift_tol,
        shift_maxiter=kcfg.shift_maxiter,
        shift_restart=kcfg.shift_restart,
        shift_solve_method=kcfg.shift_solve_method,
        shift_preconditioner=kcfg.shift_preconditioner,
        shift_selection=kcfg.shift_selection,
        shift_outer_residual_tol=kcfg.shift_outer_residual_tol,
        mode_family=kcfg.mode_family,
        fallback_method=kcfg.fallback_method,
        fallback_real_floor=kcfg.fallback_real_floor,
        certify=kcfg.certify,
        status_callback=lambda message: _status(status_callback, message),
        return_status=True,
    )
    gamma = float(jnp.real(eig))
    omega = float(-jnp.imag(eig))
    gamma, omega = apply_diagnostic_normalization(
        gamma,
        omega,
        rho_star=float(np.asarray(ctx.params.rho_star)),
        diagnostic_norm=ctx.cfg.normalization.diagnostic_norm,
    )
    _status(
        status_callback, f"Krylov solve complete: gamma={gamma:.6f} omega={omega:.6f}"
    )
    return gamma, omega, np.asarray(vec), eigen_status


def _resolve_linear_time_config(
    ctx: _LinearRuntimeContext,
    *,
    method: str | None,
    dt: float | None,
    steps: int | None,
    sample_stride: int | None,
) -> Any:
    tcfg = ctx.cfg.time
    if method is not None:
        tcfg = replace(tcfg, method=str(method))
    if dt is not None:
        tcfg = replace(tcfg, dt=float(dt))
    if steps is not None:
        tcfg = replace(tcfg, t_max=float(steps) * float(tcfg.dt))
    if sample_stride is not None:
        tcfg = replace(tcfg, sample_stride=int(sample_stride))
    if ctx.return_state_effective and ctx.solver_key == "explicit_time":
        raise ValueError(
            "return_state/quasilinear diagnostics are not supported with solver='explicit_time'"
        )
    if ctx.return_state_effective:
        tcfg = replace(tcfg, save_state=True)
    return tcfg


def _warn_if_linear_dt_exceeds_cfl(ctx: _LinearRuntimeContext, tcfg: Any) -> None:
    """Delegate the fixed-step CFL hint to the module that owns the bound."""

    from gkx.solvers_time_explicit_cfl import warn_if_fixed_dt_exceeds_cfl

    warn_if_fixed_dt_exceeds_cfl(
        grid=ctx.grid,
        geom=ctx.geom,
        params=ctx.params,
        n_laguerre=ctx.n_laguerre,
        n_hermite=ctx.n_hermite,
        tcfg=tcfg,
    )


def _validate_parallel_linear_time_path(ctx: _LinearRuntimeContext, tcfg: Any) -> None:
    need_density = ctx.fit_key in {"density", "auto"}
    parallel_strategy = (
        str(getattr(ctx.cfg.parallel, "strategy", "serial")).lower().replace("-", "_")
    )
    if parallel_strategy == "serial":
        return
    if need_density:
        raise NotImplementedError(
            "parallel linear RHS runtime path currently requires fit_signal='phi'"
        )


def _integrate_linear_density_path(
    ctx: _LinearRuntimeContext,
    *,
    tcfg: Any,
    n_steps: int,
    show_progress: bool,
) -> _LinearTrajectory:
    from gkx.solvers_time_runners import _resolve_config_collision_operator

    # This is the executable's density-diagnostic path, so the TOML
    # collision_operator selection has to be resolved here as well as on the
    # cached-phi path.
    request = solve_stats_request(tcfg.method, kind="linear")
    diag = integrate_linear_diagnostics(
        ctx.initial_state,
        ctx.grid,
        ctx.geom,
        ctx.params,
        dt=tcfg.dt,
        steps=n_steps,
        method=tcfg.method,
        terms=ctx.terms,
        sample_stride=tcfg.sample_stride,
        species_index=0,
        record_hl_energy=False,
        show_progress=show_progress,
        implicit_restart=int(tcfg.implicit_restart),
        implicit_preconditioner=tcfg.implicit_preconditioner,
        collision_operator=_resolve_config_collision_operator(
            tcfg, ctx.params, ctx.initial_state
        ),
        **request,
    )
    return _LinearTrajectory(
        g_last=diag[0],
        phi_t=diag[1],
        density_t=diag[2] if len(diag) > 2 else None,
        implicit_solve=checked_solve_summary(
            diag, request, label="linear implicit run"
        ),
    )


def _integrate_linear_cached_phi_path(
    ctx: _LinearRuntimeContext,
    *,
    tcfg: Any,
    show_progress: bool,
) -> _LinearTrajectory:
    request = solve_stats_request(tcfg.method, kind="linear")
    g_last, phi_t, *extra = integrate_linear_from_config(
        ctx.initial_state,
        ctx.grid,
        ctx.geom,
        ctx.params,
        tcfg,
        terms=ctx.terms,
        show_progress=show_progress,
        parallel=ctx.cfg.parallel,
        **request,
    )
    return _LinearTrajectory(
        g_last=g_last,
        phi_t=phi_t,
        density_t=None,
        implicit_solve=checked_solve_summary(
            extra, request, label="linear implicit run"
        ),
    )


def _linear_saved_sample_times(phi_t: np.ndarray, tcfg: Any) -> np.ndarray:
    return (
        float(tcfg.dt)
        * float(tcfg.sample_stride)
        * (np.arange(phi_t.shape[0], dtype=float) + 1.0)
    )


def _integrate_linear_time_series(
    ctx: _LinearRuntimeContext,
    *,
    tcfg: Any,
    show_progress: bool,
    status_callback: _StatusCallback,
) -> _LinearTrajectory:
    need_density = ctx.fit_key in {"density", "auto"}
    n_steps = int(round(tcfg.t_max / tcfg.dt))
    if ctx.solver_key == "explicit_time":
        _status(status_callback, "running CFL-controlled explicit integrator")
        t_explicit, phi_t = integrate_linear_explicit_from_config(
            ctx.initial_state,
            ctx.grid,
            ctx.geom,
            ctx.params,
            tcfg,
            Nl=ctx.n_laguerre,
            Nm=ctx.n_hermite,
            terms=ctx.terms,
            z_index=ctx.selection.z_index,
            show_progress=show_progress,
        )
        trajectory = _LinearTrajectory(None, phi_t, None, t_explicit)
    elif need_density:
        _status(
            status_callback,
            f"running diagnostics integrator over {n_steps} steps with sample_stride={int(tcfg.sample_stride)}",
        )
        trajectory = _integrate_linear_density_path(
            ctx,
            tcfg=tcfg,
            n_steps=n_steps,
            show_progress=show_progress,
        )
    else:
        _status(
            status_callback,
            f"running cached linear integrator over {n_steps} steps with sample_stride={int(tcfg.sample_stride)}",
        )
        trajectory = _integrate_linear_cached_phi_path(
            ctx,
            tcfg=tcfg,
            show_progress=show_progress,
        )

    return trajectory.as_numpy(tcfg)


def _fit_linear_time_series(
    ctx: _LinearRuntimeContext,
    *,
    fit_policy: _LinearFitPolicy,
    trajectory: _LinearTrajectory,
    status_callback: _StatusCallback,
) -> RuntimeLinearResult:
    times = trajectory.t
    if times is None:
        raise RuntimeError(
            "linear trajectory times must be materialized before fitting"
        )
    _status(
        status_callback,
        f"integration complete; fitting growth rate from {times.size} saved samples",
    )
    fit_result = fit_runtime_linear_diagnostics(
        t=times,
        phi_t=trajectory.phi_t,
        density_t=trajectory.density_t,
        selection=ctx.selection,
        z=np.asarray(ctx.grid.z, dtype=float),
        fit_signal=ctx.fit_key,
        mode_method=fit_policy.mode_method,
        auto_window=fit_policy.auto_window,
        tmin=fit_policy.tmin,
        tmax=fit_policy.tmax,
        window_fraction=fit_policy.window_fraction,
        min_points=fit_policy.min_points,
        start_fraction=fit_policy.start_fraction,
        growth_weight=fit_policy.growth_weight,
        require_positive=fit_policy.require_positive,
        min_amp_fraction=fit_policy.min_amp_fraction,
        window_method=fit_policy.window_method,
    )
    if ctx.fit_key == "auto":
        _status(
            status_callback,
            f"automatic fit selected signal '{fit_result.fit_signal_used}'",
        )
    gamma, omega = apply_diagnostic_normalization(
        fit_result.gamma,
        fit_result.omega,
        rho_star=float(np.asarray(ctx.params.rho_star)),
        diagnostic_norm=ctx.cfg.normalization.diagnostic_norm,
    )
    gamma_stderr = fit_result.gamma_stderr
    omega_stderr = fit_result.omega_stderr
    if gamma_stderr is not None and omega_stderr is not None:
        # Standard errors transform with the same linear reporting scale.
        gamma_stderr, omega_stderr = apply_diagnostic_normalization(
            gamma_stderr,
            omega_stderr,
            rho_star=float(np.asarray(ctx.params.rho_star)),
            diagnostic_norm=ctx.cfg.normalization.diagnostic_norm,
        )
    _status(status_callback, f"fit complete: gamma={gamma:.6f} omega={omega:.6f}")
    return RuntimeLinearResult(
        ky=float(ctx.grid.ky[ctx.selection.ky_index]),
        gamma=float(gamma),
        omega=float(omega),
        selection=ctx.selection,
        t=times,
        signal=fit_result.signal,
        field_history=trajectory.phi_t,
        state=None
        if trajectory.g_last is None or not ctx.return_state_effective
        else np.asarray(trajectory.g_last),
        z=fit_result.z,
        eigenfunction=fit_result.eigenfunction,
        fit_window_tmin=fit_result.fit_window_tmin,
        fit_window_tmax=fit_result.fit_window_tmax,
        fit_signal_used=fit_result.fit_signal_used,
        gamma_stderr=None if gamma_stderr is None else float(gamma_stderr),
        omega_stderr=None if omega_stderr is None else float(omega_stderr),
        fit_r2=fit_result.fit_r2,
        fit_settled=fit_result.fit_settled,
        implicit_solve=trajectory.implicit_solve,
    )


def _run_krylov_linear_runtime(
    ctx: _LinearRuntimeContext,
    *,
    krylov_cfg: Any | None,
    status_callback: _StatusCallback,
) -> RuntimeLinearResult:
    """Run the Krylov branch and finalize any requested quasilinear diagnostics."""
    gamma, omega, vec, eigen_status = _run_krylov_linear(
        ctx,
        krylov_cfg=krylov_cfg,
        status_callback=status_callback,
    )
    result = RuntimeLinearResult(
        ky=float(ctx.grid.ky[ctx.selection.ky_index]),
        gamma=gamma,
        omega=omega,
        selection=ctx.selection,
        state=vec if ctx.return_state_effective else None,
        eigen_status=eigen_status,
    )
    return _finalize_linear_result(
        result,
        ctx=ctx,
        state_for_quasilinear=vec,
        status_callback=status_callback,
    )


def _run_linear_runtime_branch(
    ctx: _LinearRuntimeContext,
    *,
    method: str | None,
    dt: float | None,
    steps: int | None,
    sample_stride: int | None,
    fit_policy: _LinearFitPolicy,
    krylov_cfg: Any | None,
    show_progress: bool,
    status_callback: _StatusCallback,
) -> RuntimeLinearResult:
    if ctx.solver_key == "krylov":
        return _run_krylov_linear_runtime(
            ctx,
            krylov_cfg=krylov_cfg,
            status_callback=status_callback,
        )

    _status(
        status_callback, f"starting time integration path with fit_signal={ctx.fit_key}"
    )
    time_config = _resolve_linear_time_config(
        ctx,
        method=method,
        dt=dt,
        steps=steps,
        sample_stride=sample_stride,
    )
    _validate_parallel_linear_time_path(ctx, time_config)
    # ``explicit_time`` used to be excluded from the CFL hint, which left the
    # one linear path that advances a *fixed* step explicitly as the only one
    # that overflowed without saying why. Measured on the shipped Cyclone deck
    # at (Nz,Nl,Nm)=(96,4,8) with the TimeConfig defaults (rk2, dt=0.1,
    # fixed_dt=True): ``solver="time"`` warns "requested dt=0.1 exceeds the
    # estimated CFL-stable step 0.01281 (max linear frequency 70.25)" and then
    # raises FloatingPointError, while ``solver="explicit_time"`` raised the
    # same FloatingPointError with no warning at all. The hint is therefore
    # extended to that path, and only where the step really is fixed: with
    # ``fixed_dt=False`` the same deck and dt integrate cleanly (rk2 gamma
    # 0.10126899, rk4 0.10125984 against the certified 0.10128645), so warning
    # about the adaptive controller's initial guess would be a false positive.
    # The clause is additive -- every warning that fired before still fires.
    if ctx.solver_key != "explicit_time" or bool(
        getattr(time_config, "fixed_dt", True)
    ):
        _warn_if_linear_dt_exceeds_cfl(ctx, time_config)
    trajectory = _integrate_linear_time_series(
        ctx,
        tcfg=time_config,
        show_progress=show_progress,
        status_callback=status_callback,
    )
    result = _fit_linear_time_series(
        ctx,
        fit_policy=fit_policy,
        trajectory=trajectory,
        status_callback=status_callback,
    )
    if ctx.solver_key == "auto" and not _valid_growth(
        result.gamma,
        result.omega,
        require_positive=fit_policy.require_positive,
    ):
        _status(
            status_callback, "time-path result rejected; falling back to Krylov solve"
        )
        return _run_krylov_linear_runtime(
            ctx,
            krylov_cfg=krylov_cfg,
            status_callback=status_callback,
        )

    return _finalize_linear_result(
        result,
        ctx=ctx,
        status_callback=status_callback,
    )


def run_runtime_linear(
    cfg: RuntimeConfig,
    *,
    ky_target: float = 0.3,
    Nl: int | None = None,
    Nm: int | None = None,
    solver: str = "auto",
    method: str | None = None,
    dt: float | None = None,
    steps: int | None = None,
    sample_stride: int | None = None,
    auto_window: bool = True,
    tmin: float | None = None,
    tmax: float | None = None,
    window_fraction: float = 0.4,
    min_points: int = 40,
    start_fraction: float = 0.2,
    growth_weight: float = 0.2,
    require_positive: bool = True,
    min_amp_fraction: float = 0.0,
    window_method: str = "stationary",
    krylov_cfg: KrylovConfig | None = None,
    mode_method: str = "project",
    fit_signal: str = "auto",
    return_state: bool = False,
    initial_state: Any | None = None,
    show_progress: bool = False,
    status_callback: Callable[[str], None] | None = None,
    kx_target: float | None = None,
) -> RuntimeLinearResult:
    """Run one linear point from a case-agnostic runtime config.

    ``kx_target`` picks the fitted kx (nearest grid value; default kx = 0).
    """

    Nl, Nm = _resolve_runtime_hl_dims(cfg, Nl=Nl, Nm=Nm)
    _status(status_callback, "building runtime geometry")

    ctx = _prepare_linear_runtime_context(
        cfg,
        ky_target=ky_target,
        kx_target=kx_target,
        n_laguerre=Nl,
        n_hermite=Nm,
        solver=solver,
        fit_signal=fit_signal,
        return_state=return_state,
        initial_state=initial_state,
        status_callback=status_callback,
    )
    fit_policy = _LinearFitPolicy(
        mode_method=mode_method,
        auto_window=auto_window,
        tmin=tmin,
        tmax=tmax,
        window_fraction=window_fraction,
        min_points=min_points,
        start_fraction=start_fraction,
        growth_weight=growth_weight,
        require_positive=require_positive,
        min_amp_fraction=min_amp_fraction,
        window_method=window_method,
    )

    return _run_linear_runtime_branch(
        ctx,
        method=method,
        dt=dt,
        steps=steps,
        sample_stride=sample_stride,
        fit_policy=fit_policy,
        krylov_cfg=krylov_cfg,
        show_progress=show_progress,
        status_callback=status_callback,
    )


def _indexed_control(value: float | int | np.ndarray | None, index: int, cast):
    if isinstance(value, np.ndarray):
        return cast(value[index])
    return None if value is None else cast(value)


def run_linear_scan(
    *,
    ky_values: np.ndarray,
    run_linear_fn: Callable[..., LinearRunResult],
    cfg: Any,
    Nl: int,
    Nm: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    method: str,
    solver: str,
    krylov_cfg: Any,
    window_kw: dict[str, Any],
    tmin: float | np.ndarray | None = None,
    tmax: float | np.ndarray | None = None,
    auto_window: bool = True,
    run_kwargs: dict[str, Any] | None = None,
    resolution_policy: Callable[[float], tuple[int, int]] | None = None,
    krylov_policy: Callable[[float], object] | None = None,
) -> LinearScanResult:
    """Run a deterministic pointwise linear scan over ``ky_values``."""

    rows: list[tuple[float, float, float]] = []
    for index, ky in enumerate(np.asarray(ky_values, dtype=float)):
        n_l, n_m = (
            resolution_policy(float(ky)) if resolution_policy is not None else (Nl, Nm)
        )
        result = run_linear_fn(
            ky_target=float(ky),
            cfg=cfg,
            Nl=int(n_l),
            Nm=int(n_m),
            dt=_indexed_control(dt, index, float),
            steps=_indexed_control(steps, index, int),
            method=method,
            solver=solver,
            krylov_cfg=(
                krylov_policy(float(ky)) if krylov_policy is not None else krylov_cfg
            ),
            auto_window=auto_window,
            tmin=_indexed_control(tmin, index, float),
            tmax=_indexed_control(tmax, index, float),
            **window_kw,
            **(run_kwargs or {}),
        )
        rows.append((float(result.ky), float(result.gamma), float(result.omega)))
    if not rows:
        empty = np.asarray([], dtype=float)
        return LinearScanResult(ky=empty, gamma=empty.copy(), omega=empty.copy())
    values = np.asarray(rows, dtype=float)
    return LinearScanResult(ky=values[:, 0], gamma=values[:, 1], omega=values[:, 2])
