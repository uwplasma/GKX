"""Executable workflows for reduced gyrokinetic models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from spectraxgk.diagnostics.modes import ModeSelection, select_ky_index
from spectraxgk.geometry import apply_geometry_grid_defaults
from spectraxgk.core.grid import build_spectral_grid, select_ky_grid
from spectraxgk.workflows.runtime.config import RuntimeConfig
from spectraxgk.workflows.runtime.policies import _midplane_index, _normalize_linear_solver_name
from spectraxgk.workflows.runtime.results import RuntimeLinearResult, RuntimeNonlinearResult
from spectraxgk.workflows.runtime.toml import load_toml


@dataclass(frozen=True)
class ReducedModelContract:
    """Parsed cETG/KREHM input contract used by reduced-model benchmark gates."""

    model: str
    nx: int
    ny: int
    nz: int
    Nl: int
    Nm: int
    x0: float
    y0: float
    z0: float | None
    boundary: str
    dt: float
    t_max: float | None
    cfl: float
    init_field: str
    init_amp: float
    ikpar_init: int
    adiabatic_species: str | None
    tau_fac: float | None
    z_ion: float | None
    zero_shat: bool
    hyper: bool
    D_hyper: float
    dealias_kz: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_reduced_model_contract(path: str | Path) -> ReducedModelContract:
    """Parse a reduced-model input file into a stable contract summary."""

    data = load_toml(path)
    dims = data.get("Dimensions", {})
    domain = data.get("Domain", {})
    time = data.get("Time", {})
    init = data.get("Initialization", {})
    boltz = data.get("Boltzmann", {})
    geo = data.get("Geometry", {})
    diss = data.get("Dissipation", {})
    expert = data.get("Expert", {})
    cetg = bool(data.get("Collisional_slab_ETG", {}).get("cetg", False))
    krehm = bool(data.get("KREHM", {}).get("krehm", False))

    if cetg:
        model = "cetg"
        nl = 2
        nm = 1
    elif krehm:
        model = "krehm"
        nl = int(dims.get("nlaguerre", 1))
        nm = int(dims.get("nhermite", 2))
    else:
        raise ValueError(
            f"{path} is not a recognized reduced-model input (expected cETG or KREHM marker)"
        )

    t_max = time.get("t_max")
    return ReducedModelContract(
        model=model,
        nx=int(dims["nx"]),
        ny=int(dims["ny"]),
        nz=int(dims["ntheta"]),
        Nl=nl,
        Nm=nm,
        x0=float(domain["x0"]),
        y0=float(domain["y0"]),
        z0=float(domain["z0"]) if "z0" in domain else None,
        boundary=str(domain["boundary"]),
        dt=float(time["dt"]),
        t_max=None if t_max is None else float(t_max),
        cfl=float(time.get("cfl", 1.0)),
        init_field=str(init["init_field"]),
        init_amp=float(init["init_amp"]),
        ikpar_init=int(init.get("ikpar_init", 0)),
        adiabatic_species=str(boltz["Boltzmann_type"])
        if "Boltzmann_type" in boltz
        else None,
        tau_fac=None if "tau_fac" not in boltz else float(boltz["tau_fac"]),
        z_ion=None if "Z_ion" not in boltz else float(boltz["Z_ion"]),
        zero_shat=bool(geo.get("zero_shat", False)),
        hyper=bool(diss.get("hyper", False)),
        D_hyper=float(diss.get("D_hyper", 0.0)),
        dealias_kz=bool(expert.get("dealias_kz", False)),
    )


@dataclass(frozen=True)
class CETGLinearRuntimeDeps:
    """Injected cETG workflow dependencies owned by the public runtime facade."""

    build_runtime_geometry: Callable[[RuntimeConfig], Any]
    validate_cetg_runtime_config: Callable[..., Any]
    build_initial_condition: Callable[..., Any]
    build_runtime_term_config: Callable[[RuntimeConfig], Any]
    build_cetg_model_params: Callable[..., Any]
    integrate_cetg_explicit_diagnostics_state: Callable[..., Any]
    fit_growth_rate_auto: Callable[..., tuple[float, float, float | None, float | None]]
    fit_growth_rate: Callable[..., tuple[float, float]]


@dataclass(frozen=True)
class CETGNonlinearRuntimeDeps:
    """Injected dependencies for the cETG nonlinear runtime workflow."""

    build_runtime_geometry: Callable[[RuntimeConfig], Any]
    validate_cetg_runtime_config: Callable[..., Any]
    select_nonlinear_mode_indices: Callable[..., tuple[int, int]]
    build_initial_condition: Callable[..., Any]
    build_cetg_model_params: Callable[..., Any]
    build_runtime_term_config: Callable[[RuntimeConfig], Any]
    integrate_cetg_explicit_diagnostics_state: Callable[..., Any]
    run_adaptive_runtime_chunk_loop: Callable[..., Any]
    build_runtime_nonlinear_result: Callable[..., RuntimeNonlinearResult]


@dataclass(frozen=True)
class _CETGNonlinearSetup:
    """Prepared cETG nonlinear state shared by fixed and adaptive branches."""

    grid: Any
    initial_state: Any
    params: Any
    term_config: Any
    ky_index: int
    kx_index: int
    dt: float
    sample_stride: int
    diagnostics_stride: int
    diagnostics_on: bool


def _resolved_fit_bounds(
    t_arr: np.ndarray,
    tmin_fit: float | None,
    tmax_fit: float | None,
) -> tuple[float | None, float | None]:
    if t_arr.size == 0:
        return None, None
    tmin_use = float(tmin_fit) if tmin_fit is not None else float(t_arr[0])
    tmax_use = float(tmax_fit) if tmax_fit is not None else float(t_arr[-1])
    return tmin_use, tmax_use


def _cetg_integration_kwargs(
    cfg: RuntimeConfig,
    *,
    dt: float,
    steps: int,
    method: str | None,
    sample_stride: int,
    diagnostics_stride: int,
    ky_index: int,
    kx_index: int,
    fixed_dt: bool,
) -> dict[str, Any]:
    """Return shared cETG explicit-integrator options for runtime workflows."""

    return {
        "dt": dt,
        "steps": int(steps),
        "method": str(method or cfg.time.method),
        "sample_stride": int(sample_stride),
        "diagnostics_stride": int(diagnostics_stride),
        "compressed_real_fft": bool(cfg.time.compressed_real_fft),
        "omega_ky_index": int(ky_index),
        "omega_kx_index": int(kx_index),
        "fixed_dt": bool(fixed_dt),
        "dt_min": float(cfg.time.dt_min),
        "dt_max": cfg.time.dt_max,
        "cfl": float(cfg.time.cfl),
        "cfl_fac": cfg.time.cfl_fac,
    }


def _build_cetg_nonlinear_result(
    deps: CETGNonlinearRuntimeDeps,
    *,
    diag: Any,
    fields: Any,
    state: Any,
    grid: Any,
    ky_index: int,
    kx_index: int,
    return_state: bool,
    diagnostics_on: bool,
) -> RuntimeNonlinearResult:
    """Pack cETG nonlinear diagnostics through the shared runtime result schema."""

    return deps.build_runtime_nonlinear_result(
        t=np.asarray(diag.t),
        diagnostics=diag,
        fields=fields,
        state=np.asarray(state) if return_state else None,
        ky_selected=float(np.asarray(grid.ky[ky_index])),
        kx_selected=float(np.asarray(grid.kx[kx_index])),
        summarize_fields=diagnostics_on is False,
    )


def _prepare_cetg_nonlinear_setup(
    cfg: RuntimeConfig,
    *,
    deps: CETGNonlinearRuntimeDeps,
    ky_target: float,
    kx_target: float | None,
    Nl: int,
    Nm: int,
    dt: float | None,
    sample_stride: int | None,
    diagnostics_stride: int | None,
    diagnostics: bool | None,
    status_callback: Callable[[str], None],
) -> _CETGNonlinearSetup:
    """Build the cETG nonlinear grid, state, coefficients, and output policy."""

    geom = deps.build_runtime_geometry(cfg)
    deps.validate_cetg_runtime_config(cfg, geom, Nl=Nl, Nm=Nm)
    status_callback("building spectral grid")
    grid_cfg = apply_geometry_grid_defaults(geom, cfg.grid)
    grid = build_spectral_grid(grid_cfg)
    ky_index, kx_index = deps.select_nonlinear_mode_indices(
        grid,
        ky_target=ky_target,
        kx_target=kx_target,
        use_dealias_mask=bool(cfg.time.nonlinear_dealias),
    )
    status_callback(
        f"selected nonlinear mode ky={float(np.asarray(grid.ky[ky_index])):.6g} "
        f"kx={float(np.asarray(grid.kx[kx_index])):.6g}"
    )
    status_callback("building initial condition")
    initial_state = deps.build_initial_condition(
        grid,
        geom,
        cfg,
        ky_index=ky_index,
        kx_index=kx_index,
        Nl=Nl,
        Nm=Nm,
        nspecies=1,
    )
    dt_val = float(cfg.time.dt if dt is None else dt)
    if dt_val <= 0.0:
        raise ValueError("dt must be > 0")
    setup = _CETGNonlinearSetup(
        grid=grid,
        initial_state=initial_state,
        params=deps.build_cetg_model_params(cfg, geom, Nl=Nl, Nm=Nm),
        term_config=deps.build_runtime_term_config(cfg),
        ky_index=int(ky_index),
        kx_index=int(kx_index),
        dt=dt_val,
        sample_stride=int(
            cfg.time.sample_stride if sample_stride is None else sample_stride
        ),
        diagnostics_stride=int(
            cfg.time.diagnostics_stride
            if diagnostics_stride is None
            else diagnostics_stride
        ),
        diagnostics_on=bool(cfg.time.diagnostics if diagnostics is None else diagnostics),
    )
    status_callback(
        f"nonlinear diagnostics={'on' if setup.diagnostics_on else 'off'} "
        f"sample_stride={setup.sample_stride} diagnostics_stride={setup.diagnostics_stride}"
    )
    return setup


def _run_adaptive_cetg_nonlinear(
    cfg: RuntimeConfig,
    *,
    deps: CETGNonlinearRuntimeDeps,
    setup: _CETGNonlinearSetup,
    method: str | None,
    show_progress: bool,
    return_state: bool,
    status_callback: Callable[[str], None],
) -> RuntimeNonlinearResult:
    """Run the adaptive cETG nonlinear branch through the shared chunk loop."""

    chunk_steps = 1024
    G_chunk = setup.initial_state

    def _run_cetg_chunk(chunk_show_progress: bool):
        nonlocal G_chunk
        kwargs = _cetg_integration_kwargs(
            cfg,
            dt=setup.dt,
            steps=chunk_steps,
            method=method,
            sample_stride=1,
            diagnostics_stride=1,
            ky_index=setup.ky_index,
            kx_index=setup.kx_index,
            fixed_dt=False,
        )
        if chunk_show_progress:
            kwargs["show_progress"] = True
        t_chunk, diag_chunk, G_next, fields_next = (
            deps.integrate_cetg_explicit_diagnostics_state(
                G_chunk,
                setup.grid,
                setup.params,
                setup.term_config,
                **kwargs,
            )
        )
        G_chunk = G_next
        return t_chunk, diag_chunk, G_next, fields_next

    chunk_result = deps.run_adaptive_runtime_chunk_loop(
        integrate_chunk=_run_cetg_chunk,
        t_max=float(cfg.time.t_max),
        chunk_steps=chunk_steps,
        label="cETG",
        show_progress=show_progress,
        status_callback=status_callback,
    )
    return _build_cetg_nonlinear_result(
        deps,
        diag=chunk_result.diagnostics,
        fields=chunk_result.fields,
        state=chunk_result.state,
        grid=setup.grid,
        ky_index=setup.ky_index,
        kx_index=setup.kx_index,
        return_state=return_state,
        diagnostics_on=setup.diagnostics_on,
    )


def _run_fixed_step_cetg_nonlinear(
    cfg: RuntimeConfig,
    *,
    deps: CETGNonlinearRuntimeDeps,
    setup: _CETGNonlinearSetup,
    steps: int | None,
    method: str | None,
    show_progress: bool,
    return_state: bool,
    status_callback: Callable[[str], None],
) -> RuntimeNonlinearResult:
    """Run the fixed-step cETG nonlinear branch and pack runtime diagnostics."""

    steps_val = (
        int(round(float(cfg.time.t_max) / setup.dt)) if steps is None else int(steps)
    )
    if steps_val < 1:
        raise ValueError("steps must be >= 1")
    status_callback(
        f"running cETG nonlinear integration over {steps_val} steps with dt={setup.dt:.6g}"
    )
    progress_kw = {"show_progress": True} if show_progress else {}
    _t, diag, G_final, fields_final = deps.integrate_cetg_explicit_diagnostics_state(
        setup.initial_state,
        setup.grid,
        setup.params,
        setup.term_config,
        **_cetg_integration_kwargs(
            cfg,
            dt=setup.dt,
            steps=steps_val,
            method=method,
            sample_stride=setup.sample_stride,
            diagnostics_stride=setup.diagnostics_stride,
            ky_index=setup.ky_index,
            kx_index=setup.kx_index,
            fixed_dt=bool(cfg.time.fixed_dt),
        ),
        **progress_kw,
    )
    if setup.diagnostics_on is False:
        status_callback("diagnostics disabled; returning final cETG state summary")
    return _build_cetg_nonlinear_result(
        deps,
        diag=diag,
        fields=fields_final,
        state=G_final,
        grid=setup.grid,
        ky_index=setup.ky_index,
        kx_index=setup.kx_index,
        return_state=return_state,
        diagnostics_on=setup.diagnostics_on,
    )


def run_cetg_linear_runtime(
    cfg: RuntimeConfig,
    *,
    deps: CETGLinearRuntimeDeps,
    ky_target: float,
    Nl: int,
    Nm: int,
    solver: str,
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
    return_state: bool,
    status_callback: Callable[[str], None] | None = None,
) -> RuntimeLinearResult:
    """Run the cETG reduced-model linear runtime path."""

    def _status(message: str) -> None:
        if status_callback is not None:
            status_callback(message)

    geom = deps.build_runtime_geometry(cfg)
    deps.validate_cetg_runtime_config(cfg, geom, Nl=Nl, Nm=Nm)
    _status("building spectral grid")
    grid_cfg = apply_geometry_grid_defaults(geom, cfg.grid)
    grid_full = build_spectral_grid(grid_cfg)
    ky_index = select_ky_index(np.asarray(grid_full.ky), ky_target)
    grid = select_ky_grid(grid_full, ky_index)
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))
    _status(f"selected ky index {ky_index} at ky={float(grid.ky[0]):.4f}")
    _status("building initial condition")
    g0 = deps.build_initial_condition(
        grid,
        geom,
        cfg,
        ky_index=sel.ky_index,
        kx_index=sel.kx_index,
        Nl=Nl,
        Nm=Nm,
        nspecies=1,
    )
    cetg_terms = deps.build_runtime_term_config(cfg)
    cetg_params = deps.build_cetg_model_params(cfg, geom, Nl=Nl, Nm=Nm)
    solver_key = _normalize_linear_solver_name(solver)
    if solver_key == "krylov":
        raise NotImplementedError(
            "solver='krylov' is not implemented for physics.reduced_model='cetg'"
        )
    if solver_key not in {"auto", "time", "explicit_time"}:
        raise ValueError(
            "solver must be one of {'auto', 'time', 'explicit_time', 'krylov'}"
        )
    dt_val = float(cfg.time.dt if dt is None else dt)
    if dt_val <= 0.0:
        raise ValueError("dt must be > 0")
    steps_val = (
        int(steps) if steps is not None else int(round(float(cfg.time.t_max) / dt_val))
    )
    if steps_val < 1:
        raise ValueError("steps must be >= 1")
    sample_stride_use = int(cfg.time.sample_stride if sample_stride is None else sample_stride)
    _status(f"running cETG time integration over {steps_val} steps")
    _t, diag, G_final, _fields = deps.integrate_cetg_explicit_diagnostics_state(
        g0,
        grid,
        cetg_params,
        cetg_terms,
        **_cetg_integration_kwargs(
            cfg,
            dt=dt_val,
            steps=steps_val,
            method=method,
            sample_stride=sample_stride_use,
            diagnostics_stride=1,
            ky_index=0,
            kx_index=0,
            fixed_dt=bool(cfg.time.fixed_dt),
        ),
    )
    signal = np.asarray(
        diag.phi_mode_t if diag.phi_mode_t is not None else np.zeros_like(np.asarray(diag.t))
    )
    t_arr = np.asarray(diag.t, dtype=float)
    fit_window_tmin: float | None = None
    fit_window_tmax: float | None = None
    _status(f"integration complete; fitting growth rate from {t_arr.size} saved samples")
    if t_arr.size < 2:
        gamma = (
            float(np.asarray(diag.gamma_t)[-1]) if np.asarray(diag.gamma_t).size else 0.0
        )
        omega = (
            float(np.asarray(diag.omega_t)[-1]) if np.asarray(diag.omega_t).size else 0.0
        )
    elif auto_window:
        gamma, omega, fit_window_tmin, fit_window_tmax = deps.fit_growth_rate_auto(
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
        gamma, omega = deps.fit_growth_rate(t_arr, signal, tmin=tmin, tmax=tmax)
        fit_window_tmin, fit_window_tmax = _resolved_fit_bounds(t_arr, tmin, tmax)
    _status(f"fit complete: gamma={float(gamma):.6f} omega={float(omega):.6f}")
    return RuntimeLinearResult(
        ky=float(grid.ky[0]),
        gamma=float(gamma),
        omega=float(omega),
        selection=sel,
        t=t_arr,
        signal=np.asarray(signal),
        state=np.asarray(G_final) if return_state else None,
        fit_window_tmin=fit_window_tmin,
        fit_window_tmax=fit_window_tmax,
        fit_signal_used="phi",
    )


def run_cetg_nonlinear_runtime(
    cfg: RuntimeConfig,
    *,
    deps: CETGNonlinearRuntimeDeps,
    ky_target: float,
    kx_target: float | None,
    Nl: int,
    Nm: int,
    dt: float | None,
    steps: int | None,
    method: str | None,
    sample_stride: int | None,
    diagnostics_stride: int | None,
    diagnostics: bool | None,
    return_state: bool,
    show_progress: bool,
    status_callback: Callable[[str], None] | None = None,
) -> RuntimeNonlinearResult:
    """Run the cETG reduced-model nonlinear runtime path."""

    def _status(message: str) -> None:
        if status_callback is not None:
            status_callback(message)

    setup = _prepare_cetg_nonlinear_setup(
        cfg,
        deps=deps,
        ky_target=ky_target,
        kx_target=kx_target,
        Nl=Nl,
        Nm=Nm,
        dt=dt,
        sample_stride=sample_stride,
        diagnostics_stride=diagnostics_stride,
        diagnostics=diagnostics,
        status_callback=_status,
    )
    adaptive_chunked = steps is None and not bool(cfg.time.fixed_dt)
    if adaptive_chunked:
        return _run_adaptive_cetg_nonlinear(
            cfg,
            deps=deps,
            setup=setup,
            method=method,
            show_progress=show_progress,
            return_state=return_state,
            status_callback=_status,
        )
    return _run_fixed_step_cetg_nonlinear(
        cfg,
        deps=deps,
        setup=setup,
        steps=steps,
        method=method,
        show_progress=show_progress,
        return_state=return_state,
        status_callback=_status,
    )
