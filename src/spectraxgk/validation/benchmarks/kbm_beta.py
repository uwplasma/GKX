"""KBM fixed-ky beta-scan benchmark runner."""

# ruff: noqa: F401

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Callable, Sequence

import jax.numpy as jnp
import numpy as np

from spectraxgk.diagnostics.analysis import (
    ModeSelection,
    extract_mode_time_series,
    fit_growth_rate,
    fit_growth_rate_auto,
    windowed_growth_rate_from_omega_series,
    instantaneous_growth_rate_from_phi,
    select_ky_index,
)
from spectraxgk.validation.benchmarks.defaults import (
    KBM_KRYLOV_DEFAULT,
    KBM_OMEGA_D_SCALE,
    KBM_OMEGA_STAR_SCALE,
    KBM_RHO_STAR,
)
from spectraxgk.validation.benchmarks.scan import _resolve_streaming_window
from spectraxgk.diagnostics.growth_rates import (
    _extract_mode_only_signal,
    _normalize_growth_rate,
    _select_fit_signal,
    _select_fit_signal_auto,
)
from spectraxgk.validation.benchmarks.defaults import _build_initial_condition
from spectraxgk.validation.benchmarks.defaults import LinearRunResult, LinearScanResult
from spectraxgk.validation.benchmarks.defaults import (
    _kbm_use_multi_target_krylov,
    _midplane_index,
    select_kbm_solver_auto,
)
from spectraxgk.validation.benchmarks.defaults import (
    _linked_boundary_end_damping,
    _two_species_params,
)
from spectraxgk.validation.benchmarks.scan import (
    ScanFitWindowPolicy,
    apply_auto_fit_scan_policy,
    indexed_float_value,
    indexed_scan_value,
    normalize_fit_signal,
    normalize_solver_key,
    scan_window_valid,
)
from spectraxgk.config import KBMBaseCase, TimeConfig, resolve_cfl_fac
from spectraxgk.solvers.time.diffrax import (
    integrate_linear_diffrax_streaming,
)
from spectraxgk.geometry import (
    SAlphaGeometry,
    apply_geometry_grid_defaults,
    build_flux_tube_geometry,
)
from spectraxgk.core.grid import build_spectral_grid, select_ky_grid
from spectraxgk.solvers.time.explicit import (
    ExplicitTimeConfig,
    integrate_linear_explicit_diagnostics,
)
from spectraxgk.linear import integrate_linear, integrate_linear_diagnostics
from spectraxgk.operators.linear.cache import build_linear_cache
from spectraxgk.operators.linear.params import (
    LinearParams,
    LinearTerms,
    linear_terms_to_term_config,
)
from spectraxgk.solvers.linear.krylov import KrylovConfig, dominant_eigenpair
from spectraxgk.solvers.time.runners import integrate_linear_from_config
from spectraxgk.terms.assembly import compute_fields_cached


@dataclass(frozen=True)
class _KBMBetaScanSetup:
    """Shared fixed-ky KBM beta-scan state and patchable solver policies."""

    cfg: KBMBaseCase
    grid: Any
    geom: Any
    selection: ModeSelection
    terms: LinearTerms
    reference_aligned: bool
    damp_ends_amp: float
    damp_ends_widthfrac: float
    solver_key: str
    fit_key: str
    streaming_fit: bool
    mode_only: bool
    diagnostic_norm: str
    krylov_cfg: KrylovConfig
    use_continuation: bool
    fit_policy: ScanFitWindowPolicy
    explicit_hooks: KBMBetaExplicitHooks
    krylov_hooks: KBMBetaKrylovHooks
    time_hooks: KBMBetaTimeHooks


@dataclass(frozen=True)
class _KBMBetaSample:
    """One beta point after species parameters, cache, and initial state exist."""

    beta: float
    index: int
    dt: float
    steps: int
    params: LinearParams
    cache: Any
    initial_state: Any
    solver_use: str


@dataclass(frozen=True)
class _KBMBetaContinuation:
    """Krylov continuation state carried between beta points."""

    prev_vec: Any = None
    prev_eig: Any = None


@dataclass(frozen=True)
class _KBMBetaScanOptions:
    ky_target: float
    n_laguerre: int
    n_hermite: int
    dt: float | np.ndarray
    steps: int | np.ndarray
    method: str
    time_cfg: TimeConfig | None
    kbm_target_factors: Sequence[float] | None
    kbm_beta_transition: float | None
    tmin: float | None
    tmax: float | None
    require_positive: bool
    mode_method: str
    sample_stride: int | None
    streaming_amp_floor: float
    init_species_index: int
    density_species_index: int
    fapar_override: float | None
    apar_beta_scale: float | None
    ampere_g0_scale: float | None
    bpar_beta_scale: float | None


@dataclass
class _KBMBetaScanOutput:
    beta: list[float]
    gamma: list[float]
    omega: list[float]

    @classmethod
    def empty(cls) -> "_KBMBetaScanOutput":
        return cls(beta=[], gamma=[], omega=[])

    def append(self, *, beta: float, gamma: float, omega: float) -> None:
        self.beta.append(float(beta))
        self.gamma.append(float(gamma))
        self.omega.append(float(omega))

    def result(self) -> LinearScanResult:
        return LinearScanResult(
            ky=np.array(self.beta),
            gamma=np.array(self.gamma),
            omega=np.array(self.omega),
        )


@dataclass(frozen=True)
class _KBMBetaScanRequest:
    """Raw public fixed-ky beta-scan inputs before policies are resolved."""

    betas: np.ndarray
    ky_target: float
    Nl: int
    Nm: int
    dt: float | np.ndarray
    steps: int | np.ndarray
    method: str
    cfg: KBMBaseCase | None
    time_cfg: TimeConfig | None
    solver: str
    krylov_cfg: KrylovConfig | None
    kbm_target_factors: Sequence[float] | None
    kbm_beta_transition: float | None
    tmin: float | None
    tmax: float | None
    auto_window: bool
    window_fraction: float
    min_points: int
    start_fraction: float
    growth_weight: float
    require_positive: bool
    min_amp_fraction: float
    mode_method: str
    mode_only: bool
    terms: LinearTerms | None
    sample_stride: int | None
    fit_signal: str
    ky_batch: int
    fixed_batch_shape: bool
    streaming_fit: bool
    streaming_amp_floor: float
    init_species_index: int
    density_species_index: int
    diagnostic_norm: str
    fapar_override: float | None
    apar_beta_scale: float | None
    ampere_g0_scale: float | None
    bpar_beta_scale: float | None
    reference_aligned: bool | None


def _kbm_beta_scan_request_from_locals(values: dict[str, Any]) -> _KBMBetaScanRequest:
    """Build a beta-scan request from ``run_kbm_beta_scan`` locals."""

    names = {field.name for field in fields(_KBMBetaScanRequest)}
    return _KBMBetaScanRequest(**{name: values[name] for name in names})


# Fixed-ky beta-scan path contracts and sample solvers live with the scan owner.
@dataclass(frozen=True)
class KBMBetaExplicitHooks:
    """Patchable numerical hooks used by the explicit-time KBM beta path."""

    integrate_linear_explicit_diagnostics: Callable[..., Any]
    instantaneous_growth_rate_from_phi: Callable[..., Any]
    windowed_growth_rate_from_omega_series: Callable[..., Any]
    extract_mode_time_series: Callable[..., Any]
    fit_growth_rate_auto: Callable[..., Any]
    normalize_growth_rate: Callable[..., tuple[float, float]]
    resolve_cfl_fac: Callable[..., float]


@dataclass(frozen=True)
class KBMBetaKrylovHooks:
    """Patchable numerical hooks used by the Krylov KBM beta path."""

    dominant_eigenpair: Callable[..., Any]
    use_multi_target_krylov: Callable[..., bool]
    normalize_growth_rate: Callable[..., tuple[float, float]]


@dataclass(frozen=True)
class KBMBetaTimeHooks:
    """Patchable numerical hooks used by the saved-time KBM beta path."""

    integrate_linear_diffrax_streaming: Callable[..., Any]
    integrate_linear_from_config: Callable[..., Any]
    integrate_linear_diagnostics: Callable[..., Any]
    integrate_linear: Callable[..., Any]
    resolve_streaming_window: Callable[..., tuple[float, float]]
    midplane_index: Callable[..., int]
    select_fit_signal_auto: Callable[..., tuple[Any, str, float, float]]
    extract_mode_only_signal: Callable[..., Any]
    select_fit_signal: Callable[..., Any]
    normalize_growth_rate: Callable[..., tuple[float, float]]


@dataclass(frozen=True)
class KBMBetaTimeSampleRequest:
    """Inputs for one saved-time or streaming KBM beta sample."""

    G0_jax: Any
    grid: Any
    geom: Any
    cache: Any
    params: Any
    terms: Any
    dt_i: float
    steps_i: int
    method: str
    time_cfg: Any
    sample_stride: int | None
    fit_key: str
    streaming_fit: bool
    streaming_amp_floor: float
    mode_only: bool
    mode_method: str
    sel: Any
    tmin: Any
    tmax: Any
    sample_index: int
    diagnostic_norm: str
    density_species_index: int
    fit_policy: Any


@dataclass(frozen=True)
class KBMBetaKrylovResult:
    """Krylov growth result plus continuation state for the next beta point."""

    gamma: float
    omega: float
    prev_vec: Any
    prev_eig: Any
    fallback_to_time: bool = False


_KBM_KRYLOV_FORWARD_KEYS = (
    "krylov_dim restarts omega_min_factor omega_target_factor omega_cap_factor omega_sign method "
    "power_iters power_dt shift shift_source shift_tol shift_maxiter shift_restart shift_solve_method "
    "shift_preconditioner shift_selection mode_family fallback_method fallback_real_floor"
).split()


def _dominant_kbm_eigenpair(
    hooks: KBMBetaKrylovHooks,
    G0_jax: Any,
    cache: Any,
    params: Any,
    terms: Any,
    krylov_cfg_use: Any,
    **overrides: Any,
) -> tuple[Any, Any]:
    """Call the KBM Krylov solver with one shared target/shift policy."""

    kwargs = {
        "terms": terms,
        "v_ref": None,
        "select_overlap": False,
        **{name: getattr(krylov_cfg_use, name) for name in _KBM_KRYLOV_FORWARD_KEYS},
        **overrides,
    }
    return hooks.dominant_eigenpair(G0_jax, cache, params, **kwargs)


def fit_kbm_beta_explicit_time_sample(
    *,
    G0_jax: Any,
    grid: Any,
    cache: Any,
    params: Any,
    geom: Any,
    terms: Any,
    dt_i: float,
    steps_i: int,
    time_cfg: Any,
    sample_stride: int | None,
    mode_method: str,
    sel: Any,
    diagnostic_norm: str,
    fit_policy: Any,
    hooks: KBMBetaExplicitHooks,
) -> tuple[float, float]:
    """Run and fit one explicit-time KBM beta sample."""

    explicit_mode_method = (
        mode_method if mode_method in {"z_index", "max"} else "z_index"
    )
    explicit_time_cfg = ExplicitTimeConfig(
        dt=dt_i,
        t_max=dt_i * steps_i,
        sample_stride=max(int(sample_stride or 1), 1),
        fixed_dt=bool(time_cfg.fixed_dt) if time_cfg is not None else False,
        use_dealias_mask=bool(getattr(time_cfg, "use_dealias_mask", False))
        if time_cfg is not None
        else False,
        dt_min=float(time_cfg.dt_min) if time_cfg is not None else 1.0e-7,
        dt_max=float(time_cfg.dt_max)
        if (time_cfg is not None and time_cfg.dt_max is not None)
        else None,
        cfl=float(time_cfg.cfl) if time_cfg is not None else 0.9,
        cfl_fac=(
            hooks.resolve_cfl_fac(str(time_cfg.method), time_cfg.cfl_fac)
            if time_cfg is not None
            else float(ExplicitTimeConfig.cfl_fac)
        ),
    )
    t_arr, phi_t, gamma_t, omega_t, _diagnostics = (
        hooks.integrate_linear_explicit_diagnostics(
            G0_jax,
            grid,
            cache,
            params,
            geom,
            explicit_time_cfg,
            terms=terms,
            mode_method=explicit_mode_method,
            z_index=sel.z_index,
            jit=True,
        )
    )
    if t_arr.size > 1:
        gamma, omega = _fit_explicit_growth_history(
            phi_t=phi_t,
            t_arr=t_arr,
            gamma_t=gamma_t,
            omega_t=omega_t,
            mode_method=mode_method,
            sel=sel,
            fit_policy=fit_policy,
            hooks=hooks,
        )
    else:
        gamma = float("nan")
        omega = float("nan")
    return hooks.normalize_growth_rate(gamma, omega, params, diagnostic_norm)


def _fit_explicit_growth_history(
    *,
    phi_t: Any,
    t_arr: Any,
    gamma_t: Any,
    omega_t: Any,
    mode_method: str,
    sel: Any,
    fit_policy: Any,
    hooks: KBMBetaExplicitHooks,
) -> tuple[float, float]:
    """Fit the explicit-time trace using the KBM fallback ladder."""

    phi_np = np.asarray(phi_t)
    t_np = np.asarray(t_arr, dtype=float)
    if mode_method in {"z_index", "max"}:
        try:
            gamma, omega, _gamma_t, _omega_t, _t_mid = (
                hooks.instantaneous_growth_rate_from_phi(
                    phi_np,
                    t_np,
                    sel,
                    navg_fraction=0.5,
                    mode_method=mode_method,
                )
            )
            return gamma, omega
        except ValueError:
            try:
                gamma, omega, _gamma_t, _omega_t = (
                    hooks.windowed_growth_rate_from_omega_series(
                        np.asarray(gamma_t),
                        np.asarray(omega_t),
                        sel,
                        navg_fraction=0.5,
                    )
                )
                return gamma, omega
            except ValueError:
                pass

    signal = hooks.extract_mode_time_series(phi_np, sel, method=mode_method)
    gamma, omega, _tmin, _tmax = hooks.fit_growth_rate_auto(
        t_np,
        signal,
        **{**fit_policy.auto_kwargs(), "window_method": "fixed"},
    )
    return gamma, omega


def fit_kbm_beta_time_sample(
    *,
    G0_jax: Any,
    grid: Any,
    geom: Any,
    cache: Any,
    params: Any,
    terms: Any,
    dt_i: float,
    steps_i: int,
    method: str,
    time_cfg: Any,
    sample_stride: int | None,
    fit_key: str,
    streaming_fit: bool,
    streaming_amp_floor: float,
    mode_only: bool,
    mode_method: str,
    sel: Any,
    tmin: Any,
    tmax: Any,
    sample_index: int,
    diagnostic_norm: str,
    density_species_index: int,
    fit_policy: Any,
    hooks: KBMBetaTimeHooks,
) -> tuple[float, float]:
    """Run and fit one saved-time or streaming KBM beta sample."""

    return _fit_kbm_beta_time_sample_request(
        KBMBetaTimeSampleRequest(
            G0_jax=G0_jax,
            grid=grid,
            geom=geom,
            cache=cache,
            params=params,
            terms=terms,
            dt_i=dt_i,
            steps_i=steps_i,
            method=method,
            time_cfg=time_cfg,
            sample_stride=sample_stride,
            fit_key=fit_key,
            streaming_fit=streaming_fit,
            streaming_amp_floor=streaming_amp_floor,
            mode_only=mode_only,
            mode_method=mode_method,
            sel=sel,
            tmin=tmin,
            tmax=tmax,
            sample_index=sample_index,
            diagnostic_norm=diagnostic_norm,
            density_species_index=density_species_index,
            fit_policy=fit_policy,
        ),
        hooks=hooks,
    )


def _fit_kbm_beta_time_sample_request(
    request: KBMBetaTimeSampleRequest,
    *,
    hooks: KBMBetaTimeHooks,
) -> tuple[float, float]:
    time_cfg_i = _sample_time_config(
        request.time_cfg,
        request.dt_i,
        request.steps_i,
        request.sample_stride,
    )
    if time_cfg_i is not None and time_cfg_i.use_diffrax and request.streaming_fit:
        return _fit_streaming_time_sample(
            G0_jax=request.G0_jax,
            grid=request.grid,
            geom=request.geom,
            cache=request.cache,
            params=request.params,
            terms=request.terms,
            dt_i=request.dt_i,
            steps_i=request.steps_i,
            time_cfg_i=time_cfg_i,
            fit_key=request.fit_key,
            streaming_amp_floor=request.streaming_amp_floor,
            mode_method=request.mode_method,
            tmin=request.tmin,
            tmax=request.tmax,
            sample_index=request.sample_index,
            fit_policy=request.fit_policy,
            diagnostic_norm=request.diagnostic_norm,
            density_species_index=request.density_species_index,
            hooks=hooks,
        )

    phi_t, density_t, stride = _integrate_time_sample_series(
        G0_jax=request.G0_jax,
        grid=request.grid,
        geom=request.geom,
        cache=request.cache,
        params=request.params,
        terms=request.terms,
        dt_i=request.dt_i,
        steps_i=request.steps_i,
        method=request.method,
        time_cfg_i=time_cfg_i,
        sample_stride=request.sample_stride,
        fit_key=request.fit_key,
        mode_only=request.mode_only,
        mode_method=request.mode_method,
        sel=request.sel,
        density_species_index=request.density_species_index,
        hooks=hooks,
    )
    return _fit_saved_time_sample(
        phi_t=phi_t,
        density_t=density_t,
        dt_i=request.dt_i,
        stride=stride,
        fit_key=request.fit_key,
        mode_only=request.mode_only,
        mode_method=request.mode_method,
        sel=request.sel,
        tmin=request.tmin,
        tmax=request.tmax,
        sample_index=request.sample_index,
        diagnostic_norm=request.diagnostic_norm,
        density_species_index=request.density_species_index,
        params=request.params,
        fit_policy=request.fit_policy,
        hooks=hooks,
    )


def _sample_time_config(
    time_cfg: Any,
    dt_i: float,
    steps_i: int,
    sample_stride: int | None,
) -> Any | None:
    if time_cfg is None:
        return None
    cfg_i = replace(time_cfg, dt=dt_i, t_max=dt_i * steps_i)
    if sample_stride is not None:
        cfg_i = replace(cfg_i, sample_stride=sample_stride)
    return cfg_i


def _fit_streaming_time_sample(
    *,
    G0_jax: Any,
    grid: Any,
    geom: Any,
    cache: Any,
    params: Any,
    terms: Any,
    dt_i: float,
    steps_i: int,
    time_cfg_i: Any,
    fit_key: str,
    streaming_amp_floor: float,
    mode_method: str,
    tmin: Any,
    tmax: Any,
    sample_index: int,
    fit_policy: Any,
    diagnostic_norm: str,
    density_species_index: int,
    hooks: KBMBetaTimeHooks,
) -> tuple[float, float]:
    tmin_i, tmax_i = hooks.resolve_streaming_window(
        float(time_cfg_i.t_max),
        _indexed_float_value(tmin, sample_index),
        _indexed_float_value(tmax, sample_index),
        fit_policy.start_fraction,
        fit_policy.window_fraction,
        1.0,
    )
    _, gamma_vals, omega_vals = hooks.integrate_linear_diffrax_streaming(
        G0_jax,
        grid,
        geom,
        params,
        dt=dt_i,
        steps=steps_i,
        method=time_cfg_i.diffrax_solver,
        cache=cache,
        terms=terms,
        adaptive=time_cfg_i.diffrax_adaptive,
        rtol=time_cfg_i.diffrax_rtol,
        atol=time_cfg_i.diffrax_atol,
        max_steps=time_cfg_i.diffrax_max_steps,
        progress_bar=time_cfg_i.progress_bar,
        checkpoint=time_cfg_i.checkpoint,
        tmin=tmin_i,
        tmax=tmax_i,
        fit_signal=fit_key,
        mode_ky_indices=[0],
        mode_kx_index=0,
        mode_z_index=hooks.midplane_index(grid),
        mode_method=mode_method,
        amp_floor=streaming_amp_floor,
        density_species_index=density_species_index if fit_key == "density" else None,
        return_state=False,
    )
    gamma = float(np.asarray(gamma_vals)[0])
    omega = float(np.asarray(omega_vals)[0])
    return hooks.normalize_growth_rate(gamma, omega, params, diagnostic_norm)


def _integrate_time_sample_series(
    *,
    G0_jax: Any,
    grid: Any,
    geom: Any,
    cache: Any,
    params: Any,
    terms: Any,
    dt_i: float,
    steps_i: int,
    method: str,
    time_cfg_i: Any | None,
    sample_stride: int | None,
    fit_key: str,
    mode_only: bool,
    mode_method: str,
    sel: Any,
    density_species_index: int,
    hooks: KBMBetaTimeHooks,
) -> tuple[Any, Any | None, int]:
    if time_cfg_i is not None and time_cfg_i.use_diffrax:
        stride = int(time_cfg_i.sample_stride)
        phi_t, density_t = _integrate_config_time_series(
            G0_jax=G0_jax,
            grid=grid,
            geom=geom,
            cache=cache,
            params=params,
            terms=terms,
            time_cfg_i=time_cfg_i,
            fit_key=fit_key,
            mode_only=mode_only,
            mode_method=mode_method,
            sel=sel,
            density_species_index=density_species_index,
            hooks=hooks,
        )
        return phi_t, density_t, stride

    stride = (
        int(time_cfg_i.sample_stride)
        if time_cfg_i is not None
        else 1 if sample_stride is None else int(sample_stride)
    )
    phi_t, density_t = _integrate_saved_time_series(
        G0_jax=G0_jax,
        grid=grid,
        geom=geom,
        params=params,
        cache=cache,
        terms=terms,
        dt_i=dt_i,
        steps_i=steps_i,
        method=method,
        stride=stride,
        fit_key=fit_key,
        density_species_index=density_species_index,
        hooks=hooks,
    )
    return phi_t, density_t, stride


def _integrate_config_time_series(
    *,
    G0_jax: Any,
    grid: Any,
    geom: Any,
    cache: Any,
    params: Any,
    terms: Any,
    time_cfg_i: Any,
    fit_key: str,
    mode_only: bool,
    mode_method: str,
    sel: Any,
    density_species_index: int,
    hooks: KBMBetaTimeHooks,
) -> tuple[Any, Any | None]:
    save_mode_method = mode_method if mode_method in {"z_index", "max"} else "z_index"
    _, phi_t = hooks.integrate_linear_from_config(
        G0_jax,
        grid,
        geom,
        params,
        time_cfg_i,
        cache=cache,
        terms=terms,
        save_mode=sel if mode_only else None,
        mode_method=save_mode_method,
        save_field="phi+density"
        if fit_key == "auto"
        else ("density" if fit_key == "density" else "phi"),
        density_species_index=density_species_index
        if fit_key in {"density", "auto"}
        else None,
    )
    if fit_key == "auto":
        phi_values, density_values = phi_t
        return phi_values, density_values
    return phi_t, None


def _fit_saved_time_sample(
    *,
    phi_t: Any,
    density_t: Any | None,
    dt_i: float,
    stride: int,
    fit_key: str,
    mode_only: bool,
    mode_method: str,
    sel: Any,
    tmin: Any,
    tmax: Any,
    sample_index: int,
    diagnostic_norm: str,
    density_species_index: int,
    params: Any,
    fit_policy: Any,
    hooks: KBMBetaTimeHooks,
) -> tuple[float, float]:
    phi_t_np = np.asarray(phi_t)
    density_np = None if density_t is None else np.asarray(density_t)
    if fit_key == "density" and density_np is None:
        density_np = phi_t_np
    if fit_key == "auto":
        _signal, _name, gamma, omega = hooks.select_fit_signal_auto(
            np.arange(phi_t_np.shape[0]) * dt_i * stride,
            phi_t_np,
            density_np,
            sel,
            mode_method=mode_method,
            tmin=_indexed_float_value(tmin, sample_index),
            tmax=_indexed_float_value(tmax, sample_index),
            num_windows=8,
            **fit_policy.auto_kwargs(),
        )
        return hooks.normalize_growth_rate(gamma, omega, params, diagnostic_norm)

    signal = _select_time_fit_signal(
        phi_t_np=phi_t_np,
        density_np=density_np,
        fit_key=fit_key,
        mode_only=mode_only,
        mode_method=mode_method,
        sel=sel,
        density_species_index=density_species_index,
        hooks=hooks,
    )
    return fit_policy.fit_signal(
        signal,
        idx=sample_index,
        dt=dt_i,
        stride=stride,
        params=params,
        diagnostic_norm=diagnostic_norm,
    )


def _select_time_fit_signal(
    *,
    phi_t_np: Any,
    density_np: Any | None,
    fit_key: str,
    mode_only: bool,
    mode_method: str,
    sel: Any,
    density_species_index: int,
    hooks: KBMBetaTimeHooks,
) -> Any:
    if (
        mode_only
        and fit_key == "density"
        and density_np is not None
        and density_np.ndim <= 3
    ):
        return hooks.extract_mode_only_signal(
            density_np,
            local_idx=0,
            species_index=density_species_index,
        )
    if mode_only and phi_t_np.ndim <= 2:
        return hooks.extract_mode_only_signal(phi_t_np, local_idx=0)
    return hooks.select_fit_signal(
        phi_t_np,
        density_np,
        sel,
        fit_signal=fit_key,
        mode_method=mode_method,
    )


def _integrate_saved_time_series(
    *,
    G0_jax: Any,
    grid: Any,
    geom: Any,
    params: Any,
    cache: Any,
    terms: Any,
    dt_i: float,
    steps_i: int,
    method: str,
    stride: int,
    fit_key: str,
    density_species_index: int,
    hooks: KBMBetaTimeHooks,
) -> tuple[Any, Any | None]:
    if fit_key in {"density", "auto"}:
        diag_out = hooks.integrate_linear_diagnostics(
            G0_jax,
            grid,
            geom,
            params,
            dt=dt_i,
            steps=steps_i,
            method=method,
            cache=cache,
            terms=terms,
            sample_stride=stride,
            species_index=density_species_index,
        )
        return diag_out[1], diag_out[2] if len(diag_out) > 2 else None

    _, phi_t = hooks.integrate_linear(
        G0_jax,
        grid,
        geom,
        params,
        dt=dt_i,
        steps=steps_i,
        method=method,
        cache=cache,
        terms=terms,
        sample_stride=stride,
    )
    return phi_t, None


def _select_kbm_beta_eigen_candidate(
    *,
    beta: float,
    beta_transition: float,
    eig_candidates: Sequence[Any],
    vec_candidates: Sequence[Any],
) -> tuple[Any, Any]:
    """Select the KBM branch from targeted Krylov candidates."""

    if len(eig_candidates) >= 2 and np.isfinite(beta_transition):
        idx = 1 if float(beta) >= float(beta_transition) else 0
    else:
        growth = np.real(np.asarray([complex(np.asarray(e)) for e in eig_candidates]))
        idx = (
            0
            if np.all(~np.isfinite(growth))
            else int(np.nanargmax(np.where(np.isfinite(growth), growth, -np.inf)))
        )
    return eig_candidates[idx], vec_candidates[idx]


def _solve_multi_target_kbm_eigenpair(
    *,
    beta: float,
    cfg: Any,
    G0_jax: Any,
    cache: Any,
    params: Any,
    terms: Any,
    krylov_cfg_use: Any,
    targets: Sequence[float],
    kbm_beta_transition: float | None,
    hooks: KBMBetaKrylovHooks,
) -> tuple[Any, Any]:
    """Solve all targeted KBM candidates and select the physical beta branch."""

    beta_transition = (
        float(cfg.model.beta)
        if kbm_beta_transition is None
        else float(kbm_beta_transition)
    )
    target_results = [
        _dominant_kbm_eigenpair(
            hooks,
            G0_jax,
            cache,
            params,
            terms,
            krylov_cfg_use,
            omega_target_factor=float(target),
            shift=None,
            shift_source="target",
            shift_selection="targeted",
        )
        for target in targets
    ]
    eig_candidates, vec_candidates = zip(*target_results, strict=True)
    return _select_kbm_beta_eigen_candidate(
        beta=beta,
        beta_transition=beta_transition,
        eig_candidates=eig_candidates,
        vec_candidates=vec_candidates,
    )


def _resolve_kbm_krylov_shift(
    krylov_cfg_use: Any, *, use_continuation: bool, prev_eig: Any
) -> tuple[Any, Any]:
    shift_val = krylov_cfg_use.shift
    if use_continuation and prev_eig is not None:
        shift_val = complex(np.asarray(prev_eig))
    return shift_val, krylov_cfg_use.shift_selection


def _solve_kbm_beta_selected_eigenpair(
    *,
    beta: float,
    cfg: Any,
    G0_jax: Any,
    cache: Any,
    params: Any,
    terms: Any,
    krylov_cfg_use: Any,
    use_continuation: bool,
    prev_vec: Any,
    targets: Sequence[float] | None,
    kbm_beta_transition: float | None,
    shift_val: Any,
    shift_selection: Any,
    hooks: KBMBetaKrylovHooks,
) -> tuple[Any, Any]:
    use_multi_target = hooks.use_multi_target_krylov(
        krylov_cfg_use,
        targets,
        shift=shift_val,
    )
    if use_multi_target:
        assert targets is not None
        return _solve_multi_target_kbm_eigenpair(
            beta=beta,
            cfg=cfg,
            G0_jax=G0_jax,
            cache=cache,
            params=params,
            terms=terms,
            krylov_cfg_use=krylov_cfg_use,
            targets=targets,
            kbm_beta_transition=kbm_beta_transition,
            hooks=hooks,
        )
    return _dominant_kbm_eigenpair(
        hooks,
        G0_jax,
        cache,
        params,
        terms,
        krylov_cfg_use,
        v_ref=prev_vec,
        select_overlap=use_continuation,
        shift=shift_val,
        shift_selection=shift_selection,
    )


def _normalized_kbm_beta_krylov_growth(
    eig: Any,
    *,
    krylov_cfg_use: Any,
    params: Any,
    diagnostic_norm: str,
    hooks: KBMBetaKrylovHooks,
) -> tuple[float, float]:
    gamma = float(np.real(eig))
    omega = float(-np.imag(eig))
    if krylov_cfg_use.omega_sign != 0:
        omega = float(np.sign(krylov_cfg_use.omega_sign)) * abs(omega)
    return hooks.normalize_growth_rate(gamma, omega, params, diagnostic_norm)


def solve_kbm_beta_krylov_sample(
    *,
    beta: float,
    cfg: Any,
    G0_jax: Any,
    cache: Any,
    params: Any,
    terms: Any,
    solver_key: str,
    krylov_cfg_use: Any,
    use_continuation: bool,
    prev_vec: Any,
    prev_eig: Any,
    kbm_target_factors: Sequence[float] | None,
    kbm_beta_transition: float | None,
    diagnostic_norm: str,
    is_valid_growth: Callable[[float, float], bool],
    hooks: KBMBetaKrylovHooks,
) -> KBMBetaKrylovResult:
    """Run one Krylov KBM beta sample and decide whether time fallback is needed."""

    shift_val, shift_selection = _resolve_kbm_krylov_shift(
        krylov_cfg_use,
        use_continuation=use_continuation,
        prev_eig=prev_eig,
    )
    targets: Sequence[float] | None = kbm_target_factors if kbm_target_factors else None
    eig, vec = _solve_kbm_beta_selected_eigenpair(
        beta=beta,
        cfg=cfg,
        G0_jax=G0_jax,
        cache=cache,
        params=params,
        terms=terms,
        krylov_cfg_use=krylov_cfg_use,
        use_continuation=use_continuation,
        prev_vec=prev_vec,
        targets=targets,
        kbm_beta_transition=kbm_beta_transition,
        shift_val=shift_val,
        shift_selection=shift_selection,
        hooks=hooks,
    )
    gamma, omega = _normalized_kbm_beta_krylov_growth(
        eig,
        krylov_cfg_use=krylov_cfg_use,
        params=params,
        diagnostic_norm=diagnostic_norm,
        hooks=hooks,
    )
    if solver_key == "auto" and not is_valid_growth(gamma, omega):
        return KBMBetaKrylovResult(
            gamma=gamma,
            omega=omega,
            prev_vec=prev_vec,
            prev_eig=prev_eig,
            fallback_to_time=True,
        )
    if use_continuation:
        prev_vec = vec
        prev_eig = eig
    return KBMBetaKrylovResult(
        gamma=gamma,
        omega=omega,
        prev_vec=prev_vec,
        prev_eig=prev_eig,
        fallback_to_time=False,
    )


def _indexed_float_value(value: Any, idx: int) -> float | None:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.ndim == 0:
        return float(arr)
    return float(arr[int(idx)])


def _valid_kbm_growth(
    gamma_val: float, omega_val: float, *, require_positive: bool
) -> bool:
    """Return whether a Krylov result is acceptable before time fallback."""

    if not np.isfinite(gamma_val) or not np.isfinite(omega_val):
        return False
    if require_positive and gamma_val <= 0.0:
        return False
    return True


def _validate_kbm_species_indices(
    *, init_species_index: int, density_species_index: int
) -> None:
    """Validate the two kinetic species indices used by the beta scan."""

    if init_species_index < 0 or init_species_index >= 2:
        raise ValueError("init_species_index out of range for kinetic species")
    if density_species_index < 0 or density_species_index >= 2:
        raise ValueError("density_species_index out of range for kinetic species")


def _build_kbm_beta_hooks() -> tuple[
    KBMBetaExplicitHooks, KBMBetaKrylovHooks, KBMBetaTimeHooks
]:
    """Build patchable hook bundles for the KBM beta solver paths."""

    explicit_hooks = KBMBetaExplicitHooks(
        integrate_linear_explicit_diagnostics=integrate_linear_explicit_diagnostics,
        instantaneous_growth_rate_from_phi=instantaneous_growth_rate_from_phi,
        windowed_growth_rate_from_omega_series=windowed_growth_rate_from_omega_series,
        extract_mode_time_series=extract_mode_time_series,
        fit_growth_rate_auto=fit_growth_rate_auto,
        normalize_growth_rate=_normalize_growth_rate,
        resolve_cfl_fac=resolve_cfl_fac,
    )
    krylov_hooks = KBMBetaKrylovHooks(
        dominant_eigenpair=dominant_eigenpair,
        use_multi_target_krylov=_kbm_use_multi_target_krylov,
        normalize_growth_rate=_normalize_growth_rate,
    )
    time_hooks = KBMBetaTimeHooks(
        integrate_linear_diffrax_streaming=integrate_linear_diffrax_streaming,
        integrate_linear_from_config=integrate_linear_from_config,
        integrate_linear_diagnostics=integrate_linear_diagnostics,
        integrate_linear=integrate_linear,
        resolve_streaming_window=_resolve_streaming_window,
        midplane_index=_midplane_index,
        select_fit_signal_auto=_select_fit_signal_auto,
        extract_mode_only_signal=_extract_mode_only_signal,
        select_fit_signal=_select_fit_signal,
        normalize_growth_rate=_normalize_growth_rate,
    )
    return explicit_hooks, krylov_hooks, time_hooks


def _build_kbm_beta_fit_policy(
    *,
    tmin: float | None,
    tmax: float | None,
    auto_window: bool,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
) -> ScanFitWindowPolicy:
    """Pack the shared beta-scan growth-window policy."""

    return ScanFitWindowPolicy(
        tmin=tmin,
        tmax=tmax,
        auto_window=auto_window,
        window_fraction=window_fraction,
        min_points=min_points,
        start_fraction=start_fraction,
        growth_weight=growth_weight,
        require_positive=require_positive,
        min_amp_fraction=min_amp_fraction,
        fit_growth_rate_fn=fit_growth_rate,
        fit_growth_rate_auto_fn=fit_growth_rate_auto,
        normalize_growth_rate_fn=_normalize_growth_rate,
    )


def _build_kbm_beta_scan_setup(
    *,
    cfg: KBMBaseCase | None,
    ky_target: float,
    terms: LinearTerms | None,
    reference_aligned: bool | None,
    diagnostic_norm: str,
    solver: str,
    fit_signal: str,
    streaming_fit: bool,
    mode_only: bool,
    krylov_cfg: KrylovConfig | None,
    fit_policy: ScanFitWindowPolicy,
) -> _KBMBetaScanSetup:
    """Build shared grid, geometry, policy, and hook state for a beta scan."""

    cfg_use = cfg or KBMBaseCase()
    grid_full = build_spectral_grid(cfg_use.grid)
    geom = SAlphaGeometry.from_config(cfg_use.geometry)
    terms_use = terms if terms is not None else LinearTerms(bpar=0.0)
    reference_aligned_use = bool(
        True if reference_aligned is None else reference_aligned
    )
    diagnostic_norm_use = diagnostic_norm
    if reference_aligned_use and diagnostic_norm_use == "none":
        diagnostic_norm_use = "rho_star"
    damp_ends_amp, damp_ends_widthfrac = _linked_boundary_end_damping(
        reference_aligned_use
    )
    fit_key = normalize_fit_signal(fit_signal)
    streaming_fit_use, mode_only_use = apply_auto_fit_scan_policy(
        fit_key, streaming_fit=streaming_fit, mode_only=mode_only
    )
    ky_index = select_ky_index(np.asarray(grid_full.ky), ky_target)
    grid = select_ky_grid(grid_full, ky_index)
    selection = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))
    explicit_hooks, krylov_hooks, time_hooks = _build_kbm_beta_hooks()
    krylov_cfg_use = krylov_cfg or KBM_KRYLOV_DEFAULT
    return _KBMBetaScanSetup(
        cfg=cfg_use,
        grid=grid,
        geom=geom,
        selection=selection,
        terms=terms_use,
        reference_aligned=reference_aligned_use,
        damp_ends_amp=damp_ends_amp,
        damp_ends_widthfrac=damp_ends_widthfrac,
        solver_key=normalize_solver_key(solver),
        fit_key=fit_key,
        streaming_fit=streaming_fit_use,
        mode_only=mode_only_use,
        diagnostic_norm=diagnostic_norm_use,
        krylov_cfg=krylov_cfg_use,
        use_continuation=bool(getattr(krylov_cfg_use, "continuation", False)),
        fit_policy=fit_policy,
        explicit_hooks=explicit_hooks,
        krylov_hooks=krylov_hooks,
        time_hooks=time_hooks,
    )


def _build_kbm_beta_initial_state(
    setup: _KBMBetaScanSetup,
    *,
    Nl: int,
    Nm: int,
    init_species_index: int,
) -> Any:
    """Build the selected-species KBM beta-scan initial condition."""

    state = np.zeros(
        (2, Nl, Nm, setup.grid.ky.size, setup.grid.kx.size, setup.grid.z.size),
        dtype=np.complex64,
    )
    single_species_state = _build_initial_condition(
        setup.grid,
        setup.geom,
        ky_index=setup.selection.ky_index,
        kx_index=setup.selection.kx_index,
        Nl=Nl,
        Nm=Nm,
        init_cfg=setup.cfg.init,
    )
    state[int(init_species_index)] = np.asarray(
        single_species_state, dtype=np.complex64
    )
    return jnp.asarray(state)


def _build_kbm_beta_sample(
    setup: _KBMBetaScanSetup,
    *,
    beta: float,
    sample_index: int,
    ky_target: float,
    Nl: int,
    Nm: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    init_species_index: int,
    fapar_override: float | None,
    apar_beta_scale: float | None,
    ampere_g0_scale: float | None,
    bpar_beta_scale: float | None,
) -> _KBMBetaSample:
    """Create all per-beta inputs needed by the selected solver path."""

    dt_i = float(dt[sample_index]) if isinstance(dt, np.ndarray) else float(dt)
    steps_i = (
        int(steps[sample_index]) if isinstance(steps, np.ndarray) else int(steps)
    )
    params = _two_species_params(
        setup.cfg.model,
        kpar_scale=float(setup.geom.gradpar()),
        omega_d_scale=KBM_OMEGA_D_SCALE,
        omega_star_scale=KBM_OMEGA_STAR_SCALE,
        rho_star=KBM_RHO_STAR,
        beta_override=float(beta),
        fapar_override=fapar_override,
        apar_beta_scale=apar_beta_scale,
        ampere_g0_scale=ampere_g0_scale,
        bpar_beta_scale=bpar_beta_scale,
        damp_ends_amp=setup.damp_ends_amp,
        damp_ends_widthfrac=setup.damp_ends_widthfrac,
        nhermite=Nm,
    )
    return _KBMBetaSample(
        beta=float(beta),
        index=sample_index,
        dt=dt_i,
        steps=steps_i,
        params=params,
        cache=build_linear_cache(setup.grid, setup.geom, params, Nl, Nm),
        initial_state=_build_kbm_beta_initial_state(
            setup, Nl=Nl, Nm=Nm, init_species_index=init_species_index
        ),
        solver_use=select_kbm_solver_auto(
            setup.solver_key,
            ky_target=ky_target,
            reference_aligned=setup.reference_aligned,
        ),
    )


def _fit_kbm_beta_explicit_sample(
    setup: _KBMBetaScanSetup,
    sample: _KBMBetaSample,
    *,
    time_cfg: TimeConfig | None,
    sample_stride: int | None,
    mode_method: str,
) -> tuple[float, float]:
    """Fit one beta sample with the explicit-time diagnostic path."""

    return fit_kbm_beta_explicit_time_sample(
        G0_jax=sample.initial_state,
        grid=setup.grid,
        cache=sample.cache,
        params=sample.params,
        geom=setup.geom,
        terms=setup.terms,
        dt_i=sample.dt,
        steps_i=sample.steps,
        time_cfg=time_cfg,
        sample_stride=sample_stride,
        mode_method=mode_method,
        sel=setup.selection,
        diagnostic_norm=setup.diagnostic_norm,
        fit_policy=setup.fit_policy,
        hooks=setup.explicit_hooks,
    )


def _fit_kbm_beta_krylov_sample(
    setup: _KBMBetaScanSetup,
    sample: _KBMBetaSample,
    *,
    continuation: _KBMBetaContinuation,
    kbm_target_factors: Sequence[float] | None,
    kbm_beta_transition: float | None,
    require_positive: bool,
) -> tuple[float, float, _KBMBetaContinuation, bool]:
    """Fit one beta sample with Krylov and return continuation/fallback state."""

    krylov_result = solve_kbm_beta_krylov_sample(
        beta=sample.beta,
        cfg=setup.cfg,
        G0_jax=sample.initial_state,
        cache=sample.cache,
        params=sample.params,
        terms=setup.terms,
        solver_key=setup.solver_key,
        krylov_cfg_use=setup.krylov_cfg,
        use_continuation=setup.use_continuation,
        prev_vec=continuation.prev_vec,
        prev_eig=continuation.prev_eig,
        kbm_target_factors=kbm_target_factors,
        kbm_beta_transition=kbm_beta_transition,
        diagnostic_norm=setup.diagnostic_norm,
        is_valid_growth=lambda gamma, omega: _valid_kbm_growth(
            gamma, omega, require_positive=require_positive
        ),
        hooks=setup.krylov_hooks,
    )
    if krylov_result.fallback_to_time:
        return (
            krylov_result.gamma,
            krylov_result.omega,
            continuation,
            True,
        )
    return (
        krylov_result.gamma,
        krylov_result.omega,
        _KBMBetaContinuation(
            prev_vec=krylov_result.prev_vec,
            prev_eig=krylov_result.prev_eig,
        ),
        False,
    )


def _fit_kbm_beta_saved_time_sample(
    setup: _KBMBetaScanSetup,
    sample: _KBMBetaSample,
    *,
    method: str,
    time_cfg: TimeConfig | None,
    sample_stride: int | None,
    mode_method: str,
    tmin: float | None,
    tmax: float | None,
    streaming_amp_floor: float,
    density_species_index: int,
) -> tuple[float, float]:
    """Fit one beta sample with the saved-time or streaming time path."""

    return fit_kbm_beta_time_sample(
        G0_jax=sample.initial_state,
        grid=setup.grid,
        geom=setup.geom,
        cache=sample.cache,
        params=sample.params,
        terms=setup.terms,
        dt_i=sample.dt,
        steps_i=sample.steps,
        method=method,
        time_cfg=time_cfg,
        sample_stride=sample_stride,
        fit_key=setup.fit_key,
        streaming_fit=setup.streaming_fit,
        streaming_amp_floor=streaming_amp_floor,
        mode_only=setup.mode_only,
        mode_method=mode_method,
        sel=setup.selection,
        tmin=tmin,
        tmax=tmax,
        sample_index=sample.index,
        diagnostic_norm=setup.diagnostic_norm,
        density_species_index=density_species_index,
        fit_policy=setup.fit_policy,
        hooks=setup.time_hooks,
    )


def _fit_kbm_beta_sample(
    setup: _KBMBetaScanSetup,
    sample: _KBMBetaSample,
    *,
    method: str,
    time_cfg: TimeConfig | None,
    sample_stride: int | None,
    mode_method: str,
    tmin: float | None,
    tmax: float | None,
    streaming_amp_floor: float,
    density_species_index: int,
    continuation: _KBMBetaContinuation,
    kbm_target_factors: Sequence[float] | None,
    kbm_beta_transition: float | None,
    require_positive: bool,
) -> tuple[float, float, _KBMBetaContinuation]:
    """Dispatch one beta sample through the selected solver policy."""

    solver_use = sample.solver_use
    if solver_use == "explicit_time":
        gamma, omega = _fit_kbm_beta_explicit_sample(
            setup,
            sample,
            time_cfg=time_cfg,
            sample_stride=sample_stride,
            mode_method=mode_method,
        )
        return gamma, omega, continuation

    if solver_use == "krylov":
        gamma, omega, continuation, fallback_to_time = _fit_kbm_beta_krylov_sample(
            setup,
            sample,
            continuation=continuation,
            kbm_target_factors=kbm_target_factors,
            kbm_beta_transition=kbm_beta_transition,
            require_positive=require_positive,
        )
        if not fallback_to_time:
            return gamma, omega, continuation

    gamma, omega = _fit_kbm_beta_saved_time_sample(
        setup,
        sample,
        method=method,
        time_cfg=time_cfg,
        sample_stride=sample_stride,
        mode_method=mode_method,
        tmin=tmin,
        tmax=tmax,
        streaming_amp_floor=streaming_amp_floor,
        density_species_index=density_species_index,
    )
    return gamma, omega, continuation


def _build_kbm_beta_scan_options(
    *,
    ky_target: float,
    Nl: int,
    Nm: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    method: str,
    time_cfg: TimeConfig | None,
    kbm_target_factors: Sequence[float] | None,
    kbm_beta_transition: float | None,
    tmin: float | None,
    tmax: float | None,
    require_positive: bool,
    mode_method: str,
    sample_stride: int | None,
    streaming_amp_floor: float,
    init_species_index: int,
    density_species_index: int,
    fapar_override: float | None,
    apar_beta_scale: float | None,
    ampere_g0_scale: float | None,
    bpar_beta_scale: float | None,
) -> _KBMBetaScanOptions:
    return _KBMBetaScanOptions(
        ky_target=float(ky_target),
        n_laguerre=int(Nl),
        n_hermite=int(Nm),
        dt=dt,
        steps=steps,
        method=method,
        time_cfg=time_cfg,
        kbm_target_factors=kbm_target_factors,
        kbm_beta_transition=kbm_beta_transition,
        tmin=tmin,
        tmax=tmax,
        require_positive=bool(require_positive),
        mode_method=mode_method,
        sample_stride=sample_stride,
        streaming_amp_floor=float(streaming_amp_floor),
        init_species_index=int(init_species_index),
        density_species_index=int(density_species_index),
        fapar_override=fapar_override,
        apar_beta_scale=apar_beta_scale,
        ampere_g0_scale=ampere_g0_scale,
        bpar_beta_scale=bpar_beta_scale,
    )


def _run_kbm_beta_scan_point(
    *,
    setup: _KBMBetaScanSetup,
    beta: float,
    index: int,
    options: _KBMBetaScanOptions,
    continuation: _KBMBetaContinuation,
) -> tuple[float, float, _KBMBetaContinuation]:
    sample = _build_kbm_beta_sample(
        setup,
        beta=float(beta),
        sample_index=index,
        ky_target=options.ky_target,
        Nl=options.n_laguerre,
        Nm=options.n_hermite,
        dt=options.dt,
        steps=options.steps,
        init_species_index=options.init_species_index,
        fapar_override=options.fapar_override,
        apar_beta_scale=options.apar_beta_scale,
        ampere_g0_scale=options.ampere_g0_scale,
        bpar_beta_scale=options.bpar_beta_scale,
    )
    return _fit_kbm_beta_sample(
        setup,
        sample,
        method=options.method,
        time_cfg=options.time_cfg,
        sample_stride=options.sample_stride,
        mode_method=options.mode_method,
        tmin=options.tmin,
        tmax=options.tmax,
        streaming_amp_floor=options.streaming_amp_floor,
        density_species_index=options.density_species_index,
        continuation=continuation,
        kbm_target_factors=options.kbm_target_factors,
        kbm_beta_transition=options.kbm_beta_transition,
        require_positive=options.require_positive,
    )


def _run_kbm_beta_scan_loop(
    betas: np.ndarray,
    *,
    setup: _KBMBetaScanSetup,
    options: _KBMBetaScanOptions,
) -> _KBMBetaScanOutput:
    output = _KBMBetaScanOutput.empty()
    continuation = _KBMBetaContinuation()
    for index, beta in enumerate(betas):
        gamma, omega, continuation = _run_kbm_beta_scan_point(
            setup=setup,
            beta=float(beta),
            index=index,
            options=options,
            continuation=continuation,
        )
        output.append(beta=float(beta), gamma=gamma, omega=omega)
    return output


def _run_kbm_beta_scan_request(request: _KBMBetaScanRequest) -> LinearScanResult:
    """Resolve KBM beta-scan policies and execute the fixed-ky beta sweep."""

    _validate_kbm_species_indices(
        init_species_index=request.init_species_index,
        density_species_index=request.density_species_index,
    )
    fit_policy = _build_kbm_beta_fit_policy(
        tmin=request.tmin,
        tmax=request.tmax,
        auto_window=request.auto_window,
        window_fraction=request.window_fraction,
        min_points=request.min_points,
        start_fraction=request.start_fraction,
        growth_weight=request.growth_weight,
        require_positive=request.require_positive,
        min_amp_fraction=request.min_amp_fraction,
    )
    setup = _build_kbm_beta_scan_setup(
        cfg=request.cfg,
        ky_target=request.ky_target,
        terms=request.terms,
        reference_aligned=request.reference_aligned,
        diagnostic_norm=request.diagnostic_norm,
        solver=request.solver,
        fit_signal=request.fit_signal,
        streaming_fit=request.streaming_fit,
        mode_only=request.mode_only,
        krylov_cfg=request.krylov_cfg,
        fit_policy=fit_policy,
    )
    options = _build_kbm_beta_scan_options(
        ky_target=request.ky_target,
        Nl=request.Nl,
        Nm=request.Nm,
        dt=request.dt,
        steps=request.steps,
        method=request.method,
        time_cfg=request.time_cfg,
        kbm_target_factors=request.kbm_target_factors,
        kbm_beta_transition=request.kbm_beta_transition,
        tmin=request.tmin,
        tmax=request.tmax,
        require_positive=request.require_positive,
        mode_method=request.mode_method,
        sample_stride=request.sample_stride,
        streaming_amp_floor=request.streaming_amp_floor,
        init_species_index=request.init_species_index,
        density_species_index=request.density_species_index,
        fapar_override=request.fapar_override,
        apar_beta_scale=request.apar_beta_scale,
        ampere_g0_scale=request.ampere_g0_scale,
        bpar_beta_scale=request.bpar_beta_scale,
    )
    return _run_kbm_beta_scan_loop(
        np.asarray(request.betas, dtype=float),
        setup=setup,
        options=options,
    ).result()


def run_kbm_beta_scan(
    betas: np.ndarray,
    ky_target: float = 0.3,
    Nl: int = 6,
    Nm: int = 12,
    dt: float | np.ndarray = 0.01,
    steps: int | np.ndarray = 800,
    method: str = "imex2",
    cfg: KBMBaseCase | None = None,
    time_cfg: TimeConfig | None = None,
    solver: str = "auto",
    krylov_cfg: KrylovConfig | None = None,
    kbm_target_factors: Sequence[float] | None = (0.7, 1.5),
    kbm_beta_transition: float | None = None,
    tmin: float | None = None,
    tmax: float | None = None,
    auto_window: bool = True,
    window_fraction: float = 0.4,
    min_points: int = 40,
    start_fraction: float = 0.2,
    growth_weight: float = 1.0,
    require_positive: bool = True,
    min_amp_fraction: float = 0.0,
    mode_method: str = "project",
    mode_only: bool = True,
    terms: LinearTerms | None = None,
    sample_stride: int | None = None,
    fit_signal: str = "auto",
    ky_batch: int = 4,
    fixed_batch_shape: bool = True,
    streaming_fit: bool = True,
    streaming_amp_floor: float = 1.0e-30,
    init_species_index: int = 1,
    density_species_index: int = 1,
    diagnostic_norm: str = "none",
    fapar_override: float | None = None,
    apar_beta_scale: float | None = None,
    ampere_g0_scale: float | None = None,
    bpar_beta_scale: float | None = None,
    reference_aligned: bool | None = True,
) -> LinearScanResult:
    """Run a KBM beta scan at fixed ky.

    If ``time_cfg`` is provided, its ``dt`` and ``t_max`` override ``dt``/``steps``.
    """

    return _run_kbm_beta_scan_request(_kbm_beta_scan_request_from_locals(locals()))
