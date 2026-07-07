"""KBM fixed-ky beta-scan benchmark runner."""

# ruff: noqa: F401

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Sequence

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
from spectraxgk.validation.benchmarks.batching import _resolve_streaming_window
from spectraxgk.diagnostics.growth_rates import (
    _extract_mode_only_signal,
    _normalize_growth_rate,
    _select_fit_signal,
    _select_fit_signal_auto,
)
from spectraxgk.validation.benchmarks.kbm_beta_solver_paths import (
    KBMBetaExplicitHooks,
    KBMBetaKrylovHooks,
    KBMBetaTimeHooks,
    fit_kbm_beta_explicit_time_sample,
    fit_kbm_beta_time_sample,
    solve_kbm_beta_krylov_sample,
)
from spectraxgk.validation.benchmarks.initialization import _build_initial_condition
from spectraxgk.validation.benchmarks.reference import LinearRunResult, LinearScanResult
from spectraxgk.validation.benchmarks.solver_policy import (
    _kbm_use_multi_target_krylov,
    _midplane_index,
    select_kbm_solver_auto,
)
from spectraxgk.validation.benchmarks.species import (
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
