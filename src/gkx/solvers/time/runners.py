"""Config-driven runners for time integration."""

from __future__ import annotations

from typing import Any, cast

from gkx.diagnostics.modes import ModeSelection, ModeSelectionBatch
from gkx.config import TimeConfig
from gkx.solvers.time.diffrax_linear import integrate_linear_diffrax
from gkx.solvers.time.diffrax_nonlinear import integrate_nonlinear_diffrax
from gkx.geometry import FluxTubeGeometryLike
from gkx.core.grid import SpectralGrid
from gkx.solvers.linear.integrators import integrate_linear
from gkx.operators.linear.cache_model import LinearCache
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.params import LinearParams, LinearTerms
from gkx.solvers.nonlinear.state_integration import integrate_nonlinear
from gkx.parallel.state import resolve_state_sharding
from gkx.parallel.integrators import integrate_nonlinear_sharded
from gkx.terms.config import TermConfig


def _steps_from_time(cfg: TimeConfig) -> int:
    if cfg.dt <= 0.0:
        raise ValueError("TimeConfig.dt must be > 0")
    steps = int(round(cfg.t_max / cfg.dt))
    if steps < 1:
        raise ValueError("TimeConfig.t_max must be >= dt")
    return steps


def _resolve_config_collision_operator(time_cfg: TimeConfig, params: LinearParams, G0=None):
    """Build the moment collision operator selected by ``TimeConfig``.

    Returns ``None`` for ``"none"``/``"lenard_bernstein"``, which keeps the
    built-in diagonal Lenard-Bernstein term in the linear RHS. ``"sugama"`` and
    ``"improved_sugama"`` return the dense drift-kinetic Hermite-Laguerre moment
    operator that replaces that diagonal term.
    """

    import jax.numpy as jnp

    from gkx.operators.linear.params import collision_operator_from_config

    name = str(time_cfg.collision_operator).strip().lower()
    if name in ("none", "lenard_bernstein"):
        return None
    # Single-species LinearParams carries scalar normalizations; the moment
    # operator is assembled from per-species vectors.
    density, mass, temperature = (
        jnp.atleast_1d(jnp.asarray(value))
        for value in (params.density, params.mass, params.temp)
    )
    sizes = {int(value.size) for value in (density, mass, temperature)}
    if len(sizes) != 1:
        raise ValueError(
            "collision_operator requires matching per-species density, mass, and "
            f"temperature; got sizes {sorted(sizes)}"
        )
    operator = collision_operator_from_config(
        name,
        density=density,
        mass=mass,
        temperature=temperature,
    )
    _check_moment_basis_matches_operator(operator, name, G0)
    return operator


def _check_moment_basis_matches_operator(operator, name: str, G0) -> None:
    """Reject a moment basis the tabulated collision matrix cannot act on.

    The drift-kinetic Sugama matrices are the fixed truncation of Frei, Ernst &
    Ricci (2022), Appendix C, so they only apply when the run's Hermite-Laguerre
    moment count matches. Without this check the mismatch surfaces deep in the
    RHS as an opaque einsum shape error.
    """

    if operator is None or G0 is None:
        return
    shape = getattr(G0, "shape", None)
    if shape is None or len(shape) not in {5, 6}:
        return
    offset = 0 if len(shape) == 5 else 1
    nl, nm = int(shape[offset]), int(shape[offset + 1])
    # Drift-kinetic operators carry one dense matrix; the finite-wavelength
    # operators carry Bessel-argument-indexed test/field tables instead.
    table = getattr(operator, "matrix", None)
    if table is None:
        table = getattr(operator, "test_table", None)
    if table is None:
        return
    expected = int(table.shape[-1])
    if nl * nm == expected:
        return
    raise ValueError(
        f"collision_operator={name!r} provides a {expected}-moment drift-kinetic "
        f"matrix, but the run uses Nl*Nm = {nl}*{nm} = {nl * nm} moments. "
        f"Choose Nl and Nm with Nl*Nm = {expected} (for example Nl={expected // 2}, "
        "Nm=2), or select collision_operator = \"lenard_bernstein\"."
    )


def _reject_unsupported_config_collision_operator(time_cfg: TimeConfig, path: str) -> None:
    """Fail loudly when a solver path cannot honour the selected operator.

    Silently ignoring ``collision_operator`` would report Lenard-Bernstein
    results under an advanced-operator label, so the unsupported combinations
    raise instead.
    """

    name = str(time_cfg.collision_operator).strip().lower()
    if name in ("none", "lenard_bernstein"):
        return
    raise NotImplementedError(
        f"collision_operator={name!r} is not supported by the {path} path; "
        "it is currently available on the fixed-step cached integrator "
        "(set use_diffrax = false and leave state_sharding unset)."
    )


def _validate_nonlinear_config_state_sharding(spec: str | None) -> None:
    """Keep config-level nonlinear sharding on release-gated state axes."""

    if spec is None:
        return
    key = str(spec).strip().lower()
    if key in {"", "none", "off", "false", "0"}:
        return
    if key not in {"auto", "ky", "kx"}:
        raise ValueError(
            "nonlinear TimeConfig.state_sharding currently supports only 'auto', 'ky', 'kx', or 'none'. "
            "Sharding along the z FFT axis is an exploratory domain-decomposition lane and is not a "
            "release-gated runtime path."
        )


def integrate_linear_from_config(
    G0,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    time_cfg: TimeConfig,
    *,
    cache: LinearCache | None = None,
    terms: LinearTerms | None = None,
    save_mode: ModeSelection | ModeSelectionBatch | None = None,
    mode_method: str = "z_index",
    save_field: str = "phi",
    density_species_index: int | None = None,
    show_progress: bool | None = None,
    parallel: Any | None = None,
) -> tuple:
    """Integrate the linear system using TimeConfig settings."""

    steps = _steps_from_time(time_cfg)
    show_progress_use = bool(time_cfg.progress_bar if show_progress is None else show_progress)
    parallel_strategy = "serial" if parallel is None else str(getattr(parallel, "strategy", "serial")).lower().replace("-", "_")
    if time_cfg.use_diffrax:
        if parallel_strategy != "serial":
            raise NotImplementedError("parallel linear RHS is currently supported only by the fixed-step cached integrator")
        _reject_unsupported_config_collision_operator(time_cfg, "diffrax linear")
        state_sharding = resolve_state_sharding(G0, time_cfg.state_sharding)
        return integrate_linear_diffrax(
            G0,
            grid,
            geom,
            params,
            dt=time_cfg.dt,
            steps=steps,
            method=time_cfg.diffrax_solver,
            cache=cache,
            terms=terms,
            adaptive=time_cfg.diffrax_adaptive,
            rtol=time_cfg.diffrax_rtol,
            atol=time_cfg.diffrax_atol,
            max_steps=time_cfg.diffrax_max_steps,
            show_progress=show_progress_use,
            progress_bar=show_progress_use,
            checkpoint=time_cfg.checkpoint,
            sample_stride=time_cfg.sample_stride,
            return_state=time_cfg.save_state,
            save_mode=save_mode,
            mode_method=mode_method,
            save_field=save_field,
            density_species_index=density_species_index,
            state_sharding=state_sharding,
        )
    return integrate_linear(
        G0,
        grid,
        geom,
        params,
        dt=time_cfg.dt,
        steps=steps,
        method=time_cfg.method,
        cache=cache,
        implicit_restart=time_cfg.implicit_restart,
        implicit_preconditioner=time_cfg.implicit_preconditioner,
        checkpoint=time_cfg.checkpoint,
        sample_stride=time_cfg.sample_stride,
        terms=terms,
        show_progress=show_progress_use,
        parallel=parallel,
        collision_operator=_resolve_config_collision_operator(time_cfg, params, G0),
    )


def integrate_nonlinear_from_config(
    G0,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    time_cfg: TimeConfig,
    *,
    cache: LinearCache | None = None,
    terms: TermConfig | None = None,
    show_progress: bool | None = None,
) -> tuple:
    """Integrate the nonlinear system using TimeConfig settings."""

    steps = _steps_from_time(time_cfg)
    show_progress_use = bool(time_cfg.progress_bar if show_progress is None else show_progress)
    _validate_nonlinear_config_state_sharding(time_cfg.state_sharding)
    state_sharding = resolve_state_sharding(G0, time_cfg.state_sharding)
    if time_cfg.use_diffrax:
        _reject_unsupported_config_collision_operator(time_cfg, "diffrax nonlinear")
        return integrate_nonlinear_diffrax(
            G0,
            grid,
            geom,
            params,
            dt=time_cfg.dt,
            steps=steps,
            method=time_cfg.diffrax_solver,
            cache=cache,
            terms=terms,
            adaptive=time_cfg.diffrax_adaptive,
            rtol=time_cfg.diffrax_rtol,
            atol=time_cfg.diffrax_atol,
            max_steps=time_cfg.diffrax_max_steps,
            show_progress=show_progress_use,
            progress_bar=show_progress_use,
            checkpoint=time_cfg.checkpoint,
            compressed_real_fft=time_cfg.compressed_real_fft,
            laguerre_mode=time_cfg.laguerre_nonlinear_mode,
            state_sharding=state_sharding,
        )
    if state_sharding is not None:
        _reject_unsupported_config_collision_operator(time_cfg, "sharded nonlinear")
        if cache is None:
            if G0.ndim == 5:
                nl, nm = G0.shape[0], G0.shape[1]
            elif G0.ndim == 6:
                nl, nm = G0.shape[1], G0.shape[2]
            else:
                raise ValueError("G0 must have shape (Nl, Nm, Ny, Nx, Nz) or (Ns, Nl, Nm, Ny, Nx, Nz)")
            cache = build_linear_cache(grid, geom, params, int(nl), int(nm))
        return cast(
            tuple,
            integrate_nonlinear_sharded(
                G0,
                cache,
                params,
                dt=time_cfg.dt,
                steps=steps,
                method=time_cfg.method,
                terms=terms,
                state_sharding=state_sharding,
                compressed_real_fft=time_cfg.compressed_real_fft,
                laguerre_mode=time_cfg.laguerre_nonlinear_mode,
                return_fields=True,
            ),
        )
    return cast(
        tuple,
        integrate_nonlinear(
            G0,
            grid,
            geom,
            params,
            dt=time_cfg.dt,
            steps=steps,
            method=time_cfg.method,
            cache=cache,
            terms=terms,
            checkpoint=time_cfg.checkpoint,
            compressed_real_fft=time_cfg.compressed_real_fft,
            laguerre_mode=time_cfg.laguerre_nonlinear_mode,
            show_progress=show_progress_use,
            return_fields=True,
            collision_operator=_resolve_config_collision_operator(time_cfg, params, G0),
        ),
    )
