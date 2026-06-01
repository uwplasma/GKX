"""VMEC-JAX QA optimization hooks for SPECTRAX-GK transport objectives.

The classes in this module are optional glue for examples that start from a
``vmec_jax`` fixed-boundary optimization and append a SPECTRAX-GK transport
term. They are intentionally small: VMEC-JAX remains responsible for aspect,
iota, and quasisymmetry constraints, while SPECTRAX-GK supplies the local ITG
linear/quasilinear or reduced nonlinear-window transport residual.

The gradient contract is deliberately conservative. The ``growth`` objective
uses eigenvalue-only differentiation. The quasilinear and reduced nonlinear
window objectives used by the VMEC-JAX optimizer differentiate the same solver
growth rate and combine it with differentiable geometry-level transport
weights. The eigenfunction-resolved quasilinear weights remain an audit path,
not the traced VMEC-JAX optimizer residual, because nonsymmetric eigenvector AD
is not supported by JAX.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
from types import SimpleNamespace
from typing import Any, Literal, Sequence, cast

import jax.numpy as jnp
import numpy as np

from spectraxgk.geometry.differentiable import (
    flux_tube_geometry_from_vmec_boozer_state,
    prewarm_vmec_boozer_equal_arc_cache,
)
from spectraxgk.solver_objective_gradients import (
    SOLVER_OBJECTIVE_NAMES,
    solver_growth_rate_from_geometry,
)
from spectraxgk.stellarator_objective_portfolio import aggregate_objective_portfolio
from spectraxgk.stellarator_optimization import StellaratorITGSampleSet, smooth_positive


VMECJAXTransportObjectiveKind = Literal[
    "growth",
    "quasilinear_flux",
    "nonlinear_window_heat_flux",
]


@dataclass(frozen=True)
class VMECJAXTransportObjectiveConfig:
    """Configuration for VMEC-JAX to SPECTRAX-GK objective evaluation."""

    kind: VMECJAXTransportObjectiveKind = "nonlinear_window_heat_flux"
    sample_set: StellaratorITGSampleSet = field(default_factory=StellaratorITGSampleSet)
    objective_weights: tuple[float, ...] | None = None
    ntheta: int = 24
    mboz: int = 21
    nboz: int = 21
    n_laguerre: int = 2
    n_hermite: int = 3
    nx: int = 1
    ny: int = 4
    nonlinear_csat: float = 0.85
    nonlinear_saturation_floor: float = 1.0e-10
    reference_length: float | None = None
    reference_b: float | None = None
    validate_finite: bool = True

    @property
    def gradient_scope(self) -> str:
        """Return the differentiated part of this objective."""

        if self.kind == "growth":
            return "eigenvalue_growth_ad"
        return "eigenvalue_growth_ad_with_geometry_transport_weights"

    def __post_init__(self) -> None:
        if self.kind not in ("growth", "quasilinear_flux", "nonlinear_window_heat_flux"):
            raise ValueError(f"unknown VMEC-JAX transport objective kind {self.kind!r}")
        if int(self.ntheta) < 4:
            raise ValueError("ntheta must be >= 4")
        if int(self.mboz) < 21 or int(self.nboz) < 21:
            raise ValueError("mboz and nboz must be at least 21 for paper-facing QA optimization")
        if int(self.n_laguerre) < 1 or int(self.n_hermite) < 1:
            raise ValueError("n_laguerre and n_hermite must be positive")
        if int(self.nx) < 1 or int(self.ny) < 3:
            raise ValueError("nx must be positive and ny must be at least 3")
        if float(self.nonlinear_csat) <= 0.0:
            raise ValueError("nonlinear_csat must be positive")

    def objective_options(self) -> dict[str, Any]:
        """Return SPECTRAX-GK solver options for this objective."""

        options: dict[str, Any] = {
            "ntheta": int(self.ntheta),
            "mboz": int(self.mboz),
            "nboz": int(self.nboz),
            "n_laguerre": int(self.n_laguerre),
            "n_hermite": int(self.n_hermite),
            "nx": int(self.nx),
            "ny": int(self.ny),
            "reference_length": self.reference_length,
            "reference_b": self.reference_b,
            "validate_finite": bool(self.validate_finite),
        }
        return {key: value for key, value in options.items() if value is not None}


def _reference_wout_from_context(ctx: Any) -> Any:
    """Return the minimal WOUT metadata needed by the VMEC/Boozer bridge."""

    cfg = getattr(getattr(ctx, "static", None), "cfg", None)
    nfp = int(getattr(cfg, "nfp", 1))
    return SimpleNamespace(
        signgs=int(getattr(ctx, "signgs", 1)),
        nfp=nfp,
        Aminor_p=1.0,
        phi=np.asarray([0.0, -np.pi], dtype=float),
    )


def _solver_table_to_nonlinear_window_proxy(
    table: jnp.ndarray,
    config: VMECJAXTransportObjectiveConfig,
) -> jnp.ndarray:
    """Map linear solver rows to a smooth reduced nonlinear heat-flux proxy."""

    idx = {name: i for i, name in enumerate(SOLVER_OBJECTIVE_NAMES)}
    gamma = jnp.asarray(table[..., idx["gamma"]])
    kperp_eff2 = jnp.asarray(table[..., idx["kperp_eff2"]])
    heat_weight = jnp.asarray(table[..., idx["linear_heat_flux_weight"]])
    gamma_plus = smooth_positive(gamma, beta=18.0)
    saturation = 1.0 + 2.2 * jnp.maximum(kperp_eff2, 0.0) + 0.15 * gamma_plus
    mean_energy = 2.0 * gamma_plus / jnp.maximum(
        saturation,
        jnp.asarray(config.nonlinear_saturation_floor, dtype=gamma_plus.dtype),
    )
    return float(config.nonlinear_csat) * jnp.maximum(heat_weight, 0.0) * mean_energy


def _static_grid_options_from_ky_values(
    ky_values: Sequence[float],
    *,
    min_ny: int,
) -> dict[str, Any]:
    """Resolve physical ``ky`` samples without creating traced JAX arrays."""

    values = np.asarray(tuple(float(value) for value in ky_values), dtype=float)
    if values.ndim != 1 or values.size < 1 or not np.all(np.isfinite(values)):
        raise ValueError("ky_values must be a finite non-empty vector")
    if np.any(values <= 0.0):
        raise ValueError("ky_values must be positive")
    base = float(np.min(values))
    ratios = values / base
    indices = np.rint(ratios).astype(int)
    if np.any(indices < 1) or not np.allclose(ratios, indices, rtol=5.0e-10, atol=5.0e-12):
        raise ValueError("ky_values must be positive integer multiples of their minimum value")
    if len(set(int(item) for item in indices)) != int(indices.size):
        raise ValueError("ky_values map to duplicate selected ky indices")
    ny = max(int(min_ny), 2 * int(np.max(indices)) + 2)
    return {
        "ky_base": base,
        "ly": float(2.0 * np.pi / base),
        "ny": int(ny),
        "selected_ky_indices": tuple(int(item) for item in indices),
    }


def _geometry_transport_weights(
    geom: Any,
    *,
    selected_ky_index: int,
    ly: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return trace-safe geometry weights for VMEC-JAX transport residuals."""

    theta = jnp.asarray(getattr(geom, "theta"))
    dtype = theta.dtype
    bmag = jnp.asarray(getattr(geom, "bmag_profile", jnp.ones_like(theta)), dtype=dtype)
    jac = jnp.abs(jnp.asarray(getattr(geom, "jacobian_profile", jnp.ones_like(theta)), dtype=dtype))
    norm = jnp.maximum(jnp.sum(jac), jnp.asarray(1.0e-30, dtype=dtype))
    weights = jac / norm
    gds2 = jnp.asarray(getattr(geom, "gds2_profile", jnp.ones_like(theta)), dtype=dtype)
    gds21 = jnp.asarray(getattr(geom, "gds21_profile", jnp.zeros_like(theta)), dtype=dtype)
    gds22 = jnp.asarray(getattr(geom, "gds22_profile", jnp.ones_like(theta)), dtype=dtype)
    cv = jnp.asarray(getattr(geom, "cv_profile", jnp.zeros_like(theta)), dtype=dtype)
    gb = jnp.asarray(getattr(geom, "gb_profile", jnp.zeros_like(theta)), dtype=dtype)
    cv0 = jnp.asarray(getattr(geom, "cv0_profile", jnp.zeros_like(theta)), dtype=dtype)
    gb0 = jnp.asarray(getattr(geom, "gb0_profile", jnp.zeros_like(theta)), dtype=dtype)
    mean_b = jnp.sum(weights * bmag)
    ripple = jnp.sqrt(
        jnp.sum(weights * (bmag / jnp.maximum(jnp.abs(mean_b), jnp.asarray(1.0e-30, dtype=dtype)) - 1.0) ** 2)
    )
    metric = jnp.sqrt(jnp.sum(weights * (gds2**2 + 2.0 * gds21**2 + gds22**2)))
    drift = jnp.sqrt(jnp.sum(weights * (cv**2 + gb**2 + cv0**2 + gb0**2)))
    ky = jnp.asarray(2.0 * np.pi * float(selected_ky_index) / float(ly), dtype=dtype)
    kperp_eff2 = jnp.maximum(
        ky**2 * jnp.sum(weights * jnp.maximum(gds2, jnp.asarray(1.0e-8, dtype=dtype))),
        jnp.asarray(1.0e-8, dtype=dtype),
    )
    heat_weight = 0.08 + 0.35 * ripple + 0.08 * metric + 0.22 * drift
    particle_weight = 0.25 * heat_weight
    return kperp_eff2, heat_weight, particle_weight


def _transport_feature_table_from_state(
    state: Any,
    static: Any,
    indata: Any,
    wout_reference: Any,
    config: VMECJAXTransportObjectiveConfig,
    grid_options: dict[str, Any],
) -> jnp.ndarray:
    """Return solver-objective rows with trace-safe growth-rate derivatives."""

    samples = config.sample_set
    selected_indices = cast(tuple[int, ...], grid_options["selected_ky_indices"])
    rows: list[jnp.ndarray] = []
    idx = {name: i for i, name in enumerate(SOLVER_OBJECTIVE_NAMES)}
    for torflux in samples.surfaces:
        for alpha in samples.alphas:
            geom = flux_tube_geometry_from_vmec_boozer_state(
                state,
                static,
                indata,
                wout_reference,
                torflux=float(torflux),
                alpha=float(alpha),
                ntheta=int(config.ntheta),
                mboz=int(config.mboz),
                nboz=int(config.nboz),
                reference_length=config.reference_length,
                reference_b=config.reference_b,
                validate_finite=bool(config.validate_finite),
            )
            for selected_ky_index in selected_indices:
                gamma = solver_growth_rate_from_geometry(
                    geom,
                    selected_ky_index=int(selected_ky_index),
                    n_laguerre=int(config.n_laguerre),
                    n_hermite=int(config.n_hermite),
                    nx=int(config.nx),
                    ny=int(grid_options["ny"]),
                    ly=float(grid_options["ly"]),
                )
                if config.kind == "growth":
                    row = jnp.zeros((len(SOLVER_OBJECTIVE_NAMES),), dtype=jnp.asarray(gamma).dtype)
                    row = row.at[idx["gamma"]].set(gamma)
                    rows.append(row)
                    continue
                kperp, heat_weight, particle_weight = _geometry_transport_weights(
                    geom,
                    selected_ky_index=int(selected_ky_index),
                    ly=float(grid_options["ly"]),
                )
                ql_proxy = gamma * heat_weight / jnp.maximum(
                    kperp,
                    jnp.asarray(1.0e-12, dtype=jnp.asarray(gamma).dtype),
                )
                row = jnp.zeros((len(SOLVER_OBJECTIVE_NAMES),), dtype=jnp.asarray(gamma).dtype)
                row = row.at[idx["gamma"]].set(gamma)
                row = row.at[idx["kperp_eff2"]].set(kperp)
                row = row.at[idx["linear_heat_flux_weight"]].set(heat_weight)
                row = row.at[idx["linear_particle_flux_weight"]].set(particle_weight)
                row = row.at[idx["mixing_length_heat_flux_proxy"]].set(ql_proxy)
                rows.append(row)
    if not rows:
        raise RuntimeError("VMEC-JAX transport objective produced no sample rows")
    return jnp.reshape(
        jnp.stack(rows),
        (
            len(samples.surfaces),
            len(samples.alphas),
            len(samples.ky_values),
            len(SOLVER_OBJECTIVE_NAMES),
        ),
    )


def vmec_jax_transport_objective_from_state(
    state: Any,
    static: Any,
    indata: Any,
    wout_reference: Any,
    config: VMECJAXTransportObjectiveConfig | None = None,
) -> jnp.ndarray:
    """Evaluate a scalar SPECTRAX-GK transport objective from a VMEC-JAX state."""

    cfg = config or VMECJAXTransportObjectiveConfig()
    samples = cfg.sample_set
    solver_options = cfg.objective_options()
    grid_options = _static_grid_options_from_ky_values(
        samples.ky_values,
        min_ny=int(solver_options.get("ny", cfg.ny)),
    )
    solver_options["ny"] = int(grid_options["ny"])
    solver_options["ly"] = float(grid_options["ly"])
    table = _transport_feature_table_from_state(
        state,
        static,
        indata,
        wout_reference,
        cfg,
        grid_options,
    )
    if cfg.kind == "nonlinear_window_heat_flux":
        objective_table = _solver_table_to_nonlinear_window_proxy(table, cfg)[..., None]
        weights = (1.0,) if cfg.objective_weights is None else cfg.objective_weights
    elif cfg.kind == "growth":
        objective_table = table[..., SOLVER_OBJECTIVE_NAMES.index("gamma")][..., None]
        weights = (1.0,) if cfg.objective_weights is None else cfg.objective_weights
    else:
        objective_table = table[..., SOLVER_OBJECTIVE_NAMES.index("mixing_length_heat_flux_proxy")][..., None]
        weights = (1.0,) if cfg.objective_weights is None else cfg.objective_weights
    return aggregate_objective_portfolio(
        objective_table,
        surface_weights=samples.surface_weights,
        alpha_weights=samples.alpha_weights,
        ky_weights=samples.ky_weights,
        objective_weights=weights,
        reduction=samples.reduction,
    )


@dataclass(frozen=True)
class VMECJAXSpectraxTransportObjective:
    """Least-squares objective object for ``vmec_jax`` QA optimizers."""

    config: VMECJAXTransportObjectiveConfig = field(default_factory=VMECJAXTransportObjectiveConfig)
    wout_reference: Any | None = None
    name: str = "spectraxgk_transport"

    def J(self, ctx: Any, state: Any) -> jnp.ndarray:
        """Return the scalar transport objective for VMEC-JAX callbacks."""

        wout_ref = self.wout_reference if self.wout_reference is not None else _reference_wout_from_context(ctx)
        return vmec_jax_transport_objective_from_state(
            state,
            ctx.static,
            ctx.indata,
            wout_ref,
            self.config,
        )

    def to_objective_term(self, *, target: float | np.ndarray, residual_weight: float) -> Any:
        """Return a VMEC-JAX ``ObjectiveTerm`` when VMEC-JAX is installed."""

        objective_term = getattr(importlib.import_module("vmec_jax"), "ObjectiveTerm")

        def _prepare(ctx: Any) -> Any:
            wout_ref = self.wout_reference if self.wout_reference is not None else _reference_wout_from_context(ctx)
            prewarm_vmec_boozer_equal_arc_cache(
                ctx.static,
                wout_ref,
                mboz=int(self.config.mboz),
                nboz=int(self.config.nboz),
            )
            return objective_term(
                self.name,
                self.J,
                target=target,
                weight=residual_weight,
                metadata={
                    "objective_family": "spectraxgk_transport",
                    "spectraxgk_transport_kind": self.config.kind,
                    "gradient_scope": self.config.gradient_scope,
                    "mboz": int(self.config.mboz),
                    "nboz": int(self.config.nboz),
                    "ntheta": int(self.config.ntheta),
                },
            )

        return objective_term(
            self.name,
            self.J,
            target=target,
            weight=residual_weight,
            metadata={
                "objective_family": "spectraxgk_transport",
                "spectraxgk_transport_kind": self.config.kind,
                "gradient_scope": self.config.gradient_scope,
                "mboz": int(self.config.mboz),
                "nboz": int(self.config.nboz),
                "ntheta": int(self.config.ntheta),
            },
            prepare=_prepare,
        )


def spectrax_transport_objective_tuple(
    *,
    weight: float,
    config: VMECJAXTransportObjectiveConfig | None = None,
    wout_reference: Any | None = None,
    target: float = 0.0,
) -> tuple[Any, float, float]:
    """Return a tuple that can be appended to ``LeastSquaresProblem.from_tuples``."""

    objective = VMECJAXSpectraxTransportObjective(
        config=config or VMECJAXTransportObjectiveConfig(),
        wout_reference=wout_reference,
    )
    return (objective.J, float(target), float(weight))
