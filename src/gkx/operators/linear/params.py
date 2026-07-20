"""Linear parameter, term-toggle, and validation policy helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterable

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from gkx.operators.linear.collisions import DriftKineticMomentCollisionOperator
    from gkx.terms.config import TermConfig

__all__ = [
    "COLLISION_OPERATOR_NAMES",
    "collision_operator_from_config",
    "LinearParams",
    "LinearTerms",
    "Species",
    "Preconditioner",
    "PreconditionerSpec",
    "_as_species_array",
    "_check_nonnegative",
    "_check_positive",
    "_is_tracer",
    "_resolve_implicit_preconditioner",
    "_x64_enabled",
    "linear_terms_to_term_config",
    "build_linear_params",
    "term_config_to_linear_terms",
]


@dataclass(frozen=True)
class Species:
    """Dimensionless kinetic-species inputs used to build solver parameters."""

    charge: float
    mass: float
    density: float
    temperature: float
    tprim: float
    fprim: float
    nu: float = 0.0


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LinearParams:
    """Parameters for the linear gyrokinetic operator (supports multi-species arrays)."""

    charge_sign: float | jnp.ndarray = 1.0
    density: float | jnp.ndarray = 1.0
    mass: float | jnp.ndarray = 1.0
    temp: float | jnp.ndarray = 1.0
    tau_e: float = 1.0
    vth: float | jnp.ndarray = 1.0
    rho: float | jnp.ndarray = 1.0
    kpar_scale: float = 1.0
    R_over_Ln: float | jnp.ndarray = 2.2
    R_over_LTi: float | jnp.ndarray = 6.9
    R_over_LTe: float | jnp.ndarray = 0.0
    omega_d_scale: float = 1.0
    omega_star_scale: float = 1.0
    energy_const: float = 0.0
    energy_par_coef: float = 0.5
    energy_perp_coef: float = 1.0
    nu: float | jnp.ndarray = 0.0
    nu_hermite: float = 1.0
    nu_laguerre: float = 2.0
    nu_hyper: float = 0.0
    p_hyper: float = 4.0
    nu_hyper_l: float = 0.0
    nu_hyper_m: float = 1.0
    nu_hyper_lm: float = 0.0
    p_hyper_l: float = 6.0
    p_hyper_m: float = 20.0
    p_hyper_lm: float = 6.0
    hypercollisions_const: float = 1.0
    hypercollisions_kz: float = 0.0
    D_hyper: float = 0.0
    p_hyper_kperp: float = 2.0
    damp_ends_widthfrac: float | jnp.ndarray = 0.125
    damp_ends_amp: float | jnp.ndarray = 0.1
    tz: float | jnp.ndarray = 1.0
    rho_star: float = 1.0
    beta: float = 0.0
    fapar: float = 0.0
    apar_beta_scale: float = 0.5
    ampere_g0_scale: float = 0.5
    bpar_beta_scale: float = 0.5

    def tree_flatten(self):
        children = (
            self.charge_sign,
            self.density,
            self.mass,
            self.temp,
            self.tau_e,
            self.vth,
            self.rho,
            self.kpar_scale,
            self.R_over_Ln,
            self.R_over_LTi,
            self.R_over_LTe,
            self.omega_d_scale,
            self.omega_star_scale,
            self.energy_const,
            self.energy_par_coef,
            self.energy_perp_coef,
            self.nu,
            self.nu_hermite,
            self.nu_laguerre,
            self.nu_hyper,
            self.p_hyper,
            self.nu_hyper_l,
            self.nu_hyper_m,
            self.nu_hyper_lm,
            self.p_hyper_l,
            self.p_hyper_m,
            self.p_hyper_lm,
            self.hypercollisions_const,
            self.hypercollisions_kz,
            self.D_hyper,
            self.p_hyper_kperp,
            self.damp_ends_widthfrac,
            self.damp_ends_amp,
            self.tz,
            self.rho_star,
            self.beta,
            self.fapar,
            self.apar_beta_scale,
            self.ampere_g0_scale,
            self.bpar_beta_scale,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


def build_linear_params(
    species: Iterable[Species],
    **overrides: float,
) -> LinearParams:
    """Build multi-species :class:`LinearParams` from physical inputs."""

    rows = tuple(species)
    if not rows:
        raise ValueError("species must contain at least one kinetic species")

    def array(name: str) -> jnp.ndarray:
        values = [getattr(row, name) for row in rows]
        return jnp.asarray(np.asarray(values, dtype=float))

    charge = array("charge")
    mass = array("mass")
    temperature = array("temperature")
    return LinearParams(
        charge_sign=charge,
        density=array("density"),
        mass=mass,
        temp=temperature,
        vth=jnp.sqrt(temperature / mass),
        rho=jnp.sqrt(temperature * mass) / jnp.abs(charge),
        tz=temperature / charge,
        R_over_LTi=array("tprim"),
        R_over_Ln=array("fprim"),
        nu=array("nu"),
        **overrides,
    )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LinearTerms:
    """Switches for linear-operator components (1.0 = on, 0.0 = off)."""

    streaming: float = 1.0
    mirror: float = 1.0
    curvature: float = 1.0
    gradb: float = 1.0
    diamagnetic: float = 1.0
    collisions: float = 1.0
    hypercollisions: float = 1.0
    hyperdiffusion: float = 0.0
    end_damping: float = 1.0
    apar: float = 1.0
    bpar: float = 1.0

    def tree_flatten(self):
        children = (
            self.streaming,
            self.mirror,
            self.curvature,
            self.gradb,
            self.diamagnetic,
            self.collisions,
            self.hypercollisions,
            self.hyperdiffusion,
            self.end_damping,
            self.apar,
            self.bpar,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


def linear_terms_to_term_config(
    terms: LinearTerms | None,
    *,
    nonlinear: float = 0.0,
) -> TermConfig:
    """Convert :class:`LinearTerms` into the modular :class:`TermConfig`."""

    from gkx.terms.config import TermConfig

    term_weights = terms if terms is not None else LinearTerms()
    return TermConfig(
        streaming=term_weights.streaming,
        mirror=term_weights.mirror,
        curvature=term_weights.curvature,
        gradb=term_weights.gradb,
        diamagnetic=term_weights.diamagnetic,
        collisions=term_weights.collisions,
        hypercollisions=term_weights.hypercollisions,
        hyperdiffusion=term_weights.hyperdiffusion,
        end_damping=term_weights.end_damping,
        apar=term_weights.apar,
        bpar=term_weights.bpar,
        nonlinear=nonlinear,
    )


def term_config_to_linear_terms(term_cfg: TermConfig | None) -> LinearTerms:
    """Convert modular :class:`TermConfig` into linear-only term weights."""

    from gkx.terms.config import TermConfig

    cfg = term_cfg if term_cfg is not None else TermConfig()
    return LinearTerms(
        streaming=cfg.streaming,
        mirror=cfg.mirror,
        curvature=cfg.curvature,
        gradb=cfg.gradb,
        diamagnetic=cfg.diamagnetic,
        collisions=cfg.collisions,
        hypercollisions=cfg.hypercollisions,
        hyperdiffusion=cfg.hyperdiffusion,
        end_damping=cfg.end_damping,
        apar=cfg.apar,
        bpar=cfg.bpar,
    )


def _is_tracer(x) -> bool:
    return isinstance(x, jax.core.Tracer)


def _x64_enabled() -> bool:
    return bool(getattr(jax.config, "jax_enable_x64", False))


def _check_positive(x, name: str) -> None:
    arr = jnp.asarray(x)
    if _is_tracer(x) or _is_tracer(arr):
        return
    if arr.ndim == 0:
        if float(arr) <= 0.0:
            raise ValueError(f"{name} must be > 0")
        return
    if np.any(np.asarray(arr) <= 0.0):
        raise ValueError(f"{name} must be > 0")


def _check_nonnegative(x, name: str) -> None:
    arr = jnp.asarray(x)
    if _is_tracer(x) or _is_tracer(arr):
        return
    if arr.ndim == 0:
        if float(arr) < 0.0:
            raise ValueError(f"{name} must be >= 0")
        return
    if np.any(np.asarray(arr) < 0.0):
        raise ValueError(f"{name} must be >= 0")


def _as_species_array(value: float | jnp.ndarray, ns: int, name: str) -> jnp.ndarray:
    """Ensure a parameter is a 1D array of length ns for multi-species handling."""

    arr = jnp.asarray(value)
    if arr.ndim == 0:
        arr = arr[None]
    if arr.size == 1:
        return jnp.broadcast_to(arr, (ns,))
    if int(arr.size) != int(ns):
        raise ValueError(f"{name} must have length {ns} (got {arr.size})")
    return arr


Preconditioner = Callable[[jnp.ndarray], jnp.ndarray]
PreconditionerSpec = Preconditioner | str | None


def _resolve_implicit_preconditioner(
    preconditioner: PreconditionerSpec,
) -> PreconditionerSpec:
    if preconditioner is None:
        return "auto"
    if isinstance(preconditioner, str):
        return preconditioner.strip().lower()
    return preconditioner


COLLISION_OPERATOR_NAMES: tuple[str, ...] = (
    "none",
    "lenard_bernstein",
    "sugama",
    "improved_sugama",
)


def collision_operator_from_config(
    name: str,
    *,
    density: jnp.ndarray,
    mass: jnp.ndarray,
    temperature: jnp.ndarray,
) -> DriftKineticMomentCollisionOperator | None:
    """Resolve a TOML ``collision_operator`` name to a solver collision operator.

    ``"none"`` and ``"lenard_bernstein"`` return ``None`` so the linear RHS
    keeps its built-in diagonal Lenard-Bernstein term (the solver re-enables
    ``collisions_contribution`` exactly when ``collision_operator is None``).
    ``"sugama"`` and ``"improved_sugama"`` build the dense drift-kinetic
    Hermite-Laguerre moment operator (Frei, Ernst & Ricci 2022) that replaces
    the diagonal term. ``density``/``mass``/``temperature`` are the per-species
    normalizations (length ``n_species``).
    """

    from gkx.operators.linear.collisions import DriftKineticMomentCollisionOperator

    key = name.strip().lower()
    if key in ("none", "lenard_bernstein"):
        return None
    if key == "sugama":
        return DriftKineticMomentCollisionOperator.from_species(
            density, mass, temperature
        )
    if key == "improved_sugama":
        return DriftKineticMomentCollisionOperator.from_improved_species(
            density, mass, temperature
        )
    raise ValueError(f"collision_operator must be one of {COLLISION_OPERATOR_NAMES}")
