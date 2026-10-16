"""Tabulated collision matrices and differentiable wavelength interpolation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import io
import json
from importlib import resources
from typing import Any
import jax
import jax.numpy as jnp
import numpy as np
import dataclasses

from gkx.operators.collision import (
    CollisionContext,
    CollisionOperator,
)
from gkx.operators.linear.collisions import (
    _FieldPytree,
    EqualSpeciesFiniteWavelengthCoulombOperator,
    apply_multispecies_collision_moment_matrix,
    interpolate_collision_diagonal_table,
)
from gkx.operators.linear.params import (
    BUILTIN_COLLISION_OPERATORS,
    COLLISION_OPERATOR_NAMES,
)


_FINITE_WAVELENGTH_STEM = "finite_wavelength_coulomb"
# Shipped resolutions, keyed by Hermite-Laguerre moment count (P+1)*(J+1). The
# default eight-moment table matches the drift-kinetic tables so the two
# operators can be compared at equal resolution; the larger ones exist because
# published convergence studies need considerably more.
FINITE_WAVELENGTH_MOMENT_COUNTS: tuple[int, ...] = (8, 18)
_FINITE_WAVELENGTH_BLOCKS = (
    "test_matrix",
    "field_matrix",
    "test_phi1",
    "field_phi1",
    "test_phi2",
    "field_phi2",
)


@lru_cache(maxsize=len(FINITE_WAVELENGTH_MOMENT_COUNTS))
def _finite_wavelength_coulomb_bundle(
    moments: int = 8,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load and provenance-check one shipped finite-wavelength Coulomb table.

    ``moments`` is the Hermite-Laguerre moment count ``(P+1)*(J+1)``, which must
    equal the run's ``Nl*Nm``.
    """

    if moments not in FINITE_WAVELENGTH_MOMENT_COUNTS:
        raise ValueError(
            f"no shipped finite-wavelength Coulomb table with {moments} moments; "
            f"available: {sorted(FINITE_WAVELENGTH_MOMENT_COUNTS)}. Generate more "
            "with tests/support/reference_physics.py"
        )
    suffix = "" if moments == 8 else f"_{moments}"
    data_root = resources.files("gkx").joinpath("data")
    payload = data_root.joinpath(f"{_FINITE_WAVELENGTH_STEM}{suffix}.npz").read_bytes()
    metadata = json.loads(
        data_root.joinpath(f"{_FINITE_WAVELENGTH_STEM}{suffix}.json").read_text(
            encoding="utf-8"
        )
    )
    if hashlib.sha256(payload).hexdigest() != metadata.get("sha256"):
        raise ValueError(
            "finite-wavelength Coulomb coefficient checksum does not match metadata"
        )
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    missing = set(_FINITE_WAVELENGTH_BLOCKS + ("bessel_argument_grid",)) - set(arrays)
    if missing:
        raise ValueError(
            f"finite-wavelength Coulomb tables are missing {sorted(missing)}"
        )
    return arrays, metadata


def finite_wavelength_coulomb_metadata(moments: int = 8) -> dict[str, Any]:
    """Return the provenance metadata of one shipped Coulomb table."""

    return dict(_finite_wavelength_coulomb_bundle(moments)[1])


def build_finite_wavelength_coulomb_operator(
    density: jnp.ndarray,
    mass: jnp.ndarray,
    temperature: jnp.ndarray,
    moments: int = 8,
) -> EqualSpeciesFiniteWavelengthCoulombOperator:
    r"""Build the gyrokinetic (finite-Larmor) Coulomb operator for one species.

    This is the full linearized Coulomb operator retaining finite perpendicular
    wavelength, from Frei, Ball, Hoffmann, Jorge, Ricci & Stenger (2021),
    arXiv:2104.11480, equations (3.47)--(3.50). The shipped tables are indexed
    by the Bessel argument :math:`B=k_\perp v_{\mathrm{th}}/\Omega`; the runtime
    interpolates them at :math:`B=\sqrt{2b}` from the cached ``b``, so the same
    operator covers every perpendicular wavenumber in the simulation.

    As :math:`k_\perp\to0` the tables reduce to the drift-kinetic Coulomb
    operator, which is enforced by a physics gate.

    ``moments`` selects the shipped Hermite-Laguerre resolution and must equal
    the run's ``Nl*Nm``; see ``FINITE_WAVELENGTH_MOMENT_COUNTS``.

    The shipped tables are generated at ``mass_ratio = temperature_ratio = 1``,
    so they are valid for like-species collisions; a multi-species request is
    refused rather than silently extrapolated.
    """

    density_s = jnp.atleast_1d(jnp.asarray(density))
    mass_s = jnp.atleast_1d(jnp.asarray(mass, dtype=jnp.result_type(density_s, float)))
    temperature_s = jnp.atleast_1d(
        jnp.asarray(temperature, dtype=jnp.result_type(density_s, mass_s, float))
    )
    if mass_s.shape != density_s.shape or temperature_s.shape != density_s.shape:
        raise ValueError("density, mass, and temperature must have equal length")
    for value, name in (
        (density_s, "density"),
        (mass_s, "mass"),
        (temperature_s, "temperature"),
    ):
        if not isinstance(value, jax.core.Tracer) and np.any(np.asarray(value) <= 0.0):
            raise ValueError(f"{name} must be positive")

    species_count = int(density_s.size)
    if species_count != 1:
        raise ValueError(
            "the shipped finite-wavelength Coulomb tables are generated for "
            f"like-species collisions, but {species_count} species were given; "
            "use collision_operator='sugama' or 'improved_sugama' for "
            "multispecies runs"
        )

    arrays, _ = _finite_wavelength_coulomb_bundle(moments)
    dtype = jnp.result_type(density_s, mass_s, temperature_s, float)
    # Same dimensionless normalization as the drift-kinetic assemblers.
    frequency = density_s[0] / (jnp.sqrt(mass_s[0]) * temperature_s[0] ** 1.5)
    # The stored block names are the generator's; the operator names its two
    # matrix fields *_table.
    field_names = {
        "test_matrix": "test_table",
        "field_matrix": "field_table",
        "test_phi1": "test_phi1",
        "field_phi1": "field_phi1",
        "test_phi2": "test_phi2",
        "field_phi2": "field_phi2",
    }
    return EqualSpeciesFiniteWavelengthCoulombOperator(
        bessel_argument_grid=jnp.asarray(arrays["bessel_argument_grid"], dtype=dtype),
        pair_frequency=jnp.reshape(frequency, (1, 1)).astype(dtype),
        **{
            field: jnp.asarray(arrays[stored], dtype=dtype)
            for stored, field in field_names.items()
        },
    )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class TabulatedMultispeciesCollisionOperator(_FieldPytree):
    """Finite-wavelength target/source collision matrices on a kperp grid.

    The table contains fully assembled collision-frequency-weighted blocks with
    shape ``(target, source, kperp, moment, moment)``. The runtime derives each
    target species' normalized ``kperp`` from ``sqrt(cache.b)`` and keeps the
    interpolation and matrix application inside JAX.
    """

    kperp_grid: jnp.ndarray
    matrices: jnp.ndarray

    def apply(self, context: CollisionContext) -> jnp.ndarray:
        """Interpolate and apply the table to the post-field Hamiltonian."""

        table = jnp.asarray(self.matrices)
        if table.ndim != 5:
            raise ValueError(
                "tabulated multispecies collision matrices must have target, "
                "source, kperp, and two moment axes"
            )
        kperp = jnp.sqrt(jnp.maximum(jnp.asarray(context.cache.b), 0.0))
        matrix = interpolate_collision_moment_matrix(self.kperp_grid, table, kperp)
        return apply_multispecies_collision_moment_matrix(context.hamiltonian, matrix)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class EqualSpeciesFiniteWavelengthSugamaOperator(_FieldPytree):
    r"""Finite-wavelength original/improved-Sugama tables for one species.

    The tabulated test and field matrices implement Frei et al. (2021),
    equations (3.72) and (3.79), optionally augmented by the improved field
    correction of Frei, Ernst & Ricci (2022), on the post-field nonadiabatic
    response :math:`H`. Unlike the Coulomb table, these models have no
    separately tabulated electrostatic polarization vectors.
    """

    bessel_argument_grid: jnp.ndarray
    pair_frequency: jnp.ndarray
    test_table: jnp.ndarray
    field_table: jnp.ndarray

    def apply(self, context: CollisionContext) -> jnp.ndarray:
        """Interpolate and apply the equal-species original-Sugama matrix."""

        state = jnp.asarray(context.hamiltonian)
        species_count = 1 if state.ndim == 5 else int(state.shape[0])
        if species_count != 1:
            raise ValueError(
                "equal-species finite-wavelength Sugama tables require one species"
            )
        bessel_argument = jnp.sqrt(2.0 * jnp.maximum(jnp.asarray(context.cache.b), 0.0))
        if bessel_argument.ndim < 1 or int(bessel_argument.shape[0]) != 1:
            raise ValueError("collision Bessel argument must have one species axis")
        test, field = (
            interpolate_collision_diagonal_table(
                self.bessel_argument_grid, table, bessel_argument[0]
            )
            for table in (self.test_table, self.field_table)
        )
        frequency = jnp.asarray(self.pair_frequency)
        if frequency.shape != (1, 1):
            raise ValueError("pair_frequency must have shape (1, 1)")
        matrix = (frequency[0, 0] * (test + field))[None, None, ...]
        return apply_multispecies_collision_moment_matrix(state, matrix)


def interpolate_collision_moment_matrix(
    kperp_grid: jnp.ndarray,
    matrices: jnp.ndarray,
    kperp: jnp.ndarray,
) -> jnp.ndarray:
    """Interpolate collision matrices onto a scalar or spatial ``kperp`` field.

    ``matrices`` may contain one shared table, one table per species, or one
    table per ordered target/source species pair. Their respective shapes are
    ``(kperp, modes, modes)``, ``(species, kperp, modes, modes)``, and
    ``(target, source, kperp, modes, modes)``. Values outside the tabulated
    interval use the nearest endpoint. The coefficient grid is validated on
    the host; interpolation and its derivative with respect to ``kperp``
    remain in JAX.
    """

    grid = jnp.asarray(kperp_grid)
    table = jnp.asarray(matrices)
    target = jnp.asarray(kperp, dtype=jnp.result_type(grid, table))
    if grid.ndim != 1 or int(grid.size) < 2:
        raise ValueError(
            "collision kperp grid must be one-dimensional with at least two points"
        )
    if table.ndim not in {3, 4, 5}:
        raise ValueError(
            "collision table must have shape (kperp, modes, modes) or "
            "(species, kperp, modes, modes), optionally with separate "
            "target/source species axes"
        )
    grid_axis = table.ndim - 3
    if int(table.shape[grid_axis]) != int(grid.size):
        raise ValueError("collision table kperp axis must match the coefficient grid")
    if int(table.shape[-1]) != int(table.shape[-2]):
        raise ValueError("collision table matrices must be square")
    if not isinstance(grid, jax.core.Tracer):
        host_grid = np.asarray(grid)
        if not np.all(np.isfinite(host_grid)) or not np.all(np.diff(host_grid) > 0.0):
            raise ValueError(
                "collision kperp grid must be finite and strictly increasing"
            )

    def interpolate_one(species_table: jnp.ndarray, values: jnp.ndarray) -> jnp.ndarray:
        clipped = jnp.clip(values, grid[0], grid[-1])
        left = jnp.clip(
            jnp.searchsorted(grid, clipped, side="right") - 1, 0, grid.size - 2
        )
        fraction = (clipped - grid[left]) / (grid[left + 1] - grid[left])
        interpolated = species_table[left] + fraction[..., None, None] * (
            species_table[left + 1] - species_table[left]
        )
        return jnp.moveaxis(interpolated, (-2, -1), (0, 1))

    if table.ndim == 3:
        return interpolate_one(table, target)
    species_count = int(table.shape[0])
    if table.ndim == 5:
        if int(table.shape[1]) != species_count:
            raise ValueError(
                "multispecies collision table must have equal target/source axes"
            )

        def interpolate_target_pairs(
            pair_tables: jnp.ndarray, values: jnp.ndarray
        ) -> jnp.ndarray:
            return jax.vmap(lambda pair_table: interpolate_one(pair_table, values))(
                pair_tables
            )

        if target.ndim == 0:
            return jax.vmap(lambda pairs: interpolate_target_pairs(pairs, target))(
                table
            )
        if int(target.shape[0]) != species_count:
            raise ValueError(
                "multispecies collision table requires scalar kperp or a "
                "target-species-leading kperp field"
            )
        return jax.vmap(interpolate_target_pairs)(table, target)
    if target.ndim == 0:
        return jax.vmap(lambda species_table: interpolate_one(species_table, target))(
            table
        )
    if int(target.shape[0]) != species_count:
        raise ValueError(
            "species collision table requires scalar kperp or a species-leading kperp field"
        )
    return jax.vmap(interpolate_one)(table, target)


def collision_operator_from_config(
    name: str,
    *,
    density: jnp.ndarray,
    mass: jnp.ndarray,
    temperature: jnp.ndarray,
    nu: jnp.ndarray | float = 1.0,
    moments: int = 8,
) -> CollisionOperator | None:
    """Resolve a TOML ``collision_operator`` name to a solver collision operator.

    The built-in names (``"none"``, ``"lenard_bernstein"``, ``"landau"`` and
    its alias ``"coulomb"``) return ``None``: the linear RHS applies them from
    the cache. ``"sugama"`` and ``"improved_sugama"`` build the dense
    drift-kinetic Hermite-Laguerre moment operator (Frei, Ernst & Ricci 2022)
    that replaces the built-in term. ``density``/``mass``/``temperature`` are
    the per-species normalizations (length ``n_species``).

    The assembled matrices carry only the dimensionless pair scaling
    ``n_b / (sqrt(m_a) T_a**1.5)``; the common collisionality prefactor is the
    caller's responsibility, so ``nu`` is applied here. It is the same ``nu``
    that sets the strength of the built-in Lenard-Bernstein term, which keeps
    every model on one collisionality axis.
    """

    from gkx.operators.linear.collisions import (
        DriftKineticMomentCollisionOperator,
        _assemble_drift_kinetic_sugama_matrix,
        drift_kinetic_improved_sugama_pair_matrices,
        drift_kinetic_sugama_pair_matrices,
    )

    key = name.strip().lower()
    if key in BUILTIN_COLLISION_OPERATORS:
        return None

    collisionality = jnp.asarray(nu)
    if collisionality.ndim > 1:
        raise ValueError("nu must be a scalar or a per-species vector")
    if collisionality.ndim == 1 and key in ("sugama", "improved_sugama"):
        # The test part of C_ab sits in the diagonal block, so the prefactor
        # must enter each ordered pair before that sum. Species prefactors
        # nu_a ~ Z_a^4 give the pair prefactor Z_a^2 Z_b^2 = sqrt(nu_a nu_b),
        # symmetric, so (a, b) and (b, a) still exchange momentum and energy.
        pairs = (
            drift_kinetic_sugama_pair_matrices
            if key == "sugama"
            else drift_kinetic_improved_sugama_pair_matrices
        )
        return DriftKineticMomentCollisionOperator(
            _assemble_drift_kinetic_sugama_matrix(
                density,
                mass,
                temperature,
                pairs,
                pair_scale=jnp.sqrt(collisionality[:, None] * collisionality),
            )
        )
    if collisionality.ndim == 1:
        # Single-species models: the vector is that species' prefactor.
        collisionality = collisionality.reshape((-1, 1, 1, 1))

    if key == "sugama":
        operator = DriftKineticMomentCollisionOperator.from_species(
            density, mass, temperature
        )
    elif key == "improved_sugama":
        operator = DriftKineticMomentCollisionOperator.from_improved_species(
            density, mass, temperature
        )
    elif key == "coulomb_finite_kperp":
        finite = build_finite_wavelength_coulomb_operator(
            density, mass, temperature, moments
        )
        scale = (
            jnp.reshape(jnp.asarray(nu), ())
            if jnp.asarray(nu).ndim == 0
            else (jnp.asarray(nu).reshape((-1, 1)))
        )
        return dataclasses.replace(finite, pair_frequency=finite.pair_frequency * scale)
    else:
        raise ValueError(
            f"collision_operator must be one of {COLLISION_OPERATOR_NAMES}"
        )
    return DriftKineticMomentCollisionOperator(operator.matrix * collisionality)


# ---- Linearized Landau operator at any Hermite-Laguerre resolution ----
#
# The like- and unlike-species linearized Landau operator is rotationally
# invariant, so in the orthonormal Legendre-Sonine basis
# S_lk = N_lk x^l L_k^{(l+1/2)}(x^2) P_l(xi) it is block diagonal in l. Each block
# is a pair of one-dimensional radial integrals (test part: weak form of the
# Rosenbluth diffusion; field part: multipole Rosenbluth potentials), evaluated
# by composite Gauss quadrature. An exact orthogonal map takes the Hermite-
# Laguerre box (p < Nm, j < Nl) into Legendre-Sonine functions of total degree
# l + 2k <= (Nm - 1) + 2 (Nl - 1). Velocities are in units of
# v_ta = sqrt(2 T_a / m_a); a unit pair frequency gives
# nu_D^ab(v) = [erf(x_b) - G(x_b)] / x_a^3, the GS2 ``vnewk`` convention.


def _hermite_values(x: np.ndarray, count: int) -> np.ndarray:
    values = np.zeros((count,) + x.shape)
    values[0] = 1.0
    if count > 1:
        values[1] = np.sqrt(2.0) * x
    for p in range(1, count - 1):
        values[p + 1] = (
            np.sqrt(2.0 / (p + 1)) * x * values[p]
            - np.sqrt(p / (p + 1)) * values[p - 1]
        )
    return values


def _laguerre_values(y: np.ndarray, count: int, alpha: float = 0.0) -> np.ndarray:
    values = np.zeros((count,) + y.shape)
    values[0] = 1.0
    if count > 1:
        values[1] = 1.0 + alpha - y
    for k in range(1, count - 1):
        values[k + 1] = (
            (2 * k + 1 + alpha - y) * values[k] - (k + alpha) * values[k - 1]
        ) / (k + 1)
    return values


def _sonine_norm(l: int, k: np.ndarray) -> np.ndarray:  # noqa: E741
    from scipy.special import gammaln

    return np.exp(
        0.5
        * (
            0.5 * np.log(np.pi)
            + np.log(2 * l + 1)
            + gammaln(k + 1)
            - np.log(2.0)
            - gammaln(k + l + 1.5)
        )
    )


def _hermite_laguerre_to_legendre_sonine(nl: int, nm: int) -> tuple[np.ndarray, list]:
    from scipy.special import roots_genlaguerre, roots_hermite

    degree = (nm - 1) + 2 * (nl - 1)
    t, wt = roots_hermite(degree + 2)
    y, wy = roots_genlaguerre(degree + 2, 0.0)
    par, perp = np.meshgrid(t, y, indexing="ij")
    weight = np.outer(wt, wy) / np.sqrt(np.pi)
    hl = (
        _hermite_values(par, nm)[:, None] * _laguerre_values(perp, nl)[None, :]
    ).reshape((nm * nl,) + par.shape)
    r2 = par**2 + perp
    solid = np.zeros((degree + 1,) + par.shape)  # x^l P_l(xi)
    solid[0] = 1.0
    if degree:
        solid[1] = par
    for l in range(1, degree):  # noqa: E741
        solid[l + 1] = ((2 * l + 1) * par * solid[l] - l * r2 * solid[l - 1]) / (l + 1)
    labels, sonine = [], []
    for l in range(degree + 1):  # noqa: E741
        count = (degree - l) // 2 + 1
        norm = _sonine_norm(l, np.arange(count))[:, None, None]
        sonine.append(norm * solid[l] * _laguerre_values(r2, count, l + 0.5))
        labels += [l] * count
    transform = np.einsum("aij,bij,ij->ab", hl, np.concatenate(sonine), weight)
    return transform, labels


def _composite_gauss(xmax: float, scale: float, fine: float):
    from scipy.special import roots_legendre

    edges = np.unique(
        np.concatenate(
            [
                [0.0],
                np.geomspace(1.0e-3 * fine, scale, 20)[:-1],
                np.arange(scale, xmax + scale, scale),
            ]
        )
    )
    nodes, weights = roots_legendre(16)
    lo, hi = edges[:-1, None], edges[1:, None]
    return ((hi - lo) / 2 * nodes + (hi + lo) / 2).ravel(), (
        (hi - lo) / 2 * weights
    ).ravel()


def _sonine_radial(x: np.ndarray, l: int, count: int, scale: float = 1.0):  # noqa: E741
    """Return N_lk u^l L_k(u^2) e^{-u^2/2} and its x-derivative, u = x/scale."""

    u = x / scale
    with np.errstate(divide="ignore"):
        base = np.exp(l * np.log(u) - 0.5 * u**2) if l else np.exp(-0.5 * u**2)
    lag = _laguerre_values(u**2, count + 1, l + 0.5)[:count]
    dlag = np.zeros_like(lag)
    dlag[1:] = -_laguerre_values(u**2, count, l + 1.5)[: count - 1]
    norm = _sonine_norm(l, np.arange(count))[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        l_over_u = np.where(u > 0, l / u, 0.0)
    value = norm * base * lag
    return value, norm * base * ((l_over_u - u) * lag + 2 * u * dlag) / scale


@lru_cache(maxsize=32)
def linearized_landau_pair_matrices(
    nl: int, nm: int, mass_ratio: float = 1.0, temperature_ratio: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return the unit-frequency test and field matrices of one ordered pair.

    ``C_ab[f_a, f_b] = C(f_a, F_b) + C(F_a, f_b)`` with ``mass_ratio = m_a/m_b``
    and ``temperature_ratio = T_a/T_b``, linearized about Maxwellians. Rows are
    species-``a`` moments, columns the moments of the perturbed species (``a``
    for the test part, ``b`` for the field part), in the runtime's Hermite-major
    ``p*Nl + j`` order and signed-Laguerre convention. Particles, momentum and
    energy are conserved exactly by the pair sum; the like-species matrix is
    symmetric negative semidefinite (H-theorem). With ``T_a != T_b`` the
    operator still conserves but is no longer self-adjoint.
    """

    from scipy.special import erf, roots_legendre

    mu, tau = float(mass_ratio), float(temperature_ratio)
    if mu <= 0.0 or tau <= 0.0:
        raise ValueError("mass_ratio and temperature_ratio must be positive")
    transform, labels = _hermite_laguerre_to_legendre_sonine(nl, nm)
    degree = (nm - 1) + 2 * (nl - 1)
    extent = np.sqrt(4.0 * degree + 120.0)
    s = np.sqrt(mu / tau)  # v_tb / v_ta
    x, w = _composite_gauss(extent, 1.0, min(1.0, s))
    y = x / s
    chandrasekhar = (erf(y) - 2 * y * np.exp(-y * y) / np.sqrt(np.pi)) / (2 * y * y)
    d_par, d_perp = 2 * chandrasekhar / x, (erf(y) - chandrasekhar) / x
    drift = 2 * chandrasekhar * (1.0 - tau)  # vanishes at equal temperatures
    t, wt = roots_legendre(48)
    t, wt = (t + 1) / 2, wt / 2
    v, wv = _composite_gauss(s * extent, s, s)
    moment_l = np.asarray(labels)
    blocks = [np.zeros((moment_l.size, moment_l.size)) for _ in range(2)]
    for l in range(degree + 1):  # noqa: E741
        count = (degree - l) // 2 + 1
        c = 4 * np.pi / (2 * l + 1)
        r, dr = _sonine_radial(x, l, count)
        dpsi = dr + x * r
        outer = c * w * x**2 * np.pi**-1.5
        test = -np.einsum("x,kx,qx->kq", outer * 0.5 * d_par, dpsi, dpsi)
        test -= np.einsum(
            "x,kx,qx->kq", outer * 0.5 * d_perp * l * (l + 1) / x**2, r, r
        )
        test += np.einsum("x,kx,qx->kq", outer * drift, dpsi, r)

        def perturbed(points: np.ndarray) -> np.ndarray:
            radial = _sonine_radial(points.ravel(), l, count, s)[0]
            gauss = np.exp(-0.5 * (points.ravel() / s) ** 2) * np.pi**-1.5 / s**3
            return (radial * gauss).reshape((count,) + points.shape)

        # Rosenbluth multipoles as integrals of bounded kernel ratios.
        span = np.minimum(x, s * extent)[:, None]
        inner = span * t[None, :]
        below = perturbed(inner) * ((inner / x[:, None]) ** (l + 1) * inner * span)
        t1 = np.einsum("qxt,t->qx", below, wt)
        t2 = np.einsum("qxt,xt,t->qx", below, inner**2, wt)
        inner = x[:, None] + v[None, :]
        above = perturbed(inner) * ((x[:, None] / inner) ** l * inner)
        t3 = np.einsum("qxv,v->qx", above, wv)
        t4 = np.einsum("qxv,xv,v->qx", above, inner**2, wv)
        odd = 1.0 / (2 * l - 1)
        even = (l + 1) * (l + 2) / (2 * l + 3)
        potentials = c * (
            -2 * (t1 + t3)
            + 2 * (mu - 1) * (l * t3 - (l + 1) * t1)
            + 2 * (even * (t2 + x**2 * t3) - l * (l - 1) * odd * (x**2 * t1 + t4))
        )
        source = 4 * np.pi * mu * perturbed(x) + potentials
        field = np.einsum("x,kx,qx->kq", outer * np.exp(-0.5 * x**2), r, source)
        index = np.flatnonzero(moment_l == l)
        blocks[0][np.ix_(index, index)] = test
        blocks[1][np.ix_(index, index)] = field
    sign = (-1.0) ** (np.arange(nl * nm) % nl)
    convention = transform * sign[:, None]
    return tuple(convention @ block @ convention.T for block in blocks)  # type: ignore[return-value]


def linearized_landau_species_matrices(
    nl: int, nm: int, mass: Any, temperature: Any
) -> np.ndarray:
    """Stack unit-frequency ``(test, field)`` pair matrices as ``(2, s, s, n, n)``."""

    masses = np.atleast_1d(np.asarray(mass, dtype=float))
    temps = np.broadcast_to(
        np.atleast_1d(np.asarray(temperature, dtype=float)), masses.shape
    )
    return np.stack(
        [
            np.stack(
                [
                    np.stack(
                        linearized_landau_pair_matrices(
                            nl, nm, float(ma / mb), float(ta / tb)
                        )
                    )
                    for mb, tb in zip(masses, temps)
                ],
                axis=1,
            )
            for ma, ta in zip(masses, temps)
        ],
        axis=1,
    )
