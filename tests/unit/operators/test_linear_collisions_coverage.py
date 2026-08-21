"""Validation-branch and pytree coverage for linear collision kernels.

These tests exercise the provenance checks, operator-variant guards, and
edge-case validation in :mod:`gkx.operators.linear.collisions` that the
main kernel regression suite does not reach. Every check pins the exact
contract (checksum/shape provenance, moment-axis shapes, species axes, grid
monotonicity, or pytree round-trip identity) rather than mere execution.
"""

from __future__ import annotations

import hashlib
import io
import json
from importlib import resources
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gkx.operators.collision import CollisionContext
from gkx.operators.linear import collisions
from gkx.operators.linear.collisions import (
    DriftKineticMomentCollisionOperator,
    EqualSpeciesFiniteWavelengthCoulombOperator,
    FiniteWavelengthCoulombOperator,
    apply_collision_moment_matrix,
    apply_finite_wavelength_coulomb_moment_operator,
    apply_multispecies_collision_moment_matrix,
    drift_kinetic_sugama_pair_matrices,
    interpolate_collision_diagonal_table,
    interpolate_collision_pair_table,
)
from gkx.terms.config import FieldState


def _fake_resource_files(payload: bytes, metadata_text: str):
    """Return a stand-in for ``importlib.resources.files`` over fixed bytes."""

    class _Leaf:
        def __init__(self, *, data: bytes | None = None, text: str | None = None):
            self._data = data
            self._text = text

        def read_bytes(self) -> bytes:
            assert self._data is not None
            return self._data

        def read_text(self, encoding: str = "utf-8") -> str:
            assert self._text is not None
            return self._text

    class _DataRoot:
        def joinpath(self, name: str) -> _Leaf:
            if name.endswith(".npy"):
                return _Leaf(data=payload)
            return _Leaf(text=metadata_text)

    class _Package:
        def joinpath(self, name: str) -> _DataRoot:
            return _DataRoot()

    def _files(package: str) -> _Package:
        return _Package()

    return _files


def test_collision_matrix_bundle_rejects_corrupt_provenance(monkeypatch) -> None:
    """The cached bundle fails closed on checksum and shape mismatches."""

    real_payload = (
        resources.files("gkx")
        .joinpath("data")
        .joinpath(collisions._COLLISION_MATRIX_DATA)
        .read_bytes()
    )

    # Checksum branch: genuine coefficients, metadata advertising a wrong hash.
    corrupt_hash_metadata = json.dumps(
        {"sha256": "0" * 64, "shape": [3, 8, 8], "models": ["sugama"]}
    )
    monkeypatch.setattr(
        collisions.resources,
        "files",
        _fake_resource_files(real_payload, corrupt_hash_metadata),
    )
    collisions._collision_matrix_bundle.cache_clear()
    with pytest.raises(ValueError, match="checksum does not match metadata"):
        collisions._collision_matrix_bundle()

    # Shape branch: matching checksum but a payload whose array shape disagrees
    # with the declared (3, 8, 8) provenance.
    buffer = io.BytesIO()
    np.save(buffer, np.zeros((2, 2), dtype=np.float64))
    mismatched_payload = buffer.getvalue()
    honest_hash = hashlib.sha256(mismatched_payload).hexdigest()
    wrong_shape_metadata = json.dumps(
        {"sha256": honest_hash, "shape": [3, 8, 8], "models": ["sugama"]}
    )
    monkeypatch.setattr(
        collisions.resources,
        "files",
        _fake_resource_files(mismatched_payload, wrong_shape_metadata),
    )
    collisions._collision_matrix_bundle.cache_clear()
    with pytest.raises(ValueError, match="shape does not match metadata"):
        collisions._collision_matrix_bundle()

    # Drop the poisoned (empty) cache so downstream tests reload real data.
    collisions._collision_matrix_bundle.cache_clear()


def test_sugama_pair_matrices_reject_nonpositive_temperature_ratio() -> None:
    """A positive mass ratio still requires a positive temperature ratio."""

    # A valid pair returns finite, correctly shaped (8, 8) test/field blocks.
    test_matrix, field_matrix = drift_kinetic_sugama_pair_matrices(
        jnp.asarray(1.0), jnp.asarray(1.0)
    )
    assert test_matrix.shape == (8, 8)
    assert field_matrix.shape == (8, 8)
    assert bool(jnp.all(jnp.isfinite(test_matrix)))

    with pytest.raises(ValueError, match="temperature_ratio must be positive"):
        drift_kinetic_sugama_pair_matrices(jnp.asarray(1.0), jnp.asarray(0.0))
    with pytest.raises(ValueError, match="temperature_ratio must be positive"):
        drift_kinetic_sugama_pair_matrices(jnp.asarray(2.0), jnp.asarray(-0.5))


def _diagonal_coulomb_operator() -> EqualSpeciesFiniteWavelengthCoulombOperator:
    grid = jnp.asarray([0.0, 1.0, 2.0])
    matrix = (1.0 + 0.3 * grid)[:, None, None]
    vector = (0.2 - 0.04 * grid)[:, None]
    zero_matrix = jnp.zeros_like(matrix)
    zero_vector = jnp.zeros_like(vector)
    return EqualSpeciesFiniteWavelengthCoulombOperator(
        grid,
        jnp.asarray([[0.7]]),
        matrix,
        zero_matrix,
        vector,
        zero_vector,
        zero_vector,
        zero_vector,
    )


def test_collision_operators_round_trip_through_pytree() -> None:
    """Registered collision operators flatten and unflatten without data loss."""

    dense = DriftKineticMomentCollisionOperator(
        jnp.asarray(collisions.load_collision_moment_matrix("sugama"))
    )
    dense_leaves, dense_treedef = jax.tree_util.tree_flatten(dense)
    assert len(dense_leaves) == 1
    dense_rebuilt = jax.tree_util.tree_unflatten(dense_treedef, dense_leaves)
    assert isinstance(dense_rebuilt, DriftKineticMomentCollisionOperator)
    assert bool(jnp.array_equal(dense_rebuilt.matrix, dense.matrix))

    grid = jnp.asarray([0.0, 1.0, 2.0])
    finite = FiniteWavelengthCoulombOperator(
        grid,
        jnp.ones((1, 1)),
        jnp.zeros((1, 1, 3, 3, 1, 1)),
        jnp.zeros((1, 1, 3, 3, 1, 1)),
        jnp.zeros((1, 1, 3, 3, 1)),
        jnp.zeros((1, 1, 3, 3, 1)),
        jnp.zeros((1, 1, 3, 3, 1)),
        jnp.zeros((1, 1, 3, 3, 1)),
    )
    finite_leaves, finite_treedef = jax.tree_util.tree_flatten(finite)
    finite_rebuilt = jax.tree_util.tree_unflatten(finite_treedef, finite_leaves)
    assert isinstance(finite_rebuilt, FiniteWavelengthCoulombOperator)
    assert bool(
        jnp.array_equal(
            finite_rebuilt.bessel_argument_grid, finite.bessel_argument_grid
        )
    )
    assert bool(jnp.array_equal(finite_rebuilt.test_table, finite.test_table))

    diagonal = _diagonal_coulomb_operator()
    diag_leaves, diag_treedef = jax.tree_util.tree_flatten(diagonal)
    diag_rebuilt = jax.tree_util.tree_unflatten(diag_treedef, diag_leaves)
    assert isinstance(diag_rebuilt, EqualSpeciesFiniteWavelengthCoulombOperator)
    assert bool(jnp.array_equal(diag_rebuilt.test_table, diagonal.test_table))
    assert bool(jnp.array_equal(diag_rebuilt.pair_frequency, diagonal.pair_frequency))


def test_equal_species_coulomb_apply_requires_single_species_bessel_axis() -> None:
    """The compact like-species path rejects a multi-species Bessel argument."""

    diagonal = _diagonal_coulomb_operator()
    single_species_state = jnp.ones((1, 1, 1, 1, 1), dtype=jnp.complex128)
    context = CollisionContext(
        distribution=single_species_state,
        hamiltonian=single_species_state,
        fields=FieldState(phi=jnp.zeros((1, 1, 1)), apar=None, bpar=None),
        cache=SimpleNamespace(b=jnp.zeros((2, 1, 1, 1))),
        parameters=SimpleNamespace(tz=jnp.ones(1)),
    )
    with pytest.raises(ValueError, match="Bessel argument must have one species"):
        diagonal.apply(context)


def test_interpolate_collision_diagonal_table_validates_grid_and_table() -> None:
    """The diagonal interpolator guards grid rank, table rank, and monotonicity."""

    target = jnp.asarray(0.5)
    with pytest.raises(ValueError, match="at least two points"):
        interpolate_collision_diagonal_table(
            jnp.asarray([0.0]), jnp.ones((1, 3)), target
        )
    with pytest.raises(ValueError, match="one vector or two matrix axes"):
        interpolate_collision_diagonal_table(
            jnp.asarray([0.0, 1.0]), jnp.ones(4), target
        )
    with pytest.raises(ValueError, match="axis must match the grid"):
        interpolate_collision_diagonal_table(
            jnp.asarray([0.0, 1.0]), jnp.ones((3, 4)), target
        )
    with pytest.raises(ValueError, match="matrices must be square"):
        interpolate_collision_diagonal_table(
            jnp.asarray([0.0, 1.0]), jnp.ones((2, 3, 4)), target
        )
    with pytest.raises(ValueError, match="finite and strictly increasing"):
        interpolate_collision_diagonal_table(
            jnp.asarray([1.0, 0.0]), jnp.ones((2, 3)), target
        )


def test_interpolate_collision_pair_table_validates_grid_and_table() -> None:
    """The bilinear pair interpolator guards grid, table rank, and squareness."""

    species_target = jnp.ones((2, 1))
    with pytest.raises(ValueError, match="at least two points"):
        interpolate_collision_pair_table(
            jnp.asarray([0.0]), jnp.ones((1, 1, 1, 1, 2)), jnp.ones((1,))
        )
    with pytest.raises(ValueError, match="one vector or two matrix coefficient"):
        interpolate_collision_pair_table(
            jnp.asarray([0.0, 1.0]), jnp.ones((2, 2, 2, 2)), species_target
        )
    with pytest.raises(ValueError, match="kperp axes must match the grid"):
        interpolate_collision_pair_table(
            jnp.asarray([0.0, 1.0]), jnp.ones((2, 2, 3, 3, 2)), species_target
        )
    with pytest.raises(ValueError, match="pair matrices must be square"):
        interpolate_collision_pair_table(
            jnp.asarray([0.0, 1.0]),
            jnp.ones((2, 2, 2, 2, 3, 4)),
            species_target,
        )
    with pytest.raises(ValueError, match="finite and strictly increasing"):
        interpolate_collision_pair_table(
            jnp.asarray([1.0, 0.0]), jnp.ones((2, 2, 2, 2, 2)), species_target
        )


def test_dense_collision_apply_requires_five_or_six_dimensions() -> None:
    """Both dense apply kernels reject states outside the (5, 6)-rank contract."""

    rank_four_state = jnp.ones((2, 4, 1, 1))
    with pytest.raises(ValueError, match="five or six dimensions"):
        apply_collision_moment_matrix(rank_four_state, jnp.eye(4), nu=jnp.asarray(1.0))
    with pytest.raises(ValueError, match="five or six dimensions"):
        apply_multispecies_collision_moment_matrix(
            rank_four_state, jnp.zeros((2, 2, 4, 4))
        )


def _valid_coulomb_arguments() -> dict:
    ns, nl, nm = 1, 1, 2
    mode_count = nl * nm
    spatial_shape = (1, 1, 2)
    state = (
        jnp.arange(ns * nl * nm * 2, dtype=jnp.float64).reshape(
            (ns, nl, nm) + spatial_shape
        )
        + 0.2j
    )
    matrix = 0.01 * jnp.ones((ns, ns, mode_count, mode_count))
    vector = 0.01 * jnp.ones((ns, ns, mode_count))
    return {
        "distribution": state,
        "test_matrix": matrix,
        "field_matrix": -0.6 * matrix,
        "test_phi1": vector,
        "field_phi1": -0.5 * vector,
        "test_phi2": 0.2 * vector,
        "field_phi2": -0.3 * vector,
        "phi": jnp.asarray([[[0.3, -0.2]]]),
        "pair_frequency": jnp.asarray([[0.7]]),
        "charge_over_temperature": jnp.asarray([1.3]),
    }


def _call_coulomb(arguments: dict) -> jnp.ndarray:
    return apply_finite_wavelength_coulomb_moment_operator(
        arguments["distribution"],
        arguments["test_matrix"],
        arguments["field_matrix"],
        arguments["test_phi1"],
        arguments["field_phi1"],
        arguments["test_phi2"],
        arguments["field_phi2"],
        phi=arguments["phi"],
        pair_frequency=arguments["pair_frequency"],
        charge_over_temperature=arguments["charge_over_temperature"],
    )


def test_finite_wavelength_coulomb_operator_validates_shapes() -> None:
    """The runtime Coulomb apply enforces every state, matrix, and field shape."""

    baseline = _valid_coulomb_arguments()
    result = _call_coulomb(baseline)
    assert result.shape == baseline["distribution"].shape
    assert bool(jnp.all(jnp.isfinite(result)))

    with pytest.raises(ValueError, match="five or six dimensions"):
        _call_coulomb({**baseline, "distribution": jnp.ones((1, 2, 1, 1))})
    with pytest.raises(ValueError, match="test/field matrices must have"):
        _call_coulomb({**baseline, "test_matrix": jnp.zeros((1, 1, 3, 3))})
    with pytest.raises(ValueError, match="polarization vectors must have"):
        _call_coulomb({**baseline, "test_phi1": jnp.zeros((1, 1, 3))})
    with pytest.raises(ValueError, match="charge_over_temperature must have length"):
        _call_coulomb({**baseline, "charge_over_temperature": jnp.ones(2)})
    with pytest.raises(ValueError, match="phi must have spatial shape"):
        _call_coulomb({**baseline, "phi": jnp.zeros((1, 1, 3))})
