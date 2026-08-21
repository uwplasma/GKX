from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from gkx.operators.nonlinear.rhs import (
    linear_rhs_jit_for_terms_impl,
    nonlinear_em_term_cached_impl,
    nonlinear_rhs_cached_impl,
)
from gkx.terms.config import FieldState, TermConfig


def _minimal_cache() -> SimpleNamespace:
    return SimpleNamespace(
        Jl=jnp.ones((1, 1, 1, 1, 1), dtype=jnp.float32),
        JlB=jnp.ones((1, 1, 1, 1, 1), dtype=jnp.float32),
        sqrt_m=jnp.ones((1, 1, 1, 1, 1, 1), dtype=jnp.float32),
        sqrt_m_p1=jnp.ones((1, 1, 1, 1, 1, 1), dtype=jnp.float32),
        kx_grid=jnp.zeros((1, 1), dtype=jnp.float32),
        ky_grid=jnp.zeros((1, 1), dtype=jnp.float32),
        dealias_mask=jnp.ones((1, 1), dtype=bool),
        kxfac=1.0,
        laguerre_to_grid=None,
        laguerre_to_spectral=None,
        laguerre_roots=None,
        laguerre_j0=None,
        laguerre_j1_over_alpha=None,
        b=None,
    )


def test_linear_rhs_jit_for_terms_impl_selects_narrowest_route() -> None:
    electrostatic = object()
    full = object()

    assert (
        linear_rhs_jit_for_terms_impl(
            TermConfig(apar=0.0, bpar=0.0),
            electrostatic_rhs_fn=electrostatic,  # type: ignore[arg-type]
            full_rhs_fn=full,  # type: ignore[arg-type]
            is_static_zero_fn=lambda value: float(value) == 0.0,
        )
        is electrostatic
    )
    assert (
        linear_rhs_jit_for_terms_impl(
            TermConfig(apar=1.0, bpar=0.0),
            electrostatic_rhs_fn=electrostatic,  # type: ignore[arg-type]
            full_rhs_fn=full,  # type: ignore[arg-type]
            is_static_zero_fn=lambda value: float(value) == 0.0,
        )
        is full
    )


def test_nonlinear_rhs_cached_impl_skips_bracket_when_disabled() -> None:
    G = jnp.ones((1, 1, 1, 1, 1, 2), dtype=jnp.complex64)
    fields = FieldState(phi=jnp.zeros((1, 1, 2), dtype=jnp.complex64))
    calls = {"linear": 0, "nonlinear": 0}

    def linear_rhs(G_in, *_args, **_kwargs):
        calls["linear"] += 1
        return 2.0 * G_in, fields

    def nonlinear_contribution(*_args, **_kwargs):
        calls["nonlinear"] += 1
        raise AssertionError("nonlinear contribution should not run")

    rhs, rhs_fields = nonlinear_rhs_cached_impl(
        G,
        _minimal_cache(),
        SimpleNamespace(tz=jnp.asarray([1.0]), vth=jnp.asarray([1.0])),
        TermConfig(nonlinear=0.0, apar=0.0, bpar=0.0),
        electrostatic_rhs_fn=linear_rhs,
        full_rhs_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError()),
        nonlinear_contribution_fn=nonlinear_contribution,
    )

    assert calls == {"linear": 1, "nonlinear": 0}
    np.testing.assert_allclose(np.asarray(rhs), 2.0)
    assert rhs_fields is fields


def test_nonlinear_rhs_cached_impl_accepts_custom_collision_operator() -> None:
    G = jnp.ones((1, 1, 1, 1, 1, 2), dtype=jnp.complex64)
    fields = FieldState(phi=2.0 * jnp.ones((1, 1, 2), dtype=jnp.complex64))
    seen: dict[str, object] = {}

    def linear_rhs(G_in, _cache, _params, terms, **_kwargs):
        seen["linear_collision_weight"] = terms.collisions
        seen["hypercollision_weight"] = terms.hypercollisions
        return 2.0 * G_in, fields

    class DragCollision:
        def apply(self, context):
            seen["cache"] = context.cache
            seen["parameters"] = context.parameters
            seen["fields"] = context.fields
            seen["hamiltonian"] = context.hamiltonian
            return -3.0 * context.distribution

    cache = _minimal_cache()
    parameters = SimpleNamespace(tz=jnp.asarray([1.0]), vth=jnp.asarray([1.0]))
    rhs, _ = nonlinear_rhs_cached_impl(
        G,
        cache,
        parameters,
        TermConfig(collisions=0.25, hypercollisions=0.75, nonlinear=0.0),
        collision_operator=DragCollision(),
        electrostatic_rhs_fn=linear_rhs,
        full_rhs_fn=linear_rhs,
    )

    assert seen["linear_collision_weight"] == 0.0
    assert seen["hypercollision_weight"] == 0.75
    assert seen["cache"] is cache
    assert seen["parameters"] is parameters
    assert seen["fields"] is fields
    np.testing.assert_allclose(np.asarray(seen["hamiltonian"]), 3.0)
    np.testing.assert_allclose(np.asarray(rhs), 1.25)


def test_nonlinear_rhs_cached_impl_rejects_invalid_collision_shape() -> None:
    G = jnp.ones((1, 1, 1, 1, 1, 2), dtype=jnp.complex64)
    fields = FieldState(phi=jnp.zeros((1, 1, 2), dtype=jnp.complex64))

    class InvalidCollision:
        def apply(self, context):
            return context.distribution[..., 0]

    with pytest.raises(ValueError, match="same state shape"):
        nonlinear_rhs_cached_impl(
            G,
            _minimal_cache(),
            SimpleNamespace(tz=jnp.asarray([1.0]), vth=jnp.asarray([1.0])),
            TermConfig(collisions=1.0, nonlinear=0.0),
            collision_operator=InvalidCollision(),
            electrostatic_rhs_fn=lambda state, *_args, **_kwargs: (
                jnp.zeros_like(state),
                fields,
            ),
            full_rhs_fn=lambda state, *_args, **_kwargs: (
                jnp.zeros_like(state),
                fields,
            ),
        )


def test_nonlinear_rhs_cached_impl_forwards_physical_bracket_payload() -> None:
    G = jnp.ones((1, 1, 1, 1, 1, 2), dtype=jnp.complex64)
    phi = jnp.ones((1, 1, 2), dtype=jnp.complex64)
    fields = FieldState(phi=phi, apar=2.0 * phi, bpar=3.0 * phi)
    seen: dict[str, object] = {}

    def linear_rhs(G_in, *_args, **_kwargs):
        return jnp.zeros_like(G_in), fields

    def nonlinear_contribution(G_in, **kwargs):
        seen["shape"] = tuple(G_in.shape)
        seen["apar_is_none"] = kwargs["apar"] is None
        seen["bpar_is_none"] = kwargs["bpar"] is None
        seen["weight"] = float(kwargs["weight"])
        seen["apar_weight"] = kwargs["apar_weight"]
        seen["bpar_weight"] = kwargs["bpar_weight"]
        seen["compressed_real_fft"] = kwargs["compressed_real_fft"]
        seen["laguerre_mode"] = kwargs["laguerre_mode"]
        return 4.0 * jnp.ones_like(G_in)

    rhs, _rhs_fields = nonlinear_rhs_cached_impl(
        G,
        _minimal_cache(),
        SimpleNamespace(tz=jnp.asarray([1.0]), vth=jnp.asarray([1.0])),
        TermConfig(nonlinear=0.25, apar=1.0, bpar=0.0),
        compressed_real_fft=False,
        laguerre_mode="spectral",
        electrostatic_rhs_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError()
        ),
        full_rhs_fn=linear_rhs,
        nonlinear_contribution_fn=nonlinear_contribution,
    )

    np.testing.assert_allclose(np.asarray(rhs), 4.0)
    assert seen == {
        "shape": tuple(G.shape),
        "apar_is_none": False,
        "bpar_is_none": True,
        "weight": 0.25,
        "apar_weight": 1.0,
        "bpar_weight": 0.0,
        "compressed_real_fft": False,
        "laguerre_mode": "spectral",
    }


def test_nonlinear_em_term_cached_impl_reuses_imex_bracket_payload() -> None:
    G = jnp.ones((1, 1, 1, 1, 1, 2), dtype=jnp.complex64)
    phi = jnp.ones((1, 1, 2), dtype=jnp.complex64)
    fields = FieldState(phi=phi, apar=2.0 * phi, bpar=3.0 * phi)
    seen: dict[str, object] = {"fields": 0, "nonlinear": 0}

    def fields_fn(G_in, *_args, **kwargs):
        seen["fields"] = int(seen["fields"]) + 1
        seen["external_phi"] = kwargs["external_phi"]
        np.testing.assert_allclose(np.asarray(G_in), np.asarray(G))
        return fields

    def nonlinear_contribution(G_in, **kwargs):
        seen["nonlinear"] = int(seen["nonlinear"]) + 1
        seen["shape"] = tuple(G_in.shape)
        seen["apar_is_none"] = kwargs["apar"] is None
        seen["bpar_is_none"] = kwargs["bpar"] is None
        seen["weight"] = float(kwargs["weight"])
        seen["apar_weight"] = kwargs["apar_weight"]
        seen["bpar_weight"] = kwargs["bpar_weight"]
        seen["compressed_real_fft"] = kwargs["compressed_real_fft"]
        seen["laguerre_mode"] = kwargs["laguerre_mode"]
        return 5.0 * jnp.ones_like(G_in)

    zero = nonlinear_em_term_cached_impl(
        G,
        _minimal_cache(),
        SimpleNamespace(tz=jnp.asarray([1.0]), vth=jnp.asarray([1.0])),
        TermConfig(nonlinear=0.0),
        fields_fn=fields_fn,
        nonlinear_contribution_fn=nonlinear_contribution,
    )
    np.testing.assert_allclose(np.asarray(zero), 0.0)
    assert seen == {"fields": 0, "nonlinear": 0}

    out = nonlinear_em_term_cached_impl(
        G,
        _minimal_cache(),
        SimpleNamespace(tz=jnp.asarray([1.0]), vth=jnp.asarray([1.0])),
        TermConfig(nonlinear=0.5, apar=1.0, bpar=1.0),
        external_phi=3.0,
        compressed_real_fft=False,
        laguerre_mode="spectral",
        fields_fn=fields_fn,
        nonlinear_contribution_fn=nonlinear_contribution,
    )

    np.testing.assert_allclose(np.asarray(out), 5.0)
    assert seen == {
        "fields": 1,
        "nonlinear": 1,
        "external_phi": 3.0,
        "shape": tuple(G.shape),
        "apar_is_none": False,
        "bpar_is_none": False,
        "weight": 0.5,
        "apar_weight": 1.0,
        "bpar_weight": 1.0,
        "compressed_real_fft": False,
        "laguerre_mode": "spectral",
    }
