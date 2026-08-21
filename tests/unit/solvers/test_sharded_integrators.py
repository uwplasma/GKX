from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from gkx.parallel.integrators import (
    integrate_linear_sharded,
    integrate_nonlinear_sharded,
)
from gkx.parallel import integrators as integrator_module
from gkx.terms.config import FieldState


def test_integrate_linear_sharded_rejects_nonpositive_steps() -> None:
    with pytest.raises(ValueError):
        integrate_linear_sharded(
            jnp.zeros((1, 1, 1, 1, 1), dtype=jnp.complex64), None, None, dt=0.1, steps=0
        )


def test_integrate_linear_sharded_runs_with_mocked_pjit(monkeypatch) -> None:
    calls = {"rhs": 0, "shard": 0, "put": 0}

    def fake_rhs(G, cache, params, terms=None, dt=None):
        calls["rhs"] += 1
        return jnp.ones_like(G), None

    monkeypatch.setattr("gkx.parallel.integrators.linear_rhs_cached", fake_rhs)
    monkeypatch.setattr("gkx.parallel.integrators.pjit", lambda fn, **kwargs: fn)
    monkeypatch.setattr(
        "gkx.parallel.integrators.jax.lax.with_sharding_constraint",
        lambda state, sharding: calls.__setitem__("shard", calls["shard"] + 1) or state,
    )
    monkeypatch.setattr(
        "gkx.parallel.integrators.jax.device_put",
        lambda state, sharding: calls.__setitem__("put", calls["put"] + 1) or state,
    )

    G0 = jnp.zeros((1, 1, 1, 1, 1), dtype=jnp.complex64)
    out = integrate_linear_sharded(
        G0, SimpleNamespace(), SimpleNamespace(), dt=0.5, steps=2, state_sharding="mesh"
    )
    assert out.shape == G0.shape
    assert calls["rhs"] >= 2
    assert calls["put"] == 1
    assert calls["shard"] >= 3


def test_integrate_linear_sharded_no_sharding_path(monkeypatch) -> None:
    calls = {"rhs": 0}

    def fake_rhs(G, cache, params, terms=None, dt=None):
        calls["rhs"] += 1
        return jnp.ones_like(G), None

    monkeypatch.setattr("gkx.parallel.integrators.linear_rhs_cached", fake_rhs)
    monkeypatch.setattr("gkx.parallel.integrators.pjit", lambda fn, **kwargs: fn)

    G0 = jnp.zeros((1, 1, 1, 1, 1), dtype=jnp.complex64)
    out = integrate_linear_sharded(
        G0, SimpleNamespace(), SimpleNamespace(), dt=0.25, steps=2
    )

    assert out.shape == G0.shape
    assert calls["rhs"] == 2
    assert jnp.allclose(out, 0.5)


def _cache_stub() -> SimpleNamespace:
    return SimpleNamespace(ky=jnp.asarray([0.0]), kx=jnp.asarray([0.0]))


def test_integrate_nonlinear_sharded_rejects_bad_options() -> None:
    G0 = jnp.zeros((1, 1, 1, 1, 1), dtype=jnp.complex64)
    with pytest.raises(ValueError, match="steps"):
        integrate_nonlinear_sharded(
            G0, _cache_stub(), SimpleNamespace(), dt=0.1, steps=0
        )
    with pytest.raises(ValueError, match="method"):
        integrate_nonlinear_sharded(
            G0, _cache_stub(), SimpleNamespace(), dt=0.1, steps=1, method="bad"
        )


def test_integrate_nonlinear_sharded_runs_with_mocked_pjit(monkeypatch) -> None:
    calls = {"rhs": 0, "shard": 0, "put": 0}

    def fake_rhs(
        G,
        cache,
        params,
        terms=None,
        *,
        compressed_real_fft=True,
        laguerre_mode="grid",
        external_phi=None,
    ):
        calls["rhs"] += 1
        return jnp.ones_like(G), FieldState(phi=jnp.ones((1, 1, 1), dtype=G.dtype))

    monkeypatch.setattr(
        "gkx.parallel.integrators.nonlinear_rhs_cached", fake_rhs
    )
    monkeypatch.setattr("gkx.parallel.integrators.pjit", lambda fn, **kwargs: fn)
    monkeypatch.setattr(
        "gkx.parallel.integrators.jax.lax.with_sharding_constraint",
        lambda state, sharding: calls.__setitem__("shard", calls["shard"] + 1) or state,
    )
    monkeypatch.setattr(
        "gkx.parallel.integrators.jax.device_put",
        lambda state, sharding: calls.__setitem__("put", calls["put"] + 1) or state,
    )

    G0 = jnp.zeros((1, 1, 1, 1, 1), dtype=jnp.complex64)
    G_final, fields_t = integrate_nonlinear_sharded(
        G0,
        _cache_stub(),
        SimpleNamespace(),
        dt=0.5,
        steps=2,
        method="rk2",
        state_sharding="mesh",
    )

    assert G_final.shape == G0.shape
    assert fields_t.phi.shape[0] == 2
    assert jnp.allclose(G_final, 1.0)
    assert calls["rhs"] >= 2
    assert calls["put"] == 1
    assert calls["shard"] >= 3


@pytest.mark.parametrize(
    "method", ["euler", "rk2", "rk3", "rk3_heun", "rk3_classic", "rk4", "sspx3"]
)
def test_integrate_nonlinear_sharded_explicit_methods_constant_rhs(
    monkeypatch, method: str
) -> None:
    """All explicit nonlinear sharded methods should preserve a constant RHS update."""

    calls = {"rhs": 0}

    def fake_rhs(
        G,
        cache,
        params,
        terms=None,
        *,
        compressed_real_fft=True,
        laguerre_mode="grid",
        external_phi=None,
    ):
        calls["rhs"] += 1
        return jnp.ones_like(G), FieldState(phi=jnp.ones((1, 1, 1), dtype=G.dtype))

    monkeypatch.setattr(
        "gkx.parallel.integrators.nonlinear_rhs_cached", fake_rhs
    )
    monkeypatch.setattr("gkx.parallel.integrators.pjit", lambda fn, **kwargs: fn)

    G0 = jnp.zeros((1, 1, 1, 1, 1), dtype=jnp.complex64)
    G_final, fields_t = integrate_nonlinear_sharded(
        G0,
        _cache_stub(),
        SimpleNamespace(),
        dt=0.1,
        steps=1,
        method=method,
        compressed_real_fft=False,
    )

    assert G_final.shape == G0.shape
    assert fields_t.phi.shape[0] == 1
    assert jnp.allclose(G_final, 0.1)
    assert calls["rhs"] >= 2


def test_integrate_nonlinear_sharded_final_only_path(monkeypatch) -> None:
    calls = {"rhs": 0}

    def fake_rhs(
        G,
        cache,
        params,
        terms=None,
        *,
        compressed_real_fft=True,
        laguerre_mode="grid",
        external_phi=None,
    ):
        calls["rhs"] += 1
        return 2.0 * jnp.ones_like(G), FieldState(
            phi=jnp.ones((1, 1, 1), dtype=G.dtype)
        )

    monkeypatch.setattr(
        "gkx.parallel.integrators.nonlinear_rhs_cached", fake_rhs
    )
    monkeypatch.setattr("gkx.parallel.integrators.pjit", lambda fn, **kwargs: fn)

    G0 = jnp.zeros((1, 1, 1, 1, 1), dtype=jnp.complex64)
    out = integrate_nonlinear_sharded(
        G0,
        _cache_stub(),
        SimpleNamespace(),
        dt=0.25,
        steps=2,
        method="euler",
        return_fields=False,
    )

    assert out.shape == G0.shape
    assert jnp.allclose(out, 1.0)
    # lax.scan traces one RHS site for the final-only graph. The field-history
    # path has a second RHS site to recover fields at the accepted state.
    assert calls["rhs"] == 1


def test_integrate_nonlinear_sharded_reuses_compiled_runner(monkeypatch) -> None:
    calls = {"compile": 0}

    def fake_pjit(fn, **_kwargs):
        calls["compile"] += 1
        return fn

    def fake_rhs(
        G,
        cache,
        params,
        terms=None,
        *,
        compressed_real_fft=True,
        laguerre_mode="grid",
        external_phi=None,
    ):
        del cache, params, terms, compressed_real_fft, laguerre_mode, external_phi
        return jnp.ones_like(G), FieldState(phi=jnp.ones((1, 1, 1), dtype=G.dtype))

    monkeypatch.setattr(integrator_module, "pjit", fake_pjit)
    monkeypatch.setattr(integrator_module, "nonlinear_rhs_cached", fake_rhs)
    integrator_module._compiled_nonlinear_sharded_runner.cache_clear()
    G0 = jnp.zeros((1, 1, 1, 1, 1), dtype=jnp.complex64)

    for _ in range(2):
        out = integrate_nonlinear_sharded(
            G0,
            _cache_stub(),
            SimpleNamespace(),
            dt=0.1,
            steps=2,
            method="euler",
            return_fields=False,
        )
        assert jnp.allclose(out, 0.2)

    assert calls["compile"] == 1
    integrator_module._compiled_nonlinear_sharded_runner.cache_clear()
