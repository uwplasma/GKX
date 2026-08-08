"""Unit contracts: parallel linear velocity."""

from __future__ import annotations


# ---- test_linear_parallel_dispatch.py ----

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from gkx.solvers.linear import parallel as linear_parallel
import gkx.solvers.linear.parallel_common as linear_parallel_common
import gkx.solvers.linear.parallel_electrostatic as linear_parallel_electrostatic
import gkx.solvers.linear.parallel_streaming as linear_parallel_streaming
from gkx.operators.linear.params import LinearParams, LinearTerms


def _state() -> jnp.ndarray:
    return jnp.zeros((1, 1, 1, 1, 1), dtype=jnp.complex64)


def _richer_state() -> jnp.ndarray:
    values = jnp.arange(2 * 3 * 2 * 1 * 4, dtype=jnp.float32).reshape((2, 3, 2, 1, 4))
    return values.astype(jnp.complex64)


def _cache_for_richer_state(**overrides):
    cache = SimpleNamespace(
        use_twist_shift=False,
        kz=jnp.asarray([0.0, 1.0, -1.0, 2.0], dtype=jnp.float32),
        Jl=jnp.ones((2, 2, 1, 4), dtype=jnp.float32),
        b=jnp.ones((2, 1, 4), dtype=jnp.float32),
        mask0=None,
        bgrad=jnp.ones((4,), dtype=jnp.float32),
        l=jnp.arange(2, dtype=jnp.float32),
        l4=jnp.arange(2, dtype=jnp.float32).reshape((2, 1, 1, 1)),
        m=jnp.arange(3, dtype=jnp.float32),
        sqrt_m=jnp.sqrt(jnp.arange(3, dtype=jnp.float32)),
        sqrt_m_p1=jnp.sqrt(jnp.arange(1, 4, dtype=jnp.float32)),
        cv_d=jnp.ones((2, 1, 4), dtype=jnp.float32),
        gb_d=jnp.ones((2, 1, 4), dtype=jnp.float32),
        ky=jnp.asarray([0.2, 0.5], dtype=jnp.float32),
    )
    for key, value in overrides.items():
        setattr(cache, key, value)
    return cache


def _sentinel(name: str):
    return jnp.asarray([len(name)], dtype=jnp.float32), jnp.asarray(
        [0.0], dtype=jnp.float32
    )


def _streaming_only_terms() -> LinearTerms:
    return LinearTerms(
        streaming=1.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=0.0,
        hypercollisions=0.0,
        hyperdiffusion=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )


def _electrostatic_slice_terms() -> LinearTerms:
    return LinearTerms(
        streaming=1.0,
        mirror=1.0,
        curvature=1.0,
        gradb=1.0,
        diamagnetic=1.0,
        collisions=0.0,
        hypercollisions=0.0,
        hyperdiffusion=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )


def test_linear_parallel_term_classifiers_cover_release_routes() -> None:
    assert linear_parallel._is_streaming_only_terms(_streaming_only_terms()) is True
    assert (
        linear_parallel._is_streaming_only_terms(_electrostatic_slice_terms()) is False
    )
    assert (
        linear_parallel._is_electrostatic_slice_terms(_electrostatic_slice_terms())
        is True
    )
    assert (
        linear_parallel._is_electrostatic_slice_terms(LinearTerms(collisions=1.0))
        is False
    )
    assert (
        linear_parallel._is_electrostatic_field_terms(LinearTerms(apar=0.0, bpar=0.0))
        is True
    )
    assert (
        linear_parallel._is_electrostatic_field_terms(LinearTerms(apar=1.0, bpar=0.0))
        is False
    )
    assert linear_parallel._is_mixed_electrostatic_terms(
        replace(_electrostatic_slice_terms(), hypercollisions=1.0)
    )
    assert linear_parallel._is_mixed_electrostatic_terms(
        replace(_electrostatic_slice_terms(), collisions=1.0)
    )


def test_resolve_parallel_devices_validates_explicit_and_requested_counts(
    monkeypatch,
) -> None:
    devices = [object(), object()]

    assert linear_parallel._resolve_parallel_devices(devices=devices) == devices
    assert (
        linear_parallel._resolve_parallel_devices(devices=devices, num_devices=2)
        == devices
    )
    with pytest.raises(ValueError, match="num_devices"):
        linear_parallel._resolve_parallel_devices(devices=devices, num_devices=1)
    with pytest.raises(ValueError, match="num_devices"):
        linear_parallel._resolve_parallel_devices(num_devices=0)
    monkeypatch.setattr(linear_parallel_common.jax, "devices", lambda: [object()])
    with pytest.raises(ValueError, match="only"):
        linear_parallel._resolve_parallel_devices(num_devices=2)
    with pytest.raises(ValueError, match="at least one"):
        linear_parallel._resolve_parallel_devices(devices=[])


def test_streaming_velocity_sharded_route_validates_shape_and_returns_zero_phi(
    monkeypatch,
) -> None:
    import gkx.parallel.velocity as velocity_sharding

    calls: list[tuple[tuple[int, ...], int]] = []

    def fake_streaming(arr, plan, **kwargs):
        calls.append((tuple(arr.shape), int(plan.num_devices)))
        assert kwargs["kz"].shape == (4,)
        assert kwargs["vth"] == 1.0
        assert len(kwargs["devices"]) == 1
        return jnp.ones_like(arr) * (2.0 + 0.0j)

    monkeypatch.setattr(
        velocity_sharding, "periodic_streaming_shard_map", fake_streaming
    )

    with pytest.raises(ValueError, match="G must have shape"):
        linear_parallel.linear_rhs_streaming_velocity_sharded(
            jnp.zeros((2, 3, 4), dtype=jnp.complex64),
            _cache_for_richer_state(),
            LinearParams(),
            devices=[object()],
        )

    dG, phi = linear_parallel.linear_rhs_streaming_velocity_sharded(
        _richer_state(),
        _cache_for_richer_state(),
        LinearParams(),
        devices=[object()],
    )

    assert calls == [((2, 3, 2, 1, 4), 1)]
    assert dG.shape == _richer_state().shape
    assert jnp.all(dG == -(2.0 + 0.0j))
    assert phi.shape == (2, 1, 4)
    assert jnp.all(phi == 0.0)


def test_electrostatic_streaming_field_rhs_and_sharded_phi_path(monkeypatch) -> None:
    import gkx.operators.linear.streaming as operators
    import gkx.parallel.velocity as velocity_sharding

    arr = _richer_state()
    cache = _cache_for_richer_state()
    phi = jnp.ones(arr.shape[-3:], dtype=arr.dtype) * (1.5 + 0.0j)

    field_rhs = linear_parallel._electrostatic_streaming_field_rhs(
        arr[None, ...],
        phi=phi,
        Jl=cache.Jl,
        tz=jnp.asarray([2.0], dtype=jnp.float32),
        vth=jnp.asarray([3.0], dtype=jnp.float32),
    )
    assert field_rhs.shape == (1,) + arr.shape
    assert jnp.all(field_rhs[:, :, 0] == 0.0)
    assert jnp.all(field_rhs[:, :, 2] == 0.0)
    assert jnp.allclose(field_rhs[:, :, 1], -2.25 + 0.0j)

    def fake_streaming(local_arr, plan, **kwargs):
        assert tuple(local_arr.shape) == tuple(arr.shape)
        assert int(plan.num_devices) == 1
        assert len(kwargs["devices"]) == 1
        return jnp.ones_like(local_arr) * (4.0 + 0.0j)

    def fake_grad_z(value, *, kz):
        assert kz.shape == (4,)
        return jnp.ones_like(value) * (0.25 + 0.0j)

    monkeypatch.setattr(
        velocity_sharding, "periodic_streaming_shard_map", fake_streaming
    )
    monkeypatch.setattr(operators, "grad_z_periodic", fake_grad_z)

    from gkx.parallel.velocity import build_velocity_sharding_plan

    out = linear_parallel._streaming_electrostatic_from_phi_velocity_sharded(
        arr,
        cache,
        LinearParams(kpar_scale=2.0, tz=2.0, vth=3.0),
        phi=phi,
        plan=build_velocity_sharding_plan(arr.shape, num_devices=1, axes=("hermite",)),
        devices=[object()],
    )

    assert out.shape == arr.shape
    assert jnp.allclose(out, -(4.0 + 0.0j) + 0.5)


def test_streaming_electrostatic_velocity_sharded_fail_closed_and_uses_phi(
    monkeypatch,
) -> None:
    import gkx.parallel.velocity as velocity_sharding

    arr = _richer_state()
    cache = _cache_for_richer_state()
    phi_expected = jnp.ones(arr.shape[-3:], dtype=arr.dtype) * (0.75 + 0.0j)

    with pytest.raises(NotImplementedError, match="single-species 5D"):
        linear_parallel.linear_rhs_streaming_electrostatic_velocity_sharded(
            arr[None, ...],
            cache,
            LinearParams(),
            devices=[object()],
        )
    with pytest.raises(NotImplementedError, match="periodic z grid"):
        linear_parallel.linear_rhs_streaming_electrostatic_velocity_sharded(
            arr,
            _cache_for_richer_state(use_twist_shift=True),
            LinearParams(),
            devices=[object()],
        )

    def fake_phi(local_arr, plan, **kwargs):
        assert tuple(local_arr.shape) == tuple(arr.shape)
        assert kwargs["tau_e"] == 1.0
        assert len(kwargs["devices"]) == 1
        return phi_expected

    def fake_rhs(local_arr, local_cache, local_params, *, phi, plan, devices):
        assert local_arr is not arr or tuple(local_arr.shape) == tuple(arr.shape)
        assert jnp.all(phi == phi_expected)
        assert int(plan.num_devices) == 1
        assert len(devices) == 1
        return jnp.ones_like(local_arr) * (6.0 + 0.0j)

    monkeypatch.setattr(velocity_sharding, "electrostatic_phi_shard_map", fake_phi)
    monkeypatch.setattr(
        linear_parallel_streaming,
        "_streaming_electrostatic_from_phi_velocity_sharded",
        fake_rhs,
    )

    dG, phi = linear_parallel.linear_rhs_streaming_electrostatic_velocity_sharded(
        arr,
        cache,
        LinearParams(),
        devices=[object()],
    )

    assert jnp.all(dG == 6.0 + 0.0j)
    assert jnp.all(phi == phi_expected)


def test_electrostatic_slices_velocity_sharded_fail_closed_and_weighted_routes(
    monkeypatch,
) -> None:
    import gkx.parallel.velocity as velocity_sharding

    arr = _richer_state()
    cache = _cache_for_richer_state()
    phi_expected = jnp.ones(arr.shape[-3:], dtype=arr.dtype) * (0.25 + 0.0j)
    calls: list[str] = []

    with pytest.raises(NotImplementedError, match="allows only electrostatic"):
        linear_parallel.linear_rhs_electrostatic_slices_velocity_sharded(
            arr,
            cache,
            LinearParams(),
            LinearTerms(collisions=1.0),
            devices=[object()],
        )
    with pytest.raises(NotImplementedError, match="single-species 5D"):
        linear_parallel.linear_rhs_electrostatic_slices_velocity_sharded(
            arr[None, ...],
            cache,
            LinearParams(),
            _electrostatic_slice_terms(),
            devices=[object()],
        )
    with pytest.raises(NotImplementedError, match="periodic z grid"):
        linear_parallel.linear_rhs_electrostatic_slices_velocity_sharded(
            arr,
            _cache_for_richer_state(use_twist_shift=True),
            LinearParams(),
            _electrostatic_slice_terms(),
            devices=[object()],
        )

    def fake_phi(*args, **kwargs):
        calls.append("phi")
        return phi_expected

    def fake_streaming(local_arr, *_args, **_kwargs):
        calls.append("streaming")
        return jnp.ones_like(local_arr) * (1.0 + 0.0j)

    def fake_build_h(local_arr, *_args, **_kwargs):
        calls.append("H")
        return local_arr + (0.5 + 0.0j)

    def fake_mirror(local_h, *_args, **kwargs):
        calls.append("mirror")
        assert float(kwargs["weight"]) == pytest.approx(0.25)
        return jnp.ones_like(local_h) * (2.0 * kwargs["weight"] + 0.0j)

    def fake_curv(local_h, *_args, **kwargs):
        calls.append("curv")
        assert float(kwargs["weight_curv"]) == pytest.approx(0.5)
        assert float(kwargs["weight_gradb"]) == pytest.approx(0.0)
        return jnp.ones_like(local_h) * (3.0 * kwargs["weight_curv"] + 0.0j)

    def fake_diamag(local_arr, *_args, **kwargs):
        calls.append("diamag")
        assert float(kwargs["weight"]) == pytest.approx(0.125)
        assert kwargs["b"] is cache.b
        assert kwargs["l4"] is cache.l4
        return jnp.ones_like(local_arr) * (4.0 * kwargs["weight"] + 0.0j)

    monkeypatch.setattr(velocity_sharding, "electrostatic_phi_shard_map", fake_phi)
    monkeypatch.setattr(
        linear_parallel_electrostatic,
        "_streaming_electrostatic_from_phi_velocity_sharded",
        fake_streaming,
    )
    monkeypatch.setattr(linear_parallel_electrostatic, "build_H", fake_build_h)
    monkeypatch.setattr(velocity_sharding, "mirror_drift_shard_map", fake_mirror)
    monkeypatch.setattr(velocity_sharding, "curvature_gradb_drift_shard_map", fake_curv)
    monkeypatch.setattr(velocity_sharding, "diamagnetic_drive_shard_map", fake_diamag)

    terms = LinearTerms(
        streaming=0.5,
        mirror=0.25,
        curvature=0.5,
        gradb=0.0,
        diamagnetic=0.125,
        collisions=0.0,
        hypercollisions=0.0,
        hyperdiffusion=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )
    dG, phi = linear_parallel.linear_rhs_electrostatic_slices_velocity_sharded(
        arr,
        cache,
        LinearParams(),
        terms,
        devices=[object()],
    )

    assert calls == ["phi", "streaming", "H", "mirror", "curv", "diamag"]
    assert jnp.all(phi == phi_expected)
    assert jnp.allclose(dG, 0.5 * 1.0 + 0.25 * 2.0 + 0.5 * 3.0 + 0.125 * 4.0)


def test_fused_electrostatic_slice_route_validates_decomposition_before_mesh() -> None:
    arr = _richer_state()
    cache = _cache_for_richer_state()
    terms = _electrostatic_slice_terms()

    with pytest.raises(ValueError, match="more than one Hermite chunk"):
        linear_parallel._linear_rhs_electrostatic_slices_velocity_sharded_fused(
            arr,
            cache,
            LinearParams(),
            terms,
            plan=SimpleNamespace(chunks={"m": 1}, active_axes=("m",)),
            devices=[object()],
        )

    with pytest.raises(ValueError, match="Hermite dimension must divide evenly"):
        linear_parallel._linear_rhs_electrostatic_slices_velocity_sharded_fused(
            arr,
            cache,
            LinearParams(),
            terms,
            plan=SimpleNamespace(chunks={"m": 2}, active_axes=("m",)),
            devices=[object(), object()],
        )

    with pytest.raises(NotImplementedError, match="only an active 'm' axis"):
        linear_parallel._linear_rhs_electrostatic_slices_velocity_sharded_fused(
            jnp.zeros((2, 4, 2, 1, 4), dtype=jnp.complex64),
            cache,
            LinearParams(),
            terms,
            plan=SimpleNamespace(chunks={"m": 2}, active_axes=("m", "l")),
            devices=[object(), object()],
        )


def test_linear_rhs_parallel_cached_serial_dispatch(monkeypatch) -> None:
    import gkx.operators.linear.rhs as linear_rhs_owner

    calls: list[str] = []

    def fake_serial(*args, **kwargs):
        calls.append("serial")
        assert kwargs["use_jit"] is False
        assert kwargs["use_custom_vjp"] is False
        assert kwargs["dt"] == 0.125
        return _sentinel("serial")

    monkeypatch.setattr(linear_rhs_owner, "linear_rhs_cached", fake_serial)

    out = linear_parallel.linear_rhs_parallel_cached(
        _state(),
        SimpleNamespace(),
        LinearParams(),
        parallel=None,
        use_jit=False,
        use_custom_vjp=False,
        dt=0.125,
    )

    assert calls == ["serial"]
    assert int(out[0][0]) == len("serial")


def test_linear_rhs_parallel_cached_velocity_auto_selects_electrostatic_slice(
    monkeypatch,
) -> None:
    calls: list[tuple[str, int | None]] = []

    def fake_slice(*args, **kwargs):
        calls.append(("slice", kwargs["num_devices"]))
        assert kwargs["terms"] == _electrostatic_slice_terms()
        return _sentinel("slice")

    monkeypatch.setattr(
        linear_parallel, "linear_rhs_electrostatic_slices_velocity_sharded", fake_slice
    )
    parallel = SimpleNamespace(
        strategy="velocity", backend="auto", axis="hermite", num_devices=3
    )

    out = linear_parallel.linear_rhs_parallel_cached(
        _state(),
        SimpleNamespace(),
        LinearParams(),
        terms=_electrostatic_slice_terms(),
        parallel=parallel,
    )

    assert calls == [("slice", 3)]
    assert int(out[0][0]) == len("slice")


def test_linear_rhs_parallel_cached_explicit_streaming_routes(monkeypatch) -> None:
    calls: list[str] = []

    def fake_streaming(*args, **kwargs):
        calls.append("streaming")
        assert kwargs["num_devices"] == 2
        return _sentinel("streaming")

    def fake_electrostatic(*args, **kwargs):
        calls.append("electrostatic")
        assert kwargs["num_devices"] == 2
        assert kwargs["use_custom_vjp"] is False
        return _sentinel("electrostatic")

    monkeypatch.setattr(
        linear_parallel, "linear_rhs_streaming_velocity_sharded", fake_streaming
    )
    monkeypatch.setattr(
        linear_parallel,
        "linear_rhs_streaming_electrostatic_velocity_sharded",
        fake_electrostatic,
    )
    terms = _streaming_only_terms()

    linear_parallel.linear_rhs_parallel_cached(
        _state(),
        SimpleNamespace(),
        LinearParams(),
        terms=terms,
        parallel=SimpleNamespace(
            strategy="velocity", backend="streaming_only", axis="m", num_devices=2
        ),
    )
    linear_parallel.linear_rhs_parallel_cached(
        _state(),
        SimpleNamespace(),
        LinearParams(),
        terms=terms,
        parallel=SimpleNamespace(
            strategy="velocity",
            backend="streaming_electrostatic",
            axis="hermite",
            num_devices=2,
        ),
        use_custom_vjp=False,
    )

    assert calls == ["streaming", "electrostatic"]


def test_linear_rhs_parallel_cached_rejects_unsupported_velocity_requests() -> None:
    g = _state()
    cache = SimpleNamespace()
    params = LinearParams()

    with pytest.raises(NotImplementedError, match="Hermite axis"):
        linear_parallel.linear_rhs_parallel_cached(
            g,
            cache,
            params,
            terms=_electrostatic_slice_terms(),
            parallel=SimpleNamespace(strategy="velocity", backend="auto", axis="kx"),
        )
    with pytest.raises(NotImplementedError, match="backend='auto'"):
        linear_parallel.linear_rhs_parallel_cached(
            g,
            cache,
            params,
            terms=LinearTerms(apar=1.0),
            parallel=SimpleNamespace(
                strategy="velocity", backend="auto", axis="hermite"
            ),
        )
    with pytest.raises(NotImplementedError, match="streaming-only"):
        linear_parallel.linear_rhs_parallel_cached(
            g,
            cache,
            params,
            terms=_streaming_only_terms(),
            parallel=SimpleNamespace(
                strategy="velocity", backend="streaming_only", axis="ky"
            ),
        )
    with pytest.raises(NotImplementedError, match="requires streaming-only"):
        linear_parallel.linear_rhs_parallel_cached(
            g,
            cache,
            params,
            terms=_electrostatic_slice_terms(),
            parallel=SimpleNamespace(
                strategy="velocity", backend="streaming_only", axis="hermite"
            ),
        )
    with pytest.raises(NotImplementedError, match="collision/EM"):
        linear_parallel.linear_rhs_parallel_cached(
            g,
            cache,
            params,
            terms=LinearTerms(apar=1.0),
            parallel=SimpleNamespace(
                strategy="velocity",
                backend="electrostatic_linear_slices",
                axis="hermite",
            ),
        )
    with pytest.raises(NotImplementedError, match="strategy='velocity'"):
        linear_parallel.linear_rhs_parallel_cached(
            g,
            cache,
            params,
            terms=_electrostatic_slice_terms(),
            parallel=SimpleNamespace(
                strategy="domain", backend="unknown", axis="hermite"
            ),
        )


# ---- test_velocity_sharding.py ----

from dataclasses import replace


import jax
import numpy as np

import gkx
from gkx.parallel.velocity import (
    build_velocity_sharding_plan,
    curvature_gradb_drift_reference,
    curvature_gradb_drift_shard_map,
    diamagnetic_drive_reference,
    diamagnetic_drive_shard_map,
    electrostatic_phi_reference,
    electrostatic_phi_shard_map,
    hermite_neighbor_reference,
    hermite_neighbor_shard_map,
    hermite_shift_reference,
    hermite_shift_shard_map,
    hermite_streaming_ladder_reference,
    hermite_streaming_ladder_shard_map,
    mirror_drift_reference,
    mirror_drift_shard_map,
    periodic_streaming_reference,
    periodic_streaming_shard_map,
    velocity_field_reduce_reference,
    velocity_field_reduce_shard_map,
)


def _install_fake_two_way_shard_map(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise shard-map branches without requiring CI-visible logical devices."""

    class FakeMesh:
        def __init__(self, devices, names):
            self.devices = devices
            self.names = names

    class FakeNamedSharding:
        def __init__(self, mesh, spec):
            self.mesh = mesh
            self.spec = spec

    def fake_partition_spec(*axes):
        return axes

    def local_hermite_slice(arr):
        if getattr(arr, "ndim", 0) not in (5, 6):
            return arr
        m_axis = 1 if arr.ndim == 5 else 2
        local_m = int(arr.shape[m_axis]) // 2
        index = [slice(None)] * arr.ndim
        index[m_axis] = slice(0, local_m)
        return arr[tuple(index)]

    def fake_shard_map(fn, **_kwargs):
        def mapped(arr):
            return fn(local_hermite_slice(arr))

        return mapped

    monkeypatch.setattr("jax.sharding.Mesh", FakeMesh)
    monkeypatch.setattr("jax.sharding.NamedSharding", FakeNamedSharding)
    monkeypatch.setattr("jax.sharding.PartitionSpec", fake_partition_spec)
    monkeypatch.setattr(jax, "shard_map", fake_shard_map)
    monkeypatch.setattr(jax, "device_put", lambda arr, _sharding=None: arr)
    monkeypatch.setattr(
        jax.lax, "ppermute", lambda value, *_args, **_kwargs: jnp.zeros_like(value)
    )
    monkeypatch.setattr(jax.lax, "psum", lambda value, *_args, **_kwargs: value)
    monkeypatch.setattr(jax.lax, "axis_index", lambda *_args, **_kwargs: 0)


def test_velocity_sharding_plan_prefers_species_then_hermite_for_6d_state() -> None:
    plan = build_velocity_sharding_plan((2, 4, 8, 16, 4, 32), num_devices=4)

    assert gkx.build_velocity_sharding_plan is build_velocity_sharding_plan
    assert plan.dims == ("s", "l", "m", "ky", "kx", "z")
    assert plan.chunks["s"] == 2
    assert plan.chunks["m"] == 2
    assert plan.chunks["l"] == 1
    assert plan.active_axes == ("s", "m")
    assert plan.shard_shape == (1, 4, 4, 16, 4, 32)
    assert plan.needs_hermite_exchange is True
    assert plan.needs_field_reduction is True
    assert plan.field_reduction_axes == ("s", "m")
    assert plan.communication_pattern == "hermite_ghost_exchange+field_reduce_broadcast"
    assert plan.load_balance == pytest.approx(1.0)
    assert plan.to_dict()["num_devices"] == 4


def test_velocity_sharding_plan_uses_hermite_for_single_species_5d_state() -> None:
    plan = build_velocity_sharding_plan((4, 8, 16, 4, 32), num_devices=2)

    assert plan.dims == ("l", "m", "ky", "kx", "z")
    assert plan.chunks["m"] == 2
    assert plan.active_axes == ("m",)
    assert plan.shard_shape == (4, 4, 16, 4, 32)
    assert plan.field_reduction_axes == ("m",)


def test_velocity_sharding_plan_supports_explicit_laguerre_fallback() -> None:
    plan = build_velocity_sharding_plan(
        (1, 4, 3, 16, 4, 32), num_devices=4, axes=("hermite", "laguerre")
    )

    assert plan.chunks["m"] == 2
    assert plan.chunks["l"] == 2
    assert plan.active_axes == ("l", "m")


def test_velocity_sharding_plan_rejects_invalid_requests() -> None:
    with pytest.raises(ValueError, match="5 or 6"):
        build_velocity_sharding_plan((2, 3, 4), num_devices=2)
    with pytest.raises(ValueError, match="entries"):
        build_velocity_sharding_plan((4, 0, 16, 4, 32), num_devices=1)
    with pytest.raises(ValueError, match="num_devices"):
        build_velocity_sharding_plan((4, 8, 16, 4, 32), num_devices=0)
    with pytest.raises(ValueError, match="ghost"):
        build_velocity_sharding_plan(
            (4, 8, 16, 4, 32), num_devices=1, hermite_ghost_depth=-1
        )
    with pytest.raises(ValueError, match="species sharding"):
        build_velocity_sharding_plan(
            (4, 8, 16, 4, 32), num_devices=2, axes=("species",)
        )
    with pytest.raises(ValueError, match="could not be factored"):
        build_velocity_sharding_plan((1, 2, 3, 4, 1, 8), num_devices=5)
    with pytest.raises(ValueError, match="Unknown"):
        build_velocity_sharding_plan((4, 8, 16, 4, 32), num_devices=2, axes=("banana",))
    field_only = build_velocity_sharding_plan(
        (4, 8, 16, 4, 32), num_devices=2, axes=("laguerre",)
    )
    assert field_only.communication_pattern == "field_reduce_broadcast"


def test_velocity_sharding_rejects_invalid_hermite_exchange_requests() -> None:
    state = jnp.zeros((2, 6, 3, 1, 4), dtype=jnp.complex64)
    plan = build_velocity_sharding_plan(state.shape, num_devices=2, axes=("hermite",))
    bad_shape = state[:, :5, ...]
    bad_chunks = replace(plan, chunks={**plan.chunks, "m": 0})
    uneven = replace(plan, state_shape=bad_shape.shape)
    laguerre_plan = build_velocity_sharding_plan(
        state.shape, num_devices=2, axes=("laguerre",)
    )

    with pytest.raises(ValueError, match="state shape"):
        hermite_neighbor_shard_map(bad_shape, plan)
    with pytest.raises(ValueError, match="chunk count"):
        hermite_neighbor_shard_map(state, bad_chunks)
    with pytest.raises(ValueError, match="divide evenly"):
        hermite_neighbor_shard_map(bad_shape, uneven, devices=[object(), object()])
    with pytest.raises(ValueError, match="not enough devices"):
        hermite_neighbor_shard_map(state, plan, devices=[object()])

    np.testing.assert_allclose(
        np.asarray(hermite_shift_shard_map(state, plan, offset=0)), np.asarray(state)
    )
    np.testing.assert_allclose(
        np.asarray(hermite_shift_shard_map(state, plan, offset=99)),
        np.zeros_like(np.asarray(state)),
    )
    with pytest.raises(ValueError, match="state shape"):
        hermite_shift_shard_map(bad_shape, plan, offset=1)
    with pytest.raises(NotImplementedError, match="active 'm'"):
        hermite_shift_shard_map(state, laguerre_plan, offset=1)
    with pytest.raises(ValueError, match="divide evenly"):
        hermite_shift_shard_map(
            bad_shape, uneven, offset=1, devices=[object(), object()]
        )
    with pytest.raises(NotImplementedError, match="local shard size"):
        hermite_shift_shard_map(state, plan, offset=4, devices=[object(), object()])
    with pytest.raises(ValueError, match="not enough devices"):
        hermite_shift_shard_map(state, plan, offset=1, devices=[object()])


def test_velocity_field_reduce_rejects_invalid_axes_and_plans() -> None:
    state = jnp.zeros((2, 6, 3, 1, 4), dtype=jnp.complex64)
    plan = build_velocity_sharding_plan(state.shape, num_devices=2, axes=("hermite",))
    bad_shape = state[:, :5, ...]
    uneven = replace(plan, state_shape=bad_shape.shape)
    laguerre_plan = build_velocity_sharding_plan(
        state.shape, num_devices=2, axes=("laguerre",)
    )

    with pytest.raises(ValueError, match="Unknown"):
        velocity_field_reduce_reference(state, axis="banana")
    with pytest.raises(ValueError, match="not present"):
        velocity_field_reduce_reference(state, axis="species")
    with pytest.raises(ValueError, match="state shape"):
        velocity_field_reduce_shard_map(bad_shape, plan)
    with pytest.raises(ValueError, match="Unknown"):
        velocity_field_reduce_shard_map(state, plan, axis="banana")
    with pytest.raises(ValueError, match="not present"):
        velocity_field_reduce_shard_map(state, plan, axis="species")
    with pytest.raises(NotImplementedError, match="one active reduction axis"):
        velocity_field_reduce_shard_map(state, plan, axis="laguerre")
    with pytest.raises(NotImplementedError, match="one active reduction axis"):
        velocity_field_reduce_shard_map(state, laguerre_plan)
    with pytest.raises(ValueError, match="divide evenly"):
        velocity_field_reduce_shard_map(bad_shape, uneven, devices=[object(), object()])
    with pytest.raises(ValueError, match="not enough devices"):
        velocity_field_reduce_shard_map(state, plan, devices=[object()])


def test_hermite_neighbor_reference_matches_manual_shift() -> None:
    state = jnp.arange(2 * 5 * 3 * 1 * 4, dtype=jnp.float32).reshape((2, 5, 3, 1, 4))

    lower, upper = hermite_neighbor_reference(state)

    expected_lower = np.zeros_like(np.asarray(state))
    expected_upper = np.zeros_like(np.asarray(state))
    expected_lower[:, 1:, ...] = np.asarray(state)[:, :-1, ...]
    expected_upper[:, :-1, ...] = np.asarray(state)[:, 1:, ...]
    np.testing.assert_allclose(np.asarray(lower), expected_lower)
    np.testing.assert_allclose(np.asarray(upper), expected_upper)
    assert gkx.hermite_neighbor_reference is hermite_neighbor_reference
    assert gkx.hermite_neighbor_shard_map is hermite_neighbor_shard_map


def test_hermite_neighbor_shard_map_noops_to_reference_for_single_chunk() -> None:
    state = jnp.arange(2 * 5 * 3 * 1 * 4, dtype=jnp.float32).reshape((2, 5, 3, 1, 4))
    plan = build_velocity_sharding_plan(state.shape, num_devices=1, axes=("hermite",))

    lower, upper = hermite_neighbor_shard_map(state, plan, devices=[jax.devices()[0]])
    expected_lower, expected_upper = hermite_neighbor_reference(state)

    np.testing.assert_allclose(np.asarray(lower), np.asarray(expected_lower))
    np.testing.assert_allclose(np.asarray(upper), np.asarray(expected_upper))


def test_hermite_neighbor_shard_map_matches_reference_when_logical_devices_available() -> (
    None
):
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip(
            "requires at least two JAX devices; artifact generator sets logical CPU devices"
        )
    state = (
        jnp.arange(2 * 6 * 3 * 1 * 4, dtype=jnp.float32).reshape((2, 6, 3, 1, 4)) + 1j
    ).astype(jnp.complex64)
    plan = build_velocity_sharding_plan(state.shape, num_devices=2, axes=("hermite",))

    lower, upper = hermite_neighbor_shard_map(state, plan, devices=devices[:2])
    expected_lower, expected_upper = hermite_neighbor_reference(state)

    np.testing.assert_allclose(np.asarray(lower), np.asarray(expected_lower))
    np.testing.assert_allclose(np.asarray(upper), np.asarray(expected_upper))


def test_mocked_shard_map_exercises_multi_device_velocity_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_two_way_shard_map(monkeypatch)
    state = (
        jnp.arange(2 * 6 * 3 * 1 * 4, dtype=jnp.float32).reshape((2, 6, 3, 1, 4)) + 1j
    ).astype(jnp.complex64)
    plan = build_velocity_sharding_plan(state.shape, num_devices=2, axes=("hermite",))
    devices = [object(), object()]

    lower, upper = hermite_neighbor_shard_map(state, plan, devices=devices)
    assert lower.shape == (2, 3, 3, 1, 4)
    assert upper.shape == (2, 3, 3, 1, 4)
    assert hermite_shift_shard_map(state, plan, offset=1, devices=devices).shape == (
        2,
        3,
        3,
        1,
        4,
    )
    assert hermite_shift_shard_map(state, plan, offset=-1, devices=devices).shape == (
        2,
        3,
        3,
        1,
        4,
    )
    assert velocity_field_reduce_shard_map(state, plan, devices=devices).shape == (
        2,
        3,
        1,
        4,
    )

    kz = jnp.asarray([0.0, 1.0, -1.0, -2.0], dtype=jnp.float32)
    local_coeffs = jnp.ones((1, 3, 1, 1, 1), dtype=state.dtype)
    monkeypatch.setattr(
        "gkx.parallel.velocity._hermite_ladder_coefficients",
        lambda _arr: (local_coeffs, local_coeffs, 1),
    )
    assert hermite_streaming_ladder_shard_map(
        state, plan, vth=1.2, devices=devices
    ).shape == (2, 3, 3, 1, 4)
    assert periodic_streaming_shard_map(
        state, plan, kz=kz, vth=1.2, devices=devices
    ).shape == (2, 3, 3, 1, 4)

    local_m = 3
    local_m_shape = (1, local_m, 1, 1, 1)
    ell = jnp.arange(state.shape[0], dtype=jnp.float32).reshape(
        (state.shape[0], 1, 1, 1, 1)
    )
    local_m_values = jnp.arange(local_m, dtype=jnp.float32).reshape(local_m_shape)
    mirror = mirror_drift_shard_map(
        state,
        plan,
        vth=jnp.asarray([1.0], dtype=jnp.float32),
        bgrad=jnp.ones((state.shape[-1],), dtype=jnp.float32),
        ell=ell,
        sqrt_m=jnp.sqrt(local_m_values),
        sqrt_m_p1=jnp.sqrt(local_m_values + 1.0),
        devices=devices,
    )
    assert mirror.shape == (2, 3, 3, 1, 4)
    Jl = jnp.ones(
        (state.shape[0], state.shape[2], state.shape[3], state.shape[4]),
        dtype=jnp.float32,
    )
    phi = electrostatic_phi_shard_map(
        state,
        plan,
        Jl=Jl,
        tau_e=1.0,
        charge=1.0,
        density=1.0,
        tz=1.0,
        mask0=jnp.zeros((state.shape[2], state.shape[3], state.shape[4]), dtype=bool),
        devices=devices,
    )
    assert phi.shape == (3, 1, 4)
    assert electrostatic_phi_shard_map(
        state,
        plan,
        Jl=Jl[None, ...],
        tau_e=1.0,
        devices=devices,
    ).shape == (3, 1, 4)
    diamagnetic = diamagnetic_drive_shard_map(
        state,
        plan,
        phi=phi,
        Jl=Jl,
        b=jnp.zeros((state.shape[2], state.shape[3], state.shape[4])),
        l4=jnp.arange(state.shape[0], dtype=jnp.float32).reshape(
            (state.shape[0], 1, 1, 1)
        ),
        tprim=6.9,
        fprim=2.2,
        omega_star_scale=1.0,
        ky=jnp.asarray([0.0, 0.3, 0.6], dtype=jnp.float32),
        devices=devices,
    )
    assert diamagnetic.shape == (2, 3, 3, 1, 4)


def test_hermite_shift_reference_matches_manual_offsets() -> None:
    state = jnp.arange(2 * 6 * 3 * 1 * 4, dtype=jnp.float32).reshape((2, 6, 3, 1, 4))

    upper2 = hermite_shift_reference(state, offset=2)
    lower2 = hermite_shift_reference(state, offset=-2)

    expected_upper2 = np.zeros_like(np.asarray(state))
    expected_lower2 = np.zeros_like(np.asarray(state))
    expected_upper2[:, :-2, ...] = np.asarray(state)[:, 2:, ...]
    expected_lower2[:, 2:, ...] = np.asarray(state)[:, :-2, ...]
    np.testing.assert_allclose(np.asarray(upper2), expected_upper2)
    np.testing.assert_allclose(np.asarray(lower2), expected_lower2)
    assert gkx.hermite_shift_reference is hermite_shift_reference
    assert gkx.hermite_shift_shard_map is hermite_shift_shard_map


def test_hermite_shift_shard_map_matches_reference_when_logical_devices_available() -> (
    None
):
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip(
            "requires at least two JAX devices; artifact generator sets logical CPU devices"
        )
    state = (
        jnp.arange(2 * 8 * 3 * 1 * 4, dtype=jnp.float32).reshape((2, 8, 3, 1, 4)) + 1j
    ).astype(jnp.complex64)
    plan = build_velocity_sharding_plan(state.shape, num_devices=2, axes=("hermite",))

    for offset in (-2, -1, 1, 2):
        observed = hermite_shift_shard_map(
            state, plan, offset=offset, devices=devices[:2]
        )
        expected = hermite_shift_reference(state, offset=offset)
        np.testing.assert_allclose(np.asarray(observed), np.asarray(expected))


def test_hermite_neighbor_shard_map_rejects_unsupported_multi_axis_plan() -> None:
    state = jnp.zeros((2, 4, 8, 3, 1, 4), dtype=jnp.complex64)
    plan = build_velocity_sharding_plan(state.shape, num_devices=4)

    with pytest.raises(NotImplementedError, match="only an active 'm' axis"):
        hermite_neighbor_shard_map(state, plan, devices=[jax.devices()[0]] * 4)


def test_velocity_field_reduce_reference_matches_manual_sum() -> None:
    state = jnp.arange(2 * 5 * 3 * 1 * 4, dtype=jnp.float32).reshape((2, 5, 3, 1, 4))

    reduced = velocity_field_reduce_reference(state, axis="hermite")

    np.testing.assert_allclose(np.asarray(reduced), np.asarray(state).sum(axis=1))
    assert gkx.velocity_field_reduce_reference is velocity_field_reduce_reference
    assert gkx.velocity_field_reduce_shard_map is velocity_field_reduce_shard_map


def test_velocity_field_reduce_shard_map_noops_to_reference_for_single_chunk() -> None:
    state = jnp.arange(2 * 5 * 3 * 1 * 4, dtype=jnp.float32).reshape((2, 5, 3, 1, 4))
    plan = build_velocity_sharding_plan(state.shape, num_devices=1, axes=("hermite",))

    reduced = velocity_field_reduce_shard_map(state, plan, devices=[jax.devices()[0]])
    expected = velocity_field_reduce_reference(state)

    np.testing.assert_allclose(np.asarray(reduced), np.asarray(expected))


def test_velocity_field_reduce_shard_map_matches_reference_when_logical_devices_available() -> (
    None
):
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip(
            "requires at least two JAX devices; artifact generator sets logical CPU devices"
        )
    state = (
        jnp.arange(2 * 6 * 3 * 1 * 4, dtype=jnp.float32).reshape((2, 6, 3, 1, 4)) + 1j
    ).astype(jnp.complex64)
    plan = build_velocity_sharding_plan(state.shape, num_devices=2, axes=("hermite",))

    reduced = velocity_field_reduce_shard_map(state, plan, devices=devices[:2])
    expected = velocity_field_reduce_reference(state)

    np.testing.assert_allclose(np.asarray(reduced), np.asarray(expected))


def test_species_field_reduce_shard_map_matches_reference_when_devices_available() -> (
    None
):
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("requires two logical CPU devices or two accelerators")
    state = (
        jnp.arange(2 * 2 * 3 * 2 * 1 * 4, dtype=jnp.float32).reshape((2, 2, 3, 2, 1, 4))
        + 1j
    ).astype(jnp.complex64)
    plan = build_velocity_sharding_plan(state.shape, num_devices=2, axes=("species",))

    reduced = velocity_field_reduce_shard_map(
        state, plan, axis="species", devices=devices[:2], axis_name="species"
    )
    expected = velocity_field_reduce_reference(state, axis="species")

    np.testing.assert_allclose(np.asarray(reduced), np.asarray(expected))


def _small_periodic_field_problem():
    from gkx.config import CycloneBaseCase, GridConfig
    from gkx.geometry import SAlphaGeometry
    from gkx.core.grid import build_spectral_grid
    from gkx.operators.linear.cache_builder import build_linear_cache
    from gkx.operators.linear.params import LinearParams

    cfg = CycloneBaseCase(
        grid=GridConfig(Nx=1, Ny=4, Nz=8, Lx=6.0, Ly=6.0, boundary="periodic")
    )
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(beta=0.0, fapar=0.0)
    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=6)
    z = jnp.linspace(0.0, 2.0 * jnp.pi, grid.z.size, endpoint=False)
    state = jnp.zeros(
        (2, 6, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64
    )
    state = state.at[0, 0, min(1, grid.ky.size - 1), 0, :].set(0.2 * jnp.exp(1j * z))
    state = state.at[1, 0, min(2, grid.ky.size - 1), 0, :].set(0.1 * jnp.exp(2j * z))
    return state, cache, params


def _small_kinetic_electron_problem(*, linked: bool = False):
    from gkx.config import CycloneBaseCase, GridConfig
    from gkx.geometry import SAlphaGeometry
    from gkx.core.grid import build_spectral_grid
    from gkx.operators.linear.cache_builder import build_linear_cache
    from gkx.operators.linear.params import LinearParams

    grid_config = (
        GridConfig(
            Nx=8,
            Ny=8,
            Nz=8,
            Lx=62.8,
            Ly=2.0 * np.pi,
            boundary="linked",
            y0=1.0,
        )
        if linked
        else GridConfig(Nx=1, Ny=4, Nz=8, Lx=6.0, Ly=6.0, boundary="periodic")
    )
    cfg = CycloneBaseCase(grid=grid_config)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(
        charge_sign=jnp.asarray([1.0, -1.0]),
        density=jnp.asarray([1.0, 1.0]),
        mass=jnp.asarray([1.0, 1.0 / 1836.0]),
        temp=jnp.asarray([1.0, 1.0]),
        vth=jnp.asarray([1.0, 42.0]),
        rho=jnp.asarray([1.0, 0.023]),
        fprim=jnp.asarray([2.2, 2.2]),
        tprim=jnp.asarray([6.9, 0.0]),
        tprim_e=jnp.asarray([0.0, 6.9]),
        tz=jnp.asarray([1.0, -1.0]),
        tau_e=0.0,
        beta=0.0,
        fapar=0.0,
    )
    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=6)
    z = jnp.linspace(0.0, 2.0 * jnp.pi, grid.z.size, endpoint=False)
    state = jnp.zeros(
        (2, 2, 6, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64
    )
    ky = min(1, grid.ky.size - 1)
    state = state.at[0, 0, 0, ky, 0, :].set(0.2 * jnp.exp(1j * z))
    state = state.at[1, 0, 0, ky, 0, :].set(0.07j * jnp.exp(2j * z))
    return state, cache, params, grid, geom


def test_mixed_species_hermite_linked_operator_matches_serial_trajectory() -> None:
    from gkx.operators.linear.rhs import linear_rhs_cached
    from gkx.solvers.linear.integrators import integrate_linear

    devices = jax.devices()
    if len(devices) < 4:
        pytest.skip("requires four logical CPU devices or accelerators")
    template, cache, params, grid, geom = _small_kinetic_electron_problem(linked=True)
    assert cache.use_twist_shift
    assert cache.linked_indices
    assert cache.linked_damp_profile.size > 0
    rng = np.random.default_rng(20260713)
    state = jnp.asarray(
        1.0e-3
        * (
            rng.standard_normal(template.shape)
            + 1j * rng.standard_normal(template.shape)
        ),
        dtype=template.dtype,
    )
    params = replace(
        params,
        nu=jnp.asarray([0.1, 0.2]),
        nu_hyper_l=0.03,
        nu_hyper_m=0.05,
        hypercollisions_kz=0.4,
        damp_ends_amp=0.1,
    )
    terms = replace(
        _electrostatic_slice_terms(),
        collisions=1.0,
        hypercollisions=1.0,
        end_damping=1.0,
    )
    expected_rhs, expected_phi = linear_rhs_cached(
        state,
        cache,
        params,
        terms=terms,
        dt=0.01,
        use_jit=False,
        use_custom_vjp=False,
        force_electrostatic_fields=True,
    )
    observed_rhs, observed_phi = (
        linear_parallel_streaming.linear_rhs_electrostatic_species_hermite_sharded(
            state,
            cache,
            params,
            terms=terms,
            dt=0.01,
            devices=devices[:4],
        )
    )
    assert float(jnp.linalg.norm(expected_rhs)) > 0.0
    np.testing.assert_allclose(
        np.asarray(observed_phi), np.asarray(expected_phi), rtol=4e-6, atol=4e-6
    )
    np.testing.assert_allclose(
        np.asarray(observed_rhs), np.asarray(expected_rhs), rtol=8e-5, atol=8e-6
    )

    integration = dict(dt=1.0e-5, steps=2, method="euler", cache=cache, terms=terms)
    serial_state, serial_phi = integrate_linear(
        state, grid, geom, params, **integration
    )
    mixed_state, mixed_phi = integrate_linear(
        state,
        grid,
        geom,
        params,
        parallel=SimpleNamespace(
            strategy="velocity",
            backend="electrostatic_species_hermite",
            axis="species_hermite",
            num_devices=4,
        ),
        **integration,
    )
    np.testing.assert_allclose(
        np.asarray(mixed_state), np.asarray(serial_state), rtol=9e-5, atol=9e-6
    )
    np.testing.assert_allclose(
        np.asarray(mixed_phi), np.asarray(serial_phi), rtol=9e-5, atol=9e-6
    )


def test_electrostatic_phi_reference_matches_production_field_solve() -> None:
    from gkx.operators.linear.params import LinearTerms
    from gkx.operators.linear.rhs import linear_rhs_cached

    state, cache, params = _small_periodic_field_problem()
    terms = LinearTerms(
        streaming=0.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=0.0,
        hypercollisions=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )

    _rhs, phi = linear_rhs_cached(
        state, cache, params, terms=terms, use_jit=False, use_custom_vjp=False
    )
    observed = electrostatic_phi_reference(
        state,
        Jl=cache.Jl,
        tau_e=params.tau_e,
        charge=params.charge_sign,
        density=params.density,
        tz=params.tz,
        mask0=cache.mask0,
    )

    assert float(jnp.linalg.norm(phi)) > 0.0
    np.testing.assert_allclose(
        np.asarray(observed), np.asarray(phi), rtol=2.0e-6, atol=2.0e-6
    )
    assert gkx.electrostatic_phi_reference is electrostatic_phi_reference
    assert gkx.electrostatic_phi_shard_map is electrostatic_phi_shard_map


def test_electrostatic_phi_shard_map_noops_to_reference_for_single_chunk() -> None:
    state, cache, params = _small_periodic_field_problem()
    plan = build_velocity_sharding_plan(state.shape, num_devices=1, axes=("hermite",))

    observed = electrostatic_phi_shard_map(
        state,
        plan,
        Jl=cache.Jl,
        tau_e=params.tau_e,
        charge=params.charge_sign,
        density=params.density,
        tz=params.tz,
        mask0=cache.mask0,
        devices=[jax.devices()[0]],
    )
    expected = electrostatic_phi_reference(
        state,
        Jl=cache.Jl,
        tau_e=params.tau_e,
        charge=params.charge_sign,
        density=params.density,
        tz=params.tz,
        mask0=cache.mask0,
    )

    np.testing.assert_allclose(
        np.asarray(observed), np.asarray(expected), rtol=2.0e-6, atol=2.0e-6
    )


def test_electrostatic_phi_shard_map_matches_reference_when_logical_devices_available() -> (
    None
):
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip(
            "requires at least two JAX devices; artifact generator sets logical CPU devices"
        )
    state, cache, params = _small_periodic_field_problem()
    plan = build_velocity_sharding_plan(state.shape, num_devices=2, axes=("hermite",))

    observed = electrostatic_phi_shard_map(
        state,
        plan,
        Jl=cache.Jl,
        tau_e=params.tau_e,
        charge=params.charge_sign,
        density=params.density,
        tz=params.tz,
        mask0=cache.mask0,
        devices=devices[:2],
    )
    expected = electrostatic_phi_reference(
        state,
        Jl=cache.Jl,
        tau_e=params.tau_e,
        charge=params.charge_sign,
        density=params.density,
        tz=params.tz,
        mask0=cache.mask0,
    )

    np.testing.assert_allclose(
        np.asarray(observed), np.asarray(expected), rtol=2.0e-6, atol=2.0e-6
    )


def test_species_sharded_phi_matches_production_quasineutrality() -> None:
    from gkx.operators.linear.moments import quasineutrality_phi

    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("requires two logical CPU devices or two accelerators")
    single_state, cache, _params = _small_periodic_field_problem()
    state = jnp.stack((single_state, 0.35j * single_state), axis=0)
    jl = jnp.concatenate((cache.Jl, 0.8 * cache.Jl), axis=0)
    charge = jnp.asarray([1.0, -1.0], dtype=jnp.float32)
    density = jnp.asarray([1.0, 0.9], dtype=jnp.float32)
    tz = jnp.asarray([1.0, 0.5], dtype=jnp.float32)
    tau_e = 0.0
    plan = build_velocity_sharding_plan(state.shape, num_devices=2, axes=("species",))

    observed = electrostatic_phi_shard_map(
        state,
        plan,
        Jl=jl,
        tau_e=tau_e,
        charge=charge,
        density=density,
        tz=tz,
        mask0=cache.mask0,
        devices=devices[:2],
        axis_name="species",
    )
    expected = quasineutrality_phi(state, jl, tau_e, charge, density, tz)
    expected = jnp.where(cache.mask0, 0.0, expected)
    reference = electrostatic_phi_reference(
        state,
        Jl=jl,
        tau_e=tau_e,
        charge=charge,
        density=density,
        tz=tz,
        mask0=cache.mask0,
    )

    np.testing.assert_allclose(
        np.asarray(observed), np.asarray(expected), rtol=2e-6, atol=2e-6
    )
    np.testing.assert_allclose(
        np.asarray(reference), np.asarray(expected), rtol=2e-6, atol=2e-6
    )


def test_mixed_species_hermite_electrostatic_rhs_matches_serial_production_route() -> (
    None
):
    from gkx.operators.linear.rhs import linear_rhs_cached
    from gkx.solvers.linear.integrators import integrate_linear

    devices = jax.devices()
    if len(devices) < 4:
        pytest.skip("requires four logical CPU devices or accelerators")
    template, cache, params, _grid, _geom = _small_kinetic_electron_problem()
    rng = np.random.default_rng(20260712)
    state = jnp.asarray(
        1.0e-3
        * (
            rng.standard_normal(template.shape)
            + 1j * rng.standard_normal(template.shape)
        ),
        dtype=template.dtype,
    )
    terms = _streaming_only_terms()
    expected_rhs, expected_phi = linear_rhs_cached(
        state,
        cache,
        params,
        terms=terms,
        use_jit=False,
        use_custom_vjp=False,
        force_electrostatic_fields=True,
    )
    observed_rhs, observed_phi = (
        linear_parallel_streaming.linear_rhs_electrostatic_species_hermite_sharded(
            state,
            cache,
            params,
            terms=terms,
            species_chunks=2,
            hermite_chunks=2,
            devices=devices[:4],
        )
    )
    routed_rhs, routed_phi = linear_parallel.linear_rhs_parallel_cached(
        state,
        cache,
        params,
        terms=terms,
        parallel=SimpleNamespace(
            strategy="velocity",
            backend="auto",
            axis="species_hermite",
            num_devices=4,
        ),
    )

    for phi in (observed_phi, routed_phi):
        np.testing.assert_allclose(
            np.asarray(phi), np.asarray(expected_phi), rtol=3e-6, atol=3e-6
        )
    for rhs in (observed_rhs, routed_rhs):
        np.testing.assert_allclose(
            np.asarray(rhs), np.asarray(expected_rhs), rtol=4e-5, atol=4e-6
        )

    zero_terms = replace(
        _electrostatic_slice_terms(),
        streaming=0.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
    )
    term_cases = {
        name: replace(zero_terms, **{name: 1.0})
        for name in ("mirror", "curvature", "gradb", "diamagnetic")
    }
    term_cases["collision_free_electrostatic"] = _electrostatic_slice_terms()
    for name, active_terms in term_cases.items():
        expected_term_rhs, expected_term_phi = linear_rhs_cached(
            state,
            cache,
            params,
            terms=active_terms,
            use_jit=False,
            use_custom_vjp=False,
            force_electrostatic_fields=True,
        )
        observed_term_rhs, observed_term_phi = (
            linear_parallel_streaming.linear_rhs_electrostatic_species_hermite_sharded(
                state,
                cache,
                params,
                terms=active_terms,
                devices=devices[:4],
            )
        )
        np.testing.assert_allclose(
            np.asarray(observed_term_phi),
            np.asarray(expected_term_phi),
            rtol=3e-6,
            atol=3e-6,
            err_msg=name,
        )
        np.testing.assert_allclose(
            np.asarray(observed_term_rhs),
            np.asarray(expected_term_rhs),
            rtol=6e-5,
            atol=6e-6,
            err_msg=name,
        )

    dissipative_params = replace(
        params,
        nu_hyper=0.01,
        nu_hyper_l=0.03,
        nu_hyper_m=0.05,
        nu_hyper_lm=0.02,
        hypercollisions_const=1.0,
        hypercollisions_kz=0.4,
        D_hyper=0.02,
        damp_ends_amp=0.1,
        nu=jnp.asarray([0.1, 0.2]),
    )
    dissipative_cache = replace(
        cache,
        damp_profile=jnp.linspace(0.2, 1.0, cache.damp_profile.size),
    )
    dissipation_cases = {
        "conserving_collisions": replace(zero_terms, collisions=1.0),
        "hypercollisions": replace(zero_terms, hypercollisions=1.0),
        "hyperdiffusion": replace(zero_terms, hyperdiffusion=1.0),
        "end_damping": replace(zero_terms, end_damping=1.0),
        "combined_local_dissipation": replace(
            _electrostatic_slice_terms(),
            collisions=1.0,
            hypercollisions=1.0,
            hyperdiffusion=1.0,
            end_damping=1.0,
        ),
    }
    for name, active_terms in dissipation_cases.items():
        expected_term_rhs, expected_term_phi = linear_rhs_cached(
            state,
            dissipative_cache,
            dissipative_params,
            terms=active_terms,
            dt=0.01,
            use_jit=False,
            use_custom_vjp=False,
            force_electrostatic_fields=True,
        )
        observed_term_rhs, observed_term_phi = (
            linear_parallel_streaming.linear_rhs_electrostatic_species_hermite_sharded(
                state,
                dissipative_cache,
                dissipative_params,
                terms=active_terms,
                dt=0.01,
                devices=devices[:4],
            )
        )
        assert float(jnp.linalg.norm(expected_term_rhs)) > 0.0
        np.testing.assert_allclose(
            np.asarray(observed_term_phi),
            np.asarray(expected_term_phi),
            rtol=3e-6,
            atol=3e-6,
            err_msg=name,
        )
        np.testing.assert_allclose(
            np.asarray(observed_term_rhs),
            np.asarray(expected_term_rhs),
            rtol=7e-5,
            atol=7e-6,
            err_msg=name,
        )

    adiabatic_params = replace(params, tau_e=1.0)
    adiabatic_expected_rhs, adiabatic_expected_phi = linear_rhs_cached(
        state,
        cache,
        adiabatic_params,
        terms=terms,
        use_jit=False,
        use_custom_vjp=False,
        force_electrostatic_fields=True,
    )
    adiabatic_rhs, adiabatic_phi = (
        linear_parallel_streaming.linear_rhs_electrostatic_species_hermite_sharded(
            state,
            cache,
            adiabatic_params,
            terms=terms,
            devices=devices[:4],
        )
    )
    np.testing.assert_allclose(
        np.asarray(adiabatic_phi),
        np.asarray(adiabatic_expected_phi),
        rtol=3e-6,
        atol=3e-6,
    )
    np.testing.assert_allclose(
        np.asarray(adiabatic_rhs),
        np.asarray(adiabatic_expected_rhs),
        rtol=4e-5,
        atol=4e-6,
    )

    integration_terms = dissipation_cases["combined_local_dissipation"]
    integration = dict(
        dt=1e-5,
        steps=3,
        method="euler",
        cache=dissipative_cache,
        terms=integration_terms,
    )
    serial_state, serial_phi = integrate_linear(
        state, _grid, _geom, dissipative_params, **integration
    )
    mixed_state, mixed_phi = integrate_linear(
        state,
        _grid,
        _geom,
        dissipative_params,
        parallel=SimpleNamespace(
            strategy="velocity",
            backend="electrostatic_species_hermite",
            axis="species_hermite",
            num_devices=4,
        ),
        **integration,
    )
    np.testing.assert_allclose(
        np.asarray(mixed_state), np.asarray(serial_state), rtol=5e-5, atol=5e-6
    )
    np.testing.assert_allclose(
        np.asarray(mixed_phi), np.asarray(serial_phi), rtol=5e-5, atol=5e-6
    )
    rk2_integration = dict(
        dt=1e-5,
        steps=4,
        method="rk2",
        sample_stride=2,
        cache=dissipative_cache,
        terms=integration_terms,
    )
    serial_rk2, serial_rk2_phi = integrate_linear(
        state, _grid, _geom, dissipative_params, **rk2_integration
    )
    mixed_rk2, mixed_rk2_phi = integrate_linear(
        state,
        _grid,
        _geom,
        dissipative_params,
        parallel=SimpleNamespace(
            strategy="velocity",
            backend="electrostatic_species_hermite",
            axis="species_hermite",
            num_devices=4,
        ),
        **rk2_integration,
    )
    np.testing.assert_allclose(
        np.asarray(mixed_rk2), np.asarray(serial_rk2), rtol=6e-5, atol=6e-6
    )
    np.testing.assert_allclose(
        np.asarray(mixed_rk2_phi),
        np.asarray(serial_rk2_phi),
        rtol=6e-5,
        atol=6e-6,
    )
    with pytest.raises(NotImplementedError, match="Euler and RK2"):
        integrate_linear(
            state,
            _grid,
            _geom,
            dissipative_params,
            dt=1e-5,
            steps=1,
            method="rk4",
            cache=cache,
            terms=terms,
            parallel=SimpleNamespace(
                strategy="velocity",
                backend="electrostatic_species_hermite",
                axis="species_hermite",
                num_devices=4,
            ),
        )


def test_species_sharded_linear_rhs_matches_serial_production_route() -> None:
    from gkx.operators.linear.rhs import linear_rhs_cached

    def assert_species_close(observed, expected, *, rtol=5e-5, atol=5e-6):
        for species_index in range(int(expected.shape[0])):
            np.testing.assert_allclose(
                np.asarray(observed[species_index]),
                np.asarray(expected[species_index]),
                rtol=rtol,
                atol=atol,
            )

    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("requires two logical CPU devices or two accelerators")
    state, cache, params, grid, geom = _small_kinetic_electron_problem()
    terms = _electrostatic_slice_terms()

    expected_rhs, expected_phi = linear_rhs_cached(
        state,
        cache,
        params,
        terms=terms,
        use_jit=False,
        use_custom_vjp=False,
        force_electrostatic_fields=True,
    )
    observed_rhs, observed_phi = (
        linear_parallel_electrostatic.linear_rhs_electrostatic_species_sharded(
            state,
            cache,
            params,
            terms=terms,
            devices=devices[:2],
        )
    )
    parallel = SimpleNamespace(
        strategy="velocity", backend="auto", axis="species", num_devices=2
    )
    prepared_state, prepared_cache, prepared_params = (
        linear_parallel_electrostatic.prepare_electrostatic_species_inputs(
            state, cache, params, devices=devices[:2]
        )
    )
    routed_rhs, routed_phi = jax.jit(
        lambda value: linear_parallel.linear_rhs_parallel_cached(
            value, prepared_cache, prepared_params, terms=terms, parallel=parallel
        )
    )(prepared_state)

    np.testing.assert_allclose(
        np.asarray(observed_phi), np.asarray(expected_phi), rtol=3e-6, atol=3e-6
    )
    assert_species_close(observed_rhs, expected_rhs, rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(
        np.asarray(routed_phi), np.asarray(expected_phi), rtol=3e-6, atol=3e-6
    )
    assert_species_close(routed_rhs, expected_rhs, rtol=3e-5, atol=3e-5)

    from gkx.solvers.linear.integrators import integrate_linear

    serial_state, serial_phi = integrate_linear(
        state,
        grid,
        geom,
        params,
        dt=1e-4,
        steps=9,
        method="euler",
        cache=cache,
        terms=terms,
    )
    parallel_state, parallel_phi = integrate_linear(
        state,
        grid,
        geom,
        params,
        dt=1e-4,
        steps=9,
        method="euler",
        cache=cache,
        terms=terms,
        parallel=parallel,
    )
    assert_species_close(parallel_state, serial_state)
    np.testing.assert_allclose(
        np.asarray(parallel_phi), np.asarray(serial_phi), rtol=5e-5, atol=5e-6
    )

    serial_rk2, serial_rk2_phi = integrate_linear(
        state,
        grid,
        geom,
        params,
        dt=1e-5,
        steps=6,
        method="rk2",
        cache=cache,
        terms=terms,
        sample_stride=3,
    )
    parallel_rk2, parallel_rk2_phi = integrate_linear(
        state,
        grid,
        geom,
        params,
        dt=1e-5,
        steps=6,
        method="rk2",
        cache=cache,
        terms=terms,
        sample_stride=3,
        parallel=parallel,
    )
    assert parallel_rk2_phi.shape[0] == 2
    assert_species_close(parallel_rk2, serial_rk2)
    np.testing.assert_allclose(
        np.asarray(parallel_rk2_phi), np.asarray(serial_rk2_phi), rtol=5e-5, atol=5e-6
    )
    collision_params = replace(params, nu=jnp.asarray([0.1, 0.2]))
    collision_terms = LinearTerms(
        streaming=0.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=1.0,
        hypercollisions=0.0,
        hyperdiffusion=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )
    collision_serial_rhs, collision_serial_phi = linear_rhs_cached(
        state,
        cache,
        collision_params,
        terms=collision_terms,
        use_jit=False,
        use_custom_vjp=False,
        force_electrostatic_fields=True,
    )
    assert float(jnp.linalg.norm(collision_serial_rhs)) > 0.0
    assert float(jnp.linalg.norm(collision_serial_phi)) > 0.0
    with pytest.raises(NotImplementedError, match="electrostatic linear terms"):
        linear_parallel_electrostatic.linear_rhs_electrostatic_species_sharded(
            state,
            cache,
            collision_params,
            terms=collision_terms,
            devices=devices[:2],
        )
    collision_serial_state, _ = integrate_linear(
        state,
        grid,
        geom,
        collision_params,
        dt=1e-5,
        steps=3,
        method="euler",
        cache=cache,
        terms=collision_terms,
    )
    collision_parallel_state, _ = integrate_linear(
        state,
        grid,
        geom,
        collision_params,
        dt=1e-5,
        steps=3,
        method="euler",
        cache=cache,
        terms=collision_terms,
        parallel=parallel,
    )
    assert_species_close(collision_parallel_state, collision_serial_state)
    hyper_state = state.at[:, 1, 5, :, :, :].set(1.0e-3 + 2.0e-3j)
    hyper_params = replace(params, nu_hyper_l=0.03, nu_hyper_m=0.05)
    hyper_terms = replace(collision_terms, collisions=0.0, hypercollisions=1.0)
    hyper_rhs, _ = linear_rhs_cached(
        hyper_state,
        cache,
        hyper_params,
        terms=hyper_terms,
        use_jit=False,
        use_custom_vjp=False,
        force_electrostatic_fields=True,
    )
    assert float(jnp.linalg.norm(hyper_rhs)) > 0.0
    hyper_serial_state, _ = integrate_linear(
        hyper_state,
        grid,
        geom,
        hyper_params,
        dt=1e-5,
        steps=3,
        method="euler",
        cache=cache,
        terms=hyper_terms,
    )
    hyper_parallel_state, _ = integrate_linear(
        hyper_state,
        grid,
        geom,
        hyper_params,
        dt=1e-5,
        steps=3,
        method="euler",
        cache=cache,
        terms=hyper_terms,
        parallel=parallel,
    )
    assert_species_close(hyper_parallel_state, hyper_serial_state)
    with pytest.raises(NotImplementedError, match="species-parallel IMEX"):
        integrate_linear(
            state,
            grid,
            geom,
            params,
            dt=1e-5,
            steps=1,
            method="imex",
            cache=cache,
            terms=terms,
            parallel=parallel,
        )


def test_species_pmap_electromagnetic_trajectory_matches_serial() -> None:
    from gkx.solvers.linear.integrators import integrate_linear
    from gkx.operators.linear.params import linear_terms_to_term_config
    from gkx.terms.assembly import compute_fields_cached

    if len(jax.devices()) < 2:
        pytest.skip("requires two logical CPU devices or two accelerators")
    state, cache, params, grid, geom = _small_kinetic_electron_problem()
    ky = min(1, grid.ky.size - 1)
    state = state.at[:, 0, 1, ky, 0, :].set(0.03 + 0.02j)
    params = replace(params, beta=0.01, fapar=1.0)
    terms = replace(_electrostatic_slice_terms(), apar=1.0, bpar=1.0)
    fields = compute_fields_cached(
        state,
        cache,
        params,
        terms=linear_terms_to_term_config(terms),
        use_custom_vjp=False,
    )
    assert float(jnp.linalg.norm(fields.apar)) > 0.0
    assert float(jnp.linalg.norm(fields.bpar)) > 0.0

    integration = dict(
        dt=1e-6,
        steps=3,
        method="euler",
        cache=cache,
        terms=terms,
    )
    serial_state, serial_phi = integrate_linear(
        state, grid, geom, params, **integration
    )
    parallel_state, parallel_phi = integrate_linear(
        state,
        grid,
        geom,
        params,
        parallel=SimpleNamespace(
            strategy="velocity", backend="auto", axis="species", num_devices=2
        ),
        **integration,
    )
    np.testing.assert_allclose(
        np.asarray(parallel_state),
        np.asarray(serial_state),
        rtol=8e-5,
        atol=8e-6,
    )
    np.testing.assert_allclose(
        np.asarray(parallel_phi),
        np.asarray(serial_phi),
        rtol=8e-5,
        atol=8e-6,
    )


def test_species_pmap_collision_preserves_long_wavelength_moments() -> None:
    from gkx.solvers.linear.integrators import integrate_linear
    from gkx.operators.linear.dissipation import _laguerre_temperature_coupling

    if len(jax.devices()) < 2:
        pytest.skip("requires two logical CPU devices or two accelerators")
    template, cache, params, grid, geom = _small_kinetic_electron_problem()
    rng = np.random.default_rng(20260712)
    state = jnp.zeros_like(template)
    kperp_zero_shape = state[:, :, :, 0, 0, :].shape
    values = rng.standard_normal(kperp_zero_shape) + 1j * rng.standard_normal(
        kperp_zero_shape
    )
    state = state.at[:, :, :, 0, 0, :].set(
        jnp.asarray(1.0e-3 * values, dtype=state.dtype)
    )
    params = replace(params, nu=jnp.asarray([0.1, 0.2]))
    terms = LinearTerms(
        streaming=0.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=1.0,
        hypercollisions=0.0,
        hyperdiffusion=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )

    def invariant_moments(value):
        density = jnp.sum(cache.Jl * value[:, :, 0, ...], axis=1)[:, 0, 0, :]
        momentum = jnp.sum(cache.Jl * value[:, :, 1, ...], axis=1)[:, 0, 0, :]
        _coefficients, temperature = _laguerre_temperature_coupling(
            value[:, :, 0, ...],
            value[:, :, 2, ...],
            cache.Jl,
        )
        return density, momentum, temperature[:, 0, 0, :]

    integration = dict(
        dt=1e-4,
        steps=5,
        method="euler",
        cache=cache,
        terms=terms,
    )
    serial_state, _ = integrate_linear(state, grid, geom, params, **integration)
    parallel_state, _ = integrate_linear(
        state,
        grid,
        geom,
        params,
        parallel=SimpleNamespace(
            strategy="velocity", backend="auto", axis="species", num_devices=2
        ),
        **integration,
    )
    assert float(jnp.linalg.norm(serial_state - state)) > 0.0
    for expected, serial, parallel in zip(
        invariant_moments(state),
        invariant_moments(serial_state),
        invariant_moments(parallel_state),
        strict=True,
    ):
        np.testing.assert_allclose(
            np.asarray(serial), np.asarray(expected), rtol=2e-5, atol=2e-9
        )
        np.testing.assert_allclose(
            np.asarray(parallel), np.asarray(expected), rtol=2e-5, atol=2e-9
        )


def test_species_pmap_parameter_gradient_matches_centered_difference() -> None:
    from gkx.solvers.linear.integrators import integrate_linear

    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("requires two logical CPU devices or two accelerators")
    state, cache, params, grid, geom = _small_kinetic_electron_problem()
    terms = _electrostatic_slice_terms()
    parallel = SimpleNamespace(
        strategy="velocity", backend="auto", axis="species", num_devices=2
    )
    prepared_state, prepared_cache, prepared_params = (
        linear_parallel_electrostatic.prepare_electrostatic_species_inputs(
            state,
            cache,
            params,
            devices=devices[:2],
            replicate_cache=False,
        )
    )

    def objective(ion_temperature_gradient):
        dynamic_params = replace(
            prepared_params,
            tprim=jnp.stack(
                (ion_temperature_gradient, jnp.zeros_like(ion_temperature_gradient))
            ),
        )
        final_state, _fields = integrate_linear(
            prepared_state,
            grid,
            geom,
            dynamic_params,
            dt=5.0e-4,
            steps=2,
            method="euler",
            cache=prepared_cache,
            terms=terms,
            parallel=parallel,
        )
        mode = final_state[0, 0, 0, 1, 0, :]
        probe = jnp.exp(1j * jnp.linspace(0.0, 1.0, mode.size, dtype=mode.real.dtype))
        return jnp.real(jnp.vdot(probe, mode))

    point = jnp.asarray(6.9, dtype=jnp.float32)
    tangent = jax.grad(objective)(point)
    eps = jnp.asarray(5.0e-2, dtype=point.dtype)
    finite_difference = (objective(point + eps) - objective(point - eps)) / (2.0 * eps)

    assert float(jnp.abs(tangent)) > 1.0e-12
    np.testing.assert_allclose(
        np.asarray(tangent), np.asarray(finite_difference), rtol=1.0e-2, atol=1.0e-8
    )


def test_electrostatic_phi_rejects_invalid_shapes_and_plans() -> None:
    state, cache, params = _small_periodic_field_problem()
    plan = build_velocity_sharding_plan(state.shape, num_devices=2, axes=("hermite",))
    bad_shape = state[:, :5, ...]
    uneven = replace(plan, state_shape=bad_shape.shape)
    laguerre_plan = build_velocity_sharding_plan(
        state.shape, num_devices=2, axes=("laguerre",)
    )

    multi_reference = electrostatic_phi_reference(
        state[None, ...], Jl=cache.Jl, tau_e=params.tau_e
    )
    single_reference = electrostatic_phi_reference(
        state, Jl=cache.Jl, tau_e=params.tau_e
    )
    np.testing.assert_allclose(
        np.asarray(multi_reference), np.asarray(single_reference)
    )
    with pytest.raises(ValueError, match="Jl"):
        electrostatic_phi_reference(
            state,
            Jl=jnp.ones((state.shape[0], state.shape[2], state.shape[4])),
            tau_e=params.tau_e,
        )
    species_plan = build_velocity_sharding_plan(
        state[None, ...].shape, num_devices=1, axes=("species",)
    )
    multi_sharded = electrostatic_phi_shard_map(
        state[None, ...], species_plan, Jl=cache.Jl, tau_e=params.tau_e
    )
    np.testing.assert_allclose(np.asarray(multi_sharded), np.asarray(single_reference))
    with pytest.raises(ValueError, match="state shape"):
        electrostatic_phi_shard_map(bad_shape, plan, Jl=cache.Jl, tau_e=params.tau_e)
    with pytest.raises(NotImplementedError, match="active 'm'"):
        electrostatic_phi_shard_map(
            state, laguerre_plan, Jl=cache.Jl, tau_e=params.tau_e
        )
    with pytest.raises(ValueError, match="divide evenly"):
        electrostatic_phi_shard_map(
            bad_shape,
            uneven,
            Jl=cache.Jl,
            tau_e=params.tau_e,
            devices=[object(), object()],
        )
    with pytest.raises(ValueError, match="Jl"):
        electrostatic_phi_shard_map(
            state,
            plan,
            Jl=jnp.ones((state.shape[0], state.shape[2], state.shape[4])),
            tau_e=params.tau_e,
        )
    with pytest.raises(ValueError, match="not enough devices"):
        electrostatic_phi_shard_map(
            state, plan, Jl=cache.Jl, tau_e=params.tau_e, devices=[object()]
        )


def test_mirror_and_curvature_gradb_drift_shard_maps_match_production_terms() -> None:
    from gkx.operators.linear.moments import build_H
    from gkx.terms.linear_terms import (
        curvature_gradb_contribution,
        mirror_contribution,
    )

    state, cache, params = _small_periodic_field_problem()
    phi = electrostatic_phi_reference(
        state,
        Jl=cache.Jl,
        tau_e=params.tau_e,
        charge=params.charge_sign,
        density=params.density,
        tz=params.tz,
        mask0=cache.mask0,
    )
    H = build_H(state, cache.Jl, phi, jnp.asarray([params.tz]))
    plan = build_velocity_sharding_plan(H.shape, num_devices=1, axes=("hermite",))
    vth = jnp.asarray([params.vth], dtype=jnp.float32)
    tz = jnp.asarray([params.tz], dtype=jnp.float32)

    mirror_expected = mirror_contribution(
        H[None, ...],
        vth=vth,
        bgrad=cache.bgrad,
        ell=cache.l,
        sqrt_m=cache.sqrt_m,
        sqrt_m_p1=cache.sqrt_m_p1,
        weight=jnp.asarray(1.0, dtype=jnp.float32),
    )[0]
    mirror_observed = mirror_drift_shard_map(
        H,
        plan,
        vth=vth,
        bgrad=cache.bgrad,
        ell=cache.l,
        sqrt_m=cache.sqrt_m,
        sqrt_m_p1=cache.sqrt_m_p1,
        devices=[jax.devices()[0]],
    )
    np.testing.assert_allclose(
        np.asarray(mirror_observed),
        np.asarray(mirror_expected),
        rtol=2.0e-6,
        atol=2.0e-6,
    )

    drift_expected = curvature_gradb_contribution(
        H[None, ...],
        tz=tz,
        omega_d_scale=jnp.asarray(params.omega_d_scale, dtype=jnp.float32),
        cv_d=cache.cv_d,
        gb_d=cache.gb_d,
        ell=cache.l,
        m=cache.m,
        imag=jnp.asarray(1j, dtype=H.dtype),
        weight_curv=jnp.asarray(1.0, dtype=jnp.float32),
        weight_gradb=jnp.asarray(1.0, dtype=jnp.float32),
    )[0]
    drift_observed = curvature_gradb_drift_shard_map(
        H,
        plan,
        tz=tz,
        omega_d_scale=params.omega_d_scale,
        cv_d=cache.cv_d,
        gb_d=cache.gb_d,
        ell=cache.l,
        m=cache.m,
        devices=[jax.devices()[0]],
    )
    np.testing.assert_allclose(
        np.asarray(drift_observed), np.asarray(drift_expected), rtol=2.0e-6, atol=2.0e-6
    )
    assert gkx.mirror_drift_reference is mirror_drift_reference
    assert gkx.mirror_drift_shard_map is mirror_drift_shard_map
    assert gkx.curvature_gradb_drift_reference is curvature_gradb_drift_reference
    assert gkx.curvature_gradb_drift_shard_map is curvature_gradb_drift_shard_map


def test_mirror_and_curvature_gradb_drift_shard_maps_match_reference_when_logical_devices_available() -> (
    None
):
    from gkx.operators.linear.moments import build_H

    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip(
            "requires at least two JAX devices; artifact generator sets logical CPU devices"
        )
    state, cache, params = _small_periodic_field_problem()
    phi = electrostatic_phi_reference(
        state,
        Jl=cache.Jl,
        tau_e=params.tau_e,
        charge=params.charge_sign,
        density=params.density,
        tz=params.tz,
        mask0=cache.mask0,
    )
    H = build_H(state, cache.Jl, phi, jnp.asarray([params.tz]))
    plan = build_velocity_sharding_plan(H.shape, num_devices=2, axes=("hermite",))
    vth = jnp.asarray([params.vth], dtype=jnp.float32)
    tz = jnp.asarray([params.tz], dtype=jnp.float32)

    mirror_observed = mirror_drift_shard_map(
        H,
        plan,
        vth=vth,
        bgrad=cache.bgrad,
        ell=cache.l,
        sqrt_m=cache.sqrt_m,
        sqrt_m_p1=cache.sqrt_m_p1,
        devices=devices[:2],
    )
    mirror_expected = mirror_drift_reference(
        H,
        vth=vth,
        bgrad=cache.bgrad,
        ell=cache.l,
        sqrt_m=cache.sqrt_m,
        sqrt_m_p1=cache.sqrt_m_p1,
    )
    np.testing.assert_allclose(
        np.asarray(mirror_observed),
        np.asarray(mirror_expected),
        rtol=2.0e-6,
        atol=2.0e-6,
    )

    drift_observed = curvature_gradb_drift_shard_map(
        H,
        plan,
        tz=tz,
        omega_d_scale=params.omega_d_scale,
        cv_d=cache.cv_d,
        gb_d=cache.gb_d,
        ell=cache.l,
        m=cache.m,
        devices=devices[:2],
    )
    drift_expected = curvature_gradb_drift_reference(
        H,
        tz=tz,
        omega_d_scale=params.omega_d_scale,
        cv_d=cache.cv_d,
        gb_d=cache.gb_d,
        ell=cache.l,
        m=cache.m,
    )
    np.testing.assert_allclose(
        np.asarray(drift_observed), np.asarray(drift_expected), rtol=2.0e-6, atol=2.0e-6
    )


def test_six_dimensional_mirror_and_drift_paths_preserve_species_broadcasts() -> None:
    state = (
        jnp.arange(2 * 2 * 4 * 2 * 1 * 3, dtype=jnp.float32).reshape((2, 2, 4, 2, 1, 3))
        + 1j
    ).astype(jnp.complex64)
    plan = build_velocity_sharding_plan(state.shape, num_devices=1, axes=("hermite",))
    ell = jnp.arange(state.shape[1], dtype=jnp.float32).reshape(
        (1, state.shape[1], 1, 1, 1, 1)
    )
    m = jnp.arange(state.shape[2], dtype=jnp.float32).reshape(
        (1, 1, state.shape[2], 1, 1, 1)
    )
    sqrt_m = jnp.sqrt(m)
    sqrt_m_p1 = jnp.sqrt(m + 1.0)
    vth = jnp.asarray([1.0, 1.5], dtype=jnp.float32)
    tz = jnp.asarray([1.0, -1.0], dtype=jnp.float32)
    cv_d = jnp.ones((state.shape[3], state.shape[4], state.shape[5]), dtype=jnp.float32)
    gb_d = 0.25 * cv_d

    mirror_expected = mirror_drift_reference(
        state,
        vth=vth,
        bgrad=jnp.ones((state.shape[-1],), dtype=jnp.float32),
        ell=ell,
        sqrt_m=sqrt_m,
        sqrt_m_p1=sqrt_m_p1,
    )
    mirror_observed = mirror_drift_shard_map(
        state,
        plan,
        vth=vth,
        bgrad=jnp.ones((state.shape[-1],), dtype=jnp.float32),
        ell=ell,
        sqrt_m=sqrt_m,
        sqrt_m_p1=sqrt_m_p1,
        devices=[jax.devices()[0]],
    )
    np.testing.assert_allclose(
        np.asarray(mirror_observed),
        np.asarray(mirror_expected),
        rtol=2.0e-6,
        atol=2.0e-6,
    )

    drift_expected = curvature_gradb_drift_reference(
        state,
        tz=tz,
        omega_d_scale=0.7,
        cv_d=cv_d,
        gb_d=gb_d,
        ell=ell,
        m=m,
    )
    drift_observed = curvature_gradb_drift_shard_map(
        state,
        plan,
        tz=tz,
        omega_d_scale=0.7,
        cv_d=cv_d,
        gb_d=gb_d,
        ell=ell,
        m=m,
        devices=[jax.devices()[0]],
    )
    assert drift_observed.shape == state.shape
    np.testing.assert_allclose(
        np.asarray(drift_observed), np.asarray(drift_expected), rtol=2.0e-6, atol=2.0e-6
    )


def test_diamagnetic_drive_shard_map_matches_production_term() -> None:
    from gkx.operators.linear.params import LinearTerms
    from gkx.operators.linear.rhs import linear_rhs_cached

    state, cache, params = _small_periodic_field_problem()
    phi = electrostatic_phi_reference(
        state,
        Jl=cache.Jl,
        tau_e=params.tau_e,
        charge=params.charge_sign,
        density=params.density,
        tz=params.tz,
        mask0=cache.mask0,
    )
    plan = build_velocity_sharding_plan(state.shape, num_devices=1, axes=("hermite",))

    observed = diamagnetic_drive_shard_map(
        state,
        plan,
        phi=phi,
        Jl=cache.Jl,
        b=cache.b,
        l4=cache.l4,
        tprim=params.tprim,
        fprim=params.fprim,
        omega_star_scale=params.omega_star_scale,
        ky=cache.ky,
        devices=[jax.devices()[0]],
    )
    expected, _phi = linear_rhs_cached(
        state,
        cache,
        params,
        terms=LinearTerms(
            streaming=0.0,
            mirror=0.0,
            curvature=0.0,
            gradb=0.0,
            diamagnetic=1.0,
            collisions=0.0,
            hypercollisions=0.0,
            end_damping=0.0,
            apar=0.0,
            bpar=0.0,
        ),
        use_jit=False,
        use_custom_vjp=False,
    )

    np.testing.assert_allclose(
        np.asarray(observed), np.asarray(expected), rtol=2.0e-6, atol=2.0e-6
    )
    assert gkx.diamagnetic_drive_reference is diamagnetic_drive_reference
    assert gkx.diamagnetic_drive_shard_map is diamagnetic_drive_shard_map


def test_diamagnetic_drive_shard_map_matches_reference_when_logical_devices_available() -> (
    None
):
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip(
            "requires at least two JAX devices; artifact generator sets logical CPU devices"
        )
    state, cache, params = _small_periodic_field_problem()
    phi = electrostatic_phi_reference(
        state,
        Jl=cache.Jl,
        tau_e=params.tau_e,
        charge=params.charge_sign,
        density=params.density,
        tz=params.tz,
        mask0=cache.mask0,
    )
    plan = build_velocity_sharding_plan(state.shape, num_devices=2, axes=("hermite",))

    observed = diamagnetic_drive_shard_map(
        state,
        plan,
        phi=phi,
        Jl=cache.Jl,
        b=cache.b,
        l4=cache.l4,
        tprim=params.tprim,
        fprim=params.fprim,
        omega_star_scale=params.omega_star_scale,
        ky=cache.ky,
        devices=devices[:2],
    )
    expected = diamagnetic_drive_reference(
        state,
        phi=phi,
        Jl=cache.Jl,
        b=cache.b,
        l4=cache.l4,
        tprim=params.tprim,
        fprim=params.fprim,
        omega_star_scale=params.omega_star_scale,
        ky=cache.ky,
    )

    np.testing.assert_allclose(
        np.asarray(observed), np.asarray(expected), rtol=2.0e-6, atol=2.0e-6
    )


def test_diamagnetic_drive_rejects_invalid_shapes_and_plans() -> None:
    state, cache, params = _small_periodic_field_problem()
    phi = electrostatic_phi_reference(
        state,
        Jl=cache.Jl,
        tau_e=params.tau_e,
        charge=params.charge_sign,
        density=params.density,
        tz=params.tz,
        mask0=cache.mask0,
    )
    plan = build_velocity_sharding_plan(state.shape, num_devices=2, axes=("hermite",))
    bad_shape = state[:, :5, ...]
    uneven = replace(plan, state_shape=bad_shape.shape)
    laguerre_plan = build_velocity_sharding_plan(
        state.shape, num_devices=2, axes=("laguerre",)
    )
    kwargs = dict(
        phi=phi,
        Jl=cache.Jl,
        b=cache.b,
        l4=cache.l4,
        tprim=params.tprim,
        fprim=params.fprim,
        omega_star_scale=params.omega_star_scale,
        ky=cache.ky,
    )

    with pytest.raises(NotImplementedError, match="single-species"):
        diamagnetic_drive_reference(state[None, ...], **kwargs)
    with pytest.raises(ValueError, match="Jl"):
        diamagnetic_drive_reference(
            state,
            **{
                **kwargs,
                "Jl": jnp.ones((state.shape[0], state.shape[2], state.shape[4])),
            },
        )
    jl4 = cache.Jl[0] if cache.Jl.ndim == 5 else cache.Jl
    assert (
        diamagnetic_drive_reference(state, **{**kwargs, "Jl": jl4[None, ...]}).shape
        == state.shape
    )
    with pytest.raises(NotImplementedError, match="single-species"):
        diamagnetic_drive_shard_map(state[None, ...], plan, **kwargs)
    with pytest.raises(ValueError, match="state shape"):
        diamagnetic_drive_shard_map(bad_shape, plan, **kwargs)
    with pytest.raises(NotImplementedError, match="active 'm'"):
        diamagnetic_drive_shard_map(state, laguerre_plan, **kwargs)
    with pytest.raises(ValueError, match="divide evenly"):
        diamagnetic_drive_shard_map(
            bad_shape, uneven, devices=[object(), object()], **kwargs
        )
    with pytest.raises(ValueError, match="not enough devices"):
        diamagnetic_drive_shard_map(state, plan, devices=[object()], **kwargs)


def test_hermite_streaming_ladder_reference_matches_manual_coefficients() -> None:
    state = jnp.zeros((1, 4, 1, 1, 1), dtype=jnp.complex64)
    state = state.at[0, 2, 0, 0, 0].set(3.0 + 2.0j)

    ladder = hermite_streaming_ladder_reference(state, vth=2.0)

    expected = np.zeros_like(np.asarray(state))
    expected[0, 1, 0, 0, 0] = 2.0 * np.sqrt(2.0) * (3.0 + 2.0j)
    expected[0, 3, 0, 0, 0] = 2.0 * np.sqrt(3.0) * (3.0 + 2.0j)
    np.testing.assert_allclose(np.asarray(ladder), expected, rtol=1.0e-6, atol=1.0e-6)
    assert (
        gkx.hermite_streaming_ladder_reference
        is hermite_streaming_ladder_reference
    )
    assert (
        gkx.hermite_streaming_ladder_shard_map
        is hermite_streaming_ladder_shard_map
    )


def test_hermite_streaming_ladder_shard_map_noops_to_reference_for_single_chunk() -> (
    None
):
    state = (
        jnp.arange(2 * 5 * 3 * 1 * 4, dtype=jnp.float32).reshape((2, 5, 3, 1, 4)) + 1j
    ).astype(jnp.complex64)
    plan = build_velocity_sharding_plan(state.shape, num_devices=1, axes=("hermite",))

    ladder = hermite_streaming_ladder_shard_map(
        state, plan, vth=1.5, devices=[jax.devices()[0]]
    )
    expected = hermite_streaming_ladder_reference(state, vth=1.5)

    np.testing.assert_allclose(
        np.asarray(ladder), np.asarray(expected), rtol=1.0e-6, atol=1.0e-6
    )


def test_hermite_streaming_ladder_shard_map_matches_reference_when_logical_devices_available() -> (
    None
):
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip(
            "requires at least two JAX devices; artifact generator sets logical CPU devices"
        )
    state = (
        jnp.arange(2 * 6 * 3 * 1 * 4, dtype=jnp.float32).reshape((2, 6, 3, 1, 4)) + 1j
    ).astype(jnp.complex64)
    plan = build_velocity_sharding_plan(state.shape, num_devices=2, axes=("hermite",))

    ladder = hermite_streaming_ladder_shard_map(
        state, plan, vth=1.5, devices=devices[:2]
    )
    expected = hermite_streaming_ladder_reference(state, vth=1.5)

    np.testing.assert_allclose(
        np.asarray(ladder), np.asarray(expected), rtol=1.0e-6, atol=1.0e-6
    )


def test_periodic_streaming_reference_matches_production_streaming_term() -> None:
    from gkx.core.velocity import hermite_ladder_coeffs
    from gkx.operators.linear.streaming import streaming_ladder_term

    ns, nl, nm, ny, nx, nz = 1, 2, 4, 2, 1, 8
    z = jnp.linspace(0.0, 2.0 * jnp.pi, nz, endpoint=False)
    kz = jnp.fft.fftfreq(nz, d=float(z[1] - z[0])) * 2.0 * jnp.pi
    state = jnp.zeros((ns, nl, nm, ny, nx, nz), dtype=jnp.complex64)
    state = state.at[0, 0, 1, 0, 0, :].set(jnp.exp(1j * z))
    sqrt_p, sqrt_m = hermite_ladder_coeffs(nm - 1)
    sqrt_p = sqrt_p[:nm].reshape((1, 1, nm, 1, 1, 1))
    sqrt_m = sqrt_m[:nm].reshape((1, 1, nm, 1, 1, 1))
    vth = jnp.asarray([1.5], dtype=jnp.float32)

    observed = periodic_streaming_reference(state, kz=kz, vth=vth)
    expected = streaming_ladder_term(
        state,
        kz=kz,
        vth=vth.reshape((1, 1, 1, 1, 1, 1)),
        sqrt_p=sqrt_p,
        sqrt_m=sqrt_m,
    )

    np.testing.assert_allclose(
        np.asarray(observed), np.asarray(expected), rtol=1.0e-6, atol=1.0e-6
    )
    assert gkx.periodic_streaming_reference is periodic_streaming_reference
    assert gkx.periodic_streaming_shard_map is periodic_streaming_shard_map


def test_periodic_streaming_shard_map_noops_to_reference_for_single_chunk() -> None:
    nz = 8
    z = jnp.linspace(0.0, 2.0 * jnp.pi, nz, endpoint=False)
    kz = jnp.fft.fftfreq(nz, d=float(z[1] - z[0])) * 2.0 * jnp.pi
    state = jnp.zeros((1, 2, 5, 2, 1, nz), dtype=jnp.complex64)
    state = state.at[0, 0, 2, 0, 0, :].set(jnp.exp(2j * z))
    plan = build_velocity_sharding_plan(state.shape, num_devices=1, axes=("hermite",))

    observed = periodic_streaming_shard_map(
        state, plan, kz=kz, vth=jnp.asarray([1.2]), devices=[jax.devices()[0]]
    )
    expected = periodic_streaming_reference(state, kz=kz, vth=jnp.asarray([1.2]))

    np.testing.assert_allclose(
        np.asarray(observed), np.asarray(expected), rtol=1.0e-6, atol=1.0e-6
    )


def test_periodic_streaming_shard_map_matches_reference_when_logical_devices_available() -> (
    None
):
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip(
            "requires at least two JAX devices; artifact generator sets logical CPU devices"
        )
    nz = 8
    z = jnp.linspace(0.0, 2.0 * jnp.pi, nz, endpoint=False)
    kz = jnp.fft.fftfreq(nz, d=float(z[1] - z[0])) * 2.0 * jnp.pi
    state = jnp.zeros((1, 2, 6, 2, 1, nz), dtype=jnp.complex64)
    state = state.at[0, 0, 2, 0, 0, :].set(jnp.exp(2j * z))
    plan = build_velocity_sharding_plan(state.shape, num_devices=2, axes=("hermite",))

    observed = periodic_streaming_shard_map(
        state, plan, kz=kz, vth=jnp.asarray([1.2]), devices=devices[:2]
    )
    expected = periodic_streaming_reference(state, kz=kz, vth=jnp.asarray([1.2]))

    np.testing.assert_allclose(
        np.asarray(observed), np.asarray(expected), rtol=1.0e-6, atol=1.0e-6
    )


def test_linear_rhs_parallel_cached_streaming_only_matches_serial_call_graph() -> None:
    from gkx.config import CycloneBaseCase, GridConfig
    from gkx.geometry import SAlphaGeometry
    from gkx.core.grid import build_spectral_grid
    from gkx.operators.linear.cache_builder import build_linear_cache
    from gkx.operators.linear.params import LinearParams, LinearTerms
    from gkx.operators.linear.rhs import linear_rhs_cached
    from gkx.solvers.linear.parallel import linear_rhs_parallel_cached
    from gkx.workflows.runtime.config import RuntimeParallelConfig

    cfg = CycloneBaseCase(
        grid=GridConfig(Nx=1, Ny=4, Nz=8, Lx=6.0, Ly=6.0, boundary="periodic")
    )
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=4)
    z = jnp.linspace(0.0, 2.0 * jnp.pi, grid.z.size, endpoint=False)
    state = jnp.zeros(
        (2, 4, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64
    )
    state = state.at[0, 2, min(1, grid.ky.size - 1), 0, :].set(jnp.exp(1j * z))
    terms = LinearTerms(
        streaming=1.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=0.0,
        hypercollisions=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )
    parallel = RuntimeParallelConfig(
        strategy="velocity", backend="streaming_only", axis="hermite", num_devices=1
    )

    serial, phi_serial = linear_rhs_cached(
        state, cache, params, terms=terms, use_jit=False
    )
    sharded, phi_sharded = linear_rhs_parallel_cached(
        state, cache, params, terms=terms, parallel=parallel
    )

    np.testing.assert_allclose(
        np.asarray(sharded), np.asarray(serial), rtol=2.0e-6, atol=2.0e-6
    )
    np.testing.assert_allclose(
        np.asarray(phi_sharded), np.asarray(phi_serial), rtol=0.0, atol=0.0
    )
    assert gkx.linear_rhs_parallel_cached is linear_rhs_parallel_cached
    assert (
        gkx.linear_rhs_streaming_velocity_sharded(
            state, cache, params, num_devices=1
        )[0].shape
        == state.shape
    )


def test_linear_rhs_parallel_cached_rejects_non_streaming_velocity_route() -> None:
    from gkx.operators.linear.params import LinearParams, LinearTerms
    from gkx.solvers.linear.parallel import linear_rhs_parallel_cached
    from gkx.workflows.runtime.config import RuntimeParallelConfig

    class Cache:
        kz = jnp.asarray([0.0, 1.0, -1.0])

    state = jnp.zeros((1, 4, 1, 1, 3), dtype=jnp.complex64)
    parallel = RuntimeParallelConfig(
        strategy="velocity", backend="streaming_only", axis="hermite", num_devices=1
    )

    with pytest.raises(NotImplementedError, match="streaming-only LinearTerms"):
        linear_rhs_parallel_cached(
            state, Cache(), LinearParams(), terms=LinearTerms(), parallel=parallel
        )


def test_linear_rhs_parallel_cached_electrostatic_streaming_matches_serial_call_graph() -> (
    None
):
    from gkx.config import CycloneBaseCase, GridConfig
    from gkx.geometry import SAlphaGeometry
    from gkx.core.grid import build_spectral_grid
    from gkx.operators.linear.cache_builder import build_linear_cache
    from gkx.operators.linear.params import LinearParams, LinearTerms
    from gkx.operators.linear.rhs import linear_rhs_cached
    from gkx.solvers.linear.parallel import linear_rhs_parallel_cached
    from gkx.workflows.runtime.config import RuntimeParallelConfig

    cfg = CycloneBaseCase(
        grid=GridConfig(Nx=1, Ny=4, Nz=8, Lx=6.0, Ly=6.0, boundary="periodic")
    )
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(beta=0.0, fapar=0.0)
    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=4)
    z = jnp.linspace(0.0, 2.0 * jnp.pi, grid.z.size, endpoint=False)
    state = jnp.zeros(
        (2, 4, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64
    )
    state = state.at[0, 0, min(1, grid.ky.size - 1), 0, :].set(0.2 * jnp.exp(1j * z))
    state = state.at[1, 2, min(1, grid.ky.size - 1), 0, :].set(0.1 * jnp.exp(2j * z))
    terms = LinearTerms(
        streaming=1.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=0.0,
        hypercollisions=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )
    parallel = RuntimeParallelConfig(
        strategy="velocity",
        backend="streaming_electrostatic",
        axis="hermite",
        num_devices=1,
    )

    serial, phi_serial = linear_rhs_cached(
        state, cache, params, terms=terms, use_jit=False, use_custom_vjp=False
    )
    sharded, phi_sharded = linear_rhs_parallel_cached(
        state,
        cache,
        params,
        terms=terms,
        parallel=parallel,
        use_custom_vjp=False,
    )

    assert float(jnp.linalg.norm(phi_serial)) > 0.0
    np.testing.assert_allclose(
        np.asarray(phi_sharded), np.asarray(phi_serial), rtol=2.0e-6, atol=2.0e-6
    )
    np.testing.assert_allclose(
        np.asarray(sharded), np.asarray(serial), rtol=2.0e-6, atol=2.0e-6
    )
    assert (
        gkx.linear_rhs_streaming_electrostatic_velocity_sharded(
            state, cache, params, num_devices=1
        )[0].shape
        == state.shape
    )


def test_linear_rhs_parallel_cached_electrostatic_linear_slices_match_serial_call_graph() -> (
    None
):
    from gkx.operators.linear.params import LinearTerms
    from gkx.operators.linear.rhs import linear_rhs_cached
    from gkx.solvers.linear.parallel import linear_rhs_parallel_cached
    from gkx.workflows.runtime.config import RuntimeParallelConfig

    state, cache, params = _small_periodic_field_problem()
    z = jnp.linspace(0.0, 2.0 * jnp.pi, state.shape[-1], endpoint=False)
    state = state.at[0, 2, min(1, state.shape[2] - 1), 0, :].set(0.07 * jnp.exp(2j * z))
    state = state.at[1, 3, min(1, state.shape[2] - 1), 0, :].set(0.03 * jnp.exp(3j * z))
    terms = LinearTerms(
        streaming=1.0,
        mirror=1.0,
        curvature=1.0,
        gradb=1.0,
        diamagnetic=1.0,
        collisions=0.0,
        hypercollisions=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )
    parallel = RuntimeParallelConfig(
        strategy="velocity",
        backend="electrostatic_linear_slices",
        axis="hermite",
        num_devices=1,
    )

    serial, phi_serial = linear_rhs_cached(
        state, cache, params, terms=terms, use_jit=False, use_custom_vjp=False
    )
    sharded, phi_sharded = linear_rhs_parallel_cached(
        state,
        cache,
        params,
        terms=terms,
        parallel=parallel,
        use_custom_vjp=False,
    )

    np.testing.assert_allclose(
        np.asarray(phi_sharded), np.asarray(phi_serial), rtol=2.0e-6, atol=2.0e-6
    )
    np.testing.assert_allclose(
        np.asarray(sharded), np.asarray(serial), rtol=2.0e-6, atol=2.0e-6
    )
    assert (
        gkx.linear_rhs_electrostatic_slices_velocity_sharded(
            state, cache, params, terms=terms, num_devices=1
        )[0].shape
        == state.shape
    )


def test_linear_rhs_parallel_cached_auto_backend_selects_gated_electrostatic_slices() -> (
    None
):
    from gkx.operators.linear.params import LinearTerms
    from gkx.operators.linear.rhs import linear_rhs_cached
    from gkx.solvers.linear.parallel import linear_rhs_parallel_cached
    from gkx.workflows.runtime.config import RuntimeParallelConfig

    state, cache, params = _small_periodic_field_problem()
    z = jnp.linspace(0.0, 2.0 * jnp.pi, state.shape[-1], endpoint=False)
    state = state.at[0, 2, min(1, state.shape[2] - 1), 0, :].set(0.07 * jnp.exp(2j * z))
    terms = LinearTerms(
        streaming=1.0,
        mirror=1.0,
        curvature=1.0,
        gradb=1.0,
        diamagnetic=1.0,
        collisions=0.0,
        hypercollisions=0.0,
        hyperdiffusion=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )
    parallel = RuntimeParallelConfig(
        strategy="velocity", backend="auto", axis="hermite", num_devices=1
    )

    serial, phi_serial = linear_rhs_cached(
        state, cache, params, terms=terms, use_jit=False, use_custom_vjp=False
    )
    sharded, phi_sharded = linear_rhs_parallel_cached(
        state,
        cache,
        params,
        terms=terms,
        parallel=parallel,
        use_custom_vjp=False,
    )

    np.testing.assert_allclose(
        np.asarray(phi_sharded), np.asarray(phi_serial), rtol=2.0e-6, atol=2.0e-6
    )
    np.testing.assert_allclose(
        np.asarray(sharded), np.asarray(serial), rtol=2.0e-6, atol=2.0e-6
    )


def test_linear_rhs_parallel_cached_electrostatic_linear_slices_rejects_ungated_terms() -> (
    None
):
    from gkx.operators.linear.params import LinearParams, LinearTerms
    from gkx.solvers.linear.parallel import linear_rhs_parallel_cached
    from gkx.workflows.runtime.config import RuntimeParallelConfig

    class Cache:
        use_twist_shift = False

    state = jnp.zeros((1, 4, 1, 1, 3), dtype=jnp.complex64)
    parallel = RuntimeParallelConfig(
        strategy="velocity",
        backend="electrostatic_linear_slices",
        axis="hermite",
        num_devices=1,
    )

    with pytest.raises(NotImplementedError, match="collision/EM"):
        linear_rhs_parallel_cached(
            state,
            Cache(),
            LinearParams(),
            terms=LinearTerms(collisions=1.0),
            parallel=parallel,
        )

    auto_parallel = RuntimeParallelConfig(
        strategy="velocity", backend="auto", axis="hermite", num_devices=1
    )
    with pytest.raises(NotImplementedError, match="backend='auto'"):
        linear_rhs_parallel_cached(
            state,
            Cache(),
            LinearParams(),
            terms=LinearTerms(collisions=1.0),
            parallel=auto_parallel,
        )


def test_fused_electrostatic_kernel_term_sum_and_finite_quasineutrality(
    monkeypatch,
) -> None:
    """The fused local kernel must equal the sum of its physical RHS terms."""

    full_state = (
        jnp.arange(2 * 4 * 2 * 1 * 4, dtype=jnp.float32).reshape((2, 4, 2, 1, 4))
        * (1.0e-3 + 2.0e-4j)
    ).astype(jnp.complex64)
    local_state = full_state[:, :2]
    cache = _cache_for_richer_state(Jl=0.2 * _cache_for_richer_state().Jl)
    params = LinearParams(
        charge_sign=1.0,
        density=1.0,
        tau_e=1.0,
        tz=1.0,
        vth=1.2,
        tprim=2.5,
        fprim=0.8,
        omega_d_scale=0.7,
        omega_star_scale=1.1,
        kpar_scale=0.9,
    )
    route = SimpleNamespace(
        local_m=2,
        prev_pairs=((0, 1),),
        next_pairs=((1, 0),),
    )

    monkeypatch.setattr(jax.lax, "axis_index", lambda _name: jnp.asarray(0))
    monkeypatch.setattr(jax.lax, "psum", lambda value, _name: value)
    monkeypatch.setattr(
        jax.lax,
        "ppermute",
        lambda value, _name, _pairs: jnp.zeros_like(value),
    )

    all_terms = _electrostatic_slice_terms()
    data = linear_parallel_electrostatic._fused_electrostatic_constants(
        full_state, cache, params, all_terms, local_m=2
    )
    combined_rhs, phi = (
        linear_parallel_electrostatic._fused_electrostatic_rhs_kernel(
            local_state, data, route, axis_name="m"
        )
    )

    contributions = []
    term_names = ("streaming", "mirror", "curvature", "gradb", "diamagnetic")
    for term_name in term_names:
        term_values = {name: 0.0 for name in term_names}
        term_values[term_name] = 1.0
        weights = replace(all_terms, **term_values)
        term_data = linear_parallel_electrostatic._fused_electrostatic_constants(
            full_state, cache, params, weights, local_m=2
        )
        term_rhs, term_phi = (
            linear_parallel_electrostatic._fused_electrostatic_rhs_kernel(
                local_state, term_data, route, axis_name="m"
            )
        )
        contributions.append(term_rhs)
        np.testing.assert_allclose(np.asarray(term_phi), np.asarray(phi))

    assert combined_rhs.shape == local_state.shape
    assert phi.shape == local_state.shape[-3:]
    assert np.isfinite(np.asarray(combined_rhs)).all()
    assert np.isfinite(np.asarray(phi)).all()
    np.testing.assert_allclose(
        np.asarray(combined_rhs),
        np.asarray(sum(contributions, jnp.zeros_like(combined_rhs))),
        rtol=3.0e-6,
        atol=3.0e-7,
    )
    np.testing.assert_array_equal(
        np.asarray(
            linear_parallel_electrostatic._fused_hermite_shift(
                local_state,
                offset=0,
                axis_name="m",
                prev_pairs=route.prev_pairs,
                next_pairs=route.next_pairs,
            )
        ),
        np.asarray(local_state),
    )


def test_fused_electrostatic_route_rejects_invalid_decompositions() -> None:
    state = jnp.zeros((2, 4, 2, 1, 4), dtype=jnp.complex64)
    device = jax.devices()[0]

    with pytest.raises(ValueError, match="more than one Hermite"):
        linear_parallel_electrostatic._fused_electrostatic_route_setup(
            state,
            plan=SimpleNamespace(chunks={"m": 1}, active_axes=("m",)),
            devices=[device],
            axis_name="m",
        )
    with pytest.raises(ValueError, match="divide evenly"):
        linear_parallel_electrostatic._fused_electrostatic_route_setup(
            state,
            plan=SimpleNamespace(chunks={"m": 3}, active_axes=("m",)),
            devices=[device, device, device],
            axis_name="m",
        )
    with pytest.raises(NotImplementedError, match="only an active 'm' axis"):
        linear_parallel_electrostatic._fused_electrostatic_route_setup(
            state,
            plan=SimpleNamespace(chunks={"m": 2}, active_axes=("l", "m")),
            devices=[device, device],
            axis_name="m",
        )
    with pytest.raises(ValueError, match="not enough devices"):
        linear_parallel_electrostatic._fused_electrostatic_route_setup(
            state,
            plan=SimpleNamespace(chunks={"m": 2}, active_axes=("m",)),
            devices=[device],
            axis_name="m",
        )

    if len(jax.devices()) >= 2:
        route = linear_parallel_electrostatic._fused_electrostatic_route_setup(
            state,
            plan=SimpleNamespace(chunks={"m": 2}, active_axes=("m",)),
            devices=jax.devices()[:2],
            axis_name="m",
        )
        assert route.local_m == 2
        assert route.prev_pairs == ((0, 1),)
        assert route.next_pairs == ((1, 0),)


def test_bounded_mixed_species_hermite_rhs_matches_serial_operator() -> None:
    """One complete mixed RHS compile must preserve the serial equations."""

    from gkx.operators.linear.rhs import linear_rhs_cached

    devices = jax.devices()
    if len(devices) < 4:
        pytest.skip("requires four logical CPU devices or accelerators")
    state, cache, params, _grid, _geom = _small_kinetic_electron_problem()
    terms = _streaming_only_terms()

    expected_rhs, expected_phi = linear_rhs_cached(
        state,
        cache,
        params,
        terms=terms,
        use_jit=False,
        use_custom_vjp=False,
        force_electrostatic_fields=True,
    )
    observed_rhs, observed_phi = (
        linear_parallel_streaming.linear_rhs_electrostatic_species_hermite_sharded(
            state,
            cache,
            params,
            terms=terms,
            devices=devices[:4],
        )
    )

    assert float(jnp.linalg.norm(expected_rhs)) > 0.0
    np.testing.assert_allclose(
        np.asarray(observed_phi), np.asarray(expected_phi), rtol=3.0e-6, atol=3.0e-6
    )
    np.testing.assert_allclose(
        np.asarray(observed_rhs), np.asarray(expected_rhs), rtol=4.0e-5, atol=4.0e-6
    )


def test_bounded_species_sharded_rhs_matches_serial_operator() -> None:
    """One complete species-sharded compile must preserve the serial equations."""

    from gkx.operators.linear.rhs import linear_rhs_cached

    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("requires two logical CPU devices or accelerators")
    state, cache, params, _grid, _geom = _small_kinetic_electron_problem()
    terms = _electrostatic_slice_terms()

    expected_rhs, expected_phi = linear_rhs_cached(
        state,
        cache,
        params,
        terms=terms,
        use_jit=False,
        use_custom_vjp=False,
        force_electrostatic_fields=True,
    )
    observed_rhs, observed_phi = (
        linear_parallel_electrostatic.linear_rhs_electrostatic_species_sharded(
            state,
            cache,
            params,
            terms=terms,
            devices=devices[:2],
        )
    )

    assert float(jnp.linalg.norm(expected_rhs)) > 0.0
    np.testing.assert_allclose(
        np.asarray(observed_phi), np.asarray(expected_phi), rtol=3.0e-6, atol=3.0e-6
    )
    np.testing.assert_allclose(
        np.asarray(observed_rhs), np.asarray(expected_rhs), rtol=4.0e-5, atol=4.0e-6
    )
