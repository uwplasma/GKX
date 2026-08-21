"""Unit contracts: parallel core."""

from __future__ import annotations

from importlib import import_module

from support.paths import REPO_ROOT

# ---- test_parallel.py ----

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gkx
import gkx.parallel as parallel
import gkx.parallel.batch as parallel_batch
import gkx.parallel.independent as parallel_independent


def test_parallel_public_api_exports_are_stable() -> None:
    public_names = (
        "IndependentEnsembleProvenanceReport",
        "ParallelIdentityReport",
        "batch_map",
        "batch_map_identity_report",
        "independent_ensemble_provenance_gate",
        "independent_map",
        "ky_scan_batches",
        "parallel_identity_report",
    )

    assert set(public_names) <= set(gkx.__all__)
    assert set(public_names) <= set(parallel.__all__)
    for name in public_names:
        assert getattr(gkx, name) is getattr(parallel, name)


def test_parallel_lazy_registry_matches_owner_exports() -> None:
    """Keep wheel-safe lazy exports synchronized with their numerical owners."""

    for module_name, expected_names in parallel._MODULE_EXPORTS.items():
        owner = import_module(f"gkx.parallel.{module_name}")
        assert tuple(owner.__all__) == expected_names


def test_ky_scan_batches_are_balanced_and_order_preserving() -> None:
    ky = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    chunks = parallel.ky_scan_batches(ky, n_batches=2)

    assert [chunk.tolist() for chunk in chunks] == [[0.1, 0.2, 0.3], [0.4, 0.5]]
    assert np.allclose(np.concatenate(chunks), ky)


def test_split_evenly_handles_multidimensional_batches_without_empty_chunks() -> None:
    values = np.arange(15).reshape(5, 3)
    chunks = parallel.split_evenly(values, n_parts=8)

    assert [chunk.shape for chunk in chunks] == [(1, 3)] * 5
    np.testing.assert_array_equal(np.concatenate(chunks, axis=0), values)


def test_batch_map_matches_vmap_on_single_device() -> None:
    values = jnp.linspace(0.0, 1.0, 7)

    def fn(x):
        return jnp.asarray([x, x**2 + 1.0])

    observed = parallel.batch_map(fn, values, batch_size=3, devices=[jax.devices()[0]])
    expected = jax.vmap(fn)(values)

    assert np.allclose(np.asarray(observed), np.asarray(expected))


def test_batch_map_matches_vmap_for_pytree_outputs_single_device() -> None:
    values = jnp.linspace(0.0, 1.0, 6)

    def fn(x):
        return {
            "features": jnp.asarray([x, x**2 + 1.0]),
            "moments": (x + 2.0, jnp.asarray([x - 1.0])),
        }

    observed = parallel.batch_map(fn, values, batch_size=2, devices=[jax.devices()[0]])
    expected = jax.vmap(fn)(values)

    jax.tree_util.tree_map(
        lambda obs, exp: np.testing.assert_allclose(np.asarray(obs), np.asarray(exp)),
        observed,
        expected,
    )


def test_parallel_identity_report_records_tree_errors_and_metadata() -> None:
    reference = {
        "gamma": jnp.asarray([1.0, 2.0, 4.0]),
        "omega": (jnp.asarray([0.25]),),
    }
    observed = {
        "gamma": jnp.asarray([1.0, 2.0 + 1e-6, 4.0]),
        "omega": (jnp.asarray([0.25]),),
    }

    report = parallel.parallel_identity_report(
        reference,
        observed,
        kind="unit_identity",
        problem_size=3,
        requested_workers=2,
        actual_workers=2,
        backend="cpu",
        atol=1e-5,
        rtol=1e-5,
        metadata={"observable": "linear_scan"},
    )

    payload = report.to_dict()
    assert report.identity_passed is True
    assert payload["kind"] == "unit_identity"
    assert payload["backend"] == "cpu"
    assert payload["problem_size"] == 3
    assert payload["metadata"] == {"observable": "linear_scan"}
    assert payload["max_abs_error"] > 0.0


def test_parallel_identity_report_rejects_mismatched_pytrees_and_bad_counts() -> None:
    with pytest.raises(ValueError, match="different structures"):
        parallel.parallel_identity_report(
            {"a": jnp.asarray([1.0])},
            {"b": jnp.asarray([1.0])},
            kind="bad",
            problem_size=1,
            requested_workers=1,
        )
    with pytest.raises(ValueError, match="problem_size"):
        parallel.parallel_identity_report(
            jnp.asarray([1.0]),
            jnp.asarray([1.0]),
            kind="bad",
            problem_size=0,
            requested_workers=1,
        )
    with pytest.raises(ValueError, match="actual_workers"):
        parallel.parallel_identity_report(
            jnp.asarray([1.0]),
            jnp.asarray([1.0]),
            kind="bad",
            problem_size=1,
            requested_workers=1,
            actual_workers=2,
        )


def test_batch_map_identity_report_is_serial_vmap_identity_gate() -> None:
    values = jnp.linspace(0.1, 0.9, 9)

    def fn(x):
        return {
            "mode": jnp.asarray([x, x**2, jnp.sin(x)]),
            "flux_proxy": x**3 + 0.5,
        }

    report = parallel.batch_map_identity_report(
        fn,
        values,
        batch_size=4,
        devices=[jax.devices()[0]],
        atol=0.0,
        rtol=0.0,
    )

    assert report.kind == "batch_map_serial_identity"
    assert report.identity_passed is True
    assert report.problem_size == 9
    assert report.actual_workers == 1
    assert report.metadata["batch_size"] == 4


def test_pad_to_multiple_preserves_prefix_and_reports_original_size() -> None:
    padded, original_n = parallel.pad_to_multiple(jnp.asarray([1.0, 2.0, 3.0]), 4)

    assert original_n == 3
    assert np.allclose(np.asarray(padded), np.asarray([1.0, 2.0, 3.0, 3.0]))


def test_pad_to_multiple_preserves_batch_tail_for_multidimensional_values() -> None:
    values = jnp.arange(10, dtype=jnp.float32).reshape(5, 2)
    padded, original_n = parallel.pad_to_multiple(values, 4)

    assert original_n == 5
    assert padded.shape == (8, 2)
    np.testing.assert_array_equal(np.asarray(padded[:5]), np.asarray(values))
    np.testing.assert_array_equal(
        np.asarray(padded[5:]),
        np.repeat(np.asarray(values[-1:]), 3, axis=0),
    )


def test_pad_to_multiple_noops_when_already_aligned_and_split_empty() -> None:
    values = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    padded, original_n = parallel.pad_to_multiple(values, 2)

    assert original_n == 4
    assert np.allclose(np.asarray(padded), np.asarray(values))
    assert parallel.split_evenly(np.asarray([]), 3) == []


def test_batch_map_multi_device_branch_preserves_vmap_identity(monkeypatch) -> None:
    def fake_pmap(fn, devices):
        assert len(devices) == 2

        def mapped(sharded):
            return jnp.stack([fn(shard) for shard in sharded], axis=0)

        return mapped

    monkeypatch.setattr(parallel_batch.jax, "pmap", fake_pmap)
    values = jnp.linspace(0.0, 1.0, 5)

    def fn(x):
        return jnp.asarray([x, x + 1.0])

    observed = parallel.batch_map(
        fn, values, batch_size=3, devices=[object(), object()]
    )
    expected = jax.vmap(fn)(values)

    assert np.allclose(np.asarray(observed), np.asarray(expected))


def test_batch_map_multi_device_branch_drops_padding_and_preserves_chunk_order(
    monkeypatch,
) -> None:
    seen_shards: list[np.ndarray] = []

    def fake_pmap(fn, devices):
        assert len(devices) == 3

        def mapped(sharded):
            seen_shards.append(np.asarray(sharded))
            return jnp.stack([fn(shard) for shard in sharded], axis=0)

        return mapped

    monkeypatch.setattr(parallel_batch.jax, "pmap", fake_pmap)
    values = jnp.arange(7, dtype=jnp.float32)

    def fn(x):
        return jnp.asarray([x, 10.0 * x + jnp.mod(x, 2.0)])

    observed = parallel.batch_map(
        fn, values, batch_size=5, devices=[object(), object(), object()]
    )
    expected = jax.vmap(fn)(values)

    assert np.allclose(np.asarray(observed), np.asarray(expected))
    assert [tuple(shard.shape) for shard in seen_shards] == [(3, 2), (3, 2)]
    np.testing.assert_allclose(
        seen_shards[0].reshape(-1),
        np.asarray([0, 1, 2, 3, 3, 3], dtype=float),
    )
    np.testing.assert_allclose(
        seen_shards[1].reshape(-1),
        np.asarray([4, 5, 6, 6, 6, 6], dtype=float),
    )


def test_batch_map_single_device_fallback_never_calls_pmap(monkeypatch) -> None:
    monkeypatch.setattr(
        parallel_batch.jax,
        "pmap",
        lambda *args, **kwargs: pytest.fail(
            "single-device fallback must use vmap chunks, not pmap"
        ),
    )
    values = jnp.arange(5, dtype=jnp.float32)

    def fn(x):
        return {"linear": x + 1.0, "quadratic": jnp.asarray([x**2])}

    observed = parallel.batch_map(fn, values, batch_size=2, devices=[jax.devices()[0]])
    expected = jax.vmap(fn)(values)

    jax.tree_util.tree_map(
        lambda obs, exp: np.testing.assert_allclose(np.asarray(obs), np.asarray(exp)),
        observed,
        expected,
    )


def test_batch_map_multi_device_branch_preserves_pytree_identity(monkeypatch) -> None:
    def fake_pmap(fn, devices):
        assert len(devices) == 2

        def mapped(sharded):
            return jax.tree_util.tree_map(
                lambda *parts: jnp.stack(parts, axis=0),
                *[fn(shard) for shard in sharded],
            )

        return mapped

    monkeypatch.setattr(parallel_batch.jax, "pmap", fake_pmap)
    values = jnp.linspace(0.0, 1.0, 5)

    def fn(x):
        return {"field": jnp.asarray([x, x + 1.0]), "flux": x**2}

    observed = parallel.batch_map(
        fn, values, batch_size=3, devices=[object(), object()]
    )
    expected = jax.vmap(fn)(values)

    jax.tree_util.tree_map(
        lambda obs, exp: np.testing.assert_allclose(np.asarray(obs), np.asarray(exp)),
        observed,
        expected,
    )


def test_parallel_helpers_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        parallel.split_evenly(np.arange(3), 0)
    with pytest.raises(ValueError):
        parallel.ky_scan_batches(np.ones((2, 2)), n_batches=2)
    with pytest.raises(ValueError):
        parallel.batch_map(lambda x: x, jnp.asarray([]))
    with pytest.raises(ValueError):
        parallel.batch_map(lambda x: x, jnp.asarray([1.0]), batch_size=0)
    with pytest.raises(ValueError):
        parallel.pad_to_multiple(jnp.asarray([1.0]), 0)
    with pytest.raises(ValueError):
        parallel.pad_to_multiple(jnp.asarray([]), 2)
    with pytest.raises(ValueError):
        parallel.independent_map(lambda x: x, [1], workers=0)
    with pytest.raises(ValueError):
        parallel.independent_map(lambda x: x, [1], workers=2, executor="mpi")


def test_independent_map_preserves_serial_order_and_nested_outputs() -> None:
    values = [3, 1, 2]

    def fn(value: int) -> dict[str, int]:
        return {"x": value, "x2": value * value}

    serial = parallel.independent_map(fn, values, workers=1)
    threaded = parallel.independent_map(fn, values, workers=2)

    assert serial == [{"x": 3, "x2": 9}, {"x": 1, "x2": 1}, {"x": 2, "x2": 4}]
    assert threaded == serial
    assert parallel.independent_map(fn, [], workers=3) == []


def test_independent_map_clips_thread_workers_and_accepts_executor_aliases(
    monkeypatch,
) -> None:
    records: list[tuple[int, tuple[tuple[object, int, int, str, int], ...]]] = []

    class FakeThreadPool:
        def __init__(self, *, max_workers: int):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, fn, items):
            materialized = tuple(items)
            records.append((self.max_workers, materialized))
            return [fn(item) for item in materialized]

    monkeypatch.setattr(parallel_independent, "ThreadPoolExecutor", FakeThreadPool)

    observed = parallel.independent_map(
        lambda value: value * 11, [3, 1, 4], workers=99, executor="threads"
    )

    assert observed == [33, 11, 44]
    assert records[0][0] == 3
    assert [(task[1], task[2], task[3], task[4]) for task in records[0][1]] == [
        (0, 3, "thread", 3),
        (1, 1, "thread", 3),
        (2, 4, "thread", 3),
    ]


def test_independent_worker_metadata_normalizes_aliases_and_empty_work() -> None:
    metadata = parallel.independent_worker_metadata(
        3,
        workers=8,
        executor="threads",
    )
    empty = parallel.independent_worker_metadata(
        0,
        workers=4,
        executor="processes",
    )

    assert metadata.to_dict() == {
        "requested_workers": 8,
        "actual_workers": 3,
        "problem_size": 3,
        "executor": "thread",
        "parallel_enabled": True,
    }
    assert empty.to_dict() == {
        "requested_workers": 4,
        "actual_workers": 0,
        "problem_size": 0,
        "executor": "process",
        "parallel_enabled": False,
    }
    with pytest.raises(ValueError, match="problem_size"):
        parallel.independent_worker_metadata(-1)


def test_independent_map_identity_report_records_worker_metadata() -> None:
    values = [0.1, 0.2, 0.4]

    def fn(value: float) -> dict[str, jnp.ndarray]:
        x = jnp.asarray(value)
        return {"mode": jnp.asarray([x, x**2]), "flux": jnp.asarray(x + 1.0)}

    report = parallel.independent_map_identity_report(
        fn,
        values,
        workers=5,
        executor="threads",
        atol=0.0,
        rtol=0.0,
        metadata={"case": "threaded_unit_gate"},
    )

    assert report.kind == "independent_map_serial_identity"
    assert report.backend == "python:thread"
    assert report.identity_passed is True
    assert report.problem_size == 3
    assert report.requested_workers == 5
    assert report.actual_workers == 3
    assert report.max_abs_error == 0.0
    assert report.metadata["case"] == "threaded_unit_gate"
    assert report.metadata["executor"] == "thread"
    assert report.metadata["parallel_enabled"] is True
    assert report.metadata["worker_metadata"] == {
        "requested_workers": 5,
        "actual_workers": 3,
        "problem_size": 3,
        "executor": "thread",
        "parallel_enabled": True,
    }


def test_independent_ensemble_provenance_gate_closes_uq_optimization_batching() -> None:
    values = [0.05, 0.2, 0.45, 0.8]

    def fn(value: float) -> dict[str, jnp.ndarray]:
        x = jnp.asarray(value)
        residual = x - 0.35
        return {
            "objective": jnp.asarray(residual * residual + 0.1 * x),
            "gradient_proxy": jnp.asarray([2.0 * residual + 0.1, x**2]),
            "uq_weight": jnp.asarray(1.0 / (1.0 + x * x)),
        }

    report = parallel.independent_ensemble_provenance_gate(
        fn,
        values,
        workers=99,
        executor="threads",
        workload="optimization_ensemble",
        atol=0.0,
        rtol=0.0,
        metadata={"case": "optimization_uq_batch"},
    )

    assert isinstance(report, parallel.IndependentEnsembleProvenanceReport)
    assert report.kind == "independent_ensemble_provenance_gate"
    assert report.workload == "optimization_ensemble"
    assert report.passed is True
    assert report.identity_passed is True
    assert report.ordering_passed is True
    assert report.worker_clipping_passed is True
    assert report.reconstruction_identity_passed is True
    assert report.exception_metadata_passed is True
    assert report.requested_workers == 99
    assert report.actual_workers == len(values)
    assert report.serial_indices == (0, 1, 2, 3)
    assert report.parallel_indices == report.serial_indices
    assert report.reconstructed_indices == report.serial_indices
    assert report.identity_report.max_abs_error == 0.0
    assert report.exception_metadata["index"] == 1
    assert report.exception_metadata["executor"] == "thread"
    assert report.exception_metadata["actual_workers"] == 2
    assert report.exception_metadata["original_type"] == "ValueError"
    assert report.metadata["case"] == "optimization_uq_batch"
    assert (
        report.metadata["contract"]["claim_level"] == "production_independent_batching"
    )
    assert report.to_dict()["passed"] is True


def test_independent_ensemble_provenance_gate_rejects_empty_and_bad_workloads() -> None:
    with pytest.raises(ValueError, match="at least one item"):
        parallel.independent_ensemble_provenance_gate(lambda x: x, [], workers=2)
    with pytest.raises(ValueError, match="workload"):
        parallel.independent_ensemble_provenance_gate(
            lambda x: x,
            [1.0],
            workload="independent_ky_scan",
        )


def test_independent_map_identity_helpers_are_exported_at_package_top_level() -> None:
    import gkx as sgk

    assert (
        sgk.IndependentEnsembleProvenanceReport
        is parallel.IndependentEnsembleProvenanceReport
    )
    assert sgk.IndependentMapExecutionError is parallel.IndependentMapExecutionError
    assert sgk.IndependentWorkerMetadata is parallel.IndependentWorkerMetadata
    assert (
        sgk.independent_ensemble_provenance_gate
        is parallel.independent_ensemble_provenance_gate
    )
    assert sgk.independent_worker_metadata is parallel.independent_worker_metadata
    assert (
        sgk.independent_map_identity_report is parallel.independent_map_identity_report
    )


def test_independent_map_parallel_failures_include_worker_metadata() -> None:
    def fn(value: int) -> int:
        if value == 2:
            raise ValueError("bad ky point")
        return value

    with pytest.raises(
        parallel.IndependentMapExecutionError,
        match=(
            "independent_map task 1 failed with executor='thread' "
            "and actual_workers=2: ValueError: bad ky point"
        ),
    ) as exc_info:
        parallel.independent_map(fn, [1, 2, 3], workers=2, executor="thread")

    assert exc_info.value.index == 1
    assert exc_info.value.executor == "thread"
    assert exc_info.value.actual_workers == 2


@pytest.mark.gpu
def test_cpu_gpu_short_window_gate_matches_within_tolerance() -> None:
    import os
    from dataclasses import replace
    from pathlib import Path

    if os.environ.get("GKX_DEVICE_PARITY", "").strip() not in {
        "1",
        "true",
        "yes",
    }:
        pytest.skip("Set GKX_DEVICE_PARITY=1 to enable CPU/GPU parity gate.")

    try:
        cpu_devices = jax.devices("cpu")
    except Exception:
        cpu_devices = ()
    try:
        gpu_devices = jax.devices("gpu")
    except Exception:
        gpu_devices = ()

    cpu = cpu_devices[0] if cpu_devices else None
    gpu = gpu_devices[0] if gpu_devices else None
    if cpu is None or gpu is None:
        pytest.skip("No GPU backend detected for JAX.")

    from support.paths import load_repo_script
    from gkx.runtime import run_runtime_nonlinear

    restart_helpers = load_repo_script(
        Path("tests/integration/runtime/test_restart_gate.py"),
        module_name="runtime_restart_gate_helpers",
    )
    cfg = restart_helpers._restart_base_cfg()
    cfg = replace(cfg, time=replace(cfg.time, dt=0.02))

    def _run_on(device):
        with jax.default_device(device):
            out = run_runtime_nonlinear(
                cfg,
                ky_target=0.2,
                kx_target=0.0,
                Nl=4,
                Nm=6,
                dt=0.02,
                steps=6,
                sample_stride=1,
                diagnostics_stride=1,
                return_state=True,
            )
        assert out.state is not None
        return np.asarray(out.state)

    state_cpu = _run_on(cpu)
    state_gpu = _run_on(gpu)

    # GPU FFTs can introduce small roundoff differences; gate on a stable scalar.
    norm_cpu = float(np.linalg.norm(state_cpu.ravel()))
    norm_gpu = float(np.linalg.norm(state_gpu.ravel()))
    assert norm_cpu > 0.0
    assert norm_gpu > 0.0
    assert norm_gpu == pytest.approx(norm_cpu, rel=2.0e-4, abs=1.0e-7)


# ---- test_sharding.py ----

import gkx.parallel.state as sharding_mod
from gkx.parallel.state import resolve_state_sharding


def _state_5d():
    return jnp.zeros((2, 2, 4, 1, 8), dtype=jnp.complex64)


def _state_6d():
    return jnp.zeros((1, 2, 2, 4, 1, 8), dtype=jnp.complex64)


def test_state_sharding_disabled():
    G0 = _state_5d()
    assert resolve_state_sharding(G0, None) is None
    assert resolve_state_sharding(G0, "none") is None
    assert resolve_state_sharding(G0, "off") is None
    assert resolve_state_sharding(G0, "") is None
    assert resolve_state_sharding(G0, " false ") is None
    assert resolve_state_sharding(G0, "0") is None


def test_state_sharding_invalid():
    G0 = _state_5d()
    with pytest.raises(ValueError):
        resolve_state_sharding(G0, "banana")


def test_state_sharding_single_device_noop():
    G0 = _state_6d()
    sharding = resolve_state_sharding(G0, "ky", devices=[jax.devices()[0]])
    assert sharding is None


def test_state_sharding_builds_partition_specs_with_fake_mesh(monkeypatch):
    class FakeNamedSharding:
        def __init__(self, mesh, spec):
            self.mesh = mesh
            self.spec = spec

    monkeypatch.setattr(
        sharding_mod,
        "_mesh_from_devices",
        lambda devices, axis_name: f"mesh:{axis_name}",
    )
    monkeypatch.setattr(sharding_mod, "NamedSharding", FakeNamedSharding)

    ky_sharding = resolve_state_sharding(
        _state_5d(), "auto", axis_name="batch", devices=[object(), object()]
    )
    species_sharding = resolve_state_sharding(
        _state_6d(), "species", axis_name="batch", devices=[object(), object()]
    )

    assert ky_sharding.mesh == "mesh:batch"
    assert ky_sharding.spec == sharding_mod.PartitionSpec(
        None, None, "batch", None, None
    )
    assert species_sharding.spec == sharding_mod.PartitionSpec(
        "batch", None, None, None, None, None
    )

    with pytest.raises(ValueError, match="Cannot shard"):
        resolve_state_sharding(_state_5d(), "species", devices=[object(), object()])
    with pytest.raises(ValueError, match="5 or 6 dimensions"):
        resolve_state_sharding(jnp.zeros((2, 2)), "ky", devices=[object(), object()])


@pytest.mark.parametrize(
    ("directive", "expected_spec"),
    [
        ("auto", (None, None, "batch", None, None)),
        ("ky", (None, None, "batch", None, None)),
        ("kx", (None, None, None, "batch", None)),
        ("z", (None, None, None, None, "batch")),
        ("l", ("batch", None, None, None, None)),
        ("m", (None, "batch", None, None, None)),
    ],
)
def test_state_sharding_5d_axis_map_is_explicit_with_fake_mesh(
    monkeypatch, directive, expected_spec
):
    class FakeNamedSharding:
        def __init__(self, mesh, spec):
            self.mesh = mesh
            self.spec = spec

    monkeypatch.setattr(
        sharding_mod,
        "_mesh_from_devices",
        lambda devices, axis_name: f"mesh:{axis_name}",
    )
    monkeypatch.setattr(sharding_mod, "NamedSharding", FakeNamedSharding)

    resolved = resolve_state_sharding(
        _state_5d(), directive, axis_name="batch", devices=[object(), object()]
    )

    assert resolved.mesh == "mesh:batch"
    assert resolved.spec == sharding_mod.PartitionSpec(*expected_spec)


@pytest.mark.parametrize("directive", ["species", "s"])
def test_state_sharding_6d_species_aliases_share_partition_spec(monkeypatch, directive):
    class FakeNamedSharding:
        def __init__(self, mesh, spec):
            self.mesh = mesh
            self.spec = spec

    monkeypatch.setattr(
        sharding_mod,
        "_mesh_from_devices",
        lambda devices, axis_name: f"mesh:{axis_name}",
    )
    monkeypatch.setattr(sharding_mod, "NamedSharding", FakeNamedSharding)

    resolved = resolve_state_sharding(
        _state_6d(), directive, axis_name="batch", devices=[object(), object()]
    )

    assert resolved.spec == sharding_mod.PartitionSpec(
        "batch", None, None, None, None, None
    )


def test_mesh_from_devices_uses_visible_devices_and_returns_none_for_one_device(
    monkeypatch,
):
    class FakeMesh:
        def __init__(self, devices, axis_names):
            self.devices = devices
            self.axis_names = axis_names

    monkeypatch.setattr(sharding_mod, "Mesh", FakeMesh)
    fake_devices = [object(), object(), object()]
    monkeypatch.setattr(sharding_mod.jax, "devices", lambda: fake_devices)

    mesh = sharding_mod._mesh_from_devices(None, "d")

    assert mesh is not None
    assert mesh.axis_names == ("d",)
    assert list(mesh.devices.reshape(-1)) == fake_devices
    assert sharding_mod._mesh_from_devices([object()], "d") is None


# ---- test_parallel_decomposition.py ----

from gkx.parallel.decomposition import (
    DecompositionContract,
    ReconstructionIdentityReport,
    ShardAssignment,
    build_diagnostic_nonlinear_domain_decomposition,
    build_independent_portfolio_decomposition,
    reconstruct_serial,
    serial_reconstruction_identity_report,
    shard_sequence,
)


ROOT = REPO_ROOT


def test_independent_ky_decomposition_is_deterministic_balanced_and_ordered() -> None:
    first = build_independent_portfolio_decomposition(
        7,
        requested_shards=3,
        workload="independent_ky_scan",
    )
    second = build_independent_portfolio_decomposition(
        7,
        requested_shards=3,
        workload="independent_ky_scan",
    )

    assert first == second
    assert first.production_independent_batching is True
    assert first.diagnostic_nonlinear_partition is False
    assert first.independent_work is True
    assert first.changes_solver_layout is False
    assert first.actual_shards == 3
    assert [shard.indices for shard in first.shards] == [
        (0, 1, 2),
        (3, 4),
        (5, 6),
    ]
    assert [shard.size for shard in first.shards] == [3, 2, 2]
    assert [shard.start for shard in first.shards] == [0, 3, 5]
    assert [shard.stop for shard in first.shards] == [3, 5, 7]
    assert "production independent batching" in first.claim_label
    assert (
        "not a nonlinear state-domain decomposition speedup claim" in first.claim_label
    )
    assert first.to_dict()["workload"] == "independent_ky_scan"
    assert (
        first.shards[0].to_dict()["label"].startswith("independent_ky_scan:shard_000")
    )


def test_uq_decomposition_reconstructs_serial_identity() -> None:
    values = tuple(f"member-{idx}" for idx in range(8))
    contract = build_independent_portfolio_decomposition(
        len(values),
        requested_shards=4,
        workload="uq_ensemble",
    )

    shards = shard_sequence(values, contract)
    reconstructed = reconstruct_serial(contract, shards)
    report = serial_reconstruction_identity_report(values, contract)

    assert shards == (
        ("member-0", "member-1"),
        ("member-2", "member-3"),
        ("member-4", "member-5"),
        ("member-6", "member-7"),
    )
    assert reconstructed == values
    assert report == ReconstructionIdentityReport(
        workload="uq_ensemble",
        claim_level="production_independent_batching",
        claim_label=contract.claim_label,
        n_items=8,
        requested_shards=4,
        actual_shards=4,
        identity_passed=True,
        expected_indices=tuple(range(8)),
        reconstructed_indices=tuple(range(8)),
        missing_indices=(),
        duplicate_indices=(),
        out_of_range_indices=(),
        out_of_order=False,
    )
    assert report.to_dict()["identity_passed"] is True


def test_optimization_ensemble_decomposition_uses_production_independent_contract() -> (
    None
):
    values = tuple({"candidate": idx, "objective": idx * idx} for idx in range(5))
    contract = build_independent_portfolio_decomposition(
        len(values),
        requested_shards=8,
        workload="optimization_ensemble",
    )
    report = serial_reconstruction_identity_report(values, contract)

    assert contract.workload == "optimization_ensemble"
    assert contract.claim_level == "production_independent_batching"
    assert contract.actual_shards == 5
    assert contract.independent_work is True
    assert contract.changes_solver_layout is False
    assert "independent optimization ensemble" in contract.claim_label
    assert "not a nonlinear state-domain decomposition" in contract.claim_label
    assert report.identity_passed is True
    assert reconstruct_serial(contract, shard_sequence(values, contract)) == values


def test_decomposition_handles_empty_and_oversharded_portfolios_without_empty_shards() -> (
    None
):
    empty = build_independent_portfolio_decomposition(
        0,
        requested_shards=4,
        workload="uq_ensemble",
    )
    oversharded = build_independent_portfolio_decomposition(
        3,
        requested_shards=8,
        workload="independent_ky_scan",
    )

    assert empty.actual_shards == 0
    assert empty.shards == ()
    assert serial_reconstruction_identity_report((), empty).identity_passed is True
    assert oversharded.actual_shards == 3
    assert [shard.indices for shard in oversharded.shards] == [(0,), (1,), (2,)]
    assert all(shard.size == 1 for shard in oversharded.shards)
    assert reconstruct_serial(
        oversharded, shard_sequence(("a", "b", "c"), oversharded)
    ) == (
        "a",
        "b",
        "c",
    )


def test_decomposition_rejects_invalid_counts_workloads_and_mismatched_values() -> None:
    with pytest.raises(ValueError, match="requested_shards"):
        build_independent_portfolio_decomposition(
            3,
            requested_shards=0,
            workload="independent_ky_scan",
        )
    with pytest.raises(ValueError, match="n_items"):
        build_independent_portfolio_decomposition(
            -1,
            requested_shards=1,
            workload="uq_ensemble",
        )
    with pytest.raises(ValueError, match="workload"):
        build_independent_portfolio_decomposition(
            3,
            requested_shards=1,
            workload="diagnostic_nonlinear_domain",  # type: ignore[arg-type]
        )

    contract = build_independent_portfolio_decomposition(
        3,
        requested_shards=2,
        workload="uq_ensemble",
    )
    with pytest.raises(ValueError, match="values length"):
        shard_sequence(("only-one",), contract)
    with pytest.raises(ValueError, match="actual_shards"):
        reconstruct_serial(contract, (("a", "b"),))
    with pytest.raises(ValueError, match="assignment size"):
        reconstruct_serial(contract, (("a",), ("b",)))


def test_diagnostic_nonlinear_domain_decomposition_is_split_reassemble_only() -> None:
    contract = build_diagnostic_nonlinear_domain_decomposition(
        (4, 6, 2),
        axis=-2,
        requested_shards=4,
    )
    values = tuple(range(contract.n_items))
    report = serial_reconstruction_identity_report(values, contract)

    assert contract.workload == "diagnostic_nonlinear_domain"
    assert contract.claim_level == "diagnostic_nonlinear_domain_partition"
    assert contract.production_independent_batching is False
    assert contract.diagnostic_nonlinear_partition is True
    assert contract.independent_work is False
    assert contract.changes_solver_layout is True
    assert contract.state_shape == (4, 6, 2)
    assert contract.axis == 1
    assert contract.actual_shards == 4
    assert [shard.indices for shard in contract.shards] == [
        (0, 1),
        (2, 3),
        (4,),
        (5,),
    ]
    assert all("axis_1" in shard.label for shard in contract.shards)
    assert report.identity_passed is True
    assert "diagnostic nonlinear state-domain partition" in report.claim_label
    assert "no production routing or speedup claim" in report.claim_label


def test_diagnostic_nonlinear_domain_decomposition_validates_shape_and_shards() -> None:
    oversharded = build_diagnostic_nonlinear_domain_decomposition(
        (2, 3),
        axis=1,
        requested_shards=10,
    )

    assert oversharded.actual_shards == 3
    assert [shard.indices for shard in oversharded.shards] == [(0,), (1,), (2,)]
    with pytest.raises(ValueError, match="at least one axis"):
        build_diagnostic_nonlinear_domain_decomposition(
            (),
            axis=0,
            requested_shards=1,
        )
    with pytest.raises(ValueError, match="positive"):
        build_diagnostic_nonlinear_domain_decomposition(
            (2, 0),
            axis=0,
            requested_shards=1,
        )
    with pytest.raises(ValueError, match="requested_shards"):
        build_diagnostic_nonlinear_domain_decomposition(
            (2, 3),
            axis=0,
            requested_shards=0,
        )


def test_claim_levels_separate_production_batches_from_diagnostic_domain_partitions() -> (
    None
):
    ky = build_independent_portfolio_decomposition(
        5,
        requested_shards=2,
        workload="independent_ky_scan",
    )
    uq = build_independent_portfolio_decomposition(
        5,
        requested_shards=2,
        workload="uq_ensemble",
    )
    nonlinear = build_diagnostic_nonlinear_domain_decomposition(
        (5, 4),
        axis=0,
        requested_shards=2,
    )

    assert {ky.claim_level, uq.claim_level} == {"production_independent_batching"}
    assert nonlinear.claim_level == "diagnostic_nonlinear_domain_partition"
    assert ky.independent_work and uq.independent_work
    assert not nonlinear.independent_work
    assert not ky.changes_solver_layout
    assert nonlinear.changes_solver_layout
    assert "not a nonlinear state-domain decomposition" in ky.claim_label
    assert "not a nonlinear state-domain decomposition" in uq.claim_label
    assert "no production routing or speedup claim" in nonlinear.claim_label


def test_manual_bad_assignment_report_can_expose_claim_scoped_identity_failure() -> (
    None
):
    bad_contract = DecompositionContract(
        workload="diagnostic_nonlinear_domain",
        claim_level="diagnostic_nonlinear_domain_partition",
        claim_label="diagnostic nonlinear state-domain partition contract",
        n_items=3,
        requested_shards=2,
        actual_shards=2,
        shards=(
            ShardAssignment(
                shard_id=0,
                start=0,
                stop=2,
                indices=(0, 2),
                label="bad:0",
            ),
            ShardAssignment(
                shard_id=1,
                start=2,
                stop=3,
                indices=(1,),
                label="bad:1",
            ),
        ),
        independent_work=False,
        changes_solver_layout=True,
    )
    report = serial_reconstruction_identity_report(("a", "b", "c"), bad_contract)

    assert report.identity_passed is False
    assert report.missing_indices == ()
    assert report.duplicate_indices == ()
    assert report.out_of_range_indices == ()
    assert report.out_of_order is True
    assert report.reconstructed_indices == (0, 2, 1)


def test_parallel_decomposition_contracts_are_exported_at_package_top_level() -> None:
    import gkx as sgk

    contract = sgk.build_independent_portfolio_decomposition(
        2,
        requested_shards=2,
        workload="independent_ky_scan",
    )

    assert isinstance(contract, sgk.DecompositionContract)
    assert sgk.shard_sequence(("a", "b"), contract) == (("a",), ("b",))
