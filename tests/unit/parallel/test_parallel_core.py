"""Unit contracts: parallel core."""

from __future__ import annotations

from importlib import import_module


# ---- test_parallel.py ----

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gkx
import gkx.parallel as parallel
import gkx.parallel.independent as parallel_batch
import gkx.parallel.independent as parallel_independent
from dataclasses import replace
import gkx.workflows.nonlinear as nonlinear_workflow
from gkx.config import GeometryConfig, GridConfig, InitializationConfig, TimeConfig
from gkx.runtime import run_runtime_nonlinear
from gkx.config import (
    RuntimeConfig,
    RuntimeNormalizationConfig,
    RuntimeParallelConfig,
    RuntimePhysicsConfig,
    RuntimeSpeciesConfig,
    RuntimeTermsConfig,
)
from gkx.workflows.runtime.parallel_nonlinear import (
    NonlinearParallelIdentityError,
    NonlinearParallelPlan,
    NonlinearParallelRoutingError,
    assert_nonlinear_parallel_identity,
    resolve_nonlinear_parallel_plan,
    shard_nonlinear_state,
)
import json
from pathlib import Path
from support.paths import REPO_ROOT, load_release_tool
import tomllib


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

    assert set(public_names).isdisjoint(gkx.__all__)
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
        Path("tests/integration/runtime/test_runtime_runner.py"),
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

from gkx.parallel.independent import (
    DecompositionContract,
    ReconstructionIdentityReport,
    ShardAssignment,
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


# ---- from test_parallel_nonlinear_routing.py ----
# Unit contracts: routing ``[parallel]`` into the nonlinear solver path.
#
# The multi-device cases need more than one JAX device. Run them with
# ``XLA_FLAGS=--xla_force_host_platform_device_count=4``; the wide-coverage
# runner supplies that through ``WIDE_COVERAGE_LOGICAL_CPU_DEVICES``. Without it
# they skip rather than silently passing on one device.


_RUN_KWARGS: dict[str, object] = {
    "ky_target": 0.2,
    "Nl": 3,
    "Nm": 4,
    "dt": 0.01,
    "steps": 3,
    "sample_stride": 1,
}


def _nonlinear_cfg(
    parallel: RuntimeParallelConfig | None = None, *, ky_layout: str = "full"
) -> RuntimeConfig:
    """Return a small periodic nonlinear case with ``Ny = 8``.

    The stored ky extent is 8 on the two-sided axis and ``Nyc = 5`` on the
    ``ky >= 0`` one, which no even device count divides.
    """

    cfg = RuntimeConfig(
        grid=GridConfig(
            Nx=1,
            Ny=8,
            Nz=16,
            Lx=6.28,
            Ly=6.28,
            boundary="periodic",
            ky_layout=ky_layout,
        ),
        time=TimeConfig(t_max=0.2, dt=0.01, method="rk2", sample_stride=1),
        geometry=GeometryConfig(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778),
        init=InitializationConfig(
            init_field="density", init_amp=1.0e-8, gaussian_init=False
        ),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone"),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, nonlinear=True),
        terms=RuntimeTermsConfig(nonlinear=1.0, hypercollisions=0.0, end_damping=0.0),
    )
    return cfg if parallel is None else replace(cfg, parallel=parallel)


def _live_cfg(
    parallel: RuntimeParallelConfig | None = None, *, ky_layout: str = "full"
) -> RuntimeConfig:
    """Return a deck of the same size whose state and fields actually evolve.

    :func:`_nonlinear_cfg` seeds a single mode that stays put (``phi`` is
    identically zero and the state does not change over the run), so an
    identity on it cannot fail. This one has ``Nx = 4``, a larger box and a
    multimode start at ``1e-2``; over 20 rk2 steps the state moves by about
    ``5e-4`` and ``max |phi|`` is about ``4e-3``.
    """

    cfg = _nonlinear_cfg(parallel, ky_layout=ky_layout)
    return replace(
        cfg,
        grid=replace(cfg.grid, Nx=4, Lx=31.4, Ly=31.4),
        init=replace(cfg.init, init_amp=1.0e-2, init_single=False),
    )


def _fake_plan(
    count: int = 2, *, strict_identity: bool = True
) -> NonlinearParallelPlan:
    """Return a plan with placeholder devices for host-only contract checks."""

    return NonlinearParallelPlan(
        axis="ky",
        devices=tuple(object() for _ in range(count)),
        strict_identity=strict_identity,
    )


def _require_devices(count: int) -> None:
    if len(jax.devices()) < count:
        pytest.skip(f"requires {count} logical CPU devices or accelerators")


# ---- plan resolution: nothing is accepted and then ignored ----


def test_serial_strategy_resolves_to_no_plan() -> None:
    assert resolve_nonlinear_parallel_plan(RuntimeParallelConfig()) is None
    assert resolve_nonlinear_parallel_plan(None) is None


@pytest.mark.parametrize(
    "strategy",
    ["batch", "combined_ky", "device_batch", "pmap", "pjit", "state", "velocity"],
)
def test_unsupported_strategy_raises_instead_of_running_serial(strategy: str) -> None:
    """A strategy with no nonlinear route must fail, not quietly run serially."""

    with pytest.raises(NonlinearParallelRoutingError) as excinfo:
        resolve_nonlinear_parallel_plan(
            RuntimeParallelConfig(strategy=strategy, axis="ky", num_devices=2)
        )
    message = str(excinfo.value)
    assert strategy in message
    assert "shard_map" in message and "axis='ky'" in message


def test_z_axis_is_rejected_with_the_measured_reason() -> None:
    """axis='z' names both blockers rather than pretending to be routable."""

    with pytest.raises(NonlinearParallelRoutingError) as excinfo:
        resolve_nonlinear_parallel_plan(
            RuntimeParallelConfig(strategy="shard_map", axis="z", num_devices=2)
        )
    message = str(excinfo.value)
    assert "spectral FFT along z" in message
    assert "device_z" in message
    assert "axis='ky'" in message


@pytest.mark.parametrize("axis", ["kx", "l", "laguerre"])
def test_unsupported_axis_raises(axis: str) -> None:
    with pytest.raises(NonlinearParallelRoutingError, match="axis='ky'"):
        resolve_nonlinear_parallel_plan(
            RuntimeParallelConfig(strategy="shard_map", axis=axis, num_devices=2)
        )


@pytest.mark.parametrize(
    "axis", ["species_hermite", "velocity", "s_m", "species", "m", "hermite"]
)
def test_velocity_axis_aliases_resolve_to_the_production_mesh(axis: str) -> None:
    """Every spelling of the velocity mesh routes to one canonical axis."""

    _require_devices(2)
    plan = resolve_nonlinear_parallel_plan(
        RuntimeParallelConfig(strategy="shard_map", axis=axis, num_devices=2)
    )
    assert plan is not None
    assert plan.axis == "species_hermite"


def test_auto_selects_the_species_hermite_mesh_and_says_which() -> None:
    """``auto = true`` resolves the mesh from the devices and reports it."""

    _require_devices(2)
    from gkx.workflows.runtime.parallel_nonlinear import resolve_species_hermite_mesh

    plan = resolve_nonlinear_parallel_plan(
        RuntimeParallelConfig(auto=True, num_devices=2)
    )
    assert plan is not None and plan.axis == "species_hermite" and plan.auto
    resolved = resolve_species_hermite_mesh(
        jnp.zeros((2, 4, 16, 4, 4, 4), dtype=jnp.complex64), plan
    )
    assert resolved.mesh_shape == (2, 1)
    described = resolved.describe()
    assert "2 species x 1 Hermite" in described
    assert "auto-selected" in described
    assert "no halo" in described


def test_auto_conflicting_with_an_explicit_strategy_is_an_error() -> None:
    """``auto`` must not silently overrule an explicit request."""

    with pytest.raises(ValueError, match="conflicts with"):
        RuntimeParallelConfig(auto=True, strategy="batch")
    with pytest.raises(ValueError, match="conflicts with"):
        RuntimeParallelConfig(auto=True, axis="kx")


def test_indivisible_device_count_names_the_counts_that_work() -> None:
    """A rejected mesh reports its alternatives, not only its failure."""

    _require_devices(3)
    from gkx.workflows.runtime.parallel_nonlinear import resolve_species_hermite_mesh

    plan = resolve_nonlinear_parallel_plan(
        RuntimeParallelConfig(auto=True, num_devices=3)
    )
    with pytest.raises(NonlinearParallelRoutingError) as excinfo:
        resolve_species_hermite_mesh(
            jnp.zeros((2, 4, 16, 4, 4, 4), dtype=jnp.complex64), plan
        )
    message = str(excinfo.value)
    assert "supports 1, 2, 4, 8, 16 devices" in message
    assert "divide" in message


def test_independent_worker_options_are_rejected() -> None:
    with pytest.raises(NonlinearParallelRoutingError, match="worker pool"):
        resolve_nonlinear_parallel_plan(
            RuntimeParallelConfig(
                strategy="shard_map", axis="ky", num_devices=2, backend="thread"
            )
        )
    with pytest.raises(NonlinearParallelRoutingError, match="batch_size"):
        resolve_nonlinear_parallel_plan(
            RuntimeParallelConfig(
                strategy="shard_map", axis="ky", num_devices=2, batch_size=2
            )
        )


def test_unsatisfiable_device_request_raises_a_routing_error() -> None:
    with pytest.raises(NonlinearParallelRoutingError, match="num_devices"):
        resolve_nonlinear_parallel_plan(
            RuntimeParallelConfig(
                strategy="shard_map", axis="ky", num_devices=len(jax.devices()) + 8
            )
        )


def test_plan_honours_num_devices() -> None:
    _require_devices(2)
    plan = resolve_nonlinear_parallel_plan(
        RuntimeParallelConfig(strategy="shard_map", axis="ky", num_devices=2)
    )
    assert plan is not None
    assert plan.device_count == 2
    assert plan.strict_identity is True
    assert "axis='ky'" in plan.describe()


def test_indivisible_ky_extent_is_placed_replicated_on_the_mesh() -> None:
    """JAX places arrays only in equal shares, so an uneven ky extent -- the
    ``ky >= 0`` axis's odd ``Nyc`` -- enters replicated on the same mesh."""

    _require_devices(2)
    plan = replace(_fake_plan(2), devices=tuple(jax.devices()[:2]))
    state = jnp.arange(12, dtype=jnp.complex64).reshape(1, 1, 3, 1, 4)
    placed = shard_nonlinear_state(state, plan)
    assert placed.sharding.is_fully_replicated
    assert placed.sharding.device_set == set(plan.devices)
    np.testing.assert_array_equal(np.asarray(placed), np.asarray(state))


def test_divisible_ky_extent_is_split_across_the_mesh() -> None:
    _require_devices(2)
    plan = replace(_fake_plan(2), devices=tuple(jax.devices()[:2]))
    state = jnp.zeros((1, 1, 4, 1, 4), dtype=jnp.complex64)
    placed = shard_nonlinear_state(state, plan)
    assert not placed.sharding.is_fully_replicated
    assert {shard.data.shape[-3] for shard in placed.addressable_shards} == {2}


# ---- fail-closed identity ----


def test_identity_gate_accepts_an_exact_match() -> None:
    state = jnp.ones((2, 2), dtype=jnp.complex64)
    assert_nonlinear_parallel_identity(
        serial_state=state, sharded_state=state, plan=_fake_plan()
    )


def test_identity_gate_raises_on_a_perturbed_state() -> None:
    serial = jnp.ones((2, 2), dtype=jnp.complex64)
    with pytest.raises(NonlinearParallelIdentityError) as excinfo:
        assert_nonlinear_parallel_identity(
            serial_state=serial,
            sharded_state=serial * 1.5,
            plan=_fake_plan(),
        )
    message = str(excinfo.value)
    assert "final_state" in message
    assert "discarded" in message


def test_identity_gate_raises_on_a_perturbed_diagnostic_trace() -> None:
    """A route that reproduces the state but not the traces still fails."""

    state = jnp.ones((2, 2), dtype=jnp.complex64)
    serial_diag = type("Diag", (), {"Wg_t": np.ones(4), "heat_flux_t": np.ones(4)})()
    sharded_diag = type(
        "Diag", (), {"Wg_t": np.ones(4), "heat_flux_t": np.full(4, 2.0)}
    )()
    with pytest.raises(NonlinearParallelIdentityError, match="heat_flux_t"):
        assert_nonlinear_parallel_identity(
            serial_state=state,
            sharded_state=state,
            serial_diagnostics=serial_diag,
            sharded_diagnostics=sharded_diag,
            plan=_fake_plan(),
        )


def test_identity_gate_reports_a_shape_mismatch_as_a_failure() -> None:
    with pytest.raises(NonlinearParallelIdentityError):
        assert_nonlinear_parallel_identity(
            serial_state=jnp.ones((2, 2), dtype=jnp.complex64),
            sharded_state=jnp.ones((2, 3), dtype=jnp.complex64),
            plan=_fake_plan(),
        )


# ---- end-to-end runtime routing ----


@pytest.mark.parametrize("ky_layout", ["full", "half"])
@pytest.mark.parametrize("num_devices", [2, 4])
def test_shard_map_nonlinear_run_matches_serial_diagnostics(
    num_devices: int, ky_layout: str
) -> None:
    """A sharded nonlinear run reproduces the serial run's diagnostics.

    On ``"half"`` the stored extent is ``Nyc = 5``, which neither device count
    divides; the run is accepted rather than refused (SHARD-PAD, plan F.6).
    """

    _require_devices(num_devices)
    serial = run_runtime_nonlinear(
        _live_cfg(ky_layout=ky_layout), return_state=True, **_RUN_KWARGS
    )
    sharded = run_runtime_nonlinear(
        _live_cfg(
            RuntimeParallelConfig(
                strategy="shard_map", axis="ky", num_devices=num_devices
            ),
            ky_layout=ky_layout,
        ),
        return_state=True,
        **_RUN_KWARGS,
    )
    assert np.asarray(sharded.state).shape[-3] == (8 if ky_layout == "full" else 5)
    assert serial.diagnostics is not None and sharded.diagnostics is not None
    # The live deck's field energy is nonzero, so the comparison can fail.
    assert np.max(np.abs(np.asarray(serial.diagnostics.Wphi_t))) > 0.0
    np.testing.assert_allclose(
        np.asarray(sharded.state), np.asarray(serial.state), rtol=0.0, atol=5.0e-6
    )
    for name in ("Wg_t", "Wphi_t", "heat_flux_t", "particle_flux_t"):
        np.testing.assert_allclose(
            np.asarray(getattr(sharded.diagnostics, name)),
            np.asarray(getattr(serial.diagnostics, name)),
            rtol=1.0e-4,
            atol=5.0e-6,
        )


def test_ky_route_scan_runs_replicated(monkeypatch) -> None:
    """Pin the documented limit of the ``ky`` route: its scan is not split.

    ``docs/parallelization.rst`` says the scan's input is replicated on every
    device even when the ky extent divides the device count. If a change makes
    the route partition for real, this fails, and that paragraph -- and the
    claim that an uneven extent loses nothing by entering replicated -- must be
    rewritten with it.
    """

    import gkx.solvers_nonlinear_diagnostics as diagnostics

    _require_devices(2)
    seen: list[object] = []
    original = diagnostics._run_explicit_diagnostic_scan_and_finalize

    def spy(prepared, *args, **kwargs):
        seen.append(prepared.G0.sharding)
        return original(prepared, *args, **kwargs)

    monkeypatch.setattr(diagnostics, "_run_explicit_diagnostic_scan_and_finalize", spy)
    run_runtime_nonlinear(
        _nonlinear_cfg(
            RuntimeParallelConfig(
                strategy="shard_map", axis="ky", num_devices=2, strict_identity=False
            )
        ),
        **_RUN_KWARGS,
    )
    assert len(seen) == 1
    assert seen[0].is_fully_replicated and len(seen[0].device_set) == 2


def test_shard_map_nonlinear_run_reports_the_route() -> None:
    _require_devices(2)
    messages: list[str] = []
    run_runtime_nonlinear(
        _nonlinear_cfg(
            RuntimeParallelConfig(strategy="shard_map", axis="ky", num_devices=2)
        ),
        status_callback=messages.append,
        **_RUN_KWARGS,
    )
    assert any("routing nonlinear run through" in message for message in messages)
    assert any("identity gate passed" in message for message in messages)


def _inject_shard_perturbation(monkeypatch) -> None:
    """Offset the sharded initial state far above the identity tolerance.

    The offset has to beat ``atol`` in absolute terms: the gate is an
    ``abs or rel`` test, so scaling a ``1e-8`` initial amplitude would stay
    inside the absolute tolerance and prove nothing.
    """

    original = nonlinear_workflow.shard_nonlinear_state

    def perturbed(state, plan):
        sharded = original(state, plan)
        return sharded + jnp.asarray(1.0e-3, dtype=sharded.dtype)

    monkeypatch.setattr(nonlinear_workflow, "shard_nonlinear_state", perturbed)


def test_strict_identity_raises_when_the_sharded_answer_drifts(monkeypatch) -> None:
    """Injecting a perturbation into the sharded state must fail the run."""

    _require_devices(2)
    _inject_shard_perturbation(monkeypatch)
    with pytest.raises(NonlinearParallelIdentityError, match="final_state"):
        run_runtime_nonlinear(
            _nonlinear_cfg(
                RuntimeParallelConfig(strategy="shard_map", axis="ky", num_devices=2)
            ),
            **_RUN_KWARGS,
        )


def test_strict_identity_false_skips_the_serial_reference(monkeypatch) -> None:
    """strict_identity is the only thing standing between the two behaviours."""

    _require_devices(2)
    _inject_shard_perturbation(monkeypatch)
    messages: list[str] = []
    result = run_runtime_nonlinear(
        _nonlinear_cfg(
            RuntimeParallelConfig(
                strategy="shard_map",
                axis="ky",
                num_devices=2,
                strict_identity=False,
            )
        ),
        status_callback=messages.append,
        **_RUN_KWARGS,
    )
    assert result.diagnostics is not None
    assert any("skipping the serial identity gate" in message for message in messages)


def test_unsupported_strategy_fails_the_runtime_run_before_any_work() -> None:
    with pytest.raises(NonlinearParallelRoutingError, match="shard_map"):
        run_runtime_nonlinear(
            _nonlinear_cfg(RuntimeParallelConfig(strategy="batch", axis="ky")),
            **_RUN_KWARGS,
        )


# ---- [time] state_sharding: real placement, results read back ----


def _live_context(monkeypatch, ky_layout: str):
    """Return the runtime context (initial state, grid, geometry, ...) of the live deck."""

    captured: dict = {}
    original = nonlinear_workflow._run_once

    def spy(cfg, ctx, *args, **kwargs):
        captured["ctx"] = ctx
        return original(cfg, ctx, *args, **kwargs)

    monkeypatch.setattr(nonlinear_workflow, "_run_once", spy)
    cfg = _live_cfg(ky_layout=ky_layout)
    run_runtime_nonlinear(cfg, **{**_RUN_KWARGS, "steps": 1})
    monkeypatch.setattr(nonlinear_workflow, "_run_once", original)
    return cfg, captured["ctx"]


@pytest.mark.parametrize(
    ("ky_layout", "spec", "num_devices"),
    [
        ("full", "ky", 2),
        ("full", "ky", 4),
        ("half", "ky", 2),
        ("half", "ky", 4),
        ("full", "kx", 4),
    ],
)
def test_time_state_sharding_matches_serial(
    monkeypatch, ky_layout: str, spec: str, num_devices: int
) -> None:
    """``[time] state_sharding`` runs on real devices and reproduces the serial run.

    Before the fix, the two-sided ky split failed on XLA:CPU with the FFT
    thunk's layout RET_CHECK, and the half layout (``Nyc = 5``) raised
    ``IndivisibleError`` from ``device_put``. The failure only surfaced when
    the result was read, so the test reads both outputs back.
    """

    import gkx.solvers_time_runners as runners
    from gkx.parallel.state import resolve_state_sharding

    _require_devices(num_devices)
    cfg, ctx = _live_context(monkeypatch, ky_layout)
    devices = jax.devices()[:num_devices]
    monkeypatch.setattr(
        runners,
        "resolve_state_sharding",
        lambda G0, name: resolve_state_sharding(G0, name, devices=devices),
    )

    def run(sharding: str | None):
        time_cfg = replace(
            cfg.time, dt=ctx.dt, t_max=20 * ctx.dt, state_sharding=sharding
        )
        return runners.integrate_nonlinear_from_config(
            jnp.array(ctx.G0, copy=True),
            ctx.grid,
            ctx.geom,
            ctx.params,
            time_cfg,
            terms=ctx.terms,
        )

    serial_state, serial_fields = run(None)
    state, fields = run(spec)
    serial_state = np.asarray(serial_state)
    assert np.asarray(state).shape == serial_state.shape
    assert serial_state.shape[-3] == (8 if ky_layout == "full" else 5)
    # The deck must be live, or the comparison below proves nothing.
    assert np.max(np.abs(serial_state - np.asarray(ctx.G0))) > 1.0e-5
    assert np.max(np.abs(np.asarray(serial_fields.phi))) > 1.0e-4
    np.testing.assert_allclose(np.asarray(state), serial_state, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        np.asarray(fields.phi), np.asarray(serial_fields.phi), rtol=0.0, atol=1.0e-12
    )


# ---- from test_parallel_artifacts.py ----
# Unit contracts: parallel artifacts.


ROOT = REPO_ROOT
STATIC = ROOT / "docs" / "_static"


def _load_json(name: str) -> dict:
    return json.loads((STATIC / name).read_text(encoding="utf-8"))


def _load_toml(path: Path) -> dict:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _load_parallel_checker():
    return load_release_tool("check_parallel_scaling_artifacts")


def _write_nonlinear_sharding_source_artifacts(
    tmp_path: Path, rows: list[dict]
) -> None:
    split_names = {
        "cpu": "nonlinear_sharding_strong_scaling_cpu_large.json",
        "gpu": "nonlinear_sharding_strong_scaling_gpu_xlarge.json",
    }
    for backend, name in split_names.items():
        backend_rows = [row for row in rows if row.get("backend") == backend]
        (tmp_path / name).write_text(
            json.dumps({"rows": backend_rows}),
            encoding="utf-8",
        )


def _assert_positive_stats(stats: dict) -> None:
    assert stats["min"] > 0.0
    assert stats["median"] > 0.0
    assert stats["mean"] > 0.0
    assert stats["max"] > 0.0
    assert stats["min"] <= stats["median"] <= stats["max"]
    assert stats["std"] >= 0.0


def _assert_worker_timing_payload(row: dict) -> None:
    assert row["error"] is None
    assert row["timed_wall_s"] > 0.0
    assert row["wall_s"] > 0.0
    assert row["strong_speedup_vs_1_device"] > 0.0
    assert row["parallel_efficiency"] > 0.0
    assert len(row["worker_stats"]) == row["actual_workers"]
    for worker in row["worker_stats"]:
        assert worker["samples_s"]
        assert all(sample > 0.0 for sample in worker["samples_s"])
        _assert_positive_stats(worker["stats_s"])


def test_parallel_manifests_track_current_cpu_gpu_scaling_artifacts() -> None:
    required = {
        "docs/_static/independent_ky_scan_scaling_large.json",
        "docs/_static/independent_ky_scan_scaling_large.csv",
        "docs/_static/independent_ky_scan_scaling_large.png",
        "docs/_static/independent_ky_scan_scaling_cpu_large.json",
        "docs/_static/independent_ky_scan_scaling_cpu_large.csv",
        "docs/_static/independent_ky_scan_scaling_cpu_large.png",
        "docs/_static/independent_ky_scan_scaling_gpu_large.json",
        "docs/_static/independent_ky_scan_scaling_gpu_large.csv",
        "docs/_static/independent_ky_scan_scaling_gpu_large.png",
        "docs/_static/quasilinear_uq_ensemble_scaling_large.json",
        "docs/_static/quasilinear_uq_ensemble_scaling_large.csv",
        "docs/_static/quasilinear_uq_ensemble_scaling_large.png",
        "docs/_static/quasilinear_uq_ensemble_scaling_cpu_large.json",
        "docs/_static/quasilinear_uq_ensemble_scaling_cpu_large.csv",
        "docs/_static/quasilinear_uq_ensemble_scaling_cpu_large.png",
        "docs/_static/quasilinear_uq_ensemble_scaling_gpu_large.json",
        "docs/_static/quasilinear_uq_ensemble_scaling_gpu_large.csv",
        "docs/_static/quasilinear_uq_ensemble_scaling_gpu_large.png",
        "docs/_static/parallelization_completion_status.json",
        "docs/_static/parallelization_completion_status.png",
        "docs/_static/nonlinear_sharding_strong_scaling_large.json",
        "docs/_static/nonlinear_sharding_strong_scaling_large.csv",
        "docs/_static/nonlinear_sharding_strong_scaling_large.png",
        "docs/_static/nonlinear_sharding_strong_scaling_cpu_large.json",
        "docs/_static/nonlinear_sharding_strong_scaling_cpu_large.csv",
        "docs/_static/nonlinear_sharding_strong_scaling_cpu_large.png",
        "docs/_static/nonlinear_sharding_strong_scaling_gpu_xlarge.json",
        "docs/_static/nonlinear_sharding_strong_scaling_gpu_xlarge.csv",
        "docs/_static/nonlinear_sharding_strong_scaling_gpu_xlarge.png",
        "docs/_static/nonlinear_sharding_production_speedup_gate.json",
        "docs/_static/nonlinear_sharding_production_speedup_gate.csv",
        "docs/_static/nonlinear_device_z_pencil_transport_gpu2_observable_split_profile.json",
        "docs/_static/nonlinear_device_z_pencil_transport_gpu2_observable_split_profile.csv",
        "docs/_static/nonlinear_device_z_pencil_transport_gpu2_observable_split_profile.png",
        "docs/_static/linear_rhs_parallel_slices_sweep.json",
        "docs/_static/linear_rhs_parallel_slices_sweep.csv",
        "docs/_static/linear_rhs_parallel_slices_sweep.png",
    }

    performance = _load_toml(ROOT / "tools" / "performance_optimization_manifest.toml")
    parallel_lane = next(
        lane for lane in performance["lanes"] if lane["name"] == "parallel_scaling"
    )
    validation = _load_toml(ROOT / "tools" / "validation_coverage_manifest.toml")
    validation_paths = {
        path
        for module in validation["modules"]
        if module["module"] in {"gkx.parallel.__init__", "gkx.parallel.state"}
        for path in module["artifact_paths"]
    }

    assert required <= set(parallel_lane["artifact_paths"])
    assert required <= validation_paths
    parallel_checker = _load_parallel_checker()
    validation_checker = load_release_tool("check_validation_coverage_manifest")
    assert (
        parallel_checker.RENDERED_ARTIFACT_SUFFIXES
        == validation_checker.RENDERED_ARTIFACT_SUFFIXES
    )
    for artifact in required:
        if Path(artifact).suffix in parallel_checker.RENDERED_ARTIFACT_SUFFIXES:
            continue
        assert (ROOT / artifact).exists(), artifact


def test_parallelization_completion_status_scopes_production_and_diagnostic_lanes() -> (
    None
):
    payload = _load_json("parallelization_completion_status.json")

    assert payload["kind"] == "parallelization_completion_status"
    assert payload["passed"] is True
    assert payload["production_completion_percent"] == 100.0
    assert "Release production parallelization is closed" in payload["claim_scope"]
    lanes = {lane["lane"]: lane for lane in payload["lanes"]}
    assert lanes["independent_ky_scan"]["status"] == "production_closed"
    assert lanes["quasilinear_uq_ensemble"]["status"] == "production_closed"
    assert (
        lanes["independent_ky_scan"]["source_contract"]["claim_separation_passed"]
        is True
    )
    assert lanes["independent_ky_scan"]["source_contract"]["input_backends"] == [
        "cpu",
        "gpu",
    ]
    assert (
        lanes["quasilinear_uq_ensemble"]["source_contract"]["claim_separation_passed"]
        is True
    )
    assert lanes["independent_ky_scan"]["best_speedups"]["cpu"] >= 5.0
    assert lanes["independent_ky_scan"]["best_speedups"]["gpu"] >= 1.5
    assert (
        lanes["whole_state_nonlinear_sharding"]["status"]
        == "diagnostic_closed_not_production"
    )
    assert (
        lanes["whole_state_nonlinear_sharding"]["source_contract"][
            "claim_separation_passed"
        ]
        is True
    )
    assert lanes["fft_axis_domain"]["status"] == "diagnostic_identity_closed"


def test_nonlinear_domain_parallel_identity_gate_is_scoped_and_fail_closed() -> None:
    payload = _load_json("nonlinear_domain_parallel_identity_gate.json")

    assert payload["case"] == "Nonlinear state-domain decomposition identity gate"
    assert payload["gate"]["identity_passed"] is True
    assert payload["gate"]["decomposed_path_enabled"] is True
    assert payload["gated_state_matches_serial"] is True
    assert payload["gated_state_matches_decomposed"] is True
    assert payload["gate"]["max_abs_error"] <= payload["gate"]["atol"]
    assert payload["gate"]["max_rel_error"] <= payload["gate"]["rtol"]
    assert payload["transport_window"]["gate"]["identity_passed"] is True
    assert payload["transport_window"]["gate"]["decomposed_path_enabled"] is True
    assert (
        payload["transport_window"]["gate"]["max_abs_state_error"]
        <= payload["gate"]["atol"]
    )
    assert (
        payload["transport_window"]["gate"]["mass_trace_max_abs_error"]
        <= payload["gate"]["atol"]
    )
    assert (
        payload["transport_window"]["gate"]["free_energy_trace_max_abs_error"]
        <= payload["gate"]["atol"]
    )
    assert (
        payload["transport_window"]["gate"]["flux_proxy_trace_max_abs_error"]
        <= payload["gate"]["atol"]
    )
    assert {row["metric"] for row in payload["transport_window"]["metrics"]} == {
        "mass_trace",
        "free_energy_trace",
        "boundary_flux_proxy_trace",
    }
    assert all(
        row["identity_passed"] is True for row in payload["transport_window"]["metrics"]
    )
    assert "no production routing or speedup claim" in payload["claim_scope"]


def test_nonlinear_spectral_communication_identity_gate_is_scoped_and_fail_closed() -> (
    None
):
    payload = _load_json("nonlinear_spectral_communication_identity_gate.json")

    assert payload["case"] == "Nonlinear spectral decomposition identity gate"
    assert payload["kind"] == "nonlinear_spectral_communication_identity_gate"
    assert payload["gate"]["identity_passed"] is True
    assert payload["gate"]["decomposed_path_enabled"] is True
    assert payload["gate"]["communication_identity_passed"] is True
    assert payload["gate"]["rhs_identity_passed"] is True
    assert payload["gate"]["integrator_identity_passed"] is True
    assert payload["gate"]["pencil_rhs_identity_passed"] is True
    assert payload["gate"]["pencil_transport_window_identity_passed"] is True
    assert payload["communication_gate"]["fft_max_abs_error"] <= payload["gate"]["atol"]
    assert (
        payload["communication_gate"]["bracket_max_abs_error"]
        <= payload["gate"]["atol"]
    )
    assert (
        payload["communication_gate"]["field_max_abs_error"] <= payload["gate"]["atol"]
    )
    assert payload["rhs_gate"]["rhs_max_abs_error"] <= payload["gate"]["atol"]
    assert (
        payload["integrator_gate"]["final_state_max_abs_error"]
        <= payload["gate"]["atol"]
    )
    assert (
        payload["integrator_gate"]["flux_proxy_trace_max_abs_error"]
        <= payload["gate"]["atol"]
    )
    assert payload["pencil_rhs_gate"]["rhs_max_abs_error"] <= payload["gate"]["atol"]
    assert (
        payload["pencil_transport_window_gate"]["final_state_max_abs_error"]
        <= payload["gate"]["atol"]
    )
    assert all(row["identity_passed"] is True for row in payload["rows"])
    assert {row["operator"] for row in payload["rows"]} == {
        "fft_forward_inverse",
        "nonlinear_bracket",
        "spectral_field_solve_layout",
        "logical_sharded_rhs",
        "logical_integrator_final_state",
        "logical_integrator_flux_proxy_trace",
        "pencil_fused_rhs",
        "pencil_physical_transport_window",
    }
    assert "pencil fused-bracket" in payload["claim_scope"]
    assert "physical transport-window identity gate" in payload["claim_scope"]
    assert (
        "no production distributed FFT routing or speedup claim"
        in payload["claim_scope"]
    )


def test_parallel_scaling_artifact_checker_validates_tracked_large_run_evidence() -> (
    None
):
    mod = _load_parallel_checker()

    summary = mod.validate_all()

    assert summary["n_families"] == 4
    assert summary["n_json_artifacts"] == 12
    assert summary["n_sidecars"] == 24
    assert summary["manifest_checked"] is True
    assert {family["name"] for family in summary["families"]} == {
        "independent_ky_scan",
        "quasilinear_uq_ensemble",
        "nonlinear_sharding",
        "linear_rhs_parallel_slices",
    }
    assert (
        summary["production_gate"]["name"]
        == "nonlinear_sharding_production_speedup_gate"
    )
    assert summary["production_gate"]["gate_passed"] is False
    assert summary["production_gate"]["status"] == "diagnostic_only"
    assert summary["production_gate"]["production_candidate_backends"] == ["cpu"]
    assert summary["observable_split"]["name"] == "device_z_pencil_observable_split"
    assert summary["observable_split"]["production_speedup_claim_allowed"] is False
    assert summary["observable_split"]["max_observable_gate_overhead_vs_compute"] > 1.0


def test_parallel_scaling_artifact_checker_validates_observable_split() -> None:
    mod = _load_parallel_checker()

    summary = mod.validate_device_z_pencil_observable_split(STATIC)

    assert summary["json"] == mod.OBSERVABLE_SPLIT_JSON
    assert summary["production_speedup_claim_allowed"] is False
    assert summary["max_speedup_vs_serial"] < summary["min_speedup"]
    assert summary["max_observable_gate_overhead_vs_compute"] > 10.0


def test_parallel_scaling_artifact_checker_rejects_promoted_observable_split(
    tmp_path: Path,
) -> None:
    mod = _load_parallel_checker()
    payload = _load_json(mod.OBSERVABLE_SPLIT_JSON)
    payload["summary"]["max_speedup_vs_serial"] = payload["min_speedup"]
    (tmp_path / mod.OBSERVABLE_SPLIT_JSON).write_text(
        json.dumps(payload), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="no longer below gate"):
        mod.validate_device_z_pencil_observable_split(tmp_path, check_sidecars=False)


def test_parallel_scaling_artifact_checker_rejects_failed_identity_gate(
    tmp_path: Path,
) -> None:
    mod = _load_parallel_checker()
    family = mod.ArtifactFamily(
        name="bad_identity",
        combined="bad_identity.json",
        split=(),
        expected_combined_kind="bad_identity",
        expected_split_kind=None,
        identity_claim_phrase="identity-only",
        split_identity_claim_phrase=None,
        timing_fields=("serial_median_s",),
        error_fields=("max_abs_error",),
        row_identity_key="identity_passed",
        combined_has_inputs=False,
    )
    (tmp_path / "bad_identity.json").write_text(
        json.dumps(
            {
                "kind": "bad_identity",
                "identity_passed": False,
                "claim_scope": "identity-only local test",
                "rows": [
                    {
                        "requested_devices": 1,
                        "identity_passed": True,
                        "serial_median_s": 1.0,
                        "max_abs_error": 0.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="identity_passed must be true"):
        mod.validate_family(tmp_path, family, check_sidecars=False)


def test_parallel_scaling_artifact_checker_rejects_tiny_problem_metadata(
    tmp_path: Path,
) -> None:
    mod = _load_parallel_checker()
    family = mod.ArtifactFamily(
        name="tiny",
        combined="tiny.json",
        split=(),
        expected_combined_kind="tiny_scaling",
        expected_split_kind=None,
        identity_claim_phrase="identity-only",
        split_identity_claim_phrase=None,
        timing_fields=("serial_median_s",),
        error_fields=("max_abs_error",),
        row_identity_key="identity_passed",
        combined_has_inputs=False,
        min_grid=(("Ny", 64), ("Nz", 32)),
        min_steps=100,
    )
    (tmp_path / "tiny.json").write_text(
        json.dumps(
            {
                "kind": "tiny_scaling",
                "identity_passed": True,
                "claim_scope": "identity-only local test",
                "grid": {"Ny": 16, "Nz": 32},
                "time": {"steps": 100},
                "rows": [
                    {
                        "requested_devices": 1,
                        "identity_passed": True,
                        "serial_median_s": 1.0,
                        "max_abs_error": 0.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "tiny.csv").write_text(
        "requested_devices,identity_passed,serial_median_s,max_abs_error\n"
        "1,true,1.0,0.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="grid Ny=16 is below required 64"):
        mod.validate_family(tmp_path, family, check_sidecars=False)


def test_parallel_scaling_artifact_checker_accepts_profile_source_contract(
    tmp_path: Path,
) -> None:
    mod = _load_parallel_checker()
    family = mod.ArtifactFamily(
        name="profile_contract",
        combined="profile_contract.json",
        split=(),
        expected_combined_kind="profile_contract",
        expected_split_kind=None,
        identity_claim_phrase="identity-only",
        split_identity_claim_phrase=None,
        timing_fields=("serial_median_s",),
        error_fields=("max_abs_error",),
        row_identity_key="identity_passed",
        combined_has_inputs=False,
    )
    row = {
        "requested_devices": 1,
        "actual_devices": 1,
        "backend": "gpu",
        "identity_passed": True,
        "serial_median_s": 1.0,
        "max_abs_error": 0.0,
        "source_contract_version": 1,
        "profile_command": "python scripts/profiling/profile_nonlinear_sharding.py --sharding kx",
        "profile_command_argv": [
            "python",
            "scripts/profiling/profile_nonlinear_sharding.py",
            "--sharding",
            "kx",
        ],
        "source_artifact": "docs/_static/profile.json",
        "software_versions": {
            "python": "3.11.0",
            "gkx": "test",
            "jax": "0.test",
            "jaxlib": "0.test",
            "numpy": "2.test",
        },
        "timing_warmup_repeat": {"warmups": 0, "repeats": 2},
        "profile_backend": "gpu",
        "profile_device_count": 1,
        "profile_sharding_axis": "kx",
    }
    (tmp_path / "profile_contract.json").write_text(
        json.dumps(
            {
                "kind": "profile_contract",
                "identity_passed": True,
                "claim_scope": "identity-only local test",
                "rows": [row],
            }
        ),
        encoding="utf-8",
    )

    summary = mod.validate_family(tmp_path, family, check_sidecars=False)

    assert summary["n_combined_rows"] == 1


def test_parallel_scaling_artifact_checker_rejects_stale_profile_source_contract(
    tmp_path: Path,
) -> None:
    mod = _load_parallel_checker()
    family = mod.ArtifactFamily(
        name="profile_contract",
        combined="profile_contract.json",
        split=(),
        expected_combined_kind="profile_contract",
        expected_split_kind=None,
        identity_claim_phrase="identity-only",
        split_identity_claim_phrase=None,
        timing_fields=("serial_median_s",),
        error_fields=("max_abs_error",),
        row_identity_key="identity_passed",
        combined_has_inputs=False,
    )
    row = {
        "requested_devices": 1,
        "actual_devices": 1,
        "backend": "cpu",
        "identity_passed": True,
        "serial_median_s": 1.0,
        "max_abs_error": 0.0,
        "source_contract_version": 1,
        "profile_command": "python scripts/profiling/profile_nonlinear_sharding.py --sharding kx",
        "profile_command_argv": [
            "python",
            "scripts/profiling/profile_nonlinear_sharding.py",
            "--sharding",
            "kx",
        ],
        "source_artifact": "docs/_static/profile.json",
        "software_versions": {
            "python": "3.11.0",
            "gkx": "test",
            "jax": "0.test",
            "jaxlib": "0.test",
            "numpy": "2.test",
        },
        "timing_warmup_repeat": {"warmups": 0, "repeats": 2},
        "profile_backend": "gpu",
        "profile_device_count": 1,
        "profile_sharding_axis": "kx",
    }
    (tmp_path / "profile_contract.json").write_text(
        json.dumps(
            {
                "kind": "profile_contract",
                "identity_passed": True,
                "claim_scope": "identity-only local test",
                "rows": [row],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="profile_backend must match row backend"):
        mod.validate_family(tmp_path, family, check_sidecars=False)


def test_parallel_scaling_artifact_checker_rejects_stale_production_gate(
    tmp_path: Path,
) -> None:
    mod = _load_parallel_checker()
    payload = {
        "kind": "nonlinear_sharding_production_speedup_gate",
        "claim_scope": (
            "Whole-state nonlinear sharding gate; otherwise keep it as a "
            "diagnostic identity/profiler artifact."
        ),
        "gate_passed": True,
        "production_speedup_claim_allowed": True,
        "status": "production_speedup_candidate",
        "required_backends": ["cpu", "gpu"],
        "min_devices": 2,
        "min_speedup_vs_1_device": 1.2,
        "min_parallel_efficiency": 0.5,
        "identity_atol": 1.0e-5,
        "identity_rtol": 1.0e-5,
        "best_candidates": {
            "cpu": {
                "backend": "cpu",
                "requested_devices": 2,
                "actual_devices": 2,
                "source": "docs/_static/nonlinear_sharding_strong_scaling_cpu_large.json",
                "strong_speedup_vs_1_device": 1.3,
            },
            "gpu": None,
        },
        "blockers": [],
        "rows": [
            {
                "backend": "cpu",
                "requested_devices": 2,
                "actual_devices": 2,
                "source": "docs/_static/nonlinear_sharding_strong_scaling_cpu_large.json",
                "state_sharding_active": True,
                "identity_gate_pass": True,
                "strong_speedup_vs_1_device": 1.3,
                "parallel_efficiency": 0.65,
                "max_abs_state_error": 0.0,
                "max_rel_state_error": 0.0,
                "candidate_passed": True,
                "classification": "production_candidate",
                "blockers": [],
            },
            {
                "backend": "gpu",
                "requested_devices": 2,
                "actual_devices": 2,
                "source": "docs/_static/nonlinear_sharding_strong_scaling_gpu_xlarge.json",
                "state_sharding_active": True,
                "identity_gate_pass": True,
                "strong_speedup_vs_1_device": 0.8,
                "parallel_efficiency": 0.4,
                "max_abs_state_error": 0.0,
                "max_rel_state_error": 0.0,
                "candidate_passed": False,
                "classification": "identity_preserving_regression",
                "blockers": [
                    "speedup_below_threshold",
                    "parallel_efficiency_below_threshold",
                ],
            },
        ],
    }
    _write_nonlinear_sharding_source_artifacts(tmp_path, payload["rows"])
    (tmp_path / mod.PRODUCTION_GATE_JSON).write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError, match="blockers do not match missing backend candidates"
    ):
        mod.validate_nonlinear_sharding_production_gate(tmp_path, check_sidecars=False)


def test_parallel_scaling_artifact_checker_rejects_stale_production_gate_source_row(
    tmp_path: Path,
) -> None:
    mod = _load_parallel_checker()
    source_row = {
        "backend": "cpu",
        "requested_devices": 2,
        "actual_devices": 2,
        "best_spec": "kx",
        "state_sharding_active": True,
        "identity_gate_pass": True,
        "strong_speedup_vs_1_device": 1.30,
        "max_abs_state_error": 0.0,
        "max_rel_state_error": 0.0,
    }
    gate_row = {
        **source_row,
        "source": "docs/_static/nonlinear_sharding_strong_scaling_cpu_large.json",
        "strong_speedup_vs_1_device": 1.40,
        "parallel_efficiency": 0.70,
        "candidate_passed": True,
        "classification": "production_candidate",
        "blockers": [],
    }
    _write_nonlinear_sharding_source_artifacts(
        tmp_path,
        [
            source_row,
            {
                "backend": "gpu",
                "requested_devices": 1,
                "actual_devices": 1,
                "best_spec": "auto",
                "state_sharding_active": False,
                "identity_gate_pass": True,
                "strong_speedup_vs_1_device": 1.0,
                "max_abs_state_error": 0.0,
                "max_rel_state_error": 0.0,
            },
        ],
    )
    payload = {
        "kind": "nonlinear_sharding_production_speedup_gate",
        "claim_scope": (
            "Whole-state nonlinear sharding gate; otherwise keep it as a "
            "diagnostic identity/profiler artifact."
        ),
        "gate_passed": True,
        "production_speedup_claim_allowed": True,
        "status": "production_speedup_candidate",
        "required_backends": ["cpu"],
        "min_devices": 2,
        "min_speedup_vs_1_device": 1.2,
        "min_parallel_efficiency": 0.5,
        "identity_atol": 1.0e-5,
        "identity_rtol": 1.0e-5,
        "best_candidates": {"cpu": gate_row},
        "blockers": [],
        "rows": [gate_row],
    }
    (tmp_path / mod.PRODUCTION_GATE_JSON).write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="stale relative to source artifact"):
        mod.validate_nonlinear_sharding_production_gate(tmp_path, check_sidecars=False)


def test_parallel_scaling_artifact_checker_rejects_stale_production_gate_blocker_report(
    tmp_path: Path,
) -> None:
    mod = _load_parallel_checker()
    rows = [
        {
            "backend": "cpu",
            "requested_devices": 2,
            "actual_devices": 2,
            "best_spec": "kx",
            "state_sharding_active": True,
            "identity_gate_pass": True,
            "strong_speedup_vs_1_device": 1.30,
            "parallel_efficiency": 0.65,
            "max_abs_state_error": 0.0,
            "max_rel_state_error": 0.0,
            "source": "docs/_static/nonlinear_sharding_strong_scaling_cpu_large.json",
            "candidate_passed": True,
            "classification": "production_candidate",
            "blockers": [],
        },
        {
            "backend": "gpu",
            "requested_devices": 2,
            "actual_devices": 2,
            "best_spec": "kx",
            "state_sharding_active": True,
            "identity_gate_pass": True,
            "strong_speedup_vs_1_device": 0.80,
            "parallel_efficiency": 0.40,
            "max_abs_state_error": 0.0,
            "max_rel_state_error": 0.0,
            "source": "docs/_static/nonlinear_sharding_strong_scaling_gpu_xlarge.json",
            "candidate_passed": False,
            "classification": "identity_preserving_regression",
            "blockers": [
                "speedup_below_threshold",
                "parallel_efficiency_below_threshold",
            ],
        },
    ]
    _write_nonlinear_sharding_source_artifacts(tmp_path, rows)
    payload = {
        "kind": "nonlinear_sharding_production_speedup_gate",
        "claim_scope": (
            "Whole-state nonlinear sharding gate; otherwise keep it as a "
            "diagnostic identity/profiler artifact."
        ),
        "gate_passed": False,
        "production_speedup_claim_allowed": False,
        "status": "diagnostic_only",
        "required_backends": ["cpu", "gpu"],
        "min_devices": 2,
        "min_speedup_vs_1_device": 1.2,
        "min_parallel_efficiency": 0.5,
        "identity_atol": 1.0e-5,
        "identity_rtol": 1.0e-5,
        "best_candidates": {"cpu": rows[0], "gpu": None},
        "blockers": ["gpu_production_speedup_candidate_missing"],
        "backend_blocker_report": {
            "cpu": {
                "row_count": 1,
                "candidate_row_count": 1,
                "passing_candidate_count": 1,
                "production_speedup_candidate_missing": False,
                "identity_evidence_complete": True,
                "active_identity_evidence_complete": True,
                "classification_counts": {"production_candidate": 1},
                "candidate_blocker_counts": {},
                "primary_blockers": [],
                "claim_scope": "stale",
            },
            "gpu": {
                "row_count": 1,
                "candidate_row_count": 1,
                "passing_candidate_count": 1,
                "production_speedup_candidate_missing": False,
                "identity_evidence_complete": True,
                "active_identity_evidence_complete": True,
                "classification_counts": {"identity_preserving_regression": 1},
                "candidate_blocker_counts": {},
                "primary_blockers": [],
                "claim_scope": "stale",
            },
        },
        "rows": rows,
    }
    (tmp_path / mod.PRODUCTION_GATE_JSON).write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="backend_blocker_report"):
        mod.validate_nonlinear_sharding_production_gate(tmp_path, check_sidecars=False)


def test_independent_ky_scaling_artifact_preserves_order_and_identity_scope() -> None:
    payload = _load_json("independent_ky_scan_scaling_large.json")

    assert payload["kind"] == "independent_ky_scan_scaling_combined"
    assert payload["identity_passed"] is True
    assert "not a nonlinear domain-decomposition" in payload["claim_scope"]
    for backend in {"cpu", "gpu"}:
        rows = sorted(
            (row for row in payload["rows"] if row["backend"] == backend),
            key=lambda row: row["requested_devices"],
        )
        assert rows
        reference_ky = rows[0]["ky"]
        reference_gamma = rows[0]["gamma"]
        reference_omega = rows[0]["omega"]
        for row in rows:
            assert row["identity_gate_pass"] is True
            assert row["actual_workers"] <= row["requested_devices"]
            assert row["ky"] == reference_ky
            assert row["gamma"] == reference_gamma
            assert row["omega"] == reference_omega
            assert row["max_gamma_rel_error"] == 0.0
            assert row["max_omega_abs_error"] == 0.0


def test_independent_ky_split_artifacts_are_large_solver_backed_identity_profiles() -> (
    None
):
    for backend, artifact in {
        "cpu": "independent_ky_scan_scaling_cpu_large.json",
        "gpu": "independent_ky_scan_scaling_gpu_large.json",
    }.items():
        payload = _load_json(artifact)

        assert payload["kind"] == "independent_ky_scan_strong_scaling"
        assert payload["backend"] == backend
        assert payload["identity_passed"] is True
        assert "not a nonlinear domain-decomposition" in payload["claim_scope"]
        assert len(payload["ky"]) >= 8
        assert payload["warmups"] >= 1
        assert payload["repeats"] >= 1
        assert payload["time"]["steps"] >= 200
        assert payload["grid"]["Ny"] >= 96
        assert payload["grid"]["Nz"] >= 64
        assert payload["grid"]["Nl"] >= 4
        assert payload["grid"]["Nm"] >= 8

        rows = sorted(payload["rows"], key=lambda row: row["requested_devices"])
        assert rows[0]["requested_devices"] == 1
        assert rows[-1]["requested_devices"] > 1
        reference = rows[0]
        for row in rows:
            assert row["identity_gate_pass"] is True
            assert row["actual_workers"] <= row["requested_devices"]
            assert row["ky"] == reference["ky"]
            assert row["gamma"] == reference["gamma"]
            assert row["omega"] == reference["omega"]
            assert row["max_gamma_abs_error"] == 0.0
            assert row["max_gamma_rel_error"] == 0.0
            assert row["max_omega_abs_error"] == 0.0
            _assert_worker_timing_payload(row)


def test_quasilinear_uq_scaling_artifact_preserves_member_order_and_identity_scope() -> (
    None
):
    payload = _load_json("quasilinear_uq_ensemble_scaling_large.json")

    assert payload["kind"] == "quasilinear_uq_ensemble_scaling_combined"
    assert payload["identity_passed"] is True
    assert (
        "not a promoted absolute nonlinear heat-flux predictor"
        in payload["claim_scope"]
    )
    for backend in {"cpu", "gpu"}:
        rows = sorted(
            (row for row in payload["rows"] if row["backend"] == backend),
            key=lambda row: row["requested_devices"],
        )
        assert rows
        reference_gradients = [member["tprim"] for member in rows[0]["members"]]
        reference_flux = [member["heat_flux_proxy"] for member in rows[0]["members"]]
        for row in rows:
            gradients = [member["tprim"] for member in row["members"]]
            flux = [member["heat_flux_proxy"] for member in row["members"]]
            assert row["identity_gate_pass"] is True
            assert row["actual_workers"] <= row["requested_devices"]
            assert gradients == reference_gradients
            assert flux == reference_flux
            assert row["max_heat_flux_proxy_rel_error"] == 0.0
            assert row["max_gamma_abs_error"] == 0.0


def test_quasilinear_uq_split_artifacts_are_large_solver_backed_identity_profiles() -> (
    None
):
    for backend, artifact in {
        "cpu": "quasilinear_uq_ensemble_scaling_cpu_large.json",
        "gpu": "quasilinear_uq_ensemble_scaling_gpu_large.json",
    }.items():
        payload = _load_json(artifact)

        assert payload["kind"] == "quasilinear_uq_ensemble_scaling"
        assert payload["backend"] == backend
        assert payload["identity_passed"] is True
        assert (
            "not an absolute nonlinear heat-flux validation claim"
            in payload["claim_scope"]
        )
        assert len(payload["gradients"]) >= 6
        assert len(payload["ky"]) >= 5
        assert payload["warmups"] >= 1
        assert payload["repeats"] >= 1
        assert payload["time"]["steps"] >= 1000
        assert payload["grid"]["Ny"] >= 64
        assert payload["grid"]["Nz"] >= 64
        assert payload["grid"]["Nl"] >= 3
        assert payload["grid"]["Nm"] >= 6

        rows = sorted(payload["rows"], key=lambda row: row["requested_devices"])
        assert rows[0]["requested_devices"] == 1
        assert rows[-1]["requested_devices"] > 1
        reference_gradients = [member["tprim"] for member in rows[0]["members"]]
        reference_flux = [member["heat_flux_proxy"] for member in rows[0]["members"]]
        for row in rows:
            gradients = [member["tprim"] for member in row["members"]]
            flux = [member["heat_flux_proxy"] for member in row["members"]]
            assert row["identity_gate_pass"] is True
            assert row["actual_workers"] <= row["requested_devices"]
            assert gradients == reference_gradients
            assert flux == reference_flux
            assert row["max_heat_flux_proxy_abs_error"] == 0.0
            assert row["max_heat_flux_proxy_rel_error"] == 0.0
            assert row["max_gamma_abs_error"] == 0.0
            _assert_worker_timing_payload(row)


def test_nonlinear_whole_state_scaling_artifact_is_identity_only_not_speedup_claim() -> (
    None
):
    payload = _load_json("nonlinear_sharding_strong_scaling_large.json")

    assert payload["kind"] == "nonlinear_sharding_strong_scaling_combined"
    assert payload["identity_passed"] is True
    assert payload["speedup_passed"] is False
    assert payload["status"] == "diagnostic_identity_only"
    assert payload["speedup_blockers"]
    assert "not a production speedup claim" in payload["claim_scope"]
    assert {row["backend"] for row in payload["rows"]} == {"cpu", "gpu"}
    for row in payload["rows"]:
        assert row["identity_gate_pass"] is True
        assert row["max_abs_state_error"] == 0.0
        assert row["max_rel_state_error"] == 0.0
        assert row["strong_speedup_vs_1_device"] > 0.0
        assert row["parallel_median_s"] > 0.0
        assert row["best_spec"] in {"auto", "ky", "kx"}
        if row["actual_devices"] < 2:
            assert row["state_sharding_active"] is False


def test_nonlinear_strong_scaling_split_artifacts_embed_profiler_payloads() -> None:
    artifacts = {
        "cpu": (
            "nonlinear_sharding_strong_scaling_cpu_large.json",
            {"Nx": 24, "Ny_requested": 48, "Nz": 96, "Nl": 4, "Nm": 8},
        ),
        "gpu": (
            "nonlinear_sharding_strong_scaling_gpu_xlarge.json",
            {"Nx": 48, "Ny_requested": 96, "Nz": 128, "Nl": 4, "Nm": 8},
        ),
    }

    for backend, (artifact, expected_grid) in artifacts.items():
        payload = _load_json(artifact)

        assert payload["kind"] == "nonlinear_sharding_strong_scaling_sweep"
        assert payload["backend"] == backend
        assert payload["grid"] == expected_grid
        assert payload["steps"] >= 8
        assert payload["identity_passed"] is True
        assert "not as a broad production speedup claim" in payload["claim_scope"]
        assert set(payload["profiles"]) == {
            str(row["requested_devices"]) for row in payload["rows"]
        }

        for row in payload["rows"]:
            assert row["error"] is None
            assert row["identity_gate_pass"] is True
            assert row["actual_devices"] <= row["requested_devices"]
            assert row["max_abs_state_error"] == 0.0
            assert row["max_rel_state_error"] == 0.0
            assert row["parallel_median_s"] > 0.0
            assert row["serial_median_s"] > 0.0
            assert row["same_process_speedup"] > 0.0
            assert row["strong_speedup_vs_1_device"] > 0.0
            assert row["best_spec"] in {"auto", "ky", "kx"}
            if row["actual_devices"] >= 2:
                assert row["state_sharding_active"] is True

            profile = payload["profiles"][str(row["requested_devices"])]
            assert profile["_profile_json"] == row["profile_json"]
            assert profile["default_backend"] == backend
            assert profile["device_count"] == row["actual_devices"]
            assert profile["state_shape"] == row["state_shape"]
            assert profile["identity_gate_pass"] is True
            assert "Do not use as a published runtime claim" in profile["claim_scope"]
            assert set(profile["profiler_trace"]) >= {"requested", "path", "error"}
            assert profile["profiler_trace"]["error"] is None
            _assert_positive_stats(profile["serial_stats_s"])

            best = profile["best_identity_preserving_candidate"]
            assert best["spec"] == row["best_spec"]
            assert best["identity_gate_pass"] is True
            result = profile["sharded_results"][row["best_spec"]]
            assert result["identity_gate_pass"] is True
            assert result["max_abs_state_error"] == row["max_abs_state_error"]
            assert result["max_rel_state_error"] == row["max_rel_state_error"]
            _assert_positive_stats(result["stats_s"])


def test_parallel_docs_keep_speedup_claims_tied_to_current_artifacts() -> None:
    docs = "\n".join(
        (ROOT / path).read_text(encoding="utf-8")
        for path in (
            "docs/parallelization.rst",
            "docs/performance.rst",
            "docs/testing.rst",
        )
    )

    for artifact in (
        "independent_ky_scan_scaling_large.json",
        "quasilinear_uq_ensemble_scaling_large.json",
        "nonlinear_sharding_strong_scaling_large.json",
        "linear_rhs_parallel_slices_sweep.json",
    ):
        assert artifact in docs
    compact_docs = " ".join(docs.split())
    assert "Large-run scaling acceptance checklist" in compact_docs
    assert "fresh profiler artifacts for the exact workload" in compact_docs
    assert "not a production nonlinear speedup claim" in compact_docs


def _assert_independent_ky_scaling_artifact(payload: dict) -> None:
    assert payload["identity_passed"] is True
    assert "independent ky" in payload["claim_scope"].lower()
    assert payload["grid"] == {"Nl": 4, "Nm": 8, "Nx": 1, "Ny": 128, "Nz": 96}
    assert payload["time"]["steps"] == 240
    assert len(payload["ky"]) == 64
    for row in payload["rows"]:
        assert row["identity_gate_pass"] is True
        assert row["max_gamma_rel_error"] == 0.0
        assert row["max_omega_abs_error"] == 0.0
        assert row["strong_speedup_vs_1_device"] > 0.0


def test_independent_ky_cpu_gpu_scaling_artifacts_are_identity_gated() -> None:
    cpu = _load_json("independent_ky_scan_scaling_cpu_large.json")
    gpu = _load_json("independent_ky_scan_scaling_gpu_large.json")
    combined = _load_json("independent_ky_scan_scaling_large.json")

    _assert_independent_ky_scaling_artifact(cpu)
    assert cpu["backend"] == "cpu"
    assert [row["requested_devices"] for row in cpu["rows"]] == [1, 2, 4, 8]
    assert cpu["rows"][-1]["strong_speedup_vs_1_device"] > 7.0

    _assert_independent_ky_scaling_artifact(gpu)
    assert gpu["backend"] == "gpu"
    assert [row["requested_devices"] for row in gpu["rows"]] == [1, 2]
    assert gpu["rows"][-1]["strong_speedup_vs_1_device"] > 1.8

    assert combined["identity_passed"] is True
    assert {row["backend"] for row in combined["rows"]} == {"cpu", "gpu"}
    assert "not a nonlinear domain-decomposition" in combined["claim_scope"]


def _assert_nonlinear_sharding_identity_artifact(payload: dict) -> None:
    assert payload["identity_gate_pass"] is True
    assert payload["sharding_options"] == ["auto", "kx"]
    assert "Do not use as a published runtime claim" in payload["claim_scope"]
    for axis in payload["sharding_options"]:
        result = payload["sharded_results"][axis]
        assert result["identity_gate_pass"] is True
        assert result["error"] is None
        assert result["max_abs_state_error"] == 0.0
        assert result["max_rel_state_error"] == 0.0
        assert result["diagnostic_identity_gate_pass"] is True
        assert result["max_abs_phi_error"] == 0.0
        assert result["max_rel_phi_error"] == 0.0
        assert result["max_abs_rhs_error"] == 0.0
        assert result["max_rel_rhs_error"] == 0.0


def test_nonlinear_sharding_profiles_are_identity_gated_and_scoped() -> None:
    local = _load_json("nonlinear_sharding_profile.json")
    gpu = _load_json("nonlinear_sharding_profile_gpu.json")
    benchmark_gpu = _load_json(
        "nonlinear_sharding_profile_gpu_benchmark_grid.json"
    )

    _assert_nonlinear_sharding_identity_artifact(local)
    assert local["default_backend"] == "cpu"
    assert local["state_sharding_active"] is False

    _assert_nonlinear_sharding_identity_artifact(gpu)
    assert gpu["default_backend"] == "gpu"
    assert gpu["device_count"] >= 2
    assert gpu["state_sharding_active"] is True
    assert gpu["profiler_trace"]["requested"] is True

    # The old profile is a tiny control-flow smoke. The matched benchmark grid
    # is the production-candidate gate and must fail closed on trajectory drift.
    assert benchmark_gpu["default_backend"] == "gpu"
    assert benchmark_gpu["device_count"] == 2
    assert benchmark_gpu["state_shape"] == [4, 8, 64, 192, 24]
    assert benchmark_gpu["state_sharding_active"] is True
    assert benchmark_gpu["identity_gate_pass"] is False
    assert benchmark_gpu["max_abs_state_error"] > 1.0
    assert benchmark_gpu["engineering_speedup"] < 1.0
    assert benchmark_gpu["best_identity_preserving_candidate"]["spec"] is None
    assert (
        benchmark_gpu["best_identity_preserving_candidate"]["identity_gate_pass"]
        is False
    )
    assert (
        benchmark_gpu["sharded_results"]["kx"]["diagnostic_identity_gate_pass"] is False
    )
    assert benchmark_gpu["sharded_results"]["kx"]["max_abs_rhs_error"] > 1.0
    assert benchmark_gpu["sharded_results"]["kx"]["identity_gate_pass"] is False


def test_device_z_transport_window_profiles_are_identity_gated_and_scoped() -> None:
    cpu = _load_json("nonlinear_device_z_pencil_transport_cpu4_profile.json")
    gpu = _load_json("nonlinear_device_z_pencil_transport_gpu2_profile.json")

    for payload in (cpu, gpu):
        assert payload["kind"] == "nonlinear_device_z_pencil_transport_window_profile"
        assert (
            "not yet a full production nonlinear turbulent-transport solve"
            in payload["claim_scope"]
        )
        assert payload["summary"]["all_active_identity_passed"] is True
        assert payload["summary"]["full_solver_speedup_claim_allowed"] is False
        assert payload["shape"] == [4, 16, 96, 96, 32]
        assert payload["steps"] == 4
        active_rows = [
            row for row in payload["rows"] if row["active"] and row["device_count"] > 1
        ]
        assert active_rows
        for row in active_rows:
            assert row["identity_passed"] is True
            assert row["transport_window_identity_passed"] is True
            assert row["final_state_max_abs_error"] <= payload["atol"]
            assert row["physical_flux_trace_max_abs_error"] <= payload["atol"]
            assert row["transport_window_report"]["identity_passed"] is True

    assert cpu["backend"] == "cpu"
    assert cpu["summary"]["transport_window_speedup_claim_allowed"] is True
    assert cpu["summary"]["max_speedup_vs_serial"] >= 1.5
    assert cpu["hlo"]["device_4"]["all_to_all"] == 0
    assert cpu["hlo"]["device_4"]["collective_permute"] == 0
    assert cpu["trace"]["requested"] is True

    assert gpu["backend"] == "gpu"
    assert gpu["summary"]["transport_window_speedup_claim_allowed"] is False
    assert gpu["rows"][1]["blocked_reasons"] == ["speedup_below_gate"]
    assert gpu["hlo"]["device_2"]["all_to_all"] == 0
    assert gpu["hlo"]["device_2"]["collective_permute"] == 0
    assert gpu["trace"]["requested"] is True


def test_quasilinear_uq_cpu_gpu_artifacts_have_identity_and_speedup() -> None:
    cpu = _load_json("quasilinear_uq_ensemble_scaling_cpu_large.json")
    gpu = _load_json("quasilinear_uq_ensemble_scaling_gpu_large.json")
    combined = _load_json("quasilinear_uq_ensemble_scaling_large.json")

    assert cpu["identity_passed"] is True
    assert cpu["grid"] == {"Nx": 1, "Ny": 96, "Nz": 64, "Nl": 3, "Nm": 6}
    assert cpu["time"]["steps"] == 2000
    assert cpu["time"]["fit_start_fraction"] == 0.5
    assert cpu["time"]["fit_end_fraction"] == 0.95
    assert cpu["gradients"] == [2.2, 2.4, 2.6, 2.8, 3.0, 3.2]
    assert cpu["ky"] == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert [row["requested_devices"] for row in cpu["rows"]] == [1, 2, 4, 8]
    assert all(row["identity_gate_pass"] for row in cpu["rows"])
    assert min(row["ensemble_mean_heat_flux_proxy"] for row in cpu["rows"]) > 1.0
    assert cpu["rows"][-1]["actual_workers"] == 6
    assert cpu["rows"][-1]["strong_speedup_vs_1_device"] > 5.0

    assert gpu["identity_passed"] is True
    assert gpu["grid"] == {"Nx": 1, "Ny": 96, "Nz": 64, "Nl": 3, "Nm": 6}
    assert [row["requested_devices"] for row in gpu["rows"]] == [1, 2]
    assert all(row["identity_gate_pass"] for row in gpu["rows"])
    assert min(row["ensemble_mean_heat_flux_proxy"] for row in gpu["rows"]) > 1.0
    assert gpu["rows"][-1]["strong_speedup_vs_1_device"] > 1.5

    assert combined["identity_passed"] is True
    assert combined["kind"] == "quasilinear_uq_ensemble_scaling_combined"
    assert {row["backend"] for row in combined["rows"]} == {"cpu", "gpu"}
    assert (
        "not a promoted absolute nonlinear heat-flux predictor"
        in combined["claim_scope"]
    )
