from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from gkx.workflows.runtime.policies import (
    RuntimeIndependentParallelPlan,
    _active_kx_indices,
    _active_ky_indices,
    _infer_runtime_nonlinear_steps,
    _midplane_index,
    _nearest_index_from_candidates,
    _normalize_linear_solver_name,
    _parallel_requests_combined_ky_scan,
    _runtime_external_phi,
    _runtime_independent_parallel_plan,
    _select_nonlinear_mode_indices,
    _validate_dealias_mask_shape,
    _zero_kx_index,
)


def test_nearest_index_from_candidates_locks_retained_mode_tie_order() -> None:
    values = np.asarray([0.0, 0.5, 1.0, 1.5])

    assert _nearest_index_from_candidates(values, 0.75, np.asarray([1, 2])) == 1
    assert _nearest_index_from_candidates(values, 1.4, np.asarray([0, 2])) == 2

    with pytest.raises(ValueError, match="values must be non-empty"):
        _nearest_index_from_candidates(np.asarray([]), 1.0, np.asarray([0]))
    with pytest.raises(ValueError, match="candidate indices"):
        _nearest_index_from_candidates(values, 1.0, np.asarray([], dtype=int))


def test_active_dealias_indices_fall_back_to_full_axis_for_empty_masks() -> None:
    empty_mask = np.zeros((3, 4), dtype=bool)

    np.testing.assert_array_equal(_active_ky_indices(empty_mask, 3), [0, 1, 2])
    np.testing.assert_array_equal(_active_kx_indices(empty_mask, 1, 4), [0, 1, 2, 3])

    mixed_mask = np.asarray(
        [
            [False, False, False, True],
            [False, False, False, False],
            [True, False, False, False],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(_active_ky_indices(mixed_mask, 3), [0, 2])
    np.testing.assert_array_equal(_active_kx_indices(mixed_mask, 2, 4), [0])


def test_select_nonlinear_mode_indices_uses_nearest_retained_dealiased_mode() -> None:
    grid = SimpleNamespace(
        ky=np.asarray([0.0, 0.4, 0.8]),
        kx=np.asarray([-1.0, 0.0, 1.0]),
        dealias_mask=np.asarray(
            [
                [False, False, True],
                [False, False, False],
                [True, False, False],
            ],
            dtype=bool,
        ),
    )

    ky_idx, kx_idx = _select_nonlinear_mode_indices(
        grid,
        ky_target=0.39,
        kx_target=0.2,
        use_dealias_mask=True,
    )

    assert (ky_idx, kx_idx) == (0, 2)


def test_select_nonlinear_mode_indices_validates_dealias_mask_shape() -> None:
    bad_grid = SimpleNamespace(
        ky=np.asarray([0.0, 0.4]),
        kx=np.asarray([-1.0, 0.0, 1.0]),
        dealias_mask=np.ones((2, 2), dtype=bool),
    )

    with pytest.raises(ValueError, match="dealias_mask shape"):
        _select_nonlinear_mode_indices(
            bad_grid,
            ky_target=0.4,
            kx_target=0.0,
            use_dealias_mask=True,
        )


def test_validate_dealias_mask_shape_returns_boolean_view() -> None:
    mask = _validate_dealias_mask_shape(
        np.asarray([[1, 0], [0, 1]], dtype=int),
        ky_size=2,
        kx_size=2,
    )

    assert mask.dtype == np.bool_
    np.testing.assert_array_equal(mask, [[True, False], [False, True]])


def test_runtime_independent_parallel_plan_serializes_argument_policy() -> None:
    cfg = SimpleNamespace(parallel=None)

    plan = _runtime_independent_parallel_plan(
        cfg,
        problem_size=3,
        workers=8,
        executor="threads",
    )
    empty = _runtime_independent_parallel_plan(
        cfg,
        problem_size=0,
        workers=2,
        executor="process",
    )

    assert isinstance(plan, RuntimeIndependentParallelPlan)
    assert plan.requested_workers == 8
    assert plan.effective_workers == 3
    assert plan.executor == "thread"
    assert plan.source == "arguments"
    assert plan.enabled is True
    assert plan.to_dict()["enabled"] is True
    assert empty.effective_workers == 0
    assert empty.enabled is False


def test_runtime_independent_parallel_plan_honors_batch_config_and_guards() -> None:
    cfg = SimpleNamespace(
        parallel=SimpleNamespace(
            strategy="batch",
            axis="ky",
            num_devices=4,
            batch_size=None,
            backend="processes",
        )
    )

    plan = _runtime_independent_parallel_plan(
        cfg,
        problem_size=2,
        workers=1,
        executor="thread",
    )

    assert plan.requested_workers == 4
    assert plan.effective_workers == 2
    assert plan.executor == "process"
    assert plan.strategy == "batch"
    assert plan.axis == "ky"
    assert plan.source == "runtime_config"

    with pytest.raises(ValueError, match="problem_size"):
        _runtime_independent_parallel_plan(
            SimpleNamespace(parallel=None),
            problem_size=-1,
            workers=1,
            executor="thread",
        )
    with pytest.raises(ValueError, match="workers"):
        _runtime_independent_parallel_plan(
            SimpleNamespace(parallel=None),
            problem_size=1,
            workers=0,
            executor="thread",
        )
    with pytest.raises(ValueError, match="parallel_executor"):
        _runtime_independent_parallel_plan(
            SimpleNamespace(parallel=None),
            problem_size=1,
            workers=1,
            executor="gpu",
        )
    with pytest.raises(ValueError, match="axis='ky'"):
        _runtime_independent_parallel_plan(
            SimpleNamespace(
                parallel=SimpleNamespace(strategy="batch", axis="kx", backend="auto")
            ),
            problem_size=2,
            workers=1,
            executor="thread",
        )
    with pytest.raises(ValueError, match="independent scans"):
        _runtime_independent_parallel_plan(
            SimpleNamespace(
                parallel=SimpleNamespace(strategy="batch", axis="ky", backend="mpi")
            ),
            problem_size=2,
            workers=1,
            executor="thread",
        )


def test_runtime_solver_and_combined_ky_policy_helpers_normalize_inputs() -> None:
    assert _normalize_linear_solver_name(" explicit_time ") == "explicit_time"
    assert _normalize_linear_solver_name(" Krylov ") == "krylov"

    assert _parallel_requests_combined_ky_scan(SimpleNamespace(parallel=None)) is False
    assert (
        _parallel_requests_combined_ky_scan(
            SimpleNamespace(parallel=SimpleNamespace(strategy="Combined_KY", axis="KY"))
        )
        is True
    )
    assert (
        _parallel_requests_combined_ky_scan(
            SimpleNamespace(parallel=SimpleNamespace(strategy="combined_ky", axis="kx"))
        )
        is False
    )


def test_runtime_mode_and_axis_helpers_cover_unmasked_selection() -> None:
    single_z = SimpleNamespace(z=np.asarray([0.0]), kx=np.asarray([1.0]))
    centered = SimpleNamespace(
        z=np.linspace(-1.0, 1.0, 5),
        kx=np.asarray([2.0, -0.05, 0.2]),
    )
    grid = SimpleNamespace(
        ky=np.asarray([0.0, 0.5, 1.0]),
        kx=np.asarray([-1.0, 0.2, 2.0]),
        dealias_mask=np.zeros((3, 3), dtype=bool),
    )

    assert _midplane_index(single_z) == 0
    assert _midplane_index(centered) == 3
    assert _zero_kx_index(centered) == 1
    assert _select_nonlinear_mode_indices(
        grid,
        ky_target=0.6,
        kx_target=None,
        use_dealias_mask=False,
    ) == (1, 1)


def test_runtime_step_and_external_phi_policies_are_fail_closed() -> None:
    fixed = SimpleNamespace(
        time=SimpleNamespace(fixed_dt=True, t_max=1.0, dt=0.25, dt_max=None)
    )
    adaptive_capped = SimpleNamespace(
        time=SimpleNamespace(fixed_dt=False, t_max=1.0, dt=0.2, dt_max=0.3)
    )
    adaptive_uncapped = SimpleNamespace(
        time=SimpleNamespace(fixed_dt=False, t_max=1.0, dt=0.2, dt_max=None)
    )

    assert _infer_runtime_nonlinear_steps(fixed, dt=0.125, steps=None) == 4
    assert _infer_runtime_nonlinear_steps(fixed, dt=0.125, steps=7) == 7
    assert _infer_runtime_nonlinear_steps(adaptive_capped, dt=0.2, steps=None) == 4
    assert _infer_runtime_nonlinear_steps(adaptive_uncapped, dt=0.2, steps=None) == 5
    with pytest.raises(ValueError, match="steps"):
        _infer_runtime_nonlinear_steps(fixed, dt=0.125, steps=0)

    assert (
        _runtime_external_phi(
            SimpleNamespace(expert=SimpleNamespace(source=" default ", phi_ext=3.0))
        )
        is None
    )
    assert _runtime_external_phi(
        SimpleNamespace(expert=SimpleNamespace(source="phiext_full", phi_ext=2.5))
    ) == pytest.approx(2.5)
    with pytest.raises(ValueError, match="unsupported expert.source"):
        _runtime_external_phi(
            SimpleNamespace(expert=SimpleNamespace(source="external_phi", phi_ext=1.0))
        )
