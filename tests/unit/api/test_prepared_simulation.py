"""Contract tests for the public prepared simulation.

`gkx.prepare` used to return the solver's internal
`PreparedExplicitNonlinearDiagnostics` and to raise for any linear case, so
the reusable-execution concept covered half the product and leaked a private
type while doing it. These tests pin the public replacement: both case kinds
prepare, the object reports the topology and cache facts that decide whether a
second call is cheap, and it refuses rather than guesses when asked for
something it cannot do.
"""

from __future__ import annotations

import pytest

import gkx
from gkx.api.prepared import PreparedSimulation, prepare_simulation
from gkx.workflows.runtime.config import RuntimeConfig

PREPARED_METHODS = (
    "solve",
    "scan",
    "value_and_grad",
    "warmup",
    "estimate_memory",
    "summary",
)


def _linear_case() -> RuntimeConfig:
    return RuntimeConfig()


@pytest.mark.parametrize("name", PREPARED_METHODS)
def test_prepared_advertises_its_contracted_methods(name: str) -> None:
    assert callable(getattr(PreparedSimulation, name))


def test_gkx_prepare_returns_the_public_type_not_a_solver_internal() -> None:
    """The exported name must not leak `PreparedExplicitNonlinearDiagnostics`."""

    prepared = gkx.prepare(_linear_case())
    assert isinstance(prepared, PreparedSimulation)


def test_prepare_accepts_a_linear_case() -> None:
    """This raised `prepare currently requires nonlinear physics` before."""

    prepared = prepare_simulation(_linear_case())
    assert prepared.kind == "linear"


def test_prepared_is_frozen_so_topology_cannot_drift_from_its_metadata() -> None:
    prepared = prepare_simulation(_linear_case())
    with pytest.raises(Exception):
        prepared.kind = "nonlinear"  # type: ignore[misc]


def test_state_shape_follows_the_case_grid() -> None:
    case = _linear_case()
    prepared = prepare_simulation(case, Nl=6, Nm=12)
    n_species, n_l, n_m, _n_ky, n_kx, n_z = prepared.state_shape
    assert (n_l, n_m) == (6, 12)
    assert n_kx == int(case.grid.Nx)
    assert n_z == int(case.grid.Nz)
    assert n_species >= 1


def test_estimate_memory_scales_with_the_grid_and_admits_it_is_a_floor() -> None:
    small = prepare_simulation(_linear_case(), Nl=4, Nm=8)
    large = prepare_simulation(_linear_case(), Nl=8, Nm=16)
    assert (
        large.estimate_memory()["state_bytes"] > small.estimate_memory()["state_bytes"]
    )
    assert small.estimate_memory()["is_floor_not_ceiling"] is True


def test_summary_reports_precision_devices_and_cache_state() -> None:
    summary = prepare_simulation(_linear_case()).summary()
    assert summary["precision"] in {"float32", "float64"}
    assert summary["devices"]
    assert "persistent_cache_enabled" in summary
    assert summary["compiled_at_prepare"] is False  # linear compiles per call


def test_solve_refuses_parameters_rather_than_solving_a_different_problem() -> None:
    """A prepared object must not silently change physics."""

    prepared = prepare_simulation(_linear_case())
    with pytest.raises(NotImplementedError, match="value_and_grad"):
        prepared.solve(parameters={"tprim": 3.0})


def test_scan_names_the_parameter_it_cannot_scan() -> None:
    prepared = prepare_simulation(_linear_case())
    with pytest.raises(NotImplementedError, match="ky"):
        prepared.scan("tprim", [1.0, 2.0])


def test_warmup_returns_self_so_timing_code_can_chain() -> None:
    prepared = prepare_simulation(_linear_case())
    assert prepared.warmup() is prepared


def test_value_and_grad_differentiates_a_scalar_objective() -> None:
    prepared = prepare_simulation(_linear_case())
    value, grad = prepared.value_and_grad(lambda x: 3.0 * x**2, 2.0)
    assert float(value) == pytest.approx(12.0)
    assert float(grad) == pytest.approx(12.0)


def test_print_summary_writes_every_field(capsys) -> None:
    prepare_simulation(_linear_case()).print_summary()
    printed = capsys.readouterr().out
    assert "kind" in printed and "precision" in printed
