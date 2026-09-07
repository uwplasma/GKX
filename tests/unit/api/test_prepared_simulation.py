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
from gkx.config import RuntimeConfig

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


# ---- the prepared topology is the one solve will build --------------------
#
# The deck's [run] table names the velocity resolution. Before it was carried
# on the case, the loader dropped it and prepare_simulation substituted a
# hard-coded (4, 8), so preparing the shipped Cyclone deck -- which asks for
# (16, 48) -- described a calculation nobody had requested. Reporting a
# resolution that solve will not build is worse than reporting none, because a
# recorded result then carries a resolution it was not run at.

_SHIPPED_LINEAR_DECK = "examples/linear/axisymmetric/cyclone.toml"


def _repo_root():
    from pathlib import Path

    return Path(__file__).resolve().parents[3]


def test_prepare_reports_the_resolution_the_shipped_deck_asks_for() -> None:
    case = gkx.load(_repo_root() / _SHIPPED_LINEAR_DECK)
    assert (case.run.Nl, case.run.Nm) == (16, 48), "deck under test changed"

    prepared = prepare_simulation(case)

    assert (prepared.n_laguerre, prepared.n_hermite) == (16, 48), (
        "prepare reported a velocity resolution the deck did not ask for; a "
        "summary that disagrees with the deck misdescribes the calculation"
    )


def test_an_explicit_argument_still_overrides_the_deck() -> None:
    case = gkx.load(_repo_root() / _SHIPPED_LINEAR_DECK)

    prepared = prepare_simulation(case, Nl=6, Nm=10)

    assert (prepared.n_laguerre, prepared.n_hermite) == (6, 10)


def test_a_deck_without_a_run_table_falls_back_to_the_runtime_default() -> None:
    """The fallback is the runtime's own, and it differs by kind.

    ``_CASE_LINEAR_SPECS`` defaults to (24, 12) and ``_CASE_NONLINEAR_SPECS``
    to (4, 8). Preparing must not invent a third pair.
    """

    case = _linear_case()
    assert case.run.is_empty(), "this fixture is meant to carry no [run] table"

    prepared = prepare_simulation(case)

    assert (prepared.n_laguerre, prepared.n_hermite) == (24, 12)


def test_a_deck_with_a_run_table_round_trips_through_toml(tmp_path) -> None:
    case = gkx.load(_repo_root() / _SHIPPED_LINEAR_DECK)

    written = case.to_toml(tmp_path / "resolved.toml")
    reloaded = gkx.load(written)

    assert (reloaded.run.Nl, reloaded.run.Nm) == (16, 48)
    assert reloaded.run.solver == case.run.solver


def test_a_deck_without_a_run_table_does_not_grow_one(tmp_path) -> None:
    """Decks that never had [run] keep writing byte-identical resolved decks."""

    case = _linear_case()

    text = (case.to_toml(tmp_path / "resolved.toml")).read_text(encoding="utf-8")

    assert "[run]" not in text
