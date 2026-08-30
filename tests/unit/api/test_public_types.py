"""Contract tests for the public `Case` and result types.

The plan's public-API section requires `Case` to own `replace`, `validate`,
`to_toml` and `summary`, and each result to own `save`, `plot`,
`print_summary` and `to_dataset`. Before this suite the types were frozen
dataclasses with no behaviour, so every one of those verbs lived in a helper a
caller had to know to import. These tests pin the surface and, more
importantly, pin what the surface promises: that a written case reloads to the
case that wrote it, and that a result never reports an unsaturated window as an
accepted value.
"""

from __future__ import annotations

import numpy as np
import pytest

import gkx
from gkx.workflows.runtime.config import RuntimeConfig
from gkx.workflows.runtime.results import (
    RuntimeLinearResult,
    RuntimeLinearScanResult,
    RuntimeNonlinearResult,
)
from gkx.diagnostics.modes import ModeSelection

RESULT_TYPES = (RuntimeLinearResult, RuntimeLinearScanResult, RuntimeNonlinearResult)
CASE_METHODS = ("replace", "validate", "to_toml", "summary")
RESULT_METHODS = ("save", "plot", "print_summary", "to_dataset", "summary")


def _linear() -> RuntimeLinearResult:
    return RuntimeLinearResult(
        ky=0.5,
        gamma=0.14,
        omega=0.06,
        selection=ModeSelection(ky_index=1, kx_index=0, z_index=0),
        t=np.linspace(0.0, 1.0, 5),
        signal=np.ones(5),
        z=np.linspace(-np.pi, np.pi, 4),
        eigenfunction=np.ones(4, dtype=complex),
        fit_settled=True,
    )


@pytest.mark.parametrize("name", CASE_METHODS)
def test_case_advertises_its_contracted_methods(name: str) -> None:
    assert callable(getattr(RuntimeConfig, name))


@pytest.mark.parametrize("result_type", RESULT_TYPES)
@pytest.mark.parametrize("name", RESULT_METHODS)
def test_results_advertise_their_contracted_methods(result_type, name: str) -> None:
    assert callable(getattr(result_type, name))


def test_case_is_frozen_so_replace_is_the_only_way_to_derive_one() -> None:
    case = RuntimeConfig()
    with pytest.raises(Exception):
        case.grid = None  # type: ignore[misc]


def test_replace_returns_a_validated_copy_and_leaves_the_original_alone() -> None:
    case = RuntimeConfig()
    derived = case.replace(time=case.time.__class__(t_max=12.0, dt=0.01))
    assert derived is not case
    assert float(derived.time.t_max) == 12.0
    assert float(case.time.t_max) != 12.0 or float(case.time.dt) != 0.01


def test_replace_rejects_an_invalid_case_at_the_point_it_is_built() -> None:
    """A scan that builds cases in a loop must fail on the case it built."""

    case = RuntimeConfig()
    with pytest.raises(ValueError, match="invalid case"):
        case.replace(time=case.time.__class__(t_max=-1.0))


def test_validate_rejects_simultaneous_linear_and_nonlinear() -> None:
    case = RuntimeConfig()
    physics = case.physics.__class__(linear=True, nonlinear=True)
    with pytest.raises(ValueError, match="cannot both be true"):
        RuntimeConfig(physics=physics).validate()


def test_case_round_trips_through_to_toml_and_load(tmp_path) -> None:
    """What `to_toml` writes is what `gkx.load` reads back, field for field."""

    case = RuntimeConfig()
    written = case.to_toml(tmp_path / "case.toml")
    assert written.exists()
    assert gkx.load(written).to_dict() == case.to_dict()


def test_to_toml_omits_none_rather_than_inventing_a_null() -> None:
    """TOML has no null; an absent key is how the loader spells "default"."""

    from gkx.workflows.runtime.wout import deck_text

    text = deck_text({"grid": {"Nx": 8, "Ny": None}})
    assert "Nx = 8" in text
    assert "Ny" not in text


def test_case_summary_reports_the_identifying_scalars() -> None:
    summary = RuntimeConfig().summary()
    assert summary["n_species"] >= 1
    assert set(summary["grid"]) == {"Nx", "Ny", "Nz", "y0"}


def test_linear_summary_carries_fit_status_beside_the_number() -> None:
    summary = _linear().summary()
    assert summary["kind"] == "linear"
    assert summary["gamma"] == pytest.approx(0.14)
    assert summary["fit_settled"] is True


def test_nonlinear_summary_never_hides_an_unsaturated_verdict() -> None:
    """An unsaturated window must travel with its status, not be inferred."""

    result = RuntimeNonlinearResult(
        t=np.linspace(0.0, 10.0, 3),
        diagnostics=None,
        saturation={"saturated": False, "mean_flux": 4.5, "sem": 0.3},
    )
    summary = result.summary()
    assert summary["saturated"] is False
    assert summary["heat_flux_mean"] == pytest.approx(4.5)


def test_scan_summary_locates_the_growth_peak() -> None:
    scan = RuntimeLinearScanResult(
        ky=np.array([0.1, 0.5, 0.9]),
        gamma=np.array([0.02, 0.20, 0.11]),
        omega=np.array([0.01, 0.05, 0.09]),
    )
    summary = scan.summary()
    assert summary["gamma_peak"] == pytest.approx(0.20)
    assert summary["ky_at_gamma_peak"] == pytest.approx(0.5)


def test_to_dataset_is_xarray_shaped_without_requiring_xarray() -> None:
    """The mapping is exactly what `xarray.Dataset(**payload)` accepts."""

    payload = _linear().to_dataset()
    assert set(payload) == {"coords", "data_vars", "attrs"}
    dims, values = payload["data_vars"]["signal"]
    assert dims == "t"
    assert values.shape == payload["coords"]["t"].shape


def test_to_dataset_drops_absent_arrays_rather_than_emitting_none() -> None:
    scan = RuntimeLinearScanResult(
        ky=np.array([0.1]), gamma=np.array([0.02]), omega=np.array([0.01])
    )
    payload = scan.to_dataset()
    assert set(payload["data_vars"]) == {"gamma", "omega"}


def test_print_summary_writes_every_summary_field(capsys) -> None:
    _linear().print_summary()
    printed = capsys.readouterr().out
    assert "gamma" in printed and "fit_settled" in printed
