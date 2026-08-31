"""Unit contracts for VMEC flux-tube report helper functions.

These exercise the pure metric/packing/error helpers behind the two public
``vmex_flux_tube_*_report`` entry points (which are ``# pragma: no cover``)
directly, with synthetic dict/array/``SimpleNamespace`` inputs so that no real
equilibrium solve is required.  Numeric expectations are hand-computed from the
underlying ``_array_parity_metrics``/``_scalar_parity_metrics`` contracts.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import gkx.geometry.imported_vmec as imported_vmec
import gkx.geometry.vmec_flux_tube_reports as reports


# ---------------------------------------------------------------------------
# Synthetic-input factories
# ---------------------------------------------------------------------------
def _base_profile() -> np.ndarray:
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])


def _equal_arc_core_dict(**overrides: object) -> dict[str, object]:
    """Boozer equal-arc core payload keyed as produced by the core-profiles fn."""

    data: dict[str, object] = {
        "theta": np.linspace(-np.pi, np.pi, 5),
        "bmag": _base_profile(),
        "bgrad": _base_profile(),
        "jacobian": _base_profile(),
        "gds2": _base_profile(),
        "gds21": _base_profile(),
        "gds22": _base_profile(),
        "grho": _base_profile(),
        "cvdrift": _base_profile(),
        "gbdrift": _base_profile(),
        "cvdrift0": _base_profile(),
        "gbdrift0": _base_profile(),
        "gradpar": np.array([0.7, 0.7, 0.7, 0.7, 0.7]),
        "q": 1.5,
        "s_hat": 0.4,
    }
    data.update(overrides)
    return data


def _equal_arc_imported(**overrides: object) -> SimpleNamespace:
    """Imported-VMEC reference exposing the ``*_profile``/``*_value`` attributes."""

    attrs: dict[str, object] = {
        "theta": np.linspace(-np.pi, np.pi, 5),
        "bmag_profile": _base_profile(),
        "bgrad_profile": _base_profile(),
        "jacobian_profile": _base_profile(),
        "gds2_profile": _base_profile(),
        "gds21_profile": _base_profile(),
        "gds22_profile": _base_profile(),
        "grho_profile": _base_profile(),
        "cv_profile": _base_profile(),
        "gb_profile": _base_profile(),
        "cv0_profile": _base_profile(),
        "gb0_profile": _base_profile(),
        "gradpar_value": 0.7,
        "q": 1.5,
        "s_hat": 0.4,
    }
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


def _flux_tube_namespace(**overrides: object) -> SimpleNamespace:
    """Direct/imported flux-tube geometry exposing ``_FLUX_TUBE_ARRAY_FIELDS``."""

    attrs: dict[str, object] = {
        "theta": np.linspace(-np.pi, np.pi, 5),
        "bmag_profile": _base_profile(),
        "bgrad_profile": _base_profile(),
        "gds2_profile": _base_profile(),
        "gds21_profile": _base_profile(),
        "gds22_profile": _base_profile(),
        "cv_profile": _base_profile(),
        "gb_profile": _base_profile(),
        "cv0_profile": _base_profile(),
        "gb0_profile": _base_profile(),
        "jacobian_profile": _base_profile(),
        "grho_profile": _base_profile(),
        "gradpar_value": 0.8,
        "q": 2.0,
        "s_hat": 2.0,
    }
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


def _valid_flux_tube_mapping() -> dict[str, object]:
    """A finite, contract-valid flux-tube mapping (constant gradpar)."""

    theta = np.linspace(-np.pi, np.pi, 8, endpoint=False)
    ones = np.ones_like(theta)
    zeros = np.zeros_like(theta)
    return {
        "theta": theta,
        "gradpar": 0.7 * ones,
        "bmag": 1.0 + 0.05 * np.cos(theta),
        "bgrad": 0.05 * np.sin(theta),
        "gds2": ones,
        "gds21": zeros,
        "gds22": ones,
        "cvdrift": 0.2 * np.cos(theta),
        "gbdrift": 0.2 * np.cos(theta),
        "cvdrift0": zeros,
        "gbdrift0": zeros,
        "jacobian": ones,
        "grho": ones,
        "q": 1.7,
        "s_hat": 0.4,
        "R0": 1.5,
        "nfp": 5,
    }


# ---------------------------------------------------------------------------
# Pure metric helpers
# ---------------------------------------------------------------------------
def test_normalized_max_abs_reads_numeric_or_falls_back_to_inf() -> None:
    assert reports._normalized_max_abs({"normalized_max_abs": 0.25}) == pytest.approx(0.25)
    assert reports._normalized_max_abs(
        {"normalized_max_abs": np.float64(0.5)}
    ) == pytest.approx(0.5)
    assert reports._normalized_max_abs({"normalized_max_abs": 3}) == pytest.approx(3.0)
    assert np.isinf(reports._normalized_max_abs({}))
    assert np.isinf(reports._normalized_max_abs({"normalized_max_abs": "bad"}))
    assert np.isinf(reports._normalized_max_abs({"normalized_max_abs": None}))


def test_array_metrics_from_pairs_maps_each_pair_through_parity_metrics() -> None:
    pairs = {
        "matched": (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0])),
        "mismatched": (np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0])),
    }

    metrics = reports._array_metrics_from_pairs(pairs)

    assert set(metrics) == {"matched", "mismatched"}
    assert metrics["matched"]["shape_match"] is True
    assert metrics["matched"]["max_abs"] == pytest.approx(1.0)
    # reference max is 4.0 -> normalized max-abs = 1.0 / 4.0.
    assert metrics["matched"]["normalized_max_abs"] == pytest.approx(0.25)
    assert metrics["mismatched"]["shape_match"] is False
    assert "normalized_max_abs" not in metrics["mismatched"]


def test_array_metrics_from_key_attrs_pairs_dict_keys_with_reference_attrs() -> None:
    candidate = {"x": np.array([1.0, 2.0, 3.0]), "y": np.zeros(3)}
    reference = SimpleNamespace(a_ref=np.array([1.0, 2.0, 4.0]), b_ref=np.zeros(3))
    specs = (("alpha", "x", "a_ref"), ("beta", "y", "b_ref"))

    metrics = reports._array_metrics_from_key_attrs(candidate, reference, specs)

    assert set(metrics) == {"alpha", "beta"}
    assert metrics["alpha"]["normalized_max_abs"] == pytest.approx(0.25)
    assert metrics["beta"]["max_abs"] == pytest.approx(0.0)


def test_array_pairs_from_attrs_reads_both_sides_by_attribute() -> None:
    candidate = SimpleNamespace(ca=np.array([1.0, 2.0]), cb=np.array([3.0, 4.0]))
    reference = SimpleNamespace(ra=np.array([5.0, 6.0]), rb=np.array([7.0, 8.0]))
    specs = (("first", "ca", "ra"), ("second", "cb", "rb"))

    pairs = reports._array_pairs_from_attrs(candidate, reference, specs)

    assert set(pairs) == {"first", "second"}
    np.testing.assert_array_equal(pairs["first"][0], candidate.ca)
    np.testing.assert_array_equal(pairs["first"][1], reference.ra)
    np.testing.assert_array_equal(pairs["second"][0], candidate.cb)
    np.testing.assert_array_equal(pairs["second"][1], reference.rb)


def test_worst_array_error_ignores_shape_mismatches_and_defaults_to_inf() -> None:
    metrics = {
        "a": {"normalized_max_abs": 0.02, "shape_match": True},
        "b": {"normalized_max_abs": 0.05, "shape_match": True},
        "c": {"normalized_max_abs": 0.99, "shape_match": False},
    }

    # names=None spans every entry but drops the shape-mismatched "c".
    assert reports._worst_array_error(metrics) == pytest.approx(0.05)
    assert reports._worst_array_error(metrics, ("a",)) == pytest.approx(0.02)
    assert reports._worst_array_error(metrics, ("a", "b")) == pytest.approx(0.05)
    assert np.isinf(
        reports._worst_array_error({"x": {"normalized_max_abs": 0.1, "shape_match": False}})
    )


def test_empty_equal_arc_parity_is_fail_closed_with_propagated_error() -> None:
    report = reports._empty_equal_arc_parity("some error")

    assert report["equal_arc_core_error"] == "some error"
    assert report["equal_arc_core_passed"] is False
    assert report["equal_arc_derivative_passed"] is False
    assert report["equal_arc_metric_passed"] is False
    assert report["equal_arc_drift_passed"] is False
    for key in (
        "equal_arc_core_worst_normalized_max_abs",
        "equal_arc_core_worst_scalar_rel",
        "equal_arc_derivative_worst_normalized_max_abs",
        "equal_arc_metric_worst_normalized_max_abs",
        "equal_arc_drift_worst_normalized_max_abs",
    ):
        assert np.isinf(report[key])
    for key in (
        "equal_arc_core_array_metrics",
        "equal_arc_metric_array_metrics",
        "equal_arc_drift_array_metrics",
        "equal_arc_core_scalar_metrics",
    ):
        assert report[key] == {}
    assert reports._empty_equal_arc_parity(None)["equal_arc_core_error"] is None


# ---------------------------------------------------------------------------
# Equal-arc grouping / scalar / packing helpers
# ---------------------------------------------------------------------------
def test_equal_arc_core_profiles_forwards_context_and_disables_jit(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_core(state, runtime, inp, wout, **kwargs):  # noqa: ANN001, ANN202
        captured["state"] = state
        captured["runtime"] = runtime
        captured["inp"] = inp
        captured["wout"] = wout
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(
        reports, "vmex_boozer_equal_arc_core_profiles_from_state", fake_core
    )
    ctx = SimpleNamespace(state="S", runtime="R", inp="I", wout="W")

    out = reports._equal_arc_core_profiles(
        ctx=ctx, surface_index=4, torflux=0.5, alpha=0.25, ntheta=16, mboz=21, nboz=23
    )

    assert out == {"ok": True}
    assert (captured["state"], captured["runtime"]) == ("S", "R")
    assert (captured["inp"], captured["wout"]) == ("I", "W")
    assert captured["surface_index"] == 4
    assert captured["torflux"] == 0.5
    assert captured["alpha"] == pytest.approx(0.25)
    assert isinstance(captured["alpha"], float)
    assert captured["ntheta"] == 16
    assert captured["mboz"] == 21
    assert captured["nboz"] == 23
    assert captured["jit"] is False


def test_equal_arc_array_metric_groups_split_into_core_metric_drift() -> None:
    core = _equal_arc_core_dict()
    imported = _equal_arc_imported(
        bgrad_profile=np.array([1.0, 2.0, 3.0, 4.0, 5.5]),
        gds21_profile=np.array([1.0, 2.4, 3.0, 4.0, 5.0]),
        gb_profile=np.array([1.0, 2.0, 3.0, 3.5, 5.0]),
    )

    core_m, metric_m, drift_m = reports._equal_arc_array_metric_groups(core, imported)

    assert set(core_m) == {"theta", "bmag", "bgrad", "jacobian"}
    assert set(metric_m) == {"gds2", "gds21", "gds22", "grho"}
    assert set(drift_m) == {"cvdrift", "gbdrift", "cvdrift0", "gbdrift0"}
    # theta is identical, proving the zero baseline for the plumbing.
    assert core_m["theta"]["normalized_max_abs"] == pytest.approx(0.0)
    # bgrad differs by 0.5 with reference max 5.5.
    assert core_m["bgrad"]["normalized_max_abs"] == pytest.approx(0.5 / 5.5)
    # gds21 differs by 0.4 with reference max 5.0.
    assert metric_m["gds21"]["normalized_max_abs"] == pytest.approx(0.08)
    # gbdrift differs by 0.5 with reference max 5.0.
    assert drift_m["gbdrift"]["normalized_max_abs"] == pytest.approx(0.1)


def test_equal_arc_scalar_metrics_uses_first_gradpar_sample() -> None:
    core = _equal_arc_core_dict(
        gradpar=np.array([0.75, 999.0, 999.0, 999.0, 999.0]), q=1.6, s_hat=0.5
    )
    imported = _equal_arc_imported(gradpar_value=0.75, q=1.5, s_hat=0.4)

    scalar = reports._equal_arc_scalar_metrics(core, imported)

    assert set(scalar) == {"gradpar", "q", "s_hat"}
    # The [0] index must win over the 999.0 sentinels further down the profile.
    assert scalar["gradpar"]["candidate"] == pytest.approx(0.75)
    assert scalar["gradpar"]["rel"] == pytest.approx(0.0)
    assert scalar["q"]["candidate"] == pytest.approx(1.6)
    assert scalar["q"]["reference"] == pytest.approx(1.5)
    assert scalar["q"]["rel"] == pytest.approx(0.1 / 1.5)
    assert scalar["s_hat"]["rel"] == pytest.approx(0.1 / 0.4)


def test_pack_equal_arc_parity_computes_worst_values_and_flips_flags() -> None:
    core_metrics = {
        "theta": {"normalized_max_abs": 0.001, "shape_match": True},
        "bmag": {"normalized_max_abs": 0.004, "shape_match": True},
        "bgrad": {"normalized_max_abs": 0.02, "shape_match": True},
        "jacobian": {"normalized_max_abs": 0.003, "shape_match": True},
    }
    metric_metrics = {
        "gds2": {"normalized_max_abs": 0.05, "shape_match": True},
        "gds21": {"normalized_max_abs": 0.07, "shape_match": True},
        "gds22": {"normalized_max_abs": 0.06, "shape_match": True},
        "grho": {"normalized_max_abs": 0.01, "shape_match": True},
    }
    drift_metrics = {
        "cvdrift": {"normalized_max_abs": 0.03, "shape_match": True},
        "gbdrift": {"normalized_max_abs": 0.08, "shape_match": True},
        "cvdrift0": {"normalized_max_abs": 0.02, "shape_match": True},
        "gbdrift0": {"normalized_max_abs": 0.09, "shape_match": True},
    }
    scalar_metrics = {
        "gradpar": {"rel": 0.002},
        "q": {"rel": 0.001},
        "s_hat": {"rel": 0.003},
    }

    packed = reports._pack_equal_arc_parity(
        core_metrics=core_metrics,
        metric_metrics=metric_metrics,
        drift_metrics=drift_metrics,
        scalar_metrics=scalar_metrics,
        core_tolerance=0.01,
        derivative_tolerance=0.03,
        metric_tolerance=0.08,
        drift_tolerance=0.08,
    )

    # core worst spans only theta/bmag/jacobian (bgrad is the derivative gate).
    assert packed["equal_arc_core_worst_normalized_max_abs"] == pytest.approx(0.004)
    assert packed["equal_arc_core_worst_scalar_rel"] == pytest.approx(0.003)
    assert packed["equal_arc_derivative_worst_normalized_max_abs"] == pytest.approx(0.02)
    assert packed["equal_arc_metric_worst_normalized_max_abs"] == pytest.approx(0.07)
    assert packed["equal_arc_drift_worst_normalized_max_abs"] == pytest.approx(0.09)
    assert packed["equal_arc_core_passed"] is True
    assert packed["equal_arc_derivative_passed"] is True
    assert packed["equal_arc_metric_passed"] is True
    assert packed["equal_arc_drift_passed"] is False  # 0.09 > 0.08
    assert packed["equal_arc_core_error"] is None
    assert packed["equal_arc_core_array_metrics"] is core_metrics
    # Packed schema must exactly match the fail-closed schema.
    assert set(packed) == set(reports._empty_equal_arc_parity(None))

    # A tight array tolerance flips the core gate off through the array worst.
    tight_core = reports._pack_equal_arc_parity(
        core_metrics=core_metrics,
        metric_metrics=metric_metrics,
        drift_metrics=drift_metrics,
        scalar_metrics=scalar_metrics,
        core_tolerance=0.0025,
        derivative_tolerance=0.03,
        metric_tolerance=0.08,
        drift_tolerance=0.08,
    )
    assert tight_core["equal_arc_core_passed"] is False

    # A large scalar residual flips the core gate off through the scalar worst.
    scalar_heavy = reports._pack_equal_arc_parity(
        core_metrics=core_metrics,
        metric_metrics=metric_metrics,
        drift_metrics=drift_metrics,
        scalar_metrics={"gradpar": {"rel": 0.5}, "q": {"rel": 0.0}, "s_hat": {"rel": 0.0}},
        core_tolerance=0.01,
        derivative_tolerance=0.03,
        metric_tolerance=0.08,
        drift_tolerance=0.08,
    )
    assert scalar_heavy["equal_arc_core_worst_scalar_rel"] == pytest.approx(0.5)
    assert scalar_heavy["equal_arc_core_passed"] is False


def test_equal_arc_parity_report_returns_empty_when_booz_api_unavailable() -> None:
    report = reports._equal_arc_parity_report(
        info={"booz_xform_jax_api_available": False},
        ctx=object(),
        imported=object(),
        surface_index=1,
        torflux=0.5,
        alpha=0.0,
        ntheta=8,
        mboz=21,
        nboz=21,
        core_tolerance=0.01,
        derivative_tolerance=0.03,
        metric_tolerance=0.08,
        drift_tolerance=0.08,
    )

    assert report == reports._empty_equal_arc_parity(
        "booz_xform_jax functional API is not available"
    )


def test_equal_arc_parity_report_packs_metrics_when_core_profiles_available(
    monkeypatch,
) -> None:
    core = _equal_arc_core_dict()
    # Only the drift channel differs (reference max 10.0, diff 5.0 -> 0.5).
    imported = _equal_arc_imported(gb_profile=np.array([1.0, 2.0, 3.0, 4.0, 10.0]))
    monkeypatch.setattr(
        reports, "_equal_arc_core_profiles", lambda **kwargs: core
    )

    report = reports._equal_arc_parity_report(
        info={"booz_xform_jax_api_available": True},
        ctx=object(),
        imported=imported,
        surface_index=1,
        torflux=0.5,
        alpha=0.0,
        ntheta=8,
        mboz=21,
        nboz=21,
        core_tolerance=0.01,
        derivative_tolerance=0.03,
        metric_tolerance=0.08,
        drift_tolerance=0.08,
    )

    assert report["equal_arc_core_error"] is None
    assert set(report["equal_arc_core_array_metrics"]) == {
        "theta",
        "bmag",
        "bgrad",
        "jacobian",
    }
    assert report["equal_arc_core_passed"] is True
    assert report["equal_arc_derivative_passed"] is True
    assert report["equal_arc_metric_passed"] is True
    # gbdrift diverges by 0.5 (>> 0.08 tolerance), so only the drift gate fails.
    assert report["equal_arc_drift_worst_normalized_max_abs"] == pytest.approx(0.5)
    assert report["equal_arc_drift_passed"] is False


# ---------------------------------------------------------------------------
# Sensitivity / parity report scaffolding helpers
# ---------------------------------------------------------------------------
def test_vmec_sensitivity_unavailable_report_schema_and_coercions() -> None:
    info = {"vmex_available": False}

    report = reports._vmec_sensitivity_unavailable_report(
        info=info, case_name="demo", fd_step=2.0e-6, reason="vmex is not available"
    )

    assert report == {
        "available": False,
        "backend_info": info,
        "sensitivity": None,
        "fd_step": 2.0e-6,
        "case_name": "demo",
        "reason": "vmex is not available",
    }
    coerced = reports._vmec_sensitivity_unavailable_report(
        info=info, case_name=123, fd_step=2, reason="x"
    )
    assert coerced["case_name"] == "123"
    assert coerced["fd_step"] == pytest.approx(2.0)
    assert isinstance(coerced["fd_step"], float)


def test_validate_vmec_parity_inputs_enforces_resolution_floors() -> None:
    assert reports._validate_vmec_parity_inputs(16, 21, 23) == (16, 21, 23)
    # Non-int inputs are truncated through int().
    assert reports._validate_vmec_parity_inputs(16.9, 21.0, 21.0) == (16, 21, 21)

    with pytest.raises(ValueError, match="ntheta must be >= 4"):
        reports._validate_vmec_parity_inputs(3, 21, 21)
    with pytest.raises(ValueError, match="mboz and nboz"):
        reports._validate_vmec_parity_inputs(8, 20, 21)
    with pytest.raises(ValueError, match="mboz and nboz"):
        reports._validate_vmec_parity_inputs(8, 21, 20)


def test_vmec_array_parity_unavailable_report_schema() -> None:
    info = {"vmex_available": False}

    report = reports._vmec_array_parity_unavailable_report(
        info=info, case_name="demo", reason="internal VMEC/EIK backend is not available"
    )

    assert report == {
        "available": False,
        "backend_info": info,
        "case_name": "demo",
        "reason": "internal VMEC/EIK backend is not available",
    }


def test_surface_index_and_torflux_defaults_to_mid_surface() -> None:
    ctx = SimpleNamespace(base_Rcos=np.zeros((8, 3)))

    sidx, torflux = reports._surface_index_and_torflux(ctx, None)
    assert sidx == 4  # max(1, min(8 // 2, 8 - 2))
    assert torflux == pytest.approx(4.0 / 7.0)

    sidx_explicit, torflux_explicit = reports._surface_index_and_torflux(ctx, 3)
    assert sidx_explicit == 3
    assert torflux_explicit == pytest.approx(3.0 / 7.0)

    tiny = SimpleNamespace(base_Rcos=np.zeros((2, 3)))
    sidx_tiny, torflux_tiny = reports._surface_index_and_torflux(tiny, None)
    assert sidx_tiny == 1
    assert torflux_tiny == pytest.approx(1.0)


def test_production_parity_metrics_reports_worst_core_and_scalar() -> None:
    imported = _flux_tube_namespace(
        bmag_profile=np.array([1.0, 2.0, 3.0, 4.0, 4.0]), s_hat=2.0
    )
    direct = _flux_tube_namespace(
        bmag_profile=np.array([1.0, 2.0, 3.0, 3.9, 4.0]), s_hat=2.1
    )

    array_metrics, scalar_metrics, worst_core, worst_scalar, passed = (
        reports._production_parity_metrics(
            direct=direct, imported=imported, core_tolerance=0.05, scalar_tolerance=0.06
        )
    )

    assert set(array_metrics) == {
        "theta",
        "bmag",
        "bgrad",
        "gds2",
        "gds21",
        "gds22",
        "cvdrift",
        "gbdrift",
        "cvdrift0",
        "gbdrift0",
        "jacobian",
        "grho",
    }
    assert set(scalar_metrics) == {"gradpar", "q", "s_hat"}
    # bmag differs by 0.1 with reference max 4.0 -> 0.025; all other core arrays match.
    assert array_metrics["bmag"]["normalized_max_abs"] == pytest.approx(0.025)
    assert worst_core == pytest.approx(0.025)
    # s_hat differs by 0.1 with scale 2.0 -> 0.05.
    assert worst_scalar == pytest.approx(0.05)
    assert passed is True  # 0.025 <= 0.05 and 0.05 <= 0.06

    _, _, _, _, passed_tight = reports._production_parity_metrics(
        direct=direct, imported=imported, core_tolerance=0.05, scalar_tolerance=0.04
    )
    assert passed_tight is False  # scalar worst 0.05 now exceeds tolerance


def test_pack_vmec_array_parity_report_schema_and_status() -> None:
    ctx = SimpleNamespace(input_path=Path("input.demo"), wout_path=Path("wout_demo.nc"))
    info = {"vmex_available": True}
    equal_arc = reports._empty_equal_arc_parity(None)

    kwargs = dict(
        ctx=ctx,
        info=info,
        case_name="demo",
        surface_index=4,
        torflux=0.5,
        alpha=0.0,
        ntheta=16,
        mboz=21,
        nboz=23,
        boundary="none",
        include_shear_variation=True,
        include_pressure_variation=False,
        array_metrics={"bmag": {"normalized_max_abs": 0.02, "shape_match": True}},
        scalar_metrics={"q": {"rel": 0.001}},
        equal_arc_parity=equal_arc,
        equal_arc_core_tolerance=0.01,
        equal_arc_derivative_tolerance=0.03,
        equal_arc_metric_tolerance=0.08,
        equal_arc_drift_tolerance=0.08,
        worst_core=0.02,
        worst_scalar=0.001,
        core_tolerance=0.05,
        scalar_tolerance=0.005,
        production_parity_passed=False,
    )

    report = reports._pack_vmec_array_parity_report(**kwargs)

    assert report["available"] is True
    assert report["backend_info"] is info
    assert report["source_model"] == (
        "vmex:state->tensor-flux-tube vs imported-vmec-eik"
    )
    assert report["case_name"] == "demo"
    assert report["input_path"] == "input.demo"
    assert report["wout_path"] == "wout_demo.nc"
    assert report["surface_index"] == 4
    assert report["torflux"] == pytest.approx(0.5)
    assert report["ntheta"] == 16
    assert report["mboz"] == 21
    assert report["nboz"] == 23
    assert report["boundary"] == "none"
    assert report["include_shear_variation"] is True
    assert report["include_pressure_variation"] is False
    assert report["array_metrics"]["bmag"]["normalized_max_abs"] == 0.02
    assert report["worst_core_normalized_max_abs"] == pytest.approx(0.02)
    assert report["worst_scalar_rel"] == pytest.approx(0.001)
    assert report["equal_arc_core_tolerance"] == pytest.approx(0.01)
    assert report["production_parity_passed"] is False
    assert report["status"] == "diagnostic_open"
    assert "interpretation" in report
    # The equal-arc parity dict is merged into the top-level schema.
    assert "equal_arc_core_passed" in report
    assert report["equal_arc_core_error"] is None

    passed_report = reports._pack_vmec_array_parity_report(
        **{**kwargs, "production_parity_passed": True}
    )
    assert passed_report["status"] == "passed"


def test_vmec_array_parity_options_validates_and_packs_dataclass() -> None:
    options = reports._vmec_array_parity_options(
        case_name="demo",
        surface_index=None,
        alpha=0,
        ntheta=16,
        mboz=21,
        nboz=21,
        boundary="none",
        include_shear_variation=True,
        include_pressure_variation=False,
        core_tolerance=0.05,
        scalar_tolerance=0.005,
        equal_arc_core_tolerance=0.01,
        equal_arc_derivative_tolerance=0.03,
        equal_arc_metric_tolerance=0.08,
        equal_arc_drift_tolerance=0.08,
    )

    assert isinstance(options, reports._VMECArrayParityOptions)
    assert options.case_name == "demo"
    assert options.surface_index is None
    assert options.alpha == pytest.approx(0.0)
    assert isinstance(options.alpha, float)
    assert options.ntheta == 16
    assert options.mboz == 21
    assert options.nboz == 21
    assert options.include_shear_variation is True
    assert options.include_pressure_variation is False
    assert options.core_tolerance == pytest.approx(0.05)
    assert options.equal_arc_drift_tolerance == pytest.approx(0.08)

    # Resolution-floor validation is enforced inside the options builder too.
    with pytest.raises(ValueError, match="mboz and nboz"):
        reports._vmec_array_parity_options(
            case_name="demo",
            surface_index=None,
            alpha=0.0,
            ntheta=16,
            mboz=20,
            nboz=21,
            boundary="none",
            include_shear_variation=True,
            include_pressure_variation=True,
            core_tolerance=0.05,
            scalar_tolerance=0.005,
            equal_arc_core_tolerance=0.01,
            equal_arc_derivative_tolerance=0.03,
            equal_arc_metric_tolerance=0.08,
            equal_arc_drift_tolerance=0.08,
        )


def test_pack_vmec_array_parity_result_report_bridges_options_and_result() -> None:
    options = reports._vmec_array_parity_options(
        case_name="demo",
        surface_index=None,
        alpha=0.25,
        ntheta=16,
        mboz=21,
        nboz=23,
        boundary="none",
        include_shear_variation=False,
        include_pressure_variation=True,
        core_tolerance=0.05,
        scalar_tolerance=0.005,
        equal_arc_core_tolerance=0.011,
        equal_arc_derivative_tolerance=0.03,
        equal_arc_metric_tolerance=0.08,
        equal_arc_drift_tolerance=0.08,
    )
    ctx = SimpleNamespace(input_path=Path("input.demo"), wout_path=Path("wout_demo.nc"))
    result = reports._VMECArrayParityResult(
        ctx=ctx,
        surface_index=4,
        torflux=0.5,
        array_metrics={"bmag": {"normalized_max_abs": 0.01, "shape_match": True}},
        scalar_metrics={"q": {"rel": 0.002}},
        equal_arc_parity=reports._empty_equal_arc_parity(None),
        worst_core=0.01,
        worst_scalar=0.002,
        production_parity_passed=True,
    )
    info = {"vmex_available": True}

    report = reports._pack_vmec_array_parity_result_report(
        result=result, info=info, options=options
    )

    assert report["available"] is True
    assert report["case_name"] == "demo"
    assert report["surface_index"] == 4  # from result
    assert report["torflux"] == pytest.approx(0.5)  # from result
    assert report["alpha"] == pytest.approx(0.25)  # from options
    assert report["ntheta"] == 16
    assert report["nboz"] == 23
    assert report["include_shear_variation"] is False
    assert report["include_pressure_variation"] is True
    assert report["worst_core_normalized_max_abs"] == pytest.approx(0.01)
    assert report["equal_arc_core_tolerance"] == pytest.approx(0.011)
    assert report["status"] == "passed"


def test_vmec_array_parity_dataclasses_are_frozen() -> None:
    assert dataclasses.is_dataclass(reports._VMECArrayParityOptions)
    assert dataclasses.is_dataclass(reports._VMECArrayParityResult)

    options = reports._VMECArrayParityOptions(
        case_name="demo",
        surface_index=None,
        alpha=0.0,
        ntheta=16,
        mboz=21,
        nboz=21,
        boundary="none",
        include_shear_variation=True,
        include_pressure_variation=True,
        core_tolerance=0.05,
        scalar_tolerance=0.005,
        equal_arc_core_tolerance=0.01,
        equal_arc_derivative_tolerance=0.03,
        equal_arc_metric_tolerance=0.08,
        equal_arc_drift_tolerance=0.08,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        options.alpha = 1.0

    result = reports._VMECArrayParityResult(
        ctx=object(),
        surface_index=1,
        torflux=0.5,
        array_metrics={},
        scalar_metrics={},
        equal_arc_parity={},
        worst_core=0.0,
        worst_scalar=0.0,
        production_parity_passed=False,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.surface_index = 2


def test_vmec_array_parity_backend_unavailable_reason_covers_each_branch(
    monkeypatch,
) -> None:
    assert reports._vmec_array_parity_backend_unavailable_reason(
        {"vmex_available": False}
    ) == "vmex is not available"

    monkeypatch.setattr(
        imported_vmec, "internal_vmec_backend_available", lambda: False
    )
    assert reports._vmec_array_parity_backend_unavailable_reason(
        {"vmex_available": True}
    ) == "internal VMEC/EIK backend is not available"

    monkeypatch.setattr(
        imported_vmec, "internal_vmec_backend_available", lambda: True
    )
    assert (
        reports._vmec_array_parity_backend_unavailable_reason({"vmex_available": True})
        is None
    )


def test_vmec_array_parity_error_report_stringifies_exception() -> None:
    info = {"vmex_available": True}

    report = reports._vmec_array_parity_error_report(
        info=info, case_name="demo", exc=ValueError("boom")
    )

    assert report == {
        "available": False,
        "backend_info": info,
        "case_name": "demo",
        "error": "ValueError: boom",
    }


# ---------------------------------------------------------------------------
# Sensitivity-report helpers (mapping closure, packing, direct geometry, EIK)
# ---------------------------------------------------------------------------
def test_vmec_flux_tube_mapping_fn_wires_perturbation_into_mapping(monkeypatch) -> None:
    captured: dict[str, dict[str, object]] = {}

    def fake_perturb(ctx, x, *, radial_index, mode_index):  # noqa: ANN001, ANN202
        captured["perturb"] = {
            "ctx": ctx,
            "x": x,
            "radial_index": radial_index,
            "mode_index": mode_index,
        }
        return "traced_state"

    def fake_map(state, runtime, *, surface_index, alpha, ntheta):  # noqa: ANN001, ANN202
        captured["map"] = {
            "state": state,
            "runtime": runtime,
            "surface_index": surface_index,
            "alpha": alpha,
            "ntheta": ntheta,
        }
        return {"ok": True}

    monkeypatch.setattr(reports, "_perturb_vmec_state", fake_perturb)
    monkeypatch.setattr(reports, "vmex_flux_tube_mapping_from_state", fake_map)
    ctx = SimpleNamespace(runtime="R", wout="W")

    mapping_fn = reports._vmec_flux_tube_mapping_fn(
        ctx=ctx, radial_index=1, mode_index=2, surface_index=3, alpha=0.25, ntheta=16
    )
    x = np.array([0.1, 0.2])
    out = mapping_fn(x)

    assert out == {"ok": True}
    assert captured["perturb"]["ctx"] is ctx
    np.testing.assert_array_equal(captured["perturb"]["x"], x)
    assert captured["perturb"]["radial_index"] == 1
    assert captured["perturb"]["mode_index"] == 2
    # The perturbed state is threaded into the mapping call alongside ctx handles.
    assert captured["map"]["state"] == "traced_state"
    assert captured["map"]["runtime"] == "R"
    assert captured["map"]["surface_index"] == 3
    assert captured["map"]["alpha"] == pytest.approx(0.25)
    assert isinstance(captured["map"]["alpha"], float)
    assert captured["map"]["ntheta"] == 16
    assert isinstance(captured["map"]["ntheta"], int)


def test_pack_vmec_sensitivity_report_schema() -> None:
    ctx = SimpleNamespace(
        input_path=Path("input.demo"),
        wout_path=Path("wout_demo.nc"),
        base_Rcos=np.zeros((6, 4)),
    )
    mapping = {
        "vmex": {
            "surface_index": 3,
            "iota": 0.42,
            "reference_length": 1.7,
            "reference_b": 2.3,
            "field_line_convention": "vmec",
        }
    }
    info = {"vmex_available": True}

    report = reports._pack_vmec_sensitivity_report(
        ctx=ctx,
        sensitivity={"foo": 1},
        mapping=mapping,
        params=np.array([0.1, -0.2]),
        case_name="demo",
        radial_index=1,
        mode_index=2,
        alpha=0.25,
        ntheta=16,
        fd_step=2.0e-6,
        info=info,
    )

    assert report["available"] is True
    assert report["backend_info"] is info
    assert report["source_model"] == "vmex:state->tensor-flux-tube"
    assert report["param_names"] == ["delta_Rcos", "delta_Zsin"]
    assert report["params"] == [pytest.approx(0.1), pytest.approx(-0.2)]
    assert report["input_path"] == "input.demo"
    assert report["wout_path"] == "wout_demo.nc"
    assert report["radial_index"] == 1
    assert report["mode_index"] == 2
    assert report["surface_index"] == 3
    assert report["iota"] == pytest.approx(0.42)
    assert report["alpha"] == pytest.approx(0.25)
    assert report["ntheta"] == 16
    assert report["state_shape"] == [6, 4]
    assert report["sensitivity"] == {"foo": 1}
    assert report["fd_step"] == pytest.approx(2.0e-6)
    assert report["reference_length"] == pytest.approx(1.7)
    assert report["reference_b"] == pytest.approx(2.3)
    assert report["field_line_convention"] == "vmec"
    assert "scope" in report


def test_direct_vmec_flux_tube_geometry_builds_contract_from_mapping(
    monkeypatch,
) -> None:
    mapping = _valid_flux_tube_mapping()
    captured: dict[str, object] = {}

    def fake_map(state, runtime, *, surface_index, alpha, ntheta):  # noqa: ANN001, ANN202
        captured.update(
            state=state,
            runtime=runtime,
            surface_index=surface_index,
            alpha=alpha,
            ntheta=ntheta,
        )
        return mapping

    monkeypatch.setattr(reports, "vmex_flux_tube_mapping_from_state", fake_map)
    ctx = SimpleNamespace(state="S", runtime="R", wout="W")

    geom = reports._direct_vmec_flux_tube_geometry(
        ctx=ctx, surface_index=3, alpha=0.25, ntheta=8
    )

    # The real flux-tube contract is built from the (mocked) mapping.
    assert geom.source_model == "vmex:state->tensor-flux-tube"
    np.testing.assert_allclose(np.asarray(geom.theta), np.asarray(mapping["theta"]))
    assert geom.nfp == 5
    assert captured["state"] == "S"
    assert captured["surface_index"] == 3
    assert captured["alpha"] == pytest.approx(0.25)
    assert captured["ntheta"] == 8


def test_vmec_eik_request_merges_defaults_with_overrides() -> None:
    ctx = SimpleNamespace(wout_path=Path("wout_demo.nc"))

    request = reports._vmec_eik_request(
        ctx=ctx,
        ntheta=16,
        boundary="none",
        alpha=0.25,
        torflux=0.5,
        include_shear_variation=True,
        include_pressure_variation=False,
    )

    # Overridden fields.
    assert request.vmec_file == "wout_demo.nc"
    assert request.ntheta == 16
    assert request.boundary == "none"
    assert request.alpha == pytest.approx(0.25)
    assert request.torflux == pytest.approx(0.5)
    assert request.include_shear_variation is True
    assert request.include_pressure_variation is False
    # Defaults preserved from _VMEC_EIK_DEFAULT_REQUEST.
    assert request.y0 == pytest.approx(10.0)
    assert request.z == (1.0, -1.0)
    assert request.species_type == ("ion", "electron")
