"""Unit contracts for the vmex-backed VMEC-state sensitivity helpers.

These exercise the private helpers behind the three ``# pragma: no cover``
public entry points of ``gkx.geometry.vmec_state_sensitivity`` directly,
using synthetic ``_VMECStateContext`` bundles and a monkeypatched
``gk_fieldline_geometry`` / Boozer-tables seam so that no real vmex equilibrium
solve is required.  Numeric expectations (packed observable order, hand-computed
RMS/mean values, and the AD derivatives of the two perturbed Fourier controls)
are checked against the underlying contracts rather than merely asserting that a
result exists.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import gkx.geometry.vmec_state_sensitivity as vss


# ---------------------------------------------------------------------------
# Synthetic state / context factories
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class _FakeSpectralState:
    """Minimal ``dataclasses.replace``-able stand-in for a vmex SpectralState."""

    R_cos: jnp.ndarray
    Z_sin: jnp.ndarray


def _make_state_context(
    *, ns: int = 4, nmode: int = 2, resolution: tuple[int, int, int] = (5, 6, 7)
) -> vss._VMECStateContext:
    """Build a synthetic context whose base coefficient tables start at zero."""

    base_Rcos = jnp.zeros((ns, nmode))
    base_Zsin = jnp.zeros((ns, nmode))
    res = SimpleNamespace(ns=resolution[0], ntheta=resolution[1], nzeta=resolution[2])
    return vss._VMECStateContext(
        input_path=Path("input.synthetic"),
        wout_path=Path("in-memory:synthetic"),
        inp=SimpleNamespace(tag="inp"),
        runtime=SimpleNamespace(resolution=res),
        wout=SimpleNamespace(tag="wout"),
        state=_FakeSpectralState(R_cos=base_Rcos, Z_sin=base_Zsin),
        base_Rcos=base_Rcos,
        base_Zsin=base_Zsin,
    )


# ---------------------------------------------------------------------------
# Pure grid / observable packers
# ---------------------------------------------------------------------------
def test_vmec_state_metric_grid_shape_reads_runtime_resolution() -> None:
    runtime = SimpleNamespace(resolution=SimpleNamespace(ns=9, ntheta=12, nzeta=4))

    assert vss._vmec_state_metric_grid_shape(runtime) == [9, 12, 4]


def test_metric_tensor_observable_fn_packs_pest_metric_observables() -> None:
    ctx = _make_state_context()
    captured: dict[str, object] = {}

    def gk(state, runtime, **kwargs):  # noqa: ANN001, ANN202
        captured["state"] = state
        captured["runtime"] = runtime
        captured.update(kwargs)
        return {
            "jacobian": jnp.asarray([1.0, 1.0]),
            "gds2": jnp.asarray([2.0, 4.0]),
            "gds22": jnp.asarray([0.5, 1.5]),
            "gds21": jnp.asarray([3.0, 4.0]),
            "grho": jnp.asarray([1.0, 3.0]),
            "gradpar": jnp.asarray([0.7, 0.7]),
            "s_hat": jnp.asarray(0.35),
        }

    observable_fn = vss._metric_tensor_observable_fn(
        ctx=ctx,
        turbulence_mod=SimpleNamespace(gk_fieldline_geometry=gk),
        radial_index=1,
        mode_index=1,
        surface_index=3,
        ntheta=16,
        rms_epsilon=jnp.asarray(0.0),
    )
    observables = np.asarray(observable_fn(jnp.asarray([0.001, 0.002])))

    # order: rms(jacobian), mean(gds2), mean(gds22), rms(gds21), mean(grho),
    #        mean(gradpar), s_hat
    np.testing.assert_allclose(
        observables,
        [1.0, 3.0, 1.0, np.sqrt(12.5), 2.0, 0.7, 0.35],
        rtol=1e-9,
        atol=1e-9,
    )
    # the perturbation reached the seam at [radial_index, mode_index]
    assert float(captured["state"].R_cos[1, 1]) == pytest.approx(0.001)
    assert float(captured["state"].Z_sin[1, 1]) == pytest.approx(0.002)
    assert captured["runtime"] is ctx.runtime
    assert captured["s_index"] == 3
    assert captured["ntheta"] == 16


def test_field_line_tensor_observable_fn_packs_drift_observables() -> None:
    ctx = _make_state_context()
    captured: dict[str, object] = {}

    def gk(state, runtime, **kwargs):  # noqa: ANN001, ANN202
        captured["state"] = state
        captured.update(kwargs)
        return {
            "bmag": jnp.asarray([1.0, 3.0]),
            "epsilon": jnp.asarray(0.3),
            "bgrad": jnp.asarray([3.0, 4.0]),
            "cvdrift": jnp.asarray([1.0, 1.0]),
            "gbdrift": jnp.asarray([2.0, 0.0]),
            "gbdrift0": jnp.asarray([1.0, 1.0]),
            "jacobian": jnp.asarray([3.0, 4.0]),
        }

    observable_fn = vss._field_line_tensor_observable_fn(
        ctx=ctx,
        turbulence_mod=SimpleNamespace(gk_fieldline_geometry=gk),
        radial_index=1,
        mode_index=1,
        surface_index=2,
        alpha=0.25,
        ntheta=16,
        rms_epsilon=jnp.asarray(0.0),
    )
    observables = np.asarray(observable_fn(jnp.asarray([0.001, 0.002])))

    # order: mean(bmag), epsilon, rms(bgrad), rms(cvdrift), rms(gbdrift),
    #        rms(gbdrift0), rms(jacobian)
    np.testing.assert_allclose(
        observables,
        [2.0, 0.3, np.sqrt(12.5), 1.0, np.sqrt(2.0), 1.0, np.sqrt(12.5)],
        rtol=1e-9,
        atol=1e-9,
    )
    assert float(captured["state"].R_cos[1, 1]) == pytest.approx(0.001)
    assert captured["s_index"] == 2
    assert captured["alpha"] == pytest.approx(0.25)
    assert captured["ntheta"] == 16


def test_tensor_sensitivity_payload_packs_observables_and_ad_fd_diagnostics() -> None:
    def observables(p):  # noqa: ANN001, ANN202
        return jnp.asarray([p[0] + 2.0 * p[1], p[0] * p[0] - p[1]])

    payload = vss._tensor_sensitivity_payload(
        observable_fn=observables,
        params=jnp.asarray([0.3, -0.2]),
        fd_step=1.0e-4,
        observable_names=("obs_linear", "obs_quadratic"),
        relative_floor=1.0e-10,
    )

    assert payload["observable_names"] == ["obs_linear", "obs_quadratic"]
    np.testing.assert_allclose(payload["observables"], [-0.1, 0.29], atol=1e-9)
    np.testing.assert_allclose(
        payload["jacobian_ad"], [[1.0, 2.0], [0.6, -1.0]], atol=1e-6
    )
    np.testing.assert_allclose(payload["jacobian_fd"], payload["jacobian_ad"], atol=1e-4)
    assert float(payload["max_abs_ad_fd_error"]) < 1e-4
    assert payload["conditioning"]["jacobian_shape"] == [2, 2]


def test_boozer_flux_tube_report_payload_packs_booz_xform_metadata() -> None:
    sensitivity = {"observable_names": ["mean_bmag"], "max_abs_ad_fd_error": 1.0e-9}
    booz_meta = {
        "bmnc_b": np.asarray([1.0, 0.1, -0.05]),
        "ixm_b": np.asarray([0, 1, 2]),
        "ixn_b": np.asarray([0, 0, 4]),
        "iota_b": np.asarray(0.37),
    }

    payload = vss._boozer_flux_tube_report_payload(
        sensitivity=sensitivity, booz_meta=booz_meta, mboz=3, nboz=1, ntheta=24
    )

    assert payload["sensitivity"] is sensitivity
    assert payload["mboz"] == 3
    assert payload["nboz"] == 1
    assert payload["ntheta"] == 24
    assert payload["bmnc_b"] == [1.0, 0.1, -0.05]
    assert payload["ixm_b"] == [0, 1, 2]
    assert payload["ixn_b"] == [0, 0, 4]
    assert payload["iota_b"] == pytest.approx(0.37)
    assert isinstance(payload["iota_b"], float)


# ---------------------------------------------------------------------------
# Metadata packing and fail-closed orchestration
# ---------------------------------------------------------------------------
def test_vmec_state_sensitivity_report_from_run_merges_metadata_and_payload() -> None:
    ctx = _make_state_context(ns=3, nmode=4)
    run = vss._VMECStateSensitivityReportRun(
        ctx=ctx,
        radial_index=1,
        mode_index=2,
        surface_index=0,
        payload={"source_model": "m", "sensitivity": {"x": 1}, "extra": 7},
    )

    report = vss._vmec_state_sensitivity_report_from_run(
        backend_info={"vmex_available": True},
        run=run,
        case_name="case",
        params=jnp.asarray([0.1, -0.2]),
        fd_step=3.0e-5,
    )

    assert report["available"] is True
    assert report["param_names"] == ["delta_Rcos", "delta_Zsin"]
    np.testing.assert_allclose(report["params"], [0.1, -0.2])
    assert report["state_shape"] == [3, 4]
    assert report["radial_index"] == 1
    assert report["mode_index"] == 2
    assert report["surface_index"] == 0
    assert report["fd_step"] == pytest.approx(3.0e-5)
    assert report["case_name"] == "case"
    # payload keys are merged on top of the shared metadata
    assert report["source_model"] == "m"
    assert report["sensitivity"] == {"x": 1}
    assert report["extra"] == 7


def test_optional_vmec_state_sensitivity_report_returns_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        vss,
        "discover_differentiable_geometry_backends",
        lambda: {"vmex_available": False},
    )

    def build_run(_p):  # noqa: ANN001, ANN202
        raise AssertionError("build_run must not run when the backend is missing")

    report = vss._optional_vmec_state_sensitivity_report(
        params=None,
        default_param=1.0e-3,
        case_name="case",
        fd_step=2.0e-5,
        backend_available=lambda info: bool(info.get("vmex_available")),
        unavailable_reason="vmex is not available",
        build_run=build_run,
    )

    assert report == {
        "available": False,
        "backend_info": {"vmex_available": False},
        "sensitivity": None,
        "fd_step": 2.0e-5,
        "case_name": "case",
        "reason": "vmex is not available",
    }


def test_optional_vmec_state_sensitivity_report_returns_failed(monkeypatch) -> None:
    monkeypatch.setattr(
        vss,
        "discover_differentiable_geometry_backends",
        lambda: {"vmex_available": True},
    )

    def build_run(_p):  # noqa: ANN001, ANN202
        raise ValueError("boom")

    report = vss._optional_vmec_state_sensitivity_report(
        params=None,
        default_param=1.0e-3,
        case_name="case",
        fd_step=1.0e-6,
        backend_available=lambda info: bool(info.get("vmex_available")),
        unavailable_reason="unused",
        build_run=build_run,
    )

    assert report["available"] is False
    assert report["error"] == "ValueError: boom"
    assert report["sensitivity"] is None
    assert report["case_name"] == "case"
    assert report["backend_info"] == {"vmex_available": True}


def test_optional_vmec_state_sensitivity_report_returns_full_report(monkeypatch) -> None:
    monkeypatch.setattr(
        vss,
        "discover_differentiable_geometry_backends",
        lambda: {"vmex_available": True},
    )
    ctx = _make_state_context(ns=2, nmode=3)
    seen: dict[str, object] = {}

    def build_run(p):  # noqa: ANN001, ANN202
        seen["p"] = np.asarray(p).tolist()
        return vss._VMECStateSensitivityReportRun(
            ctx=ctx,
            radial_index=1,
            mode_index=1,
            surface_index=0,
            payload={"sensitivity": {"k": 2}, "source_model": "s"},
        )

    report = vss._optional_vmec_state_sensitivity_report(
        params=None,
        default_param=1.0e-3,
        case_name="case",
        fd_step=4.0e-5,
        backend_available=lambda info: bool(info.get("vmex_available")),
        unavailable_reason="unused",
        build_run=build_run,
    )

    # params=None resolves to the length-two default vector before build_run
    assert seen["p"] == [1.0e-3, 1.0e-3]
    assert report["available"] is True
    assert report["backend_info"] == {"vmex_available": True}
    np.testing.assert_allclose(report["params"], [1.0e-3, 1.0e-3])
    assert report["state_shape"] == [2, 3]
    assert report["sensitivity"] == {"k": 2}
    assert report["source_model"] == "s"
    assert report["fd_step"] == pytest.approx(4.0e-5)


# ---------------------------------------------------------------------------
# Context loaders (index resolution + backend seams)
# ---------------------------------------------------------------------------
def test_load_vmec_geom_sensitivity_context_resolves_metric_indices(monkeypatch) -> None:
    ctx = _make_state_context(ns=5, nmode=3)
    turbulence = SimpleNamespace(name="turbulence")
    seen: dict[str, object] = {}

    def fake_load(case_name):  # noqa: ANN001, ANN202
        seen["case_name"] = case_name
        return ctx

    monkeypatch.setattr(vss, "_load_vmec_state_context", fake_load)
    monkeypatch.setattr(vss, "_import_vmex_turbulence", lambda: turbulence)

    out_ctx, out_mod, ridx, midx, sidx = vss._load_vmec_geom_sensitivity_context(
        case_name="circular_tokamak",
        radial_index=None,
        mode_index=1,
        surface_index=None,
        surface_grid="metric",
    )

    assert out_ctx is ctx
    assert out_mod is turbulence
    assert seen["case_name"] == "circular_tokamak"
    # ns=5 -> ridx = 5 // 2 = 2 ; metric default surface = min(ridx - 1, ns - 1) = 1
    assert (ridx, midx, sidx) == (2, 1, 1)


def test_load_vmec_boozer_sensitivity_context_imports_tables_seam(monkeypatch) -> None:
    ctx = _make_state_context(ns=5, nmode=3)
    monkeypatch.setattr(vss, "_load_vmec_state_context", lambda _c: ctx)
    fake_tables = ModuleType("vmex.core.boozer_tables")
    monkeypatch.setitem(sys.modules, "vmex.core.boozer_tables", fake_tables)

    out_ctx, out_mod, ridx, midx, sidx = vss._load_vmec_boozer_sensitivity_context(
        case_name="circular_tokamak",
        radial_index=None,
        mode_index=1,
        surface_index=None,
    )

    assert out_ctx is ctx
    assert out_mod is fake_tables
    # ns=5 -> ridx = 2 ; half-mesh default surface = min(ridx - 1, ns - 2) = 1
    assert (ridx, midx, sidx) == (2, 1, 1)


# ---------------------------------------------------------------------------
# Boozer flux-tube mapping wiring + run bundling
# ---------------------------------------------------------------------------
def test_vmec_to_boozer_mapping_fn_wires_state_through_boozer_bridge(monkeypatch) -> None:
    ctx = _make_state_context(ns=4, nmode=2)
    boozer_tables_mod = ModuleType("vmex.core.boozer_tables")
    inputs_sentinel = object()
    mapping_sentinel = {"booz_xform": {"iota_b": 0.4}, "marker": "mapping"}
    seen_inputs: dict[str, object] = {}
    seen_mapping: dict[str, object] = {}

    def fake_inputs(state, runtime, **kwargs):  # noqa: ANN001, ANN202
        seen_inputs["state"] = state
        seen_inputs["runtime"] = runtime
        seen_inputs.update(kwargs)
        return inputs_sentinel

    def fake_mapping(inputs, **kwargs):  # noqa: ANN001, ANN202
        seen_mapping["inputs"] = inputs
        seen_mapping.update(kwargs)
        return mapping_sentinel

    monkeypatch.setattr(vss, "_boozer_xform_inputs_from_state", fake_inputs)
    monkeypatch.setattr(vss, "booz_xform_flux_tube_mapping_from_inputs", fake_mapping)

    mapping_fn = vss._vmec_to_boozer_mapping_fn(
        ctx=ctx,
        boozer_tables_mod=boozer_tables_mod,
        radial_index=1,
        mode_index=1,
        surface_index=0,
        mboz=2,
        nboz=0,
        ntheta=16,
    )
    result = mapping_fn(jnp.asarray([0.001, 0.002]))

    assert result is mapping_sentinel
    # state perturbed at [radial_index, mode_index] and threaded to the seam
    assert float(seen_inputs["state"].R_cos[1, 1]) == pytest.approx(0.001)
    assert float(seen_inputs["state"].Z_sin[1, 1]) == pytest.approx(0.002)
    assert seen_inputs["runtime"] is ctx.runtime
    assert seen_inputs["inp"] is ctx.inp
    assert seen_inputs["wout"] is ctx.wout
    assert seen_inputs["boozer_tables_mod"] is boozer_tables_mod
    assert seen_inputs["ns_full"] == 4
    # bridge receives the produced inputs plus the hard-wired shear / jit flags
    assert seen_mapping["inputs"] is inputs_sentinel
    assert seen_mapping["mboz"] == 2
    assert seen_mapping["nboz"] == 0
    assert seen_mapping["ntheta"] == 16
    assert seen_mapping["surface_index"] == 0
    assert seen_mapping["magnetic_shear"] == pytest.approx(0.35)
    assert seen_mapping["jit"] is False


def test_run_vmec_boozer_flux_tube_report_bundles_sensitivity_and_meta(
    monkeypatch,
) -> None:
    ctx = _make_state_context()
    fake_tables = ModuleType("vmex.core.boozer_tables")
    fake_sensitivity = {"observable_names": ["mean_bmag"], "max_abs_ad_fd_error": 0.0}
    booz_meta = {
        "bmnc_b": np.asarray([1.0, 0.1]),
        "ixm_b": np.asarray([0, 1]),
        "ixn_b": np.asarray([0, 0]),
        "iota_b": 0.42,
    }
    seen_load: dict[str, object] = {}
    seen_builder: dict[str, object] = {}
    seen_report: dict[str, object] = {}

    def fake_load(**kwargs):  # noqa: ANN001, ANN202
        seen_load.update(kwargs)
        return ctx, fake_tables, 1, 1, 0

    def fake_mapping_fn(_params):  # noqa: ANN001, ANN202
        return {"booz_xform": booz_meta}

    def fake_builder(**kwargs):  # noqa: ANN001, ANN202
        seen_builder.update(kwargs)
        return fake_mapping_fn

    def fake_report(mapping_fn, params, **kwargs):  # noqa: ANN001, ANN202
        seen_report["mapping_fn"] = mapping_fn
        seen_report["params"] = np.asarray(params).tolist()
        seen_report.update(kwargs)
        return fake_sensitivity

    monkeypatch.setattr(vss, "_load_vmec_boozer_sensitivity_context", fake_load)
    monkeypatch.setattr(vss, "_vmec_to_boozer_mapping_fn", fake_builder)
    monkeypatch.setattr(vss, "geometry_sensitivity_report", fake_report)

    run = vss._run_vmec_boozer_flux_tube_report(
        params=jnp.asarray([1.0e-3, 1.0e-3]),
        case_name="circular_tokamak",
        radial_index=None,
        mode_index=1,
        surface_index=None,
        fd_step=1.0e-5,
        mboz=2,
        nboz=0,
        ntheta=16,
    )

    assert isinstance(run, vss._VMECStateSensitivityReportRun)
    assert run.ctx is ctx
    assert (run.radial_index, run.mode_index, run.surface_index) == (1, 1, 0)
    assert run.payload["sensitivity"] is fake_sensitivity
    assert run.payload["mboz"] == 2
    assert run.payload["nboz"] == 0
    assert run.payload["ntheta"] == 16
    np.testing.assert_allclose(run.payload["bmnc_b"], [1.0, 0.1])
    assert run.payload["ixm_b"] == [0, 1]
    assert run.payload["ixn_b"] == [0, 0]
    assert run.payload["iota_b"] == pytest.approx(0.42)
    # context loader received the request unchanged
    assert seen_load["case_name"] == "circular_tokamak"
    assert seen_load["radial_index"] is None
    assert seen_load["mode_index"] == 1
    assert seen_load["surface_index"] is None
    # mapping builder wired with the resolved indices + boozer module
    assert seen_builder["ctx"] is ctx
    assert seen_builder["boozer_tables_mod"] is fake_tables
    assert seen_builder["radial_index"] == 1
    assert seen_builder["surface_index"] == 0
    assert seen_builder["mboz"] == 2
    # the sensitivity gate is tagged with the state->boozer source model
    assert seen_report["mapping_fn"] is fake_mapping_fn
    assert seen_report["source_model"] == (
        "vmex:state->booz_xform_jax:field-line-bmag"
    )
    assert seen_report["fd_step"] == pytest.approx(1.0e-5)
    np.testing.assert_allclose(seen_report["params"], [1.0e-3, 1.0e-3])


# ---------------------------------------------------------------------------
# Metric-tensor payload + run wrapper
# ---------------------------------------------------------------------------
def test_metric_tensor_report_payload_builds_pest_metric_sensitivity() -> None:
    ctx = _make_state_context(ns=4, nmode=2, resolution=(5, 6, 7))

    def gk(state, runtime, **kwargs):  # noqa: ANN001, ANN202
        r = jnp.asarray(state.R_cos)[1, 1]
        z = jnp.asarray(state.Z_sin)[1, 1]
        return {
            "jacobian": jnp.asarray([1.0, 1.0]) + r,
            "gds2": jnp.asarray([2.0, 4.0]) + z,
            "gds22": jnp.asarray([0.5, 1.5]),
            "gds21": jnp.asarray([3.0, 4.0]) + r,
            "grho": jnp.asarray([1.0, 3.0]),
            "gradpar": jnp.asarray([0.7, 0.7]),
            "s_hat": jnp.asarray(0.35) + z,
        }

    payload = vss._metric_tensor_report_payload(
        ctx=ctx,
        turbulence_mod=SimpleNamespace(gk_fieldline_geometry=gk),
        params=jnp.asarray([0.0, 0.0]),
        radial_index=1,
        mode_index=1,
        surface_index=2,
        ntheta=16,
        fd_step=1.0e-5,
        rms_epsilon=0.0,
    )

    assert payload["source_model"] == "vmex:state->metric-tensors"
    assert payload["observable_names"] == list(vss._VMEC_METRIC_OBSERVABLE_NAMES)
    np.testing.assert_allclose(
        payload["observables"],
        [1.0, 3.0, 1.0, np.sqrt(12.5), 2.0, 0.7, 0.35],
        atol=1e-9,
    )
    assert np.asarray(payload["jacobian_ad"]).shape == (7, 2)
    assert float(payload["max_abs_ad_fd_error"]) < 1e-6
    # sqrt(g) observable (row 0) tracks the R_cos control (param 0); the mean
    # g_ss observable (row 1) tracks the Z_sin control (param 1)
    np.testing.assert_allclose(payload["jacobian_ad"][0], [1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(payload["jacobian_ad"][1], [0.0, 1.0], atol=1e-6)
    assert payload["ntheta"] == 16
    assert payload["metric_grid_shape"] == [5, 6, 7]
    assert payload["rms_epsilon"] == 0.0


def test_run_vmec_metric_tensor_sensitivity_wraps_payload(monkeypatch) -> None:
    ctx = _make_state_context()
    turbulence = SimpleNamespace(name="turbulence")
    sentinel_payload = {"source_model": "vmex:state->metric-tensors", "tag": "m"}
    seen_load: dict[str, object] = {}
    seen_payload: dict[str, object] = {}

    def fake_load(**kwargs):  # noqa: ANN001, ANN202
        seen_load.update(kwargs)
        return ctx, turbulence, 1, 1, 0

    def fake_payload(**kwargs):  # noqa: ANN001, ANN202
        seen_payload.update(kwargs)
        return sentinel_payload

    monkeypatch.setattr(vss, "_load_vmec_geom_sensitivity_context", fake_load)
    monkeypatch.setattr(vss, "_metric_tensor_report_payload", fake_payload)

    run = vss._run_vmec_metric_tensor_sensitivity(
        params=jnp.asarray([0.0, 0.0]),
        case_name="circular_tokamak",
        radial_index=None,
        mode_index=1,
        surface_index=None,
        ntheta=16,
        fd_step=1.0e-5,
        rms_epsilon=1.0e-24,
    )

    assert isinstance(run, vss._VMECStateSensitivityReportRun)
    assert run.ctx is ctx
    assert (run.radial_index, run.mode_index, run.surface_index) == (1, 1, 0)
    assert run.payload is sentinel_payload
    assert seen_load["surface_grid"] == "metric"
    assert seen_load["case_name"] == "circular_tokamak"
    assert seen_payload["ctx"] is ctx
    assert seen_payload["turbulence_mod"] is turbulence
    assert seen_payload["radial_index"] == 1
    assert seen_payload["mode_index"] == 1
    assert seen_payload["surface_index"] == 0
    assert seen_payload["ntheta"] == 16
    assert seen_payload["fd_step"] == pytest.approx(1.0e-5)
    assert seen_payload["rms_epsilon"] == pytest.approx(1.0e-24)


# ---------------------------------------------------------------------------
# Field-line payload + run wrapper
# ---------------------------------------------------------------------------
def test_field_line_tensor_report_payload_builds_field_line_sensitivity() -> None:
    ctx = _make_state_context(ns=4, nmode=2, resolution=(5, 6, 7))

    def gk(state, runtime, **kwargs):  # noqa: ANN001, ANN202
        r = jnp.asarray(state.R_cos)[1, 1]
        z = jnp.asarray(state.Z_sin)[1, 1]
        return {
            "bmag": jnp.asarray([1.0, 3.0]) + r,
            "epsilon": jnp.asarray(0.3) + z,
            "bgrad": jnp.asarray([3.0, 4.0]) + r,
            "cvdrift": jnp.asarray([1.0, 1.0]) + z,
            "gbdrift": jnp.asarray([2.0, 0.0]) + r,
            "gbdrift0": jnp.asarray([1.0, 1.0]),
            "jacobian": jnp.asarray([3.0, 4.0]),
            "vmex": {"field_line_convention": "pest", "iota": 0.42},
        }

    payload = vss._field_line_tensor_report_payload(
        ctx=ctx,
        turbulence_mod=SimpleNamespace(gk_fieldline_geometry=gk),
        params=jnp.asarray([0.0, 0.0]),
        radial_index=1,
        mode_index=1,
        surface_index=2,
        alpha=0.25,
        ntheta=16,
        fd_step=1.0e-6,
        b2_floor=1.0e-24,
        rms_epsilon=0.0,
    )

    assert payload["source_model"] == "vmex:state->field-line-metric-and-b"
    assert payload["field_line_convention"] == "pest"
    assert payload["observable_names"] == list(vss._VMEC_FIELD_LINE_OBSERVABLE_NAMES)
    np.testing.assert_allclose(
        payload["observables"],
        [2.0, 0.3, np.sqrt(12.5), 1.0, np.sqrt(2.0), 1.0, np.sqrt(12.5)],
        atol=1e-9,
    )
    assert np.asarray(payload["jacobian_ad"]).shape == (7, 2)
    assert float(payload["max_abs_ad_fd_error"]) < 1e-6
    # mean |B| (row 0) tracks the R_cos control; epsilon (row 1) tracks Z_sin
    np.testing.assert_allclose(payload["jacobian_ad"][0], [1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(payload["jacobian_ad"][1], [0.0, 1.0], atol=1e-6)
    assert payload["iota"] == pytest.approx(0.42)
    assert payload["alpha"] == pytest.approx(0.25)
    assert payload["ntheta"] == 16
    assert payload["metric_grid_shape"] == [5, 6, 7]
    assert payload["b2_floor"] == pytest.approx(1.0e-24)
    assert payload["rms_epsilon"] == 0.0


def test_run_vmec_field_line_tensor_sensitivity_wraps_payload(monkeypatch) -> None:
    ctx = _make_state_context()
    turbulence = SimpleNamespace(name="turbulence")
    sentinel_payload = {"source_model": "vmex:state->field-line-metric-and-b"}
    seen_load: dict[str, object] = {}
    seen_payload: dict[str, object] = {}

    def fake_load(**kwargs):  # noqa: ANN001, ANN202
        seen_load.update(kwargs)
        return ctx, turbulence, 1, 1, 0

    def fake_payload(**kwargs):  # noqa: ANN001, ANN202
        seen_payload.update(kwargs)
        return sentinel_payload

    monkeypatch.setattr(vss, "_load_vmec_geom_sensitivity_context", fake_load)
    monkeypatch.setattr(vss, "_field_line_tensor_report_payload", fake_payload)

    run = vss._run_vmec_field_line_tensor_sensitivity(
        params=jnp.asarray([0.0, 0.0]),
        case_name="nfp4_QH_warm_start",
        radial_index=None,
        mode_index=1,
        surface_index=None,
        alpha=0.0,
        ntheta=16,
        fd_step=1.0e-6,
        b2_floor=1.0e-24,
        rms_epsilon=1.0e-24,
    )

    assert isinstance(run, vss._VMECStateSensitivityReportRun)
    assert run.ctx is ctx
    assert (run.radial_index, run.mode_index, run.surface_index) == (1, 1, 0)
    assert run.payload is sentinel_payload
    assert seen_load["surface_grid"] == "field_line"
    assert seen_load["case_name"] == "nfp4_QH_warm_start"
    assert seen_payload["ctx"] is ctx
    assert seen_payload["turbulence_mod"] is turbulence
    assert seen_payload["alpha"] == pytest.approx(0.0)
    assert seen_payload["surface_index"] == 0
    assert seen_payload["b2_floor"] == pytest.approx(1.0e-24)
    assert seen_payload["rms_epsilon"] == pytest.approx(1.0e-24)
