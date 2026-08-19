"""Finite-beta parity between the wout runtime path and the vmex-state bridge.

GKX evaluates VMEC flux-tube geometry two ways: the wout runtime path
(``imported_vmec`` + ``vmec_field_line_sampling``) and the differentiable
vmex-state bridge (``vmec_boozer_core`` + ``vmec_boozer_drifts``) that the
adjoint and objective stack uses.  The bridge used to alias ``gbdrift`` to
``cvdrift`` and drop ``mu0 dp/ds`` from the normal curvature, so every
boundary-coefficient gradient ignored finite-beta drift physics.  These tests
pin the two paths together at finite beta and pin the zero-beta behaviour that
the pressure terms must leave untouched.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from gkx.geometry.numerics import _array_parity_metrics
from gkx.geometry.vmec_boozer_drifts import (
    _MU_0,
    boozer_pressure_gradient,
    raw_drift_profiles,
)

_FINITE_BETA_CASES = (
    ("LandremanPaul2021_QA_beta2", 0.25),
    ("LandremanPaul2021_QA_beta2", 0.64),
    ("LandremanPaul2021_QA_beta2_current", 0.64),
)
# The bridge and the runtime path disagree at this level on quantities that
# carry no pressure at all (bgrad, gds2), so it is the bridge's own numerical
# floor rather than a pressure tolerance. Before the pressure terms landed
# cvdrift missed by 2.1e-1 to 2.6e-1, an order of magnitude above this.
_DRIFT_TOLERANCE = 2.0e-2
_METRIC_TOLERANCE = 8.0e-2


def _stub_drift_inputs(d_pressure_ds: float) -> tuple[object, ...]:
    """Build the minimal duck-typed bundle ``raw_drift_profiles`` consumes."""

    theta = jnp.linspace(-jnp.pi, jnp.pi, 8, dtype=jnp.float64)[:-1]
    mod_b = 1.0 + 0.1 * jnp.cos(theta)
    request = SimpleNamespace(
        base_Rcos=jnp.zeros((3, 2), dtype=jnp.float64), torflux=0.5
    )
    scales = SimpleNamespace(length=0.3, magnetic_field=1.2)
    profiles = SimpleNamespace(
        boozer_g=jnp.asarray(1.4, dtype=jnp.float64),
        boozer_i=jnp.asarray(0.07, dtype=jnp.float64),
        iota_safe=jnp.asarray(0.42, dtype=jnp.float64),
        d_iota_ds=jnp.asarray(0.11, dtype=jnp.float64),
        s_hat=jnp.asarray(-0.26, dtype=jnp.float64),
        d_pressure_ds=jnp.asarray(d_pressure_ds, dtype=jnp.float64),
    )
    equal_arc = SimpleNamespace(
        mod_b_safe=mod_b,
        sqrt_g_booz=(1.4 + 0.42 * 0.07) / (mod_b * mod_b),
    )
    spectral = SimpleNamespace(
        d_mod_b_d_theta=-0.1 * jnp.sin(theta),
        d_mod_b_d_phi=0.03 * jnp.cos(theta),
        d_mod_b_d_s=0.2 + 0.05 * jnp.cos(theta),
    )
    state = SimpleNamespace(
        spectral=spectral,
        eps=jnp.asarray(1.0e-30, dtype=jnp.float64),
        etf=jnp.asarray(-0.013, dtype=jnp.float64),
        etf_safe=jnp.asarray(-0.013, dtype=jnp.float64),
        local_shear_l1=0.05 * jnp.sin(theta),
        shear_phase=theta,
        metric_bmag_sq=mod_b * mod_b,
    )
    return request, scales, profiles, equal_arc, state


def test_zero_pressure_restores_the_zero_beta_drift_alias() -> None:
    """At zero beta the bridge must reproduce the old ``gbdrift = cvdrift``."""

    drifts = raw_drift_profiles(*_stub_drift_inputs(0.0))

    np.testing.assert_array_equal(
        np.asarray(drifts.gbdrift), np.asarray(drifts.cvdrift)
    )
    np.testing.assert_array_equal(
        np.asarray(drifts.gbdrift0), np.asarray(drifts.cvdrift0)
    )


def test_pressure_moves_the_curvature_drift_and_leaves_the_grad_b_drift() -> None:
    """``mu0 dp/ds`` belongs to the curvature, not to ``grad B``.

    Force balance puts the pressure gradient in ``kappa`` and nowhere else, so
    the same term that shifts ``cvdrift`` is subtracted back out of ``gbdrift``.
    The grad-B drift must therefore be numerically identical with and without
    pressure, and the split must equal the analytic offset.
    """

    d_pressure_ds = -3.1e4
    vacuum = raw_drift_profiles(*_stub_drift_inputs(0.0))
    finite = raw_drift_profiles(*_stub_drift_inputs(d_pressure_ds))

    grad_b = np.asarray(finite.gbdrift)
    # The invariant is exactness, so the tolerance tracks the working precision
    # rather than assuming float64: the suite runs both ways.
    tol = 8.0 * float(np.finfo(grad_b.dtype).eps)
    np.testing.assert_allclose(
        grad_b, np.asarray(vacuum.gbdrift), rtol=tol, atol=tol
    )
    np.testing.assert_allclose(
        np.asarray(finite.cvdrift0), np.asarray(vacuum.cvdrift0), rtol=tol, atol=tol
    )

    _request, scales, _profiles, _equal_arc, state = _stub_drift_inputs(d_pressure_ds)
    expected_offset = (
        2.0
        * scales.magnetic_field
        * scales.length**2
        * np.sqrt(0.5)
        * _MU_0
        * d_pressure_ds
        * np.sign(float(state.etf))
        / (float(state.etf_safe) * np.asarray(state.metric_bmag_sq))
    )
    np.testing.assert_allclose(
        np.asarray(finite.gbdrift) - np.asarray(finite.cvdrift),
        expected_offset,
        rtol=max(tol, 1e-10),
    )
    assert np.max(np.abs(np.asarray(finite.cvdrift) - np.asarray(vacuum.cvdrift))) > 0.0


def test_pressure_gradient_is_exact_for_a_quadratic_vmec_profile() -> None:
    """``p(s) = p0 (1-s)^2`` differentiates exactly on the VMEC half mesh."""

    ns = 50
    p0 = 2.2e4
    s_full = np.linspace(0.0, 1.0, ns)
    s_half = 0.5 * (s_full[:-1] + s_full[1:])
    pres = np.concatenate([[0.0], p0 * (1.0 - s_half) ** 2])
    wout = SimpleNamespace(pres=pres)

    for s_value in (0.25, 0.64):
        got = float(
            boozer_pressure_gradient(wout, s_value=s_value, dtype=jnp.float64)
        )
        assert got == pytest.approx(-2.0 * p0 * (1.0 - s_value), rel=2e-3)


def test_pressure_gradient_vanishes_for_a_vacuum_equilibrium() -> None:
    """A wout with no pressure profile must not perturb the vacuum drifts."""

    assert (
        float(
            boozer_pressure_gradient(
                SimpleNamespace(), s_value=0.5, dtype=jnp.float64
            )
        )
        == 0.0
    )
    assert (
        float(
            boozer_pressure_gradient(
                SimpleNamespace(pres=np.zeros(50)), s_value=0.5, dtype=jnp.float64
            )
        )
        == 0.0
    )


def test_pressure_gradient_rejects_a_malformed_profile() -> None:
    """Silently dropping pressure is the defect this module exists to remove."""

    with pytest.raises(ValueError, match="pressure profile"):
        boozer_pressure_gradient(
            SimpleNamespace(pres=np.zeros((4, 2))), s_value=0.5, dtype=jnp.float64
        )


def _finite_beta_artifact_dir() -> Path:
    raw = os.environ.get("GKX_FINITE_BETA_VMEC_DIR", "").strip()
    if not raw:
        pytest.skip(
            "Set GKX_FINITE_BETA_VMEC_DIR to a directory holding "
            "input.LandremanPaul2021_QA_beta2[_current] and their wout files "
            "to enable the finite-beta VMEC/Boozer parity gate."
        )
    directory = Path(raw).expanduser()
    if not directory.is_dir():
        pytest.skip(f"GKX_FINITE_BETA_VMEC_DIR is not a directory: {directory}")
    return directory


def _finite_beta_case_paths(directory: Path, case: str) -> tuple[Path, Path]:
    input_path = directory / f"input.{case}"
    wout_path = directory / f"wout_{case}.nc"
    missing = [p.name for p in (input_path, wout_path) if not p.is_file()]
    if missing:
        pytest.skip(f"missing finite-beta artifacts in {directory}: {missing}")
    return input_path, wout_path


def _imported_runtime_geometry(wout_path: Path, *, torflux: float, ntheta: int):
    from gkx.geometry import load_imported_geometry_netcdf
    from gkx.geometry.imported_vmec import generate_vmec_eik_internal
    from gkx.geometry.vmec_flux_tube_reports import _VMEC_EIK_DEFAULT_REQUEST

    payload = dict(_VMEC_EIK_DEFAULT_REQUEST)
    payload.update(
        {
            "vmec_file": str(wout_path),
            "ntheta": int(ntheta),
            "boundary": "none",
            "alpha": 0.0,
            "torflux": float(torflux),
            "include_shear_variation": False,
            "include_pressure_variation": False,
        }
    )
    with tempfile.TemporaryDirectory(prefix="gkx_finite_beta_parity_") as tmp:
        eik_path = Path(tmp) / "finite_beta.eik.nc"
        generate_vmec_eik_internal(
            output_path=eik_path, request=SimpleNamespace(**payload)
        )
        return load_imported_geometry_netcdf(eik_path)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize(("case", "torflux"), _FINITE_BETA_CASES)
def test_finite_beta_drifts_agree_between_wout_and_state_paths(
    case: str, torflux: float
) -> None:
    """The differentiable bridge must reproduce the runtime path at ~2% beta.

    Both routes describe one equilibrium, so a disagreement here is a missing
    physical term rather than a convention difference: this gate is what caught
    the bridge computing the grad-B drift and labelling it the curvature drift.
    """

    pytest.importorskip("vmex")
    pytest.importorskip("booz_xform_jax")
    directory = _finite_beta_artifact_dir()
    input_path, wout_path = _finite_beta_case_paths(directory, case)

    from gkx.geometry.vmec_boozer_core import (
        load_solved_vmex_case,
        vmex_boozer_equal_arc_core_profiles_from_state,
    )

    ntheta = 32
    inp, state, runtime, wout = load_solved_vmex_case(str(input_path))
    bridge = vmex_boozer_equal_arc_core_profiles_from_state(
        state,
        runtime,
        inp,
        wout,
        surface_index=None,
        torflux=torflux,
        alpha=0.0,
        ntheta=ntheta,
        mboz=24,
        nboz=24,
        jit=False,
    )
    imported = _imported_runtime_geometry(wout_path, torflux=torflux, ntheta=ntheta)
    if imported.theta.shape[0] == ntheta + 1:
        imported = imported.trim_terminal_theta_point()

    tolerances = {
        "bmag": 1.0e-3,
        "gds2": _METRIC_TOLERANCE,
        "gds21": _METRIC_TOLERANCE,
        "gds22": _METRIC_TOLERANCE,
        "grho": _METRIC_TOLERANCE,
        "cvdrift": _DRIFT_TOLERANCE,
        "gbdrift": _DRIFT_TOLERANCE,
        # cvdrift0/gbdrift0 are grad-psi components carrying no pressure term:
        # they were already at 2.2e-2 here before the pressure work and moved
        # by nothing, so they are held to the metric floor they actually share
        # with gds21/gds22 rather than to the drift tolerance.
        "cvdrift0": _METRIC_TOLERANCE,
        "gbdrift0": _METRIC_TOLERANCE,
    }
    attributes = {
        "bmag": "bmag_profile",
        "gds2": "gds2_profile",
        "gds21": "gds21_profile",
        "gds22": "gds22_profile",
        "grho": "grho_profile",
        "cvdrift": "cv_profile",
        "gbdrift": "gb_profile",
        "cvdrift0": "cv0_profile",
        "gbdrift0": "gb0_profile",
    }
    for name, tolerance in tolerances.items():
        metrics = _array_parity_metrics(
            np.asarray(bridge[name]), getattr(imported, attributes[name])
        )
        assert bool(metrics["shape_match"]), f"{name} shape mismatch"
        assert float(metrics["normalized_max_abs"]) <= tolerance, (
            f"{name} parity {metrics['normalized_max_abs']:.3e} exceeds {tolerance:.1e}"
        )

    # The split itself is the finite-beta physics: assert it is present in the
    # bridge and the same size as the runtime path, not merely that both are
    # close to zero.
    bridge_split = np.asarray(bridge["cvdrift"]) - np.asarray(bridge["gbdrift"])
    runtime_split = np.asarray(imported.cv_profile) - np.asarray(imported.gb_profile)
    drift_amplitude = float(np.max(np.abs(np.asarray(imported.cv_profile))))
    assert np.max(np.abs(runtime_split)) > 0.1 * drift_amplitude
    np.testing.assert_allclose(
        bridge_split, runtime_split, rtol=0.0, atol=1.0e-3 * drift_amplitude
    )
