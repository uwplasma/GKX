"""Inactive HNGC cost contract and independent local-shear/pressure oracles."""
from types import SimpleNamespace

import numpy as np
import pytest

from gkx.geometry import vmec_boozer_derivatives as vd
from gkx.geometry import vmec_state_controls as controls


@pytest.mark.parametrize("flags", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("psi", [-0.01, 0.01])
@pytest.mark.parametrize("pressure_slope", [0.0, -700.0])
@pytest.mark.parametrize("shat", [0.0, -1e-8, 0.2])
def test_hngc_integrations_are_needed_only_for_enabled_corrections(
    monkeypatch, flags, psi, pressure_slope, shat
):
    s, iota = 0.25, 0.6
    native_diota = -shat * iota / (2 * s)
    const = lambda value: lambda x: np.full_like(np.asarray(x), value, dtype=float)
    vs = SimpleNamespace(
        phiedge=-2 * np.pi * psi, Aminor_p=0.2, nfp=1,
        raxis_cc=np.array([3.0]), mnbooz=2, xm_b=np.array([0, 1]), xn_b=np.array([0, 0]),
        rmnc_b=[const(3.0), const(0.1)], zmns_b=[const(0.0), const(0.1)],
        numns_b=[const(0.0), const(0.0)], d_rmnc_b_d_s=[const(0.0), const(0.2)],
        d_zmns_b_d_s=[const(0.0), const(0.2)], d_numns_b_d_s=[const(0.0), const(0.0)],
        gmnc_b=[const(-2.0), const(0.1)], bmnc_b=[const(1.0), const(0.1)],
        d_bmnc_b_d_s=[const(0.02), const(0.01)], Gfun=const(-2.0), Ifun=const(0.03),
        iota=const(iota), d_iota_d_s=const(native_diota), d_pressure_d_s=const(pressure_slope),
    )
    scalars = controls._fieldline_scalar_profiles(
        vs, s_val=s, alpha=0.2, iota_input=0.65, s_hat_input=0.1
    )
    samples = controls._sample_fieldline_boozer_state(
        vs, scalars, theta1d=np.linspace(-np.pi, np.pi, 9), isaxisym=False
    )
    hngc = vd._hngc_mode_corrections(scalars, samples)
    geometry = vd._fieldline_metric_geometry(scalars, samples)
    inv = 0.2 + samples.phi_b**2
    lam = 0.1 + 0.3 * samples.phi_b
    calls, observed = [], {}

    def expensive(*args, **kwargs):
        assert any(flags), "disabled corrections must not integrate the flux surface"
        calls.append(kwargs)
        return vd._FieldlineHNGCIntegrals(1.7, 0.4, inv, lam)

    original_drifts = vd._fieldline_metric_drifts

    def capture_drifts(**kwargs):
        observed.update(kwargs)
        return original_drifts(**kwargs)

    monkeypatch.setattr(vd, "_fieldline_hngc_integrals", expensive)
    monkeypatch.setattr(vd, "_fieldline_metric_drifts", capture_drifts)
    vd._fieldline_metric_coefficients(
        scalars, samples, hngc, s_val=s, betaprim=0.003,
        include_shear_variation=flags[0], include_pressure_variation=flags[1],
        res_theta=201, res_phi=201,
    )
    assert len(calls) == int(any(flags))
    if calls:
        assert calls[0] == {"res_theta": 201, "res_phi": 201}
    # Independent Hegna--Nakajima formula; disabled terms are exact zeros.
    diota = (-0.65 * 0.1 / (2 * s) - native_diota) if flags[0] else 0.0
    drive = 0.003 * scalars.B_reference**2 / (4 * np.sqrt(s))
    dp = drive - vd._MU_0 * pressure_slope if flags[1] else 0.0
    phi = samples.phi_b - scalars.zeta_center
    expected_D = (
        diota * (inv / 1.7 - phi)
        - dp * samples.Vprime[:, None, None]
        * (scalars.G + scalars.iota * scalars.boozer_i)[:, None, None]
        * (lam - 0.4 * inv / 1.7)
    ) / psi
    ratio = geometry.alpha_gradients.grad_alpha_dot_grad_psi / geometry.alpha_gradients.g_sup_psi_psi
    shear = observed["shear"]
    np.testing.assert_allclose(shear.D_HNGC, expected_D, rtol=1e-13, atol=1e-14)
    if not any(flags):
        np.testing.assert_array_equal(shear.D_HNGC, np.zeros_like(phi))
    # L0 subtracts large equal native-shear terms; bound roundoff by their scale.
    roundoff = 32 * np.finfo(float).eps * max(1.0, np.max(abs(ratio)), np.max(abs(native_diota * phi / psi)))
    np.testing.assert_allclose(shear.L0, -ratio - native_diota * phi / psi, rtol=1e-13, atol=roundoff)
    np.testing.assert_allclose(shear.L1, ratio - diota * phi / psi - expected_D, rtol=1e-13, atol=roundoff)
    # The guard must preserve native pressure physics even with variation disabled.
    np.testing.assert_array_equal(observed["d_pressure_d_s"], np.array([pressure_slope]))
    expected_pfac = drive / (vd._MU_0 * (pressure_slope if abs(pressure_slope) >= 1e-30 else 1e-8)) if flags[1] else 1.0
    assert observed["pfac"] == pytest.approx(expected_pfac)
    assert observed["sfac"] == pytest.approx(shat / 0.1 if flags[0] else 1.0)
