"""Miller geometry backend, eik-file, and low-level kernel tests."""

from __future__ import annotations


# Backend kernel finite-difference and periodic-extension contracts.

import numpy as np
import pytest

from gkx.geometry.kernels import (
    finite_diff_nonuniform,
    centered_reflected_difference,
    weighted_centered_difference,
    extend_nperiod_data,
    reflect_and_append,
    nperiod_contract,
    nperiod_mask,
)


def test_nperiod_helpers_contract_arrays() -> None:
    theta = np.array([-4.0, -1.0, 0.0, 1.0, 4.0])
    values = np.arange(theta.size)
    mask = np.asarray(nperiod_mask(theta, 1.0))
    assert mask.tolist() == [False, True, True, True, False]
    contracted_values, contracted_theta = nperiod_contract.__wrapped__(
        values, theta, 1.0
    )
    np.testing.assert_allclose(np.asarray(contracted_values), [1, 2, 3])
    np.testing.assert_allclose(np.asarray(contracted_theta), [-1.0, 0.0, 1.0])


def test_finite_diff_nonuniform_matches_quadratic_derivative() -> None:
    grid = np.array([0.0, 0.5, 1.5, 3.0])
    values = grid**2
    diff = np.asarray(finite_diff_nonuniform(values, grid))
    np.testing.assert_allclose(diff[1:-1], 2.0 * grid[1:-1], atol=1.0e-6)
    assert np.isfinite(diff[[0, -1]]).all()
    np.testing.assert_allclose(
        np.asarray(finite_diff_nonuniform(np.array([1.0, 2.0]), np.array([0.0, 1.0]))),
        [0.0, 0.0],
    )


def test_centered_reflected_difference_handles_1d_and_2d_axes() -> None:
    arr = np.array([1.0, 2.0, 4.0, 7.0])
    np.testing.assert_allclose(
        np.asarray(centered_reflected_difference(arr, axis="l", parity="e")),
        [0.0, 3.0, 5.0, 0.0],
    )
    np.testing.assert_allclose(
        np.asarray(centered_reflected_difference(arr, axis="r")), [2.0, 3.0, 5.0, 6.0]
    )
    arr2 = np.arange(12.0).reshape(3, 4)
    out_r = np.asarray(centered_reflected_difference(arr2, axis="r"))
    out_l = np.asarray(centered_reflected_difference(arr2, axis="l", parity="o"))
    assert out_r.shape == arr2.shape
    assert out_l.shape == arr2.shape
    with pytest.raises(ValueError):
        centered_reflected_difference(np.ones((2, 2, 2)), axis="l")
    with pytest.raises(ValueError):
        centered_reflected_difference(arr, axis="bad")


def test_weighted_centered_difference_handles_weighted_derivatives() -> None:
    grid = np.array([0.0, 1.0, 2.0, 4.0])
    values = grid**2
    out_l = np.asarray(weighted_centered_difference(values, grid, axis="l", parity="o"))
    assert out_l.shape == values.shape
    assert np.isfinite(out_l).all()

    arr2 = np.vstack([values, values + 1.0])
    grid2 = np.vstack([grid, grid])
    out2 = np.asarray(weighted_centered_difference(arr2, grid2, axis="r"))
    assert out2.shape == arr2.shape

    with pytest.raises(ValueError):
        weighted_centered_difference(np.ones(3), np.ones(4), axis="l")
    with pytest.raises(ValueError):
        weighted_centered_difference(np.ones((2, 2, 2)), np.ones((2, 2, 2)), axis="l")
    with pytest.raises(ValueError):
        weighted_centered_difference(values, grid, axis="bad")


def test_nperiod_extension_and_reflection_helpers() -> None:
    base = np.array([0.0, 1.0, 2.0])
    even = np.asarray(extend_nperiod_data(base, 2, istheta=False, parity="e"))
    odd = np.asarray(extend_nperiod_data(base, 2, istheta=False, parity="o"))
    theta = np.asarray(extend_nperiod_data(base, 2, istheta=True))
    assert even.size == 7
    assert odd.size == 7
    assert theta.size == 7
    np.testing.assert_allclose(
        np.asarray(reflect_and_append(base, "e")), [2.0, 1.0, 0.0, 1.0, 2.0]
    )
    np.testing.assert_allclose(
        np.asarray(reflect_and_append(base, "o")), [-2.0, -1.0, 0.0, 1.0, 2.0]
    )


def test_backend_kernel_exports_canonical_helpers() -> None:
    arr = np.array([1.0, 2.0, 4.0, 7.0])
    np.testing.assert_allclose(
        np.asarray(centered_reflected_difference(arr, axis="r")),
        np.asarray(centered_reflected_difference(arr, axis="r")),
    )
    assert callable(centered_reflected_difference)


# Internal Miller backend request, collocation, and NetCDF contracts.

from pathlib import Path
from types import SimpleNamespace


from gkx.geometry.analytic import MillerCoreParams, build_collocation_surfaces
from gkx.geometry.kernels import _safe_denom, cumulative_trapezoid
from gkx.geometry.imported_miller import (
    _request_attr,
    generate_miller_eik_internal,
)


def _request() -> SimpleNamespace:
    return SimpleNamespace(
        ntheta=24,
        nperiod=1,
        rhoc=0.5,
        q=1.4,
        s_hat=0.8,
        R0=3.0,
        R_geo=3.0,
        shift=0.0,
        akappa=1.0,
        tri=0.0,
        akappri=0.0,
        tripri=0.0,
        betaprim=0.0,
    )


def test_safe_denom_and_request_attr() -> None:
    assert _safe_denom(0.0) > 0.0
    assert _safe_denom(-0.0) > 0.0
    arr = _safe_denom(np.array([0.0, -1.0e-40, 2.0]))
    assert np.all(np.abs(arr) > 0.0)

    req = SimpleNamespace(q=1.4, s_hat=0.9)
    assert _request_attr(req, "qinp", "q") == 1.4
    assert _request_attr(req, "shat", "s_hat") == 0.9
    with pytest.raises(AttributeError):
        _request_attr(req, "missing_a", "missing_b")


def test_cumulative_trapezoid_supports_1d_and_2d() -> None:
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])
    np.testing.assert_allclose(cumulative_trapezoid(y, x), [0.0, 0.5, 2.0])

    y2 = np.vstack([y, 2.0 * y])
    out = cumulative_trapezoid(y2, x, axis=1)
    np.testing.assert_allclose(out[0], [0.0, 0.5, 2.0])
    np.testing.assert_allclose(out[1], [0.0, 1.0, 4.0])

    x2 = np.vstack([x, x + 0.5])
    out2 = cumulative_trapezoid(y2, x2, axis=1)
    assert out2.shape == y2.shape

    with pytest.raises(ValueError):
        cumulative_trapezoid(np.array([1.0, 2.0]), np.array([[0.0, 1.0]]))
    with pytest.raises(NotImplementedError):
        cumulative_trapezoid(np.ones((2, 2)), np.ones(2), axis=0)
    with pytest.raises(ValueError):
        cumulative_trapezoid(np.ones((2, 2, 2)), np.ones((2, 2, 2)))


def test_miller_collocation_preserves_signed_shift_derivative() -> None:
    params = MillerCoreParams(
        ntgrid=8,
        nperiod=1,
        rhoc=0.5,
        qinp=1.4,
        shat=0.8,
        rmaj=3.0,
        r_geo=3.0,
        shift=-0.2,
        akappa=1.0,
        tri=0.0,
        akappri=0.0,
        tripri=0.0,
        betaprim=0.0,
        delrho=1.0e-3,
    )

    state = build_collocation_surfaces(params)
    r_midplane = np.asarray(state["r"])[:, 0]
    rho = np.asarray(state["rho"])
    d_rgeom_drho = np.gradient(r_midplane - rho, rho)[1]

    assert d_rgeom_drho == pytest.approx(params.shift, rel=1.0e-6)


def test_generate_miller_eik_internal_writes_netcdf(tmp_path: Path) -> None:
    out = generate_miller_eik_internal(
        output_path=tmp_path / "miller.eiknc.nc", request=_request()
    )
    assert out.exists()

    import netCDF4 as nc

    with nc.Dataset(out) as ds:
        assert "theta" in ds.variables
        assert "bmag" in ds.variables
        assert ds.variables["theta"][:].size > 0
        assert float(ds.variables["q"].getValue()) == pytest.approx(1.4)
        assert float(ds.variables["shat"].getValue()) == pytest.approx(0.8)


# Runtime Miller eik request and generation contracts.

from unittest.mock import MagicMock

from gkx.config import (
    GeometryConfig,
    GridConfig,
    InitializationConfig,
    TimeConfig,
)
from gkx.geometry.miller_eik import (
    build_miller_geometry_request,
    generate_runtime_miller_eik,
)
from gkx.workflows.runtime.config import (
    RuntimeConfig,
    RuntimeNormalizationConfig,
    RuntimePhysicsConfig,
    RuntimeSpeciesConfig,
)


def _miller_runtime_cfg(
    tmp_path: Path, *, geometry_file: str | None = None
) -> RuntimeConfig:
    return RuntimeConfig(
        grid=GridConfig(
            Nx=32,
            Ny=16,
            Nz=24,
            Lx=62.8,
            Ly=62.8,
            boundary="linked",
            y0=10.0,
            ntheta=24,
            nperiod=1,
        ),
        time=TimeConfig(
            t_max=1.0, dt=0.1, method="rk3", use_diffrax=False, fixed_dt=True
        ),
        geometry=GeometryConfig(
            model="miller",
            geometry_file=geometry_file,
            q=1.4,
            s_hat=0.8,
            rhoc=0.5,
            R0=2.77778,
            R_geo=2.77778,
            shift=0.0,
            akappa=1.0,
            akappri=0.0,
            tri=0.0,
            tripri=0.0,
            betaprim=0.0,
        ),
        init=InitializationConfig(init_field="density", init_amp=1.0e-6),
        species=(
            RuntimeSpeciesConfig(
                name="ion", charge=1.0, mass=1.0, tprim=2.49, fprim=0.8
            ),
        ),
        physics=RuntimePhysicsConfig(
            linear=False,
            nonlinear=True,
            adiabatic_electrons=True,
            tau_e=1.0,
            electrostatic=True,
            electromagnetic=False,
            beta=0.0,
            collisions=False,
        ),
        normalization=RuntimeNormalizationConfig(
            contract="cyclone", diagnostic_norm="rho_star"
        ),
    )


def test_build_miller_geometry_request_creates_expected_request(tmp_path: Path) -> None:
    cfg = _miller_runtime_cfg(tmp_path)
    request = build_miller_geometry_request(cfg)

    assert request.q == 1.4
    assert request.s_hat == 0.8
    assert request.rhoc == 0.5
    assert request.ntheta == 24
    assert request.nperiod == 1


def test_generate_runtime_miller_eik_invokes_internal_generator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_path = tmp_path / "geom.eiknc.nc"
    cfg = _miller_runtime_cfg(tmp_path, geometry_file=str(out_path))

    mock_gen = MagicMock(return_value=out_path.resolve())
    monkeypatch.setattr(
        "gkx.geometry.miller_eik.generate_miller_eik_internal", mock_gen
    )
    out_path.write_bytes(b"stale")
    out = generate_runtime_miller_eik(cfg, force=True)

    assert out == out_path.resolve()
    assert mock_gen.called
    _, kwargs = mock_gen.call_args
    request = kwargs["request"]
    assert request.ntheta == 24
    assert request.q == 1.4


def test_generate_runtime_miller_eik_reuses_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_path = tmp_path / "cached.eiknc.nc"
    out_path.write_bytes(b"cached")
    cfg = _miller_runtime_cfg(tmp_path, geometry_file=str(out_path))
    mock_gen = MagicMock()
    monkeypatch.setattr(
        "gkx.geometry.miller_eik.generate_miller_eik_internal", mock_gen
    )

    out = generate_runtime_miller_eik(cfg)

    assert out == out_path.resolve()
    mock_gen.assert_not_called()


def test_internal_miller_request_attr_accepts_runtime_aliases() -> None:
    class Req:
        q = 1.4
        s_hat = 0.8

    req = Req()
    assert _request_attr(req, "qinp", "q") == 1.4
    assert _request_attr(req, "shat", "s_hat") == 0.8
