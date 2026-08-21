from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import gkx.geometry.vmec_boozer_derivatives as vmec_derivatives
import gkx.geometry.backend_discovery as vmec_backend_discovery
import gkx.geometry.imported_vmec as vmec_facade
import gkx.geometry.vmec_field_line_sampling as vmec_fieldline_numerics
from gkx.geometry.imported_vmec import _Struct
from gkx.geometry.imported_vmec import (
    _apply_flux_tube_cut,
    _booz_read_wout_square_layout_failure,
    _booz_xform_jax_search_paths,
    _equal_arc_remap,
    _import_booz_backend,
    _import_module_with_search_paths,
    _vmec_fieldlines,
    _vmec_splines,
    dermv,
    generate_vmec_eik_internal,
    internal_vmec_backend_available,
    nperiod_set,
    write_vmec_eik_netcdf,
)


def test_imported_vmec_reuses_focused_backend_contracts() -> None:
    assert vmec_facade.internal_vmec_backend_available is (
        vmec_backend_discovery.internal_vmec_backend_available
    )
    assert (
        vmec_facade._import_booz_backend is vmec_backend_discovery._import_booz_backend
    )
    assert vmec_facade.nperiod_set is vmec_fieldline_numerics.nperiod_set
    assert vmec_facade.dermv is vmec_fieldline_numerics.dermv
    assert vmec_facade._vmec_splines is vmec_fieldline_numerics._vmec_splines


def test_vmec_struct_accepts_named_fields_and_remains_mutable() -> None:
    geom = _Struct(theta=np.array([0.0]), nfp=5)

    np.testing.assert_allclose(geom.theta, [0.0])
    assert geom.nfp == 5

    geom.iota = 0.41
    assert geom.iota == pytest.approx(0.41)

    with pytest.raises(AttributeError, match="missing"):
        _ = geom.missing


def test_booz_search_paths_include_env_and_src(monkeypatch, tmp_path: Path) -> None:
    checkout = tmp_path / "booz_xform_jax"
    (checkout / "src").mkdir(parents=True)
    monkeypatch.setenv("GKX_BOOZ_XFORM_JAX_PATH", str(checkout))
    paths = _booz_xform_jax_search_paths()
    assert checkout.resolve() in paths
    assert (checkout / "src").resolve() in paths


def test_import_module_with_search_paths_loads_temp_module(tmp_path: Path) -> None:
    pkg = tmp_path / "mods"
    pkg.mkdir()
    (pkg / "demo_mod.py").write_text("VALUE = 7\n", encoding="utf-8")
    sys.modules.pop("demo_mod", None)
    mod = _import_module_with_search_paths("demo_mod", [pkg])
    assert mod.VALUE == 7
    sys.modules.pop("demo_mod", None)


def test_import_module_with_search_paths_replaces_namespace_without_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout_root = tmp_path / "checkout"
    pkg = checkout_root / "booz_xform_jax"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("class Booz_xform:\n    pass\n", encoding="utf-8")
    monkeypatch.setitem(
        sys.modules,
        "booz_xform_jax",
        SimpleNamespace(
            __name__="booz_xform_jax", __path__=[str(tmp_path / "namespace")]
        ),
    )

    mod = _import_module_with_search_paths(
        "booz_xform_jax", [checkout_root], required_attr="Booz_xform"
    )

    assert hasattr(mod, "Booz_xform")
    assert Path(mod.__file__).resolve() == (pkg / "__init__.py").resolve()
    sys.modules.pop("booz_xform_jax", None)


def test_import_module_with_search_paths_raises_on_missing(tmp_path: Path) -> None:
    with pytest.raises(ImportError):
        _import_module_with_search_paths("missing_demo_mod", [tmp_path / "missing"])


def test_import_booz_backend_falls_back_to_booz_xform(monkeypatch) -> None:
    monkeypatch.setattr(
        "gkx.geometry.backend_discovery._booz_xform_jax_search_paths",
        lambda: [],
    )
    monkeypatch.setattr(
        "gkx.geometry.backend_discovery._import_module_with_search_paths",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ImportError("jax backend missing")
        ),
    )

    marker = SimpleNamespace(name="fallback", Booz_xform=object)

    def _import_module(name: str):
        if name == "booz_xform":
            return marker
        raise ImportError(name)

    monkeypatch.setattr(
        "gkx.geometry.backend_discovery.importlib.import_module",
        _import_module,
    )
    assert _import_booz_backend() is marker


def test_import_booz_backend_honors_fortran_override(monkeypatch) -> None:
    marker = SimpleNamespace(name="forced-fortran", Booz_xform=object)
    monkeypatch.setenv("GKX_BOOZ_BACKEND", "fortran")
    monkeypatch.setattr(
        "gkx.geometry.backend_discovery._import_booz_xform_backend",
        lambda: marker,
    )
    monkeypatch.setattr(
        "gkx.geometry.backend_discovery._import_booz_xform_jax_backend",
        lambda: (_ for _ in ()).throw(
            AssertionError("jax backend should not be imported")
        ),
    )

    assert _import_booz_backend() is marker


def test_square_layout_failure_matcher_is_specific() -> None:
    assert _booz_read_wout_square_layout_failure(
        ValueError(
            "rmnc0 has unexpected shape (50, 50); one dimension must equal ns=50"
        )
    )
    assert not _booz_read_wout_square_layout_failure(
        ValueError("rmnc0 has unexpected shape (50, 49)")
    )
    assert not _booz_read_wout_square_layout_failure(
        RuntimeError(
            "rmnc0 has unexpected shape (50, 50); one dimension must equal ns=50"
        )
    )


def test_import_booz_backend_honors_jax_override(monkeypatch) -> None:
    marker = SimpleNamespace(name="forced-jax", Booz_xform=object)
    monkeypatch.setenv("GKX_BOOZ_BACKEND", "jax")
    monkeypatch.setattr(
        "gkx.geometry.backend_discovery._import_booz_xform_jax_backend",
        lambda: marker,
    )
    monkeypatch.setattr(
        "gkx.geometry.backend_discovery._import_booz_xform_backend",
        lambda: (_ for _ in ()).throw(
            AssertionError("booz_xform should not be imported")
        ),
    )

    assert _import_booz_backend() is marker


def test_import_booz_backend_rejects_unknown_override(monkeypatch) -> None:
    monkeypatch.setenv("GKX_BOOZ_BACKEND", "unexpected-backend")

    with pytest.raises(ValueError, match="GKX_BOOZ_BACKEND"):
        _import_booz_backend()


def test_import_booz_backend_reports_missing_backends(monkeypatch) -> None:
    monkeypatch.delenv("GKX_BOOZ_BACKEND", raising=False)
    monkeypatch.setattr(
        "gkx.geometry.backend_discovery._import_booz_xform_jax_backend",
        lambda: (_ for _ in ()).throw(ImportError("jax missing")),
    )
    monkeypatch.setattr(
        "gkx.geometry.backend_discovery._import_booz_xform_backend",
        lambda: (_ for _ in ()).throw(ImportError("booz missing")),
    )

    with pytest.raises(
        ImportError, match="booz_xform_jax/booz_xform backend unavailable"
    ):
        _import_booz_backend()


def test_internal_vmec_backend_available_uses_backend_probe(monkeypatch) -> None:
    monkeypatch.setattr(
        "gkx.geometry.backend_discovery._import_booz_backend",
        lambda: object(),
    )
    assert internal_vmec_backend_available() is True
    monkeypatch.setattr(
        "gkx.geometry.backend_discovery._import_booz_backend",
        lambda: (_ for _ in ()).throw(ImportError("missing")),
    )
    assert internal_vmec_backend_available() is False


def test_nperiod_set_and_dermv(monkeypatch) -> None:
    monkeypatch.setattr(
        "gkx.geometry.vmec_field_line_sampling.nperiod_contract",
        lambda values, theta, npol: (values[1:-1], theta[1:-1]),
    )
    values, theta = nperiod_set(
        np.array([0.0, 1.0, 2.0]), np.array([-2.0, 0.0, 2.0]), 1.0
    )
    np.testing.assert_allclose(values, [1.0])
    np.testing.assert_allclose(theta, [0.0])

    out = dermv(np.array([0.0, 1.0, 4.0, 9.0]), np.array([0.0, 1.0, 2.0, 3.0]))
    np.testing.assert_allclose(out[1:-1], [2.0, 4.0], atol=1.0e-6)

    with pytest.raises(ValueError):
        nperiod_set(np.array([1.0, 2.0]), np.array([1.0]), 1.0)
    with pytest.raises(ValueError):
        dermv(np.ones((2, 2)), np.ones((2, 2)))
    with pytest.raises(ValueError):
        dermv(np.ones(2), np.ones(3))


def test_apply_flux_tube_cut_none_and_unknown() -> None:
    theta = np.linspace(-np.pi, np.pi, 7)
    base = np.linspace(1.0, 2.0, theta.size)
    geo = SimpleNamespace(
        bmag=base[None, None, :],
        gradpar_theta_b=np.abs(base)[None, None, :],
        cvdrift=base[None, None, :],
        gbdrift=base[None, None, :],
        cvdrift0=base[None, None, :],
        gbdrift0=base[None, None, :],
        gds2=(base + 1.0)[None, None, :],
        gds21=np.linspace(-1.0, 1.0, theta.size)[None, None, :],
        gds22=(base + 2.0)[None, None, :],
        grho=base[None, None, :],
        R_b=(base + 3.0)[None, None, :],
        Z_b=(base - 1.0)[None, None, :],
        grad_x=np.stack([base, base + 1.0, base + 2.0])[:, None, None, :],
        grad_y=np.stack([base + 3.0, base + 4.0, base + 5.0])[:, None, None, :],
        s_hat_input=0.8,
    )

    theta_cut, arrays = _apply_flux_tube_cut(
        theta,
        geo,
        ntheta=theta.size,
        flux_tube_cut="none",
        npol_min=None,
        which_crossing=0,
        y0=1.0,
        x0=1.0,
        jtwist_in=None,
    )
    np.testing.assert_allclose(theta_cut, theta)
    assert arrays["grad_x"].shape == (3, theta.size)
    assert arrays["b_vec"].shape == (3, theta.size)

    with pytest.raises(ValueError):
        _apply_flux_tube_cut(
            theta,
            geo,
            ntheta=theta.size,
            flux_tube_cut="bad",
            npol_min=None,
            which_crossing=0,
            y0=1.0,
            x0=1.0,
            jtwist_in=None,
        )


def test_equal_arc_remap_returns_constant_gradpar() -> None:
    theta = np.linspace(-np.pi, np.pi, 7)
    arrays = {
        "theta_PEST": theta.copy(),
        "bmag": np.linspace(1.0, 2.0, theta.size),
        "gradpar": np.full(theta.size, 2.0),
        "cvdrift": np.linspace(0.0, 1.0, theta.size),
        "gbdrift": np.linspace(1.0, 0.0, theta.size),
        "cvdrift0": np.linspace(0.5, 1.5, theta.size),
        "gbdrift0": np.linspace(1.5, 0.5, theta.size),
        "gds2": np.linspace(2.0, 3.0, theta.size),
        "gds21": np.linspace(-0.5, 0.5, theta.size),
        "gds22": np.linspace(3.0, 4.0, theta.size),
        "grho": np.linspace(0.8, 1.2, theta.size),
        "Rplot": np.linspace(5.0, 6.0, theta.size),
        "Zplot": np.linspace(-1.0, 1.0, theta.size),
        "grad_x": np.vstack(
            [np.ones(theta.size), 2.0 * np.ones(theta.size), 3.0 * np.ones(theta.size)]
        ),
        "grad_y": np.vstack(
            [
                4.0 * np.ones(theta.size),
                5.0 * np.ones(theta.size),
                6.0 * np.ones(theta.size),
            ]
        ),
    }

    gradpar_eqarc, out = _equal_arc_remap(theta, arrays, ntheta=9)
    assert np.isfinite(gradpar_eqarc)
    np.testing.assert_allclose(out["gradpar"], np.full(9, gradpar_eqarc))
    assert out["theta"].shape == (9,)
    assert out["grad_x"].shape == (3, 9)
    assert out["b_vec"].shape == (3, 9)
    assert np.isfinite(out["scale"])


def _mock_geo_for_cut(
    theta: np.ndarray,
    *,
    gds21: np.ndarray | None = None,
    gbdrift0: np.ndarray | None = None,
) -> SimpleNamespace:
    base = np.linspace(1.0, 2.0, theta.size)
    gds21_arr = gds21 if gds21 is not None else np.linspace(-1.0, 1.0, theta.size)
    gbdrift0_arr = (
        gbdrift0 if gbdrift0 is not None else np.linspace(-1.0, 1.0, theta.size)
    )
    return SimpleNamespace(
        bmag=base[None, None, :],
        gradpar_theta_b=np.abs(base)[None, None, :],
        cvdrift=base[None, None, :],
        gbdrift=base[None, None, :],
        cvdrift0=base[None, None, :],
        gbdrift0=gbdrift0_arr[None, None, :],
        gds2=(base + 1.0)[None, None, :],
        gds21=gds21_arr[None, None, :],
        gds22=np.ones(theta.size)[None, None, :],
        grho=base[None, None, :],
        R_b=(base + 3.0)[None, None, :],
        Z_b=(base - 1.0)[None, None, :],
        grad_x=np.stack([base, base + 1.0, base + 2.0])[:, None, None, :],
        grad_y=np.stack([base + 3.0, base + 4.0, base + 5.0])[:, None, None, :],
        s_hat_input=0.5,
    )


class _ScalarWithData:
    def __init__(self, value: float | int) -> None:
        self.data = np.array(value)


class _FakeVar:
    def __init__(self, data: object, *, with_data: bool = False) -> None:
        self._data = np.asarray(data)
        self._with_data = with_data

    def __getitem__(self, item: object) -> object:
        if self._data.ndim == 0:
            out = self._data
        else:
            out = self._data[item]
        if self._with_data:
            return _ScalarWithData(out)
        return out


class _FakeNCScalarVar:
    def __init__(self, value: float | int) -> None:
        self.value = value

    def __getitem__(self, _item: object) -> np.ndarray:
        return np.array(self.value)


class _FakeNCDataset:
    def __init__(self, mpol: int = 2, ntor: int = 1) -> None:
        self.variables = {
            "mpol": _FakeNCScalarVar(mpol),
            "ntor": _FakeNCScalarVar(ntor),
        }
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeBoozXform:
    def __init__(self) -> None:
        self.verbose = 0
        self.mboz: int | None = None
        self.nboz: int | None = None
        self.read_path: str | None = None
        self.ran = False

    def read_wout(self, path: str) -> None:
        self.read_path = path

    def run(self) -> None:
        self.ran = True


class _SquareLayoutFailingBoozXform(_FakeBoozXform):
    def read_wout(self, path: str) -> None:
        self.read_path = path
        raise ValueError(
            "rmnc0 has unexpected shape (50, 50); one dimension must equal ns=50"
        )


def _const_callable(value: float):
    def _inner(s: object, _value: float = value) -> np.ndarray:
        return np.full_like(np.asarray(s, dtype=float), _value, dtype=float)

    return _inner


def _fake_vmec_spline_struct() -> SimpleNamespace:
    return SimpleNamespace(
        d_pressure_d_s=_const_callable(0.2),
        iota=_const_callable(0.8),
        d_iota_d_s=_const_callable(0.1),
        Gfun=_const_callable(1.5),
        Ifun=_const_callable(0.3),
        phiedge=-2.0 * np.pi,
        Aminor_p=1.2,
        nfp=5,
        raxis_cc=np.array([3.0]),
        xm_b=np.array([0.0, 1.0]),
        xn_b=np.array([0.0, 0.0]),
        mnbooz=2,
        mboz=4,
        nboz=2,
        rmnc_b=[_const_callable(4.0), _const_callable(0.2)],
        zmns_b=[_const_callable(0.0), _const_callable(0.1)],
        numns_b=[_const_callable(0.0), _const_callable(0.05)],
        d_rmnc_b_d_s=[_const_callable(0.0), _const_callable(0.0)],
        d_zmns_b_d_s=[_const_callable(0.0), _const_callable(0.0)],
        d_numns_b_d_s=[_const_callable(0.0), _const_callable(0.0)],
        gmnc_b=[_const_callable(1.0), _const_callable(0.0)],
        bmnc_b=[_const_callable(1.3), _const_callable(0.1)],
        d_bmnc_b_d_s=[_const_callable(0.0), _const_callable(0.0)],
    )


@pytest.mark.parametrize(
    ("flux_tube_cut", "geo", "expected_cut", "jtwist_in"),
    [
        (
            "gds21",
            _mock_geo_for_cut(
                np.linspace(-2.0, 2.0, 9), gds21=np.linspace(-2.0, 2.0, 9) - 0.5
            ),
            0.5,
            None,
        ),
        (
            "gbdrift0",
            _mock_geo_for_cut(
                np.linspace(-2.0, 2.0, 9), gbdrift0=np.linspace(-2.0, 2.0, 9) - 0.75
            ),
            0.75,
            None,
        ),
        (
            "aspect",
            _mock_geo_for_cut(
                np.linspace(-2.0, 2.0, 9), gds21=np.linspace(-2.0, 2.0, 9)
            ),
            1.0,
            1,
        ),
    ],
)
def test_apply_flux_tube_cut_branch_specific_roots(
    flux_tube_cut: str,
    geo: SimpleNamespace,
    expected_cut: float,
    jtwist_in: int | None,
) -> None:
    theta = np.linspace(-2.0, 2.0, 9)
    theta_cut, arrays = _apply_flux_tube_cut(
        theta,
        geo,
        ntheta=11,
        flux_tube_cut=flux_tube_cut,
        npol_min=None,
        which_crossing=0,
        y0=1.0,
        x0=1.0,
        jtwist_in=jtwist_in,
    )

    assert theta_cut[0] == pytest.approx(-expected_cut, abs=1.0e-6)
    assert theta_cut[-1] == pytest.approx(expected_cut, abs=1.0e-6)
    assert arrays["theta"].shape == (11,)
    assert arrays["grad_x"].shape == (3, 11)


def test_apply_flux_tube_cut_reports_missing_crossings() -> None:
    theta = np.linspace(-2.0, 2.0, 9)
    geo = _mock_geo_for_cut(theta, gds21=np.ones_like(theta))

    with pytest.raises(ValueError, match="No positive gds21 flux-tube crossing"):
        _apply_flux_tube_cut(
            theta,
            geo,
            ntheta=11,
            flux_tube_cut="gds21",
            npol_min=None,
            which_crossing=0,
            y0=1.0,
            x0=1.0,
            jtwist_in=None,
        )


def test_apply_flux_tube_cut_reports_out_of_range_crossing() -> None:
    theta = np.linspace(-2.0, 2.0, 9)
    geo = _mock_geo_for_cut(theta, gds21=theta - 0.5)

    with pytest.raises(ValueError, match="which_crossing=2"):
        _apply_flux_tube_cut(
            theta,
            geo,
            ntheta=11,
            flux_tube_cut="gds21",
            npol_min=None,
            which_crossing=2,
            y0=1.0,
            x0=1.0,
            jtwist_in=None,
        )


def test_write_vmec_eik_netcdf_writes_expected_variables(tmp_path: Path) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    path = tmp_path / "geom.eik.nc"
    theta = np.linspace(-np.pi, np.pi, 5)
    bmag = np.linspace(1.0, 2.0, theta.size)
    gradpar = np.full(theta.size, 0.5)
    profiles = {
        "theta": theta,
        "theta_PEST": theta + 0.1,
        "bmag": bmag,
        "gradpar": gradpar,
        "grho": np.full(theta.size, 1.2),
        "gds2": np.linspace(2.0, 3.0, theta.size),
        "gds21": np.linspace(-0.5, 0.5, theta.size),
        "gds22": np.linspace(3.0, 4.0, theta.size),
        "gbdrift": np.linspace(0.1, 0.2, theta.size),
        "gbdrift0": np.linspace(0.2, 0.3, theta.size),
        "cvdrift": np.linspace(0.3, 0.4, theta.size),
        "cvdrift0": np.linspace(0.4, 0.5, theta.size),
        "Rplot": np.linspace(5.0, 6.0, theta.size),
        "Zplot": np.linspace(-1.0, 1.0, theta.size),
        "grad_x": np.vstack(
            [np.ones(theta.size), 2.0 * np.ones(theta.size), 3.0 * np.ones(theta.size)]
        ),
        "grad_y": np.vstack(
            [
                4.0 * np.ones(theta.size),
                5.0 * np.ones(theta.size),
                6.0 * np.ones(theta.size),
            ]
        ),
        "b_vec": np.vstack(
            [np.ones(theta.size), np.zeros(theta.size), np.zeros(theta.size)]
        ),
        "dpsidrho": 2.0,
        "kxfac": 1.7,
        "Rmaj": 5.5,
        "q": 1.4,
        "shat": 0.8,
        "scale": 2.0,
        "alpha": 0.25,
        "zeta_center": 0.125,
        "nfp": 5,
    }

    write_vmec_eik_netcdf(path, profiles, request=SimpleNamespace())

    with netcdf4.Dataset(path) as ds:
        np.testing.assert_allclose(ds.variables["theta"][:], theta)
        np.testing.assert_allclose(ds.variables["theta_PEST"][:], theta + 0.1)
        np.testing.assert_allclose(ds.variables["bmag"][:], bmag)
        np.testing.assert_allclose(ds.variables["gradpar"][:], gradpar)
        np.testing.assert_allclose(ds.variables["grad_x"][:, :], profiles["grad_x"])
        assert float(ds.variables["kxfac"].getValue()) == pytest.approx(1.7)
        assert float(ds.variables["Rmaj"].getValue()) == pytest.approx(5.5)
        assert int(ds.variables["nfp"].getValue()) == 5
        expected_jacob = 1.0 / abs(
            (1.0 / abs(profiles["dpsidrho"])) * gradpar[0] * bmag
        )
        np.testing.assert_allclose(ds.variables["jacob"][:], expected_jacob)


def test_vmec_splines_builds_interpolants_and_metadata() -> None:
    s_half = np.array([0.125, 0.375, 0.625, 0.875])
    booz_obj = SimpleNamespace(
        mnboz=2,
        rmnc_b=np.vstack([1.0 + s_half, 2.0 * s_half]),
        zmns_b=np.vstack([0.5 * s_half, -s_half]),
        numns_b=np.vstack([0.2 * s_half, 0.3 + 0.1 * s_half]),
        gmnc_b=np.vstack([1.0 + 0.5 * s_half, 0.1 * s_half]),
        bmnc_b=np.vstack([2.0 + 2.0 * s_half, 0.5 - 0.25 * s_half]),
        Boozer_G=5.0 + 0.2 * s_half,
        Boozer_I=1.0 - 0.1 * s_half,
        xm_b=np.array([0, 1]),
        xn_b=np.array([0, 5]),
        mboz=8,
        nboz=6,
    )
    nc_obj = SimpleNamespace(
        variables={
            "ns": _FakeVar(5, with_data=True),
            "pres": _FakeVar([0.0, 1.0, 2.0, 3.0, 4.0]),
            "phi": _FakeVar([0.0, 2.0 * np.pi, 4.0 * np.pi, 6.0 * np.pi, 8.0 * np.pi]),
            "iotas": _FakeVar([0.0, 0.5, 0.6, 0.7, 0.8]),
            "Aminor_p": _FakeVar(1.7),
            "nfp": _FakeVar(5),
            "raxis_cc": _FakeVar([3.2, 0.1]),
        }
    )

    out = _vmec_splines(nc_obj, booz_obj)

    assert out.mnbooz == 2
    assert out.mboz == 8
    assert out.nboz == 6
    assert out.nfp == 5
    np.testing.assert_allclose(out.raxis_cc, [3.2, 0.1])
    assert out.Aminor_p == pytest.approx(1.7)
    assert out.phiedge == pytest.approx(8.0 * np.pi)
    assert out.rmnc_b[0](0.5) == pytest.approx(1.5, abs=1.0e-10)
    assert out.d_rmnc_b_d_s[0](0.5) == pytest.approx(1.0, abs=1.0e-10)
    assert out.bmnc_b[1](0.5) == pytest.approx(0.375, abs=1.0e-10)
    assert out.d_bmnc_b_d_s[1](0.5) == pytest.approx(-0.25, abs=1.0e-10)
    assert out.psi(0.5) == pytest.approx(2.5, abs=1.0e-10)
    assert out.d_psi_d_s(0.5) == pytest.approx(4.0, abs=1.0e-10)
    assert out.iota(0.5) == pytest.approx(0.65, abs=1.0e-10)
    assert out.d_iota_d_s(0.5) == pytest.approx(0.4, abs=1.0e-10)


def test_vmec_fieldline_helper_angle_and_denominator_policies() -> None:
    theta = np.array([[[0.0, 1.0]]])
    phi = np.array([[[0.25, 0.25]]])
    xm = np.array([1.0, 2.0])
    xn = np.array([0.0, 1.0])

    angle = vmec_fieldline_numerics._boozer_mode_angle(xm, xn, theta, phi, flipit=False)
    np.testing.assert_allclose(angle[0, 0, 0], theta[0, 0])
    np.testing.assert_allclose(angle[1, 0, 0], 2.0 * theta[0, 0] - 0.25)

    flipped = vmec_fieldline_numerics._boozer_mode_angle(
        xm, xn, theta, phi, flipit=True
    )
    np.testing.assert_allclose(flipped[0] - angle[0], np.pi)
    np.testing.assert_allclose(flipped[1] - angle[1], 2.0 * np.pi)

    safe = vmec_derivatives._safe_mode_denominator(
        np.array([0.0, 2.0, 3.0]), np.array([0.0, 1.0, 1.0]), np.array([0.5])
    )
    assert safe.shape == (1, 2)
    assert safe[0, 0] == pytest.approx(1.0e-30)
    assert safe[0, 1] == pytest.approx(0.5)


def test_vmec_fieldline_boozer_mode_sum_preserves_surface_axis() -> None:
    coeff = np.array([[1.0, 2.0], [3.0, 4.0]])
    basis = np.arange(2 * 2 * 3 * 4, dtype=float).reshape(2, 2, 3, 4)

    out = vmec_fieldline_numerics._boozer_mode_sum(coeff, basis)

    assert out.shape == (2, 3, 4)
    np.testing.assert_allclose(
        out[0], coeff[0, 0] * basis[0, 0] + coeff[0, 1] * basis[1, 0]
    )
    np.testing.assert_allclose(
        out[1], coeff[1, 0] * basis[0, 1] + coeff[1, 1] * basis[1, 1]
    )


def test_vmec_fieldline_boozer_trig_basis_preserves_mode_axis() -> None:
    xm = np.array([0.0, 1.0, 2.0])
    xn = np.array([0.0, -1.0, 3.0])
    angle = np.linspace(0.0, 0.5, 3 * 2 * 4).reshape(3, 2, 4)

    cosangle, sinangle, mcos, msin, ncos, nsin = (
        vmec_fieldline_numerics._boozer_trig_basis(xm, xn, angle)
    )

    assert cosangle.shape == angle.shape
    assert sinangle.shape == angle.shape
    np.testing.assert_allclose(cosangle, np.cos(angle))
    np.testing.assert_allclose(sinangle, np.sin(angle))
    np.testing.assert_allclose(mcos[2], 2.0 * np.cos(angle[2]))
    np.testing.assert_allclose(msin[1], np.sin(angle[1]))
    np.testing.assert_allclose(ncos[1], -np.cos(angle[1]))
    np.testing.assert_allclose(nsin[2], 3.0 * np.sin(angle[2]))


def test_vmec_fieldline_tensor_helpers_match_circular_surface() -> None:
    theta = np.linspace(-np.pi, np.pi, 9)[None, None, :]
    phi = theta.copy()
    xm = np.array([0.0, 1.0])
    xn = np.array([0.0, 0.0])
    angle = vmec_fieldline_numerics._boozer_mode_angle(xm, xn, theta, phi, flipit=False)
    cosangle, sinangle, mcos, msin, ncos, nsin = (
        vmec_fieldline_numerics._boozer_trig_basis(xm, xn, angle)
    )
    r0, r1, z1 = 2.0, 0.2, 0.3

    tensors = vmec_fieldline_numerics._fieldline_boozer_tensors(
        rmnc_b=np.array([[r0, r1]]),
        zmns_b=np.array([[0.0, z1]]),
        numns_b=np.array([[0.0, 0.0]]),
        d_rmnc_b_d_s=np.array([[0.05, 0.01]]),
        d_zmns_b_d_s=np.array([[0.0, 0.02]]),
        d_numns_b_d_s=np.array([[0.0, 0.0]]),
        gmnc_b=np.array([[1.0, 0.0]]),
        bmnc_b=np.array([[1.0, 0.1]]),
        d_bmnc_b_d_s=np.array([[0.01, 0.02]]),
        cosangle_b=cosangle,
        sinangle_b=sinangle,
        mcosangle_b=mcos,
        msinangle_b=msin,
        ncosangle_b=ncos,
        nsinangle_b=nsin,
    )

    expected_r = r0 + r1 * np.cos(theta[0, 0])
    np.testing.assert_allclose(tensors.R_b[0, 0], expected_r)
    np.testing.assert_allclose(tensors.d_R_b_d_theta_b[0, 0], -r1 * np.sin(theta[0, 0]))
    np.testing.assert_allclose(tensors.d_Z_b_d_theta_b[0, 0], z1 * np.cos(theta[0, 0]))
    np.testing.assert_allclose(
        tensors.d_B_b_d_s[0, 0], 0.01 + 0.02 * np.cos(theta[0, 0])
    )

    cartesian = vmec_fieldline_numerics._fieldline_cartesian_derivatives(
        tensors=tensors, phi_b=phi
    )
    np.testing.assert_allclose(
        cartesian.d_X_d_phi_b[0, 0],
        -expected_r * np.sin(theta[0, 0]),
    )
    gradients = vmec_fieldline_numerics._fieldline_coordinate_gradients(
        tensors=tensors,
        cartesian=cartesian,
        edge_toroidal_flux_over_2pi=2.0,
    )
    np.testing.assert_allclose(
        gradients.grad_psi_Z,
        cartesian.d_X_d_theta_b * cartesian.d_Y_d_phi_b
        - cartesian.d_Y_d_theta_b * cartesian.d_X_d_phi_b,
    )


def test_vmec_fieldline_alpha_gradient_and_local_shear_helpers() -> None:
    shape = (1, 1, 3)
    gradients = vmec_fieldline_numerics._FieldlineCoordinateGradients(
        grad_psi_X=np.ones(shape),
        grad_psi_Y=2.0 * np.ones(shape),
        grad_psi_Z=3.0 * np.ones(shape),
        grad_theta_b_X=0.5 * np.ones(shape),
        grad_theta_b_Y=0.25 * np.ones(shape),
        grad_theta_b_Z=-0.75 * np.ones(shape),
        grad_phi_b_X=0.1 * np.ones(shape),
        grad_phi_b_Y=0.2 * np.ones(shape),
        grad_phi_b_Z=0.3 * np.ones(shape),
    )
    phi_b = np.array([[[0.0, 0.5, 1.0]]])
    zeta_center = 0.1
    d_iota_d_s = np.array([0.2])
    iota = np.array([0.4])
    etf = 2.0

    alpha = vmec_fieldline_numerics._fieldline_alpha_gradients(
        gradients=gradients,
        phi_b=phi_b,
        zeta_center=zeta_center,
        d_iota_d_s=d_iota_d_s,
        iota=iota,
        edge_toroidal_flux_over_2pi=etf,
    )

    radial_shear = -(phi_b - zeta_center) * d_iota_d_s[:, None, None] / etf
    expected_x = radial_shear * gradients.grad_psi_X + 0.5 - iota[:, None, None] * 0.1
    expected_y = radial_shear * gradients.grad_psi_Y + 0.25 - iota[:, None, None] * 0.2
    expected_z = radial_shear * gradients.grad_psi_Z - 0.75 - iota[:, None, None] * 0.3
    expected_g = 1.0**2 + 2.0**2 + 3.0**2
    expected_dot = expected_x + 2.0 * expected_y + 3.0 * expected_z

    np.testing.assert_allclose(alpha.grad_alpha_X, expected_x)
    np.testing.assert_allclose(alpha.grad_alpha_Y, expected_y)
    np.testing.assert_allclose(alpha.grad_alpha_Z, expected_z)
    np.testing.assert_allclose(alpha.g_sup_psi_psi, expected_g)
    np.testing.assert_allclose(alpha.grad_alpha_dot_grad_psi, expected_dot)

    d_iota_d_s_1 = np.array([0.3])
    d_pressure_d_s_1 = np.array([0.2])
    Vprime = np.array([1.2])
    G = np.array([2.0])
    boozer_i = np.array([0.1])
    intinv_g = np.array([[[0.1, 0.2, 0.3]]])
    int_lam_div_g = np.array([[[0.05, 0.07, 0.11]]])
    D1 = 1.7
    D2 = 0.4
    shear = vmec_fieldline_numerics._fieldline_local_shear(
        edge_toroidal_flux_over_2pi=etf,
        d_iota_d_s=d_iota_d_s,
        d_iota_d_s_1=d_iota_d_s_1,
        d_pressure_d_s_1=d_pressure_d_s_1,
        Vprime=Vprime,
        G=G,
        iota=iota,
        boozer_i=boozer_i,
        phi_b=phi_b,
        zeta_center=zeta_center,
        intinv_g=intinv_g,
        int_lam_div_g=int_lam_div_g,
        D1=D1,
        D2=D2,
        g_sup_psi_psi=alpha.g_sup_psi_psi,
        grad_alpha_dot_grad_psi=alpha.grad_alpha_dot_grad_psi,
    )
    expected_D = (
        d_iota_d_s_1[:, None, None] * (intinv_g / D1 - phi_b + zeta_center)
        - d_pressure_d_s_1[:, None, None]
        * Vprime[:, None, None]
        * (G[:, None, None] + iota[:, None, None] * boozer_i[:, None, None])
        * (int_lam_div_g - D2 * intinv_g / D1)
    ) / etf
    expected_L0 = -(
        expected_dot / expected_g
        + d_iota_d_s[:, None, None] * (phi_b - zeta_center) / etf
    )
    expected_L1 = (
        -d_iota_d_s_1[:, None, None] * (phi_b - zeta_center) / etf
        + expected_dot / expected_g
        - expected_D
    )

    np.testing.assert_allclose(shear.D_HNGC, expected_D)
    np.testing.assert_allclose(shear.L0, expected_L0)
    np.testing.assert_allclose(shear.L1, expected_L1)


def test_vmec_fieldline_helper_coordinates_and_axisym_flip_policy() -> None:
    theta1d = np.array([0.0, 1.0])
    alpha_arr = np.array([0.0, 0.5])
    iota = np.array([0.5, 1.0])

    theta_b, phi_b = vmec_derivatives._fieldline_boozer_coordinates(
        theta1d, alpha_arr, iota
    )

    assert theta_b.shape == (2, 2, 2)
    np.testing.assert_allclose(theta_b[1, 0], theta1d)
    np.testing.assert_allclose(phi_b[0, 1], (theta1d - 0.5) / 0.5)
    np.testing.assert_allclose(
        theta_b - iota[:, None, None] * phi_b,
        np.broadcast_to(alpha_arr[None, :, None], theta_b.shape),
    )

    xm = np.array([1.0])
    xn = np.array([0.0])
    rmnc_b = np.array([[1.0]])
    zmns_b = np.array([[0.25]])

    assert not vmec_derivatives._axisym_flip_required(
        isaxisym=False,
        xm_b=xm,
        xn_b=xn,
        theta_b=theta_b[:1, :1],
        phi_b=phi_b[:1, :1],
        rmnc_b=rmnc_b,
        zmns_b=zmns_b,
    )
    assert vmec_derivatives._axisym_flip_required(
        isaxisym=True,
        xm_b=xm,
        xn_b=xn,
        theta_b=theta_b[:1, :1],
        phi_b=phi_b[:1, :1],
        rmnc_b=rmnc_b,
        zmns_b=zmns_b,
    )


def test_vmec_fieldline_helper_surface_average_and_centered_integral() -> None:
    theta_grid = np.linspace(-np.pi, np.pi, 21)
    phi_grid = np.linspace(-np.pi, np.pi, 19)
    constant = np.full((phi_grid.size, theta_grid.size), 2.5)

    assert vmec_derivatives._surface_average_2d(
        constant, theta_grid, phi_grid
    ) == pytest.approx(2.5)

    theta = np.linspace(-np.pi, np.pi, 17)
    fieldline = theta[None, None, :]
    centered = vmec_derivatives._centered_fieldline_integral(
        np.ones_like(fieldline), fieldline, theta
    )
    np.testing.assert_allclose(centered[0, 0], theta, atol=1.0e-12)


def test_vmec_fieldline_reference_scale_and_override_policies() -> None:
    vs = SimpleNamespace(Aminor_p=2.0, raxis_cc=np.array([3.5]))

    L_reference, B_reference, R_mag_ax = vmec_derivatives._validated_reference_scales(
        vs, edge_toroidal_flux_over_2pi=-4.0
    )

    assert L_reference == pytest.approx(2.0)
    assert B_reference == pytest.approx(2.0)
    assert R_mag_ax == pytest.approx(3.5)

    with pytest.raises(ValueError, match="positive finite minor radius"):
        vmec_derivatives._validated_reference_scales(
            SimpleNamespace(Aminor_p=0.0, raxis_cc=np.array([3.5])),
            edge_toroidal_flux_over_2pi=-4.0,
        )

    assert vmec_derivatives._input_iota_shear(
        np.array([0.62]), np.array([0.0]), None, None
    ) == (pytest.approx(0.62), pytest.approx(1.0e-8))
    assert vmec_derivatives._input_iota_shear(
        np.array([0.62]), np.array([0.2]), 0.7, 0.3
    ) == (pytest.approx(0.7), pytest.approx(0.3))


def test_vmec_fieldline_hngc_shear_and_pressure_correction_policies() -> None:
    d_iota_d_s_1, sfac = vmec_derivatives._hngc_shear_correction(
        s_val=0.5,
        iota=np.array([0.6]),
        shat=np.array([0.2]),
        iota_input_val=0.7,
        s_hat_input_val=0.4,
        include_shear_variation=True,
    )

    np.testing.assert_allclose(d_iota_d_s_1, np.array([-0.16]))
    assert sfac == pytest.approx(0.5)

    disabled_shear, disabled_sfac = vmec_derivatives._hngc_shear_correction(
        s_val=0.5,
        iota=np.array([0.6]),
        shat=np.array([0.2]),
        iota_input_val=0.7,
        s_hat_input_val=0.4,
        include_shear_variation=False,
    )

    np.testing.assert_allclose(disabled_shear, np.zeros(1))
    assert disabled_sfac == pytest.approx(1.0)

    d_pressure_d_s_1, pfac = vmec_derivatives._hngc_pressure_correction(
        s_val=0.25,
        betaprim=0.2,
        B_reference=2.0,
        d_pressure_d_s=np.array([0.5]),
        include_pressure_variation=True,
    )

    drive = 0.2 * 2.0**2 / (4.0 * np.sqrt(0.25))
    np.testing.assert_allclose(
        d_pressure_d_s_1,
        np.array([drive - vmec_fieldline_numerics._MU_0 * 0.5]),
    )
    assert pfac == pytest.approx(drive / (vmec_fieldline_numerics._MU_0 * 0.5))

    floored_pressure, floored_pfac = vmec_derivatives._hngc_pressure_correction(
        s_val=0.25,
        betaprim=0.2,
        B_reference=2.0,
        d_pressure_d_s=np.array([0.0]),
        include_pressure_variation=True,
    )

    np.testing.assert_allclose(floored_pressure, np.array([drive]))
    assert floored_pfac == pytest.approx(
        drive / (vmec_fieldline_numerics._MU_0 * 1.0e-8)
    )

    disabled_pressure, disabled_pfac = vmec_derivatives._hngc_pressure_correction(
        s_val=0.25,
        betaprim=0.2,
        B_reference=2.0,
        d_pressure_d_s=np.array([0.5]),
        include_pressure_variation=False,
    )

    np.testing.assert_allclose(disabled_pressure, np.zeros(1))
    assert disabled_pfac == pytest.approx(1.0)


def test_vmec_fieldline_helper_flux_surface_hngc_averages() -> None:
    d1, d2 = vmec_derivatives._flux_surface_hngc_averages(
        xm_b=np.array([0.0, 1.0]),
        xn_b=np.array([0.0, 0.0]),
        flipit=False,
        lambmnc_b=np.array([[0.0, 0.0]]),
        rmnc_b=np.array([[3.0, 0.4]]),
        zmns_b=np.array([[0.0, 0.4]]),
        numns_b=np.array([[0.0, 0.0]]),
        gmnc_b=np.array([[1.0, 0.0]]),
        res_theta=31,
        res_phi=29,
    )

    assert np.isfinite(d1)
    assert d1 > 0.0
    assert d2 == pytest.approx(0.0, abs=1.0e-14)


def test_vmec_fieldline_helper_samples_boozer_mode_table() -> None:
    s = np.array([0.25, 0.75])

    def _family(scale: float) -> list:
        return [lambda x, j=j: scale * (j + np.asarray(x)) for j in range(3)]

    vs = SimpleNamespace(
        mnbooz=3,
        rmnc_b=_family(1.0),
        zmns_b=_family(2.0),
        numns_b=_family(3.0),
        d_rmnc_b_d_s=_family(4.0),
        d_zmns_b_d_s=_family(5.0),
        d_numns_b_d_s=_family(6.0),
        gmnc_b=_family(7.0),
        bmnc_b=_family(8.0),
        d_bmnc_b_d_s=_family(9.0),
    )

    rmnc_b, zmns_b, *_, d_bmnc_b_d_s = (
        vmec_fieldline_numerics._sample_boozer_mode_table(vs, s, ns=2)
    )

    assert rmnc_b.shape == (2, 3)
    np.testing.assert_allclose(rmnc_b[:, 2], 2.0 + s)
    np.testing.assert_allclose(zmns_b[:, 1], 2.0 * (1.0 + s))
    np.testing.assert_allclose(d_bmnc_b_d_s[:, 0], 9.0 * s)


def test_vmec_fieldlines_respects_overrides_and_closes_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_nc = _FakeNCDataset(mpol=3, ntor=2)
    fake_backend = SimpleNamespace(Booz_xform=_FakeBoozXform)

    monkeypatch.setitem(
        sys.modules,
        "netCDF4",
        SimpleNamespace(Dataset=lambda *_args, **_kwargs: fake_nc),
    )
    monkeypatch.setattr(
        "gkx.geometry.vmec_state_controls._import_booz_backend",
        lambda: fake_backend,
    )
    monkeypatch.setattr(
        "gkx.geometry.vmec_state_controls._vmec_splines",
        lambda _nc, _booz: _fake_vmec_spline_struct(),
    )

    out = _vmec_fieldlines(
        vmec_fname="dummy.nc",
        s_val=0.5,
        betaprim=0.01,
        alpha=0.2,
        include_shear_variation=False,
        include_pressure_variation=False,
        theta1d=np.linspace(-np.pi, np.pi, 9),
        isaxisym=True,
        iota_input=0.9,
        s_hat_input=0.0,
        res_theta=21,
        res_phi=21,
    )

    assert fake_nc.closed is True
    assert out.iota_input == pytest.approx(0.9)
    assert out.s_hat_input == pytest.approx(1.0e-8)
    assert out.zeta_center == pytest.approx(-0.2 / 0.8)
    assert out.nfp == 5
    assert out.L_reference == pytest.approx(1.2)
    assert out.B_reference == pytest.approx(2.0 / (1.2**2))
    assert out.dpsidrho == pytest.approx(np.sqrt(2.0))
    assert out.theta_b.shape == (1, 1, 9)
    assert out.theta_PEST.shape == (1, 1, 9)
    assert out.grad_x.shape == (3, 1, 1, 9)
    assert out.grad_y.shape == (3, 1, 1, 9)
    assert np.isfinite(out.bmag).all()
    assert np.isfinite(out.gds2).all()
    assert np.isfinite(out.gbdrift).all()


def test_vmec_fieldlines_falls_back_for_square_vmex_wout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_nc = _FakeNCDataset(mpol=3, ntor=2)
    jax_backend = SimpleNamespace(Booz_xform=_SquareLayoutFailingBoozXform)
    fortran_backend = SimpleNamespace(Booz_xform=_FakeBoozXform)
    calls: dict[str, object] = {}

    def _fake_import_backend(preferred: str | None = None) -> object:
        if preferred == "booz_xform":
            return fortran_backend
        return jax_backend

    def _fake_splines(_nc: object, booz_obj: object) -> SimpleNamespace:
        calls["booz_obj"] = booz_obj
        return _fake_vmec_spline_struct()

    monkeypatch.delenv("GKX_BOOZ_BACKEND", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "netCDF4",
        SimpleNamespace(Dataset=lambda *_args, **_kwargs: fake_nc),
    )
    monkeypatch.setattr(
        "gkx.geometry.vmec_state_controls._import_booz_backend",
        _fake_import_backend,
    )
    monkeypatch.setattr(
        "gkx.geometry.vmec_state_controls._vmec_splines", _fake_splines
    )

    out = _vmec_fieldlines(
        vmec_fname="square-vmec-jax.nc",
        s_val=0.5,
        betaprim=0.01,
        alpha=0.2,
        include_shear_variation=False,
        include_pressure_variation=False,
        theta1d=np.linspace(-np.pi, np.pi, 9),
        isaxisym=True,
        iota_input=0.9,
        s_hat_input=0.0,
        res_theta=21,
        res_phi=21,
    )

    assert fake_nc.closed is True
    assert isinstance(calls["booz_obj"], _FakeBoozXform)
    assert not isinstance(calls["booz_obj"], _SquareLayoutFailingBoozXform)
    assert out.iota_input == pytest.approx(0.9)


def test_vmec_fieldlines_does_not_fallback_when_booz_backend_is_forced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_nc = _FakeNCDataset(mpol=3, ntor=2)
    jax_backend = SimpleNamespace(Booz_xform=_SquareLayoutFailingBoozXform)

    def _fake_import_backend(preferred: str | None = None) -> object:
        assert preferred is None
        return jax_backend

    monkeypatch.setenv("GKX_BOOZ_BACKEND", "jax")
    monkeypatch.setitem(
        sys.modules,
        "netCDF4",
        SimpleNamespace(Dataset=lambda *_args, **_kwargs: fake_nc),
    )
    monkeypatch.setattr(
        "gkx.geometry.vmec_state_controls._import_booz_backend",
        _fake_import_backend,
    )

    with pytest.raises(ValueError, match="rmnc0 has unexpected shape"):
        _vmec_fieldlines(
            vmec_fname="square-vmec-jax.nc",
            s_val=0.5,
            betaprim=0.01,
            alpha=0.2,
            include_shear_variation=False,
            include_pressure_variation=False,
            theta1d=np.linspace(-np.pi, np.pi, 9),
            isaxisym=True,
            iota_input=0.9,
            s_hat_input=0.0,
            res_theta=21,
            res_phi=21,
        )
    assert fake_nc.closed is True


def test_vmec_fieldlines_rejects_degenerate_reference_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_nc = _FakeNCDataset(mpol=3, ntor=2)
    fake_backend = SimpleNamespace(Booz_xform=_FakeBoozXform)
    bad_vs = _fake_vmec_spline_struct()
    bad_vs.Aminor_p = 0.0

    monkeypatch.setitem(
        sys.modules,
        "netCDF4",
        SimpleNamespace(Dataset=lambda *_args, **_kwargs: fake_nc),
    )
    monkeypatch.setattr(
        "gkx.geometry.vmec_state_controls._import_booz_backend",
        lambda: fake_backend,
    )
    monkeypatch.setattr(
        "gkx.geometry.vmec_state_controls._vmec_splines",
        lambda _nc, _booz: bad_vs,
    )

    with pytest.raises(ValueError, match="Aminor_p"):
        _vmec_fieldlines(
            vmec_fname="dummy.nc",
            s_val=0.5,
            betaprim=0.01,
            alpha=0.2,
            include_shear_variation=False,
            include_pressure_variation=False,
            theta1d=np.linspace(-np.pi, np.pi, 9),
            isaxisym=True,
            iota_input=0.9,
            s_hat_input=0.0,
            res_theta=21,
            res_phi=21,
        )
    assert fake_nc.closed is True


def test_generate_vmec_eik_internal_maps_boundary_and_computes_betaprim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, object] = {}

    theta_out = np.linspace(-np.pi, np.pi, 9)
    arrays_equal_arc = {
        "theta": theta_out,
        "theta_PEST": theta_out,
        "bmag": np.linspace(1.0, 2.0, theta_out.size),
        "gradpar": np.full(theta_out.size, 0.5),
        "cvdrift": np.linspace(0.0, 1.0, theta_out.size),
        "gbdrift": np.linspace(1.0, 2.0, theta_out.size),
        "cvdrift0": np.linspace(2.0, 3.0, theta_out.size),
        "gbdrift0": np.linspace(3.0, 4.0, theta_out.size),
        "gds2": np.linspace(4.0, 5.0, theta_out.size),
        "gds21": np.linspace(-1.0, 1.0, theta_out.size),
        "gds22": np.linspace(5.0, 6.0, theta_out.size),
        "grho": np.full(theta_out.size, 1.1),
        "Rplot": np.linspace(4.0, 6.0, theta_out.size),
        "Zplot": np.linspace(-1.0, 1.0, theta_out.size),
        "grad_x": np.vstack(
            [
                np.ones(theta_out.size),
                2.0 * np.ones(theta_out.size),
                3.0 * np.ones(theta_out.size),
            ]
        ),
        "grad_y": np.vstack(
            [
                4.0 * np.ones(theta_out.size),
                5.0 * np.ones(theta_out.size),
                6.0 * np.ones(theta_out.size),
            ]
        ),
        "b_vec": np.vstack(
            [
                np.ones(theta_out.size),
                np.zeros(theta_out.size),
                np.zeros(theta_out.size),
            ]
        ),
        "scale": 1.0,
    }

    def _mock_fieldlines(**kwargs):
        calls["fieldlines"] = kwargs
        return SimpleNamespace(
            dpsidrho=2.0,
            iota_input=0.5,
            s_hat_input=0.8,
            nfp=5,
            alpha=0.125,
            zeta_center=0.25,
        )

    def _mock_cut(**kwargs):
        calls["cut"] = kwargs
        return np.linspace(-1.0, 1.0, 9), {"theta_PEST": np.linspace(-1.0, 1.0, 9)}

    def _mock_remap(**kwargs):
        calls["remap"] = kwargs
        return 0.5, arrays_equal_arc

    def _mock_write(
        path: Path, profiles: dict[str, object], *, request: object
    ) -> None:
        calls["write"] = {"path": path, "profiles": profiles, "request": request}
        Path(path).write_bytes(b"mock eik data")

    monkeypatch.setattr(
        "gkx.geometry.imported_vmec._vmec_fieldlines", _mock_fieldlines
    )
    monkeypatch.setattr(
        "gkx.geometry.imported_vmec._apply_flux_tube_cut", _mock_cut
    )
    monkeypatch.setattr(
        "gkx.geometry.imported_vmec._equal_arc_remap", _mock_remap
    )
    monkeypatch.setattr(
        "gkx.geometry.imported_vmec.write_vmec_eik_netcdf", _mock_write
    )

    request = SimpleNamespace(
        vmec_file=str(tmp_path / "wout_test.nc"),
        torflux=0.64,
        beta=0.02,
        alpha=0.1,
        include_shear_variation=True,
        include_pressure_variation=False,
        npol=1.0,
        npol_min=None,
        ntheta=8,
        isaxisym=False,
        boundary="fix aspect",
        which_crossing=None,
        betaprim=None,
        z=(1.0, -1.0),
        dens=(2.0, 1.0),
        temp=(3.0, 0.5),
        tprim=(4.0, 5.0),
        fprim=(6.0, 7.0),
        y0=2.5,
        x0=None,
        jtwist=3,
    )

    out = generate_vmec_eik_internal(
        output_path=tmp_path / "out.eik.nc", request=request
    )

    assert out == (tmp_path / "out.eik.nc").resolve()
    assert calls["fieldlines"]["betaprim"] == pytest.approx(
        -0.02 * (2.0 * 3.0 * (4.0 + 6.0) + 1.0 * 0.5 * (5.0 + 7.0))
    )
    assert calls["cut"]["flux_tube_cut"] == "aspect"
    assert calls["cut"]["which_crossing"] == -1
    assert calls["cut"]["ntheta"] == 9
    assert calls["cut"]["x0"] == pytest.approx(2.5)
    profiles = calls["write"]["profiles"]
    assert profiles["q"] == pytest.approx(2.0)
    assert profiles["shat"] == pytest.approx(0.8)
    assert profiles["Rmaj"] == pytest.approx(5.0)
    assert profiles["alpha"] == pytest.approx(0.125)
    assert profiles["nfp"] == 5
