from __future__ import annotations

import os
from pathlib import Path
import sys
import netCDF4 as nc
import numpy as np
import pytest
from unittest.mock import MagicMock
from dataclasses import replace

from gkx.config import (
    GeometryConfig,
    GridConfig,
    InitializationConfig,
    TimeConfig,
)
from gkx.workflows.runtime.config import (
    RuntimeConfig,
    RuntimeNormalizationConfig,
    RuntimePhysicsConfig,
    RuntimeSpeciesConfig,
)
from gkx.geometry import load_imported_geometry_netcdf
from gkx.geometry.vmec_eik import (
    build_vmec_geometry_request,
    default_vmec_eik_output_path,
    generate_runtime_vmec_eik,
)
import gkx.geometry.imported_vmec as vmec_backend
from gkx.geometry.imported_vmec import internal_vmec_backend_available


def _vmec_runtime_cfg(
    tmp_path: Path, *, geometry_file: str | None = None
) -> RuntimeConfig:
    vmec_path = tmp_path / "wout_test.nc"
    vmec_path.write_text("stub", encoding="utf-8")
    return RuntimeConfig(
        grid=GridConfig(
            Nx=1,
            Ny=8,
            Nz=32,
            Lx=62.8,
            Ly=62.8,
            boundary="linked",
            y0=10.0,
            ntheta=32,
            nperiod=1,
        ),
        time=TimeConfig(
            t_max=1.0, dt=0.1, method="rk4", use_diffrax=False, fixed_dt=True
        ),
        geometry=GeometryConfig(
            model="vmec",
            vmec_file=str(vmec_path),
            geometry_file=geometry_file,
            torflux=0.64,
            npol=2.0,
            alpha=0.1,
        ),
        init=InitializationConfig(init_field="density", init_amp=1.0e-6),
        species=(
            RuntimeSpeciesConfig(
                name="ion", charge=1.0, mass=1.0, tprim=3.0, fprim=1.0
            ),
        ),
        physics=RuntimePhysicsConfig(
            linear=True,
            nonlinear=False,
            adiabatic_electrons=True,
            tau_e=1.0,
            electrostatic=True,
            electromagnetic=False,
            beta=0.0,
            collisions=False,
        ),
        normalization=RuntimeNormalizationConfig(
            contract="kinetic", diagnostic_norm="rho_star"
        ),
    )


def _write_minimal_eik_cache(path: Path) -> None:
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("z", 1)
        for name in ("theta", "bmag", "gradpar"):
            ds.createVariable(name, "f8", ("z",))[:] = [1.0]
        ds.createVariable("q", "f8").assignValue(1.4)
        ds.createVariable("shat", "f8").assignValue(0.8)


def test_build_vmec_geometry_request_creates_expected_request(tmp_path: Path) -> None:
    cfg = _vmec_runtime_cfg(tmp_path)
    request = build_vmec_geometry_request(cfg)

    assert request.vmec_file == str(Path(cfg.geometry.vmec_file).resolve())
    assert request.torflux == 0.64
    assert request.npol == 2.0
    assert request.ntheta == 32
    assert request.alpha == 0.1
    assert request.y0 == pytest.approx(10.0)
    assert request.z == (1.0, -1.0)  # Ion + adiabatic electron


def test_build_vmec_geometry_request_expands_env_vmec_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    vmec_path = tmp_path / "wout_env.nc"
    vmec_path.write_text("stub", encoding="utf-8")
    monkeypatch.setenv("HSX_VMEC_FILE", str(vmec_path))
    cfg = _vmec_runtime_cfg(tmp_path)
    cfg = replace(cfg, geometry=replace(cfg.geometry, vmec_file="$HSX_VMEC_FILE"))

    request = build_vmec_geometry_request(cfg)

    assert request.vmec_file == str(vmec_path.resolve())


@pytest.mark.integration
def test_vmec_roundtrip_gate_is_deterministic(tmp_path: Path) -> None:
    vmec_file = os.environ.get("GKX_VMEC_FILE", "").strip()
    if not vmec_file:
        pytest.skip("Set GKX_VMEC_FILE to enable VMEC roundtrip parity gate.")

    geometry_helper_repo = (
        os.environ.get("GKX_GEOMETRY_HELPER_REPO", "").strip() or None
    )
    cfg = _vmec_runtime_cfg(tmp_path)
    cfg = replace(
        cfg,
        grid=replace(cfg.grid, Nz=64, ntheta=64),
        geometry=replace(
            cfg.geometry,
            vmec_file=vmec_file,
            geometry_file=None,
            npol=1.0,
            alpha=0.0,
            geometry_helper_repo=geometry_helper_repo,
        ),
    )

    out1 = tmp_path / "geom1.eik.nc"
    out2 = tmp_path / "geom2.eik.nc"
    generate_runtime_vmec_eik(cfg, output_path=out1, force=True)
    generate_runtime_vmec_eik(cfg, output_path=out2, force=True)

    g1 = load_imported_geometry_netcdf(out1)
    g2 = load_imported_geometry_netcdf(out2)

    # This is a determinism gate, not a physics gate: any drift here will break
    # VMEC-backed parity workflows in hard-to-debug ways.
    for name in (
        "theta",
        "bmag_profile",
        "gds2_profile",
        "gds21_profile",
        "gds22_profile",
        "cv_profile",
        "gb_profile",
        "jacobian_profile",
        "grho_profile",
    ):
        a = np.asarray(getattr(g1, name))
        b = np.asarray(getattr(g2, name))
        np.testing.assert_allclose(a, b, rtol=0.0, atol=0.0)


def test_build_vmec_geometry_request_infers_npol_from_nperiod(tmp_path: Path) -> None:
    cfg = _vmec_runtime_cfg(tmp_path)
    cfg = replace(
        cfg,
        grid=replace(cfg.grid, nperiod=3),
        geometry=replace(cfg.geometry, npol=None),
    )

    request = build_vmec_geometry_request(cfg)

    assert request.npol == pytest.approx(5.0)


def test_default_vmec_eik_output_path_tracks_vmec_file_metadata(tmp_path: Path) -> None:
    cfg = _vmec_runtime_cfg(tmp_path)
    request = build_vmec_geometry_request(cfg)

    first = default_vmec_eik_output_path(request)
    vmec_path = Path(request.vmec_file)
    vmec_path.write_text("updated-stub", encoding="utf-8")
    os.utime(
        vmec_path,
        ns=(vmec_path.stat().st_atime_ns, vmec_path.stat().st_mtime_ns + 1_000_000),
    )

    second = default_vmec_eik_output_path(request)

    assert first.parent.name == "vmec_eik"
    assert first.suffixes == [".eik", ".nc"]
    assert first != second


def test_atomic_vmec_eik_write_replaces_final_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_path = tmp_path / "geom.eik.nc"
    out_path.write_text("old", encoding="utf-8")
    temp_paths: list[Path] = []

    def fake_write(path: Path, _profiles: dict, *, request: object) -> None:
        assert request == "request"
        assert path != out_path
        temp_paths.append(path)
        path.write_text("new", encoding="utf-8")

    monkeypatch.setattr(
        "gkx.geometry.imported_vmec.write_vmec_eik_netcdf", fake_write
    )

    vmec_backend._write_vmec_eik_netcdf_atomically(out_path, {}, request="request")

    assert out_path.read_text(encoding="utf-8") == "new"
    assert temp_paths
    assert not temp_paths[0].exists()


def test_generate_runtime_vmec_eik_invokes_internal_generator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_path = tmp_path / "geom.eik.nc"
    cfg = _vmec_runtime_cfg(tmp_path, geometry_file=str(out_path))

    mock_gen = MagicMock(return_value=out_path.resolve())
    monkeypatch.setattr(
        "gkx.geometry.vmec_eik.generate_vmec_eik_internal", mock_gen
    )
    monkeypatch.setattr(
        "gkx.geometry.vmec_eik.internal_vmec_backend_available", lambda: True
    )

    out = generate_runtime_vmec_eik(cfg)

    assert out == out_path.resolve()
    assert mock_gen.called
    _, kwargs = mock_gen.call_args
    request = kwargs["request"]
    assert request.ntheta == 32
    assert request.vmec_file == str(Path(cfg.geometry.vmec_file).resolve())


def test_generate_runtime_vmec_eik_reuses_default_cache_without_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _vmec_runtime_cfg(tmp_path, geometry_file=None)
    expected_out = tmp_path / "cached.eik.nc"
    _write_minimal_eik_cache(expected_out)

    monkeypatch.setattr(
        "gkx.geometry.vmec_eik.default_vmec_eik_output_path",
        lambda _request: expected_out,
    )
    monkeypatch.setattr(
        "gkx.geometry.vmec_eik.internal_vmec_backend_available", lambda: False
    )
    mock_gen = MagicMock(
        side_effect=AssertionError("cached VMEC geometry should be reused")
    )
    monkeypatch.setattr(
        "gkx.geometry.vmec_eik.generate_vmec_eik_internal", mock_gen
    )

    out = generate_runtime_vmec_eik(cfg)

    assert out == expected_out.resolve()
    assert not mock_gen.called


def test_generate_runtime_vmec_eik_regenerates_invalid_default_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _vmec_runtime_cfg(tmp_path, geometry_file=None)
    expected_out = tmp_path / "cached.eik.nc"
    expected_out.write_text("partial-not-netcdf", encoding="utf-8")

    monkeypatch.setattr(
        "gkx.geometry.vmec_eik.default_vmec_eik_output_path",
        lambda _request: expected_out,
    )
    monkeypatch.setattr(
        "gkx.geometry.vmec_eik.internal_vmec_backend_available", lambda: True
    )
    mock_gen = MagicMock(return_value=expected_out.resolve())
    monkeypatch.setattr(
        "gkx.geometry.vmec_eik.generate_vmec_eik_internal", mock_gen
    )

    out = generate_runtime_vmec_eik(cfg)

    assert out == expected_out.resolve()
    assert mock_gen.called


def test_generate_runtime_vmec_eik_uses_default_output_when_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _vmec_runtime_cfg(tmp_path, geometry_file=None)
    expected_out = tmp_path / "cached.eik.nc"

    monkeypatch.setattr(
        "gkx.geometry.vmec_eik.default_vmec_eik_output_path",
        lambda _request: expected_out,
    )
    monkeypatch.setattr(
        "gkx.geometry.vmec_eik.internal_vmec_backend_available", lambda: True
    )
    mock_gen = MagicMock(return_value=expected_out.resolve())
    monkeypatch.setattr(
        "gkx.geometry.vmec_eik.generate_vmec_eik_internal", mock_gen
    )

    out = generate_runtime_vmec_eik(cfg)

    assert out == expected_out.resolve()
    _, kwargs = mock_gen.call_args
    assert Path(kwargs["output_path"]) == expected_out


@pytest.mark.parametrize(
    ("backend", "expected_message"),
    [
        ("mystery", "Unknown geometry backend"),
    ],
)
def test_generate_runtime_vmec_eik_rejects_invalid_backends(
    tmp_path: Path, backend: str, expected_message: str
) -> None:
    cfg = _vmec_runtime_cfg(tmp_path)
    cfg = replace(cfg, geometry=replace(cfg.geometry, geometry_backend=backend))

    with pytest.raises(ValueError, match=expected_message):
        generate_runtime_vmec_eik(cfg)


def test_generate_runtime_vmec_eik_requires_internal_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _vmec_runtime_cfg(tmp_path)
    monkeypatch.setattr(
        "gkx.geometry.vmec_eik.internal_vmec_backend_available", lambda: False
    )

    with pytest.raises(
        RuntimeError, match="Internal VMEC geometry backend dependencies are missing"
    ):
        generate_runtime_vmec_eik(cfg)


def test_internal_vmec_backend_available_detects_env_provided_booz_xform_jax(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pkg_root = tmp_path / "booz_xform_jax_checkout"
    pkg_dir = pkg_root / "src" / "booz_xform_jax"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text(
        "class Booz_xform:\n    pass\n", encoding="utf-8"
    )

    for name in ("booz_xform_jax", "booz_xform"):
        sys.modules.pop(name, None)
    original_sys_path = list(sys.path)
    monkeypatch.setenv("GKX_BOOZ_XFORM_JAX_PATH", str(pkg_root))
    monkeypatch.delenv("BOOZ_XFORM_JAX_PATH", raising=False)

    try:
        assert internal_vmec_backend_available() is True
    finally:
        sys.path[:] = original_sys_path
        for name in ("booz_xform_jax", "booz_xform"):
            sys.modules.pop(name, None)
