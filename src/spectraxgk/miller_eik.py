"""GX-backed Miller to ``*.eiknc.nc`` generation helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

from spectraxgk.runtime_config import RuntimeConfig
from spectraxgk.vmec_eik import resolve_gx_python, resolve_gx_repo


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CACHE_DIR = _REPO_ROOT / ".cache" / "spectrax" / "miller_eik"


@dataclass(frozen=True)
class GXMillerGeometryRequest:
    """Minimal GX Miller geometry-generation contract."""

    ntheta: int
    nperiod: int
    boundary: str
    y0: float
    rhoc: float
    qinp: float
    shat: float
    Rmaj: float
    R_geo: float
    shift: float
    akappa: float
    akappri: float
    tri: float
    tripri: float
    betaprim: float
    gx_repo: str | None = None
    gx_python: str | None = None


def _format_toml_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isfinite(value):
            return repr(float(value))
        raise ValueError(f"Non-finite TOML float: {value!r}")
    raise TypeError(f"Unsupported TOML scalar type: {type(value)!r}")


def resolve_gx_miller_script(gx_repo: str | Path | None = None) -> Path:
    """Return the GX ``gx_geo.py`` Miller geometry script path."""

    repo = resolve_gx_repo(gx_repo)
    script = repo / "geometry_modules" / "miller" / "gx_geo.py"
    if not script.exists():
        raise FileNotFoundError(f"GX Miller geometry script not found: {script}")
    return script


def _infer_miller_ntheta(cfg: RuntimeConfig) -> int:
    if cfg.grid.ntheta is not None:
        return int(cfg.grid.ntheta)
    return int(cfg.grid.Nz)


def _infer_miller_nperiod(cfg: RuntimeConfig) -> int:
    if cfg.grid.nperiod is not None:
        return int(cfg.grid.nperiod)
    if cfg.grid.zp is not None:
        zp = int(cfg.grid.zp)
        return max((zp + 1) // 2, 1)
    return 1


def build_gx_miller_geometry_request(cfg: RuntimeConfig) -> GXMillerGeometryRequest:
    """Build a GX Miller generation request from a runtime config."""

    if str(cfg.geometry.model).strip().lower() != "miller":
        raise ValueError("geometry.model must be 'miller' for GX Miller geometry generation")

    y0 = float(cfg.grid.y0) if cfg.grid.y0 is not None else float(cfg.grid.Ly) / (2.0 * math.pi)
    ntheta = _infer_miller_ntheta(cfg)
    if ntheta < 2:
        raise ValueError("Miller geometry generation requires ntheta >= 2")

    return GXMillerGeometryRequest(
        ntheta=ntheta,
        nperiod=_infer_miller_nperiod(cfg),
        boundary=str(cfg.grid.boundary),
        y0=y0,
        rhoc=float(cfg.geometry.rhoc),
        qinp=float(cfg.geometry.q),
        shat=float(cfg.geometry.s_hat),
        Rmaj=float(cfg.geometry.R0),
        R_geo=float(cfg.geometry.R0 if cfg.geometry.R_geo is None else cfg.geometry.R_geo),
        shift=float(cfg.geometry.shift),
        akappa=float(cfg.geometry.akappa),
        akappri=float(cfg.geometry.akappri),
        tri=float(cfg.geometry.tri),
        tripri=float(cfg.geometry.tripri),
        betaprim=float(0.0 if cfg.geometry.betaprim is None else cfg.geometry.betaprim),
        gx_repo=cfg.geometry.gx_repo,
        gx_python=cfg.geometry.gx_python,
    )


def default_miller_eik_output_path(
    request: GXMillerGeometryRequest,
    *,
    gx_repo: str | Path | None = None,
) -> Path:
    """Return a stable cache path for a Miller-generated ``*.eiknc.nc`` file."""

    payload = {
        **asdict(request),
        "gx_repo": str(resolve_gx_repo(gx_repo or request.gx_repo)),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return _DEFAULT_CACHE_DIR / f"miller_{digest}.eiknc.nc"


def write_gx_miller_geometry_input(request: GXMillerGeometryRequest, path: str | Path) -> Path:
    """Write the minimal TOML input that GX's Miller helper expects."""

    out = Path(path)
    lines = [
        "debug = false",
        "",
        "[Dimensions]",
        f"ntheta = {request.ntheta}",
        f"nperiod = {request.nperiod}",
        "",
        "[Domain]",
        f"boundary = {_format_toml_scalar(request.boundary)}",
        f"y0 = {_format_toml_scalar(float(request.y0))}",
        "",
        "[Geometry]",
        'geo_option = "miller"',
        f"rhoc = {_format_toml_scalar(float(request.rhoc))}",
        f"qinp = {_format_toml_scalar(float(request.qinp))}",
        f"shat = {_format_toml_scalar(float(request.shat))}",
        f"Rmaj = {_format_toml_scalar(float(request.Rmaj))}",
        f"R_geo = {_format_toml_scalar(float(request.R_geo))}",
        f"shift = {_format_toml_scalar(float(request.shift))}",
        f"akappa = {_format_toml_scalar(float(request.akappa))}",
        f"akappri = {_format_toml_scalar(float(request.akappri))}",
        f"tri = {_format_toml_scalar(float(request.tri))}",
        f"tripri = {_format_toml_scalar(float(request.tripri))}",
        f"betaprim = {_format_toml_scalar(float(request.betaprim))}",
        "",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def generate_gx_miller_eik(
    request: GXMillerGeometryRequest,
    *,
    output_path: str | Path | None = None,
    gx_repo: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Generate a GX-compatible Miller ``*.eiknc.nc`` file."""

    out = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else default_miller_eik_output_path(request, gx_repo=gx_repo)
    )
    if out.exists() and not force:
        return out

    script = resolve_gx_miller_script(gx_repo or request.gx_repo)
    gx_python = resolve_gx_python(request.gx_python)
    out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="spectrax_miller_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        input_path = tmpdir_path / "gx_miller_geometry.in"
        eik_out_path = tmpdir_path / "gx_miller_geometry.eik.out"
        eik_nc_path = tmpdir_path / "gx_miller_geometry.eiknc.nc"
        write_gx_miller_geometry_input(request, input_path)
        proc = subprocess.run(
            [gx_python, str(script), str(input_path), str(eik_out_path)],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(script.parent),
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "GX Miller geometry generation failed:\n"
                f"command: {gx_python} {script} ...\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )
        if not eik_nc_path.exists():
            raise RuntimeError(f"GX Miller geometry generation did not create the expected output: {eik_nc_path}")
        shutil.copy2(eik_nc_path, out)
    return out


def generate_runtime_miller_eik(
    cfg: RuntimeConfig,
    *,
    output_path: str | Path | None = None,
    gx_repo: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Generate or reuse a GX-compatible Miller ``*.eiknc.nc`` file from a runtime config."""

    request = build_gx_miller_geometry_request(cfg)
    resolved_output = output_path
    if resolved_output is None and cfg.geometry.geometry_file is not None:
        resolved_output = cfg.geometry.geometry_file
    force_runtime = force or resolved_output is not None
    return generate_gx_miller_eik(request, output_path=resolved_output, gx_repo=gx_repo, force=force_runtime)
