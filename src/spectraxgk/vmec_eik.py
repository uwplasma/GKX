"""GX-backed VMEC to ``*.eik.nc`` generation helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from spectraxgk.config import GX_REFERENCE_ELECTRON_MASS
from spectraxgk.runtime_config import RuntimeConfig


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CACHE_DIR = _REPO_ROOT / ".cache" / "spectrax" / "vmec_eik"
_DEFAULT_GX_REPOS = (
    Path("/path/to/reference-code"),
    Path("/home/user/GX"),
)


@dataclass(frozen=True)
class GXVmecGeometryRequest:
    """Minimal GX VMEC geometry-generation contract."""

    vmec_file: str
    ntheta: int
    boundary: str
    y0: float
    x0: float | None
    jtwist: int | None
    beta: float
    alpha: float
    torflux: float
    npol: float
    npol_min: float | None
    isaxisym: bool
    which_crossing: int | None
    include_shear_variation: bool
    include_pressure_variation: bool
    betaprim: float | None
    z: tuple[float, ...]
    mass: tuple[float, ...]
    dens: tuple[float, ...]
    temp: tuple[float, ...]
    tprim: tuple[float, ...]
    fprim: tuple[float, ...]
    vnewk: tuple[float, ...]
    species_type: tuple[str, ...]
    gx_repo: str | None = None


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


def _format_toml_array(values: tuple[object, ...]) -> str:
    return "[" + ", ".join(_format_toml_scalar(value) for value in values) + "]"


def resolve_gx_repo(explicit_repo: str | Path | None = None) -> Path:
    """Resolve the GX repository used for VMEC geometry generation."""

    if explicit_repo is not None:
        repo = Path(explicit_repo).expanduser().resolve()
        if not repo.exists():
            raise FileNotFoundError(f"GX repository does not exist: {repo}")
        return repo

    gx_repo_env = os.environ.get("GX_REPO")
    if gx_repo_env:
        env_repo = Path(gx_repo_env).expanduser().resolve()
        if env_repo.exists():
            return env_repo

    for candidate in _DEFAULT_GX_REPOS:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not locate a GX repository. Set geometry.gx_repo or the GX_REPO environment variable."
    )


def resolve_gx_vmec_script(gx_repo: str | Path | None = None) -> Path:
    """Return the GX ``gx_geo_vmec.py`` script path."""

    repo = resolve_gx_repo(gx_repo)
    script = repo / "geometry_modules" / "pyvmec" / "gx_geo_vmec.py"
    if not script.exists():
        raise FileNotFoundError(f"GX VMEC geometry script not found: {script}")
    return script


def _infer_vmec_npol(cfg: RuntimeConfig) -> float:
    if cfg.geometry.npol is not None:
        return float(cfg.geometry.npol)
    if cfg.grid.nperiod is not None:
        return float(2 * int(cfg.grid.nperiod) - 1)
    return 1.0


def build_gx_vmec_geometry_request(cfg: RuntimeConfig) -> GXVmecGeometryRequest:
    """Build a GX VMEC generation request from a runtime config."""

    if cfg.geometry.vmec_file is None:
        raise ValueError("geometry.vmec_file must be set when geometry.model='vmec'")
    if cfg.geometry.torflux is None:
        raise ValueError("geometry.torflux must be set when geometry.model='vmec'")

    beta = float(cfg.physics.beta)
    if beta != 0.0 and cfg.geometry.betaprim is None:
        has_adiabatic_species = bool(cfg.physics.adiabatic_electrons or cfg.physics.adiabatic_ions)
        if has_adiabatic_species:
            raise ValueError(
                "geometry.betaprim must be set for VMEC generation when beta!=0 and adiabatic species are present"
            )

    species = tuple(cfg.species)
    if not species:
        raise ValueError("RuntimeConfig.species must contain at least one species")

    y0 = float(cfg.grid.y0) if cfg.grid.y0 is not None else float(cfg.grid.Ly) / (2.0 * math.pi)
    x0 = float(cfg.grid.Lx) / (2.0 * math.pi) if float(cfg.grid.Lx) > 0.0 else None
    ntheta = int(cfg.grid.ntheta) if cfg.grid.ntheta is not None else int(cfg.grid.Nz)
    if ntheta < 2:
        raise ValueError("VMEC geometry generation requires ntheta >= 2")

    z = [float(sp.charge) for sp in species]
    mass = [float(sp.mass) for sp in species]
    dens = [float(sp.density) for sp in species]
    temp = [float(sp.temperature) for sp in species]
    tprim = [float(sp.tprim) for sp in species]
    fprim = [float(sp.fprim) for sp in species]
    vnewk = [float(sp.nu) for sp in species]
    species_type = ["electron" if float(sp.charge) < 0.0 else "ion" for sp in species]

    if cfg.physics.adiabatic_electrons and not any(val < 0.0 for val in z):
        z.append(-1.0)
        mass.append(GX_REFERENCE_ELECTRON_MASS)
        dens.append(1.0)
        temp.append(1.0 / max(float(cfg.physics.tau_e), 1.0e-30))
        tprim.append(0.0)
        fprim.append(0.0)
        vnewk.append(0.0)
        species_type.append("electron")

    return GXVmecGeometryRequest(
        vmec_file=str(Path(cfg.geometry.vmec_file).expanduser().resolve()),
        ntheta=ntheta,
        boundary=str(cfg.grid.boundary),
        y0=y0,
        x0=x0,
        jtwist=cfg.grid.jtwist,
        beta=beta,
        alpha=float(cfg.geometry.alpha),
        torflux=float(cfg.geometry.torflux),
        npol=_infer_vmec_npol(cfg),
        npol_min=None if cfg.geometry.npol_min is None else float(cfg.geometry.npol_min),
        isaxisym=bool(cfg.geometry.isaxisym),
        which_crossing=cfg.geometry.which_crossing,
        include_shear_variation=bool(cfg.geometry.include_shear_variation),
        include_pressure_variation=bool(cfg.geometry.include_pressure_variation),
        betaprim=None if cfg.geometry.betaprim is None else float(cfg.geometry.betaprim),
        z=tuple(z),
        mass=tuple(mass),
        dens=tuple(dens),
        temp=tuple(temp),
        tprim=tuple(tprim),
        fprim=tuple(fprim),
        vnewk=tuple(vnewk),
        species_type=tuple(species_type),
        gx_repo=cfg.geometry.gx_repo,
    )


def default_vmec_eik_output_path(
    request: GXVmecGeometryRequest,
    *,
    gx_repo: str | Path | None = None,
) -> Path:
    """Return a stable cache path for a VMEC-generated ``*.eik.nc`` file."""

    vmec_path = Path(request.vmec_file).expanduser().resolve()
    stat = vmec_path.stat()
    payload = {
        **asdict(request),
        "gx_repo": str(resolve_gx_repo(gx_repo or request.gx_repo)),
        "vmec_file": str(vmec_path),
        "vmec_size": stat.st_size,
        "vmec_mtime_ns": stat.st_mtime_ns,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    stem = vmec_path.stem.removeprefix("wout_")
    return _DEFAULT_CACHE_DIR / f"{stem}_{digest}.eik.nc"


def write_gx_vmec_geometry_input(request: GXVmecGeometryRequest, path: str | Path) -> Path:
    """Write the minimal TOML input that GX's ``gx_geo_vmec.py`` expects."""

    out = Path(path)
    lines = [
        "debug = false",
        "",
        "[Dimensions]",
        f"ntheta = {request.ntheta}",
        "",
        "[Domain]",
        f"boundary = {_format_toml_scalar(request.boundary)}",
        f"y0 = {_format_toml_scalar(float(request.y0))}",
    ]
    if request.x0 is not None:
        lines.append(f"x0 = {_format_toml_scalar(float(request.x0))}")
    if request.jtwist is not None:
        lines.append(f"jtwist = {int(request.jtwist)}")

    lines.extend(
        [
            "",
            "[Physics]",
            f"beta = {_format_toml_scalar(float(request.beta))}",
            "",
            "[Geometry]",
            'geo_option = "vmec"',
            f"vmec_file = {_format_toml_scalar(request.vmec_file)}",
            f"alpha = {_format_toml_scalar(float(request.alpha))}",
            f"npol = {_format_toml_scalar(float(request.npol))}",
            f"torflux = {_format_toml_scalar(float(request.torflux))}",
        ]
    )
    if request.npol_min is not None:
        lines.append(f"npol_min = {_format_toml_scalar(float(request.npol_min))}")
    if request.isaxisym:
        lines.append("isaxisym = true")
    if request.which_crossing is not None:
        lines.append(f"which_crossing = {int(request.which_crossing)}")
    if request.include_shear_variation:
        lines.append("include_shear_variation = true")
    if request.include_pressure_variation:
        lines.append("include_pressure_variation = true")
    if request.betaprim is not None:
        lines.append(f"betaprim = {_format_toml_scalar(float(request.betaprim))}")

    lines.extend(
        [
            "",
            "[species]",
            f"z = {_format_toml_array(request.z)}",
            f"mass = {_format_toml_array(request.mass)}",
            f"dens = {_format_toml_array(request.dens)}",
            f"temp = {_format_toml_array(request.temp)}",
            f"tprim = {_format_toml_array(request.tprim)}",
            f"fprim = {_format_toml_array(request.fprim)}",
            f"vnewk = {_format_toml_array(request.vnewk)}",
            f"type = {_format_toml_array(request.species_type)}",
            "",
        ]
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def generate_gx_vmec_eik(
    request: GXVmecGeometryRequest,
    *,
    output_path: str | Path | None = None,
    gx_repo: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Generate a GX-compatible ``*.eik.nc`` file from a VMEC ``wout`` file."""

    out = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else default_vmec_eik_output_path(request, gx_repo=gx_repo)
    )
    if out.exists() and not force:
        return out

    script = resolve_gx_vmec_script(gx_repo or request.gx_repo)
    out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="spectrax_vmec_") as tmpdir:
        input_path = Path(tmpdir) / "gx_vmec_geometry.toml"
        write_gx_vmec_geometry_input(request, input_path)
        proc = subprocess.run(
            [sys.executable, str(script), str(input_path), str(out)],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(script.parent),
        )
    if proc.returncode != 0:
        raise RuntimeError(
            "GX VMEC geometry generation failed:\n"
            f"command: {sys.executable} {script} ...\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    if not out.exists():
        raise RuntimeError(f"GX VMEC geometry generation did not create the requested output: {out}")
    return out


def generate_runtime_vmec_eik(
    cfg: RuntimeConfig,
    *,
    output_path: str | Path | None = None,
    gx_repo: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Generate or reuse a GX-compatible ``*.eik.nc`` file from a runtime config."""

    request = build_gx_vmec_geometry_request(cfg)
    resolved_output = output_path
    if resolved_output is None and cfg.geometry.geometry_file is not None:
        resolved_output = cfg.geometry.geometry_file
    return generate_gx_vmec_eik(request, output_path=resolved_output, gx_repo=gx_repo, force=force)
