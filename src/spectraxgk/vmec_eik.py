"""GX-backed VMEC to ``*.eik.nc`` generation helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path

from spectraxgk.config import GX_REFERENCE_ELECTRON_MASS
from spectraxgk.from_gx.vmec import generate_vmec_eik_internal, internal_vmec_backend_available
from spectraxgk.runtime_config import RuntimeConfig


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CACHE_DIR = _REPO_ROOT / ".cache" / "spectrax" / "vmec_eik"


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


def _infer_vmec_npol(cfg: RuntimeConfig) -> float:
    if cfg.geometry.npol is not None:
        return float(cfg.geometry.npol)
    if cfg.grid.nperiod is not None:
        return float(2 * int(cfg.grid.nperiod) - 1)
    return 1.0


def _resolve_runtime_vmec_file(vmec_file: str) -> Path:
    """Resolve a runtime VMEC path with env/user expansion."""

    expanded = Path(os.path.expandvars(vmec_file)).expanduser()
    if expanded.is_absolute():
        return expanded.resolve()

    cwd_candidate = expanded.resolve()
    return cwd_candidate


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
    # Match GX VMEC defaults: unless the user exposes an explicit VMEC x0 control,
    # leave x0 unset so the geometry helper chooses the flux-tube cut.
    x0 = None
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
        vmec_file=str(_resolve_runtime_vmec_file(cfg.geometry.vmec_file)),
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
    )


def default_vmec_eik_output_path(
    request: GXVmecGeometryRequest,
) -> Path:
    """Return a stable cache path for a VMEC-generated ``*.eik.nc`` file."""

    vmec_path = Path(request.vmec_file).expanduser().resolve()
    stat = vmec_path.stat()
    payload = {
        **asdict(request),
        "vmec_file": str(vmec_path),
        "vmec_size": stat.st_size,
        "vmec_mtime_ns": stat.st_mtime_ns,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    stem = vmec_path.stem.removeprefix("wout_")
    return _DEFAULT_CACHE_DIR / f"{stem}_{digest}.eik.nc"


def generate_runtime_vmec_eik(
    cfg: RuntimeConfig,
    *,
    output_path: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Generate or reuse an internal-backend ``*.eik.nc`` file from a runtime config."""

    request = build_gx_vmec_geometry_request(cfg)
    resolved_output = output_path
    if resolved_output is None and cfg.geometry.geometry_file is not None:
        resolved_output = cfg.geometry.geometry_file
    # For runtime VMEC workflows, an explicit geometry_file is an output target,
    # not a signal to reuse whatever happened to be on disk from a previous run.
    backend = str(cfg.geometry.geometry_backend).strip().lower()
    if not backend:
        backend = "auto"

    if backend == "gx":
        raise ValueError(
            "geometry_backend='gx' is no longer supported for runtime VMEC geometry generation. "
            "Use geometry_backend='internal' (or 'auto')."
        )

    if backend not in {"auto", "internal"}:
        raise ValueError(
            f"Unknown geometry backend {cfg.geometry.geometry_backend!r}. "
            "Expected one of: 'auto', 'internal'."
        )

    if not internal_vmec_backend_available():
        raise RuntimeError(
            "Internal VMEC geometry backend dependencies are missing. "
            "Install JAX plus either booz_xform_jax or booz_xform."
        )

    if resolved_output is None:
        resolved_output = default_vmec_eik_output_path(request)
    return generate_vmec_eik_internal(output_path=resolved_output, request=request)
