"""Miller to ``*.eiknc.nc`` generation helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

from spectraxgk.from_gx.miller import generate_miller_eik_internal, internal_miller_backend_available
from spectraxgk.runtime_config import RuntimeConfig


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CACHE_DIR = _REPO_ROOT / ".cache" / "spectrax" / "miller_eik"
_MILLER_EIK_CACHE_VERSION = 2


@dataclass(frozen=True)
class MillerGeometryRequest:
    """Miller geometry-generation contract."""

    ntheta: int
    nperiod: int
    boundary: str
    y0: float
    rhoc: float
    q: float
    s_hat: float
    Rmaj: float
    R_geo: float
    shift: float
    akappa: float
    akappri: float
    tri: float
    tripri: float
    betaprim: float


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


def build_miller_geometry_request(cfg: RuntimeConfig) -> MillerGeometryRequest:
    """Build a Miller generation request from a runtime config."""

    if str(cfg.geometry.model).strip().lower() != "miller":
        raise ValueError("geometry.model must be 'miller' for Miller geometry generation")

    y0 = float(cfg.grid.y0) if cfg.grid.y0 is not None else float(cfg.grid.Ly) / (2.0 * math.pi)
    ntheta = _infer_miller_ntheta(cfg)
    if ntheta < 2:
        raise ValueError("Miller geometry generation requires ntheta >= 2")

    return MillerGeometryRequest(
        ntheta=ntheta,
        nperiod=_infer_miller_nperiod(cfg),
        boundary=str(cfg.grid.boundary),
        y0=y0,
        rhoc=float(cfg.geometry.rhoc),
        q=float(cfg.geometry.q),
        s_hat=float(cfg.geometry.s_hat),
        Rmaj=float(cfg.geometry.R0),
        R_geo=float(cfg.geometry.R0 if cfg.geometry.R_geo is None else cfg.geometry.R_geo),
        shift=float(cfg.geometry.shift),
        akappa=float(cfg.geometry.akappa),
        akappri=float(cfg.geometry.akappri),
        tri=float(cfg.geometry.tri),
        tripri=float(cfg.geometry.tripri),
        betaprim=float(0.0 if cfg.geometry.betaprim is None else cfg.geometry.betaprim),
    )


def default_miller_eik_output_path(
    request: MillerGeometryRequest,
) -> Path:
    """Return a stable cache path for a Miller-generated ``*.eiknc.nc`` file."""

    payload = {"cache_version": _MILLER_EIK_CACHE_VERSION, **asdict(request)}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return _DEFAULT_CACHE_DIR / f"miller_{digest}.eiknc.nc"


def generate_runtime_miller_eik(
    cfg: RuntimeConfig,
    *,
    output_path: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Generate or reuse an internal-backend Miller ``*.eiknc.nc`` file from a runtime config."""

    request = build_miller_geometry_request(cfg)
    resolved_output = output_path
    if resolved_output is None and cfg.geometry.geometry_file is not None:
        resolved_output = cfg.geometry.geometry_file
    backend = str(cfg.geometry.geometry_backend).strip().lower()
    if not backend:
        backend = "auto"

    if backend == "gx":
        raise ValueError(
            "geometry_backend='gx' is no longer supported for runtime Miller geometry generation. "
            "Use geometry_backend='internal' (or 'auto')."
        )

    if backend not in {"auto", "internal"}:
        raise ValueError(
            f"Unknown geometry backend {cfg.geometry.geometry_backend!r}. "
            "Expected one of: 'auto', 'internal'."
        )

    if not internal_miller_backend_available():
        raise RuntimeError(
            "Internal Miller geometry backend dependencies are missing. "
            "Install JAX to enable the in-repo backend."
        )

    if resolved_output is None:
        resolved_output = default_miller_eik_output_path(request)
    return generate_miller_eik_internal(output_path=resolved_output, request=request)


# Compatibility aliases kept for existing callers and tests that still import
# the older GX-prefixed API names.
GXMillerGeometryRequest = MillerGeometryRequest
build_gx_miller_geometry_request = build_miller_geometry_request
