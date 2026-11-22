"""VMEC to ``*.eik.nc`` generation helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import importlib
import json
import os
import math
from pathlib import Path
from typing import Callable

import numpy as np

from gkx.config import REFERENCE_ELECTRON_MASS
from gkx.geometry.backend_discovery import geometry_precision
from gkx.geometry.imported_vmec import (
    generate_vmec_eik_internal,
    internal_vmec_backend_available,
)
from gkx.config import RuntimeConfig


_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CACHE_DIR = _REPO_ROOT / ".cache" / "gkx" / "vmec_eik"
_REQUIRED_EIK_VARIABLES = ("theta", "bmag", "gradpar", "q", "shat")


@dataclass(frozen=True)
class VmecGeometryRequest:
    """VMEC geometry-generation contract."""

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


def build_vmec_geometry_request(cfg: RuntimeConfig) -> VmecGeometryRequest:
    """Build a VMEC generation request from a runtime config."""

    if str(cfg.geometry.model).strip().lower() != "vmec":
        raise ValueError("geometry.model must be 'vmec' for VMEC geometry generation")

    if cfg.geometry.vmec_file is None:
        raise ValueError("geometry.vmec_file must be set for VMEC geometry generation")
    if cfg.geometry.torflux is None:
        raise ValueError("geometry.torflux must be set for VMEC geometry generation")

    beta = float(cfg.physics.beta)
    species = tuple(cfg.species)
    if not species:
        raise ValueError("RuntimeConfig.species must contain at least one species")

    y0 = (
        float(cfg.grid.y0)
        if cfg.grid.y0 is not None
        else float(cfg.grid.Ly) / (2.0 * math.pi)
    )
    # Match imported VMEC defaults: unless the user exposes an explicit VMEC x0 control,
    # leave x0 unset so the geometry helper chooses the flux-tube cut.
    x0 = None
    npol = _infer_vmec_npol(cfg)
    # ``grid.ntheta`` counts points per poloidal turn; the line holds ``npol`` turns.
    ntheta = round(int(cfg.grid.ntheta) * npol) if cfg.grid.ntheta else int(cfg.grid.Nz)
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
        mass.append(REFERENCE_ELECTRON_MASS)
        dens.append(1.0)
        temp.append(1.0 / max(float(cfg.physics.tau_e), 1.0e-30))
        tprim.append(0.0)
        fprim.append(0.0)
        vnewk.append(0.0)
        species_type.append("electron")

    return VmecGeometryRequest(
        vmec_file=str(_resolve_runtime_vmec_file(cfg.geometry.vmec_file)),
        ntheta=ntheta,
        boundary=str(cfg.grid.boundary),
        y0=y0,
        x0=x0,
        jtwist=cfg.grid.jtwist,
        beta=beta,
        alpha=float(cfg.geometry.alpha),
        torflux=float(cfg.geometry.torflux),
        npol=npol,
        npol_min=None
        if cfg.geometry.npol_min is None
        else float(cfg.geometry.npol_min),
        isaxisym=bool(cfg.geometry.isaxisym),
        which_crossing=cfg.geometry.which_crossing,
        include_shear_variation=bool(cfg.geometry.include_shear_variation),
        include_pressure_variation=bool(cfg.geometry.include_pressure_variation),
        betaprim=None
        if cfg.geometry.betaprim is None
        else float(cfg.geometry.betaprim),
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
    request: VmecGeometryRequest,
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
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    stem = vmec_path.stem.removeprefix("wout_")
    return _DEFAULT_CACHE_DIR / f"{stem}_{digest}.eik.nc"


def _resolve_output_path(path: str | Path) -> Path:
    return Path(os.path.expandvars(str(path))).expanduser().resolve()


def _is_reusable_vmec_eik_cache(path: Path) -> bool:
    if not (path.exists() and path.is_file() and path.stat().st_size > 0):
        return False
    try:
        import netCDF4 as nc

        with nc.Dataset(path, "r") as ds:
            return all(name in ds.variables for name in _REQUIRED_EIK_VARIABLES)
    except Exception:
        return False


def generate_runtime_vmec_eik(
    cfg: RuntimeConfig,
    *,
    output_path: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Generate or reuse an internal-backend ``*.eik.nc`` file from a runtime config."""

    model = str(cfg.geometry.model).strip().lower()
    if model in {"desc", "gvec"}:
        wout = _convert_equilibrium(model, cfg.geometry.vmec_file, _DEFAULT_CACHE_DIR)
        cfg = replace(
            cfg, geometry=replace(cfg.geometry, model="vmec", vmec_file=str(wout))
        )
    request = build_vmec_geometry_request(cfg)
    resolved_output = output_path
    default_cache_output = False
    if resolved_output is None and cfg.geometry.geometry_file is not None:
        resolved_output = cfg.geometry.geometry_file
    # An explicit geometry_file selects the generated VMEC geometry output.
    backend = str(cfg.geometry.geometry_backend).strip().lower()
    if not backend:
        backend = "auto"

    if backend not in {"auto", "internal"}:
        raise ValueError(
            f"Unknown geometry backend {cfg.geometry.geometry_backend!r}. "
            "Expected one of: 'auto', 'internal'."
        )

    if resolved_output is None:
        resolved_output = default_vmec_eik_output_path(request)
        default_cache_output = True
    resolved_path = _resolve_output_path(resolved_output)

    if (
        default_cache_output
        and not force
        and _is_reusable_vmec_eik_cache(resolved_path)
    ):
        return resolved_path

    with geometry_precision():
        if not internal_vmec_backend_available():
            raise RuntimeError(
                "Internal VMEC geometry backend dependencies are missing. Install "
                "JAX plus booz_xform_jax, or provide a booz_xform-compatible shim."
            )
        return generate_vmec_eik_internal(output_path=resolved_path, request=request)


# Default EIK request template. This is a geometry default rather than report
# machinery, so it stays in the package even though the parity report builders
# that consumed it were removed from scripts/.
_VMEC_EIK_DEFAULT_REQUEST: dict[str, object] = {
    "y0": 10.0,
    "x0": None,
    "jtwist": None,
    "beta": 0.0,
    "npol": 1.0,
    "npol_min": None,
    "isaxisym": False,
    "which_crossing": None,
    "betaprim": None,
    "z": (1.0, -1.0),
    "mass": (1.0, 2.7e-4),
    "dens": (1.0, 1.0),
    "temp": (1.0, 1.0),
    "tprim": (3.0, 0.0),
    "fprim": (1.0, 0.0),
    "vnewk": (0.0, 0.0),
    "species_type": ("ion", "electron"),
}


# ---- DESC and GVEC equilibria as VMEC wout files ----
# A DESC or GVEC equilibrium is sampled on flux surfaces in its own straight-field-line
# angles, Fourier analysed in the VMEC convention and written as a stellarator-symmetric
# wout, so every coefficient comes from the VMEC code path. desc-opt and gvec are
# imported only when selected.


def _modes(mpol: int, ntor: int, nfp: int) -> tuple[np.ndarray, np.ndarray]:
    mn = [(m, n) for m in range(mpol) for n in range(0 if m == 0 else -ntor, ntor + 1)]
    return np.array([m for m, _ in mn], float), np.array(
        [n * nfp for _, n in mn], float
    )


def write_wout(
    path: str | Path,
    *,
    nfp: int,
    mpol: int,
    ntor: int,
    ns: int,
    sample: Callable[..., dict[str, np.ndarray]],
    iota: Callable[[np.ndarray], np.ndarray],
    pres: Callable[[np.ndarray], np.ndarray],
    phiedge: float,
    aminor: float,
    raxis: float,
) -> Path:
    """Write a stellarator-symmetric wout from a surface sampler.

    ``sample(fields, s, theta, zeta)`` returns the requested fields among R, Z,
    lam (theta* - theta), B, Bu, Bv (covariant components) on the tensor grid
    (s, theta, zeta) for the VMEC angles theta and zeta (zeta in [0, 2 pi / nfp)).
    """

    import netCDF4 as nc

    mpol, ntor = int(mpol), int(ntor)
    mn_b, nn_b = 2 * mpol, 2 * ntor
    xm, xn = _modes(mpol, ntor, nfp)
    xmq, xnq = _modes(mn_b, nn_b, nfp)
    nth, nze = 2 * mn_b + 2, 2 * nn_b + 2
    th = np.linspace(0.0, 2.0 * np.pi, nth, endpoint=False)
    ze = np.linspace(0.0, 2.0 * np.pi / nfp, nze, endpoint=False)
    s_full = np.linspace(0.0, 1.0, ns)
    s_half = 0.5 * (s_full[:-1] + s_full[1:])
    T, Z_ = np.meshgrid(th, ze, indexing="ij")

    def angle(m: np.ndarray, n: np.ndarray) -> np.ndarray:
        return m[None, :] * T.ravel()[:, None] - n[None, :] * Z_.ravel()[:, None]

    pc = np.linalg.pinv(np.cos(angle(xm, xn)))
    ps = np.linalg.pinv(np.sin(angle(xm, xn)))
    pcq = np.linalg.pinv(np.cos(angle(xmq, xnq)))

    full = sample(("R", "Z"), s_full[1:], th, ze)
    half = sample(("lam", "B", "Bu", "Bv"), s_half, th, ze)

    def coef(a: np.ndarray, p: np.ndarray) -> np.ndarray:
        return np.asarray(a).reshape(a.shape[0], -1) @ p.T

    rmnc, zmns = coef(full["R"], pc), coef(full["Z"], ps)
    rmnc = np.vstack([np.zeros(len(xm)), rmnc])
    zmns = np.vstack([np.zeros(len(xm)), zmns])
    lmns = np.vstack([np.zeros(len(xm)), coef(half["lam"], ps)])
    bmnc = np.vstack([np.zeros(len(xmq)), coef(half["B"], pcq)])
    bu = np.vstack([np.zeros(len(xmq)), coef(half["Bu"], pcq)])
    bv = np.vstack([np.zeros(len(xmq)), coef(half["Bv"], pcq)])
    # Axis: extrapolate the m=0 harmonics linearly in rho = sqrt(s) from two surfaces.
    r1, r2 = np.sqrt(s_full[1]), np.sqrt(s_full[2])
    for a in (rmnc, zmns):
        a[0] = a[1] - (a[2] - a[1]) * r1 / (r2 - r1)
        a[0, xm > 0] = 0.0

    out = Path(path)
    with nc.Dataset(out, "w") as d:
        d.createDimension("radius", ns)
        d.createDimension("mn_mode", len(xm))
        d.createDimension("mn_mode_nyq", len(xmq))
        d.createDimension("n_tor", ntor + 1)

        def put(name: str, value, dims: tuple[str, ...] = (), dt: str = "f8") -> None:
            d.createVariable(name, dt, dims)[:] = value

        for k, v in dict(
            nfp=nfp,
            ns=ns,
            mpol=mpol,
            ntor=ntor,
            mnmax=len(xm),
            mnmax_nyq=len(xmq),
            lasym__logical__=0,
        ).items():
            put(k, v, dt="i4")
        put("xm", xm, ("mn_mode",))
        put("xn", xn, ("mn_mode",))
        put("xm_nyq", xmq, ("mn_mode_nyq",))
        put("xn_nyq", xnq, ("mn_mode_nyq",))
        for name, a in (("rmnc", rmnc), ("zmns", zmns), ("lmns", lmns)):
            put(name, a, ("radius", "mn_mode"))
        for name, a in (("bmnc", bmnc), ("bsubumnc", bu), ("bsubvmnc", bv)):
            put(name, a, ("radius", "mn_mode_nyq"))
        for name, a in (
            ("iotas", np.r_[0.0, iota(s_half)]),
            ("pres", np.r_[0.0, pres(s_half)]),
            ("phi", phiedge * s_full),
            ("phipf", np.full(ns, phiedge / (2 * np.pi))),
            ("chi", np.zeros(ns)),
        ):
            put(name, a, ("radius",))
        put("Aminor_p", aminor)
        put(
            "raxis_cc",
            raxis * np.ones(ntor + 1) * (np.arange(ntor + 1) == 0),
            ("n_tor",),
        )
    return out


def _cached(
    path: Path, kind: str, convert: Callable[[Path], Path], cache: Path
) -> Path:
    stat = path.stat()
    key = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{kind}"
    out = (
        cache
        / f"{path.stem}_{kind}_{hashlib.sha256(key.encode()).hexdigest()[:12]}.wout.nc"
    )
    if not out.exists():
        cache.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp.nc")
        convert(tmp)
        os.replace(tmp, out)
    return out


def desc_to_wout(h5: str | Path, out: str | Path, *, ns: int = 16) -> Path:
    """Write a wout from a DESC equilibrium (``.h5``)."""

    import desc.io
    from desc.grid import LinearGrid

    eq = desc.io.load(str(h5))
    eq = eq[-1] if hasattr(eq, "__len__") else eq
    if not eq.sym:
        raise ValueError("DESC geometry requires a stellarator-symmetric equilibrium")
    keys = dict(R="R", Z="Z", lam="lambda", B="|B|", Bu="B_theta", Bv="B_zeta")

    def sample(fields, s, th, ze):
        grid = LinearGrid(rho=np.sqrt(s), theta=len(th), zeta=len(ze), NFP=eq.NFP)
        data = eq.compute([keys[k] for k in fields], grid=grid)
        shape = (len(ze), len(s), len(th))  # DESC meshgrid order: zeta, rho, theta
        return {k: data[keys[k]].reshape(shape).transpose(1, 2, 0) for k in fields}

    def flux(name):
        def f(s):
            return eq.compute(name, grid=LinearGrid(rho=np.sqrt(s), M=0, N=0))[name]

        return f

    return write_wout(
        out,
        nfp=eq.NFP,
        mpol=eq.M + 1,
        ntor=eq.N,
        ns=ns,
        sample=sample,
        iota=flux("iota"),
        pres=flux("p"),
        phiedge=float(eq.Psi),
        aminor=float(eq.compute("a")["a"]),
        raxis=float(np.sum(eq.axis.R_n)),
    )


def gvec_to_wout(state_file: str | Path, out: str | Path, *, ns: int = 16) -> Path:
    """Write a wout from a GVEC state (``*State*.dat``) next to its ``parameter.ini``.

    GVEC runs with zeta = -phi, so the VMEC angle v is -zeta and the iota,
    covariant B_v and edge flux change sign.
    """

    import gvec

    state_file = Path(state_file)
    params = state_file.with_name("parameter.ini")
    if not params.exists():
        params = next(state_file.parent.glob("*.ini"), params)
    st = gvec.load_state(params, state_file)
    if [st.parameters.get(f"{k}_sin_cos") for k in ("X1", "X2", "LA")] != [
        "_cos_",
        "_sin_",
        "_sin_",
    ]:
        raise ValueError("GVEC geometry requires a stellarator-symmetric state")
    mpol, ntor = st.get_mn_max("X1")
    mpol += 1

    def sample(fields, s, th, ze):
        need = {"R": "X1", "Z": "X2", "lam": "LA", "B": "mod_B"}
        names = [need[f] for f in fields if f in need]
        if "Bu" in fields:
            names += ["B", "e_theta", "e_zeta"]
        ev = st.evaluate(*names, rho=np.sqrt(s), theta=th, zeta=-ze)
        out = {f: ev[need[f]].values for f in fields if f in need}
        if "Bu" in fields:
            out["Bu"] = (ev.B * ev.e_theta).sum("xyz").values
            out["Bv"] = -(ev.B * ev.e_zeta).sum("xyz").values
        return out

    def flux(name, sign=1.0):
        return lambda s: (
            sign * st.evaluate(name, rho=np.sqrt(s), theta=None, zeta=None)[name].values
        )

    th = np.linspace(0.0, 2.0 * np.pi, 400, endpoint=False)
    edge = st.evaluate(
        "X1",
        "X2",
        "dX1_dt",
        "dX2_dt",
        "Phi_edge",
        rho=np.array([1.0]),
        theta=th,
        zeta=np.zeros(1),
    )
    area = (
        abs(
            np.mean(
                edge.X1.values * edge.dX2_dt.values
                - edge.X2.values * edge.dX1_dt.values
            )
        )
        * np.pi
    )
    axis = st.evaluate("X1", rho=np.zeros(1), theta=np.zeros(1), zeta=np.zeros(1))
    return write_wout(
        out,
        nfp=st.nfp,
        mpol=mpol,
        ntor=ntor,
        ns=ns,
        sample=sample,
        iota=flux("iota", -1.0),
        pres=flux("p"),
        phiedge=-2.0 * np.pi * float(edge.Phi_edge),
        aminor=float(np.sqrt(area / np.pi)),
        raxis=float(axis.X1.values.item()),
    )


def _convert_equilibrium(
    model: str, path: str | Path | None, cache: Path, **kw
) -> Path:
    """Return a cached wout for a ``desc`` or ``gvec`` equilibrium file."""

    if path is None:
        raise ValueError(
            f"geometry.model = '{model}' requires an equilibrium file path"
        )
    src = Path(os.path.expandvars(str(path))).expanduser()
    to_wout = {"desc": desc_to_wout, "gvec": gvec_to_wout}[model]

    def convert(out: Path) -> Path:
        try:
            importlib.import_module(model)
        except ImportError as err:
            raise ImportError(
                f"geometry.model = '{model}' needs the optional package: "
                f"pip install 'gkx[{model}]'"
            ) from err
        return to_wout(src, out, **kw)

    return _cached(src, model, convert, cache / "wout")
