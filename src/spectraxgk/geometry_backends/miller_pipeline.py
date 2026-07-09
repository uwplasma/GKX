"""High-level internal Miller-to-EIK pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from spectraxgk.geometry_backends.miller_core import (
    MillerCoreParams,
    build_collocation_surfaces,
    compute_equal_arc_theta,
    compute_primary_gradients,
    compute_straight_field_theta,
    rebuild_straight_theta_state,
)
from spectraxgk.geometry_backends.miller_numerics import (
    _safe_denom,
    cumulative_trapezoid,
    nperiod_data_extend,
)
from spectraxgk.geometry_backends.miller_profiles import assemble_miller_profiles


def internal_miller_backend_available() -> bool:
    """Return True when internal Miller backend dependencies are present."""

    try:
        import jax  # noqa: F401
        import jax.numpy as jnp  # noqa: F401
    except Exception:
        return False
    return True


def _request_attr(request: Any, *names: str) -> Any:
    """Return the first available attribute from a Miller request."""

    for name in names:
        if hasattr(request, name):
            return getattr(request, name)
    raise AttributeError(f"Miller request is missing all aliases: {', '.join(names)}")


@dataclass(frozen=True)
class _MillerGeometryNormalizations:
    dpsidrho_arr: np.ndarray
    dpsidrho: float
    bpol: np.ndarray
    bmag: np.ndarray


def _miller_params_from_request(request: Any) -> MillerCoreParams:
    return MillerCoreParams(
        ntgrid=int(int(request.ntheta) / 2 + 1),
        nperiod=int(request.nperiod),
        rhoc=float(_request_attr(request, "rhoc")),
        qinp=float(_request_attr(request, "qinp", "q")),
        shat=float(_request_attr(request, "shat", "s_hat")),
        rmaj=float(_request_attr(request, "Rmaj", "R0")),
        r_geo=float(_request_attr(request, "R_geo")),
        shift=float(_request_attr(request, "shift")),
        akappa=float(_request_attr(request, "akappa")),
        tri=float(_request_attr(request, "tri")),
        akappri=float(_request_attr(request, "akappri")),
        tripri=float(_request_attr(request, "tripri")),
        betaprim=float(_request_attr(request, "betaprim")),
    )


def _miller_geometry_normalizations(
    params: MillerCoreParams, state: dict[str, Any], gradients: dict[str, Any]
) -> _MillerGeometryNormalizations:
    r = np.asarray(state["r"], dtype=float)
    qfac = np.asarray(state["qfac"], dtype=float)
    theta_common = np.asarray(state["theta_common_mag_axis"], dtype=float)
    jac = np.asarray(gradients["jac"], dtype=float)
    drhod_r = np.asarray(gradients["drhod_r"], dtype=float)
    drhod_z = np.asarray(gradients["drhod_z"], dtype=float)

    jac_r_theta_arr = np.abs(
        2.0 * cumulative_trapezoid(jac / _safe_denom(r), theta_common, axis=1)[:, -1]
    )
    dpsidrho_arr = -(params.r_geo / np.abs(2.0 * np.pi * qfac)) * jac_r_theta_arr
    dpsidrho = float(dpsidrho_arr[1])
    bpol = (
        np.abs(dpsidrho)
        * np.sqrt(drhod_r[1] ** 2 + drhod_z[1] ** 2)
        / _safe_denom(r[1])
    )
    btor = params.r_geo / _safe_denom(r[1])
    return _MillerGeometryNormalizations(
        dpsidrho_arr=np.asarray(dpsidrho_arr, dtype=float),
        dpsidrho=dpsidrho,
        bpol=bpol,
        bmag=np.sqrt(bpol**2 + btor**2),
    )


def _miller_equal_arc_grid(
    params: MillerCoreParams, theta_st: np.ndarray, straight_state: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return compute_equal_arc_theta(
        theta_straight=theta_st[1],
        gradpar=np.asarray(straight_state["gradpar_center"], dtype=float),
        bmag=np.asarray(straight_state["bmag_center"], dtype=float),
        bpol=np.asarray(straight_state["bpol_center"], dtype=float),
        nperiod=params.nperiod,
    )


def _assemble_miller_profiles_for_request(request: Any) -> dict[str, Any]:
    params = _miller_params_from_request(request)
    state = build_collocation_surfaces(params)
    gradients = compute_primary_gradients(state)
    normalizations = _miller_geometry_normalizations(params, state, gradients)
    theta_common = np.asarray(state["theta_common_mag_axis"], dtype=float)
    r = np.asarray(state["r"], dtype=float)

    theta_st = compute_straight_field_theta(
        f_const=float(params.r_geo),
        dpsidrho=normalizations.dpsidrho_arr,
        jac=np.asarray(gradients["jac"], dtype=float),
        r=r,
        theta_common=theta_common,
    )
    straight_state = rebuild_straight_theta_state(
        params=params,
        state=state,
        theta_st=theta_st,
        dpsidrho=normalizations.dpsidrho,
        f_const=float(params.r_geo),
    )
    theta_target_ex, gradpar_target_ex, theta_source_ex = _miller_equal_arc_grid(
        params, theta_st, straight_state
    )
    return assemble_miller_profiles(
        params=params,
        state=state,
        gradients=gradients,
        straight_state=straight_state,
        theta_st_center=theta_st[1],
        theta_st_ex=nperiod_data_extend(theta_st[1], params.nperiod, istheta=1),
        theta_source_ex=theta_source_ex,
        theta_target_ex=theta_target_ex,
        gradpar_target_ex=gradpar_target_ex,
        bmag_center=normalizations.bmag,
        bpol_center=normalizations.bpol,
        dpsidrho=normalizations.dpsidrho,
    )


def write_miller_eik_netcdf(
    path: Path, profiles: dict[str, np.ndarray | float]
) -> None:
    """Write root-level imported Miller ``*.eiknc.nc`` output."""

    try:
        import netCDF4 as nc
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise ImportError(
            "netCDF4 is required for internal Miller eik writeout"
        ) from exc

    theta = np.asarray(profiles["theta"], dtype=float)
    ntheta2 = int(theta.shape[0] - 1)
    if ntheta2 < 2:
        raise ValueError("Insufficient theta samples for Miller eik writeout")

    with nc.Dataset(path, "w") as ds:
        ds.createDimension("z", ntheta2)

        def _vec(name: str) -> Any:
            return ds.createVariable(name, "f8", ("z",))

        _vec("theta")[:] = np.asarray(profiles["theta"], dtype=float)[:-1]
        _vec("bmag")[:] = np.asarray(profiles["bmag"], dtype=float)[:-1]
        _vec("gradpar")[:] = np.asarray(profiles["gradpar"], dtype=float)[:-1]
        _vec("grho")[:] = np.asarray(profiles["grho"], dtype=float)[:-1]
        _vec("gds2")[:] = np.asarray(profiles["gds2"], dtype=float)[:-1]
        _vec("gds21")[:] = np.asarray(profiles["gds21"], dtype=float)[:-1]
        _vec("gds22")[:] = np.asarray(profiles["gds22"], dtype=float)[:-1]
        _vec("gbdrift")[:] = np.asarray(profiles["gbdrift"], dtype=float)[:-1]
        _vec("gbdrift0")[:] = np.asarray(profiles["gbdrift0"], dtype=float)[:-1]
        _vec("cvdrift")[:] = np.asarray(profiles["cvdrift"], dtype=float)[:-1]
        _vec("cvdrift0")[:] = np.asarray(profiles["cvdrift0"], dtype=float)[:-1]
        _vec("jacob")[:] = np.asarray(profiles["jacob"], dtype=float)[:-1]
        _vec("Rplot")[:] = np.asarray(profiles["Rplot"], dtype=float)[:-1]
        _vec("Zplot")[:] = np.asarray(profiles["Zplot"], dtype=float)[:-1]
        _vec("aprime")[:] = np.asarray(profiles["aprime"], dtype=float)[:-1]

        ds.createVariable("drhodpsi", "f8").assignValue(float(profiles["drhodpsi"]))
        ds.createVariable("kxfac", "f8").assignValue(float(profiles["kxfac"]))
        ds.createVariable("Rmaj", "f8").assignValue(float(profiles["Rmaj"]))
        ds.createVariable("q", "f8").assignValue(float(profiles["q"]))
        ds.createVariable("shat", "f8").assignValue(float(profiles["shat"]))


def generate_miller_eik_internal(
    *, output_path: str | Path, request: Any | None = None
) -> Path:
    """Internal Miller->EIK pipeline entry point (in progress)."""

    if request is None:
        raise NotImplementedError(
            "Internal Miller geometry backend requires runtime request data. "
            "Current status: low-level Miller geometry numerics are ported; final EIK writeout is pending."
        )

    profiles = _assemble_miller_profiles_for_request(request)
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    write_miller_eik_netcdf(out, profiles)
    return out


__all__ = [
    "_request_attr",
    "generate_miller_eik_internal",
    "internal_miller_backend_available",
    "write_miller_eik_netcdf",
]
