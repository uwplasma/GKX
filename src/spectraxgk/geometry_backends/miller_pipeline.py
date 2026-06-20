"""High-level internal Miller-to-EIK pipeline."""

from __future__ import annotations

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
from spectraxgk.geometry_backends.miller_io import write_miller_eik_netcdf
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


def generate_miller_eik_internal(
    *, output_path: str | Path, request: Any | None = None
) -> Path:
    """Internal Miller->EIK pipeline entry point (in progress)."""

    if request is None:
        raise NotImplementedError(
            "Internal Miller geometry backend requires runtime request data. "
            "Current status: low-level Miller geometry numerics are ported; final EIK writeout is pending."
        )

    params = MillerCoreParams(
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
    state = build_collocation_surfaces(params)
    gradients = compute_primary_gradients(state)

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
    bmag = np.sqrt(bpol**2 + btor**2)
    theta_st = compute_straight_field_theta(
        f_const=float(params.r_geo),
        dpsidrho=np.asarray(dpsidrho_arr, dtype=float),
        jac=jac,
        r=r,
        theta_common=theta_common,
    )
    straight_state = rebuild_straight_theta_state(
        params=params,
        state=state,
        theta_st=theta_st,
        dpsidrho=float(dpsidrho),
        f_const=float(params.r_geo),
    )
    theta_eqarc_target_ex, gradpar_eqarc_ex, theta_eqarc_source_ex = (
        compute_equal_arc_theta(
            theta_straight=theta_st[1],
            gradpar=np.asarray(straight_state["gradpar_center"], dtype=float),
            bmag=np.asarray(straight_state["bmag_center"], dtype=float),
            bpol=np.asarray(straight_state["bpol_center"], dtype=float),
            nperiod=params.nperiod,
        )
    )
    profiles = assemble_miller_profiles(
        params=params,
        state=state,
        gradients=gradients,
        straight_state=straight_state,
        theta_st_center=theta_st[1],
        theta_st_ex=nperiod_data_extend(theta_st[1], params.nperiod, istheta=1),
        theta_source_ex=theta_eqarc_source_ex,
        theta_target_ex=theta_eqarc_target_ex,
        gradpar_target_ex=gradpar_eqarc_ex,
        bmag_center=bmag,
        bpol_center=bpol,
        dpsidrho=float(dpsidrho),
    )
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    write_miller_eik_netcdf(out, profiles)
    return out


__all__ = [
    "_request_attr",
    "generate_miller_eik_internal",
    "internal_miller_backend_available",
]
