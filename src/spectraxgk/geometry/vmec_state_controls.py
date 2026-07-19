"""Shared vmex state control helpers for differentiable geometry gates."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace as dc_replace
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from spectraxgk.geometry.backend_discovery import (
    _booz_read_wout_square_layout_failure,
    _import_booz_backend,
    _new_booz_object,
)
from spectraxgk.geometry.vmec_boozer_core import (
    load_solved_vmex_case,
    resolve_vmex_case_input_path,
)
from spectraxgk.geometry.vmec_boozer_derivatives import (
    _axisym_flip_required,
    _fieldline_boozer_coordinates,
    _input_iota_shear,
    _validated_reference_scales,
)
from spectraxgk.geometry.vmec_field_line_sampling import (
    _boozer_mode_angle,
    _boozer_trig_basis,
    _fieldline_boozer_tensors,
    _sample_boozer_mode_table,
    _vmec_splines,
)


VMEC_BOOZER_STATE_PARAMETER_NAMES = ("Rcos_mid_surface_m1",)
VMEC_BOOZER_STATE_PARAMETER_FAMILIES = ("Rcos", "Rsin", "Zcos", "Zsin", "Lcos", "Lsin")
#: Public family strings (config/report values) -> vmex SpectralState attributes.
_VMEC_STATE_FAMILY_ATTRS = {
    "Rcos": "R_cos",
    "Rsin": "R_sin",
    "Zcos": "Z_cos",
    "Zsin": "Z_sin",
    "Lcos": "L_cos",
    "Lsin": "L_sin",
}
#: Provenance marker: vmex equilibria are solved in memory, without a wout file.
VMEC_STATE_IN_MEMORY_WOUT_PATH = "in-memory:vmex.optimize.solve_equilibrium"


def _new_boozer_object_with_auto_fallback(
    primary_backend: Any, vmec_fname: str | Path, nc_obj: Any
) -> Any:
    """Create a Boozer transform object, using the classic reader if needed.

    Some vmex-written WOUT files expose a square ``(radius, mode)`` layout that
    old ``booz_xform_jax`` readers reject as ambiguous. In automatic backend
    mode, the imported-geometry path can safely fall back to the classic
    ``booz_xform`` reader; explicit backend selections remain fail-fast.
    """

    try:
        return _new_booz_object(primary_backend, str(vmec_fname))
    except Exception as exc:
        auto_backend = os.environ.get("SPECTRAX_BOOZ_BACKEND", "auto").strip().lower()
        if auto_backend in {"", "auto"} and _booz_read_wout_square_layout_failure(exc):
            try:
                fallback = _import_booz_backend("booz_xform")
                return _new_booz_object(fallback, str(vmec_fname))
            except Exception:
                nc_obj.close()
                raise
        nc_obj.close()
        raise


@dataclass(frozen=True)
class _VMECStateContext:
    """Solved vmex example state plus coefficient arrays used by AD gates.

    ``base_Rcos``/``base_Zsin`` mirror the vmex ``R_cos``/``Z_sin`` spectral
    tables; ``wout_path`` is the in-memory provenance marker because vmex
    solves the equilibrium directly instead of reading a wout file.
    """

    input_path: Any
    wout_path: Any
    inp: Any
    runtime: Any
    wout: Any
    state: Any
    base_Rcos: jnp.ndarray
    base_Zsin: jnp.ndarray


def _load_vmec_state_context(case_name: str) -> _VMECStateContext:
    """Solve a bundled vmex example and expose differentiable state arrays."""

    input_path = resolve_vmex_case_input_path(str(case_name))
    inp, state, runtime, wout = load_solved_vmex_case(str(case_name))
    base_Rcos = jnp.asarray(state.R_cos)
    base_Zsin = jnp.asarray(state.Z_sin)
    if base_Rcos.ndim != 2 or base_Zsin.ndim != 2:
        raise RuntimeError("vmex state R_cos/Z_sin arrays must be two-dimensional")
    return _VMECStateContext(
        input_path=input_path,
        wout_path=VMEC_STATE_IN_MEMORY_WOUT_PATH,
        inp=inp,
        runtime=runtime,
        wout=wout,
        state=state,
        base_Rcos=base_Rcos,
        base_Zsin=base_Zsin,
    )


def _vmec_state_family_attribute(parameter_family: str) -> str:
    """Map a public Fourier-family string to its vmex state attribute name."""

    family = str(parameter_family)
    attribute = _VMEC_STATE_FAMILY_ATTRS.get(family)
    if attribute is None:
        raise ValueError(
            "parameter_family must be one of "
            f"{', '.join(VMEC_BOOZER_STATE_PARAMETER_FAMILIES)}"
        )
    return attribute


def _vmec_boozer_state_array(state: Any, parameter_family: str) -> jnp.ndarray:
    """Return a validated VMEC state coefficient table for one Fourier family."""

    attribute = _vmec_state_family_attribute(parameter_family)
    if not hasattr(state, attribute):
        raise RuntimeError(f"vmex state does not expose {attribute}")
    array = jnp.asarray(getattr(state, attribute))
    if array.ndim != 2 or int(array.shape[1]) < 2:
        raise RuntimeError(
            f"vmex state {attribute} array must expose at least one non-axisymmetric mode"
        )
    return array


def _replace_vmec_boozer_state_coefficient(
    state: Any,
    parameter_family: str,
    base_array: jnp.ndarray,
    radial_index: int,
    mode_index: int,
    delta: Any,
) -> Any:
    """Return ``state`` with one VMEC/Boozer Fourier coefficient incremented."""

    return dc_replace(
        state,
        **{
            _vmec_state_family_attribute(parameter_family): base_array.at[
                int(radial_index),
                int(mode_index),
            ].add(delta)
        },
    )


def _vmec_boozer_state_parameter_name(
    parameter_family: str,
    radial_index: int,
    mode_index: int,
    *,
    default_mid_surface: int,
) -> str:
    """Name the state coefficient used by reports and finite-difference gates."""

    family = str(parameter_family)
    if int(radial_index) == int(default_mid_surface):
        return f"{family}_mid_surface_m{int(mode_index)}"
    return f"{family}_r{int(radial_index)}_m{int(mode_index)}"


def _resolve_vmec_state_indices(
    base_Rcos: jnp.ndarray,
    *,
    radial_index: int | None,
    mode_index: int,
    surface_index: int | None,
    surface_grid: str,
) -> tuple[int, int, int]:
    """Resolve coefficient and surface indices for VMEC-state sensitivity gates."""

    ns_full = int(base_Rcos.shape[0])
    ridx = ns_full // 2 if radial_index is None else int(radial_index)
    midx = int(mode_index)
    if not (0 <= ridx < ns_full):
        raise ValueError("radial_index is outside the VMEC state radial grid")
    if not (0 <= midx < int(base_Rcos.shape[1])):
        raise ValueError("mode_index is outside the VMEC state mode table")

    if surface_grid == "half_mesh":
        default_sidx = max(0, min(ridx - 1, ns_full - 2))
        surface_count = ns_full - 1
        error = "surface_index is outside the VMEC half-mesh Boozer surface grid"
    elif surface_grid == "field_line":
        default_sidx = max(1, min(ridx, ns_full - 2))
        surface_count = ns_full
        error = "surface_index is outside the VMEC metric radial grid"
    elif surface_grid == "metric":
        default_sidx = max(0, min(ridx - 1, ns_full - 1))
        surface_count = ns_full
        error = "surface_index is outside the VMEC metric radial grid"
    else:
        raise ValueError(f"unknown VMEC surface grid {surface_grid!r}")

    sidx = default_sidx if surface_index is None else int(surface_index)
    if not (0 <= sidx < surface_count):
        raise ValueError(error)
    return int(ridx), int(midx), int(sidx)


def _perturb_vmec_state(
    ctx: _VMECStateContext,
    x: jnp.ndarray,
    *,
    radial_index: int,
    mode_index: int,
) -> Any:
    """Return a VMEC state with two Fourier controls perturbed by ``x``."""

    return dc_replace(
        ctx.state,
        R_cos=ctx.base_Rcos.at[radial_index, mode_index].add(x[0]),
        Z_sin=ctx.base_Zsin.at[radial_index, mode_index].add(x[1]),
    )


def _length_two_params(params: jnp.ndarray | None, default: float) -> jnp.ndarray:
    """Normalize optional VMEC control perturbations to a length-two vector."""

    p = jnp.asarray([default, default] if params is None else params, dtype=jnp.float64)
    if p.ndim != 1 or int(p.shape[0]) != 2:
        raise ValueError("params must be a length-2 vector")
    return p


@dataclass(frozen=True)
class _VMECFieldlineScalars:
    s: np.ndarray
    ns: int
    alpha_arr: np.ndarray
    d_pressure_d_s: np.ndarray
    iota: np.ndarray
    d_iota_d_s: np.ndarray
    shat: np.ndarray
    nfp: int
    edge_toroidal_flux_over_2pi: float
    toroidal_flux_sign: float
    L_reference: float
    B_reference: float
    R_mag_ax: float
    zeta_center: float
    iota_input_val: float
    s_hat_input_val: float
    G: np.ndarray
    boozer_i: np.ndarray


@dataclass(frozen=True)
class _BoozerFieldlineSamples:
    xm_b: np.ndarray
    xn_b: np.ndarray
    rmnc_b: np.ndarray
    zmns_b: np.ndarray
    numns_b: np.ndarray
    d_rmnc_b_d_s: np.ndarray
    d_zmns_b_d_s: np.ndarray
    d_numns_b_d_s: np.ndarray
    gmnc_b: np.ndarray
    bmnc_b: np.ndarray
    d_bmnc_b_d_s: np.ndarray
    theta_b: np.ndarray
    phi_b: np.ndarray
    flipit: bool
    tensors: Any
    R_b: np.ndarray
    Z_b: np.ndarray
    nu_b: np.ndarray
    Vprime: np.ndarray
    mnmax_b: int


@dataclass(frozen=True)
class _BoozerModeProfiles:
    rmnc_b: np.ndarray
    zmns_b: np.ndarray
    numns_b: np.ndarray
    d_rmnc_b_d_s: np.ndarray
    d_zmns_b_d_s: np.ndarray
    d_numns_b_d_s: np.ndarray
    gmnc_b: np.ndarray
    bmnc_b: np.ndarray
    d_bmnc_b_d_s: np.ndarray


@dataclass(frozen=True)
class _BoozerFieldlineCoordinates:
    theta_b: np.ndarray
    phi_b: np.ndarray
    flipit: bool


@dataclass(frozen=True)
class _BoozerTrigSamples:
    cosangle_b: np.ndarray
    sinangle_b: np.ndarray
    mcosangle_b: np.ndarray
    msinangle_b: np.ndarray
    ncosangle_b: np.ndarray
    nsinangle_b: np.ndarray


def _load_vmec_boozer_splines(vmec_fname: str | Path) -> tuple[Any, Any]:
    """Open VMEC data, run Boozer transform, and build radial splines."""

    bxform = _import_booz_backend()
    from netCDF4 import Dataset as _NC

    nc_obj = _NC(str(vmec_fname), "r")
    try:
        mpol = int(nc_obj.variables["mpol"][:])
        ntor = int(nc_obj.variables["ntor"][:])
        booz_obj = _new_boozer_object_with_auto_fallback(bxform, vmec_fname, nc_obj)
        booz_obj.mboz = int(2 * mpol)
        booz_obj.nboz = int(2 * ntor)
        booz_obj.run()
        return nc_obj, _vmec_splines(nc_obj, booz_obj)
    except Exception:
        nc_obj.close()
        raise


def _fieldline_scalar_profiles(
    vs: Any,
    *,
    s_val: float,
    alpha: float,
    iota_input: float | None,
    s_hat_input: float | None,
) -> _VMECFieldlineScalars:
    """Sample scalar VMEC profiles and normalize reference scales."""

    s = np.array([s_val])
    alpha_arr = np.array([alpha])
    d_pressure_d_s = vs.d_pressure_d_s(s)
    iota = vs.iota(s)
    d_iota_d_s = vs.d_iota_d_s(s)
    shat = (-2.0 * s / iota) * d_iota_d_s
    edge_toroidal_flux_over_2pi = float(-vs.phiedge / (2.0 * np.pi))
    toroidal_flux_sign = float(np.sign(edge_toroidal_flux_over_2pi))
    L_reference, B_reference, R_mag_ax = _validated_reference_scales(
        vs, edge_toroidal_flux_over_2pi
    )
    iota_input_val, s_hat_input_val = _input_iota_shear(
        iota, shat, iota_input, s_hat_input
    )
    return _VMECFieldlineScalars(
        s=s,
        ns=1,
        alpha_arr=alpha_arr,
        d_pressure_d_s=d_pressure_d_s,
        iota=iota,
        d_iota_d_s=d_iota_d_s,
        shat=shat,
        nfp=vs.nfp,
        edge_toroidal_flux_over_2pi=edge_toroidal_flux_over_2pi,
        toroidal_flux_sign=toroidal_flux_sign,
        L_reference=float(L_reference),
        B_reference=float(B_reference),
        R_mag_ax=float(R_mag_ax),
        zeta_center=-alpha / float(iota[0]),
        iota_input_val=iota_input_val,
        s_hat_input_val=s_hat_input_val,
        G=vs.Gfun(s),
        boozer_i=vs.Ifun(s),
    )


def _sample_boozer_mode_profiles(
    vs: Any, scalars: _VMECFieldlineScalars
) -> _BoozerModeProfiles:
    """Sample Boozer coefficients and their radial derivatives."""

    (
        rmnc_b,
        zmns_b,
        numns_b,
        d_rmnc_b_d_s,
        d_zmns_b_d_s,
        d_numns_b_d_s,
        gmnc_b,
        bmnc_b,
        d_bmnc_b_d_s,
    ) = _sample_boozer_mode_table(vs, scalars.s, scalars.ns)
    return _BoozerModeProfiles(
        rmnc_b=rmnc_b,
        zmns_b=zmns_b,
        numns_b=numns_b,
        d_rmnc_b_d_s=d_rmnc_b_d_s,
        d_zmns_b_d_s=d_zmns_b_d_s,
        d_numns_b_d_s=d_numns_b_d_s,
        gmnc_b=gmnc_b,
        bmnc_b=bmnc_b,
        d_bmnc_b_d_s=d_bmnc_b_d_s,
    )


def _fieldline_coordinates_and_flip(
    *,
    theta1d: np.ndarray,
    scalars: _VMECFieldlineScalars,
    xm_b: np.ndarray,
    xn_b: np.ndarray,
    profiles: _BoozerModeProfiles,
    isaxisym: bool,
) -> _BoozerFieldlineCoordinates:
    """Return field-line Boozer coordinates and axisymmetric orientation."""

    theta_b, phi_b = _fieldline_boozer_coordinates(
        theta1d, scalars.alpha_arr, scalars.iota
    )
    flipit = _axisym_flip_required(
        isaxisym=isaxisym,
        xm_b=xm_b,
        xn_b=xn_b,
        theta_b=theta_b,
        phi_b=phi_b,
        rmnc_b=profiles.rmnc_b,
        zmns_b=profiles.zmns_b,
    )
    return _BoozerFieldlineCoordinates(
        theta_b=theta_b,
        phi_b=phi_b,
        flipit=bool(flipit),
    )


def _fieldline_trig_samples(
    xm_b: np.ndarray, xn_b: np.ndarray, coords: _BoozerFieldlineCoordinates
) -> _BoozerTrigSamples:
    """Return Boozer angle basis arrays used by tensor mode sums."""

    angle_b = _boozer_mode_angle(
        xm_b, xn_b, coords.theta_b, coords.phi_b, flipit=coords.flipit
    )
    (
        cosangle_b,
        sinangle_b,
        mcosangle_b,
        msinangle_b,
        ncosangle_b,
        nsinangle_b,
    ) = _boozer_trig_basis(xm_b, xn_b, angle_b)
    return _BoozerTrigSamples(
        cosangle_b=cosangle_b,
        sinangle_b=sinangle_b,
        mcosangle_b=mcosangle_b,
        msinangle_b=msinangle_b,
        ncosangle_b=ncosangle_b,
        nsinangle_b=nsinangle_b,
    )


def _sample_fieldline_boozer_state(
    vs: Any,
    scalars: _VMECFieldlineScalars,
    *,
    theta1d: np.ndarray,
    isaxisym: bool,
) -> _BoozerFieldlineSamples:
    """Build Boozer mode tables, field-line coordinates, and tensor sums."""

    xm_b = vs.xm_b
    xn_b = vs.xn_b
    profiles = _sample_boozer_mode_profiles(vs, scalars)
    coords = _fieldline_coordinates_and_flip(
        theta1d=theta1d,
        scalars=scalars,
        xm_b=xm_b,
        xn_b=xn_b,
        profiles=profiles,
        isaxisym=isaxisym,
    )
    trig = _fieldline_trig_samples(xm_b, xn_b, coords)
    tensors = _fieldline_boozer_tensors(
        rmnc_b=profiles.rmnc_b,
        zmns_b=profiles.zmns_b,
        numns_b=profiles.numns_b,
        d_rmnc_b_d_s=profiles.d_rmnc_b_d_s,
        d_zmns_b_d_s=profiles.d_zmns_b_d_s,
        d_numns_b_d_s=profiles.d_numns_b_d_s,
        gmnc_b=profiles.gmnc_b,
        bmnc_b=profiles.bmnc_b,
        d_bmnc_b_d_s=profiles.d_bmnc_b_d_s,
        cosangle_b=trig.cosangle_b,
        sinangle_b=trig.sinangle_b,
        mcosangle_b=trig.mcosangle_b,
        msinangle_b=trig.msinangle_b,
        ncosangle_b=trig.ncosangle_b,
        nsinangle_b=trig.nsinangle_b,
    )
    return _BoozerFieldlineSamples(
        xm_b=xm_b,
        xn_b=xn_b,
        rmnc_b=profiles.rmnc_b,
        zmns_b=profiles.zmns_b,
        numns_b=profiles.numns_b,
        d_rmnc_b_d_s=profiles.d_rmnc_b_d_s,
        d_zmns_b_d_s=profiles.d_zmns_b_d_s,
        d_numns_b_d_s=profiles.d_numns_b_d_s,
        gmnc_b=profiles.gmnc_b,
        bmnc_b=profiles.bmnc_b,
        d_bmnc_b_d_s=profiles.d_bmnc_b_d_s,
        theta_b=coords.theta_b,
        phi_b=coords.phi_b,
        flipit=coords.flipit,
        tensors=tensors,
        R_b=tensors.R_b,
        Z_b=tensors.Z_b,
        nu_b=tensors.nu_b,
        Vprime=profiles.gmnc_b[:, 0],
        mnmax_b=profiles.rmnc_b.shape[1],
    )


__all__ = [
    "VMEC_BOOZER_STATE_PARAMETER_FAMILIES",
    "VMEC_BOOZER_STATE_PARAMETER_NAMES",
    "VMEC_STATE_IN_MEMORY_WOUT_PATH",
    "_VMECStateContext",
    "_length_two_params",
    "_load_vmec_state_context",
    "_perturb_vmec_state",
    "_replace_vmec_boozer_state_coefficient",
    "_resolve_vmec_state_indices",
    "_vmec_boozer_state_array",
    "_vmec_boozer_state_parameter_name",
]
