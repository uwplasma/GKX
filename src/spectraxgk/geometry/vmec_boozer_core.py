"""vmex (VMEC-in-JAX) to Boozer equal-arc core-profile bridge."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import importlib
from importlib import resources as importlib_resources
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from spectraxgk.geometry.core import FluxTubeGeometryData
from spectraxgk.geometry.flux_tube_contract import flux_tube_geometry_from_mapping
from spectraxgk.geometry.numerics import (
    _boozer_half_mesh_s_grid,
    _cumulative_trapezoid,
    _evaluate_boozer_cosine_series_on_field_line,
    _interp_equal_arc_profile,
    _interp_radial,
    _radial_derivative_array,
    _radial_derivative_profile,
)
from spectraxgk.geometry.vmec_boozer_constants import (
    _VMEC_BOOZER_PARITY_MIN_MODE_COUNT,
    _cached_booz_xform_constants,
    _grid_mode_limits,
    prewarm_vmec_boozer_equal_arc_cache,
)
from spectraxgk.geometry.vmec_boozer_derivatives import (
    boozer_cartesian_derivatives,
    boozer_coordinate_gradients,
    evaluate_boozer_field_line_derivatives,
)


def _import_vmex_boozer_modules() -> tuple[Any, Any]:
    """Import the vmex Boozer-tables seam and the booz_xform_jax API."""

    try:
        boozer_tables_mod = importlib.import_module("vmex.core.boozer_tables")
        bx = importlib.import_module("booz_xform_jax.jax_api")
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "vmex and booz_xform_jax are required for the VMEC/Boozer bridge"
        ) from exc
    return boozer_tables_mod, bx


def resolve_vmex_case_input_path(case_name: str) -> Path:
    """Resolve a case name (path, vmex packaged resource, or tracked example).

    Accepts an existing filesystem path, a ``vmex.resources/input.<name>``
    resource (e.g. ``nfp4_QH_warm_start``), or ``examples/vmec/input.<name>``.
    """

    direct = Path(str(case_name)).expanduser()
    if direct.is_file():
        return direct.resolve()
    name = str(case_name).removeprefix("input.")
    try:
        packaged_dir: Any = importlib_resources.files("vmex.resources")
    except ModuleNotFoundError:  # pragma: no cover - vmex resources unavailable
        packaged_dir = Path("/nonexistent")
    repo_examples = Path(__file__).resolve().parents[3] / "examples" / "vmec"
    for candidate_dir in (packaged_dir, repo_examples):
        candidate = candidate_dir / f"input.{name}"
        if candidate.is_file():
            return Path(str(candidate)).resolve()
    known = {
        str(entry.name).removeprefix("input.")
        for candidate_dir in (packaged_dir, repo_examples)
        if candidate_dir.is_dir()
        for entry in candidate_dir.iterdir()
        if str(entry.name).startswith("input.")
    }
    raise ValueError(
        f"unknown VMEC/Boozer case {case_name!r}; pass an input-file path or one of: "
        + ", ".join(sorted(known))
    )


@lru_cache(maxsize=8)
def _load_solved_vmex_case_cached(input_path: str) -> tuple[Any, Any, Any, Any]:
    """Parse and converge one VMEC input file (cached; solves are expensive)."""

    _import_vmex_boozer_modules()
    vmex = importlib.import_module("vmex")
    eq = vmex.optimize.solve_equilibrium(vmex.VmecInput.from_file(input_path))
    return eq.inp, eq.state, eq.runtime, eq.wout


def load_solved_vmex_case(case_name: str) -> tuple[Any, Any, Any, Any]:
    """Solve a named VMEC case with vmex; return ``(inp, state, runtime, wout)``.

    The canonical loader for in-memory differentiable VMEC/Boozer geometry:
    the parsed :class:`vmex.VmecInput`, the converged spectral state, the
    matching solver runtime, and the host-NumPy wout dataset.
    """

    return _load_solved_vmex_case_cached(str(resolve_vmex_case_input_path(case_name)))


@dataclass(frozen=True)
class _BoozXformInputs:
    """Duck-typed half-mesh input bundle for ``booz_xform_jax.jax_api``."""

    rmnc: jnp.ndarray
    zmns: jnp.ndarray
    lmns: jnp.ndarray
    bmnc: jnp.ndarray
    bsubumnc: jnp.ndarray
    bsubvmnc: jnp.ndarray
    iota: jnp.ndarray
    xm: np.ndarray
    xn: np.ndarray
    xm_nyq: np.ndarray
    xn_nyq: np.ndarray
    nfp: int
    bmns: Any = None


@dataclass(frozen=True)
class _BoozerCoreRequest:
    ntheta: int
    mboz: int
    nboz: int
    base_Rcos: jnp.ndarray
    ns_full: int
    surface_index: int
    torflux: float


@dataclass(frozen=True)
class _ReferenceScales:
    length: float
    magnetic_field: float
    edge_toroidal_flux_over_2pi: float


@dataclass(frozen=True)
class _BoozerRadialProfiles:
    bmnc_b: jnp.ndarray
    rmnc_b: jnp.ndarray
    zmns_b: jnp.ndarray
    numns_b: jnp.ndarray
    d_bmnc_b_d_s: jnp.ndarray
    d_rmnc_b_d_s: jnp.ndarray
    d_zmns_b_d_s: jnp.ndarray
    d_numns_b_d_s: jnp.ndarray
    iota: jnp.ndarray
    d_iota_ds: jnp.ndarray
    iota_safe: jnp.ndarray
    s_hat: jnp.ndarray
    boozer_i: jnp.ndarray
    boozer_g: jnp.ndarray


@dataclass(frozen=True)
class _EqualArcFieldLine:
    theta_closed: jnp.ndarray
    theta_uniform_closed: jnp.ndarray
    theta_eqarc: jnp.ndarray
    theta: jnp.ndarray
    mod_b_safe: jnp.ndarray
    sqrt_g_booz: jnp.ndarray
    gradpar_eqarc: jnp.ndarray
    gradpar: jnp.ndarray
    bmag: jnp.ndarray
    bmag_safe: jnp.ndarray
    bgrad: jnp.ndarray
    jacobian: jnp.ndarray


@dataclass(frozen=True)
class _MetricDriftProfiles:
    gds2: jnp.ndarray
    gds21: jnp.ndarray
    gds22: jnp.ndarray
    grho: jnp.ndarray
    cvdrift: jnp.ndarray
    gbdrift: jnp.ndarray
    cvdrift0: jnp.ndarray
    gbdrift0: jnp.ndarray


@dataclass(frozen=True)
class _MetricDifferentialState:
    spectral: Any
    etf: jnp.ndarray
    etf_safe: jnp.ndarray
    eps: jnp.ndarray
    g_sup_psi_psi_safe: jnp.ndarray
    shear_phase: jnp.ndarray
    local_shear_l1: jnp.ndarray
    metric_bmag_sq: jnp.ndarray


@dataclass(frozen=True)
class _RawMetricProfiles:
    gds2: jnp.ndarray
    gds21: jnp.ndarray
    gds22: jnp.ndarray
    grho: jnp.ndarray


@dataclass(frozen=True)
class _RawDriftProfiles:
    cvdrift: jnp.ndarray
    cvdrift0: jnp.ndarray


def _resolve_boozer_core_request(
    state: Any,
    *,
    surface_index: int | None,
    torflux: float | None,
    ntheta: int,
    mboz: int,
    nboz: int,
    surface_stencil_width: int | None,
) -> _BoozerCoreRequest:
    ntheta_int = int(ntheta)
    if ntheta_int < 4:
        raise ValueError("ntheta must be >= 4")
    mboz_int = int(mboz)
    nboz_int = int(nboz)
    if (
        mboz_int < _VMEC_BOOZER_PARITY_MIN_MODE_COUNT
        or nboz_int < _VMEC_BOOZER_PARITY_MIN_MODE_COUNT
    ):
        raise ValueError(
            "mboz and nboz must both be >= "
            f"{_VMEC_BOOZER_PARITY_MIN_MODE_COUNT} for VMEC/Boozer parity gates"
        )

    base_Rcos = jnp.asarray(state.R_cos)
    if base_Rcos.ndim != 2:
        raise RuntimeError("vmex state R_cos array must be two-dimensional")
    ns_full = int(base_Rcos.shape[0])
    if ns_full < 3:
        raise RuntimeError("vmex state needs at least three radial surfaces")

    sidx = (
        max(1, min(ns_full // 2, ns_full - 2))
        if surface_index is None
        else int(surface_index)
    )
    if not (0 < sidx < ns_full - 1):
        raise ValueError("surface_index must be an interior VMEC radial index")
    s_value = (
        float(sidx) / float(max(ns_full - 1, 1)) if torflux is None else float(torflux)
    )
    if not (0.0 < s_value < 1.0):
        raise ValueError("torflux must lie inside (0, 1)")
    if surface_stencil_width is not None and int(surface_stencil_width) < 3:
        raise ValueError("surface_stencil_width must be >= 3 when provided")
    return _BoozerCoreRequest(
        ntheta=ntheta_int,
        mboz=mboz_int,
        nboz=nboz_int,
        base_Rcos=base_Rcos,
        ns_full=ns_full,
        surface_index=sidx,
        torflux=s_value,
    )


def _resolve_reference_scales(
    wout: Any,
    *,
    reference_length: float | None,
    reference_b: float | None,
) -> _ReferenceScales:
    raw_length = (
        float(getattr(wout, "Aminor_p", 1.0))
        if reference_length is None
        else float(reference_length)
    )
    length = raw_length if np.isfinite(raw_length) and abs(raw_length) > 0.0 else 1.0
    phi_profile = np.asarray(getattr(wout, "phi", [0.0, np.pi]), dtype=float)
    edge_toroidal_flux_over_2pi = -float(phi_profile[-1]) / (2.0 * np.pi)
    if reference_b is None:
        raw_b = 2.0 * abs(edge_toroidal_flux_over_2pi) / (length * length)
        magnetic_field = raw_b if np.isfinite(raw_b) and abs(raw_b) > 0.0 else 1.0
    else:
        magnetic_field = float(reference_b)
    magnetic_field = (
        magnetic_field
        if np.isfinite(magnetic_field) and abs(magnetic_field) > 0.0
        else 1.0
    )
    return _ReferenceScales(
        length=length,
        magnetic_field=magnetic_field,
        edge_toroidal_flux_over_2pi=edge_toroidal_flux_over_2pi,
    )


def _surface_indices_for_stencil(
    *,
    surface_stencil_width: int | None,
    ns_full: int,
    torflux: float,
) -> jnp.ndarray | None:
    if surface_stencil_width is None:
        return None
    ns_b_est = max(1, int(ns_full) - 1)
    width = min(int(surface_stencil_width), ns_b_est)
    center = int(round(float(torflux) * float(ns_b_est) - 0.5))
    half_width = width // 2
    start = max(0, min(center - half_width, ns_b_est - width))
    return jnp.arange(start, start + width, dtype=jnp.int32)


def _boozer_xform_inputs_from_state(
    state: Any, runtime: Any, *, inp: Any, wout: Any,
    boozer_tables_mod: Any, ns_full: int,
) -> _BoozXformInputs:
    """Stack traceable vmex Boozer tables over all half-mesh rows ``1..ns-1``."""

    if bool(getattr(getattr(runtime, "resolution", None), "lasym", False)):
        raise NotImplementedError(
            "the vmex Boozer-tables bridge is stellarator-symmetric only"
        )
    rows = [
        boozer_tables_mod.boozer_input_tables(state, runtime, int(j))
        for j in range(1, int(ns_full))
    ]
    nfp_raw = getattr(inp, "nfp", None)
    if nfp_raw is None:
        nfp_raw = getattr(wout, "nfp", 1)
    stacked = {
        key: jnp.stack([jnp.asarray(row[key]) for row in rows])
        for key in ("rmnc", "zmns", "lmns", "bmnc", "bsubumnc", "bsubvmnc", "iota")
    }
    xm = np.asarray(rows[0]["xm"], dtype=np.int32)
    xn = np.asarray(rows[0]["xn"], dtype=np.int32)
    return _BoozXformInputs(
        xm=xm, xn=xn, xm_nyq=xm, xn_nyq=xn, nfp=int(nfp_raw), bmns=None, **stacked
    )


def _run_boozer_transform_from_state(
    state: Any,
    runtime: Any,
    inp: Any,
    wout: Any,
    request: _BoozerCoreRequest,
    *,
    jit: bool,
    surface_stencil_width: int | None,
) -> tuple[dict[str, Any], jnp.ndarray | None]:
    boozer_tables_mod, bx = _import_vmex_boozer_modules()
    surface_indices = _surface_indices_for_stencil(
        surface_stencil_width=surface_stencil_width,
        ns_full=request.ns_full,
        torflux=request.torflux,
    )
    inputs = _boozer_xform_inputs_from_state(
        state,
        runtime,
        inp=inp,
        wout=wout,
        boozer_tables_mod=boozer_tables_mod,
        ns_full=request.ns_full,
    )
    asym = bool(inputs.bmns is not None)
    try:
        resolution = runtime.resolution
        ntheta1, nzeta = int(resolution.ntheta1), int(resolution.nzeta)
        m_max, n_max = _grid_mode_limits(ntheta1, nzeta)
        if (n_max + 1) + m_max * (2 * n_max + 1) != int(inputs.xm.size):
            raise ValueError("cached mode table does not match vmex Boozer tables")
        constants, grids = _cached_booz_xform_constants(
            nfp=int(inputs.nfp),
            ntheta1=ntheta1,
            nzeta=nzeta,
            mboz=request.mboz,
            nboz=request.nboz,
            asym=asym,
        )
    except (AttributeError, ModuleNotFoundError, ValueError):
        constants, grids = bx.prepare_booz_xform_constants_from_inputs(
            inputs=inputs,
            mboz=request.mboz,
            nboz=request.nboz,
            asym=asym,
        )
    out = bx.booz_xform_from_inputs(
        inputs=inputs,
        constants=constants,
        grids=grids,
        surface_indices=surface_indices,
        jit=bool(jit),
    )
    return out, surface_indices


def _boozer_surface_grid(
    out: dict[str, Any],
    *,
    base_dtype: Any,
    ns_full: int,
) -> tuple[jnp.ndarray, jnp.ndarray, float]:
    bmnc_b_all = jnp.asarray(out["bmnc_b"], dtype=base_dtype)
    if bmnc_b_all.ndim != 2:
        raise RuntimeError(
            "booz_xform_jax bmnc_b output must have shape (surface, mode)"
        )
    ns_b = int(bmnc_b_all.shape[0])
    if ns_b < 2:
        raise RuntimeError("booz_xform_jax output needs at least two radial surfaces")
    ns_b_full = max(int(ns_full) - 1, int(ns_b))
    s_half = _boozer_half_mesh_s_grid(
        out.get("jlist"),
        ns_b=ns_b,
        ns_b_full=ns_b_full,
        dtype=base_dtype,
    )
    radial_spacing = 1.0 / float(max(ns_b_full, 1))
    return bmnc_b_all, s_half, radial_spacing


def _interpolate_boozer_radial_profiles(
    out: dict[str, Any],
    request: _BoozerCoreRequest,
    *,
    s_half: jnp.ndarray,
    radial_spacing: float,
) -> _BoozerRadialProfiles:
    dtype = request.base_Rcos.dtype
    s_value = request.torflux
    bmnc_b = _interp_radial(jnp.asarray(out["bmnc_b"], dtype=dtype), s_half, s_value)

    def interp_profile(name: str) -> jnp.ndarray:
        return _interp_radial(jnp.asarray(out[name], dtype=dtype), s_half, s_value)

    def interp_derivative(name: str) -> jnp.ndarray:
        return _interp_radial(
            _radial_derivative_array(
                jnp.asarray(out[name], dtype=dtype), radial_spacing
            ),
            s_half,
            s_value,
        )

    iota_profile = jnp.asarray(out["iota_b"], dtype=dtype)
    iota = _interp_radial(iota_profile, s_half, s_value)
    d_iota_ds = _interp_radial(
        _radial_derivative_profile(iota_profile, radial_spacing),
        s_half,
        s_value,
    )
    iota_safe = jnp.where(
        jnp.abs(iota) < 1.0e-12,
        jnp.sign(iota + 1.0e-30) * 1.0e-12,
        iota,
    )
    s_hat = -2.0 * jnp.asarray(s_value, dtype=dtype) * d_iota_ds / iota_safe
    boozer_i = _interp_radial(jnp.asarray(out["buco_b"], dtype=dtype), s_half, s_value)
    boozer_g = _interp_radial(jnp.asarray(out["bvco_b"], dtype=dtype), s_half, s_value)
    return _BoozerRadialProfiles(
        bmnc_b=bmnc_b,
        rmnc_b=interp_profile("rmnc_b"),
        zmns_b=interp_profile("zmns_b"),
        numns_b=-interp_profile("pmns_b"),
        d_bmnc_b_d_s=interp_derivative("bmnc_b"),
        d_rmnc_b_d_s=interp_derivative("rmnc_b"),
        d_zmns_b_d_s=interp_derivative("zmns_b"),
        d_numns_b_d_s=-interp_derivative("pmns_b"),
        iota=iota,
        d_iota_ds=d_iota_ds,
        iota_safe=iota_safe,
        s_hat=s_hat,
        boozer_i=boozer_i,
        boozer_g=boozer_g,
    )


def _build_equal_arc_field_line(
    out: dict[str, Any],
    request: _BoozerCoreRequest,
    scales: _ReferenceScales,
    profiles: _BoozerRadialProfiles,
    *,
    alpha: float,
) -> _EqualArcFieldLine:
    dtype = request.base_Rcos.dtype
    theta_closed = jnp.linspace(-jnp.pi, jnp.pi, request.ntheta + 1, dtype=dtype)
    mod_b, _dmod_b_dtheta = _evaluate_boozer_cosine_series_on_field_line(
        theta_closed,
        coeffs=profiles.bmnc_b,
        ixm_b=jnp.asarray(out["ixm_b"]),
        ixn_b=jnp.asarray(out["ixn_b"]),
        iota=profiles.iota_safe,
        alpha=float(alpha),
    )
    eps = jnp.asarray(1.0e-30, dtype=dtype)
    mod_b_safe = jnp.maximum(jnp.abs(mod_b), eps)
    sqrt_g_booz = (profiles.boozer_g + profiles.iota_safe * profiles.boozer_i) / (
        mod_b_safe * mod_b_safe
    )
    gradpar_raw = jnp.abs(
        jnp.asarray(float(scales.length), dtype=dtype)
        * profiles.iota_safe
        / jnp.maximum(jnp.abs(mod_b_safe * sqrt_g_booz), eps)
    )
    inv_gradpar_int = _cumulative_trapezoid(1.0 / gradpar_raw, theta_closed)
    gradpar_eqarc = 2.0 * jnp.pi / jnp.maximum(inv_gradpar_int[-1], eps)
    theta_eqarc = gradpar_eqarc * inv_gradpar_int - jnp.pi
    theta_uniform_closed = jnp.linspace(-jnp.pi, jnp.pi, request.ntheta + 1, dtype=dtype)
    bmag_closed = jnp.asarray(
        _interp_equal_arc_profile(
            theta_uniform_closed,
            theta_eqarc,
            mod_b_safe / float(scales.magnetic_field),
        )
    )
    theta = theta_uniform_closed[:-1]
    bmag = bmag_closed[:-1]
    gradpar = gradpar_eqarc * jnp.ones_like(theta)
    dtheta = 2.0 * jnp.pi / float(request.ntheta)
    bmag_safe = jnp.maximum(jnp.abs(bmag), eps)
    wave_number = 2.0 * jnp.pi * jnp.fft.fftfreq(request.ntheta, d=float(dtheta))
    dbmag_dtheta = jnp.real(jnp.fft.ifft(1j * wave_number * jnp.fft.fft(bmag)))
    bgrad = gradpar_eqarc * dbmag_dtheta / bmag_safe
    dpsidrho = (
        2.0
        * jnp.sqrt(jnp.asarray(request.torflux, dtype=dtype))
        * jnp.asarray(scales.edge_toroidal_flux_over_2pi, dtype=dtype)
    )
    drhodpsi = 1.0 / jnp.maximum(jnp.abs(dpsidrho), eps)
    jacobian = 1.0 / jnp.maximum(jnp.abs(drhodpsi * gradpar_eqarc * bmag_safe), eps)
    return _EqualArcFieldLine(
        theta_closed=theta_closed,
        theta_uniform_closed=theta_uniform_closed,
        theta_eqarc=theta_eqarc,
        theta=theta,
        mod_b_safe=mod_b_safe,
        sqrt_g_booz=sqrt_g_booz,
        gradpar_eqarc=gradpar_eqarc,
        gradpar=gradpar,
        bmag=bmag,
        bmag_safe=bmag_safe,
        bgrad=bgrad,
        jacobian=jacobian,
    )


def _metric_differential_state(
    out: dict[str, Any],
    request: _BoozerCoreRequest,
    scales: _ReferenceScales,
    profiles: _BoozerRadialProfiles,
    equal_arc: _EqualArcFieldLine,
    *,
    alpha: float,
) -> _MetricDifferentialState:
    """Evaluate Boozer differential geometry used by metrics and drifts."""

    dtype = request.base_Rcos.dtype
    eps = jnp.asarray(1.0e-30, dtype=dtype)
    spectral = evaluate_boozer_field_line_derivatives(
        out,
        theta_closed=equal_arc.theta_closed,
        alpha=alpha,
        iota_safe=profiles.iota_safe,
        base_dtype=dtype,
        bmnc_b=profiles.bmnc_b,
        d_bmnc_b_d_s=profiles.d_bmnc_b_d_s,
        rmnc_b=profiles.rmnc_b,
        d_rmnc_b_d_s=profiles.d_rmnc_b_d_s,
        zmns_b=profiles.zmns_b,
        d_zmns_b_d_s=profiles.d_zmns_b_d_s,
        numns_b=profiles.numns_b,
        d_numns_b_d_s=profiles.d_numns_b_d_s,
    )
    cartesian = boozer_cartesian_derivatives(spectral)
    etf = jnp.asarray(scales.edge_toroidal_flux_over_2pi, dtype=dtype)
    etf_floor = jnp.asarray(1.0e-12, dtype=dtype)
    etf_safe = jnp.where(jnp.abs(etf) < etf_floor, jnp.sign(etf + eps) * etf_floor, etf)
    gradients = boozer_coordinate_gradients(
        spectral=spectral,
        cartesian=cartesian,
        sqrt_g_booz=equal_arc.sqrt_g_booz,
        etf_safe=etf_safe,
    )
    grad_psi_x = gradients.grad_psi_x
    grad_psi_y = gradients.grad_psi_y
    grad_psi_z = gradients.grad_psi_z
    g_sup_psi_psi = grad_psi_x**2 + grad_psi_y**2 + grad_psi_z**2
    g_sup_psi_psi_safe = jnp.maximum(g_sup_psi_psi, eps)

    zeta_center = -jnp.asarray(float(alpha), dtype=dtype) / profiles.iota_safe
    shear_phase = spectral.phi_b - zeta_center
    grad_alpha_x = (
        -shear_phase * profiles.d_iota_ds * grad_psi_x / etf_safe
        + gradients.grad_theta_x
        - profiles.iota_safe * gradients.grad_phi_x
    )
    grad_alpha_y = (
        -shear_phase * profiles.d_iota_ds * grad_psi_y / etf_safe
        + gradients.grad_theta_y
        - profiles.iota_safe * gradients.grad_phi_y
    )
    grad_alpha_z = (
        -shear_phase * profiles.d_iota_ds * grad_psi_z / etf_safe
        + gradients.grad_theta_z
        - profiles.iota_safe * gradients.grad_phi_z
    )
    grad_alpha_dot_grad_psi = (
        grad_alpha_x * grad_psi_x
        + grad_alpha_y * grad_psi_y
        + grad_alpha_z * grad_psi_z
    )
    local_shear_l1 = grad_alpha_dot_grad_psi / g_sup_psi_psi_safe
    metric_bmag_sq = equal_arc.mod_b_safe * equal_arc.mod_b_safe
    return _MetricDifferentialState(
        spectral=spectral,
        etf=etf,
        etf_safe=etf_safe,
        eps=eps,
        g_sup_psi_psi_safe=g_sup_psi_psi_safe,
        shear_phase=shear_phase,
        local_shear_l1=local_shear_l1,
        metric_bmag_sq=metric_bmag_sq,
    )


def _raw_metric_profiles(
    request: _BoozerCoreRequest,
    scales: _ReferenceScales,
    profiles: _BoozerRadialProfiles,
    state: _MetricDifferentialState,
) -> _RawMetricProfiles:
    """Compute raw Boozer metric coefficients before equal-arc remap."""

    dtype = request.base_Rcos.dtype
    s_arr = jnp.asarray(request.torflux, dtype=dtype)
    L = jnp.asarray(float(scales.length), dtype=dtype)
    Bref = jnp.asarray(float(scales.magnetic_field), dtype=dtype)
    gds2 = (
        (
            state.metric_bmag_sq / state.g_sup_psi_psi_safe
            + state.g_sup_psi_psi_safe * state.local_shear_l1**2
        )
        * L
        * L
        * s_arr
    )
    gds21 = state.g_sup_psi_psi_safe * state.local_shear_l1 * profiles.s_hat / Bref
    gds22 = (
        state.g_sup_psi_psi_safe
        * profiles.s_hat
        * profiles.s_hat
        / (L * L * Bref * Bref * s_arr)
    )
    grho = jnp.sqrt(state.g_sup_psi_psi_safe / (L * L * Bref * Bref * s_arr))
    return _RawMetricProfiles(gds2=gds2, gds21=gds21, gds22=gds22, grho=grho)


def _raw_drift_profiles(
    request: _BoozerCoreRequest,
    scales: _ReferenceScales,
    profiles: _BoozerRadialProfiles,
    equal_arc: _EqualArcFieldLine,
    state: _MetricDifferentialState,
) -> _RawDriftProfiles:
    """Compute raw curvature drift coefficients before equal-arc remap."""

    dtype = request.base_Rcos.dtype
    s_arr = jnp.asarray(request.torflux, dtype=dtype)
    L = jnp.asarray(float(scales.length), dtype=dtype)
    Bref = jnp.asarray(float(scales.magnetic_field), dtype=dtype)
    boozer_current_sum = profiles.boozer_g + profiles.iota_safe * profiles.boozer_i
    d_sqrt_g_booz_d_theta = (
        -2.0
        * boozer_current_sum
        * state.spectral.d_mod_b_d_theta
        / (equal_arc.mod_b_safe**3)
    )
    d_sqrt_g_booz_d_phi = (
        -2.0
        * boozer_current_sum
        * state.spectral.d_mod_b_d_phi
        / (equal_arc.mod_b_safe**3)
    )
    curvature_numerator = (
        profiles.boozer_g * d_sqrt_g_booz_d_theta
        - profiles.boozer_i * d_sqrt_g_booz_d_phi
    )
    curvature_denom = 2.0 * equal_arc.sqrt_g_booz * boozer_current_sum
    curvature_denom_safe = jnp.where(
        jnp.abs(curvature_denom) < state.eps,
        jnp.sign(curvature_denom + state.eps) * state.eps,
        curvature_denom,
    )
    kappa_g = curvature_numerator / curvature_denom_safe
    local_shear_l0 = -(
        state.local_shear_l1 + profiles.d_iota_ds / state.etf_safe * state.shear_phase
    )
    kappa_n = (
        state.spectral.d_mod_b_d_s / (equal_arc.mod_b_safe * state.etf_safe)
        + local_shear_l0 * kappa_g
    )
    b_cross_kappa_dot_grad_alpha = (
        kappa_n + kappa_g * state.local_shear_l1
    ) * state.metric_bmag_sq
    b_cross_kappa_dot_grad_psi = kappa_g * state.metric_bmag_sq
    toroidal_flux_sign = jnp.sign(state.etf)
    sqrt_s = jnp.sqrt(s_arr)
    cvdrift0 = (
        -b_cross_kappa_dot_grad_psi
        * 2.0
        * profiles.s_hat
        / jnp.maximum(state.metric_bmag_sq * sqrt_s, state.eps)
        * toroidal_flux_sign
    )
    cvdrift = (
        -2.0
        * Bref
        * L
        * L
        * sqrt_s
        * b_cross_kappa_dot_grad_alpha
        / state.metric_bmag_sq
        * toroidal_flux_sign
    )
    return _RawDriftProfiles(cvdrift=cvdrift, cvdrift0=cvdrift0)


def _equal_arc_open_profile(
    equal_arc: _EqualArcFieldLine, profile: jnp.ndarray
) -> jnp.ndarray:
    return _interp_equal_arc_profile(
        equal_arc.theta_uniform_closed,
        equal_arc.theta_eqarc,
        profile,
    )[:-1]


def _pack_metric_drift_profiles(
    request: _BoozerCoreRequest,
    equal_arc: _EqualArcFieldLine,
    metrics: _RawMetricProfiles,
    drifts: _RawDriftProfiles,
) -> _MetricDriftProfiles:
    """Remap raw metric/drift coefficients onto the open equal-arc grid."""

    dtype = request.base_Rcos.dtype
    drift_loader_factor = jnp.asarray(0.5, dtype=dtype)
    cvdrift = drift_loader_factor * _equal_arc_open_profile(equal_arc, drifts.cvdrift)
    cvdrift0 = drift_loader_factor * _equal_arc_open_profile(
        equal_arc, drifts.cvdrift0
    )
    return _MetricDriftProfiles(
        gds2=_equal_arc_open_profile(equal_arc, metrics.gds2),
        gds21=_equal_arc_open_profile(equal_arc, metrics.gds21),
        gds22=_equal_arc_open_profile(equal_arc, metrics.gds22),
        grho=_equal_arc_open_profile(equal_arc, metrics.grho),
        cvdrift=cvdrift,
        gbdrift=cvdrift,
        cvdrift0=cvdrift0,
        gbdrift0=cvdrift0,
    )


def _build_metric_and_drift_profiles(
    out: dict[str, Any],
    request: _BoozerCoreRequest,
    scales: _ReferenceScales,
    profiles: _BoozerRadialProfiles,
    equal_arc: _EqualArcFieldLine,
    *,
    alpha: float,
) -> _MetricDriftProfiles:
    state = _metric_differential_state(
        out,
        request,
        scales,
        profiles,
        equal_arc,
        alpha=alpha,
    )
    metrics = _raw_metric_profiles(request, scales, profiles, state)
    drifts = _raw_drift_profiles(request, scales, profiles, equal_arc, state)
    return _pack_metric_drift_profiles(request, equal_arc, metrics, drifts)


def _assemble_core_mapping(
    request: _BoozerCoreRequest,
    scales: _ReferenceScales,
    profiles: _BoozerRadialProfiles,
    equal_arc: _EqualArcFieldLine,
    metric_drift: _MetricDriftProfiles,
    *,
    surface_stencil_width: int | None,
    surface_indices: jnp.ndarray | None,
) -> dict[str, Any]:
    eps = jnp.asarray(1.0e-30, dtype=request.base_Rcos.dtype)
    return {
        "theta": equal_arc.theta,
        "theta_equal_arc_closed": equal_arc.theta_eqarc,
        "theta_uniform_closed": equal_arc.theta_uniform_closed,
        "gradpar": equal_arc.gradpar,
        "bmag": equal_arc.bmag,
        "bgrad": equal_arc.bgrad,
        "jacobian": equal_arc.jacobian,
        "gds2": metric_drift.gds2,
        "gds21": metric_drift.gds21,
        "gds22": metric_drift.gds22,
        "cvdrift": metric_drift.cvdrift,
        "gbdrift": metric_drift.gbdrift,
        "cvdrift0": metric_drift.cvdrift0,
        "gbdrift0": metric_drift.gbdrift0,
        "grho": metric_drift.grho,
        "q": 1.0 / jnp.maximum(jnp.abs(profiles.iota_safe), eps),
        "s_hat": profiles.s_hat,
        "iota": profiles.iota,
        "torflux": float(request.torflux),
        "surface_index": int(request.surface_index),
        "reference_length": float(scales.length),
        "reference_b": float(scales.magnetic_field),
        "mboz": int(request.mboz),
        "nboz": int(request.nboz),
        "surface_stencil_width": None
        if surface_stencil_width is None
        else int(surface_stencil_width),
        "boozer_surface_indices": None
        if surface_indices is None
        else [int(x) for x in np.asarray(surface_indices)],
        "field_line_convention": "Boozer theta, alpha=theta-iota*zeta, equal-arc remap",
        "scope": (
            "Boozer equal-arc bmag/gradpar/Jacobian plus zero-beta metric/drift parity; "
            "finite-beta pressure corrections and broad-equilibrium drift gates remain open"
        ),
    }


def _core_mapping_from_boozer_output(
    out: dict[str, Any],
    request: _BoozerCoreRequest,
    scales: _ReferenceScales,
    *,
    alpha: float,
    surface_stencil_width: int | None,
    surface_indices: jnp.ndarray | None,
) -> dict[str, Any]:
    """Assemble solver-facing core profiles from Boozer transform output."""

    _bmnc_b_all, s_half, radial_spacing = _boozer_surface_grid(
        out,
        base_dtype=request.base_Rcos.dtype,
        ns_full=request.ns_full,
    )
    profiles = _interpolate_boozer_radial_profiles(
        out,
        request,
        s_half=s_half,
        radial_spacing=radial_spacing,
    )
    equal_arc = _build_equal_arc_field_line(
        out,
        request,
        scales,
        profiles,
        alpha=alpha,
    )
    metric_drift = _build_metric_and_drift_profiles(
        out,
        request,
        scales,
        profiles,
        equal_arc,
        alpha=alpha,
    )
    return _assemble_core_mapping(
        request,
        scales,
        profiles,
        equal_arc,
        metric_drift,
        surface_stencil_width=surface_stencil_width,
        surface_indices=surface_indices,
    )


def vmec_jax_boozer_equal_arc_core_profiles_from_state(  # pragma: no cover
    state: Any,
    runtime: Any,
    inp: Any,
    wout: Any,
    *,
    surface_index: int | None = None,
    torflux: float | None = None,
    alpha: float = 0.0,
    ntheta: int = 32,
    mboz: int = _VMEC_BOOZER_PARITY_MIN_MODE_COUNT,
    nboz: int = _VMEC_BOOZER_PARITY_MIN_MODE_COUNT,
    jit: bool = False,
    surface_stencil_width: int | None = None,
    reference_length: float | None = None,
    reference_b: float | None = None,
) -> dict[str, Any]:
    """Return Boozer equal-arc core profiles from a solved ``vmex`` state.

    ``state``/``runtime``/``inp``/``wout`` come from
    :func:`load_solved_vmex_case` (or any :class:`vmex.optimize.Equilibrium`).
    The bridge keeps the imported VMEC/EIK conventions for the scalar/core
    field-line quantities and the zero-beta Boozer metric/drift terms from
    ``booz_xform_jax`` output; finite-beta pressure corrections and
    broader-equilibrium drift gates remain separate promotion steps.
    """

    request = _resolve_boozer_core_request(
        state,
        surface_index=surface_index,
        torflux=torflux,
        ntheta=ntheta,
        mboz=mboz,
        nboz=nboz,
        surface_stencil_width=surface_stencil_width,
    )
    scales = _resolve_reference_scales(
        wout, reference_length=reference_length, reference_b=reference_b
    )
    out, surface_indices = _run_boozer_transform_from_state(
        state,
        runtime,
        inp,
        wout,
        request,
        jit=jit,
        surface_stencil_width=surface_stencil_width,
    )
    return _core_mapping_from_boozer_output(
        out,
        request,
        scales,
        alpha=alpha,
        surface_stencil_width=surface_stencil_width,
        surface_indices=surface_indices,
    )


def flux_tube_geometry_from_vmec_boozer_state(  # pragma: no cover
    state: Any,
    runtime: Any,
    inp: Any,
    wout: Any,
    *,
    surface_index: int | None = None,
    torflux: float | None = None,
    alpha: float = 0.0,
    ntheta: int = 32,
    mboz: int = _VMEC_BOOZER_PARITY_MIN_MODE_COUNT,
    nboz: int = _VMEC_BOOZER_PARITY_MIN_MODE_COUNT,
    jit: bool = False,
    surface_stencil_width: int | None = None,
    reference_length: float | None = None,
    reference_b: float | None = None,
    source_model: str = "mode21_vmec_boozer_state",
    validate_finite: bool = True,
) -> FluxTubeGeometryData:
    """Build solver-ready geometry directly from an in-memory vmex/Boozer state."""

    mapping = vmec_jax_boozer_equal_arc_core_profiles_from_state(
        state,
        runtime,
        inp,
        wout,
        surface_index=surface_index,
        torflux=torflux,
        alpha=alpha,
        ntheta=ntheta,
        mboz=mboz,
        nboz=nboz,
        jit=jit,
        surface_stencil_width=surface_stencil_width,
        reference_length=reference_length,
        reference_b=reference_b,
    )
    return flux_tube_geometry_from_mapping(
        mapping,
        source_model=source_model,
        validate_finite=validate_finite,
    )


__all__ = [
    "flux_tube_geometry_from_vmec_boozer_state",
    "load_solved_vmex_case",
    "prewarm_vmec_boozer_equal_arc_cache",
    "resolve_vmex_case_input_path",
    "vmec_jax_boozer_equal_arc_core_profiles_from_state",
]
