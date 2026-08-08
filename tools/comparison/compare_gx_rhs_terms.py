#!/usr/bin/env python3
"""Write and compare term-resolved RHS diagnostic dumps."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import jax.numpy as jnp
import jax
from netCDF4 import Dataset

from gkx.benchmarking.shared import (
    CYCLONE_OMEGA_D_SCALE,
    CYCLONE_OMEGA_STAR_SCALE,
    CYCLONE_RHO_STAR,
    KBM_OMEGA_D_SCALE,
    KBM_OMEGA_STAR_SCALE,
    KBM_RHO_STAR,
    KINETIC_OMEGA_D_SCALE,
    KINETIC_OMEGA_STAR_SCALE,
    KINETIC_RHO_STAR,
    TEM_OMEGA_D_SCALE,
    TEM_OMEGA_STAR_SCALE,
    TEM_RHO_STAR,
    _apply_reference_hypercollisions,
    _build_initial_condition,
    _two_species_params,
)
from gkx.config import CycloneBaseCase, GeometryConfig, GridConfig, KBMBaseCase
from gkx.geometry import SlabGeometry, SAlphaGeometry, apply_imported_geometry_grid_defaults, load_imported_geometry_netcdf
from gkx.core.grid import build_spectral_grid, select_ky_grid
from gkx.workflows.runtime.toml import load_runtime_from_toml
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.moments import build_H
from gkx.operators.linear.params import LinearParams, LinearTerms, linear_terms_to_term_config
from gkx.operators.linear.params import _as_species_array
from gkx.runtime import build_runtime_geometry, build_runtime_linear_params, build_runtime_term_config
from gkx.terms.linear_terms import (
    collisions_contribution,
    curvature_gradb_contribution,
    diamagnetic_contribution,
    end_damping_contribution,
    hypercollisions_contribution,
    mirror_contribution,
    linked_streaming_contribution,
)
from gkx.terms.assembly import assemble_rhs_terms_cached, compute_fields_cached
from gkx.terms.config import TermConfig
from gkx.operators.linear.params import build_linear_params

try:
    from tools.comparison.compare_gx_imported_linear import (
        _load_gx_input_contract,
        _read_gx_output_bool,
        _resolve_imported_boundary,
        _resolve_imported_real_fft_ny,
        _resolve_internal_geometry_source,
    )
except ModuleNotFoundError:  # Direct execution adds this script directory.
    from compare_gx_imported_linear import (  # type: ignore[no-redef]
        _load_gx_input_contract,
        _read_gx_output_bool,
        _resolve_imported_boundary,
        _resolve_imported_real_fft_ny,
        _resolve_internal_geometry_source,
    )


def _load_shape(path: Path) -> dict[str, int]:
    data: dict[str, int] = {}
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) == 2:
            data[parts[0]] = int(parts[1])
    return data


def _load_bin(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.complex64)
    if raw.size != int(np.prod(shape)):
        raise ValueError(f"{path} size {raw.size} does not match expected {shape}")
    return raw.reshape(shape)


def _load_field(path: Path, nyc: int, nx: int, nz: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.complex64)
    expected = nyc * nx * nz
    if raw.size != expected:
        raise ValueError(f"{path} size {raw.size} does not match expected {expected}")
    ky_idx = np.arange(nyc)[:, None, None]
    kx_idx = np.arange(nx)[None, :, None]
    z_idx = np.arange(nz)[None, None, :]
    idxyz = ky_idx + nyc * (kx_idx + nx * z_idx)
    return raw[idxyz.ravel()].reshape(nyc, nx, nz)


def _cast_cache(cache: Any, *, real_dtype: jnp.dtype, complex_dtype: jnp.dtype) -> Any:
    def _cast(x):
        if isinstance(x, jnp.ndarray):
            if jnp.issubdtype(x.dtype, jnp.complexfloating):
                return x.astype(complex_dtype)
            if jnp.issubdtype(x.dtype, jnp.floating):
                return x.astype(real_dtype)
        return x

    return jax.tree_util.tree_map(_cast, cache)


def _reshape_gx(
    raw: np.ndarray,
    *,
    nspec: int,
    nl: int,
    nm: int,
    nyc: int,
    nx: int,
    nz: int,
) -> np.ndarray:
    nR = nyc * nx * nz
    # GX flattens moments with moment_idx = l + Nl * m (m-major).
    arr = raw.reshape((nspec, nm, nl, nR)).transpose(0, 2, 1, 3)
    ky_idx = np.arange(nyc)[:, None, None]
    kx_idx = np.arange(nx)[None, :, None]
    z_idx = np.arange(nz)[None, None, :]
    idxyz = ky_idx + nyc * (kx_idx + nx * z_idx)
    arr = arr[..., idxyz.ravel()]
    return arr.reshape((nspec, nl, nm, nyc, nx, nz))


def _summary(label: str, ref: np.ndarray, test: np.ndarray) -> None:
    diff = test - ref
    max_ref = float(np.max(np.abs(ref)))
    max_test = float(np.max(np.abs(test)))
    max_diff = float(np.max(np.abs(diff)))
    if max_ref == 0.0:
        max_rel = float("nan")
        rms_rel = float("nan")
    else:
        thresh = max_ref * 1.0e-12
        mask = np.abs(ref) > thresh
        if not np.any(mask):
            max_rel = float("nan")
            rms_rel = float("nan")
        else:
            rel = diff[mask] / ref[mask]
            max_rel = float(np.max(np.abs(rel)))
            rms_rel = float(
                np.sqrt(np.mean(np.abs(diff[mask]) ** 2))
                / (np.sqrt(np.mean(np.abs(ref[mask]) ** 2)) + 1.0e-30)
            )
    idx_diff = np.unravel_index(int(np.argmax(np.abs(diff))), diff.shape)
    ref_at = ref[idx_diff]
    test_at = test[idx_diff]
    print(
        f"{label:12s} max|ref|={max_ref:.3e} max|test|={max_test:.3e} "
        f"max|diff|={max_diff:.3e} max|rel|={max_rel:.3e} rms_rel={rms_rel:.3e} "
        f"idx={idx_diff} ref={ref_at:.3e} test={test_at:.3e}"
    )


def _infer_y0(ky: np.ndarray) -> float:
    ky_pos = np.asarray(ky, dtype=float)
    ky_pos = ky_pos[ky_pos > 0.0]
    if ky_pos.size == 0:
        raise ValueError("Need at least one positive ky value to infer y0")
    return float(1.0 / np.min(ky_pos))


def _manual_linear_contributions_from_fields(
    G: jnp.ndarray,
    cache: Any,
    params: LinearParams,
    term_cfg: TermConfig,
    *,
    phi: np.ndarray | jnp.ndarray,
    apar: np.ndarray | jnp.ndarray,
    bpar: np.ndarray | jnp.ndarray,
):
    """Assemble linear term contributions using externally supplied fields.

    GX term dumps are most useful when GKX evaluates the operator on the
    exact same ``G, phi, apar, bpar`` state rather than on a recomputed field
    solve. Keep this helper close to the comparison tool so tests can lock its
    argument/shape contract to the current linear-term APIs.
    """

    G_arr = jnp.asarray(G)
    out_dtype = jnp.result_type(G_arr, jnp.complex64)
    G_arr = jnp.asarray(G_arr, dtype=out_dtype)
    real_dtype = jnp.real(jnp.empty((), dtype=out_dtype)).dtype
    imag = jnp.asarray(1j, dtype=out_dtype)

    ns = int(G_arr.shape[0]) if G_arr.ndim == 6 else 1
    tz = _as_species_array(params.tz, ns, "tz").astype(real_dtype)
    vth = _as_species_array(params.vth, ns, "vth").astype(real_dtype)
    tprim = _as_species_array(params.tprim, ns, "tprim").astype(real_dtype)
    fprim = _as_species_array(params.fprim, ns, "fprim").astype(real_dtype)
    nu = _as_species_array(params.nu, ns, "nu").astype(real_dtype)
    nu_hyper = jnp.asarray(params.nu_hyper, dtype=real_dtype)
    nu_hyper_l = jnp.asarray(params.nu_hyper_l, dtype=real_dtype)
    nu_hyper_m = jnp.asarray(params.nu_hyper_m, dtype=real_dtype)
    nu_hyper_lm = jnp.asarray(params.nu_hyper_lm, dtype=real_dtype)
    hypercollisions_const = jnp.asarray(params.hypercollisions_const, dtype=real_dtype)
    hypercollisions_kz = jnp.asarray(params.hypercollisions_kz, dtype=real_dtype)
    damp_amp = jnp.asarray(params.damp_ends_amp, dtype=real_dtype)

    omega_d_scale = jnp.asarray(params.omega_d_scale, dtype=real_dtype)
    omega_star_scale = jnp.asarray(params.omega_star_scale, dtype=real_dtype)
    kpar_scale = jnp.asarray(params.kpar_scale, dtype=real_dtype)

    w_stream = jnp.asarray(term_cfg.streaming, dtype=real_dtype)
    w_mirror = jnp.asarray(term_cfg.mirror, dtype=real_dtype)
    w_curv = jnp.asarray(term_cfg.curvature, dtype=real_dtype)
    w_gradb = jnp.asarray(term_cfg.gradb, dtype=real_dtype)
    w_dia = jnp.asarray(term_cfg.diamagnetic, dtype=real_dtype)
    w_coll = jnp.asarray(term_cfg.collisions, dtype=real_dtype)
    w_hyper = jnp.asarray(term_cfg.hypercollisions, dtype=real_dtype)
    w_damp = jnp.asarray(term_cfg.end_damping, dtype=real_dtype)

    Jl = jnp.asarray(cache.Jl, dtype=real_dtype)
    JlB = jnp.asarray(cache.JlB, dtype=real_dtype)
    phi_j = jnp.asarray(phi, dtype=out_dtype)
    apar_j = jnp.asarray(apar, dtype=out_dtype)
    bpar_j = jnp.asarray(bpar, dtype=out_dtype)

    H = build_H(G_arr, Jl, phi_j, tz, apar=apar_j, vth=vth, bpar=bpar_j, JlB=JlB)
    zero = jnp.zeros_like(G_arr)

    contrib: dict[str, jnp.ndarray] = {}
    contrib["streaming"] = linked_streaming_contribution(
        G_arr,
        phi=phi_j,
        apar=apar_j,
        bpar=bpar_j,
        Jl=Jl,
        JlB=JlB,
        tz=tz,
        kz=cache.kz,
        dz=cache.dz,
        vth=vth,
        sqrt_p=cache.sqrt_p,
        sqrt_m=cache.sqrt_m_ladder,
        kpar_scale=kpar_scale,
        weight=w_stream,
        linked_indices=cache.linked_indices,
        linked_kz=cache.linked_kz,
        linked_inverse_permutation=cache.linked_inverse_permutation,
        linked_full_cover=cache.linked_full_cover,
        linked_gather_map=cache.linked_gather_map,
        linked_gather_mask=cache.linked_gather_mask,
        linked_use_gather=cache.linked_use_gather,
        use_twist_shift=cache.use_twist_shift,
    )
    contrib["mirror"] = mirror_contribution(
        H,
        vth=vth,
        bgrad=cache.bgrad,
        ell=cache.l,
        sqrt_m=cache.sqrt_m,
        sqrt_m_p1=cache.sqrt_m_p1,
        weight=w_mirror,
    )
    contrib["curvature"] = curvature_gradb_contribution(
        H,
        tz=tz,
        omega_d_scale=omega_d_scale,
        cv_d=cache.cv_d,
        gb_d=cache.gb_d,
        ell=cache.l,
        m=cache.m,
        imag=imag,
        weight_curv=w_curv,
        weight_gradb=jnp.asarray(0.0, dtype=real_dtype),
    )
    contrib["gradb"] = curvature_gradb_contribution(
        H,
        tz=tz,
        omega_d_scale=omega_d_scale,
        cv_d=cache.cv_d,
        gb_d=cache.gb_d,
        ell=cache.l,
        m=cache.m,
        imag=imag,
        weight_curv=jnp.asarray(0.0, dtype=real_dtype),
        weight_gradb=w_gradb,
    )
    contrib["diamagnetic"] = diamagnetic_contribution(
        zero,
        phi=phi_j,
        apar=apar_j,
        bpar=bpar_j,
        Jl=Jl,
        b=cache.b,
        JlB=JlB,
        l4=cache.l4,
        tprim=tprim,
        fprim=fprim,
        tz=tz,
        vth=vth,
        omega_star_scale=omega_star_scale,
        ky=cache.ky,
        imag=imag,
        weight=w_dia,
    )
    contrib["collisions"] = collisions_contribution(
        H,
        G=G_arr,
        Jl=Jl,
        JlB=JlB,
        b=cache.b,
        nu=nu,
        lb_lam=cache.lb_lam,
        weight=w_coll,
    )
    contrib["hypercollisions"] = hypercollisions_contribution(
        G_arr,
        vth=vth,
        nu_hyper=nu_hyper,
        nu_hyper_l=nu_hyper_l,
        nu_hyper_m=nu_hyper_m,
        nu_hyper_lm=nu_hyper_lm,
        hyper_ratio=cache.hyper_ratio,
        ratio_l=cache.ratio_l,
        ratio_m=cache.ratio_m,
        ratio_lm=cache.ratio_lm,
        mask_const=cache.mask_const,
        mask_kz=cache.mask_kz,
        m_pow=cache.m_pow,
        m_norm_kz_factor=cache.m_norm_kz_factor,
        kz=cache.kz,
        kpar_scale=kpar_scale,
        hypercollisions_const=hypercollisions_const,
        hypercollisions_kz=hypercollisions_kz,
        weight=w_hyper,
        linked_indices=cache.linked_indices,
        linked_kz=cache.linked_kz,
        linked_inverse_permutation=cache.linked_inverse_permutation,
        linked_full_cover=cache.linked_full_cover,
        linked_gather_map=cache.linked_gather_map,
        linked_gather_mask=cache.linked_gather_mask,
        linked_use_gather=cache.linked_use_gather,
    )
    contrib["end_damping"] = end_damping_contribution(
        H,
        ky=cache.ky,
        damp_profile=cache.damp_profile,
        linked_damp_profile=cache.linked_damp_profile,
        damp_amp=damp_amp,
        weight=w_damp,
    )
    fields = compute_fields_cached(G_arr, cache, params, terms=term_cfg, use_custom_vjp=False)
    return fields, contrib


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gx-dir", type=Path, required=True, help="Directory with rhs_stream.bin, rhs_linear.bin")
    parser.add_argument("--gx-out", type=Path, required=True, help="GX .out.nc file to map ky indices")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Runtime TOML config. When set, geometry/physics come from the runtime path instead of --case.",
    )
    parser.add_argument(
        "--gx-input",
        type=Path,
        default=None,
        help="GX input file for imported-geometry term audits.",
    )
    parser.add_argument(
        "--geometry-file",
        type=Path,
        default=None,
        help="Imported GX geometry source (.out.nc/.eik.nc); required with --gx-input for non-slab cases.",
    )
    parser.add_argument("--case", type=str, default="cyclone", choices=("cyclone", "kbm"))
    parser.add_argument("--ky", type=float, default=0.3)
    parser.add_argument("--Nl", type=int, default=None, help="Laguerre resolution (defaults to dump metadata)")
    parser.add_argument("--Nm", type=int, default=None, help="Hermite resolution (defaults to dump metadata)")
    parser.add_argument("--Ny", type=int, default=24)
    parser.add_argument("--Nz", type=int, default=96)
    parser.add_argument("--y0", type=float, default=None, help="Perpendicular box parameter (defaults to GX ky grid)")
    parser.add_argument("--ntheta", type=int, default=None)
    parser.add_argument("--nperiod", type=int, default=None)
    return parser


def _build_runtime_compare_context(
    config_path: Path,
    *,
    nx: int,
    ny_full: int,
    nz: int,
    nm: int,
    ky_vals: np.ndarray,
    y0_override: float | None,
):
    cfg, _data = load_runtime_from_toml(config_path)
    y0_use = float(y0_override) if y0_override is not None else _infer_y0(ky_vals)
    cfg_use = replace(
        cfg,
        grid=replace(
            cfg.grid,
            Nx=int(nx),
            Ny=int(ny_full),
            Nz=int(nz),
            y0=float(y0_use),
        ),
    )
    geom = build_runtime_geometry(cfg_use)
    grid_cfg = apply_imported_geometry_grid_defaults(geom, cfg_use.grid)
    grid_full = build_spectral_grid(grid_cfg)
    params = build_runtime_linear_params(cfg_use, Nm=nm, geom=geom)
    term_cfg = replace(build_runtime_term_config(cfg_use), hypercollisions=0.0, end_damping=0.0)
    return cfg_use, geom, grid_full, params, term_cfg


def _build_imported_compare_context(
    gx_out: Path,
    gx_input: Path,
    geometry_file: Path | None,
    *,
    nx: int,
    nz: int,
    nm: int,
    ky_vals: np.ndarray,
    y0_override: float | None,
):
    gx_contract = _load_gx_input_contract(gx_input)
    y0_use = float(y0_override) if y0_override is not None else _infer_y0(ky_vals)
    ny_full = _resolve_imported_real_fft_ny(ky_vals, gx_contract)
    boundary_eff = _resolve_imported_boundary(
        str(gx_contract.boundary),
        zero_shat=bool(gx_contract.zero_shat),
    )
    if gx_contract.geo_option == "slab":
        gx_contract = replace(
            gx_contract,
            zero_shat=_read_gx_output_bool(gx_out, "zero_shat", default=gx_contract.zero_shat),
        )
        boundary_eff = _resolve_imported_boundary(
            str(gx_contract.boundary),
            zero_shat=bool(gx_contract.zero_shat),
        )
        geom = SlabGeometry.from_config(
            GeometryConfig(model="slab", s_hat=float(gx_contract.s_hat), zero_shat=bool(gx_contract.zero_shat))
        )
    else:
        if geometry_file is None:
            raise ValueError("--geometry-file is required with --gx-input for non-slab imported cases")
        geom = load_imported_geometry_netcdf(_resolve_internal_geometry_source(geometry_file=geometry_file, runtime_config=None))

    lx = 2.0 * np.pi * y0_use if boundary_eff == "periodic" else 62.8
    grid_cfg = GridConfig(
        Nx=int(nx),
        Ny=int(ny_full),
        Nz=int(nz),
        Lx=float(lx),
        Ly=2.0 * np.pi * float(y0_use),
        boundary=str(boundary_eff),
        y0=float(y0_use),
        nperiod=max(1, int(gx_contract.nperiod)),
        ntheta=max(1, int(gx_contract.ntheta)),
    )
    grid_full = build_spectral_grid(apply_imported_geometry_grid_defaults(geom, grid_cfg))
    params = build_linear_params(
        gx_contract.species,
        tau_e=float(gx_contract.tau_e),
        kpar_scale=float(geom.gradpar()),
        beta=float(gx_contract.beta),
        fapar=float(gx_contract.fapar),
    )
    if gx_contract.hypercollisions:
        params = _apply_reference_hypercollisions(params, nhermite=int(nm))
    params = replace(
        params,
        D_hyper=float(gx_contract.D_hyper),
        damp_ends_amp=float(gx_contract.damp_ends_amp),
        damp_ends_widthfrac=float(gx_contract.damp_ends_widthfrac),
    )
    terms = LinearTerms(
        hypercollisions=1.0 if gx_contract.hypercollisions else 0.0,
        hyperdiffusion=1.0 if gx_contract.hyper else 0.0,
        apar=1.0 if float(gx_contract.fapar) > 0.0 else 0.0,
        bpar=1.0 if float(gx_contract.fbpar) > 0.0 else 0.0,
    )
    return gx_contract, geom, grid_full, params, linear_terms_to_term_config(terms)


def run_compare(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args()

    shape_path = args.gx_dir / "rhs_terms_shape.txt"
    stream_path = args.gx_dir / "rhs_stream.bin"
    linear_path = args.gx_dir / "rhs_linear.bin"
    if not shape_path.exists() or not stream_path.exists() or not linear_path.exists():
        raise FileNotFoundError("Missing GX rhs dump files")

    shape = _load_shape(shape_path)
    nspec = shape.get("nspec", 1)
    nl = shape.get("nl", int(args.Nl) if args.Nl is not None else 1)
    nm = shape.get("nm", int(args.Nm) if args.Nm is not None else 1)
    Nl_use = int(args.Nl) if args.Nl is not None else int(nl)
    Nm_use = int(args.Nm) if args.Nm is not None else int(nm)
    nyc = shape.get("nyc", args.Ny // 2 + 1)
    nx = shape.get("nx", 1)
    nz = shape.get("nz", args.Nz)
    gx_shape = (nspec, nl, nm, nyc, nx, nz)

    gx_stream = _reshape_gx(_load_bin(stream_path, gx_shape), nspec=nspec, nl=nl, nm=nm, nyc=nyc, nx=nx, nz=nz)
    gx_linear = _reshape_gx(_load_bin(linear_path, gx_shape), nspec=nspec, nl=nl, nm=nm, nyc=nyc, nx=nx, nz=nz)
    gx_mirror = _reshape_gx(_load_bin(args.gx_dir / "rhs_mirror.bin", gx_shape), nspec=nspec, nl=nl, nm=nm, nyc=nyc, nx=nx, nz=nz)
    gx_curv = _reshape_gx(_load_bin(args.gx_dir / "rhs_curv.bin", gx_shape), nspec=nspec, nl=nl, nm=nm, nyc=nyc, nx=nx, nz=nz)
    gx_gradb = _reshape_gx(_load_bin(args.gx_dir / "rhs_gradb.bin", gx_shape), nspec=nspec, nl=nl, nm=nm, nyc=nyc, nx=nx, nz=nz)
    gx_dia = _reshape_gx(_load_bin(args.gx_dir / "rhs_diamagnetic.bin", gx_shape), nspec=nspec, nl=nl, nm=nm, nyc=nyc, nx=nx, nz=nz)
    gx_coll = _reshape_gx(_load_bin(args.gx_dir / "rhs_collisions.bin", gx_shape), nspec=nspec, nl=nl, nm=nm, nyc=nyc, nx=nx, nz=nz)
    gx_g_path = args.gx_dir / "g_state.bin"
    gx_hyper_delta: np.ndarray | None = None
    hyper_shape_path = args.gx_dir / "hypercollisions_shape.txt"
    hyper_delta_path = args.gx_dir / "hypercollisions_rhs_delta.bin"
    if hyper_shape_path.exists() and hyper_delta_path.exists():
        hyper_shape = _load_shape(hyper_shape_path)
        hyper_gx_shape = (
            hyper_shape.get("nspec", nspec),
            hyper_shape.get("nl", nl),
            hyper_shape.get("nm", nm),
            hyper_shape.get("nyc", nyc),
            hyper_shape.get("nx", nx),
            hyper_shape.get("nz", nz),
        )
        gx_hyper_delta = _reshape_gx(
            _load_bin(hyper_delta_path, hyper_gx_shape),
            nspec=hyper_gx_shape[0],
            nl=hyper_gx_shape[1],
            nm=hyper_gx_shape[2],
            nyc=hyper_gx_shape[3],
            nx=hyper_gx_shape[4],
            nz=hyper_gx_shape[5],
        )

    with Dataset(args.gx_out, "r") as root:
        ky_vals = np.asarray(root.groups["Grids"].variables["ky"][:], dtype=float)
    y0_use = float(args.y0) if args.y0 is not None else _infer_y0(ky_vals)
    ky_idx = int(np.argmin(np.abs(ky_vals - float(args.ky))))
    gx_stream = gx_stream[:, :, :, ky_idx : ky_idx + 1, :, :]
    gx_linear = gx_linear[:, :, :, ky_idx : ky_idx + 1, :, :]
    gx_mirror = gx_mirror[:, :, :, ky_idx : ky_idx + 1, :, :]
    gx_curv = gx_curv[:, :, :, ky_idx : ky_idx + 1, :, :]
    gx_gradb = gx_gradb[:, :, :, ky_idx : ky_idx + 1, :, :]
    gx_dia = gx_dia[:, :, :, ky_idx : ky_idx + 1, :, :]
    gx_coll = gx_coll[:, :, :, ky_idx : ky_idx + 1, :, :]
    if gx_hyper_delta is not None:
        gx_hyper_delta = gx_hyper_delta[:, :, :, ky_idx : ky_idx + 1, :, :]

    cfg: Any
    if args.gx_input is not None:
        cfg = None
        _gx_contract, geom, grid_full, params, term_cfg = _build_imported_compare_context(
            args.gx_out,
            args.gx_input,
            args.geometry_file,
            nx=nx,
            nz=nz,
            nm=Nm_use,
            ky_vals=ky_vals,
            y0_override=args.y0,
        )
    elif args.config is not None:
        cfg, geom, grid_full, params, term_cfg = _build_runtime_compare_context(
            args.config,
            nx=nx,
            ny_full=int(2 * (nyc - 1)),
            nz=nz,
            nm=Nm_use,
            ky_vals=ky_vals,
            y0_override=args.y0,
        )
    elif args.case == "cyclone":
        cfg = CycloneBaseCase(
            grid=GridConfig(
                Nx=nx,
                Ny=args.Ny,
                Nz=args.Nz,
                Lx=62.8,
                Ly=62.8,
                boundary="linked",
                y0=y0_use,
                ntheta=None,
                nperiod=None,
            )
        )
        geom = SAlphaGeometry.from_config(cfg.geometry)
        params = LinearParams(
            fprim=cfg.model.fprim,
            tprim=cfg.model.tprim_i,
            tprim_e=cfg.model.tprim_e,
            omega_d_scale=CYCLONE_OMEGA_D_SCALE,
            omega_star_scale=CYCLONE_OMEGA_STAR_SCALE,
            rho_star=CYCLONE_RHO_STAR,
            kpar_scale=float(geom.gradpar()),
            nu=cfg.model.nu_i,
            damp_ends_amp=0.0,
            damp_ends_widthfrac=0.0,
        )
        params = _apply_reference_hypercollisions(params, nhermite=Nm_use)
        grid_full = build_spectral_grid(cfg.grid)
        term_cfg = TermConfig(hypercollisions=0.0, end_damping=0.0)
    else:
        ntheta = args.ntheta if args.ntheta is not None else 32
        nperiod = args.nperiod if args.nperiod is not None else 2
        cfg = KBMBaseCase(
            grid=GridConfig(
                Nx=nx,
                Ny=args.Ny,
                Nz=args.Nz,
                Lx=62.8,
                Ly=62.8,
                boundary="linked",
                y0=y0_use,
                ntheta=ntheta,
                nperiod=nperiod,
            )
        )
        geom = SAlphaGeometry.from_config(cfg.geometry)
        params = _two_species_params(
            cfg.model,
            kpar_scale=float(geom.gradpar()),
            omega_d_scale=KBM_OMEGA_D_SCALE,
            omega_star_scale=KBM_OMEGA_STAR_SCALE,
            rho_star=KBM_RHO_STAR,
            nhermite=Nm_use,
        )
        grid_full = build_spectral_grid(cfg.grid)
        term_cfg = TermConfig(hypercollisions=0.0, end_damping=0.0, bpar=0.0)

    ky_index = int(np.argmin(np.abs(np.asarray(grid_full.ky) - float(args.ky))))
    grid = select_ky_grid(grid_full, ky_index)
    cache = build_linear_cache(grid, geom, params, Nl_use, Nm_use)
    cache = _cast_cache(cache, real_dtype=jnp.float32, complex_dtype=jnp.complex64)
    if gx_g_path.exists():
        gx_g = _reshape_gx(_load_bin(gx_g_path, gx_shape), nspec=nspec, nl=nl, nm=nm, nyc=nyc, nx=nx, nz=nz)
        gx_g = gx_g[:, :, :, ky_idx : ky_idx + 1, :, :]
        G0 = jnp.asarray(gx_g.astype(np.complex64))
    else:
        G0 = _build_initial_condition(
            grid,
            geom,
            ky_index=0,
            kx_index=0,
            Nl=Nl_use,
            Nm=Nm_use,
            init_cfg=cfg.init,
        )
    phi_path = args.gx_dir / "phi.bin"
    if phi_path.exists():
        apar_path = args.gx_dir / "apar.bin"
        bpar_path = args.gx_dir / "bpar.bin"
        phi = _load_field(phi_path, nyc, nx, nz)[ky_idx : ky_idx + 1, :, :]
        apar = _load_field(apar_path, nyc, nx, nz)[ky_idx : ky_idx + 1, :, :]
        bpar = _load_field(bpar_path, nyc, nx, nz)[ky_idx : ky_idx + 1, :, :]

        G = G0 if G0.ndim == 6 else G0[None, ...]
        fields, contrib = _manual_linear_contributions_from_fields(
            jnp.asarray(G),
            cache,
            params,
            term_cfg,
            phi=phi,
            apar=apar,
            bpar=bpar,
        )
    else:
        _rhs_total, fields, contrib = assemble_rhs_terms_cached(G0, cache, params, terms=term_cfg)

    def _with_species(arr: jnp.ndarray | np.ndarray) -> np.ndarray:
        arr_np = np.asarray(arr)
        if arr_np.ndim == 5:
            return arr_np[None, ...]
        return arr_np

    gkx_stream = _with_species(contrib["streaming"])
    gkx_mirror = _with_species(contrib["mirror"])
    gkx_curv = _with_species(contrib["curvature"])
    gkx_gradb = _with_species(contrib["gradb"])
    gkx_dia = _with_species(contrib["diamagnetic"])
    gkx_coll = _with_species(contrib["collisions"])
    gkx_hyper = _with_species(contrib["hypercollisions"])

    gkx_linear = (
        gkx_mirror + gkx_curv + gkx_gradb + gkx_dia + gkx_coll
    )

    _summary("streaming", gx_stream, gkx_stream)
    _summary("mirror", gx_mirror, gkx_mirror)
    _summary("curvature", gx_curv, gkx_curv)
    _summary("gradb", gx_gradb, gkx_gradb)
    _summary("diamag", gx_dia, gkx_dia)
    _summary("collisions", gx_coll, gkx_coll)
    if gx_hyper_delta is not None:
        _summary("hyper", gx_hyper_delta, gkx_hyper)
    _summary("linear_sum", gx_linear, gkx_linear)
    if phi_path.exists():
        gkx_phi = np.asarray(fields.phi)
        gkx_apar = np.asarray(fields.apar) if fields.apar is not None else None
        gkx_bpar = np.asarray(fields.bpar) if fields.bpar is not None else None
        _summary("phi", phi, gkx_phi)
        if gkx_apar is not None:
            _summary("apar", apar, gkx_apar)
        if gkx_bpar is not None:
            _summary("bpar", bpar, gkx_bpar)


def _case_config(name: str, args) -> tuple[object, object, int, float, float, float]:
    case = name.lower()
    if case == "cyclone":
        cfg = CycloneBaseCase(
            grid=GridConfig(
                Nx=args.Nx,
                Ny=args.Ny,
                Nz=args.Nz,
                Lx=args.Lx,
                Ly=args.Ly,
                boundary=args.boundary,
                y0=args.y0,
                ntheta=args.ntheta,
                nperiod=args.nperiod,
            )
        )
        geom = SAlphaGeometry.from_config(
            replace(cfg.geometry, drift_scale=args.drift_scale)
        )
        params = LinearParams(
            fprim=cfg.model.fprim,
            tprim=cfg.model.tprim_i,
            tprim_e=cfg.model.tprim_e,
            omega_d_scale=CYCLONE_OMEGA_D_SCALE,
            omega_star_scale=CYCLONE_OMEGA_STAR_SCALE,
            rho_star=CYCLONE_RHO_STAR,
            kpar_scale=float(geom.gradpar()),
            nu=cfg.model.nu_i,
            damp_ends_amp=0.0,
            damp_ends_widthfrac=0.0,
        )
        params = _apply_reference_hypercollisions(params, nhermite=args.Nm)
        return (
            cfg,
            params,
            0,
            CYCLONE_OMEGA_D_SCALE,
            CYCLONE_OMEGA_STAR_SCALE,
            CYCLONE_RHO_STAR,
        )
    if case == "etg":
        cfg, _ = load_runtime_from_toml(
            Path(__file__).resolve().parents[2]
            / "examples/linear/axisymmetric/etg.toml"
        )
        cfg = replace(
            cfg,
            grid=GridConfig(
                Nx=args.Nx,
                Ny=args.Ny,
                Nz=args.Nz,
                Lx=args.Lx,
                Ly=args.Ly,
                boundary=args.boundary,
                y0=args.y0,
                ntheta=args.ntheta,
                nperiod=args.nperiod,
            ),
            species=(replace(cfg.species[0], tprim=float(args.tprim_e)),),
        )
        geom = SAlphaGeometry.from_config(
            replace(cfg.geometry, drift_scale=args.drift_scale)
        )
        params = build_runtime_linear_params(cfg, Nm=args.Nm, geom=geom)
        return (
            cfg,
            params,
            0,
            float(params.omega_d_scale),
            float(params.omega_star_scale),
            float(params.rho_star),
        )
    if case == "kinetic":
        cfg, _raw = load_runtime_from_toml(
            Path(__file__).resolve().parents[2]
            / "examples"
            / "linear"
            / "axisymmetric"
            / "runtime_kinetic_electron.toml"
        )
        cfg = replace(
            cfg,
            grid=GridConfig(
                Nx=args.Nx,
                Ny=args.Ny,
                Nz=args.Nz,
                Lx=args.Lx,
                Ly=args.Ly,
                boundary=args.boundary,
                y0=args.y0,
                ntheta=args.ntheta,
                nperiod=args.nperiod,
            ),
            geometry=replace(cfg.geometry, drift_scale=args.drift_scale),
        )
        geom = build_runtime_geometry(cfg)
        params = build_runtime_linear_params(cfg, Nm=args.Nm, geom=geom)
        return (
            cfg,
            params,
            1,
            KINETIC_OMEGA_D_SCALE,
            KINETIC_OMEGA_STAR_SCALE,
            KINETIC_RHO_STAR,
        )
    if case == "tem":
        cfg, _raw = load_runtime_from_toml(
            Path(__file__).resolve().parents[2]
            / "examples"
            / "linear"
            / "axisymmetric"
            / "runtime_tem.toml"
        )
        cfg = replace(
            cfg,
            grid=GridConfig(
                Nx=args.Nx,
                Ny=args.Ny,
                Nz=args.Nz,
                Lx=args.Lx,
                Ly=args.Ly,
                boundary=args.boundary,
                y0=args.y0,
                ntheta=args.ntheta,
                nperiod=args.nperiod,
            ),
            geometry=replace(cfg.geometry, drift_scale=args.drift_scale),
        )
        geom = build_runtime_geometry(cfg)
        params = build_runtime_linear_params(cfg, Nm=args.Nm, geom=geom)
        return cfg, params, 1, TEM_OMEGA_D_SCALE, TEM_OMEGA_STAR_SCALE, TEM_RHO_STAR
    if case == "kbm":
        cfg = KBMBaseCase(
            grid=GridConfig(
                Nx=args.Nx,
                Ny=args.Ny,
                Nz=args.Nz,
                Lx=args.Lx,
                Ly=args.Ly,
                boundary=args.boundary,
                y0=args.y0,
                ntheta=args.ntheta,
                nperiod=args.nperiod,
            )
        )
        geom = SAlphaGeometry.from_config(
            replace(cfg.geometry, drift_scale=args.drift_scale)
        )
        params = _two_species_params(
            cfg.model,
            kpar_scale=float(geom.gradpar()),
            omega_d_scale=KBM_OMEGA_D_SCALE,
            omega_star_scale=KBM_OMEGA_STAR_SCALE,
            rho_star=KBM_RHO_STAR,
            damp_ends_amp=0.0,
            damp_ends_widthfrac=0.0,
            nhermite=args.Nm,
        )
        return cfg, params, 0, KBM_OMEGA_D_SCALE, KBM_OMEGA_STAR_SCALE, KBM_RHO_STAR
    raise ValueError(f"Unknown case '{name}'")


def _build_seed_state(
    *,
    cfg,
    geom,
    grid,
    params,
    Nl: int,
    Nm: int,
    init_species_index: int,
) -> np.ndarray:
    """Build a single- or multi-species startup state matching ``params``."""

    G0_single = _build_initial_condition(
        grid,
        geom,
        ky_index=0,
        kx_index=0,
        Nl=Nl,
        Nm=Nm,
        init_cfg=cfg.init,
    )
    ns = int(np.atleast_1d(np.asarray(params.charge_sign)).shape[0])
    if ns == 1:
        return np.asarray(G0_single, dtype=np.complex64)
    if init_species_index < 0 or init_species_index >= ns:
        raise ValueError("init_species_index out of range for multi-species seed")
    G0 = np.zeros(
        (ns, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
    )
    G0[int(init_species_index)] = np.asarray(G0_single, dtype=np.complex64)
    return G0


def run_write(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=str, default="cyclone")
    parser.add_argument("--ky", type=float, default=0.3)
    parser.add_argument("--Nx", type=int, default=1)
    parser.add_argument("--Ny", type=int, default=24)
    parser.add_argument("--Nz", type=int, default=96)
    parser.add_argument("--Lx", type=float, default=62.8)
    parser.add_argument("--Ly", type=float, default=62.8)
    parser.add_argument("--boundary", type=str, default="linked")
    parser.add_argument("--y0", type=float, default=20.0)
    parser.add_argument("--ntheta", type=int, default=32)
    parser.add_argument("--nperiod", type=int, default=2)
    parser.add_argument("--Nl", type=int, default=48)
    parser.add_argument("--Nm", type=int, default=16)
    parser.add_argument("--drift-scale", type=float, default=1.0)
    parser.add_argument("--tprim_e", type=float, default=6.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    cfg, params, _init_species_index, *_ = _case_config(args.case, args)
    geom = SAlphaGeometry.from_config(
        replace(cfg.geometry, drift_scale=args.drift_scale)
    )
    grid_full = build_spectral_grid(cfg.grid)
    ky_index = int(np.argmin(np.abs(np.asarray(grid_full.ky) - float(args.ky))))
    grid = select_ky_grid(grid_full, ky_index)

    G0 = _build_seed_state(
        cfg=cfg,
        geom=geom,
        grid=grid,
        params=params,
        Nl=args.Nl,
        Nm=args.Nm,
        init_species_index=_init_species_index,
    )
    cache = build_linear_cache(grid, geom, params, args.Nl, args.Nm)
    term_cfg = TermConfig()
    rhs_total, fields, contrib = assemble_rhs_terms_cached(
        G0, cache, params, terms=term_cfg
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        rhs_total=np.asarray(rhs_total),
        phi=np.asarray(fields.phi),
        apar=np.asarray(fields.apar) if fields.apar is not None else None,
        bpar=np.asarray(fields.bpar) if fields.bpar is not None else None,
        streaming=np.asarray(contrib["streaming"]),
        mirror=np.asarray(contrib["mirror"]),
        curvature=np.asarray(contrib["curvature"]),
        gradb=np.asarray(contrib["gradb"]),
        diamagnetic=np.asarray(contrib["diamagnetic"]),
        collisions=np.asarray(contrib["collisions"]),
        hypercollisions=np.asarray(contrib["hypercollisions"]),
        end_damping=np.asarray(contrib["end_damping"]),
    )
    print(f"Wrote {out}")


def main(argv: list[str] | None = None) -> None:
    """Dispatch RHS diagnostic writing or comparison."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("compare", "write"))
    args, remainder = parser.parse_known_args(argv)
    if args.mode == "compare":
        run_compare(remainder)
    else:
        run_write(remainder)


if __name__ == "__main__":
    main()
