#!/usr/bin/env python3
"""Compare runtime startup, diagnostic-state, and evolution-window data.

The three subcommands share one exact-state comparison implementation while
keeping the startup, saved-state, and evolved-window contracts explicit.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys

import jax.numpy as jnp
import numpy as np
import pandas as pd
from netCDF4 import Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.comparison.compare_gx_rhs_terms import (
    _infer_y0, _load_bin, _load_field, _load_shape, _reshape_gx, _summary,
)
from gkx.core.grid import build_spectral_grid, select_real_fft_ky_grid
from gkx.diagnostics import (
    magnetic_vector_potential_energy, distribution_free_energy,
    electrostatic_field_energy, heat_flux_total, heat_flux_species,
    particle_flux_total, particle_flux_species, fieldline_quadrature_weights,
)
from gkx.geometry import (
    apply_imported_geometry_grid_defaults, ensure_flux_tube_geometry_data,
)
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.runtime import (
    _build_initial_condition, _species_to_linear, build_runtime_geometry,
    build_runtime_linear_params, build_runtime_term_config, run_runtime_nonlinear,
)
from gkx.terms.assembly import compute_fields_cached
from gkx.workflows.runtime.toml import load_runtime_from_toml


def _default_comparison_repo() -> Path | None:
    for candidate in (
        os.environ.get("REFERENCE_GK_REPO"),
        str(REPO_ROOT.parent / "GX"),
        str(REPO_ROOT.parent / "gx"),
    ):
        if candidate:
            path = Path(candidate).expanduser().resolve()
            if path.exists():
                return path
    return None


def _linear_stress_cases(repo: Path) -> dict[str, tuple[Path, Path]]:
    benchmark = repo / "benchmarks" / "linear"
    return {
        "kaw": (
            benchmark / "KAW" / "kaw_betahat10.0_kp0.01_correct.out.nc",
            benchmark / "KAW" / "kaw_betahat10.0_kp0.01.in",
        ),
        "cyclone_ke": (
            benchmark / "ITG_cyclone" / "itg_miller_kinetic_electrons_correct.out.nc",
            benchmark / "ITG_cyclone" / "itg_miller_kinetic_electrons.in",
        ),
        "kbm_miller": (
            benchmark / "KBM" / "kbm_miller_correct.out.nc",
            benchmark / "KBM" / "kbm_miller.in",
        ),
    }


def build_stress_matrix_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the linear comparison stress matrix.")
    parser.add_argument(
        "--comparison-repo", "--gx-repo", dest="comparison_repo", type=Path
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path("tools_out") / "stress_matrix_linear"
    )
    parser.add_argument(
        "--cases", nargs="*", default=["kaw", "cyclone_ke", "kbm_miller"]
    )
    parser.add_argument("--Nl", type=int, default=8)
    parser.add_argument("--Nm", type=int, default=16)
    return parser


def _run_linear_stress_case(
    *, name: str, output: Path, input_file: Path, out_csv: Path, nl: int, nm: int
) -> pd.DataFrame:
    for path in (output, input_file):
        if not path.exists():
            raise FileNotFoundError(f"missing comparison benchmark file: {path}")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parent / "compare_gx_imported_linear.py"),
            "fields",
            "--gx", str(output),
            "--geometry-file", str(output),
            "--gx-input", str(input_file),
            "--Nl", str(nl),
            "--Nm", str(nm),
            "--out", str(out_csv),
        ],
        check=True,
    )
    frame = pd.read_csv(out_csv)
    frame.insert(0, "case", name)
    return frame


def main_stress_matrix() -> None:
    args = build_stress_matrix_parser().parse_args()
    repo = args.comparison_repo or _default_comparison_repo()
    if repo is None:
        raise SystemExit(
            "comparison repository not found; pass --comparison-repo or set REFERENCE_GK_REPO"
        )
    cases = _linear_stress_cases(repo.expanduser().resolve())
    unknown = sorted(set(args.cases).difference(cases))
    if unknown:
        raise SystemExit(f"unknown cases {unknown}; choose from {sorted(cases)}")
    outdir = args.outdir.expanduser().resolve()
    frames = []
    for name in args.cases:
        output, input_file = cases[name]
        frames.append(
            _run_linear_stress_case(
                name=name,
                output=output,
                input_file=input_file,
                out_csv=outdir / f"{name}.csv",
                nl=int(args.Nl),
                nm=int(args.Nm),
            )
        )
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined_path = outdir / "combined.csv"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(combined_path, index=False)
    print(combined.to_string(index=False))
    print(f"saved {combined_path}")


def _select_ky_block(arr: np.ndarray, ky_idx: int) -> np.ndarray:
    if arr.ndim < 3:
        raise ValueError("Expected an array with a ky axis in the third-to-last position")
    slicer = [slice(None)] * arr.ndim
    slicer[-3] = slice(ky_idx, ky_idx + 1)
    return arr[tuple(slicer)]


def _full_ny_from_positive_ky(ky_vals: np.ndarray) -> int:
    ky_arr = np.asarray(ky_vals, dtype=float)
    if ky_arr.ndim != 1 or ky_arr.size == 0:
        raise ValueError("ky_vals must be a non-empty 1D array")
    return int(3 * (ky_arr.size - 1) + 1)


def build_startup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gx-dir", type=Path, required=True, help="Directory containing GX field dump binaries")
    parser.add_argument("--gx-out", type=Path, required=True, help="GX .out.nc file for ky metadata")
    parser.add_argument("--config", type=Path, required=True, help="Runtime TOML config used by GKX")
    parser.add_argument("--ky", type=float, required=True, help="ky value to compare")
    parser.add_argument("--kx-target", type=float, default=0.0, help="kx target within the selected ky block")
    parser.add_argument("--y0", type=float, default=None, help="Optional y0 override; defaults to GX ky metadata")
    return parser


def main_startup() -> None:
    args = build_startup_parser().parse_args()

    shape = _load_shape(args.gx_dir / "field_shape.txt")
    nspec = shape["nspec"]
    nl = shape["nl"]
    nm = shape["nm"]
    nyc = shape["nyc"]
    nx = shape["nx"]
    nz = shape["nz"]
    gx_shape = (nspec, nl, nm, nyc, nx, nz)

    gx_g = _reshape_gx(
        _load_bin(args.gx_dir / "field_g_state.bin", gx_shape),
        nspec=nspec,
        nl=nl,
        nm=nm,
        nyc=nyc,
        nx=nx,
        nz=nz,
    )
    gx_phi = _load_field(args.gx_dir / "field_phi.bin", nyc, nx, nz)
    gx_apar = None
    gx_apar_path = args.gx_dir / "field_apar.bin"
    if gx_apar_path.exists():
        gx_apar = _load_field(gx_apar_path, nyc, nx, nz)

    with Dataset(args.gx_out, "r") as root:
        ky_vals = np.asarray(root.groups["Grids"].variables["ky"][:], dtype=float)

    cfg, _data = load_runtime_from_toml(args.config)
    y0_use = float(args.y0) if args.y0 is not None else _infer_y0(ky_vals)
    full_ny = _full_ny_from_positive_ky(ky_vals)
    cfg_use = replace(
        cfg,
        grid=replace(
            cfg.grid,
            Nx=int(nx),
            Ny=int(full_ny),
            Nz=int(nz),
            y0=float(y0_use),
        ),
    )

    geom = build_runtime_geometry(cfg_use)
    grid_cfg = apply_imported_geometry_grid_defaults(geom, cfg_use.grid)
    grid_full = build_spectral_grid(grid_cfg)
    ky_index = int(np.argmin(np.abs(np.asarray(grid_full.ky) - float(args.ky))))
    kx_index = int(np.argmin(np.abs(np.asarray(grid_full.kx, dtype=float) - float(args.kx_target))))

    params = build_runtime_linear_params(cfg_use, Nm=nm, geom=geom)
    g0 = _build_initial_condition(
        grid_full,
        geom,
        cfg_use,
        ky_index=ky_index,
        kx_index=kx_index,
        Nl=nl,
        Nm=nm,
        nspecies=len(_species_to_linear(cfg_use.species)),
    )
    cache = build_linear_cache(grid_full, geom, params, nl, nm)
    term_cfg = build_runtime_term_config(cfg_use)
    sp_fields = compute_fields_cached(jnp.asarray(g0), cache, params, terms=term_cfg)

    ky_idx_gx = int(np.argmin(np.abs(ky_vals - float(args.ky))))
    gx_g_slice = _select_ky_block(gx_g, ky_idx_gx)
    gx_phi_slice = _select_ky_block(gx_phi, ky_idx_gx)
    sp_g_slice = _select_ky_block(np.asarray(g0, dtype=np.complex64), ky_index)
    sp_phi_slice = _select_ky_block(np.asarray(sp_fields.phi, dtype=np.complex64), ky_index)
    _summary("g_state", gx_g_slice.astype(np.complex64), sp_g_slice)
    _summary("phi", gx_phi_slice.astype(np.complex64), sp_phi_slice)

    if gx_apar is not None and sp_fields.apar is not None:
        gx_apar_slice = _select_ky_block(gx_apar, ky_idx_gx)
        sp_apar_slice = _select_ky_block(np.asarray(sp_fields.apar, dtype=np.complex64), ky_index)
        _summary("apar", gx_apar_slice.astype(np.complex64), sp_apar_slice)


def _load_real_field(path: Path, nyc: int, nx: int, nz: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    expected = nyc * nx * nz
    if raw.size != expected:
        raise ValueError(f"{path} size {raw.size} does not match expected {expected}")
    ky_idx = np.arange(nyc)[:, None, None]
    kx_idx = np.arange(nx)[None, :, None]
    z_idx = np.arange(nz)[None, None, :]
    idxyz = ky_idx + nyc * (kx_idx + nx * z_idx)
    return raw[idxyz.ravel()].reshape(nyc, nx, nz)


def _load_real_vector(path: Path, size: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size != size:
        raise ValueError(f"{path} size {raw.size} does not match expected {size}")
    return raw.astype(np.float32)


def _load_real_vector_auto(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size == 0:
        raise ValueError(f"{path} is empty")
    return raw.astype(np.float32)


def _load_species_state(
    gx_dir: Path,
    *,
    nspec: int,
    nl: int,
    nm: int,
    nyc: int,
    nx: int,
    nz: int,
    time_index: int,
) -> np.ndarray:
    pieces: list[np.ndarray] = []
    n_expected = nl * nm * nyc * nx * nz
    for ispec in range(nspec):
        path = gx_dir / f"diag_state_G_s{ispec}_t{time_index}.bin"
        raw = np.fromfile(path, dtype=np.complex64)
        if raw.size != n_expected:
            raise ValueError(f"{path} size {raw.size} does not match expected {n_expected}")
        pieces.append(raw)
    return _reshape_gx(
        np.stack(pieces, axis=0),
        nspec=nspec,
        nl=nl,
        nm=nm,
        nyc=nyc,
        nx=nx,
        nz=nz,
    )


def _maybe_load_field(path: Path, nyc: int, nx: int, nz: int) -> np.ndarray | None:
    if not path.exists():
        return None
    return _load_field(path, nyc, nx, nz)


def _gx_diag_scalar(group, name: str, time_index: int) -> float:
    arr = np.asarray(group.variables[name][time_index], dtype=float)
    return float(np.sum(arr))


def _gx_diag_species(group, name: str, time_index: int) -> np.ndarray | None:
    if name not in group.variables:
        return None
    arr = np.asarray(group.variables[name][time_index], dtype=float)
    return np.atleast_1d(np.squeeze(arr))


def _diag_row(
    G: np.ndarray,
    phi: np.ndarray,
    apar: np.ndarray,
    bpar: np.ndarray,
    *,
    cache,
    grid,
    params,
    vol_fac: jnp.ndarray,
    flux_fac: jnp.ndarray,
    flux_scale: float,
    wphi_scale: float,
) -> dict[str, float | np.ndarray]:
    G_j = jnp.asarray(G)
    phi_j = jnp.asarray(phi)
    apar_j = jnp.asarray(apar)
    bpar_j = jnp.asarray(bpar)
    return {
        "Wg": float(distribution_free_energy(G_j, grid, params, vol_fac)),
        "Wphi": float(electrostatic_field_energy(phi_j, cache, params, vol_fac, wphi_scale=wphi_scale)),
        "Wapar": float(magnetic_vector_potential_energy(apar_j, cache, vol_fac)),
        "heat": float(heat_flux_total(G_j, phi_j, apar_j, bpar_j, cache, grid, params, flux_fac, flux_scale=flux_scale)),
        "pflux": float(
            particle_flux_total(G_j, phi_j, apar_j, bpar_j, cache, grid, params, flux_fac, flux_scale=flux_scale)
        ),
        "heat_s": np.asarray(
            heat_flux_species(
                G_j,
                phi_j,
                apar_j,
                bpar_j,
                cache,
                grid,
                params,
                flux_fac,
                flux_scale=flux_scale,
            ),
            dtype=float,
        ),
        "pflux_s": np.asarray(
            particle_flux_species(
                G_j,
                phi_j,
                apar_j,
                bpar_j,
                cache,
                grid,
                params,
                flux_fac,
                flux_scale=flux_scale,
            ),
            dtype=float,
        ),
    }


def _rel_err(test: float, ref: float) -> float:
    denom = max(abs(ref), 1.0e-30)
    return float(abs(test - ref) / denom)


def build_diagnostic_state_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gx-dir", type=Path, required=True, help="Directory containing GX diag_state dump binaries")
    parser.add_argument("--gx-out", type=Path, required=True, help="GX .out.nc file for dimensions/diagnostics")
    parser.add_argument("--config", type=Path, required=True, help="Runtime TOML config used by GKX")
    parser.add_argument("--time-index", type=int, required=True, help="GX diagnostic time index to compare")
    parser.add_argument("--y0", type=float, default=None, help="Optional y0 override; defaults to GX ky metadata")
    parser.add_argument("--out", type=Path, default=None, help="Optional CSV summary output")
    return parser


def main_diagnostic_state() -> None:
    args = build_diagnostic_state_parser().parse_args()

    with Dataset(args.gx_out, "r") as root:
        diag = root.groups["Diagnostics"]
        nl = int(root.dimensions["l"].size)
        nm = int(root.dimensions["m"].size)
        nspec = int(root.dimensions["s"].size)
        if args.time_index < 0 or args.time_index >= int(root.dimensions["time"].size):
            raise ValueError(f"time_index={args.time_index} outside [0, {int(root.dimensions['time'].size) - 1}]")
        gx_diag = {
            "Wg": _gx_diag_scalar(diag, "Wg_kyst", args.time_index),
            "Wphi": _gx_diag_scalar(diag, "Wphi_kyst", args.time_index),
            "Wapar": _gx_diag_scalar(diag, "Wapar_kyst", args.time_index),
            "heat": _gx_diag_scalar(diag, "HeatFlux_kyst", args.time_index),
            "pflux": _gx_diag_scalar(diag, "ParticleFlux_kyst", args.time_index),
        }
        gx_heat_s = _gx_diag_species(diag, "HeatFlux_st", args.time_index)
        gx_pflux_s = _gx_diag_species(diag, "ParticleFlux_st", args.time_index)
        t_val = float(np.asarray(root.groups["Grids"].variables["time"][args.time_index], dtype=float))

    gx_kx = _load_real_vector_auto(args.gx_dir / f"diag_state_kx_t{args.time_index}.bin")
    gx_ky = _load_real_vector_auto(args.gx_dir / f"diag_state_ky_t{args.time_index}.bin")
    nyc = int(gx_ky.size)
    nx = int(gx_kx.size)
    phi_raw = np.fromfile(args.gx_dir / f"diag_state_phi_t{args.time_index}.bin", dtype=np.complex64)
    if phi_raw.size % max(nyc * nx, 1) != 0:
        raise ValueError(
            f"diag_state_phi_t{args.time_index}.bin size {phi_raw.size} is not divisible by nyc*nx={nyc*nx}"
        )
    nz = int(phi_raw.size // (nyc * nx))

    gx_G = _load_species_state(
        args.gx_dir,
        nspec=nspec,
        nl=nl,
        nm=nm,
        nyc=nyc,
        nx=nx,
        nz=nz,
        time_index=args.time_index,
    )
    gx_phi = _load_field(args.gx_dir / f"diag_state_phi_t{args.time_index}.bin", nyc, nx, nz)
    gx_apar = _maybe_load_field(args.gx_dir / f"diag_state_apar_t{args.time_index}.bin", nyc, nx, nz)
    gx_bpar = _maybe_load_field(args.gx_dir / f"diag_state_bpar_t{args.time_index}.bin", nyc, nx, nz)
    gx_kperp2 = _load_real_field(args.gx_dir / f"diag_state_kperp2_t{args.time_index}.bin", nyc, nx, nz)
    gx_fluxfac = _load_real_vector(args.gx_dir / f"diag_state_fluxfac_t{args.time_index}.bin", nz)

    cfg, _data = load_runtime_from_toml(args.config)
    y0_use = float(args.y0) if args.y0 is not None else _infer_y0(gx_ky)
    full_ny = int(2 * (nyc - 1))
    cfg_use = replace(
        cfg,
        grid=replace(
            cfg.grid,
            Nx=int(nx),
            Ny=int(full_ny),
            Nz=int(nz),
            y0=float(y0_use),
        ),
    )

    geom = build_runtime_geometry(cfg_use)
    grid_cfg = apply_imported_geometry_grid_defaults(geom, cfg_use.grid)
    grid_full = build_spectral_grid(grid_cfg)
    grid = select_real_fft_ky_grid(grid_full, gx_ky.astype(np.float32))
    params = build_runtime_linear_params(cfg_use, Nm=nm, geom=geom)
    geom_eff = ensure_flux_tube_geometry_data(geom, grid.z)
    cache = build_linear_cache(grid, geom_eff, params, nl, nm)
    term_cfg = build_runtime_term_config(cfg_use)
    sp_fields = compute_fields_cached(jnp.asarray(gx_G), cache, params, terms=term_cfg)

    vol_fac, flux_fac = fieldline_quadrature_weights(geom_eff, grid)

    sp_phi = np.asarray(sp_fields.phi, dtype=np.complex64)
    sp_apar = (
        np.asarray(sp_fields.apar, dtype=np.complex64)
        if sp_fields.apar is not None
        else np.zeros_like(gx_phi, dtype=np.complex64)
    )
    sp_bpar = (
        np.asarray(sp_fields.bpar, dtype=np.complex64)
        if sp_fields.bpar is not None
        else np.zeros_like(gx_phi, dtype=np.complex64)
    )
    gx_apar_use = gx_apar if gx_apar is not None else np.zeros_like(gx_phi, dtype=np.complex64)
    gx_bpar_use = gx_bpar if gx_bpar is not None else np.zeros_like(gx_phi, dtype=np.complex64)

    _summary("kperp2", gx_kperp2.astype(np.float32), np.asarray(cache.kperp2, dtype=np.float32))
    _summary("fluxfac", gx_fluxfac.astype(np.float32), np.asarray(flux_fac, dtype=np.float32))
    if gx_kx.shape == grid.kx.shape:
        _summary("kx", gx_kx.astype(np.float32), np.asarray(grid.kx, dtype=np.float32))
    if gx_ky.shape == grid.ky.shape:
        _summary("ky", gx_ky.astype(np.float32), np.asarray(grid.ky, dtype=np.float32))
    _summary("phi", gx_phi.astype(np.complex64), sp_phi)
    if gx_apar is not None or sp_fields.apar is not None:
        _summary("apar", gx_apar_use.astype(np.complex64), sp_apar)
    if gx_bpar is not None or sp_fields.bpar is not None:
        _summary("bpar", gx_bpar_use.astype(np.complex64), sp_bpar)

    flux_scale = float(cfg_use.normalization.flux_scale)
    wphi_scale = float(cfg_use.normalization.wphi_scale)
    sp_dump = _diag_row(
        gx_G,
        gx_phi,
        gx_apar_use,
        gx_bpar_use,
        cache=cache,
        grid=grid,
        params=params,
        vol_fac=vol_fac,
        flux_fac=flux_fac,
        flux_scale=flux_scale,
        wphi_scale=wphi_scale,
    )
    sp_solve = _diag_row(
        gx_G,
        sp_phi,
        sp_apar,
        sp_bpar,
        cache=cache,
        grid=grid,
        params=params,
        vol_fac=vol_fac,
        flux_fac=flux_fac,
        flux_scale=flux_scale,
        wphi_scale=wphi_scale,
    )

    print(f"time_index={args.time_index} t={t_val:.8f}")
    print("metric     gx_out        gkx_dump  rel_dump      gkx_solve rel_solve")
    rows: list[dict[str, float | str]] = []
    for key in ("Wg", "Wphi", "Wapar", "heat", "pflux"):
        gx_val = float(gx_diag[key])
        dump_val = float(sp_dump[key])
        solve_val = float(sp_solve[key])
        rel_dump = _rel_err(dump_val, gx_val)
        rel_solve = _rel_err(solve_val, gx_val)
        rows.append(
            {
                "time_index": float(args.time_index),
                "t": t_val,
                "metric": key,
                "gx_out": gx_val,
                "gkx_dump": dump_val,
                "rel_dump": rel_dump,
                "gkx_solve": solve_val,
                "rel_solve": rel_solve,
            }
        )
        print(
            f"{key:8s} {gx_val: .6e} {dump_val: .6e} {rel_dump: .3e} "
            f"{solve_val: .6e} {rel_solve: .3e}"
        )

    if gx_heat_s is not None:
        print("heat_flux_species gx_out=", np.asarray(gx_heat_s, dtype=float), "gkx_dump=", sp_dump["heat_s"])
    if gx_pflux_s is not None:
        print("particle_flux_species gx_out=", np.asarray(gx_pflux_s, dtype=float), "gkx_dump=", sp_dump["pflux_s"])

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)


def build_window_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gx-dir", type=Path, required=True, help="Directory containing GX diag_state dump binaries")
    parser.add_argument("--gx-out", type=Path, required=True, help="GX .out.nc file for dimensions/diagnostics")
    parser.add_argument("--config", type=Path, required=True, help="Runtime TOML config used by GKX")
    parser.add_argument("--time-index-start", type=int, required=True, help="GX diagnostic time index for the start state")
    parser.add_argument("--time-index-stop", type=int, required=True, help="GX diagnostic time index for the target state")
    parser.add_argument("--steps", type=int, default=None, help="Optional maximum step count override for the runtime audit")
    parser.add_argument("--ky", type=float, default=None, help="Optional ky to use for gamma/omega diagnostics")
    parser.add_argument("--y0", type=float, default=None, help="Optional y0 override; defaults to GX ky metadata")
    parser.add_argument("--out", type=Path, default=None, help="Optional CSV summary output")
    return parser


def main_window() -> None:
    args = build_window_parser().parse_args()

    with Dataset(args.gx_out, "r") as root:
        diag = root.groups["Diagnostics"]
        grids = root.groups["Grids"]
        nl = int(root.dimensions["l"].size)
        nm = int(root.dimensions["m"].size)
        nspec = int(root.dimensions["s"].size)
        ntime = int(root.dimensions["time"].size)
        if args.time_index_start < 0 or args.time_index_start >= ntime:
            raise ValueError(f"time_index_start={args.time_index_start} outside [0, {ntime - 1}]")
        if args.time_index_stop < 0 or args.time_index_stop >= ntime:
            raise ValueError(f"time_index_stop={args.time_index_stop} outside [0, {ntime - 1}]")
        t_vals = np.asarray(grids.variables["time"][:], dtype=float)
        gx_diag_stop = {
            "Wg": _gx_diag_scalar(diag, "Wg_kyst", args.time_index_stop),
            "Wphi": _gx_diag_scalar(diag, "Wphi_kyst", args.time_index_stop),
            "Wapar": _gx_diag_scalar(diag, "Wapar_kyst", args.time_index_stop),
            "heat": _gx_diag_scalar(diag, "HeatFlux_kyst", args.time_index_stop),
            "pflux": _gx_diag_scalar(diag, "ParticleFlux_kyst", args.time_index_stop),
        }
        gx_heat_s = _gx_diag_species(diag, "HeatFlux_st", args.time_index_stop)
        gx_pflux_s = _gx_diag_species(diag, "ParticleFlux_st", args.time_index_stop)
        t_start = float(t_vals[args.time_index_start])
        t_stop = float(t_vals[args.time_index_stop])
        dt_window = float(t_stop - t_start)

    gx_kx = _load_real_vector_auto(args.gx_dir / f"diag_state_kx_t{args.time_index_start}.bin")
    gx_ky = _load_real_vector_auto(args.gx_dir / f"diag_state_ky_t{args.time_index_start}.bin")
    nyc = int(gx_ky.size)
    nx = int(gx_kx.size)
    phi_raw = np.fromfile(args.gx_dir / f"diag_state_phi_t{args.time_index_start}.bin", dtype=np.complex64)
    if phi_raw.size % max(nyc * nx, 1) != 0:
        raise ValueError(
            f"diag_state_phi_t{args.time_index_start}.bin size {phi_raw.size} is not divisible by nyc*nx={nyc*nx}"
        )
    nz = int(phi_raw.size // (nyc * nx))

    gx_G_start = _load_species_state(
        args.gx_dir,
        nspec=nspec,
        nl=nl,
        nm=nm,
        nyc=nyc,
        nx=nx,
        nz=nz,
        time_index=args.time_index_start,
    )
    gx_phi_start = _load_field(args.gx_dir / f"diag_state_phi_t{args.time_index_start}.bin", nyc, nx, nz)
    gx_apar_start = _maybe_load_field(args.gx_dir / f"diag_state_apar_t{args.time_index_start}.bin", nyc, nx, nz)
    gx_bpar_start = _maybe_load_field(args.gx_dir / f"diag_state_bpar_t{args.time_index_start}.bin", nyc, nx, nz)

    cfg, _data = load_runtime_from_toml(args.config)
    y0_use = float(args.y0) if args.y0 is not None else _infer_y0(gx_ky)
    full_ny = int(2 * (nyc - 1))
    cfg_use = replace(
        cfg,
        grid=replace(cfg.grid, Nx=int(nx), Ny=int(full_ny), Nz=int(nz), y0=float(y0_use)),
    )

    geom = build_runtime_geometry(cfg_use)
    grid_cfg = apply_imported_geometry_grid_defaults(geom, cfg_use.grid)
    grid_full = build_spectral_grid(grid_cfg)
    grid = select_real_fft_ky_grid(grid_full, gx_ky.astype(np.float32))
    geom_eff = ensure_flux_tube_geometry_data(geom, grid.z)
    params = build_runtime_linear_params(cfg_use, Nm=nm, geom=geom)
    cache = build_linear_cache(grid, geom_eff, params, nl, nm)
    term_cfg = build_runtime_term_config(cfg_use)

    sp_fields_start = compute_fields_cached(jnp.asarray(gx_G_start, dtype=jnp.complex64), cache, params, terms=term_cfg)
    sp_phi_start = np.asarray(sp_fields_start.phi, dtype=np.complex64)
    sp_apar_start = (
        np.asarray(sp_fields_start.apar, dtype=np.complex64)
        if sp_fields_start.apar is not None
        else np.zeros_like(gx_phi_start, dtype=np.complex64)
    )
    sp_bpar_start = (
        np.asarray(sp_fields_start.bpar, dtype=np.complex64)
        if sp_fields_start.bpar is not None
        else np.zeros_like(gx_phi_start, dtype=np.complex64)
    )
    gx_apar_start_use = gx_apar_start if gx_apar_start is not None else np.zeros_like(gx_phi_start, dtype=np.complex64)
    gx_bpar_start_use = gx_bpar_start if gx_bpar_start is not None else np.zeros_like(gx_phi_start, dtype=np.complex64)

    _summary("start_g_state", gx_G_start.astype(np.complex64), np.asarray(gx_G_start, dtype=np.complex64))
    _summary("start_phi", gx_phi_start.astype(np.complex64), sp_phi_start)
    if gx_apar_start is not None or sp_fields_start.apar is not None:
        _summary("start_apar", gx_apar_start_use.astype(np.complex64), sp_apar_start)
    if gx_bpar_start is not None or sp_fields_start.bpar is not None:
        _summary("start_bpar", gx_bpar_start_use.astype(np.complex64), sp_bpar_start)

    ky_target = (
        float(args.ky)
        if args.ky is not None
        else float(next((val for val in np.asarray(grid.ky, dtype=float) if abs(val) > 0.0), 0.0))
    )
    cfg_run = replace(
        cfg_use,
        time=replace(
            cfg_use.time,
            t_max=dt_window,
            sample_stride=1,
            diagnostics_stride=1,
        ),
        init=replace(
            cfg_use.init,
            init_file=str(args.gx_dir / f"diag_state_G_s0_t{args.time_index_start}.bin"),
            init_file_scale=1.0,
            init_file_mode="replace",
        ),
    )
    result = run_runtime_nonlinear(
        cfg_run,
        ky_target=ky_target,
        Nl=nl,
        Nm=nm,
        steps=args.steps,
        diagnostics=True,
    )
    if result.diagnostics is None:
        raise RuntimeError("runtime window audit requires diagnostics output")
    diag_use = result.diagnostics
    t_arr = np.asarray(diag_use.t, dtype=float)
    idx_match = int(np.argmin(np.abs(t_arr - dt_window))) if t_arr.size else 0
    t_match = float(t_arr[idx_match]) if t_arr.size else 0.0
    sp_diag_stop = {
        "Wg": float(np.asarray(diag_use.Wg_t)[idx_match]),
        "Wphi": float(np.asarray(diag_use.Wphi_t)[idx_match]),
        "Wapar": float(np.asarray(diag_use.Wapar_t)[idx_match]),
        "heat": float(np.asarray(diag_use.heat_flux_t)[idx_match]),
        "pflux": float(np.asarray(diag_use.particle_flux_t)[idx_match]),
    }
    sp_diag_species: dict[str, np.ndarray] = {}
    if diag_use.heat_flux_species_t is not None:
        heat_s = np.asarray(diag_use.heat_flux_species_t)
        sp_diag_species["heat_s"] = np.asarray(heat_s[idx_match], dtype=float)
    if diag_use.particle_flux_species_t is not None:
        pflux_s = np.asarray(diag_use.particle_flux_species_t)
        sp_diag_species["pflux_s"] = np.asarray(pflux_s[idx_match], dtype=float)

    print(
        f"time_index_start={args.time_index_start} t_start={t_start:.8f} "
        f"time_index_stop={args.time_index_stop} t_stop={t_stop:.8f} "
        f"delta_t={dt_window:.8f} steps={args.steps if args.steps is not None else -1} "
        f"gkx_t_match={t_match:.8f}"
    )
    print("metric     gx_stop       gkx      rel")
    rows: list[dict[str, float | str]] = []
    for key in ("Wg", "Wphi", "Wapar", "heat", "pflux"):
        gx_val = float(gx_diag_stop[key])
        sp_val = sp_diag_stop[key]
        rel = _rel_err(sp_val, gx_val)
        rows.append(
            {
                "time_index_start": float(args.time_index_start),
                "time_index_stop": float(args.time_index_stop),
                "t_start": t_start,
                "t_stop": t_stop,
                "delta_t": dt_window,
                "steps": float(args.steps) if args.steps is not None else -1.0,
                "gkx_t": t_match,
                "metric": key,
                "gx_stop": gx_val,
                "gkx": sp_val,
                "rel": rel,
            }
        )
        print(f"{key:8s} {gx_val: .6e} {sp_val: .6e} {rel: .3e}")
    if gx_heat_s is not None:
        print(
            "heat_flux_species gx_stop=",
            np.asarray(gx_heat_s, dtype=float),
            "gkx=",
            sp_diag_species.get("heat_s", np.asarray([])),
        )
    if gx_pflux_s is not None:
        print(
            "particle_flux_species gx_stop=",
            np.asarray(gx_pflux_s, dtype=float),
            "gkx=",
            sp_diag_species.get("pflux_s", np.asarray([])),
        )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)

_COMMANDS = {
    "startup": main_startup,
    "diagnostic-state": main_diagnostic_state,
    "window": main_window,
    "stress-matrix": main_stress_matrix,
}


def main(argv: list[str] | None = None) -> None:
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw or raw[0] in {"-h", "--help"}:
        choices = ", ".join(_COMMANDS)
        print(f"usage: compare_runtime.py <command> [options]\ncommands: {choices}")
        return
    command, *command_args = raw
    try:
        command_main = _COMMANDS[command]
    except KeyError as exc:
        choices = ", ".join(_COMMANDS)
        raise SystemExit(f"unknown command {command!r}; choose one of: {choices}") from exc
    previous_argv = sys.argv
    try:
        sys.argv = [f"{previous_argv[0]} {command}", *command_args]
        command_main()
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
