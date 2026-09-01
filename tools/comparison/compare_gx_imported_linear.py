#!/usr/bin/env python3
"""Compare imported-geometry linear fields, growth dumps, and time windows."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from netCDF4 import Dataset

from gkx.benchmarking_shared import _apply_reference_hypercollisions
from gkx.config import GeometryConfig, GridConfig, InitializationConfig, resolve_cfl_fac
from gkx.geometry import (
    SlabGeometry,
    apply_imported_geometry_grid_defaults,
    ensure_flux_tube_geometry_data,
    zero_shear_enabled,
    load_imported_geometry_netcdf,
)
from gkx.core_velocity import gamma0
from gkx.core_grid import (
    build_spectral_grid,
    select_ky_grid,
    select_real_fft_ky_grid,
)
from gkx.diagnostics import (
    distribution_free_energy,
    electrostatic_field_energy,
    fieldline_quadrature_weights,
    magnetic_vector_potential_energy,
)
from gkx.diagnostics.analysis import (
    ModeSelection,
    instantaneous_growth_rate_from_phi,
    select_ky_index,
)
from gkx.solvers_time_explicit import (
    ExplicitTimeConfig,
    _instantaneous_growth_rate_step,
    _linear_frequency_bound,
    _diagnostic_midplane_index,
    _linear_term_config,
    _linear_explicit_step,
)
from gkx.workflows.runtime.toml import load_toml
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.params import LinearTerms
from gkx.runtime import (
    _build_initial_condition as _build_runtime_initial_condition,
    _load_initial_state_from_file,
)
from gkx.workflows.runtime.toml import load_runtime_from_toml
from gkx.geometry.miller_eik import generate_runtime_miller_eik
from gkx.workflows.runtime.config import RuntimeConfig, RuntimeSpeciesConfig
from gkx.operators.linear.params import Species, build_linear_params
from gkx.terms.assembly import assemble_rhs_cached
from gkx.geometry.vmec_eik import generate_runtime_vmec_eik


def _reshape_saved_state(
    raw: np.ndarray,
    *,
    nspec: int,
    nl: int,
    nm: int,
    nyc: int,
    nx: int,
    nz: int,
) -> np.ndarray:
    """Map flattened comparison dumps to ``(s,l,m,ky,kx,z)`` layout."""

    arr = raw.reshape((nspec, nm, nl, nyc * nx * nz)).transpose(0, 2, 1, 3)
    ky_idx = np.arange(nyc)[:, None, None]
    kx_idx = np.arange(nx)[None, :, None]
    z_idx = np.arange(nz)[None, None, :]
    indices = ky_idx + nyc * (kx_idx + nx * z_idx)
    return arr[..., indices.ravel()].reshape((nspec, nl, nm, nyc, nx, nz))


def _load_field(path: Path, nyc: int, nx: int, nz: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.complex64)
    expected = nyc * nx * nz
    if raw.size != expected:
        raise ValueError(f"{path} size {raw.size} does not match expected {expected}")
    ky_idx = np.arange(nyc)[:, None, None]
    kx_idx = np.arange(nx)[None, :, None]
    z_idx = np.arange(nz)[None, None, :]
    indices = ky_idx + nyc * (kx_idx + nx * z_idx)
    return raw[indices.ravel()].reshape(nyc, nx, nz)


def _load_real_vector_auto(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size == 0:
        raise ValueError(f"{path} is empty")
    return raw


def _load_species_state(
    directory: Path,
    *,
    nspec: int,
    nl: int,
    nm: int,
    nyc: int,
    nx: int,
    nz: int,
    time_index: int,
) -> np.ndarray:
    expected = nl * nm * nyc * nx * nz
    pieces = []
    for species_index in range(nspec):
        path = directory / f"diag_state_G_s{species_index}_t{time_index}.bin"
        raw = np.fromfile(path, dtype=np.complex64)
        if raw.size != expected:
            raise ValueError(
                f"{path} size {raw.size} does not match expected {expected}"
            )
        pieces.append(raw)
    return _reshape_saved_state(
        np.stack(pieces), nspec=nspec, nl=nl, nm=nm, nyc=nyc, nx=nx, nz=nz
    )


def _maybe_load_field(path: Path, nyc: int, nx: int, nz: int) -> np.ndarray | None:
    return _load_field(path, nyc, nx, nz) if path.exists() else None


@dataclass(frozen=True)
class GXInputContract:
    Nx: int
    Ny: int
    nperiod: int
    ntheta: int
    npol: float | None
    alpha: float | None
    torflux: float | None
    nlaguerre: int
    nhermite: int
    boundary: str
    geo_option: str
    s_hat: float
    zero_shat: bool
    y0: float
    fapar: float
    fbpar: float
    species: tuple[Species, ...]
    tau_e: float
    beta: float
    dt: float | None
    scheme: str
    nwrite: int
    init_field: str
    init_amp: float
    init_single: bool
    ikx_single: int
    iky_single: int
    gaussian_init: bool
    gaussian_width: float
    gaussian_envelope_constant: float
    gaussian_envelope_sine: float
    kpar_init: float
    random_seed: int
    init_electrons_only: bool
    random_init: bool
    hypercollisions: bool
    hyper: bool
    D_hyper: float
    damp_ends_amp: float
    damp_ends_widthfrac: float
    restart_with_perturb: bool
    restart_scale: float


def _file_cache_token(path: Path | None) -> dict[str, str | int | None]:
    if path is None:
        return {"path": None, "size": None, "mtime_ns": None}
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _load_gx_reference(
    path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    def _read_ky_series(diag_group, name: str) -> np.ndarray:
        data = np.asarray(diag_group.variables[name][:], dtype=float)
        if data.ndim == 3:
            if data.shape[1] == 1 or name.startswith("Wapar_"):
                return np.asarray(data[:, 0, :], dtype=float)
            return np.asarray(np.sum(data, axis=1), dtype=float)
        raise ValueError(f"Unexpected shape for {name}: {data.shape}")

    root = Dataset(path, "r")
    try:
        grids = root.groups["Grids"]
        diag = root.groups["Diagnostics"]
        time = np.asarray(grids.variables["time"][:], dtype=float)
        ky = np.asarray(grids.variables["ky"][:], dtype=float)
        kx = np.asarray(grids.variables["kx"][:], dtype=float)
        omega = np.asarray(diag.variables["omega_kxkyt"][:], dtype=float)
        Wg = _read_ky_series(diag, "Wg_kyst")
        Wphi = _read_ky_series(diag, "Wphi_kyst")
        Wapar = _read_ky_series(diag, "Wapar_kyst")
        if "Phi2_kyt" in diag.variables:
            Phi2 = np.asarray(diag.variables["Phi2_kyt"][:], dtype=float)
        else:
            Phi2 = np.full((time.size, ky.size), np.nan, dtype=float)
    finally:
        root.close()
    return time, ky, kx, omega, Wg, Wphi, Wapar, Phi2


def _read_gx_output_bool(path: Path, name: str, *, default: bool = False) -> bool:
    root = Dataset(path, "r")
    try:
        inputs = root.groups.get("Inputs")
        if inputs is None or name not in inputs.variables:
            return bool(default)
        return bool(np.asarray(inputs.variables[name][:]).item())
    finally:
        root.close()


def _infer_real_fft_ny(ky: np.ndarray) -> int:
    positive = ky[ky > 0.0]
    if positive.size == 0:
        raise ValueError("GX reference does not contain positive ky modes")
    # Invert GX's "dealiased positive ky count" convention.
    #
    # Most GX benchmark outputs store only the dealiased non-negative ky block.
    # To build a full FFT grid that:
    # - contains the requested smallest positive ky, and
    # - does not immediately mask it out under GX's strict 2/3 rule
    # we need Ny >= 4 even when only one positive ky is present.
    return max(4, int(3 * (positive.size - 1) + 1))


def _infer_y0(ky: np.ndarray) -> float:
    positive = ky[ky > 0.0]
    if positive.size == 0:
        raise ValueError("GX reference does not contain positive ky modes")
    return float(1.0 / np.min(positive))


def _resolve_imported_real_fft_ny(
    gx_ky: np.ndarray, gx_contract: GXInputContract | None
) -> int:
    """Return the full real-FFT ``Ny`` implied by the stored GX ``ky`` grid."""

    inferred = _infer_real_fft_ny(gx_ky)
    if gx_contract is None:
        return inferred
    ny_input = int(gx_contract.Ny)
    if ny_input <= 0:
        return inferred
    full_ny_from_input = max(4, int(3 * (ny_input - 1) + 1))
    # GX input files store the dealiased non-negative ky count (`nky`), while
    # the imported GKX grid needs the full real-FFT layout. For the GX
    # 2/3-rule contract, the non-negative block has length `floor(Ny/3) + 1`,
    # so the inverse mapping is `Ny = 3 * (nky - 1) + 1`.
    if ny_input == int(np.asarray(gx_ky).size):
        return full_ny_from_input
    # Raw `diag_state_ky` dumps keep the full non-negative real-FFT block,
    # including the Nyquist row when present. Map that back onto the same full
    # `Ny` implied by the GX input contract instead of inverting it as though
    # it were the dealiased NetCDF ky axis.
    raw_real_fft_block = full_ny_from_input // 2 + 1
    if int(np.asarray(gx_ky).size) == raw_real_fft_block:
        return full_ny_from_input
    return inferred


def _resolve_imported_boundary(boundary: str, *, zero_shat: bool) -> str:
    """Return the effective GX boundary contract for imported linear runs.

    GX promotes near-zero magnetic shear to ``zero_shat`` and, in that path,
    forces periodic parallel boundary conditions.
    """

    if bool(zero_shat):
        return "periodic"
    return str(boundary)


def _load_gx_input_contract(path: Path) -> GXInputContract:
    data = load_toml(path)
    dims = data.get("Dimensions", {})
    domain = data.get("Domain", {})
    physics = data.get("Physics", {})
    time = data.get("Time", {})
    init = data.get("Initialization", {})
    expert = data.get("Expert", {})
    geometry = data.get("Geometry", {})
    species_raw = data.get("species", {})
    boltz = data.get("Boltzmann", {})
    diagnostics = data.get("Diagnostics", {})
    dissipation = data.get("Dissipation", {})

    nspecies = int(dims.get("nspecies", 1))

    def _species_array(name: str, *, default: float) -> np.ndarray:
        raw = species_raw.get(name)
        if raw is None:
            return np.full(nspecies, default, dtype=float)
        arr = np.asarray(raw, dtype=float)
        if arr.ndim == 0:
            arr = np.full(nspecies, float(arr), dtype=float)
        if arr.size < nspecies:
            raise ValueError(
                f"{path} species.{name} has {arr.size} entries, expected at least {nspecies}"
            )
        return np.asarray(arr[:nspecies], dtype=float)

    charge = _species_array("z", default=1.0)
    mass = _species_array("mass", default=1.0)
    dens = _species_array("dens", default=1.0)
    temp = _species_array("temp", default=1.0)
    tprim = _species_array("tprim", default=0.0)
    fprim = _species_array("fprim", default=0.0)
    nu = _species_array("vnewk", default=0.0)
    species = tuple(
        Species(
            charge=float(charge[i]),
            mass=float(mass[i]),
            density=float(dens[i]),
            temperature=float(temp[i]),
            tprim=float(tprim[i]),
            fprim=float(fprim[i]),
            nu=float(nu[i]),
        )
        for i in range(nspecies)
    )

    add_boltz = bool(boltz.get("add_Boltzmann_species", False))
    boltz_type = str(boltz.get("Boltzmann_type", "")).strip().lower()
    tau_e = (
        float(boltz.get("tau_fac", 0.0))
        if add_boltz and boltz_type == "electrons"
        else 0.0
    )
    kpar_init = (
        float(init["ikpar_init"])
        if "ikpar_init" in init
        else float(init.get("kpar_init", 0.0))
    )

    s_hat = float(geometry.get("shat", 0.0))
    zero_shat = zero_shear_enabled(
        s_hat,
        zero_shat=bool(geometry.get("zero_shat", False)),
        threshold=float(geometry.get("zero_shat_threshold", 1.0e-5)),
    )

    return GXInputContract(
        Nx=int(dims.get("nkx", dims.get("nx", 1))),
        Ny=int(dims.get("nky", dims.get("ny", 0))),
        nperiod=int(dims.get("nperiod", 1)),
        ntheta=int(dims.get("ntheta", 0)),
        npol=float(geometry["npol"]) if "npol" in geometry else None,
        alpha=float(geometry["alpha"]) if "alpha" in geometry else None,
        torflux=float(geometry["torflux"]) if "torflux" in geometry else None,
        nlaguerre=int(dims.get("nlaguerre", 8)),
        nhermite=int(dims.get("nhermite", 16)),
        boundary=str(domain.get("boundary", "linked")),
        geo_option=str(geometry.get("geo_option", "s-alpha")).strip().lower(),
        s_hat=s_hat,
        zero_shat=zero_shat,
        y0=float(domain["y0"]) if "y0" in domain else float("nan"),
        fapar=float(
            physics.get("fapar", 1.0 if float(physics.get("beta", 0.0)) > 0.0 else 0.0)
        ),
        fbpar=float(
            physics.get("fbpar", 1.0 if float(physics.get("beta", 0.0)) > 0.0 else 0.0)
        ),
        species=species,
        tau_e=tau_e,
        beta=float(physics.get("beta", 0.0)),
        dt=None if "dt" not in time else float(time["dt"]),
        scheme=str(time.get("scheme", "rk4")).strip().lower(),
        nwrite=max(1, int(diagnostics.get("nwrite", 1))),
        init_field=str(init.get("init_field", "density")).strip().lower(),
        init_amp=float(init.get("init_amp", 1.0e-5)),
        init_single=bool(expert.get("init_single", False)),
        ikx_single=int(expert.get("ikx_single", 0)),
        iky_single=int(expert.get("iky_single", 1)),
        gaussian_init=bool(init.get("gaussian_init", False)),
        gaussian_width=float(init.get("gaussian_width", 0.5)),
        gaussian_envelope_constant=float(
            init.get("gaussian_envelope_constant_coefficient", 1.0)
        ),
        gaussian_envelope_sine=float(
            init.get("gaussian_envelope_sine_coefficient", 0.0)
        ),
        kpar_init=kpar_init,
        random_seed=int(init.get("random_seed", 22)),
        init_electrons_only=bool(init.get("init_electrons_only", False)),
        random_init=bool(init.get("random_init", False)),
        hypercollisions=bool(dissipation.get("hypercollisions", False)),
        hyper=bool(dissipation.get("hyper", False)),
        D_hyper=float(dissipation.get("D_hyper", 0.0)),
        damp_ends_amp=float(dissipation.get("damp_ends_amp", 0.1)),
        damp_ends_widthfrac=float(dissipation.get("damp_ends_widthfrac", 1.0 / 8.0)),
        restart_with_perturb=bool(data.get("restart_with_perturb", False)),
        restart_scale=float(data.get("scale", 1.0)),
    )


def _resolve_internal_geometry_source(
    *,
    geometry_file: Path | None,
    runtime_config: Path | None,
    gx_contract: GXInputContract | None = None,
) -> Path:
    """Resolve geometry for the GKX run without sourcing it from GX output files."""

    if geometry_file is not None:
        return geometry_file.expanduser().resolve()

    if runtime_config is not None:
        cfg, _ = load_runtime_from_toml(runtime_config.expanduser().resolve())
        if gx_contract is not None:
            ntheta = int(gx_contract.ntheta)
            nperiod = int(gx_contract.nperiod)
            cfg = replace(
                cfg,
                grid=replace(
                    cfg.grid,
                    boundary=_resolve_imported_boundary(
                        gx_contract.boundary,
                        zero_shat=bool(gx_contract.zero_shat),
                    ),
                    y0=float(gx_contract.y0),
                    ntheta=ntheta if ntheta > 0 else cfg.grid.ntheta,
                    nperiod=nperiod if nperiod > 0 else cfg.grid.nperiod,
                ),
            )
            if str(cfg.geometry.model).strip().lower() == "vmec":
                cfg = replace(
                    cfg,
                    geometry=replace(
                        cfg.geometry,
                        alpha=float(gx_contract.alpha)
                        if gx_contract.alpha is not None
                        else cfg.geometry.alpha,
                        torflux=float(gx_contract.torflux)
                        if gx_contract.torflux is not None
                        else cfg.geometry.torflux,
                        npol=float(gx_contract.npol)
                        if gx_contract.npol is not None
                        else cfg.geometry.npol,
                    ),
                )
        model = str(cfg.geometry.model).strip().lower()
        if model == "vmec":
            return generate_runtime_vmec_eik(cfg, force=True).expanduser().resolve()
        if model == "miller":
            return generate_runtime_miller_eik(cfg, force=True).expanduser().resolve()
        raise ValueError(
            f"--runtime-config must use geometry.model='vmec' or 'miller' for internal generation; got {cfg.geometry.model!r}"
        )

    raise ValueError(
        "No geometry source for GKX run. Provide either --geometry-file or --runtime-config "
        "(VMEC runtime TOML)."
    )


def _runtime_species_tuple(
    species: tuple[Species, ...],
) -> tuple[RuntimeSpeciesConfig, ...]:
    return tuple(
        RuntimeSpeciesConfig(
            name=f"species_{idx}",
            charge=float(sp.charge),
            mass=float(sp.mass),
            density=float(sp.density),
            temperature=float(sp.temperature),
            tprim=float(sp.tprim),
            fprim=float(sp.fprim),
            nu=float(sp.nu),
            kinetic=True,
        )
        for idx, sp in enumerate(species)
    )


def _build_imported_linear_terms(gx_contract: GXInputContract | None) -> LinearTerms:
    terms = LinearTerms()
    if gx_contract is None:
        return terms
    return replace(
        terms,
        hypercollisions=1.0 if gx_contract.hypercollisions else 0.0,
        hyperdiffusion=1.0 if gx_contract.hyper else 0.0,
        apar=1.0 if float(gx_contract.fapar) > 0.0 else 0.0,
        bpar=1.0 if float(gx_contract.fbpar) > 0.0 else 0.0,
    )


def _build_imported_initial_condition(
    *,
    grid,
    geom,
    gx_contract: GXInputContract | None,
    species: tuple[Species, ...],
    ky_index: int,
    kx_index: int,
    Nl: int,
    Nm: int,
):
    if gx_contract is None:
        init_cfg = InitializationConfig(
            gaussian_init=True,
            init_field="density",
            init_amp=1.0e-10,
        )
        seed_ky = ky_index
        seed_kx = kx_index
    else:
        if gx_contract.random_init:
            raise NotImplementedError(
                "Imported linear comparison does not yet support GX random_init=true startup"
            )
        init_cfg = InitializationConfig(
            init_field=gx_contract.init_field,
            init_amp=gx_contract.init_amp,
            init_single=gx_contract.init_single,
            random_seed=gx_contract.random_seed,
            gaussian_init=gx_contract.gaussian_init,
            gaussian_width=gx_contract.gaussian_width,
            gaussian_envelope_constant=gx_contract.gaussian_envelope_constant,
            gaussian_envelope_sine=gx_contract.gaussian_envelope_sine,
            kpar_init=gx_contract.kpar_init,
            init_electrons_only=gx_contract.init_electrons_only,
        )
        seed_ky = gx_contract.iky_single if gx_contract.init_single else ky_index
        seed_kx = gx_contract.ikx_single if gx_contract.init_single else kx_index
    cfg = RuntimeConfig(
        init=init_cfg,
        species=_runtime_species_tuple(species),
    )
    return _build_runtime_initial_condition(
        grid,
        geom,
        cfg,
        ky_index=int(seed_ky),
        kx_index=int(seed_kx),
        Nl=Nl,
        Nm=Nm,
        nspecies=len(species),
    )


def _mean_rel_error(
    lhs: np.ndarray, rhs: np.ndarray, *, floor_fraction: float
) -> float:
    lhs = np.asarray(lhs, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    scale = max(float(np.nanmax(np.abs(rhs))), 1.0e-12)
    floor = floor_fraction * scale
    denom = np.maximum(np.abs(rhs), floor)
    return float(np.mean(np.abs(lhs - rhs) / denom))


def _match_local_kx_index(grid_kx: np.ndarray, gx_kx_value: float) -> int:
    return int(np.argmin(np.abs(np.asarray(grid_kx, dtype=float) - float(gx_kx_value))))


def _select_gx_kx_index(gx_kx: np.ndarray, gx_contract: GXInputContract | None) -> int:
    gx_kx_arr = np.asarray(gx_kx, dtype=float)
    if gx_contract is not None and gx_contract.init_single:
        if 0 <= int(gx_contract.ikx_single) < int(gx_kx_arr.size):
            return int(gx_contract.ikx_single)
    return int(np.argmin(np.abs(gx_kx_arr)))


def _cached_hermitian_mode_weight(cache, *, use_dealias: bool) -> jnp.ndarray:
    ky = jnp.asarray(cache.ky)
    has_negative = jnp.any(ky < 0.0)
    fac = jnp.where(has_negative, 1.0, jnp.where(ky == 0.0, 1.0, 2.0))
    fac = fac[:, None] * jnp.ones((1, cache.kx.size), dtype=fac.dtype)
    if use_dealias:
        fac = fac * cache.dealias_mask.astype(fac.dtype)
    return fac


def _gx_kyst_fac_mask_cached(cache, *, use_dealias: bool) -> jnp.ndarray:
    """Return GX kyst fac*mask on a full GKX ky layout.

    GX stores ``*_kyst`` diagnostics on the positive-rFFT ky half only, while the
    evolved GKX state may carry the full ``±ky`` layout. To compare a full
    state against GX ``*_kyst`` data, keep the positive-ky rows with the Hermitian
    factor of 2, keep ky=0 with unit weight, and exclude ky<0 entirely.
    """

    ky = jnp.asarray(cache.ky)
    fac = jnp.where(ky > 0.0, 2.0, jnp.where(ky == 0.0, 1.0, 0.0))
    fac = fac[:, None] * jnp.ones((1, cache.kx.size), dtype=fac.dtype)
    if use_dealias:
        fac = fac * cache.dealias_mask.astype(fac.dtype)
    return fac


def _species_array(val: float | jnp.ndarray, ns: int) -> jnp.ndarray:
    arr = jnp.asarray(val)
    if arr.ndim == 0:
        return jnp.broadcast_to(arr, (ns,))
    return arr


def _distribution_free_energy_by_ky(
    G: jnp.ndarray, cache, params, vol_fac: jnp.ndarray, *, use_dealias: bool = True
) -> jnp.ndarray:
    Gs = G if G.ndim == 6 else G[None, ...]
    ns = Gs.shape[0]
    nt = _species_array(params.density, ns) * _species_array(params.temp, ns)
    fac = _gx_kyst_fac_mask_cached(cache, use_dealias=use_dealias)
    weight = fac[:, :, None] * vol_fac[None, None, :]
    contrib = 0.5 * (jnp.abs(Gs) ** 2) * nt[:, None, None, None, None, None]
    contrib = contrib * weight[None, None, None, :, :, :]
    return jnp.sum(contrib, axis=(0, 1, 2, 4, 5))


def _electrostatic_field_energy_by_ky(
    phi: jnp.ndarray, cache, params, vol_fac: jnp.ndarray, *, use_dealias: bool = True
) -> jnp.ndarray:
    fac = _gx_kyst_fac_mask_cached(cache, use_dealias=use_dealias)
    weight = fac[:, :, None] * vol_fac[None, None, :]
    rho = jnp.asarray(params.rho)
    if rho.ndim == 0:
        rho = rho[None]
    phi2 = jnp.abs(phi) ** 2
    wphi = jnp.zeros((phi.shape[0],), dtype=jnp.real(phi).dtype)
    for rho_s in rho:
        b = cache.kperp2 * (rho_s * rho_s)
        contrib = 0.5 * phi2 * (1.0 - gamma0(b)) * weight
        wphi = wphi + jnp.sum(contrib, axis=(1, 2))
    return wphi


def _magnetic_vector_potential_energy_by_ky(
    apar: jnp.ndarray, cache, vol_fac: jnp.ndarray, *, use_dealias: bool = True
) -> jnp.ndarray:
    fac = _gx_kyst_fac_mask_cached(cache, use_dealias=use_dealias)
    weight = fac[:, :, None] * vol_fac[None, None, :]
    bmag2 = cache.bmag[None, None, :] ** 2 if cache.kperp2_bmag else 1.0
    contrib = 0.5 * (jnp.abs(apar) ** 2) * cache.kperp2 * bmag2 * weight
    return jnp.sum(contrib, axis=(1, 2))


def _gx_Phi2_by_ky(phi: jnp.ndarray, vol_fac: jnp.ndarray) -> jnp.ndarray:
    weight = vol_fac[None, None, :]
    return jnp.sum(jnp.abs(phi) ** 2 * weight, axis=(1, 2))


def _integrate_target_mode_series(
    *,
    G0: jnp.ndarray,
    grid,
    geom,
    cache,
    params,
    time_cfg: ExplicitTimeConfig,
    terms: LinearTerms,
    mode_method: str,
    ky_index: int,
    kx_index: int,
    reference_times: np.ndarray,
    output_steps: np.ndarray,
    return_phi_samples: bool = False,
) -> tuple[np.ndarray, ...]:
    if mode_method not in {"z_index", "max", "project", "svd"}:
        raise ValueError(
            "mode_method must be one of {'z_index', 'max', 'project', 'svd'}"
        )

    term_cfg = _linear_term_config(terms)
    mask = jnp.asarray(grid.dealias_mask, dtype=bool)
    z_index = _diagnostic_midplane_index(grid.z.size)
    dt = float(time_cfg.dt)
    dt_min = float(time_cfg.dt_min)
    dt_max = float(time_cfg.dt_max) if time_cfg.dt_max is not None else dt
    target_times = np.asarray(reference_times, dtype=float)
    target_samples = int(target_times.size)
    if target_samples <= 0:
        raise ValueError("reference_times must request at least one sample")
    if np.any(np.diff(target_times) < -1.0e-14):
        raise ValueError("reference_times must be monotonically nondecreasing")
    output_idx = np.asarray(output_steps, dtype=int)
    if output_idx.ndim != 1:
        raise ValueError("output_steps must be a 1D index array")
    if output_idx.size <= 0:
        raise ValueError("output_steps must request at least one sample")
    if np.any(output_idx < 0) or np.any(output_idx >= target_samples):
        raise ValueError("output_steps must index reference_times")
    if np.any(np.diff(output_idx) < 0):
        raise ValueError("output_steps must be monotonically nondecreasing")
    max_target_time = float(target_times[-1])
    dt_ceiling = float(time_cfg.dt_max) if time_cfg.dt_max is not None else dt
    dt_floor = max(min(dt, dt_ceiling), float(time_cfg.dt_min))
    base_steps = int(np.ceil(max_target_time / max(dt_floor, 1.0e-12)))
    if time_cfg.fixed_dt:
        max_steps = max(base_steps + 8 * target_samples + 8, 8 * target_samples + 8)
    else:
        # Imported audits can encounter much smaller reconstructed adaptive steps
        # than the initial dt guess. Use a generous safety factor so we do not
        # misclassify a lane because the harness ran out of budget before
        # reaching the requested GX sample times.
        max_steps = max(base_steps * 256 + 1024, 32 * target_samples + 32)
    vol_fac = cache.jacobian / jnp.sum(cache.jacobian)

    G = jnp.asarray(G0)
    t = 0.0
    step = 0

    geom_eff = ensure_flux_tube_geometry_data(geom, grid.z)
    omega_max = _linear_frequency_bound(
        grid, geom_eff, params, G.shape[-5], G.shape[-4]
    )
    wmax = float(np.sum(omega_max))
    if not time_cfg.fixed_dt and wmax > 0.0:
        dt_guess = float(time_cfg.cfl_fac) * float(time_cfg.cfl) / wmax
        dt = min(max(dt_guess, dt_min), dt_max)

    _, fields0 = assemble_rhs_cached(G, cache, params, terms=term_cfg)
    phi_prev_sample = fields0.phi
    t_prev_sample = t

    def _step(G_state, cache_state, params_state, term_cfg_state, dt_state):
        return _linear_explicit_step(
            G_state,
            cache_state,
            params_state,
            term_cfg_state,
            dt_state,
            method=time_cfg.method,
        )

    stepper = jax.jit(_step, donate_argnums=(0,))

    gamma_list: list[float] = []
    omega_list: list[float] = []
    Wg_list: list[float] = []
    Wphi_list: list[float] = []
    Wapar_list: list[float] = []
    Phi2_list: list[float] = []
    collect_phi_samples = bool(return_phi_samples) or mode_method in {"project", "svd"}
    phi_samples: list[np.ndarray] = []
    phi_sample_times: list[float] = []
    if collect_phi_samples:
        phi0_sel = np.asarray(fields0.phi)[
            ky_index : ky_index + 1, kx_index : kx_index + 1, :
        ]
        phi_samples.append(np.asarray(phi0_sel, dtype=np.complex64))
        phi_sample_times.append(float(t))

    target_idx = 0
    while target_idx < target_samples and step < max_steps:
        dt_step = dt
        if not time_cfg.fixed_dt and wmax > 0.0:
            dt_guess = float(time_cfg.cfl_fac) * float(time_cfg.cfl) / wmax
            dt_step = min(max(dt_guess, dt_min), dt_max)
        next_target = float(target_times[target_idx])
        remaining = next_target - t
        if remaining <= 1.0e-14:
            remaining = 0.0
        elif dt_step > remaining:
            dt_step = max(remaining, float(time_cfg.dt_min))

        G, fields = stepper(G, cache, params, term_cfg, dt_step)
        step += 1
        t += dt_step

        if t >= next_target - 1.0e-12:
            phi = fields.phi
            apar = fields.apar if fields.apar is not None else jnp.zeros_like(phi)
            dt_sample = max(t - t_prev_sample, 0.0)
            if mode_method in {"project", "svd"}:
                gamma = jnp.zeros(phi.shape[:2], dtype=jnp.real(phi).dtype)
                omega = jnp.zeros(phi.shape[:2], dtype=jnp.real(phi).dtype)
            else:
                gamma, omega = _instantaneous_growth_rate_step(
                    phi,
                    phi_prev_sample,
                    dt_sample,
                    z_index=z_index,
                    mask=mask,
                    mode_method=mode_method,
                )
            Wg = _distribution_free_energy_by_ky(G, cache, params, vol_fac)
            Wphi = _electrostatic_field_energy_by_ky(phi, cache, params, vol_fac)
            Wapar = _magnetic_vector_potential_energy_by_ky(apar, cache, vol_fac)
            Phi2 = _gx_Phi2_by_ky(phi, vol_fac)
            gamma_list.append(float(np.asarray(gamma)[ky_index, kx_index]))
            omega_list.append(float(np.asarray(omega)[ky_index, kx_index]))
            Wg_list.append(float(np.asarray(Wg)[ky_index]))
            Wphi_list.append(float(np.asarray(Wphi)[ky_index]))
            Wapar_list.append(float(np.asarray(Wapar)[ky_index]))
            Phi2_list.append(float(np.asarray(Phi2)[ky_index]))
            if collect_phi_samples:
                phi_sel = np.asarray(phi)[
                    ky_index : ky_index + 1, kx_index : kx_index + 1, :
                ]
                phi_samples.append(np.asarray(phi_sel, dtype=np.complex64))
                phi_sample_times.append(float(t))
            target_idx += 1
            phi_prev_sample = phi
            t_prev_sample = t

    if len(gamma_list) != target_samples:
        raise RuntimeError(
            "Imported-linear integration produced "
            f"{len(gamma_list)} samples, expected {target_samples} "
            f"(dt={dt}, max_steps={max_steps})"
        )
    if mode_method in {"project", "svd"}:
        phi_t = np.asarray(phi_samples, dtype=np.complex64)
        t_arr = np.asarray(phi_sample_times, dtype=float)
        _gamma_avg, _omega_avg, gamma_t, omega_t, _t_mid = (
            instantaneous_growth_rate_from_phi(
                phi_t,
                t_arr,
                ModeSelection(ky_index=0, kx_index=0, z_index=z_index),
                use_last=False,
                mode_method=mode_method,
            )
        )
        gamma_list = list(np.asarray(gamma_t, dtype=float))
        omega_list = list(np.asarray(omega_t, dtype=float))
    gamma_arr = np.asarray(gamma_list, dtype=float)
    omega_arr = np.asarray(omega_list, dtype=float)
    Wg_arr = np.asarray(Wg_list, dtype=float)
    Wphi_arr = np.asarray(Wphi_list, dtype=float)
    Wapar_arr = np.asarray(Wapar_list, dtype=float)
    Phi2_arr = np.asarray(Phi2_list, dtype=float)
    out = (
        gamma_arr[output_idx],
        omega_arr[output_idx],
        Wg_arr[output_idx],
        Wphi_arr[output_idx],
        Wapar_arr[output_idx],
        Phi2_arr[output_idx],
    )
    if return_phi_samples:
        return (
            *out,
            np.asarray(phi_sample_times, dtype=float),
            np.asarray(phi_samples, dtype=np.complex64),
        )
    return out


def _run_single_ky(
    *,
    ky_target: float,
    geom,
    grid_full,
    params,
    time_cfg: ExplicitTimeConfig,
    gx_contract: GXInputContract | None,
    species: tuple[Species, ...],
    Nl: int,
    Nm: int,
    reference_times: np.ndarray,
    output_steps: np.ndarray,
    mode_method: str,
    kx_index: int,
    terms: LinearTerms,
    G0_override: jnp.ndarray | None = None,
    return_phi_samples: bool = False,
) -> tuple[np.ndarray, ...]:
    ky_index = select_ky_index(np.asarray(grid_full.ky), ky_target)
    if G0_override is None:
        G0_full = _build_imported_initial_condition(
            grid=grid_full,
            geom=geom,
            gx_contract=gx_contract,
            species=species,
            ky_index=ky_index,
            kx_index=kx_index,
            Nl=Nl,
            Nm=Nm,
        )
    else:
        G0_full = jnp.asarray(G0_override)
    use_full_grid = gx_contract is not None
    if use_full_grid:
        G0 = G0_full
        grid = grid_full
        ky_diag_index = ky_index
    else:
        G0 = G0_full[:, :, :, ky_index : ky_index + 1, :, :]
        grid = select_ky_grid(grid_full, ky_index)
        ky_diag_index = 0
    cache = build_linear_cache(grid, geom, params, Nl, Nm)
    return _integrate_target_mode_series(
        G0=G0,
        grid=grid,
        geom=geom,
        cache=cache,
        params=params,
        time_cfg=time_cfg,
        terms=terms,
        mode_method=mode_method,
        ky_index=ky_diag_index,
        kx_index=kx_index,
        reference_times=reference_times,
        output_steps=output_steps,
        return_phi_samples=return_phi_samples,
    )


def _write_scan_rows(rows: list[dict[str, float]], out: Path | None) -> pd.DataFrame:
    df = pd.DataFrame(rows).sort_values("ky").reset_index(drop=True)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
    return df


def _infer_gx_linear_dt(
    gx_time: np.ndarray, gx_contract: GXInputContract | None
) -> float:
    """Infer the underlying GX timestep from saved diagnostic times."""

    if gx_contract is not None and gx_contract.dt is not None:
        return float(gx_contract.dt)

    time_arr = np.asarray(gx_time, dtype=float)
    if time_arr.size == 0:
        raise ValueError("gx_time cannot be empty")
    nwrite = 1 if gx_contract is None else max(1, int(gx_contract.nwrite))
    positive = time_arr[time_arr > 0.0]
    if time_arr.size >= 2:
        diffs = np.diff(time_arr)
        positive_diffs = diffs[diffs > 0.0]
        if positive_diffs.size > 0:
            return float(np.median(positive_diffs) / float(nwrite))
    if positive.size > 0:
        return float(positive[0] / float(nwrite))
    raise ValueError("Could not infer a positive GX timestep from diagnostic times")


def _gx_has_uniform_linear_dt(
    gx_time: np.ndarray,
    gx_contract: GXInputContract | None,
    *,
    rtol: float = 1.0e-6,
    atol: float = 1.0e-12,
) -> bool:
    """Return True when the GX diagnostic times imply a uniform underlying dt.

    Imported-linear audits compare against a specific GX run. When the saved
    times correspond to a constant underlying timestep, reusing that inferred dt
    is a more faithful contract than re-estimating an adaptive CFL step from
    the reconstructed GKX state.
    """

    time_arr = np.asarray(gx_time, dtype=float)
    if time_arr.size <= 1:
        return True
    diffs = np.diff(time_arr)
    positive_diffs = diffs[diffs > 0.0]
    if positive_diffs.size <= 1:
        return True
    nwrite = 1 if gx_contract is None else max(1, int(gx_contract.nwrite))
    dt_steps = positive_diffs / float(nwrite)
    dt_ref = float(np.median(dt_steps))
    close = np.isclose(dt_steps, dt_ref, rtol=rtol, atol=atol)
    if bool(np.all(close)):
        return True
    # GX commonly ends a linear run with one truncated final interval when
    # `time + dt > t_max`; that should not force the audit back onto the CFL
    # estimator for the entire trajectory.
    if dt_steps.size >= 2 and bool(np.all(close[:-1])):
        return True
    return False


def _build_sample_steps(
    gx_time: np.ndarray,
    *,
    sample_step_stride: int,
    max_samples: int | None,
    sample_window: str = "head",
) -> np.ndarray:
    steps = np.arange(np.asarray(gx_time).size, dtype=int)
    stride = max(1, int(sample_step_stride))
    if stride > 1:
        steps = steps[::stride]
    if max_samples is not None:
        nkeep = max(0, int(max_samples))
        if sample_window == "tail":
            steps = steps[-nkeep:]
        else:
            steps = steps[:nkeep]
    if steps.size == 0:
        raise ValueError(
            "Selected sample window is empty; increase --max-samples or lower --sample-step-stride"
        )
    return np.asarray(steps, dtype=int)


def _series_cache_path(
    *,
    cache_dir: Path,
    gx_path: Path,
    geometry_file: Path | None,
    gx_input: Path | None,
    ky_target: float,
    Nl: int,
    Nm: int,
    mode_method: str,
    rel_floor_fraction: float,
    sample_steps: np.ndarray,
) -> Path:
    payload = {
        "version": 2,
        "gx": _file_cache_token(gx_path),
        "geometry_file": _file_cache_token(geometry_file),
        "gx_input": _file_cache_token(gx_input),
        "ky_target": float(ky_target),
        "Nl": int(Nl),
        "Nm": int(Nm),
        "mode_method": str(mode_method),
        "rel_floor_fraction": float(rel_floor_fraction),
        "sample_steps": [int(v) for v in np.asarray(sample_steps, dtype=int)],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    tag = f"{float(ky_target):0.4f}".replace(".", "p")
    return cache_dir / f"ky_{tag}_{digest}.npz"


def _load_cached_ky_series(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as data:
        return (
            np.asarray(data["gamma"], dtype=float),
            np.asarray(data["omega"], dtype=float),
            np.asarray(data["Wg"], dtype=float),
            np.asarray(data["Wphi"], dtype=float),
            np.asarray(data["Wapar"], dtype=float),
            np.asarray(data["Phi2"], dtype=float),
        )


def _save_cached_ky_series(
    path: Path,
    *,
    gamma: np.ndarray,
    omega: np.ndarray,
    Wg: np.ndarray,
    Wphi: np.ndarray,
    Wapar: np.ndarray,
    Phi2: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        gamma=np.asarray(gamma, dtype=float),
        omega=np.asarray(omega, dtype=float),
        Wg=np.asarray(Wg, dtype=float),
        Wphi=np.asarray(Wphi, dtype=float),
        Wapar=np.asarray(Wapar, dtype=float),
        Phi2=np.asarray(Phi2, dtype=float),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare GX linear diagnostics against GKX using imported GX/VMEC geometry."
    )
    parser.add_argument(
        "--gx", type=Path, required=True, help="Path to the GX .out.nc file"
    )
    parser.add_argument(
        "--geometry-file",
        type=Path,
        default=None,
        help="Optional path to a GX/VMEC geometry file (for example *.eik.nc). "
        "If omitted, pass --runtime-config to generate VMEC geometry internally.",
    )
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=None,
        help="Optional GKX runtime TOML used to generate VMEC geometry internally when --geometry-file is omitted.",
    )
    parser.add_argument(
        "--gx-input",
        type=Path,
        default=None,
        help="Optional GX input file used to infer boundary/grid defaults for imported geometry cases.",
    )
    parser.add_argument(
        "--init-file",
        type=Path,
        default=None,
        help="Optional raw GX state file to use as the exact initial condition.",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Optional CSV output path"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional directory for per-ky cached trajectory/result arrays.",
    )
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Reuse matching per-ky cached trajectory/result arrays when available.",
    )
    parser.add_argument(
        "--ky",
        type=float,
        nargs="*",
        default=None,
        help="Specific ky values to compare",
    )
    parser.add_argument(
        "--sample-step-stride",
        type=int,
        default=1,
        help="Subsample the saved GX diagnostic sample indices by this stride before scoring.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="If set, only score the first N selected GX diagnostic samples.",
    )
    parser.add_argument(
        "--sample-window",
        choices=("head", "tail"),
        default="head",
        help="When --max-samples is set, select the first or last N stride-filtered GX diagnostic samples.",
    )
    parser.add_argument("--Nl", type=int, default=None)
    parser.add_argument("--Nm", type=int, default=None)
    parser.add_argument("--tprim", type=float, default=3.0)
    parser.add_argument("--fprim", type=float, default=1.0)
    parser.add_argument("--tau-e", type=float, default=1.0, dest="tau_e")
    parser.add_argument("--damp-ends-amp", type=float, default=0.1)
    parser.add_argument("--damp-ends-widthfrac", type=float, default=1.0 / 8.0)
    parser.add_argument(
        "--mode-method", choices=("z_index", "max", "project", "svd"), default="z_index"
    )
    parser.add_argument(
        "--rel-floor-fraction",
        type=float,
        default=1.0e-2,
        help="Relative-error floor as a fraction of the peak reference magnitude for each series",
    )
    return parser


def run_fields(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    (
        gx_time,
        gx_ky,
        gx_kx,
        gx_omega,
        distribution_free_energy,
        electrostatic_field_energy,
        magnetic_vector_potential_energy,
        gx_Phi2,
    ) = _load_gx_reference(args.gx)
    positive_ky = gx_ky[gx_ky > 0.0]
    ky_values = (
        positive_ky
        if args.ky is None or len(args.ky) == 0
        else np.asarray(args.ky, dtype=float)
    )
    sample_steps = _build_sample_steps(
        gx_time,
        sample_step_stride=int(args.sample_step_stride),
        max_samples=args.max_samples,
        sample_window=str(args.sample_window),
    )
    ref_stop = int(sample_steps[-1]) + 1
    reference_times = np.asarray(gx_time[:ref_stop], dtype=float)

    boundary = "linked"
    y0 = _infer_y0(gx_ky)
    nx = 1
    ny = _resolve_imported_real_fft_ny(gx_ky, None)
    nperiod = 1
    ntheta = 0
    gx_contract: GXInputContract | None = None
    species = [
        Species(
            charge=1.0,
            mass=1.0,
            density=1.0,
            temperature=1.0,
            tprim=float(args.tprim),
            fprim=float(args.fprim),
        )
    ]
    tau_e = float(args.tau_e)
    beta = 0.0
    if args.gx_input is not None:
        gx_contract = _load_gx_input_contract(args.gx_input)
        boundary = str(gx_contract.boundary)
        y0 = float(gx_contract.y0) if np.isfinite(float(gx_contract.y0)) else y0
        nx = max(1, int(gx_contract.Nx))
        ny = _resolve_imported_real_fft_ny(gx_ky, gx_contract)
        nperiod = max(1, int(gx_contract.nperiod))
        ntheta_in = int(gx_contract.ntheta)
        if ntheta_in > 0:
            ntheta = ntheta_in
        species = list(gx_contract.species)
        tau_e = float(gx_contract.tau_e)
        beta = float(gx_contract.beta)
        sample_steps = _build_sample_steps(
            gx_time,
            sample_step_stride=int(args.sample_step_stride),
            max_samples=args.max_samples,
            sample_window=str(args.sample_window),
        )
        ref_stop = int(sample_steps[-1]) + 1
        reference_times = np.asarray(gx_time[:ref_stop], dtype=float)
    dt = _infer_gx_linear_dt(gx_time, gx_contract)
    if gx_contract is not None and gx_contract.geo_option == "slab":
        gx_contract = replace(
            gx_contract,
            zero_shat=_read_gx_output_bool(
                args.gx, "zero_shat", default=gx_contract.zero_shat
            ),
        )
        geom = SlabGeometry.from_config(
            GeometryConfig(
                model="slab",
                s_hat=float(gx_contract.s_hat),
                zero_shat=bool(gx_contract.zero_shat),
            )
        )
        nz = int(gx_contract.ntheta) if int(gx_contract.ntheta) > 0 else 16
    else:
        geometry_source = _resolve_internal_geometry_source(
            geometry_file=args.geometry_file,
            runtime_config=args.runtime_config,
            gx_contract=gx_contract,
        )
        geom = load_imported_geometry_netcdf(geometry_source)
        nz = int(np.asarray(geom.theta).size)
    if ntheta <= 0:
        ntheta = nz

    lx = 62.8
    boundary_eff = _resolve_imported_boundary(
        boundary,
        zero_shat=bool(gx_contract.zero_shat) if gx_contract is not None else False,
    )
    if boundary_eff == "periodic":
        lx = 2.0 * np.pi * y0

    grid_cfg = GridConfig(
        Nx=nx,
        Ny=ny,
        Nz=nz,
        Lx=lx,
        Ly=2.0 * np.pi * y0,
        boundary=boundary_eff,
        y0=y0,
        nperiod=nperiod,
        ntheta=ntheta,
    )
    grid_full = build_spectral_grid(
        apply_imported_geometry_grid_defaults(geom, grid_cfg)
    )

    nl_use = (
        int(args.Nl)
        if args.Nl is not None
        else int(gx_contract.nlaguerre if gx_contract is not None else 8)
    )
    nm_use = (
        int(args.Nm)
        if args.Nm is not None
        else int(gx_contract.nhermite if gx_contract is not None else 16)
    )
    init_state = None
    if args.init_file is not None:
        init_state = jnp.asarray(
            _load_initial_state_from_file(
                args.init_file.expanduser().resolve(),
                nspecies=len(species),
                Nl=nl_use,
                Nm=nm_use,
                ny=int(np.asarray(grid_full.ky).size),
                nx=int(np.asarray(grid_full.kx).size),
                nz=int(np.asarray(grid_full.z).size),
            )
        )

    params = build_linear_params(
        species,
        tau_e=tau_e,
        kpar_scale=float(geom.gradpar()),
        beta=beta,
        fapar=(
            float(gx_contract.fapar)
            if gx_contract is not None
            else (1.0 if beta > 0.0 else 0.0)
        ),
    )
    terms = _build_imported_linear_terms(gx_contract)
    if gx_contract is not None:
        if gx_contract.hypercollisions:
            params = _apply_reference_hypercollisions(params, nhermite=nm_use)
        params = replace(
            params,
            D_hyper=float(gx_contract.D_hyper),
            damp_ends_amp=float(gx_contract.damp_ends_amp),
            damp_ends_widthfrac=float(gx_contract.damp_ends_widthfrac),
        )
    else:
        params = _apply_reference_hypercollisions(params, nhermite=nm_use)
        params = replace(
            params,
            damp_ends_amp=float(args.damp_ends_amp),
            damp_ends_widthfrac=float(args.damp_ends_widthfrac),
        )
    time_cfg = ExplicitTimeConfig(
        dt=dt,
        t_max=float(gx_time[-1]),
        method=(gx_contract.scheme if gx_contract is not None else "rk4"),
        sample_stride=(gx_contract.nwrite if gx_contract is not None else 1),
        fixed_dt=bool(
            (gx_contract is not None and gx_contract.dt is not None)
            or _gx_has_uniform_linear_dt(gx_time, gx_contract)
        ),
        cfl_fac=resolve_cfl_fac(
            (gx_contract.scheme if gx_contract is not None else "rk4"), None
        ),
    )

    rows: list[dict[str, float]] = []
    cache_dir = (
        None if args.cache_dir is None else args.cache_dir.expanduser().resolve()
    )
    for ky_target in ky_values:
        ky_idx = int(np.argmin(np.abs(gx_ky - float(ky_target))))
        gx_kx_idx = _select_gx_kx_index(gx_kx, gx_contract)
        kx_idx = _match_local_kx_index(
            np.asarray(grid_full.kx), float(gx_kx[gx_kx_idx])
        )
        cache_path = None
        if cache_dir is not None:
            cache_path = _series_cache_path(
                cache_dir=cache_dir,
                gx_path=args.gx,
                geometry_file=(
                    geometry_source
                    if gx_contract is None or gx_contract.geo_option != "slab"
                    else None
                ),
                gx_input=args.gx_input,
                ky_target=float(ky_target),
                Nl=nl_use,
                Nm=nm_use,
                mode_method=str(args.mode_method),
                rel_floor_fraction=float(args.rel_floor_fraction),
                sample_steps=sample_steps,
            )
        if cache_path is not None and args.reuse_cache and cache_path.exists():
            gamma, omega, Wg, Wphi, Wapar, Phi2 = _load_cached_ky_series(cache_path)
        else:
            gamma, omega, Wg, Wphi, Wapar, Phi2 = _run_single_ky(
                ky_target=float(ky_target),
                geom=geom,
                grid_full=grid_full,
                params=params,
                time_cfg=time_cfg,
                gx_contract=gx_contract,
                species=tuple(species),
                Nl=nl_use,
                Nm=nm_use,
                reference_times=reference_times,
                output_steps=sample_steps,
                mode_method=args.mode_method,
                kx_index=kx_idx,
                terms=terms,
                G0_override=init_state,
            )
            if cache_path is not None:
                _save_cached_ky_series(
                    cache_path,
                    gamma=gamma,
                    omega=omega,
                    Wg=Wg,
                    Wphi=Wphi,
                    Wapar=Wapar,
                    Phi2=Phi2,
                )
        omega_ref = gx_omega[sample_steps, ky_idx, gx_kx_idx, 0]
        gamma_ref = gx_omega[sample_steps, ky_idx, gx_kx_idx, 1]
        row = {
            "ky": float(ky_target),
            "kx_ref": float(gx_kx[gx_kx_idx]),
            "kx_local": float(np.asarray(grid_full.kx)[kx_idx]),
            "peak_abs_omega_ref": float(np.max(np.abs(omega_ref))),
            "mean_abs_omega": float(np.mean(np.abs(omega - omega_ref))),
            "mean_rel_omega": _mean_rel_error(
                omega, omega_ref, floor_fraction=float(args.rel_floor_fraction)
            ),
            "peak_abs_gamma_ref": float(np.max(np.abs(gamma_ref))),
            "mean_abs_gamma": float(np.mean(np.abs(gamma - gamma_ref))),
            "mean_rel_gamma": _mean_rel_error(
                gamma, gamma_ref, floor_fraction=float(args.rel_floor_fraction)
            ),
            "mean_abs_Wg": float(
                np.mean(np.abs(Wg - distribution_free_energy[sample_steps, ky_idx]))
            ),
            "mean_rel_Wg": _mean_rel_error(
                Wg,
                distribution_free_energy[sample_steps, ky_idx],
                floor_fraction=1.0e-6,
            ),
            "mean_abs_Wphi": float(
                np.mean(np.abs(Wphi - electrostatic_field_energy[sample_steps, ky_idx]))
            ),
            "mean_rel_Wphi": _mean_rel_error(
                Wphi,
                electrostatic_field_energy[sample_steps, ky_idx],
                floor_fraction=1.0e-6,
            ),
            "mean_abs_Wapar": float(
                np.mean(
                    np.abs(
                        Wapar - magnetic_vector_potential_energy[sample_steps, ky_idx]
                    )
                )
            ),
            "mean_rel_Wapar": _mean_rel_error(
                Wapar,
                magnetic_vector_potential_energy[sample_steps, ky_idx],
                floor_fraction=1.0e-6,
            ),
            "omega_last": float(omega[-1]),
            "omega_ref_last": float(omega_ref[-1]),
            "gamma_last": float(gamma[-1]),
            "gamma_ref_last": float(gamma_ref[-1]),
        }
        phi2_ref = gx_Phi2[sample_steps, ky_idx]
        if np.any(np.isfinite(phi2_ref)):
            row.update(
                {
                    "mean_abs_Phi2": float(np.mean(np.abs(Phi2 - phi2_ref))),
                    "mean_rel_Phi2": _mean_rel_error(
                        Phi2, phi2_ref, floor_fraction=1.0e-6
                    ),
                    "Phi2_last": float(Phi2[-1]),
                    "Phi2_ref_last": float(phi2_ref[-1]),
                }
            )
        rows.append(row)
        _write_scan_rows(rows, args.out)

    df = _write_scan_rows(rows, args.out)
    print(df.to_string(index=False))
    if args.out is not None:
        print(f"saved {args.out}")


def build_growth_dump_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--gx-dir-start",
        type=Path,
        required=True,
        help="Directory containing the start diag_state_* dump set.",
    )
    p.add_argument(
        "--gx-dir-stop",
        type=Path,
        required=True,
        help="Directory containing the stop diag_state_* dump set.",
    )
    p.add_argument(
        "--gx-restart-start",
        type=Path,
        default=None,
        help="Optional GX restart.nc file holding the exact start distribution state for late-window replay.",
    )
    p.add_argument(
        "--gx-out",
        type=Path,
        required=True,
        help="GX .out.nc file for times and omega_kxkyt.",
    )
    p.add_argument(
        "--gx-input",
        type=Path,
        required=True,
        help="GX input file describing the imported contract.",
    )
    p.add_argument(
        "--geometry-file",
        type=Path,
        required=True,
        help="Imported geometry file used by GKX.",
    )
    p.add_argument(
        "--time-index-start", type=int, required=True, help="GX diagnostic start index."
    )
    p.add_argument(
        "--time-index-stop", type=int, required=True, help="GX diagnostic stop index."
    )
    p.add_argument(
        "--ky",
        type=float,
        default=None,
        help="Optional ky value to score. Defaults to the smallest positive ky.",
    )
    p.add_argument(
        "--kx",
        type=float,
        default=0.0,
        help="Optional kx value to score. Defaults to 0.",
    )
    p.add_argument("--out", type=Path, default=None, help="Optional CSV output path.")
    return p


def _gx_growth_pair(
    phi_now: np.ndarray, phi_prev: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    z_index = _diagnostic_midplane_index(phi_now.shape[-1])
    phi_now_j = jnp.asarray(phi_now)
    phi_prev_j = jnp.asarray(phi_prev)
    mask = jnp.ones(phi_now.shape[:2], dtype=bool)
    gamma, omega = _instantaneous_growth_rate_step(
        phi_now_j,
        phi_prev_j,
        dt,
        z_index=z_index,
        mask=mask,
        mode_method="z_index",
    )
    return np.asarray(gamma, dtype=float), np.asarray(omega, dtype=float)


def _select_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(np.asarray(values, dtype=float) - float(target))))


def _load_growth_dt(path: Path) -> float:
    raw64 = np.fromfile(path, dtype=np.float64)
    if raw64.size == 1:
        return float(raw64[0])
    raw32 = np.fromfile(path, dtype=np.float32)
    if raw32.size == 1:
        return float(raw32[0])
    raise ValueError(f"unexpected growth dt payload in {path}")


def _gx_active_kx_count(nx_full: int) -> int:
    return 1 + 2 * ((int(nx_full) - 1) // 3)


def _gx_active_ky_count(ny_full: int) -> int:
    return 1 + ((int(ny_full) - 1) // 3)


def _expand_gx_restart_state_to_full_positive_ky(
    state_active: np.ndarray,
    *,
    ny_full: int,
    nx_full: int,
) -> np.ndarray:
    state_active = np.asarray(state_active, dtype=np.complex64)
    if state_active.ndim != 6:
        raise ValueError(f"restart state must have rank 6, got {state_active.shape}")
    nspec, nl, nm, naky, nakx, nz = state_active.shape
    nyc_full = int(ny_full) // 2 + 1
    expected_naky = _gx_active_ky_count(int(ny_full))
    expected_nakx = _gx_active_kx_count(int(nx_full))
    if naky != expected_naky:
        raise ValueError(
            f"restart Nky={naky} does not match ny_full={ny_full} (expected {expected_naky})"
        )
    if nakx != expected_nakx:
        raise ValueError(
            f"restart Nkx={nakx} does not match nx_full={nx_full} (expected {expected_nakx})"
        )

    out = np.zeros((nspec, nl, nm, nyc_full, int(nx_full), nz), dtype=np.complex64)
    split = 1 + ((int(nx_full) - 1) // 3)
    out[..., :naky, :split, :] = state_active[..., :split, :]
    if int(nx_full) > 1:
        for i in range(2 * int(nx_full) // 3 + 1, int(nx_full)):
            it = i - 2 * int(nx_full) // 3 + ((int(nx_full) - 1) // 3)
            out[..., :naky, i, :] = state_active[..., it, :]
    return out


def _load_gx_restart_state(path: Path) -> np.ndarray:
    with Dataset(path, "r") as root:
        if "G" not in root.variables:
            raise ValueError(f"restart file {path} does not contain variable 'G'")
        raw = np.asarray(root.variables["G"][:], dtype=float)
    if raw.ndim != 7 or raw.shape[-1] != 2:
        raise ValueError(f"unexpected GX restart G shape {raw.shape}")
    state = raw[..., 0] + 1j * raw[..., 1]
    # GX restart layout: (species, m, l, z, kx, ky) -> GKX: (species, l, m, ky, kx, z)
    return np.asarray(np.transpose(state, (0, 2, 1, 5, 4, 3)), dtype=np.complex64)


def _load_gx_restart_time(path: Path) -> float:
    with Dataset(path, "r") as root:
        if "time" not in root.variables:
            raise ValueError(f"restart file {path} does not contain variable 'time'")
        return float(np.asarray(root.variables["time"][:], dtype=float).reshape(-1)[0])


def run_growth_dump(argv: list[str] | None = None) -> None:
    args = build_growth_dump_parser().parse_args(argv)

    gx_contract = _load_gx_input_contract(args.gx_input)
    with Dataset(args.gx_out, "r") as root:
        gx_time = np.asarray(root.groups["Grids"].variables["time"][:], dtype=float)
        gx_omega = np.asarray(
            root.groups["Diagnostics"].variables["omega_kxkyt"][:], dtype=float
        )
        nl = int(root.dimensions["l"].size)
        nm = int(root.dimensions["m"].size)
        nspec = int(root.dimensions["s"].size)
    growth_phi_prev_path = (
        args.gx_dir_stop / f"diag_growth_phi_prev_t{args.time_index_stop}.bin"
    )
    growth_phi_path = args.gx_dir_stop / f"diag_growth_phi_t{args.time_index_stop}.bin"
    growth_dt_path = args.gx_dir_stop / f"diag_growth_dt_t{args.time_index_stop}.bin"
    growth_kx_path = args.gx_dir_stop / f"diag_growth_kx_t{args.time_index_stop}.bin"
    growth_ky_path = args.gx_dir_stop / f"diag_growth_ky_t{args.time_index_stop}.bin"
    stop_has_growth = all(
        path.exists()
        for path in (
            growth_phi_prev_path,
            growth_phi_path,
            growth_dt_path,
            growth_kx_path,
            growth_ky_path,
        )
    )
    start_has_state = (
        args.gx_dir_start / f"diag_state_G_s0_t{args.time_index_start}.bin"
    ).exists()
    if stop_has_growth:
        if args.time_index_stop < args.time_index_start:
            raise ValueError(
                "time-index-stop must be >= time-index-start in growth-dump mode"
            )
    elif args.time_index_stop <= args.time_index_start:
        raise ValueError("time-index-stop must be greater than time-index-start")

    if stop_has_growth:
        gx_kx = _load_real_vector_auto(growth_kx_path)
        gx_ky = _load_real_vector_auto(growth_ky_path)
    else:
        gx_kx = _load_real_vector_auto(
            args.gx_dir_start / f"diag_state_kx_t{args.time_index_start}.bin"
        )
        gx_ky = _load_real_vector_auto(
            args.gx_dir_start / f"diag_state_ky_t{args.time_index_start}.bin"
        )
    nyc = int(gx_ky.size)
    nx = int(gx_kx.size)
    phi_seed_path = (
        growth_phi_prev_path
        if stop_has_growth
        else args.gx_dir_start / f"diag_state_phi_t{args.time_index_start}.bin"
    )
    phi_raw = np.fromfile(phi_seed_path, dtype=np.complex64)
    if phi_raw.size % max(nyc * nx, 1) != 0:
        raise ValueError(
            f"{phi_seed_path.name} size {phi_raw.size} is not divisible by nyc*nx={nyc * nx}"
        )
    nz = int(phi_raw.size // (nyc * nx))

    if stop_has_growth:
        gx_phi_start = _load_field(growth_phi_prev_path, nyc, nx, nz)
        gx_phi_stop = _load_field(growth_phi_path, nyc, nx, nz)
        target = _load_growth_dt(growth_dt_path)
        gx_apar_stop = None
        gx_bpar_stop = None
        gx_G_start = None
        restart_time = None
        restart_state_active = None
        if args.gx_restart_start is not None:
            restart_state_active = _load_gx_restart_state(args.gx_restart_start)
            restart_time = _load_gx_restart_time(args.gx_restart_start)
        elif start_has_state:
            gx_G_start = _load_species_state(
                args.gx_dir_start,
                nspec=nspec,
                nl=nl,
                nm=nm,
                nyc=nyc,
                nx=nx,
                nz=nz,
                time_index=args.time_index_start,
            )
    else:
        gx_G_start = _load_species_state(
            args.gx_dir_start,
            nspec=nspec,
            nl=nl,
            nm=nm,
            nyc=nyc,
            nx=nx,
            nz=nz,
            time_index=args.time_index_start,
        )
        gx_phi_start = _load_field(
            args.gx_dir_start / f"diag_state_phi_t{args.time_index_start}.bin",
            nyc,
            nx,
            nz,
        )
        gx_phi_stop = _load_field(
            args.gx_dir_stop / f"diag_state_phi_t{args.time_index_stop}.bin",
            nyc,
            nx,
            nz,
        )
        gx_apar_stop = _maybe_load_field(
            args.gx_dir_stop / f"diag_state_apar_t{args.time_index_stop}.bin",
            nyc,
            nx,
            nz,
        )
        gx_bpar_stop = _maybe_load_field(
            args.gx_dir_stop / f"diag_state_bpar_t{args.time_index_stop}.bin",
            nyc,
            nx,
            nz,
        )

    y0 = (
        float(gx_contract.y0)
        if np.isfinite(float(gx_contract.y0))
        else _infer_y0(gx_ky)
    )
    ny_full = _resolve_imported_real_fft_ny(gx_ky, gx_contract)
    if gx_contract.geo_option == "slab":
        geom = SlabGeometry.from_config(
            GeometryConfig(
                model="slab",
                s_hat=float(gx_contract.s_hat),
                zero_shat=bool(gx_contract.zero_shat),
            )
        )
    else:
        geom = load_imported_geometry_netcdf(
            _resolve_internal_geometry_source(
                geometry_file=args.geometry_file, runtime_config=None
            )
        )

    boundary_eff = _resolve_imported_boundary(
        gx_contract.boundary, zero_shat=bool(gx_contract.zero_shat)
    )
    lx = 2.0 * np.pi * y0 if boundary_eff == "periodic" else 62.8
    grid_cfg = apply_imported_geometry_grid_defaults(
        geom,
        GridConfig(
            Nx=int(nx),
            Ny=int(ny_full),
            Nz=int(nz),
            Lx=lx,
            Ly=2.0 * np.pi * y0,
            boundary=boundary_eff,
            y0=y0,
            nperiod=max(1, int(gx_contract.nperiod)),
            ntheta=max(1, int(gx_contract.ntheta)),
        ),
    )
    grid_full = build_spectral_grid(grid_cfg)
    grid = select_real_fft_ky_grid(grid_full, gx_ky.astype(np.float32))

    if stop_has_growth and args.gx_restart_start is not None:
        if restart_state_active is None:
            raise ValueError(
                "restart_state_active must be available for restart-based growth replay"
            )
        gx_G_start = _expand_gx_restart_state_to_full_positive_ky(
            restart_state_active,
            ny_full=ny_full,
            nx_full=int(nx),
        )
        gx_G_start = np.asarray(gx_G_start, dtype=np.complex64) * np.complex64(
            gx_contract.restart_scale
        )
        if gx_contract.restart_with_perturb:
            gx_G_start = gx_G_start + np.asarray(
                _build_imported_initial_condition(
                    grid=grid,
                    geom=geom,
                    gx_contract=gx_contract,
                    species=gx_contract.species,
                    ky_index=0,
                    kx_index=0,
                    Nl=nl,
                    Nm=nm,
                ),
                dtype=np.complex64,
            )

    params = build_linear_params(
        gx_contract.species,
        tau_e=float(gx_contract.tau_e),
        kpar_scale=float(geom.gradpar()),
        beta=float(gx_contract.beta),
        fapar=float(gx_contract.fapar),
    )
    terms = _build_imported_linear_terms(gx_contract)
    if gx_contract.hypercollisions:
        params = _apply_reference_hypercollisions(params, nhermite=nm)
    params = replace(
        params,
        D_hyper=float(gx_contract.D_hyper),
        damp_ends_amp=float(gx_contract.damp_ends_amp),
        damp_ends_widthfrac=float(gx_contract.damp_ends_widthfrac),
    )
    cache = build_linear_cache(grid, geom, params, nl, nm)
    dt = _infer_gx_linear_dt(gx_time, gx_contract)
    time_cfg = ExplicitTimeConfig(
        dt=dt,
        t_max=float(gx_time[args.time_index_stop] - gx_time[args.time_index_start])
        if not stop_has_growth
        else float(target),
        method=str(gx_contract.scheme),
        sample_stride=max(1, int(gx_contract.nwrite)),
        fixed_dt=bool(
            (gx_contract.dt is not None)
            or _gx_has_uniform_linear_dt(gx_time, gx_contract)
        ),
        cfl_fac=resolve_cfl_fac(str(gx_contract.scheme), None),
    )
    dt_min = float(time_cfg.dt_min)
    dt_max = (
        float(time_cfg.dt_max) if time_cfg.dt_max is not None else float(time_cfg.dt)
    )

    gamma_gx_dump, omega_gx_dump = _gx_growth_pair(gx_phi_stop, gx_phi_start, target)
    if stop_has_growth and gx_G_start is None:
        gamma_sp_dump, omega_sp_dump = gamma_gx_dump, omega_gx_dump
    else:
        G = jnp.asarray(gx_G_start, dtype=jnp.complex64)
        omega_max = _linear_frequency_bound(grid, geom, params, nl, nm)
        wmax = float(np.sum(omega_max))
        t = 0.0
        phi_prev_step = gx_phi_start

        def _step(G_state, cache_state, params_state, term_cfg_state, dt_state):
            return _linear_explicit_step(
                G_state,
                cache_state,
                params_state,
                term_cfg_state,
                dt_state,
                method=time_cfg.method,
            )

        stepper = jax.jit(_step, donate_argnums=(0,))
        term_cfg = _linear_term_config(terms)
        if stop_has_growth and args.gx_restart_start is not None:
            if restart_time is None:
                raise ValueError(
                    "restart_time must be available for restart-based growth replay"
                )
            target = float(gx_time[args.time_index_stop] - restart_time)
            step_dt = float(_load_growth_dt(growth_dt_path))
            if step_dt <= 0.0:
                raise ValueError("growth dump dt must be > 0")
            nsteps_float = target / step_dt
            nsteps = int(np.rint(nsteps_float))
            if nsteps < 1 or not np.isclose(
                target, step_dt * nsteps, rtol=1.0e-6, atol=1.0e-10
            ):
                raise ValueError(
                    "restart-based growth replay requires a uniform late window; "
                    f"got target={target:.12g}, step_dt={step_dt:.12g}, ratio={nsteps_float:.12g}"
                )
            for _ in range(nsteps):
                G, fields = stepper(G, cache, params, term_cfg, step_dt)
                t += step_dt
                if t < target - 0.5 * step_dt:
                    phi_prev_step = np.asarray(fields.phi, dtype=np.complex64)
            gamma_sp_dump, omega_sp_dump = _gx_growth_pair(
                np.asarray(fields.phi, dtype=np.complex64), phi_prev_step, step_dt
            )
        else:
            while t < target - 1.0e-12:
                dt_step = float(time_cfg.dt)
                if not time_cfg.fixed_dt and wmax > 0.0:
                    dt_guess = float(time_cfg.cfl_fac) * float(time_cfg.cfl) / wmax
                    dt_step = min(max(dt_guess, dt_min), dt_max)
                remaining = target - t
                if dt_step > remaining:
                    dt_step = max(remaining, dt_min)
                G, fields = stepper(G, cache, params, term_cfg, dt_step)
                t += dt_step

            sp_phi_stop = np.asarray(fields.phi, dtype=np.complex64)
            sp_apar_stop = (
                np.asarray(fields.apar, dtype=np.complex64)
                if fields.apar is not None
                else np.zeros_like(gx_phi_stop, dtype=np.complex64)
            )
            sp_bpar_stop = (
                np.asarray(fields.bpar, dtype=np.complex64)
                if fields.bpar is not None
                else np.zeros_like(gx_phi_stop, dtype=np.complex64)
            )
            _ = (gx_apar_stop, gx_bpar_stop, sp_apar_stop, sp_bpar_stop)
            gamma_sp_dump, omega_sp_dump = _gx_growth_pair(
                sp_phi_stop, gx_phi_start, target
            )

    ky_target = (
        float(args.ky) if args.ky is not None else float(np.min(gx_ky[gx_ky > 0.0]))
    )
    ky_idx = _select_index(gx_ky, ky_target)
    kx_idx = _select_index(gx_kx, float(args.kx))

    row = {
        "time_index_start": int(args.time_index_start),
        "time_index_stop": int(args.time_index_stop),
        "t_start": float(gx_time[args.time_index_start]),
        "t_restart_start": (
            float(restart_time)
            if stop_has_growth and args.gx_restart_start is not None
            else np.nan
        ),
        "t_stop": float(gx_time[args.time_index_stop]),
        "delta_t": float(target),
        "compare_mode": (
            "growth_dump"
            if stop_has_growth and gx_G_start is None
            else (
                "growth_restart_replay"
                if stop_has_growth and args.gx_restart_start is not None
                else ("growth_replay" if stop_has_growth else "state_replay")
            )
        ),
        "ky": float(gx_kx.size and gx_ky[ky_idx]),
        "kx": float(gx_kx[kx_idx]),
        "omega_out": float(gx_omega[args.time_index_stop, ky_idx, kx_idx, 0]),
        "gamma_out": float(gx_omega[args.time_index_stop, ky_idx, kx_idx, 1]),
        "omega_gx_dump": float(omega_gx_dump[ky_idx, kx_idx]),
        "gamma_gx_dump": float(gamma_gx_dump[ky_idx, kx_idx]),
        "omega_sp_dump": float(omega_sp_dump[ky_idx, kx_idx]),
        "gamma_sp_dump": float(gamma_sp_dump[ky_idx, kx_idx]),
    }
    row["abs_omega_out_vs_gx_dump"] = abs(row["omega_out"] - row["omega_gx_dump"])
    row["abs_gamma_out_vs_gx_dump"] = abs(row["gamma_out"] - row["gamma_gx_dump"])
    row["abs_omega_sp_vs_gx_dump"] = abs(row["omega_sp_dump"] - row["omega_gx_dump"])
    row["abs_gamma_sp_vs_gx_dump"] = abs(row["gamma_sp_dump"] - row["gamma_gx_dump"])
    row["rel_omega_sp_vs_gx_dump"] = row["abs_omega_sp_vs_gx_dump"] / max(
        abs(row["omega_gx_dump"]), 1.0e-12
    )
    row["rel_gamma_sp_vs_gx_dump"] = row["abs_gamma_sp_vs_gx_dump"] / max(
        abs(row["gamma_gx_dump"]), 1.0e-12
    )

    df = pd.DataFrame([row])
    print(df.to_string(index=False))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"saved {args.out}")


def build_window_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--gx-dir",
        type=Path,
        required=True,
        help="Directory containing GX diag_state dump binaries",
    )
    p.add_argument(
        "--gx-out",
        type=Path,
        required=True,
        help="GX .out.nc file for dimensions and times",
    )
    p.add_argument(
        "--gx-input",
        type=Path,
        required=True,
        help="GX input file describing the imported contract",
    )
    p.add_argument(
        "--geometry-file",
        type=Path,
        required=True,
        help="Imported geometry file used by GKX",
    )
    p.add_argument(
        "--time-index-start", type=int, required=True, help="GX diag_state start index"
    )
    p.add_argument(
        "--time-index-stop", type=int, required=True, help="GX diag_state stop index"
    )
    p.add_argument("--out", type=Path, default=None, help="Optional CSV summary output")
    return p


def _rel_err(test: np.ndarray, ref: np.ndarray) -> float:
    ref_abs = np.abs(ref)
    denom = np.maximum(ref_abs, 1.0e-30)
    return float(np.max(np.abs(test - ref) / denom))


def _print_array_summary(label: str, ref: np.ndarray, test: np.ndarray) -> None:
    """Print compact absolute and relative errors for one saved state array."""

    diff = test - ref
    max_ref = float(np.max(np.abs(ref)))
    mask = np.abs(ref) > max_ref * 1.0e-12
    if max_ref == 0.0 or not np.any(mask):
        max_rel = rms_rel = float("nan")
    else:
        max_rel = float(np.max(np.abs(diff[mask] / ref[mask])))
        rms_rel = float(
            np.sqrt(np.mean(np.abs(diff[mask]) ** 2))
            / (np.sqrt(np.mean(np.abs(ref[mask]) ** 2)) + 1.0e-30)
        )
    idx_diff = np.unravel_index(int(np.argmax(np.abs(diff))), diff.shape)
    print(
        f"{label:12s} max|ref|={max_ref:.3e} "
        f"max|test|={float(np.max(np.abs(test))):.3e} "
        f"max|diff|={float(np.max(np.abs(diff))):.3e} "
        f"max|rel|={max_rel:.3e} rms_rel={rms_rel:.3e} idx={idx_diff} "
        f"ref={ref[idx_diff]:.3e} test={test[idx_diff]:.3e}"
    )


def _gx_phi2_total(phi: jnp.ndarray, vol_fac: jnp.ndarray) -> float:
    return float(jnp.sum(jnp.abs(phi) ** 2 * vol_fac[None, None, :]))


def run_window(argv: list[str] | None = None) -> None:
    args = build_window_parser().parse_args(argv)

    gx_contract = _load_gx_input_contract(args.gx_input)
    with Dataset(args.gx_out, "r") as root:
        nl = int(root.dimensions["l"].size)
        nm = int(root.dimensions["m"].size)
        nspec = int(root.dimensions["s"].size)
        gx_time = np.asarray(root.groups["Grids"].variables["time"][:], dtype=float)
    if args.time_index_start < 0 or args.time_index_start >= gx_time.size:
        raise ValueError(
            f"time-index-start={args.time_index_start} outside [0, {gx_time.size - 1}]"
        )
    if args.time_index_stop < 0 or args.time_index_stop >= gx_time.size:
        raise ValueError(
            f"time-index-stop={args.time_index_stop} outside [0, {gx_time.size - 1}]"
        )
    if args.time_index_stop <= args.time_index_start:
        raise ValueError("time-index-stop must be greater than time-index-start")

    gx_kx = _load_real_vector_auto(
        args.gx_dir / f"diag_state_kx_t{args.time_index_start}.bin"
    )
    gx_ky = _load_real_vector_auto(
        args.gx_dir / f"diag_state_ky_t{args.time_index_start}.bin"
    )
    nyc = int(gx_ky.size)
    nx = int(gx_kx.size)
    phi_raw = np.fromfile(
        args.gx_dir / f"diag_state_phi_t{args.time_index_start}.bin", dtype=np.complex64
    )
    if phi_raw.size % max(nyc * nx, 1) != 0:
        raise ValueError(
            f"diag_state_phi_t{args.time_index_start}.bin size {phi_raw.size} is not divisible by nyc*nx={nyc * nx}"
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
    gx_phi_start = _load_field(
        args.gx_dir / f"diag_state_phi_t{args.time_index_start}.bin", nyc, nx, nz
    )
    gx_apar_start = _maybe_load_field(
        args.gx_dir / f"diag_state_apar_t{args.time_index_start}.bin", nyc, nx, nz
    )
    gx_bpar_start = _maybe_load_field(
        args.gx_dir / f"diag_state_bpar_t{args.time_index_start}.bin", nyc, nx, nz
    )
    gx_G_stop = _load_species_state(
        args.gx_dir,
        nspec=nspec,
        nl=nl,
        nm=nm,
        nyc=nyc,
        nx=nx,
        nz=nz,
        time_index=args.time_index_stop,
    )
    gx_phi_stop = _load_field(
        args.gx_dir / f"diag_state_phi_t{args.time_index_stop}.bin", nyc, nx, nz
    )
    gx_apar_stop = _maybe_load_field(
        args.gx_dir / f"diag_state_apar_t{args.time_index_stop}.bin", nyc, nx, nz
    )
    gx_bpar_stop = _maybe_load_field(
        args.gx_dir / f"diag_state_bpar_t{args.time_index_stop}.bin", nyc, nx, nz
    )

    y0 = (
        float(gx_contract.y0)
        if np.isfinite(float(gx_contract.y0))
        else _infer_y0(gx_ky)
    )
    ny_full = _resolve_imported_real_fft_ny(gx_ky, gx_contract)
    if gx_contract.geo_option == "slab":
        geom = SlabGeometry.from_config(
            GeometryConfig(
                model="slab",
                s_hat=float(gx_contract.s_hat),
                zero_shat=bool(gx_contract.zero_shat),
            )
        )
    else:
        geom = load_imported_geometry_netcdf(
            _resolve_internal_geometry_source(
                geometry_file=args.geometry_file, runtime_config=None
            )
        )

    boundary_eff = _resolve_imported_boundary(
        gx_contract.boundary, zero_shat=bool(gx_contract.zero_shat)
    )
    lx = 2.0 * np.pi * y0 if boundary_eff == "periodic" else 62.8
    grid_cfg = apply_imported_geometry_grid_defaults(
        geom,
        GridConfig(
            Nx=int(nx),
            Ny=int(ny_full),
            Nz=int(nz),
            Lx=lx,
            Ly=2.0 * np.pi * y0,
            boundary=boundary_eff,
            y0=y0,
            nperiod=max(1, int(gx_contract.nperiod)),
            ntheta=max(1, int(gx_contract.ntheta)),
        ),
    )
    grid_full = build_spectral_grid(grid_cfg)
    grid = select_real_fft_ky_grid(grid_full, gx_ky.astype(np.float32))

    params = build_linear_params(
        gx_contract.species,
        tau_e=float(gx_contract.tau_e),
        kpar_scale=float(geom.gradpar()),
        beta=float(gx_contract.beta),
        fapar=float(gx_contract.fapar),
    )
    terms = _build_imported_linear_terms(gx_contract)
    if gx_contract.hypercollisions:
        params = _apply_reference_hypercollisions(params, nhermite=nm)
    params = replace(
        params,
        D_hyper=float(gx_contract.D_hyper),
        damp_ends_amp=float(gx_contract.damp_ends_amp),
        damp_ends_widthfrac=float(gx_contract.damp_ends_widthfrac),
    )

    cache = build_linear_cache(grid, geom, params, nl, nm)
    dt = _infer_gx_linear_dt(gx_time, gx_contract)
    time_cfg = ExplicitTimeConfig(
        dt=dt,
        t_max=float(gx_time[args.time_index_stop] - gx_time[args.time_index_start]),
        method=str(gx_contract.scheme),
        sample_stride=max(1, int(gx_contract.nwrite)),
        fixed_dt=bool(
            (gx_contract.dt is not None)
            or _gx_has_uniform_linear_dt(gx_time, gx_contract)
        ),
        cfl_fac=resolve_cfl_fac(str(gx_contract.scheme), None),
    )
    dt_min = float(time_cfg.dt_min)
    dt_max = (
        float(time_cfg.dt_max) if time_cfg.dt_max is not None else float(time_cfg.dt)
    )

    G = jnp.asarray(gx_G_start, dtype=jnp.complex64)
    omega_max = _linear_frequency_bound(grid, geom, params, nl, nm)
    wmax = float(np.sum(omega_max))
    t = 0.0
    target = float(time_cfg.t_max)

    def _step(G_state, cache_state, params_state, term_cfg_state, dt_state):
        return _linear_explicit_step(
            G_state,
            cache_state,
            params_state,
            term_cfg_state,
            dt_state,
            method=time_cfg.method,
        )

    stepper = jax.jit(_step, donate_argnums=(0,))
    term_cfg = _linear_term_config(terms)
    step_count = 0
    while t < target - 1.0e-12:
        dt_step = float(time_cfg.dt)
        if not time_cfg.fixed_dt and wmax > 0.0:
            dt_guess = float(time_cfg.cfl_fac) * float(time_cfg.cfl) / wmax
            dt_step = min(max(dt_guess, dt_min), dt_max)
        remaining = target - t
        if dt_step > remaining:
            dt_step = max(remaining, dt_min)
        G, fields = stepper(G, cache, params, term_cfg, dt_step)
        t += dt_step
        step_count += 1

    sp_G_stop = np.asarray(G, dtype=np.complex64)
    sp_phi_stop = np.asarray(fields.phi, dtype=np.complex64)
    sp_apar_stop = (
        np.asarray(fields.apar, dtype=np.complex64)
        if fields.apar is not None
        else np.zeros_like(gx_phi_stop, dtype=np.complex64)
    )
    sp_bpar_stop = (
        np.asarray(fields.bpar, dtype=np.complex64)
        if fields.bpar is not None
        else np.zeros_like(gx_phi_stop, dtype=np.complex64)
    )
    gx_apar_stop_use = (
        gx_apar_stop
        if gx_apar_stop is not None
        else np.zeros_like(gx_phi_stop, dtype=np.complex64)
    )
    gx_bpar_stop_use = (
        gx_bpar_stop
        if gx_bpar_stop is not None
        else np.zeros_like(gx_phi_stop, dtype=np.complex64)
    )

    print(
        f"time_index_start={args.time_index_start} t_start={gx_time[args.time_index_start]:.8f} "
        f"time_index_stop={args.time_index_stop} t_stop={gx_time[args.time_index_stop]:.8f} "
        f"delta_t={target:.8f} steps={step_count} t_match={t:.8f}"
    )
    _print_array_summary(
        "start_phi",
        gx_phi_start.astype(np.complex64),
        gx_phi_start.astype(np.complex64),
    )
    if gx_apar_start is not None:
        _print_array_summary(
            "start_apar",
            gx_apar_start.astype(np.complex64),
            gx_apar_start.astype(np.complex64),
        )
    if gx_bpar_start is not None:
        _print_array_summary(
            "start_bpar",
            gx_bpar_start.astype(np.complex64),
            gx_bpar_start.astype(np.complex64),
        )
    _print_array_summary("stop_g_state", gx_G_stop.astype(np.complex64), sp_G_stop)
    _print_array_summary("stop_phi", gx_phi_stop.astype(np.complex64), sp_phi_stop)
    if gx_apar_stop is not None or fields.apar is not None:
        _print_array_summary(
            "stop_apar", gx_apar_stop_use.astype(np.complex64), sp_apar_stop
        )
    if gx_bpar_stop is not None or fields.bpar is not None:
        _print_array_summary(
            "stop_bpar", gx_bpar_stop_use.astype(np.complex64), sp_bpar_stop
        )

    vol_fac, _flux_fac = fieldline_quadrature_weights(geom, grid)
    distribution_free_energy_stop = float(
        distribution_free_energy(jnp.asarray(gx_G_stop), grid, params, vol_fac)
    )
    sp_Wg_stop = float(
        distribution_free_energy(jnp.asarray(sp_G_stop), grid, params, vol_fac)
    )
    electrostatic_field_energy_stop = float(
        electrostatic_field_energy(jnp.asarray(gx_phi_stop), cache, params, vol_fac)
    )
    sp_Wphi_stop = float(
        electrostatic_field_energy(jnp.asarray(sp_phi_stop), cache, params, vol_fac)
    )
    magnetic_vector_potential_energy_stop = float(
        magnetic_vector_potential_energy(jnp.asarray(gx_apar_stop_use), cache, vol_fac)
    )
    sp_Wapar_stop = float(
        magnetic_vector_potential_energy(jnp.asarray(sp_apar_stop), cache, vol_fac)
    )
    gx_Phi2_stop = _gx_phi2_total(jnp.asarray(gx_phi_stop), vol_fac)
    sp_Phi2_stop = _gx_phi2_total(jnp.asarray(sp_phi_stop), vol_fac)

    rows = [
        {"metric": "g_state", "rel": _rel_err(sp_G_stop, gx_G_stop)},
        {"metric": "phi", "rel": _rel_err(sp_phi_stop, gx_phi_stop)},
        {"metric": "apar", "rel": _rel_err(sp_apar_stop, gx_apar_stop_use)},
        {"metric": "bpar", "rel": _rel_err(sp_bpar_stop, gx_bpar_stop_use)},
        {
            "metric": "Wg",
            "gx_stop": distribution_free_energy_stop,
            "gkx": sp_Wg_stop,
            "rel": abs(sp_Wg_stop - distribution_free_energy_stop)
            / max(abs(distribution_free_energy_stop), 1.0e-30),
        },
        {
            "metric": "Wphi",
            "gx_stop": electrostatic_field_energy_stop,
            "gkx": sp_Wphi_stop,
            "rel": abs(sp_Wphi_stop - electrostatic_field_energy_stop)
            / max(abs(electrostatic_field_energy_stop), 1.0e-30),
        },
        {
            "metric": "Wapar",
            "gx_stop": magnetic_vector_potential_energy_stop,
            "gkx": sp_Wapar_stop,
            "rel": abs(sp_Wapar_stop - magnetic_vector_potential_energy_stop)
            / max(abs(magnetic_vector_potential_energy_stop), 1.0e-30),
        },
        {
            "metric": "Phi2",
            "gx_stop": gx_Phi2_stop,
            "gkx": sp_Phi2_stop,
            "rel": abs(sp_Phi2_stop - gx_Phi2_stop) / max(abs(gx_Phi2_stop), 1.0e-30),
        },
    ]
    print("metric     rel")
    for row in rows[:4]:
        print(f"{row['metric']:8s} {float(row['rel']): .3e}")
    print("diag       gx_stop       gkx      rel")
    for row in rows[4:]:
        print(
            f"{row['metric']:8s} {float(row['gx_stop']): .6e} {float(row['gkx']): .6e} {float(row['rel']): .3e}"
        )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)


def main(argv: list[str] | None = None) -> None:
    """Dispatch one imported-linear comparison workflow."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fields", "growth-dump", "window"))
    args, remainder = parser.parse_known_args(argv)
    if args.mode == "fields":
        run_fields(remainder)
    elif args.mode == "growth-dump":
        run_growth_dump(remainder)
    else:
        run_window(remainder)


if __name__ == "__main__":
    main()
