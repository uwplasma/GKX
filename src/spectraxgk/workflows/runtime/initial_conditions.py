"""Runtime initial-condition and restart-state construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, cast

import jax.numpy as jnp
import numpy as np

from spectraxgk.artifacts.restart import load_netcdf_restart_state
from spectraxgk.core.grid import SpectralGrid
from spectraxgk.geometry import FluxTubeGeometryLike
from spectraxgk.operators.linear.cache import build_linear_cache
from spectraxgk.operators.linear.params import LinearParams
from spectraxgk.workflows.runtime.config import RuntimeConfig
from spectraxgk.workflows.runtime.initial_phi import _density_moments_for_target_phi

_GLIBC_RAND_MAX = float((1 << 31) - 1)
_INITIAL_FIELD_MOMENTS: dict[str, tuple[int, int]] = {
    "density": (0, 0),
    "upar": (0, 1),
    "tpar": (0, 2),
    "tperp": (1, 0),
    "qpar": (0, 3),
    "qperp": (1, 1),
}
_ALL_FIELD_SCALES: dict[str, float] = {
    "density": 1.0,
    "upar": 1.0,
    "tpar": 1.0 / np.sqrt(2.0),
    "tperp": 1.0,
    "qpar": 1.0 / np.sqrt(6.0),
    "qperp": 1.0,
}
_VALID_INIT_FIELDS = {"all", "phi", *_INITIAL_FIELD_MOMENTS.keys()}


def _build_gaussian_profile(
    z: np.ndarray,
    *,
    kx: float,
    ky: float,
    s_hat: float,
    width: float,
    envelope_constant: float,
    envelope_sine: float,
) -> np.ndarray:
    if ky == 0.0:
        return np.zeros_like(z)
    theta0 = kx / (s_hat * ky)
    env = envelope_constant + envelope_sine * np.sin(z - theta0)
    return env * np.exp(-(((z - theta0) / width) ** 2))


def _build_single_phi_gaussian_profile(
    z: np.ndarray,
    *,
    kx: float,
    ky: float,
    s_hat: float,
    width: float,
    envelope_constant: float,
    envelope_sine: float,
) -> np.ndarray:
    """Return a single-mode Gaussian potential profile along the flux tube.

    The W7-X zonal-flow benchmark prescribes a Gaussian electrostatic-potential
    perturbation centered in the middle of the flux tube. A multi-mode
    ballooning-angle Gaussian initializer is undefined for ``ky=0`` because its
    center uses ``kx / (s_hat * ky)``; for the zonal case the physically stated
    center is therefore the tube midpoint, ``z=0``.
    """

    if ky != 0.0 and s_hat != 0.0:
        return _build_gaussian_profile(
            z,
            kx=kx,
            ky=ky,
            s_hat=s_hat,
            width=width,
            envelope_constant=envelope_constant,
            envelope_sine=envelope_sine,
        )
    center = 0.0
    env = envelope_constant + envelope_sine * np.sin(z - center)
    return env * np.exp(-(((z - center) / width) ** 2))


def _reshape_netcdf_state(
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
    arr = raw.reshape((nspec, nm, nl, nR)).transpose(0, 2, 1, 3)
    ky_idx = np.arange(nyc)[:, None, None]
    kx_idx = np.arange(nx)[None, :, None]
    z_idx = np.arange(nz)[None, None, :]
    idxyz = ky_idx + nyc * (kx_idx + nx * z_idx)
    arr_reordered = arr[..., idxyz.ravel()]
    return arr_reordered.reshape((nspec, nl, nm, nyc, nx, nz))


def _expand_ky(arr: np.ndarray, *, nyc: int) -> np.ndarray:
    ny_full = 2 * (nyc - 1)
    if ny_full <= 0 or arr.shape[-3] == ny_full:
        return arr
    if nyc <= 2:
        return arr
    pos = arr
    neg = np.conj(pos[..., 1 : nyc - 1, :, :])
    neg = neg[..., ::-1, :, :]
    nx = pos.shape[-2]
    if nx > 1:
        kx_neg = np.concatenate(([0], np.arange(nx - 1, 0, -1)))
        neg = neg[..., kx_neg, :]
    return np.concatenate([pos, neg], axis=-3)


def _enforce_full_ky_hermitian(arr: np.ndarray) -> np.ndarray:
    """Mirror positive-`ky` content into the negative branch for full FFT grids."""

    state = np.asarray(arr, dtype=np.complex64)
    ny = int(state.shape[-3])
    if ny <= 1:
        return state
    nyc = ny // 2 + 1
    neg_hi = nyc - 1 if (ny % 2) == 0 else nyc
    if neg_hi <= 1:
        return state
    neg = np.conj(state[..., 1:neg_hi, :, :])[..., ::-1, :, :]
    nx = int(state.shape[-2])
    if nx > 1:
        kx_neg = np.concatenate(([0], np.arange(nx - 1, 0, -1)))
        neg = neg[..., kx_neg, :]
    state[..., nyc:, :, :] = neg
    return state


def _load_initial_state_from_file(
    path: Path,
    *,
    nspecies: int,
    Nl: int,
    Nm: int,
    ny: int,
    nx: int,
    nz: int,
) -> np.ndarray:
    if path.suffix.lower() == ".nc":
        return load_netcdf_restart_state(
            path,
            nspecies=nspecies,
            Nl=Nl,
            Nm=Nm,
            ny=ny,
            nx=nx,
            nz=nz,
        )
    raw = np.fromfile(path, dtype=np.complex64)
    nyc = ny // 2 + 1
    expected_nyc = nspecies * Nl * Nm * nyc * nx * nz
    expected_full = nspecies * Nl * Nm * ny * nx * nz
    if raw.size == expected_nyc:
        arr = _reshape_netcdf_state(
            raw, nspec=nspecies, nl=Nl, nm=Nm, nyc=nyc, nx=nx, nz=nz
        )
        return _expand_ky(arr, nyc=nyc)
    if raw.size == expected_full:
        return raw.reshape((nspecies, Nl, Nm, ny, nx, nz))
    raise ValueError(
        f"init_file size {raw.size} does not match expected {expected_nyc} (nyc) or {expected_full} (full)"
    )


def _centered_glibc_random_pairs(seed: int, count: int) -> np.ndarray:
    """Return centered random pairs using glibc `rand()` semantics."""

    if count <= 0:
        return np.empty((0, 2), dtype=np.float64)

    seed_use = 1 if int(seed) == 0 else int(seed)
    state = np.zeros(344 + 2 * count, dtype=np.uint64)
    state[0] = np.uint64(seed_use)
    for i in range(1, 31):
        state[i] = np.uint64((16807 * int(state[i - 1])) % int(_GLIBC_RAND_MAX))
    for i in range(31, 34):
        state[i] = state[i - 31]
    for i in range(34, state.size):
        state[i] = (state[i - 31] + state[i - 3]) & np.uint64(0xFFFFFFFF)

    rand_vals = (state[344:] >> np.uint64(1)).astype(np.float64, copy=False)
    half = 0.5 * _GLIBC_RAND_MAX
    inv = 1.0 / _GLIBC_RAND_MAX
    pairs = np.empty((count, 2), dtype=np.float64)
    for i in range(count):
        pairs[i, 0] = (rand_vals[2 * i] - half) * inv
        pairs[i, 1] = (rand_vals[2 * i + 1] - half) * inv
    return pairs


def _dealiased_initial_mode_pairs(grid: SpectralGrid) -> list[tuple[int, int]]:
    """Return the dealiased startup-loop `(kx, ky)` pairs for multimode initial conditions."""

    nx = int(np.asarray(grid.kx).size)
    ny = int(np.asarray(grid.ky).size)
    kx_max = 1 + (nx - 1) // 3
    ky_max = 1 + (ny - 1) // 3
    return [
        (int(kx_i), int(ky_i)) for kx_i in range(kx_max) for ky_i in range(1, ky_max)
    ]


def _periodic_zp_from_grid(z: np.ndarray) -> float:
    """Return periodic `Zp` from the discrete theta grid."""

    z_arr = np.asarray(z, dtype=float)
    if z_arr.size <= 1:
        return 1.0
    dz = float(z_arr[1] - z_arr[0])
    period = abs(dz) * float(z_arr.size)
    if period <= 0.0:
        return 1.0
    return period / (2.0 * np.pi)


def _validate_initialization(cfg: RuntimeConfig) -> tuple[str, str]:
    init_field = cfg.init.init_field.lower()
    if init_field not in _VALID_INIT_FIELDS:
        raise ValueError(
            "init_field must be one of {'density','upar','tpar','tperp','qpar','qperp','all','phi'}"
        )
    if cfg.init.gaussian_width <= 0.0:
        raise ValueError("gaussian_width must be > 0")
    init_file_mode = cfg.init.init_file_mode.strip().lower()
    if init_file_mode not in {"replace", "add"}:
        raise ValueError("init_file_mode must be one of {'replace', 'add'}")
    return init_field, init_file_mode


def _scaled_restart_state(
    cfg: RuntimeConfig,
    grid: SpectralGrid,
    *,
    Nl: int,
    Nm: int,
    nspecies: int,
) -> np.ndarray | None:
    if cfg.init.init_file is None:
        return None
    loaded_state = _load_initial_state_from_file(
        Path(cfg.init.init_file),
        nspecies=nspecies,
        Nl=Nl,
        Nm=Nm,
        ny=grid.ky.size,
        nx=grid.kx.size,
        nz=grid.z.size,
    )
    return np.asarray(loaded_state, dtype=np.complex64) * np.complex64(
        float(cfg.init.init_file_scale)
    )


def _species_targets(cfg: RuntimeConfig, nspecies: int) -> tuple[int, ...]:
    if nspecies == 1:
        return (0,)
    if not cfg.init.init_electrons_only:
        return tuple(range(nspecies))

    electron_indices = tuple(
        i for i, sp in enumerate(cfg.species[:nspecies]) if float(sp.charge) < 0.0
    )
    return electron_indices or (nspecies - 1,)


def _single_mode_values(
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    cfg: RuntimeConfig,
    *,
    init_field: str,
    ky_index: int,
    kx_index: int,
) -> np.ndarray:
    z = np.asarray(grid.z)
    z_period = _periodic_zp_from_grid(z)
    z_phase = np.cos(float(cfg.init.kpar_init) * z / z_period)
    amp = float(cfg.init.init_amp)
    if cfg.init.init_single and cfg.init.gaussian_init and init_field == "phi":
        profile = _build_single_phi_gaussian_profile(
            z,
            kx=float(grid.kx[kx_index]),
            ky=float(grid.ky[ky_index]),
            s_hat=float(geom.s_hat),
            width=float(cfg.init.gaussian_width),
            envelope_constant=float(cfg.init.gaussian_envelope_constant),
            envelope_sine=float(cfg.init.gaussian_envelope_sine),
        )
        return amp * profile.astype(np.complex64, copy=False)
    return amp * z_phase.astype(np.complex64, copy=False)


def _validate_named_moment_resolution(init_field: str, *, Nl: int, Nm: int) -> None:
    if init_field in {"all", "phi"}:
        return
    l_idx, m_idx = _INITIAL_FIELD_MOMENTS[init_field]
    if l_idx >= Nl or m_idx >= Nm:
        raise ValueError("init_field moment exceeds (Nl, Nm) resolution")


@dataclass
class _InitialConditionBuilder:
    grid: SpectralGrid
    geom: FluxTubeGeometryLike
    cfg: RuntimeConfig
    Nl: int
    Nm: int
    state: np.ndarray
    species_targets: tuple[int, ...]
    build_runtime_linear_params_fn: Callable[..., LinearParams]
    phi_seed_context: tuple[object, LinearParams] | None = None

    def set_mode(
        self, l_idx: int, m_idx: int, ky_i: int, kx_i: int, vals_k: np.ndarray
    ) -> None:
        if l_idx >= self.Nl or m_idx >= self.Nm:
            return
        for s_idx in self.species_targets:
            self.state[s_idx, l_idx, m_idx, ky_i, kx_i, :] = vals_k

    def set_named_mode_scaled(
        self, field_name: str, ky_i: int, kx_i: int, vals_k: np.ndarray
    ) -> None:
        l_idx, m_idx = _INITIAL_FIELD_MOMENTS[field_name]
        self.set_mode(l_idx, m_idx, ky_i, kx_i, vals_k * _ALL_FIELD_SCALES[field_name])

    def set_named_mode_raw(
        self, field_name: str, ky_i: int, kx_i: int, vals_k: np.ndarray
    ) -> None:
        l_idx, m_idx = _INITIAL_FIELD_MOMENTS[field_name]
        self.set_mode(l_idx, m_idx, ky_i, kx_i, vals_k)

    def set_phi_mode(self, ky_i: int, kx_i: int, vals_k: np.ndarray) -> None:
        if self.Nl < 1 or self.Nm < 1:
            raise ValueError(
                "init_field='phi' requires at least one Laguerre and one Hermite moment"
            )
        if self.phi_seed_context is None:
            phi_params = self.build_runtime_linear_params_fn(
                self.cfg, Nm=self.Nm, geom=self.geom
            )
            self.phi_seed_context = (
                build_linear_cache(self.grid, self.geom, phi_params, self.Nl, self.Nm),
                phi_params,
            )
        cache, phi_params = self.phi_seed_context
        seeds = _density_moments_for_target_phi(
            np.asarray(vals_k, dtype=np.complex64),
            cache=cache,
            params=phi_params,
            ky_i=int(ky_i),
            kx_i=int(kx_i),
            species_targets=self.species_targets,
        )
        for s_idx, seed_vals in seeds.items():
            self.state[s_idx, 0, 0, ky_i, kx_i, :] = seed_vals

    def seed_field(
        self, init_field: str, ky_i: int, kx_i: int, vals_k: np.ndarray
    ) -> None:
        if init_field == "all":
            for field_name in _INITIAL_FIELD_MOMENTS:
                self.set_named_mode_scaled(field_name, ky_i, kx_i, vals_k)
        elif init_field == "phi":
            self.set_phi_mode(ky_i, kx_i, vals_k)
        else:
            self.set_named_mode_raw(init_field, ky_i, kx_i, vals_k)


def _seed_gaussian_multimode(
    builder: _InitialConditionBuilder,
    *,
    init_field: str,
    amp: float,
) -> None:
    z = np.asarray(builder.grid.z)
    nx = builder.grid.kx.size
    for kx_i, ky_i in _dealiased_initial_mode_pairs(builder.grid):
        ky_k = float(builder.grid.ky[ky_i])
        if ky_k == 0.0:
            continue
        profile_k = _build_gaussian_profile(
            z,
            kx=abs(float(builder.grid.kx[kx_i])),
            ky=ky_k,
            s_hat=float(builder.geom.s_hat),
            width=float(builder.cfg.init.gaussian_width),
            envelope_constant=float(builder.cfg.init.gaussian_envelope_constant),
            envelope_sine=float(builder.cfg.init.gaussian_envelope_sine),
        )
        vals_k = amp * profile_k * (1.0 + 1.0j)
        builder.seed_field(init_field, ky_i, kx_i, vals_k)
        if kx_i != 0:
            builder.seed_field(init_field, ky_i, int(nx - kx_i), vals_k)


def _seed_random_multimode(
    builder: _InitialConditionBuilder,
    *,
    init_field: str,
    amp: float,
) -> None:
    _validate_named_moment_resolution(init_field, Nl=builder.Nl, Nm=builder.Nm)
    z = np.asarray(builder.grid.z)
    z_phase = np.cos(float(builder.cfg.init.kpar_init) * z / _periodic_zp_from_grid(z))
    nx = builder.grid.kx.size
    active_modes = _dealiased_initial_mode_pairs(builder.grid)
    rand_pairs = amp * _centered_glibc_random_pairs(
        int(builder.cfg.init.random_seed), len(active_modes)
    )
    for (kx_i, ky_i), (ra, rb) in zip(active_modes, rand_pairs, strict=True):
        vals_k = ((rb + 1j * ra) if kx_i == 0 else (ra + 1j * rb)) * z_phase
        builder.seed_field(init_field, ky_i, kx_i, vals_k)
        if kx_i != 0:
            vals_neg = (rb + 1j * ra) * z_phase
            builder.seed_field(init_field, ky_i, int(nx - kx_i), vals_neg)


def _finalize_initial_state(
    grid: SpectralGrid,
    state: np.ndarray,
    *,
    loaded_state: np.ndarray | None,
    init_file_mode: str,
) -> jnp.ndarray:
    if grid.ky.size > 1 and np.any(np.asarray(grid.ky) < 0.0):
        state = _enforce_full_ky_hermitian(state)
    if loaded_state is None:
        return jnp.asarray(state)
    if init_file_mode == "replace":
        return jnp.asarray(loaded_state)
    return jnp.asarray(cast(np.ndarray, loaded_state + state))


def _build_initial_condition_impl(
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    cfg: RuntimeConfig,
    *,
    ky_index: int,
    kx_index: int,
    Nl: int,
    Nm: int,
    nspecies: int,
    build_runtime_linear_params_fn: Callable[..., LinearParams],
) -> jnp.ndarray:
    init_field, init_file_mode = _validate_initialization(cfg)
    state: np.ndarray = np.zeros(
        (nspecies, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size),
        dtype=np.complex64,
    )
    loaded_state = _scaled_restart_state(cfg, grid, Nl=Nl, Nm=Nm, nspecies=nspecies)
    amp = float(cfg.init.init_amp)
    builder = _InitialConditionBuilder(
        grid=grid,
        geom=geom,
        cfg=cfg,
        Nl=Nl,
        Nm=Nm,
        state=state,
        species_targets=_species_targets(cfg, nspecies),
        build_runtime_linear_params_fn=build_runtime_linear_params_fn,
    )

    if cfg.init.gaussian_init and not cfg.init.init_single:
        _seed_gaussian_multimode(builder, init_field=init_field, amp=amp)
    elif not cfg.init.init_single and not cfg.init.gaussian_init:
        _seed_random_multimode(builder, init_field=init_field, amp=amp)
    else:
        _validate_named_moment_resolution(init_field, Nl=Nl, Nm=Nm)
        vals = _single_mode_values(
            grid,
            geom,
            cfg,
            init_field=init_field,
            ky_index=ky_index,
            kx_index=kx_index,
        )
        builder.seed_field(init_field, ky_index, kx_index, vals)

    return _finalize_initial_state(
        grid,
        state,
        loaded_state=loaded_state,
        init_file_mode=init_file_mode,
    )
