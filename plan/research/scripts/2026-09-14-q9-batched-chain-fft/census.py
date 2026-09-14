"""Chain-class census per grid, and the padding/tiling algebra check.

Prints (nChains, nLinks) per class for the Cyclone nonlinear deck at three
grids, the chain samples current layout transforms, and what the two
single-launch layouts would cost:
  pad  : every chain zero-padded to the longest length (one FFT length)
  tile : every chain repeated periodically to lcm of lengths (exact)
Then checks numerically that padding is not the operator and tiling is.
"""

from dataclasses import replace
import math
import json
import sys

import numpy as np

from gkx.core_grid import build_spectral_grid
from gkx.geometry import apply_imported_geometry_grid_defaults
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.runtime import build_runtime_geometry, build_runtime_linear_params
from gkx.workflows.runtime.toml import load_runtime_from_toml

cfg, _ = load_runtime_from_toml(
    "examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml"
)
rows = []
for nx, ny, nz in ((32, 32, 24), (64, 64, 24), (96, 96, 48)):
    c = replace(cfg, grid=replace(cfg.grid, Nx=nx, Ny=ny, Nz=nz, ntheta=nz))
    geom = build_runtime_geometry(c)
    grid = build_spectral_grid(apply_imported_geometry_grid_defaults(geom, c.grid))
    params = build_runtime_linear_params(c, Nm=2, geom=geom)
    cache = build_linear_cache(grid, geom, params, 1, 2)
    classes = [tuple(int(v) for v in np.shape(idx)) for idx in cache.linked_indices]
    nzg = int(grid.z.size)
    lengths = [nl for _, nl in classes]
    samples = sum(nc * nl * nzg for nc, nl in classes)
    nmax = max(lengths)
    pad = sum(nc for nc, _ in classes) * nmax * nzg
    lcm = math.lcm(*lengths)
    tile = sum(nc for nc, _ in classes) * lcm * nzg
    row = dict(
        grid=f"{nx}x{ny}x{nz}",
        Nz=nzg,
        classes=classes,
        n_classes=len(classes),
        chain_samples=samples,
        pad_samples=pad,
        pad_ratio=round(pad / samples, 3),
        lcm_links=lcm,
        tile_samples=tile,
        tile_ratio=round(tile / samples, 3),
    )
    rows.append(row)
    print(json.dumps(row))

# Algebra check in float64: the spectral derivative on Z_N of a chain of
# N = L*Nz samples, against (a) zero padding to M and (b) periodic tiling to M.
rng = np.random.default_rng(0)
dz = 0.26
for n, m in ((24, 48), (48, 120), (72, 120), (24, 72), (96, 288)):
    f = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    k_n = 2 * np.pi * np.fft.fftfreq(n, d=dz)
    ref = np.fft.ifft(1j * k_n * np.fft.fft(f))
    k_m = 2 * np.pi * np.fft.fftfreq(m, d=dz)
    padded = np.concatenate([f, np.zeros(m - n)])
    pad_out = np.fft.ifft(1j * k_m * np.fft.fft(padded))[:n]
    msg = dict(N=n, M=m, pad_rel=float(np.linalg.norm(pad_out - ref) / np.linalg.norm(ref)))
    if m % n == 0:
        tiled = np.tile(f, m // n)
        tile_out = np.fft.ifft(1j * k_m * np.fft.fft(tiled))[:n]
        msg["tile_rel"] = float(np.linalg.norm(tile_out - ref) / np.linalg.norm(ref))
        # Bin j*(M/N) of the tiled transform carries (M/N)*fhat_N[j] and the
        # same wavenumber, Nyquist sign included; every other bin is zero.
        fh_n = np.fft.fft(f)
        fh_m = np.fft.fft(tiled)
        r = m // n
        msg["tile_bins_rel"] = float(
            np.linalg.norm(fh_m[::r] - r * fh_n) / np.linalg.norm(r * fh_n)
        )
        msg["tile_offbins_max"] = float(np.max(np.abs(np.delete(fh_m, np.arange(0, m, r)))))
        msg["tile_k_equal"] = bool(np.array_equal(k_m[::r], k_n))
    print(json.dumps(msg))

if len(sys.argv) > 1:
    with open(sys.argv[1], "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
