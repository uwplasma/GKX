"""Where does stacked-vs-separate linked FFT lose bits? (proto trees only)

For the nl64 cache: jit(_linked_fft_apply(stack or tuple, ("grad","abs")))[i]
vs jit(_linked_fft_apply(x_i, op_i)); then the same on the raw pieces:
fft of a (2, ...) array vs fft of each slot, multiply, ifft.
"""

import sys
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import gkx
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear import streaming as S
from tools.profiling.profile_runtime_kernels import (
    apply_imported_geometry_grid_defaults,
    build_runtime_geometry,
    build_runtime_linear_params,
    build_spectral_grid,
    load_runtime_from_toml,
)

print("gkx", gkx.__file__, "x64", jax.config.read("jax_enable_x64"))
n, nl, nm = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
cfg, _ = load_runtime_from_toml(Path("examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml"))
cfg = replace(cfg, grid=replace(cfg.grid, Nx=n, Ny=n, Nz=24))
geom = build_runtime_geometry(cfg)
grid = build_spectral_grid(apply_imported_geometry_grid_defaults(geom, cfg.grid))
params = build_runtime_linear_params(cfg, Nm=nm, geom=geom)
cache = build_linear_cache(grid, geom, params, nl, nm)
cdt = jnp.complex128 if jax.config.read("jax_enable_x64") else jnp.complex64
rng = np.random.default_rng(3)
shape = (1, nl, nm, grid.ky.size, grid.kx.size, grid.z.size)
a = jnp.asarray(rng.standard_normal(shape) + 1j * rng.standard_normal(shape), cdt)
b = jnp.asarray(rng.standard_normal(shape) + 1j * rng.standard_normal(shape), cdt)
route = dict(
    linked_inverse_permutation=cache.linked_inverse_permutation,
    linked_full_cover=cache.linked_full_cover,
    linked_gather_map=cache.linked_gather_map,
    linked_gather_mask=cache.linked_gather_mask,
    linked_use_gather=cache.linked_use_gather,
)
li, lk = cache.linked_indices, cache.linked_kz


def eq(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return bool(np.array_equal(x, y)), float(np.linalg.norm(x - y) / np.linalg.norm(x))


sep_a = jax.jit(lambda x: S._linked_fft_apply(x, li, lk, operator="grad", **route))(a)
sep_b = jax.jit(lambda x: S._linked_fft_apply(x, li, lk, operator="abs", **route))(b)
st = jax.jit(lambda x, y: S._linked_fft_apply(jnp.stack([x, y]), li, lk, operator=("grad", "abs"), **route))(a, b)
print("stack  grad", eq(sep_a, st[0]), "abs", eq(sep_b, st[1]))
try:
    tu = jax.jit(lambda x, y: S._linked_fft_apply((x, y), li, lk, operator=("grad", "abs"), **route))(a, b)
    print("tuple  grad", eq(sep_a, tu[0]), "abs", eq(sep_b, tu[1]))
except Exception as exc:  # proto1 has no tuple route
    print("tuple route unavailable:", type(exc).__name__)

# raw pieces per class
for (idx, kz) in zip(li, lk):
    nc, nlk = idx.shape
    x = jnp.asarray(rng.standard_normal((2, 1, nl, nm, nc, nlk * 24)) + 0j, cdt)
    f_st = jax.jit(lambda v: jnp.fft.fft(v, axis=-1))(x)
    f_0 = jax.jit(lambda v: jnp.fft.fft(v, axis=-1))(x[1])
    m_abs = S._fft_abs_multiplier(kz, f_0)
    mult = jax.jit(lambda v: S._linked_fft_multiplier(kz, v, ("grad", "abs")) * v)(f_st)
    mult_0 = jax.jit(lambda v: m_abs * v)(f_0)
    i_st = jax.jit(lambda v: jnp.fft.ifft(v, axis=-1))(mult)
    i_0 = jax.jit(lambda v: jnp.fft.ifft(v, axis=-1))(mult_0)
    print((nc, nlk), "fft", eq(f_0, f_st[1])[0], "mult", eq(mult_0, mult[1])[0], "ifft", eq(i_0, i_st[1])[0])
