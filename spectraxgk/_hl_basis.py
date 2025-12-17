# spectraxgk/_hl_basis.py
import math
from functools import lru_cache

import jax
import jax.numpy as jnp

__all__ = [
    "twothirds_mask",
    "kgrid_fftshifted",
    "laguerre_L_all",
    "J_l_all",
    "alpha_tensor",
    "conjugate_index_fftshifted",
]

def twothirds_mask(Ny: int, Nx: int, Nz: int):
    """Boolean mask in fftshifted ordering keeping |mode| <= N//3 per dim."""
    def centered_modes(N):
        k = jnp.fft.fftfreq(N) * N
        return jnp.fft.fftshift(k)

    ky = centered_modes(Ny)[:, None, None]
    kx = centered_modes(Nx)[None, :, None]
    kz = centered_modes(Nz)[None, None, :]

    cy, cx, cz = Ny // 3, Nx // 3, Nz // 3
    return (jnp.abs(ky) <= cy) & (jnp.abs(kx) <= cx) & (jnp.abs(kz) <= cz)

def kgrid_fftshifted(Lx, Ly, Lz, Nx, Ny, Nz):
    """
    Return fftshifted physical wavenumber grids (rad / length) compatible with arrays
    stored in fftshift order.
    """
    kx_1d = jnp.fft.fftshift(jnp.fft.fftfreq(Nx, d=Lx / Nx)) * (2 * jnp.pi)
    ky_1d = jnp.fft.fftshift(jnp.fft.fftfreq(Ny, d=Ly / Ny)) * (2 * jnp.pi)
    kz_1d = jnp.fft.fftshift(jnp.fft.fftfreq(Nz, d=Lz / Nz)) * (2 * jnp.pi)
    ky, kx, kz = jnp.meshgrid(ky_1d, kx_1d, kz_1d, indexing="ij")
    return kx, ky, kz

def conjugate_index_fftshifted(i: int, N: int) -> int:
    """
    Index map for the conjugate (-k) mode in *fftshifted* ordering.
    For odd N: i' = (N-1) - i ; for even N: i' = (N - i) mod N.
    """
    if (N % 2) == 1:
        return (N - 1) - i
    return (N - i) % N

def laguerre_L_all(b, Nl: int):
    """
    Compute Laguerre polynomials L_0..L_{Nl-1} at b (broadcastable array).
    Recurrence:
      L0=1
      L1=1-b
      L_{l+1}=((2l+1-b)L_l - l L_{l-1})/(l+1)
    """
    b = jnp.asarray(b)
    L0 = jnp.ones_like(b)
    if Nl == 1:
        return L0[None, ...]
    L1 = 1.0 - b
    if Nl == 2:
        return jnp.stack([L0, L1], axis=0)

    def body(carry, l):
        Lm1, Lm = carry
        Lp = ((2.0 * l + 1.0 - b) * Lm - l * Lm1) / (l + 1.0)
        return (Lm, Lp), Lp

    (_, _), rest = jax.lax.scan(body, (L0, L1), jnp.arange(1, Nl - 1))
    return jnp.concatenate([jnp.stack([L0, L1], axis=0), rest], axis=0)

def J_l_all(b, Nl: int):
    """
    Gyroaveraging coefficients used in Laguerre–Hermite GK:
      J_l(b) = exp(-b/2) * L_l(b)
    """
    L = laguerre_L_all(b, Nl)
    return jnp.exp(-0.5 * b)[None, ...] * L

def _log_fact(n: int) -> float:
    return math.lgamma(n + 1)

@lru_cache(maxsize=None)
def _alpha_tensor_cached(Nl: int):
    """
    CPU-side cached computation of alpha_{k l n}. Nl is small (typically <= 16).
    Returned as a Python nested list of floats; converted to jnp in alpha_tensor().
    """
    out = [[[0.0 for _ in range(Nl)] for _ in range(Nl)] for _ in range(Nl)]
    for k in range(Nl):
        for ell in range(Nl):
            for n in range(Nl):
                if n < abs(k - ell) or n > (k + ell):
                    continue
                j_min = max(0, int(math.ceil((k + ell - n) / 2)))
                j_max = min(k, ell, k + ell - n)
                acc = 0.0
                for j in range(j_min, j_max + 1):
                    a = k + ell - j
                    b = k - j
                    c = ell - j
                    d = 2 * j - k - ell + n
                    e = k + ell - n - j
                    if min(a, b, c, d, e) < 0:
                        continue
                    log_term = (
                        _log_fact(a)
                        + (2 * j - k - ell + n) * math.log(2.0)
                        - (_log_fact(b) + _log_fact(c) + _log_fact(d) + _log_fact(e))
                    )
                    acc += math.exp(log_term)
                out[k][ell][n] = acc
    return out

def alpha_tensor(Nl: int):
    """
    Precompute alpha_{k l n} Laguerre triple-product coefficients.
    Cached so you don’t redo Python loops every simulation.
    """
    dtype = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32
    return jnp.array(_alpha_tensor_cached(Nl), dtype=dtype)
