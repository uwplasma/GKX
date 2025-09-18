"""
Fourier–Hermite mode-bank evolution:
- Build H_k for each k, add collisions, evolve c_k(t) by Diffrax or Eig.
- Initial conditions:
  landau: c0_k(0) = amplitude * δ_{k, k0}; two_stream: use the same scaffold but different seed if desired.
"""

from typing import Tuple
import numpy as np
import jax
import jax.numpy as jnp
from hermite_ops import streaming_block_fourier, field_one_sided_fourier, build_collision_matrix

# Optional Diffrax
try:
    from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt, PIDController
    HAS_DIFFRAX = True
except Exception:
    HAS_DIFFRAX = False

@jax.jit
def eig_cache(A: jnp.ndarray):
    w, V = jnp.linalg.eig(A)
    Vinv = jnp.linalg.inv(V)
    return w, V, Vinv

@jax.jit
def evolve_cached(w, V, Vinv, c0, ts):
    alpha = Vinv @ c0
    phases = jnp.exp(w[:, None] * ts[None, :])
    return V @ (phases * alpha[:, None])

def rhs_real(t, y, A):
    # For Diffrax: real 2N system for one k
    N = A.shape[0]
    x = y[:N]; z = y[N:]
    # dot c = A c; split real/imag: A = (-i H - C) is complex
    dx = (jnp.real(A) @ x - jnp.imag(A) @ z) + 0.0
    dz = (jnp.imag(A) @ x + jnp.real(A) @ z) + 0.0
    return jnp.concatenate([dx, dz])

def build_A_k(k: float, N: int, nu0: float, hyper_p: int, cutoff: int) -> jnp.ndarray:
    H = streaming_block_fourier(k, N) + field_one_sided_fourier(k, N)
    C = build_collision_matrix(N, nu0, hyper_p, cutoff)
    return (-1j) * H - C

def run_bank(kvals: np.ndarray, N: int, nu0: float, hyper_p: int, cutoff: int,
             backend: str, tmax: float, nt: int,
             amplitude: float, seed_c1: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (ts, C_knt, Ek_kt), where:
      ts: (nt,)
      C_knt: (Nk, N, nt) Hermite coefficients per k
      Ek_kt: (Nk, nt) electric field per k, with E_k = i c_{0k} / k
    """
    ts = jnp.linspace(0.0, tmax, nt, dtype=jnp.float64)
    Nk = len(kvals)
    C_store = []
    E_store = []
    for k in kvals:
        A = build_A_k(float(k), N, nu0, hyper_p, cutoff)
        c0 = jnp.zeros((N,), dtype=jnp.complex128).at[0].set(amplitude)
        if seed_c1:
            c0 = c0.at[1].set(0.1 * amplitude)
        if backend == "eig":
            w, V, Vinv = eig_cache(A)
            C = evolve_cached(w, V, Vinv, c0, ts)  # (N, nt)
        else:
            if not HAS_DIFFRAX:
                raise RuntimeError("Diffrax not installed.")
            y0 = jnp.concatenate([jnp.real(c0), jnp.imag(c0)])
            sol = diffeqsolve(
                ODETerm(rhs_real), Tsit5(),
                t0=0.0, t1=float(tmax), dt0=1e-3,
                y0=y0, args=A,
                stepsize_controller=None, saveat=SaveAt(ts=ts),
                max_steps=2_000_000
            )
            Y = sol.ys
            Xr, Xi = Y[:, :N], Y[:, N:]
            C = (Xr + 1j * Xi).T
        Ek = 1j * C[0, :] / k
        C_store.append(np.asarray(C))
        E_store.append(np.asarray(Ek))
    C_knt = np.stack(C_store, axis=0)
    Ek_kt = np.stack(E_store, axis=0)
    return np.asarray(ts), C_knt, Ek_kt
