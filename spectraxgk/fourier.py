# fourier.py
"""
Fourier–Hermite mode bank.
Build H_k and C once, form A_k = (-i H_k - C), then evolve either by Diffrax
(ODE per k, real 2N system) or by batched eigendecomposition.
Returns consistent keys: {"k", "C_knt", "Ek_kt"}.
"""

from typing import Tuple
from functools import partial
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from hermite_ops import (
    streaming_block_fourier,  # H_stream(k,N)
    field_one_sided_fourier,  # H_field(k,N)
    build_collision_matrix    # C(N; nu0, hyper_p, cutoff)
)

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

@partial(jax.jit, static_argnames=["N", "hyper_p", "cutoff"])
def _A_from_k(k: float, N: int, nu0: float, hyper_p: int, cutoff: int) -> jnp.ndarray:
    """Helper used by vmap: build H(k) = H_stream(k) + H_field(k)."""
    H = streaming_block_fourier(k, N) + field_one_sided_fourier(k, N)           # (N,N) float64
    C = build_collision_matrix(N, nu0, hyper_p, cutoff)                         # (N,N) float64
    A = (-1j) * H.astype(jnp.complex128) - C.astype(jnp.complex128)             # (N,N) complex128
    return A


def _batch_build_A(kvals: jnp.ndarray, N: int, nu0: float, hyper_p: int, cutoff: int) -> jnp.ndarray:
    # vmap ONLY over k; N/hyper_p/cutoff stay python-statics for the jitted _A_from_k
    build_one = lambda kk: _A_from_k(kk, N, nu0, hyper_p, cutoff)
    return jax.vmap(build_one, in_axes=(0,))(kvals)  # (Nk, N, N)


# ---------------- Eig backend (batched) ----------------
@jax.jit
def _eig_evolve_batch(A_k: jnp.ndarray, c0: jnp.ndarray, ts: jnp.ndarray) -> jnp.ndarray:
    """
    Batched eigendecomposition and evolution across k.
      A_k: (Nk, N, N) complex
      c0:  (N,)       complex  (same initial condition for all k)
      ts:  (nt,)      real
    Returns:
      C_knt: (Nk, N, nt) complex
    """
    def one_k(A):
        w, V = jnp.linalg.eig(A)
        Vinv = jnp.linalg.inv(V)
        alpha = Vinv @ c0
        phases = jnp.exp(w[:, None] * ts[None, :])       # (N, nt)
        C = V @ (phases * alpha[:, None])                # (N, nt)
        return C

    return jax.vmap(one_k, in_axes=0)(A_k)               # (Nk, N, nt)


# ---------------- Diffrax backend (per-k ODE) ----------------
def rhs_real(t, y, A):
    """
    Real 2N state for dot c = (-i H - C) c.
    Since Re(A)=-C, Im(A)=-H:
      dx/dt = -C x + H y
      dy/dt = -H x - C y
    """
    N = A.shape[0]
    x = y[:N]; z = y[N:]
    Ar, Ai = jnp.real(A), jnp.imag(A)
    dx = Ar @ x - Ai @ z
    dz = Ai @ x + Ar @ z
    return jnp.concatenate([dx, dz])


def run_bank(kvals: jnp.ndarray, N: int, nu0: float, hyper_p: int, cutoff: int,
             backend: str, tmax: float, nt: int,
             amplitude: float, seed_c1: bool) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Return (ts, C_knt, Ek_kt), where:
      ts: (nt,)
      C_knt: (Nk, N, nt) Hermite coefficients per k
      Ek_kt: (Nk, nt) electric field per k, with E_k = i c_{0k} / k
    """
    kvals = kvals.astype(jnp.float64)
    ts = jnp.linspace(0.0, tmax, nt, dtype=jnp.float64)

    # Build all A(k) at once (fast & JIT friendly)
    A_k = _batch_build_A(kvals, N, nu0, hyper_p, cutoff)  # (Nk, N, N) complex128

    # Initial state (complex128)
    amp = jnp.asarray(amplitude, jnp.float64)
    base = jnp.zeros((N,), dtype=jnp.complex128).at[0].set(amp)
    if seed_c1:
        base = base.at[1].set(0.1 * amp)

    def solve_one(A, k):
        if backend == "eig":
            w, V, Vinv = eig_cache(A)
            C = evolve_cached(w, V, Vinv, base, ts)                       # (N, nt)
        else:
            if not HAS_DIFFRAX:
                raise RuntimeError("Diffrax not installed.")
            # from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt, PIDController
            y0 = jnp.concatenate([jnp.real(base), jnp.imag(base)])
            controller = PIDController(rtol=1e-7, atol=1e-10, jump_ts=ts)
            sol = diffeqsolve(
                ODETerm(rhs_real), Tsit5(),
                t0=0.0, t1=float(tmax), dt0=1e-3,
                y0=y0, args=A,
                stepsize_controller=controller, saveat=SaveAt(ts=ts),
                max_steps=2_000_000
            )
            Y = sol.ys  # (nt, 2N)
            Xr, Xi = Y[:, :N], Y[:, N:]
            C = (Xr + 1j * Xi).T
        Ek = 1j * C[0, :] / k
        return C, Ek

    # vmap over k-dimension
    C_knt, Ek_kt = jax.vmap(solve_one, in_axes=(0, 0))(A_k, kvals)   # ((Nk,N,nt),(Nk,nt))
    return jnp.asarray(ts), jnp.asarray(C_knt), jnp.asarray(Ek_kt)
