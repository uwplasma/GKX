from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import lax, vmap

from .basis import hermite_coupling_factors, lb_eigenvalues


class StreamingOperator(eqx.Module):
    """Hermite streaming along straight B; Fourier in z with kpar.

    dC_{n,m}/dt += -i * (kpar*vth/sqrt(2)) * [ sqrt(n+1) C_{n+1,m} + sqrt(n) C_{n-1,m} ]
    """
    Nn: int = eqx.field(static=True)
    Nm: int = eqx.field(static=True)
    kpar: float
    vth: float
    sqrt_n: jnp.ndarray
    sqrt_np1: jnp.ndarray

    def __init__(self, Nn: int, Nm: int, kpar: float, vth: float):
        object.__setattr__(self, "Nn", Nn)
        object.__setattr__(self, "Nm", Nm)
        object.__setattr__(self, "kpar", kpar)
        object.__setattr__(self, "vth", vth)
        sn, snp1 = hermite_coupling_factors(Nn)
        object.__setattr__(self, "sqrt_n", sn)
        object.__setattr__(self, "sqrt_np1", snp1)

    @property
    def a(self) -> jnp.ndarray:
        return self.kpar * self.vth / jnp.sqrt(2.0)

    def __call__(self, C: jnp.ndarray) -> jnp.ndarray:
        """Complex-space streaming RHS."""
        # nearest-neighbor Hermite coupling (same Laguerre m)
        upper = jnp.pad(C[1:, :], ((0, 1), (0, 0))) * self.sqrt_np1[:, None]
        lower = jnp.pad(C[:-1, :], ((1, 0), (0, 0))) * self.sqrt_n[:, None]
        return -1j * self.a * (upper + lower)

class StreamingOperatorKS(eqx.Module):
    """Streaming with multiple Fourier modes: expects C[k, n, m]."""
    Nn: int = eqx.field(static=True)
    Nm: int = eqx.field(static=True)
    accepts_3d: bool = eqx.field(static=True, default=True)
    ks: jnp.ndarray          # (Nk,)
    vth: float
    sqrt_n: jnp.ndarray
    sqrt_np1: jnp.ndarray

    def __init__(self, ks: jnp.ndarray, Nn: int, Nm: int, vth: float):
        object.__setattr__(self, "ks", ks)
        object.__setattr__(self, "Nn", Nn)
        object.__setattr__(self, "Nm", Nm)
        object.__setattr__(self, "vth", vth)
        sn, snp1 = hermite_coupling_factors(Nn)
        object.__setattr__(self, "sqrt_n", sn)
        object.__setattr__(self, "sqrt_np1", snp1)

    def _couple_nm(self, A_nm: jnp.ndarray) -> jnp.ndarray:
        # A_nm has shape (Nn, Nm) for a single k-slice
        upper = jnp.pad(A_nm[1:, :], ((0, 1), (0, 0))) * self.sqrt_np1[:, None]
        lower = jnp.pad(A_nm[:-1, :], ((1, 0), (0, 0))) * self.sqrt_n[:, None]
        return upper + lower

    def __call__(self, C: jnp.ndarray) -> jnp.ndarray:
        # C shape: (Nk, Nn, Nm)
        assert C.ndim == 3, "StreamingOperatorKS expects C[k,n,m]"
        # Apply Hermite neighbor coupling per k, then scale by -i * k_j * vth / sqrt(2)
        S = vmap(self._couple_nm, in_axes=0, out_axes=0)(C)   # (Nk,Nn,Nm)
        factor = -1j * self.ks * (self.vth / jnp.sqrt(2.0))       # (Nk,)
        return S * factor[:, None, None]


class LenardBernstein(eqx.Module):
    """Diagonal Lenard–Bernstein model in Hermite–Laguerre space: dC/dt += -nu*(α n + β m) C."""
    Nn: int = eqx.field(static=True)
    Nm: int = eqx.field(static=True)
    nu: float
    alpha: float = eqx.field(static=True, default=1.0)
    beta:  float = eqx.field(static=True, default=2.0)

    def __call__(self, C: jnp.ndarray) -> jnp.ndarray:
        lam = lb_eigenvalues(self.Nn, self.Nm, self.alpha, self.beta)
        return -self.nu * lam * C


class ElectrostaticDrive(eqx.Module):
    """Very simple electrostatic drive:
       E_∥ ~ C_{0,0} / kpar; dC_{1,0}/dt += - coef * sqrt(2) * i * E_∥
    """
    Nn: int = eqx.field(static=True)
    Nm: int = eqx.field(static=True)
    kpar: float
    coef: float

    def __call__(self, C: jnp.ndarray) -> jnp.ndarray:
        k = self.kpar
        E = jnp.where(k != 0.0, 1j * C[0, 0] / k, 0.0 + 0.0j)
        dC = jnp.zeros_like(C)
        dC = dC.at[1, 0].add(-1 * self.coef * jnp.sqrt(2.0) * E)
        return dC

class NonlinearConvolution(eqx.Module):
    """Quadratic E–a convolution in Fourier space, only Laguerre m=0 is active.

    State expected (nonlinear mode): C[k, n, m], complex.
      E_k = i * a_{k, n=0, m=0} / k
      N_{k, n>=1, m=0} = sqrt(2n) * sum_{k'} E_{k'} * a_{k-k', n-1, m=0}
                       = sqrt(2n) * IFFT[ FFT(E) * FFT(a[:, n-1]) ]

    Dealiasing options:
      - two_thirds: zero spectral coefficients for |idx| > floor(Nk/3)
      - houli:     multiply by exp(-36 * (|idx|/ (Nk/2))^36)
      - none:      do nothing

    Notes:
      * Assumes a uniform FFT grid in k (use klist of equally spaced values).
      * Leaves Laguerre m>0 untouched; only m=0 updated.
    """
    Nk: int = eqx.field(static=True)
    Nn: int = eqx.field(static=True)
    Nm: int = eqx.field(static=True)
    accepts_3d: bool = eqx.field(static=True, default=True)
    nl_filter: str = eqx.field(static=True, default="two_thirds")  # "two_thirds"|"houli"|"none"
    ks: jnp.ndarray  # (Nk,) real k-values, used only to build E; FFT uses index space

    def __init__(self, ks: jnp.ndarray, Nn: int, Nm: int, nl_filter: str = "two_thirds"):
        Nk = int(ks.shape[0])
        object.__setattr__(self, "Nk", Nk)
        object.__setattr__(self, "Nn", Nn)
        object.__setattr__(self, "Nm", Nm)
        object.__setattr__(self, "nl_filter", nl_filter)
        object.__setattr__(self, "ks", ks)

    def _dealise(self, Xh: jnp.ndarray) -> jnp.ndarray:
        """Apply chosen spectral filter to a Fourier array Xh[k] (complex)."""
        if self.nl_filter == "none":
            return Xh
        Nk = self.Nk
        idx = jnp.fft.fftfreq(Nk) * Nk  # integer-like indices in [-Nk/2, Nk/2)
        abs_idx = jnp.abs(idx)
        if self.nl_filter == "two_thirds":
            cutoff = Nk / 3.0   # keep |idx| <= Nk/3
            mask = (abs_idx <= cutoff)
            return Xh * mask.astype(Xh.dtype)
        elif self.nl_filter == "houli":
            # Hou–Li filter: exp(-36 (|k|/k_max)^36). Use index space: k_max ~ Nk/2.
            kmax = Nk / 2.0
            sigma = jnp.exp(-36.0 * (abs_idx / kmax) ** 36)
            return Xh * sigma.astype(Xh.dtype)
        else:
            return Xh

    def __call__(self, C: jnp.ndarray) -> jnp.ndarray:
        # Accept both (Nn,Nm) (linear path) and (Nk,Nn,Nm) (nonlinear path).
        if C.ndim == 2:
            # No nonlinear effect in single-k runs; return zeros.
            return jnp.zeros_like(C)

        assert C.ndim == 3, "NonlinearConvolution expects C[k,n,m] for nonlinear runs."
        Nk, Nn, Nm = self.Nk, self.Nn, self.Nm
        assert Nm >= 1, "Nonlinear term currently defined only for Laguerre m=0."

        a = C[:, :, 0]                  # (Nk, Nn) Hermite at Laguerre m=0
        k = self.ks                     # (Nk,)

        # E_k = i * a_{k,0}/k, guarded at k=0
        a0 = a[:, 0]
        E = jnp.where(k != 0.0, 1j * a0 / k, 0.0 + 0.0j)   # (Nk,)

        Eh = jnp.fft.fft(E, axis=0)     # (Nk,)
        # Prepare output N on (Nk,Nn), m=0 only
        N = jnp.zeros_like(a)

        def body(m, Ncur):
            # m >= 1 uses a[:, m-1]
            a_prev = a[:, m-1]                          # (Nk,)
            Ah = jnp.fft.fft(a_prev, axis=0)
            Prod = Eh * Ah
            Prod_f = self._dealise(Prod)
            conv = jnp.fft.ifft(Prod_f, axis=0)         # (Nk,), complex
            return Ncur.at[:, m].set(jnp.sqrt(2.0 * m) * conv)

        # Loop m=1..Nn-1; m=0 remains zero
        N = lax.fori_loop(1, Nn, body, N)

        # Place back into full C-shape, only m=0 affected
        dC = jnp.zeros_like(C)
        dC = dC.at[:, :, 0].add(N)
        return dC