"""Integrating-factor (Lawson) explicit steppers: the magnetic drift is advanced exactly.

The drift is diagonal in ``(species, ky, kx, z)`` and acts on the moments as
``-i (a C (x) I + b I (x) B)``: ``a`` and ``b`` carry the curvature and grad-B rates, ``C`` and
``B`` the truncated Hermite and Laguerre matrices.  Both are real symmetric, so
``exp(-i h (a C + b B))`` is a diagonal phase ``exp(-i h (a c_m + b beta_l))`` between two
orthogonal moment transforms.  A Lawson step is the explicit Runge-Kutta step of
``N(G) = F(G) - L_d G`` in the frame turning with the drift,
``y' = exp(-t L_d) N(exp(t L_d) y)``, so the drift leaves the explicit stability limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from solvax.ode import RK4_2, erk_step

from gkx.terms.assembly import _species_arrays, _term_weights
from gkx.terms.linear_terms import curvature_term, gradb_term

# Heun's third-order method, the tableau of the explicit ``rk3`` step (no embedded estimate).
_HEUN3 = RK4_2._replace(
    a=((1 / 3,), (0.0, 2 / 3)), b=(0.25, 0.0, 0.75), e=(0.0,) * 3, c=(0.0, 1 / 3, 2 / 3)
)
_PAIRS = {"rk3": _HEUN3, "rk4": RK4_2, "rk4-pi": RK4_2}
# The default GPU matmul (TF32) loses 1e-3 per transform; up to this many moments one dense
# (Nl Nm)^2 transform beats the two small ones.
_HI, _KRON_MOMENTS = jax.lax.Precision.HIGHEST, 128


@dataclass(frozen=True)
class DriftFactor:
    """The drift exponential: moment transforms and the per-mode drift frequencies."""

    vc: jnp.ndarray  # (Nm, Nm) eigenvectors of the truncated Hermite curvature matrix
    vb: jnp.ndarray  # (Nl, Nl) eigenvectors of the truncated Laguerre grad-B matrix
    wc: jnp.ndarray  # (S, 1, Nm, ky, kx, z) a * eigenvalue of C
    wb: jnp.ndarray  # (S, Nl, 1, ky, kx, z) b * eigenvalue of B

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """Moments to the eigenbasis of the drift."""
        return self._transform(x, self.vc.T, self.vb.T)

    def inverse(self, x: jnp.ndarray) -> jnp.ndarray:
        return self._transform(x, self.vc, self.vb)

    def _transform(
        self, x: jnp.ndarray, vc: jnp.ndarray, vb: jnp.ndarray
    ) -> jnp.ndarray:
        if x.shape[1] * x.shape[2] <= _KRON_MOMENTS:
            flat = x.reshape(x.shape[0], x.shape[1] * x.shape[2], -1)
            kron = jnp.kron(vb, vc)
            return jnp.einsum("ab,sbn->san", kron, flat, precision=_HI).reshape(x.shape)
        x = jnp.einsum("mj,slmyxz->sljyxz", vc.T, x, precision=_HI)
        return jnp.einsum("lk,sljyxz->skjyxz", vb.T, x, precision=_HI)

    def rotate(self, x: jnp.ndarray, t: Any) -> jnp.ndarray:
        """``exp(t L_d) x`` in the eigenbasis: the product of the Hermite and Laguerre phases."""
        return x * jnp.exp(-1j * t * self.wc) * jnp.exp(-1j * t * self.wb)


def _eigh(term: Callable[..., Any], coord: Any, n: int, axis: int):
    """Eigenpairs of a moment-axis drift matrix, read off the production routine itself."""
    eye = jnp.eye(n).reshape((1, n, n, 1, 1) if axis == 1 else (n, 1, n, 1, 1))
    out = np.asarray(term(eye, coord), dtype=float)
    out = out[0, :, :, 0, 0] if axis == 1 else out[:, 0, :, 0, 0]
    return np.linalg.eigh(0.5 * (out + out.T))


def build_drift_factor(
    cache: Any, params: Any, terms: Any, ns: int, real_dtype: Any
) -> DriftFactor:
    """Diagonalize the truncated drift operator of ``cache`` for ``terms``."""
    if getattr(cache, "groups", None) is not None:
        raise NotImplementedError(
            "Lawson steppers need one moment layout for all species"
        )
    lam_c, vc = _eigh(curvature_term, cache.m, int(cache.m.shape[1]), 1)
    lam_b, vb = _eigh(gradb_term, cache.l, int(cache.l.shape[0]), 0)
    weights = _term_weights(params, terms, real_dtype)
    real = lambda x: jnp.asarray(x, real_dtype)  # noqa: E731
    tz_od = real(_species_arrays(params, ns, real_dtype).tz) * real(
        params.omega_d_scale
    )
    scale = tz_od[:, None, None, None, None, None]
    return DriftFactor(
        real(vc),
        real(vb),
        scale * weights.curvature * real(cache.cv_d) * real(lam_c)[:, None, None, None],
        scale
        * weights.gradb
        * real(cache.gb_d)
        * real(lam_b)[:, None, None, None, None],
    )


def attach_drift_factor(
    rhs_fn: Callable[..., Any], prepared: Any, params: Any, terms: Any
):
    """Return ``rhs_fn`` carrying the drift exponential the ``lawson-*`` steppers read."""
    ns = 1 if jnp.ndim(prepared.G0) == 5 else prepared.G0.shape[0]

    def rhs_with_drift(state: jnp.ndarray) -> Any:
        return rhs_fn(state)

    rhs_with_drift.drift_factor = build_drift_factor(  # type: ignore[attr-defined]
        prepared.cache, params, terms, ns, prepared.real_dtype
    )
    return rhs_with_drift


def lawson_step(
    G: jnp.ndarray,
    dG: jnp.ndarray,
    h: jnp.ndarray,
    method: str,
    rhs_fn: Callable[[jnp.ndarray], tuple[jnp.ndarray, object]],
    project_state: Callable[[jnp.ndarray], jnp.ndarray],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """One Lawson step from ``G`` with first-stage slope ``dG``: ``(G_new, err)``.

    ``rhs_fn`` carries the drift factor (:func:`attach_drift_factor`).  ``err`` is the
    embedded estimate of the tableau (zero for ``rk3``) in the eigenbasis, an orthogonal
    image of the moments, so its norm is the norm of the moment error.
    """
    factor: DriftFactor = rhs_fn.drift_factor  # type: ignore[attr-defined]
    up = (lambda x: x[None]) if G.ndim == 5 else (lambda x: x)
    down = (lambda x: x[0]) if G.ndim == 5 else (lambda x: x)
    drift = 1j * (factor.wc + factor.wb)

    def slope(t: Any, y: jnp.ndarray, dy: jnp.ndarray) -> jnp.ndarray:
        """``exp(-t L_d) N`` at the stage ``exp(t L_d) y``, with ``N = F - L_d Y``."""
        return factor.rotate(factor.forward(up(dy)), -t) + drift * y

    def stage(t: Any, y: jnp.ndarray) -> jnp.ndarray:
        state = project_state(down(factor.inverse(factor.rotate(y, t))))
        return slope(t, y, rhs_fn(state)[0])

    w = factor.forward(up(G))
    pair = _PAIRS[method.removeprefix("lawson-")]
    y, err, _ = erk_step(stage, 0.0, w, h, pair, k1=slope(0.0, w, dG))
    return down(factor.inverse(factor.rotate(y, h))), err
