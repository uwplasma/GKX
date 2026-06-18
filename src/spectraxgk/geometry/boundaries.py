"""Boundary-condition helpers for flux-tube geometry.

The zero-shear promotion is shared by analytic geometry construction and by
runtime grid-default policy. Keeping it isolated avoids cycles between analytic
geometry models and sampled/imported geometry contracts.
"""

from __future__ import annotations

ZERO_SHAT_THRESHOLD = 1.0e-5


def zero_shear_enabled(
    s_hat: float,
    *,
    zero_shat: bool = False,
    threshold: float = ZERO_SHAT_THRESHOLD,
) -> bool:
    """Return the effective zero-shear state."""

    return bool(zero_shat) or abs(float(s_hat)) < float(threshold)


def effective_boundary(
    boundary: str,
    *,
    s_hat: float,
    zero_shat: bool = False,
    threshold: float = ZERO_SHAT_THRESHOLD,
) -> str:
    """Return the effective boundary after zero-shear promotion."""

    if zero_shear_enabled(s_hat, zero_shat=zero_shat, threshold=threshold):
        return "periodic"
    return str(boundary)
