"""Term-wise RHS assembly for the full gyrokinetic system.

This package provides a modular structure for assembling the nonlinear,
electromagnetic multispecies gyrokinetic RHS from individual terms. The
term modules expose a stable API for streaming, mirror, curvature/grad-B,
diamagnetic drive, collisions, hyper-collisions, end damping, and the
pseudo-spectral nonlinear bracket.
"""

from typing import TYPE_CHECKING

from gkx.terms.config import FieldState, TermConfig

if TYPE_CHECKING:  # pragma: no cover
    from gkx.terms.assembly import (
        assemble_rhs,
        assemble_rhs_cached,
        assemble_rhs_cached_jit,
        assemble_rhs_terms_cached,
    )

__all__ = [
    "FieldState",
    "TermConfig",
    "assemble_rhs",
    "assemble_rhs_cached",
    "assemble_rhs_cached_jit",
    "assemble_rhs_terms_cached",
]


def __getattr__(name: str):
    if name in {"assemble_rhs", "assemble_rhs_cached", "assemble_rhs_cached_jit", "assemble_rhs_terms_cached"}:
        from gkx.terms import assembly

        return getattr(assembly, name)
    raise AttributeError(name)
