"""Term-wise RHS assembly for the full gyrokinetic system.

This package provides a modular structure for assembling the nonlinear,
electromagnetic multispecies gyrokinetic RHS from individual terms. The
term modules expose a stable API for streaming, mirror, curvature/grad-B,
diamagnetic drive, collisions, hyper-collisions, and end damping, with
placeholders for nonlinear terms.
"""

from typing import TYPE_CHECKING

from spectraxgk.terms.config import FieldState, TermConfig

if TYPE_CHECKING:  # pragma: no cover
    from spectraxgk.terms.assembly import (
        assemble_rhs,
        assemble_rhs_cached,
        assemble_rhs_cached_jit,
        assemble_rhs_terms_cached,
    )
    from spectraxgk.terms.integrators import integrate_nonlinear

__all__ = [
    "FieldState",
    "TermConfig",
    "assemble_rhs",
    "assemble_rhs_cached",
    "assemble_rhs_cached_jit",
    "assemble_rhs_terms_cached",
    "integrate_nonlinear",
]


def __getattr__(name: str):
    if name in {"assemble_rhs", "assemble_rhs_cached", "assemble_rhs_cached_jit", "assemble_rhs_terms_cached"}:
        from spectraxgk.terms import assembly

        return getattr(assembly, name)
    if name == "integrate_nonlinear":
        from spectraxgk.terms import integrators

        return integrators.integrate_nonlinear
    raise AttributeError(name)
