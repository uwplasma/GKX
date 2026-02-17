"""Term-wise RHS assembly for the full gyrokinetic system.

This package provides a modular structure for assembling the nonlinear,
electromagnetic multispecies gyrokinetic RHS from individual terms. The
term modules expose a stable API for streaming, mirror, curvature/grad-B,
diamagnetic drive, collisions, hyper-collisions, and end damping, with
placeholders for nonlinear terms.
"""

from spectraxgk.terms.config import FieldState, TermConfig
from spectraxgk.terms.assembly import assemble_rhs, assemble_rhs_cached, assemble_rhs_cached_jit
from spectraxgk.terms.integrators import integrate_nonlinear

__all__ = [
    "FieldState",
    "TermConfig",
    "assemble_rhs",
    "assemble_rhs_cached",
    "assemble_rhs_cached_jit",
    "integrate_nonlinear",
]
