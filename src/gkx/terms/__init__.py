"""Lazy compatibility facade for term-wise gyrokinetic RHS assembly."""

from gkx.terms.config import FieldState, TermConfig

__all__ = [
    "FieldState",
    "TermConfig",
    "assemble_rhs_cached",
    "assemble_rhs_cached_jit",
    "assemble_rhs_terms_cached",
]


def __getattr__(name: str):
    if name in {
        "assemble_rhs_cached",
        "assemble_rhs_cached_jit",
        "assemble_rhs_terms_cached",
    }:
        from gkx.terms import assembly

        return getattr(assembly, name)
    raise AttributeError(name)
