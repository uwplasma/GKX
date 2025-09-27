from __future__ import annotations
from typing import Protocol, Any
import jax.numpy as jnp
from dataclasses import dataclass
import numpy as np


class ComplexTerm(Protocol):
    """Any physics operator that maps complex HL coefficients to a complex RHS."""
    def __call__(self, C: jnp.ndarray) -> jnp.ndarray: ...


@dataclass
class Result:
    t: np.ndarray
    C: np.ndarray  # (nt, Nn, Nm) complex
    meta: dict[str, Any]
