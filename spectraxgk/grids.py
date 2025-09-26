from __future__ import annotations
from dataclasses import dataclass
import jax
import jax.numpy as jnp

@dataclass
class FourierZ:
    kpar: float

@dataclass
class DG1D:
    # Placeholder for future DG discretization in x/y/z
    nx: int
    L: float