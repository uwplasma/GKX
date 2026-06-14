"""Core contracts for SPECTRAX-GK refactor and extension points.

The ``spectraxgk.core`` package is intentionally small and dependency-light.
It holds typed contracts that describe public extension boundaries without
pulling solver kernels, optional geometry backends, or runtime I/O into import
paths used by tests and documentation.
"""

from .contracts import (
    DifferentiabilityContract,
    ExtensionPointContract,
    ModuleRefactorContract,
    ShapeContract,
    ValidationGateContract,
)

__all__ = [
    "DifferentiabilityContract",
    "ExtensionPointContract",
    "ModuleRefactorContract",
    "ShapeContract",
    "ValidationGateContract",
]
