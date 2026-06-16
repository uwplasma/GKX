"""Operator kernels and compatibility exports.

The public ``spectraxgk.operators`` surface remains intentionally small while
implementation modules live in domain subpackages.
"""

from __future__ import annotations

from spectraxgk.operators.linear import hermite_streaming

__all__ = ["hermite_streaming"]
