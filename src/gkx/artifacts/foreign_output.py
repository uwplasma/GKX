"""Dispatch to readers for other gyrokinetic codes' output bundles.

``gkx --plot`` accepts a bundle written by another code so that a cross-code
comparison is one command rather than a bespoke script. The readers themselves
live in their own modules, one per code; this indirection keeps their names out
of the plotting path, so the core figure code never learns which codes exist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

__all__ = ["foreign_output_plotter"]


def _readers() -> list[tuple[Callable[[Path], bool], Callable[..., Any]]]:
    """Return the (recognizer, plotter) pairs, imported on demand."""

    from gkx.artifacts import gx_output

    return [(gx_output.is_gx_output, gx_output.plot_gx_output)]


def foreign_output_plotter(path: str | Path) -> Callable[..., Any] | None:
    """Return a plotter for ``path`` when another code wrote it, else ``None``."""

    candidate = Path(path)
    for recognizes, plot in _readers():
        if recognizes(candidate):
            return plot
    return None
