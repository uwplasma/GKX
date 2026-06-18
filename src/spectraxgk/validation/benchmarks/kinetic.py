"""Kinetic-electron benchmark facade."""

from __future__ import annotations

from spectraxgk.validation.benchmarks.kinetic_linear import run_kinetic_linear
from spectraxgk.validation.benchmarks.kinetic_scan import run_kinetic_scan

__all__ = ["run_kinetic_linear", "run_kinetic_scan"]
