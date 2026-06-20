"""Diffrax-based time-integrator facade for gyrokinetic systems."""

from __future__ import annotations

from spectraxgk.solvers.time.diffrax_core import (
    _adjoint,
    _assemble_rhs,
    _base_complex_dtype,
    _density_from_G_cached,
    _is_imex_solver,
    _is_implicit_solver,
    _pack_complex_state,
    _progress_meter,
    _require_diffrax,
    _save_with_phi,
    _solver_from_name,
    _stepsize_controller,
    _unpack_complex_state,
)
from spectraxgk.solvers.time.diffrax_linear import integrate_linear_diffrax
from spectraxgk.solvers.time.diffrax_nonlinear import integrate_nonlinear_diffrax
from spectraxgk.solvers.time.diffrax_streaming import integrate_linear_diffrax_streaming

__all__ = [
    "_adjoint",
    "_assemble_rhs",
    "_base_complex_dtype",
    "_density_from_G_cached",
    "_is_imex_solver",
    "_is_implicit_solver",
    "_pack_complex_state",
    "_progress_meter",
    "_require_diffrax",
    "_save_with_phi",
    "_solver_from_name",
    "_stepsize_controller",
    "_unpack_complex_state",
    "integrate_linear_diffrax",
    "integrate_linear_diffrax_streaming",
    "integrate_nonlinear_diffrax",
]
