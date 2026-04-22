"""Adaptive runtime chunk execution helpers.

These helpers own the repeated GX-style adaptive chunk loop used by the runtime
drivers. Keeping the loop outside ``runtime.py`` makes the execution layer
smaller without changing the accepted diagnostics truncation/stride behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np

from spectraxgk.diagnostics import SimulationDiagnostics
from spectraxgk.runtime_diagnostics import concat_gx_diagnostics, stride_gx_diagnostics, truncate_gx_diagnostics
from spectraxgk.terms.config import FieldState


@dataclass(frozen=True)
class AdaptiveChunkResult:
    """Concatenated result from one adaptive GX-style chunk loop."""

    diagnostics: SimulationDiagnostics
    state: Any
    fields: FieldState


def run_adaptive_gx_chunk_loop(
    *,
    integrate_chunk: Callable[[bool], tuple[Any, SimulationDiagnostics, Any, FieldState | None]],
    t_max: float,
    chunk_steps: int,
    label: str,
    show_progress: bool = False,
    status_callback: Callable[[str], None] | None = None,
    diagnostics_stride: int = 1,
    max_chunks: int = 100000,
) -> AdaptiveChunkResult:
    """Run repeated diagnostic chunks until ``t_max`` is reached.

    ``integrate_chunk`` must return ``(t_chunk, diag_chunk, state, fields)`` in
    the same contract used by the runtime integrators.
    """

    def _status(message: str) -> None:
        if status_callback is not None:
            status_callback(message)

    state_chunk = None
    t_elapsed = 0.0
    diag_chunks: list[SimulationDiagnostics] = []
    fields_final: FieldState | None = None
    _status(f"starting adaptive {label} integration in chunks of {chunk_steps} steps up to t_max={float(t_max):.6g}")

    for chunk in range(max_chunks):
        _t_chunk, diag_chunk, state_chunk, fields_final = integrate_chunk(show_progress)
        diag_chunk = replace(diag_chunk, t=np.asarray(diag_chunk.t) + t_elapsed)
        diag_chunks.append(diag_chunk)
        t_next = float(np.asarray(diag_chunk.t)[-1])
        if t_next <= t_elapsed + 1.0e-12:
            raise RuntimeError(f"adaptive {label} runtime made no time-step progress")
        t_elapsed = t_next
        _status(f"completed {label} chunk {chunk + 1}: t={t_elapsed:.6g}/{float(t_max):.6g}")
        if t_elapsed >= float(t_max):
            break
    else:
        raise RuntimeError(f"adaptive {label} runtime exceeded chunk limit before reaching t_max")

    diag = concat_gx_diagnostics(diag_chunks)
    diag = truncate_gx_diagnostics(diag, t_max=float(t_max))
    if int(max(diagnostics_stride, 1)) > 1:
        diag = stride_gx_diagnostics(diag, stride=int(max(diagnostics_stride, 1)))
    if fields_final is None:
        raise RuntimeError(f"adaptive {label} runtime did not produce final fields")
    return AdaptiveChunkResult(
        diagnostics=diag,
        state=state_chunk,
        fields=fields_final,
    )
