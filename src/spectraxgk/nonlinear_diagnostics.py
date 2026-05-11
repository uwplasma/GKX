"""Diagnostic packing and sampling helpers for nonlinear integrations."""

from __future__ import annotations

import numpy as np

from spectraxgk.diagnostics import ResolvedDiagnostics

__all__ = [
    "_pack_resolved_diagnostics",
    "_sample_axis0",
    "_sample_indices_with_final",
]


def _pack_resolved_diagnostics(resolved_t: tuple[np.ndarray, ...]) -> ResolvedDiagnostics:
    return ResolvedDiagnostics(
        Phi2_kxt=resolved_t[0],
        Phi2_kyt=resolved_t[1],
        Phi2_kxkyt=resolved_t[2],
        Phi2_zt=resolved_t[3],
        Phi2_zonal_t=resolved_t[4],
        Phi2_zonal_kxt=resolved_t[5],
        Phi2_zonal_zt=resolved_t[6],
        Phi_zonal_mode_kxt=resolved_t[7],
        Phi_zonal_line_kxt=resolved_t[8],
        Wg_kxst=resolved_t[9],
        Wg_kyst=resolved_t[10],
        Wg_kxkyst=resolved_t[11],
        Wg_zst=resolved_t[12],
        Wg_lmst=resolved_t[13],
        Wphi_kxst=resolved_t[14],
        Wphi_kyst=resolved_t[15],
        Wphi_kxkyst=resolved_t[16],
        Wphi_zst=resolved_t[17],
        Wapar_kxst=resolved_t[18],
        Wapar_kyst=resolved_t[19],
        Wapar_kxkyst=resolved_t[20],
        Wapar_zst=resolved_t[21],
        HeatFlux_kxst=resolved_t[22],
        HeatFlux_kyst=resolved_t[23],
        HeatFlux_kxkyst=resolved_t[24],
        HeatFlux_zst=resolved_t[25],
        HeatFluxES_kxst=resolved_t[26],
        HeatFluxES_kyst=resolved_t[27],
        HeatFluxES_kxkyst=resolved_t[28],
        HeatFluxES_zst=resolved_t[29],
        HeatFluxApar_kxst=resolved_t[30],
        HeatFluxApar_kyst=resolved_t[31],
        HeatFluxApar_kxkyst=resolved_t[32],
        HeatFluxApar_zst=resolved_t[33],
        HeatFluxBpar_kxst=resolved_t[34],
        HeatFluxBpar_kyst=resolved_t[35],
        HeatFluxBpar_kxkyst=resolved_t[36],
        HeatFluxBpar_zst=resolved_t[37],
        ParticleFlux_kxst=resolved_t[38],
        ParticleFlux_kyst=resolved_t[39],
        ParticleFlux_kxkyst=resolved_t[40],
        ParticleFlux_zst=resolved_t[41],
        ParticleFluxES_kxst=resolved_t[42],
        ParticleFluxES_kyst=resolved_t[43],
        ParticleFluxES_kxkyst=resolved_t[44],
        ParticleFluxES_zst=resolved_t[45],
        ParticleFluxApar_kxst=resolved_t[46],
        ParticleFluxApar_kyst=resolved_t[47],
        ParticleFluxApar_kxkyst=resolved_t[48],
        ParticleFluxApar_zst=resolved_t[49],
        ParticleFluxBpar_kxst=resolved_t[50],
        ParticleFluxBpar_kyst=resolved_t[51],
        ParticleFluxBpar_kxkyst=resolved_t[52],
        ParticleFluxBpar_zst=resolved_t[53],
        TurbulentHeating_kxst=resolved_t[54],
        TurbulentHeating_kyst=resolved_t[55],
        TurbulentHeating_kxkyst=resolved_t[56],
        TurbulentHeating_zst=resolved_t[57],
    )


def _sample_indices_with_final(length: int, stride: int) -> slice | np.ndarray:
    """Return strided sample indices while always retaining the final step."""

    n = int(length)
    stride_i = int(max(stride, 1))
    if stride_i <= 1 or n <= 1:
        return slice(None)
    idx = np.arange(0, n, stride_i, dtype=int)
    if idx.size == 0 or int(idx[-1]) != n - 1:
        idx = np.concatenate([idx, np.asarray([n - 1], dtype=int)])
    return idx


def _sample_axis0(arr, indices: slice | np.ndarray):
    return arr[indices, ...]
