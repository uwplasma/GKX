"""Canonical normalization contracts for benchmark families.

This module centralizes the case-level calibration knobs that were previously
spread across benchmark runners and tool scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


DiagnosticNorm = Literal["none", "gx", "rho_star"]


@dataclass(frozen=True)
class NormalizationContract:
    """Case-level normalization parameters.

    Attributes
    ----------
    case:
        Canonical case key (e.g. ``"cyclone"``).
    rho_star:
        Multiplier applied to ``kx``/``ky`` in drift and drive terms.
    omega_d_scale:
        Curvature / grad-B / mirror scaling.
    omega_star_scale:
        Diamagnetic drive scaling.
    diagnostic_norm_default:
        Default post-processing normalization for reported ``(gamma, omega)``.
    """

    case: str
    rho_star: float
    omega_d_scale: float
    omega_star_scale: float
    diagnostic_norm_default: DiagnosticNorm = "none"


CYCLONE_NORMALIZATION = NormalizationContract(
    case="cyclone",
    rho_star=1.0,
    omega_d_scale=1.0,
    omega_star_scale=1.0,
)

ETG_NORMALIZATION = NormalizationContract(
    case="etg",
    rho_star=1.0,
    omega_d_scale=0.4,
    omega_star_scale=0.8,
)

KINETIC_NORMALIZATION = NormalizationContract(
    case="kinetic",
    rho_star=1.0,
    omega_d_scale=1.0,
    omega_star_scale=1.0,
)

TEM_NORMALIZATION = NormalizationContract(
    case="tem",
    rho_star=1.0,
    omega_d_scale=1.0,
    omega_star_scale=1.0,
)

KBM_NORMALIZATION = NormalizationContract(
    case="kbm",
    rho_star=1.0,
    omega_d_scale=1.0,
    omega_star_scale=0.8,
)


_CONTRACTS: dict[str, NormalizationContract] = {
    "cyclone": CYCLONE_NORMALIZATION,
    "etg": ETG_NORMALIZATION,
    "kinetic": KINETIC_NORMALIZATION,
    "tem": TEM_NORMALIZATION,
    "kbm": KBM_NORMALIZATION,
}

_ALIASES: dict[str, str] = {
    "kinetic_itg": "kinetic",
    "kinetic-electron": "kinetic",
}


def get_normalization_contract(case: str) -> NormalizationContract:
    """Return the canonical normalization contract for ``case``."""

    key = case.strip().lower()
    key = _ALIASES.get(key, key)
    try:
        return _CONTRACTS[key]
    except KeyError as exc:
        valid = ", ".join(sorted(_CONTRACTS))
        raise ValueError(f"Unknown normalization case '{case}'. Valid keys: {valid}") from exc


def apply_diagnostic_normalization(
    gamma: float,
    omega: float,
    *,
    rho_star: float,
    diagnostic_norm: str,
) -> tuple[float, float]:
    """Apply reporting-space normalization to growth rates/frequencies."""

    mode = diagnostic_norm.strip().lower()
    if mode in {"none", ""}:
        return float(gamma), float(omega)
    if mode in {"gx", "rho_star"}:
        scale = float(rho_star)
        return float(gamma) * scale, float(omega) * scale
    raise ValueError(f"Unknown diagnostic_norm '{diagnostic_norm}'")
