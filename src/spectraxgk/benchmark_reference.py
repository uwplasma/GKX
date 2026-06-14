"""Reference data containers and loaders for benchmark lanes."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources

import numpy as np

from spectraxgk.analysis import ModeSelection


__all__ = [
    "CycloneComparison",
    "CycloneReference",
    "CycloneRunResult",
    "CycloneScanResult",
    "LinearRunResult",
    "LinearScanResult",
    "_load_reference_with_header",
    "compare_cyclone_to_reference",
    "load_cyclone_reference",
    "load_cyclone_reference_kinetic",
    "load_etg_reference",
    "load_kbm_reference",
    "load_tem_reference",
]


@dataclass(frozen=True)
class CycloneReference:
    ky: np.ndarray
    omega: np.ndarray
    gamma: np.ndarray


@dataclass(frozen=True)
class CycloneRunResult:
    t: np.ndarray
    phi_t: np.ndarray
    gamma: float
    omega: float
    ky: float
    selection: ModeSelection


@dataclass(frozen=True)
class CycloneScanResult:
    ky: np.ndarray
    gamma: np.ndarray
    omega: np.ndarray


@dataclass(frozen=True)
class CycloneComparison:
    ky: float
    gamma: float
    omega: float
    gamma_ref: float
    omega_ref: float
    rel_gamma: float
    rel_omega: float


@dataclass(frozen=True)
class LinearRunResult:
    t: np.ndarray
    phi_t: np.ndarray
    gamma: float
    omega: float
    ky: float
    selection: ModeSelection
    gamma_t: np.ndarray | None = None
    omega_t: np.ndarray | None = None


@dataclass(frozen=True)
class LinearScanResult:
    ky: np.ndarray
    gamma: np.ndarray
    omega: np.ndarray


def _load_csv_reference(filename: str) -> CycloneReference:
    data_path = resources.files("spectraxgk").joinpath("data", filename)
    arr = np.loadtxt(str(data_path), delimiter=",", skiprows=1)
    ky = arr[:, 0]
    omega = arr[:, 1]
    gamma = arr[:, 2]
    return CycloneReference(ky=ky, omega=omega, gamma=gamma)


def load_cyclone_reference() -> CycloneReference:
    """Load Cyclone base case reference data (adiabatic electrons)."""

    return _load_csv_reference("cyclone_reference_adiabatic.csv")


def _load_reference_with_header(filename: str) -> CycloneReference:
    """Load reference CSVs with columns ky,gamma,omega."""

    data_path = resources.files("spectraxgk").joinpath("data", filename)
    arr = np.genfromtxt(str(data_path), delimiter=",", names=True, dtype=float)
    ky = np.atleast_1d(np.asarray(arr["ky"], dtype=float))
    gamma = np.atleast_1d(np.asarray(arr["gamma"], dtype=float))
    omega = np.atleast_1d(np.asarray(arr["omega"], dtype=float))
    return CycloneReference(ky=ky, omega=omega, gamma=gamma)


def load_cyclone_reference_kinetic() -> CycloneReference:
    """Load Cyclone base case reference data (kinetic electrons)."""

    return _load_csv_reference("cyclone_reference_kinetic.csv")


def load_kbm_reference() -> CycloneReference:
    """Load KBM reference data (finite beta, kinetic electrons)."""

    return _load_csv_reference("kbm_reference.csv")


def load_etg_reference() -> CycloneReference:
    """Load ETG reference data for the tracked two-species ETG lane."""

    return _load_csv_reference("etg_reference.csv")


def load_tem_reference() -> CycloneReference:
    """Load the provisional TEM reference digitized from the literature.

    This lane remains an extended stress case while the literature case
    definition is being reconstructed.
    """

    return _load_csv_reference("tem_reference.csv")


def compare_cyclone_to_reference(
    result: CycloneRunResult, reference: CycloneReference
) -> CycloneComparison:
    """Compare a Cyclone run result against the reference data set."""

    idx = int(np.argmin(np.abs(reference.ky - result.ky)))
    gamma_ref = float(reference.gamma[idx])
    omega_ref = float(reference.omega[idx])
    rel_gamma = (result.gamma - gamma_ref) / gamma_ref if gamma_ref != 0.0 else np.nan
    rel_omega = (result.omega - omega_ref) / omega_ref if omega_ref != 0.0 else np.nan
    return CycloneComparison(
        ky=float(reference.ky[idx]),
        gamma=result.gamma,
        omega=result.omega,
        gamma_ref=gamma_ref,
        omega_ref=omega_ref,
        rel_gamma=rel_gamma,
        rel_omega=rel_omega,
    )
