#!/usr/bin/env python3
"""Generate the shipped finite-wavelength (gyrokinetic) Coulomb collision tables.

The finite-Larmor-radius Coulomb operator of Frei, Ball, Hoffmann, Jorge, Ricci
& Stenger (2021), arXiv:2104.11480, equations (3.47)--(3.50), is evaluated in
independent Dirichlet/Fourier quadrature on a Bessel-argument grid. Every node
must pass quadrature refinement before publication. The full J0 polarization
source is retained; this does not change the finite-basis field closure.
Only float64 tables are used by the runtime; --digits controls the optional
independent multiprecision drift-kinetic check, not quadrature precision.

The Bessel argument is ``B = k_perp v_th / Omega``. The runtime cache stores
``b = k_perp**2 T m / (q B_ref)**2``, so it interpolates at ``B = sqrt(2 b)``.

Usage::

    python tools/artifacts/build_finite_wavelength_coulomb_data.py
    python tools/artifacts/build_finite_wavelength_coulomb_data.py --digits 40 --check
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "tools" / "artifacts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools" / "artifacts"))

DATA_DIR = REPO_ROOT / "src" / "gkx" / "data"
STEM = "finite_wavelength_coulomb"

# Denser at small B, where the kernels vary fastest, and extending to k_perp
# rho well past the ion-scale drive.
BESSEL_ARGUMENTS: tuple[float, ...] = (
    0.0,
    0.125,
    0.25,
    0.375,
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
)
# (P, J) = (3, 1) gives the eight Hermite-Laguerre moments used by the
# drift-kinetic tables, so the two operators can be compared at equal
# resolution and both satisfy the runtime's Nl*Nm moment count.
MAXIMUM_HERMITE_ORDER = 3
MAXIMUM_LAGUERRE_ORDER = 1
BLOCK_NAMES = (
    "test_matrix",
    "field_matrix",
    "test_phi1",
    "field_phi1",
    "test_phi2",
    "field_phi2",
)


def build_tables(
    digits: int,
    worker_count: int,
    hermite: int = MAXIMUM_HERMITE_ORDER,
    laguerre: int = MAXIMUM_LAGUERRE_ORDER,
) -> dict[str, np.ndarray]:
    """Generate only reachable like-species diagonals, with all-node refinement."""

    from build_linear_validation_artifacts import (
        like_species_field_particle_fourier,
        like_species_test_particle_gram_matrices,
        like_species_test_particle_polarization,
    )

    if digits < 1 or worker_count < 1:
        raise ValueError("digits and workers must be positive")
    coarse, refined = [], []
    for target, (nr, nx, fr, fx, fa) in zip(
        (coarse, refined), ((96, 48, 64, 48, 48), (192, 64, 96, 64, 64)), strict=True
    ):
        c0, d = like_species_test_particle_gram_matrices(
            hermite, laguerre, radial_nodes=nr, pitch_nodes=nx
        )
        for b in BESSEL_ARGUMENTS:
            field, field_phi = like_species_field_particle_fourier(
                hermite,
                laguerre,
                b,
                radial_nodes=fr,
                pitch_nodes=fx,
                azimuthal_nodes=fa,
            )
            test_phi = like_species_test_particle_polarization(
                hermite, laguerre, b, radial_nodes=nr, pitch_nodes=nx
            )
            zero = np.zeros_like(test_phi)
            target.append((c0 - b * b * d, field, zero, zero, test_phi, field_phi))
    signs = np.tile((-1.0) ** np.arange(laguerre + 1), hermite + 1)
    result = {}
    for k, name in enumerate(BLOCK_NAMES):
        values = np.asarray([row[k] for row in coarse])
        reference = np.asarray([row[k] for row in refined])
        if not (np.isfinite(values).all() and np.isfinite(reference).all()):
            raise ValueError(f"{name}: non-finite quadrature coefficients")
        if not np.allclose(values, reference, rtol=1e-10, atol=1e-12):
            raise ValueError(f"{name}: quadrature refinement failed")
        convention = signs[:, None] * signs[None, :] if k < 2 else signs
        result[name] = np.ascontiguousarray(reference * convention)
    return result


def write_artifacts(
    diagonals: dict[str, np.ndarray],
    digits: int,
    hermite: int = MAXIMUM_HERMITE_ORDER,
    laguerre: int = MAXIMUM_LAGUERRE_ORDER,
    stem: str = STEM,
    output_dir: Path | None = None,
    drift_kinetic_checked: bool = False,
) -> tuple[Path, Path]:
    """Write the checksummed ``.npz``/``.json`` pair into the package data."""

    if any(not np.isfinite(value).all() for value in diagonals.values()):
        raise ValueError("Coulomb candidate contains non-finite coefficients")
    buffer = io.BytesIO()
    np.savez(
        buffer,
        bessel_argument_grid=np.asarray(BESSEL_ARGUMENTS, dtype=float),
        **diagonals,
    )
    payload = buffer.getvalue()
    metadata = {
        "kind": "gkx_finite_wavelength_coulomb_coefficients",
        "source": "Frei, Ball, Hoffmann, Jorge, Ricci & Stenger (2021), arXiv:2104.11480",
        "equations": "3.47-3.50",
        "claim_scope": "research_like_species_finite_larmor_coulomb",
        "bessel_argument": "B = k_perp v_th / Omega; runtime interpolates at sqrt(2 b)",
        "bessel_argument_grid": list(BESSEL_ARGUMENTS),
        "maximum_hermite_order": hermite,
        "maximum_laguerre_order": laguerre,
        "moment_order": "hermite_major_index=p*(J+1)+j",
        "laguerre_convention": "gkx_signed_runtime",
        "mass_ratio": 1.0,
        "temperature_ratio": 1.0,
        "precision_decimal_digits": np.finfo(np.float64).precision,
        "reference_decimal_digits": digits,
        "drift_kinetic_check_passed": drift_kinetic_checked,
        "generation_method": "like_species_dirichlet_fourier_quadrature",
        "quadrature_nodes": {"test": [192, 64], "field": [96, 64, 64]},
        "comparison_nodes": {"test": [96, 48], "field": [64, 48, 48]},
        "quadrature_rtol": 1e-10,
        "quadrature_atol": 1e-12,
        "polarization_source": "full_J0_not_truncated_H",
        "dtype": "float64",
        "shapes": {name: list(value.shape) for name, value in diagonals.items()},
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    destination = DATA_DIR if output_dir is None else output_dir
    destination.mkdir(parents=True, exist_ok=True)
    data_path = destination / f"{stem}.npz"
    metadata_path = destination / f"{stem}.json"
    data_path.write_bytes(payload)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return data_path, metadata_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--digits",
        type=int,
        default=60,
        help="digits for the optional DK reference check",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="legacy option; quadrature is streamed serially",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="candidate output directory; defaults to package data",
    )
    parser.add_argument(
        "--hermite",
        type=int,
        default=MAXIMUM_HERMITE_ORDER,
        help="maximum Hermite order P; the moment count is (P+1)*(J+1)",
    )
    parser.add_argument(
        "--laguerre",
        type=int,
        default=MAXIMUM_LAGUERRE_ORDER,
        help="maximum Laguerre order J",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default=None,
        help="output basename; defaults to the moment count for non-default orders",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the b->0 limit against the drift-kinetic Coulomb matrices",
    )
    args = parser.parse_args(argv)

    moments = (args.hermite + 1) * (args.laguerre + 1)
    stem = args.stem
    if stem is None:
        stem = (
            STEM
            if (args.hermite, args.laguerre)
            == (MAXIMUM_HERMITE_ORDER, MAXIMUM_LAGUERRE_ORDER)
            else f"{STEM}_{moments}"
        )

    start = time.time()
    diagonals = build_tables(args.digits, args.workers, args.hermite, args.laguerre)
    if args.check:
        from build_linear_validation_artifacts import (
            coulomb_drift_kinetic_moment_matrices,
        )

        sign = np.asarray(
            [
                (-1.0) ** lag
                for _h in range(args.hermite + 1)
                for lag in range(args.laguerre + 1)
            ]
        )
        convention = sign[:, None] * sign[None, :]
        dk_test, dk_field = (
            np.asarray(matrix, dtype=float)
            for matrix in coulomb_drift_kinetic_moment_matrices(
                args.hermite,
                args.laguerre,
                1.0,
                1.0,
                digits=args.digits,
            )[:2]
        )
        test_error = np.abs(diagonals["test_matrix"][0] * convention - dk_test).max()
        field_error = np.abs(diagonals["field_matrix"][0] * convention - dk_field).max()
        print(f"b->0 reduction: test {test_error:.3e}, field {field_error:.3e}")
        if not (test_error <= 1.0e-10 and field_error <= 1.0e-10):
            print("FAIL: b->0 limit does not match the drift-kinetic operator")
            return 1
        print("b->0 reduction OK")
    data_path, metadata_path = write_artifacts(
        diagonals,
        args.digits,
        args.hermite,
        args.laguerre,
        stem,
        args.output_dir,
        drift_kinetic_checked=args.check,
    )
    elapsed = time.time() - start

    print(
        f"generated {moments} moments in {elapsed:.1f}s with refined float64 quadrature"
    )
    for name, value in diagonals.items():
        print(f"  {name}: {value.shape}")
    print(f"wrote {data_path} ({data_path.stat().st_size} bytes)")
    print(f"wrote {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
