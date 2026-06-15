"""Standalone VMEC flux-tube geometry generation implementation."""

from __future__ import annotations

from pathlib import Path


def generate_vmec_eik(
    cfg_data: dict,
    output_path: str | Path,
):
    """Generate VMEC geometry coefficients and save to NetCDF."""

    try:
        from netCDF4 import Dataset
    except ImportError:
        raise ImportError("netCDF4 is required for VMEC geometry generation")

    # Try booz_xform_jax first, then fallback to booz_xform.
    try:
        __import__("booz_xform_jax")
    except ImportError:
        try:
            __import__("booz_xform")
        except ImportError:
            raise ImportError(
                "Either booz_xform or booz_xform_jax is required for VMEC geometry generation"
            )

    # Use a Boozer transform backend to build the compact VMEC geometry file
    # needed by standalone geometry examples.

    # Let's save a stub file for now to verify the integration.
    ds = Dataset(output_path, "w")
    try:
        ds.createDimension("z", 16)  # dummy size
        # Add necessary variables...
    finally:
        ds.close()
