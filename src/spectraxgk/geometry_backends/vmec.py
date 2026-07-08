"""Public VMEC imported-geometry backend facade.

The implementation is split into focused modules for dependency discovery,
field-line assembly, flux-tube remapping, NetCDF writeout, and orchestration.
This facade keeps the public import surface small while implementation owners
stay focused enough to test and review module by module.
"""

from __future__ import annotations

from spectraxgk.geometry_backends.vmec_backend_discovery import (
    _booz_read_wout_square_layout_failure,
    _booz_xform_jax_search_paths,
    _import_booz_backend,
    _import_booz_xform_backend,
    _import_booz_xform_jax_backend,
    _import_module_with_search_paths,
    _new_booz_object,
    internal_vmec_backend_available,
)
from spectraxgk.geometry_backends.vmec_fieldlines import _vmec_fieldlines, _vmec_splines
from spectraxgk.geometry_backends.vmec_io import (
    _write_vmec_eik_netcdf_atomically,
    write_vmec_eik_netcdf,
)
from spectraxgk.geometry_backends.vmec_numerics import dermv, nperiod_set
from spectraxgk.geometry_backends.vmec_pipeline import generate_vmec_eik_internal
from spectraxgk.geometry_backends.vmec_remap import _apply_flux_tube_cut, _equal_arc_remap
from spectraxgk.geometry_backends.vmec_splines import _Struct

__all__ = [
    "_Struct",
    "_apply_flux_tube_cut",
    "_booz_read_wout_square_layout_failure",
    "_booz_xform_jax_search_paths",
    "_equal_arc_remap",
    "_import_booz_backend",
    "_import_booz_xform_backend",
    "_import_booz_xform_jax_backend",
    "_import_module_with_search_paths",
    "_new_booz_object",
    "_vmec_fieldlines",
    "_vmec_splines",
    "_write_vmec_eik_netcdf_atomically",
    "dermv",
    "generate_vmec_eik_internal",
    "internal_vmec_backend_available",
    "nperiod_set",
    "write_vmec_eik_netcdf",
]
