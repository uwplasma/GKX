"""Internal imported-geometry backends used by SPECTRAX-GK.

The modules in this package generate and postprocess Miller/VMEC flux-tube
geometry into the solver-facing imported-geometry contract. They are internal
implementation details; public callers should prefer ``spectraxgk.geometry``,
``spectraxgk.geometry.miller_eik``, and ``spectraxgk.geometry.vmec_eik``.
"""

from spectraxgk.geometry_backends.miller import internal_miller_backend_available
from spectraxgk.geometry_backends.vmec import internal_vmec_backend_available

__all__ = ["internal_miller_backend_available", "internal_vmec_backend_available"]
