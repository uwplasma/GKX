"""Internal geometry backends progressively ported from GX.

These modules host JAX-friendly implementations intended to remove the hard
runtime dependency on an external GX repository.
"""

from spectraxgk.from_gx.vmec import internal_vmec_backend_available
from spectraxgk.from_gx.miller import internal_miller_backend_available

__all__ = ["internal_vmec_backend_available", "internal_miller_backend_available"]
