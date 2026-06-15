"""Legacy compatibility package for renamed internal geometry backends.

New SPECTRAX-GK internals should import from ``spectraxgk.geometry_backends``.
This package remains only to avoid breaking older user scripts and archived
comparison tooling.
"""

from spectraxgk.geometry_backends.miller import internal_miller_backend_available
from spectraxgk.geometry_backends.vmec import internal_vmec_backend_available

__all__ = ["internal_vmec_backend_available", "internal_miller_backend_available"]
