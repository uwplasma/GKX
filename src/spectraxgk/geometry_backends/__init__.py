"""Internal VMEC imported-geometry backend used by SPECTRAX-GK.

Miller imported-geometry generation now lives in ``spectraxgk.geometry`` beside
the public Miller EIK helpers. This package remains as the VMEC implementation
namespace until the VMEC backend is folded into ``spectraxgk.geometry``.
"""

from spectraxgk.geometry_backends.vmec import internal_vmec_backend_available

__all__ = ["internal_vmec_backend_available"]
