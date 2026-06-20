"""SPECTRAX-GK: a JAX gyrokinetic solver with Hermite-Laguerre velocity space."""

from spectraxgk._version import __version__
from spectraxgk import api as _api

for _name in _api.__all__:
    globals()[_name] = getattr(_api, _name)

__all__ = ["__version__", *_api.__all__]

del _api, _name
