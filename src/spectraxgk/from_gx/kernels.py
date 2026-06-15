"""Legacy compatibility imports for geometry-backend kernel helpers."""

from spectraxgk.geometry_backends import kernels as _backend

globals().update(
    {
        name: getattr(_backend, name)
        for name in dir(_backend)
        if not name.startswith("__")
    }
)
__all__ = [name for name in globals() if not name.startswith("__")]
