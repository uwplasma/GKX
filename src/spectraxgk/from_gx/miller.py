"""Legacy compatibility imports for the internal Miller geometry backend."""

from spectraxgk.geometry_backends import miller as _backend

globals().update(
    {
        name: getattr(_backend, name)
        for name in dir(_backend)
        if not name.startswith("__")
    }
)
__all__ = [name for name in globals() if not name.startswith("__")]
