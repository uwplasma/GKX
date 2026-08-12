"""The repository's only ``tomllib``/``tomli`` fallback.

``tomllib`` joined the standard library in Python 3.11. GKX supports 3.10, where
the same parser ships as the ``tomli`` distribution -- declared in
``pyproject.toml`` as a runtime dependency under the ``python_version < '3.11'``
marker, because ``src`` reads TOML at runtime and not only under ``[dev]``.

Import this module instead of ``tomllib``::

    from gkx.utils import tomlcompat as tomllib

The aliased form keeps ``tomllib.load``/``tomllib.loads`` call sites unchanged.
Resolving the fallback here once, rather than repeating a ``try``/``except``
per file, is what lets ``test_no_unguarded_tomllib_imports`` fail a bare
``import tomllib`` anywhere else in the tree -- the defect that made the suite
uncollectable on the declared floor while CI only ever ran 3.11.
"""

from __future__ import annotations

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - taken only on Python 3.10
    import tomli as tomllib  # type: ignore[import-not-found,no-redef]

TOMLDecodeError = tomllib.TOMLDecodeError
load = tomllib.load
loads = tomllib.loads

__all__ = ["TOMLDecodeError", "load", "loads"]
