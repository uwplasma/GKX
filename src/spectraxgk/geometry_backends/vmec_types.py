"""Lightweight containers shared by VMEC geometry modules."""

from __future__ import annotations

from typing import Any


class _Struct:
    """Mutable attribute bag, designed for mutable geometry assembly."""

    def __init__(self, **fields: Any) -> None:
        """Attach named fields while keeping downstream mutation explicit."""

        self.__dict__.update(fields)

    def __getattr__(self, name: str) -> Any:
        """Expose dynamically assembled geometry attributes to static checkers."""

        raise AttributeError(name)
