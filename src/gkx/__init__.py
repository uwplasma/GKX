"""GKX: a JAX gyrokinetic solver with Hermite-Laguerre velocity space."""

from __future__ import annotations

import os
from importlib import import_module
from typing import Any

from gkx._version import __version__


def _cpu_thread_defaults() -> None:
    """Keep XLA:CPU off its multithreaded Eigen pool on fewer than four CPUs.

    With 2 or 3 schedulable CPUs (``taskset``, small containers and CI
    runners), nonlinear runs at 32x32x16 deadlocked inside XLA:CPU with every
    thread parked (jax 0.11.2, 5/5 attempts); 1, 4, 5, 6, 8 and 16 CPUs did
    not, and the single-threaded pool ran them. From four CPUs up the pool is
    left on: at 16 cores it is 1.6x faster on that grid. An explicit
    ``xla_cpu_multi_thread_eigen`` in ``XLA_FLAGS`` always wins. JAX reads the
    variable when it creates the CPU client, so setting it here, before any
    computation, is in time.
    """

    flags = os.environ.get("XLA_FLAGS", "")
    getaff = getattr(os, "sched_getaffinity", None)
    ncpu = len(getaff(0)) if getaff else (os.cpu_count() or 1)
    if "xla_cpu_multi_thread_eigen" not in flags and 1 < ncpu < 4:
        os.environ["XLA_FLAGS"] = f"{flags} --xla_cpu_multi_thread_eigen=false".strip()


_cpu_thread_defaults()

_api = import_module("gkx.api")
__all__ = list(_api.__all__)
_EXPORT_TARGETS: dict[str, tuple[str, str]] = dict(_api._EXPORT_TARGETS)
__all__ = ["__version__", *__all__]


def __getattr__(name: str) -> Any:
    """Lazily resolve public API exports without importing the full solver stack."""

    if name == "__version__":
        return __version__
    try:
        module_name, attr_name = _EXPORT_TARGETS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
