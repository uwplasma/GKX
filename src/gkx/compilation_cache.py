"""Persistent JAX compilation cache, on by default.

Compiled kernels are reused across processes, which cuts the first chunk of a
nonlinear run several-fold. ``import gkx`` installs the cache. Set
``GKX_JAX_CACHE=0`` to disable it and ``GKX_JAX_CACHE_DIR`` to choose the
directory. The default is ``$XDG_CACHE_HOME/gkx/jax`` (``~/.cache/gkx/jax``),
separated by JAX version. A cache directory the user already configured, through
``JAX_COMPILATION_CACHE_DIR`` or ``jax.config``, is left alone.

JAX keys entries by the compiled program, compilation options and device.
Unwritable or corrupt entries fall back to compilation. Delete the directory to
reclaim disk space.
"""

from __future__ import annotations

import os
from importlib import metadata
from pathlib import Path
import sys
from typing import Mapping

ENABLE_ENV_VAR = "GKX_JAX_CACHE"
DIRECTORY_ENV_VAR = "GKX_JAX_CACHE_DIR"

_FALSEY = frozenset({"0", "off", "false", "no", "none", "disable", "disabled"})


def compilation_cache_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Return whether the persistent compilation cache should be installed."""

    environ = os.environ if env is None else env
    raw = str(environ.get(ENABLE_ENV_VAR, "")).strip().lower()
    return raw not in _FALSEY


def compilation_cache_directory(env: Mapping[str, str] | None = None) -> Path:
    """Return the cache directory: ``GKX_JAX_CACHE_DIR`` or the XDG user cache."""

    environ = os.environ if env is None else env
    override = str(environ.get(DIRECTORY_ENV_VAR, "")).strip()
    if override:
        return Path(override).expanduser()
    xdg = str(environ.get("XDG_CACHE_HOME", "")).strip()
    return (Path(xdg) if xdg else Path.home() / ".cache") / "gkx" / "jax"


def enable_persistent_compilation_cache(
    env: Mapping[str, str] | None = None,
) -> Path | None:
    """Point JAX at a persistent compilation cache; return the directory used.

    Returns ``None`` when the cache is disabled, when the user already set a JAX
    cache directory, or when the directory cannot be prepared. Never raises: a
    cache is an optimization, and losing it must not lose a run.
    """

    environ = os.environ if env is None else env
    if not compilation_cache_enabled(environ) or environ.get(
        "JAX_COMPILATION_CACHE_DIR"
    ):
        return None
    try:
        directory = compilation_cache_directory(environ)
        if DIRECTORY_ENV_VAR not in environ:
            directory /= f"jax-{metadata.version('jax')}"
        directory.mkdir(parents=True, exist_ok=True)
        jax = sys.modules.get("jax")
        if jax is None:  # JAX reads these when it is first imported
            os.environ["JAX_COMPILATION_CACHE_DIR"] = str(directory)
            os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
        elif not jax.config.jax_compilation_cache_dir:
            jax.config.update("jax_compilation_cache_dir", str(directory))
            jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
        else:
            return None
    except Exception as exc:  # pragma: no cover - depends on the environment
        print(
            f"warning: persistent compilation cache disabled ({exc}); "
            f"set {ENABLE_ENV_VAR}=0 to silence this",
            file=sys.stderr,
            flush=True,
        )
        return None
    return directory


__all__ = [
    "DIRECTORY_ENV_VAR",
    "ENABLE_ENV_VAR",
    "compilation_cache_directory",
    "compilation_cache_enabled",
    "enable_persistent_compilation_cache",
]
