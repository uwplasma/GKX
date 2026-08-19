"""Persistent JAX compilation cache for the GKX executable.

A GKX run is compile-dominated at the sizes people actually iterate on. On an
office GPU a 100-step nonlinear case spends ~22.9 s of a ~25 s integrator wall
inside XLA compilation, so the accelerator only wins end to end past roughly
seventy steps; on a laptop CPU the same compile costs ~14 s. That cost is paid
again on every fresh process even when nothing about the problem changed, which
is the common case while a user edits a TOML and re-runs.

JAX can persist compiled executables across processes, and this module turns
that on by default for the executable. Two environment variables are the
escape: ``GKX_JAX_CACHE=0`` disables the cache entirely, and
``GKX_JAX_CACHE_DIR`` moves it. The default location mirrors how the repository
already caches generated ``*.eik.nc`` geometry -- a ``.cache/gkx`` directory
beside the source tree -- and falls back to ``~/.cache/gkx/jax`` when that tree
is read-only, which is what an installed wheel sees.

On correctness: the cache key is XLA's, not ours. It is computed from the
lowered HLO module, the compilation options, and the backend/device the
executable was built for, so a changed grid, changed physics, or a different
device is a different key rather than a stale hit. The one thing that key does
not span is the toolchain itself, so entries are additionally namespaced by the
installed ``jax`` version: upgrading JAX starts a new directory instead of
reading executables built by a different compiler. ``jax_raise_persistent_cache_errors``
is left at its default of ``False`` so a corrupt or unwritable entry degrades
to a recompile rather than failing a simulation.

The directory only grows, at roughly 1.6 MB per distinct problem shape. JAX
0.9 exposes no writable size cap, so clearing it is `rm -rf` on the directory
this module reports; nothing is lost but the next cold compile.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Mapping

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CACHE_ROOT = _REPO_ROOT / ".cache" / "gkx" / "jax"
_FALLBACK_CACHE_ROOT = Path.home() / ".cache" / "gkx" / "jax"

ENABLE_ENV_VAR = "GKX_JAX_CACHE"
DIRECTORY_ENV_VAR = "GKX_JAX_CACHE_DIR"

_FALSEY = frozenset({"0", "off", "false", "no", "none", "disable", "disabled"})


def compilation_cache_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Return whether the persistent compilation cache should be installed."""

    environ = os.environ if env is None else env
    raw = str(environ.get(ENABLE_ENV_VAR, "")).strip().lower()
    return raw not in _FALSEY


def _writable_root(root: Path) -> bool:
    """Return whether ``root`` can be created and written to."""

    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    return os.access(root, os.W_OK)


def compilation_cache_directory(env: Mapping[str, str] | None = None) -> Path:
    """Return the directory that should hold persisted executables.

    An explicit ``GKX_JAX_CACHE_DIR`` wins and is used verbatim. Otherwise the
    repository-local ``.cache/gkx/jax`` is preferred so the cache is discarded
    with the checkout, with ``~/.cache/gkx/jax`` as the fallback for installs
    whose package tree is not writable.
    """

    environ = os.environ if env is None else env
    override = str(environ.get(DIRECTORY_ENV_VAR, "")).strip()
    if override:
        return Path(override).expanduser()
    if _writable_root(_DEFAULT_CACHE_ROOT):
        return _DEFAULT_CACHE_ROOT
    return _FALLBACK_CACHE_ROOT


def _versioned_cache_directory(root: Path) -> Path:
    """Namespace a cache root by the installed JAX version."""

    import jax

    return root / f"jax-{jax.__version__}"


def enable_persistent_compilation_cache(
    env: Mapping[str, str] | None = None,
) -> Path | None:
    """Point JAX at a persistent compilation cache; return the directory used.

    Returns ``None`` when the cache is disabled by ``GKX_JAX_CACHE``, when JAX
    is not importable, or when the directory cannot be prepared. Never raises:
    a cache is an optimization, and losing it must not lose a run.
    """

    if not compilation_cache_enabled(env):
        return None
    try:
        import jax

        directory = _versioned_cache_directory(compilation_cache_directory(env))
        directory.mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(directory))
        jax.config.update("jax_enable_compilation_cache", True)
        # JAX defaults to persisting only compiles that took longer than a
        # second, on the theory that a cheap compile is not worth a disk round
        # trip. That heuristic is wrong for GKX, and measurably so. A 16x16x32
        # nonlinear step compiles ~320 separate kernels; the fused scan is only
        # 2.1 s of a 13.5 s total and every one of the rest is a ~20 ms compile
        # that the default threshold refuses to store. Keeping the threshold
        # saves 3.9 s of that 13.5 s; dropping it to zero saves 12.6 s, for
        # 1.6 MB of cache. The threshold is what was making a re-run of an
        # unchanged case still feel cold.
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    except Exception as exc:  # pragma: no cover - depends on the environment
        # Say so rather than failing silently: a cache that did not install is
        # a run that will keep paying a cold compile, and the user should know
        # which of the two they are getting.
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
