"""Shared JAX profiler options for trace tools."""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any


def git_source_state(root: Path) -> dict[str, Any]:
    """Return source revision and dirty state for profiler artifacts."""

    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"git_revision": "unknown", "git_dirty": None}
    return {"git_revision": revision, "git_dirty": dirty}


def make_profile_options(*, python_tracer_level: int = 0, host_tracer_level: int = 0):
    """Return JAX profile options with explicit tracer levels.

    The default levels avoid the optional TensorFlow Python-trace hook, which is
    not present on the lightweight `office` profiling environment.
    """

    import jax.profiler as jprof

    opts = jprof.ProfileOptions()
    opts.python_tracer_level = int(python_tracer_level)
    opts.host_tracer_level = int(host_tracer_level)
    return opts
