"""Capability probe for the unreleased SOLVAX eigenpair API.

The adaptive eigensolver consumes SOLVAX interfaces that are not in a released
package yet, so its tests skip when those interfaces are absent. That default is
right for a contributor who installed GKX from PyPI, and wrong everywhere the
paired branch is supposed to be under test: a skipped suite and a passing suite
are indistinguishable at the exit code, so a run that silently tested nothing
reports success.

Set ``GKX_REQUIRE_PAIRED_SOLVAX=1`` to convert the skip into a collection error.
CI sets it on the paired job, so a mis-pinned or stale SOLVAX turns the job red
instead of green-with-no-tests. Use the same variable locally after installing
the paired branch.
"""

from __future__ import annotations

import os

import pytest
import solvax


def _paired_solvax_required() -> bool:
    return os.environ.get("GKX_REQUIRE_PAIRED_SOLVAX", "").strip().lower() not in (
        "",
        "0",
        "false",
        "no",
    )


def missing_solvax_symbols(*names: str) -> tuple[str, ...]:
    """Return the requested SOLVAX callables that this installation lacks."""

    return tuple(name for name in names if not callable(getattr(solvax, name, None)))


def requires_paired_solvax(*names: str) -> pytest.MarkDecorator:
    """Skip unless every named SOLVAX callable is importable.

    Raises instead of skipping when ``GKX_REQUIRE_PAIRED_SOLVAX`` is set, which
    surfaces as a collection error naming the missing symbols and the installed
    version -- the two facts needed to tell "not installed" from "wrong commit".
    """

    missing = missing_solvax_symbols(*names)
    if missing and _paired_solvax_required():
        raise RuntimeError(
            "GKX_REQUIRE_PAIRED_SOLVAX is set but the installed SOLVAX "
            f"{getattr(solvax, '__version__', 'unknown')} at {solvax.__file__} "
            f"lacks: {', '.join(missing)}. Install the paired branch, e.g. "
            "pip install --no-deps -e ../solvax"
        )
    return pytest.mark.skipif(
        bool(missing),
        reason=(
            "requires the SOLVAX eigenpair API (missing: "
            f"{', '.join(missing) or 'none'})"
        ),
    )
