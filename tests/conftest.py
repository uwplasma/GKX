from __future__ import annotations

import sys
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parent
ROOT = TESTS_ROOT.parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))


def _precision_environment() -> tuple[bool, list[str]]:
    """Return whether float64 is in force, and which switches are missing."""

    import os

    checks = (
        (
            "JAX_ENABLE_X64=true",
            os.environ.get("JAX_ENABLE_X64", "").strip().lower() in {"1", "true"},
        ),
        ("GKX_X64=1", os.environ.get("GKX_X64", "").strip() == "1"),
    )
    missing = [name for name, enabled in checks if not enabled]
    return not missing, missing


def pytest_report_header(config) -> str:
    """Record which precision the suite ran at, for anyone reading the log."""

    enabled, missing = _precision_environment()
    if enabled:
        return "gkx precision: float64 (JAX_ENABLE_X64=true GKX_X64=1), as CI runs"
    return "gkx precision: DEFAULT (float32) -- missing " + " ".join(missing)


def pytest_configure(config) -> None:
    """Warn, unmissably, when the suite is about to run at the wrong precision.

    Parts of this suite assert float64-scale tolerances: finite-difference
    agreement for the nonlinear window gradients, Krylov preconditioner
    conditioning, the electromagnetic zonal solve. At default precision 17 of
    them fail on the format rather than on the physics, and they read as broken
    solvers to anyone who does not already know that.

    This cannot go through ``pytest_report_header`` or a warning alone, because
    ``pytest.ini`` passes ``-q --disable-warnings`` and a plain ``pytest`` run --
    exactly the one a newcomer types -- suppresses both. So it is written
    straight to stderr, and only when something is missing: a correctly
    configured run stays silent. Nothing is skipped or loosened.
    """

    enabled, missing = _precision_environment()
    if enabled:
        return
    sys.stderr.write(
        "\n"
        "  gkx: running at DEFAULT (float32) precision -- missing "
        + " ".join(missing)
        + "\n"
        "  Tests asserting float64-scale tolerances (nonlinear window AD/FD,\n"
        "  Krylov conditioning, the EM zonal solve) will fail on precision, not\n"
        "  on physics. CI runs:\n"
        "      MPLBACKEND=Agg JAX_ENABLE_X64=true GKX_X64=1 pytest\n"
        "\n"
    )
