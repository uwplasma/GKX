from __future__ import annotations

import os
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

#: The XLA:CPU FFT is only handed to a thread pool when this flag is true, and
#: the pool's reduction order is not reproducible run to run.
DETERMINISTIC_CPU_FFT_FLAG = "--xla_cpu_multi_thread_eigen=false"


def _pin_deterministic_cpu_fft() -> None:
    """Make the CPU FFT reproducible before any JAX backend is created.

    Several tests here assert *bitwise* equality rather than a tolerance --
    ``test_prepared_and_runtime_nonlinear_routes_are_bitwise_identical`` and
    the restart round trip among them -- and the research gates that compare a
    branch against ``main`` assert it across dozens of arrays.  Under the
    multithreaded XLA:CPU FFT thunk that is not a property the code controls:
    #248 measured two runs of *unmodified* ``main``, minutes apart, differing
    by 1.4e-10 on the float32 nonlinear RHS with no code change between them,
    and every such pair came back bitwise with this flag set.  Pinning it here
    keeps those gates measuring the code instead of the thread pool.

    The flag is only added when the caller has not already spoken about it, so
    an explicit ``XLA_FLAGS`` still wins.  ``conftest.py`` is imported before
    any test module, and JAX creates its CPU client lazily on first use, so the
    variable is in place before the flag is read.
    """

    flags = os.environ.get("XLA_FLAGS", "")
    if "xla_cpu_multi_thread_eigen" in flags:
        return
    os.environ["XLA_FLAGS"] = f"{flags} {DETERMINISTIC_CPU_FFT_FLAG}".strip()


_pin_deterministic_cpu_fft()


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

    Parts of this suite assert float64-scale tolerances: Krylov preconditioner
    conditioning, the electromagnetic zonal solve. When this banner was added
    (#217), 17 tests failed at default precision on the format rather than on
    the physics, and they read as broken solvers to anyone who does not already
    know that. The nonlinear window AD/FD matrix was among them; it now checks
    its float32 adjoint against a float64 reference instead.

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
        "  Tests asserting float64-scale tolerances (Krylov conditioning, the\n"
        "  EM zonal solve) will fail on precision, not on physics. CI runs:\n"
        "      MPLBACKEND=Agg JAX_ENABLE_X64=true GKX_X64=1 pytest\n"
        "\n"
    )
