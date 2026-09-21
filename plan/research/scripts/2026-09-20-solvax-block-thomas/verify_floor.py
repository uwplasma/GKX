"""The lowest SOLVAX release GKX can declare, checked wheel by wheel.

With no arguments this is the driver: for each version in ``VERSIONS`` it
downloads the wheel from PyPI (``pip download --no-deps --only-binary=:all:
--index-url https://pypi.org/simple``) into a temporary directory, checks its
SHA-256 against the digest PyPI's JSON API publishes, unpacks it, and runs the
worker below on it in a fresh interpreter, printing one JSON line per version.

``--check DIR`` is the worker, run against one unpacked wheel with ``DIR`` first
on ``sys.path`` so that wheel wins over whatever SOLVAX is installed. It checks
that every name GKX's ``src/`` imports from SOLVAX exists (the list was
enumerated from the AST, not grepped), and then runs GKX's *own*
``_block_thomas_factors`` / ``_block_thomas_substitute`` on a random
block-tridiagonal system with Hermite-banded couplings: the result must equal
``solvax.block_thomas_solve_ops`` bitwise and solve the dense system to 1e-12.

Run from the repository root:

    PYTHONPATH=$PWD/src:$PWD python plan/research/scripts/2026-09-20-solvax-block-thomas/verify_floor.py
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import os
import sys
from pathlib import Path

os.environ["JAX_ENABLE_X64"] = "true"
os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO = Path(__file__).resolve().parents[4]
VERSIONS = [f"0.{minor}.0" for minor in range(12, 25)]


def driver() -> None:
    import hashlib
    import subprocess
    import tempfile
    import urllib.request
    import zipfile

    with urllib.request.urlopen("https://pypi.org/pypi/solvax/json") as reply:
        releases = json.load(reply)["releases"]
    with tempfile.TemporaryDirectory() as tmp:
        for version in VERSIONS:
            subprocess.run(
                [sys.executable, "-m", "pip", "download", "--no-deps", "-q",
                 "--only-binary=:all:", "--index-url", "https://pypi.org/simple",
                 f"solvax=={version}", "-d", tmp],
                check=True, capture_output=True,
            )  # fmt: skip
            wheel = Path(tmp) / f"solvax-{version}-py3-none-any.whl"
            digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
            published = {
                f["filename"]: f["digests"]["sha256"] for f in releases[version]
            }
            unpacked = Path(tmp) / version
            zipfile.ZipFile(wheel).extractall(unpacked)
            worker = subprocess.run(
                [sys.executable, __file__, "--check", str(unpacked)],
                capture_output=True, text=True,
            )  # fmt: skip
            line = next(
                (ln for ln in worker.stdout.splitlines() if ln.startswith("{")),
                json.dumps(
                    {"version": version, "PASS": False, "error": worker.stderr[-300:]}
                ),
            )
            result = json.loads(line)
            result["sha256"] = digest
            result["sha256_matches_pypi"] = digest == published[wheel.name]
            result["PASS"] = bool(result["PASS"] and result["sha256_matches_pypi"])
            result.pop("wheel", None)
            print(json.dumps(result, sort_keys=True), flush=True)


if len(sys.argv) == 1:
    driver()
    sys.exit(0)
assert sys.argv[1] == "--check", sys.argv
WHEEL = os.path.abspath(sys.argv[2])
sys.path.insert(0, WHEEL)  # the wheel under test wins over site-packages
sys.path.insert(1, str(REPO / "src"))

report: dict = {"wheel": os.path.basename(WHEEL)}

import solvax  # noqa: E402

report["version"] = getattr(solvax, "__version__", "?")
report["from_wheel"] = os.path.abspath(solvax.__file__).startswith(WHEEL)

# Every name GKX's src/ imports from solvax (AST-enumerated), plus the one
# accessed as an attribute.
NAMES = [
    "adaptive_eigenpair",
    "estimate_rk4_timestep",
    "exponential_eigenpairs",
    "SpluFactorization",
    "sparse_eigenpairs",
    "sparse_operator_matrix",
    "KrylovSolution",
    "gmres",
    "linear_solve",
    "tridiagonal_solve",
    "eigenpair_reverse",
    "propagator_eigenpairs",
    "chunked_jacfwd",
    "block_thomas_factor_ops",
]
# Used by the test that pins GKX's substitution against SOLVAX's.
TEST_NAMES = ["block_thomas_solve_ops"]
report["missing_src"] = [n for n in NAMES if not hasattr(solvax, n)]
report["missing_test"] = [n for n in TEST_NAMES if not hasattr(solvax, n)]

ok_functional = False
detail = ""
if not report["missing_src"] and not report["missing_test"]:
    try:
        sig = inspect.signature(solvax.block_thomas_factor_ops)
        assert "store" in sig.parameters, f"no store kwarg: {sig}"

        import jax
        import jax.numpy as jnp
        import numpy as np

        # GKX's real code paths, not a re-implementation of them.
        import gkx.solvers_linear_precond_pr3 as pr3

        rng = np.random.default_rng(0)
        nblocks, nl, nm = 5, 6, 7
        full = np.zeros((nblocks, nl, nm, nl, nm), np.complex128)
        for a in range(nl):
            full[:, a, :, a, :] = (
                rng.standard_normal((nblocks, nm, nm))
                + 1j * rng.standard_normal((nblocks, nm, nm))
                + 6.0 * np.eye(nm)
            )
            for b in (a - 1, a + 1):
                if 0 <= b < nl:
                    for o in (-1, 0, 1):  # Hermite half-width 1
                        idx = np.arange(nm)
                        ok = (idx + o >= 0) & (idx + o < nm)
                        full[:, a, idx[ok], b, (idx + o)[ok]] = rng.standard_normal(
                            (nblocks, int(ok.sum()))
                        )
        no_phi = full.reshape(nblocks, nl * nm, nl * nm)
        shift = 0.3 - 0.2j
        _d, lo, up = pr3._laguerre_bands(no_phi, nl, nm, shift)
        p = pr3._coupling_halfwidth(lo, up, float(np.abs(no_phi).max()))
        assert p == 1, p

        schur = pr3._block_thomas_factors(no_phi, nl, nm, shift, p)
        assert dataclasses.is_dataclass(schur), "factors are not a dataclass"
        assert schur.pivots is None, "store='inverse' kept pivots"
        assert schur.store == "inverse"
        assert tuple(schur.blocks.shape) == (nblocks, nl, nm, nm), schur.blocks.shape

        rhs = rng.standard_normal((nblocks, nl * nm)) + 1j * rng.standard_normal(
            (nblocks, nl * nm)
        )
        ours = np.asarray(pr3._block_thomas_substitute(schur, nl, nm)(jnp.asarray(rhs)))
        theirs = np.asarray(
            jax.vmap(solvax.block_thomas_solve_ops)(
                schur, jnp.asarray(rhs).reshape(nblocks, nl, nm)
            ).reshape(nblocks, nl * nm)
        )
        mat = no_phi - shift * np.eye(nl * nm)
        resid = np.linalg.norm(
            np.einsum("bij,bj->bi", mat, ours) - rhs
        ) / np.linalg.norm(rhs)
        report["bitwise_vs_solvax_solve"] = bool(np.array_equal(ours, theirs))
        report["residual_vs_dense"] = float(resid)
        ok_functional = bool(report["bitwise_vs_solvax_solve"] and resid < 1e-12)
    except Exception as exc:  # noqa: BLE001
        detail = f"{type(exc).__name__}: {exc}"[:300]

report["functional_ok"] = ok_functional
if detail:
    report["error"] = detail
report["PASS"] = bool(
    report["from_wheel"]
    and not report["missing_src"]
    and not report["missing_test"]
    and ok_functional
)
print(json.dumps(report))
