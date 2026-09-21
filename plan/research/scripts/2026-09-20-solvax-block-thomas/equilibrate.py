# ruff: noqa: E402
"""Does SOLVAX's Ruiz equilibration rescue the ``hermite-line`` stall?

Q28 (#261) measured, on the shipped Cyclone deck at ``(Nz, Nl, Nm) = (96, 4, 8)``
(``d96``) and on the very first right-hand side the outer Arnoldi generates,
that ``hermite-line`` does not converge: 300 unrestarted FGMRES iterations
leave a relative residual of 0.9047 and 600 leave 0.5493, where ``pr3-cm``
reaches the 1e-4 tolerance in 252. This asks whether that stall is a *scaling*
problem that :func:`solvax.equilibrate` (Ruiz, SOLVAX >= 0.24) can remove.

The case construction is Q28's ``diagnose.py``, imported unchanged, so the
operator, deck, seed, shift and right-hand side are the ones Q28 measured.

``solvax.equilibrate`` scales an *assembled* sparse matrix, so ``B = A - sigma
I`` is sampled into CSR with :func:`solvax.sparse_operator_matrix`; ``d96`` is
small enough for that. With ``D_r``, ``D_c`` its row and column scalings, the
arms are, all from ``x = 0``, all unrestarted:

``hl``          ``B``, right preconditioner ``M^-1`` (the Q28 baseline)
``hl+ruiz``     ``D_r B D_c`` with ``M`` scaled consistently, ``D_c^-1 M^-1
                D_r^-1``, then ``x = D_c y``. The preconditioned operator is
                ``D_r (B M^-1) D_r^-1``: a diagonal *similarity* of the
                baseline's, so the spectrum is unchanged and only the norm GMRES
                minimizes moves.
``hl+ruiz-row`` ``D_r B`` with the *unscaled* ``M^-1``: a left row scaling that
                is not a similarity, the one arm here that can change the
                spectrum GMRES sees.
``ruiz``        ``D_r B D_c`` and no preconditioner: Ruiz as a preconditioner in
                its own right.
``none``        ``B`` and no preconditioner, the reference for ``ruiz``.
``pr3``         ``B`` with ``pr3-cm``: the positive control.
``pr3+ruiz``    ``pr3-cm`` scaled consistently, to check scaling does not hurt a
                preconditioner that works.

Every residual reported is the **true, unscaled** ``||b - B x|| / ||b||``;
the scaled arms' own stopping norm is ``||D_r (b - B x)||`` and is not what the
outer eigensolve needs. One fresh process. Output: progress lines and one
``RESULT {json}`` line.

A defect this run turned up, and why every scaled arm runs twice
------------------------------------------------------------------
``solvax.equilibrate`` (0.24.0) begins ``matrix.tocsr().astype(np.float64)``,
which on a complex matrix **discards the imaginary part** (numpy says so with a
``ComplexWarning``, and nothing else does). ``B`` here is complex128, so the
row and column maxima it equilibrates are those of ``Re B``, not ``B``: on a
two-by-two check an entry ``1e-6 + 1j``, of magnitude one, gets its row scaled
by 1000. Ruiz depends only on entry magnitudes, so ``equilibrate(abs(B))``
computes the correct scaling for a complex matrix. Arms suffixed ``[|B|]`` use
that, and are the fair test of the idea; arms suffixed ``[Re B]`` use what the
shipped call returns when handed ``B`` itself, and are recorded because that is
what a caller gets.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-19-pr3cm-src"))

import jax
import jax.numpy as jnp
import numpy as np
import solvax

import diagnose as dg  # noqa: E402  (Q28's harness, reused unchanged)

from gkx.solvers_linear_krylov import KrylovConfig, _normalized_config
from gkx.solvers_linear_krylov_algorithms import (
    _apply_operator,
    _projected_flat_operator,
    build_shift_invert_preconditioner,
)
from gkx.solvers_linear_precond_pr3 import build_pr3_factors

BUDGETS = (60, 150, 300, 600)
TOL = 1.0e-4  # the shipped inner tolerance Q28 measured against


def main() -> None:
    bk = dg.bk
    case = bk.Case("d96", None)
    rec: dict = {
        "case": "d96",
        "n": case.n,
        "shape": list(case.shape),
        "solvax": solvax.__version__,
        "jax": jax.__version__,
        "x64": bool(jax.config.read("jax_enable_x64")),
        "git": bk.git_state(),
        "env": {
            k: os.environ.get(k)
            for k in ("JAX_ENABLE_X64", "JAX_PLATFORMS", "XLA_FLAGS", "OMP_NUM_THREADS")
        },
    }
    cfg = _normalized_config(dict(KrylovConfig(method="shift_invert").__dict__))
    sigma, v_init, covered = dg._first_rhs(case, cfg)
    sigma = jnp.asarray(sigma, dtype=v_init.dtype)
    rec["sigma"] = [float(np.real(sigma)), float(np.imag(sigma))]
    shape, size = v_init.shape, v_init.size
    bk.log(f"d96 n={size} sigma={complex(np.asarray(sigma))}")

    @jax.jit
    def matvec(x_flat):
        x = x_flat.reshape(shape)
        return (
            _apply_operator(x, case.cache, case.params, case.T0) - sigma * x
        ).reshape(size)

    def precond_for(mode):
        factors = None
        if mode == "pr3-cm":
            factors, _meta = build_pr3_factors(
                v_init, case.cache, case.params, case.T0, sigma
            )
        _p, raw = build_shift_invert_preconditioner(
            v_init, case.cache, case.params, case.T0, sigma, mode, factors
        )
        return jax.jit(_projected_flat_operator(raw, covered, shape))

    b = v_init.reshape(size)
    b = b / jnp.linalg.norm(b)

    def true_rel(x):
        return float(np.asarray(jnp.linalg.norm(b - matvec(x)) / jnp.linalg.norm(b)))

    # --- assemble and equilibrate B = A - sigma I ------------------------------
    t = time.perf_counter()
    b_csr = solvax.sparse_operator_matrix(matvec, b, batch_size=64)
    assemble_s = time.perf_counter() - t
    probe = np.random.default_rng(0).standard_normal(size) + 0j
    assembly_defect = float(
        np.linalg.norm(b_csr @ probe - np.asarray(matvec(jnp.asarray(probe))))
        / np.linalg.norm(np.asarray(matvec(jnp.asarray(probe))))
    )
    magnitude = abs(b_csr)  # real, same magnitudes: what Ruiz actually needs
    rec["equilibration"] = {
        "assemble_seconds": assemble_s,
        "nnz": int(b_csr.nnz),
        "assembly_defect": assembly_defect,
        # entries whose real part is much smaller than their modulus are the
        # ones the float64 cast misjudges
        "entries_re_below_1e-3_of_modulus": int(
            (np.abs(b_csr.data.real) < 1.0e-3 * np.abs(b_csr.data)).sum()
        ),
    }
    scalings: dict = {}
    for label, source in (("|B|", magnitude), ("Re B", b_csr)):
        t = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            eq = solvax.equilibrate(source)
        dr = np.asarray(eq.row_scale)
        dc = np.asarray(eq.column_scale)
        # eq.matrix is always real; compare it against |D_r B D_c|, which is
        # what it is when the scaling was computed from the right magnitudes
        expected = abs(magnitude.multiply(dr[:, None]).multiply(dc[None, :]))
        rec["equilibration"][label] = {
            "seconds": time.perf_counter() - t,
            "warnings": sorted({f"{w.category.__name__}: {w.message}" for w in caught}),
            "matrix_defect_vs_abs_DrBDc": float(abs(abs(eq.matrix) - expected).max()),
            "original_spread": float(eq.original_spread),
            "spread": float(eq.spread),
            "row_scale_range": [float(dr.min()), float(dr.max())],
            "column_scale_range": [float(dc.min()), float(dc.max())],
            "row_max_after_range": [
                float(v)
                for v in np.quantile(
                    np.asarray(expected.max(axis=1).todense()).ravel(), [0.0, 1.0]
                )
            ],
        }
        scalings[label] = (jnp.asarray(dr), jnp.asarray(dc))
        bk.log(f"equilibration[{label}] {rec['equilibration'][label]}")
    # How far from equilibrated B already is: its own row/column maxima.
    row_max = np.asarray(magnitude.max(axis=1).todense()).ravel()
    col_max = np.asarray(magnitude.max(axis=0).todense()).ravel()
    rec["equilibration"]["B_row_max_range"] = [
        float(row_max.min()),
        float(row_max.max()),
    ]
    rec["equilibration"]["B_col_max_range"] = [
        float(col_max.min()),
        float(col_max.max()),
    ]
    bk.log(
        f"B row max in [{row_max.min():.4g}, {row_max.max():.4g}], "
        f"col max in [{col_max.min():.4g}, {col_max.max():.4g}]"
    )

    hl = precond_for("hermite-line")
    pr3 = precond_for("pr3-cm")

    def scaled(dr_j, dc_j):
        def matvec_s(v):
            return dr_j * matvec(dc_j * v)

        def consistent(pre):
            return lambda v: pre(v / dr_j) / dc_j

        return matvec_s, consistent

    # (matvec, precond, rhs, recover-x)
    arms: dict = {
        "hl": (matvec, hl, b, lambda y: y),
        "none": (matvec, None, b, lambda y: y),
        "pr3": (matvec, pr3, b, lambda y: y),
    }
    for label, (dr_j, dc_j) in scalings.items():
        mv_s, consistent = scaled(dr_j, dc_j)

        def rec_c(y, _dc=dc_j):
            return _dc * y

        def row_mv(v, _dr=dr_j):
            return _dr * matvec(v)

        arms |= {
            f"hl+ruiz[{label}]": (mv_s, consistent(hl), dr_j * b, rec_c),
            f"hl+ruiz-row[{label}]": (row_mv, hl, dr_j * b, lambda y: y),
            f"ruiz[{label}]": (mv_s, None, dr_j * b, rec_c),
            f"pr3+ruiz[{label}]": (mv_s, consistent(pr3), dr_j * b, rec_c),
        }

    results: dict = {}
    for name, (mv, pre, rhs, recover) in arms.items():
        row: dict = {"curve": []}
        # residual after a fixed number of unrestarted iterations (no early exit)
        for budget in BUDGETS:
            t = time.perf_counter()
            sol = solvax.gmres(
                mv, rhs, precond=pre, rtol=1e-14, restart=budget, max_restarts=1
            )
            row["curve"].append(
                {
                    "iterations": int(np.asarray(sol.iterations)),
                    "true_relative_residual": true_rel(recover(sol.x)),
                    "seconds": time.perf_counter() - t,
                }
            )
        # iterations to the shipped 1e-4, in the arm's own stopping norm, then
        # checked in the true one
        sol = solvax.gmres(
            mv, rhs, precond=pre, rtol=TOL, restart=max(BUDGETS), max_restarts=1
        )
        row["to_tol"] = {
            "iterations": int(np.asarray(sol.iterations)),
            "converged_own_norm": bool(np.asarray(sol.converged)),
            "true_relative_residual": true_rel(recover(sol.x)),
        }
        results[name] = row
        bk.log(
            f"{name:22s} "
            + " ".join(
                f"{c['iterations']}:{c['true_relative_residual']:.4g}"
                for c in row["curve"]
            )
            + f" | to 1e-4: {row['to_tol']}"
        )
    rec["arms"] = results
    print("RESULT " + json.dumps(rec, sort_keys=True))


if __name__ == "__main__":
    main()
