Numerical defaults from SOLVAX
==============================

GKX delegates physics-independent numerics to `SOLVAX
<https://github.com/uwplasma/SOLVAX>`_. This page records which of its
facilities GKX uses, which it deliberately does not, and what the open
candidates are worth -- measured rather than asserted, so that a default can be
changed on evidence and not on preference.

What GKX consumes today
-----------------------

.. list-table::
   :header-rows: 1

   * - SOLVAX entry point
     - Where GKX uses it
   * - ``gmres``
     - implicit linear time stepping (``solvers/linear/implicit.py``) and the
       nonlinear IMEX field solve (``solvers/nonlinear/imex.py``)
   * - ``linear_solve``
     - implicit-function-theorem adjoint of the IMEX solve
   * - ``tridiagonal_solve``
     - batched Hermite-line inverse inside the shifted preconditioner
   * - ``chunked_jacfwd``
     - bounded-memory geometry Jacobians (``geometry/autodiff_checks.py``)

Four of SOLVAX's 113 public entry points. That ratio is not evidence of
anything on its own -- most of SOLVAX exists for problems GKX does not have --
and the one place it was tested, the shift-invert inner solve below, the
unmigrated incumbent turned out to be the right choice.

The shift-invert inner solve: measured, and the incumbent wins
--------------------------------------------------------------

``solvers/linear/krylov_algorithms.py`` imports ``gmres`` from
``jax.scipy.sparse.linalg``, not from SOLVAX. This is deliberate and recorded in
:doc:`solvers`: the branch-continuity gate on that lane is open, so it was never
promoted. The open question was whether migrating would buy anything.

Shift-invert Arnoldi issues ``krylov_dim * restarts`` solves against the *same*
shifted operator, varying only the right-hand side, and starts each one cold --
the sequence-of-related-systems that Krylov recycling targets.
The candidates were measured on the production operator with the physics-aware
Hermite-line preconditioner active and a stated shift offset, counting
matrix-vector products.

.. list-table:: Cyclone s-alpha, 1% shift offset, error against the dense reference
   :header-rows: 1

   * - inner solver
     - matvecs
     - ``n=384`` (ratio 21)
     - ``n=1536`` (ratio 48)
   * - exact LU (harness control)
     - --
     - ``1.56e-15``
     - ``5.48e-15``
   * - **jax.scipy GMRES (incumbent)**
     - **320**
     - **9.12e-15**
     - **4.39e-15**
   * - ``solvax.gmres``
     - 256
     - ``1.06e-11``
     - ``2.95e-11``
   * - ``solvax.gcrot`` (FIFO recycling)
     - 760
     - ``3.12e-12``
     - ``2.70e-12``
   * - ``solvax.gcrot`` (harmonic / GCRO-DR)
     - 760
     - ``2.99e-13``
     - ``7.99e-12``

**Verdict: change nothing.** Every candidate converges. The incumbent matches an
exact direct inner solve to within a few times machine epsilon, at the second
lowest matrix-vector count. Recycling costs 2.4x the matrix-vector products and
is one to three orders of magnitude less accurate: the recycle space buys nothing
here because the preconditioned shifted system is already well conditioned, and
its extra operator applications per restart are pure overhead.

That GKX consumes four of 113 SOLVAX entry points is therefore not, by itself,
evidence of a missed opportunity on this path.

Wall-clock times are not reported: the variants share one process and the first
JAX-backed rung absorbs compilation, which makes the ordering an artifact of the
harness rather than a property of the solvers.

How this was got wrong first
----------------------------

Recorded because the failure mode is reusable, not for penance. An earlier
version of this page reported the opposite conclusion -- that plain GMRES stalled
near ``1e-2`` while recycling reached machine precision. Two defects produced it:

1. The tool passed ``preconditioner="auto"``. That is a valid value for
   ``dominant_eigenpair`` but not one of the names
   ``_build_shift_invert_precond`` matches, and that function returns
   ``(None, None)`` for an unrecognised name rather than raising. Every run was
   **unpreconditioned** while the docs claimed otherwise.
2. The shift was the exact dense rightmost eigenvalue, making
   :math:`A - \sigma I` singular by construction. Unpreconditioned GMRES stalls
   on that -- the stall *was* the reported finding -- while a working
   preconditioner inverts the near-null direction and returns NaN, which is how
   this finally surfaced.

The tool carried an exact-LU control throughout, and it caught an unrelated
branch-selection bug. It could not catch either of these, because it bypasses
both the preconditioner and the conditioning of the shifted solve. **A control
that shares an assumption with the thing it checks validates nothing.**

A third attempt took the shift from a genuinely coarser rung, which is what
production continuation does. That failed too: at sizes where a dense reference
fits there is no room on the resolution ladder, and coarsening ``(2,4)`` to
``(1,2)`` moved the eigenvalue 6.5 magnitudes, so every solver correctly
converged to a different eigenvalue. Hence the stated offset, which makes shift
quality an independent variable instead of an accident.

Open candidates
---------------

Each of these is a hypothesis with a stated reason, not a plan of record.

``mixed_precision`` + ``iterative_refinement``
    Apply the preconditioner in single precision and compute residuals in
    double. GPUs favour fp32 by a wide margin and the preconditioner does not
    need to be accurate -- only useful. CI runs x64 throughout, so a precision
    regression would surface rather than hide. Untested.

``chunked_jacrev`` / ``auto_chunk_size``
    GKX chunks forward-mode Jacobians only. Stellarator optimization
    differentiates few outputs with respect to many boundary coefficients, which
    is the reverse-mode case, and that is where peak memory binds. Untested.

``p_multigrid``
    Coarsen in polynomial degree: a coarse level is a lower ``(Nl, Nm)``
    truncation, which GKX already constructs for its convergence ladder, and the
    coarse space is exactly a subspace of the fine one, so restriction is
    truncation and prolongation is zero-padding. That makes the transfer
    operators exact by construction rather than approximations. Untested, and
    the most promising of the three.

Provenance
----------

The harness that produced the table, ``tools/campaigns/shift_invert_recycling.py``,
was **removed after it answered its question**. It existed to decide one thing --
whether to migrate the shift-invert inner solve -- and the answer was no. Keeping
380 lines of benchmark in the tree to defend a decision to change nothing is the
wrong trade; it is recoverable from commit ``3aa1591b`` if the question reopens.

What survives is the number that matters and the reason to trust it: the exact-LU
control reached machine precision on the same harness that produced the rest of
the column, so the comparison was measured rather than assumed.
