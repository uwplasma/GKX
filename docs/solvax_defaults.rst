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

Four of SOLVAX's 113 public entry points. The gap is not by itself a defect --
most of SOLVAX exists for problems GKX does not have -- but two of the unused
facilities address costs GKX measurably pays, and those are recorded below.

The shift-invert inner solve is not migrated
--------------------------------------------

``solvers/linear/krylov_algorithms.py`` imports ``gmres`` from
``jax.scipy.sparse.linalg``, not from SOLVAX. This is deliberate and recorded in
:doc:`solvers`: the branch-continuity gate on that lane is open, so it was never
promoted. What was missing is the size of the difference.

Shift-invert Arnoldi issues ``krylov_dim * restarts`` solves against the *same*
shifted operator, varying only the right-hand side, and starts each one cold.
``tools/campaigns/shift_invert_recycling.py`` measures the alternatives on the
production operator, counting matrix-vector products.

.. warning::

   **The comparison table below is retracted.** It was produced by a harness
   with two defects, both found after publication:

   1. The tool passed ``preconditioner="auto"``, which is a valid value for
      ``dominant_eigenpair`` but **not** one of the names
      ``_build_shift_invert_precond`` matches. That function returns
      ``(None, None)`` for an unrecognised name rather than raising, so the runs
      were **unpreconditioned** while reporting that the Hermite-line inverse was
      active.
   2. The shift was set to the exact dense rightmost eigenvalue, making
      :math:`A - \sigma I` singular by construction. Unpreconditioned GMRES only
      stalls on that -- which is what produced the ``1e-2`` "stall" readings --
      whereas a working preconditioner inverts the near-null direction and
      returns NaN.

   The exact-LU control could not catch either defect, because it bypasses both
   the preconditioner and the conditioning of the shifted solve. A control that
   shares an assumption with the thing it checks validates nothing.

   The tool now takes a **stated** relative shift offset, so shift quality is an
   independent variable rather than an accident, and raises if the
   preconditioner resolves to ``None``. Numbers will be restored here only after
   a run passes its controls. **Nothing in this section should be cited until
   then**, and no default was changed on the strength of it.

.. list-table:: Cyclone s-alpha, error against the dense reference
   :header-rows: 1

   * - inner solver
     - matvecs
     - ``n=384`` (ratio 21)
     - ``n=1536`` (ratio 48)
   * - exact LU (harness control)
     - --
     - ``1.42e-15``
     - ``5.44e-15``
   * - ``jax.scipy`` GMRES (incumbent)
     - 320
     - ``2.27e-02``
     - ``1.19e-02``
   * - ``solvax.gmres``
     - 256
     - ``4.91e-04``
     - ``9.96e-03``
   * - ``solvax.gcrot`` (FIFO recycling)
     - 760
     - ``1.42e-15``
     - ``5.60e-15``
   * - ``solvax.gcrot`` (harmonic / GCRO-DR)
     - 760
     - ``1.48e-15``
     - ``5.60e-15``

"Ratio" is the spectral radius over the wanted eigenvalue's magnitude, the
measure of how interior the target is.

The apparent reading -- that recycling reproduced an exact direct inner solve
while plain GMRES stalled near ``1e-2`` -- does not survive the defects listed in
the warning above. The ``1e-2`` figures are what an *unpreconditioned* GMRES does
on a *singular* shifted system, which is not a statement about GMRES.

Whether recycling helps a correctly preconditioned, non-singular shifted solve
is therefore still **open**. It is worth re-measuring, because the underlying
argument stands on its own: shift-invert really does issue many solves against
one operator, and that really is what recycling targets.

Wall-clock times from that run are not reported: the variants share one process
and the first JAX-backed rung absorbs compilation, which makes the ordering an
artifact of the harness rather than a property of the solvers.

The exact-LU row is a control, not a candidate. An outer Arnoldi that cannot
reach the dense reference with exact inner solves makes every other row
meaningless, and this one earned its place immediately: it failed at ``5.09`` and
exposed a branch-selection bug in the measurement tool. Shift-invert concentrates
the eigenvalue nearest the shift into the largest :math:`|\theta|`; the tool had
selected on :math:`\max \operatorname{Re}\lambda` after mapping back through
:math:`\lambda = \sigma + 1/\theta`, which lets an inaccurate Ritz value with
small :math:`|\theta|` map to a spurious :math:`\lambda` right of the true
rightmost eigenvalue.

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

Reproducing the table
---------------------

.. code-block:: bash

   python tools/campaigns/shift_invert_recycling.py \
       --n-laguerre 2 --n-hermite 4 --nz 12 \
       --krylov-dim 16 --restarts 4 --maxiter 2000 --restart 60 \
       --output docs/_static/shift_invert_recycling.json

and for the second column, ``--n-laguerre 4 --n-hermite 6 --nz 16``.

The exact-LU control runs first and must report an error at machine precision.
If it does not, no other row in the output is interpretable.
