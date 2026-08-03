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
production operator with the physics-aware Hermite-line preconditioner left on,
counting matrix-vector products.

.. list-table:: Cyclone s-alpha, ``(Nl,Nm)=(2,4)``, ``nz=12``, ``n=384``
   :header-rows: 1

   * - inner solver
     - matvecs
     - error vs dense
   * - exact LU (harness control)
     - --
     - ``1.42e-15``
   * - ``jax.scipy`` GMRES (incumbent)
     - 320
     - ``2.27e-02``
   * - ``solvax.gmres``
     - 256
     - ``4.91e-04``
   * - ``solvax.gcrot`` (FIFO recycling)
     - 760
     - ``1.42e-15``
   * - ``solvax.gcrot`` (harmonic / GCRO-DR)
     - 760
     - ``1.48e-15``

Both recycling strategies reproduce an exact direct inner solve; neither plain
GMRES does. **Recycling is buying convergence here, not speed** -- it costs 2.4x
the matrix-vector products. The cheaper FIFO strategy is as accurate as harmonic
deflation at this size, so harmonic Ritz deflation is not yet justified.

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

The exact-LU control runs first and must report an error at machine precision.
If it does not, no other row in the output is interpretable.
