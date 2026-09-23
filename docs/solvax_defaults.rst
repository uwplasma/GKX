Numerical defaults from SOLVAX
==============================

GKX delegates physics-independent numerics to `SOLVAX
<https://github.com/uwplasma/SOLVAX>`_ (``solvax>=0.26.0``). This page lists the
SOLVAX functions GKX calls and where it calls them. Physics, preconditioners,
certification gates and branch selection stay in GKX.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - SOLVAX function
     - where GKX calls it
   * - ``gmres``
     - backward-Euler implicit linear stepping
       (``gkx.solvers_linear_implicit``); the nonlinear IMEX field solve
       (``gkx.solvers_nonlinear_imex``); every shifted inner solve of
       shift-invert Arnoldi (``gkx.solvers_linear_krylov_algorithms``); the
       propagator fixed-point solve of the adaptive eigenpair sensitivity
       (``gkx.objectives.core``)
   * - ``KrylovSolution``
     - result type of the implicit and IMEX GMRES solves, read into
       ``ImplicitSolveStats``
   * - ``linear_solve``
     - IMEX implicit solve and its implicit-function-theorem adjoint
       (``gkx.solvers_nonlinear_imex``)
   * - ``tridiagonal_solve``
     - batched Hermite-line inverse (``gkx.solvers_linear_implicit``), used by
       the implicit time-step preconditioner and by the ``hermite-line`` and
       ``field-corrected`` shift-invert preconditioners
   * - ``block_thomas_factor_ops``
     - Schur elimination of the z-local block in the ``pr3-cm`` preconditioner
       (``gkx.solvers_linear_precond_pr3``); GKX's forward substitution is
       pinned bitwise to ``block_thomas_solve_ops`` on the same factors
   * - ``estimate_rk4_timestep``, ``adaptive_eigenpair``
     - stable-step estimate and residual-certified propagator eigensolve behind
       ``method="adaptive"`` (``gkx.solvers_linear_adaptive_propagator``)
   * - ``exponential_eigenpairs``
     - optional exponential-Krylov filter of the adaptive eigensolve and of the
       differentiable eigenpair's adjoint candidates
   * - ``propagator_eigenpairs``, ``eigenpair_reverse``
     - candidate extraction and implicit reverse rule of the differentiable
       eigenpair (``gkx.objectives.core``; see
       :doc:`differentiable_eigensolver`)
   * - ``sparse_operator_matrix``, ``SpluFactorization``,
       ``sparse_eigenpairs``
     - bounded-batch sparse assembly, shifted factorization and eigenpairs of
       ``method="sparse_shift_invert"`` (``gkx.solvers_linear_krylov``)
   * - ``chunked_jacfwd``
     - bounded-memory geometry Jacobians (``gkx.geometry.autodiff_checks``)

The eigensolver and sparse functions are imported inside the function that
uses them, so importing GKX does not load them.

A shift-invert inner-solver comparison once published on this page is
retired: it ran unpreconditioned, on a shifted system that was singular by
construction, so its accuracy ranking of plain and recycled GMRES says nothing
about the production solve.
