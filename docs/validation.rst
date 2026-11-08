Validation
==========

Validation separates operator identities, numerical convergence and physical
comparison. Passing one layer does not establish the others. Run checks on
the source revision, precision and backend used for the intended calculation.

Operator and integration checks
-------------------------------

The test suite covers spectral derivatives, basis and quadrature identities,
field solves, collision invariants and dissipation, explicit and implicit
time integration, restart continuity and diagnostic normalization. Tests for
optimized or parallel routes compare them with the serial reference operator.

Run focused tests from the repository root, for example:

.. code-block:: bash

   python -m pytest tests/validation/physics_gates/test_collision_physics.py
   python -m pytest tests/unit/objectives/test_autodiff_solver_objectives.py

These commands execute bounded checks. They do not rerun a complete physical
convergence campaign. See :doc:`testing` for the test groups and coverage policy.

Linear physical comparisons
----------------------------

Landau damping, wave dispersion, instability growth and zonal response each
need a declared analytical or independent numerical reference. State the
normalization, collision model and observable used by the reference. In a
finite Hermite expansion, phase mixing can recur; distinguish a transient
initial-value fit from an eigenmode of the regularized finite operator.

For initial-value fits, vary the analysis window and diagnostic cadence and
check competing modes. For eigenmodes, inspect the full-operator residual,
branch identity and conditioning. Increase velocity and spatial resolution,
and vary timestep, domain and numerical regularization independently.
Residual certification is an algebraic check on the chosen discretization.

Cross-code protocol
-------------------

Match geometry profiles and coordinates, normalization, species, kinetic or
adiabatic responses, electromagnetic terms, grid and boundary conditions,
initialization, precision, timestep policy, collisions and regularization.
Check raw fields and diagnostics before fitting rates. Compare complex-field
phase conventions as well as amplitudes; a frequency sign or reference-speed
conversion can otherwise look like a physical discrepancy.

Use the formulas in :doc:`conventions` and :doc:`normalization`. GKX, GX, GS2
and stella have different discretizations and collision-model options. A
matched finite-resolution result supports that specific case; it does not
establish a collisionless limit, a converged nonlinear transport prediction,
or agreement across all kinetic-electron and electromagnetic regimes.
Report unresolved discrepancies with the configuration and convergence tests
needed to distinguish their causes.

Physical reference cases include electron-temperature-gradient turbulence
[Dorland00]_ [Jenko00]_, stellarator calculations [Sanchez21]_
[Papadopoulos23]_, quasilinear stellarator turbulence [Pueschel16]_ and stellarator
transport and zonal-flow studies [Hegna18]_ [McKinney19]_ [Tiwari25]_. Match
the published model and analysis protocol before interpreting agreement.

Nonlinear transport
-------------------

Specify the saturation criterion, averaging interval and sampling cadence.
Check stationarity independently for every trace, account for autocorrelation
when estimating uncertainty, and repeat seeds. Save energy and resolved
field/flux spectra so spatial and velocity-space convergence can be assessed.
Compare smaller timesteps and longer windows as well as higher resolutions.
See :doc:`stellarator_optimization` for the paired-seed uncertainty estimator.

Derivative and performance checks
---------------------------------

Compare derivatives with finite differences across perturbation sizes and
window lengths. A finite-window adjoint supports a local trajectory
sensitivity; assess a design change with independent long simulations.
See :doc:`nonlinear_autodiff` and :doc:`differentiable_eigensolver`.

Measure performance only after checking numerical agreement. Separate
compilation from warm execution, wait for device completion and report the
hardware, software versions, allocation, workload and tolerances. See
:doc:`performance` and :doc:`parallelization`.

Reproducing results
-------------------

Keep input decks, analysis commands and source revisions with a shared result.
Write generated simulation, benchmark and test outputs outside version control.
Source physics tables and comparison input decks remain required inputs.
A published figure or summary cannot substitute for executable setup and
analysis instructions. The supported feature scope is in :doc:`release_scope`.
