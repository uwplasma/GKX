Features and limitations
========================

GKX solves linear and nonlinear gyrokinetic flux-tube problems with JAX.
Input keys and defaults are defined in :doc:`inputs`; numerical methods are
described in :doc:`numerics` and :doc:`solvers`.

Supported workflows
-------------------

- Analytic slab, s-alpha and Miller geometry; VMEC/VMEX equilibria and imported
  flux tubes for tokamak and stellarator calculations.
- Kinetic species with electrostatic or electromagnetic fields, optional
  adiabatic responses, collisions and declared numerical regularization.
- Initial-value integration, linear eigenmodes, restartable runtime output,
  independent parameter scans and opt-in nonlinear species--Hermite meshes.
- Differentiable geometry and solver objectives, residual-certified linear
  eigenpair sensitivities and finite-window nonlinear derivatives.

Use the checked-in examples for configuration and the tests for executable
contracts. Keep generated simulation, benchmark and test outputs outside
version control. Source collision tables and input decks are required inputs.

Numerical limits
----------------

A certified eigenpair establishes the residual of the discretized operator.
Physical convergence also requires resolution, domain, collision and
regularization studies. At a branch crossing, mode identity and derivative
conditioning need independent checks. An initial-value fit also depends on
the fit window and contamination by competing modes.

A finite nonlinear derivative describes the executed discrete trajectory and
its selected diagnostic window. It does not establish the derivative of an
infinite-time turbulent average. Verify finite differences over a suitable
step-size range and use independent, longer runs and replicated seeds to
assess a proposed design change. See :doc:`nonlinear_autodiff`.

Quasilinear observables and fitted saturation models require calibration and
holdout checks for the intended geometry and regime. They do not by themselves
establish absolute nonlinear heat flux or transfer to another regime.

Parallel execution defaults to serial. Independent batches preserve separate
solver calls; a species--Hermite mesh partitions one nonlinear state under
explicit support checks. Distributed perpendicular FFTs and whole-state
field-line decomposition are outside the supported nonlinear runtime. See
:doc:`parallelization` for the configuration and validation requirements.

Comparisons
-----------

A cross-code comparison needs matched geometry, normalization, species, grid,
boundaries, initialization, precision, timestep policy, collisions,
regularization, diagnostic definition and analysis window. A discrepancy at
one finite resolution cannot establish which code has the converged answer.
A bounded test or one successful case supports only the tested configuration.
