Algorithms
==========

GKX expands each kinetic species in Hermite--Laguerre velocity moments and
Fourier modes in the perpendicular plane. The state has axes
``(species, l, m, ky, kx, z)``. Streaming couples adjacent Hermite moments;
mirror and drift terms couple velocity moments; field equations recover the
electrostatic and electromagnetic fields from distribution moments.
See :doc:`theory`, :doc:`linear_model` and :doc:`operators` for the equations.

Velocity truncation and boundaries
----------------------------------

A truncated Hermite ladder reflects phase-mixed structure. Collisions,
hypercollisions and closure choices control that reflection and must be
reported as part of the model. Increase Hermite and Laguerre resolutions
independently and check the regularization sensitivity of the desired
observable. See :doc:`numerics` for closure, quadrature and nodal-grid options.

Twist-and-shift boundaries connect perpendicular modes into parallel chains.
Supplied states must satisfy the chain-cover projection; otherwise decoupled
rows can contaminate the nonlinear convolution. Periodic and linked cases
have distinct streaming and projection contracts. End damping is specified
as a rate and requires a parallel-domain sensitivity check.

Linear eigenmodes
-----------------

An eigensolver candidate is accepted only after checking its residual against
the original operator. Residual certification establishes the discrete mode;
resolution convergence, branch identity and physical interpretation require
separate checks. A zero or non-finite candidate cannot certify.

Dense methods store :math:`O(n^2)` operator entries. Matrix-free methods store
a bounded basis with :math:`O(nm)` storage. Shift-invert methods also require
an accurate shifted solve. The configured automatic policy selects a route
under memory and residual checks; explicit routes remain available for
controlled comparisons. See :doc:`solvers` for defaults and preconditioners,
and :doc:`differentiable_eigensolver` for mode tracking and sensitivities.

Nonlinear integration
---------------------

The nonlinear bracket uses FFTs and a pseudo-spectral convolution with the
two-thirds dealiasing rule. This controls quadratic aliasing; resolution,
conservation and transport-window checks remain necessary. Timestep
controllers enforce their configured stability policy, while fixed-step runs
require a suitable explicit timestep. See :doc:`numerics` for method and
controller parameters and :doc:`parallelization` for device layouts.

Differentiation
---------------

JAX differentiates geometry, operators and diagnostics. Certified eigenpair
sensitivities use implicit rules with residual, overlap and conditioning
checks. Nonlinear sensitivities differentiate a finite executed trajectory
with checkpointing; they depend on the initial state and diagnostic window.
Use finite differences and independent forward simulations to assess them.
See :doc:`nonlinear_autodiff` for the discrete adjoint and memory policy.

Precision and verification
--------------------------

Choose precision explicitly when comparing solvers or backends. Eigensolver
conditioning and long trajectories can require float64 even when a bounded
float32 test passes. Verify optimized kernels against the reference operator
and record numerical tolerances with any timing result. The test workflow is
in :doc:`testing`; physical comparison requirements are in :doc:`validation`.
