Differentiable matrix-free eigenmodes
=====================================

Status
------

GKX has an opt-in, matrix-free eigensolver for linear and quasilinear
observables.  It is accurate, differentiable, branch-aware, and has
``O(n m)`` storage for state size ``n`` and Krylov dimension ``m``.  It is
not yet the default solver: high velocity-space resolutions remain limited by
the cold cost of the explicit propagator filter.

This distinction is intentional.  The qualification records below establish
correctness and differentiation.  They do not claim that the current cold
wall time is competitive with mature preconditioned gyrokinetic eigensolvers.

Why the dense path stops scaling
--------------------------------

The validation path materializes the complex linear operator and computes all
eigenpairs.  Its matrix storage grows as ``O(n^2)``.  At the qualified QI rung
``n = 494,592``, a complex128 dense matrix alone would require about 3.6 TiB.
The matrix-free path retains only a small basis and applies the production
linear RHS directly.

The solver forms an RK4 approximation to ``exp(T A)``, builds an Arnoldi
subspace of that propagator, and ranks its leading candidates by the
continuous-operator Rayleigh quotient.  Every returned vector is certified
with

.. math::

   \frac{\lVert A v - \lambda v\rVert_2}
        {|\lambda|\lVert v\rVert_2} < \epsilon .

The timestep comes from broadband Arnoldi probes of the original operator.
If the inferred RK4 filter is unstable or the continuous residual misses the
tolerance, the solve fails closed.

Branch continuation
-------------------

Maximum growth is not a smooth branch label.  At a growth-rate crossing, a
plain ``argmax(Re(lambda))`` switches modes.  GKX can instead retain several
certified candidates and select by the phase-invariant overlap with the
previous right eigenvector, or by the biorthogonal overlap with its left
eigenvector.  The implementation also gates the overlap and complex spectral
gap.

The checked-in branch-crossing record scans a real ITG crossing, observes the
dense growth-order exchange, follows the requested subdominant branch, and
matches the dense right and left eigenpairs:

* :download:`branch-crossing qualification
  <_static/adaptive_propagator_branch_crossing_validation.json>`

Implicit reverse-mode differentiation
-------------------------------------

The iteration is not differentiated.  For a simple eigenpair with
``w^H v = 1``, the eigenvalue derivative is

.. math::

   d\lambda = w^H (dA) v .

Eigenvector observables use a bordered reduced-resolvent solve.  Reverse mode
therefore costs a primal eigenpair, an adjoint eigenvector, and one transposed
bordered solve; it does not tape thousands of propagator steps.  GKX rejects a
branch when ``||w|| ||v|| / |w^H v|`` exceeds the configured condition limit,
because an exceptional point does not have a trustworthy single-eigenvector
gradient.

The cold CPU value-and-gradient record at ``n = 6,144`` agrees with dense AD
to ``1.5e-10`` relative error and used 142.75 s versus 171.81 s for the dense
calculation:

* :download:`objective and gradient qualification
  <_static/adaptive_objective_gradient_cold_n6144_validation.json>`

Qualified physics and resolution
--------------------------------

The reduced dense-oracle matrix covers ITG, ETG, TEM, and KBM; electrostatic
and electromagnetic fields; periodic and linked layouts; circular, Miller,
QHS, and QI geometry.  Each value and directional gradient passes:

* :download:`physics-matrix qualification
  <_static/adaptive_objective_physics_matrix.json>`

The QI full-frequency ladder reaches ``(N_l,N_m)=(84,92)`` and
``n=494,592`` with continuous residuals below ``5.3e-13``:

* :download:`QI frequency-convergence qualification
  <_static/adaptive_propagator_convergence_qi_frequency_extension_validation.json>`

That same record exposes the open performance gate: its four cold GPU solves
took 917--1504 s and 7,905--8,935 RK4 steps per propagator application.  The
high-order Hermite streaming spectrum, not JAX compilation or dense storage,
is the dominant cost.

Usage
-----

.. code-block:: python

   import jax
   from gkx.objectives import (
       AdaptiveLinearEigensolverConfig,
       solver_objective_vector_from_geometry,
   )

   config = AdaptiveLinearEigensolverConfig(
       tolerance=1e-9,
       candidate_count=2,
   )

   def growth(boundary):
       geometry = build_flux_tube_geometry(boundary)
       observables = solver_objective_vector_from_geometry(
           geometry,
           n_laguerre=12,
           n_hermite=16,
           eigensolver="adaptive-propagator",
           adaptive_config=config,
       )
       return observables[0]

   value, gradient = jax.value_and_grad(growth)(boundary)

The geometry builder must be JAX-transformable.  During a future optimization
campaign, pass continuation vectors between nearby configurations and keep the
condition, overlap, gap, residual, and velocity-resolution gates active.

Reproduce the retained evidence
-------------------------------

.. code-block:: bash

   JAX_ENABLE_X64=1 python tools/campaigns/validate_adaptive_objective_physics_matrix.py
   JAX_ENABLE_X64=1 python tools/campaigns/validate_adaptive_objective_gradient.py
   JAX_ENABLE_X64=1 python tools/campaigns/validate_adaptive_branch_continuation.py
   JAX_ENABLE_X64=1 python tools/campaigns/validate_adaptive_propagator_convergence.py \
     --device qi --required-observable frequency \
     --resolution 72,80 --resolution 76,84 \
     --resolution 80,88 --resolution 84,92 \
     --output docs/_static/adaptive_propagator_convergence_qi_frequency_extension_validation.json

Next performance work
---------------------

The production blocker is a preconditioner for the oscillatory
Hermite-streaming spectrum.  The next candidate must beat the current cold
operator-application count while preserving the original-operator residual,
branch-crossing, QI frequency, and implicit-gradient gates.  In particular,
mixed precision or a transformed residual is not an acceptable substitute for
the continuous complex128 certificate.

Relevant precedents are the matrix-free Jacobi--Davidson implementation in
GENE, SLEPc's Krylov--Schur/Jacobi--Davidson methods, and timestepper Arnoldi.
GKX's evidence supports the same conclusion as the GENE literature: an
effective physics-aware preconditioner is required before this path should be
advertised as a universally fast cold eigensolver.
