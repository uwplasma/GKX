Differentiable matrix-free eigenmodes
=====================================

Status
------

GKX has an opt-in, matrix-free eigensolver for linear and quasilinear
observables.  It is accurate, differentiable, branch-aware, and has
``O(n m)`` storage for state size ``n`` and Krylov dimension ``m``.  It is
not yet the default solver: targeted and continued modes now have a fast
physics-aware shift-invert path, but high-resolution cold branch discovery
remains limited by the explicit propagator filter.

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

Physics-aware targeted solves
-----------------------------

For a supplied target or continuation shift, ``dominant_eigenpair`` can use
right-preconditioned FGMRES inside shift-invert Arnoldi.  The
``"hermite-line"`` preconditioner inverts

.. math::

   D(\ell,m,k_x,k_y) + S(k_z,m)

with one FFT and a batched tridiagonal solve.  Keeping :math:`D+S` additive is
essential; applying a point inverse and streaming inverse as a product is a
different operator and loses the high-Hermite scaling.  The linked variant
uses the same solve on complete twist-and-shift chains.

``"field-corrected"`` applies the Woodbury identity to the linear field map.
It supports electrostatic and electromagnetic fields, multiple kinetic
species, and periodic or linked layouts.  Field columns are mapped
sequentially, and only the solved response factor is retained, cutting the two
tall factors and batched-column peak of the earlier implementation.

This design combines three established ideas: retaining the stiff parallel
kinetic block in gyrokinetic eigenvalue preconditioning [Merz12]_, exploiting
the sparse Laguerre--Hermite streaming hierarchy [MDL17]_, and correcting a
kinetic inverse with a reduced moment/field model [Chen14]_.  GKX's contribution
is a JAX-native structured realization whose primal and tangent paths use the
same matrix-free operators.

On the retained QA targeted-solve qualification, the line inverse reduced a
representative complex128 shifted solve from more than 1,024 iterations with
diagonal damping to 39 at ``n=768`` and 63 at ``n=1,536``.  At ``n=4,480`` the
complete eigenpair took 12.40 s cold and 11.37 s warm on an RTX A4000, compared
with 25.19 s for the same-device dense full-spectrum solve.  The eigenvalue
error was ``9.2e-12`` and the continuous-operator residual ``1.8e-10``.  Cold
time includes JIT compilation after geometry/cache construction; the shift was
carried from the adjacent certified branch.

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

The remaining production blocker is the full-frequency QI cold solve.  At
extreme :math:`(N_\ell,N_m)`, its residual is concentrated in the high-moment
mirror and grad-B tail.  The sign-changing mirror coefficient has zero
field-line mean, so neither the Fourier/Hermite line nor a positive diagonal
envelope retains trapped-orbit transport.  Local mirror-only block solves,
velocity-collocation diagonals, defect polynomials, and low-moment coarse
corrections were rejected because they increased memory or failed to contract
the original residual.

The next research candidate must invert the coupled parallel-orbit principal
operator—streaming and mirror force together—then apply the existing field
capacitance correction.  It must beat the explicit cold operator count while
preserving the branch-crossing, QI frequency, original-operator residual, peak
memory, and implicit-gradient gates.  Mixed precision or a transformed
residual is not a substitute for the continuous complex128 certificate.
