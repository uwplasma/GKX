Differentiable matrix-free eigenmodes
=====================================

Scope
-----

GKX provides an opt-in eigensolver for linear and quasilinear objectives.  It
applies the production gyrokinetic right-hand side without forming its dense
matrix, retains competing modes near crossings, and differentiates accepted
simple eigenpairs with an implicit adjoint.  The path supports ITG, ETG, TEM,
and KBM models; electrostatic and electromagnetic fields; one or more kinetic
species; periodic and twist-and-shift domains; and analytic, Miller, and VMEC
geometry.

The eigenpair extraction and its reverse rule come from SOLVAX
(``propagator_eigenpairs``, ``adaptive_eigenpair``, ``eigenpair_reverse``; see
:doc:`solvax_defaults`), which installs from the released package.

The matrix-free storage is ``O(n m)`` for state size ``n`` and subspace size
``m``, versus ``O(n^2)`` for dense validation.  A complex128 matrix for the
largest tested QI truncation, ``n = 494,592``, would require about 3.6 TiB.

Algorithm and acceptance
------------------------

Cold discovery builds a restarted Arnoldi space of a stable RK4 polynomial of
the full operator.  A continuous-operator Rayleigh quotient recovers frequency,
and every accepted vector must satisfy

.. math::

   \frac{\lVert A v-\lambda v\rVert_2}
        {\max(\lVert A v\rVert_2,|\lambda|\lVert v\rVert_2)} < \epsilon .

That ratio is built from operator applications in the working dtype, so it
bottoms out a few machine epsilons above zero rather than at zero.  A requested
:math:`\epsilon` below that floor is not a strict gate but an unreachable one:
it certifies nothing while the restart budget burns on a pair that is already
as good as the precision allows.  Every residual gate is therefore raised to
``1e3`` times the epsilon of the dtype it measures
(``certifiable_residual_tolerance``): ``1.19e-4`` in complex64 and
``2.2e-13`` in complex128.  Every shipped default is above the complex128
floor, so complex128 gates pass through unchanged.  Solutions report the
residual actually achieved, so the accepted quality is visible rather than
implied by the requested tolerance.

Near a growth-rate crossing, candidate vectors may be selected by overlap with
the previous right mode or by biorthogonal overlap with its left mode.  The
overlap, complex spectral gap, residual, and eigenpair condition number
(``condition_limit``, default ``1e8``) are independent fail-closed gates.  Only
the residual gates carry a precision floor: the overlap and gap gates compare
normalized ``O(1)`` quantities, which resolve identically at either precision.

For a supplied target or continuation shift, ``dominant_eigenpair`` can instead
use right-preconditioned shift-invert Arnoldi.  The default ``"auto"`` policy
inverts the additive diagonal-plus-Hermite-streaming symbol with an FFT and
batched tridiagonal solve for electrostatic systems.  Electromagnetic systems
go directly to ``"field-corrected"``, which adds the exact low-moment field map
through a Woodbury capacitance solve.  If the cheaper electrostatic solve fails
the original-operator residual, auto retries it with the same field correction
before invoking a generic fallback.  Field columns are probed sequentially, so
setup memory does not grow with velocity resolution.

This follows the established use of the stiff parallel kinetic block in
gyrokinetic preconditioning [Merz12]_ and the sparse Laguerre--Hermite
hierarchy [MDL17]_.  The implementation is JAX-native and uses the same
matrix-free physics in primal and tangent calculations.

Cold discovery
--------------

The optional exponential path (``exponential_krylov_dim``) approximates each
action of :math:`\exp(TA)` in an inner Arnoldi space and performs the
leading-mode extraction in a second, smaller Arnoldi space.  This is the
standard matrix-free projection of a large matrix exponential [HL97]_; it
avoids the explicit RK4 stability limit without forming :math:`A` or
:math:`\exp(TA)`.  GKX always recomputes the Rayleigh value and residual with
the original continuous operator.

It is opt-in, not a resolution-independent default: at large hard truncations
the inner exponential space must grow with the streaming/mirror spectral
radius, and the original-residual gate rejects an under-resolved projection.
Exponential cold discovery does not replace continuation selection; the two
cannot be combined in one call.

Differentiation
---------------

Solver iterations are not unrolled through reverse mode.  For a simple
eigenpair normalized by ``w^H v = 1``,

.. math::

   d\lambda = w^H(dA)v .

Eigenvector-dependent observables use the corresponding bordered
reduced-resolvent solve.  Reverse mode therefore needs an adjoint eigenmode and
one sensitivity solve rather than an iteration tape.  GKX rejects exceptional
or poorly separated modes (``branch_gap_floor``, ``condition_limit``) because a
single-mode derivative is then not well defined.  A value-only call does not
compute the left mode; that cost is paid only when reverse mode requests the
custom VJP.  The sensitivity solve fails closed: a solve that misses its
acceptance returns NaN rather than an inaccurate gradient.

Sparse shift-invert
-------------------

``method="sparse_shift_invert"`` assembles the operator 64 matrix-free columns
at a time, discards entries below ``1e-14``, factors the complete shifted
streaming--mirror--field operator (no mirror closure or reduced field
approximation), and solves near the shift with shift-invert implicitly
restarted Arnoldi [LSY98]_.  Every candidate is certified with the unmodified
JAX operator, and the call raises when none passes.  SOLVAX applies the same LU
to the conjugate-transpose shifted system, so the left mode that
``eigenpair_reverse`` needs does not require a second factorization; the eager
native primal is not differentiated, while the JAX operator supplies
:math:`(dA)v` and the bordered sensitivity equation.

The route requires a supplied or coarse-grid ``shift``.  It is a cold
factor-and-solve fallback for continuation, not target-free branch discovery,
and it factors on the CPU.

Velocity resolution
-------------------

Residual convergence for each finite matrix is **not** velocity-space
convergence.  On a collisionless hard truncation the leading eigenvalue drifts
with ``(Nl, Nm)``, and explicit stability is set by the high-moment
streaming/mirror spectrum, so the matrix-free cold path stays opt-in and
universal cold-solve speed is not claimed.  A hard-truncation ladder is judged
only after the physical collisional velocity-space cutoff is resolved.

The reflectionless Hermite closure [Kanekar15]_
(``hermite_closure="reflectionless"``, see :doc:`numerics`) is a slab
phase-mixing sink and is not used as a substitute for that cutoff.  Toroidal
low-moment closures must represent parallel phase mixing together with trapping
and curvature resonance [MDL17]_, so an accurate slab sink is not presented as
a general stellarator closure.

The unit and integration tests cover dense parity, original-operator
residuals, continuation selection, implicit gradients, field correction,
periodic and linked layouts, and CPU JIT behavior.  Dense parity and residuals
are the acceptance criteria; timings are not.

Usage
-----

.. code-block:: python

   import jax
   from gkx import (
       AdaptiveLinearEigensolverConfig,
       solver_objective_vector_from_geometry,
   )

   settings = AdaptiveLinearEigensolverConfig(
       tolerance=1e-9,
       candidate_count=2,
       max_restarts=14,
       exponential_krylov_dim=96,
       exponential_horizon=5.0,
   )

   def objective(boundary):
       geometry = build_flux_tube_geometry(boundary)  # your geometry builder
       values = solver_objective_vector_from_geometry(
           geometry,
           n_laguerre=12,
           n_hermite=16,
           eigensolver="adaptive-propagator",
           adaptive_config=settings,
       )
       return values[-1]  # quasilinear transport objective

   value, gradient = jax.value_and_grad(objective)(initial_boundary)

Pass continuation data between nearby design points and keep the residual,
overlap, gap, conditioning, and velocity-resolution checks active throughout
an optimization campaign.

For a supplied continuation shift, use the fail-closed cold policy directly:

.. code-block:: python

   value, mode = dominant_eigenpair(
       seed,
       cache,
       linear_params,
       terms=linear_terms,
       method="shift_invert",
       shift=previous_value,
       shift_source="reference",
       shift_preconditioner="auto",
       shift_outer_residual_tol=1e-7,
   )
