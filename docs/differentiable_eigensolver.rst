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

The adaptive objective currently requires the differentiable eigenpair API in
SOLVAX pull request 65.  GKX CI tests the paired repositories at an immutable
SOLVAX commit; the feature becomes installable from released packages after
that API is merged and released.  Published SOLVAX already supports the
shift-invert and Hermite-line path.

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

Near a growth-rate crossing, candidate vectors may be selected by overlap with
the previous right mode or by biorthogonal overlap with its left mode.  The
overlap, complex spectral gap, residual, and eigenpair condition number are
independent fail-closed gates.

For a supplied target or continuation shift, ``dominant_eigenpair`` can instead
use right-preconditioned shift-invert Arnoldi.  ``"hermite-line"`` inverts the
additive diagonal-plus-Hermite-streaming symbol with an FFT and batched
tridiagonal solve.  ``"field-corrected"`` adds the exact low-moment field map
through a Woodbury capacitance solve.  It maps field columns sequentially and
retains one tall response factor, keeping setup memory bounded.

This follows the established use of the stiff parallel kinetic block in
gyrokinetic preconditioning [Merz12]_ and the sparse Laguerre--Hermite
hierarchy [MDL17]_.  The implementation is JAX-native and uses the same
matrix-free physics in primal and tangent calculations.

Cold discovery
--------------

The optional exponential path approximates each action of
:math:`\exp(TA)` in an inner Arnoldi space and performs the leading-mode
extraction in a second, smaller Arnoldi space.  This is the standard
matrix-free projection of a large matrix exponential [HL97]_; it avoids the
explicit RK4 stability limit without forming :math:`A` or
:math:`\exp(TA)`.  GKX always recomputes the Rayleigh value and residual with
the original continuous operator.

On the retained complex128 QI case (``Nl=4``, ``Nm=8``, ``n=1,536``; RTX
A4000), inner/outer dimensions 96/24, horizon 5, and 14 restarts reached
``3.1e-10`` residual in 61.32 s cold.  The stability-limited path took
110.12 s and stopped at ``3.2e-6``.  This is an opt-in qualified point, not a
resolution-independent default: at large hard truncations the inner
exponential space must grow with the streaming/mirror spectral radius, and
the original-residual gate rejects an under-resolved projection.

Differentiation
---------------

Solver iterations are not unrolled through reverse mode.  For a simple
eigenpair normalized by ``w^H v = 1``,

.. math::

   d\lambda = w^H(dA)v .

Eigenvector-dependent observables use the corresponding bordered
reduced-resolvent solve.  Reverse mode therefore needs an adjoint eigenmode and
one sensitivity solve rather than an iteration tape.  GKX rejects exceptional
or poorly separated modes because a single-mode derivative is then not
well-defined.  A value-only call does not compute the left mode; that cost is
paid only when reverse mode requests the custom VJP.

Measured boundary
-----------------

On the retained complex128 targeted test (RTX A4000), the Hermite line reduced
a shifted solve from more than 1,024 iterations with diagonal damping to 39 at
``n=768`` and 63 at ``n=1,536``.  At ``n=4,480`` the complete continued
eigenpair took 12.40 s cold and 11.37 s warm, versus 25.19 s for the
same-device dense full spectrum.  Its eigenvalue error was ``9.2e-12`` and its
full-operator residual ``1.8e-10``.  These are supplied-shift measurements,
not cold branch-discovery timings.

The largest hard-truncated QI matrices also produced residuals below
``6e-13``, but their frequencies continued to drift as velocity resolution
increased.  Residual convergence for each finite matrix is **not**
velocity-space convergence.  Moreover, the ``n=494,592`` cold solve took
1,504 s because explicit stability is controlled by the high-moment
streaming/mirror spectrum.  Consequently this path remains opt-in and no
full-frequency QI convergence or universally fast cold solve is claimed.

The production unit and integration tests cover dense parity, original-operator
residuals, continuation selection, implicit gradients, field correction,
periodic and linked layouts, and CPU JIT behavior.  GPU timing is hardware
evidence, not a numerical acceptance substitute.

Usage
-----

.. code-block:: python

   import jax
   from gkx.objectives import (
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
       geometry = build_flux_tube_geometry(boundary)
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
