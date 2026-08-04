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

The differentiable eigenpair API this path uses ships in SOLVAX 0.12.0, so the
feature installs from released packages; ``pyproject.toml`` requires that
version.  GKX CI no longer pins a SOLVAX commit.

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
use right-preconditioned shift-invert Arnoldi.  The default ``"auto"`` policy
inverts the additive diagonal-plus-Hermite-streaming symbol with an FFT and
batched tridiagonal solve for electrostatic systems.  Electromagnetic systems
go directly to ``"field-corrected"``, which adds the exact low-moment field map
through a Woodbury capacitance solve.  If the cheaper electrostatic solve fails
the original-operator residual, auto retries it with the same field correction
before invoking a generic fallback.  Field columns are mapped sequentially and
only one tall response factor is retained, keeping setup memory bounded.

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

On the reduced ``n=1,536`` QI eigenpair, the line-first cold solve took 9.64 s
with ``2.6e-11`` residual, versus 11.72 s for eager field correction.  Fresh
processes on an 18-core Xeon W-2295 took 12.28 s and 13.08 s respectively.
Reduced ITG, ETG, finite-electron-mass TEM, Miller, KBM-Apar, and
KBM-Apar-Bpar gates all recovered their dense target with residual at most
``2.3e-10``; auto chose the exact field path for both KBM cases.  Timings are
descriptive hardware measurements, while dense parity and residuals are the
acceptance criteria.

The collisional QI runtime contract was also followed past the physical
velocity-space cutoff.  A bounded sparse qualification built 64 matrix-free
operator columns at a time, discarded entries below ``1e-14``, solved near the
continued branch with shift-invert implicitly restarted Arnoldi [LSY98]_, and
certified the result with the unmodified operator:

.. list-table:: Collisional QI full-frequency ladder
   :header-rows: 1

   * - ``(Nl,Nm)``
     - state size
     - growth rate
     - frequency
     - residual
   * - ``(8,16)``
     - 6,144
     - -0.0473847353
     - -0.0010064988
     - ``8.5e-14``
   * - ``(10,20)``
     - 9,600
     - -0.0472648235
     - -0.0010649098
     - ``1.2e-13``
   * - ``(12,24)``
     - 13,824
     - -0.0472192892
     - -0.0010702563
     - ``2.4e-13``
   * - ``(14,28)``
     - 18,816
     - -0.0472042045
     - -0.0010606607
     - ``2.3e-13``

The final two relative frequency changes, 0.50 and 0.90 percent, satisfy the
predeclared two-consecutive-rung 1 percent gate.  The largest matrix retained
2.29 million nonzeros and took 10.10 s to assemble plus 420.18 s for the
native sparse eigensolve on the qualification host.

The same algorithm is the opt-in ``method="sparse_shift_invert"`` production
fallback.  It factors the complete shifted streaming--mirror--field operator;
there is no mirror closure or reduced field approximation in that solve.  At
``n=6,144``, four candidates took 3.09 s to assemble, 38.05 s to factor with
the fill-reducing default ordering, and 1.64 s for Arnoldi, with ``8.5e-14``
original-operator residual.  The LU contained 9.29 million nonzeros.  A tested
minimum-degree ordering was still factoring after 225 s and was rejected.

One LU also applies the conjugate-transpose shifted inverse.  The four-candidate
left solve reused it in 1.81 s with ``2.2e-13`` residual, so the left mode
needed by ``eigenpair_reverse`` does not require another 38.05 s factorization.
This is the standard implicit derivative posture: the eager native primal is
not differentiated, while the original JAX operator supplies :math:`(dA)v`
and the bordered sensitivity equation.  An unshifted rightmost sparse solve
exceeded 100 s at only ``n=1,536`` and was stopped.  The production path
consequently requires a supplied or coarse-grid shift; it is a fast cold
factor-and-solve fallback, not a claim of target-free branch discovery.

The distinct collisionless hard-truncation stress test still drifted with
velocity resolution.  Its ``n=494,592`` solve took 1,504 s because explicit
stability is controlled by the high-moment streaming/mirror spectrum.
Residual convergence for each finite matrix is **not** velocity-space
convergence.  The matrix-free cold path therefore remains opt-in and universal
cold-solve speed is not claimed.

An outgoing slab Hermite-tail closure [Kanekar15]_ was also screened rather
than promoted.  On the collisionless QI operator its frequency changed by
10.7 percent between ``(Nl,Nm)=(4,8)`` and ``(6,12)``, and neither structured
inverse certified ``(8,16)`` despite much larger inner spaces.  This is
consistent with the Laguerre--Hermite analysis [MDL17]_: toroidal low-moment
closures must represent parallel phase mixing together with trapping and
curvature resonance.  Production hard-truncation ladders are therefore judged
only after the physical collisional cutoff is resolved; an accurate slab sink
is not presented as a general stellarator closure.

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
