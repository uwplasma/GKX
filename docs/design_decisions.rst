Design Decisions
================

Why GKX is built the way it is. Each choice has a cost; the cost is stated
next to it. The measurements behind these choices are in :doc:`algorithms`.

Flux tube
---------

GKX solves the gyrokinetic equation in a thin tube that follows one field
line. The tube is long along the field and a few gyroradii across, so it
resolves the turbulence scale at a small fraction of the cost of a full
surface or volume. The price is that profile variation across the tube is
neglected (local limit) and the parallel ends need a boundary condition: GKX
uses twist-and-shift linking, solved only on the modes each chain couples.
For stellarators, a set of tubes at different ``alpha`` samples a surface.

Hermite-Laguerre velocity space
-------------------------------

The distribution function is expanded in Hermite polynomials in
:math:`v_\parallel` and Laguerre polynomials in :math:`\mu`. Velocity space
becomes two spectral indices, the gyroaverage becomes a sum, and low-order
moments are the fluid quantities, so a coarse truncation is a meaningful model
rather than an under-resolved grid. The whole state is one array, which suits
JAX. The cost: a truncated Hermite ladder reflects free energy back from its
last moment (recurrence), so GKX damps the highest moments with
hypercollisions by default and says so. The same representation is used by
GX, which makes GX the closest parity reference.

JAX
---

Writing the solver in JAX gives one code path for CPU and GPU and exact
derivatives of the discretized equations, including the equilibrium when
VMEX supplies it. Reverse mode through an iterative solve is replaced by
implicit rules (eigenpairs) and checkpointed adjoints (time windows), each with
its own gate. The costs: compile time on the first call (cached afterwards),
static array shapes, and ``float32`` graphs that are not always bit-for-bit
reproducible.

Explicit and implicit time stepping
-----------------------------------

The nonlinear step is explicit Runge-Kutta with a CFL controller: the
:math:`E\times B` bracket changes every step and an implicit nonlinear solve
would dominate the cost. Stiff linear damping (collisions, hyperdiffusion) can
be taken implicitly with IMEX, and linear initial-value runs can use
backward Euler with GMRES. For linear growth rates the eigensolvers are
usually faster and more precise than time stepping.

``k_y`` layouts
---------------

The fields are real, so the negative-``k_y`` half of the spectrum is the
complex conjugate of the positive half. The default ``full`` layout stores
both halves; ``half`` stores only :math:`k_y \ge 0`. On CPU ``half`` runs an
RK3 step in 0.54--0.60 of the time, but the checkpointed heat-flux gradient
takes 1.28 times as long, so ``full`` stays the default. One module, :mod:`gkx.core_ky_layout`, owns every conversion and
the row weights used in reductions, so no other code writes a Hermitian
completion by hand.

Collisions
----------

Lenard-Bernstein/Dougherty is the default: cheap, diagonal in the Hermite
index up to conserving corrections. Sugama and linearized Coulomb operators
are available when the collision physics matters, each checked against its
published closed form, conservation laws and H-theorem. They are generated
for like-species collisions; a multispecies Coulomb request is refused rather
than extrapolated.

Certify, do not trust
---------------------

Every returned eigenpair is checked against the unprojected operator, and a
failed check is an error, not a number. Under-resolved or unsaturated runs
warn. Published numbers are tied to artifacts and recomputed in CI through
the evidence ledger (:doc:`verification_matrix`).
