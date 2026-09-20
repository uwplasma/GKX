Solvers
=======

Time integration
----------------

The linear solver supports explicit Euler, RK2, and RK4 updates inside a JAX
``scan`` loop.  RK4 is used in the Cyclone harness.

Two options address stiff Hermite--Laguerre systems:

* ``method="imex"`` treats collision and hyper-diffusion damping implicitly
  while keeping drift and streaming terms explicit.
* ``method="implicit"`` applies backward Euler through SOLVAX FGMRES.  Its
  physical preconditioners include diagonal damping and an FFT/Hermite-line
  inverse of the additive diagonal-plus-streaming symbol.  Linked chains
  retain the twist-and-shift layout.

The same line inverse is available to shift-invert Arnoldi.  The default
``shift_preconditioner="auto"`` uses it for electrostatic models, selects the
Woodbury field correction directly for electromagnetic models, and retries a
residual-rejected electrostatic pair with field correction.  Explicit
``"hermite-line"`` and ``"field-corrected"`` choices remain available.

Complex FGMRES uses unitary Givens rotations and reports the physical residual
``||b - A x||``.  Users set its tolerance, restart length, iteration limit,
and preconditioner; a failed residual gate is never converted into a finite
success value.

When the matrix-free inner solve still stalls at high velocity resolution,
``method="sparse_shift_invert"`` assembles the sparse operator 64 columns at a
time and factors the complete shifted operator through the optional SciPy
backend.  It requires ``shift`` from a coarse solve or continuation point and
certifies every candidate with the original JAX operator.  The SOLVAX bridge
can reuse the same factors for the adjoint inverse in implicit eigenpair AD.
This path is eager and CPU-factorized; it is not the default for small or
target-free solves.

On a linked boundary the eigen routes solve on the modes the linked chains
couple.  Only the linked parallel derivative, its kz hypercollisions and the
chain end damping couple ``(ky, kx)`` modes, and they act on chain members; a
kx row outside GX's dealiased ``Nakx`` set therefore has no coupling to the
chains in either direction and contributes only undamped drift eigenvalues
next to a physical shift.  Seeds and every Krylov, power-iteration and inner
GMRES vector are projected onto the chain modes, and ``sparse_shift_invert``
assembles only their columns (2560 of 4096 unknowns on the Nx=8 linked Cyclone
pilot).  Returned eigenvectors keep the full state shape with exact zeros on
the other rows, and certification still applies the unprojected operator, so
any unexpected coupling fails the residual gate instead of being hidden.
Periodic boundaries and full-cover linked grids (for example Nx=1 at one ky)
are unchanged; a seed with no chain component raises ``ValueError``.

State intake on linked boundaries
---------------------------------

On a linked boundary the runtime zeroes the rows the chains never reach in
every state it did not build itself: a user ``initial_state`` passed to
``run_runtime_linear``, and a restart or init file named by ``init.init_file``.
This is the contract GX enforces by masking after ``restart_read``.  The
runtime's own initial conditions seed only the dealiased
``1 + 2 * ((Nx - 1) // 3)`` kx rows, which the chains cover, so nothing that
GKX builds is changed by the rule; periodic decks and full-cover linked grids
have no mask at all and trace exactly as before.

The mask is applied **at intake only**, not after each step.  The chains are
closed under the whole right-hand side: a state that is zero outside them has a
right-hand side that is exactly zero there too, in the linear operator and in
the nonlinear bracket, whose output the two-thirds mask restricts to exactly
the covered rows on a full grid.  Once the intake mask has run, the time loop
keeps those rows at exact zero without doing any work, so no per-step operation
is added to the runtime graph.

The rule is not cosmetic.  Off-chain content is decoupled and undamped, so no
growth-rate fit and no certified eigenpair reads it -- but
:func:`~gkx.operators.moments.distribution_free_energy` and the resolved ky/kx
spectra sum over every row, and a linear ky-selected grid carries an all-true
dealias mask, so the sums would count it forever.  On a nonlinear grid the ExB
bracket takes the *unmasked* state into real space, so off-chain content also
aliases back onto the chain rows and changes the physics.

:func:`~gkx.operators.linear.cache_builder.linked_chain_cover_mask` returns the
covered ``(ky, kx)`` modes for a deck, and
:func:`~gkx.operators.linear.cache_builder.mask_off_chain_rows` applies them to
a state; both cost ``O(Nz + Nky * Nkx)`` and return ``None`` and the state
unchanged when every row is reachable.  A caller that already holds a built
cache -- every nonlinear entry point does -- reads the same cover off it with
:func:`~gkx.operators.linear.linked.linked_cover_mask_from_cache` and applies it
with :func:`~gkx.operators.linear.linked.mask_supplied_state`, which costs
nothing beyond the cache it already built.

Supplied states below the runtime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The same rule holds one floor down, at the library entry points that accept a
state the caller built: ``gkx.prepare`` and the prepared object's ``run`` /
``run_arrays`` (including ``PreparedSimulation.solve(initial_state=...)``),
``integrate_nonlinear_explicit_diagnostics_state``, and the differentiable
objective :func:`~gkx.solvers_nonlinear_state_integration.nonlinear_heat_flux_window`.
It also holds at the raw drivers under them,
:func:`~gkx.solvers_nonlinear_state_integration.integrate_nonlinear` and
:func:`~gkx.solvers_nonlinear_state_integration.integrate_nonlinear_cached`
(queue row Q29), which is where the rule stops being a property of the doors a
user happens to use and becomes a property of the library.  Each **projects**
the state it is given onto the linked chain cover; none of them rejects it.

At the raw drivers the projection is applied once, in
``integrate_nonlinear_cached``, which ``integrate_nonlinear`` builds a cache for
and then delegates to, so both doors take the same view of the same array.  It
runs before the scan is built, so the compiled scan is the graph it was: the
added work is a single ``where`` whose instruction count does not grow with
``steps``.

Projection rather than rejection, for two reasons.  These are differentiable
entry points: the prepared object's ``run_arrays`` is the traced boundary, and
the objective's state is an array a design loop hands in, so an error condition
would have to read values a tracer does not have -- and on a concrete array it
would force a host synchronization on every call, on the route whose whole
purpose is a reused compiled graph.  And the runtime above them already
projects, so rejecting here would make two doors into the same solver disagree
about the same array.

What the projection does and does not change:

* it is a ``where`` against a mask fixed by the deck's topology, never a branch
  on the state's values, so it holds under ``jit`` and under reverse-mode AD;
* the cotangent of a supplied state is **exactly zero** on the off-chain rows
  and bitwise unchanged on the chain rows, which is the same statement as those
  rows carrying no physics forward;
* ``nonlinear_heat_flux_window`` keeps its documented gradient contract -- the
  saturated state is detached and only ``geom`` and ``params`` carry
  derivatives -- and the projection changes the trajectory the window starts
  on, not what is differentiated;
* periodic decks and full-cover linked grids get ``None`` and the same state
  object back, so they trace exactly as before.

The reason to project here is the bracket, not the diagnostics.  On a **full**
nonlinear grid the chain cover equals the two-thirds dealias mask, so the free
energy and its spectra already exclude the off-chain rows and are not inflated
by them; what does change is the physics.  Measured on a linked Cyclone grid at
Nx8/Ny8/Nz16 with a saturated-amplitude broadband state, one right-hand side's
chain rows move ``2.37e-02`` relative when off-chain content is present, and a
three-step trajectory from it is a different trajectory.  The effect is
quadratic in amplitude -- ``2.37e-04`` at ``max|G| = 1e-2`` -- so a linear-seed
state shows almost nothing and a saturated restart shows all of it.

Eigenpair certification
-----------------------

``KrylovConfig()`` and ``dominant_eigenpair`` default to ``method="adaptive"``,
the residual-certified eigensolve that runtime linear runs already use for
generic contracts.  No method returns a pair above its original-operator
residual gate
:math:`\lVert Av-\lambda v\rVert/\max(\lVert Av\rVert,|\lambda|\lVert v\rVert)`:

* ``adaptive``, ``shift_invert`` and ``sparse_shift_invert`` raise when their
  gates reject the pair;
* ``power``, ``propagator`` and ``arnoldi`` return a Rayleigh or Ritz pair
  without a convergence test of their own.  The pair is checked against
  ``certifiable_residual_tolerance(shift_outer_residual_tol, dtype)``, the same
  gate as shift-invert, and a failing pair raises ``RuntimeError`` with the
  residual and tolerance;
* ``certify=False`` (on ``KrylovConfig`` or ``dominant_eigenpair``) is the
  explicit opt-out for these three raw routes only.  The pair is returned and
  the status callback reports it as uncertified, with its residual.  It does
  not relax the other gates, and shift-invert seeds and fallbacks are
  unchanged;
* a zero or non-finite eigenvector has infinite residual, so a breakdown that
  returns ``(0, 0)`` cannot pass any gate.

Why ``adaptive`` is the default, and not shift-invert
-----------------------------------------------------

Queue row Q26 (2026-09-19) re-examined this choice against the rewritten §5.1
adoption gate, which adopts a reproducible cost reduction that loses no
accuracy even at 15--30%.  ``adaptive`` stays the default, and the reason is not
that shift-invert is merely slower:

* the shift is **not** the obstacle.  ``shift_source="propagator"`` derives a
  shift from a short propagator run, so the default path can choose one with no
  user input;
* the inner solve is.  On the shipped Cyclone deck
  (``examples/linear/axisymmetric/cyclone.toml``) at ``Nl=4, Nm=8``,
  ``KrylovConfig(method="shift_invert")`` leaves **48 of 48** inner FGMRES
  solves unconverged at maximum relative residual 34 against a 1e-4 inner
  tolerance, and the outer pair is rejected at residual 0.99 against its gate.
  That is not a budget shortfall: raising ``shift_maxiter`` from the default 50
  to 2000 -- 96,000 inner iterations instead of 2,880 -- leaves the residual at
  35.  It is not a precision effect either; the float64 run gives 48/48
  unconverged at residual 34 and an outer residual of 0.987;
* the route **fails closed**, as Q12 requires, so no user receives an
  uncertified pair from it.  On the same deck and rung ``adaptive`` certifies at
  residual 4.0e-15 against the 1e-9 float64 gate.

So shift-invert is not a candidate default until its inner solve converges on a
production chain.  Q7 (#236) and Q21 (#255) both studied that inner solve, but
on a research harness whose shift-invert is a different algorithm from this one:
it uses SOLVAX ``gcrot`` with subspace recycling, the ``pr3-cm`` structured
preconditioner, an unrestarted Arnoldi and a per-step original-operator residual
with early exit.  This module uses restarted FGMRES with no recycling, the
``hermite-line``/``field-corrected`` preconditioners, a fixed restart count and
no per-step residual.  Neither of Q21's two levers transfers as a result: an
inner-tolerance schedule cannot reduce a cost that no tolerance is setting when
every solve is budget-capped and none converges, and the exact block-Thomas +
Sherman--Morrison apply accelerates the dense z-local block of ``pr3-cm``, which
this module does not build.  Landing either one means landing ``pr3-cm`` here
first, which is a separate adoption.

API change (queue row Q12, 2026-09-13): ``KrylovConfig.method`` previously
defaulted to ``"propagator"`` and ``dominant_eigenpair(method=...)`` to
``"power"``.  Both raw routes returned unchecked pairs.  On the linked Cyclone
deck, a bare ``KrylovConfig()`` returned the wrong branch with relative
residual 0.98--1.00.  Code that relies on the raw routes must now name the
method, and must pass ``certify=False`` if it knowingly accepts an
unconverged pair.

Solver status on results
------------------------

Runtime results carry the status of the solve that produced them, so a number
never travels without its evidence (queue row Q15):

* ``RuntimeLinearResult.eigen_status`` (``EigenSolveStatus``) is set on Krylov
  runs.  ``residual`` is the original-operator relative residual above of the
  returned pair, ``tolerance`` the gate its route applied, ``certified``
  whether it passed and ``route`` the method that produced the pair; a
  shift-invert fallback names its fallback method.  ``inner`` summarizes the
  inner FGMRES solves of the shift-invert build that produced the pair
  (converged, unconverged and total solves, maximum relative residual,
  iterations, tolerance and preconditioner) and is ``None`` for other routes.
  ``dominant_eigenpair(..., return_status=True)`` returns it as a third
  element.  Every gate raises, so ``certified`` is ``False`` only for a raw
  route called with ``certify=False``.
* ``implicit_solve`` (``ImplicitSolveSummary``) on ``RuntimeLinearResult`` for
  ``method="implicit"`` and on ``RuntimeNonlinearResult`` for IMEX runs
  summarizes every GMRES solve of the run: the largest SOLVAX true relative
  residual ``||b - A x|| / ||b||``, the most iterations one solve used, the
  number of solves and the number that did not converge.
* ``summary()`` flattens both into scalar ``eigen_*`` and ``implicit_*`` keys,
  which are ``None`` where no such solve ran.

The time scans carry ``ImplicitSolveStats``, four scalars, through
``lax.scan`` and ``fori_loop``; no host callback runs inside a traced scan.

**Non-convergence fails closed at the host boundary.**  A traced step cannot
raise, so ``run_runtime_linear`` and ``run_runtime_nonlinear`` check the
carried stats after the scan returns and raise ``RuntimeError`` with the
unconverged count, the maximum relative residual and the maximum iterations,
as the eigen gates refuse an uncertified pair.  Neither returns a trajectory
with an unconverged implicit step.  The library integrators stay traceable and
do not raise: ``integrate_linear``, ``integrate_linear_diagnostics``,
``integrate_linear_from_config``, ``integrate_nonlinear``,
``integrate_nonlinear_from_config``, ``integrate_nonlinear_imex_cached``,
``integrate_nonlinear_imex_diagnostics``,
``integrate_nonlinear_explicit_diagnostics``, ``integrate_nonlinear_sheared``
and ``integrate_nonlinear_sheared_transport`` accept
``return_solve_stats=True`` and append the stats (``None`` for methods without
an implicit solve) for callers that decide for themselves; without it they
return what they returned before.  For
``integrate_nonlinear_sheared_transport`` the stats arrive as the returned
trace's ``solve_stats`` field, which stays ``None`` when they were not asked
for.  A caller that wants the runtime's refusal applies
:func:`~gkx.solvers_linear_implicit.require_converged_implicit_solves` to the
stats itself; that is the one gate, wherever it is called from.

The IMEX diagnostics route and the sheared IMEX route carry the stats through
their own scans (queue row Q29): the diagnostic carry grows one
``ImplicitSolveStats`` leaf, folded once per implicit solve -- three times per
step under ``sspx3`` -- and the sheared scans grow that leaf only on the
``method="imex"`` route, so the explicit sheared methods keep exactly the carry
they had.

**Saved summaries carry the status too.**  ``*.summary.json`` -- written by a
prefix run, by the ``--out`` JSON form and as the sidecar of a NetCDF bundle --
now records the same flat ``eigen_*`` and ``implicit_*`` keys the in-memory
result reports, with the same names and the same ``None`` where no such solve
ran.  Before this, a run read back from disk had no convergence channel at all,
so a reader could not tell a certified eigenpair or a converged implicit scan
from an unchecked one.  Each status type spells its own keys --
:func:`~gkx.solvers_linear_implicit.implicit_solve_payload` and
``gkx.solvers_linear_krylov.eigen_status_payload`` -- and both the result's
``summary()`` and the artifact writers read those, so the two cannot drift
apart.

**Cost.**  Callers that do not request the stats compile the graph they
compiled before, because XLA removes the unused carry.  Requesting them reads
SOLVAX's recomputed true residual, which XLA otherwise removes, so it costs one
operator application per implicit solve; the scalar folds add about twenty
instructions.  On a 4x4x8, Nl=2, Nm=4 implicit linear scan the optimized HLO
grows from 3834 to 4443 instructions (FFT 16 to 20); a fold of the iteration
count alone adds 19.  No jit static argument or compile key is added.

One nonlinear diagnostics graph
-------------------------------

There are two ways into an explicit nonlinear diagnostics run, and they are
one reference route (queue row Q18, plan section 5.3).  ``gkx.prepare`` keeps
a compiled scan and runs it through ``simulation.run``;
``integrate_nonlinear_explicit_diagnostics_state`` is the function entry point,
and the one ``run_runtime_nonlinear`` reaches on its fixed-window, chunked and
sharded routes.  **Both compile the same jitted graph and return bitwise
identical arrays for the same inputs**, in ``float32`` and under
``JAX_ENABLE_X64``.  Compare them with an exact equality, not a tolerance;
``tests/unit/nonlinear/test_nonlinear.py`` does, over every output array of
both entry points, for RK3 and RK4 and for fixed and adaptive steps.  Value and
reverse-mode gradient match bit for bit as well.

The one remaining difference between them is the state dtype.  A prepared
object freezes the dtype of the ``G0`` it was built with and casts every later
state to it; the function entry point takes the dtype of the ``G0`` it is
handed.  Feeding a promoted state to one and not the other -- for example by
scaling ``G0`` with a ``float64`` scalar under ``JAX_ENABLE_X64`` -- runs the
two at different precisions.  That is a precision change, not a route
difference.

**What this replaces.**  Until GKX 2.1.0 the function entry point ran its scan
outside ``jit``.  XLA then compiled the scan primitive on its own, with the
cache, parameter and policy arrays its body closes over as operands, and
optimized the pre-scan field solve and the first diagnostic as separate
modules.  The two routes agreed only to roundoff.  On a 100-step Cyclone gate
at 16x16x24, Nl=2, Nm=4, over RK3 and RK4 crossed with adaptive steps, an
implicit collision split and a fixed mode, none of the twelve cases was
bitwise in either precision.  The final state differed by 8.0e-8 to 2.1e-7
relative in ``float32`` and by 1.5e-21 to 1.6e-7 under x64; most diagnostics
stayed near 1e-7, the zonal potential diagnostics reached 2.1e-6, and the
growth-rate and frequency ratios 1.2e-3.  The turbulent-heating diagnostics,
which are a near-cancelling difference of two larger terms, reached 0.93
relative in ``float32`` and 1.41 under x64: for those the relative difference
between two roundoff-equivalent routes is unbounded, and only the terms they
are built from can be compared meaningfully.  Pinning that spread as a
tolerance was the rejected option.

**Cost.**  No user opts in and no step costs more.  On the shipped Cyclone deck
at 32x32x24, Nl=2, Nm=4, the graph the function entry point compiles goes from
the bare scan module to the prepared one: for RK3, ``copy`` 226 to 85,
``concatenate`` 44 to 38, ``transpose`` 64 to 66, ``fft`` 43 to 44, and the
bytes those copies and concatenates write 89,062,692 to 89,479,380 (+0.47%);
for RK4, ``copy`` 246 to 105 and 116,996,388 to 117,413,076 bytes.  The
captured graph does carry its cache, parameter and policy arrays as module
literals, 6,553,260 bytes at that resolution against 7,140 for the bare scan,
and that is the price of the contract.  It is repaid in process memory,
because one jit replaces the roughly sixty extra small modules the eager route
compiled around its scan: eight successive chunked calls at 16x16x24 peaked at
826 MB of resident memory instead of 1,200 MB.  The number of compilations per
call is unchanged at one.  ``profile_runtime_kernels.py nonlinear-step-hlo``
reports ``--route diagnostics`` and ``--route runtime`` as that one graph and
keeps the retired placement under ``--route eager-scan``.

Differentiable eigenmodes
-------------------------

``adaptive_propagator_eigenpair`` is the opt-in matrix-free path for linear and
quasilinear objectives.  It:

* estimates a stable RK4 step from broadband spectral probes;
* extracts several leading-growth candidates from a long-horizon propagator;
* certifies every pair against the original continuous operator;
* tracks a requested branch with right or biorthogonal overlap; and
* uses an implicit reverse rule for eigenvalue and eigenvector observables.

It avoids the ``O(n^2)`` dense matrix, but the explicit filter is still slow at
high Hermite resolution.  The collisional QI full-frequency ladder is closed;
the separate collisionless stress case remains unresolved.  Qualification,
cold timings, and the remaining implementation boundary are documented in
:doc:`differentiable_eigensolver`.

Optional damping
----------------

The linear operator supports:

* Lenard--Bernstein diagonal damping with rate ``nu``;
* velocity-space hyper-collisions controlled by ``nu_hyper_l``,
  ``nu_hyper_m``, ``nu_hyper_lm`` and their exponents; and
* smooth field-aligned end damping controlled by ``damp_ends_widthfrac`` and
  ``damp_ends_amp``.

All may be disabled or refined independently in resolution studies.

Performance caching
-------------------

``LinearCache`` stores geometry-dependent gyroaverages, drift coefficients,
linked-chain maps, and masks outside the time loop.  ``build_linear_cache``
constructs it, while the runtime integration entry points build and reuse it
automatically.

Growth-rate extraction
----------------------

For

.. math::

   \phi(t) \approx \exp[(\gamma - i\omega)t],

GKX fits :math:`\log|\phi|` and unwrapped phase against time.
``fit_growth_rate_auto`` scans for the most exponential window and is used by
the Cyclone harness when ``auto_window=True``.
