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
``"hermite-line"``, ``"field-corrected"`` and ``"pr3-cm"`` choices remain
available; ``"pr3-cm"`` is the structured splitting described in
:doc:`numerics`, and is the one with host-side setup and structural
preconditions.

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

Queue rows Q26 (2026-09-19) and Q28 (2026-09-20) re-examined this choice against
the rewritten §5.1 adoption gate, which adopts a reproducible cost reduction
that loses no accuracy even at 15--30%.  ``adaptive`` stays the default.

Q26 recorded the shipped shift-invert route failing closed on the shipped
Cyclone deck (``examples/linear/axisymmetric/cyclone.toml``) at ``Nl=4, Nm=8``:
48 of 48 inner FGMRES solves unconverged, the outer pair rejected at residual
0.99, and neither the budget nor the precision responsible.  Q28 separated that
into three causes and fixed two of them:

* **the initial guess**, now removed.  The shifted FGMRES seeded itself with
  ``x0 = M^-1 b``.  Under right preconditioning the first Krylov vector *is*
  ``M^-1 b``, so a cycle from zero already minimizes over a space containing
  that point; the guess cannot help, and with a weak ``M`` it starts the solve
  behind ``x = 0``.  On the first right-hand side of that deck it started 25.4
  times behind for ``hermite-line``, 20.7 for ``field-corrected`` and 3.53 for
  ``damping``.  That factor is what the reported inner residual of 34 was.  The
  honest figure from zero is about 1 -- still a failure, for the next reason;
* **the preconditioner**, now optional.  ``hermite-line`` does not converge at
  that size: 600 unrestarted iterations leave the inner residual at 0.549 and
  it never reaches the 1e-4 tolerance, while ``pr3-cm`` reaches it in 252.  With
  ``shift_preconditioner="pr3-cm"`` the route **certifies** on that deck --
  residual 3.25e-07 at ``Nl=4, Nm=8`` and 1.67e-07 at ``Nl=8, Nm=24``, both
  agreeing with the ``adaptive`` control to eight significant figures in the
  growth rate -- where it previously failed closed;
* **the restart length**, unchanged and still a trap.  At the default
  ``shift_restart=20`` even ``pr3-cm`` stalls: 300 iterations as GMRES(20) x 15
  leave 0.177, while the same 252 iterations in one cycle converge.  A user
  selecting ``pr3-cm`` has to raise ``shift_restart`` and ``shift_maxiter`` with
  it.

So shift-invert now has a configuration that works, and it is still not the
default, because it is not cheaper.  Measured against a same-session
``adaptive`` control with every arm certified against the original operator, in
matvec-equivalents to a certified pair: about 57,000 against 33,916 at
``Nl=4, Nm=8`` and about 220,000 against 96,700 at ``Nl=8, Nm=24`` -- 1.7 to 2.3
times **more**, and more at every preconditioner apply cost observed over
eighteen interleaved timing rounds.  The gate adopts a cost reduction, and this
is a cost increase.  The route is improved; the default is unchanged.

The gap is entirely the preconditioner apply.  The inner-iteration counts alone,
13,996 and 24,188, are *below* the control's operator applications, so a free
apply would make shift-invert 2.4 and 4.0 times cheaper instead.  The route is
apply-bound, not iteration-bound.

Of Q21's two inner-solve levers, one is now in ``src/`` and one is not.  The
exact block-Thomas plus Sherman--Morrison solve of ``pr3-cm``'s z-local block is
here, and is the same preconditioner: identical inner-iteration counts,
identical certified residuals and applies agreeing to 7.4e-16 against the dense
control, with factors 2.60 times smaller at ``Nl=8, Nm=24``.  Its *apply* cost
changes sign with block size: the dense inverse was cheaper in seven of nine
interleaved rounds at ``Nl*Nm = 32`` and block-Thomas in nine of nine at 192, so
``shift_precond_block_solve="dense"`` stays reachable and is the better choice
on a small velocity grid.  The inexact-Krylov
tolerance schedule is not, but its blocker has changed.  Q26's reason was that
no tolerance was setting the cost -- every solve was budget-capped.  With
``pr3-cm`` the inner solves converge (0 of 96 unconverged, against 96 of 96 for
``hermite-line``), so a tolerance does set it; what is missing is a per-Arnoldi-
step original-operator residual to drive the schedule, which this module's
restarted loop does not compute.

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

One adjoint window graph
------------------------

:func:`~gkx.solvers_nonlinear_state_integration.nonlinear_heat_flux_window` is
the route a transport optimization takes, and through GKX 2.2.0 it
**recompiled on every call** (queue row Q30).  It ran its scans eagerly, and an
eager ``lax.scan`` or ``lax.cond`` is dispatched on the jaxpr its body was just
traced into; a jaxpr compares by identity, so a jaxpr rebuilt per call missed
every lowering cache below it.  On the shipped Cyclone deck at 16x16x12,
Nl=2, Nm=4, a six-step RK3 window differentiated with ``jax.value_and_grad``,
that was
thirteen XLA modules -- four scans and nine conds -- on the first call and on
every call after it, and 60 to 74 per cent of the wall time of each objective
evaluation.

The differentiated scan now runs as one ``jax.jit`` graph.  Everything that has
to be read on the host stays outside it: the linked cache build, whose
``jtwist`` is an integer read off the shear and refuses a traced geometry; the
quadrature weights; the chain-cover projection of the supplied state; and the
Hermitian projector's ``ky`` axis layout, which crosses the boundary as a
hashable signature rather than as a grid.  Every array is an **argument**, not
a captured constant, which is what lets the same executable serve the next
geometry an optimizer proposes instead of recompiling for it.  Measured on
XLA:CPU with jax 0.10.2, two runs of three timed calls each:

.. list-table::
   :header-rows: 1
   :widths: 22 24 54

   * - ``ky`` layout
     - compilations per call
     - wall per call
   * - two-sided
     - 13 -> **0**
     - 4.92-5.36 s -> 0.139-0.141 s (**37x**)
   * - half-spectrum
     - 13 -> **0**
     - 7.04-7.36 s -> 0.112-0.116 s (**63x**)

The first call is cheaper too, 11.0 s to 7.3 s and 12.4 s to 6.8 s, because one
graph replaces the thirteen and the small modules around them.

**Identity.**  The value and both adjoint components are bitwise unchanged
against the eager route in twenty of twenty-six gate configurations -- every
one that uses the default ``checkpoint=True`` except RK2 under x64 -- across
RK2/RK3/RK4, two-sided and half ``ky``, a window tail, the uncompressed FFT
path, exact Laguerre quadrature, and an electromagnetic KBM deck, in
``float32`` and under ``JAX_ENABLE_X64``.  The six that move do so by 1 to 11
``float32`` ulps and 1 to 3 double ulps, all in the value rather than in the
gradient except one.  They are roundoff, and they are structural rather than
incidental: a graph that can be reused across geometries has to take its grid
and its quadrature weights as operands, and on the eager route those were
concrete arrays whose products were folded on the host before the scan body was
traced.  The window's own measured ``float32`` roundoff against ``float64`` is
48 ulps (queue row Q14), four times the largest deviation here.

**What this does not change.**  The discrete adjoint, the checkpoint schedule,
the gradient contract, and the warning at the divergence knee are the ones GKX
2.1.0 shipped.  The mean over the window is still taken outside the graph,
because inside it XLA fuses the divide into the scan's accumulation: with the
divide inside, RK2 and the electromagnetic deck moved by one ``float32`` ulp
and RK4 by one double ulp, and with it outside those four are bitwise.

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
