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
unchanged when every row is reachable.  Library entry points below the runtime
(``gkx.prepare``, the nonlinear ``simulation.run(initial_state)`` route) do not
apply it; a state handed to those is taken as given.

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
``integrate_nonlinear_from_config`` and ``integrate_nonlinear_imex_cached``
accept ``return_solve_stats=True`` and append the stats (``None`` for methods
without an implicit solve) for callers that decide for themselves; without it
they return what they returned before.  The IMEX diagnostics scan and the
sheared IMEX route do not carry the stats yet.

**Cost.**  Callers that do not request the stats compile the graph they
compiled before, because XLA removes the unused carry.  Requesting them reads
SOLVAX's recomputed true residual, which XLA otherwise removes, so it costs one
operator application per implicit solve; the scalar folds add about twenty
instructions.  On a 4x4x8, Nl=2, Nm=4 implicit linear scan the optimized HLO
grows from 3834 to 4443 instructions (FFT 16 to 20); a fold of the iteration
count alone adds 19.  No jit static argument or compile key is added.

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
