Solvers
=======

Time integration
----------------

Linear methods run inside a JAX ``scan``:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - ``method``
     - update
   * - ``euler``, ``rk2``, ``rk4``, ``sspx3``
     - explicit
   * - ``imex``
     - first-order diagonal IMEX: collision and hyper-diffusion damping
       implicit, drift and streaming explicit
   * - ``imex2``
     - the same diagonal split as ARS(2,2,2), second order
   * - ``implicit``
     - backward Euler solved with SOLVAX GMRES.  Preconditioners: diagonal
       damping, or an FFT/Hermite-line inverse of the additive
       diagonal-plus-streaming symbol.  Linked chains keep the
       twist-and-shift layout.

``integrate_linear`` defaults to ``rk4``.  A runtime deck that sets ``dt`` gets
``rk2`` at that fixed step; a deck without ``dt`` gets ``rk4`` with the CFL
controller at ``cfl = 0.9`` (:class:`gkx.config.TimeConfig`).

Shift-invert preconditioners
----------------------------

Shift-invert Arnoldi solves each shifted system with complex FGMRES (unitary
Givens rotations), starting from zero.  It reports the physical residual
``||b - A x||``; tolerance, restart length, iteration limit and preconditioner
are user settings, and a failed residual gate is never turned into a finite
success value.

``shift_preconditioner`` selects the right preconditioner:

* ``"auto"`` (default) uses ``"hermite-line"`` for electrostatic models and
  ``"field-corrected"`` (Woodbury low-moment field correction) for
  electromagnetic ones, and retries a residual-rejected electrostatic pair with
  field correction;
* ``"hermite-line"`` and ``"field-corrected"`` can be named directly;
* ``"pr3-cm"`` is the structured Peaceman--Rachford splitting described in
  :doc:`numerics`.  It is the only choice with host-side setup and structural
  preconditions.

``pr3-cm``
~~~~~~~~~~

``pr3-cm`` splits the shifted operator into a streaming half, inverted by the
Hermite line solve, and a z-local half, a batch of ``(l, m)`` blocks, and runs
three double sweeps between them.  Setup costs ``Nl * Nm`` probes of the
z-local operator and one host factorization per shift; the factors enter the
jitted solve as an operand.

``shift_precond_block_solve`` chooses how the z-local block is inverted:

``"auto"`` (default)
   exact block-Thomas elimination in the Laguerre index plus one
   Sherman--Morrison field correction when the block is Laguerre-tridiagonal
   plus rank one; otherwise the dense batched inverse, with the reason recorded
   in ``EigenSolveStatus.inner["preconditioner_setup"]``.
``"block-thomas"``
   the exact solve, or an error when the structure does not hold.
``"dense"``
   the dense batched inverse.

The Schur elimination is SOLVAX ``block_thomas_factor_ops`` (hence
``solvax>=0.22.0``).  GKX keeps the Hermite-banded coupling action, an
unrolled forward substitution pinned bitwise to SOLVAX
``block_thomas_solve_ops`` on the same factors, and the Sherman--Morrison
correction.  Against the previous in-tree factors, full applies agree to
``7.1e-16``--``8.9e-16`` and factor storage is 1.43x--2.61x smaller over
``Nl*Nm = 36..768`` (``plan/log.md``, 2026-09-20).  The record supports no
speed claim.

The structural checks (z-locality, rank one, off-tridiagonal and Hermite-band
width) are floored at ``64 * eps`` of the probe precision: inert in float64,
``7.6e-6`` in float32.  A z-coupled block is refused, not sent to the dense
fallback, because it is not the operator the splitting describes.

Measured on the shipped Cyclone deck (``examples/linear/axisymmetric/cyclone.toml``)
at ``Nz=96``:

* ``pr3-cm`` reaches the ``1e-4`` inner tolerance in 252 unrestarted
  iterations; ``hermite-line`` leaves 0.549 after 600;
* with ``pr3-cm`` the shift-invert pair certifies (residual ``3.25e-07`` at
  ``Nl=4, Nm=8``, ``1.67e-07`` at ``Nl=8, Nm=24``) and agrees with
  ``adaptive`` to eight significant figures in the growth rate;
* at the default ``shift_restart=20`` even ``pr3-cm`` stalls (GMRES(20) x 15
  leaves 0.177).  Raise ``shift_restart`` and ``shift_maxiter`` together with
  it.

Linked boundaries
~~~~~~~~~~~~~~~~~

On a linked boundary the eigen routes solve on the modes the linked chains
couple.  Only the linked parallel derivative, its kz hypercollisions and the
chain end damping couple ``(ky, kx)`` modes, so a kx row outside the dealiased
``Nakx`` set has no coupling to the chains and contributes only undamped drift
eigenvalues next to a physical shift.  Seeds and every Krylov, power-iteration
and inner GMRES vector are projected onto the chain modes, and
``sparse_shift_invert`` assembles only their columns (2560 of 4096 unknowns on
the Nx=8 linked Cyclone pilot).  Returned eigenvectors keep the full state
shape with exact zeros on the other rows, and certification applies the
unprojected operator, so unexpected coupling fails the residual gate.  Periodic
boundaries and full-cover linked grids (for example Nx=1 at one ky) are
unchanged; a seed with no chain component raises ``ValueError``.

Sparse shift-invert
~~~~~~~~~~~~~~~~~~~

``method="sparse_shift_invert"`` assembles the sparse operator 64 columns at a
time and factors the complete shifted operator through the optional SciPy
backend.  It needs ``shift`` from a coarse solve or continuation point and
certifies every candidate with the original JAX operator.  The SOLVAX bridge
reuses the same factors for the adjoint inverse in implicit eigenpair AD.  It is
eager and CPU-factorized.

State intake on linked boundaries
---------------------------------

On a linked boundary the runtime zeroes the rows the chains never reach in
every state it did not build itself: a user ``initial_state`` passed to
``run_runtime_linear``, and a restart or init file named by ``init.init_file``.
This matches GX, which masks after ``restart_read``.  The runtime's own initial
conditions seed only the dealiased ``1 + 2 * ((Nx - 1) // 3)`` kx rows, which
the chains cover, so nothing GKX builds is changed; periodic decks and
full-cover linked grids have no mask.

The mask is applied **at intake only**.  The chains are closed under the whole
right-hand side: a state that is zero outside them has a right-hand side that
is exactly zero there, in the linear operator and in the nonlinear bracket,
whose output the two-thirds mask restricts to the covered rows on a full grid.
No per-step operation is added.

Why the mask is needed: off-chain content is decoupled and undamped, so no
growth-rate fit or certified eigenpair reads it, but
:func:`~gkx.operators.moments.distribution_free_energy` and the resolved ky/kx
spectra sum over every row, and a linear ky-selected grid carries an all-true
dealias mask, so the sums would count it forever.  On a nonlinear grid the ExB
bracket takes the *unmasked* state into real space, so off-chain content also
aliases back onto the chain rows and changes the physics.

:func:`~gkx.operators.linear.cache_builder.linked_chain_cover_mask` returns the
covered ``(ky, kx)`` modes for a deck, and
:func:`~gkx.operators.linear.cache_builder.mask_off_chain_rows` applies them to
a state; both cost ``O(Nz + Nky * Nkx)`` and return ``None`` and the state
unchanged when every row is reachable.  A caller holding a built cache (every
nonlinear entry point does) reads the same cover with
:func:`~gkx.operators.linear.linked.linked_cover_mask_from_cache` and applies it
with :func:`~gkx.operators.linear.linked.mask_supplied_state`.

Supplied states below the runtime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The same rule holds at the library entry points that accept a caller-built
state: ``gkx.prepare`` and the prepared object's ``run`` / ``run_arrays``
(including ``PreparedSimulation.solve(initial_state=...)``),
``integrate_nonlinear_explicit_diagnostics_state``, the differentiable
objective :func:`~gkx.solvers_nonlinear_state_integration.nonlinear_heat_flux_window`,
and the raw drivers
:func:`~gkx.solvers_nonlinear_state_integration.integrate_nonlinear` and
:func:`~gkx.solvers_nonlinear_state_integration.integrate_nonlinear_cached`.
Each **projects** the state onto the linked chain cover; none rejects it.  At
the raw drivers the projection runs once, in ``integrate_nonlinear_cached``
(``integrate_nonlinear`` builds a cache and delegates), before the scan is
built: one ``where`` whose instruction count does not grow with ``steps``.

Projection, not rejection: these entry points are traced, so an error
condition would have to read values a tracer does not have, and on a concrete
array it would force a host synchronization on every call.  The runtime above
them already projects, so rejecting here would make two doors into the same
solver disagree.

What the projection does and does not change:

* it is a ``where`` against a mask fixed by the deck's topology, never a branch
  on the state's values, so it holds under ``jit`` and reverse-mode AD;
* the cotangent of a supplied state is **exactly zero** on the off-chain rows
  and bitwise unchanged on the chain rows;
* ``nonlinear_heat_flux_window`` keeps its gradient contract (the saturated
  state is detached; only ``geom`` and ``params`` carry derivatives); the
  projection changes the trajectory the window starts on, not what is
  differentiated;
* periodic decks and full-cover linked grids get ``None`` and the same state
  object back.

On a **full** nonlinear grid the chain cover equals the two-thirds dealias
mask, so free energy and spectra already exclude off-chain rows; what the
projection protects is the bracket.  On a linked Cyclone grid at Nx8/Ny8/Nz16
with a saturated-amplitude broadband state, one right-hand side's chain rows
move ``2.37e-02`` relative when off-chain content is present.  The effect is
quadratic in amplitude (``2.37e-04`` at ``max|G| = 1e-2``), so a linear-seed
state shows almost nothing and a saturated restart shows all of it.

Eigenpair certification
-----------------------

``KrylovConfig()`` and ``dominant_eigenpair`` default to ``method="adaptive"``,
the residual-certified eigensolve runtime linear runs use for generic
contracts.  ``power_iters`` defaults to 40 in both.  No method returns a pair
above its original-operator residual gate
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
  not relax the other gates;
* a zero or non-finite eigenvector has infinite residual, so a breakdown that
  returns ``(0, 0)`` cannot pass any gate.

``KrylovConfig.method`` previously defaulted to ``"propagator"`` and
``dominant_eigenpair(method=...)`` to ``"power"``; both returned unchecked
pairs, and on the linked Cyclone deck a bare ``KrylovConfig()`` returned the
wrong branch with relative residual 0.98--1.00.  Code that relies on the raw
routes must now name the method, and pass ``certify=False`` if it knowingly
accepts an unconverged pair.

Why ``adaptive`` is the default, and not shift-invert
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Shift-invert with ``pr3-cm`` certifies on the shipped Cyclone deck but is not
cheaper.  Against a same-session ``adaptive`` control, every arm certified
against the original operator, in matvec-equivalents to a certified pair:

.. list-table::
   :header-rows: 1

   * - ``(Nz, Nl, Nm)``
     - ``adaptive``
     - best certified ``pr3-cm``
     - ratio
   * - ``(96, 4, 8)``
     - 33,916
     - about 57,000 (dense block solve)
     - 1.7x more
   * - ``(96, 8, 24)``
     - 96,700
     - about 220,000 (block-Thomas)
     - 2.3x more

Shift-invert costs more at every preconditioner apply cost observed over
eighteen interleaved timing rounds.  The gap is the preconditioner apply: the
inner-iteration counts alone, 13,996 and 24,188, are *below* the control's
operator applications, so the route is apply-bound, not iteration-bound.  An
inexact-Krylov tolerance schedule is not implemented; it needs a
per-Arnoldi-step original-operator residual, which the restarted loop does not
compute.

Solver status on results
------------------------

Runtime results carry the status of the solve that produced them:

* ``RuntimeLinearResult.eigen_status`` (``EigenSolveStatus``) is set on Krylov
  runs.  ``residual`` is the original-operator relative residual of the
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
  which are ``None`` where no such solve ran.  ``*.summary.json`` (written by a
  prefix run, by the ``--out`` JSON form and as the sidecar of a NetCDF bundle)
  records the same keys.  Both read
  :func:`~gkx.solvers_linear_implicit.implicit_solve_payload` and
  ``gkx.solvers_linear_krylov.eigen_status_payload``, so they cannot drift
  apart.

The time scans carry ``ImplicitSolveStats``, four scalars, through
``lax.scan`` and ``fori_loop``; no host callback runs inside a traced scan.
The IMEX diagnostics and sheared IMEX scans fold one ``ImplicitSolveStats``
leaf per implicit solve (three per step under ``sspx3``); the explicit sheared
methods keep their carry unchanged.

**Non-convergence fails closed at the host boundary.**  A traced step cannot
raise, so ``run_runtime_linear`` and ``run_runtime_nonlinear`` check the
carried stats after the scan returns and raise ``RuntimeError`` with the
unconverged count, the maximum relative residual and the maximum iterations.
Neither returns a trajectory with an unconverged implicit step.  The library
integrators stay traceable and do not raise: ``integrate_linear``,
``integrate_linear_diagnostics``, ``integrate_linear_from_config``,
``integrate_nonlinear``, ``integrate_nonlinear_from_config``,
``integrate_nonlinear_imex_cached``, ``integrate_nonlinear_imex_diagnostics``,
``integrate_nonlinear_explicit_diagnostics``, ``integrate_nonlinear_sheared``
and ``integrate_nonlinear_sheared_transport`` accept
``return_solve_stats=True`` and append the stats (``None`` for methods without
an implicit solve).  ``integrate_nonlinear_sheared_transport`` returns them as
the trace's ``solve_stats`` field.  A caller that wants the runtime's refusal
applies :func:`~gkx.solvers_linear_implicit.require_converged_implicit_solves`
to the stats.

**Cost.**  Callers that do not request the stats compile the same graph, because
XLA removes the unused carry.  Requesting them reads SOLVAX's recomputed true
residual, one operator application per implicit solve, plus about twenty
instructions of scalar folds.  On a 4x4x8, Nl=2, Nm=4 implicit linear scan the
optimized HLO grows from 3834 to 4443 instructions (FFT 16 to 20).  No jit
static argument or compile key is added.

One nonlinear diagnostics graph
-------------------------------

``gkx.prepare`` keeps a compiled scan and runs it through ``simulation.run``;
``integrate_nonlinear_explicit_diagnostics_state`` is the function entry point,
and the one ``run_runtime_nonlinear`` reaches on its fixed-window, chunked and
sharded routes.  **Both compile the same jitted graph and return bitwise
identical arrays for the same inputs**, in ``float32`` and under
``JAX_ENABLE_X64``.  Compare them with exact equality;
``tests/unit/nonlinear/test_nonlinear.py`` does, over every output array, for
RK3 and RK4 and for fixed and adaptive steps.  Value and reverse-mode gradient
match bit for bit.

The one remaining difference is the state dtype.  A prepared object freezes the
dtype of the ``G0`` it was built with and casts every later state to it; the
function entry point takes the dtype it is handed.  Scaling ``G0`` by a
``float64`` scalar under ``JAX_ENABLE_X64`` and feeding it to only one of them
runs the two at different precisions.

**Cost.**  Nothing to opt into, and no step costs more.  On the shipped Cyclone
deck at 32x32x24, Nl=2, Nm=4, against a scan compiled outside ``jit``: RK3
``copy`` ops fall from 226 to 85 and the bytes written by copies and
concatenates rise 0.47%; the graph carries its cache, parameter and policy
arrays as module literals (6,553,260 bytes).  One jit replaces about sixty
small modules, so eight successive chunked calls at 16x16x24 peak at 826 MB
resident instead of 1,200 MB.  ``profile_runtime_kernels.py
nonlinear-step-hlo`` reports ``--route diagnostics`` and ``--route runtime`` as
that one graph and keeps the out-of-jit placement under ``--route eager-scan``.

One adjoint window graph
------------------------

:func:`~gkx.solvers_nonlinear_state_integration.nonlinear_heat_flux_window`, the
route a transport optimization takes, runs its differentiated scan as one
``jax.jit`` graph that is compiled once and reused for matching shapes, dtypes,
topology and static options (GKX 2.3.0; earlier releases recompiled on every
call).  Host-side work stays outside the graph: the linked cache build (its
``jtwist`` is an integer read off the shear), the quadrature weights, the
chain-cover projection of the supplied state, and the Hermitian projector's
``ky`` layout, which crosses the boundary as a hashable signature.  Every array
is an **argument**, not a captured constant, so the same executable serves the
next geometry an optimizer proposes.

Shipped Cyclone deck at 16x16x12, Nl=2, Nm=4, six-step RK3 window
differentiated with ``jax.value_and_grad``, XLA:CPU, jax 0.10.2, two runs of
three timed calls:

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

The first call also falls, 11.0 s to 7.3 s and 12.4 s to 6.8 s.

**Identity.**  Value and both adjoint components are bitwise unchanged against
the per-call route in 20 of 26 gate configurations (every one using the default
``checkpoint=True`` except RK2 under x64), across RK2/RK3/RK4, two-sided and
half ``ky``, a window tail, the uncompressed FFT path, exact Laguerre
quadrature and an electromagnetic KBM deck, in ``float32`` and x64.  The other
six move by 1 to 11 ``float32`` ulps or 1 to 3 double ulps, inside the window's
measured ``float32``-vs-``float64`` roundoff of 48 ulps.  They follow from
taking the grid and quadrature weights as operands.  The window mean is taken
outside the graph: inside it XLA fuses the divide into the scan's
accumulation, and four configurations that are bitwise with the divide outside
move by one ulp.

The discrete adjoint, checkpoint schedule, gradient contract and the warning at
the divergence knee are described in :doc:`nonlinear_autodiff`.

Differentiable eigenmodes
-------------------------

``adaptive_propagator_eigenpair`` is the opt-in matrix-free path for linear and
quasilinear objectives.  It:

* estimates a stable RK4 step from broadband spectral probes;
* extracts several leading-growth candidates from a long-horizon propagator;
* certifies every pair against the original continuous operator;
* tracks a requested branch with right or biorthogonal overlap; and
* uses an implicit reverse rule for eigenvalue and eigenvector observables.

It avoids the ``O(n^2)`` dense matrix, but the explicit filter is slow at high
Hermite resolution.  Details are in :doc:`differentiable_eigensolver`.

Optional damping
----------------

The linear operator supports:

* Lenard--Bernstein diagonal damping with rate ``nu``;
* velocity-space hyper-collisions controlled by ``nu_hyper_l``,
  ``nu_hyper_m``, ``nu_hyper_m_const``, ``nu_hyper_lm`` and their exponents.
  The Laguerre channels reach the distribution only through the
  constant-coefficient branch, so they need ``hypercollisions_const``; see
  :ref:`velocity-regularization`; and
* smooth field-aligned end damping controlled by ``damp_ends_widthfrac`` and
  ``damp_ends_amp``.

Each can be disabled or refined independently in resolution studies.

Performance caching
-------------------

``LinearCache`` stores geometry-dependent gyroaverages, drift coefficients,
linked-chain maps, and masks outside the time loop.  ``build_linear_cache``
constructs it; the runtime integration entry points build and reuse it
automatically.

Growth-rate extraction
----------------------

For

.. math::

   \phi(t) \approx \exp[(\gamma - i\omega)t],

GKX fits :math:`\log|\phi|` and unwrapped phase against time.
``fit_growth_rate_auto`` scans for the most exponential window and is used by
the Cyclone harness when ``auto_window=True``.
