Numerics
========

Spectral discretization
-----------------------

Perpendicular spatial coordinates are discretized with Fourier modes on a
uniform grid in :math:`x` and :math:`y`, while the parallel coordinate is
resolved in real space along the field line. The velocity space uses a
Hermite-Laguerre basis. The resulting data layout for a single species is

``(N_l, N_m, N_y, N_x, N_z)``.

Algorithm mapping (numerics → code)
-----------------------------------

The core numerical algorithms and their implementation entry points are:

- **Hermite–Laguerre pseudo-spectral expansion**:
  :mod:`gkx.core_velocity`.
- **Gyroaverage / polarization**:
  :func:`gkx.core_velocity.J_l_all`,
  :func:`gkx.operators.linear.moments.quasineutrality_phi`.
- **Centered periodic derivative in z**:
  :func:`gkx.operators.linear.moments.grad_z_periodic`.
- **Hermite ladder streaming**:
  :func:`gkx.operators.linear.moments.streaming_term`.
- **Curvature / grad-B / mirror couplings**:
  :func:`gkx.operators.linear.rhs.linear_rhs_cached`,
  :func:`gkx.geometry.SAlphaGeometry.drift_components`,
  :func:`gkx.geometry.SAlphaGeometry.bgrad`.
- **Diamagnetic drive**:
  :func:`gkx.operators.linear.moments.diamagnetic_drive_coeffs`.
- **Time integration (explicit RK, IMEX)**:
  :func:`gkx.solvers_linear_integrators.integrate_linear`.
- **CFL-controlled RK4 (adaptive step control, streaming diagnostics)**:
  :func:`gkx.integrate_linear_explicit`
  (implemented in :func:`gkx.solvers_time_explicit.integrate_linear_explicit`).
- **Config-driven runner**:
  :func:`gkx.solvers_time_runners.integrate_linear_from_config`.
- **Implicit solve (Backward Euler + GMRES)**:
  :func:`gkx.solvers_linear_integrators.integrate_linear`.
- **Structured Hermite-line solves and bounded-memory Jacobians**:
  `SOLVAX <https://github.com/uwplasma/SOLVAX>`_ provides the reusable
  tridiagonal and autodiff primitives; GKX retains physical layout,
  coefficients, tolerances, and acceptance policy.
- **Nonlinear IMEX (implicit linear + explicit nonlinear)**:
  :func:`gkx.solvers_nonlinear_state_integration.integrate_nonlinear`.

JAX execution model
-------------------

The implementation leverages the following JAX primitives:

- **JIT compilation**: ``jax.jit`` is used in
  :func:`gkx.solvers_linear_integrators.integrate_linear` to stage time-stepping
  kernels.
- **Loop fusion**: ``jax.lax.scan`` drives the time integration loop.
- **FFT grids**: ``jax.numpy.fft.fftfreq`` is used in
  :func:`gkx.core_grid.build_spectral_grid`.
- **Sparse Krylov solver**: ``solvax.gmres`` is used for implicit linear and
  nonlinear IMEX time steps through one shared GKX policy adapter.
  Nonlinear IMEX reverse mode wraps the tolerance-controlled solve with
  ``solvax.linear_solve``. Its implicit-function VJP solves the transposed
  linear system instead of differentiating dynamic GMRES iterations; plain and
  checkpointed two-step trajectories agree with centered finite differences.
  A separate gate rebuilds the gyrokinetic cache and matrix-free operator from
  a traced :math:`R/L_{T_i}` and verifies that the VJP includes both right-hand
  side and operator dependence.
  Shift-invert eigenmode extraction temporarily retains the prior JAX GMRES
  route because its branch-continuity gate has not passed with the replacement.
- **Backend-aware Hermite line solve**: ``solvax.tridiagonal_solve`` uses a
  deterministic Thomas recurrence on CPU and the fused JAX/vendor path on
  accelerators. GKX moves only the Hermite system axis; all remaining
  dimensions are independent line-solve columns.
- **Memory-bounded sensitivities**: ``solvax.chunked_jacfwd`` underlies the
  geometry gradient report when ``jacobian_chunk_size`` is set. Chunking
  changes batching and peak memory, not the mathematical JVP columns. Without
  a chunk request, ``jacobian_mode="auto"`` uses forward mode for few controls
  and reverse mode for few observables; the resolved mode is recorded and
  checked against finite differences.
- **Stencil operations**: ``jax.numpy.roll`` and ``jax.numpy.pad`` implement
  the centered ``z`` derivative and Hermite/Laguerre ladder couplings in
  :func:`gkx.operators.linear.moments.grad_z_periodic`,
  :func:`gkx.operators.linear.moments.streaming_term`,
  :func:`gkx.operators.linear.moments.apply_hermite_v`,
  :func:`gkx.operators.linear.moments.apply_laguerre_x`.

These links are clickable in the HTML docs via the ``viewcode`` extension.

Structured solver dependency contract
-------------------------------------

GKX requires ``solvax>=0.12.0``; ``pyproject.toml`` is the only place that
floor is declared. Version 0.12.0 is the first release that exports every
SOLVAX name GKX imports: the eigenpair, propagator, and sparse-operator
interfaces (``adaptive_eigenpair``, ``eigenpair_reverse``,
``estimate_rk4_timestep``, ``exponential_eigenpairs``,
``propagator_eigenpairs``, ``sparse_eigenpairs``, ``sparse_operator_matrix``)
first ship there, while the Krylov and structured-solve interfaces (``gmres``,
``linear_solve``, ``tridiagonal_solve``, ``chunked_jacfwd``,
``SpluFactorization``) are older. CI installs the newest released SOLVAX, so
it tests the latest release rather than the floor. Generic numerical
algebra lives in SOLVAX; gyrokinetic state layout, linked-boundary assembly,
preconditioner coefficients, eigenbranch tracking, transport windows, and
physics gates remain in GKX.

The admitted migration covers the Hermite-line tridiagonal solve,
memory-chunked geometry Jacobians, and implicit linear/nonlinear time-step
GMRES. Implicit time stepping now exposes one FGMRES algorithm with explicit
tolerance, restart, iteration-limit, and physical-preconditioner controls;
the obsolete backend-name selector has been removed.

Shift-invert remains explicitly excluded. In its current streaming test, both
inner solvers stagnate above the requested tolerance and their small solution
difference changes the selected outer eigenbranch. The prior JAX route remains
in that one call site until preconditioning/recycling improvements pass inner
residual, eigenpair residual, eigenvalue, and eigenvector-overlap gates.

Time integration algorithms
---------------------------

The linear solver supports:

- **Forward Euler** (``method="euler"``) and **RK2/RK4** explicit schemes for
  non-stiff runs.
- **reference-compatible RK4 with CFL step control**
  (``integrate_linear_explicit``). The timestep is recomputed from the linear
  max-frequency estimate using the benchmark-locked CFL rule, and growth rates
  are extracted from the midplane ``phi`` ratio using the same diagnostic
  convention as the tracked comparison data.
- **IMEX (semi-implicit)** where the collisional/hyper-diffusion terms are
  treated implicitly and the remaining terms explicitly.
- **Backward Euler + GMRES** in ``method="implicit"`` for stiff scans, with a
  diagonal preconditioner that includes damping and drift/mirror diagonals.
- **IMEX (implicit linear operator + explicit nonlinear term)** in
  ``method="imex"`` for nonlinear runs, using the same GMRES-based linear
  solve and preconditioner. Reverse derivatives use the converged-system
  implicit derivative, so the gradient does not depend on the number of Krylov
  iterations except through primal/transpose solve accuracy.

These are all implemented in :func:`gkx.solvers_linear_integrators.integrate_linear` and
share the cached operator data assembled by
:func:`gkx.operators.linear.cache_builder.build_linear_cache`.

.. _first-run-defaults:

What a first run gets
---------------------

Queue row Q26 (2026-09-19) audited the defaults that decide whether a run
started from a shipped deck, or from a deck that omits a section, is fast and
accurate without expert flags. The results below are measurements, not
intentions; the host was heavily contended throughout (1-minute load 48--173
under another user's job), so every statement rests on load-independent
evidence -- residuals, certification gates, iteration counts, growth rates and
dtypes -- and no wall time is quoted as a result.

**Eigen route.** ``adaptive``, unchanged. See
"Why ``adaptive`` is the default, and not shift-invert" in :doc:`solvers`:
``shift_invert`` leaves 48 of 48 inner solves unconverged
and is rejected at outer residual 0.99 on the shipped Cyclone deck, in float32
and float64 alike.

**Precision.** ``complex64``, unchanged, and correct: on
``examples/linear/axisymmetric/cyclone.toml`` at its own resolution the float32
run returns :math:`\gamma = 0.09309106`, :math:`\omega = 0.28203276` against the
float64 :math:`0.09309117`, :math:`0.28203273` -- agreement to 1.2e-6 relative.
What changed is that ``JAX_ENABLE_X64=true`` now reaches the run at all; see
:doc:`inputs`. Float64 costs about 17% more resident memory on that deck
(0.90 -> 1.05 GiB) and buys the float64 certification gate: residual 1.07e-14
against 1e-9, where float32 reports 5.55e-6 against 1.19e-4.

**Time integrator and step size.** A deck that omits ``[time]`` took the
``TimeConfig`` dataclass defaults, ``method="rk2"``, ``dt=0.1``,
``fixed_dt=True``, and on the Cyclone geometry that **overflowed**: both rk2
and rk4 raise ``FloatingPointError`` at ``dt=0.1`` held fixed, because the
CFL-stable step there is 0.0128. Q26 diagnosed that failure on every linear
path -- the fixed-step CFL hint previously skipped ``solver="explicit_time"``,
the one linear path that advances a fixed step explicitly, so that path
overflowed with nothing said -- but left the defaults alone. Queue row Q28
changed them; see :ref:`q28-defaults` below.

Set ``dt`` for your case, or set ``fixed_dt = false`` and let the CFL controller
choose it. ``cfl_fac`` resolves to 1.73 for rk3/sspx3, 2.82 for rk4 and 1.0
otherwise, so rk4 takes a 2.82x longer stable step for twice the work per step.

**Resolution.** ``Nl`` and ``Nm`` omitted from ``[run]`` fell back to 24 and 12.
The Cyclone deck's own comment records that pair as reporting :math:`\gamma`
about 4.5% low for that case, which Q26 reproduced: 0.08893 at 24/12
against 0.09309 at the deck's 16/48. Q28 changed that fallback; see
:ref:`q28-defaults`. Always set ``Nl`` and ``Nm`` regardless.

**Unchanged, and measured or inspected to be right.** Two-thirds dealiasing on
for nonlinear runs (``nonlinear_dealias = true``); ``dealias_kz = false``;
``collisions`` and ``hypercollisions`` enabled as terms with per-species
``nu = 0``, so a deck opts into collisionality rather than inheriting one;
``diagnostics_stride = sample_stride = 1``; nonlinear chunking at
``min(steps, 128)``; restart cadence from ``nsave = 10000``; the persistent
compilation cache on with ``min_compile_time_secs = 0``, which is what makes a
second run of the same deck skip compilation.

**The CPU FFT thread pool stays on.** GKX does not set
``--xla_cpu_multi_thread_eigen`` for user runs anywhere in ``src``; only
``tests/conftest.py`` pins it false, for bitwise reproducibility across the test
matrix. That is the right split: queue row Q9's idle-host rerun measured the
single-threaded pool 8--10% slower per step, so pinning it for users would cost
speed to buy a determinism users have not asked for. This row did not re-time it
-- a thread-pool effect is purely a timing measurement and the host was
contended -- so the default rests on Q9's idle-host numbers, not on a fresh one.

.. _q28-defaults:

The three defaults Q26 could not set
------------------------------------

Queue row Q28 (2026-09-20) measured the three defaults Q26 recorded as
suspicious but left alone for want of a measurement, and set each of them. The
host carried four sibling lanes throughout at 1-minute load 28--141, so no
wall time is quoted as a result: the cost numbers below are propagator
applies, step counts, right-hand-side evaluations and velocity-space degrees
of freedom, all of which are independent of load.

.. list-table::
   :header-rows: 1
   :widths: 26 14 14 46

   * - default
     - old
     - new
     - accuracy
   * - ``KrylovConfig.power_iters``
     - 200
     - 40
     - residual 9.50e-01 against 9.47e-01; neither certifies
   * - ``TimeConfig`` pairing, deck chose no ``dt``
     - rk2, fixed
     - rk4, CFL
     - -1.72e-04 against -2.63e-04 vs the eigensolve
   * - ``Nl``/``Nm`` linear fallback
     - (24, 12)
     - (12, 24)
     - -4.405% against +0.400% of the tracked GX golden

**power_iters: 200 → 40, and the two entry points now agree.** ``KrylovConfig``
said 200 and the ``dominant_eigenpair`` signature said 40 -- one route, two
costs, and two compilations of an iteration-static scan. Measured on the
shipped Cyclone deck at ``(Nl, Nm) = (4, 8)``, ``ky = 0.3``, float32, against
that rung's certified adaptive eigenpair :math:`\gamma = 0.10128645`:

=======  ==========  =========  =========
applies  gamma       residual   certified
=======  ==========  =========  =========
40       0.18119842  9.501e-01  no
80       0.22636196  9.255e-01  no
200      0.12442227  9.467e-01  no
400      0.09790193  9.214e-01  no
1000     0.08590183  4.358e-01  no
2000     0.09348834  1.105e-01  no
5000     0.10113729  6.182e-03  no
10000    0.10106588  5.959e-03  no
=======  ==========  =========  =========

Five times the applies moves the residual by 0.4% of an O(1) quantity, and the
gate is 1.19e-4: the pair is rejected at both. The ladder stalls near 6e-3 and
never certifies, so the value cannot be chosen for accuracy and is chosen for
cost. Nothing shipped takes this route -- no deck or Krylov contract selects
``method="power"``, ``shift_source="power"`` or ``fallback_method="power"``,
and every shipped contract sets ``power_iters`` explicitly -- and with the
default ``certify=True`` the route raises at 40 and at 200 alike, so the change
makes a rejection cheaper rather than making an answer different.

**Time: the pairing is defaulted, not either field.** ``ExplicitTimeConfig.dt``
is a required field and ``TimeConfig.dt`` is a defaulted one. That is the whole
of the real difference between the two surfaces: a caller of the library struct
has always chosen a step, a deck may not have. Measured on the same deck and
rung, with the ``TimeConfig`` dataclass defaults:

===================  ==========  =====  =========  ==================
method / policy      gamma       steps  RHS evals  rel. error
===================  ==========  =====  =========  ==================
rk2, fixed dt=0.1    fails       --     --         FloatingPointError
rk4, fixed dt=0.1    fails       --     --         FloatingPointError
rk2, CFL-controlled  0.10126899  7806   15612      -1.72e-04
rk3, CFL-controlled  0.10126041  4512   13536      -2.57e-04
rk4, CFL-controlled  0.10125984  2768   11072      -2.63e-04
===================  ==========  =====  =========  ==================

The defaulted ``dt = 0.1`` is about 7.8x the CFL-stable 0.01281 here, so both
fixed-step arms overflow whatever the scheme: changing ``method`` alone would
not have fixed it. With the controller on, all three schemes agree to better
than 3e-4 and rk4 reaches the horizon in **29.1% fewer right-hand-side
evaluations**, exactly :math:`4/(2.82 \times 2)`.

So a deck that chooses no ``dt`` now gets the CFL controller and rk4 with it,
and a deck that chooses ``dt`` keeps rk2 at that fixed step -- a fixed step
gives rk4 no step-size compensation for its four stages and would simply cost
twice as much. ``fixed_dt`` is **not** flipped globally: fourteen shipped decks
and parity fixtures omit it and depend on ``True``, and several back
evidence-ledger rows. Every shipped deck sets ``dt``, so none of them moves,
and a test asserts that so a future deck cannot drift onto the controller by
accident.

**Resolution: (24, 12) → (12, 24), at the same cost.** Certified adaptive
eigensolves on the shipped Cyclone deck at its own ``ky = 0.3``, float32,
against the tracked GX golden :math:`\gamma = 0.09302951` in
``src/gkx/data/cyclone_reference_adiabatic.csv``:

====  ====  =====  ==========  ==============  ========
Nl    Nm    Nl*Nm  gamma       rel. to golden  residual
====  ====  =====  ==========  ==============  ========
8     24    192    0.09877566  +6.177%         3.16e-06
8     32    256    0.09801412  +5.358%         3.97e-06
24    12    288    0.08893196  -4.405%         2.39e-06
12    24    288    0.09340143  **+0.400%**     3.27e-06
12    32    384    0.09263792  -0.421%         3.93e-06
16    32    512    0.09284505  -0.198%         4.13e-06
24    24    576    0.09368346  +0.703%         3.67e-06
16    48    768    0.09309106  +0.066%         5.55e-06
====  ====  =====  ==========  ==============  ========

Every rung is certified against the original operator at a 1.19e-4 float32
gate. ``(12, 24)`` carries the same :math:`N_l N_m` as ``(24, 12)`` -- the same
cost per propagator apply -- for eleven times less error, so this is a strict
improvement and not a trade. It is a balance, not "more Hermite wins":
``Nl = 8`` is worse than both at either Hermite count, because the parallel
phase mixing that sets an ITG rate needs Hermite resolution while the FLR
response still needs enough Laguerre. Spending beyond 288 is not bought either
-- ``(12, 32)`` at 1.33x the cost and ``(24, 24)`` at 2x both land further out
than 288 does, and only ``(16, 48)`` at 2.67x clearly improves on it -- and a
cost increase of that size is not a decision this fallback should make on a
deck's behalf. A deck whose physics runs the other way, such as the shipped ETG
decks at ``Nl = 24``/``Nm = 8``, must still say so, and all of them do.

**Limits.** All three measurements are one deck, one ``ky`` and one geometry.
The resolution ladder is certified but not a convergence proof: the sequence is
non-monotone through 288--384, so ``+0.400%`` is where this case lands at that
budget, not an error bound for another case. Nothing here replaces setting
``Nl`` and ``Nm`` in ``[run]``.

Distributed state sharding
--------------------------

Progress reporting is disabled by default; enable it by setting
``TimeConfig.progress_bar=True`` (or ``progress_bar=True`` in the integrator
call).

For distributed parallelization, set ``TimeConfig.state_sharding = "auto"``
(or ``"ky"`` / ``"kx"`` for the release-gated nonlinear path) to partition the
packed state array over multiple JAX devices. This is honored by the
fixed-step nonlinear scan through ``integrate_nonlinear_sharded``. When only one device is visible, the
parallelization request is ignored and the run proceeds on a single device
while preserving the same control-flow path for identity testing. The nonlinear
config path intentionally rejects ``"z"`` sharding because the current
multi-device FFT-axis decomposition has not passed the identity gate.
On macOS you can emulate multiple CPU devices with
``XLA_FLAGS=--xla_force_host_platform_device_count=2`` for non-FFT-axis
parallelization checks, but the nonlinear whole-state ``pjit`` profile skips
active multi-device CPU sharding by default because current JAX/XLA CPU FFT
layouts can abort before Python can catch a failure. Multi-GPU artifacts remain
the release reference for active nonlinear state-sharding diagnostics, and
production nonlinear speedup claims still require separate identity,
transport-window, and profiler gates.

For scan workloads, the default path is the custom fixed-step ``imex2``
owner. This keeps stepping shape-stable and improves throughput for multi-ky
scans.

Nonlinear FFT bracket
---------------------

The nonlinear :math:`E\times B` term is evaluated pseudospectrally using
FFT-based derivatives in the perpendicular plane. By default GKX uses
the compressed real-FFT path (``TimeConfig.compressed_real_fft = true``), which computes
gradients from the Nyquist-compressed (``N_y/2+1``) spectrum using the
benchmark-compatible compressed wavenumber layout: non-negative ``k_y`` (including positive
Nyquist when ``N_y`` is even) and a positive Nyquist multiplier on the ``k_x``
axis when ``N_x`` is even. The result is then expanded back to full
:math:`k_y`. This matches the tracked nonlinear reference layout and minimizes
memory traffic. Set ``compressed_real_fft = false`` to use the full complex FFT bracket
instead.

For electromagnetic nonlinear runs, GKX stacks the gyro-averaged
potentials ``J0*phi``, ``J0*apar``, and the ``bpar`` correction into a single
FFT batch. This collapses multiple rFFT/iFFT passes into one pipeline per
step and reuses the same real-space gradients for all channels.
Laguerre/Bessel factors on the benchmark quadrature grid (``J0`` and ``J1/alpha``) are
precomputed once per grid and cached in the linear operator, so the nonlinear
kernel only applies them via inexpensive elementwise multiplies.
For nonlinear runs that do not require the benchmark quadrature grid, set
``TimeConfig.laguerre_nonlinear_mode="spectral"`` to skip the Laguerre
quadrature transform and instead use the spectral gyroaverage factors ``Jl``
directly. The default ``"grid"`` mode applies the quadrature
transform.

The ``ky`` layout contract
--------------------------

Every spectral array in GKX has the shape ``(..., ky, kx, z)``, and its ``ky``
axis is in one of two layouts. :mod:`gkx.core_ky_layout` states the rule and
owns every conversion between them; nothing else may write a Hermitian
completion by hand.

``full``
  ``Nky = Ny``, the two-sided ``fftfreq`` order
  :math:`[0, 1, \dots, N_y/2 - 1, -N_y/2, \dots, -1]`. Half of it is
  redundant, because a real field obeys the reality condition
  :math:`F(-k_y, -k_x, z) = F^{*}(k_y, k_x, z)`, so the negative rows are
  rebuilt in the bracket and after every Runge--Kutta stage. This was the
  evolved state's layout through 2.2.0 and is still reachable, with
  ``[grid] ky_layout = "full"``, for reproducing such a run bit for bit.

``half``
  ``Nky = Nyc = 1 + Ny // 2``, the non-negative ``rfftfreq`` rows. The reality
  condition holds by construction. **This is the default after 2.2.0.** GX,
  stella and GS2 evolve this layout, and GKX already used it for restart
  files, NetCDF output, the ``ky`` spectra and real-space snapshots.

``Ny`` means the same thing in both, and it is the one that matters to a deck:
the length of the physical ``y`` axis, and so the resolution of the run.
``Nyc`` is a storage count. A deck that names ``Ny = 64`` resolves the same
wavenumbers on either axis.

The boundary between the two sits at I/O and at the nonlinear bracket, not in
the middle of the step. Restart files store ``Nyc`` rows and
:func:`gkx.core_ky_layout.to_full` widens them on read; the compressed bracket
computes on ``Nyc`` rows
(:func:`gkx.operators.nonlinear.brackets._spectral_bracket_half_core`) and
widens its result once at the end; the Hermitian projector applied after each
Runge--Kutta stage is exactly ``to_full(to_half(G))``.

Three properties of the contract are easy to get wrong, so they are stated and
tested rather than rederived at each call site.

**Nyc does not determine Ny.** Both :math:`N_y = 2(N_{yc}-1)` and
:math:`N_y = 2N_{yc}-1` give the same :math:`N_{yc}`, so a stored half-spectrum
array cannot say how long its own full axis is. Every widening takes
``ny_full`` explicitly. Inferring the even branch is what made an odd-``Ny``
raw restart expand to ``Ny - 1`` rows.

**The self-conjugate rows are not made real by the layout.** Row ``0`` always,
and row ``Ny/2`` when ``Ny`` is even, are their own conjugate partners; on them
the reality condition becomes a constraint *within* the row,
:math:`F(k_y^{sc}, k_x) = F^{*}(k_y^{sc}, -k_x)`. Storing only ``ky >= 0`` does
not enforce it. A real inverse transform imposes it silently by discarding the
anti-symmetric part, but an operator that writes those rows directly must call
:func:`gkx.core_ky_layout.symmetrize_self_conjugate_rows`.

**Reductions need row weights, not a factor of two.** For any quantity with
:math:`Q(-k_y) = Q(k_y)` -- :math:`|F|^2`, :math:`\mathrm{Re}(F G^{*})`, every
flux -- the sum over the full axis equals the
:func:`gkx.core_ky_layout.ky_row_weights` weighted sum over the half axis: 2 on
the paired rows, 1 on the self-conjugate rows. A blanket factor of two
over-counts an even grid's Nyquist row.

That weight has one owner. :func:`gkx.core_ky_layout.hermitian_mode_weights`
is the traced ``(k_y, k_x)`` form used by every quadrature, energy and spectral
reduction, and :func:`gkx.core_ky_layout.transport_mode_weights` is the
representative weight the flux kernels use, which carry the pair factor of two
themselves and so take 1 on a folded pair and 0.5 on a self-conjugate row. The
rule was written out three times before queue row Q24, and the copies gave an
even grid's Nyquist row the paired weight 2 on a half axis. Finding that row
needs :math:`N_y`, which a half axis cannot supply, so
:class:`gkx.core_grid.SpectralGrid` and the linear cache carry ``ny_full``:
the length of the two-sided axis their rows were taken from, or ``None`` when
the rows are a selection of modes rather than a complete axis. The two-sided
weights are unchanged on every row.

One convention in that rule is the state switch's own, and is settled here:
an even grid's Nyquist row is a flux **representative in both layouts**, at the
self-conjugate weight. The two-sided rule used to select the ``ky > 0`` rows,
which drops a row stored once as :math:`-N_y/2`, while the half axis stores the
same row as :math:`+N_y/2` and counted it, so the flux named different sums in
the two layouts. It may not: the flux kernel carries an explicit
:math:`\mathrm{i}k_y`, so that row's contribution is *odd* under a sign choice
that is pure convention. Counting it is safe because the contribution is
identically zero on any state representing a real field --- on a self-conjugate
row the reality condition reads :math:`F(k_x) = F^{*}(-k_x)`, under which the
summand is odd in :math:`k_x` and cancels pairwise --- and no run reaches the
row in any case, since two-thirds dealiasing zeroes everything at or above
:math:`N_y/3`.

Plan 5.3 N3 moved the evolved state to the ``half`` layout, which removes the
per-stage completion that the repository XLA profile attributes 41.9 per cent
of step time to, and after 2.2.0 that layout is what a run takes. The layout
travels with the configuration: :attr:`gkx.config.GridConfig.ky_layout` is a
deck key, and :func:`gkx.core_grid.build_spectral_grid` reads it unless a
caller overrides it, so the solver, the diagnostics, the restart writer and
the NetCDF writer --- which each build their own grid from the same config ---
cannot end up on different axes. Measured on idle pinned cores, the RK3 step
runs at 0.60x the two-sided one at ``64x64x24`` and 0.54x at ``32x32x24``;
see :doc:`performance` for the A/B and for the one number that moves the other
way.

Two things the layout does **not** change, and both are gated rather than
asserted. The published NetCDF bundle is unchanged, variable for variable and
dimension for dimension: the file has always stored the dealiased
``ky >= 0`` block, and the writer divides the pair weight back out of the
reductions that carry it
(``Phi2``, ``Wg``, ``Wphi``, ``Wapar``, ``TurbulentHeating``) so a published
row means what it always meant. And a restart file written by either axis
loads onto either axis, including files written by 2.2.0 and earlier, because the block
they store is the same block.

One reduction did have to change to keep that promise, and it is worth stating
because it looks like it should not have. The sum over ``ky`` **at fixed**
``kx`` is not layout-invariant under the pair weight: the reality condition
pairs :math:`(k_y, k_x)` with :math:`(-k_y, -k_x)`, not with
:math:`(-k_y, k_x)`, so a weighted sum over the stored rows charges the
partner's share to the mirrored ``kx`` column and returns the ``kx``-reflected
spectrum. On one field at ``16x16x24`` that is an 11 per cent error on
``Phi2_kxt`` and 13 per cent on ``Wphi_kxst`` --- plausible-looking, and
wrong. The half-axis reductions therefore add the ``kx`` mirror of their
paired-row part, which restores the two-sided answer to roundoff. The
two-sided expressions are untouched.

Equilibrium-flow shearing coordinates
--------------------------------------

The validated coordinate kernel
:func:`gkx.operators.nonlinear.projection.advance_shearing_coordinates`
follows a shearing wave according to

.. math::

   k_x^*(t) = k_x(0) - k_y\,\gamma_E t.

This model isolates perpendicular equilibrium-flow decorrelation. It does not
include a parallel-velocity-gradient drive, which is a distinct physical term
and requires its own normalization, instability, and transport gates
[Schekochihin12]_ [Ball19]_. The
matched comparison campaign uses the same scope: continuous :math:`k_x^*`
geometry updates, nearest-cell remapping, and the residual nonlinear FFT phase,
without claiming a toroidal-rotation or parallel-flow-shear model.

When the displacement crosses half a radial Fourier cell, the state is moved
to the nearest :math:`k_x` mode. The residual sub-cell displacement is retained
both in the effective wavenumber and in the real-space phase
:math:`\exp(i\,\delta k_x x)`. Modes leaving the two-thirds retained band are
zeroed rather than wrapped around the FFT grid. Retaining this residual phase
implements the corrected-remap principle and avoids the non-convergent smeared
nonlinear coupling of integer-only wavevector remapping [McMillan19]_. Tests
cover zero-shear identity,
the analytic shearing-wave trajectory, norm-preserving forward/inverse remaps,
the de-alias boundary, and JAX tangents with respect to both :math:`\gamma_E`
and the radial scale against centered finite differences.

The integer nearest-mode decision is piecewise constant and therefore uses a
stopped tangent. Continuous effective wavenumbers and phases remain
differentiable between the measure-zero remap events. This kernel is not yet a
shipped equilibrium-flow-shear model. The periodic/linked-boundary cache updater
:func:`gkx.operators.linear.cache_builder.update_linear_cache_for_sheared_kx`
already rebuilds :math:`k_\perp^2`, drift frequencies, gyroaverages, Bessel
tables, field-solve inputs, bracket multipliers, and hyperdiffusion from the
two-dimensional effective :math:`k_x` grid. Its zero-shear arrays and complete
linear RHS reproduce the static-cache path, and its nonzero-shear tangent agrees
with finite differences. The full-complex nonlinear bracket uses split
transforms to apply the residual radial phase between the :math:`k_x` and
:math:`k_y` FFTs; a canonical-coordinate invariance test verifies that the
Poisson bracket is unchanged by the shear-coordinate transformation. The
full-complex state is projected onto its Hermitian subspace after every remap
and Runge--Kutta stage, matching the real-field constraint implicit in the
production real-FFT layout. Without this projection a physical pilot accumulated
a 3.15% conjugate-symmetry defect by :math:`t=5`; the corrected trajectory keeps
that residual at machine zero. The compressed bracket can also evaluate this
fractional state in canonical shearing coordinates: the common residual phase
of the distribution and potential cancels from their Poisson bracket, leaving
the row-relative :math:`k_x` mesh. Direct full/compressed bracket, JAX-tangent,
three-step state, and heat-flux gates agree within ``2e-5`` relative tolerance.
The
research function
:func:`gkx.solvers_nonlinear_state_integration.integrate_nonlinear_sheared` verifies zero-shear
trajectory identity and cumulative full-step remapping. Its midpoint RK2 and
three-stage Heun RK3 routes evaluate each RHS in the correctly remapped stage
coordinate basis and return each derivative to the step basis before combining
stages. Both recover their designed orders on a physical drift/diamagnetic
trajectory. A
fixed-window Cyclone-like linear ITG pilot is converged to below 1% under a
factor-two timestep refinement and reduces the final potential norm by more
than 20% when :math:`\gamma_E=1`. This is the expected decorrelation direction
when the shearing rate exceeds the instability rate [Biglari90]_ [Waltz95]_,
but it is not a nonlinear transport validation.

Standard linked flux tubes use the cache-normalized radial spacing selected by
the twist-and-shift construction. The equilibrium-flow displacement is constant
along a fixed-:math:`k_y` linked chain, so the chain topology and endpoint
separation are unchanged while :math:`k_\perp`, drifts, gyroaverages, and field
operators are rebuilt. RK2 and RK3 recover the established linked trajectory
exactly at zero shear. At nonzero shear, tests preserve every linked-neighbor
spacing and compare the cache tangent with a centered finite difference.
Non-twist flux tubes remain unsupported because their radial coordinate is
:math:`z` dependent.

The fixed-step ``method="imex"`` route applies explicit nonlinear forcing in the
current sheared basis, remaps its right-hand side and warm start to
:math:`t_{n+1}`, rebuilds the matrix-free endpoint operator, and solves

.. math::

   [I-\Delta t L(t_{n+1})]G_{n+1}
   = G_n^* + \Delta t\,N(G_n,t_n)^*.

The tolerance-controlled solve uses the shared SOLVAX implicit derivative rule.
It is exactly identical to the static linked IMEX trajectory at zero shear,
recovers first-order convergence on a physical nonzero-shear trajectory, and
passes endpoint heat-flux plus JVP/VJP finite-difference gates. Adaptive sheared
IMEX and custom collision operators remain explicitly rejected.

State-only campaigns may set ``return_fields=False``. This follows the main
nonlinear-integrator contract and avoids the endpoint field/RHS evaluation and
field-history allocation on every step; the default retains field histories for
diagnostic compatibility. State-only and field-returning trajectories satisfy
the same zero-shear identity gate.

Field-returning and transport scans reuse each accepted endpoint RHS and field
solve as the next step's initial evaluation. The carried physical time ensures
the reused derivative and shearing basis are identical. This removes one
redundant RHS evaluation per step without altering the Runge--Kutta tableau;
the state-only path remains separate so it never computes fields solely for
reuse.

:func:`gkx.solvers_nonlinear_state_integration.integrate_nonlinear_sheared_transport` records the
canonical per-species gyro-Bohm heat flux at every accepted step. It uses
the same flux-surface quadrature and transport kernel as production nonlinear
diagnostics and evaluates that kernel with the instantaneous sheared cache. The
returned ``ShearedTransportTrace`` stores only the final distribution plus time,
and heat-flux traces, avoiding distribution- and field-history allocations.
Its default ``differentiable=True`` path traces the JAX-native field solve so
both forward JVP and reverse gradient reach the transport objective; both agree
with a centered finite difference in the validated mini-case. Setting
``differentiable=False`` selects the faster custom-VJP production field solve,
and a numerical-identity gate confirms that this policy switch does not alter
the trajectory or heat flux.

Setting ``fixed_dt=False`` applies the same production nonlinear CFL policy
used by the main diagnostic integrator. The accepted step combines conservative
linear-frequency bounds with the instantaneous pseudo-spectral
:math:`E\times B` frequency and is clipped by ``dt_min`` and ``dt_max``. The
trace's ``time`` array then records the nonuniform accepted-time grid, while
``steps`` is an explicit work budget. Time, step size, shearing remaps, fields,
and transport remain inside the JAX scan so tangents propagate through the
piecewise-smooth adaptive policy away from clipping and remap transitions.
Long campaigns can continue from ``final_state`` using the previous terminal
``time`` and accepted step as ``initial_time`` and ``initial_dt``. A chunked
versus single-scan identity gate covers the absolute shearing basis, state, and
heat-flux trace.

Treatment effects are evaluated with
:func:`gkx.matched_nonlinear_transport_report`, not by comparing two
instantaneous chaotic traces. The baseline and treatment first pass independent
post-transient finite-sample, running-mean drift, terminal-mean, block count,
and conservative SEM gates. Only then is the relative mean reduction reported;
its uncertainty separation uses the quadrature sum of the two block/bootstrap
SEMs. A drifting source window therefore blocks a treatment claim even when its
provisional mean is lower.

The first full-grid internal transport campaign uses ``64x64x24`` spatial
resolution, ``Nl=4``, ``Nm=8``, periodic ``x0=y0=28.2``, adaptive Heun RK3,
``dt_max=0.02``, and x64 precision. Over the independently selected
``t=[240,300]`` window, both the unsheared and ``gamma_E=0.01`` traces pass the
finite-sample, 12-block, running-drift, terminal-mean, and bootstrap-SEM gates.
Their heat fluxes are ``10.5009 +/- 0.0949`` and ``9.8603 +/- 0.0569``, giving a
``6.10%`` reduction with ``5.79`` quadrature-SEM separation. Moving the lower
window bound from ``t=240`` through ``t=280`` preserves the reduction direction
(``4.46--6.28%``). This closes the internal saturated-transport check for the
periodic research path.

A clean external comparison campaign used the same ``64x64x24``, ``Nl=4``,
``Nm=8``, periodic-domain contract and evolved both references from identical
initial states to ``t=300``. Over ``t=[240,300]``, the comparison traces pass the
same finite-sample, stationarity, and uncertainty checks but give
``5.9963 +/- 0.0321`` without shear and ``6.0014 +/- 0.0416`` with shear. The
corresponding ``-0.084%`` reduction is only ``-0.10`` combined SEM and disagrees
with the internal ``6.10%`` response. This is negative parity evidence, not a
flow-suppression result. A dealiased real-field regression shows that the
full-complex and compressed-real Poisson brackets agree before and immediately
after both integer and fractional remaps. A short physical sheared integration
also reproduces the full-complex state and heat-flux trace with the canonical
compressed bracket. A subsequent source audit localized a time-discretization
difference: the external adaptive RK3 route advances shear once with the
previous step size before selecting the next ``dt`` and holds that coordinate
basis fixed across all RK stages. GKX advances accepted physical time
and evaluates each stage in its exact shearing basis. The ``-0.084%`` result is
therefore negative cross-discretization evidence, not a model-identical parity
failure. Linked-boundary and fixed-step IMEX implementation gates now pass.

A bounded fixed-step source-localization probe confirms that the external
shearing path is active when the time policy is controlled. On a deterministic
reduced periodic Cyclone-like case with :math:`\gamma_E=0.5`, the terminal
``Phi2`` treatment ratios are ``0.64032``, ``0.64051``, and ``0.64060`` for
``dt=0.02``, ``0.01``, and ``0.005``. The corresponding startup heat-flux
ratios change from ``0.5079`` to ``0.5142`` and are not used as transport
evidence. This closes only the short fixed-step response check.

The subsequent full-resolution fixed-step campaign used the same
``64x64x24``, ``Nl=4``, ``Nm=8``, periodic weak-shear contract through
``t=300``. The internal fixed-IMEX baseline and treatment remained finite, but
both failed the predeclared ``t=[240,300]`` stationarity policy. Their means are
``15.4508 +/- 0.2628`` and ``16.1948 +/- 0.1602``, a 4.82% increase rather than
the required reduction. An independent fixed-RK4 comparison completed the same
grid, timestep, duration, seed, dissipation, and diagnostic contract. Both of
its windows pass the drift, terminal-mean, block, and SEM gates, with
``11.7154 +/- 0.2157`` and ``14.6236 +/- 0.1407``. This is a 24.82% increase,
resolved by 11.29 combined SEM. Integrator-specific absolute trajectories are
not claimed identical; the important result is that neither fixed-step audit
supports the earlier adaptive suppression claim.

RK3 remains useful for bounded research campaigns because it expands the stable
explicit operating envelope without changing the coordinate or transport
definitions. This path is a numerical foundation, not a promoted physical
model. The compressed-real bracket is an opt-in research route, while non-twist
and linked boundaries remain distinct: linked standard tubes are gated, while
non-twist tubes fail closed. The failed matched-response gate keeps flow shear
out of input files and executable claims. The compact machine-readable record
is ``docs/_static/flow_shear_fixed_step_response_gate.json``; large raw states
and comparison outputs are intentionally not tracked.

De-aliasing and hyperdiffusion
------------------------------

Nonlinear brackets are filtered using the standard ``2/3`` de-alias mask. The
mask lives on the spectral grid and is applied after each bracket evaluation.
Additional numerical stabilization is provided by hyperdiffusion in
:math:`k_\perp` (``TermConfig.hyperdiffusion`` / ``D_hyper`` settings), which
acts as a scale-selective damping term and is treated implicitly in IMEX
schemes.

Performance tuning
------------------

GKX includes several performance-oriented options that preserve
end-to-end JAX differentiability:

- **Batched ky scans**: pass ``ky_batch>1`` to the benchmark scan helpers to
  integrate multiple ky values at once using a sliced ky grid. Set
  ``fixed_batch_shape=True`` (default) to edge-pad the final batch and avoid
  recompilation on short tail batches.
- **Stacked FFT channels**: nonlinear brackets batch ``phi/apar/bpar`` into a
  single FFT pipeline so the spatial derivatives are computed once and reused
  across fields. This removes redundant transforms and reduces FFT calls.
- **Donation and parallelized buffers**: time integrators donate state buffers
  in JIT-compiled paths to reduce allocations.
- **Implicit preconditioning hooks**: ``implicit_preconditioner`` accepts
  ``"auto"/"diag"/"physics"/"block"`` (full diagonal preconditioner),
  ``"damping"`` (collisional/hyper-only), ``"pas"`` (PAS line preconditioner),
  ``"pas-coarse"`` (line + coarse correction in kx/linked-kx chains),
  ``"hermite-line"``
  (Hermite streaming line solve in ``m`` at fixed :math:`k_z`), or
  ``"hermite-line-coarse"`` (Hermite line solve + kx-coarse correction), or
  ``"identity"`` to disable preconditioning.
- **Shift-invert preconditioning hooks**: the shift-invert Krylov solver uses
  GMRES solves for ``(A - \sigma I)^{-1}``. Configure
  ``KrylovConfig.shift_preconditioner`` to accelerate these solves with
  ``"auto"`` (line-first electrostatic, field-corrected electromagnetic, with
  a certified electrostatic retry), ``"damping"`` (element-wise
  collisional/hyper damping),
  ``"hermite-line"`` (FFT in :math:`z` plus a tridiagonal Hermite solve for
  the additive diagonal-and-streaming symbol), ``"field-corrected"`` (the
  Hermite line plus the exact linear field response in a Woodbury capacitance
  solve), or ``"pr3-cm"`` (below). The complex shift scaling is shared with
  backward Euler, and right-preconditioned FGMRES minimizes the original
  shifted-system residual, starting from ``x = 0``: under right preconditioning
  the first Krylov vector is already :math:`M^{-1}b`, so a guess of
  :math:`M^{-1}b` searches a subset of what the first cycle searches anyway,
  and with a weak :math:`M` it starts the solve behind the trivial guess.
  The line path costs :math:`O(n)` storage and work. The field correction maps
  response columns sequentially and stores one state-by-field factor, trading
  a larger setup for lower Krylov counts in field-dominated cases.
- **The** ``pr3-cm`` **structured preconditioner** (queue row Q28) splits the
  operator into two halves that are each invertible in closed form and sweeps
  between them. The Hermite line solve above already inverts streaming,
  hypercollisions and the *z*-mean of the drift diagonal exactly; everything
  else -- the exact :math:`\omega_d(z)`, the mirror term, the drive, the end
  damping, collisions and the local field response -- is *z*-local, so it is a
  batch of dense :math:`(l, m)` blocks, one per ``(species, ky, kx, z)``. With
  :math:`s_1 = \sigma/2 - \alpha` one Peaceman--Rachford double sweep of the
  two shifted halves costs one solve of each and no operator application at
  all, and ``pr3-cm`` is three such sweeps started from zero. The parameter
  follows Q7's scalar symbol rule :math:`\alpha = -\sqrt{s_1 d}` unless
  ``KrylovConfig.shift_precond_alpha`` names one.

  The z-local block is solved exactly by block-Thomas in the Laguerre index
  plus one Sherman--Morrison correction, which stores :math:`3 N_l N_m^2`
  entries instead of the dense :math:`(N_l N_m)^2`. That is exact only while
  the block is *l*-tridiagonal with a rank-one field part, so the build
  measures both properties at every shift and
  ``KrylovConfig.shift_precond_block_solve`` decides what happens when they do
  not hold: ``"auto"`` falls back to the dense batched inverse -- the same
  preconditioner, a different apply cost -- and records why in
  ``EigenSolveStatus.inner["preconditioner_setup"]``, ``"block-thomas"``
  refuses rather than falling back, and ``"dense"`` is the control the fallback
  is measured against. A collision operator that couples the Laguerre index
  beyond :math:`l \pm 1` is the common reason for the fallback; the shipped
  Cyclone deck runs at :math:`\nu = 0` and measures the largest
  off-tridiagonal entry at exactly zero.

  The exact solve always stores less -- :math:`3 N_l N_m^2` against
  :math:`(N_l N_m)^2`, measured at 2.60x fewer bytes for
  :math:`N_l N_m = 192` -- but its *apply* cost changes sign with block size:
  :math:`N_l` sequential batched :math:`N_m \times N_m` solves were slower than
  one batched :math:`32 \times 32` product in seven of nine interleaved timing
  rounds, and faster than a :math:`192 \times 192` one in nine of nine. Two
  rungs do not calibrate a crossover, so ``"auto"`` takes the exact solve
  whenever the structure allows and ``"dense"`` is the better choice on a small
  velocity grid.

  Setting it up costs :math:`N_l N_m` probes of the z-local operator and a host
  factorization, done once per shift and handed to the compiled Arnoldi as an
  operand. It also requires the non-streaming operator to *be* z-local, which
  the build checks against the operator itself on a random vector and
  **refuses** rather than approximates. A grid carrying zonal
  :math:`(k_y = 0, k_x > 0)` rows under adiabatic electrons is the case that
  fails: their :math:`\langle\phi\rangle` is a sum over *z*. The linear eigen
  route escapes it because it reduces the grid to one non-zero :math:`k_y`.
  Every returned pair is checked with the matrix-free relative residual
  :math:`\lVert Av-\lambda v\rVert/
  \max(\lVert Av\rVert,|\lambda|\lVert v\rVert)`. Configure the acceptance
  threshold with ``KrylovConfig.shift_outer_residual_tol``; the default is
  ``1e-6``. Rejected primary and fallback pairs raise instead of returning a
  plausible frequency with an unconverged eigenvector.
- **Arnoldi breakdown policy**: a candidate basis direction is retained only
  when its norm exceeds a dtype-scaled threshold relative to the applied
  operator. Exact and numerical happy breakdown therefore terminate the
  resolved subspace instead of amplifying roundoff into a spurious mode.
- **Physical Ritz refinement**: after selecting a shift-invert Ritz vector,
  the solver recomputes its eigenvalue with the physical-operator Rayleigh
  quotient. For a fixed vector this scalar minimizes the Euclidean residual;
  it removes avoidable error from mapping an inexact inverse Ritz value through
  ``lambda = sigma + 1 / mu``. The outer residual gate remains mandatory.
- **Interior-eigenvalue restart boundary**: an augmented retained-Ritz
  prototype was tested against the physical KBM operator and removed. At
  matched ``Nl=8, Nm=24`` resolution, retaining two and four nearby vectors
  selected damped or opposite-frequency branches and changed the outer
  residual from ``0.881`` to ``0.922`` and ``0.991``. Merely carrying several
  Ritz vectors is therefore not treated as Krylov--Schur: a future retained
  implementation must perform ordered Schur compression or an equivalently
  branch-preserving correction. Two further physical discriminators were also
  rejected at reduced ``Nl=4, Nm=8`` resolution. An exact projected
  Jacobi--Davidson correction solved its correction equation to relative
  residual ``0.035`` but worsened the eigenpair residual from ``0.742`` to
  ``0.876`` on the first step. Ordered complex-Schur compression retaining four
  vectors selected a damped branch and changed the residual from ``0.755`` to
  ``0.956`` while costing about ``61`` seconds per restart. A propagated seed
  reduced the residual to ``0.521`` but selected the wrong damped branch. These
  failures rule out one-vector projection, generic thick restart, and seed
  changes as release repairs. The next candidate must couple the field and
  low-moment blocks or use a two-sided interior-eigenvalue correction, and must
  pass branch identity, physical residual, runtime, and peak-memory gates.
  A bounded A4000 check at ``Nl=8, Nm=24`` reached only ``0.429`` after three
  projected corrections from ``0.975`` and moved to the wrong high-frequency
  branch; the projected systems themselves remained poorly converged. The
  failure therefore persists beyond the smallest CPU discriminator.
  A two-sided variant also failed because its adjoint inverse iterations had
  residuals between ``0.67`` and ``2.12``; its first accurately solved
  projected correction still worsened the physical residual to ``0.870``. A
  matrix-free low-moment block preconditioner that retained the self-consistent
  field response cost ``240`` seconds versus ``30`` seconds for the damping
  baseline and returned residual ``0.972`` on the wrong branch. Future work
  must therefore assemble and factor a genuinely reduced field/moment Schur
  block rather than nesting another full-operator Krylov solve.
  That reduced-block route was subsequently tested with a ``1536 x 1536``
  complex field/moment block. Assembly took ``0.49`` seconds, factorization
  ``0.15`` seconds, and storage ``36`` MiB, so construction was not the
  bottleneck. However, diagonal and Hermite-line high-moment complements
  returned physical residuals ``0.936`` and ``0.999`` on incorrect branches,
  each taking about ``55`` seconds and more than ``1.3`` GB resident memory.
  The omitted high-moment drift/mirror coupling is therefore material; adding
  more coarse layers is not an accepted path without a new spectral
  transformation and an independently converged physical discriminator.
  JAX's current Schur primitive is CPU-only, so it is not used in the CPU/GPU
  solver API. See the
  `SLEPc Krylov--Schur documentation
  <https://slepc.upv.es/release/manualpages/EPS/EPSKRYLOVSCHUR.html>`_ and
  `harmonic extraction guidance
  <https://slepc.upv.es/release/manualpages/EPS/EPS_HARMONIC.html>`_.
- **Targeted shift-invert mode selection**: set ``KrylovConfig.mode_family``
  (for example ``"cyclone"``, ``"etg"``, ``"kbm"``) and
  ``KrylovConfig.shift_selection`` to stabilize branch selection in stiff
  spectra. KBM uses the positive reported-frequency convention in the tracked
  benchmark table, so its matrix eigenvalue target lies on the negative
  imaginary axis. ``KrylovConfig.fallback_method`` controls the automatic
  fallback policy when shift-invert returns a non-finite, strongly damped, or
  high-residual mode.
- **Reusable IMEX operators**: nonlinear IMEX runs can prebuild and reuse the
  matrix-free linear operator with
  :func:`gkx.operators.nonlinear.policies.build_nonlinear_imex_operator` and pass it to
  :func:`gkx.solvers_nonlinear_state_integration.integrate_nonlinear_imex_cached` via
  ``implicit_operator``. When ``apar=bpar=0``, the IMEX fixed-point and
  post-step field paths use the same electrostatic compiled linear-RHS route as
  the explicit nonlinear RHS, avoiding unused electromagnetic Hamiltonian
  branches while preserving the generic-RHS identity gate.

Automatic solver + fit-signal selection
---------------------------------------

For newcomer-friendly runs, the benchmark and runtime drivers accept
``solver="auto"`` and ``fit_signal="auto"``. The auto solver tries the
preferred path for the case (time integration for ion-scale Cyclone/KBM
benchmarks, Krylov for ETG) and falls back to the alternative if the returned
``(gamma, omega)`` is non-finite or violates ``require_positive``. The auto
fit-signal choice computes both ``phi`` and density moment time traces (when
available), scores each using the same windowing rules (``R^2`` of log-amplitude
and phase fits plus an optional growth-rate weight), and selects the higher
score. To make this decision robust, auto mode disables streaming fits and
stores the minimal time traces needed for the comparison.

Advanced users can override these defaults in TOML or Python drivers by setting
``solver="time"``/``"krylov"`` and ``fit_signal="phi"``/``"density"`` together
with custom fit-window parameters.

Implementation note:

- **Cached hypercollision factors**: the linear cache now stores the Hermite–
  Laguerre hypercollision ratios and masks to avoid repeated power operations
  inside the RHS assembly.

Custom collision operators
--------------------------

The shipped conserving long-wavelength model is independently exposed as
``drift_kinetic_dougherty_contribution``. It implements Appendix C, equation
(C6), of `Frei, Hoffmann & Ricci (2022)
<https://doi.org/10.1017/S0022377822000344>`_ in GKX's
Hermite--Laguerre ordering. The test suite verifies exact agreement between
that equation-level kernel and the production finite-Larmor-radius operator at
``b=0``, its density/momentum/thermal null moments, non-positive quadratic
rate, and collision-frequency JVP against centered finite differences. At
finite ``b``, the suite independently reconstructs the gyroaveraged flow and
temperature moments in Mandell et al. equations (3.39)--(3.42), verifies the
free-energy dissipation identity in equation (4.10), and measures first-order
convergence to the drift-kinetic equation as ``b`` tends to zero. This is not a
Sugama/Coulomb promotion: those operators require the complete
mass/temperature-ratio-dependent test- and field-particle coefficients and
their own ITG, zonal, conductivity, entropy, and convergence gates.

``conservative_full_f_dougherty_cross_moments`` separately implements only the
cross-species primitive-moment targets in equations (2.11)--(2.12) of
`Francisquez et al. (2022) <https://doi.org/10.1017/S0022377822000289>`_. The
tests check those formulas directly, pairwise momentum and energy conservation,
positive target temperatures, equal-species limits, and AD/FD agreement. The
multispecies gate covers several spatial samples and :math:`d_v=1,2,3`, and
checks that a uniform velocity offset shifts only the target flow. It is useful
when developing a future nonlinear full-distribution collision operator, but it
is not itself that operator and is not routed through the current delta-f
gyrokinetic evolution.

The full finite-Larmor-radius coefficient formulas contain deeply nested,
cancellation-sensitive sums. Following the implementation guidance in the
same reference, the planned advanced operator will generate those coefficients
offline with multiple-precision arithmetic, record normalization/provenance
and checksums with the resulting tables, and load only compact arrays into the
JAX runtime. Direct evaluation of the published sums in runtime ``float64`` is
not an accepted implementation path. The generated tables must first pass
symmetry, conservation, adjointness, and entropy gates before any transport
benchmark can promote the operator.

The exact Bessel--Laguerre kernel entering those sums is already implemented as
``core.velocity.bessel_laguerre_kernels``. Its recurrence is differentiable,
avoids factorial overflow, and is independently checked by velocity-space
quadrature and the finite-:math:`b` truncation behavior reported by
`Frei et al. (2021) <https://doi.org/10.1017/S0022377821000830>`_. The
associated-Laguerre prefactors for :math:`J_m`, :math:`m=0,1,2`, additionally
reconstruct the independent Bessel functions over velocity space. The
Appendix-A Coulomb speed integrals :math:`e_k` and :math:`E_k` are generated at
80-digit precision and checked independently by improper quadrature for several
orders and unequal thermal-speed ratios. Their test- and field-particle speed
moments from equations (A5) and (A13) additionally match direct Maxwellian
quadrature for unequal masses and temperatures, including equal-species
density and momentum invariant endpoints. Generalized-Laguerre monomial
coefficients from equation (3.10) reconstruct independent polynomials through
order eight. The cancellation-sensitive basis transforms and test-/field-
particle matrix contractions remain offline-generator work; these exact
primitives must not be interpreted as a complete finite-:math:`b` Sugama or
Coulomb operator. The required transform formulas have been located in
`Jorge, Ricci & Loureiro (2017) <https://arxiv.org/abs/1709.01411>`_, Appendix
A, equation (A4), and `Jorge, Frei & Ricci (2019)
<https://arxiv.org/abs/1906.03252>`_, Appendix B, equations (B5)--(B6); their
printed normalization is checked against the defining basis identity rather
than treated as an independent source of truth.

That isotropic transform is now implemented with 80-digit nested sums and only
casts at the generated-table boundary. Independent velocity quadrature and
forward/inverse shell products pass through total degree 12, including a shell
with condition number above :math:`10^8`. The finite-:math:`m` extension remains
implemented as lower-triangular parity blocks through reduced degree six for
:math:`m=0,1,2,3`. A convention factor :math:`2(-1)^m` is required for direct
velocity projection and pointwise basis reconstruction. Because literal
equation (B6) does not invert the finite-:math:`m` blocks, the generator inverts
the full forward matrix with 80-digit arithmetic as an independent oracle.
Frei et al. (2021), equation (3.33), includes the required weighted Laguerre
contraction and matches that inverse for all tested :math:`m=0,1,2,3` blocks;
it is the scalar inverse used for subsequent sums. Complete collision
contractions and conductivity, ITG, zonal-response, and velocity-resolution
gates remain required before runtime promotion.

The associated-Laguerre product coefficients in equations (3.36)--(3.37) and
(3.44)--(3.45) are evaluated by the same multiprecision generator and
reconstruct their defining products pointwise. Equation (3.35)'s finite-
:math:`b` gyro-moment-to-spherical-moment map then combines those products,
the finite-:math:`m` basis transform, and the exact :math:`K_n(b)` kernel.
Independent Bessel-weighted velocity projection verifies six coefficients
through :math:`m=3`; agreement between 20- and 32-term sums provides a local
truncation gate. These intermediate coefficients are not direct runtime APIs.

At the drift-kinetic endpoint the generator removes Bessel orders
:math:`n>0` and azimuthal harmonics :math:`m>0` before forming contractions;
their coefficients vanish exactly at :math:`b=0`. A bitwise regression against
the unspecialized pre-change path and a tracked Bessel-order-independence gate
protect the algebra. For the converged eight-mode
:math:`(p_{\max},j_{\max})=(6,3)` probe this reduces generation from 11.1 to
4.24 seconds without changing either test- or field-particle matrices.
The collision-table subcommands also avoid importing the runtime, JAX, or any
device backend; benchmark and solver dependencies are loaded only by the
subcommands that use them. This keeps offline arbitrary-precision generation
CPU-only even on GPU hosts. The development extra installs ``gmpy2`` so
``mpmath`` can use its GMP backend for these exact offline tables; it is not a
runtime dependency and does not alter stored coefficients. A matched
:math:`(5,2)` office probe retained the checksum and reduced generation by
18% relative to the pure-Python multiprecision backend.
The general finite-:math:`b` contraction remains too expensive for the full
conductivity resolution. It now evaluates equation (A4)'s basis transform as
an exact polynomial projection followed by analytic Gaussian and Laguerre
moments, replacing a six-deep combinatorial sum. Independent velocity
quadrature and inverse-shell gates are unchanged. After factoring
:math:`s_\perp^m`, the finite-:math:`m` associated basis is also transformed by
direct polynomial projection; the separately factorized equation-(B5) overlap
is retained as an independent oracle. A representative :math:`(5,2)`
two-wavelength build falls from 26.41 to 12.44 seconds with the same checksum;
:math:`(7,3)` falls from 411.22 to 247.04 seconds on the office CPU. Remaining
contractions now evaluate Laguerre products through cached analytic moments,
hoist wavelength/angular factors, group all radial indices sharing one speed
contraction, and collapse equation (3.33)'s auxiliary Laguerre sum by
orthogonality into one weighted polynomial projection. With shared pair
caches, the exact :math:`(9,4)`, Bessel-argument :math:`B=0.5` build falls from
464.05 to 183.59 seconds while preserving all six arrays and checksum
-152.93360627939981. Matrices take 156.68 seconds and the four polarization
vectors 26.91 seconds. These transformations reorder exact multiprecision
algebra; they do not truncate the collision model.

Correct full-block indexing gives 3.74% and 1.28% test/field matrix changes
between :math:`(7,3)` and :math:`(9,4)`, with nonzero polarization changes
below :math:`1.2\times10^{-8}`. The GMP-backed :math:`(12,5)` point then
completes in 556.21 seconds; its common :math:`(9,4)` test/field blocks change
by 2.87% and 1.07%, and polarization changes remain below
:math:`10^{-11}`. ``collision_finite_wavelength_generation_hierarchy.json``
records the prospective 5% intermediate-resolution coefficient pass. The
transport gate below applies the stricter growth-rate protocol rather than
promoting coefficient convergence by itself.
This distinction matters: Frei, Hoffmann & Ricci normalize the Bessel argument
as :math:`B=k_\perp\sqrt{2\tau}`. At :math:`\tau=1`, the existing
:math:`B=0.5` hierarchy corresponds to paper :math:`k_\perp=0.5/\sqrt{2}`;
their :math:`k_\perp=0.5` convergence point requires :math:`B=1/\sqrt{2}`.
The runtime interpolation uses the same convention explicitly through
:math:`B=\sqrt{2b_\mathrm{cache}}`.
At that required wavelength, exact :math:`(7,3)`, :math:`(9,4)`, and
:math:`(12,5)` builds complete in 43.48, 145.71, and 557.87 seconds on the
office CPU. The common test/field matrix changes are 5.87%/2.35% from
:math:`(7,3)` to :math:`(9,4)`, then 4.38%/1.92% from :math:`(9,4)` to
:math:`(12,5)`; the latter passes the prospective 5% intermediate gate.
Polarization-vector changes are below :math:`2.2\times10^{-9}` at the latter
step. These are coefficient-hierarchy results, not a collisional-ITG
acceptance claim. The paper-facing growth scan below now supplies the required
demonstrably equivalent converged resolution.

The exact generator subsequently contracts the Bessel expansion with each
Laguerre product before applying the inverse Hermite--Laguerre transform. It
also caches the shared Poisson/Bessel kernels and uses the associated basis's
exact reduced-degree support. On the local CPU, the same archived
:math:`(7,3)`, :math:`(9,4)`, and :math:`(12,5)` tables rebuild in 11.35,
37.27, and 139.42 seconds. Every one of the six generated arrays is bitwise
identical to its archived GMP result. These timings establish generator
efficiency and identity only; a bounded :math:`(15,6)` probe still required
350.84 seconds for its matrix before polarization, so the paper endpoint
requires the planned shared-precompute Hermite decomposition rather than an
unbounded serial run.

That decomposition now forks only complete output-Hermite matrix rows after
serial polarization has populated the shared transform and speed caches.
At :math:`(12,5)`, four local CPU workers reduce the exact total from 139.42
to 110.13 seconds; all six arrays remain bitwise identical. The matrix tail
takes 46.26 seconds after 63.88 seconds of shared polarization precomputation.
This is a scoped offline-generator improvement, not linear strong scaling:
the measured serial fraction is retained explicitly and polarization is not
forked. Eight matrix workers subsequently completed the exact
:math:`(15,6)` table in 261.39 seconds, including 142.93 seconds of serial
polarization work; its checksum is -423.2127750524206.

The drift-kinetic generator has a separate polynomial-support decomposition.
It evaluates independent multiprecision speed moments with fork workers and
optionally performs only the final dense contraction in float64. The latter
changes the P12/J5 matrices by :math:`1.4\times10^{-16}` relative L2. On the
36-core validation host, 28 workers generated the inclusive P24/J10 Coulomb
endpoint (275 moments) in 181.24 seconds. Its checksum is
``-1365.8775659269347``; an independently generated P20/J5 common block agrees
to :math:`7.3\times10^{-17}` relative L2, while density, momentum, energy,
symmetry, and non-positive-spectrum checks pass to roundoff.

The runtime-level scan below applies those exact tables through the complete
solved-field RHS at the paper's homogeneous-slab parameters. It confirms
collisional stabilization and closes the equivalent growth-convergence gate.
From :math:`(12,5)` to :math:`(15,6)`, the maximum growth-rate change is 1.99%
over all collision frequencies and 0.037% over the unstable
:math:`\nu\geq0.03` interval, both below the fixed 5% gate. An independent
collisionless hierarchy checks the remaining high-order endpoint: the
:math:`(15,6)` to :math:`(18,6)` change is 0.59%. Thus the expensive finite-
collision :math:`(18,6)` table is not required to establish equivalent growth
convergence. This closes the scoped homogeneous-slab ITG gate; it does not
promote the finite-wavelength operator to input files because the independent
collisional zonal-response gate remains open.
Machine-readable values and every gate are retained in
:download:`collision_finite_wavelength_itg_convergence.json
<_static/collision_finite_wavelength_itg_convergence.json>`.

**Generated figure.** Exact paper-wavelength slab-ITG hierarchy. The :math:`(15,6)`
finite-
collision scan and independent :math:`(18,6)` collisionless endpoint pass
the fixed equivalent-convergence gates.


The panel is regenerated with the existing artifact owner after exact table
archives are built with ``build_finite_wavelength_coulomb_pair_tables``::

   python tools/artifacts/build_linear_validation_artifacts.py collision-itg \
     --table finite_b_P7_J3.npz --table finite_b_P9_J4.npz \
     --table finite_b_P12_J5.npz --table finite_b_P15_J6.npz

For a fixed-wavelength zonal-response audit, generate a compact endpoint
archive instead of a full two-dimensional interpolation table::

   python tools/artifacts/build_linear_validation_artifacts.py collision-endpoint \
     --out finite_b_zonal_P24_J10_kx020.npz \
     --bessel-argument 0.282842712474619 --maximum-hermite-order 24 \
     --maximum-laguerre-order 10 --maximum-angular-bessel-order 4 \
     --maximum-bessel-laguerre-order 6 --digits 32 --worker-count 16

This command records every truncation, timing, and checksum in the archive.
It is a coefficient diagnostic, not a complete zonal table: on the paper
Miller surface the local Bessel argument varies along the field line. For the
one-species zonal problem, generate the required equal-target/source diagonal
table by repeating ``--bessel-argument`` over a grid that covers the measured
field-line interval::

   python tools/artifacts/build_linear_validation_artifacts.py collision-diagonal-table \
     --out finite_b_zonal_P24_J10_diagonal.npz \
     --bessel-argument 0.126 --bessel-argument 0.140 \
     --bessel-argument 0.155 --bessel-argument 0.254 \
     --bessel-argument 0.282 --bessel-argument 0.311 \
     --maximum-hermite-order 24 --maximum-laguerre-order 10 \
     --maximum-angular-bessel-order 4 \
     --maximum-bessel-laguerre-order 6 --digits 32 --worker-count 16

The generator prints each expensive phase, shares wavelength-independent
Coulomb speed coefficients across the complete grid, and writes coefficients
in the runtime Laguerre convention. A nested B-grid trace comparison remains
mandatory because the illustrative grid above is coverage, not an interpolation
convergence claim.

Run one table through the common zonal integrator with::

   python tools/artifacts/build_zonal_flow_artifacts.py \
     simulate-collisional-zonal-finite-b \
     --config benchmarks/collisional_zonal_response.toml \
     --table-archive finite_b_zonal_P24_J10_diagonal.npz \
     --model coulomb \
     --kx 0.1 --out-csv finite_b_zonal_kx010.csv \
     --dt 0.005 --maximum-normalized-time 30 --sample-stride 10 --nz 32

The runner rejects the wrong archive scope or Laguerre convention, malformed
resolution metadata, non-finite coefficients, and tables that do not cover the
actual field-line Bessel-argument range. The same command with ``--kx 0.2``
produces the second Figure-13 wavelength once the table and moment hierarchy
have passed their nested convergence gates. Select ``--model original_sugama``
or ``--model improved_sugama`` with the corresponding converted archive; a
model/archive mismatch fails before integration.

For the :math:`k_x=0.2` run, add ``--out-sections-csv sections.csv``. The
integrator advances first to :math:`t\nu=5`, retains that state, and then
continues to :math:`t\nu=30`, so producing Figure 14 does not repeat the first
part of the trace. At the outboard midplane it reconstructs the modulus of the
gyrocenter perturbation from Frei, Ernst & Ricci (2022), equation (52),

.. math::

   g_i/F_{Mi} = \sum_{p,j} N_i^{pj}
      \frac{H_p(s_{\parallel i})}{\sqrt{2^p p!}}L_j(x_i),

with the runtime's equivalent signed-Laguerre convention. The output contains
the cuts in :math:`s_{\parallel i}` at :math:`x_i=0` and in :math:`x_i` at
:math:`s_{\parallel i}=0`, each normalized to its maximum. A density-only
manufactured state independently recovers the two Maxwellian cuts. The
publication panel overlays those analytical Maxwellians as dashed references,
following Figure 14 of the source paper.

The tracked lower-hierarchy interpolation pilot is
``docs/_static/collision_finite_wavelength_zonal_b_grid_pilot.json``. At
:math:`(P,J)=(7,3)` through :math:`t\nu=2`, refining from four to six B-grid
points changes the normalized traces by relative :math:`L_2` errors
:math:`1.40\times10^{-4}` at :math:`k_x=0.1` and
:math:`3.68\times10^{-4}` at :math:`k_x=0.2`. This passes the declared
:math:`10^{-3}` interpolation gate but deliberately does not promote moment
resolution or the paper's :math:`t\nu=30` trace.

Velocity-space convergence is tracked independently in
``docs/_static/collision_finite_wavelength_zonal_moment_hierarchy.json``.
Adjacent physical traces are normalized by their common initial potential and
must satisfy both a 5% relative :math:`L_2` bound and a 5% maximum-deviation
bound at :math:`k_x=0.1` and :math:`0.2`. The P7/J3 to P12/J5 changes are
19.2% and 12.0% in relative :math:`L_2`; P12/J5 to P15/J6 still changes by
8.90% and 9.93%. The latter maximum deviations are 6.89% and 5.82%.
The four-point P18/J7 extension costs 189.06 seconds and changes the P15/J6
traces by 3.86% at :math:`k_x=0.1` and 7.16% at :math:`k_x=0.2`; the first
wavelength passes both criteria while the second still fails relative
:math:`L_2`. A four-wavelength P21/J8 table then completes in 584.73 seconds
with the decomposition described below. Its change from P18/J7 is 2.37% at
:math:`k_x=0.1` and 5.60% at :math:`k_x=0.2`; both maximum-deviation tests pass,
but the second relative-:math:`L_2` test narrowly remains open. Consequently
the tracked hierarchy shows monotone convergence but is not a substitute for
the paper-required P24/J10 traces.
Reproduce the report with::

   python tools/artifacts/build_zonal_flow_artifacts.py \
     collisional-zonal-moment-gate \
     --level 7 3 p7_kx010.csv p7_kx020.csv \
     --level 12 5 p12_kx010.csv p12_kx020.csv \
     --level 15 6 p15_kx010.csv p15_kx020.csv \
     --level 18 7 p18_kx010.csv p18_kx020.csv \
     --level 21 8 p21_kx010.csv p21_kx020.csv \
     --out-json collision_finite_wavelength_zonal_moment_hierarchy.json

An adjacent hierarchy can be derived from one authoritative high-order table
without recomputing or interpolating any coefficient. The projection keeps the
principal Hermite--Laguerre subspace in the runtime's Hermite-major ordering,
projects every collision matrix and polarization vector with the same indices,
and records the parent resolution and checksum::

   python tools/artifacts/build_linear_validation_artifacts.py \
     collision-project-table \
     --source P24_J10_coulomb.npz \
     --out P21_J8_coulomb.npz \
     --maximum-hermite-order 21 \
     --maximum-laguerre-order 8

This is a Galerkin truncation study of the same generated operator. It cannot
expand an archive, silently change its wavelength grid, or support a physics
claim beyond the source archive's declared collision model.
The resulting paper-endpoint comparison is retained in
:download:`collision_finite_wavelength_zonal_P21_P24_gate.json
<_static/collision_finite_wavelength_zonal_P21_P24_gate.json>`. Both traces
reach :math:`t\nu=30`: the P21/J8-to-P24/J10 relative :math:`L_2` changes are
4.93% at :math:`k_x=0.1` and 1.72% at :math:`k_x=0.2`, with maximum absolute
changes 0.0188 and 0.0070. All four values pass the prospectively fixed 5%
adjacent-hierarchy gate. This closes the moment-resolution question while
leaving the independent three-operator literature ordering as the physical
acceptance test.

The bounded archive generator uses a separate, prospectively gated
optimization. Every transform, collision moment, and projection coefficient is
still evaluated at the requested multiprecision; only the final dense
projection-vector contraction is performed in the float64 precision of the
stored table. The tracked P12/J5 six-wavelength gate in
``docs/_static/collision_finite_wavelength_table_contraction_gate.json`` gives
relative errors below :math:`6.4\times10^{-16}` across all matrices and
polarization vectors, and reduces eight-worker generation from 86.79 to 40.67
seconds (2.13x). The matched P12/J5 :math:`k_x=0.1` physical trace is bitwise
identical in every saved real and imaginary sample. This optimization affects
offline table generation, not the collision equations or runtime operator.
The same archive path factors the Bessel/Laguerre projection that is independent
of spherical and Hermite indices. A matched P18/J7, :math:`B=0.16` endpoint
retains checksum ``-604.0543094294402`` while reducing matrix assembly from
141.18 to 138.91 seconds and total generation from 235.12 to 229.70 seconds.
This measured but modest reduction is retained; it does not by itself make the
P24/J10 campaign tractable.

For high-order tables, independent Bessel-argument points can also be assigned
to separate processes. The default ``--wavelength-worker-count 1`` retains the
lower-memory shared algebra cache. Setting, for example,
``--worker-count 28 --wavelength-worker-count 4`` runs four points concurrently
with seven inner angular/row workers per point. Every point still shares the
wavelength-independent multiprecision speed coefficients, while its mutable
algebra cache remains process-local. A serial-versus-decomposed archive test
gates all six stored arrays to roundoff; this is an offline table-generation
strategy and is not evidence for simulation-runtime strong scaling.

If a complete high-order table cannot finish inside one bounded process, the
same equations can be generated as resumable angular shards. The production
command constructs the wavelength-independent speed coefficients once, forks
all ``M=0,...,4`` writers from that copy-on-write cache, and combines the
complete set in deterministic angular order::

   python tools/artifacts/build_linear_validation_artifacts.py \
     collision-shared-angular-table \
     --out finite_b_zonal_P24_J10_diagonal.npz \
     --bessel-argument 0.12 --bessel-argument 0.16 \
     --bessel-argument 0.24 --bessel-argument 0.32 \
     --maximum-hermite-order 24 --maximum-laguerre-order 10 \
     --maximum-angular-bessel-order 4 \
     --maximum-bessel-laguerre-order 6 --digits 32 \
     --worker-count 30 --wavelength-worker-count 2

The command retains sibling ``*_m0.npz`` through ``*_m4.npz`` files as
restartable evidence. Repeating the shared-table command validates and reuses
matching finite shards, skips precomputation when all are complete, and
redistributes the total worker budget across only the missing harmonics.
Existing complete shards can also be recombined without regeneration using::

   python tools/artifacts/build_linear_validation_artifacts.py \
     collision-combine-angular-shards \
     --shard finite_b_zonal_P24_J10_diagonal_m0.npz \
     --shard finite_b_zonal_P24_J10_diagonal_m1.npz \
     --shard finite_b_zonal_P24_J10_diagonal_m2.npz \
     --shard finite_b_zonal_P24_J10_diagonal_m3.npz \
     --shard finite_b_zonal_P24_J10_diagonal_m4.npz \
     --out finite_b_zonal_P24_J10_diagonal.npz

At P24/J10, even one angular harmonic evaluated at two wavelengths exceeded
the 600-second campaign bound. The bounded production checkpoint is therefore
one ``(B,m)`` equation block. Generate each block with a single
``--bessel-argument``, ``--angular-order m``, and
``--wavelength-worker-count 1``; combine the five harmonics at each wavelength
with ``collision-combine-angular-shards``. Finally concatenate the complete
single-wavelength tables without recomputing coefficients::

   python tools/artifacts/build_linear_validation_artifacts.py \
     collision-combine-wavelength-tables \
     --table finite_b_zonal_P24_J10_B012.npz \
     --table finite_b_zonal_P24_J10_B016.npz \
     --table finite_b_zonal_P24_J10_B024.npz \
     --table finite_b_zonal_P24_J10_B032.npz \
     --out finite_b_zonal_P24_J10_diagonal.npz

Both combiners verify complete angular ownership, matching equation and
resolution metadata, finite arrays, and a strictly ordered nonoverlapping
wavelength grid. This 20-block layout is exact additive checkpointing of the
published equations; it neither changes the collision model nor weakens the
P24/J10 acceptance criteria.

The combiner rejects missing/duplicate harmonics, metadata mismatches, and
non-finite arrays. A unit-level equation gate verifies that the ordered sum of
single-harmonic matrices and polarization vectors reproduces the monolithic
archive to roundoff. This is checkpointing of an additive analytical sum, not
a reduced collision model. The production-topology P12/J5 check is retained in
``collision_finite_wavelength_angular_shard_gate.json``: both matrix relative
:math:`L_2` errors are below :math:`5.8\times10^{-16}`, and all four
polarization vectors are bitwise identical.

The radial order six follows the convergence statement in
`Frei, Ernst & Ricci (2022) <https://arxiv.org/abs/2202.06293>`_; the separate
angular cutoff must pass a nested endpoint or observable-level convergence
check before use. The current P12/J5 check at
:math:`B=\sqrt{2}\,0.2` changes the most sensitive matrix norm by
:math:`1.24\times10^{-7}` between angular orders four and six. The command
generates exact Coulomb coefficients only. Original-Sugama field terms are
then projected independently from equations (3.65), (3.68)--(3.69), and
(3.79)--(3.80) of the `2021 gyro-moment collision derivation
<https://arxiv.org/abs/2104.11480>`_. In the runtime's orthonormal signed-
Laguerre basis their matrix is assembled as

.. math::

   C^{F,\mathrm{OS}}_{ij}(B)
   = -\frac{A_i^\parallel(B)A_j^\parallel(B)}{\gamma}
     -\frac{A_i^\perp(B)A_j^\perp(B)}{\gamma}
     -\frac{A_i^E(B)A_j^E(B)}{\eta},

where each :math:`A` is a direct velocity projection of the corresponding
parallel-flow, perpendicular-flow, or energy test response in equation
(3.65). Product Gauss--Hermite/Laguerre rules with 80 and 96 nodes must agree
before the higher-order matrix is accepted. At :math:`B=0` this independent
route recovers the verified drift-kinetic C6 field matrix to roundoff. At
finite :math:`B` the same channels are not gyrocenter collision invariants:
their nonzero action is the classical gyro-diffusion expected from the
gyroaveraged operator. The improved model adds the separate Braginskii-moment
correction of Frei, Ernst & Ricci (2022), equations (61)--(69), under the same
quadrature-convergence contract.

An independent homogeneous-slab matrix reconstruction now evaluates equations
(2.14)--(2.18) directly at :math:`k_\perp=0.5`,
:math:`k_\parallel=0.1`, :math:`\eta=3`, and :math:`\tau=1`. It exposed and
fixed a separate truncation-boundary error: the distribution moment
:math:`N_{p,J+1}` is zero, but the analytically known Bessel coefficient
:math:`K_{J+1}` in equation (2.16a) is not. Retaining that coefficient makes
every collisionless runtime matrix entry agree with the published hierarchy
to roundoff in x64 and within :math:`2\times10^{-6}` in default precision.
The drift-kinetic generator still evaluates
collapsed equations
(3.53)--(3.56) directly: its :math:`(20,5)` response is the validated transport
path, whereas finite-:math:`b` still requires the independent zonal-response
gate. These are local algorithm timings, not portable runtime-performance
claims.

That contraction is now implemented offline. Equations (3.48)--(3.49) produce
test and field matrices in Hermite-major order, while equations (3.41) and
(3.50) produce separately coupled polarization vectors for target and source
species. The :math:`b=0` endpoint recovers the independent drift-kinetic
Coulomb coefficients and passes symmetry, negative-semidefinite, density,
momentum, and energy gates. Direct :math:`J_0J_m` quadrature verifies the
polarization coefficient. Runtime promotion still requires the independent
collisional zonal-response benchmark.

The tracked ``collision_operator_verification.json`` and matching panel in
:doc:`operators` turn this into a numerical gate rather than a visual claim.
The present level covers arbitrary-precision coefficient generation, direct
manufactured velocity-space projection, Bessel-sum convergence, the three
collision-invariant null modes, and non-positive entropy production. Its
coupled spherical/radial scan rejects the former low-order cutoff at 29%
relative error and admits :math:`(p_{\max},j_{\max})=(8,4)` at
:math:`8.68\times10^{-7}` against a converged :math:`(9,4)` reference. The
drift-kinetic driven response now has its separate paper-scale
Hermite/Laguerre scan. Finite-:math:`b` runtime-table convergence,
Spitzer--Härm and Braginskii transport, collision-frequency convergence, and
finite-:math:`b` slab ITG have separate closed gates; collisionless versus
collisional zonal response is the remaining promotion gate. Finite-:math:`b`
gyrocenter density is deliberately not treated as a local invariant: the
tracked test, field, and combined :math:`O(b^2)` density-row gates resolve the
classical gyro-diffusion discussed after equation (3.5) of Frei et al. (2021).
Particle-space conservation would require evaluating the pre-gyroaverage
operator at fixed particle position; it cannot be reconstructed from the
gyrophase-independent runtime matrix.

The finite-wavelength ITG promotion target follows the convergence section of
`the 2022 finite-wavelength ITG study by Frei, Hoffmann & Ricci
<https://arxiv.org/abs/2201.02860>`_. Its first
gate is the :math:`k_\perp=0.5`, :math:`k_\parallel=0.1`, :math:`\eta=3`,
:math:`R_B=0.1` collisionality scan through :math:`(P,J)=(18,6)`. The
independent follow-ups hold :math:`P=18` for the perpendicular-wavenumber/J
scan, use :math:`J=10` for the weak-collision P-scan and :math:`P=32` for the
J-scan, and vary magnetic-gradient strength separately. Analytical peak
estimates are context, not numerical acceptance data.

The strongly collisional stage now has a differentiable constrained-response
solver rather than a matrix-distance surrogate. It removes stated invariant
modes, solves the remaining dense moment system on device, and differentiates
the resulting current through the same JAX solve. Analytic damping, long-time
matrix-exponential, JIT, and centered finite-difference checks close this
algorithmic layer. The direct Coulomb hierarchy is resolved through
:math:`(P,J)=(20,5)`: its largest current change from :math:`(15,5)` is
:math:`1.66\times10^{-4}` over :math:`Z=1,2,5,10,100`, below the fixed 0.5%
gate. The Spitzer--Härm normalization is closed for this unmagnetized,
equal-temperature drift-kinetic problem. With
:math:`\widehat E=eE/(m_ev_{Te}\nu_{ee})`, the computed ratio
:math:`(u_e/v_{Te})/\widehat E` is
:math:`\sigma_\parallel/[n_e e^2/(m_e\nu_{ee})]`. Substituting the collision
frequency and Spitzer conductivity gives the high-charge limit
:math:`64/[3\,2^{3/2}\pi Z]`. The :math:`Z=100` Coulomb result is 7.453% below
this limit, inside the prospectively fixed 8% gate. The source is also converted
to the paper's :math:`eE/(\sqrt{m_eT_e}\nu_{ee})=10^{-3}` convention; Coulomb,
original Sugama, and improved Sugama all reach their matrix steady state by
:math:`t\nu_{ee}=50` and remain linear over fields from :math:`10^{-4}` to
:math:`10^{-2}`.

The original-Sugama hierarchy is generated at equal temperature from the
Coulomb test matrix plus the unique self-adjoint low-rank momentum/energy
restoration. The improved hierarchy evaluates the Coulomb Braginskii
:math:`N` matrix, removes its momentum-restoring Schur complement, and applies
the exact drift-kinetic basis transforms through :math:`K=5` in
multiprecision. The two constructions recover the independently tabulated C6
and C103 coefficients and reproduce the published low- and high-charge current
ordering without fitting response data. At the converged endpoint, the largest
improved-to-Coulomb current difference is 0.307%, the
:math:`K=4\rightarrow5` change is 0.439%, and the final velocity-hierarchy
change is 0.0237%.

Rosenbluth--Hinton residual flow is collisionless and is therefore not used as
a substitute for the Hinton--Rosenbluth collisional damping test.
The collisional gate follows Figures 12--14 of `Frei, Ernst & Ricci (2022)
<https://arxiv.org/abs/2202.06293>`_ rather than an unrelated collisionless
zonal benchmark. It uses ion--ion collisions at :math:`q=1.4`,
:math:`\epsilon=0.1`, and :math:`\nu_i^*=3.13`, with the paper's converged
:math:`(P,J)=(24,10)` hierarchy. Required outputs are the drift-kinetic
:math:`k_x=0.05` response, finite-Larmor :math:`k_x=0.1` and 0.2 responses,
Coulomb/original-Sugama/improved-Sugama ordering, the Xiao long-time estimate

.. math::

   R_z(\infty) =
   \frac{\epsilon^2/q^2}{1 + \epsilon^2/q^2},

and parallel/perpendicular sections of the perturbed distribution at
:math:`t\nu=5`. The stated :math:`q` and :math:`\epsilon` give
:math:`R_z(\infty)=0.00508`; this is deliberately distinct from the
collisionless Rosenbluth--Hinton estimate. Lower moment counts or collisionless
residual agreement are
development evidence, not substitutes for that converged gate.

The paper normalization is converted explicitly rather than fitted,

.. math::

   \nu = \frac{\nu_i^*\epsilon^{3/2}}{\sqrt{2}q}
       = 0.0499921,

so a trace through :math:`t\nu=30` evolves to solver time approximately 600.
The canonical geometry and time contract is
``benchmarks/collisional_zonal_response.toml``. Its Miller surface sets
``rhoc/R0=0.1`` explicitly; changing only the analytic ``epsilon`` metadata
would leave the generated surface at the wrong aspect ratio. The dense
drift-kinetic matrix acts on evolved gyrocenter :math:`g` moments, following
Frei, Ernst & Ricci (2022), Eq. (73), rather than on the post-field
Hamiltonian. One drift-kinetic model trace
is reproduced from an offline P24/J10 matrix archive with::

   python tools/artifacts/build_zonal_flow_artifacts.py \
     simulate-collisional-zonal-dk \
     --config benchmarks/collisional_zonal_response.toml \
     --model-archive collisional_zonal_dk_p24j10.npz \
     --model coulomb --out-csv coulomb_zonal_trace.csv

The runner rejects a requested radial mode if it is outside the grid's active
linear spectrum, records imaginary-response contamination and the tail median,
and keeps the generated dense archive outside the repository. Coulomb,
original-Sugama, and improved-Sugama runs use the same initial state, grid, and
time discretization.

The drift-kinetic Figure-12 subset has late-window medians 0.00565 (original
Sugama), 0.00572 (Coulomb), and 0.00585 (improved Sugama, :math:`K=5`) against
the Xiao estimate 0.00508. It is incorporated into the complete panel below
rather than retained as a duplicate figure and trace bundle.

The acceptance contract is implemented by the existing zonal-artifact owner::

   python tools/artifacts/build_zonal_flow_artifacts.py collisional-zonal \
     --traces collisional_zonal_traces.csv \
     --sections collisional_zonal_velocity_sections.csv \
     --out-json collisional_zonal_gate.json \
     --out-png collisional_zonal_gate.png

Trace rows contain ``model,kx,t_nu,response,p_max,j_max``. Velocity-section
rows add ``coordinate,abscissa,normalized_distribution`` and are evaluated at
:math:`k_x=0.2`, :math:`t\nu=5`. The gate requires all three collision models,
all three radial wavenumbers, normalized-time coverage through 30, the
:math:`(24,10)` hierarchy, the Xiao residual at :math:`k_x=0.05`, OS/IS/Coulomb
late-time ordering at finite wavelength, improved-Sugama proximity to Coulomb
over :math:`t\nu\leq10`, and both velocity sections. Missing data fail closed;
the tool does not infer a pass from a lower-resolution or collisionless trace,
and the command exits nonzero while any gate remains open.

**Generated figure.** Complete Figures 12--14 protocol at :math:`(P,J)=(24,10)`. The
finite-wavelength late responses obey original Sugama < improved Sugama <
Coulomb at :math:`k_x\rho_i=0.1` and 0.2; the improved model has the smaller
early-window error relative to Coulomb at both wavenumbers. Both
:math:`t\nu=5` velocity sections and the drift-kinetic Xiao-residual gate
pass.


The full-resolution campaign contains 73,824 trace rows. Its exact verdict is
retained in :download:`the JSON report
<_static/collision_finite_wavelength_zonal_response.json>`, and the compact
:download:`velocity sections
<_static/collision_finite_wavelength_zonal_velocity_sections.csv>` are not
decimated. Dense coefficient archives, raw traces, and logs remain external
campaign artifacts rather than adding more than six megabytes of duplicative
data to the repository.

Nonlinear full-distribution Landau collisions are a separate future model, not
an extension flag on this linearized matrix. A dense precomputed collision
tensor would have prohibitive basis scaling. The planned route follows the
one-centre Coulomb/Talmi formulation of `Jorge et al. (2026)
<https://arxiv.org/abs/2606.31035>`_: approximate the Boys kernel with a
controlled sum of exponentials, contract the resulting separable factors
matrix-free, rotate between oscillator and Hermite--Laguerre bases, and project
roundoff-level particle, momentum, and energy defects. Generic separable
Kronecker contractions and differentiable constrained solves belong in
`SOLVAX <https://github.com/uwplasma/SOLVAX>`_; gyrokinetic normalization,
species coupling, and Landau physics remain GKX responsibilities. Any
implementation must reproduce manufactured dense low-order tensors, quadrature
refinement, Maxwellian stationarity, conservation, entropy production,
isotropic relaxation, two-species equilibration, JVP/VJP checks, and measured
memory scaling before runtime exposure.

For unlike species, runtime interpolation is two-dimensional: target
:math:`k_\perp\rho_a` controls the outer/test factors and source
:math:`k_\perp\rho_b` controls the field-particle moment map. The pure JAX
kernel contracts the resulting test block with :math:`G_a`, the field block
with :math:`G_b`, and the four polarization vectors with the solved
:math:`\phi`. This ordering is JIT- and JVP-gated and prevents the common error
of applying the complete finite-wavelength matrix to the nonadiabatic
Hamiltonian response, which would count the pullback field terms twice.

Python workflows may supply any JAX-compatible object implementing
``apply(context)`` to ``linear_rhs``, ``linear_rhs_cached``,
``integrate_linear``, or ``nonlinear_rhs_cached`` through the
``collision_operator`` keyword. The returned array is the unit-weight
collisional contribution and must match the distribution-state shape. The
configured collision term weight multiplies this contribution; built-in
collisions are disabled, while hypercollisions remain independent.

Custom operators currently support serial explicit and IMEX linear
integration, cached nonlinear RHS evaluation, and serial explicit nonlinear
state integration. Implicit solves, diagnostic scans, and decomposed state
integration reject this option until their operator, observable, and
preconditioner contracts can include the same model exactly. This boundary is
appropriate for differentiable collision-model research because state, cache,
and parameter arrays remain inside the JAX trace while artifact writing and
configuration parsing remain outside it.

The context contains both the evolved distribution :math:`G` and the
post-field Hamiltonian response :math:`H`, plus the solved fields, cache, and
parameters. This is required for gyroaveraged field-particle terms and keeps
the callback differentiable without repeating the field solve.

An operator may additionally satisfy ``SplitCollisionOperator`` by defining
``split_step(context, dt)``. This advertises a valid
finite-time update; the general runtime does not route it until the
operator-specific invariant and entropy gates pass. Built-in
``collision_split`` applies only to diagonal hypercollisions. Conserving
collisions stay in the assembled RHS so their low-order field-particle terms
cannot be dropped by diagonal operator splitting.

Gyroaverage and polarization
----------------------------

The Laguerre gyroaverage coefficients follow the Laguerre–Hermite convention
used in Hermite–Laguerre gyrokinetic moment closures,

.. math::

   J_\ell(b) = \frac{1}{\ell!}\left(-\frac{b}{2}\right)^\ell e^{-b/2},

with :math:`b = k_\perp^2 \rho^2`. This definition is consistent with the
Laguerre projection of the gyroaveraged potential in the Hermite–Laguerre
closure used by the linear operator.

Parallel streaming
------------------

The streaming operator is applied in real space using a spectral periodic
derivative in :math:`z` (FFT-based, via ``jax.numpy.fft``) and the Hermite
ladder coupling

.. math::

   \mathcal{L}_m[H] = \sqrt{m+1} H_{m+1} + \sqrt{m} H_{m-1}.

In the current linked-FFT formulation we apply the parallel derivative to the
non-adiabatic moments plus explicit field terms before the Hermite ladder is
applied. In other words, the streamed quantity is

.. math::

   \tilde{G}_{\ell m} = G_{\ell m}
   + \frac{Z_s}{T_s} J_\ell \phi\,\delta_{m0}
   - \frac{Z_s v_{th}}{T_s} J_\ell A_\parallel\,\delta_{m1}
   + J_\ell^B B_\parallel\,\delta_{m0},

so that the current streaming term uses :math:`\partial_z \tilde{G}` instead of
the full :math:`H_{\ell m}` derivative. This matches the ordering and ghost
exchange used by GX’s ``grad_parallel_linked`` operator.

Landau damping against the exact kinetic roots
----------------------------------------------

The sharpest available check on the Hermite representation is the slab
gyrokinetic ion-acoustic dispersion relation with adiabatic electrons at
:math:`k_\perp \to 0`,

.. math::

   1 + \frac{T_i}{T_e} + \zeta Z(\zeta) = 0,
   \qquad
   \zeta = \frac{\omega}{k_\parallel \sqrt{2T_i/m_i}},

with :math:`Z` the Fried-Conte plasma dispersion function. It has no free
parameters.

.. list-table:: GKX's linear operator vs the exact roots
   :header-rows: 1

   * - case
     - exact
     - GKX
     - error
   * - :math:`T_e/T_i=1`, :math:`\omega`
     - 2.045904866
     - 2.047220793
     - 0.064%
   * - :math:`T_e/T_i=1`, :math:`\gamma`
     - -0.851330459
     - -0.849234188
     - 0.246%
   * - :math:`T_e/T_i=10`, :math:`\omega`
     - 3.728834801
     - 3.728993838
     - 0.004%
   * - :math:`T_e/T_i=10`, :math:`\gamma`
     - -0.058337421
     - -0.058339802
     - 0.004%

.. figure:: _static/landau_damping_validation.png
   :width: 100%

   Everything shown is produced by ``linear_rhs_cached``, GKX's production
   linear operator; panel (b) eigen-decomposes that same operator by applying it
   to Hermite basis vectors.

Measurement pitfalls
~~~~~~~~~~~~~~~~~~~~~

Four separate traps produce plausible but wrong numbers here. Each is gated in
``tests/validation/physics_gates/test_hermite_hierarchy_physics.py``.

**1. A collisionless truncated Hermite system cannot Landau damp at all.**
Free streaming is anti-Hermitian, so the truncated matrix has a purely real
spectrum -- measured :math:`2.8\times10^{-14}`. Whatever a fit returns from a
collisionless run is a transient that ends at recurrence, not an asymptotic
rate. The gate asserts both that the spectrum is real *and* that collisions make
it genuinely damped, so it cannot be satisfied by an operator that does nothing.

**2. The Landau root is not an eigenvalue.** It is a pole of the analytically
continued response function. Taking the least-damped eigenvalue of the
collisional operator and extrapolating gives **76% error** at
:math:`T_e/T_i = 1`, because at strong damping the discrete modes are ballistic
rather than collective. Landau damping is an initial-value phenomenon and must
be measured as one: add Lenard-Bernstein collisions, measure the decay at
several small :math:`\nu`, and extrapolate :math:`\nu \to 0`.

**3. A density perturbation with no initial flow is a standing wave.** It splits
into left- and right-going sound waves, so the signal stays real and beats
through zeros. Unwrapping its phase returns :math:`\omega \approx 0.13` instead
of :math:`2.05`. Fitting :math:`A e^{\gamma t}\cos(\omega t + \phi)` returns the
root.

**4. Temperature-ratio conventions.** GKX's ``tau_e`` is :math:`T_i/T_e`. Passing
the reciprocal is invisible at :math:`T_e = T_i` and gives a **2620% error** at
:math:`T_e/T_i = 10`. The thermal speed carries a second convention: :math:`Z`
takes :math:`\zeta = \omega/(k\sqrt{2T/m})` while GKX normalizes to
:math:`v_{ti} = \sqrt{T/m}`, a factor :math:`\sqrt2` that still looks physical.

Panel (a) also shows the crossover that follows from trap 1: the Landau pole
dominates for roughly two decades, after which the ballistic part of the initial
condition leaves a plateau, and the revival at
:math:`t_{\mathrm{rec}} = 2\sqrt{N_m}` ends the useful window entirely.

Regenerate with ``tools/artifacts/build_landau_damping_figure.py``.

Hermite closure and recurrence
------------------------------

The ladder :math:`\mathcal{L}_m` above couples :math:`m` to :math:`m\pm1`, so it
must be terminated at the last retained moment :math:`m = M`. The default
termination sets :math:`G_{M+1} = 0`. That is a *reflecting* boundary in Hermite
space: free energy streaming up the ladder reaches the end and propagates back
down, returning to the low moments as **recurrence** at

.. math::

   t_{\mathrm{rec}} \simeq \frac{2\sqrt{M}}{k_\parallel v_{th}}.

Because :math:`t_{\mathrm{rec}}` grows only as :math:`\sqrt{M}`, refining the
velocity grid is an inefficient remedy; the end of the ladder must instead
absorb the outgoing flux.

GKX provides two absorbing treatments.

**Hypercollisions** (default, ``[physics] hypercollisions = true``) apply a
scale-selective damping :math:`\propto (m/M)^{p}` that acts over a band of high
moments. Their strength is set by ``nu_hyper`` and ``p_hyper``; see
`De-aliasing and hyperdiffusion`_.

**Reflectionless closure** (opt-in) applies the outgoing-wave condition of
Kanekar, Schekochihin, Dorland & Loureiro, *J. Plasma Phys.* **81**, 305810104
(2015), Eq. (4.36),

.. math::

   G_{M+1} = -\,i\,\mathrm{sgn}(k_\parallel)\,R_{M+1}\,G_M,
   \qquad
   R_{M+1} = \frac{M}{\sqrt{2(M+1)}}
             \frac{\Gamma(M/2)}{\Gamma\!\left((M+1)/2\right)},

which adds a single term to the last moment,

.. math::

   \left.\frac{\partial G_M}{\partial t}\right|_{\mathrm{closure}}
   = -\,R_{M+1}\sqrt{M+1}\; v_{th}\,\lvert k_\parallel\rvert\, G_M .

Three properties make this attractive as a closure rather than as a tuned
dissipation:

- :math:`R_{M+1} \to 1 - 1/(4M)` as :math:`M` grows, so the absorption becomes
  *exact* with increasing resolution instead of requiring retuning.
- :math:`R_{M+1} > 0` makes the term a strictly positive-definite sink of free
  energy, so it cannot inject energy at any resolution.
- It is confined to :math:`m = M` exactly, so unlike a band operator it cannot
  bias the resolved moments.

At :math:`M = 3` the coefficient evaluates to :math:`R_4 = 0.921318`, exactly
reproducing the Hammett-Perkins three-pole coefficient :math:`\sqrt{8/\pi}/\sqrt3`,
which is an independent check on the family.

Measured on the free-streaming hierarchy at :math:`k_\parallel v_{ti}=1`, from an
initial :math:`\lvert g_0 \rvert = 1`. Two metrics are reported because the first
one alone is misleading: revival suppression rewards *any* strong damping, so a
closure that flattens the whole hierarchy would score perfectly on it. The second
column charges for perturbing physics that is still resolved, measured against a
converged :math:`M = 1024` run over :math:`t < t_{\mathrm{rec}}`. Hypercollisions
use the GX Appendix B normalization; without it the comparison is meaningless.

.. list-table:: Revived :math:`\lvert g_0 \rvert` / error on the resolved window
   :header-rows: 1

   * - :math:`M`
     - hard truncation
     - hypercollisions
     - reflectionless
   * - 16
     - 0.9987 / 1.0
     - 0.0009 / 8.5e-4
     - 0.0398 / 4.0e-2
   * - 32
     - 0.9994 / 1.0
     - 0.0003 / 2.7e-4
     - 0.0277 / 2.8e-2
   * - 64
     - 0.9997 / 1.0
     - 0.0002 / 2.0e-4
     - 0.0194 / 1.9e-2
   * - 128
     - 0.9998 / 1.0
     - 0.00002 / 4.9e-5
     - 0.0136 / 1.4e-2

A hard truncation returns the pulse **essentially intact**: free streaming is
anti-Hermitian, so :math:`\lVert g \rVert` is conserved and :math:`\lvert g_0
\rvert` can never exceed its initial value -- the reflection is as complete as
that bound allows, and the closure supplies no dissipation at all.

On these measurements a hypercollision at the GX normalization **outperforms the
reflectionless closure on both metrics**. The closure's merits are structural
rather than numerical: it carries no free parameter, and it is confined to
:math:`m = M`, so it cannot bias resolved moments the way a band operator can.
It is not a drop-in improvement on a well-tuned hypercollision.

**Generated figure.** Recurrence and the two absorbing treatments. Panel (d) shows
:math:`R_{M+1}\to1`, i.e. absorption that becomes exact with resolution.


The closure is selected through the Python API,
``linked_streaming_contribution(..., hermite_closure="reflectionless")``;
``"truncation"`` remains the default. Gates:
``tests/validation/physics_gates/test_hermite_hierarchy_physics.py``.

Curvature, grad-B, and mirror couplings
---------------------------------------

The magnetic drift terms follow a Laguerre-Hermite stencil: curvature
(``cv``) couples Hermite indices :math:`m\pm 2`, grad-:math:`B` (``gb``) couples
Laguerre indices :math:`\ell\pm 1`, and the mirror term couples :math:`m\pm 1`
and :math:`\ell\pm 1` with a :math:`b^\prime(\theta)` prefactor. These couplings
are applied directly to the gyrokinetic variable :math:`H_{\ell m}` built from
the non-adiabatic moments and the gyroaveraged potential.

Putting the pieces together, the linear operator is assembled from:

- **Streaming**: :math:`v_{th}\,\partial_z` with Hermite ladder couplings.
- **Mirror**: :math:`b'(\theta)` coupling across :math:`(\ell\pm1, m\pm1)`.
- **Curvature drift**: ``cv_d`` coupling across :math:`m\pm2`.
- **Grad-B drift**: ``gb_d`` coupling across :math:`\ell\pm1`.
- **Diamagnetic drive**: :math:`\omega_*` energy-weighted source in ``m=0,2``.

Operator toggles start from :class:`gkx.operators.linear.params.LinearTerms` and are
converted into one canonical :class:`gkx.terms.TermConfig` through
:func:`gkx.operators.linear.params.linear_terms_to_term_config`. The same modular RHS
path is then used by fixed-step linear integrators, Krylov operator
applications, and nonlinear IMEX linear solves.

The RHS is assembled in :mod:`gkx.terms` via
:func:`gkx.terms.assemble_rhs_cached`, which sums per-term kernels
(streaming, mirror, drifts, diamagnetic drive, collisions, hyper-collisions,
and end damping). This keeps the physics core branch-free and easier to extend,
while preserving JAX differentiability and performance.

For the explicit equations and per-parameter operator definitions, see
:doc:`operators`.

Field solve and electromagnetic coupling
----------------------------------------

Electrostatic runs solve quasineutrality for :math:`\phi` with optional
Boltzmann response (``tau_e``). Electromagnetic runs solve the coupled
quasineutrality/perpendicular-Ampere system for :math:`(\phi, B_\parallel)` and
then compute :math:`A_\parallel` from parallel Ampere’s law. The implementation
is in :mod:`gkx.terms.fields` and is called from
:func:`gkx.terms.assemble_rhs_cached`.

Normalization control
---------------------

``LinearParams`` exposes a ``rho_star`` factor that scales the perpendicular
wave numbers used in the drift and drive terms. This allows fine adjustments
of the effective :math:`k_\perp \rho` without changing the FFT grid spacing.

Diamagnetic drive
-----------------

The diamagnetic drive is written in the standard energy form,

.. math::

   \mathcal{D}_{\ell m} = i \omega_*\, J_\ell(b)\, \phi
   \left[1 + \eta_i \left(\mathcal{E}_{\ell m} - \frac{3}{2}\right)\right],

where :math:`\omega_* = k_y \, a/L_n`, :math:`\eta_i = (a/L_T)/(a/L_n)`, and
:math:`\mathcal{E}_{\ell m}` is the Hermite–Laguerre energy operator applied to
the basis. The coefficients are generated by
:func:`gkx.operators.linear.moments.diamagnetic_drive_coeffs`.

Time integration
----------------

The linear system is integrated using explicit fixed-step schemes (Euler, RK2,
RK4) implemented inside a ``jax.lax.scan`` loop. For higher-order Hermite-Laguerre
scans, the ``imex`` and ``implicit`` options provide additional stability by
treating damping terms implicitly. RK4 remains the default for the Cyclone
harness.

Boundary damping
----------------

For field-aligned domains with extended :math:`z` coverage, the linear operator
optionally applies a smooth end-cap damping profile (matching the analytic
linked-boundary taper used in flux-tube calculations). The damping profile is
controlled by:

- ``damp_ends_widthfrac``: fraction of the domain used for the taper.
- ``damp_ends_amp``: damping amplitude applied to :math:`H_{\ell m}`. Linear
  callers supplying ``dt`` use a per-step strength; nonlinear/RHS callers
  without ``dt`` use a rate. The exact scalar RK amplification and the pending
  normalization migration are documented in :doc:`operators`.

The damping is only applied to nonzonal modes (:math:`k_y>0`) and can be
disabled by setting ``damp_ends_amp = 0`` in ``LinearParams``.

Dealiasing
----------

Nonlinear E×B terms use the 2/3 de-aliasing rule in perpendicular Fourier space,
consistent with standard pseudo-spectral practice. The current implementation
applies the mask before and after the real-space bracket evaluation.

Nonlinear Electromagnetic Terms
-------------------------------

The nonlinear kernel evaluates gyro-averaged Poisson brackets in spectral space
and converts to real space only for the perpendicular derivatives. The E×B term
advects each Hermite–Laguerre moment with a gyro-averaged potential
:math:`\chi = J_0 \phi + J_1 b_\parallel` (implemented via :math:`J_l` and
:math:`J_l^B` in the Laguerre basis). The electromagnetic flutter contribution
uses :math:`\{g_m, J_0 A_\parallel\}` and couples adjacent Hermite moments with the
standard ladder factors, matching the GX nonlinear formulation. See
FC82, AL80, and GX in :doc:`references` for the governing electromagnetic
gyrokinetic equations and GX's implementation details.
