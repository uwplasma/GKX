Numerics
========

This page describes how GKX discretizes and integrates the operator stated in
:doc:`operators`, the defaults a run takes when a deck leaves a choice open,
and the numerical checks behind them. Solver selection and eigenvalue
certification are covered in :doc:`solvers`.

Spectral discretization
-----------------------

Perpendicular coordinates are Fourier modes on a uniform grid in :math:`x` and
:math:`y`; the parallel coordinate is resolved in real space along the field
line, and differentiated spectrally. Velocity space uses a Hermite-Laguerre
basis. The state layout for one species is

``(N_l, N_m, N_y, N_x, N_z)``.

Algorithm mapping (numerics → code)
-----------------------------------

- **Hermite–Laguerre pseudo-spectral expansion**:
  :mod:`gkx.core_velocity`.
- **Gyroaverage / polarization**:
  :func:`gkx.core_velocity.J_l_all`,
  :func:`gkx.operators.linear.moments.quasineutrality_phi`.
- **Spectral periodic derivative in z**:
  :func:`gkx.operators.linear.moments.grad_z_periodic`.
- **Hermite ladder streaming**:
  :func:`gkx.operators.linear.moments.streaming_term`.
- **Curvature / grad-B / mirror couplings**:
  :func:`gkx.operators.linear.rhs.linear_rhs_cached`,
  :func:`gkx.geometry.SAlphaGeometry.drift_components`,
  :func:`gkx.geometry.SAlphaGeometry.bgrad`.
- **Diamagnetic drive**:
  :func:`gkx.operators.linear.moments.diamagnetic_drive_coeffs`.
- **Time integration (explicit RK, IMEX, backward Euler)**:
  :func:`gkx.solvers_linear_integrators.integrate_linear`.
- **CFL-controlled explicit integration**:
  :func:`gkx.integrate_linear_explicit`
  (implemented in :func:`gkx.solvers_time_explicit.integrate_linear_explicit`).
- **Config-driven runner**:
  :func:`gkx.solvers_time_runners.integrate_linear_from_config`.
- **Nonlinear integration (explicit RK, IMEX)**:
  :func:`gkx.solvers_nonlinear_state_integration.integrate_nonlinear`.
- **Structured solves and bounded-memory Jacobians**:
  `SOLVAX <https://github.com/uwplasma/SOLVAX>`_ provides the reusable Krylov,
  tridiagonal, block-tridiagonal and autodiff primitives; GKX keeps the
  physical layout, coefficients, tolerances and acceptance policy.

JAX execution model
-------------------

- **JIT compilation**: ``jax.jit`` stages the time-stepping kernels in
  :func:`gkx.solvers_linear_integrators.integrate_linear`.
- **Loop fusion**: ``jax.lax.scan`` drives the time integration loop.
- **FFT grids**: ``jax.numpy.fft.fftfreq`` builds the wavenumbers in
  :func:`gkx.core_grid.build_spectral_grid`.
- **Krylov solves**: ``solvax.gmres`` (right-preconditioned FGMRES) serves the
  implicit linear and nonlinear IMEX time steps and the shift-invert inner
  solves, through one GKX policy adapter per call site. Nonlinear IMEX reverse
  mode wraps the tolerance-controlled solve in ``solvax.linear_solve``, whose
  implicit-function VJP solves the transposed system instead of
  differentiating the GMRES iterations. Plain and checkpointed two-step
  trajectories agree with centered finite differences, and a separate gate
  rebuilds the cache and matrix-free operator from a traced
  :math:`R/L_{T_i}` to check that the VJP carries both right-hand-side and
  operator dependence.
- **Hermite line solve**: ``solvax.tridiagonal_solve`` uses a deterministic
  Thomas recurrence on CPU and the fused JAX/vendor path on accelerators. GKX
  moves only the Hermite axis; every other dimension is an independent column.
- **Memory-bounded sensitivities**: ``solvax.chunked_jacfwd`` underlies the
  geometry gradient report when ``jacobian_chunk_size`` is set. Chunking
  changes batching and peak memory, not the JVP columns. Without a chunk
  request, ``jacobian_mode="auto"`` uses forward mode for few controls and
  reverse mode for few observables; the resolved mode is recorded and checked
  against finite differences.
- **Ladder couplings**: ``jax.numpy.roll`` and ``jax.numpy.pad`` implement the
  Hermite/Laguerre ladder couplings in
  :func:`gkx.operators.linear.moments.apply_hermite_v` and
  :func:`gkx.operators.linear.moments.apply_laguerre_x`.

These links are clickable in the HTML docs via the ``viewcode`` extension.

Structured solver dependency contract
-------------------------------------

GKX requires ``solvax>=0.22.0``; ``pyproject.toml`` is the only place that
floor is declared. Version 0.22.0 is the first release that exports every
SOLVAX name GKX imports. The binding one is ``block_thomas_factor_ops``, the
operator-coupling Schur elimination behind the ``pr3-cm`` preconditioner's
exact z-block solve, which first ships there together with
``block_thomas_solve_ops``, the solve the tests pin GKX's unrolled
substitution against. The eigenpair, propagator, and sparse-operator
interfaces (``adaptive_eigenpair``, ``eigenpair_reverse``,
``estimate_rk4_timestep``, ``exponential_eigenpairs``,
``propagator_eigenpairs``, ``sparse_eigenpairs``, ``sparse_operator_matrix``)
are older, from 0.12.0, and the Krylov and structured-solve interfaces
(``gmres``, ``linear_solve``, ``tridiagonal_solve``, ``chunked_jacfwd``,
``SpluFactorization``) older still. The floor was checked against the released
wheels rather than inferred: every PyPI release from 0.12.0 to 0.21.0 lacks
``block_thomas_factor_ops``, and the ``pr3-cm`` unit tests fail on 0.21.0 and
pass on 0.22.0. CI installs the newest released SOLVAX, so it tests the latest
release rather than the floor.

Generic numerical algebra lives in SOLVAX; gyrokinetic state layout,
linked-boundary assembly, preconditioner coefficients, eigenbranch tracking,
transport windows, and physics gates stay in GKX. Implicit time stepping
exposes one FGMRES algorithm with explicit tolerance, restart, iteration-limit
and physical-preconditioner controls.

Time integration algorithms
---------------------------

The linear integrators, all in
:func:`gkx.solvers_linear_integrators.integrate_linear` and sharing the cached
operator data of :func:`gkx.operators.linear.cache_builder.build_linear_cache`:

- **Explicit** ``method="euler"``, ``"rk2"``, ``"rk4"``, ``"sspx3"``.
- **CFL-controlled explicit** (``integrate_linear_explicit``,
  ``ExplicitTimeConfig``): the step is recomputed from the linear
  max-frequency estimate with the benchmark-locked CFL rule, and growth rates
  come from the midplane ``phi`` ratio with the same convention as the tracked
  comparison data. Its defaults are ``method="rk4"``, ``fixed_dt=False``,
  ``cfl=0.9``.
- **Diagonal IMEX** ``method="imex"`` (first order) and ``method="imex2"``: the
  diagonal collision and hypercollision damping is implicit and every other
  term explicit.
- **Backward Euler + GMRES** ``method="implicit"``, preconditioned by the
  physical preconditioner named in ``implicit_preconditioner`` (see
  `Linear solver options`_).

``imex2`` is the `ARS(2,2,2) split
<https://doi.org/10.1016/S0168-9274(97)00056-1>`_ (section 2.6). For a diagonal
non-negative damping operator :math:`D` and explicit remainder :math:`E`, with
:math:`\gamma=1-1/\sqrt{2}` and :math:`\delta=1-1/(2\gamma)`:

.. math::

   Y &= (I + \gamma hD)^{-1}\left(y_n + \gamma hE(y_n)\right), \\
   y_{n+1} &= (I + \gamma hD)^{-1}\left[y_n
      + h\delta E(y_n) + h(1-\delta)E(Y) - h(1-\gamma)DY\right].

The split is second order and L-stable for pure damping. For linear :math:`E`
with :math:`D=0` its stability polynomial equals explicit midpoint's, with the
same stability restriction; coupled explicit/implicit stability is problem
dependent. Besides being a user method, ``imex2`` is the one-step propagator
inside the Krylov ``power`` route and the single-step propagator filter.

Nonlinear runs use an explicit scheme (``euler``, ``rk2``, the ``rk3``
variants, ``rk4``, ``sspx3``, ``k10``) or ``method="imex"``, which treats the linear operator implicitly with the same
GMRES solve and preconditioners and the nonlinear term explicitly. Reverse
derivatives of the IMEX route use the converged-system implicit derivative, so
the gradient does not depend on the Krylov iteration count except through
solve accuracy. The CFL prefactor ``cfl_fac`` resolves to 1.73 for
``rk3``/``sspx3``, 2.82 for ``rk4`` and 1.0 otherwise.

.. _first-run-defaults:

What a first run gets
---------------------

These are the defaults that decide whether a run started from a shipped deck,
or from a deck that leaves a section out, is fast and accurate without expert
flags. Each rests on a load-independent measurement -- residuals,
certification gates, step counts, growth rates -- recorded in
:ref:`q28-defaults` and ``plan/log.md``.

**Eigen route.** ``KrylovConfig.method = "adaptive"``, the residual-certified
propagator eigensolve. See "Why ``adaptive`` is the default, and not
shift-invert" in :doc:`solvers`.

**Precision.** ``complex64`` by default. On
``examples/linear/axisymmetric/cyclone.toml`` at its own resolution the float32
run returns :math:`\gamma = 0.09309106`, :math:`\omega = 0.28203276` against
float64's :math:`0.09309117`, :math:`0.28203273`, agreement to 1.2e-6 relative.
``JAX_ENABLE_X64=true`` widens the initial state to ``complex128`` and so
applies the float64 certification gate: residual 1.07e-14 against 1e-9, where
float32 reports 5.55e-6 against 1.19e-4. On that deck float64 costs about 17%
more resident memory (0.90 to 1.05 GiB).

**Time integrator and step size.** A deck that sets ``[time] dt`` keeps
``method = "rk2"`` at that fixed step. A deck that sets no ``dt`` gets
``method = "rk4"`` with ``fixed_dt = false``: the CFL controller chooses the
step, and the defaulted ``dt`` is only its initial guess. An explicit
``method`` or ``fixed_dt`` in the deck still wins. Every fixed-step linear path,
including ``solver = "explicit_time"``, warns when ``dt`` exceeds the estimated
CFL-stable step.

**Resolution.** When ``Nl`` and ``Nm`` are not set, a linear run takes
``(Nl, Nm) = (12, 24)`` and a nonlinear run ``(4, 8)``, from the CLI, from
``gkx.solve`` and from the runtime alike. Set them for your case regardless;
the fallback is a sensible starting point, not a convergence statement.

**Nonlinear horizon.** Diagnosed nonlinear runs stop at saturation by default
(``[time] run_to = "saturation"``), with ``t_max`` as the hard cap; see
`Saturation stop policy`_.

**Unchanged, and measured or inspected to be right.** Two-thirds dealiasing on
for nonlinear runs (``nonlinear_dealias = true``); ``dealias_kz = false``;
``collisions`` and ``hypercollisions`` enabled as terms with per-species
``nu = 0``, so a deck opts into collisionality rather than inheriting one;
``diagnostics_stride = sample_stride = 1``; nonlinear chunking at
``min(steps, 128)``; restart cadence from ``nsave = 10000``; the persistent
compilation cache on with ``min_compile_time_secs = 0``, which is what makes a
second run of the same deck skip compilation.

**The CPU FFT thread pool stays on.** GKX sets no
``--xla_cpu_multi_thread_eigen`` flag for user runs anywhere in ``src``; only
``tests/conftest.py`` pins it false, for bitwise reproducibility across the test
matrix. On an idle host the single-threaded pool measured 8--10% slower per
step, so pinning it for users would cost speed to buy a determinism they have
not asked for.

.. _q28-defaults:

Measurements behind the defaults
--------------------------------

All three measurements use the shipped Cyclone deck, float32, and count
propagator applies, steps, right-hand-side evaluations or velocity-space
degrees of freedom rather than wall time.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - default
     - value
     - accuracy
   * - ``KrylovConfig.power_iters``
     - 40
     - residual 9.50e-01; 200 applies give 9.47e-01; neither certifies
   * - ``TimeConfig`` pairing when the deck chose no ``dt``
     - rk4, CFL-controlled
     - -2.63e-04 relative to the certified eigensolve
   * - ``Nl``/``Nm`` linear fallback
     - (12, 24)
     - +0.400% of the tracked GX golden

**power_iters = 40.** ``KrylovConfig`` and the ``dominant_eigenpair`` signature
share one value, so the route compiles its iteration-static scan once. At
``(Nl, Nm) = (4, 8)``, ``ky = 0.3``, against that rung's certified adaptive
eigenpair :math:`\gamma = 0.10128645`:

=======  ==========  =========  =========
applies  gamma       residual   certified
=======  ==========  =========  =========
40       0.18119842  9.501e-01  no
200      0.12442227  9.467e-01  no
1000     0.08590183  4.358e-01  no
5000     0.10113729  6.182e-03  no
10000    0.10106588  5.959e-03  no
=======  ==========  =========  =========

The gate is 1.19e-4 and the ladder stalls near 6e-3, so no value certifies and
the count is chosen for cost. No shipped deck or Krylov contract selects
``method="power"``, ``shift_source="power"`` or ``fallback_method="power"``, and
with the default ``certify=True`` the route raises rather than return an
uncertified pair.

**Time pairing.** ``ExplicitTimeConfig.dt`` is a required field and
``TimeConfig.dt`` a defaulted one, so only a deck can leave the step unchosen.
On the same deck and rung, with the ``TimeConfig`` dataclass defaults:

===================  ==========  =====  =========  ==================
method / policy      gamma       steps  RHS evals  rel. error
===================  ==========  =====  =========  ==================
rk2, fixed dt=0.1    fails       --     --         FloatingPointError
rk4, fixed dt=0.1    fails       --     --         FloatingPointError
rk2, CFL-controlled  0.10126899  7806   15612      -1.72e-04
rk3, CFL-controlled  0.10126041  4512   13536      -2.57e-04
rk4, CFL-controlled  0.10125984  2768   11072      -2.63e-04
===================  ==========  =====  =========  ==================

The defaulted ``dt = 0.1`` is about 7.8x the CFL-stable 0.01281, so both
fixed-step arms overflow whatever the scheme. With the controller on, all three
schemes agree to better than 3e-4 and rk4 reaches the horizon in **29.1% fewer
right-hand-side evaluations**, exactly :math:`4/(2.82 \times 2)`. At a fixed
step rk4 has no step-size compensation for its four stages, so a deck that
chooses ``dt`` keeps rk2. ``fixed_dt`` is not flipped globally: fourteen
shipped decks and parity fixtures omit it and depend on ``True``. Every shipped
deck sets ``dt``, and ``tests/integration/runtime/test_runtime_config.py``
asserts it so no deck drifts onto the controller by accident.

**Resolution (12, 24).** Certified adaptive eigensolves at the deck's own
``ky = 0.3``, against the tracked GX golden :math:`\gamma = 0.09302951` in
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

Every rung is certified against the original operator at the 1.19e-4 float32
gate. ``(12, 24)`` has the same :math:`N_l N_m` as ``(24, 12)``, so the same
cost per propagator apply, for eleven times less error. It is a balance, not
"more Hermite wins": ``Nl = 8`` is worse than both at either Hermite count,
because the parallel phase mixing that sets an ITG rate needs Hermite
resolution while the FLR response still needs enough Laguerre. Only
``(16, 48)``, at 2.67x the cost, clearly improves on 288, and a cost increase of
that size is not a fallback's decision to make. A deck whose physics runs the
other way, such as the shipped ETG decks at ``Nl = 24``/``Nm = 8``, says so.

**Limits.** All three measurements are one deck, one ``ky`` and one geometry.
The resolution ladder is certified but is not a convergence proof: it is
non-monotone through 288--384, so ``+0.400%`` is where this case lands at that
budget, not an error bound for another case.

.. _saturation-stop:

Saturation stop policy
----------------------

A diagnosed nonlinear run with ``[time] run_to = "saturation"`` (the default)
integrates in chunks and, after each chunk, asks
:func:`gkx.diagnostics.saturation.saturation_stop_decision` whether the
accumulated heat-flux trace has produced a mean worth stopping on. The run stops
when every gate passes, or when it reaches ``t_max`` (or the step budget), which
stops the loop without claiming saturation. :doc:`inputs` describes the
controls; this is the algorithm.

1. **Spin-up removal.** Everything before the trace first reaches its own
   median is dropped. During linear growth the flux sits below its saturated
   level, so once a plateau exists the median lies on it and the first
   crossing marks the end of growth. The overshoot decay that follows is left
   to the stationarity gate rather than trimmed.
2. **Correlated error.** The window mean's standard error uses the Sokal
   integrated autocorrelation time :math:`\tau` (shared with post-hoc window
   analysis in :mod:`gkx.diagnostics.analysis`):
   :math:`\mathrm{SEM} = s/\sqrt{n_\mathrm{eff}}` with
   :math:`n_\mathrm{eff} = \min(n,\; n\,\Delta t/(2\tau))`.

The run is saturated only when all of the following hold:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Gate (reason recorded when it fails)
     - Condition
   * - ``tau_ac_unresolved``
     - the autocorrelation crosses zero inside the window, and
       :math:`\tau > \Delta t`; a trace that decorrelates within one sample has
       shown noise, not a correlation time
   * - retained samples
     - at least 256 samples after spin-up (``min_samples``, floored at 256)
   * - ``window_below_min_window``
     - window span at least :math:`20\tau`, and at least
       ``saturation_min_window`` when set
   * - ``rel_sem_above_threshold``
     - :math:`\mathrm{SEM}/|\text{mean}| \le` ``saturation_rel_sem``
       (default 0.05)
   * - ``window_not_stationary``
     - first- and second-half means agree within twice their combined SEM
   * - ``guard_not_stationary``, ``Wg_guard_not_stationary``
     - the same half-window agreement holds for the field energy ``Wphi`` and
       the free energy ``Wg`` over the same window; the guards have no SEM
       gate, they stop a flat-looking flux from ending a run whose energies
       still drift
   * - ``flux_indistinguishable_from_zero``
     - :math:`|\text{mean}| > 10^{-9}`, so a trace that never leaves zero (a
       zonal-response case, for instance) runs to ``t_max`` instead of stopping
       in its first chunk

The twice-SEM stationarity factor is deliberate: a one-SEM gate would reject
about a third of genuinely stationary windows and only prolong runs until they
pass by chance. These are operational finite-sample safeguards, not a
sequential-confidence guarantee, and they are not qualified for irregular
adaptive-time sampling. The decision, its window, ``mean``, ``sem``,
``tau_ac`` and the failing ``reasons`` are returned with the result and written
under ``saturation`` in the summary JSON.

A run with diagnostics off, or whose whole fixed-step budget is shorter than
the minimum sample count, runs to ``t_max``. A prepared simulation
(``gkx.prepare``) compiles one scan of a fixed length, so it refuses
``run_to = "saturation"`` unless given ``steps=N`` or a deck with
``run_to = "t_max"``.

Distributed state sharding
--------------------------

Progress reporting is off by default; set ``TimeConfig.progress_bar=True`` (or
``progress_bar=True`` in the integrator call) to enable it.

For distributed runs, set ``TimeConfig.state_sharding = "auto"`` (or ``"ky"`` /
``"kx"``) to partition the packed nonlinear state over several JAX devices
through ``integrate_nonlinear_sharded``. With one visible device the request is
ignored and the run proceeds on that device through the same control flow. The
config path rejects ``"z"`` sharding, because the multi-device FFT-axis
decomposition has not passed its identity gate, and the sharded path refuses a
non-default ``collision_operator``. On macOS,
``XLA_FLAGS=--xla_force_host_platform_device_count=2`` emulates several CPU
devices for non-FFT-axis checks, but the nonlinear whole-state ``pjit`` profile
skips active multi-device CPU sharding by default because JAX/XLA CPU FFT
layouts can abort before Python can catch the failure. Multi-GPU artifacts are
the reference for active nonlinear state-sharding diagnostics, and a speedup
claim needs its own identity, transport-window and profiler gates.

Nonlinear FFT bracket
---------------------

The nonlinear :math:`E\times B` term is evaluated pseudospectrally with
FFT-based perpendicular derivatives. By default
(``TimeConfig.compressed_real_fft = true``) GKX computes gradients from the
Nyquist-compressed (``N_y/2+1``) spectrum in the benchmark-compatible layout:
non-negative ``k_y`` (including positive Nyquist when ``N_y`` is even) and a
positive Nyquist multiplier on the ``k_x`` axis when ``N_x`` is even. The
result is expanded back to the full :math:`k_y` axis once. This matches the
tracked nonlinear reference layout and minimizes memory traffic. Set
``compressed_real_fft = false`` for the full complex FFT bracket.

Electromagnetic runs stack the gyroaveraged ``J0*phi``, ``J0*apar`` and the
``bpar`` correction into one FFT batch, so one rFFT/iFFT pipeline per step
serves every channel. The Laguerre/Bessel factors on the quadrature grid
(``J0`` and ``J1/alpha``) are precomputed once per grid and cached. With
``TimeConfig.laguerre_nonlinear_mode = "spectral"`` the bracket skips the
Laguerre quadrature transform and uses the spectral gyroaverage factors ``Jl``
directly; the default ``"grid"`` applies the transform.

The ``ky`` layout contract
--------------------------

Every spectral array in GKX has the shape ``(..., ky, kx, z)``, and its ``ky``
axis is in one of two layouts. :mod:`gkx.core_ky_layout` states the rule and
owns every conversion between them; nothing else writes a Hermitian completion
by hand.

``full``
  ``Nky = Ny``, the two-sided ``fftfreq`` order
  :math:`[0, 1, \dots, N_y/2 - 1, -N_y/2, \dots, -1]`. The runtime evolves this
  layout. Half of it is redundant, because a real field obeys the reality
  condition :math:`F(-k_y, -k_x, z) = F^{*}(k_y, k_x, z)`.

``half``
  ``Nky = Nyc = 1 + Ny // 2``, the non-negative ``rfftfreq`` rows. The reality
  condition holds by construction. GX, stella and GS2 evolve this layout, and
  GKX uses it for restart files, NetCDF output, the ``ky`` spectra and
  real-space snapshots.

The boundary between the two sits at I/O and at the nonlinear bracket, not in
the middle of the step. Restart files store ``Nyc`` rows and
:func:`gkx.core_ky_layout.to_full` widens them on read; the compressed bracket
computes on ``Nyc`` rows
(:func:`gkx.operators.nonlinear.brackets._spectral_bracket_half_core`) and
widens its result once; the Hermitian projector applied after each
Runge--Kutta stage is exactly ``to_full(to_half(G))``.

Three properties of the contract are easy to get wrong, so they are stated and
tested rather than rederived at each call site.

**Nyc does not determine Ny.** Both :math:`N_y = 2(N_{yc}-1)` and
:math:`N_y = 2N_{yc}-1` give the same :math:`N_{yc}`, so a stored half-spectrum
array cannot say how long its own full axis is. Every widening takes
``ny_full`` explicitly; inferring the even branch would expand an odd-``Ny``
restart to ``Ny - 1`` rows.

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
representative weight the flux kernels use; they carry the pair factor of two
themselves and so take 1 on a folded pair and 0.5 on a self-conjugate row.
Finding the Nyquist row needs :math:`N_y`, which a half axis cannot supply, so
:class:`gkx.core_grid.SpectralGrid` and the linear cache carry ``ny_full``: the
length of the two-sided axis their rows were taken from, or ``None`` when the
rows are a selection of modes rather than a complete axis.

An even grid's Nyquist row is a flux **representative in both layouts**, at the
self-conjugate weight. Its flux contribution is odd under the sign convention
of that row, because the flux kernel carries an explicit :math:`\mathrm{i}k_y`,
but it is identically zero on any state representing a real field: on a
self-conjugate row the reality condition reads :math:`F(k_x) = F^{*}(-k_x)`,
under which the summand is odd in :math:`k_x` and cancels pairwise. No run
reaches the row in any case, since two-thirds dealiasing zeroes everything at
or above :math:`N_y/3`.

Below the runtime the ``half`` layout works end to end:
:func:`gkx.core_grid.build_spectral_grid` takes a ``ky_layout``, the bracket
computes on the stored rows without widening, and the per-stage projector is
the identity. The runtime does not select it; :doc:`performance` records what
it saves.

Equilibrium-flow shearing coordinates
-------------------------------------

This is a research path, not an input-file option. The coordinate kernel
:func:`gkx.operators.nonlinear.projection.advance_shearing_coordinates` follows
a shearing wave,

.. math::

   k_x^*(t) = k_x(0) - k_y\,\gamma_E t.

It models perpendicular equilibrium-flow decorrelation only; it does not
include a parallel-velocity-gradient drive, which needs its own normalization,
instability and transport gates [Schekochihin12]_ [Ball19]_. Shear is expected
to decorrelate turbulence once the shearing rate exceeds the instability rate
[Biglari90]_ [Waltz95]_; the response gate at the end of this section tests
that expectation.

When the displacement crosses half a radial Fourier cell, the state moves to
the nearest :math:`k_x` mode. The residual sub-cell displacement is kept in
both the effective wavenumber and the real-space phase
:math:`\exp(i\,\delta k_x x)`, and modes leaving the two-thirds retained band
are zeroed rather than wrapped. Keeping the residual phase implements the
corrected-remap principle and avoids the non-convergent smeared coupling of
integer-only remapping [McMillan19]_. The integer nearest-mode decision is
piecewise constant and uses a stopped tangent; the continuous wavenumbers and
phases stay differentiable between remap events. Tests cover zero-shear
identity, the analytic shearing-wave trajectory, norm-preserving forward and
inverse remaps, the dealias boundary, and JAX tangents with respect to
:math:`\gamma_E` and the radial scale against centered finite differences.

:func:`gkx.operators.linear.cache_builder.update_linear_cache_for_sheared_kx`
rebuilds :math:`k_\perp^2`, drift frequencies, gyroaverages, Bessel tables,
field-solve inputs, bracket multipliers and hyperdiffusion from the
two-dimensional effective :math:`k_x` grid, for periodic and standard linked
tubes. On a linked tube the displacement is constant along a fixed-:math:`k_y`
chain, so the chain topology is unchanged; non-twist tubes are refused because
their radial coordinate depends on :math:`z`. The full-complex bracket applies
the residual phase between the :math:`k_x` and :math:`k_y` FFTs, and the
compressed bracket evaluates the same state in canonical shearing coordinates;
the two agree within ``2e-5`` relative. The full-complex state is projected onto
its Hermitian subspace after every remap and stage.

The integrators are
:func:`gkx.solvers_nonlinear_state_integration.integrate_nonlinear_sheared`
(midpoint RK2 and Heun RK3, each stage evaluated in its own remapped basis, and
fixed-step ``method="imex"`` solving
:math:`[I-\Delta t L(t_{n+1})]G_{n+1} = G_n^* + \Delta t\,N(G_n,t_n)^*`) and
:func:`gkx.solvers_nonlinear_state_integration.integrate_nonlinear_sheared_transport`,
which records the per-species gyro-Bohm heat flux at every accepted step in a
``ShearedTransportTrace`` without storing distribution or field histories.
``fixed_dt=False`` applies the production nonlinear CFL policy. Both RK routes
recover their designed orders, IMEX recovers first order, and all three are
identical to the static paths at zero shear. Sheared IMEX refuses a custom
collision operator.

Treatment effects are judged with :func:`gkx.matched_nonlinear_transport_report`,
not by comparing two chaotic traces: baseline and treatment must each pass
post-transient finite-sample, running-mean drift, terminal-mean, block-count
and SEM gates before a relative reduction and its quadrature-SEM separation
are reported.

**The response gate fails, so flow shear stays out of input files.**
``docs/_static/flow_shear_fixed_step_response_gate.json`` records a Cyclone ITG,
adiabatic-electron, ``64x64x24``, ``Nl=4``, ``Nm=8`` periodic case at
``dt = 0.02`` to ``t = 300``, windowed over ``t = [240, 300]``, with
:math:`\gamma_E = 0` against :math:`0.01`:

.. list-table::
   :header-rows: 1
   :widths: 30 25 25 20

   * - integrator
     - :math:`\gamma_E = 0`
     - :math:`\gamma_E = 0.01`
     - windows pass
   * - GKX fixed-step IMEX
     - 15.451 ± 0.263
     - 16.195 ± 0.160
     - no
   * - independent fixed-step RK4
     - 11.715 ± 0.216
     - 14.624 ± 0.141
     - yes

Both show an increase in heat flux (4.82% and 24.82%) rather than the required
5% reduction at two combined SEMs. This rejects the weak-shear suppression
claim; it does not reject the gated coordinate, remap, operator, integrator or
derivative identities above.

De-aliasing and hyperdiffusion
------------------------------

Nonlinear brackets use the standard ``2/3`` de-alias mask, stored on the
spectral grid and applied inside the bracket. Perpendicular hyperdiffusion
(``D_hyper``, ``p_hyper_kperp`` and the ``hyperdiffusion`` term weight; see
:doc:`operators`) is available as scale-selective damping and is off by
default.

Linear solver options
---------------------

- **Batched ky scans**: pass ``ky_batch>1`` to the benchmark scan helpers to
  integrate several ky values at once on a sliced ky grid.
  ``fixed_batch_shape=True`` (default) edge-pads the final batch so a short
  tail batch does not recompile.
- **Donated buffers**: time integrators donate state buffers in JIT-compiled
  paths to reduce allocations.
- **Implicit preconditioners** (``implicit_preconditioner``):
  ``"auto"``/``"diag"``/``"physics"``/``"block"`` (full diagonal),
  ``"damping"`` (collisional/hyper only), ``"pas"`` (PAS line),
  ``"pas-coarse"`` (line plus coarse correction in kx/linked-kx chains),
  ``"hermite-line"`` (Hermite streaming line solve in ``m`` at fixed
  :math:`k_z`), ``"hermite-line-coarse"`` (line solve plus kx-coarse
  correction), or ``"identity"``.
- **Shift-invert preconditioners** (``KrylovConfig.shift_preconditioner``) for
  the inner solves of ``(A - \sigma I)^{-1}``: ``"auto"`` (default) takes
  ``"hermite-line"`` on electrostatic runs, with a ``"field-corrected"``
  retry, and ``"field-corrected"`` on electromagnetic ones; ``"damping"``
  (element-wise collisional/hyper damping); ``"hermite-line"`` (FFT in
  :math:`z` plus a tridiagonal Hermite solve of the additive
  diagonal-and-streaming symbol, :math:`O(n)` storage and work);
  ``"field-corrected"`` (the Hermite line plus the exact linear field response
  in a Woodbury capacitance solve, trading a larger setup for fewer Krylov
  iterations in field-dominated cases); or ``"pr3-cm"`` (below). The complex
  shift scaling is shared with backward Euler.
- **Outer acceptance**: every returned shift-invert pair is checked with the
  matrix-free relative residual
  :math:`\lVert Av-\lambda v\rVert/\max(\lVert Av\rVert,|\lambda|\lVert v\rVert)`
  against ``KrylovConfig.shift_outer_residual_tol`` (default ``1e-6``).
  Rejected primary and fallback pairs raise instead of returning a plausible
  frequency with an unconverged eigenvector.
- **Arnoldi breakdown**: a candidate basis direction is kept only when its norm
  exceeds a dtype-scaled threshold relative to the applied operator, so exact
  and numerical happy breakdown end the subspace instead of amplifying roundoff
  into a spurious mode.
- **Physical Ritz refinement**: after selecting a shift-invert Ritz vector, the
  solver recomputes its eigenvalue as the physical-operator Rayleigh quotient,
  which minimizes the Euclidean residual for that vector and removes the error
  of mapping an inexact inverse Ritz value through
  ``lambda = sigma + 1 / mu``.
- **Targeted mode selection**: ``KrylovConfig.mode_family`` (``"cyclone"``,
  ``"etg"``, ``"kbm"``) and ``KrylovConfig.shift_selection`` stabilize branch
  selection in stiff spectra. KBM uses the positive reported-frequency
  convention of the tracked benchmark table, so its matrix eigenvalue target
  lies on the negative imaginary axis. ``KrylovConfig.fallback_method``
  (default ``"propagator"``) applies when shift-invert returns a non-finite,
  strongly damped or high-residual mode. Generic retained-Ritz thick restarts,
  projected Jacobi--Davidson corrections and reduced field/moment Schur
  complements were tried for interior KBM eigenpairs and all selected the wrong
  branch; none is shipped.
- **Reusable IMEX operators**: nonlinear IMEX runs can prebuild the matrix-free
  linear operator with
  :func:`gkx.operators.nonlinear.policies.build_nonlinear_imex_operator` and pass
  it to
  :func:`gkx.solvers_nonlinear_state_integration.integrate_nonlinear_imex_cached`
  as ``implicit_operator``. With ``apar = bpar = 0`` the IMEX fixed-point and
  post-step field paths use the same electrostatic compiled RHS as the explicit
  nonlinear RHS.

Shift-invert inner solves
^^^^^^^^^^^^^^^^^^^^^^^^^

The inner solves are right-preconditioned FGMRES, so the least-squares norm is
the physical residual :math:`\lVert b - (A-\sigma I)x\rVert`, and they start
from :math:`x = 0`. Under right preconditioning the first Krylov vector is
already :math:`M^{-1}b`, so a cycle from zero minimizes over a space that
contains that point, and seeding with :math:`M^{-1}b` can only start the solve
behind the trivial guess when :math:`M` is weak. Measured on the first
right-hand side of the outer Arnoldi on the shipped Cyclone deck at
``(Nz, Nl, Nm) = (96, 4, 8)``, float64, as relative residuals:

.. list-table::
   :header-rows: 1
   :widths: 28 18 18 18 18

   * - preconditioner
     - :math:`x_0 = M^{-1}b`
     - GMRES(20) x 3 from :math:`M^{-1}b`
     - GMRES(20) x 3 from 0
     - 300 unrestarted from 0
   * - ``hermite-line``
     - 25.4
     - 9.18
     - 0.9951
     - 0.9047
   * - ``field-corrected``
     - 20.7
     - 13.0
     - 0.9995
     - --
   * - ``damping``
     - 3.53
     - 2.16
     - 0.8575
     - --
   * - ``pr3-cm``
     - 1.29
     - 0.304
     - 0.2990
     - 9.92e-05 (converged in 252)

The seeded guess inflated the reported inner residual by up to 25x without
changing the solve. With it removed, ``hermite-line`` still needs more than 600
unrestarted iterations for a ``1e-4`` tolerance and ``pr3-cm`` needs 252; at
the default ``shift_restart = 20`` even ``pr3-cm`` stalls, so raise
``shift_restart`` and ``shift_maxiter`` together. :doc:`solvers` records what
this means for the choice of eigen route.

The ``pr3-cm`` preconditioner
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``pr3-cm`` (``src/gkx/solvers_linear_precond_pr3.py``) splits the operator into
two halves that are each invertible in closed form and sweeps between them.
The Hermite line solve inverts streaming, hypercollisions and the *z*-mean of
the drift diagonal exactly. Everything else -- the exact :math:`\omega_d(z)`,
the mirror term, the drive, the end damping, collisions and the local field
response -- is *z*-local, so it is a batch of dense :math:`(l, m)` blocks, one
per ``(species, ky, kx, z)``. With :math:`s_1 = \sigma/2 - \alpha` one
Peaceman--Rachford double sweep of the two shifted halves costs one solve of
each and no operator application, and ``pr3-cm`` is three such sweeps started
from zero. The parameter follows the scalar symbol rule
:math:`\alpha = -\sqrt{s_1 d}` (``s_1`` the streaming-plus-hypercollision
symbol bound at the smallest non-zero :math:`|k_z|`, ``d`` the spectral radius
of the local blocks) unless ``KrylovConfig.shift_precond_alpha`` names one.

The z-local block is solved exactly by block-Thomas elimination in the Laguerre
index plus one Sherman--Morrison correction for the rank-one field part.
``solvax.block_thomas_factor_ops`` owns the Schur elimination. The Laguerre
couplings are banded in Hermite, so the factors store :math:`N_l N_m^2` Schur
inverses plus :math:`2 N_l (2p+1) N_m` coupling diagonals for Hermite
half-width :math:`p` (measured as 1 on the tested operator). That is 1.5x
smaller than a dense-band factorization at :math:`N_l N_m = 36` and 2.7x at
768, where the dense :math:`(N_l N_m)^2` inverse is 3.0x and 14.2x larger.

The exact solve holds only while the block is *l*-tridiagonal with a rank-one
field part and a Hermite band narrower than ``Nm``. The build measures all three
at every shift, and ``KrylovConfig.shift_precond_block_solve`` decides what
happens when they fail: ``"auto"`` (default) falls back to the dense batched
inverse -- the same preconditioner, a costlier apply -- and records why in
``EigenSolveStatus.inner["preconditioner_setup"]``; ``"block-thomas"`` refuses
instead of falling back; ``"dense"`` is the control. A collision operator that
couples the Laguerre index beyond :math:`l \pm 1` is the usual cause of the
fallback.

Setup costs :math:`N_l N_m` probes of the z-local operator and a host
factorization, once per shift, handed to the compiled Arnoldi as an operand.
The build also requires the non-streaming operator to *be* z-local, checks it
against the operator on a random vector, and **refuses** rather than
approximates when it is not. A grid carrying zonal
:math:`(k_y = 0, k_x > 0)` rows under adiabatic electrons fails that check,
because their :math:`\langle\phi\rangle` is a sum over *z*; the linear eigen
route escapes it by reducing the grid to one non-zero :math:`k_y`.

Automatic solver and fit-signal selection
-----------------------------------------

The runtime linear drivers accept ``solver = "auto"`` (the default) and
``fit_signal = "auto"``. The auto solver runs the time path and falls back to
the Krylov eigensolve when the fitted ``(gamma, omega)`` is non-finite or
violates ``require_positive``. The KBM benchmark helper resolves ``"auto"`` to
the solver locked for each tracked ``ky``. The auto fit signal computes both the
``phi`` and density time traces, scores each with the same windowing rules
(``R^2`` of the log-amplitude and phase fits plus an optional growth-rate
weight), and keeps the higher score; auto mode therefore disables streaming
fits and stores the minimal traces the comparison needs.

Set ``solver = "time"``, ``"explicit_time"`` or ``"krylov"`` and
``fit_signal = "phi"`` or ``"density"`` to override these, together with custom
fit-window parameters.

Custom collision operators
--------------------------

The shipped collision models, their verification and the Python collision
protocol are documented in :doc:`operators`. A custom operator is any
JAX-compatible object implementing ``apply(context)``, passed as
``collision_operator=`` to ``linear_rhs``, ``linear_rhs_cached``,
``integrate_linear``, or ``nonlinear_rhs_cached``. It returns the unit-weight
collisional contribution with the state shape; the collision term weight
multiplies it, the built-in collision term is switched off, and hypercollisions
stay independent.

Custom operators run on serial explicit and IMEX linear integration (without
donated state buffers), cached linear and nonlinear RHS evaluation, and serial
explicit nonlinear integration. Implicit linear solves, nonlinear IMEX, sheared
IMEX and decomposed state integration refuse them. Built-in
``collision_split`` applies only to diagonal hypercollisions; conserving
collisions stay in the assembled RHS so their low-order field-particle terms
cannot be dropped by diagonal operator splitting.

The finite-Larmor-radius collision coefficients contain deeply nested,
cancellation-sensitive sums. GKX generates them offline in multiple-precision
arithmetic (``tools/artifacts/build_linear_validation_artifacts.py`` and
``tools/artifacts/build_finite_wavelength_coulomb_data.py``), records
normalization, provenance and checksums with each table, and loads only compact
arrays into the JAX runtime. The frequency-normalization trap is a real one:
Frei, Hoffmann & Ricci write the Bessel argument as :math:`B=k_\perp\sqrt{2\tau}`,
so their :math:`k_\perp=0.5` point is :math:`B=1/\sqrt{2}` at :math:`\tau=1`;
the runtime interpolates at :math:`B=\sqrt{2b}` from the cached :math:`b`.

The tracked collision evidence is:

- ``docs/_static/collision_operator_verification.json``: offline Coulomb
  algebra, conservation, entropy and truncation gates;
- ``docs/_static/collision_response_convergence.json``: the drift-kinetic
  driven-current hierarchy through :math:`(P,J)=(20,5)`;
- ``docs/_static/collision_finite_wavelength_itg_convergence.json``: the
  homogeneous-slab ITG growth-convergence gate at the paper wavelength;
- ``docs/_static/collision_finite_wavelength_zonal_response.json`` and
  ``collision_finite_wavelength_zonal_velocity_sections.csv``: the collisional
  zonal-response protocol of Frei, Ernst & Ricci (2022), Figures 12--14, at
  :math:`(P,J)=(24,10)`, with the moment-hierarchy
  (``collision_finite_wavelength_zonal_moment_hierarchy.json``,
  ``collision_finite_wavelength_zonal_P21_P24_gate.json``) and interpolation
  (``collision_finite_wavelength_zonal_b_grid_pilot.json``) gates behind it.

The zonal protocol uses ion--ion collisions at :math:`q=1.4`,
:math:`\epsilon=0.1`, :math:`\nu_i^*=3.13`, whose Xiao long-time residual is

.. math::

   R_z(\infty) = \frac{\epsilon^2/q^2}{1 + \epsilon^2/q^2} = 0.00508,

deliberately distinct from the collisionless Rosenbluth--Hinton value. The
paper normalization converts as
:math:`\nu = \nu_i^*\epsilon^{3/2}/(\sqrt{2}q) = 0.0499921`, so
:math:`t\nu=30` is solver time of about 600. The geometry and time contract is
``benchmarks/collisional_zonal_response.toml``, and
``tools/artifacts/build_zonal_flow_artifacts.py`` owns the traces and the gate.

Landau damping against the exact kinetic roots
----------------------------------------------

The sharpest available check on the Hermite representation is the slab
gyrokinetic ion-acoustic dispersion relation with adiabatic electrons at
:math:`k_\perp \to 0`,

.. math::

   1 + \frac{T_i}{T_e} + \zeta Z(\zeta) = 0,
   \qquad
   \zeta = \frac{\omega}{k_\parallel \sqrt{2T_i/m_i}},

with :math:`Z` the Fried-Conte plasma dispersion function, solved from
``scipy.special.wofz`` to double precision. It has no free parameters.
``test_landau_root_recovered_by_collisional_extrapolation`` evolves GKX's
production linear operator at :math:`N_m = 64` and requires the
:math:`\nu \to 0` extrapolation to land within 1% in :math:`\gamma` and 0.5% in
:math:`\omega` of the exact root, at :math:`T_e/T_i = 1` and 10.

.. figure:: _static/landau_damping_validation.png
   :width: 100%

   Everything shown is produced by ``linear_rhs_cached``, GKX's production
   linear operator; panel (b) eigen-decomposes that same operator by applying it
   to Hermite basis vectors.

Measurement pitfalls
^^^^^^^^^^^^^^^^^^^^

Four traps produce plausible but wrong numbers here. Each is gated in
``tests/validation/physics_gates/test_hermite_hierarchy_physics.py``.

**1. A collisionless truncated Hermite system cannot Landau damp.** Free
streaming is anti-Hermitian, so the truncated matrix has a purely real
spectrum; the gate requires :math:`\max|\mathrm{Re}\,\lambda| < 10^{-11}`.
Whatever a fit returns from a collisionless run is a transient that ends at
recurrence, not an asymptotic rate. The gate also requires collisions to make
the spectrum genuinely damped, so an operator that does nothing cannot pass.

**2. The Landau root is not an eigenvalue.** It is a pole of the analytically
continued response function. The least-damped eigenvalue of the collisional
operator, extrapolated, misses it badly at :math:`T_e/T_i = 1`, because at
strong damping the discrete modes are ballistic rather than collective. Landau
damping is an initial-value phenomenon and is measured as one: add
Lenard-Bernstein collisions, measure the decay at several small :math:`\nu`,
and extrapolate :math:`\nu \to 0`.

**3. A density perturbation with no initial flow is a standing wave.** It splits
into left- and right-going sound waves, so the signal stays real and beats
through zeros, and unwrapping its phase returns a frequency far below the
root. Fitting :math:`A e^{\gamma t}\cos(\omega t + \phi)` returns the root.

**4. Temperature-ratio conventions.** GKX's ``tau_e`` is :math:`T_i/T_e`.
Passing the reciprocal is invisible at :math:`T_e = T_i` and moves the frequency
to the :math:`T_e/T_i = 0.1` branch at :math:`T_e/T_i = 10`;
``test_temperature_ratio_convention_is_pinned`` requires the
:math:`T_e/T_i = 10` frequency within 2%. The thermal speed carries a second
convention: :math:`Z` takes :math:`\zeta = \omega/(k\sqrt{2T/m})` while GKX
normalizes to :math:`v_{ti} = \sqrt{T/m}`, a factor :math:`\sqrt2` that still
looks physical.

Panel (a) shows the crossover that follows from trap 1: the Landau pole
dominates for a while, the ballistic part of the initial condition then leaves
a plateau, and the revival at :math:`t_{\mathrm{rec}} = 2\sqrt{N_m}` ends the
useful window. ``test_recurrence_time_follows_the_square_root_law`` locates
the revival at the same multiple of :math:`2\sqrt{N_m}` for :math:`N_m = 16`
and 64. Regenerate the figure with
``tools/artifacts/build_landau_damping_figure.py``.

Hermite closure and recurrence
------------------------------

The streaming ladder couples :math:`m` to :math:`m\pm1`, so it must be
terminated at the last retained moment :math:`m = M`. The default termination
sets :math:`G_{M+1} = 0`. That is a *reflecting* boundary in Hermite space:
free energy streaming up the ladder reaches the end and propagates back down,
returning to the low moments as **recurrence** at

.. math::

   t_{\mathrm{rec}} \simeq \frac{2\sqrt{M}}{k_\parallel v_{th}}.

Because :math:`t_{\mathrm{rec}}` grows only as :math:`\sqrt{M}`, refining the
velocity grid is an inefficient remedy; the end of the ladder must absorb the
outgoing flux instead. A hard truncation absorbs nothing: free streaming
conserves :math:`\lVert g \rVert`, so :math:`\lvert g_0 \rvert` cannot exceed
its initial value, and the gate requires the truncated revival above 0.99 of
it.

GKX provides two absorbing treatments.

**Hypercollisions** (on by default, ``[physics] hypercollisions = true``) damp
a band of high moments. A TOML run uses the :math:`|k_z|`-scaled Hermite
branch, :math:`\propto \nu_{hyper,m}\,|k_z|\,(m/M)^{p_m}`, set by
``nu_hyper_m`` and ``p_hyper_m``; see the Hypercollisions section of
:doc:`operators` for every branch and its defaults.

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

Three properties make this a closure rather than a tuned dissipation:

- :math:`R_{M+1} \to 1 - 1/(4M)` as :math:`M` grows, so the absorption becomes
  exact with increasing resolution instead of requiring retuning.
- :math:`R_{M+1} > 0` makes the term a strictly positive-definite sink of free
  energy, so it cannot inject energy at any resolution.
- It is confined to :math:`m = M`, so unlike a band operator it cannot bias
  the resolved moments.

With three retained moments (:math:`M = 2`) the coefficient is
:math:`R_3 = \sqrt{8/\pi}/\sqrt{3} = 0.921318`, the Hammett-Perkins three-pole
coefficient, which is an independent check on the family.

``test_absorbing_closures_beat_truncation_on_both_metrics`` measures both
treatments on the free-streaming hierarchy at :math:`k_\parallel v_{ti}=1`,
:math:`M = 64`, from :math:`\lvert g_0 \rvert = 1`, with hypercollisions at the
GX Appendix B normalization. Each must cut the revived :math:`\lvert g_0\rvert`
below a tenth of the truncation's, *and* keep the error before recurrence below
0.1 against a converged reference. The second metric is what makes the
comparison fair: revival suppression alone rewards any strong damping, so a
closure that flattened the whole hierarchy would score perfectly on it.

The closure is selected through the Python API,
``linked_streaming_contribution(..., hermite_closure="reflectionless")``;
``"truncation"`` is the default. It carries no free parameter, but it is not a
drop-in replacement for a well-tuned hypercollision.

Operator assembly
-----------------

The equations of every term are in :doc:`operators`; this section records how
they are put together.

Operator toggles start from :class:`gkx.operators.linear.params.LinearTerms` and
are converted into one canonical :class:`gkx.terms.TermConfig` through
:func:`gkx.operators.linear.params.linear_terms_to_term_config`.
:func:`gkx.terms.assemble_rhs_cached` sums the per-term kernels (streaming,
mirror, drifts, diamagnetic drive, collisions, hypercollisions and end damping),
and the same modular RHS serves the fixed-step linear integrators, Krylov
operator applications, and nonlinear IMEX linear solves.

- **Gyroaverage.** :math:`J_\ell(b) = (1/\ell!)(-b/2)^\ell e^{-b/2}` with
  :math:`b = k_\perp^2 \rho^2`, the Laguerre projection of the gyroaveraged
  potential.
- **Streaming.** The parallel derivative is spectral (FFT-based) and is applied
  to the non-adiabatic moments plus the explicit field terms,
  :math:`\tilde{G}_{\ell m} = G_{\ell m} + (Z_s/T_s) J_\ell \phi\,\delta_{m0}
  - (Z_s v_{th}/T_s) J_\ell A_\parallel\,\delta_{m1}
  + J_\ell^B B_\parallel\,\delta_{m0}`, before the Hermite ladder
  :math:`\sqrt{m+1} H_{m+1} + \sqrt{m} H_{m-1}`. This matches the ordering and
  ghost exchange of GX's ``grad_parallel_linked``.
- **Drifts and mirror.** Curvature couples :math:`m\pm 2`, grad-:math:`B`
  couples :math:`\ell\pm 1`, and the mirror term couples :math:`m\pm 1` and
  :math:`\ell\pm 1` with a :math:`b'(\theta)` prefactor, all acting on
  :math:`H_{\ell m}`.
- **Diamagnetic drive.** The energy form
  :math:`\mathcal{D}_{\ell m} = i \omega_*\, J_\ell(b)\, \phi
  \left[1 + \eta_i (\mathcal{E}_{\ell m} - 3/2)\right]`, with
  :math:`\omega_* = k_y a/L_n` and :math:`\eta_i = (a/L_T)/(a/L_n)`; the
  coefficients come from
  :func:`gkx.operators.linear.moments.diamagnetic_drive_coeffs`.
- **Field solve.** Electrostatic runs solve quasineutrality for :math:`\phi`
  with an optional Boltzmann response (``tau_e``). Electromagnetic runs solve
  the coupled quasineutrality/perpendicular-Ampère system for
  :math:`(\phi, B_\parallel)` and then :math:`A_\parallel` from parallel
  Ampère's law, in :mod:`gkx.terms.fields`.
- **Normalization.** ``LinearParams.rho_star`` scales the perpendicular
  wavenumbers used in the drift and drive terms, which adjusts the effective
  :math:`k_\perp \rho` without changing the FFT grid spacing.
- **End damping.** A smooth taper over ``damp_ends_widthfrac`` of each end of
  the field line, applied to non-zonal modes only; its strength and units are
  described in :doc:`operators`.
- **Nonlinear terms.** Gyroaveraged Poisson brackets advect each moment with
  :math:`\chi = J_0 \phi + J_1 b_\parallel` (through :math:`J_\ell` and
  :math:`J_\ell^B` in the Laguerre basis), and the flutter term
  :math:`\{g_m, J_0 A_\parallel\}` couples adjacent Hermite moments with the
  standard ladder factors, following the GX nonlinear formulation. See FC82,
  AL80, and GX in :doc:`references`.
