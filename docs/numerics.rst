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

JAX execution model
-------------------

``jax.jit`` stages the integrators and ``jax.lax.scan`` drives the time loop.
Implicit time steps and shift-invert inner solves use ``solvax.gmres``
(right-preconditioned FGMRES); nonlinear IMEX reverse mode wraps it in
``solvax.linear_solve``, whose implicit-function VJP solves the transposed
system instead of differentiating the iterations. The Hermite line solve uses
``solvax.tridiagonal_solve``. ``solvax.chunked_jacfwd`` bounds memory for
geometry Jacobians; chunking changes batching, not the columns. Generic
algebra lives in SOLVAX (the floor is declared once, in ``pyproject.toml``);
state layout, linked-boundary assembly, preconditioner coefficients, branch
tracking, transport windows and physics gates stay in GKX. Module names are in
:doc:`design`.

Time integration algorithms
---------------------------

Linear integrators (:func:`gkx.solvers_linear_integrators.integrate_linear`)
share the cached operator data of
:func:`gkx.operators.linear.cache_builder.build_linear_cache`:

- **Explicit** ``rk2``, ``rk4``, ``sspx3``; **CFL-controlled**
  (``integrate_linear_explicit``), which recomputes the step from the linear
  max-frequency estimate (defaults ``rk4``, ``fixed_dt=False``, ``cfl=0.9``).
- **Diagonal IMEX** ``imex`` (first order) and ``imex2``: diagonal collision
  and hypercollision damping implicit, everything else explicit.
- **Backward Euler + GMRES** ``implicit``, with the physical preconditioner
  named in ``implicit_preconditioner``.

``imex2`` is the `ARS(2,2,2) split <https://doi.org/10.1016/S0168-9274(97)00056-1>`_.
For a diagonal non-negative damping :math:`D` and explicit remainder :math:`E`,
with :math:`\gamma=1-1/\sqrt{2}` and :math:`\delta=1-1/(2\gamma)`:

.. math::

   Y &= (I + \gamma hD)^{-1}\left(y_n + \gamma hE(y_n)\right), \\
   y_{n+1} &= (I + \gamma hD)^{-1}\left[y_n
      + h\delta E(y_n) + h(1-\delta)E(Y) - h(1-\gamma)DY\right].

It is second order and L-stable for pure damping; with :math:`D=0` its
stability polynomial is explicit midpoint's. Nonlinear runs use an explicit
scheme (``rk2``, ``rk3``, ``rk4``, ``sspx3``),
``imex`` (linear operator implicit by GMRES, bracket explicit) or the opt-in
``imex-ars3`` (:doc:`guidance`). The CFL prefactor ``cfl_fac`` is 1.73 for
``rk3``/``sspx3``, 2.82 for ``rk4``, 1.0 otherwise. Reverse derivatives of the
IMEX route use the converged-system implicit derivative.

.. _first-run-defaults:

What a first run gets
---------------------

Defaults for a deck that leaves a choice open, each backed by a
load-independent measurement (:ref:`q28-defaults`):

- **Eigen route:** ``auto``, banded direct shift-invert where it fits, else
  ``adaptive``, the residual-certified propagator eigensolve (:doc:`solvers`).
- **Precision:** ``complex64``. On ``01_linear_tokamak/case_full.toml`` float32
  gives :math:`\gamma = 0.09309106` against float64's :math:`0.09309117`
  (1.2e-6 relative); ``JAX_ENABLE_X64=true`` applies the float64 certification
  gate (1e-9 against 1.19e-4) at about 17% more memory.
- **Step:** a deck that sets ``[time] dt`` keeps ``rk2`` at that step; a deck
  without ``dt`` gets ``rk4`` with the CFL controller. Fixed-step linear paths
  warn when ``dt`` exceeds the CFL estimate.
- **Resolution:** ``(Nl, Nm) = (12, 24)`` linear, ``(4, 8)`` nonlinear, a
  starting point and not a convergence statement.
- **Nonlinear horizon:** diagnosed runs stop at saturation, with ``t_max`` as
  the cap (:ref:`saturation-stop`).
- **Unchanged:** two-thirds dealiasing; ``collisions`` and ``hypercollisions``
  enabled as terms with per-species ``nu = 0``; the persistent compilation
  cache on, so a second run skips compilation; the CPU FFT thread pool stays
  on (single-threaded measured 8--10% slower).

.. _q28-defaults:

Measurements behind the defaults
--------------------------------

Shipped Cyclone deck, float32, counting propagator applies, steps and
right-hand-side evaluations rather than wall time.

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

**power_iters.** At (4, 8), ``ky = 0.3`` the power ladder stalls near a residual
of 6e-3 (5000 applies) against the 1.19e-4 gate, so no count certifies and 40
is chosen for cost. No shipped deck selects the power route, and with
``certify=True`` it raises rather than return an uncertified pair.

**Time pairing.** The defaulted ``dt = 0.1`` is 7.8x the CFL-stable 0.01281, so
fixed-step rk2 and rk4 overflow. With the controller, rk2, rk3 and rk4 agree
within 3e-4 and rk4 needs 29.1% fewer right-hand-side evaluations (11072
against 15612 for rk2), exactly :math:`4/(2.82\times2)`. A deck that chooses
``dt`` keeps rk2, and every shipped deck sets ``dt``.

**Resolution.** Certified eigensolves at ``ky = 0.3`` against the GX golden
:math:`\gamma = 0.09302951`:

====  ====  ==========  ==============
Nl    Nm    gamma       rel. to golden
====  ====  ==========  ==============
8     24    0.09877566  +6.177%
24    12    0.08893196  -4.405%
12    24    0.09340143  **+0.400%**
12    32    0.09263792  -0.421%
16    32    0.09284505  -0.198%
24    24    0.09368346  +0.703%
16    48    0.09309106  +0.066%
====  ====  ==========  ==============

All rungs certify at the 1.19e-4 gate. The ladder is not monotone, so
(12, 24) is where this case lands at that cost, not an error bound for another
case. Parallel phase mixing needs Hermite resolution while the FLR response
needs enough Laguerre; a deck whose physics runs the other way, such as the
ETG decks at ``Nl = 24``, ``Nm = 8``, says so.

.. _saturation-stop:

Saturation stop policy
----------------------

A diagnosed nonlinear run with ``run_to = "saturation"`` (the default)
integrates in chunks and asks
:func:`gkx.diagnostics.saturation.saturation_stop_decision` after each whether
the heat-flux trace supports a mean. It stops when every gate passes, or at
``t_max``, without claiming saturation. Spin-up is removed up to the first
crossing of the trace's own median. The window mean's standard error uses the
Sokal integrated autocorrelation time :math:`\tau`:
:math:`\mathrm{SEM} = s/\sqrt{n_\mathrm{eff}}`,
:math:`n_\mathrm{eff} = \min(n,\; n\,\Delta t/(2\tau))`. All of these must hold:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Gate (reason recorded when it fails)
     - Condition
   * - ``tau_ac_unresolved``
     - the autocorrelation crosses zero inside the window and :math:`\tau > \Delta t`
   * - retained samples
     - at least 256 after spin-up
   * - ``window_below_min_window``
     - window span at least :math:`20\tau` and ``saturation_min_window``
   * - ``rel_sem_above_threshold``
     - :math:`\mathrm{SEM}/|\text{mean}| \le` ``saturation_rel_sem`` (0.05)
   * - ``window_not_stationary``
     - first- and second-half means agree within twice their combined SEM
   * - ``guard_not_stationary``, ``Wg_guard_not_stationary``
     - the same half-window test for ``Wphi`` and ``Wg``
   * - ``flux_indistinguishable_from_zero``
     - :math:`|\text{mean}| > 10^{-9}`

These are finite-sample safeguards, not a sequential-confidence guarantee, and
are not qualified for irregular adaptive sampling. The decision and its
reasons are written under ``saturation`` in the summary JSON. A prepared
simulation (``gkx.prepare``) compiles a fixed length, so it refuses
``run_to = "saturation"`` unless given ``steps=N``.

Distributed state sharding
--------------------------

``TimeConfig.state_sharding = "auto"`` (or ``"ky"``/``"kx"``) partitions the
packed nonlinear state over several JAX devices; with one device it is ignored.
``"z"`` is rejected (the FFT-axis decomposition has not passed its identity
gate) and the sharded path refuses a non-default ``collision_operator``. Active
multi-device CPU sharding is skipped on macOS because XLA CPU FFT layouts can
abort. A speedup claim needs its own identity, transport-window and profiler
gates.

Nonlinear FFT bracket
---------------------

The :math:`E\times B` term is pseudospectral. By default
(``compressed_real_fft = true``) gradients come from the ``N_y/2+1``
non-negative-:math:`k_y` spectrum, matching the reference layout and minimizing
memory traffic. Electromagnetic runs stack ``J0*phi``, ``J0*apar`` and the
``bpar`` correction into one FFT batch.

*Limitation (#336).* The ``"grid"`` Laguerre bracket advects with the node-space
:math:`J_0\phi` and keeps the first ``Nl`` coefficients, while quasineutrality
uses the analytic :math:`J_\ell`. The retained energy is then conserved only to
about :math:`7\times10^{-8}` of the bilinear scale at ``Nl = Nm = 4``. Treat
it as a resolution diagnostic, converged in ``Nl`` (Mandell, Dorland &
Landreman 2018, section 4.1).

The ``ky`` layout contract
--------------------------

:mod:`gkx.core_ky_layout` owns every conversion between two layouts of the
``ky`` axis. ``full`` is the two-sided ``fftfreq`` order with ``Nky = Ny``; the
runtime evolves it. ``half`` is the non-negative ``rfftfreq`` rows,
``Nyc = 1 + Ny // 2``; GX, stella and GS2 evolve it, and GKX uses it for
restarts, NetCDF output, spectra and snapshots. A real field obeys
:math:`F(-k_y, -k_x, z) = F^{*}(k_y, k_x, z)`. Three rules:

- ``Nyc`` does not determine ``Ny``: widening takes ``ny_full`` explicitly.
- Row 0, and row ``Ny/2`` for even ``Ny``, are self-conjugate:
  :math:`F(k_x) = F^{*}(-k_x)` within the row. Storing ``ky >= 0`` does not
  enforce it; operators that write those rows call
  ``symmetrize_self_conjugate_rows``.
- Reductions of quantities even in :math:`k_y` use
  ``ky_row_weights`` (2 on paired rows, 1 on self-conjugate rows), never a
  blanket factor of two, which over-counts the Nyquist row. A flux on that row
  vanishes for a real field, and two-thirds dealiasing zeroes it anyway.

``[grid] ky_layout = "half"`` is the default; ``"full"`` is the opt-out. On
Cyclone at ``24x24x24`` to ``64x64x24`` the two layouts agree to 2e-16 (x64) and
2e-7 (float32) in the nonlinear diagnostics, with even and odd ``Nyc``, with
``state_sharding = "ky"`` on four forced CPU devices, and bitwise in the
heat-flux window value and gradient; ``"half"`` runs the window gradient in 0.68
(CPU) to 0.72 (GPU) of the two-sided time (:doc:`performance`).

Equilibrium-flow shearing coordinates
-------------------------------------

A research path, not an input option.
:func:`gkx.operators.nonlinear.projection.advance_shearing_coordinates`
follows :math:`k_x^*(t) = k_x(0) - k_y\gamma_E t`. When the displacement
crosses half a radial cell the state moves to the nearest :math:`k_x`, the
sub-cell residual is kept in the wavenumber and the phase
:math:`\exp(i\,\delta k_x x)` (the corrected remap of [McMillan19]_), and
modes leaving the two-thirds band are zeroed. The remap is piecewise constant
with a stopped tangent; wavenumbers and phases stay differentiable between
remaps. Non-twist tubes are refused. There is no parallel-velocity-gradient
drive [Schekochihin12]_ [Ball19]_. Shear is expected to decorrelate turbulence
once the shearing rate exceeds the instability rate [Biglari90]_ [Waltz95]_;
the gate below tests that. Integrators:
``integrate_nonlinear_sheared`` (RK2, RK3, fixed-step IMEX) and
``integrate_nonlinear_sheared_transport``. Treatment effects are judged with
``matched_nonlinear_transport_report``, not by comparing two chaotic traces.

**The response gate fails, so flow shear stays out of input files.**
``docs/_static/flow_shear_fixed_step_response_gate.json`` (Cyclone ITG,
``64x64x24``, ``Nl=4``, ``Nm=8``, ``dt = 0.02``, window :math:`t = [240, 300]`):

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

Both increase the heat flux (4.82% and 24.82%) instead of the required 5%
reduction at two combined SEMs, which rejects the weak-shear suppression claim.
The coordinate, remap, operator and derivative identities are not rejected.

.. _end-damping:

Parallel boundary and end damping
---------------------------------

A linked (twist-and-shift) chain is a finite piece of the infinite ballooning
line. Its spectral :math:`z` derivative is periodic over the chain, so what
streams out of one end would re-enter at the other. GKX absorbs it in a layer
at each end of every non-zonal chain:

.. math::

   \partial_t h \;\mathrel{+}= -\nu_{end}\, d(z)\, h, \qquad
   d = 1 - \frac{2x^2}{1 + x^4},\quad x = \frac{\text{distance from the chain end}}{L_w},

with :math:`L_w` = ``damp_ends_widthfrac`` (0.125) of the chain. The damped
quantity is the non-adiabatic :math:`h`, as in GX.

**Contract.** :math:`\nu_{end}` is a rate in :math:`v_{ti}/a` units and the
same on every route: time integration (any step, fixed or adaptive), Krylov,
eigen, shift-invert and nonlinear. No timestep enters. ``[time]
damp_ends_rate`` sets it. The default is

.. math::

   \nu_{end} = C\,\frac{\max_s v_{ts}\,|\hat b\cdot\nabla z|}{w\,L_z},\qquad C = 30,

that is, 30 e-folds per thermal transit of the fastest kinetic species across a
layer of one segment (:math:`L_z` is the stored :math:`z` extent). Each run
prints the rate it uses (``end damping: rate ...``). A deck that sets only the
legacy ``damp_ends_amp`` is refused.

**Why a rate.** GX applies ``damp_ends_amp/dt`` (before February 2023,
:math:`5 v_{max}/L`). Read as a rate, :math:`A/\Delta t` makes a timestep-free
eigenvalue depend on a step it does not have. GS2 [KRT95]_ and stella [BPL19]_
need no absorber (upwinded implicit streaming sets :math:`h = 0` for incoming
particles); GENE truncates the :math:`k_x` connection and uses parallel
hyperdiffusion [PDJ10]_. A Hermite basis does not separate :math:`v_\parallel`
directions, so GKX keeps GX's layer and fixes only its units.

**Measured sensitivity** (ky 0.3; ``plan/research/2026-10-15-end-damping``;
:math:`\gamma` in deck units):

.. list-table::
   :header-rows: 1

   * - deck (Nl, Nm), route
     - nperiod
     - C = 3
     - C = 10
     - C = 30
     - C = 100
   * - adiabatic Cyclone (8, 16), Krylov
     - 2 / 3 / 4
     - 0.09825 / 0.09831 / 0.09831
     - 0.09829 / 0.09831 / 0.09831
     - 0.09831 / 0.09831 / 0.09831
     - 0.09825 / 0.09831 / 0.09831
   * - kinetic-electron Cyclone (8, 16), time
     - 2
     - 0.16786
     - 0.16792
     - 0.16803
     - 0.16807
   * - KBM s-alpha (4, 8), sparse eigen
     - 2 / 3
     - 0.2364 / 0.2382
     - 0.2441 / 0.2497
     - 0.2426 / 0.2502
     - 0.2407 / 0.2499

Below :math:`C \approx 1` the kinetic-electron and KBM rates are not converged
(KE :math:`\gamma` = 1.59 at C = 0, 0.174 at C = 1). Kinetic electrons at
(8, 16) are not converged in domain length (nperiod 3 gives 0.288); that is a
resolution question, not an absorber one. At ``nperiod = 1`` the layer is a
quarter of the line and the adiabatic rate keeps falling with C
(0.0923 at C = 10, 0.0817 at C = 300): use ``nperiod >= 2``.

**Explicit steps.** :math:`\nu_{end}` is explicit, so :math:`\nu_{end}\Delta t`
must stay inside the scheme's stability region. It is below 0.3 on the shipped
``nperiod >= 2`` linear decks; an ``nperiod = 1`` KBM deck (rate 598)
blew up at :math:`\Delta t = 10^{-3}` and runs at :math:`5\times10^{-4}`. The
CFL estimate does not yet include :math:`\nu_{end}`.

**Checking a result.** Rerun at ``damp_ends_rate`` 3x above and below the
printed rate and at ``nperiod + 1``. A converged growth rate moves by less
than its quoted tolerance in all three.

.. [KRT95] M. Kotschenreuther, G. Rewoldt, W. M. Tang, Comput. Phys. Commun. 88, 128 (1995).
.. [BPL19] M. Barnes, F. I. Parra, M. Landreman, J. Comput. Phys. 391, 365 (2019).
.. [PDJ10] M. J. Pueschel, T. Dannert, F. Jenko, Comput. Phys. Commun. 181, 1428 (2010).

De-aliasing and hyperdiffusion
------------------------------

Nonlinear brackets use the standard ``2/3`` de-alias mask, stored on the
spectral grid and applied inside the bracket. Perpendicular hyperdiffusion
(``D_hyper``, ``p_hyper_kperp`` and the ``hyperdiffusion`` term weight; see
:doc:`operators`) is available as scale-selective damping and is off by
default. :math:`k_{\perp,\max}` is taken from the full grid's dealiased corner,
also when a linear run keeps only a subset of ``ky`` rows.


Linear solver options
---------------------

- **Batched ``ky`` scans:** ``ky_batch>1`` integrates several ``ky`` at once;
  ``fixed_batch_shape=True`` pads the last batch to avoid recompiling.
- **Implicit preconditioners** (``implicit_preconditioner``): ``"auto"``/
  ``"diag"``/``"physics"``/``"block"``, ``"damping"``, ``"pas"``,
  ``"pas-coarse"``, ``"hermite-line"``, ``"hermite-line-coarse"``,
  ``"identity"``.
- **Shift-invert preconditioners** (``KrylovConfig.shift_preconditioner``):
  ``"damping"``; ``"hermite-line"`` (FFT in :math:`z` plus a tridiagonal
  Hermite solve of the diagonal-plus-streaming symbol, :math:`O(n)`);
  ``"field-corrected"`` (the line plus the exact field response by a Woodbury
  capacitance solve); ``"pr3-cm"`` (below). ``"auto"`` is ``"pr3-cm"`` with the
  inner FGMRES raised to restart 150 and maxiter 600 (:doc:`algorithms`).
- **Acceptance:** every shift-invert pair is checked against the matrix-free
  relative residual
  :math:`\lVert Av-\lambda v\rVert/\max(\lVert Av\rVert,|\lambda|\lVert v\rVert)`
  with ``shift_outer_residual_tol`` (``1e-6``); rejected pairs raise. The
  eigenvalue is refreshed as the physical Rayleigh quotient.
- **Branch selection:** ``KrylovConfig.mode_family`` (``"cyclone"``, ``"etg"``,
  ``"kbm"``) and ``shift_selection``; ``fallback_method`` defaults to
  ``"propagator"``. Retained-Ritz thick restarts, projected Jacobi--Davidson
  and reduced Schur complements were tried for interior KBM eigenpairs, all
  chose the wrong branch, and none is shipped.
- **Reusable IMEX operators:** ``build_nonlinear_imex_operator`` prebuilds the
  matrix-free operator for ``integrate_nonlinear_imex_cached``.

The inner solves are right-preconditioned FGMRES started from zero. On the
first right-hand side of the outer Arnoldi (Cyclone, ``(Nz, Nl, Nm) = (96, 4,
8)``, float64) three GMRES(20) cycles from zero leave relative residuals of
0.995 (``hermite-line``), 0.9995 (``field-corrected``), 0.86 (``damping``) and
0.299 (``pr3-cm``); only ``pr3-cm`` converges unrestarted (9.9e-5 in 252
iterations), so raise ``shift_restart`` and ``shift_maxiter`` together.

``pr3-cm`` (``solvers_linear_precond_pr3.py``) splits the operator into two
halves, each invertible in closed form, and sweeps between them. The Hermite
line solve inverts streaming, hypercollisions and the *z*-mean of the drift.
The rest (exact :math:`\omega_d(z)`, mirror, drive, end damping, collisions,
local field response) is *z*-local: dense :math:`(l, m)` blocks per
``(species, ky, kx, z)``. One Peaceman--Rachford double sweep costs one solve of
each and no operator application; ``pr3-cm`` is three sweeps from zero, with
:math:`\alpha = -\sqrt{s_1 d}` unless ``shift_precond_alpha`` is set. The
*z*-local block is solved by block-Thomas elimination in the Laguerre index plus
a Sherman--Morrison correction for the rank-one field part
(``solvax.block_thomas_factor_ops``), 1.5x to 2.7x smaller than a dense-band
factor. If the block is not *l*-tridiagonal, ``shift_precond_block_solve =
"auto"`` falls back to the dense batched inverse and records why; the build
also checks that the non-streaming operator is *z*-local and refuses otherwise
(zonal rows under adiabatic electrons fail it).

Automatic solver and fit-signal selection
-----------------------------------------

``solver = "auto"`` (default) runs the time path and falls back to the Krylov
eigensolve when the fitted ``(gamma, omega)`` is non-finite or violates
``require_positive``. ``fit_signal = "auto"`` fits both the ``phi`` and density
traces with the same windowing rules and keeps the higher score. Set
``solver = "time"``, ``"explicit_time"`` or ``"krylov"`` and ``fit_signal =
"phi"`` or ``"density"`` to override.

Custom collision operators
--------------------------

Shipped models and the Python protocol are in :doc:`operators`. A custom
operator is any JAX-compatible object with ``apply(context)``, passed as
``collision_operator=`` to ``linear_rhs``, ``linear_rhs_cached``,
``integrate_linear`` or ``nonlinear_rhs_cached``. It runs on serial explicit and
IMEX linear integration and serial explicit nonlinear integration; implicit
linear solves, nonlinear IMEX, sheared IMEX and decomposed integration refuse
it. The finite-Larmor-radius coefficients are cancellation-sensitive, so they
are generated offline in multiple precision with their provenance and loaded as
compact arrays. Frei, Hoffmann & Ricci write the Bessel argument as
:math:`B=k_\perp\sqrt{2\tau}`; the runtime interpolates at :math:`B=\sqrt{2b}`.
Evidence files are ``docs/_static/collision_*.json``.

Landau damping against the exact kinetic roots
----------------------------------------------

The slab ion-acoustic dispersion relation with adiabatic electrons at
:math:`k_\perp \to 0`,

.. math::

   1 + \frac{T_i}{T_e} + \zeta Z(\zeta) = 0,
   \qquad
   \zeta = \frac{\omega}{k_\parallel \sqrt{2T_i/m_i}},

with :math:`Z` the plasma dispersion function (``scipy.special.wofz``), has no
free parameters. ``test_landau_root_recovered_by_collisional_extrapolation``
evolves the production operator at :math:`N_m = 64` and requires the
:math:`\nu \to 0` extrapolation within 1% in :math:`\gamma` and 0.5% in
:math:`\omega` of the root, at :math:`T_e/T_i = 1` and 10.

.. figure:: _static/landau_damping_validation.png
   :width: 100%

   Produced by ``linear_rhs_cached``; panel (b) eigen-decomposes the same
   operator.

Four traps give plausible but wrong numbers (all gated in
``tests/validation/physics_gates/test_collision_physics.py``):

1. A collisionless truncated Hermite system cannot Landau damp: free streaming
   is anti-Hermitian, so the spectrum is real (:math:`\max|\mathrm{Re}\,\lambda|
   < 10^{-11}`) and any fitted rate is a transient that ends at recurrence.
2. The Landau root is a pole of the continued response, not an eigenvalue. Add
   collisions, measure the decay at several small :math:`\nu`, extrapolate.
3. A density perturbation with no flow is a standing wave; fit
   :math:`A e^{\gamma t}\cos(\omega t + \phi)`, not an unwrapped phase.
4. ``tau_e`` is :math:`T_i/T_e`; the reciprocal is invisible at :math:`T_e=T_i`.
   GKX normalizes to :math:`v_{ti}=\sqrt{T/m}`, not :math:`\sqrt{2T/m}`.

Hermite closure and recurrence
------------------------------

Terminating the ladder with :math:`G_{M+1} = 0` is a reflecting boundary: free
energy returns to the low moments as recurrence at
:math:`t_{\mathrm{rec}} \simeq 2\sqrt{M}/(k_\parallel v_{th})`. Because that
grows only as :math:`\sqrt{M}`, the end of the ladder must absorb. GKX has two
treatments.

**Hypercollisions** (default, ``[physics] hypercollisions = true``) damp a band
of high moments with the :math:`|k_z|`-scaled branch
:math:`\propto \nu_{hyper,m}|k_z|(m/M)^{p_m}` (``nu_hyper_m``, ``p_hyper_m``;
:doc:`operators`).

**Reflectionless closure** (opt-in, Python API
``hermite_closure="reflectionless"``) applies the outgoing-wave condition of
Kanekar et al., *J. Plasma Phys.* **81**, 305810104 (2015), Eq. (4.36):

.. math::

   G_{M+1} = -\,i\,\mathrm{sgn}(k_\parallel)\,R_{M+1}\,G_M,
   \qquad
   R_{M+1} = \frac{M}{\sqrt{2(M+1)}}
             \frac{\Gamma(M/2)}{\Gamma\!\left((M+1)/2\right)},

a single sink :math:`-R_{M+1}\sqrt{M+1}\,v_{th}|k_\parallel|\,G_M` on the last
moment. :math:`R_{M+1} \to 1 - 1/(4M)`, so the absorption becomes exact with
resolution; :math:`R_{M+1} > 0` makes it a positive-definite energy sink; and it
touches only :math:`m = M`. At :math:`M = 2` it reduces to the Hammett-Perkins
three-pole coefficient 0.921318. The test
``test_absorbing_closures_beat_truncation_on_both_metrics`` requires each
treatment to cut the revived :math:`|g_0|` below a tenth of truncation's and
to keep the pre-recurrence error below 0.1, since suppression alone rewards any
strong damping. It carries no free parameter but is not a drop-in replacement
for a tuned hypercollision.

Operator assembly
-----------------

Equations are in :doc:`operators`. ``LinearTerms`` toggles convert to one
``TermConfig`` (``linear_terms_to_term_config``), and
``gkx.terms.assemble_rhs_cached`` sums the per-term kernels; the same RHS serves
the linear integrators, Krylov applications and nonlinear IMEX solves.
``LinearParams.rho_star`` scales the perpendicular wavenumbers in the drift and
drive terms without changing the FFT spacing.

Nodal magnetic-moment grid
--------------------------

``[run] mu_vperp_max`` (``LinearParams.mu_vperp_max``) replaces the Laguerre
expansion in :math:`\mu B` by values at the nodes of a quadrature rule in the
fixed variable :math:`\hat\mu=\mu B_0`; ``Nl`` is then the number of nodes.
Unset keeps the Laguerre path, bitwise unchanged. The construction follows the
v2 representation of GX (Hermite spectral, :math:`\hat\mu` nodal; GX ``docs/Numerics.rst``,
"V2: Hermite plus nodal magnetic moment") on the Hermite-Laguerre formulation of
Mandell, Dorland and Landreman, *J. Plasma Phys.* **84**, 905840108 (2018).

Quadrature
~~~~~~~~~~

Let :math:`\{\hat\mu_j,w_j\}_{j=1}^{n}` satisfy

.. math::

   \int_0^{X}p(\hat\mu)\,e^{-\hat\mu}\,d\hat\mu=\sum_j w_j\,p(\hat\mu_j)
   \quad(\deg p\le 2n-1),\qquad X=\tfrac12 v_{\perp,\max}^2 .

The :math:`\hat\mu_j` are the eigenvalues of the Jacobi matrix of the monic
polynomials orthogonal for :math:`e^{-\hat\mu}` on :math:`[0,X]`, and
:math:`w_j=m_0 v_{0j}^2` with :math:`m_0=1-e^{-X}` and :math:`v_{0j}` the first
component of the eigenvector (Golub and Welsch, *Math. Comp.* **23**, 221 (1969)).
The recurrence coefficients come from the Stieltjes (Lanczos) procedure on a
256-point Gauss-Legendre discretization of the weight. ``vperp_max = 4`` gives
:math:`X=8`: the neglected Maxwellian tail is :math:`e^{-8}=3\times10^{-4}`, and
the density moment of the rule is :math:`m_0=0.99966`. ``vperp_max = inf`` is
Gauss-Laguerre, with :math:`w_je^{\hat\mu_j}` computed directly because
:math:`w_j` underflows. A finite cutoff shortens the interval the nodes must
cover, which is the reason to prefer it at small :math:`n`; its cost is the
:math:`m_0` deficit and a 2-4% dependence of converged rates on
``vperp_max`` (tables below).

Representation
~~~~~~~~~~~~~~

With :math:`x_j(z)=\mu_jB(z)=\hat\mu_jB/B_0` and :math:`S_j=e^{-x_j/2}`,

.. math::

   g(v_\parallel,\hat\mu_j,z)=S_j(z)\sum_{m}\phi^m(v_\parallel)\,G_{jm}(z),
   \qquad F_M^\perp=e^{-x_j}=S_j^2 .

The perpendicular measure is :math:`B\,d\mu=(B/B_0)\,d\hat\mu`. For any
:math:`f` that is smooth in :math:`\hat\mu`,

.. math::

   \int d^2v_\perp\,F_M^\perp f=\int d\hat\mu\,\frac{B}{B_0}e^{-\hat\mu B/B_0}f
   \simeq\sum_j\frac{B}{B_0}\,w_je^{\hat\mu_j}\,S_j^2\,f_j .

Parallel advection
~~~~~~~~~~~~~~~~~~

At fixed :math:`\mu`, :math:`\mathcal L=v_\parallel\nabla_\parallel-\mu\nabla_\parallel B\,\partial_{v_\parallel}`
and :math:`\nabla_\parallel S_j=-\tfrac12x_jb\,S_j` with
:math:`b=\nabla_\parallel\ln B`. With :math:`v\phi^m=\sqrt{m+1}\phi^{m+1}+\sqrt m\phi^{m-1}`
and :math:`\partial_v\phi^m=-\sqrt{m+1}\phi^{m+1}`, projecting
:math:`\mathcal L(S_j\sum\phi^mH_m)` on :math:`\phi^m` and dividing by :math:`S_j` gives

.. math::

   (\mathcal L H)_m=\sqrt{m+1}\,\big(\nabla_\parallel-\tfrac{x}{2}b\big)H_{m+1}
   +\sqrt m\,\big(\nabla_\parallel+\tfrac{x}{2}b\big)H_{m-1}.

The factor :math:`S_j` splits the logarithmic derivative of the weight evenly
between the two couplings (a full Maxwellian would put all of it on one side),
so that the mirror term :math:`\tfrac x2b[\sqrt mH_{m-1}-\sqrt{m+1}H_{m+1}]` is
antisymmetric in :math:`m`. Writing
:math:`\sqrt{m+1}\,S^{-1}\nabla_\parallel(S\,\cdot)+\sqrt m\,S\nabla_\parallel(S^{-1}\cdot)`,
the operator is the weak (conservative) form of the phase-space flux at
fixed :math:`\mu`; no :math:`\hat\mu`-derivative appears because the nodes do
not move. :math:`H=G+\text{field terms}` is the Hamiltonian view of the
Laguerre code: :math:`H_0=G_0+Z\tau^{-1}S_jJ_0\varphi+\ldots`,
:math:`H_1=G_1-Z\tau^{-1}v_tS_jJ_0A_\parallel`.

Conservation
~~~~~~~~~~~~

Per node and without fields, :math:`\partial_t(S_jG_{j0})=-v_t\nabla_\parallel(S_jH_{j1})`,
so :math:`\int dz\,S_jG_{j0}` is conserved by streaming plus mirror; the energy
:math:`\int dz\,S_j[(\tfrac12+x_j)G_{j0}+G_{j2}/\sqrt2]` is conserved too, since
:math:`v_\parallel^2/2+\mu B` is invariant along the unperturbed orbit.
Momentum is not: the mirror force changes it by
:math:`-\mu\nabla_\parallel B\int g`. Magnetic drifts have a nonzero divergence
and do not conserve the node density. Both identities hold to spectral accuracy
for resolved states (tests ``test_weak_mirror_conserves_density_per_node``).

Scaled amplitude and fields
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The field sums are

.. math::

   n=\sum_j\frac{B}{B_0}w_je^{\hat\mu_j}\,S_jJ_{0j}\,G_{j0},\qquad
   \Gamma_0=\sum_j\frac{B}{B_0}w_je^{\hat\mu_j}\,S_j^2J_{0j}^2,

(the Maxwellian response carries :math:`S_j^2=e^{-x_j}`, the evolved moment one
:math:`S_j` because :math:`G` already carries the other), and likewise
:math:`j_\parallel=v_t\sum(\ldots)J_{0j}G_{j1}` and, for :math:`B_\parallel`,
:math:`J_0\to2x_jJ_1(\alpha_j)/\alpha_j`, with :math:`\alpha_j^2=2x_jk_\perp^2\rho^2`
at the local field. Quasineutrality
:math:`\sum_sn_sZ_s\,n_s[G]=q\,\varphi` with :math:`q=\tau_e+\sum_sn_sZ_s^2T_s^{-1}(1-\Gamma_{0s})`,
Ampere's law :math:`(k_\perp^2B^2+\beta\,c\sum_sn_sZ_s^2m_s^{-1}\Gamma_{0s})A_\parallel=\beta c'j_\parallel`
and the :math:`B_\parallel` solve are algebraic in these sums, so they are quadratures.

The state stored is :math:`\tilde G_{jm}=a_jG_{jm}` with
:math:`a_j=\sqrt{(B/B_0)\,w_je^{\hat\mu_j}}`. Since :math:`(B/B_0)w_je^{\hat\mu_j}S_jJ_{0j}G_j=(a_jS_jJ_{0j})(a_jG_j)`, every sum above is :math:`\sum_jJ_j\tilde G_j` with the single array
:math:`J_j=a_jS_jJ_0(\alpha_j)` (and :math:`J^B_j=2x_ja_jS_jJ_1(\alpha_j)/\alpha_j`):

.. math::

   n=\sum_jJ_j\tilde G_{j0},\quad\Gamma_0=\sum_jJ_j^2,\quad
   \tilde H_{j0}=\tilde G_{j0}+Z\tau^{-1}J_j\varphi,

which are exactly the Laguerre expressions with :math:`J_\ell\to J_j`. Field
solves, the Hamiltonian view and the streaming operand therefore need no change.
Since :math:`a_j\propto\sqrt B`, :math:`a\nabla_\parallel(\cdot/a)=\nabla_\parallel-\tfrac12b`,
and the advection becomes

.. math::

   (\mathcal L\tilde H)_m=\sqrt{m+1}\,\big(\nabla_\parallel-\tfrac{x+1}{2}b\big)\tilde H_{m+1}
   +\sqrt m\,\big(\nabla_\parallel+\tfrac{x-1}{2}b\big)\tilde H_{m-1}:

the only change is the node-local coefficients :math:`(x_j\pm1)/2`. The other
terms are diagonal in :math:`j`: grad-B multiplies by :math:`x_j`; the curvature
stencil is unchanged; the diamagnetic drive is
:math:`\omega_*J_j[L_n^{-1}+(x_j-1)L_T^{-1}]` at :math:`m=0`, :math:`\omega_*J_jL_T^{-1}/\sqrt2`
at :math:`m=2`, and the :math:`A_\parallel` drive carries :math:`x_j` in place of
:math:`x_j-1` (the Laguerre-index shift of the existing code is :math:`2\ell+1` there).

Free energy
~~~~~~~~~~~

Let :math:`\langle u,v\rangle=\int dz\,B_0B^{-1}\sum_{jm}\bar u_{jm}v_{jm}`. For
:math:`\tilde G=aG` this is :math:`\int dz\sum_jw_je^{\hat\mu_j}|G_{jm}|^2`, the
phase-space integral :math:`\int d^3x\,d^3v\,g^2/F_M` per flux-tube element
(:math:`d^3x\propto dz/B`, :math:`d^3v\propto B\,d\mu`). The coupling terms of
:math:`\mathcal L` between :math:`m` and :math:`m+1` give, for the real part,

.. math::

   \mathrm{Re}\!\int\!\frac{dz}{B}\sqrt{m+1}\Big[\bar G_m(\nabla-\tfrac{x+1}{2}b)G_{m+1}
   +\bar G_{m+1}(\nabla+\tfrac{x-1}{2}b)G_m\Big]
   =\int dz\,\mathrm{Re}(\bar G_mG_{m+1})\Big[-\big(\tfrac1B\big)'-\tfrac bB\Big]=0

because :math:`(1/B)'=-b/B`. The weight :math:`1/B` is the unique one for which
the coefficients :math:`(x\pm1)/2` cancel, and the :math:`x`-dependence cancels
identically. The drifts are real, diagonal in :math:`(j,z)` and symmetric in
:math:`m`, hence skew-adjoint; Hermite truncation (:math:`G_M=0`) removes no
cross term. So :math:`\mathcal L` is skew-adjoint and :math:`\mathrm{Re}\langle H,\mathcal LH\rangle=0`.
Writing :math:`H=G+F` with :math:`F` the field terms and using
:math:`\sum_jJ_j\partial_tG_{j0}=\partial_t(q\varphi)` for the electrostatic part
and :math:`\sum_jJ_j\partial_tG_{j1}=\kappa\,\partial_tA_\parallel/v_t` for
:math:`A_\parallel` (:math:`q=\tau_e+\sum n_sZ_s^2T_s^{-1}(1-\Gamma_0)`,
:math:`\kappa=(k_\perp^2B^2/\beta+c\sum n_sZ_s^2m_s^{-1}\Gamma_0)/c'`),

.. math::

   W=\tfrac12\langle G,G\rangle+\tfrac12\int\frac{dz}{B}\big(q|\varphi|^2-\kappa|A_\parallel|^2\big),
   \qquad \frac{dW}{dt}=0\quad(\text{collisionless, undriven}),

and with the diamagnetic drive :math:`dW/dt=\mathrm{Re}\langle H,\partial_tG|_{\rm drive}\rangle`.
With :math:`B_\parallel` the solve is the symmetric system

.. math::

   q\,\varphi+q_b\,b=n,\qquad q_b\,\varphi-\kappa_B\,b=P,\qquad
   P=\sum_jJ^B_j\tilde G_{j0},

with :math:`q_b=-\Gamma_{01}=-\sum_jJ_jJ^B_j` and
:math:`\kappa_B=B^2/\beta'+\Gamma_{11}`, :math:`\Gamma_{11}=\sum_j(J^B_j)^2`
(the code solves the row-scaled form; :math:`\beta'` is ``bpar_beta_scale * beta``).
The :math:`b`-part of :math:`F_0` pairs with :math:`P`, so
:math:`\mathrm{Re}\langle F,\partial_tG\rangle=\tfrac12\partial_t\int\frac{dz}{B}\,[q|\varphi|^2+2q_b\mathrm{Re}(\bar\varphi b)-\kappa_B|b|^2]` and
:math:`W` above holds with that field energy (species carry the weights :math:`n_sT_s`
in :math:`\langle\cdot,\cdot\rangle`). It is checked to round-off for ES,
:math:`A_\parallel` and :math:`B_\parallel` (single species), on the nodal and on the Laguerre
grid; a nodal two-species :math:`B_\parallel` case with weights :math:`n_sT_s` also passes (``test_collisionless_free_energy_is_conserved_without_drive``); the form is
evaluated on band-limited :math:`z` modes, for which the discrete derivative obeys
the product rule against :math:`1/B`.

Relation to the Laguerre expansion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For :math:`B=B_0` and the Gauss-Laguerre rule, :math:`\tilde G_j=\sum_\ell O_{j\ell}G_\ell`
with :math:`O_{j\ell}=\sqrt{w_j}\,(-1)^\ell L_\ell(\hat\mu_j)` orthogonal
(:math:`n` nodes integrate :math:`L_\ell L_{\ell'}e^{-\hat\mu}` exactly for
:math:`\ell,\ell'<n`). The multiplication :math:`x` is the Jacobi matrix in
:math:`\ell` and the diagonal :math:`\hat\mu_j` in :math:`j`, which are similar, and
:math:`J_\ell=O^{\!\top}J_j` at :math:`k_\perp\to0`, so all terms except the mirror
coincide with the Laguerre ones, and the mirror vanishes at :math:`b=0`. The full
operator, fields and drives agree to :math:`10^{-14}` at :math:`\varepsilon=0`,
:math:`k_\perp\to0` (``test_gauss_laguerre_nodes_reduce_to_the_laguerre_operator_at_zero_kperp``).
For :math:`B\ne B_0` the Laguerre path carries :math:`\mu B` as the coordinate and
has :math:`\ell\pm1` couplings in the mirror and the Bessel factors; the nodal path
has neither, and the two agree as :math:`n,N_\ell\to\infty`. A nodal state is
converted to Laguerre moments by :math:`P_\ell=\sum_j(-1)^\ell L_\ell(x_j)J_j\tilde G_j`
(``mu_grid_laguerre_modes``); the same modes seed ``init_field`` on all nodes.

Dougherty collisions
~~~~~~~~~~~~~~~~~~~~

``collision_operator = "lenard_bernstein"`` with ``nu > 0`` is available on the grid.
The Hermite part is diagonal (:math:`-\nu\nu_h m`) and the finite-Larmor shift
:math:`-\nu b` is unchanged. The perpendicular part is the Lenard-Bernstein
operator at fixed :math:`z`, with :math:`\beta=B/B_0` and
:math:`u=e^{\beta\hat\mu}g=e^{\beta\hat\mu/2}G`,

.. math::

   \partial_tu=\frac{2}{\beta}\,e^{\beta\hat\mu}\partial_{\hat\mu}\!\big[\hat\mu\,e^{-\beta\hat\mu}\partial_{\hat\mu}u\big],

whose eigenfunctions are :math:`u=L_\ell(\beta\hat\mu)` with rate :math:`-2\ell`
for every :math:`\beta`. It is discretized by summation by parts on the nodes,

.. math::

   M\,\partial_tu=-D^{\!\top}\mathrm{diag}(\kappa)\,D\,u,\quad
   M_j=w_je^{\hat\mu_j(1-\beta)},\ \ \kappa_j=\tfrac2\beta\hat\mu_jM_j,

with :math:`D` the Lagrange derivative matrix of the nodes. Then
:math:`\mathbf 1^{\!\top}M\partial_tu=0` (the quadrature density is conserved exactly),
:math:`u\equiv1` (the local Maxwellian) is a null vector, and
:math:`u^{\!\top}M\partial_tu\le0`, which is the discrete H-theorem for the perpendicular
operator (the scaled amplitude :math:`\tilde G` is Euclidean in this norm). For
Gauss-Laguerre nodes the spectrum is :math:`0,2,4,\ldots` to round-off at
:math:`\beta=1`; with the cutoff it is :math:`0,2.03,4.44,7.84,\ldots` for
:math:`n=8` and the largest rate (outer nodes) is 172 at :math:`n=8` and 1.1e3 at :math:`n=12`. Momentum and energy are restored as in the
Laguerre operator, with the Laguerre weight :math:`(\ell J_{\ell-1}+2\ell J_\ell+(\ell+1)J_{\ell+1})`
of :math:`(x-1)J` replaced by the node weight :math:`(x_j-1)J_j`: the momentum, energy
and perpendicular-flow terms
:math:`\nu J\bar u_\parallel`, :math:`\nu[2(x-1)J+\sqrt2J]\bar t` and
:math:`\nu bJ^B\bar u_\perp` with :math:`\bar t=\tfrac{\sqrt2}3\sum JH_2+\tfrac23\sum(x-1)JH_0`.
For Gauss-Laguerre nodes at :math:`B=B_0` the whole operator is the Laguerre
Dougherty operator under :math:`O` to :math:`10^{-10}`, and its invariants
(:math:`\sum JH_0`, :math:`\sum JH_1`, :math:`\sqrt2\sum JH_2+2\sum(x-1)JH_0`) are
conserved to round-off with :math:`\mathrm{Re}\langle H,CH\rangle<0`. For other
:math:`B` or with the cutoff the quadrature is not exact for the energy weight and the
invariants are conserved to 2e-5 (Gauss-Laguerre, :math:`\varepsilon=0.18`) and 2e-2
(cutoff) of the largest rate; the cutoff operator can have a growing collisional mode
of rate :math:`3\times10^{-4}\nu` at :math:`k_\perp\to0`. Projections onto the exact
nodal invariants conserve to round-off but are anti-dissipative at finite
:math:`k_\perp` (rate :math:`+0.1\nu`) and were not kept. At finite :math:`k_\perp` the
operator is dissipative and its gyroaveraged invariants are not conserved, as for Laguerre
(``test_nodal_dougherty_conserves_and_dissipates``; the Cyclone rate at :math:`\nu=0.2` is
0.0591 for 8 nodes and 0.0595 for ``Nl = 16``). ``collision_model = "landau"`` with ``nu > 0`` is refused.

Nonlinear bracket
~~~~~~~~~~~~~~~~~

The :math:`E\times B` term is local in :math:`z` and in :math:`\hat\mu`, and :math:`a_j`
depends on :math:`z` only, so on the nodes it is
:math:`-\{J_0(\alpha_j)\varphi,\tilde G_j\}` with :math:`\alpha_j^2=2x_jk_\perp^2\rho^2`
evaluated at the local field. This is the existing grid path of the bracket with an
identity transform: ``laguerre_to_grid`` and ``laguerre_to_spectral`` are the identity and
``laguerre_j0`` holds :math:`J_{0}(\alpha_j)`. The heat-flux weight is
:math:`(x_j+\tfrac12)J_j` for :math:`(x-1)J+\tfrac32J`. Nonlinear runs are electrostatic only
(the :math:`B_\parallel` correction would need :math:`x_j` in place of the node value in the
bracket). Cyclone adiabatic-electron heat flux (64x64x24, ``Nm = 8``, :math:`t=600`, one seed,
Sokal-window statistics, accepted at 20 :math:`\tau`): Laguerre ``Nl`` 4 / 8 / 16 give
:math:`8.86\pm0.30` / :math:`8.84\pm0.23` / :math:`8.77\pm0.16`; 8 nodes give
:math:`8.85\pm0.26`; 4 nodes give :math:`7.56\pm0.34` (window under 20 :math:`\tau`). The heat flux is
converged at the shipped ``Nl = 4``, so the grid brings no accuracy gain here and 8 nodes
cost about 2-3 times the 4 moments (shared GPU).

Scope: linear runs, electrostatic and electromagnetic; nonlinear runs, electrostatic.
``nu_hyper``, ``nu_hyper_l``, ``nu_hyper_lm`` and ``hypercollisions_const`` are refused
(they act on the Laguerre index); the :math:`|k_z|` Hermite hypercollisions are unchanged.
Velocity-parallel routes are refused. ``init_field`` (including ``"phi"``) seeds every node.

References: GX ``docs/Numerics.rst`` (V2); Mandell, Dorland, Landreman 2018;
Golub and Welsch 1969; Gautschi, *Orthogonal Polynomials* (2004), Ch. 2-3 (Stieltjes
and Lanczos procedures); Kanekar et al., *J. Plasma Phys.* **81**, 305810104 (2015)
for the Hermite closure.

Measured against the Laguerre expansion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cyclone ITG (s-alpha, adiabatic electrons, ``Nm = 32``, ``vperp_max = 4``), the
Hermite-banded eigensolver, one laptop CPU (4 threads, shared; wall times carry
about 30% noise, wall includes compilation, memory is the peak resident set).
``Delta`` is the growth rate relative to the converged Laguerre value
(``Nl = 24``: :math:`\gamma=0.0928` at :math:`k_y=0.3`, 0.0798 at 0.4).

.. list-table::
   :header-rows: 1

   * - grid
     - n
     - :math:`\gamma` (0.3)
     - Delta
     - wall (s)
     - memory (GB)
     - :math:`\gamma` (0.4)
     - Delta
     - wall (s)
   * - Laguerre
     - 8
     - 0.0981
     - +5.6%
     - 92
     - 5.7
     - 0.0912
     - +14%
     - 86
   * - Laguerre
     - 16
     - 0.0928
     - 0.0%
     - 430
     - 14.5
     - 0.0778
     - -2.5%
     - 354
   * - Laguerre
     - 24
     - 0.0928
     - ref
     - 1457
     - 19.9
     - 0.0798
     - ref
     - 939
   * - nodal
     - 4
     - 0.0808
     - -13%
     - 32
     - 2.8
     - 0.0578
     - -28%
     - 33
   * - nodal
     - 8
     - 0.0941
     - +1.4%
     - 79
     - 5.3
     - 0.0794
     - -0.5%
     - 77
   * - nodal
     - 12
     - 0.0920
     - -0.8%
     - 189
     - 9.5
     - 0.0800
     - +0.3%
     - 222
   * - nodal
     - 16
     - 0.0918
     - -1.1%
     - 421
     - 15.2
     - 0.0766
     - -4.1%
     - 424

Eight nodes match the converged Laguerre rate to 1.4% and 0.5%, where the
Laguerre rung of the same accuracy is ``Nl = 16`` (5.4x the wall, 2.7x the
memory) at :math:`k_y=0.3` and ``Nl = 24`` (12x, 4x) at 0.4. The cost per
state element is that of the Laguerre path, so the gain is the node count.
At 0.4 the 16-node rate is 4% below the 12-node one; with ``vperp_max = 6``
the 12- and 16-node rates are 0.0812 and 0.0813 (+1.8% and +1.9%) and at 0.3
the 12-node rate is 0.0940 (+1.3%), so the cutoff moves a converged value by
about 2-4%. The Laguerre ladder moves by 2.5-14% between its rungs.
The tutorial
deck (``Nz = 32, Nm = 8``) gives 0.0999 for 8 nodes and 0.1001 for ``Nl = 16``.

Collisionless TEM (Dannert and Jenko 2005, kinetic electrons,
:math:`A_\parallel`, ``Nm = 24``, :math:`k_y\rho_s=0.3`; GS2 0.869 is
``Nm``-limited here), same solver with a shift at the mode:

.. list-table::
   :header-rows: 1

   * - n
     - 4
     - 8
     - 12
     - 16
   * - Laguerre :math:`\gamma`
     - 0.480
     - 0.611
     - 0.552
     - 0.565
   * - nodal :math:`\gamma`
     - 0.514
     - 0.561
     - 0.547
     - 0.548

The nodal ladder settles to within 0.4% from 12 nodes; the Laguerre one moves by
up to 27% between rungs. At equal ``n`` the TEM cost is similar, not lower
(wall, memory: ``n = 8`` 404 s, 8.5 GB nodal against 433 s, 8.5 GB Laguerre;
``n = 12`` 1024 s, 18.3 GB against 533 s, 15.2 GB; ``n = 16`` 2459 s, 22.4 GB
against 2045 s, 23.4 GB), so on this case the gain is a ladder that converges,
not time. GX v2 reports 0.564 and 0.553 for 8 and 12 nodes with its own rule,
and 0.0934 and 0.0795 for Cyclone at 8 nodes.

At 4 nodes the TEM rate is not converged: 0.505, 0.514, 0.360, 0.411 and 0.415
for ``vperp_max`` 3, 4, 5, 6 and ``inf``, so the 0.612 that GX v2 reports at
4 nodes is within that spread. The 12-node TEM run took 1024 s and 2022 s on two
identical runs (a memory-bound, shared host), so the factor 2 against Laguerre
at that size is timing noise.

Zonal-flow residual (Rosenbluth-Hinton; :math:`k_y=0`, :math:`k_x\rho=0.05`,
adiabatic electrons, s-alpha :math:`q=1.4`, :math:`\varepsilon=0.18`, ``Nm = 16``,
late-time mean after the GAM): 0.0929 and 0.0943 for Laguerre ``Nl`` = 8 and 16;
0.0939 and 0.0940 for 8 and 12 nodes (``test_zonal_flow_residual_on_the_mu_grid_matches_laguerre``).
The small-:math:`\varepsilon` formula :math:`1/(1+1.6q^2/\sqrt\varepsilon)=0.119`
does not apply at :math:`\varepsilon=0.18`; the check is the agreement with Laguerre.

Other cases (office CPU, banded route, ``Nm`` as stated; growth rates):

.. list-table::
   :header-rows: 1

   * - case
     - Laguerre ``Nl`` 8 / 12
     - nodes 8 / 12
   * - TEM, :math:`\nu_e=0.071` (``Nm = 24``; GS2 0.5054)
     - 0.4169 / 0.4106
     - 0.4002 / 0.4010
   * - TEM, :math:`\nu_e=0.283` (GS2 0.1510)
     - 0.0548 / 0.0589
     - 0.0537 / 0.0535
   * - KBM, :math:`\beta=0.015`, :math:`k_y=0.3` (``Nm = 24``)
     - 0.3155 / 0.3164
     - 0.2988 / 0.2989
   * - Miller kinetic electrons, :math:`k_y=0.4` (``Nm = 24``)
     - 0.2652 / 0.2669
     - 0.2648 / 0.2650
   * - W7-X ITG, :math:`k_y\rho=2.1` (``Nm = 16``)
     - 0.2178 / (24: 0.2176)
     - 0.2168 / 0.2168

The nodal ladders are flat in every case. For the KBM case the nodal rate is 5.6%
below the Laguerre one at either resolution; the GS2 comparison of the evidence
ledger puts the Laguerre ``(8, 24)`` rate 7.3% above GS2, which the nodal rate
reduces to about 1.6%. The collisional TEM rates are far below GS2 for both
representations (``Nm`` limit; GS2's collision operator is not Dougherty). W7-X at
:math:`k_y\rho=2.1` is converged at 4 nodes (0.2184).
