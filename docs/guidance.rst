Resolution, Step and Solver Guidance
====================================

What to set, in what order, and how to check it. Values are starting points
for Cyclone-like cases; the check at the end of each section is the test.
Measurements are in :doc:`numerics`, :doc:`solvers` and :doc:`examples`.

Defaults at a glance
--------------------

What you get when you change nothing, and when to change it.

.. list-table::
   :header-rows: 1
   :widths: 24 28 48

   * - Choice
     - Default
     - Change it when
   * - Time stepper
     - no ``dt`` (deck): nonlinear ``auto``, which is ``lawson-rk4-pi`` where the drift is at least 56% of the linear CFL bound and the step cap is at least 2.6 times the ``rk4-pi`` bound step, ``rk4-pi`` elsewhere (:ref:`where it pays <lawson-limits>`); nonlinear kinetic electrons at :math:`\beta \le 3\times10^{-4}` ``imex-ark3`` (adaptive, falls back to ``rk4-pi`` with a warning if its factor check refuses); linear and bare ``TimeConfig()`` ``rk4`` + CFL. A ``dt`` gives ``rk2`` at that step
     - a published step (``rk3`` at fixed ``dt``); a stepper you want regardless of the rule (:ref:`advanced pins <advanced-steppers>`); the speedups are in the table below
   * - Linear growth rates
     - ``solver`` ``auto``: banded shift-invert, else ``adaptive``, certified
     - one mode of a hard branch (``shift-invert``)
   * - ``ky`` layout
     - ``half`` (``1 + Ny//2`` rows)
     - ``full`` only to read a two-sided array from an older run
   * - Velocity space
     - linear (12, 24), nonlinear (4, 8); with kinetic electrons per species: linear ions (12, 32), electrons (6, 48), nonlinear ions (4, 8), electrons (4, 16)
     - a deck or call that names ``Nl``/``Nm``; TEM, KBM: see the sections below
   * - Collisions
     - linearized Landau; hypercollisions on the last Hermite moments
     - ``nu = 0`` collisionless runs keep the hypercollisions
   * - End damping
     - one rate, ``damp_ends_rate``, on every route
     - never with the step size; see :doc:`numerics`
   * - Precision
     - ``float32``
     - ``float64`` for eigensolves, conditioning and residual checks
   * - Threads and devices
     - XLA's pool; Eigen pool off below four CPUs; one device
     - nothing on CPU; GPUs: ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` when shared

Default stepper: measured cost
------------------------------

Deck with no ``dt`` and no ``method``, measured with ``rk4-pi`` (the rule that now
selects ``lawson-rk4-pi`` where it pays came later, :ref:`where it pays <lawson-limits>`), against
``rk4`` + CFL, one A4000, float32,
wall time including setup, heat flux over the saturated window (transient
dropped; error is :math:`\sigma/\sqrt{n_\mathrm{eff}}` with the Sokal
correlation time, at least 20 correlation times).

.. list-table::
   :header-rows: 1

   * - Run
     - Default
     - Wall, default / ``rk4`` + CFL
     - :math:`Q` default / ``rk4`` + CFL
   * - Cyclone, adiabatic, 64x64, (4, 8), t = 1000
     - ``rk4-pi``
     - 103 s / 135 s (1.3x)
     - 8.51 +- 0.14 / 8.87 +- 0.15
   * - QHS, adiabatic, 48x48, (4, 8), t = 600
     - ``rk4-pi``
     - 111 s / 107 s (none)
     - 9.82 +- 0.10 / 9.91 +- 0.09
   * - Cyclone, kinetic electrons, :math:`\beta` 1e-4, 32x32, (4, 8), t = 400
     - ``imex-ark3``
     - 450 s / 1434 s (3.2x)
     - 1129 +- 14 / 1068 +- 18 (``rk4-pi`` 1162 +- 11 in 1123 s)
   * - Kinetic electrons, :math:`\beta` 1e-3, t = 60
     - ``rk4-pi``; ``imex-ark3`` measured 188 s against 129 s (0.7x)
     - (imex) 188 s / 129 s
     - (imex) 905 +- 50 / 919 +- 72
   * - Kinetic electrons, :math:`\beta` 0.015, t = 60
     - ``rk4-pi``
     - 182 s / 310 s (1.7x)
     - 18900 +- 800 / 19500 +- 1000

Setup (compile and factor build, 30 to 130 s) is a fixed part of each wall time,
so longer runs gain more. The linear eigensolver ``auto`` is the certified route
but not faster than the time route at (4, 8) or (12, 24) on one GPU (23 s against
10 s and 44 s against 10 s, Cyclone ``ky`` 0.3); with kinetic electrons the time
route without a ``dt`` does not yet integrate (non-finite), so use the eigensolver
or give a ``dt``.

Choose the time stepper
-----------------------

GKX advances :math:`\partial_t g = L g + N(g)`. Write :math:`L = S + R`, where
:math:`S` holds the stiff terms (parallel streaming with its dissipation, and
the field response) and :math:`R` the drifts, mirror, drive and collisions.
:math:`N` is the :math:`E\times B` bracket.

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Scheme
     - Use it for
     - Limit
   * - ``rk4`` + CFL (linear default without ``dt``)
     - any deck; the controller sets the step from the stiffest explicit term
     - step follows electron streaming when electrons are kinetic
   * - ``auto`` (nonlinear default without ``dt``)
     - ``lawson-rk4-pi`` or ``rk4-pi`` from the drift share of the linear CFL bound and the step cap, chosen at the start of the run
     - the choice is printed; pin a stepper to override it
   * - ``rk4-pi``
     - RK4 with an embedded error estimate and an I controller
     - 1 to 2x fewer steps than ``rk4`` + CFL at the same flux
   * - ``lawson-rk3``, ``lawson-rk4``, ``lawson-rk4-pi``
     - the curvature and grad-B drift advanced exactly (:ref:`lawson-steppers`): tokamak decks and large ``Nl * Nm``, where the drift sets the CFL step
     - 1.5 to 2.5 times fewer steps than ``rk4-pi`` and 0.5 to 0.9 of its wall time on Cyclone, Miller, EM and (8, 16); no gain, or a loss, on HSX, W7-X QI or with kinetic electrons
   * - ``imex-ark3`` (adaptive, nonlinear kinetic electrons at low beta)
     - chunk-level adaptive step, no electron-streaming limit
     - refuses decks whose factor check fails (some stellarators); no gain at :math:`\beta \gtrsim 10^{-3}`
   * - ``rk2``/``rk3`` at fixed ``dt``
     - reproducing a published step, parity runs
     - warns when ``dt`` exceeds the CFL estimate
   * - ``imex``, ``imex2``
     - linear runs with stiff collisions or hypercollisions
     - diagonal damping only; streaming stays explicit
   * - ``imex-ars3``
     - nonlinear kinetic electrons at 32x32 and up
     - fixed ``dt``, chosen by hand; factor memory and build time
   * - ``solver = "krylov"``/``"adaptive"``
     - linear growth rates: no time step at all
     - one mode per solve

.. _advanced-steppers:

**Advanced options.** The defaults need none of these; each is a deliberate pin.
``method = "rk4-pi"`` or ``"lawson-rk4-pi"`` fixes the stepper whatever the rule
chooses (the Lawson methods also accept ``cfl_fac``, and ``lawson-rk3`` and
``lawson-rk4`` exist for comparison with ``rk3`` and ``rk4``); a ``dt`` with
``fixed_dt = true`` fixes the step, which a restart that must reproduce an
uninterrupted run needs (``examples/12_restart_and_analysis``); ``imex-ars3``
takes a fixed ``dt`` chosen by hand; the sharded routes take ``rk4-pi``.
``examples/13_advanced_time_stepping`` runs the pins side by side and
``examples/05_kinetic_electrons`` compares the implicit and explicit steppers
with kinetic electrons.

**Accuracy of the adaptive IMEX step.** ``imex-ark3`` keeps the embedded
ARK3(2) error of each step below ``rtol`` = 1e-3 of the state norm. Against
``rk4-pi`` (Cyclone, kinetic electrons, :math:`m_e/m_i` = 1/1836, 32x32x16,
(4, 16), t = 120, A4000, heat-flux window mean over t = 30-120 with the Sokal
standard error of each run, three random seeds):

.. list-table::
   :header-rows: 1

   * - ``rtol``
     - Steps
     - Wall
     - :math:`Q_i` against ``rk4-pi``
     - :math:`Q_e` against ``rk4-pi``
   * - ``rk4-pi``
     - 37200
     - 334-342 s
     - 503, 507, 506
     - 709, 724, 749
   * - 1e-2 (before)
     - 960-1090
     - 165-201 s
     - +34, +32, +30 (+6%)
     - +47, +39, +17 (+5%)
   * - 1e-3
     - 2020-2140
     - 204-222 s
     - +3, -3, +10 (+0.7%)
     - +12, +8, -33 (-0.6%)
   * - 1e-4 (one seed)
     - 4830
     - 349 s
     - -20 (-4%)
     - +11 (+2%)

Each window mean carries a standard error of 5-25, so one seed does not resolve
the 1e-2 offset; all three seeds have it, and the 1e-3 offsets scatter about zero.
Wall times include the 100-200 s of setup. The first turbulent burst is a
trajectory comparison, which only holds until the peak: ``rk4`` runs at
CFL 0.9, 0.45 and 0.225 (16x16x16, 1/100) agree to 0.1% through the onset and
peak, then differ by 10-30% by t = 26-30 (chaotic divergence), so errors of 40%
after the peak are not a property of the step. Up to the peak, against ``rk4`` CFL 0.45
(window 12-22) or ``rk4-pi`` (32x32x16, window 10-20), the error of
:math:`Q(t)` is:

.. list-table::
   :header-rows: 1

   * - Case
     - 1e-2
     - 1e-3
     - 1e-4
     - 3e-5
   * - 16x16x16, 1/100: error, steps to t = 30
     - 6.7%, 384
     - 4.7%, 544
     - 3.9%, 1088
     - 0.2%, 1760
   * - 16x16x16, 1/1836: error, steps to t = 25
     - 4.2%, 256
     - 8.4%, 288
     - 0.4%, 608
     - n/a
   * - 16x16x16, 1/3670: error, steps to t = 25
     - 3.0%, 288
     - 4.0%, 320
     - 1.0%, 672
     - n/a
   * - 32x32x16, 1/1836: error, steps to t = 22
     - 5.2%, 224
     - 5.2%, 256
     - 2.6%, 416
     - 0.4%, 704

The onset time (:math:`Q` = 1e2 to 2e4) is within 0.05 of the reference
(t = 12-20) at every tolerance, and a QA stellarator (16x16x32, t = 60) agrees
over its first burst to 0.3% at 1e-2 and 0.1% at 1e-4. The peak error plateaus
for 1e-2 to 1e-3 and drops only at 1e-4 and below, where the saturated phase
takes 5x the steps of 1e-2, so 1e-3 is the default: it is the loosest
tolerance whose seed-mean saturated flux agrees with ``rk4-pi``. The bracket
constant (``dt`` bound ``17 / w_nl``) does not change the first-burst error: 2.5,
4 and 8.5 at 1e-2 give 5.3%, 5.2% and 5.2% (320, 256, 224 steps; 224 at 17), and
8.5 and 34 at 1e-4 repeat the result of 17 (2.6%, 416 steps). A bound of 4 costs
2.6x the steps in the saturated phase (one seed) for about the flux offset of
1e-3, so it stays 17. The error is not float32: the same 16x16x16, 1/100 runs in
``float64`` give 14.7% against 14.5% at 1e-2 (window 12-23). With a fixed step
``imex-ars3`` at 0.05 is 0.2% off at t = 20 and at 0.02 it is within 0.1% to
t = 21 (0.6% at 22), while 0.1 and 0.15 go non-finite at the burst, so a fixed
step needs the margin the adaptive controller gives.

**Why IMEX helps.** With kinetic electrons the explicit step is set by
electron streaming, :math:`\Delta t \lesssim 1/(v_{te}k_{z,\max})`, about
:math:`\sqrt{m_i/m_e}\approx 60` times shorter than the ion scale. In
``imex-ars3`` the operator :math:`S` is inverted exactly per twist-shift chain
(block Thomas plus a Woodbury correction for the fields), so
:math:`\Delta t` is set by :math:`R` and :math:`N`. On Cyclone with kinetic
electrons at 64x64x24 this gave 10x shorter time to solution than RK3 at the
same heat flux. At a 16x16 tutorial grid or a small QA deck it does not pay:
the factor build dominates.

**Choosing the IMEX step.** ``dt`` is not found by a controller. 0.05 is
stable at 32x32x16 and non-finite at 64x64x24, where 0.035 and 0.025 work.
Rerun at half the step and compare the heat flux. Set
``XLA_PYTHON_CLIENT_PREALLOCATE=false`` on a GPU.

**Collisions in the IMEX step.** Explicit ARK3 is stable to
:math:`\rho(C)\,\mathrm{d}t \approx 2.5`, with :math:`\rho(C)` the spectral
radius of the Landau matrix: :math:`\nu_e` times 2.9, 8.7, 16 and 24 at
:math:`(N_l, N_m)` = (2, 4), (4, 8), (4, 16) and (8, 16). With
``collision_scheme = "implicit"`` (the default) GKX measures :math:`\rho(C)` at
startup and, only if :math:`\rho(C)\,\mathrm{d}t > 2`, takes collisions out of
the RHS and advances them by a backward-Euler sub-step after each step (dense
per point, field-particle part by a Woodbury correction, density, momentum and
energy conserved to rounding). Otherwise nothing changes: the run is the
explicit one. At ``dt`` 0.05 the switch is :math:`\nu_e \approx 4.6` at (4, 8),
:math:`\nu_i \approx 0.08` for :math:`m_e/m_i = 2.7\times10^{-4}`. Below it
the sub-step costs 10% per step on small grids and buys nothing; above it
explicit goes non-finite at fixed ``dt``, and an adaptive run is held to
:math:`\mathrm{d}t \lesssim 2.5/\rho(C)`. The sub-step is first order (Lie): its
splitting error is outside the embedded estimate, so with ``fixed_dt = false``
:math:`\mathrm{d}t` is also capped at :math:`5/\rho(C)`. Measured on
8x10x16 flux tubes, :math:`\beta` 1e-4 unless stated, (4, 8), ARK3, 15 time
units; error is the heat flux at the end against adaptive ``rk4`` (CFL 0.3), the
electrostatic row against explicit at ``dt`` 0.0125:

.. list-table::
   :header-rows: 1
   :widths: 30 8 8 8 14 14 18

   * - Deck
     - :math:`\nu_e`
     - ``dt``
     - :math:`\rho(C)\mathrm{d}t`
     - :math:`Q` error, explicit
     - :math:`Q` error, implicit
     - wall, explicit / implicit
   * - Cyclone, linked
     - 0.6
     - 0.2
     - 1.0
     - 0.0%
     - 0.2%
     - 41 / 46 s
   * - Cyclone, linked
     - 3
     - 0.05
     - 1.3
     - 0.0%
     - 0.7%
     - 74 / 75 s
   * - Cyclone, linked
     - 3
     - 0.2
     - 5.3
     - non-finite
     - 1.7%
     - -- / 45 s
   * - Cyclone, linked
     - 12
     - 0.05
     - 5.3
     - non-finite
     - 2.4%
     - -- / 76 s
   * - Cyclone, linked
     - 12
     - 0.2
     - 21
     - non-finite
     - 4.9%
     - -- / 56 s
   * - Cyclone, :math:`\beta` 1e-2
     - 12
     - 0.05, 0.2
     - 5.3, 21
     - non-finite
     - 2.1%, 5.1%
     - -- / 65, 38 s
   * - Cyclone, electrostatic
     - 12
     - 0.05, 0.2
     - 5.3, 21
     - non-finite
     - 1.7%, 2.9%
     - -- / 68, 47 s
   * - Cyclone, periodic
     - 12
     - 0.05, 0.2
     - 5.3, 21
     - non-finite
     - 0.1%, 0.4%
     - -- / 70, 55 s
   * - QA stellarator
     - 12
     - 0.05, 0.2
     - 5.3, 21
     - non-finite
     - 3.2%, 8.7%
     - -- / 88, 62 s
   * - W7-X
     - 0.6
     - 0.2
     - 1.0
     - 1.3%
     - 3.0%
     - 60 / 67 s
   * - W7-X
     - 12
     - 0.0125, 0.025
     - 1.3, 2.7
     - 0.1%, --
     - 2.7%, 5.7%
     - 257 / 276, 144 s
   * - W7-X
     - 12
     - 0.05, 0.2
     - 5.3, 21
     - non-finite
     - 10%, 26%
     - -- / 97, 61 s

The growth rate of the Cyclone decks is within 0.35% at every row. The error
grows with :math:`\rho(C)\,\mathrm{d}t`, as :math:`\sqrt{\mathrm{d}t}` on the
growing Cyclone, QA and periodic modes and about linearly on the transient W7-X
flux, so a flux that matters at a few percent at high :math:`\nu_e` wants
``dt`` near the explicit bound or ``collision_scheme = "explicit"``. Adaptive
nonlinear Cyclone, :math:`\beta` 1e-4, 16x16x16, (4, 8), t = 24 on two cores:
:math:`\nu_e` = 0.6 explicit 411 s and implicit 366 s, first-burst flux within
0.3% (the default is the explicit run); :math:`\nu_e` = 12 explicit 979 s at
``dt`` 0.025, default (implicit, ``dt`` 0.05 at the cap) 490 s and 2.2% in the
first burst. Adiabatic-electron decks are not affected: the chain IMEX factor
check refuses them with or without the sub-step.

Resolution
----------

- **Velocity space.** Linear runs default to :math:`(N_l, N_m) = (12, 24)`,
  nonlinear runs to (4, 8). Vary :math:`N_m` first. Kinetic electrons and TEM
  need more: TEM growth rates still change at (16, 48).
- **Per species.** A kinetic-electron deck that names no resolution gives the
  ions and the electrons their own counts; see `Velocity space per species`_.
- **Perpendicular.** Nonlinear production runs use 64-96 points in
  :math:`x` and :math:`y`; ``gkx --estimate`` prints the grid the geometry
  asks for, with the reason.
- **Parallel.** :math:`N_z` = 24-64 per field-period segment, and
  ``nperiod >= 2`` for any linear run with linked boundaries.
- **Check.** The heat flux or growth rate changes less than your tolerance
  when :math:`N_m` doubles.

Velocity space per species
--------------------------

Ions and electrons converge at different moment counts, and one count for both
pays the larger for each. ``[[species]] Nl`` and ``Nm`` (:ref:`inputs
<per-species-moments>`) give each species its own, and the right-hand side runs
each at that size, so the cost is the sum over species of
:math:`N_l N_m` rather than :math:`N_s \max N_l \max N_m`.

**What each species needs** (Cyclone s-alpha, :math:`m_e/m_i = 2.7\times10^{-4}`,
:math:`\beta = 10^{-4}`, :math:`N_z = 32`, certified Krylov eigenvalue at
:math:`k_y\rho_i = 0.3`; the other species held at (12, 48); reference uniform
(12, 64), :math:`\gamma = 0.1394`):

.. list-table::
   :header-rows: 1
   :widths: 30 14 14 14 14 14

   * - Electron :math:`N_l` (:math:`N_m = 48`)
     - 1
     - 2
     - 4
     - 6
     - 12
   * - :math:`\gamma` against reference
     - -26%
     - -13%
     - -1.1%
     - +0.2%
     - +0.3%
   * - Electron :math:`N_m` (:math:`N_l = 4`)
     - 16
     - 24
     - 32
     - 48
     -
   * - :math:`\gamma` against reference
     - +5.1%
     - +4.6%
     - +1.9%
     - -1.1%
     -
   * - Ion :math:`N_m` (:math:`N_l = 12`)
     - 8
     - 12
     - 16
     - 24
     - 32
   * - :math:`\gamma` against reference
     - +11%
     - -3.4%
     - -4.7%
     - +1.8%
     - +1.7%
   * - Ion :math:`N_l` (:math:`N_m = 48`)
     - 4
     - 6
     - 8
     - 10
     - 12
   * - :math:`\gamma` against reference
     - -22%
     - +7%
     - -2%
     - -5%
     - +0.3%

Trapped-electron physics needs a few Laguerre moments and many Hermite ones;
the ions need Laguerre moments for the finite Larmor radius and fewer Hermite
ones. This branch keeps a scatter of a few percent for ion :math:`N_l` between
6 and 10 even at (12, 48): ITG and trapped-electron branches sit close at
:math:`k_y\rho_i = 0.3`, so judge a choice by the table below, not by one
entry.

**Fallback and check.** Ions (12, 32), electrons (6, 48) against the uniform
(12, 64) reference (uniform (12, 48) is within 0.3% on the ITG point):

.. list-table::
   :header-rows: 1

   * - Deck
     - Reference :math:`\gamma`
     - Per species
     - Difference
   * - Cyclone ITG, :math:`k_y\rho_i = 0.3`
     - 0.13940
     - 0.14161
     - +1.6%
   * - Cyclone ITG, :math:`k_y\rho_i = 0.2`
     - 0.0807 (12, 48)
     - 0.0798
     - -1.1%
   * - TEM, :math:`a/L_{n}` 2, :math:`a/L_{Te}` 4, :math:`k_y\rho_i = 0.4`
     - 0.4492 (12, 48)
     - 0.4495
     - +0.1%
   * - ETG-range, :math:`k_y\rho_i = 30`
     - 19.381
     - 19.514
     - +0.7%
   * - Landreman-Paul QA stellarator, :math:`k_y\rho_i = 0.3`
     - 0.22193
     - 0.22217
     - +0.1%

The reference moves by 5% between (8, 32) and (12, 48) on the TEM deck, so use
no smaller counts there. The cost is :math:`12\cdot32 + 6\cdot48 = 672`
moments per grid point against 1152 for uniform (12, 48) and 1536 for
(12, 64). Smaller counts (ions (8, 24), electrons (4, 32)) are within 1% on the
ITG and stellarator decks and 12% low on the TEM deck.

Nonlinear: ions (4, 8) and electrons (4, 16) are the fallback. The ions keep the
count of the ion-only fallback; the electrons keep the Hermite moments that
their fast streaming and trapped orbits use. The table scans both species
around that point: Cyclone with kinetic electrons, :math:`m_e/m_i = 1/1836`,
electrostatic, 32x32x16, the default ``imex-ark3``, :math:`Q_i` and :math:`Q_e`
averaged over t = 30-120 (the flux is stationary from t = 30), with one or two
random seeds as the Seeds column says. Each entry is the mean over the seeds
with the Sokal standard error of each run (the larger of that and the seed
spread when there are two seeds); the difference is against ions (8, 32) and
electrons (6, 48), and its error includes the reference's. Two seeds of
(4, 16) / (4, 16) differ by 3.7% in :math:`Q_i` and 0.4% in :math:`Q_e`, which
is the resolution of the scan.

.. list-table::
   :header-rows: 1
   :widths: 24 10 8 12 14 12 14

   * - Ions / electrons, :math:`(N_l, N_m)`
     - Moments per point
     - Seeds
     - :math:`Q_i`
     - Against reference
     - :math:`Q_e`
     - Against reference
   * - (8, 32) / (6, 48)
     - 544
     - 1
     - 695 +- 14
     - +0.0 +- 2.9%
     - 1258 +- 20
     - +0.0 +- 2.2%
   * - (4, 16) / (4, 16)
     - 128
     - 2
     - 716 +- 13
     - +3.1 +- 2.7%
     - 1268 +- 15
     - +0.9 +- 2.0%
   * - (4, 8) / (4, 16)
     - 96
     - 2
     - 715 +- 12
     - +3.0 +- 2.7%
     - 1277 +- 14
     - +1.5 +- 1.9%
   * - (4, 8) / (4, 8)
     - 64
     - 1
     - 738 +- 26
     - +6.3 +- 4.2%
     - 1308 +- 30
     - +4.0 +- 2.9%
   * - (4, 8) / (2, 8)
     - 48
     - 1
     - 707 +- 13
     - +1.8 +- 2.8%
     - 1267 +- 16
     - +0.8 +- 2.0%
   * - (4, 16) / (2, 16)
     - 96
     - 1
     - 723 +- 14
     - +4.1 +- 2.9%
     - 1298 +- 14
     - +3.2 +- 1.9%
   * - (4, 16) / (6, 16)
     - 160
     - 1
     - 677 +- 10
     - -2.5 +- 2.5%
     - 1271 +- 12
     - +1.1 +- 1.9%
   * - (4, 16) / (4, 8)
     - 96
     - 2
     - 696 +- 36
     - +0.2 +- 5.5%
     - 1245 +- 13
     - -1.0 +- 1.9%
   * - (4, 16) / (2, 8)
     - 80
     - 2
     - 706 +- 15
     - +1.6 +- 3.0%
     - 1252 +- 13
     - -0.5 +- 1.9%
   * - (4, 16) / (4, 24)
     - 160
     - 1
     - 684 +- 26
     - -1.5 +- 4.3%
     - 1294 +- 17
     - +2.9 +- 2.1%
   * - (4, 16) / (6, 24)
     - 208
     - 1
     - 668 +- 34
     - -3.9 +- 5.4%
     - 1257 +- 25
     - -0.1 +- 2.5%
   * - (4, 16) / (2, 24)
     - 112
     - 1
     - 716 +- 16
     - +3.1 +- 3.1%
     - 1284 +- 16
     - +2.1 +- 2.0%
   * - (4, 16) / (6, 8)
     - 112
     - 1
     - 726 +- 9
     - +4.6 +- 2.5%
     - 1264 +- 16
     - +0.5 +- 2.0%
   * - (6, 16) / (4, 16)
     - 160
     - 1
     - 724 +- 15
     - +4.2 +- 3.0%
     - 1295 +- 19
     - +3.0 +- 2.2%
   * - (6, 8) / (4, 16)
     - 112
     - 1
     - 720 +- 18
     - +3.7 +- 3.3%
     - 1282 +- 15
     - +2.0 +- 2.0%

Every pair is within 6.3% of the reference in both fluxes, and neither the
electron :math:`N_l` (2 to 6), the electron :math:`N_m` (8 to 24) nor the ion
:math:`N_l` and :math:`N_m` shows a trend larger than that scatter. The
fallback agrees with the reference to 3.0% and 1.5% and with the uniform
(4, 16) it replaces (3.1% and 0.9%), with 96 moments per point instead of 128.
Uniform (4, 8), the generic fallback, is the least accurate pair in the table
(+6.3% and +4.0%, one seed) and is not used. Ions (4, 8) with electrons (2, 8),
48 moments per point, also sit inside the scatter; they are not adopted because
this scan is electrostatic and one seed deep there.

The electromagnetic deck of ``examples/05_kinetic_electrons`` (:math:`\beta`
1e-4 with :math:`A_\parallel`, :math:`m_e/m_i = 2.7\times10^{-4}`, 32x32x16,
same window) against uniform (4, 16), two seeds where marked:

.. list-table::
   :header-rows: 1
   :widths: 24 10 8 12 14 12 14

   * - Ions / electrons, :math:`(N_l, N_m)`
     - Moments per point
     - Seeds
     - :math:`Q_i`
     - Against uniform (4, 16)
     - :math:`Q_e`
     - Against uniform (4, 16)
   * - (4, 16) / (4, 16)
     - 128
     - 2
     - 467 +- 10
     - +0.0 +- 2.9%
     - 669 +- 13
     - +0.0 +- 2.8%
   * - (4, 8) / (4, 16)
     - 96
     - 2
     - 465 +- 15
     - -0.6 +- 3.7%
     - 657 +- 32
     - -1.7 +- 5.2%
   * - (4, 8) / (4, 8)
     - 64
     - 1
     - 505 +- 12
     - +8.0 +- 3.4%
     - 699 +- 19
     - +4.5 +- 3.4%

The fallback is within 1.7% of the uniform (4, 16) in both fluxes, inside the
error of 3-5%; uniform (4, 8) is +8.0% and +4.5% in one seed. The scans do not
cover a stellarator, a larger :math:`k_y` range or a collisional deck, where
the ion :math:`N_l` and the electron :math:`N_m` named in the deck should be
checked as for any resolution.

**Cost of per-species counts.** One A4000, float32, 64x64x24, explicit ``rk3`` at
a fixed step (time per step from a long and a short run), peak device memory:

.. list-table::
   :header-rows: 1

   * - Counts
     - ms per step
     - Peak memory
   * - uniform (4, 16)
     - 34.9
     - 0.89 GB
   * - ions (4, 8), electrons (4, 16)
     - 31.3 (1.12x)
     - 0.67 GB
   * - uniform (8, 32)
     - 161.3
     - 3.29 GB
   * - ions (8, 24), electrons (4, 32)
     - 110.3 (1.46x)
     - 2.32 GB
   * - ions (8, 16), electrons (4, 32)
     - 84.4 (1.91x)
     - 2.26 GB

One right-hand side at (12, 48): 147.6 ms and 5.5 GB uniform, 86.5 ms and
3.1 GB with ions (12, 32) and electrons (6, 48). Small states are launch-bound
(two groups launch twice the kernels), so the gain grows with the state. The
state arrays keep the ``[run]`` extents, which sets the floor on memory. At
64x64x24 and (4, 16) that floor is 52 MB (two species, complex64) of the 0.67 GB
peak: the peak is the right-hand-side temporaries, which the groups already
shrink, and storing each species at its own counts would save 13 MB more, so
the state stays dense.

**Implicit (IMEX) factor.** ``imex-ars3``/``imex-ark3`` build the block-Thomas
factor, its field coupling and the collision sub-step per species group, at the
counts each species keeps (zero rows above them stay the identity), so memory and
solve time follow the counts. One A4000, float32, state at the ``[run]`` counts,
electrons (4, 32), factor build plus one solve and one collision sub-step:

.. list-table::
   :header-rows: 1

   * - Grid, ions
     - Factor, padded / per group
     - Peak memory
     - Solve, ms
     - Build, s
   * - 32x32x16, uniform (4, 32)
     - 0.93 / 0.93 GB
     - 1.29 / 1.29 GB
     - 24.5 / 24.0
     - 63 / 62
   * - 32x32x16, ions (4, 16)
     - 0.93 / 0.85 GB
     - 1.30 / 1.22 GB
     - 25.5 / 21.0
     - 65 / 63
   * - 32x32x16, ions (8, 16)
     - 3.22 / 0.96 GB
     - 5.56 / 1.61 GB
     - 48.2 / 22.2
     - 75 / 68
   * - 64x64x24, ions (4, 16)
     - 8.99 / 8.16 GB
     - 10.65 / 9.98 GB
     - 137 / 117
     - 183 / 136
   * - 64x64x24, ions (8, 16)
     - out of memory (16 GB)
     - 12.33 GB
     - 283
     - 166

The electron mirror blocks (Laguerre-coupled) dominate the factor, so the gain is
large when the state's ``Nl`` is set by the ions. Uniform counts give the same
factor to round-off (solve 2e-14 relative, collision sub-step bitwise). On a
loaded 10-core CPU the 32x32x16 ions (8, 16) build took 254 s against 3819 s
padded (16 against 9 GB resident). The state and restart
arrays keep the ``[run]`` extents (zeros above a species' count). Hermite-sharded
routes, a custom collision operator and the per-term diagnostics refuse unequal
counts with ``NotImplementedError``.

Kinetic electrons
-----------------

- Leave ``fixed_dt = false`` with an explicit scheme; read the bound in
  ``diagnostics.cfl_scales``.
- Use a small finite :math:`\beta` with :math:`A_\parallel`
  (:math:`\beta = 10^{-4}`). At :math:`\beta = 0` the electrostatic
  :math:`\omega_H` mode cuts the step a further 6 times.
- For growth rates use ``solver = "krylov"`` or ``"adaptive"``.
- Time: allow :math:`10/\gamma` for the linear phase, saturation by
  :math:`t\approx 60`-100, then 100-200 for the average, or
  ``run_to = "saturation"``.
- **Check.** Rerun with ``damp_ends_rate`` 3x up and down and with
  ``nperiod + 1`` (below).

Electromagnetic and KBM
-----------------------

- :math:`\beta` in the deck sets :math:`A_\parallel`; check the mode against
  the next :math:`N_m`.
- The linear KBM branch moves with the end-damping rate and with ``nperiod``.
  At ``nperiod = 1`` the layer is a quarter of the line: use 2 or more.
- The Miller KBM needs more points along the line than the reference deck's
  ``ntheta = 32``. At :math:`\beta=0.015`, :math:`k_y=0.3`, ``nperiod = 2``, ``(16, 48)``,
  :math:`\gamma` is 0.3128, 0.2999, 0.2953 and 0.2920 at ``ntheta`` 32, 48, 64
  and 96, falling as :math:`1/n_\theta^2`; use 64 or more and rerun at 1.5x.
  ``Nm`` 96 moves it by 0.5%. The comparison with GS2 is in :doc:`validation`
  (``X-kbm-beta-gs2``).
- A time run that stops without a stationary growth window falls back to a
  log-linear window of at least 30% of the samples and warns. Read the slope of
  the late trace before trusting a rate from such a run; an electron-only start
  (``init_electrons_only``) shows a faster transient for the first 10 to 20 time units.
- Explicit steps must keep :math:`\nu_{end}\Delta t < 0.3`. An ``nperiod = 1``
  KBM deck (rate 598) blew up at :math:`\Delta t = 10^{-3}` and ran at
  :math:`5\times10^{-4}`.
- Eigenvalues: ``auto`` (banded, else ``adaptive``) certifies KBM; a shift-invert
  with a poor shift can certify a non-dominant branch.

End damping
-----------

One rate, ``[time] damp_ends_rate``, applies on every route and no time step
enters it. The default is 30 e-folds per thermal transit of the fastest kinetic
species across the end layer, and every run prints it. Below about 1 e-fold
the kinetic-electron and KBM rates are not converged. Rerun at 3x above and
below the printed rate and at ``nperiod + 1``; a converged rate moves by less
than its tolerance in all three (:ref:`end-damping`).

Collisions
----------

The default operator is the linearized Landau operator (:doc:`operators`). It
needs the velocity resolution of the case; a factor is built once (0.5 s at
(4, 8), tens of seconds at (16, 48)) and cached. Set ``nu`` per species;
``lenard_bernstein`` is cheaper and not accurate at high :math:`\nu`
(21% low on a Cyclone scan at :math:`\nu_i = 0.1`).

Eigensolver
-----------

``auto`` (default) runs the Hermite-banded direct shift-invert where its blocks
fit in memory and ``adaptive`` otherwise; ``adaptive`` is Arnoldi on the
time-advance map, as GS2 does with SLEPc. Both certify the pair against the full
operator. Use matrix-free shift-invert only for a chosen interior branch
(:doc:`algorithms`).
