Scope of a sharded IMEX step for kinetic electrons
==================================================

The Hermite-sharded nonlinear route (:ref:`unequal-moment-mesh`) advances the explicit
steppers on several devices. The default stepper of a kinetic-electron deck at
:math:`\beta \le 3\times10^{-4}`, adaptive ``imex-ark3``, keeps the whole-state placement.
This note asks whether that gap is worth closing and answers with measurements.

**Decision: (C), neither.** The sharded explicit route stays an opt-in for decks that
name an explicit stepper, and the sharded IMEX step is not built until one of the two
triggers at the end holds. Reasons, in the order of the questions:

1. With kinetic electrons the explicit step is bounded by the end absorber at the default
   rate and, with the absorber removed from the bound, still by electron streaming and by
   the bracket at a third (``rk4-pi``) to a sixth (``rk4``) of the IMEX bracket factor. The
   step ratio (15 to 3400 explicit steps per IMEX step) is far beyond what two devices
   recover, and on this interconnect the sharded explicit step is not faster than the
   single-device step at 32x32x16.
2. An exact treatment of the absorber exists and is stable at the streaming limit, but it
   needs the field-coupled part of the absorber, not its diagonal, and it is first-order
   accurate. It removes the absorber cap, not the streaming and bracket caps.
3. Sharding IMEX itself is feasible (420 to 540 lines, a Hermite relay or SPIKE solve). On
   two PCIe-connected devices the collectives of one step cost as much as the work they save
   at 32x32x16, and the model gains 1.6 times at 64x64x16 with :math:`(N_l, N_m) = (8, 16)`.

Conditions
----------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - item
     - value
   * - code
     - the Hermite-sharded route of the stacked parent pull requests, with the IMEX
       multi-chunk fix and the absorber-aware explicit step bound applied locally for the
       measurements (neither is part of this change)
   * - decks
     - ``examples/05_kinetic_electrons/case.toml`` (Cyclone, :math:`m_e/m_i = 2.7\times10^{-4}`,
       kinetic ions and electrons, :math:`(N_l, N_m) = (4, 16)` for both species, the
       per-species default), electromagnetic at :math:`\beta = 10^{-4}`, and the same deck
       electrostatic; no collisions, which the sharded route does not apply
   * - grids
     - :math:`32\times32\times16` for radii, step bounds and the explicit profiles,
       :math:`16\times16\times16` for the rate scan, 8 x 8 perpendicular modes for the
       two-device IMEX trace
   * - devices
     - two RTX A4000 (PCIe, no direct peer link), shared with other jobs
   * - precision
     - ``float32`` / ``complex64`` on the devices, ``float64`` on a CPU for operator analysis

Wall times to a fixed :math:`t` were not taken. The devices were not idle: another job ran
at up to 97% utilization on one of them during the profiled calls. The host load average
stayed above 130 on 36 cores with swap at 97%, which stretches every host-side phase (factor
build, compile, dispatch, collective launch skew) by an order of magnitude and killed two
:math:`32\times32` IMEX runs and the profiled :math:`16\times16` one. The numbers below are
device-side kernel times read from profiler traces of the explicit routes, isolated
collective latencies, compiled flop and byte counts of the IMEX pieces on a CPU, and steps
needed from the controllers' step bounds.

Step 1: cost per step and steps per unit time
---------------------------------------------

The explicit routes were profiled on the devices: the kernel time of a warm call at two
horizons (:math:`t = 0.004` and :math:`0.01`) gives the cost per step by difference, which
removes the setup kernels. The IMEX step could not be profiled (below) and is estimated
from the compiled cost analysis of its pieces on a CPU, for the 32x32x16 deck
(:math:`(N_l, N_m) = (4, 16)`, two species, ``complex64``): one bracket right-hand side
0.67 GB and 1.4 GFLOP (1.5 ms at the 448 GB/s of the device), one stiff matvec 0.43 GB, one
chain solve 2.9 GB (6.4 ms) and 502 MB of factors. An RK4 step is four right-hand sides,
5.9 ms by that model; an ARK3 step is three right-hand sides, four matvecs and three
solves, 27.6 ms (49.8 ms when the factor check asks for refinement solves). The model
reproduces the explicit step it was checked against (4.6 ms measured on one device). It is
a bandwidth bound: at this size the explicit step is a train of about 600 kernels of 8
microseconds, launch-bound rather than bandwidth-bound, which is why halving the data over
two devices does not halve the time.

.. list-table:: 32x32x16, (4, 16), EM :math:`\beta = 10^{-4}`, two A4000 (ES in brackets where it differs)
   :header-rows: 1
   :widths: 33 17 13 17 20

   * - route
     - :math:`dt`
     - steps per unit :math:`t`
     - device ms per step
     - device s per unit :math:`t`
   * - (a) ``rk4-pi``, one device, default rate (profiled)
     - 2.9e-5 (mean)
     - 3.4e4
     - 4.6 measured
     - 158
   * - (b) ``rk4-pi`` sharded, two devices, default rate (profiled)
     - 5.1e-5 (mean)
     - 2.0e4
     - 33 per device measured (28 in collectives, waiting on launch skew); 4.8 compute + 2.5 to 4 collectives at isolated latencies
     - 660 measured; 150 to 180 at isolated latencies
   * - explicit RK4 with the exponential of the absorber, one device (prototype, step bound = streaming)
     - 3.4e-3 (8.1e-4)
     - 294 (1235)
     - 4.6 plus six exponentials (not measured)
     - 1.4 (5.7) plus the exponentials
   * - the same, two devices
     - 3.4e-3 (8.1e-4)
     - 294 (1235)
     - 7.3 to 8.8 plus the exponentials
     - 2.1 to 2.6 (9.0 to 10.9) plus the exponentials
   * - (a) ``imex-ark3``, one device
     - 0.05 to 0.1
     - 10 to 20
     - 27.6 model (49.8)
     - 0.28 to 0.55 (0.5 to 1.0)
   * - (c) ``imex-ark3``, whole-state placement, two devices
     - 0.05 to 0.1
     - 10 to 20
     - not measured
     - not measured
   * - ``imex-ark3`` sharded over Hermite, two devices (model, step 3)
     - 0.05 to 0.1
     - 10 to 20
     - 29 to 48 (the compute term does not halve when launch-bound)
     - 0.29 to 0.96

Reading the table:

* The IMEX step is 0.05 to 0.1 on the tutorial deck (mean 0.15, range 0.04 to 0.23 over
  :math:`t = 40`, :math:`16^3`, one run) and 0.05 in the saturated phase of the 32x32 decks
  (:doc:`guidance`); the explicit step at the default rate is 3e-5. **IMEX takes 1700 to
  3400 times fewer steps**, and its modeled step costs 6 times more.
* The explicit step with the absorber out of the bound is bounded by the streaming radius
  (825 EM, 3458 ES, measured by Arnoldi). It is still 15 to 125 times shorter than the IMEX
  step: the IMEX controller takes the bracket at :math:`17/\omega_{nl}`, the explicit
  RK4 at :math:`2.7/\omega_{nl}`.
* The sharded explicit step is slower than the single-device step at this size, and
  measured compute per device does not fall (4.8 ms against 4.6). The profile counts 13
  all-reduces, 2 halo exchanges and 1 all-gather per step. On this PCIe pair a collective
  costs 0.1 to 0.4 ms in isolation (psum 0.16 ms at 4 to 70 kB, 0.37 ms at 557 kB, 1.1 ms at
  2.2 MB), 2.5 to 4 ms per step; with the host contended the collective kernels wait for the
  peer's launch and take 28 of the 33 ms.
* The whole-state placement of IMEX is not idle on the second device: in a two-step call at
  8 x 8 perpendicular modes the trace shows 177 all-reduce, 77 send-receive and 44 all-gather kernels. XLA
  partitions the step without knowing the Hermite structure. No timing was taken.
* Wall times to a fixed :math:`t` could not be taken: the devices were shared (another job
  at up to 97% utilization on one of them), the host load average stayed above 130, and the
  :math:`32\times32` IMEX runs and the profiled :math:`16\times16\times16` one were killed
  by the memory manager (swap at 97%). A cold :math:`16\times16\times16` IMEX call took 20
  minutes of host time before the first step.

Step 2: the end absorber
------------------------

**It is not diagonal in the state.** The absorber damps the non-adiabatic
:math:`H = G + B\,\mathrm{fields}(G)`, so :math:`A = -e\,(I + B F)` with :math:`e(k_y, z)` one
rate per grid point, :math:`F` the field solve and :math:`B` its return into the
:math:`m = 0, 1` moments. :math:`BF` has rank :math:`n_f` (:math:`\varphi`, :math:`A_\parallel`)
at each point and a spectral radius 724 times :math:`\nu_{end}` on the small box below, 470 times on the
:math:`16\times16\times16` deck and 140 times on the :math:`32\times32\times16` deck.

.. list-table:: Spectral radii, EM :math:`\beta = 10^{-4}`, :math:`\nu_{end}` = 598
   :header-rows: 1
   :widths: 46 18 18 18

   * - quantity
     - 4x4x8 box, (2, 3)
     - 32x32x16, (4, 16) EM
     - 32x32x16, (4, 16) ES
   * - streaming and hypercollisions :math:`\rho(S)`
     - 261
     - 825
     - 3458
   * - analytic streaming bound used before the absorber was measured
     - --
     - 1002
     - 4305
   * - diagonal of the absorber :math:`\rho(D)`
     - 598
     - 598
     - 598
   * - absorber alone :math:`\rho(A)`
     - 4.3e5
     - --
     - --
   * - :math:`\rho(S + A)`
     - 4.3e5
     - 8.4e4
     - 8.4e4
   * - :math:`\rho(S + A - D)`: what an integrating factor on the diagonal leaves explicit
     - 4.3e5
     - --
     - --
   * - RK4 step :math:`2.785/\rho` with the absorber
     - 6.4e-6
     - 3.3e-5
     - 3.3e-5
   * - RK4 step without the absorber
     - 1.1e-2
     - 3.4e-3
     - 8.1e-4

The integrating factor on the diagonal of the absorber leaves the explicit step where it
was. The one that works is on all of :math:`A`: because :math:`e` is one number per point,
:math:`\exp(-s(I + BF)) = e^{-s}\,(I + B\,g(sK)\,F)` with :math:`K = FB` an :math:`n_f\times n_f`
matrix per point and :math:`g(sK) = (e^{-sK} - I)K^{-1}`. One application costs a field
solve and one :math:`B`. The prototype is
``benchmarks/absorber/absorber_exponential.py``; the exponential agrees with a dense
``expm`` to :math:`3\times10^{-16}` (float64, :math:`h\rho(A)` up to 430) and
``tests/unit/nonlinear/test_absorber_exponential.py`` keeps that comparison in float32.

.. list-table:: Spectral radius of the one-step amplification (linear, 4x4x8 box, :math:`\nu_{end}` = 598, :math:`\rho(S) = 261`)
   :header-rows: 1
   :widths: 16 16 22 22 24

   * - :math:`h`
     - :math:`h\rho(S)`
     - explicit RK4
     - integrating factor on :math:`D`
     - integrating factor on :math:`A`
   * - 5.4e-6
     - 0.001
     - 1.0
     - 1.0
     - 1.0
   * - 1.5e-5
     - 0.004
     - 38
     - 38
     - 1.0
   * - 3.9e-5
     - 0.010
     - 2.8e3
     - 2.7e3
     - 1.0
   * - 2.1e-3
     - 0.54
     - 2.6e10
     - 7.7e9
     - 1.0
   * - 5.6e-3
     - 1.45
     - 1.4e12
     - 5.0e10
     - 1.0
   * - 1.5e-2
     - 3.9
     - 7.4e13
     - 9.4e9
     - 5.4

The integrating factor on :math:`A` is stable up to :math:`h\rho(S) \approx 2.6`, the RK4
limit of :math:`S` alone: 1600 times the step of the explicit route on this box. It is not
free: on the nonlinear box of ``benchmarks/absorber/absorber_nonlinear.py`` (8x8 perpendicular
modes, 16 parallel points, :math:`(N_l, N_m) = (2, 4)`, bracket 19% of the right-hand side) the state error against RK4 at :math:`h = 2\times10^{-6}`
(whose self-difference to :math:`h/2` is :math:`3.5\times10^{-9}`) is

.. list-table:: Integrating factor on :math:`A`, nonlinear 8x8x8 box, state error at :math:`t = 0.05`
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - :math:`h`
     - :math:`h\rho(S)`
     - steps
     - error, whole state
     - error, :math:`\varphi`
   * - 5e-4
     - 0.40
     - 100
     - 0.7%
     - 1.5%
   * - 1e-3
     - 0.79
     - 50
     - 1.4%
     - 3.0%
   * - 2e-3
     - 1.6
     - 25
     - 2.6%
     - 5.6%
   * - 4e-3
     - 3.2
     - 12
     - unstable
     - unstable


The error is first order in :math:`h` in the interior as well as in the absorber layers,
which is the order reduction of an integrating factor with a stiff operator that does not
commute with streaming. At the streaming limit it is 2 to 5 percent of the state, against
:math:`10^{-3}` per step for the IMEX controller at a step 100 times larger.

**Lowering the rate.** The rate fixes the cap linearly (the gain
:math:`\rho(S + A)/\nu_{end}` is independent of the rate) and fixes the physics.
Tutorial deck, :math:`16\times16\times16`, IMEX at the default controller, :math:`t = 40`
(the deck is still in the linear phase, so these are early-time values):

.. list-table:: Heat flux :math:`Q_i / Q_e` (gyroBohm), 16x16x16 EM deck, IMEX
   :header-rows: 1
   :widths: 14 22 22 22 20

   * - :math:`\nu_{end}`
     - :math:`t = 10`
     - :math:`t = 20`
     - :math:`t = 40`
     - explicit step
   * - 598
     - 1.0e-6 / 9e-9
     - 6.3e-7 / 7.0e-8
     - 3.8e-6 / 5.2e-7
     - 9.8e-6
   * - 60
     - 1.2e-6 / 2.6e-8
     - 8.9e-7 / 1.1e-7
     - 3.4e-6 / 5.6e-7
     - 9.8e-5
   * - 6
     - 2.6e-6 / 4.9e-7
     - 6.0e-6 / 1.8e-6
     - 1.9e-5 / 6.3e-6
     - 1.0e-3
   * - 0.1
     - 8.6 / 7.7
     - 1.2e4 / 1.4e4
     - 1.5e3 / 2.0e3
     - 3.3e-3


A rate of 60 moves the ion heat flux by +11%, +41% and -8% at :math:`t = 10, 20, 40`, which
is the size of the transient's own variation and says nothing about saturation, and lifts the
explicit step from :math:`9.8\times10^{-6}` to :math:`9.8\times10^{-5}`, still 35 times below
the streaming limit. A rate of 6 brings the absorber to within a factor 3.4 of the streaming
radius (step :math:`1.0\times10^{-3}`) but changes the heat flux by a factor 5 at :math:`t = 40`, and
0.1, the value the example used to pin, grows without bound. Lowering the rate does not
remove the cap before it changes the answer.

Step 3: sharding the IMEX step
------------------------------

**What a step needs.** One adaptive ARK3 step is four stages: stage 1 is explicit and
each of stages 2 to 4 solves :math:`(I - \gamma\,dt\,L)\,y = r` (``ChainImplicitLinear.solve``,
plus one refinement solve when the build check asks for it), with :math:`L` the stiff linear
operator (streaming with its dissipation, the field response, the end absorber and the electron
mirror). Per step that is three solves, four matvecs with :math:`L`, three bracket
right-hand sides, the embedded error norm and, between chunks, the step controller
(``ImexDtControl``: one probe step and an Arnoldi radius of the explicit remainder).

.. list-table:: Which operators split over Hermite blocks
   :header-rows: 1
   :widths: 30 22 48

   * - operator
     - per-block
     - what the sharded step needs
   * - :math:`L` matvec, bracket right-hand side, projector, ``ky`` rebuild
     - yes, with a width-2 halo
     - the terms of the explicit route (halo ``ppermute`` and the :math:`m \le 3` head
       ``psum``), with the explicit term set restricted to :math:`L`
   * - chain gather and scatter over :math:`(k_y, k_x)`
     - yes
     - nothing: the chains index :math:`(k_y, k_x, z)`, not :math:`m`
   * - block Thomas over Hermite in ``_thomas_solve``
     - **no**
     - the off-diagonal blocks couple :math:`m \pm 1` with a ladder scalar times the
       parallel-gradient matrix :math:`D`: a forward and a back substitution across blocks
   * - Woodbury correction (:math:`W`, capacitance, :math:`U`)
     - yes, after one reduction
     - :math:`W` sums :math:`(s, l, m \le 3)` into :math:`n_f` fields per chain point: the
       same ``psum`` as the field solve; the capacitance is replicated and :math:`U` writes
       :math:`m \le 1`, which sits on the first block
   * - collision sub-step (``_LocalCollisions``)
     - no
     - dense per point in :math:`(l, m)` of a species: needs an all-to-all of the state, and
       the sharded explicit route excludes active collisions already
   * - factor build, refactor on a step change
     - not as written
     - the Schur inverses ``dinv`` chain from :math:`m = 0` upward, so shard :math:`d` waits
       for shard :math:`d-1`; the SPIKE form below builds the blocks independently

**Communication.** A relay solve passes the boundary vector (one :math:`m` row for all
:math:`(s, l)`, 8704 points per row on the 32x32x16 deck, 557 kB in ``complex64``) across
each block boundary on the forward and on the back sweep of each of the two Thomas passes:
four hops of 0.56 MB per solve for two devices, **2.2 MB and about 1.4 ms** at the measured
0.35 ms per hop, plus the field ``psum`` (0.17 ms). The devices alternate (half the
solve idle). A SPIKE (partitioned) solve gives each block its own Thomas, two precomputed
"spike" matrices per block, and one all-gather of the two boundary rows per pass, so both
devices work at once: the same volume, 1 all-gather per pass, and about 1.2 times the local
flops. Per step, three solves cost 4 to 5 ms of collectives (7 to 9 ms with the refinement
solves), and the explicit terms add what the explicit route pays (13 reductions, 2 halo
exchanges and 1 gather per RK4 step in the trace above, 2.5 to 4 ms at isolated
latencies). The factor memory halves per device (0.22 GB on the 32x32x16 deck, 3.9 GB at 64x64x16
with :math:`(N_l, N_m) = (8, 16)`, where the electron mirror makes :math:`N_l N` blocks), which
is a capacity gain only where a single device runs out.

**Size.** About 420 lines (relay) to 540 (SPIKE) in ``src``: a packed-carry stage that applies ``L`` and the
bracket (reuse of the grouped route, 60), the Hermite-split ``dinv`` blocks and the sharded
solve with the Woodbury ``psum`` (170 relay, 290 SPIKE), the ARK3 step and error estimate on
the packed carry (70), the controller probe and Arnoldi on the packed carry (60), the
routing, the status line and the identity gate (60), and a second set of tests against the
serial IMEX step (120 in ``tests``).


Recommendation
--------------

**(C): keep the sharded explicit route opt-in and do not build the sharded IMEX step yet.**

* (B) fails on every count. With the absorber at the default rate the explicit step costs
  158 device-seconds per unit time against 0.3 to 1.0 for IMEX, and no division over two
  devices closes a factor 150 to 500. With the absorber treated exactly (the prototype) it
  costs 1.4 s (EM) or 5.7 s (ES) on one device, still 1.5 to 20 times IMEX, because the
  bracket sets a step 6 times shorter than the IMEX controller takes and electron streaming
  sets a shorter one still. The sharded explicit step does not beat the single-device one at
  32x32x16 on this interconnect (7.3 to 8.8 ms against 4.6 even at isolated latencies).
  Lowering the rate to 60 buys a factor 10 of step and changes the early heat flux by 10 to
  40%; the factor needed is 150.
* (A) is the only sharded design worth costing, and its gain depends on the deck. The model
  puts the sharded step at 29 to 48 ms against 27.6 ms on one device at 32x32x16
  (collectives 15 to 20 ms against 13.8 ms of work, which halves only if the step is
  bandwidth-bound), and at about 190 ms against 307 ms (1.6 times) at 64x64x16 with
  :math:`(N_l, N_m) = (8, 16)`, where the solve traffic grows as :math:`N^2` and the boundary
  vectors as :math:`N`. That deck needs 7.2 GB of factors on one device. The cost is 420
  lines (relay) to 540 (SPIKE) in a package whose line budget is falling toward 45000.
* The prototype of question 2 stays in ``benchmarks/absorber`` and changes no default.

**Revisit (A) when either holds:**

1. a production deck's factors and state no longer fit one device (the 64x64x16,
   :math:`(8, 16)` deck is at 7.2 of 16 GB), or a single-device IMEX step above about 100 ms
   makes a 1.5 to 1.7 times gain on two devices worth 500 lines;
2. the devices share a link with collective latency below 30 microseconds (the
   13 to 40 collectives per step would then cost under 1 ms), which moves the break-even
   to the 32x32 decks.

Before building it, take the measurement this note could not: the whole-state IMEX step
on two quiet devices at 64x64x16, wall time to :math:`t = 60`, against one device. If XLA's own
partitioning already gives a gain there, the sharded solve has less to add.

Reproduce
---------

* Radii and step bounds: ``measured_streaming_frequency`` in
  ``gkx.operators.nonlinear.policies`` (``absorber_vs=0.0`` adds the absorber) on the deck
  of ``examples/05_kinetic_electrons``.
* Dense radii, the amplification table and the exponential of the absorber:
  ``python benchmarks/absorber/absorber_scope.py 598`` (float64, CPU, a few minutes).
* Compiled flop and byte counts: ``jax.jit(f).lower(G).compile().cost_analysis()`` of
  ``nonlinear_rhs_cached``, ``ChainImplicitLinear.solve`` and ``.linear``.
* Device kernel times and collective counts: ``jax.profiler.trace`` with
  ``ProfileOptions(python_tracer_level=0)`` around a warm call, summed per stream.
* Collective latency: one ``fori_loop`` of 400 ``psum``, ``ppermute`` or ``all_gather`` of an
  array of the given size in a two-device ``shard_map``.

