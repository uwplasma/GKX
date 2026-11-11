Validation
==========

Validation separates operator identities, numerical convergence and physical
comparison. Passing one layer does not establish the others. Run checks on
the source revision, precision and backend used for the intended calculation.

Operator and integration checks
-------------------------------

The test suite covers spectral derivatives, basis and quadrature identities,
field solves, collision invariants and dissipation, explicit and implicit
time integration, restart continuity and diagnostic normalization. Tests for
optimized or parallel routes compare them with the serial reference operator.

Run focused tests from the repository root, for example:

.. code-block:: bash

   python -m pytest tests/validation/physics_gates/test_collision_physics.py
   python -m pytest tests/unit/objectives/test_autodiff_solver_objectives.py

These commands execute bounded checks. They do not rerun a complete physical
convergence campaign. See :doc:`testing` for the test groups and coverage policy.

Linear physical comparisons
----------------------------

Landau damping, wave dispersion, instability growth and zonal response each
need a declared analytical or independent numerical reference. State the
normalization, collision model and observable used by the reference. In a
finite Hermite expansion, phase mixing can recur; distinguish a transient
initial-value fit from an eigenmode of the regularized finite operator.

For initial-value fits, vary the analysis window and diagnostic cadence and
check competing modes. For eigenmodes, inspect the full-operator residual,
branch identity and conditioning. Increase velocity and spatial resolution,
and vary timestep, domain and numerical regularization independently.
Residual certification is an algebraic check on the chosen discretization.

Cross-code protocol
-------------------

Match geometry profiles and coordinates, normalization, species, kinetic or
adiabatic responses, electromagnetic terms, grid and boundary conditions,
initialization, precision, timestep policy, collisions and regularization.
Check raw fields and diagnostics before fitting rates. Compare complex-field
phase conventions as well as amplitudes; a frequency sign or reference-speed
conversion can otherwise look like a physical discrepancy.

Use the formulas in :doc:`conventions` and :doc:`normalization`. GKX, GX, GS2
and stella have different discretizations and collision-model options. A
matched finite-resolution result supports that specific case; it does not
establish a collisionless limit, a converged nonlinear transport prediction,
or agreement across all kinetic-electron and electromagnetic regimes.
Report unresolved discrepancies with the configuration and convergence tests
needed to distinguish their causes.

Physical reference cases include electron-temperature-gradient turbulence
[Dorland00]_ [Jenko00]_, stellarator calculations [Sanchez21]_
[Papadopoulos23]_, quasilinear stellarator turbulence [Pueschel16]_ and stellarator
transport and zonal-flow studies [Hegna18]_ [McKinney19]_ [Tiwari25]_. Match
the published model and analysis protocol before interpreting agreement.

Comparison with GX, GS2 and stella
----------------------------------

The three comparison figures in the README plot GKX, GX, GS2 and stella on the
same flux-tube problems.
``docs/figures/four_code_comparison.json`` holds every plotted value, and
``python docs/figures/four_code_comparison.py`` redraws the figures from it.
The raw runs are not scripted in the repository. The GKX decks are in
``benchmarks/cases`` and ``examples``; the GX, GS2 and stella columns need
those codes installed and the settings listed here.

**Problems and units.** Circular Miller Cyclone (``a/L_T = 2.49``,
``a/L_n = 0.8``, ``q = 1.4``, ``s_hat = 0.8``) with adiabatic or kinetic
electrons (``T_e = T_i``, ``m_i/m_e = 3670``), and the W7-X standard VMEC
configuration (torflux 0.64, adiabatic electrons), all in GX units. GS2 and
stella use ``sqrt(2T/m)`` as the reference speed; their ``gamma``, ``omega``,
``k_y``, time and heat flux are converted with ``gamma_GX = sqrt(2) gamma_GS2``,
``k_y,GX = k_y,GS2/sqrt(2)``, ``t_GX = t_GS2/sqrt(2)`` and
``Q_GX = 2 sqrt(2) Q_GS2``.

**Linear growth rates.** Each code runs at its own converged resolution: the
cheapest whose growth rate stays within 1% of the code's highest-resolution
value over ``k_y rho_i = 0.2`` to ``0.5``.

* GKX and GX: ``(Nl, Nm) = (12, 32)``, ``Nz = 144``.
* GS2: 48 ``theta`` per ``2 pi`` x 3, 16 energies, 41 pitch angles.
* stella: ``nzed = 48`` x 3, ``nvgrid = 24``, ``nmu = 12``.

The adiabatic Cyclone growth rate at ``k_y rho_i = 0.3`` is 0.1255 (GKX, GX),
0.1254 (GS2) and 0.1252 (stella). GKX and GX share the Hermite-Laguerre model
and agree to 0.1% for ``k_y rho_i >= 0.2``. GS2 and stella lie 1% below them at
``k_y rho_i = 0.4`` and 3.5% at 0.6, and with kinetic electrons they stay
within 6%. In the ``A_parallel`` scan GS2 is within 1% up to ``beta_ref = 0.01``
and 5-6% lower on the ballooning branch above it. On W7-X, stella is 2-4% below
for ``k_y rho_i >= 1``; for ``k_y rho_i <= 0.6`` near-degenerate modes make the
fitted branch depend on the code.

**Cost.** The linear time is one ``k_y`` to ``t v_ti/a = 60`` (GS2 and stella:
the same physical time), taken from the initial-value route for GKX and GX. The
certified eigensolver takes 43 s at ``(12, 32)`` on the same GPU
(:doc:`guidance`). GKX and GX ran on one RTX A4000, GS2 and stella on MPI ranks
of an 18-core Xeon W-2295 (4 ranks for the linear timings, 8 for the nonlinear
runs). The GPU and the CPU were shared with other jobs throughout.

.. list-table::
   :header-rows: 1
   :widths: 9 17 12 17 20 15 10

   * - Code
     - Hardware
     - DOF per ``k_y``
     - Linear time per ``k_y``
     - Nonlinear time to ``t = 400``
     - Nonlinear ``Q_i``
     - ``gamma``, ``k_y rho_i = 0.3``
   * - GKX
     - RTX A4000
     - 55,296
     - 4.2 s (10 s with compilation)
     - 523 s
     - 6.9 +- 0.2
     - 0.1255
   * - GX
     - RTX A4000
     - 55,296
     - 47 s
     - 1610 s (from the step rate)
     - 6.6 +- 0.3
     - 0.1255
   * - GS2
     - 4 cores linear, 8 cores nonlinear
     - 190,240
     - 24 s
     - about 33 h (264 core-h); run to ``t = 100``
     - not reached
     - 0.1254
   * - stella
     - 4 cores linear, 8 cores nonlinear
     - 83,520
     - 29 s
     - about 16 h (128 core-h); run to ``t = 140``
     - 6.1 +- 0.8 over ``t = 104 .. 140``, not converged
     - 0.1252

**Nonlinear Cyclone.** The deck of the GX paper uses
``(Nx, Ny, Nz) = (192, 64, 24)``, ``y0 = 28.2`` and adiabatic electrons, with
127 x 22 Fourier modes.

* GKX and GX use ``(Nl, Nm) = (4, 8)``, 2.1 M degrees of freedom, in single
  precision.
* GS2 uses 21 pitch angles x 8 energies x 25 ``theta`` (23.5 M; its growth rate
  is 1.2% below the converged value at ``k_y rho_i = 0.3``).
* stella uses 24 ``v_par`` x 6 ``mu`` x 25 ``z`` (10.1 M; 3% below).
* GS2 and stella have no end damping or hypercollisions. All four codes use
  hyperdiffusion with ``D = 0.05``.
* GS2 halves its time step to 0.01 (code units) at the first peak under its
  default CFL control; stella keeps 0.02.

GKX reaches ``t v_ti/a = 400`` in 523 s (``Q_i = 6.9 +- 0.2``) and GX in 1610 s
(``6.6 +- 0.3``; the paper gives 7.2 +- 0.8). The GKX run pinned the end damping
at 0.1; the default rate gives ``6.8 +- 0.2`` over two seeds
(:ref:`common-deck`). GS2 and stella take 2.3 and 2.0 s
per step on 8 ranks. In 5.9 h (47 core-hours each) they reach ``t = 100`` and
``140``; at that step rate and time step they need about 33 h and 16 h (264 and
128 core-hours) for ``t = 400``. These full-length times are extrapolations,
not measurements. Both codes reproduce the first peak and the relaxation that
follows. The stella mean over ``t = 104 .. 140`` is 6.1 +- 0.8 (GKX 7.3, GX 6.7
over the same interval) but spans five autocorrelation times, and GS2 has not
left the relaxation. The heat flux uses the same estimator for every code.

**Speed benchmark deck.** ``benchmarks/cases/cyclone_nonlinear_t400.toml``
(64x64x24, ``(Nl, Nm) = (4, 8)``) measures speed; it is not a converged
calculation. The saturated heat flux is ``Q = 7.3`` at ``Nx = 64`` and 5.2 at
``Nx = 192`` (9.1 and 5.0 with the end damping pinned at 0.1), and ``(4, 8)``
overestimates the linear growth rate by 9 to 86% at ``k_y = 0.3`` to 0.5.

Nonlinear transport
-------------------

Specify the saturation criterion, averaging interval and sampling cadence.
Check stationarity independently for every trace, account for autocorrelation
when estimating uncertainty, and repeat seeds. Save energy and resolved
field/flux spectra so spatial and velocity-space convergence can be assessed.
Compare smaller timesteps and longer windows as well as higher resolutions.
See :doc:`stellarator_optimization` for the paired-seed uncertainty estimator.

Derivative and performance checks
---------------------------------

Compare derivatives with finite differences across perturbation sizes and
window lengths. A finite-window adjoint supports a local trajectory
sensitivity; assess a design change with independent long simulations.
See :doc:`nonlinear_autodiff` and :doc:`differentiable_eigensolver`.

Measure performance only after checking numerical agreement. Separate
compilation from warm execution, wait for device completion and report the
hardware, software versions, allocation, workload and tolerances. See
:doc:`performance` and :doc:`parallelization`.

Reproducing results
-------------------

Keep input decks, analysis commands and source revisions with a shared result.
Write generated simulation, benchmark and test outputs outside version control.
Source physics tables and comparison input decks remain required inputs.
A published figure or summary cannot substitute for executable setup and
analysis instructions. The supported feature scope is in :doc:`release_scope`.
