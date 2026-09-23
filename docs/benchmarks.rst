Benchmarks
==========

GKX's benchmark figures form a compact atlas that follows the usual gyrokinetic
comparison layout:

- linear growth-rate and real-frequency overlays versus ``k_y`` (or ``beta``);
- nonlinear time traces of heat flux, free energy, electrostatic field energy,
  and magnetic field energy when present;
- panels that separate exact diagnostic closures from broader stress lanes.

The pass/fail status of each benchmark lane lives in the evidence ledger.
Status: see :doc:`verification_matrix`. The release-level claim boundary is
in :doc:`release_scope`. This page describes how the figures are produced and
how to read them; it does not assign statuses.

Benchmark directory
-------------------

The root ``benchmarks/`` directory holds lightweight drivers, runtime TOML
inputs and result-index pointers. It is separate from ``examples/``: examples
teach workflows, benchmarks reproduce validation panels and comparison traces.
Generated outputs go to ``tools_out/`` or another scratch directory. Only
reviewed, compressed summary figures and small CSV/JSON metadata are tracked in
``docs/_static``.

``benchmarks/`` stays at the scale of small scripts and manifests; a contract
test caps its tracked payload at 200 kB. NetCDF files, restart
files, logs, profiler traces and exploratory plots stay outside git.

Drivers:

.. code-block:: bash

   python scripts/benchmarks/cyclone_linear_benchmark.py --outdir tools_out/cyclone_benchmark
   python scripts/benchmarks/kbm_linear_comparison.py
   gkx run --config benchmarks/cases/secondary_slab.toml
   python scripts/benchmarks/secondary_slab_workflow.py

The Cyclone driver takes its integrator settings from
``examples/01_linear_tokamak/case_full.toml`` (``t_max = 150``) and fits the last
30% of the horizon, ``t = 105--150``, well clear of the startup transient. A
contract test keeps the fit window tied to the deck's ``t_max``.

The KBM driver plots the reviewed fixed-beta ``ky`` comparison table
``docs/_static/comparison/kbm_reference_candidates.csv``. Regenerate that table
from a matched external output with ``scripts/comparison/compare_gx_kbm.py``.

Regenerate the atlas figures with:

.. code-block:: bash

   python scripts/artifacts/make_benchmark_atlas.py

The atlas builder reads its inputs from ``tools/benchmark_atlas_manifest.toml``
and writes a machine-readable summary to
``tools_out/benchmark_atlas_summary.json``, which records the provenance of
each panel.

Capability and matched-comparison contract
------------------------------------------

``benchmarks/capability_matrix.toml`` is the machine-readable source of truth
for feature scope. It prevents two errors: implying support because a related
equation exists, and diagnosing a solver mismatch before the two runs use the
same physical and numerical contract. The status column below is the
``status`` field of that file.

.. list-table:: Capability status (``benchmarks/capability_matrix.toml``)
   :header-rows: 1
   :widths: 34 20 46

   * - Capability
     - ``status``
     - Evidence
   * - Electrostatic flux-tube dynamics
     - ``validated``
     - Cyclone, Cyclone Miller, W7-X and HSX linear/nonlinear atlas gates
   * - Nonlinear :math:`E\times B` bracket
     - ``validated_scoped``
     - operator identity, de-aliasing, invariant and transport-window gates
   * - Electromagnetic ``A_parallel``/``B_parallel``
     - ``validated_scoped``
     - KBM and KAW linear gates; KBM nonlinear window gate
   * - Boltzmann and kinetic species
     - ``validated_scoped``
     - adiabatic-electron ITG atlas; kinetic-electron Cyclone and TEM are
       stress lanes
   * - Hermite-Laguerre velocity basis
     - ``validated``
     - orthonormality, streaming, gyroaverage and velocity-resolution gates
   * - s-alpha, Miller and VMEC geometry
     - ``validated_scoped``
     - Cyclone, Cyclone Miller, circular VMEC, W7-X and HSX comparisons
   * - Periodic and linked parallel boundaries; VMEC flux-tube boundary
       selection
     - ``validated_scoped``
     - exact-periodic, continuous-drift and fixed-aspect policy tests;
       fixed-aspect W7-X/HSX lanes
   * - Time integration and adaptive time policy
     - ``validated`` / ``validated_scoped``
     - observed-order, stability, restart and fixed-CFL gates
   * - Dominant-eigenmode solver
     - ``validated_scoped``
     - Krylov-versus-time evolution, residual, branch-continuity and
       eigenfunction gates
   * - Restart and resolved diagnostics
     - ``validated``
     - restart identity, schema, heat-flux, field-energy and resolved-spectrum
       tests
   * - Conserving Dougherty-like collisions
     - ``validated_limited_model``
     - moment damping and low-order momentum/temperature corrections; limited
       for trapped-particle and multispecies collisional physics
   * - Drift-kinetic six-gyromoment Sugama and Coulomb operators
     - ``validated_scoped``
     - published coefficient matrices, conservation, relaxation, dissipation
       and JVP/finite-difference gates
   * - Full linearized Sugama/Coulomb hierarchy
     - ``planned_research_lane``
     - needs field-particle terms, conductivity, zonal-damping and
       collisional-ITG benchmarks
   * - Multispecies conserving Dougherty operator
     - ``planned``
     - species-coupled conservation, null-space, adjointness and entropy gates
       required
   * - Independent ``k_y`` scans and UQ ensembles
     - ``validated``
     - CPU/GPU numerical-identity and strong-scaling gates
   * - Nonlinear multi-device domain decomposition
     - ``blocked``
     - the benchmark-grid whole-state route runs at 0.211x and fails
       trajectory identity
   * - Species/Hermite multi-device decomposition
     - ``planned``
     - diagnostic route only; see :doc:`parallelization`
   * - JAX autodiff and implicit gradients
     - ``validated_scoped``
     - AD/FD, tangent, implicit-eigenpair, conditioning and covariance gates
   * - Differentiable VMEC/Boozer optimization
     - ``validated_scoped``
     - ``vmex``/``booz_xform_jax`` parity, gradient holdout and scoped QA
       optimization gates
   * - Equilibrium :math:`E\times B` flow shear
     - ``planned_research_lane``
     - numerical gates pass; the fixed-step transport-response gate rejects the
       physical model (below)
   * - KREHM, Vlasov-Poisson, collisional-ETG, Beer/Smith closures,
       long-wavelength field limit, forcing and transport coupling
     - ``not_shipped``
     - outside the full-gyrokinetic release claim

A matched comparison must record the equations and normalization, geometry
coefficients and parallel boundary, species conventions, perpendicular and
velocity grids, initial condition and seed, precision and de-aliasing,
integrator and timestep policy, collision/dissipation settings, diagnostic
normalization, and the fit or transport window. A visually similar input file
is insufficient. This matters most for nonlinear saturation, where short
state-level agreement does not establish a converged heat-flux comparison.

Comparison-code provenance
^^^^^^^^^^^^^^^^^^^^^^^^^^

The comparison contract names GX revision ``bc2fe552`` with aggregate source
fingerprint ``sha256:bfaaadfa...20b``. The instrumented office source tree is
not a Git checkout and has a different fingerprint, ``sha256:436e403e...a004``;
the two provenances must not be interchanged. The comparison binary is a clean
rebuild of that revision against one OpenMPI 4.1.6, parallel netCDF 4.9.2 and
HDF5 1.14.5 stack; older binaries that mixed system and local libraries are
excluded. The full values are in the ``[metadata]`` block of
``benchmarks/capability_matrix.toml``.

The canonical Cyclone s-alpha probe runs 2145 steps to ``t = 10`` and reports
``(gamma, omega) = (0.101814, 0.286777)`` at ``ky = 0.3``. That reading is taken
while the mode is still ringing and is not a parity target. The same deck run to
its own ``t_max = 150`` settles at ``(0.093049, 0.281991)``, matching the
reference output GX ships for that case. A comparison against the ``t = 10``
value shows an apparent gap that is entirely transient.

``docs/_static/cyclone_runtime_parity_refresh.json`` records a bounded RK4 check
to ``t = 20`` at ``ky = 0.3``, ``N_l = 16``, ``N_m = 48``. The reference
late-window mean is ``gamma = 0.0958 +/- 0.0105``, ``omega = 0.2811 +/- 0.0144``;
GKX returns ``gamma = 0.0908``, ``omega = 0.2783``, relative differences of 5.3%
and 1.0%, inside the reference's own late-window spread. Its wall times are not
a speedup: the reference advanced 12 ``ky`` modes, GKX one.

Scope relative to GX
^^^^^^^^^^^^^^^^^^^^

GX is the mature baseline for conventional GPU nonlinear initial-value runs and
species/Hermite multi-device execution. GKX's distinct scope is its Python/JAX
API, differentiable objectives, implicit gradient paths, CPU execution, and
in-memory ``vmex``/``booz_xform_jax`` geometry. GKX does not claim the complete
finite-wavelength multispecies linearized Landau hierarchy: its reduced
Sugama/Coulomb slice is a separately gated research boundary.

Feature parity is intentionally not blanket parity. The required comparison
scope is the standard electrostatic/electromagnetic gyrokinetic system, not
GX's optional KREHM, Vlasov--Poisson, collisional-ETG, forcing,
transport-coupling or Beer/Smith closure paths. Differentiable eigen/objective
solves and the in-memory JAX geometry chain are GKX extensions rather than
comparison requirements.

Equilibrium :math:`E\times B` flow shear has validated coordinate, operator,
timestep and derivative foundations, but the predeclared fixed-step
transport-response gate fails: the internal fixed-IMEX windows drift and give a
4.82% increase, while independently stationary fixed-RK4 comparison windows
give a 24.82% increase. It is therefore a Python research API with no
input-file option. The negative-evidence record is
``docs/_static/flow_shear_fixed_step_response_gate.json``.

Tracked results index
---------------------

The root-level result index is ``benchmarks/results/manifest.toml``. It points
to the promoted benchmark figures and machine-readable tables without moving or
duplicating large run products. A contract test requires every entry's name,
path and claim scope to appear in this table; a result that is not listed here
stays unpromoted.

.. list-table:: Promoted benchmark result artifacts
   :header-rows: 1
   :widths: 22 30 18 30

   * - Result
     - Artifact (tracked, or regenerable render)
     - Claim scope
     - Regeneration path
   * - Core linear benchmark atlas
     - ``docs/_static/benchmark_core_linear_atlas.png`` (regenerable render;
       not tracked in git)
     - headline linear validation atlas
     - ``python scripts/artifacts/make_benchmark_atlas.py``
   * - Core nonlinear benchmark atlas
     - ``docs/_static/benchmark_core_nonlinear_atlas.png`` (regenerable
       render; not tracked in git)
     - headline nonlinear validation atlas
     - ``python scripts/artifacts/make_benchmark_atlas.py``
   * - README benchmark summary panel
     - ``docs/_static/benchmark_readme_panel.png`` (regenerable render; not
       tracked in git)
     - compact publication-facing benchmark summary
     - ``python scripts/artifacts/make_benchmark_atlas.py``
   * - Extended linear stress matrix
     - ``docs/_static/benchmark_extended_linear_panel.png`` (regenerable
       render; not tracked in git)
     - stress and provisional lanes, not headline validation claims
     - ``python scripts/artifacts/make_benchmark_atlas.py``
   * - Runtime and memory comparison
     - ``docs/_static/runtime_memory_benchmark.png``
     - tracked wall-time and memory comparison rows
     - ``python scripts/benchmarks/benchmark_runtime_memory.py --summary-glob ...``
   * - Runtime and memory result rows
     - ``docs/_static/runtime_memory_results_ship_refresh.csv``
     - machine-readable runtime/memory rows used by the tracked panel
     - ``python scripts/benchmarks/benchmark_runtime_memory.py``
   * - Runtime and memory summary
     - ``docs/_static/runtime_memory_summary_ship_refresh.json``
     - machine-readable summary for runtime/memory panel generation
     - ``python scripts/benchmarks/benchmark_runtime_memory.py``
   * - Core linear atlas inputs
     - ``tools/benchmark_atlas_manifest.toml``
     - manifest of small tracked benchmark inputs
     - ``python scripts/artifacts/make_benchmark_atlas.py``
   * - Reference-code linear parity matrix
     - ``docs/_static/gkx_gx_linear_parity_matrix.png`` (regenerable render;
       not tracked in git)
     - cross-code linear growth-rate and frequency parity across tokamak and stellarator cases
     - ``python scripts/comparison/build_gx_parity_matrix.py``
   * - Reference-code linear parity rows
     - ``docs/_static/gkx_gx_linear_parity_matrix.csv``
     - machine-readable per-wavenumber parity rows with convergence flags
     - ``python scripts/comparison/build_gx_parity_matrix.py``
   * - Reference-code linear parity summary
     - ``docs/_static/gkx_gx_linear_parity_matrix.json``
     - per-case resolution, provenance and cost metadata behind the parity rows
     - ``python scripts/comparison/build_gx_parity_matrix.py``

The four atlas panels are regenerable renders, not git-tracked files: their
checksums are recorded under the ``regenerate_on_demand`` action in
``tools/release_artifact_manifest.toml``. PDF copies are written next to each
PNG for manuscript use.

Metrics
-------

The atlas uses one small set of observables throughout:

- growth rate ``gamma`` and real frequency ``omega``;
- ion heat flux;
- free energy;
- electrostatic field energy (output variable ``Wphi``);
- magnetic field energy when ``A_parallel`` or ``B_parallel`` is active.

The README packs these into one validation panel plus one separate
runtime/memory panel.

Atlas contents
--------------

The core panels show the full-gyrokinetic lanes used in the regression
workflow:

- Cyclone ITG, s-alpha, linear and nonlinear (the standard ITG benchmark
  [Dimits00]_);
- KBM linear and nonlinear;
- W7-X VMEC linear and nonlinear;
- HSX VMEC linear and nonlinear (the Nuhrenberg-Zille QHS deck);
- Cyclone Miller geometry linear and nonlinear.

The linear master panel also shows ETG, KAW and the KBM Miller late-growth
replay. The extended strip shows Cyclone with kinetic electrons and TEM, which
are stress lanes rather than validation claims. Appearing in a panel is not a
validation status. Status: see :doc:`verification_matrix`; several of the core
linear lanes are ``provisional`` there.

The nonlinear panels are window comparisons: Cyclone, Cyclone Miller, KBM,
W7-X and HSX pass a mean-relative window gate against self-run GX
(``docs/_static/nonlinear_<case>_gate_summary.json``). That is not a
statistical validation of saturated transport (see :doc:`release_scope`). HSX
uses the ``t <= 50`` trace and W7-X the ``t <= 200`` trace; both are
long-window comparisons, not small-tolerance claims for every late-time sample.

The full-gyrokinetic ETG nonlinear pilot (ETG turbulence as in [Dorland00]_
[Jenko00]_) appears in the summary panel but is a short-window pilot and is
outside the nonlinear claim.

Reading the panels
------------------

The atlas mixes two kinds of evidence:

- benchmark overlays over scanned parameters such as ``k_y`` or ``beta``;
- exact-window or exact-diagnostic closures on selected lanes.

Only the exact-window closures are strict small-tolerance gates. In the
tracked set these are the KAW exact diagnostic window
(``docs/_static/kaw_exact_growth_dump.csv``) and the KBM Miller late-growth
replay (``docs/_static/kbm_miller_exact_growth_dump.csv``). The scanned panels
are coverage figures, not universal ``rtol <= 3e-2`` claims for every tile.

Benchmark-specific replay settings stay in the builders under ``tools/``. They
are not generic runtime defaults for the solver or the example drivers.

Reusable reference loaders and comparison policies live in
``gkx.benchmarking_shared``; time stepping, scans, geometry and physical
operators use the same runtime and solver APIs as ordinary simulations.
Pointwise scans and representative-mode extraction belong to
``gkx.workflows.linear``, which runs one linear solve per ``k_y`` and owns
iteration, result assembly, mode selection and fit-window extraction.
Benchmark drivers supply only case policy.

Supplementary closure artifacts
-------------------------------

Some lanes are tracked as machine-readable closure artifacts rather than atlas
tiles. Their comparison figures are regenerable renders.

- ``docs/_static/nonlinear_cyclone_short_gate_summary.json``: short nonlinear
  Cyclone replay with the short-reference dissipation contract. It is an
  exploratory short-transient diagnostic; the remaining mismatch is localized
  in resolved ``k_y`` field-energy diagnostics.
- ``docs/_static/comparison/secondary_reference_out_compare.csv``: secondary
  stage-2 mode table from the dense ``kh01a`` GX replay.
- ``docs/_static/nonlinear_w7x_gate_summary.json``,
  ``docs/_static/nonlinear_hsx_gate_summary.json`` and
  ``docs/_static/nonlinear_kbm_gate_summary.json``: long-window nonlinear
  lanes. Figures: ``scripts/comparison/make_reference_panels.py``.
- KBM eigenfunction overlap on the tracked KBM candidate table, a
  branch-identity diagnostic. Figure:
  ``scripts/artifacts/generate_linear_reference_overlays.py overlap-summary``.
  The raw mode-shape evidence is tracked as JSON gate reports and GKX traces
  under ``docs/_static/reference_modes/``, with frozen GX raw-mode bundles under
  ``docs/_static/comparison/reference_modes/``.

Extended stress matrix
----------------------

The extended linear panel keeps lanes visible without folding them into the
headline set:

- Cyclone kinetic electrons;
- TEM (literature-digitized reference);
- KBM Miller exact late-growth window.

The kinetic-electron scan is defined by
``examples/05_kinetic_electrons/case_full.toml`` and integrates
with fixed-step RK4 through the unified runtime API. Its reference seed,
linked-boundary damping, species and electromagnetic toggles are explicit in
that file.

The TEM input is ``benchmarks/cases/tem_linear.toml`` (fixed-step
RK2, electron-only Gaussian moment initialization), run through the same scan
path users call. The shipped ``tem_reference.csv`` is digitized from the
literature rather than taken from a GX benchmark output, and the exact case
definition behind it is not reconstructed, so TEM is a stress lane, not a
parity result. The audit ``docs/_static/tem_branch_parity_audit.json`` records
the open branch mismatch: maximum absolute relative growth-rate mismatch
``4.25``, maximum absolute relative frequency mismatch ``3.3`` after excluding
the near-zero reference denominator, one growth-rate sign mismatch, three
frequency sign mismatches, and a frequency-branch Spearman coefficient near
``-0.986``.

Case groups
-----------

Tokamak core cases
^^^^^^^^^^^^^^^^^^

- Cyclone ITG: linear ``gamma``/``omega`` scans and nonlinear transport traces
- KBM: linear branch-following scans and nonlinear transport traces
- Cyclone Miller geometry: imported linear scan and nonlinear time-trace audit

Stellarator core cases
^^^^^^^^^^^^^^^^^^^^^^

- W7-X VMEC: short-window linear scans and nonlinear transport traces
- HSX VMEC: short-window linear scans and nonlinear transport traces

Reduced and stress cases
^^^^^^^^^^^^^^^^^^^^^^^^

- ETG: linear benchmark overlays
- KAW: exact late-time diagnostic reconstruction and exact-window energy audit
- KBM Miller: exact late-time growth replay on the tracked low-``k_y`` branch
- Cyclone kinetic electrons: extended linear stress lane
- TEM: extended linear stress lane
