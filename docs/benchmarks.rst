Benchmarks
==========

GKX’s benchmark figures are organized as a compact atlas instead of a
case-by-case gallery. The layout follows the standard gyrokinetic comparison
pattern:

- linear growth-rate and real-frequency overlays versus ``k_y`` (or ``beta``),
- nonlinear time traces of heat flux, free energy, electrostatic field energy,
  and magnetic field energy when present,
- compact panels that separate exact diagnostic closures from broader stress
  lanes.

The figures in this page are generated directly from tracked CSV assets,
curated comparison traces, and the small root-level result index under
``benchmarks/results``. Large run directories are deliberately excluded from
git.

Figure generation
-----------------

Lightweight benchmark drivers, runtime TOML inputs, and result-index pointers
live in the repository root under ``benchmarks/``. This is the canonical
benchmark entry-point directory for users and developers. It is intentionally
separated from ``examples/``: examples teach workflows, while benchmarks
reproduce validation panels and paper-facing comparison traces. Generated
outputs should go to ``tools_out/`` or another scratch directory; only reviewed,
compressed summary figures and small CSV/JSON metadata are tracked in
``docs/_static``.

The repository-size contract for this directory is deliberately strict:
``benchmarks/`` should stay at the scale of small scripts and manifests, not
simulation products. The tracked result manifest under
``benchmarks/results/manifest.toml`` is the docs-facing index for promoted
figures and tables, while NetCDF files, restart files, logs, profiler traces,
and exploratory plots remain outside git.

Quick driver examples:

.. code-block:: bash

   python benchmarks/cyclone_linear_benchmark.py --outdir tools_out/cyclone_benchmark
   python benchmarks/kbm_linear_comparison.py
   python -m gkx.cli run-runtime-linear --config benchmarks/runtime_secondary_slab.toml
   python benchmarks/secondary_slab_workflow.py

The Cyclone publication driver fits the terminal ``t=7--10`` interval. A
fresh trajectory audit showed that the previous automatic window could select
the short ``t=5.07--5.38`` startup transition and understate the ``ky=0.3``
growth rate by more than a factor of two, even though the late-time mode
converged to the tracked branch.

The KBM plotting driver reads the reviewed fixed-beta ``ky`` comparison table.
Use ``tools/comparison/compare_gx_kbm.py`` with a matched external output to
regenerate that table; branch selection remains a transitional time-history
policy until the generic runtime reproduces the full scan. The generic runtime
now provides the same CFL-controlled trajectory and can refit multiple branch
extractors from one stored field history; the remaining gate is the converged
full-grid replay, not missing runtime functionality.

Regenerate the atlas figures with:

.. code-block:: bash

   python tools/artifacts/make_benchmark_atlas.py

The atlas builder now reads its inputs from
``tools/benchmark_atlas_manifest.toml`` and writes a machine-readable summary to
``tools_out/benchmark_atlas_summary.json`` so the panel provenance stays
explicit.
Future velocity-space convergence panels should use the same JSON-ready
gate-report convention before they are promoted into the publication stack.

Capability and matched-comparison contract
------------------------------------------

``benchmarks/capability_matrix.toml`` is the machine-readable source of truth
for feature scope. It prevents two common errors: implying support because a
related equation exists, and diagnosing a solver mismatch before the two runs
actually use the same physical and numerical contract.

.. list-table:: Required-core and extension status
   :header-rows: 1
   :widths: 28 18 54

   * - Capability family
     - Status
     - Current claim
   * - Electrostatic flux-tube dynamics and nonlinear ExB bracket
     - validated, scoped by case
     - Cyclone, Miller, W7-X, and HSX state/observable gates
   * - Electromagnetic ``A_parallel``/``B_parallel`` dynamics
     - validated, scoped by case
     - KAW/KBM linear and KBM nonlinear gates
   * - Boltzmann and kinetic species in a Hermite--Laguerre basis
     - validated, scoped by branch
     - adiabatic ITG is core; kinetic-electron/TEM remains a stress lane
   * - Analytic, Miller, and imported VMEC geometry
     - validated, scoped by equilibrium
     - axisymmetric and selected W7-X/HSX comparisons
   * - Restart, spectra, heat flux, and field-energy diagnostics
     - validated
     - schema plus numerical-identity and windowed-statistics gates
   * - Independent ``k_y`` scans and UQ ensembles
     - production validated
     - CPU/GPU identity and strong-scaling evidence
   * - Nonlinear multi-device domain decomposition
     - blocked
     - current benchmark-grid whole-state route is slower and fails identity
   * - JAX autodiff, implicit gradients, and VMEC/Boozer optimization
     - validated, scoped
     - AD/FD, conditioning, covariance, geometry parity, and holdout gates
   * - Conserving Dougherty and linearized Sugama/Coulomb collisions
     - validated reduced research boundary
     - conserving Dougherty-like runtime model plus published reduced original/
       improved-Sugama and Coulomb matrix, invariant, dissipation, relaxation,
       and derivative gates. Full finite-:math:`b` multispecies promotion still
       requires generated hierarchy, conductivity, ITG, zonal, and resolution
       evidence.
   * - VMEC exact-periodic, continuous-drift, and fixed-aspect boundaries
     - validated, scoped
     - policy tests plus fixed-aspect W7-X/HSX comparison lanes
   * - Equilibrium :math:`E\times B` flow shear
     - numerical research API validated; physical model unshipped
     - zero-shear, shearing-wave, remap/dealias, cache, linear-suppression,
       AD/FD, linked-boundary, and fixed-step IMEX gates pass. The final fixed-
       step ``64x64x24`` response audit rejects promotion: the internal windows
       drift and show a 4.82% increase, while independently stationary comparison
       windows show a 24.82% increase. No input-file option is exposed.
   * - Specialized KREHM, Vlasov--Poisson, collisional-ETG, and Beer/Smith closures
     - not shipped
     - separate reduced models are outside the full-gyrokinetic release claim
   * - Species/Hermite multi-device operator execution
     - validated electrostatic operator route, scoped
     - periodic and linked ``2 species x 2 Hermite`` identity gates pass;
       mixed electromagnetic and four-physical-device evidence remain open

A matched comparison must record the equations and normalization, geometry
coefficients and parallel boundary, species conventions, perpendicular and
velocity grids, initial condition and seed, precision and de-aliasing,
integrator and timestep policy, collision/dissipation settings, diagnostic
normalization, and the fit or transport window. A visually similar input file
is insufficient. This rule is especially important for nonlinear saturation,
where short state-level agreement does not establish a converged heat-flux
comparison.

The promoted comparison contract was audited at GX revision ``bc2fe552``. A
fresh office clone confirms that revision and has aggregate source fingerprint
``sha256:bfaaadfa...20b``. The long-lived instrumented office source tree is not
a Git checkout and has fingerprint ``sha256:436e403e...a004``; these two
provenances must not be interchanged. The older binaries mixed system OpenMPI
and netCDF with local HDF5 and are excluded. An isolated clean-revision rebuild
now links one local OpenMPI 4.1.6, parallel netCDF 4.9.2, and HDF5 1.14.5 stack.
The canonical Cyclone s-alpha probe completed 2,145 steps to ``t=10`` in 23.1 s
and wrote valid netCDF/restart outputs; at ``ky=0.3`` it reports
``(gamma, omega)=(0.101814, 0.286777)``. That pair is a smoke-test reading taken
while the mode is still ringing, not a converged eigenvalue, and it must not be
used as a parity target. Running the same deck to its own ``t_max=150`` settles
at ``(gamma, omega)=(0.093049, 0.281991)``, which agrees with the
``*_correct.out.nc`` regression reference GX ships for that case to 0.03 per
cent. Between those horizons ``gamma`` swings 0.018, 0.101, 0.086, 0.115 before
settling at 0.093, so a comparison against the ``t=10`` value shows an
apparent 8.6 per cent gap that is entirely transient. GX remains the mature baseline for conventional GPU nonlinear
initial-value runs and species/Hermite multi-device execution. GKX's
distinct validated scope is its Python/JAX API, differentiable objectives,
implicit gradient paths, CPU execution, and in-memory ``vmex``/
``booz_xform_jax`` integration. GKX does not claim the complete finite-
wavelength multispecies linearized Landau hierarchy: its published reduced
Sugama/Coulomb slice remains a separately gated Python research boundary.

Feature parity is intentionally not blanket parity. GKX's required
comparison scope is the standard electrostatic/electromagnetic gyrokinetic
system, not GX's optional KREHM, Vlasov--Poisson, collisional-ETG, forcing,
transport-coupling, or Beer/Smith closure paths. Conversely, differentiable
eigen/objective solves and the in-memory JAX geometry chain are GKX
extensions rather than comparison requirements.

The audit also identifies equilibrium :math:`E\times B` flow shear as a
scientifically useful extension rather than a compatibility checkbox. Its
coordinate, operator, timestep, and derivative foundations are validated, but
the predeclared fixed-step transport-response gate failed. The implementation
therefore remains a Python research API rather than an input-file feature. The
compact negative-evidence record is
``docs/_static/flow_shear_fixed_step_response_gate.json``; raw states and large
comparison outputs remain off-repository.

Tracked results index
---------------------

The root-level result index is ``benchmarks/results/manifest.toml``. It points
to the promoted benchmark figures and machine-readable tables without moving or
duplicating large run products. The current tracked result set is:

.. list-table:: Promoted benchmark result artifacts
   :header-rows: 1
   :widths: 22 30 18 30

   * - Result
     - Tracked artifact
     - Claim scope
     - Regeneration path
   * - Core linear benchmark atlas
     - ``docs/_static/benchmark_core_linear_atlas.png``
     - headline linear validation atlas
     - ``python tools/artifacts/make_benchmark_atlas.py``
   * - Core nonlinear benchmark atlas
     - ``docs/_static/benchmark_core_nonlinear_atlas.png``
     - headline nonlinear validation atlas
     - ``python tools/artifacts/make_benchmark_atlas.py``
   * - README benchmark summary panel
     - ``docs/_static/benchmark_readme_panel.png``
     - compact publication-facing benchmark summary
     - ``python tools/artifacts/make_benchmark_atlas.py``
   * - Extended linear stress matrix
     - ``docs/_static/benchmark_extended_linear_panel.png``
     - stress and provisional lanes, not headline validation claims
     - ``python tools/artifacts/make_benchmark_atlas.py``
   * - Runtime and memory comparison
     - ``docs/_static/runtime_memory_benchmark.png``
     - tracked wall-time and memory comparison rows
     - ``python benchmarks/performance/benchmark_runtime_memory.py --summary-glob ...``
   * - Runtime and memory result rows
     - ``docs/_static/runtime_memory_results_ship_refresh.csv``
     - machine-readable runtime/memory rows used by the tracked panel
     - ``python benchmarks/performance/benchmark_runtime_memory.py``
   * - Runtime and memory summary
     - ``docs/_static/runtime_memory_summary_ship_refresh.json``
     - machine-readable summary for runtime/memory panel generation
     - ``python benchmarks/performance/benchmark_runtime_memory.py``
   * - Core linear atlas inputs
     - ``tools/benchmark_atlas_manifest.toml``
     - manifest of small tracked benchmark inputs
     - ``python tools/artifacts/make_benchmark_atlas.py``
   * - Reference-code linear parity matrix
     - ``docs/_static/gkx_gx_linear_parity_matrix.png``
     - cross-code linear growth-rate and frequency parity across tokamak and stellarator cases
     - ``python tools/comparison/build_gx_parity_matrix.py``
   * - Reference-code linear parity rows
     - ``docs/_static/gkx_gx_linear_parity_matrix.csv``
     - machine-readable per-wavenumber parity rows with convergence flags
     - ``python tools/comparison/build_gx_parity_matrix.py``
   * - Reference-code linear parity summary
     - ``docs/_static/gkx_gx_linear_parity_matrix.json``
     - per-case resolution, provenance and cost metadata behind the parity rows
     - ``python tools/comparison/build_gx_parity_matrix.py``

This keeps the repository light: ``benchmarks/`` stores only drivers and
pointers, ``docs/_static`` stores reviewed compact figures/tables, and raw
solver output directories remain untracked. The tracked ``benchmarks/`` payload
is intentionally on the order of tens of kilobytes.

The manifest above is the docs-facing source of truth for promoted benchmark
results. If a new result is added to ``benchmarks/results/manifest.toml``, it
must either appear in this table or remain unpromoted in scratch storage.

This produces the tracked atlas panels:

- ``docs/_static/benchmark_core_linear_atlas.png``
- ``docs/_static/benchmark_core_nonlinear_atlas.png``
- ``docs/_static/benchmark_readme_panel.png``
- ``docs/_static/benchmark_extended_linear_panel.png``

PDF copies are emitted alongside each PNG for manuscript workflows.

Tracked benchmark metrics
-------------------------

The benchmark atlas uses the same small set of physically interpretable metrics
throughout:

- growth rate ``gamma``
- real frequency ``omega``
- ion heat flux
- free energy
- electrostatic field energy (output variable ``Wphi``)
- magnetic field energy when ``A_parallel`` or ``B_parallel`` are active

At the README level, these metrics are intentionally packed into one compact
publication panel plus one separate runtime/memory panel. The atlas therefore
answers two questions:

- which branches and diagnostics are being tracked for validation,
- which shipped cases have measured CPU/GPU/runtime-memory coverage.

A fresh bounded Cyclone check on 2026-07-10 used RK4 to ``t=20`` at
``ky=0.3``, ``N_l=16``, and ``N_m=48`` on an RTX A4000. The reference
late-window mean was ``gamma=0.09582`` and ``omega=0.28106``; the unified
runtime with the benchmark-aligned midplane observable returned
``gamma=0.09076`` and ``omega=0.27828``. Relative errors are 5.3% and 1.0%,
respectively. The compact machine-readable record is
``docs/_static/cyclone_runtime_parity_refresh.json``. Its wall times must not be
read as a speedup: the reference invocation advanced 12 ky modes, while the
GKX invocation advanced one.

Primary publication set
-----------------------

The headline validation set is limited to the full-gyrokinetic lanes that are
already used in the current validation and regression workflow:

- Cyclone ITG linear and nonlinear
- KBM linear and nonlinear
- W7-X VMEC linear and nonlinear
- HSX VMEC linear and nonlinear
- Cyclone Miller geometry linear and nonlinear

The README atlas also includes one extended linear strip for exploratory or
stress lanes that are still useful to show publicly without folding them into
the primary validation claim:

- Cyclone kinetic electrons
- TEM

**Generated figure.** Linear benchmark master panel. This panel keeps the headline
linear
coverage on one page: Cyclone ITG, ETG, KBM, W7-X, HSX, Cyclone Miller,
KAW, and the KBM Miller late-growth replay.


**Generated figure.** Nonlinear benchmark master panel. This panel groups the tracked
nonlinear
overlays used in the public benchmark set: Cyclone, KBM, W7-X, HSX, and
Cyclone Miller.


For the current release pass, Cyclone, KBM, W7-X, HSX, and Cyclone Miller are
treated as the acceptable nonlinear validation set in the main atlas. The
short-window full-GK ETG pilot remains documented in the examples and testing
notes, but it is intentionally kept out of the primary publication panel
because it is a pilot rather than a headline transport benchmark. TEM and KAW
are intentionally kept out of the active parity claim until their separate
recovery work is finished.

The global release claim boundary is summarized in :doc:`release_scope`. Use
that page when deciding whether a benchmark panel supports a README, release
note, or manuscript claim.

Interpretation of validation
----------------------------

The atlas intentionally mixes two classes of evidence:

- broad benchmark overlays over scanned parameters such as ``k_y`` or
  ``beta``,
- exact-window or exact-diagnostic closures on selected lanes.

Only the exact-window closures should be read as strict small-tolerance validation
gates. In the current tracked set those are:

- KAW exact diagnostic window
- KBM Miller late-growth replay

The broader scanned benchmark panels are coverage figures, not universal
``rtol <= 3e-2`` claims for every tile. They remain valuable because they show
which branches and diagnostics are being tracked across the codebase.

Benchmark-specific replay knobs used to regenerate these figures stay confined
to the benchmark builders in ``tools/``. They are not promoted into generic
runtime defaults for the solver or the shipped example drivers.

Benchmark runner internals
--------------------------

Reusable reference loaders and comparison policies live in
``gkx.benchmarking.shared``; timestepping, scans, geometry, and physical
operators use the same runtime and solver APIs as ordinary simulations.
Case-level reproduction policy stays in root ``benchmarks/`` drivers rather
than creating a second installed solver stack.

Generic pointwise scans and representative-mode extraction are owned by
``gkx.workflows.linear``. They deliberately call one linear solve per
``k_y`` and accept optional resolution and Krylov policies. Benchmark drivers
provide case policy; the reusable workflow owns iteration, result assembly,
mode selection, and fit-window extraction. The former ``scan_fn`` argument was
removed because it was accepted but never used.

For the current stellarator nonlinear pair, the tracked public figures should
also be read asymmetrically:

- HSX nonlinear is currently acceptable on the best validated ``t <= 50`` trace
  and remains part of the public benchmark set.
- W7-X nonlinear is currently acceptable on the refreshed ``t <= 200`` trace
  and remains part of the public benchmark set. It should still be read as a
  long-window benchmark closure rather than a universal small-tolerance claim
  for every late-time sample.

README summary panel
--------------------

**Generated figure.** Publication-facing benchmark summary. The shipped
summary/publication stack
now includes the closed short-window full-GK ETG nonlinear pilot alongside
the tokamak and stellarator headline lanes, while the top-level README atlas
remains compact with one validation image and one separate runtime/memory
image.


Supplementary closure figures
-----------------------------

Some parity lanes are tracked as supplementary closure artifacts rather than as
headline atlas tiles. Current examples include:

- ``docs/_static/nonlinear_cyclone_short_resolved_audit_t5.png`` for the
  corrected short nonlinear Cyclone replay, which now uses the explicit
  short-reference dissipation contract and localizes the remaining mismatch in
  resolved ``k_y`` field-energy diagnostics.
- ``docs/_static/comparison/secondary_reference_out_compare.csv`` for the refreshed secondary
  stage-2 mode table built from the dense ``kh01a`` GX replay.
- ``docs/_static/nonlinear_w7x_diag_compare_t200.png``,
  ``docs/_static/hsx_nonlinear_compare_t50_true.png``, and
  ``docs/_static/nonlinear_kbm_diag_compare_t100_refresh.png`` for the
  refreshed long-window nonlinear publication figures.
- ``docs/_static/etg_fullgk_pilot_compare_dt1e4_gaussian_match.png`` for the
  closed short-window full-GK ETG nonlinear pilot that now appears in the
  shipped summary/publication panels.
- ``docs/_static/kbm_eigenfunction_overlap_summary.png`` for the current
  eigenfunction-overlap summary on the tracked KBM GX candidate table. This is
  the first compact overlap artifact in the manuscript-facing stack and should
  be read as a branch-identity diagnostic. The raw mode-shape overlays are now
  tracked separately as ``docs/_static/kbm_eigenfunction_reference_overlay_ky0p3000.png``
  and ``docs/_static/w7x_eigenfunction_reference_overlay_ky0p3000.png`` with
  JSON gate reports under ``docs/_static/reference_modes/``.

Extended stress matrix
----------------------

Not every benchmark lane belongs in the headline validation claim. The
extended linear stress matrix keeps exploratory or still-evolving lanes
visible without mixing them into the primary publication set.

The current extended panel covers:

- Cyclone kinetic electrons
- TEM (literature-backed stress lane)
- KBM Miller exact late growth window

The kinetic-electron scan is defined by
``examples/linear/axisymmetric/runtime_kinetic_electron.toml`` and runs through
the unified runtime API. Its effective reference seed, linked-boundary damping,
species, and electromagnetic toggles are explicit in that file rather than
being applied by a hidden benchmark wrapper.

**Generated figure.** Extended linear stress matrix. These lanes remain visible for
solver stress
testing, but they are intentionally separated from the main publication
panel.


The TEM row is provisional. The shipped ``tem_reference.csv`` is digitized from
the literature rather than sourced from a GX benchmark dump, and the exact case
definition behind that digitized curve is still being reassembled. It should be
read as a tracked stress lane, not as a closed parity result. The executable
audit ``docs/_static/tem_branch_parity_audit.{png,pdf,json,csv}`` records the
open branch mismatch explicitly: maximum absolute relative growth-rate
mismatch ``4.25``, maximum absolute relative frequency mismatch ``3.3`` after
excluding the near-zero reference denominator, one growth-rate sign mismatch,
three frequency sign mismatches, and a frequency-branch Spearman coefficient
near ``-0.986``.

The authoritative executable input is
``examples/linear/axisymmetric/runtime_tem.toml``. It uses the unified runtime
schema, including electron-only Gaussian moment initialization, and
``benchmarks/tem_linear_benchmark.py`` runs that file through the same runtime
scan path exposed to users. A state/parameter/RHS identity test protects the
migration from the former case-specific runner.

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

Notes on interpretation
-----------------------

The atlas is split deliberately. The README stays limited to a compact
publication panel, while this page keeps the larger linear, nonlinear, and
stress figures visible without claiming that every scanned point is an exact
closure.
