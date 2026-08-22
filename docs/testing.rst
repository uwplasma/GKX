Testing
=======

Testing philosophy
------------------

GKX enforces high coverage on critical solver modules and requires
physics-based checks for each numerical component. The test suite is designed
to be:

- **pedagogic**: each test explains the concept being validated
- **deterministic**: no stochastic outcomes or tolerance drift
- **future-proof**: targeted at invariants and well-posed regressions

Current testing target
----------------------

The package-wide target is 95% coverage, but the coverage number is a guardrail
rather than the scientific objective. New tests should be accepted because they
protect one of the following contracts:

- an implemented equation or reduced physical limit;
- a numerical method, convergence rate, or conservation/free-energy identity;
- a geometry, normalization, or diagnostic convention;
- a benchmark artifact and its documented fit/window policy;
- an autodiff contract checked against finite differences, tangent tests, or an
  adjoint consistency relation;
- a regression for a bug found in parity, restart, runtime, plotting, or
  geometry-adapter work.

Long reference-code runs and office/GPU comparisons should not be hidden inside
the default local suite. They should live behind explicit manifests or CI/manual
lanes so local tests remain fast enough for routine development.

The test-tree ownership map lives in ``tests/README.md``. New tests should go
into ``unit``, ``integration``, ``validation``, ``tools``, ``release``, or
``support`` according to the contract they protect. Do not add new flat
``tests/test_*.py`` files, and do not add one-file-per-script wrappers when a
parametrized family test can protect the same artifact or tool contract.
The test tree is organized by domain and currently satisfies its topology
guard. Raw file count is not a release objective: parametrization is preferred
when it makes one physical or numerical contract easier to understand, while a
separate test file is preferred when it gives a distinct research claim a clear
owner. Deleting a shallow compatibility test is acceptable when the underlying
legacy behavior has been removed; weakening physics, numerical, artifact, or
release gates is not.

The refactor branch also carries a machine-readable validation/coverage
manifest at ``tools/validation_coverage_manifest.toml``. It is checked by
``tools/release/check_validation_coverage_manifest.py`` and maps each critical module
to reference anchors, physics contracts, numerical contracts, fast tests,
tracked artifacts, and next tests. This is the working guardrail for reaching
95% package-wide coverage without adding shallow tests that do not validate the
implemented physics or numerics.

Source-layout hygiene is checked separately by
``tools/package_architecture_manifest.toml`` and
``tools/release/check_package_architecture_manifest.py``. That guard enforces
that manifest and prevents new root-level prefix modules
such as ``runtime_*``, ``nonlinear_*``, ``vmex_*``, ``quasilinear_*``, or
``benchmark_*`` from being added without an explicit migration entry. This keeps
the package moving toward domain packages while the validation manifest keeps
scientific ownership and coverage traceable.
The same architecture manifest enforces source complexity budgets. New
hand-written modules above the default line budget, and new oversized public
facades, fail unless they have a reviewed baseline, reduction target, reason,
and domain owner. This replaces the earlier raw source-file-count target, which
encouraged unrelated behavior to accumulate in giant compatibility modules.

The manifest now has two levels of coverage ownership:

- direct ``[[modules]]`` rows for public, high-risk, or actively refactored
  surfaces that need their own contracts and artifact traceability;
- ``owned_modules`` entries for smaller implementation modules whose fast-test
  responsibility is intentionally carried by a direct row.

The checker inventories ``src/gkx`` and fails if a package module is not
directly listed, owned by a listed row, or explicitly excluded as package
plumbing such as ``__init__.py`` or version metadata. This makes source
extractions fail fast until the coverage owner, fast tests, and next-test debt
are declared. New manifest tests for this policy should stay cheap and live in
``tests/release/test_release_gates.py`` or
``tests/test_refactor_coverage_*.py``.

Manifest paths are intentionally concrete. ``fast_tests`` and
``artifact_paths`` must name files, not directories or placeholder buckets, and
list fields must not repeat the same module, test, artifact, contract, or next
test. The optional Cobertura XML pass also rejects duplicate measured entries
for the same package module so coverage enforcement cannot depend on whichever
duplicate XML row happened to be parsed last.

The wide CI matrix also feeds the manifest checker with ``coverage-wide.xml``.
That pass enforces the declared package-wide coverage target and writes the
measured summary to ``docs/_static/validation_coverage_manifest_summary.json``.
Module-level rows in that summary are a debt map: they identify direct and
owned modules below their row target, but release blocking remains tied to the
package-wide gate unless the CI command is explicitly upgraded to
``--enforce-module-coverage``.

Optional external-backend artifact builders that require local ``vmex`` or
``booz_xform_jax`` checkouts are kept out of the default package-wide coverage
denominator when the public CI cannot install or execute those repositories.
Their fast contracts are still covered by mocked backend tests and low-level
geometry/numerics tests, while the real physics claims are validated by the
tracked JSON/PDF artifact gates documented below. This avoids treating
unavailable optional backends as missing unit coverage while preserving the
requirement that every differentiable-geometry claim has an explicit
finite-difference or parity artifact.

Nonlinear matrix release gates
------------------------------

Broad nonlinear turbulent-flux optimization claims use fail-closed matrix and
portfolio tools rather than manual figure selection.
``tools/release/check_nonlinear_transport_gates.py matrix-portfolio``
selects only a passing family before publication artifacts are indexed. The current
tracked max-mode-5 campaign is negative
evidence: accepted QA/ESS passed only ``9/18`` samples, projected weight
``1e-3`` failed early, and projected weight ``5e-4`` increased heat flux on its
first completed sample. The negative ledger is
``docs/_static/broad_nonlinear_transport_matrix_negative_evidence.json``; it
keeps scoped single-point examples from being misread as broad optimized
stellarator claims.

Test categories
---------------

- **Basis tests**: orthonormality and recurrence checks.
- **Operator tests**: Hermite ladder streaming and mode extraction.
- **Benchmark tests**: loading reference data and growth-rate fitting.
- **Physics sanity checks**: conservation properties under simplified limits.
- **Response-function tests**: zonal-flow residuals, GAM damping, and late-time
  envelopes.
- **Spectral tests**: fluctuation spectra and windowed nonlinear statistics.
- **Autodiff tests**: tangent, finite-difference, and inverse/UQ consistency.

Unit tests (numerical invariants)
---------------------------------

Representative unit checks include:

- **Hermite/Laguerre ladder identities**:
  :func:`gkx.operators.linear.moments.apply_hermite_v`,
  :func:`gkx.operators.linear.moments.apply_laguerre_x`.
- **Quasineutrality consistency**:
  :func:`gkx.operators.linear.moments.quasineutrality_phi`.
- **Streaming term validation**:
  :func:`gkx.operators.linear.moments.grad_z_periodic`,
  :func:`gkx.operators.linear.moments.streaming_term`.
- **Growth-rate fitting windows**:
  :func:`gkx.diagnostics.growth_rates.select_fit_window`,
  :func:`gkx.diagnostics.growth_rates.fit_growth_rate_auto`.
- **Grid construction and normalization**:
  :func:`gkx.core.grid.build_spectral_grid`.
- **Normalization contract consistency**:
  :func:`gkx.diagnostics.normalization.get_normalization_contract`,
  :func:`gkx.diagnostics.normalization.apply_diagnostic_normalization`.
- **Modular RHS equivalence**:
  :func:`gkx.operators.linear.params.linear_terms_to_term_config`,
  :func:`gkx.terms.assemble_rhs_cached`,
  :func:`gkx.operators.linear.rhs.linear_rhs_cached`.

These tests live in ``tests/unit/linear/test_linear.py`` and
``tests/unit/core/test_core_numerics.py`` and ``tests/unit/operators/test_terms_assembly.py`` and are
designed to fail deterministically if a discretization, assembly path, or
normalization changes.

Physics regression tests
------------------------

The physics-focused tests exercise reduced or symmetry limits that should
remain invariant across refactors:

- **Term toggles**: :class:`gkx.operators.linear.params.LinearTerms` switches individual
  operator components without changing the equation structure.
- **Mirror/curvature activation**: nonzero drift terms create nonzero response
  when streaming and drive are turned off.
- **Diamagnetic drive structure**: the energy-weighted drive produces a
  nonzero response when gradients are enabled and vanishes at :math:`k_y=0`.
- **Normalization scaling**: ``rho_star`` rescales the cached :math:`k_y`
  values exactly.
- **End-cap damping**: the linked-boundary taper only affects :math:`k_y>0`
  modes and vanishes when ``damp_ends_amp = 0``.

These checks are in ``tests/unit/linear/test_linear.py`` and are meant to be future-proof
physics invariants.

Benchmark regression tests
--------------------------

Benchmark regression tests validate the Cyclone base case reference dataset and
growth-rate extraction pipeline:

- Loading the reference CSV via :func:`gkx.benchmarking.shared.load_cyclone_reference`.
- Running short linear scans from the canonical
  ``examples/linear/axisymmetric/cyclone.toml`` input via
  :func:`gkx.runtime.run_runtime_scan`.
- Requiring independent-mode and combined-:math:`k_y` execution to agree at
  machine precision before either path is used for performance measurements.
- Reduced ky regression with tightened tolerances on the field-aligned grid.

These tests live in ``tests/validation/benchmarks/`` and
``tests/unit/operators/test_operator_kernels.py``.

Literature-anchored response and spectrum tests
-----------------------------------------------

The next research-facing additions should follow the published benchmark
observables rather than inventing repo-local metrics:

- **Rosenbluth-Hinton / GAM response in shaped tokamaks**: use the shaped
  benchmark conventions summarized by Merlo et al. to track residual levels and
  GAM damping alongside the linear shaping scan.
- **W7-X zonal-flow response**: use the stella/GENE W7-X benchmark conventions
  for residual level and damping envelope.
- **W7-X fluctuation spectra**: follow the W7-X Doppler-reflectometry
  comparison work for density and zonal-flow frequency spectra. The current
  closed artifact is a simulation-spectrum diagnostic; experimental transfer
  functions remain outside the release claim.
- **Electromagnetic stellarator verification**: adopt a heavy-electron
  electromagnetic lane before realistic-mass claims, following the GENE-3D
  verification pattern.

These should be implemented as reproducible, script-owned figure/artifact
lanes, not as ad hoc notebooks.

The first reusable tooling for this lane now exists:

- :func:`gkx.diagnostics.zonal_validation.zonal_flow_response_metrics`
- :func:`gkx.artifacts.io.load_diagnostic_time_series`
- :func:`gkx.diagnostics.validation_gates.evaluate_scalar_gate`
- :func:`gkx.diagnostics.validation_gates.observed_order_gate_report`
- :func:`gkx.diagnostics.validation_gates.branch_continuity_gate_report`
- :func:`gkx.diagnostics.validation_gates.eigenfunction_gate_report`
- :func:`gkx.diagnostics.validation_gates.linear_metrics_gate_report`
- :func:`gkx.diagnostics.validation_gates.nonlinear_window_gate_report`
- :func:`gkx.diagnostics.validation_gates.zonal_response_gate_report`
- :func:`gkx.diagnostics.zonal_validation.reference_residual_table`
- :func:`gkx.diagnostics.zonal_validation.tail_trace_metrics`
- :func:`gkx.artifacts.plotting.zonal_flow_response_figure`
- ``tools/artifacts/build_zonal_flow_artifacts.py`` with ``response-csv`` and ``response-output`` modes
- ``tools/artifacts/build_zonal_flow_artifacts.py miller-panel``
- ``tools/artifacts/build_w7x_zonal_validation_artifacts.py response-panel``
- ``tools/artifacts/build_w7x_zonal_validation_artifacts.py contract``
- ``tools/artifacts/build_w7x_zonal_validation_artifacts.py state-convention``
- ``tools/artifacts/build_zonal_flow_artifacts.py objective-gate``
- ``tools/artifacts/plot_w7x_fluctuation_spectrum_panel.py``

The gate-report helpers are intentionally small and JSON-ready. They should be
used by manuscript refresh scripts so every reported artifact has the same
observable, reference, absolute/relative tolerance, and pass/fail convention.
The companion coverage manifest should be updated when a new gate helper,
artifact script, or refactor extraction changes module ownership or test
responsibility.
``tools/artifacts/build_zonal_flow_artifacts.py miller-panel`` writes two such
reports into its JSON metadata. ``gate_report`` is the asserted one: the
residual, GAM frequency, and signed GAM damping against **GKX's own converged
Nm=144 values**, plus the recurrence, window, and trace-completeness conditions
that make those three numbers measurements at all. ``literature_comparison``
holds the same three observables against the Merlo Case-III paper-scale
read-off at the published tolerances, is reported rather than gated, and
currently does not pass.
``tools/artifacts/generate_linear_reference_overlays.py kbm`` writes the same gate structure for
the raw KBM eigenfunction overlay, using a strict overlap/relative-L2 policy.
The current refreshed KBM overlay passes that policy with overlap ``0.999985``
and relative ``L^2`` mismatch ``0.00721`` against the frozen GX raw mode.
``tools/artifacts/generate_linear_reference_overlays.py w7x`` applies the same raw-mode policy to
the imported W7-X linear benchmark at ``k_y rho_i = 0.3``. It refreshes the
frozen finite GX raw-mode bundle when a matching ``.big.nc`` file is supplied
and writes ``docs/_static/w7x_eigenfunction_reference_overlay_ky0p3000.png``
plus JSON/CSV companions. The current artifact passes with overlap
``0.9999999994`` and relative ``L^2`` mismatch ``3.33e-5``.
``tools/comparison/compare_gx_nonlinear.py diagnostics --summary-json`` now emits a
matching gate report for nonlinear diagnostic comparison figures, using the
window mean relative mismatch as the scalar acceptance metric. The summary
writer now accepts case/source labels, explicit ``tmin/tmax`` windows, and
writes strict JSON, replacing nonfinite absolute-gate relative errors with
``null``. The tracked release-window summaries cover Cyclone, Cyclone Miller,
KBM, HSX, and W7-X. The older short Cyclone diagnostic remains available as an
exploratory startup/resolved-spectrum audit, but it is not counted in the
release-gate index.
Observed-order and branch-continuity gate helpers are also available so
velocity-space convergence panels and branch-followed scan tables can use the
same JSON-ready acceptance convention.
``tools/artifacts/build_linear_validation_artifacts.py observed-order`` is the generic no-rerun path for
CSV-backed convergence studies: it reads either an explicit step column or a
resolution column, writes an observed-order JSON gate report, and can generate
a log-log convergence figure. The tracked Cyclone velocity-space convergence
artifact lives at ``docs/_static/cyclone_resolution_observed_order.json`` and
``docs/_static/cyclone_resolution_observed_order.png``. It uses an office/GPU
``ky=0.30`` time-path sweep through ``(Nl,Nm)=(4,8),(6,12),(12,24),(16,32)``
with ``tmax=150`` and passes the strict pairwise-order and final-error gates.
``tools/comparison/compare_gx_kbm.py --branch-summary-json`` wires that convention into
the KBM branch-following workflow by summarizing adjacent ``gamma``/``omega``
jumps and successive eigenfunction-overlap continuity for the selected branch.
``tools/artifacts/build_linear_validation_artifacts.py kbm-branch`` provides the corresponding
no-rerun artifact path: it reads the existing selected KBM candidate table and
writes ``docs/_static/kbm_branch_gate_summary.json`` with the same strict gate
schema. The current continuity-first selected branch passes the adjacent
growth/frequency jump and successive-overlap gates.
``tools/release/check_validation_coverage_manifest.py gate-index`` scans tracked JSON metadata and writes
``docs/_static/validation_gate_index.json``, ``.csv``, ``.png``, and ``.pdf`` so the docs
always have one compact pass/open view of the currently materialized release
validation gates. The current JSON index has ``17/18`` tracked reports passing,
with the quasilinear model-selection status intentionally open until a
candidate passes the strict uncertainty and transport-error gates.
Exploratory diagnostics can set ``gate_index_include=false``
to remain documented without being treated as release blockers.
``tools/artifacts/build_nonlinear_validation_panels.py window-statistics`` provides the companion
manuscript-facing statistics panel for the nonlinear GX comparison gates by
plotting the per-diagnostic ``mean_rel_abs`` and ``max_rel_abs`` values from
those same tracked JSON summaries.
The ``feasibility`` mode of the same command is the analogous tool for new
finite nonlinear pilots that do not yet have a reference comparison or
production-resolution convergence gate. It writes PNG/PDF/JSON/CSV artifacts
with explicit ``claim_level`` and ``promotion_gate.passed = false`` metadata,
so exploratory external-VMEC runs can be documented without being promoted to
transport validation claims.

``tools/artifacts/plot_external_vmec_nonlinear_convergence_gate.py`` is the promotion
gate for those pilots once at least two grid levels exist. It replays the
pilot JSON/CSV traces, compares common and least-trending late windows,
requires enough samples, bounds relative heat-flux trend and coefficient of
variation, and finally checks pairwise grid-refined heat-flux agreement. The
tracked CTH-like external-VMEC artifact intentionally fails this gate and sets
``gate_index_include=false`` because it is a research-planning negative result,
not a release-blocking validation gate.

``tools/artifacts/plot_external_vmec_nonlinear_convergence_gate.py time-horizon`` is the companion
time-horizon stability gate for modified-protocol holdout repairs. It consumes
the JSON outputs from the high-grid convergence gate at several final times,
requires every input grid gate to pass, and then checks that the high-grid
averaged common-window and least-trending heat-flux means are stable across
horizons. The gate is deliberately necessary-only: even a passing
time-horizon figure writes ``promotion_gate.passed = false`` until independent
replicate, seed, timestep, and admission-policy evidence exists.

``tools/release/check_vmec_boozer_gates.py high-grid-admission`` is the final scoped
exception gate for the rare case where the full grid ladder fails only because
the lowest grid is not converged. It requires the failed full-grid JSON sidecar
to contain only common/least grid-difference failures, requires the retained
high-grid labels to match the passing high-grid convergence gates, requires a
passed late time-horizon gate, and requires a passed seed/timestep replicated
nonlinear-window ensemble with finite nonzero transport. This policy follows
the literature practice of using saturated time traces, resolution ladders, and
uncertainty estimates for nonlinear turbulent fluxes rather than relying on a
single startup window or a single seed [Dimits00]_ [GX]_ [GonzalezJerez22]_
[Hoffmann23]_ [Oberparleiter16]_. A passing high-grid admission JSON makes a
case eligible for a scoped high-grid holdout role only; it explicitly does not
claim full ``n48/n64/n80`` convergence or promote an absolute quasilinear
transport model.

Every produced ``*.out.nc`` file is checked with
``tools/release/check_nonlinear_transport_gates.py runtime-outputs``.
That gate verifies the grouped NetCDF contains ``Grids/time`` and the requested
heat-flux diagnostic, checks finite monotone time samples, enforces optional
``tmin/tmax`` coverage, and fails closed for restart-only or metadata-only
artifacts. It is the first campaign-level smoke check after a long office GPU
batch exits with ``rc=0``.
``tools/release/check_nonlinear_optimization_gates.py production-guard`` then consumes those
replicated long-window ensembles together with the reduced optimization and
startup finite-difference artifacts. It is the fail-closed check that allows
release-safe scoped wording while blocking production nonlinear turbulent-flux
optimization promotion until optimized equilibria have replicated
post-transient transport-window audits.
The strict rerun-WOUT top-12 QA edge audit is the current reference negative
transfer example: ``docs/_static/strict_qa_top12_edge_matched_nonlinear_transport.json``
records passing baseline and candidate replicated-window ensembles but a
failed matched promotion gate, with ``0.58%`` relative reduction and
uncertainty z-score ``0.20``. The companion
``docs/_static/strict_qa_top12_edge_redesign_report.json`` confirms that the
18-point reduced objective passes surface/field-line/``k_y`` coverage, so the
next blocker is predictive transfer margin and uncertainty separation rather
than a missing sample dimension. These artifacts are intentionally tracked so
future transport-objective redesigns can be judged against a real long-window
nonlinear failure, not a startup proxy.
The tracked optimized-QA/ESS ``ZBS(1,0)`` example is deliberately kept as a
fail-closed regression: the real ``vmex`` re-equilibrated ``t=[450,900]``
baseline/plus/minus ensembles pass their replicated transport-window gates and
the initial three-replicate central finite difference is local, but
``gradient_uncertainty_rel = 0.655`` and therefore does not promote a
turbulence-gradient claim. A seed-5 follow-up for the same ``ZBS(1,0)``
bracket also remains blocked: the response fraction weakens to about ``0.037``,
``gradient_uncertainty_rel`` rises to about ``1.18``, and ``fd_asymmetry_rel``
is about ``0.520``. The companion ``RBC(1,1)`` and ``ZBS(1,1)`` controls fail
the locality/asymmetry gates. The central-FD artifact now includes
diagnostic-only paired-replicate rows when matching seed or timestep labels are
available; these rows are useful for identifying sign reversals or weak
responses, but they do not relax the production gates. A future passing
artifact must satisfy both uncertainty and locality thresholds without
weakening either threshold.
For future perturbation refreshes, keep each coefficient/amplitude in a
distinct artifact slug such as
``docs/_static/qa_ess_zbs10_rel5_nonlinear_gradient_zbs_1_0_central_fd_gradient_gate.*``.
Do not promote new prose until
``tools/release/check_nonlinear_optimization_gates.py gradient-evidence`` reports
``passed = true`` and the JSON sidecar sets
``nonlinear_turbulence_gradient_gate = true``. Until then, describe the result
as a bounded production-candidate finite-difference audit, not as a nonlinear
turbulence-gradient claim.
The current QA/ESS composite profile-direction follow-up demonstrates this
policy. The targeted ``plus_delta`` cross variants ``seed22_dt0p05``,
``seed32_dt0p04``, and ``seed33_dt0p05`` completed and all six plus-state
outputs passed the runtime-output gate. The extended plus ensemble still fails
the spread gate with ``mean_rel_spread = 0.166`` against the ``0.15`` limit,
and the central finite-difference artifact remains blocked by
``fd_asymmetry_rel = 2.84`` and ``gradient_uncertainty_rel = 1.22``. That
artifact is tracked as
``docs/_static/qa_ess_descent_profile_rel2_nonlinear_gradient_plus_delta_followup_central_fd_gradient_gate.json``.
It is a regression target for the fail-closed workflow and a design input for
the next campaign, not promotion evidence.

The tracked QA/ESS controls are deliberately negative regression cases.
``ZBS(1,1)`` is statistically cleaner but nonlocal, ``ZBS(1,0)`` can be
local but remains variance limited, and the larger ``RBC(1,1)`` bracket
worsens asymmetry. The completed overdetermined audit records full runtime
coverage and zero promoted controls; it is evidence against weakening the
locality or uncertainty criteria, not a missing workflow.

``tools/release/check_nonlinear_transport_gates.py matrix-portfolio`` is the final selector
when several candidate families have been audited. It consumes one or more
aggregate matrix reports, chooses only a passing broad matrix family, and
records strict ``t=1500`` growth/QL/nonlinear-window matched comparisons as
excluded negative-transfer evidence. This prevents the release process from
counting negative strict rows or single-point matched audits toward the broad
nonlinear turbulent-flux optimization claim.

External-VMEC holdouts are retained only after the high-grid admission,
time-window, and replicate gates pass. Candidate launch planning is not release
evidence and is no longer a tracked artifact. Stable and near-marginal branches
remain useful linear evidence, but cannot enter nonlinear calibration without a
separate converged transport audit.

``tools/artifacts/build_qi_branch_refinement_gate.py`` is the focused companion for that
near-marginal QI evidence. It checks finite low-``k_y`` branch rows, contiguous
positive support, optional Krylov consistency, and the same nonlinear-launch
growth threshold. A failed launch-growth subgate is a useful documented result,
not a release failure, because it prevents QI feasibility scans from being
misread as transport validation.

``tools/release/check_quasilinear_promotion_guardrails.py calibration-inputs`` is the corresponding
calibration-admission guard. It scans quasilinear train/holdout reports and
requires every non-audit nonlinear artifact to match a passed nonlinear gate.
This makes validation provenance executable: finite-but-unconverged pilots can
be documented in the docs, but they cannot silently become calibration or
optimization data. The public CI runs this audit during the docs/packaging
job, and the fast test suite checks the current tracked train/holdout reports
against the same gate index.

``tools/release/check_quasilinear_promotion_guardrails.py`` is the higher-level
absolute-flux promotion guard. It scans the tracked quasilinear reports plus
the claim-scope docs, fails if a promoted report lacks train/holdout points,
finite nonlinear window statistics, a passed holdout gate, or calibration
policy metadata, and writes
``docs/_static/quasilinear_promotion_guardrails.json`` with a normal
``gate_report`` for the validation index. This is not a runtime/TOML
absolute-flux predictor; it is a fast metadata and wording guard that prevents
overclaiming current diagnostics.
The model-development figure scripts for saturation-rule sweeps,
shape-aware saturation, and uncertainty-aware candidate scoring also validate
their nonlinear summary inputs by default and serialize an ``input_validation``
block into the tracked JSON artifacts.

The diagnostics stream now also carries ``Diagnostics/Phi_zonal_mode_kxt``, a
signed complex zonal-potential history reduced over ``z`` with the same volume
weights used elsewhere. That is the primitive to use for manuscript-grade
Rosenbluth-Hinton / GAM work. ``Diagnostics/Phi2_zonal_t`` remains useful as a
zonal-energy proxy for intermediate checks, but it is no longer the target
observable for the final paper lane.

The case-specific shaped-Miller lane for this benchmark is reproducible
through ``benchmarks/runtime_miller_zonal_response.toml`` and
``tools/artifacts/build_zonal_flow_artifacts.py miller-panel``, and its frozen
artifact lives in ``docs/_static/miller_zonal_response_pilot.png`` with the
gate in the companion ``.json``. The physics contract is Merlo et al. Case III:
adiabatic electrons, zero gradients, ``k_xρ_i≈0.05``, ``k_y=0``, and an initial
ion-density perturbation.

**The baseline is the converged run, not the agreeing one.** The artifact now
uses ``Nz=32``, ``Nl=4``, ``Nm=144``, ``dt=0.0025``, and runs to ``t=60 a/v_i``.
It reports ``residual = 0.2059``, ``ω_GAM R0 / v_i = 2.345``, and
``γ_GAM R0 / v_i = -0.184``. Two conditions set that protocol and both are
gated rather than described:

* **Recurrence.** The quiet point of the trace -- the gap between the damped
  primary GAM and the regrowing velocity-space recurrence -- moves as
  ``t_quiet ≈ 5.5 sqrt(Nm)``, measured at ``26.0``, ``38.8``, and ``55.5`` for
  ``Nm = 24``, ``48``, and ``96``. **Every analysis window must close before it.**
  The residual window is the last 30% of the trace, ``t ∈ [42, 60]``, so the run
  needs ``Nm ≳ 120``. At the retired ``Nm=24`` baseline that whole window sat
  1.6--2.3x *past* recurrence onset, and the scatter inside it
  (``residual_std/residual = 1.20``) exceeded the number being gated. The
  artifact now gates ``residual_scatter_ratio ≤ 0.25`` (it measures ``0.143``)
  and ``analysis_window_past_recurrence = 0``.
* **Timestep.** The Hermite streaming CFL scales as ``sqrt(Nm)``, and ``Nm=144``
  at the old ``dt=0.005`` goes non-finite at ``t=46.5``. Refining the Hermite
  resolution therefore requires ``dt ≤ 0.0025``. Residual and frequency are
  timestep-converged to better than ``0.3%`` on both ladders.

Also gated: ``run_to`` must stay ``"t_max"``. The default is ``"saturation"``,
which for a zero-gradient relaxation run declares the identically-zero heat
flux converged inside the first chunk and truncates the trace at ``t ≈ 6``
without raising -- one sixth of a GAM period, with the residual then read off
whatever the last 30% of *that* happens to be. The artifact gates
``trace_completeness = 1`` so a silently short run cannot be published.

That was one deck. ``tests/release/test_release_gates.py`` carries the audit of
all of them: every shipped TOML reaching the nonlinear runtime either pins
``run_to`` in its own ``[time]`` block or is listed with the measured
first-chunk stop decision that cleared it to run under the default, and a new
deck fails the gate until somebody measures it.

**Agreement with the published read-off is reported, not asserted.** At
converged resolution GKX disagrees with Merlo Case III by more than the
paper-scale tolerances on two of the three quantities:

.. list-table::
   :header-rows: 1

   * - quantity
     - GKX converged
     - Merlo read-off
     - gap
     - paper-scale tolerance
     - literature comparison
   * - ``residual``
     - ``0.206 ± 0.006``
     - ``0.19``
     - ``0.016``
     - ``0.015``
     - **fails**
   * - ``ω_GAM R0/v_i``
     - ``2.345 ± 0.05``
     - ``2.24``
     - ``0.105``
     - ``0.10``
     - **fails**
   * - ``γ_GAM R0/v_i``
     - ``-0.184 ± 0.010``
     - ``-0.17``
     - ``0.014``
     - ``0.03``
     - passes

Both failures are marginal (``1.06x`` and ``1.05x`` tolerance) and both are
monotone in ``Nm``: the residual rises ``0.2045 → 0.2059 → 0.2082`` and the
frequency rises ``2.340 → 2.345 → 2.353`` across ``Nm = 96/144/192``, so
further refinement widens the gap rather than closing it. The tolerances have
**not** been widened to absorb this. The tracked ``gate_report`` asserts GKX's
own converged numbers and the stability conditions above; the paper comparison
is recorded separately under ``literature_comparison`` with
``paper_scale_gate_passed: false``, so the claim boundary is visible in the
artifact itself.

The most likely physical explanation was tested and **falsified**: Merlo
Table III lists ``α_MHD = 0.5425`` while the runtime TOML sets
``betaprim = 0.0``. Carrying it across gives
``betaprim = -α_MHD / (q² R0) = -0.1012``, and rerunning at converged
resolution reproduces the baseline trace **bit for bit** (max ``|Δφ| = 0``
over 2401 samples). The reason is structural: ``betaprim`` changes ``gds2``,
``gds21``, ``gbdrift``, ``cvdrift``, and ``aprime``, and leaves ``gds22``,
``gbdrift0``, ``cvdrift0``, ``bmag``, ``gradpar``, and ``jacob`` exactly
unchanged -- every coefficient it touches is multiplied by ``k_y``, and this
benchmark runs at ``k_y = 0``. The dropped ``α_MHD`` cannot be the source of
the offset for a purely radial zonal mode.

Finally, ``γ_GAM`` is measured with an estimator that cannot be swung by one
sample. The retired fit took up to four extrema per branch and fitted
``log|amplitude|`` through them, so a shallow extremum that one diagnostic
output cadence resolved and another did not entered the fit: halving the output
cadence at *identical physics* moved the shipped value from ``-0.1745`` to
``-0.2645``, a 52% swing and three times its own gate tolerance. The current
``period_rms_envelope`` mode takes the sliding RMS of the trace about its own
running one-period mean -- which for ``C(t) + A e^{-γt} cos(ωt+φ)`` is
``A e^{-γt}`` times a constant, whatever the slowly-varying offset ``C`` does --
and fits its log with weights ``A²`` over a window stated in GAM periods rather
than in samples. Measured spreads on the ``Nm=144`` trace: over a factor-8
cadence span with sampling-phase offsets, ``0.0003`` for the envelope estimator
against ``0.0379`` for the extrema fit; over ``fit_window_tmax ∈ [22, 35]``,
which moves ``ω_GAM`` by ``0.24``, ``0.0015`` against ``0.018``. On the
``Nm=24`` trace that produced the original 52% swing, the envelope estimator
moves by ``0.00013``.

The next literature lane now has a dedicated runtime contract as well:
``benchmarks/runtime_w7x_zonal_response_vmec.toml`` and
``tools/artifacts/build_w7x_zonal_validation_artifacts.py response-panel`` define the W7-X high-mirror
bean-tube zonal-flow relaxation benchmark from the stella/GENE paper. The
tool sweeps ``k_x rho_i`` over ``[0.05, 0.07, 0.10, 0.30]``. The runtime
contract seeds the published electrostatic-potential perturbation with
``init_field = "phi"`` and a Gaussian profile, while the panel extracts the
unweighted signed line-average diagnostic ``Phi_zonal_line_kxt``. The paper
text states that the line-average trace is normalized to its value at ``t=0``;
the caption also mentions the maximum value, but the source figure is clipped
at the initial point. The paper-facing default is therefore
``--initial-normalization=line_first`` and ``--time-scale=1``. The ``init_amp``
normalization and non-unit time-scale options are retained as explicit audits,
not as the validation contract. The default early-time fit-window cap is an
explicit analysis policy chosen to isolate the initial GAM before the slower
stellarator-specific oscillation. The generator forces a periodic radial box
for this ``k_y=0`` zonal response so the selected ``k_x rho_i`` values match
the published test-4 targets exactly; this avoids the linked-boundary
aspect-ratio override that is appropriate for drift-wave flux-tube runs but
wrong for this radial zonal scan.

The current frozen VMEC-backed artifact lives at
``docs/_static/w7x_zonal_response_panel.png`` with strict JSON metadata at
``docs/_static/w7x_zonal_response_panel.json``. The tracked combined trace CSV
``docs/_static/w7x_zonal_response_panel.traces.csv`` is written next to the
figure so comparison and audit scripts can be rerun without office-only
per-``k_x`` directories. It is a long-window run: ``k_x rho_i=0.05`` reaches
``t≈3460`` and the other three wavelengths reach ``t≈1980``. After the
paper-faithful line-first normalization, the late residuals are about
``0.0189``, ``0.137``, ``0.0938``, and ``0.526`` for ``k_x rho_i = 0.05``,
``0.07``, ``0.10``, and ``0.30``.
``tools/artifacts/build_w7x_zonal_reference_artifacts.py digitize`` now extracts the stella/GENE Fig. 11
main traces and inset residual levels from the arXiv source ``figs/ZF.pdf``.
The resulting reference artifacts are
``docs/_static/w7x_zonal_reference_digitized.csv``,
``docs/_static/w7x_zonal_reference_digitized_residuals.csv``,
``docs/_static/w7x_zonal_reference_digitized.json``, and
``docs/_static/w7x_zonal_reference_digitized.png``. The comparison contract is
implemented in ``tools/artifacts/build_w7x_zonal_reference_artifacts.py compare`` and materialized at
``docs/_static/w7x_zonal_reference_compare.png`` with JSON metadata in
``docs/_static/w7x_zonal_reference_compare.json``. The current long-window
artifact passes the time-coverage gate for all four wavelengths, but the
residual gate only passes at ``k_x rho_i=0.05`` and the late-envelope gate
fails by orders of magnitude. A previous ``init_amp``-normalized audit happened
to pass residual values for all four wavelengths, but that comparison is no
longer treated as a validation result because it does not follow the paper text
normalization. A later ``gaussian_width=4`` probe matched the clipped apparent
initial level of Fig. 11 better than the tracked width-1 profile, but the
source figure shows that the apparent ``0.8`` start is a plot-limit artifact,
not a reliable normalization target. The tracked TOML therefore keeps
``gaussian_width=1``, matching the source expression ``exp[-(z-z0)^2]``.

The runtime path now has three safeguards for this lane. First, strided nonlinear
diagnostics always retain the final step, so long traces do not silently stop
one stride before the intended horizon. Second, checkpointed artifact
generation validates each chunk for non-finite diagnostics, state, and fields
before writing or continuing. This makes high-moment W7-X recurrence sweeps
fail fast instead of running thousands of extra steps after a NaN. Third,
default VMEC/eik cache outputs are reused when valid and generated through a
unique temporary netCDF followed by atomic replacement, so parallel W7-X
validation sweeps cannot observe or corrupt a partially written geometry file.
A bounded
``k_x rho_i=0.07``, ``Nl=16``, ``Nm=64``, ``dt=0.05`` probe remained finite to
``t≈200`` and a post-fix ``t≈50`` rerun verified nonzero signed line-average
diagnostics through the retained final sample. A separate external-restart
artifact bug was then isolated to double-condensing already-active ``kx``/``ky``
diagnostic axes when appending loaded history. The writer now accepts either
full spectral axes or already-active GX output axes, and a W7-X VMEC external
resume smoke verified nonzero ``Phi_zonal_line_kxt`` and
``Phi_zonal_mode_kxt`` throughout the appended tail. A higher-moment follow-up
with ``Nl=16``, ``Nm=64``, ``dt=0.05`` then restart-continued the
``k_x rho_i=0.07`` trace to ``t≈100`` with finite diagnostics and nonzero
signed line/mode samples across the post-restart tail. A full four-wavelength
refresh at the same moment resolution also reached ``t≈100`` with finite,
nonzero signed traces for every target ``k_x rho_i``. A width-4 full-window
low-moment audit reached the digitized windows but flipped the residual sign at
``k_x rho_i=0.07``, ``0.10``, and ``0.30``. The remaining open item is
therefore not restart diagnostic continuity; it is the W7-X zonal damping,
closure, and velocity-space recurrence behavior under the paper-facing
line-first normalization.
``tools/artifacts/build_w7x_zonal_validation_artifacts.py contract`` turns the same tracked CSV/JSON
artifacts into ``docs/_static/w7x_zonal_contract_audit.png``. That panel is a
publication-facing diagnostic of the open mismatch rather than a release gate;
its JSON metadata has ``gate_index_include=false`` so the validation index does
not count it as closed.
The ``state-convention`` mode of the same command closes the state-level
initializer and observable convention layer for the same paper-facing setup.
At ``k_x rho_i=0.07``, ``Nl=16``, and ``Nm=64``, the recovered Gaussian
potential has relative ``L2`` error ``1.85e-6``, off-target spectral potential
content is zero to the reported precision, and the signed line-average and
volume-average helper diagnostics agree with manual reductions to about
``2e-16``. The line-first initial level is ``0.28209 init_amp`` while the
volume-weighted level is ``0.28450 init_amp``; that explicit difference is why
the paper-facing observable must remain ``Phi_zonal_line_kxt`` normalized by
its first nonzero sample.
The ``sweep`` mode of the same command then performs the bounded
recurrence sweep requested for the paper lane without changing initializer or
normalization conventions. Moment resolution and closure source are varied
separately at ``k_x rho_i=0.07`` over the common ``t v_t/a <= 100`` window.
The no-closure rows give mean absolute reference errors ``0.295`` for
``Nl=8,Nm=32``, ``0.276`` for ``Nl=12,Nm=48``, and ``0.283`` for
``Nl=16,Nm=64``. At fixed ``Nl=16,Nm=64``, constant-source closure suppresses
the final Hermite-tail fraction from ``0.388`` to ``0.062`` but worsens the
trace mean absolute error to ``0.291``; the ``k_z``-weighted closure remains
close to no closure. This separates the remaining recurrence/closure problem
from a state-convention error.
The newest constant-hypercollision follow-up keeps the paper-facing
normalization and compares ``nu_hyper_m=0.01`` and ``0.03`` at
``Nl=16,Nm=64`` to ``t v_t/a=100``. Increasing ``nu_hyper_m`` lowers the final
Hermite-tail fraction from ``0.220`` to ``0.099`` and lowers the free-energy
ratio from ``0.759`` to ``0.600``, but the mean trace error remains
``0.289`` and the late-window standard deviation remains more than four times
the digitized reference. The W7-X zonal lane therefore remains a physical
closure/recurrence problem, not a normalization problem and not a simple
constant-damping fix.
The mixed Laguerre-Hermite closure audit then tests the best bounded closure
candidate under a moment-resolution increase. At ``Nl=16,Nm=64`` and
``dt=0.05``, the mixed closure gives mean absolute trace error ``0.2753`` and
late-window standard-deviation ratio ``4.24``. Raising the resolution to
``Nl=24,Nm=96`` requires ``dt=0.025`` for a finite run; it lowers the
late-window standard-deviation ratio slightly to ``4.11`` and further reduces
the Hermite/Laguerre tail fractions, but the trace error remains ``0.2768``.
The more aggressive ``Nl=32,Nm=128`` run still becomes non-finite by
``t v_t/a≈10`` even at ``dt=0.025``. This separates a real high-moment
time-step limitation from the larger physical result: the current mixed
closure does not converge toward the digitized W7-X trace in a way that can be
promoted as validation.
``tools/artifacts/build_w7x_zonal_validation_artifacts.py response-panel`` now exposes explicit
``--nu-hyper``, ``--nu-hyper-l``, ``--nu-hyper-m``, ``--nu-hyper-lm``,
``--p-hyper-*``, ``--hypercollisions-const``, ``--hypercollisions-kz``,
``--enable-hypercollisions``, and ``--gaussian-width`` overrides so future
closure probes can be launched from the tracked benchmark tool rather than from
unrecorded local TOML edits. Non-unit Gaussian widths remain initializer
audits, not validation defaults.

**Generated figure.** W7-X high-mirror bean-tube zonal-flow response for the stella/GENE
test-4
target ``k_x rho_i`` values. The response is normalized to the first
nonzero line-average sample, following the paper text. The red dashed line
is the late-window residual estimate and the shaded band is the common
initial-GAM extraction window.


**Generated figure.** Digitized stella/GENE reference traces from the W7-X benchmark
paper's
Fig. 11. The horizontal lines are residual levels read from the figure
insets and are the reference targets for the next long-window GKX
zonal-response gate.


**Generated figure.** Current W7-X zonal comparison gate. Time coverage passes for all
four
wavelengths, but the paper-normalized residuals and late-window envelopes
remain open validation issues.


**Generated figure.** Publication-facing audit of the open W7-X test-4 zonal-response
lane. The
top row separates residual and late-envelope discrepancies; the bottom row
overlays representative paper-normalized traces against the digitized
stella/GENE mean. This figure is intended to localize the remaining
velocity-space recurrence / closure problem, not to claim validation closure.


**Generated figure.** Velocity-space tail audit for existing W7-X test-4 outputs. The
long
``Nl=8``, ``Nm=32`` traces have large late normalized-trace variance and
visible Hermite/Laguerre tail content. The short ``Nl=16``, ``Nm=64`` run
reduces the early trace envelope but does not by itself close the
long-window recurrence question.


**Generated figure.** Bounded closure ladder for ``k_x rho_i=0.07``. Constant Hermite,
``k_z``-weighted Hermite, mixed Laguerre-Hermite, Laguerre-only, and
isotropic hypercollision families are compared with the no-closure baseline.
Some variants reduce mean trace error or velocity-space tails, but none
improves the trace and late-envelope recurrence metrics together.


**Generated figure.** State-level W7-X test-4 convention audit. The runtime path
recovers the
paper Gaussian potential initializer, selects only the requested zonal
spectral mode, and verifies that the signed line-average and
volume-weighted zonal observables are intentionally distinct but internally
consistent.


**Generated figure.** Bounded W7-X test-4 recurrence sweep at ``k_x rho_i=0.07``. The
left trace
panel varies moment resolution with no closure; the right trace panel varies
closure source at fixed high resolution. The bottom panels show that tail
suppression alone does not yet close the literature-trace mismatch.


**Generated figure.** Constant-Hermite-hypercollision follow-up for ``k_x rho_i=0.07``.
Stronger
constant damping reduces Hermite-tail and free-energy metrics but does not
reduce the long-window trace error or recurrence envelope enough to match
the digitized stella/GENE reference. This is a documented negative result
that motivates a more physical closure/operator study.


**Generated figure.** Mixed Laguerre-Hermite closure resolution audit for ``k_x
rho_i=0.07``. The
``Nl=24,Nm=96`` run is finite only with the smaller ``dt=0.025`` and lowers
the late-window variability modestly, but it does not improve the trace
error relative to ``Nl=16,Nm=64``. The omitted ``Nl=32,Nm=128`` point is a
tracked non-finite result under the same closure family, so this remains an
open physics/numerics lane rather than a closed W7-X zonal validation.


Diffrax and nonlinear smoke tests
---------------------------------

Diffrax integration and the nonlinear driver are exercised with fast smoke
tests:

- ``tests/unit/solvers/test_diffrax_integrators_core.py`` runs explicit, IMEX, streaming, and branch-coverage diffrax solver contracts
  on tiny grids.
- ``tests/unit/solvers/test_diffrax_integrators_core.py`` hardens branch coverage for
  diffrax helper paths (solver selection, save modes, streaming fits, IMEX
  branches, parallelization, and validation errors).
- ``tests/unit/solvers/test_linear_krylov_core.py`` hardens matrix-free Krylov internals
  (mode-family targeting, shift-invert preconditioner selection, fallback
  policy, and dominant eigenpair wrappers).
- ``tests/integration/examples/test_examples.py`` verifies shipped example workflows:
  autodiff inverse/UQ demos, implicit quasilinear sensitivity checks,
  independent-ky parallelization examples, the config-driven diffrax runner, and
  short nonlinear scans through the assembled E×B nonlinear bracket.
- ``tests/unit/nonlinear/test_nonlinear_exb.py`` exercises the nonlinear bracket sign,
  real-FFT path, flutter coupling, scalar/precomputed gyroaverage paths, and
  EM component accounting. The targeted nonlinear-term tranche covers the
  pseudo-spectral bracket and electromagnetic decomposition branches without
  launching benchmark-size turbulence runs.
- ``tests/unit/nonlinear/test_nonlinear_helpers_extra.py`` locks the higher-level nonlinear
  diagnostic contracts: Hermitian real-FFT projection, signed-mode masks,
  explicit Runge-Kutta variants, fixed-mode frequency extraction, collision
  splitting, and IMEX nonlinear terms.
- The same nonlinear helper suite owns the equilibrium-flow-shear research
  gates. It checks analytic shearing-wave motion, integer and fractional
  corrected remaps, de-aliasing, Hermitian projection, periodic and linked
  zero-shear trajectory identity, linked-chain invariance, physical RK2/RK3
  observed order, and canonical heat-flux reconstruction. The fixed-step IMEX
  route additionally has endpoint-field, first-order convergence, x64 dtype,
  and forward/reverse derivative checks against centered finite differences.
  Unsupported adaptive IMEX, custom-collision, and non-twist combinations fail
  closed. These bounded tests verify equations and algorithms; they do not
  replace a post-transient transport-window gate. The completed fixed-step gate
  failed physical-model promotion, so no input-file option is exposed.
- ``tests/validation/nonlinear/test_nonlinear_window_artifact_contracts.py``
  locks the compact fixed-step flow-shear evidence as negative: the internal
  windows are nonstationary, the independent comparison windows pass, both
  matched effects have the wrong sign for the predeclared suppression gate, and
  the artifact forbids input-file exposure.
- ``tests/unit/linear/test_linear_helpers_extra.py`` verifies that the
  time-dependent linear cache rebuilds every sheared perpendicular operator,
  reproduces the static cache at zero shear, preserves linked radial spacing,
  and has a shear tangent consistent with centered finite differences.
- ``tests/integration/runtime/test_runtime_config.py`` and ``tests/integration/runtime/test_runtime_runner.py`` verify
  unified runtime TOML loading and case-agnostic linear runs (Cyclone/ETG/KBM)
  through the same solver path.
- ``tests/integration/runtime/test_runtime_config.py`` also locks the public nonlinear stellarator
  runtime contract, including the absence of adaptive-step truncation caps and
  the presence of default ``tools_out/...`` artifact paths for W7-X and HSX.

Parallelization identity gates
------------------------------

Independent scan and ensemble parallelization is tested before it is used for
performance claims:

- ``tests/unit/parallel/test_parallel_core.py`` locks the ``batch_map`` / ``ky_scan_batches``
  helper semantics, including deterministic padding, one-device fallback, and
  pytree outputs used by UQ and sensitivity workflows.
- ``tests/unit/parallel/test_parallel_linear_velocity.py`` locks the species/Hermite
  velocity-decomposition planner. These tests verify load balance metadata,
  Hermite ghost-exchange flags, and field-reduction axes before any production
  ``shard_map`` implementation can use that layout. The same test file also
  covers the full-array Hermite-neighbor reference and one-device fallback for
  the communication kernel.
- ``tests/unit/solvers/test_sharded_integrators.py`` locks the sharded linear RK2 wrapper in
  both no-sharding and explicit-sharding modes using a mocked RHS and mocked
  ``pjit``. It also locks the fixed-step nonlinear state-sharded wrapper,
  including final-state-only profiling mode and the config-runner route through
  ``TimeConfig.state_sharding``. These are numerical-identity and control-flow
  gates, not speedup claims.
- ``tests/unit/parallel/test_parallel_nonlinear.py`` locks the diagnostic
  nonlinear decomposition gates. It covers one-cell halo chunks for a bounded
  local stencil plus split/reassemble spectral layout identity for FFT round
  trip, pseudo-spectral bracket, and field-solve layout.
  Both fail closed and carry no production routing or speedup claim.
- ``tests/tools/artifacts/test_transport_artifact_tools.py`` tests the
  parallel identity artifact family: velocity reduction, Hermite exchange and
  streaming, electrostatic field/drive/drift routes, linear-RHS parallel
  routes, independent ``k_y`` batching, logical CPU batching, and quasilinear
  runtime batching. It replaces the previous one-file-per-gate tests with one
  parametrized artifact-family suite.
- ``tests/unit/parallel/test_parallel_artifacts.py`` locks the tracked large-run
  scaling artifacts themselves. It requires the performance and validation
  manifests to list the CPU/GPU split artifacts, verifies serial numerical
  identity for independent ``k_y`` and quasilinear/UQ rows, checks that
  nonlinear whole-state sharding embeds per-device profiler/profile payloads,
  and fails if docs detach speedup wording from the current artifact set.
- ``tools/artifacts/generate_parallel_identity_gate.py ky-scan`` runs the actual linear solver
  serially and with fixed-shape ``k_y`` batching, then writes
  ``docs/_static/parallel_ky_scan_gate.{png,pdf,csv,json}``. The JSON gate
  requires numerical identity for growth rate and frequency; the speedup value
  is reported separately for engineering tracking.
- ``tools/artifacts/generate_parallel_identity_gate.py logical-cpu`` exercises
  ``RuntimeParallelConfig`` and ``batch_map`` over logical CPU devices with a
  structured JAX-native scan output. Its artifact
  ``docs/_static/logical_cpu_parallel_scan_gate.{png,pdf,csv,json}`` is an API
  identity gate, not a gyrokinetic physics benchmark.
- ``tools/artifacts/generate_velocity_parallel_gates.py hermite-exchange`` runs the first actual
  ``jax.shard_map`` communication-kernel gate for nearest-neighbor Hermite
  ghost exchange and writes
  ``docs/_static/hermite_exchange_gate.{png,pdf,csv,json}``. This is a
  prerequisite for production velocity-space decomposition, but it is not a
  nonlinear runtime speedup claim.
- ``tools/artifacts/generate_velocity_parallel_gates.py field-reduce`` runs the matching
  ``jax.shard_map`` field-reduction gate with ``lax.psum`` over the Hermite
  mesh and writes
  ``docs/_static/velocity_field_reduce_gate.{png,pdf,csv,json}``. Its
  tolerance is a float32 communication/reduction-tree tolerance, not a physics
  acceptance tolerance.
- ``tools/artifacts/generate_electrostatic_parallel_gates.py field-reduce`` applies that reduction
  pattern to the production electrostatic quasineutrality density moment and
  writes ``docs/_static/electrostatic_field_reduce_gate.{png,pdf,csv,json}``.
  It is currently scoped to single-species periodic electrostatic cases.
- ``tools/artifacts/generate_velocity_parallel_gates.py hermite-ladder`` combines the Hermite
  exchange with the actual ``sqrt(m+1)`` / ``sqrt(m)`` streaming-ladder
  coefficients and writes
  ``docs/_static/hermite_streaming_ladder_gate.{png,pdf,csv,json}``. This is
  the last isolated communication/coefficient gate before a linear streaming
  microkernel can be wired.
- ``tools/artifacts/generate_electrostatic_parallel_gates.py drift`` gates the single-species
  periodic electrostatic mirror and curvature/grad-B drift slices against the
  production linear RHS. It uses offset-1 and offset-2 Hermite exchanges and
  writes ``docs/_static/electrostatic_drift_gate.{png,pdf,csv,json}``.
- ``tools/artifacts/generate_electrostatic_parallel_gates.py diamagnetic`` gates the
  single-species periodic electrostatic diamagnetic drive against the
  production diamagnetic-only linear RHS. It uses the Hermite-sharded
  electrostatic field reduction plus local ``m=0`` and ``m=2`` drive masks and
  writes ``docs/_static/electrostatic_diamagnetic_gate.{png,pdf,csv,json}``.
- ``tools/artifacts/generate_velocity_parallel_gates.py periodic-streaming`` adds the periodic
  spectral parallel derivative and compares the shard-map path directly
  against ``gkx.operators.linear.streaming.streaming_ladder_term``. Its artifact
  ``docs/_static/periodic_streaming_microkernel_gate.{png,pdf,csv,json}``
  gates the first opt-in linear streaming microkernel before full RHS wiring.
- ``tools/artifacts/generate_linear_rhs_parallel_gates.py streaming`` routes the same sharded
  periodic streaming kernel through production ``linear_rhs_cached`` with all
  non-streaming terms and electromagnetic channels disabled. Its artifact
  ``docs/_static/linear_rhs_streaming_gate.{png,pdf,csv,json}`` is the first
  full-call-graph linear-RHS identity gate for velocity-space streaming.
- ``tools/artifacts/generate_linear_rhs_parallel_gates.py streaming-electrostatic`` repeats that
  gate with an ``m=0`` density perturbation and nonzero electrostatic ``phi``.
  Its artifact
  ``docs/_static/linear_rhs_streaming_electrostatic_gate.{png,pdf,csv,json}``
  gates the field-reduction-to-streaming call graph for the current
  single-species periodic electrostatic route.
- ``tools/artifacts/generate_linear_rhs_parallel_gates.py electrostatic-slices`` compares the
  composed opt-in ``backend="electrostatic_linear_slices"`` route against
  serial ``linear_rhs_cached`` with streaming, mirror, curvature, grad-B, and
  diamagnetic drive enabled. Its artifact
  ``docs/_static/linear_rhs_electrostatic_slices_gate.{png,pdf,csv,json}``
  is the current single-species periodic electrostatic linear-RHS identity
  gate for velocity-space parallelization.
- ``tools/profiling/profile_linear_rhs_parallel_slices.py`` times that same composed
  route on a larger bounded CPU workload and writes
  ``docs/_static/linear_rhs_parallel_slices_profile.{png,pdf,csv,json}``.
  The tracked profile is explicitly an engineering artifact, not a publication
  speedup claim; it uses a Hermite-heavy workload and a float32
  reduction-order tolerance so the stricter composed identity gate remains the
  release correctness check. The office GPU companion artifact
  ``docs/_static/linear_rhs_parallel_slices_profile_gpu.{png,pdf,csv,json}``
  is currently a negative performance baseline: it passes identity but is much
  slower than the single-GPU serial JIT path.
- ``tools/profiling/profile_nonlinear_sharding.py`` runs a bounded fixed-step nonlinear
  serial-vs-sharded final-state comparison and writes
  ``docs/_static/nonlinear_sharding_profile.json`` locally and
  ``docs/_static/nonlinear_sharding_profile_office_gpu.json`` for a tiny
  two-GPU smoke. The controlling transport-grid regression artifact is
  ``docs/_static/nonlinear_sharding_profile_office_gpu_benchmark_grid.json``;
  it currently fails identity and speedup, so the route remains blocked. The
  candidate nonlinear axes are ``auto``/``ky`` and ``kx``;
  ``z``-axis FFT sharding remains an exploratory domain-decomposition lane and
  must pass its own identity gate before it can be exposed as a runtime option.
  This keeps nonlinear state-sharding work profiler-backed while preventing
  unsupported runtime claims from entering the README.

Nonlinear parity snapshots
--------------------------

Recent GX parity spot checks are tracked outside the automated test suite:

- **Cyclone nonlinear short replay**: the GX `cyclone_salpha_short.in` replay
  (`dt=0.05`, `t_max=5`, collisions off, diagnostics stride 1) now uses the
  explicit short-reference runtime contract in
  ``examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear_short.toml``.
  The main short-run drift turned out to be configuration-level: the replay
  needed ``p_hyper = 2`` and no end damping to match the public GX short input.
  With that contract restored, the tracked comparison improves to
  ``mean_rel_abs(Wphi) ~= 2.11e-1`` and
  ``mean_rel_abs(HeatFlux) ~= 2.51e-1``. The resolved audit remains in
  ``docs/_static/nonlinear_cyclone_short_resolved_audit_t5.{png,csv}``, where
  ``Wphi_kyst`` is still the dominant residual mismatch.
- **Secondary (`kh01a`)**: the tracked secondary comparison now uses a dense
  real GX run (`kh01a_shortdense.out.nc`, 10 samples in ``omega_kxkyt``) and
  the rebuilt ``secondary_reference_out_compare.csv``. The comparison helper now uses
  the GX file horizon automatically in ``out-nc`` mode, so it no longer mixes a
  short GX replay with a ``t_max = 100`` GKX stage-2 run. On the matched
  short window, growth rates match tightly (``max rel_gamma ~= 1.87e-4``) and
  the non-zonal ``omega`` modes also close tightly
  (``rel_omega ~= 3.23e-4`` and ``9.92e-4`` on the ``k_y = 0.1`` sidebands).
  The only large relative ``omega`` values left are the effectively zero-
  frequency ``k_y = 0`` sidebands, where the absolute mismatch stays
  ``O(1e-6)``.
- **W7-X nonlinear (`t \\approx 200`)**: the refreshed long-window NetCDF-backed
  comparison now closes at
  ``mean_rel_abs(Phi2) ~= 9.74e-2``,
  ``mean_rel_abs(Wg) ~= 3.20e-2``,
  ``mean_rel_abs(Wphi) ~= 3.02e-2``,
  ``mean_rel_abs(HeatFlux) ~= 4.53e-2``.
- **W7-X fluctuation spectrum**: ``tools/artifacts/plot_w7x_fluctuation_spectrum_panel.py``
  reuses the same gated nonlinear NetCDF artifact and writes
  ``docs/_static/w7x_fluctuation_spectrum_panel.{png,pdf,json,csv}``. The JSON
  records the time window, dominant nonzonal ``k_y``, dominant heat-flux
  ``k_y``, dominant zonal ``k_x``, and ``claim_level``. This is a reproducible
  simulation diagnostic and explicitly not a Doppler-reflectometry transfer-
  function validation.
- **W7-X/TEM extension status**:
  ``tools/artifacts/build_tem_validation_artifacts.py w7x-extension`` reads the W7-X fluctuation panel
  plus the current TEM branch audit and writes
  ``docs/_static/w7x_tem_extension_status.{png,pdf,json,csv}``. It closes only
  the simulation-spectrum estimator. Its ``axisymmetric-branch`` mode
  writes ``docs/_static/tem_branch_parity_audit.{png,pdf,json,csv}`` from the
  tracked TEM mismatch table. TEM linear parity remains open with maximum
  absolute relative growth-rate mismatch about ``4.25``, maximum absolute
  relative frequency mismatch about ``3.3`` when near-zero reference
  denominators are excluded, one growth-rate sign mismatch, three frequency
  sign mismatches, and an inverted frequency-branch rank ordering
  (Spearman ``≈ -0.986``). Because this reference is a provisional literature
  digitization rather than a direct case dump, the audit blocks broad TEM
  claims but is not a standalone tuning target. W7-X multi-alpha,
  multi-surface, and kinetic-electron nonlinear windows remain unstarted.
- **HSX nonlinear (`t = 50`)**: the refreshed comparison closes at
  ``mean_rel_abs(Wg) ~= 2.75e-2``,
  ``mean_rel_abs(Wphi) ~= 3.61e-2``,
  ``mean_rel_abs(HeatFlux) ~= 2.91e-2``.
- **KBM nonlinear (`t = 100`)**: the refreshed long-window comparison closes at
  roughly ``9.3e-3`` mean-relative error across
  ``Wg/Wphi/Wapar/HeatFlux/ParticleFlux``.

**Generated figure.** W7-X nonlinear fluctuation-spectrum diagnostic from the gated
``t≈200``
VMEC-backed run. The panel summarizes resolved simulation spectra and is
intentionally scoped below an experimental Doppler-reflectometry comparison.


**Generated figure.** Executable TEM branch audit. The growth-rate and frequency
branches fail
simultaneously, with the frequency branch ordered oppositely to the
digitized reference over the tracked low-``k_y`` interval.


**Generated figure.** Executable status of the W7-X fluctuation/TEM extension lane. The
released
simulation-spectrum diagnostic is closed, but TEM linear parity,
alpha/surface-resolved W7-X scans, and kinetic-electron nonlinear windows
remain open before broad W7-X/TEM validation claims.


Linear physics checks
---------------------

Before nonlinear validation, we exercise linear physics checks grounded in
published benchmarks and trend tests:

- **ITG/Cyclone base case**: reproduce the standard Cyclone base case growth
  rates and frequencies across a reduced ky scan. [Dimits00]_ [Lin99]_
- **GX term-by-term audit**: use the term-dump tooling to compare GKX
  streaming and linear-kernel RHS components against GX for a single Cyclone
  state (see ``tools/comparison/compare_gx_rhs_terms.py write`` and
  ``tools/comparison/compare_gx_rhs_terms.py compare``).
- **GX nonlinear term audit (KBM/Cyclone)**: compare nonlinear
  derivative, bracket, electromagnetic split, and total RHS dumps using
  ``tools/comparison/compare_gx_nonlinear.py terms``. The tool supports GX dump folders
  with ``nl_apar.bin``/``nl_bpar.bin`` and can infer shape metadata when
  ``rhs_terms_shape.txt`` is absent.
- **ETG linear instability**: verify that growth rates remain positive across
  reduced electron-scale gradients and that the real frequency follows the
  electron diamagnetic direction. [Dorland00]_ [Jenko00]_
- **KBM beta scan**: verify the transition between ITG-like and KBM branches
  in a fixed-:math:`k_y` beta sweep against the tracked benchmark reference and
  exact-diagnostic audits.

Running tests
-------------

.. code-block:: bash

   pytest

What a bare ``pytest`` selects
------------------------------

``pytest.ini`` deselects one marker, ``slow``, and nothing else. A bare
``pytest`` therefore collects 2498 of the 2502 tests in the tree; the four it
skips are the three finite-beta VMEC/Boozer parity cases and one
adaptive-eigensolver literature case, all of which are also gated on external
checkouts. The wide package-coverage shards apply the same ``-m "not slow"``
filter, so the routine local selection and the CI coverage lane now agree.

The default used to be ``-m "not integration"``, and three files carried a
file-level ``pytestmark = pytest.mark.integration``:
``tests/unit/linear/test_linear.py``,
``tests/unit/nonlinear/test_nonlinear.py``, and
``tests/integration/runtime/test_runtime_runner.py``. A bare ``pytest``
collected **zero** tests from all three. That silently removed the published
Sugama/Coulomb collision matrices, the slab-ITG gyro-moment hierarchy check,
the collision free-energy identities, and every finite-difference gradient
contract from the local loop -- and, because the quick-test shards inherit
``addopts``, from the CI shards that name those files as well. The physics
only ever ran in the wide coverage matrix, which clears ``addopts``
altogether. Re-including those 204 tests costs about seven minutes locally.

Use the markers for what they now mean:

- ``slow``: a test whose cost is not worth paying in a routine loop. It is
  excluded by default *and* by the wide coverage shards, so marking a test
  ``slow`` removes it from the package-coverage denominator. Do not reach for
  it to quiet a test that merely takes a minute.
- ``integration``: a label for broader end-to-end or workflow tests. It no
  longer deselects anything. Prefer expressing the category with the
  directory (``tests/integration/``) and keep the marker for cases that also
  need to be selectable by name.
- ``gpu``: exercises a device-specific execution path.

If you need the old fast loop, ask for it explicitly rather than changing the
default, for example ``pytest tests/unit -x -m "not slow"`` or the bounded
per-file runner described under `CI split: fast PR vs manual full`_.

Contracts a local ``pytest`` cannot catch
-----------------------------------------

Five CI gates enforce contracts that no test in ``tests/`` will fail on for
you, because they are checked by manifest scripts, by the docs build, or on a
different interpreter. Each one is cheap to run locally; run them before
pushing rather than discovering them from a red PR.

**1. The repo-wide line budget.** The total physical line count of
``src/gkx/**/*.py`` must not exceed the ``installable_source_python_lines``
baseline in ``tools/package_architecture_manifest.toml``. The baseline is
currently 95456 lines and the tree measures exactly 95456: there is **no
headroom**, so any net addition to the installable package fails
``repo-hygiene`` until the baseline is raised.

.. code-block:: bash

   python tools/release/check_package_architecture_manifest.py
   # current count, the same way the checker counts it:
   find src/gkx -name '*.py' -exec cat {} + | wc -l

Failure reads ``installable_source_python_lines: line count regressed to N,
above baseline 95456``. The fix is either to delete as much as you added or to
raise the baseline in the manifest with an inline ``# old -> new: reason``
comment saying what the lines buy. That comment is the point of the gate;
several existing entries record exactly which physics a line increase paid
for.

**2. The per-module 1000-line cap.** ``[complexity_policy]`` in the same
manifest caps each hand-written module at ``default_max_lines = 1000``, and
each listed public facade (``api/__init__.py``, ``benchmarks.py``,
``linear.py``, ``nonlinear.py``, ``quasilinear.py``, ``runtime.py``) at 500.
Going over requires a reviewed ``[[complexity_policy.exceptions]]`` entry with
``baseline_lines``, ``target_lines``, and a ``reason``. Three modules sit above
the cap on recorded exceptions, and several sit within a handful of lines of
it -- ``operators/linear/collisions.py`` at 999 and
``solvers/linear/krylov_algorithms.py`` at 997 -- so a two-line addition there
is a CI failure, not a rounding error.

.. code-block:: bash

   python tools/release/check_package_architecture_manifest.py
   find src/gkx -name '*.py' -exec wc -l {} + | sort -rn | head -20

Failure reads ``<path>: complexity regressed to N lines, above baseline M``.

**3. The coverage-owner manifest.** Every module under ``src/gkx`` must be
reachable from ``tools/validation_coverage_manifest.toml``: as a direct
``[[modules]]`` row, inside some row's ``owned_modules``, or in
``coverage_inventory.excluded_modules``. Extracting a helper into a new file
therefore fails CI until you declare who owns its coverage. Modules of 2000
source lines or more additionally need their own direct row. ``fast_tests`` and
``artifact_paths`` must name files, never directories, and no list may repeat
an entry.

.. code-block:: bash

   python tools/release/check_validation_coverage_manifest.py
   pytest tests/release/test_release_gates.py -k "manifest"

**4. The** ``docs/api.rst`` **automodule requirement.** Every
``.. automodule:: gkx...`` directive in ``docs/api.rst`` must resolve to a real
module *and* be tracked by the coverage manifest, and a documented package
``__init__`` may not be excluded from coverage unless it is in the reviewed
exception set. Renaming or moving a public module without editing
``docs/api.rst`` fails the release-gate test; adding a directive for an
untracked module fails it too. Separately, the docs job builds with ``-W``, so
any Sphinx warning from a stale directive is an error.

.. code-block:: bash

   pytest tests/release/test_release_gates.py \
     -k "documented_public_api or large_modules_have_direct_manifest_rows"
   python -m sphinx -W -b html docs docs/_build/html

**5. The** ``python-floor`` **job.** Every other job pins 3.11 explicitly;
this one installs at the declared ``requires-python`` floor and collects the
whole suite, because collection is where an import that does not exist on the
floor actually fails. It is also where the "no per-module ``tomllib``/``tomli``
try/except" rule is enforced: TOML reads go through ``gkx.utils.tomlcompat``,
the repository's only shim.

.. code-block:: bash

   python -c "import gkx; import gkx.cli"
   gkx --help > /dev/null
   pytest -q --collect-only --disable-warnings > /dev/null
   pytest -q --disable-warnings tests/release/test_release_gates.py \
     -k "tomllib or toml_shim or repo_hygiene_gates"

The whole checklist, in the order CI runs it:

.. code-block:: bash

   python tools/release/check_repository_size_manifest.py
   python tools/release/check_package_architecture_manifest.py
   python tools/release/check_validation_coverage_manifest.py
   mypy
   pytest -q --collect-only --disable-warnings > /dev/null
   pytest tests/release/test_release_gates.py
   python -m sphinx -W -b html docs docs/_build/html

Benchmark reproducibility stack
-------------------------------

The public CI and the tracked benchmark atlas are currently validated against a
tested numerical stack. Every job prints it, so the authoritative record is the
"Print benchmark stack" step of a recent green run rather than this page; at
the time of writing it resolved to:

- ``jax`` 0.10.2 (``pyproject`` floor ``jax>=0.10.1``)
- ``jaxlib`` 0.10.2 (``pyproject`` floor ``jaxlib>=0.10.1``)
- ``numpy`` 2.4.6
- ``diffrax`` 0.7.2
- ``equinox`` 0.13.8

The jax floor is a hard one, not a preference: ``gkx.objectives.core`` calls
``lax_linalg.eig(..., enable_eigvec_derivs=True)``, which first shipped in jax
0.10.1. On an older jax the solver-objective and adaptive-eigenmode tests fail
with ``TypeError: eig() got an unexpected keyword argument``, which is an
environment symptom rather than a physics regression. Check ``python -c
"import jax; print(jax.__version__)"`` before investigating such a failure.

This is not a claim that newer releases are unsupported. It is a statement
about benchmark reproducibility. Near-marginal or branch-sensitive lanes such
as TEM, ETG runtime scans, and some imported-linear stellarator cases can move
materially under newer JAX/NumPy combinations even when the code still runs.
When investigating parity regressions, reproduce the issue on the tested stack
first before changing solver logic.
For runtime-example parity reproduction across recent precision-policy changes,
also set ``JAX_ENABLE_X64=1``. Default precision can be faster while still
moving parity-sensitive linear example outputs.

Stress-matrix parity gates
--------------------------

In addition to unit/regression tests, GKX includes a small set of
"stress-matrix" gates meant to catch parity regressions early (before tracked
benchmark figures move):

- **Restart parity**: ``tests/integration/runtime/test_restart_gate.py`` verifies that a nonlinear
  run resumed from a compatible restart reproduces the same final state as a
  continuous run. This now covers both the raw binary state path and the
  nonlinear ``*.restart.nc`` bundle path, together with append-on-restart
  history preservation in ``*.out.nc``.
- **CPU/GPU short-window parity** (optional): ``tests/unit/parallel/test_parallel_core.py -k cpu_gpu``
  compares a short nonlinear trajectory norm on CPU vs GPU. Enable explicitly:

  .. code-block:: bash

     GKX_DEVICE_PARITY=1 pytest -q tests/unit/parallel/test_parallel_core.py -k cpu_gpu

- **VMEC roundtrip determinism** (optional): ``tests/unit/geometry/test_vmec_eik.py -k roundtrip``
  regenerates an ``*.eik.nc`` from a provided VMEC file twice and asserts the
  imported geometry arrays are bitwise identical. Enable explicitly:

  .. code-block:: bash

     GKX_VMEC_FILE=/path/to/wout.nc pytest -q tests/unit/geometry/test_vmec_eik.py -k roundtrip

For developer workflows that require local reference benchmark NetCDFs or dump
artifacts, use:

- ``tools/comparison/compare_runtime.py stress-matrix`` (KAW, Cyclone kinetic electrons, KBM Miller)
- ``tools/comparison/compare_gx_imported_linear.py window`` (exact imported-linear one-window replay against reference ``diag_state`` dumps)
- ``tools/comparison/build_exact_state_audit.py run`` (manifest-driven wrapper around the exact-state audit tools)
- ``tools/comparison/build_exact_state_audit.py report`` (no-rerun W7-X exact-state convention audit panel)

The current full-GK nonlinear ETG lane is now explicitly tracked as a pilot
runtime contract via
``examples/nonlinear/axisymmetric/runtime_etg_nonlinear.toml``. Reduced
collisional-ETG runtime paths have been retired from ``main``; future ETG
parity work should use the maintained full-GK runtime.

For ETG nonlinear audit runs, use dense short-window overrides first:

.. code-block:: bash

   JAX_ENABLE_X64=1 gkx examples/nonlinear/axisymmetric/runtime_etg_nonlinear.toml \
     --steps 10 \
     --sample-stride 1 \
     --diagnostics-stride 1

This lane is currently expensive enough that short persisted windows are the
right first diagnostic step before attempting long production horizons.

The ETG short-window startup mismatch was traced to the GX input contract, not
the nonlinear ETG operator. GX reads ``init_single`` from ``[Expert]`` rather
than ``[Initialization]``, so the audited GX pilot was actually using the
Gaussian startup branch. The shipped runtime ETG pilot now matches that
contract with ``gaussian_init = true``, ``init_single = false``,
``Lx = 1.25``, and ``kz``-proportional hypercollisions. On the matched
``Nx=10``, ``Ny=22``, ``ntheta=16``, ``Nl=4``, ``Nm=4``, ``dt=1e-4``,
``t_max=0.001`` pilot, the refreshed short-window comparison lands at
``mean_rel_abs(Wg) ~= 1.31e-2`` and ``mean_rel_abs(Wphi) ~= 5.18e-3``, with
the final heat-flux point within a few percent of GX.

The targeted imported-linear wrapper and the underlying
``compare_gx_imported_linear.py`` owns the ``fields``, ``growth-dump``, and
``window`` comparison modes. Its field comparator supports two important controls
for honest stress-lane scoring without changing the default full-window
behavior:

- ``--sample-step-stride``: subsample the saved diagnostic sample indices
  before scoring.
- ``--max-samples``: truncate scoring to the first N selected samples.

The lower-level comparator also supports ``--cache-dir`` plus ``--reuse-cache``
to persist per-``ky`` trajectory/result arrays (``gamma``, ``omega``,
``Wg``, ``Wphi``, ``Wapar``) as compressed ``.npz`` files keyed by the actual
reference file, geometry file, reference input, selected ``ky``, Hermite/Laguerre
resolution, mode selector, and sample-window contract. This makes the
stress-lane tooling incremental instead of rerunning a full lane every time.
It now also writes absolute diagnostic-error columns and the reference
``|gamma|`` / ``|omega|`` scales alongside the relative metrics. That matters
for near-marginal imported-linear stellarator lanes such as HSX, where
``mean_rel_gamma`` can look large simply because the reference growth rate is
close to zero even while the absolute growth-rate mismatch and the field-energy
diagnostics remain small.

For VMEC-backed exact-state audits, the runtime bridge now prefers a local
``booz_xform_jax`` checkout and injects a temporary ``booz_xform`` compatibility
shim only into the external geometry-helper subprocess. This preserves the
audited reference workflow while avoiding a host-level dependency on the original ``booz_xform``
Python package.

The bridge auto-discovers ``booz_xform_jax`` from
``BOOZ_XFORM_JAX_PATH`` / ``GKX_BOOZ_XFORM_JAX_PATH`` or from a checkout placed
next to the GKX workspace. When a specific
Python environment is needed for the helper subprocesses, set
``geometry.geometry_helper_python`` in the runtime TOML. On ``office``, the normal audited
path is:

For differentiable VMEC/Boozer gradient audits, the ``booz_xform_jax`` checkout
must include upstream commit ``1d5e8c`` or newer. The gate is intentionally
strict because older checkouts can pass value/parity tests while returning
non-finite reverse-mode cotangents for inactive zero-mode Fourier branches.

.. code-block:: bash

   export BOOZ_XFORM_JAX_PATH=/path/to/booz_xform_jax
   export GKX_VENV_PYTHON=/path/to/venv/bin/python
   export GKX_OFFICE_ROOT=/path/to/GKX
   W7X_VMEC_FILE=/path/to/wout_w7x.nc \
   HSX_VMEC_FILE=/path/to/wout_HSX_QHS_vac.nc \
   "$GKX_VENV_PYTHON" tools/comparison/build_exact_state_audit.py run \
     --manifest tools/exact_state_lanes.office.toml \
     --outdir tools_out/exact_state_audit_office

The tracked ``office`` manifest now pins these audit lanes to
``JAX_PLATFORMS=cpu``. These are parity/reference jobs, not performance runs,
and CPU pinning avoids spurious GPU ``RESOURCE_EXHAUSTED`` failures when
``booz_xform_jax`` or grid-default assembly would otherwise grab a busy device.

The current ``office`` exact-state manifest now includes:

- startup audits for Cyclone, KBM, W7-X, and HSX
- late dumped-state audits for Cyclone Miller, Cyclone runtime, W7-X, and KBM

The tracked W7-X exact-state convention panel is generated by
``tools/comparison/build_exact_state_audit.py report`` from the ``office`` W7-X startup and
late diagnostic-state dumps. It closes the VMEC geometry, Fourier-grid,
fieldsolve, and scalar-diagnostic convention layer against GX with a
``1e-4`` pointwise relative-error gate: startup ``g_state``/``phi`` are below
``7.4e-7``, late ``kperp2``/``fluxfac``/``kx``/``ky``/``phi`` arrays have
maximum finite relative error ``4.62e-5`` with ``phi`` RMS relative error
``3.77e-7``, and late scalar diagnostics are below ``1.8e-7``. This panel is
not a replacement for the open W7-X zonal-response literature lane; it rules
out the geometry/diagnostic convention layer as the source of that separate
recurrence/damping-envelope mismatch.

**Generated figure.** W7-X nonlinear exact-state convention audit. Startup state, late
dumped
geometry/field arrays, and re-evaluated scalar diagnostics are compared
directly against GX dumps from the same VMEC equilibrium and nonlinear
runtime contract.


For KBM specifically, the startup audit, late dumped-state audit, nonlinear
term replay, and first RK4 partial-step replay now all close on the shipped
nonlinear config for the current release pass. The remaining KBM work is
therefore future long-window cleanup rather than a blocking startup-state,
diagnostic-reconstruction, or first-step assembly mismatch.

If the helper must be forced to another interpreter, set ``geometry.geometry_helper_python``
in the runtime TOML used by the audit and rerun the same command. The old
environment-variable override is no longer documented because the preferred
path is the internal ``booz_xform_jax`` backend.

CI split: fast PR vs manual full
--------------------------------

CI is split into two tiers to keep pull requests fast while preserving full
physics rigor:

- **Fast PR/push tier**: the quick-test matrix runs targeted test subsets
  across fundamentals, release artifacts, linear core, runtime, nonlinear, and
  parallel/autodiff contracts. This catches solver and dtype regressions
  quickly.
- **Wide coverage tier**: CI runs the 24 top-level coverage shards as a matrix,
  uploads the per-shard ``coverage.py`` data, then combines the artifacts in one
  final ``wide-coverage`` check that enforces the package-wide ``>=95%`` target.
  The same helper, ``tools/release/run_test_gates.py wide-coverage``, is used locally and in
  CI so the threshold is not weakened when the job is parallelized. Each shard
  has its own timeout so a single slow validation slice cannot become an
  unbounded release job. Files whose tests are gated on a device count run
  whole at that count, in one command on their own shard. Selecting such a file
  by test name instead would leave every unnamed device gate unrun while the
  shard still reported success, and splitting it into node-id batches would
  make each process re-pay the compiles the whole file pays once. The
  combine step also requires labeled coverage data
  for every CI shard and writes ``coverage-wide-shard-manifest.json`` before
  refreshing the package-wide Codecov flag.
  Optional VMEC/Boozer artifact builders remain validated by their tracked
  offline artifact gates and mocked CI contracts, not by importing unavailable
  external repositories in the public coverage job.
- **Manual full tier**: full ``pytest`` suite plus strict coverage gates:
  ``gkx.terms >= 90%`` and per-module core gates for
  ``solvers/linear/krylov.py`` and the ``solvers/time/diffrax_*`` owner modules.

This keeps iteration latency low for development and still enforces complete
coverage and regression checks on demand without relying on scheduled runners.

Gates are independent, and a skip is not a pass
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every job in ``.github/workflows/ci.yml`` checks out and installs for itself,
so the only ``needs:`` edge left in the workflow is the real one:
``wide-coverage`` consumes the coverage artifacts that ``wide-coverage-shards``
uploads. Nothing else is ordered.

That is a deliberate correction rather than a preference. ``quick-tests`` used
to declare ``needs: mypy``, so a red type check did not fail the tests -- it
*skipped* them, along with the coverage, docs, and packaging jobs behind them.
GitHub scores a skipped required check as satisfied, so the run reported no
test signal instead of a failing one, and #62 reached main with seven genuine
``NameError`` sites in ``src/gkx/parallel/integrators.py``. A first-stage gate
that silently disables everything behind it converts one red light into none.

The ``ci-required`` job closes the same hole from the other side. It depends on
every gate, runs with ``if: always()``, and fails if any of them reports
``failure``, ``cancelled``, or ``skipped``. Require that single check in branch
protection: it is the one status whose green means every gate actually ran and
actually passed.

For bounded local feedback, use the per-file runner:

.. code-block:: bash

   python tools/release/run_test_gates.py fast

It enforces both a per-file timeout and a whole-run timeout of 300 seconds by
default, then reports any remaining files as ``not_run(total_timeout)`` instead
of leaving orphaned pytest children. Use ``--total-timeout 0`` only for an
explicit full sequential local pass.

The same wide gate can be run locally in one process with:

.. code-block:: bash

   python tools/release/run_test_gates.py wide-coverage \
     --shards 48 \
     --timeout 300 \
     --fail-under 95 \
     --pytest-arg=-o \
     --pytest-arg=addopts= \
     --pytest-arg=-m \
     --pytest-arg="not slow"

Raise ``--timeout`` for the shard owning a logical-CPU file: it runs every one
of its device gates at four devices in a single command and takes nine to
sixteen minutes, which is why CI passes ``--timeout 1800``. Coverage tracing is
not what costs that; the same file takes the same time uninstrumented.

On local machines where every pytest process must stay below the five-minute
release timeout, run one shard at a time and combine afterward. This is the
same data-flow used by CI, except CI runs the ``--only-shard`` jobs in
parallel and downloads the resulting coverage artifacts before the
``--combine-only`` gate:

.. code-block:: bash

   python -m coverage erase
   for shard in $(seq 1 48); do
     python tools/release/run_test_gates.py wide-coverage \
       --shards 48 \
       --timeout 300 \
       --only-shard "${shard}" \
       --keep-existing-coverage \
       --skip-combine \
       --pytest-arg=-o \
       --pytest-arg=addopts= \
       --pytest-arg=-m \
       --pytest-arg="not slow"
   done
   python tools/release/run_test_gates.py wide-coverage \
     --shards 48 \
     --combine-only \
     --fail-under 95 \
     --pytest-arg=-o \
     --pytest-arg=addopts= \
     --pytest-arg=-m \
     --pytest-arg="not slow"

Core modular coverage gate
--------------------------

To keep the modular RHS path future-proof, CI also enforces a dedicated
coverage gate for ``gkx.terms``:

.. code-block:: bash

   pytest -q tests/unit/operators/test_terms_assembly.py \
          tests/unit/operators/test_linear_streaming.py \
          tests/unit/operators/test_terms_fields.py \
          tests/unit/solvers/test_nonlinear_explicit_scan.py \
          --maxfail=1 --disable-warnings \
          --cov=src/gkx/terms \
          --cov-fail-under=90

This guard ensures term-wise kernels, field solves, custom-VJP behavior, and
assembly plumbing stay highly covered while the rest of the benchmark and
cross-code harness keeps evolving.

Core solver coverage gates
--------------------------

CI also enforces dedicated per-module thresholds for the two linear solver
engines that are most likely to regress during algorithm work:

- ``gkx.solvers.linear.krylov`` (matrix-free Arnoldi/shift-invert path)
- ``gkx.solvers.time.diffrax_linear``/``diffrax_nonlinear``/``diffrax_core`` (Diffrax explicit/IMEX/implicit paths)

The gate runs focused tests and checks each module from ``coverage-core.xml``:

.. code-block:: bash

   pytest -q tests/unit/solvers/test_linear_krylov_core.py \
          tests/unit/solvers/test_diffrax_integrators_core.py \
          --maxfail=1 --disable-warnings \
          --cov=src/gkx \
          --cov-report=xml:coverage-core.xml

Both modules are required to stay at or above 90% line coverage in CI.
