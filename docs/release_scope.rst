Release Scope and Claim Boundaries
==================================

This page is the canonical claim-scope checklist for the current development
state. It keeps README, documentation, release notes, and manuscript drafts
aligned with the tracked artifacts in ``docs/_static``. If a claim is not
listed here or in the referenced gate JSON, treat it as unpromoted.
Here, "current" means artifact-backed and release/manuscript scoped. "Deferred"
means visible as an audit or planning lane but unavailable for release notes,
README highlights, abstracts, or paper conclusions until a later gate promotes
it.

Current scoped claims
---------------------

Claim scope for this release is intentionally artifact-limited: each
release-ready claim below must be backed by the cited tracked figure, JSON
report, test, or workflow gate. Open manuscript physics lanes stay visible in
the guardrail artifacts, but they are not promoted by the release-readiness
score.

.. list-table::
   :header-rows: 1
   :widths: 28 18 54

   * - Lane
     - Status
     - Supported claim
   * - Linear and nonlinear benchmark atlas
     - release-ready for named cases
     - Linear growth/frequency/eigenfunction and nonlinear window statistics
       are validated for the tracked release cases. The nonlinear window
       statistics gate includes only Cyclone, Cyclone Miller, KBM, W7-X, and
       HSX. ETG nonlinear pilots and KAW/TEM stress lanes are not part of the
       release nonlinear parity claim unless a later gate index admits them.
   * - Runtime/refactor artifact contract
     - release-ready as infrastructure
     - The large runtime and diagnostics refactor is covered as a behavior
       preservation claim: extracted startup, chunk, result, validation-gate,
       and artifact helpers keep the public runtime and NetCDF restart/append
       contracts stable. This does not promote new physics validation,
       nonlinear optimization, or performance claims.
   * - Quasilinear diagnostics
     - release-ready as diagnostics
     - Electrostatic linear heat/particle weights, spectra, and model-selection
       artifacts are reproducible. The refreshed 12-case train/holdout
       calibration report rejects the one-constant absolute-flux family, with
       CTH-like and shaped-pressure external VMEC admitted only through
       explicit high-grid policies. Simple one-scalar saturation rules are
       rejected on the expanded sweep. The ``spectral_envelope_ridge``
       candidate closes the declared core portfolio after excluding the
       Solovev and shaped-pressure stress outliers from that scoped claim:
       core mean relative error is about ``0.280``, held-out core error is
       about ``0.275``, and interval coverage is ``10/10``. It is retained as
       a scoped model-development and optimization-screening result. The full
       12-case universal predictor remains unpromoted because the stress cases
       and rank/correlation gates do not pass. No runtime/TOML absolute-flux
       predictor, universal nonlinear transport model, or user-facing
       saturation law is promoted.
       Electromagnetic quasilinear field-channel normalization and KBM
       calibration remain future gates.
   * - Differentiable geometry
     - release-ready for equal-arc parity and reduced QH/Li383 gates
     - The ``vmex -> booz_xform_jax -> GKX`` bridge is validated
       for equal-arc field-line parity where the current ``mboz=nboz=21``
       parity artifact passes QH, QI, and shaped-pressure finite-beta rows.
       The fixed-resolution QI row now passes after the Boozer half-mesh
       convention fix, with drift mismatch about ``7.13e-2`` against the
       ``8e-2`` tolerance, and the evaluated QI ``ntheta=8,16`` variants pass.
       The shaped-pressure finite-beta eigenfrequency-gradient and
       quasilinear-gradient gates also pass with max relative
       AD/finite-difference errors about ``6.4e-11`` and ``2.1e-4``.
       The shaped-pressure finite-beta reduced nonlinear-window estimator gate
       also passes with max relative error about ``2.1e-4``.
       This is still not a broad QI transport, finite-beta nonlinear
       transport-gradient, or optimization claim. Reduced frequency,
       quasilinear, and nonlinear-window-estimator gradients pass
       AD/finite-difference gates on QH and Li383. The actual nonlinear
       finite-difference audits are startup plumbing checks with false
       transport-average gates; they do not validate production turbulence
       gradients.
   * - VMEC/Boozer reduced objectives
     - release-ready for reduced gradient and UQ plumbing
     - The public in-memory objective path supports reduced linear frequency,
       electrostatic quasilinear proxy, and smooth nonlinear-window-estimator
       objectives through the mode-21 VMEC/Boozer bridge. The QH and Li383
       holdout matrix is the citeable gate for these reduced objectives. The
       QL-seeded nonlinear-gradient state-control screen has admitted
       ``Rsin_mid_surface_m1`` and ``Zcos_mid_surface_m1`` as internal
       VMEC-state controls. The first symmetric ``RBC/ZBS`` state-to-input
       response fails closed, while the ``LASYM=true`` ``RBS/ZBC`` response
       passes as a conditioned launch mapping. This row still does not promote
       multi-surface/multi-alpha optimization, calibrated absolute
       quasilinear flux prediction, or converged nonlinear heat-flux gradients.
   * - Stellarator optimization examples
     - release-ready as reduced examples plus selected optimized-equilibrium audit
     - The examples demonstrate differentiable reduced ITG objectives, UQ, and
       AD/finite-difference checks. The nonlinear objective is a reduced
       window-estimator path, not a nonlinear turbulence-gradient path. The
       selected optimized QA equilibrium now has a converged post-transient
       seed/timestep transport-window audit, so the production guard is closed
       for that scoped audit. Broad multi-surface nonlinear optimization and
       nonlinear turbulence gradients remain unpromoted. A VMEC-JAX
       transport-gradient diagnostic now also shows a measurable local
       boundary gradient for the aspect-6 QA restart and a solved-gate
       projected line-search bracket: the best accepted reduced transport
       metric improves by ``3.55%``, while the next larger step is rejected by
       the QS gate. This supports gate-aware projected admission. The matched
       long-window audit of that earlier aspect-6 accepted projected step is negative: both
       seed/timestep ensembles pass, but the ensemble mean heat flux changes
       from ``9.833`` to ``9.891`` (relative reduction ``-0.00585``). It is
       therefore not a nonlinear turbulent-flux optimization claim. The
       companion redesign report fails closed and requires a multi-surface,
       multi-field-line, multi-``k_y`` objective before another nonlinear
       audit can be used for promotion. A later strict top-12 QA edge audit
       uses that 18-point objective coverage but still fails promotion
       (``0.58%`` relative reduction, uncertainty z-score ``0.20``). The
       subsequent broad max-mode-5 matrix campaign is also negative: accepted
       QA/ESS passes only ``9/18`` samples, projected weight ``1e-3`` fails
       early with ``1/18`` passing samples and mean reduction below ``2%``, and
       projected weight ``5e-4`` increases heat flux on the first completed
       sample. The release scope therefore remains reduced-objective and
       scoped-audit evidence only.
   * - Parallelization
     - production-ready for independent work
     - Independent ``k_y`` scans, quasilinear spectra, sensitivity batches, and
       UQ ensembles preserve serial ordering and have solver-backed scaling
       artifacts. Runtime scan TOMLs may use ``[parallel] strategy = "batch"``
       with ``axis = "ky"`` for this independent scan path. Whole-state
       nonlinear sharding is a correctness/profiler gate only.
   * - Performance
     - release-ready for scoped profiler evidence
     - Runtime/memory panels, RHS profiler artifacts, and state-sharding
       identity checks are tracked. No broad nonlinear multi-GPU speedup or
       production domain-decomposition claim is made.

Explicitly unpromoted claims
----------------------------

Do not make these claims from the current artifacts:

- universal or user-facing absolute quasilinear flux prediction;
- treating refactor/test coverage as new physics validation or as a nonlinear
  performance claim;
- using ``spectral_envelope_ridge`` as a shipped runtime or TOML saturation
  option;
- electromagnetic quasilinear transport calibration for KBM;
- broad multi-surface production nonlinear heat-flux stellarator optimization;
  the tracked broad matrix campaign failed all selected candidate families and
  is recorded as negative evidence in
  ``docs/_static/broad_nonlinear_transport_matrix_negative_evidence.json``;
- production nonlinear optimization without converged post-transient audits of
  optimized equilibria; the selected QA optimized-equilibrium audit is the
  current scoped exception;
- treating the historical strict QA full-sweep matched audit that stopped near
  ``t=400`` as a nonlinear holdout or optimized-transport success; it remains
  launch-contract evidence only;
- treating the newly admitted true ``t=1500`` growth-objective,
  quasilinear-objective, or nonlinear-window-objective QA candidate triplets as
  an optimization-success or quasilinear-calibration claim. The matched strict
  QA baseline now passes the same ``t=[1100,1500]`` postprocess, and all three
  candidate comparisons fail the ``4%`` reduction gate: growth gives only
  ``0.60%`` reduction (``z=0.26``), while quasilinear and nonlinear-window give
  ``-0.49%`` (``z=-0.19``) and ``-0.25%`` (``z=-0.09``), respectively.
- converged nonlinear transport gradients through ``vmex`` and
  ``booz_xform_jax``;
- launching nonlinear-gradient campaigns directly from admitted VMEC-state
  controls without a separate state-to-input mapping artifact;
- treating compact nonlinear finite-difference startup audits as saturated
  transport averages;
- treating reduced nonlinear-window estimators or startup finite-difference
  audits as optimized-equilibrium nonlinear heat-flux audit bars;
- multi-surface, multi-alpha, or multi-``k_y`` stellarator optimization from
  the current reduced single-fixture objective evidence;
- broad W7-X validation beyond the tracked single-flux-tube ITG windows;
- broad QI validation beyond the fixed-resolution mode-21 equal-arc parity row;
- citing even the fixed-resolution QI mode-21 row when the latest regenerated
  parity artifact fails, errors, or is missing;
- W7-X TEM / kinetic-electron validation;
- W7-X long-window zonal recurrence/damping closure;
- nonlinear multi-GPU speedup from whole-state sharding;
- FFT-axis nonlinear domain decomposition.

Release figure and artifact inventory
-------------------------------------

Use this inventory when deciding which figures can support release notes,
README claims, or manuscript claims.

.. list-table::
   :header-rows: 1
   :widths: 24 46 30

   * - Claim family
     - Current release/manuscript artifacts
     - Boundary
   * - Benchmark validation
     - ``benchmark_core_linear_atlas.png``,
       ``benchmark_core_nonlinear_atlas.png``,
       ``nonlinear_window_statistics.{png,json}``, and
       ``validation_gate_index.{png,json}``
     - Nonlinear release parity is the five-case window-statistics set only.
       Stress, pilot, and non-indexed example figures are not promoted.
   * - Quasilinear diagnostics and model selection
     - ``quasilinear_*_spectrum.*``,
       ``quasilinear_validated_calibration_inputs.*``,
       ``quasilinear_stellarator_train_holdout.*``,
       ``external_vmec_dshape_replicates/dshape_replicate_t250_ensemble_gate.*``,
       ``external_vmec_circular_replicates/circular_replicate_t700_ensemble_gate.*``,
       ``quasilinear_saturation_rule_sweep.*``,
       ``quasilinear_candidate_uncertainty.*``, and
       ``quasilinear_dataset_sufficiency.*``
     - Electrostatic diagnostics and manuscript model selection are in scope.
       Runtime absolute-flux prediction and electromagnetic calibration are not.
   * - Autodiff and differentiable geometry
     - ``autodiff_inverse_growth.*``, ``autodiff_inverse_twomode.*``,
       ``differentiable_geometry_bridge.*``, ``vmec_boozer_parity_matrix.*``,
       ``vmec_boozer_gradient_holdout_matrix.*``,
       ``vmec_boozer_differentiability_claim_guard.*``,
       ``nonlinear_window_fd_audit.*``, and
       ``vmec_boozer_nonlinear_window_fd_audit.*``
     - Reduced AD/finite-difference gates are in scope. Production nonlinear
       turbulence-gradient and broad optimized-equilibrium heat-flux claims are
       not; the selected QA optimized-equilibrium replicated audit is covered by
       the scoped stellarator-optimization row rather than by this general AD
       inventory row.
   * - VMEC/Boozer objective and optimization checklist
     - ``vmec_boozer_solver_frequency_gradient_gate.*``,
       ``vmec_boozer_quasilinear_gradient_gate.*``,
       ``vmec_boozer_nonlinear_window_gradient_gate.*``,
       ``vmec_boozer_li383_solver_frequency_gradient_gate.*``,
       ``vmec_boozer_li383_quasilinear_gradient_gate.*``,
       ``vmec_boozer_li383_nonlinear_window_gradient_gate.*``,
       ``vmec_boozer_gradient_holdout_matrix.*``,
       ``vmec_boozer_multi_point_objective_gate.*``,
       ``vmec_boozer_reduced_portfolio_guard.json``,
       ``vmec_boozer_aggregate_line_search_comparison.*``,
       ``vmec_boozer_aggregate_alpha_holdout_gate.*``,
       ``vmec_boozer_aggregate_surface_holdout_gate.*``,
       ``vmec_boozer_second_equilibrium_aggregate_gate.*``,
       ``vmec_boozer_aggregate_holdout_promotion_gate.json``,
       ``nonlinear_gradient_ql_seed_screen.*``,
       ``nonlinear_gradient_state_control_runbook.*``,
       ``nonlinear_gradient_state_to_input_mapping_campaign.*``,
       ``nonlinear_gradient_state_to_input_mapping_response.*``,
       ``nonlinear_gradient_asymmetric_state_to_input_mapping_campaign.*``,
       ``nonlinear_gradient_asymmetric_state_to_input_mapping_response.*``,
       ``nonlinear_gradient_state_control_short_bracket_launch.*``,
       ``nonlinear_gradient_state_control_short_bracket_launch_status.*``,
       ``nonlinear_gradient_state_control_short_bracket_nonlinear_audit_status.*``,
       ``vmex_transport_gradient_diagnostic.json``,
       ``vmex_transport_gradient_line_search.*``,
       ``nonlinear_window_ensemble_readiness_manifest.json``,
       ``nonlinear_window_convergence_reports/*.json``,
       ``stellarator_itg_optimization_comparison.*``, and
       ``stellarator_itg_optimization_uq.*``
     - These artifacts support reduced objective differentiability, optimizer
       plumbing, local UQ, explicit nonlinear ensemble-readiness blockers, and
       a checked state-control launch guard with a retained fail-closed symmetric negative control. They do not support
       calibrated saturated-flux prediction, production nonlinear turbulence
       gradients, direct VMEC-state launches, or optimized-equilibrium nonlinear
       audits beyond the selected QA candidate documented below.
   * - Scope guardrails
     - ``benchmarks/references/gkx_1_7_release_contract.json``,
       ``technical_release_status.json``, ``release_readiness.json``, and
       ``w7x_tem_extension_status.json``
     - The frozen contract records release-scoped claims and baseline API and
       performance evidence. It also release-gates canonical numeric
       fingerprints for the promoted nonlinear, KBM branch, collision,
       restart, geometry, derivative, and optimization references, so textual
       renaming cannot hide a numerical change. It does not promote deferred
       physics by itself.
   * - Performance and parallelization
     - ``runtime_memory_benchmark.*``,
       ``independent_ky_scan_scaling_large.*``,
       ``quasilinear_uq_ensemble_scaling_large.*``, and
       ``parallelization_completion_status.*``, plus
       ``nonlinear_sharding_*``
     - Independent-work parallelization and profiler localization are in scope.
       Whole-state nonlinear sharding is not a production speedup claim.

Artifact-backed details
-----------------------

Runtime/refactor state:

- The current large refactor has extracted runtime startup, diagnostics,
  adaptive chunks, result assembly, validation-gate helpers, zonal-validation
  helpers, parallelization policy helpers, and runtime artifact boundaries into
  smaller tested modules. This is a maintainability and public-behavior
  preservation lane.
- Restartable nonlinear NetCDF append now normalizes loaded diagnostics to the
  persisted schema before concatenation. Monitored complex mode traces that are
  transient in memory and not written to ``*.out.nc`` remain absent on reload,
  so continuation artifacts do not mix persisted and non-persisted diagnostic
  fields.
- These refactor checks support release engineering only. They do not change
  the benchmark, quasilinear, QI, nonlinear optimization, or performance claim
  surface without the artifact gates listed below.

Quasilinear model-selection state:

- ``docs/_static/quasilinear_stellarator_train_holdout_report.json``:
  nonlinear inputs are valid, but the one-constant absolute-flux model remains
  ``passed = false`` with held-out mean relative error about ``6.49``.
- ``tools/release/check_nonlinear_transport_gates.py convergence`` and
  ``gkx.diagnostics.transport_windows`` provide the reusable late-window
  convergence metadata required before any future holdout report can be
  promoted to ``calibrated_absolute_flux``. This is a metadata/finite-window
  guardrail over existing traces, not a substitute for new long nonlinear
  simulations.
- ``gkx.diagnostics.transport_windows.nonlinear_window_ensemble_report`` provides
  the next guardrail for replicated windows: seed, initial-condition, timestep,
  or restart variants must have individually passed late-window reports and
  mutually consistent late means before a nonlinear turbulent-flux optimization
  artifact can claim robustness. ``tools/release/check_nonlinear_transport_gates.py ensemble``
  is the tracked artifact wrapper for this gate.
- ``tools/release/check_nonlinear_transport_gates.py readiness`` converts tracked
  transport-window summaries into explicit convergence-report JSON files and a
  readiness manifest. The older global
  ``docs/_static/nonlinear_window_ensemble_readiness_manifest.json`` remains a
  base-window manifest, but the current D-shaped and circular case-local
  replicate campaigns now pass their own ensemble gates. Those case-local
  artifacts supersede the stale global missing-replicate message for those two
  cases. The QH VMEC/Boozer held-out surface/field-line campaign and the
  selected optimized-equilibrium audit now also pass their local seed/timestep
  ensemble gates.
- ``tools/release/check_vmec_boozer_gates.py aggregate-holdout`` now requires a passed
  replicated nonlinear-window ensemble artifact in addition to aggregate
  finite-difference, line-search, and held-out surface/field-line evidence
  before any optimized-equilibrium production nonlinear heat-flux claim can be
  promoted. Single-window convergence reports remain necessary but insufficient
  for that claim level.
- ``tools/release/check_nonlinear_optimization_gates.py production-guard`` is the explicit
  production nonlinear turbulent-flux optimization guard. Its tracked artifact,
  ``docs/_static/production_nonlinear_optimization_guard.json``, passes release
  safety because reduced/startup estimators are blocked and three long
  post-transient replicated holdout ensembles pass: D-shaped VMEC, circular
  VMEC, and QH VMEC/Boozer. The selected optimized QA equilibrium contributes
  one accepted ``t=[350,700]`` seed/timestep replicated transport-window audit,
  and the strict ``t=1500`` growth/QL/nonlinear-window candidates now close the
  optimized-equilibrium trace-count requirement with four qualifying
  ensembles. The scoped production nonlinear turbulent-flux optimization guard
  now promotes under its explicit ``2%`` long-window matched-audit policy:
  three matched baseline-to-optimized audits pass with positive
  uncertainty-separated heat-flux reductions. This is scoped candidate
  evidence, not a broad multi-surface nonlinear transport-optimization claim.
- ``tools/release/check_nonlinear_optimization_gates.py gradient-evidence`` is the stricter
  nonlinear turbulence-gradient claim gate. The tracked
  ``docs/_static/nonlinear_turbulence_gradient_evidence_status.json`` artifact
  passes the replicated long-window uncertainty side but fails closed on the
  gradient side. The current tracked production-candidate artifact is the
  optimized-QA/ESS ``ZBS(1,0)`` 7.5% follow-up at ``t=[450,900]``: all twelve
  runtime outputs pass, the baseline and minus replicated ensembles pass, and
  the central finite difference is both response-resolved
  (``response_fraction = 0.0319``) and local
  (``fd_asymmetry_rel = 0.044``). It still fails promotion because the plus
  ensemble spread is ``0.196 > 0.15`` and the propagated uncertainty is
  ``gradient_uncertainty_rel = 1.81 > 0.5``. The earlier overdetermined
  optimized-QA/ESS ``RBC(1,1)`` 3% campaign and seed follow-up also remain
  failed production candidates: all runtime-output and replicated-window gates
  pass, but ``gradient_uncertainty_rel = 0.683`` remains above the ``0.5`` gate.
  The companion ``ZBS(1,1)`` 3% overdetermined campaign passes uncertainty but
  remains nonlocal, while the overdetermined ``ZBS(1,0)`` bracket is not
  response-resolved. The status artifacts therefore record complete runtime
  coverage where expected and zero promoted controls, so this remains a failed
  production-candidate gate rather than a missing campaign.
  Until a paired post-transient artifact passes all response, asymmetry,
  conditioning, and propagated uncertainty gates, nonlinear turbulence-gradient
  evidence remains explicitly unpromoted.
- Future perturbation refreshes must use distinct artifact slugs rather than
  overwriting the tracked failed candidate. For example, a new coefficient or
  amplitude campaign should write a slug such as
  ``docs/_static/qa_ess_zbs11_rel5_nonlinear_gradient_zbs_1_1_central_fd_gradient_gate.*``
  and a matching refreshed
  ``nonlinear_turbulence_gradient_evidence_status.json``. Release prose can
  promote the result only if the central finite-difference artifact passes
  and the evidence-status JSON reports the production gradient gate as true;
  otherwise it remains a documented production-candidate audit.

- The tracked QA/ESS campaign is complete negative evidence, not an unfinished
  launch plan: all required long-window runtime outputs exist, but no boundary
  control passes every finite-difference gate. ``ZBS(1,1)`` is nonlocal,
  ``ZBS(1,0)`` is variance limited, and increasing the ``RBC(1,1)``
  amplitude worsens asymmetry.
- ``docs/_static/quasilinear_saturation_rule_sweep.json``:
  no simple saturation rule is accepted. On the expanded saturation sweep,
  the linear-weight fit is the least-bad simple rule with mean held-out
  relative error about ``4.42``; the positive-growth mixing-length rule is
  about ``6.49`` and the training-mean null is about ``1.80``.
- ``docs/_static/quasilinear_candidate_uncertainty.json``:
  no candidate is accepted as a universal runtime absolute-flux predictor on
  the expanded 12-case electrostatic-compatible candidate portfolio.
  ``spectral_envelope_ridge`` has full-ledger mean relative error about
  ``0.697`` and interval coverage ``11/12``; with the declared Solovev and
  shaped-pressure stress outliers outside the scoped claim, its core-portfolio
  mean relative error is about ``0.280`` and held-out core error is about
  ``0.275``.
- ``docs/_static/quasilinear_candidate_regularization_sweep.json``:
  the ridge-penalty sensitivity audit does not rescue that near miss. The best
  tested setting is now ``lambda = 0.5`` with mean relative error about
  ``0.689`` and held-out mean about ``0.764``; no tested penalty is accepted as
  an absolute-flux predictor.
- ``docs/_static/quasilinear_stellarator_usefulness.json``:
  the current stellarator-facing synthesis is scoped as
  ``scoped_model_skill_summary_not_runtime_absolute_flux_predictor``. It
  records HSX/W7-X as admitted finite nonlinear holdouts where the simple
  positive-growth mixing-length rule predicts zero, records CTH-like and
  shaped-pressure as scoped high-grid external-VMEC admissions, keeps QA at
  matched-nonlinear-audit-only scope, and keeps QH excluded until grid/window
  convergence passes. No model-selection result is currently accepted as a
  universal stellarator absolute-flux predictor.
- ``docs/_static/quasilinear_screening_skill.json``:
  the current correlation/ranking synthesis is scoped as
  ``screening_correlation_model_development_not_absolute_flux_promotion``. It
  records no accepted screening model on the expanded 12-case candidate
  portfolio. The least-bad ``spectral_envelope_ridge`` candidate has
  full/held-out Spearman correlations about ``0.636``/``0.624`` and pairwise
  order accuracies about ``0.697``/``0.689``, below the ``0.75`` gates. The
  declared core portfolio passes the transport/coverage diagnostic but remains
  just below the strict rank gate. Screening skill is therefore not promoted as
  a runtime saturation law or universal absolute-flux predictor.
- ``docs/_static/quasilinear_holdout_gap_report.json``:
  absolute-flux promotion remains explicitly blocked. The
  ``absolute_flux_promotion_requirements`` and
  ``screening_promotion_requirements`` blocks quantify the current gaps after
  admitting the shaped-pressure external-VMEC high-grid holdout: the absolute
  train/holdout mean relative error is about ``6.49`` against the ``0.35``
  gate, no full-portfolio or held-out-only screening model is currently
  accepted, and the independent-holdout-count blocker is closed. Screening
  promotion still fails the rank/correlation and transport-error gates. The external-VMEC-family
  and non-axisymmetric external-VMEC-family coverage requirements are
  satisfied, but these are evidence prerequisites, not a promoted runtime
  absolute-flux option.
- ``docs/_static/external_vmec_shaped_tokamak_pressure_dt0p04_high_grid_admission_gate.json``:
  shaped-tokamak pressure is now admitted only as a scoped high-grid holdout.
  The full ``n48/n64/n80`` ladder fails because the coarse ``n48`` trace moves
  the heat-flux window by about ``0.469``. The retained ``n64/n80`` gates pass
  at ``t=450`` and ``t=650``; the high-grid time-horizon gate passes; and the
  ``n80`` seed/timestep ensemble passes on ``t=[325,650]`` with mean heat flux
  about ``7.16``, mean-relative spread ``0.0939``, and combined SEM/mean
  ``0.0463``. This does not claim full ``n48/n64/n80`` convergence and does
  not promote an absolute quasilinear-flux predictor.

Nonlinear benchmark state:

- ``docs/_static/nonlinear_window_statistics.json`` records five passed
  release-window cases. KBM and HSX use tightened gates, Cyclone Miller is
  tighter than the broad release envelope, while Cyclone and W7-X remain at the
  ``0.10`` release envelope pending paper-level retuning.
- ``docs/_static/validation_gate_index.json`` currently records ``17`` passed
  gate-indexed reports and ``1`` open report. The open report is the
  quasilinear model-selection status, which is intentionally not promoted to
  an absolute-flux predictor. The index is an audit view, not a blanket
  promotion of every figure under ``docs/_static``.
- ``docs/_static/nonlinear_transport_time_horizon_audit.json`` separates
  long post-transient transport windows from startup finite-difference and
  reduced-envelope checks. Startup windows must never be described as saturated
  heat-flux averages.

Differentiable-geometry state:

- ``docs/_static/vmec_boozer_parity_matrix.json`` is the source of truth for
  the multi-equilibrium zero-beta equal-arc field-line convention gate at
  ``mboz=nboz=21``. The current regenerated artifact passes QH, fixed-
  resolution QI, and shaped-tokamak rows. The QI row
  ``nfp3_QI_fixed_resolution_final`` has drift mismatch about ``7.13e-2``
  against the ``8e-2`` release tolerance, and evaluated QI ``ntheta=8,16``
  robustness variants pass. The full declared QI seed campaign is still
  artifact-limited because three QI input variants have no bundled ``wout``
  reference. The builder rejects ``mboz,nboz < 21`` so QI is not silently
  evaluated on the under-resolved low-mode setting.
- ``docs/_static/vmec_boozer_gradient_holdout_matrix.json`` passes reduced
  linear, quasilinear, and nonlinear-window-estimator gradient gates on QH and
  Li383 with maximum relative mismatch about ``2.7e-2``.
- The VMEC/Boozer objective artifact checklist for README and manuscript use is
  the parity matrix, the six single-equilibrium frequency/quasilinear/reduced
  nonlinear-window gradient-gate figures, the combined holdout matrix, the
  multi-alpha aggregate objective gate, the reduced-portfolio provenance guard,
  the growth-vs-quasilinear line-search comparison, the positive reduced
  alpha-heldout and surface-heldout splits, the Li383 second-equilibrium
  aggregate gate, the blocked aggregate promotion JSON,
  ``nonlinear_gradient_ql_seed_screen.*``,
  ``nonlinear_gradient_state_control_runbook.*``,
  ``nonlinear_gradient_state_to_input_mapping_campaign.*``,
  ``nonlinear_gradient_state_to_input_mapping_response.*``,
  ``nonlinear_gradient_asymmetric_state_to_input_mapping_campaign.*``,
  ``nonlinear_gradient_asymmetric_state_to_input_mapping_response.*``,
  ``nonlinear_gradient_state_control_short_bracket_launch.*``,
  ``nonlinear_gradient_state_control_short_bracket_launch_status.*``,
  ``nonlinear_gradient_state_control_short_bracket_nonlinear_audit_status.*``,
  ``nonlinear_gradient_state_control_bracket_sweep_status.*``,
  ``optimized_equilibrium_replicates/optimized_equilibrium_replicate_t700_ensemble_gate.*``,
  ``qa_no_ess_reference_replicates/qa_no_ess_reference_t700_ensemble_gate.*``,
  ``qa_no_ess_to_optimized_nonlinear_audit.*``,
  ``qa_ess_zbs10_rel7p5_nonlinear_gradient_zbs_1_0_central_fd_gradient_gate.*``,
  ``qa_ess_zbs10_rel7p5_variance_reduction_plan.*``,
  and the reduced stellarator ITG optimization/UQ panels. This checklist is the
  current boundary between objective plumbing, checked state-control launch
  guards, and transport prediction.
- ``docs/_static/vmec_boozer_reduced_portfolio_guard.json`` is the
  artifact-level guard that ties the backend-free portfolio reducer to real
  VMEC/Boozer rows. It requires VMEC/Boozer path/mode provenance, two
  field-line ``alpha`` values, two ``k_y`` samples, finite aggregate FD fields,
  finite growth/QL AD/FD objective gates, and an explicit non-production
  nonlinear claim boundary.
- ``docs/_static/nonlinear_window_fd_audit.json`` and
  ``docs/_static/vmec_boozer_nonlinear_window_fd_audit.json`` pass only startup
  finite-difference plumbing checks. Both record ``transport_average_gate =
  false``.
- Finite-beta drift reconstruction, converged nonlinear turbulence gradients,
  broader VMEC/Boozer nonlinear transport-gradient validation, and broader
  optimized-equilibrium nonlinear audits beyond the selected QA candidate remain
  future promotion gates.

Parallelization and performance state:

- The current-branch representative linear/nonlinear refresh admits two local
  CPU rows and passes CPU/GPU numerical checks. Its office-GPU timing rows are
  explicitly blocked by measured device contention, so this refresh does not
  promote a new speed claim.
- ``docs/_static/independent_ky_scan_scaling_large.json`` and
  ``docs/_static/quasilinear_uq_ensemble_scaling_large.json`` support
  production independent-work parallelization for scans and ensembles.
- ``docs/_static/parallelization_completion_status.json`` is the release
  closure ledger for parallelization: production independent-work CPU/GPU
  scaling is closed, while nonlinear domain and FFT-axis decomposition remain
  diagnostic.
- ``docs/_static/nonlinear_sharding_strong_scaling_large.json`` is an identity
  and profiler-direction artifact. It shows whole-state nonlinear sharding is
  identity-correct but not a production speedup path for the current
  decomposition.
- ``docs/_static/nonlinear_domain_parallel_identity_gate.json`` and
  ``docs/_static/nonlinear_spectral_communication_identity_gate.json`` are
  diagnostic identity gates for local halo chunks and spectral
  split/reassemble communication layout, respectively. They are correctness
  prerequisites for future nonlinear domain decomposition, not runtime
  distributed-FFT or nonlinear speedup claims.
- ``docs/_static/nonlinear_sharding_profile_office_gpu.json`` is a tiny smoke
  artifact. The controlling benchmark-grid result is
  ``docs/_static/nonlinear_sharding_profile_office_gpu_benchmark_grid.json``;
  it is slower than serial and fails final-state identity, so whole-state
  nonlinear sharding is blocked from production use and speedup claims.
- ``docs/_static/nonlinear_device_z_pencil_transport_gpu2_observable_split_profile.json``
  is the final performance artifact for this release tranche. It passes
  serial-vs-sharded identity on the auto-chunked two-GPU transport-window
  diagnostic, but compute-only speedup remains below the promotion gate and the
  observable gate is dominated by scalar diagnostic overhead. This closes the
  current performance work as diagnostic evidence and defers production
  nonlinear domain decomposition to a future fused RHS/update diagnostic route.

Deferred manuscript lanes
-------------------------

The current manuscript/readme scope intentionally defers:

- W7-X zonal long-window recurrence/damping closure under the paper-facing
  initializer and observable;
- W7-X multi-flux-tube, multi-surface, and TEM / kinetic-electron validation;
- experimental W7-X fluctuation-spectrum claims through diagnostic transfer
  functions.

These release-scope decisions are frozen in
``benchmarks/references/gkx_1_7_release_contract.json``; detailed W7-X extension
state remains in ``docs/_static/w7x_tem_extension_status.json``. W7-X zonal
recurrence and TEM/kinetic-electron extension are explicitly deferred. The W7-X
fluctuation-spectrum panel is a validated simulation diagnostic only; it is not
an experimental density-spectrum validation.

Pre-release checklist
---------------------

Before tagging a new public release:

1. Run the fast shard set, docs build, package build, repo hygiene, mypy, and
   wide coverage matrix.
2. Confirm the coverage workflow reports the package-wide ``95%`` gate and
   that ``coverage-wide-shard-manifest.json`` has labeled data for every wide
   coverage shard.
3. Confirm README and this page agree with
   ``benchmarks/references/gkx_1_7_release_contract.json``.
4. Confirm runtime/performance claims point to fresh profiler artifacts for
   the exact backend, device count, problem size, and identity tolerance being
   claimed.
5. Bump the package version before tagging; PyPI rejects duplicate versions.
