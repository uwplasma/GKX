Manuscript Figures
==================

Purpose
-------

This page tracks the target figure set for the future GKX paper. A
figure is only ready for manuscript use when it has:

- one owning script,
- one reproducible artifact path,
- a declared reference,
- a declared acceptance status.

Current Readiness Snapshot
--------------------------

The current scoped manuscript stack is ready for claims about validated
quasilinear diagnostics/model selection, nonlinear-window comparison gates,
compact nonlinear startup-window finite-difference plumbing audits, mode-21
VMEC/Boozer geometry parity, reduced differentiable stellarator ITG
optimization examples, and linear/quasilinear VMEC/Boozer AD-vs-finite-
difference gradient gates on QH and Li383. The required release CI stack is the
quick-shard, docs/packaging, mypy, repo-hygiene, fast-coverage, and
wide-coverage matrix; treat the latest ``main`` run as the source of truth before
tagging. The frozen ``benchmarks/references/gkx_1_7_release_contract.json``
records five active release lanes as closed and two lanes as explicitly deferred: W7-X zonal recurrence/damping and TEM / kinetic-electron stellarator
extension.

The broader plan is not fully closed. The current quasilinear figures are
publication-ready as diagnostics, model-selection evidence, and explicit
negative promotion gates, but they do not support a calibrated absolute-flux
predictor. The stellarator optimization figures are publication-ready for
reduced differentiable optimization/UQ plumbing and gradient validation, but
not yet for broad production nonlinear heat-flux optimization; the selected QA
optimized-equilibrium audit is a bounded positive audit, not a broad turbulent
optimization result. Those stronger claims require
converged post-transient nonlinear heat-flux windows, VMEC/Boozer nonlinear
turbulence-gradient or robust finite-difference gates, local-gradient
conditioning, and nonlinear audits of additional optimized equilibria. W7-X
zonal recurrence and TEM/kinetic-electron stellarator validation remain
deferred from the current manuscript scope.

The latest manuscript-stack additions are deliberately contract-level figures:
``docs/_static/quasilinear_holdout_gap_report.png`` with CSV/JSON/PDF
companions states exactly why absolute-flux promotion remains blocked;
``docs/_static/vmec_boozer_aggregate_objective_gate.png`` together with
``docs/_static/vmec_boozer_reduced_portfolio_guard.json`` validates the
aggregate reducer on real VMEC/Boozer rows and checks provenance, sample
coverage, objective columns, and AD/FD diagnostics; and
``docs/_static/parallel_decomposition_status.png`` keeps production
independent-work parallelization claims separated from diagnostic nonlinear
whole-state/domain sharding metadata. The newer
``docs/_static/nonlinear_gradient_state_control_runbook.png`` is a claim
guardrail rather than a physics result: it shows that the QL-seeded
``Rsin_mid_surface_m1`` and ``Zcos_mid_surface_m1`` controls must be mapped to
perturbable VMEC input directions before nonlinear-gradient launches. The
newest ``LASYM=true`` ``RBS/ZBC`` response artifact provides that mapping with
rank ``2`` and condition number about ``1.02``, so the runbook now passes for
checked short-bracket launches. The older
``docs/_static/nonlinear_gradient_state_to_input_mapping_response.png`` is a
negative measured-response figure: the current stellarator-symmetric
``RBC/ZBS`` input perturbations do not move those asymmetric ``Rsin/Zcos``
state controls. The companion
``docs/_static/nonlinear_gradient_asymmetric_state_to_input_mapping_response.png``
is the positive symmetry-compatible mapping figure. The new
``docs/_static/nonlinear_gradient_state_control_short_bracket_launch_status.png``
then records that the mapped-control VMEC launch decks solved normally and that
bounded nonlinear campaign manifests are prepared, without promoting nonlinear
transport-gradient evidence. The follow-up
``docs/_static/nonlinear_gradient_state_control_short_bracket_nonlinear_audit_status.png``
records the first actual bounded nonlinear audit: all runtime and ensemble
window gates pass, but both finite-difference gradients fail closed because the
``1e-3`` bracket response is unresolved and asymmetric.

The companion solved-boundary guardrail
``docs/_static/vmex_qa_transport_candidate_comparison.png`` is not a
promoted optimization result. It documents the VMEC-JAX/GKX objective
assembly and WOUT-writing path, while deliberately failing closed when the
transport-weight refinement degrades the solved WOUT profile-iota and
quasisymmetry margins. The refreshed builder also treats gates reconstructed
from ``history.json`` and ``wout_final.nc`` as advisory only; paper-facing
admission requires an authoritative final ``solved_wout_gate.json``. A future
solved-boundary optimization figure must pass this full solved-candidate gate
before launching long-window nonlinear transport audits.
For VMEC-JAX replay-sensitive runs, the figure artifact must also state which
WOUT is authoritative. If ``wout_final_rerun.nc`` is chosen, require the
separate rerun-WOUT aspect/iota/QS gate and use that WOUT in the transport
audit commands; do not mix optimizer-state geometry with rerun-state
transport metrics.

Current Vs Deferred Figure Inventory
------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 24 42 34

   * - Figure group
     - Current manuscript use
     - Deferred or blocked interpretation
   * - Benchmark atlas and nonlinear windows
     - Release atlas, nonlinear window statistics, eigenfunction overlays, and
       gate index support the scoped linear/nonlinear validation narrative.
     - ETG nonlinear pilots, TEM/KAW stress lanes, extra W7-X flux tubes, and
       non-indexed figures are not promoted release parity claims.
   * - Quasilinear diagnostics
     - Electrostatic spectra, shape gates, calibration provenance, negative
       simple-rule gates, and the ``spectral_envelope_ridge`` uncertainty panel
       support a model-selection result.
     - No runtime/TOML absolute-flux predictor, universal saturation law, or
       electromagnetic/KBM quasilinear calibration is promoted; absolute-flux runtime promotion remains blocked.
   * - Autodiff and VMEC/Boozer gradients
     - Inverse/UQ demos, zero-beta equal-arc parity, solver-ready gradients, and
       mode-21 QH/Li383 linear, quasilinear, and reduced nonlinear-window
       estimator gates are in scope.
     - Compact nonlinear FD audits are startup plumbing checks only; production
       nonlinear transport gradients and broader optimized-equilibrium audits
       beyond the selected QA candidate remain future gates.
   * - Performance and parallelization
     - Runtime/memory figures, independent ``k_y`` scan scaling,
       quasilinear/UQ ensemble scaling, and nonlinear RHS profiler artifacts are
       release-facing engineering evidence.
     - Production parallelization is the independent-work path. Whole-state
       nonlinear sharding and nonlinear domain sharding are identity/profiler
       evidence only unless the exact workload passes promotion gates.
   * - W7-X zonal and TEM guardrails
     - Open-research and TEM-status panels are useful guardrails for the paper
       plan and release notes.
     - W7-X long-window zonal recurrence, W7-X experimental fluctuation-spectrum
       validation, W7-X TEM/kinetic-electron nonlinear windows, and broad
       multi-flux-tube stellarator validation are deferred.

Core Validation Figures
-----------------------

.. list-table::
   :header-rows: 1

   * - Figure
     - Owning script
     - Status
     - Notes
   * - Linear benchmark master panel
     - ``tools/artifacts/make_benchmark_atlas.py``
     - Closed for the release atlas; paper-level extensions remain scoped
     - Cyclone ITG, ETG, KBM, W7-X, HSX, and shaped tokamak coverage are represented in ``docs/_static/benchmark_core_linear_atlas.png``. TEM/kinetic-electron branch parity and additional W7-X multi-flux-tube scans remain outside the current release claim.
   * - Eigenfunction validation panel
     - ``tools/artifacts/generate_linear_reference_overlays.py overlap-summary``, ``tools/artifacts/generate_linear_reference_overlays.py reference-overlay``, ``tools/artifacts/generate_linear_reference_overlays.py kbm``, and ``tools/artifacts/generate_linear_reference_overlays.py w7x``
     - Closed for KBM and W7-X raw overlays
     - first shipped overlap artifact is ``docs/_static/kbm_eigenfunction_overlap_summary.png``; frozen raw GX bundles now exist for KBM and W7-X under ``docs/_static/comparison/reference_modes/``. The closed KBM raw overlay is ``docs/_static/kbm_eigenfunction_reference_overlay_ky0p3000.png`` with overlap ``0.999985`` and relative ``L^2`` mismatch ``0.00721``. The closed W7-X raw overlay is ``docs/_static/w7x_eigenfunction_reference_overlay_ky0p3000.png`` with overlap ``0.9999999994`` and relative ``L^2`` mismatch ``3.33e-5`` against the finite GX ``t≈2`` raw-mode bundle. Both overlay generators write JSON gate reports with ``overlap >= 0.95`` and ``relative L^2 <= 0.25`` requirements. ``tools/comparison/compare_gx_kbm.py --branch-summary-json`` writes branch-continuity gate metadata for selected KBM scans, and ``tools/artifacts/build_linear_validation_artifacts.py kbm-branch`` refreshes the no-rerun tracked artifact ``docs/_static/kbm_branch_gate_summary.json`` from ``docs/_static/comparison/kbm_reference_candidates.csv``. The current continuity-first branch summary passes the strict adjacent growth/frequency jump and successive-overlap gates.
   * - Nonlinear transport panel
     - ``tools/comparison/make_reference_panels.py summary`` / ``tools/comparison/make_reference_panels.py publication``
     - Closed for release-window gates; stricter manuscript tightening remains open
     - Cyclone, Miller, KBM, W7-X, HSX with matched windows. Current component artifacts: ``docs/_static/nonlinear_cyclone_diag_compare_t400.png``, ``docs/_static/nonlinear_cyclone_miller_diag_compare_t122.png``, ``docs/_static/nonlinear_kbm_diag_compare_t400_stats.png``, ``docs/_static/nonlinear_w7x_diag_compare_t200.png``, ``docs/_static/hsx_nonlinear_compare_t50_true.png``. ``tools/comparison/compare_gx_nonlinear.py diagnostics --summary-json`` writes machine-readable mean-relative-mismatch gates for each plotted diagnostic with explicit transport-window bounds. The materialized release-window metadata are ``docs/_static/nonlinear_cyclone_gate_summary.json``, ``docs/_static/nonlinear_cyclone_miller_gate_summary.json``, ``docs/_static/nonlinear_kbm_gate_summary.json``, ``docs/_static/nonlinear_hsx_gate_summary.json``, and ``docs/_static/nonlinear_w7x_gate_summary.json``; all pass the current release gate. This is sufficient for the release validation atlas. Paper-level tightening remains open where case-specific references support narrower tolerances, and the older short Cyclone ``t=5`` diagnostic remains documented as an exploratory startup/resolved-spectrum audit, not a release gate.
   * - W7-X exact-state convention audit
     - ``tools/comparison/build_exact_state_audit.py run`` and ``tools/comparison/build_exact_state_audit.py report``
     - Closed
     - current artifact base: ``docs/_static/w7x_exact_state_audit.png`` with CSV/JSON/PDF companions. It compares W7-X nonlinear VMEC startup state, late geometry/field arrays, and re-evaluated scalar diagnostics directly against GX exact-state dumps. The maximum finite pointwise relative error is ``4.62e-5`` under the explicit ``1e-4`` convention gate, while scalar diagnostics are below ``1.8e-7``. This closes the geometry/diagnostic convention layer but does not close the separate W7-X zonal-response recurrence lane.
   * - Windowed-statistics summary
     - ``tools/artifacts/build_nonlinear_validation_panels.py window-statistics``
     - Closed for current release-window gates
     - current artifact base: ``docs/_static/nonlinear_window_statistics.png`` with CSV/JSON/PDF companions. It summarizes the per-diagnostic ``mean_rel_abs`` and ``max_rel_abs`` statistics from the frozen nonlinear GX comparison gate JSONs for Cyclone, Cyclone Miller, KBM, W7-X, and HSX. Exploratory/short-run diagnostics are explicitly excluded with ``gate_index_include=false``.
   * - Nonlinear startup-window finite-difference audit
     - ``tools/artifacts/build_nonlinear_window_fd_audit.py``
     - Closed only as compact startup plumbing; transport-average and gradient promotion open
     - current artifact base: ``docs/_static/nonlinear_window_fd_audit.png`` with CSV/JSON/PDF companions. It runs actual compact GKX nonlinear Cyclone startup windows at ``a/LTi = base +/- step`` plus a repeated base point, then checks finite outputs, repeatability, monotonic drive response, startup-window coefficient of variation, startup-window trend, and resolved central finite-difference response. The tracked response/base fraction is about ``0.111``. Its ``transport_average_gate`` is false because the run is too short for a post-transient running average; it is not a production nonlinear heat-flux, VMEC/Boozer nonlinear state-gradient, or optimized-equilibrium transport claim.
   * - VMEC/Boozer nonlinear startup finite-difference audit
     - ``tools/artifacts/build_vmec_boozer_nonlinear_window_fd_audit.py``
     - Closed only as VMEC/Boozer geometry-perturbed startup plumbing; transport-average and local-gradient promotion open
     - current artifact base: ``docs/_static/vmec_boozer_nonlinear_window_fd_audit.png`` with CSV/JSON/PDF companions. It starts from the real mode-21 ``vmex -> booz_xform_jax`` QH state bridge, writes perturbed sampled geometries to temporary NetCDF files, and runs compact nonlinear startup windows at ``Rcos_mid_surface_m1 = base +/- 1e-5`` plus a repeated base point. The gate checks finite outputs, deterministic repeatability, bounded startup-window coefficient of variation and trend, resolved geometry perturbation, and resolved central finite-difference response; response/base is about ``0.040``. Its ``transport_average_gate`` is false and the forward/backward response is asymmetric, so this is a startup observable-path audit rather than a promoted transport average, local nonlinear gradient, or optimized-equilibrium transport claim.

   * - Validation gate index
     - ``tools/release/check_validation_coverage_manifest.py gate-index``
     - Current release-gate audit with quasilinear model-selection deliberately open
     - current artifact base: ``docs/_static/validation_gate_index.png`` and ``docs/_static/validation_gate_index.json``. This is not a physics result by itself; it is the audit panel for materialized release-window gates. The current index records ``17/18`` passed: the shaped-pressure external-VMEC high-grid admission gate is included as a passed scoped holdout, while ``docs/_static/quasilinear_model_selection_status.json`` remains open because the required spectral-envelope candidate misses the strict transport-error gate and is not promoted as an absolute-flux predictor.
   * - Quasilinear spectrum panel
     - ``tools/artifacts/plot_quasilinear_diagnostics.py spectrum``
     - Electrostatic diagnostic closed; absolute-flux prediction not promoted
     - current artifact bases: ``docs/_static/quasilinear_cyclone_spectrum.png``, ``docs/_static/quasilinear_cyclone_miller_spectrum.png``, ``docs/_static/quasilinear_hsx_spectrum.png``, and ``docs/_static/quasilinear_w7x_spectrum.png`` with CSV/JSON/PDF companions. They show electrostatic linear weights and explicitly uncalibrated mixing-length outputs from ``examples/linear/axisymmetric/runtime_cyclone_quasilinear.toml``, ``examples/linear/axisymmetric/runtime_cyclone_miller_quasilinear.toml``, ``examples/linear/non-axisymmetric/runtime_hsx_linear_quasilinear.toml``, and ``examples/linear/non-axisymmetric/runtime_w7x_linear_quasilinear_vmec.toml``. Scan spectra use requested ``ky`` for the x-axis and retain signed selected-mode coordinates as ``mode_ky`` when applicable. W7-X was generated from an external VMEC benchmark file via ``W7X_VMEC_FILE``; the equilibrium itself is not shipped. Absolute saturated-flux claims remain open until a held-out nonlinear calibration report passes.
   * - Quasilinear calibration audit
     - ``tools/artifacts/plot_quasilinear_calibration.py report`` and ``tools/artifacts/plot_quasilinear_calibration.py``
     - Initial train/holdout artifact closed as a failed model-transfer gate
     - current artifact bases: ``docs/_static/quasilinear_cyclone_calibration_audit.png``, ``docs/_static/quasilinear_cyclone_miller_calibration_audit.png``, ``docs/_static/quasilinear_cyclone_miller_train_holdout.png``, ``docs/_static/quasilinear_hsx_train_holdout.png``, ``docs/_static/quasilinear_w7x_train_holdout.png``, the manuscript-facing combined panel ``docs/_static/quasilinear_stellarator_train_holdout.png``, and the input-provenance audit ``docs/_static/quasilinear_validated_calibration_inputs.png`` with JSON companions. The current one-constant train/holdout report fits the heat-flux scale on Cyclone and the external-VMEC ITERModel case, then scores ten held-out windows: Cyclone Miller, HSX, W7-X, D-shaped external VMEC, up-down asymmetric external VMEC, circular external VMEC, CTH-like external VMEC, shaped-pressure external VMEC, replicated QP external VMEC, and replicated Solovev external VMEC. The CTH-like and shaped-pressure rows are admitted only under explicit high-grid policies that exclude failed coarse-grid traces, while the QP and Solovev rows are matched to passed replicated nonlinear summary gates. The report intentionally remains ``calibration_dataset`` with ``passed = false`` because held-out errors exceed the ``0.35`` gate; the current holdout mean relative error is about ``6.49`` for the positive-growth mixing-length family. The input audit confirms that every current train/holdout nonlinear artifact maps to a passed nonlinear gate or to a scoped high-grid admission gate, while failed QH and older feasibility pilots remain excluded. This closes the NetCDF/CSV/high-grid-admission calibration machinery and provenance gate but not a calibrated absolute-flux claim.
   * - External-VMEC nonlinear holdouts
     - ``tools/release/check_vmec_boozer_gates.py high-grid-admission``
     - Solovev replicated holdout admitted as negative absolute-QL evidence; QH remains negative evidence
     - retained artifacts: ``docs/_static/external_vmec_holdouts/solovev_reference_repair_dt002_amp1em5_n48_t250/solovev_n48_t250_ensemble_gate.json`` and the CTH-like, shaped-pressure, and QH convergence gates. CTH-like and shaped-pressure pass only their explicitly scoped high-grid policies. QH fails the relaxed 20% ``n64/n80`` gate through ``t=700``. Solovev passes its seed/timestep ensemble with ``<Q_i>=1.409`` and relative spread ``0.1599``. These are converged holdout outcomes, not a launch plan or an absolute-flux promotion.
   * - Quasilinear promotion guardrail audit
     - ``tools/release/check_quasilinear_promotion_guardrails.py``
     - Fast metadata gate closed; nonlinear simulation validation remains delegated to the source gates
     - current artifact: ``docs/_static/quasilinear_promotion_guardrails.json``. It scans the train/holdout calibration reports, saturation-model reports, nonlinear input-validation blocks, promotion gates, claim-scope README/docs wording, the quasilinear lane in ``benchmarks/references/gkx_1_7_release_contract.json``, and the manuscript quasilinear model-development figure index. It requires finite nonlinear window means and standard deviations for train/holdout calibration points, explicit nonlinear and quasilinear artifact provenance, JSON sidecars for the tracked model-development figures, scoped non-absolute claim levels, explicit failed-baseline or blocker metadata, passed held-out gates before any ``calibrated_absolute_flux`` claim, and a frozen release-contract quasilinear lane that remains scoped as diagnostic/model-selection evidence rather than a runtime absolute-flux predictor. This is deliberately a wording and metadata guard, not a calibrated absolute-flux claim or a replacement for nonlinear convergence simulations.
   * - Release claim-scope ledger
     - ``docs/release_scope.rst``
     - Closed as documentation guardrail
     - This documentation page centralizes the current claim boundaries across validation, quasilinear model selection, differentiable geometry, parallelization, performance, and deferred W7-X/TEM lanes. It should be updated whenever a new artifact promotes or demotes a claim.
   * - Quasilinear spectrum-shape gate
     - ``tools/artifacts/plot_quasilinear_diagnostics.py shape-gate``
     - HSX, W7-X, and Cyclone Miller gates closed; Cyclone retained as a failed model gate
     - current artifact bases: ``docs/_static/quasilinear_hsx_spectrum_shape_gate.png``, ``docs/_static/quasilinear_w7x_spectrum_shape_gate.png``, ``docs/_static/quasilinear_cyclone_miller_spectrum_shape_gate.png``, and ``docs/_static/quasilinear_cyclone_spectrum_shape_gate.png`` with JSON/PDF companions. They compare normalized linear heat-flux-weight spectra against normalized nonlinear resolved ``HeatFlux_kyst`` spectra. HSX passes with ``TV≈0.112`` and cosine ``≈0.971``; W7-X passes with ``TV≈0.056`` and cosine ``≈0.992``; Cyclone Miller passes with ``TV≈0.094`` and cosine ``≈0.983``; Cyclone is kept as a failed gate with ``TV≈0.215`` and cosine ``≈0.896``. This supports spectrum-shape diagnostics while identifying a real saturation/window/model limitation before absolute saturated-flux claims. KBM is deferred from this gate because the current quasilinear diagnostic validates electrostatic channels only and the KBM lane is electromagnetic.
   * - Zonal-flow / GAM response panel
     - ``tools/artifacts/build_zonal_flow_artifacts.py`` (``response-csv``, ``response-output``, ``objective-gate``, ``miller-panel``, and ``collisional-zonal-dk`` modes), ``tools/artifacts/build_w7x_zonal_validation_artifacts.py response-panel``, ``tools/artifacts/build_w7x_zonal_reference_artifacts.py digitize``, ``tools/artifacts/build_w7x_zonal_reference_artifacts.py compare``, and ``tools/artifacts/build_w7x_zonal_validation_artifacts.py`` (``contract`` and ``state-convention`` modes)
     - Open
     - The complete P24/J10 Coulomb, original-Sugama, and improved-Sugama Figures 12--14 protocol is closed in ``docs/_static/collision_finite_wavelength_zonal_response.png`` with an exact JSON verdict and compact velocity-section companion: all traces reach ``t nu=30``, pass the Xiao residual and finite-wavelength tail-ordering gates, and reproduce the published velocity-section relationships. The broader stellarator-zonal panel remains open: it should combine shaped-tokamak Rosenbluth-Hinton-style residuals with W7-X residual/damping envelopes using one figure convention; use signed ``Phi_zonal_mode_kxt`` or case-specific signed line averages for publication claims and keep ``Phi2_zonal_t`` only as an intermediate cross-check. The current Merlo Case-III artifact is ``docs/_static/miller_zonal_response_pilot.png`` from the initial-density setup at ``Nz=32``, ``Nl=4``, ``Nm=24``, ``dt=0.005``, and ``t≈60``. With Rosenbluth-Hinton first-sample normalization it gives ``residual≈0.192`` against the Merlo et al. Figs. 12/16 read-off of about ``0.19``; a literature-faithful common fit window ``t≈30`` with separate positive/negative-extrema damping fits gives ``γ_GAM R0 / v_i≈-0.176`` against the paper-scale read-off near ``-0.17``; and Hilbert-phase frequency extraction on that same window gives ``ω_GAM R0 / v_i≈2.20`` against the paper-scale read-off near ``2.24``. A higher-moment audit lowers the recurrence ratio but over-damps the GAM, while weak hypercollision scans are effectively inert, so the frozen Merlo artifact remains on the current ``Nm=24`` baseline. The W7-X side now uses the potential initializer, signed line-average observable, paper-facing line-first normalization, and no hidden time-axis scaling. The tracked long-window W7-X artifact is ``docs/_static/w7x_zonal_response_panel.png`` with replayable traces in ``docs/_static/w7x_zonal_response_panel.traces.csv``; it reaches the digitized Fig. 11 windows, but ``docs/_static/w7x_zonal_reference_compare.json`` remains open because residuals fail at ``k_x rho_i=0.07``, ``0.10``, and ``0.30`` and the late envelopes are much larger than the digitized stella/GENE traces. ``docs/_static/w7x_zonal_contract_audit.png`` is now the paper-facing diagnostic panel for that open mismatch and is intentionally excluded from the release gate index. ``docs/_static/w7x_zonal_state_convention_audit.png`` closes the paper-facing state convention layer: the recovered Gaussian potential has relative ``L2`` error ``1.85e-6``, off-target spectral content is zero to reported precision, and the diagnostic helpers agree with manual line/volume reductions near ``2e-16``. ``docs/_static/w7x_zonal_moment_tail_audit.png``, ``docs/_static/w7x_zonal_closure_ladder_kx070.png``, ``docs/_static/w7x_zonal_recurrence_sweep_kx070.png``, and ``docs/_static/w7x_zonal_hypercollision_probe_kx070.png`` are companion open diagnostics; together they support a recurrence / moment-closure hypothesis and show that weak or constant closure can reduce velocity-space tails without closing the paper trace. The refreshed closure ladder now covers constant Hermite, ``k_z``-weighted Hermite, mixed Laguerre-Hermite, Laguerre-only, and isotropic hypercollision families at ``0.01`` and ``0.03``. The best mean trace error is the isotropic ``nu_hyper=0.01`` row at about ``0.2755`` versus baseline ``0.2861``, but its late-window standard-deviation ratio is about ``4.25`` versus baseline ``4.10``. Thus no bounded closure family improves trace error, late-envelope recurrence, and moment-tail metrics simultaneously. The W7-X generator exposes explicit hypercollision and Gaussian-width audit overrides so future closure probes are reproducible from the tracked tool. A newer high-moment four-wavelength audit under ``tools_out/zonal_response/w7x_publication_nl16_nm64_dt005_t100`` verifies finite signed traces to ``t≈100`` after restart-continuation fixes. The tracked W7-X TOML keeps ``gaussian_width=1`` because the benchmark source writes the initializer as ``exp[-(z-z0)^2]``; wider profiles and non-unit time scales are retained only as audits. The lane remains open pending a more physical W7-X damping/closure and velocity-space recurrence fix under the paper-facing convention.
   * - W7-X fluctuation-spectrum panel
     - ``tools/artifacts/plot_w7x_fluctuation_spectrum_panel.py`` and ``tools/artifacts/build_tem_validation_artifacts.py w7x-extension``
     - Initial simulation-spectrum diagnostic closed; TEM/multi-flux validation open
     - current artifact bases: ``docs/_static/w7x_fluctuation_spectrum_panel.png``, ``docs/_static/tem_branch_parity_audit.png``, and ``docs/_static/w7x_tem_extension_status.png`` with CSV/JSON/PDF companions. The fluctuation panel is regenerated from the gated W7-X nonlinear ``t≈200`` NetCDF artifact, requires the corresponding nonlinear gate summary to pass before plotting, and records ``gate_index_include=false`` because it is a diagnostic figure rather than an additional release gate. It shows normalized ``k_y`` spectra for ``|\phi|^2``, ``W_\phi``, and ``|Q_i|``, the time-averaged ``k_x``-``k_y`` fluctuation-power map, the signed heat-flux spectrum, and a windowed temporal spectrum for the dominant nonzonal and zonal traces. The TEM audit explicitly keeps TEM linear parity open: maximum absolute relative growth-rate mismatch is about ``4.25``, maximum absolute relative frequency mismatch is about ``3.3`` away from the near-zero reference denominator, and the frequency branch has Spearman coefficient about ``-0.986``. Because the TEM reference is a provisional literature digitization rather than a direct case dump, this artifact blocks broad W7-X/TEM validation claims without being a standalone tuning target. The extension-status panel also keeps W7-X multi-alpha/multi-surface scans and kinetic-electron nonlinear windows open. This closes the reproducible simulation-spectrum panel needed for the current manuscript stack but not broad W7-X/TEM validation.
   * - Velocity-space convergence panel
     - ``tools/artifacts/build_linear_validation_artifacts.py observed-order`` plus dedicated full convergence refresh script to add
     - Open
     - should follow observed-order convergence evidence and write an observed-order gate report through ``gkx.diagnostics.validation_gates.observed_order_gate_report`` so convergence rate and final-grid error are tracked explicitly. The current atlas summary already records a machine-readable high-vs-low Cyclone grid convergence gate for the tracked convergence tile. The CSV-backed Cyclone velocity-space artifact is ``docs/_static/cyclone_resolution_observed_order.png`` with metadata in ``docs/_static/cyclone_resolution_observed_order.json``; the current office/GPU ``ky=0.30`` sweep passes the strict pairwise-order and final-error gate.
   * - Stellarator validation panel
     - dedicated script to add
     - Open
     - W7-X multi-flux-tube + zonal-flow response + HSX summary as needed; add heavy-electron EM verification before realistic-electron EM claims
   * - Performance panel
     - existing performance tooling
     - Closed for release-level scoped claims
     - current artifact bases: ``docs/_static/runtime_memory_benchmark.png``, ``docs/_static/nonlinear_rhs_profile_miller.png``, ``docs/_static/nonlinear_rhs_profile_stellarator_runtime.png``, ``docs/_static/full_nonlinear_rhs_trace_summary.json``, ``docs/_static/full_nonlinear_rhs_trace_gpu_summary.json``, ``docs/_static/nonlinear_sharding_profile_office_gpu_benchmark_grid.json``, and ``docs/_static/nonlinear_device_z_pencil_transport_gpu2_observable_split_profile.json``. The release claim is limited to current cold/warm runtime accounting, CPU/GPU nonlinear RHS hot-path localization, W7-X/HSX runtime-mode stellarator smoke profiles, and fail-closed numerical-identity gates. Whole-state sharding fails identity on the benchmark grid, and the device-z pencil route remains below the two-GPU speedup gate, so neither is a production nonlinear domain-decomposition speedup claim.
   * - Parallelization identity gate
     - ``tools/artifacts/generate_parallel_identity_gate.py ky-scan``
     - Closed for independent Cyclone ``k_y`` batching
     - current artifact base: ``docs/_static/parallel_ky_scan_gate.png`` with CSV/JSON/PDF companions. This is a real linear-solver gate: it compares serial and fixed-shape ``k_y``-batched Cyclone scans, requires numerical identity for ``gamma`` and ``omega``, and reports batch speedup separately from the acceptance criterion.

Differentiable-Physics Figures
------------------------------

.. list-table::
   :header-rows: 1

   * - Figure
     - Owning script
     - Status
     - Notes
   * - Sensitivity-analysis figure
     - ``examples/theory_and_demos/quasilinear_implicit_sensitivity.py``
     - Initial quasilinear eigenbranch gate closed
     - current artifact base: ``docs/_static/quasilinear_implicit_sensitivity.png`` with JSON/PDF companions. It differentiates a tiny Cyclone linear-RHS quasilinear objective ``[gamma, omega, kperp_eff^2, Qhat_i, Q_i^ML]`` with respect to ``[a/Ln, a/LTi]`` using the implicit left/right eigenpair system and checks the result against nearest-branch central finite differences. This is a differentiability/conditioning gate, not an absolute nonlinear-flux validation.
   * - Solver-objective geometry-gradient gate
     - ``tools/artifacts/build_solver_objective_gradient_gate.py``, ``tools/artifacts/build_solver_objective_gradient_gate.py vmec-boozer frequency``, ``tools/artifacts/build_solver_objective_gradient_gate.py vmec-boozer quasilinear``, ``tools/artifacts/build_solver_objective_gradient_gate.py vmec-boozer nonlinear-window``, ``tools/artifacts/build_vmec_boozer_gradient_holdout_matrix.py``, ``tools/artifacts/build_vmec_boozer_aggregate_objective_gate.py multi-point``, ``tools/artifacts/build_vmec_boozer_aggregate_objective_gate.py line-search-comparison``, ``tools/artifacts/build_vmec_boozer_aggregate_holdout_gate.py alpha``, ``tools/artifacts/build_vmec_boozer_aggregate_holdout_gate.py surface``, ``tools/artifacts/build_vmec_boozer_aggregate_objective_gate.py second-equilibrium``, ``tools/release/check_vmec_boozer_gates.py aggregate-holdout``, ``tools/release/check_nonlinear_optimization_gates.py production-guard``, ``tools/artifacts/build_nonlinear_window_fd_audit.py``, and ``tools/artifacts/build_vmec_boozer_nonlinear_window_fd_audit.py``
     - Solver-ready linear-RHS gradient gate closed; mode-21 VMEC/Boozer state-to-solver eigenfrequency, quasilinear heat-flux-weight, and reduced nonlinear-window estimator gates closed for QH plus Li383; compact and VMEC/Boozer-perturbed nonlinear startup FD audits closed only as plumbing checks; multi-alpha reduced aggregate objective plumbing passes; selected optimized-equilibrium post-transient transport-window audit closed as one scoped matched audit; earlier aspect-6 projected transport-gradient candidate and strict rerun-WOUT top-12 QA candidate audited as negative long-window transfer results; production nonlinear promotion now has the optimized-equilibrium ensemble count closed and is scoped-promoted by three matched audits passing the explicit 2% late-window reduction policy
     - current artifact bases: ``docs/_static/solver_objective_gradient_gate.png``, ``docs/_static/vmec_boozer_solver_frequency_gradient_gate.png``, ``docs/_static/vmec_boozer_quasilinear_gradient_gate.png``, ``docs/_static/vmec_boozer_nonlinear_window_gradient_gate.png``, ``docs/_static/vmec_boozer_li383_nonlinear_window_gradient_gate.png``, ``docs/_static/vmec_boozer_gradient_holdout_matrix.png``, ``docs/_static/vmec_boozer_multi_point_objective_gate.png``, ``docs/_static/vmec_boozer_aggregate_line_search_comparison.png``, ``docs/_static/vmec_boozer_aggregate_alpha_holdout_gate.png``, ``docs/_static/vmec_boozer_aggregate_surface_holdout_gate.png``, ``docs/_static/vmec_boozer_second_equilibrium_aggregate_gate.png``, ``docs/_static/vmec_boozer_holdout_transport/vmec_boozer_qh_torflux078_alpha120_holdout_ensemble_gate.png``, ``docs/_static/vmec_boozer_aggregate_holdout_promotion_gate.json``, ``docs/_static/production_nonlinear_optimization_guard.png``, ``docs/_static/optimized_equilibrium_replicates/optimized_equilibrium_replicate_t700_ensemble_gate.png``, ``docs/_static/qa_projected_transport_step1e3_matched_comparison.png``, ``docs/_static/qa_projected_transport_step1e3_redesign_report.json``, ``docs/_static/strict_qa_top12_edge_prelaunch_gate.json``, ``docs/_static/strict_qa_top12_edge_matched_nonlinear_transport.png``, ``docs/_static/strict_qa_top12_edge_matched_nonlinear_transport.json``, ``docs/_static/strict_qa_top12_edge_redesign_report.json``, ``docs/_static/strict_qa_rerun_baseline_ensemble_gate.json``, ``docs/_static/strict_qa_top12_step1p25em3_candidate_ensemble_gate.json``, ``docs/_static/nonlinear_window_fd_audit.png``, and ``docs/_static/vmec_boozer_nonlinear_window_fd_audit.png`` with CSV/JSON/PDF companions where available. The first differentiates actual electrostatic linear-RHS eigenpair observables with respect to solver-ready geometry arrays using the implicit left/right eigenpair system and checks ``gamma``, ``omega``, ``<k_perp^2>``, linear heat/particle-flux weights, and a mixing-length heat-flux proxy against nearest-branch central finite differences. The VMEC/Boozer frequency gate starts from a real ``vmex`` state coefficient, maps through ``booz_xform_jax`` with ``mboz=nboz=21``, builds the GKX linear RHS, and verifies the eigenfrequency gradient. The VMEC/Boozer quasilinear gate uses a richer ``Nl=2, Nm=3`` moment basis and checks ``gamma``, ``omega``, ``<k_perp^2>``, ``Q_i`` weight, and ``gamma Q_i/kperp^2`` against finite differences. The nonlinear-window estimator gates feed those observables into a smooth RK2 late-window envelope and check heat-flux mean, coefficient of variation, and normalized trend gradients. The multi-equilibrium matrix repeats the frequency, quasilinear, and estimator gates on the tracked QH and Li383 fixtures. The multi-alpha aggregate artifact shows reduced quasilinear objective sensitivity across two field lines and two ``k_y`` samples, while the growth-vs-quasilinear comparison shows that the two reduced objectives can select different VMEC coefficient directions. The alpha-heldout and surface-heldout splits pass reduced generalization checks, Li383 passes the second-equilibrium aggregate finite-difference plus line-search gate, and the QH held-out VMEC/Boozer transport artifact closes the aggregate promotion gate. The production nonlinear optimization guard now includes D-shaped, circular, and QH VMEC/Boozer replicated long-window holdouts, the optimized-equilibrium ``t=[350,700]`` seed/timestep audit, and three accepted matched baseline-to-optimized audits under the explicit ``2%`` late-window reduction policy. The no-ESS-to-optimized QA/ESS audit gives ``18.4%`` reduction, and the two max-mode-5 projected-weight audits give ``2.68%`` and ``3.35%``. The earlier aspect-6 projected transport-gradient candidate also has passed baseline/candidate seed/timestep ensembles, but the matched comparison gives a relative reduction of ``-0.00585`` and is not promoted. The stricter rerun-WOUT top-12 QA edge candidate improves the 18-point reduced metric by ``2.29%`` and passes both long-window ensemble gates, but its matched ``t=[350,700]`` nonlinear comparison gives only ``0.58%`` reduction with uncertainty z-score ``0.20`` and is also not promoted; the prelaunch gate now records that this reduced margin would be blocked against the calibrated ``4%`` threshold, so the next blocker is predictive transfer margin. These negative transfers require a better-conditioned multi-surface, multi-alpha transport objective before another expensive nonlinear audit. The compact nonlinear FD audits run actual nonlinear windows and verify startup conditioning/response only; their transport-average gates are false because they do not discard a long transient or demonstrate running-mean convergence. A surface-stencil path is available for memory-bounded diagnostics; nonlinear turbulence-gradient and broader multi-surface optimization claims remain separate gates.
   * - True ``t=1500`` strict QA matched nonlinear audits
     - ``tools/release/check_nonlinear_transport_gates.py runtime-outputs``
     - Baseline, growth-objective, quasilinear-objective, and nonlinear-window-objective triplets admitted as robust long-window signals; all three transport-candidate optimization claims rejected by matched comparison
     - current artifact bases: ``docs/_static/vmec_qa_t1500_replicates/qa_baseline_scipy_t1500_ensemble_gate.png``, ``docs/_static/vmec_qa_t1500_replicates/growth_from_strict_baseline_t1500_ensemble_gate.png``, ``docs/_static/vmec_qa_t1500_replicates/quasilinear_from_strict_baseline_t1500_ensemble_gate.png``, ``docs/_static/vmec_qa_t1500_replicates/nonlinear_window_from_strict_baseline_t1500_ensemble_gate.png``, ``docs/_static/vmec_qa_t1500_baseline_to_growth_comparison.png``, ``docs/_static/vmec_qa_t1500_baseline_to_quasilinear_comparison.png``, and ``docs/_static/vmec_qa_t1500_baseline_to_nonlinear_window_comparison.png`` with JSON companions. All use true full-horizon ``t=1500`` office runs, two seeds plus one timestep variant, and the strict ``t=[1100,1500]`` averaging window. The strict QA baseline passes with ``<Q_i> = 11.580``, mean relative spread ``0.0381``, and combined SEM/mean ``0.0195``. The growth candidate passes as a trace with ``<Q_i> = 11.510`` but fails the matched reduction gate: ``0.60%`` relative reduction, ``z=0.26``, below the ``4%`` threshold. The quasilinear candidate passes as a trace with ``<Q_i> = 11.636`` but is slightly worse than baseline: ``-0.49%``, ``z=-0.19``. The nonlinear-window candidate also passes as a trace with ``<Q_i> = 11.609`` but is slightly worse than baseline: ``-0.25%``, ``z=-0.09``. These panels demonstrate saturated trace robustness and a negative optimization-transfer result for all strict-QA candidate rows, not a successful nonlinear turbulent-flux optimization.
   * - Inverse/UQ figure
     - ``examples/theory_and_demos/autodiff_inverse_growth.py``, ``examples/theory_and_demos/autodiff_inverse_twomode.py``, and ``tools/artifacts/plot_stellarator_optimization_uq.py``
     - Scoped inverse and UQ validation closed; global identifiability and production optimization claims remain scoped
     - current artifact bases: ``docs/_static/autodiff_inverse_growth.png``, ``docs/_static/autodiff_inverse_twomode.png``, and ``docs/_static/stellarator_itg_optimization_uq.png`` with JSON/PDF/CSV companions. The inverse examples check AD/finite-difference consistency and conditioning for one- and two-parameter reduced objectives, while the UQ panel reports local Gauss-Newton covariance, rank, and sensitivity-map diagnostics. These figures validate the differentiable workflow and uncertainty reporting, not global uniqueness or production nonlinear stellarator optimization.
   * - Solved VMEC-JAX QA geometry figure
     - upstream ``vmex`` ``QA_optimization.py`` workflow plus local panel
       stitch from solved-boundary and Boozer diagnostics
     - Solved-equilibrium geometry visual ready for README/docs baseline context
     - current artifact base:
       ``docs/_static/vmex_qa_solved_boundary_boozer_panel.png``. The
       figure compares the initial and optimized solved VMEC LCFS surfaces
       colored by ``|B|`` and the corresponding Boozer-LCFS ``|B|`` contours.
       This is the manuscript-facing geometry visual for the QA baseline. It
       is not a nonlinear heat-flux optimization claim and should not be
       conflated with the reduced synthetic max-mode-1 optimization panels.
   * - Development-only optimization-plumbing figure
     - ``examples/theory_and_demos/reduced_stellarator_itg/compare_stellarator_itg_optimizations.py`` and ``tools/artifacts/plot_stellarator_optimization_uq.py``
     - Initial differentiable objective-reduction and weighted-residual UQ gates closed for development diagnostics; full VMEC/Boozer/GK optimization open
     - primary artifact base: ``docs/_static/stellarator_itg_optimization_uq.png`` with JSON sidecars and individual diagnostic panels ``docs/_static/stellarator_itg_growth_optimization.png``, ``docs/_static/stellarator_itg_quasilinear_optimization.png``, and ``docs/_static/stellarator_itg_nonlinear_optimization.png``. The supporting reduced comparison sidecar ``docs/_static/stellarator_itg_optimization_comparison.json`` records objective histories and reduction ratios, but its companion PNG is a synthetic reduced max-mode-1 surface diagnostic and is not a solved-geometry optimization figure. These files live under ``examples/theory_and_demos/reduced_stellarator_itg`` rather than ``examples/optimization``. The UQ panel computes covariance from the final weighted objective residual Jacobian and shows derivative parity, control uncertainty, covariance projection, and rank/conditioning diagnostics. The production QA optimization examples are the VMEC-JAX-style scripts in ``examples/optimization``; they remain separate from this diagnostic artifact stack and still require solved-WOUT gates plus converged nonlinear audits before transport-optimization claims.

Quasilinear Model-Selection Figures
-----------------------------------

These quasilinear model-development figures are retained as static evidence;
their generator scripts were retired, but the compressed panels and their
JSON/CSV companions remain tracked. Each documents a scoped negative or
claim-boundary result, not a runtime absolute-flux predictor:

- ``docs/_static/quasilinear_saturation_rule_sweep.png`` (with JSON companion):
  simple-rule saturation sweep on two training cases and ten held-out windows;
  positive-growth mixing length, linear heat-flux weight, and absolute-growth
  all fail the held-out absolute-flux gate.
- ``docs/_static/quasilinear_shape_aware_saturation.png`` (with JSON companion):
  shared spectrum-shape exponent model, a manuscript-facing negative result
  ruling out a one-exponent envelope.
- ``docs/_static/quasilinear_candidate_uncertainty.png`` (with JSON companion):
  leave-one-geometry-out candidate scoring with training-residual prediction
  intervals; the best candidate ``spectral_envelope_ridge`` misses the 0.35
  transport gate.
- ``docs/_static/quasilinear_candidate_regularization_sweep.png`` (with CSV/JSON
  companions): ridge-penalty sweep confirming no tested penalty passes the gate.
- ``docs/_static/quasilinear_dataset_sufficiency.png`` (with JSON companion):
  audit of the validated nonlinear windows behind the candidate models.
- ``docs/_static/quasilinear_model_selection_status.png`` (with CSV/JSON
  companion): consolidated claim-boundary panel, ``passed=false`` with the
  dataset-sufficiency, uncertainty, and transport-error blockers.
- ``docs/_static/quasilinear_stellarator_usefulness.png`` (with CSV/JSON
  companions): stellarator usefulness/limitation summary across HSX, W7-X,
  CTH-like, and shaped-pressure windows.
- ``docs/_static/quasilinear_screening_skill.png`` (with CSV/JSON companions):
  rank/correlation screening skill; no model passes the full-portfolio and
  held-out rank gates.

Caption Policy
--------------

Every manuscript-facing figure should answer these questions directly in the
caption:

1. what case and model are shown,
2. what horizon or fit window is used,
3. what the reference is,
4. what agreement was expected,
5. what agreement was found.

Open Issues Before Drafting
---------------------------

- close the W7-X zonal-flow residual and late-envelope mismatch under the
  paper-facing line-first normalization; current time coverage is sufficient,
  but residuals fail at three wavelengths and late envelopes remain too large
- either close the long-time recurrence follow-up for the shaped-tokamak
  Rosenbluth-Hinton / GAM response benchmark or explicitly scope publication
  claims to the benchmark-scale pre-recurrence metrics now frozen in the
  Merlo Case-III artifact
- extend the W7-X fluctuation-spectrum diagnostic to a Doppler-reflectometry
  transfer-function comparison if experimental-facing claims enter the paper
- extend eigenfunction-overlap metrics beyond the closed KBM and W7-X raw
  overlays where additional literature-backed mode-shape references are useful
- tighten the current windowed nonlinear statistics panel with stricter case-specific gates where reference windows justify them
- tie ETG nonlinear claims to the benchmark literature or keep them framed as a pilot
- add or refine publication-ready zonal-flow closure figures before making W7-X
  recurrence claims
- add an experimental transfer-function fluctuation-spectrum panel only if
  experimental-facing W7-X claims enter the paper
