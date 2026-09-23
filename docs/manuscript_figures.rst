:orphan:

Manuscript Figures
==================

This page indexes candidate figures for the GKX paper: the command that owns
each figure and the tracked JSON evidence behind it. PNG/PDF renders are not
tracked in git (except where stated) and are rebuilt by the owning command;
every quoted number comes from the JSON or CSV under ``docs/_static``. Claim
status follows ``tools/evidence_ledger.toml`` and :doc:`release_scope`.

A figure is ready for the manuscript when it has one owning script, one
tracked artifact, a declared reference, and a declared acceptance status. Its
caption states the case and model, the time window or fit window, the
reference, the expected agreement, and the agreement found.

Validation figures
------------------

.. list-table::
   :header-rows: 1
   :widths: 22 34 44

   * - Figure
     - Owning command
     - Tracked evidence and status
   * - Linear benchmark atlas
     - ``scripts/artifacts/make_benchmark_atlas.py``
     - Inputs indexed by ``tools/benchmark_atlas_manifest.toml``. Cyclone,
       ETG, KBM, W7-X, HSX, and shaped-tokamak panels; TEM parity is outside
       the release claim.
   * - Eigenfunction overlays
     - ``scripts/artifacts/generate_linear_reference_overlays.py kbm`` / ``w7x``
     - ``docs/_static/reference_modes/kbm_eigenfunction_reference_overlay_ky0p3000.json``
       (overlap 0.999985, relative :math:`L^2` 0.00721) and
       ``docs/_static/reference_modes/w7x_eigenfunction_reference_overlay_ky0p3000.json``
       (0.9999999994, 3.33e-5); gate overlap >= 0.95, :math:`L^2` <= 0.25.
   * - Nonlinear transport windows
     - ``scripts/comparison/compare_gx_nonlinear.py diagnostics --summary-json``;
       ``scripts/artifacts/build_nonlinear_validation_panels.py window-statistics``
     - ``docs/_static/nonlinear_{cyclone,cyclone_miller,kbm,hsx,w7x}_gate_summary.json``
       all pass; ``docs/_static/nonlinear_window_statistics.json``.
   * - Validation gate index
     - ``scripts/checks/check_validation_coverage_manifest.py gate-index``
     - ``docs/_static/validation_gate_index.json``: 17 of 18 pass;
       ``docs/_static/quasilinear_model_selection_status.json`` is open.
   * - W7-X exact-state convention audit
     - ``scripts/comparison/build_exact_state_audit.py report``
     - ``docs/_static/w7x_exact_state_audit.json``: max finite pointwise
       relative error 4.62e-5 under a 1e-4 gate. Closes the geometry and
       diagnostic convention layer only.
   * - Velocity-space convergence
     - ``scripts/artifacts/build_linear_validation_artifacts.py observed-order``
     - ``docs/_static/cyclone_resolution_observed_order.json``; passes.
   * - Merlo Case III zonal response
     - ``scripts/artifacts/build_zonal_flow_artifacts.py miller-panel``
     - ``docs/_static/miller_zonal_response_pilot.json``. Converged GKX values
       are gated; the literature comparison fails on residual (0.206 vs 0.19)
       and GAM frequency (2.345 vs 2.24), passes on damping. Scope claims to
       GKX's converged value and state the gap.
   * - W7-X zonal response
     - ``scripts/artifacts/build_w7x_zonal_validation_artifacts.py response-panel``;
       ``scripts/artifacts/build_w7x_zonal_reference_artifacts.py compare``
     - ``docs/_static/w7x_zonal_response_panel.json``,
       ``docs/_static/w7x_zonal_reference_compare.json``: open (residual fails
       at three of four wavelengths; late envelopes fail).
   * - Collisional finite-wavelength zonal response
     - ``scripts/artifacts/build_zonal_flow_artifacts.py collisional-zonal-dk``
     - ``docs/_static/collision_finite_wavelength_zonal_response.json``.
   * - W7-X fluctuation spectrum
     - ``scripts/artifacts/plot_w7x_fluctuation_spectrum_panel.py``
     - ``docs/_static/w7x_fluctuation_spectrum_panel.json``: simulation
       spectra from the gated W7-X nonlinear run; not a Doppler-reflectometry
       comparison.
   * - TEM branch audit
     - ``scripts/artifacts/build_tem_validation_artifacts.py``
     - ``docs/_static/tem_branch_parity_audit.json``: open (max relative growth
       error 4.25; frequency error 3.3 where the reference exceeds 0.2).
   * - Parallel identity gate
     - ``scripts/artifacts/generate_parallel_identity_gate.py ky-scan``
     - ``docs/_static/parallel_ky_scan_gate.json``: serial vs batched Cyclone
       :math:`k_y` scan, identity gated; speedup reported separately.
   * - Performance
     - see :doc:`performance`
     - ``docs/_static/runtime_memory_benchmark.png`` (tracked) and the profiler
       JSON listed there. Whole-state and device-z nonlinear sharding are not
       speedup claims.

Differentiable-physics figures
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 34 44

   * - Figure
     - Owning command
     - Tracked evidence and status
   * - Quasilinear implicit sensitivity
     - ``examples/theory_and_demos/quasilinear_implicit_sensitivity.py``
     - ``docs/_static/quasilinear_implicit_sensitivity.json``: implicit
       eigenpair derivatives against nearest-branch central differences.
   * - Solver-objective and VMEC/Boozer gradient gates
     - ``scripts/artifacts/build_solver_objective_gradient_gate.py``
     - ``docs/_static/solver_objective_gradient_gate.json`` and the
       ``docs/_static/vmec_boozer_*_gradient_gate.json`` family (QH and Li383).
   * - Nonlinear startup finite-difference audits
     - ``scripts/artifacts/build_nonlinear_window_fd_audit.py``;
       ``scripts/artifacts/build_vmec_boozer_nonlinear_window_fd_audit.py``
     - ``docs/_static/nonlinear_window_fd_audit.json``,
       ``docs/_static/vmec_boozer_nonlinear_window_fd_audit.json``: startup
       plumbing only; ``transport_average_gate`` is false.
   * - Inverse and UQ
     - ``examples/theory_and_demos/autodiff_inverse_growth.py``,
       ``examples/theory_and_demos/autodiff_inverse_twomode.py``,
       ``scripts/artifacts/plot_stellarator_optimization_uq.py``
     - ``docs/_static/autodiff_inverse_growth_summary.json``,
       ``docs/_static/stellarator_itg_optimization_uq.json``.
   * - Reduced stellarator ITG optimization
     - ``examples/theory_and_demos/reduced_stellarator_itg/compare_stellarator_itg_optimizations.py``
     - ``docs/_static/stellarator_itg_optimization_comparison.json``; see the
       note below.
   * - Solved-boundary QA candidate guardrail
     - ``examples/optimization``
     - ``docs/_static/vmex_qa_transport_candidate_comparison.json``: fails
       closed on solved-WOUT iota and quasisymmetry margins; not a promoted
       optimization result.

The reduced comparison sidecar records objective histories on a synthetic
max-mode-1 surface; its companion PNG
is not a solved-geometry optimization figure.
The production QA optimization examples are the VMEC-JAX-style scripts
in ``examples/optimization``; transport-optimization claims from them still
require solved-WOUT gates and converged nonlinear audits.

Quasilinear model-selection figures
-----------------------------------

These figures record scoped negative or claim-boundary results. The JSON files
are the artifacts of record; only the usefulness PNG is tracked.
There is no runtime/TOML absolute-flux predictor;
absolute-flux runtime promotion remains blocked.
The train/holdout report is not a calibrated absolute-flux claim:
held-out mean relative error is 6.49 against a 0.35 gate.

- ``docs/_static/quasilinear_stellarator_train_holdout.png``; JSON
  ``docs/_static/quasilinear_stellarator_train_holdout.json``: one-constant
  train/holdout calibration, ``passed = false``.
- ``docs/_static/quasilinear_saturation_rule_sweep.png``; JSON
  ``docs/_static/quasilinear_saturation_rule_sweep.json``: simple saturation
  rules all fail the held-out gate.
- ``docs/_static/quasilinear_shape_aware_saturation.png``; JSON
  ``docs/_static/quasilinear_shape_aware_saturation.json``: a shared
  spectrum-shape exponent is ruled out.
- ``docs/_static/quasilinear_candidate_uncertainty.png``; JSON
  ``docs/_static/quasilinear_candidate_uncertainty.json``: leave-one-geometry-out
  scoring; the best candidate, ``spectral_envelope_ridge``, misses the 0.35
  transport gate.
- ``docs/_static/quasilinear_candidate_regularization_sweep.png``; JSON
  ``docs/_static/quasilinear_candidate_regularization_sweep.json``: no tested
  ridge penalty passes.
- ``docs/_static/quasilinear_dataset_sufficiency.png``; JSON
  ``docs/_static/quasilinear_dataset_sufficiency.json``: audit of the
  nonlinear windows behind the candidates.
- ``docs/_static/quasilinear_model_selection_status.png``; JSON
  ``docs/_static/quasilinear_model_selection_status.json``: ``passed=false``
  with dataset-sufficiency, uncertainty, and transport-error blockers.
- ``docs/_static/quasilinear_stellarator_usefulness.png`` (tracked, with a
  JSON companion): stellarator usefulness and limitations across HSX, W7-X,
  CTH-like, and shaped-pressure windows.
- ``docs/_static/quasilinear_screening_skill.png``; JSON
  ``docs/_static/quasilinear_screening_skill.json``: no model passes the
  full-portfolio and held-out rank gates.
- ``docs/_static/quasilinear_holdout_gap_report.png``; JSON
  ``docs/_static/quasilinear_holdout_gap_report.json``: why absolute-flux
  promotion is blocked.

``scripts/checks/check_quasilinear_promotion_guardrails.py`` reads this list:
each PNG path must appear here with its JSON companion, and each JSON must
carry a scoped, non-absolute claim level.
