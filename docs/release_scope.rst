Release Scope and Claim Boundaries
==================================

.. note::

   Snapshot: GKX 2.3.0, 2026-09-21. The status of every public number is its
   row in the evidence ledger, shown on :doc:`verification_matrix`. A claim
   that is neither in the ledger nor on this page is not made.

   The 2.0.0 end-damping regression
   (`issue 192 <https://github.com/uwplasma/GKX/issues/192>`_) is fixed by
   `PR 197 <https://github.com/uwplasma/GKX/pull/197>`_, which restores the
   per-step contract and reproduces the recorded artifact bit-identically.
   Numbers recorded with 2.0.0 time integration still need repaired-build
   evidence before they are cited, and the GX references behind the
   provisional linear rows have not yet been regenerated with the per-unit-time
   end-damping rate (`issue 194 <https://github.com/uwplasma/GKX/issues/194>`_).

Claim scope
-----------

.. list-table::
   :header-rows: 1
   :widths: 24 18 58

   * - Lane
     - Status
     - Supported claim
   * - Linear benchmarks
     - per ledger row
     - KAW (analytic), ETG and W7-X (self-run GX) are ``passing``. Cyclone
       s-alpha, Cyclone Miller and KBM are ``provisional``: their GX goldens
       were produced above the ``dampEnds_linked`` launch cap and leave the top
       Hermite moments without end damping. HSX is ``provisional`` until the
       GX run arguments are recorded. TEM and Miller kinetic electrons are not
       validated.
   * - Nonlinear benchmarks
     - window comparison only
     - Cyclone, Cyclone Miller, KBM, W7-X and HSX pass a ``0.10`` mean-relative
       window comparison against self-run GX (``docs/_static/nonlinear_*_gate_summary.json``).
       This is not a statistical validation of saturated transport. ETG
       nonlinear pilots and KAW/TEM stress lanes are outside it.
   * - Quasilinear diagnostics
     - diagnostics only
     - Electrostatic linear heat/particle weights, spectra and model-selection
       artifacts are reproducible. The 12-case train/holdout report rejects the
       one-constant absolute-flux family, and simple one-scalar saturation
       rules are rejected. ``spectral_envelope_ridge`` fits the declared core
       portfolio with the Solovev and shaped-pressure stress outliers outside
       the scoped claim: core mean relative error about ``0.280``, held-out
       core error about ``0.275``, interval coverage ``10/10``. It is retained
       as a scoped model-development and optimization-screening result. The
       full 12-case predictor fails the stress cases and the rank/correlation
       gates. No runtime/TOML absolute-flux predictor, universal nonlinear
       transport model or user-facing saturation law is promoted.
       Electromagnetic quasilinear normalization and KBM calibration are not
       done.
   * - Differentiable geometry
     - equal-arc parity and reduced QH/Li383 gradients
     - ``vmex -> booz_xform_jax -> GKX`` matches equal-arc field-line
       geometry at ``mboz = nboz = 21`` for the QH, fixed-resolution QI and
       shaped-pressure finite-beta rows of
       ``docs/_static/vmec_boozer_parity_matrix.json`` (QI drift mismatch
       ``7.13e-2`` against an ``8e-2`` tolerance). Reduced frequency,
       quasilinear and nonlinear-window-estimator gradients pass
       AD/finite-difference gates on QH and Li383 (largest relative mismatch
       about ``2.7e-2``) and on the shaped-pressure case (``6.4e-11`` and
       ``2.1e-4``). Nonlinear finite-difference audits of the startup window
       have ``transport_average_gate = false``; they are plumbing checks.
   * - Stellarator optimization examples
     - reduced objectives; nonlinear transport open
     - The examples demonstrate differentiable reduced ITG objectives, UQ and
       AD/finite-difference checks. Every nonlinear heat-flux optimization
       audit so far is negative or unresolved (see below); production
       optimization is not promoted.
   * - Parallelization
     - production for independent work
     - Independent ``k_y`` scans, quasilinear spectra, sensitivity batches and
       UQ ensembles preserve serial results and have scaling artifacts. Runtime
       scan TOMLs use ``[parallel] strategy = "batch"`` with ``axis = "ky"``.
       Whole-state nonlinear sharding is an identity/profiler gate only.
   * - Performance
     - profiler evidence
     - Runtime/memory panels, RHS profiles and sharding identity checks are
       tracked. No nonlinear multi-GPU speedup or domain-decomposition claim is
       made.

Explicitly unpromoted claims
----------------------------

- universal or user-facing absolute quasilinear flux prediction, including
  ``spectral_envelope_ridge`` as a runtime or TOML saturation option;
- electromagnetic quasilinear transport calibration for KBM;
- nonlinear heat-flux stellarator optimization, and production nonlinear
  optimization without auditable raw post-transient runs, per-trace
  stationarity, autocorrelation-aware uncertainty, and resolution and spectral
  convergence;
- converged nonlinear transport gradients through ``vmex`` and
  ``booz_xform_jax``;
- startup finite-difference audits or reduced nonlinear-window estimators used
  as saturated transport averages or optimized-equilibrium audit bars;
- multi-surface, multi-alpha or multi-``k_y`` stellarator optimization from the
  current single-fixture objective evidence;
- W7-X validation beyond the tracked single-flux-tube ITG windows, and QI
  validation beyond the fixed-resolution mode-21 equal-arc parity row;
- W7-X TEM / kinetic-electron validation;
- W7-X long-window zonal recurrence/damping closure;
- nonlinear multi-GPU speedup from whole-state sharding, and FFT-axis
  nonlinear domain decomposition;
- treating refactor or test coverage as physics validation.

Negative and blocking evidence
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Artifact (``docs/_static/``)
     - Result
   * - ``quasilinear_stellarator_train_holdout_report.json``
     - one-constant absolute-flux model ``passed = false``; held-out mean
       relative error about ``6.49``
   * - ``quasilinear_saturation_rule_sweep.json``
     - no simple rule accepted; best (linear-weight fit) held-out error about
       ``4.42``, mixing-length ``6.49``, training-mean null ``1.80``
   * - ``quasilinear_candidate_uncertainty.json``
     - ``spectral_envelope_ridge``: full-ledger mean relative error about
       ``0.697``, coverage ``11/12``
   * - ``quasilinear_candidate_regularization_sweep.json``
     - best ridge penalty ``lambda = 0.5``: mean relative error ``0.689``,
       held-out ``0.764``; none accepted
   * - ``quasilinear_screening_skill.json``
     - Spearman ``0.636``/``0.624`` and pairwise order accuracy
       ``0.697``/``0.689`` (full/held-out), below the ``0.75`` gates
   * - ``quasilinear_holdout_gap_report.json``
     - absolute train/holdout error ``6.49`` against the ``0.35`` gate
   * - ``production_nonlinear_optimization_guard.json``
     - release safety passes (reduced estimators blocked; D-shaped, circular
       and QH VMEC holdout ensembles pass); production optimization is not
       promoted because optimized and matched summaries lack hashed raw
       sources and per-trace stationarity, autocorrelation, resolution and
       spectral gates
   * - ``nonlinear_turbulence_gradient_evidence_status.json``
     - no control passes: QA/ESS ``ZBS(1,0)`` 7.5% is resolved
       (``response_fraction = 0.0319``) and local (``fd_asymmetry_rel =
       0.044``) but fails spread (``0.196 > 0.15``) and propagated uncertainty
       (``1.81 > 0.5``); ``RBC(1,1)`` 3% fails uncertainty (``0.683``);
       ``ZBS(1,1)`` is nonlocal
   * - ``broad_nonlinear_transport_matrix_negative_evidence.json``
     - every candidate family of the broad matrix campaign fails
   * - ``nonlinear_sharding_profile_office_gpu_benchmark_grid.json``
     - whole-state sharding fails final-state identity and is slower than
       serial

Deferred lanes
--------------

- W7-X zonal long-window recurrence/damping closure under the paper-facing
  initializer and observable;
- W7-X multi-flux-tube, multi-surface and TEM / kinetic-electron validation
  (``docs/_static/w7x_tem_extension_status.json``);
- experimental W7-X fluctuation-spectrum comparison through a diagnostic
  transfer function. ``docs/_static/w7x_fluctuation_spectrum_panel.json`` is a
  simulation diagnostic only.

The frozen record of these decisions is
``benchmarks/references/gkx_1_7_release_contract.json``.
