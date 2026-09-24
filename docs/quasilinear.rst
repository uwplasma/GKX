Quasilinear Transport
=====================

GKX computes quasilinear transport diagnostics from a linear eigenstate or a
late-time linear state. The linear diagnostic and the saturation model are kept
separate:

* **linear weights** are amplitude-normalized heat and particle fluxes computed
  with the same diagnostic kernels as runtime simulations;
* **saturation rules** are named, serialized model assumptions that convert a
  linear mode into a trend-level saturated estimate;
* **calibrated absolute flux** requires nonlinear training and holdout
  validation and cannot be inferred from an uncalibrated rule.

Status
------

The quasilinear layer is a scoped core-portfolio diagnostic and
optimization-screening tool. It is not a runtime/TOML absolute-flux predictor,
and the tracked calibration reports are not a calibrated absolute-flux claim.

* One-constant saturation rules fail the held-out transport gate on the 12-case
  portfolio (2 training, 10 held-out nonlinear windows): held-out mean relative
  error 6.49 for positive-growth mixing length, 4.42 for the raw linear weight,
  6.85 for absolute-growth mixing length, against a gate of 0.35.
* The best reduced candidate, ``spectral_envelope_ridge``, reaches
  leave-one-geometry-out mean relative error 0.697 with interval coverage
  11/12. It is not exposed as a runtime saturation law.
* On the declared 10-case core portfolio (Solovev and shaped-pressure external
  VMEC excluded as stress outliers), the same candidate passes the transport
  and coverage gate: mean relative error 0.280, held-out 0.275, maximum 0.575,
  coverage 10/10. The core rank-screening gate fails (Spearman 0.745 against
  0.75).

The numbers come from ``docs/_static/quasilinear_saturation_rule_sweep.json``,
``docs/_static/quasilinear_candidate_uncertainty.json`` and the
``core_portfolio_gate`` block of ``docs/_static/quasilinear_error_anatomy.json``.

Literature anchors
------------------

The separation of linear weights from a saturation rule follows nonlinear tests
of quasilinear transport [Waltz09]_, the QuaLiKiz derivation [Stephens21]_,
profile-evolution use [Citrin17]_ and the review [Staebler24]_. Parker et al.
[Parker23]_ compare saturation rules as model assumptions rather than
consequences of the linear solve. SAT3 [Dudding22]_ and SAT3-NN [Sar26]_ use
spectrum-aware, database-calibrated saturation instead of one mixing-length
constant. For stellarator optimization GKX uses quasilinear fluxes as research
diagnostics and optimization proxies, following [Jorge24]_.

Configuration and outputs
-------------------------

Only the electrostatic channel is implemented; ``channels`` other than
``"es"`` raise an error
(``gkx.diagnostics.quasilinear_transport.normalize_quasilinear_channels``).

.. code-block:: toml

   [quasilinear]
   enabled = true
   mode = "weights"                 # or "saturated"
   saturation_rule = "none"
   amplitude_normalization = "phi_rms"
   kperp_average = "phi_weighted"
   csat = 1.0
   gamma_floor = 0.0
   include_stable_modes = false
   channels = ["es"]

The diagnostic writes:

* ``*.quasilinear.summary.json``: growth rate, frequency, normalization,
  ``kperp_eff2``, species weights and saturation metadata;
* ``*.quasilinear_species.csv``: species-resolved heat and particle flux
  weights and, when requested, saturated estimates;
* ``*.quasilinear_spectrum.csv``: one row per ``ky`` for ``gkx scan`` runs.

In the spectrum file ``ky`` is the requested scan coordinate (used for ordering
and plotting) and ``mode_ky`` is the signed grid mode the linear solve selected.
Keeping both stops negative-branch aliases from reordering linked-boundary or
imported-geometry spectra.

Running it
----------

.. code-block:: bash

   gkx run \
     --config examples/08_quasilinear/case_full.toml \
     --out tools_out/cyclone_quasilinear

or enable the diagnostic for another linear TOML from the command line:

.. code-block:: bash

   gkx run \
     --config examples/01_linear_tokamak/case_full.toml \
     --quasilinear \
     --ql-mode saturated \
     --ql-saturation-rule mixing_length \
     --ql-normalization phi_rms \
     --ql-csat 1.0 \
     --out tools_out/cyclone_quasilinear

For a ``ky`` spectrum use ``gkx scan``. ``--workers`` runs the per-``ky``
solves in parallel and keeps the serial ordering of the spectrum:

.. code-block:: bash

   gkx scan \
     --config examples/08_quasilinear/case_full.toml \
     --ky-values 0.1,0.2,0.3,0.4 \
     --quasilinear \
     --workers 2 \
     --out tools_out/cyclone_quasilinear_scan

   # retired generator; restore it first: git show f005418bf575:scripts/artifacts/plot_quasilinear_diagnostics.py > scripts/artifacts/plot_quasilinear_diagnostics.py
   python scripts/artifacts/plot_quasilinear_diagnostics.py spectrum \
     --spectrum tools_out/cyclone_quasilinear_scan.quasilinear_spectrum.csv \
     --out tools_out/quasilinear_cyclone_spectrum.png

The serial/worker identity gate for this path
(``docs/_static/quasilinear_runtime_parallel_gate.json``) checks ordered
identity of the linear heat-flux weight and the saturated estimate; its timing
fields are engineering metadata, not a speedup claim:

.. code-block:: bash

   JAX_ENABLE_X64=1 python scripts/artifacts/generate_parallel_identity_gate.py quasilinear-runtime \
     --workers 2 --ky 0.1 0.2 \
     --out-prefix docs/_static/quasilinear_runtime_parallel_gate

The Miller companion spectrum uses ``ky`` values resolved by the nonlinear
run's ``Ny = 64`` grid:

.. code-block:: bash

   gkx scan \
     --config benchmarks/cases/cyclone_miller_quasilinear.toml \
     --ky-values 0.1,0.2,0.3,0.4,0.5 \
     --quasilinear \
     --out docs/_static/quasilinear_cyclone_miller_spectrum_scan

Model
-----

Linear eigenproblem
^^^^^^^^^^^^^^^^^^^

For a fixed flux tube and perpendicular mode the linear runtime solves the
matrix-free system

.. math::

   \frac{\partial G}{\partial t} = \mathcal{L}(\mathbf{p}) G,
   \qquad
   \mathcal{L} v_j = \lambda_j v_j,
   \qquad
   \lambda_j = \gamma_j - i\omega_j,

where ``G`` is the Hermite-Laguerre gyrocenter moment state, ``v_j`` a right
eigenvector, ``gamma`` the growth rate and ``omega`` the mode frequency
reported by the executable. The operator is assembled in
:mod:`gkx.terms.assembly` from the term modules under :mod:`gkx.terms`.

Field solve and linear weights
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fields are reconstructed from ``G`` with
``gkx.terms.assembly.compute_fields_cached``. The quasilinear diagnostic uses
``phi`` and sets ``A_parallel = B_parallel = 0``. The electrostatic heat-flux
weight contracts the radial :math:`E\times B` velocity with the
Hermite-Laguerre pressure moment:

.. math::

   v_{E,x,k} = i k_y \phi_k,
   \qquad
   \overline{p}_{s,k} =
   \sum_\ell \left(J_{\ell s}^{(\mathrm{fac})} G_{\ell,0,s,k}
   + \frac{1}{\sqrt{2}} J_{\ell s} G_{\ell,2,s,k}\right),
   \qquad
   Q_{s,k}^{(\mathrm{ES})} =
   \Re\left[v_{E,x,k}^* \overline{p}_{s,k}\right] W_k .

``J_{ls}`` are the Laguerre gyroaverage coefficients of species ``s`` and
``W_k`` includes the positive-``ky`` Hermitian factor, the dealias mask, the
flux-surface Jacobian and ``grad rho`` weight, species density and temperature
factors, and the diagnostic flux scale. The particle-flux weight uses the
density moment,

.. math::

   \overline{n}_{s,k} = \sum_\ell J_{\ell s} G_{\ell,0,s,k},
   \qquad
   \Gamma_{s,k}^{(\mathrm{ES})} =
   \Re\left[v_{E,x,k}^* \overline{n}_{s,k}\right] W_{\Gamma,k},

and is zero for one-ion adiabatic-electron cases. The kernels are
:func:`gkx.diagnostics.heat_flux_species`,
:func:`gkx.diagnostics.particle_flux_species` and
``_heat_flux_channel_contrib_species`` in :mod:`gkx.operators.moments`.

Amplitude normalization and effective scale
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

   k_{\perp,\mathrm{eff}}^2 =
   \frac{\langle k_\perp^2 |\phi|^2 \rangle}{\langle |\phi|^2 \rangle},
   \qquad
   \widehat{Q}_{s} =
   \frac{\sum_k Q_{s,k}^{(\mathrm{ES})}}{\mathcal{N}_\phi},
   \qquad
   \widehat{\Gamma}_{s} =
   \frac{\sum_k \Gamma_{s,k}^{(\mathrm{ES})}}{\mathcal{N}_\phi}.

The averages use the runtime spectral and flux-tube volume weights, so the
weights are invariant under eigenfunction phase rotation and amplitude
rescaling. The default normalization is

.. math::

   \mathcal{N}_\phi =
   \sum_{k_x,k_y,z} w_{k_x,k_y,z} |\phi_{k_x,k_y}(z)|^2,

with the weights of
:func:`gkx.diagnostics.quasilinear_transport.spectral_phi_weights`, whose
``(k_y, k_x)`` factor is :func:`gkx.core_ky_layout.hermitian_mode_weights`
(see :doc:`numerics`).

``amplitude_normalization`` accepts:

* ``phi_rms``: weighted ``|phi|^2`` average (default);
* ``phi_midplane``: maximum midplane ``|phi|^2``;
* ``field_energy``: electrostatic field energy.

Saturation rules
^^^^^^^^^^^^^^^^

``saturation_rule`` sets the squared amplitude :math:`A_k^2`; the saturated
fluxes are :math:`Q_{s,k}^{(\mathrm{sat})} = A_k^2 \widehat{Q}_{s,k}` and
:math:`\Gamma_{s,k}^{(\mathrm{sat})} = A_k^2 \widehat{\Gamma}_{s,k}`.

.. list-table::
   :header-rows: 1
   :widths: 30 34 36

   * - Rule
     - :math:`A_k^2`
     - Use
   * - ``none``
     - not computed
     - linear weights only
   * - ``mixing_length``
     - :math:`C_{\rm sat}\max(\gamma_k-\gamma_{\rm floor},0)/k_{\perp,{\rm eff},k}^2`
     - baseline rule
   * - ``lapillonne_2011``
     - same as ``mixing_length``
     - named alias kept for input compatibility
   * - ``linear_weight``
     - :math:`C_{\rm sat}`
     - tests whether the weight spectrum alone transfers across geometries
   * - ``absolute_growth_mixing_length``
     - :math:`C_{\rm sat}|\gamma_k|/k_{\perp,{\rm eff},k}^2`
     - stress test that gives stable branches nonzero intensity

``include_stable_modes = true`` drops the ``max(., 0)`` clamp. A rule returns
zero when ``kperp_eff2`` is non-positive or non-finite. These rules are
baselines for software validation and sensitivity studies; none is a
predictive absolute-flux model.

``gkx.diagnostics.quasilinear_transport.quasilinear_feature_objective`` exposes
the same rules on feature vectors ``[gamma, kperp_eff2, flux_weight]``; the unit
tests check its Jacobians against central finite differences.

Implementation map
------------------

.. list-table::
   :header-rows: 1

   * - Layer
     - Source
     - Responsibility
   * - Quasilinear weights
     - :mod:`gkx.diagnostics.quasilinear_transport`
     - ``k_perp`` scale, heat and particle weights, saturated outputs
   * - Diagnostic kernels
     - :mod:`gkx.diagnostics`, :mod:`gkx.operators.fluxes`
     - heat, particle and field-energy contractions shared by linear and
       nonlinear paths
   * - Runtime plumbing
     - :mod:`gkx.runtime`, :mod:`gkx.workflows.runtime.artifacts`
     - single-run and scan execution, command-line overrides, JSON/CSV writing
   * - Input schema
     - :mod:`gkx.config` (``RuntimeQuasilinearConfig``),
       :mod:`gkx.workflows.runtime.toml`
     - ``[quasilinear]`` parsing and round-trip serialization
   * - Calibration reports
     - :mod:`gkx.diagnostics.quasilinear_calibration`
     - train/holdout/audit schemas, spectrum integration, nonlinear-window
       ingestion, scale fitting and scoring
   * - Plotting tools
     - ``scripts/artifacts/plot_quasilinear_diagnostics.py`` (retired; :ref:`retired-generators`),
       ``scripts/artifacts/plot_quasilinear_calibration.py``
     - spectrum, shape-gate and calibration figures
   * - Differentiability gates
     - :mod:`gkx.objectives.autodiff_validation`
     - finite-difference checks, dense operator fixtures and implicit
       isolated-eigenpair sensitivities

Algorithm
---------

One linear mode:

.. code-block:: text

   build grid, geometry, species and linear cache
   solve the eigenproblem or fit the late-time linear state
   reconstruct phi; compute kperp_eff2 from |phi|^2 weights
   contract heat and particle fluxes with the runtime diagnostic kernels
   divide by the amplitude normalization; optionally apply a saturation rule
   write summary JSON and species CSV

A ``ky`` scan repeats this per requested ``ky``, selects the closest grid mode,
stores ``ky`` and ``mode_ky``, and writes ``*.scan.csv`` and
``*.quasilinear_spectrum.csv``.

Calibration against nonlinear windows:

.. code-block:: text

   load a nonlinear diagnostic CSV or NetCDF over a declared time window
   compute late-window convergence metadata after transient removal
   require finite mean, running-mean drift, block-bootstrap SEM and provenance
   integrate the linear quasilinear spectrum
   create train, holdout or audit points
   optionally fit one multiplicative scale on train points only
   score holdouts against the mean-relative-error and window gates

Differentiability
-----------------

Production linear solves are matrix-free. Dense matrices are built only in
small validation fixtures by
:func:`gkx.objectives.autodiff_validation.explicit_complex_operator_matrix`.
Eigenvalue sensitivities use JAX derivatives of the matrix entries and the
isolated-branch relation

.. math::

   \frac{\partial \lambda}{\partial p_i}
   =
   w^\dagger \frac{\partial \mathcal{L}}{\partial p_i} v,
   \qquad
   w^\dagger v = 1,

where ``v`` and ``w`` are right and left eigenvectors. Eigenfunction-dependent
observables use the implicit perturbation system

.. math::

   \begin{bmatrix}
   \mathcal{L} - \lambda I & -v \\
   w^\dagger & 0
   \end{bmatrix}
   \begin{bmatrix}
   \partial_i v \\
   \partial_i \lambda
   \end{bmatrix}
   =
   \begin{bmatrix}
   -(\partial_i \mathcal{L})v \\
   0
   \end{bmatrix},

with gauge ``w^\dagger \partial_i v = 0``, which makes the derivative unique for
phase-invariant observables. ``jnp.linalg.eig`` does not differentiate
non-Hermitian eigenvectors unless the caller opts in through
``lax_linalg.eig(..., enable_eigvec_derivs=True)`` (``jax >= 0.10.1``); the
implicit left/right path is the validation route for quasilinear observables.

``examples/08_quasilinear/implicit_sensitivity.py`` applies it to
a small Cyclone linear-RHS fixture. The differentiated observable is
:math:`[\gamma, \omega, k_{\perp,\mathrm{eff}}^2, \widehat{Q}_i,
Q_i^{(\mathrm{ML})}]`, where :math:`Q_i^{(\mathrm{ML})}` is the uncalibrated
mixing-length heat-flux proxy, with respect to ``[a/L_n, a/L_Ti]``. Against
central finite differences that follow the nearest isolated branch, the tracked
artifact ``docs/_static/quasilinear_implicit_sensitivity.json`` records maximum
relative derivative error 1.2e-2 at eigenvalue gap 0.24.

.. code-block:: bash

   python examples/08_quasilinear/implicit_sensitivity.py

Tests
-----

The fast suite covers:

* TOML and command-line plumbing for ``[quasilinear]``;
* phase and amplitude invariance of the linear weights;
* rejection of electromagnetic channels;
* summary and species artifact serialization, and ordered worker scans;
* AD-versus-finite-difference checks of ``quasilinear_feature_objective`` and
  of isolated-eigenvalue and implicit eigenpair sensitivities on small dense
  GKX linear-RHS fixtures;
* the promotion guardrail described below.

Calibration reports
-------------------

Calibration artifacts use :mod:`gkx.diagnostics.quasilinear_calibration`, so
train, holdout and audit points share one schema. A report is promoted to
``calibrated_absolute_flux`` only when it has at least one training point, at
least one holdout, passed finite late-window convergence metadata for every
holdout, and a passed holdout mean-relative-error gate. Otherwise it is demoted
to ``calibration_dataset`` or ``training_or_audit_only``.

The one-constant fit is

.. math::

   Q^{\rm QL}_{i}
   = C_{\rm sat} \sum_{k_y} A^2(k_y)\,\widehat Q_i(k_y)\,\Delta k_y,
   \qquad
   C_{\rm sat}
   = \frac{\sum_{j \in {\rm train}} q_j^{\rm raw} Q_j^{\rm NL}}
          {\sum_{j \in {\rm train}} (q_j^{\rm raw})^2},

where :math:`q_j^{\rm raw}` is the unscaled spectrum sum and
:math:`Q_j^{\rm NL}` the nonlinear window mean. The gate is

.. math::

   \left\langle
   \frac{|Q^{\rm pred} - Q^{\rm NL}|}{\max(|Q^{\rm NL}|, Q_{\rm floor})}
   \right\rangle_{\rm holdout}
   \le 0.35 .

Window metadata comes from ``gkx.diagnostics.transport_windows`` or
``scripts/checks/check_nonlinear_transport_gates.py convergence`` and records the
transient cutoff, late-window mean and standard deviation, running-mean drift,
block-bootstrap SEM, sample counts and source provenance. Replicated windows are
combined with ``gkx.diagnostics.transport_windows.nonlinear_window_ensemble_report``
(command line: ``check_nonlinear_transport_gates.py ensemble``), which requires
every input window to be promotion-ready and checks the relative spread of the
late-window means and the combined SEM. ``check_nonlinear_transport_gates.py
matched-windows`` compares two passed windows, for example with and without an
intervention, and reports the relative reduction with its quadrature-SEM
separation.

The report builder rejects points whose rule differs from the report's
``saturation_rule``, non-finite predictions or window means, negative window
standard deviations, missing spectrum columns, all-non-finite samples and
non-positive ``delta_ky``.

``calibration_point_from_nonlinear_window_summary`` converts a nonlinear window
summary into a calibration point. CSV inputs use the ``t`` column and the
selected heat-flux column (usually ``heat_flux``). NetCDF inputs use
``Grids/time`` and map ``heat_flux`` to ``Diagnostics/HeatFlux_st`` and
``heat_flux_es``/``heat_flux_apar``/``heat_flux_bpar`` to the matching
``Diagnostics/HeatFlux*_st`` variables. Species are summed by default; pass
``--species-index`` to the report builder for an ion-only or electron-only
target. The window is the summary's ``tmin``/``tmax`` when present, otherwise
the full finite range.

.. code-block:: bash

   python scripts/artifacts/plot_quasilinear_calibration.py report \
     --spectrum docs/_static/quasilinear_cyclone_spectrum_scan.quasilinear_spectrum.csv \
     --nonlinear-summary docs/_static/nonlinear_cyclone_gate_summary.json \
     --split audit \
     --case cyclone_long_window \
     --geometry cyclone \
     --electron-model adiabatic \
     --saturation-rule mixing_length \
     --out docs/_static/quasilinear_cyclone_calibration_audit_report.json

   python scripts/artifacts/plot_quasilinear_calibration.py \
     --report docs/_static/quasilinear_cyclone_calibration_audit_report.json \
     --out tools_out/quasilinear_cyclone_calibration_audit.png

With ``C_sat = 1`` the mixing-length rule underpredicts the Cyclone nonlinear
heat flux by orders of magnitude, so this audit report stays
``training_or_audit_only``.

Guardrails
^^^^^^^^^^

``scripts/checks/check_quasilinear_promotion_guardrails.py calibration-inputs``
matches each train/holdout point's ``nonlinear_artifact`` to tracked, passed
nonlinear gate metadata, so an exploratory or non-converged pilot cannot enter
calibration. ``scripts/checks/check_quasilinear_promotion_guardrails.py``
without a subcommand audits the calibration and model-selection JSON reports
(finite window statistics, train/holdout provenance, passed held-out gates
before ``calibrated_absolute_flux``), scans documentation for scope markers and
positive absolute-flux wording, and checks that each quasilinear figure in the
manuscript index has a JSON sidecar and a non-absolute claim level. Its output
is ``docs/_static/quasilinear_promotion_guardrails.json``.

Calibration portfolio
---------------------

The combined report ``docs/_static/quasilinear_stellarator_train_holdout_report.json``
fits on two training windows and scores ten held-out windows, all adiabatic
electron, electrostatic ITG. :math:`\langle Q_i\rangle` is the admitted
nonlinear window mean in gyro-Bohm units.

.. list-table::
   :header-rows: 1
   :widths: 24 10 40 12

   * - Case
     - Split
     - Admission
     - :math:`\langle Q_i\rangle`
   * - Cyclone s-alpha
     - train
     - long-window nonlinear gate
     - 6.67
   * - ITERModel external VMEC
     - train
     - ``n48``/``n64`` convergence at ``t = 350``
     - 22.05
   * - Cyclone Miller
     - holdout
     - long-window nonlinear gate
     - 4.26
   * - HSX (QHS deck)
     - holdout
     - ``t <= 50`` nonlinear gate
     - 5.09
   * - W7-X
     - holdout
     - ``t <= 200`` nonlinear gate
     - 5.38
   * - D-shaped external VMEC
     - holdout
     - ``n48``/``n64`` convergence at ``t = 250``
     - 18.48
   * - Up-down asymmetric external VMEC
     - holdout
     - ``n48``/``n64`` convergence at ``t = 450``
     - 7.76
   * - Circular external VMEC
     - holdout
     - ``n48``/``n64`` convergence at ``t = 450``
     - 19.43
   * - CTH-like external VMEC
     - holdout
     - ``n64``/``n80`` high-grid admission; ``n80`` seed/timestep ensemble on
       ``t=[350,700]`` (spread 0.041, SEM/mean 0.052)
     - 9.60
   * - Shaped-pressure external VMEC
     - holdout
     - ``n64``/``n80`` high-grid admission; ``n80`` ensemble on
       ``t=[325,650]`` (spread 0.094, SEM/mean 0.046)
     - 7.16
   * - QP ``nfp2`` external VMEC
     - holdout
     - replicated ``t = 250`` ensemble (spread 0.071)
     - 16.40
   * - Solovev external VMEC
     - holdout
     - replicated ``n48``/``t250`` ensemble, 20% spread tolerance (spread 0.160)
     - 1.41

CTH-like and shaped-pressure are admitted only under coarse-grid exclusion:
their full ``n48/n64/n80`` ladders fail (shaped-pressure grid shift 0.469
against 0.15), so neither is a full-ladder convergence claim. The admission
sidecars are
``docs/_static/external_vmec_cth_like_modified_high_grid_admission_gate.json``
and
``docs/_static/external_vmec_shaped_tokamak_pressure_dt0p04_high_grid_admission_gate.json``;
the ensemble gates are named in each point's ``nonlinear_artifact``.

Excluded windows and the reason for each are listed under
``excluded_candidates`` in ``docs/_static/quasilinear_holdout_gap_report.json``.
They include:

* KBM: electromagnetic, while the quasilinear channels are electrostatic;
* nfp4 QH external VMEC: grid shifts 0.523/0.480 (``n32``/``n48``) and
  0.630/0.704 (``n48``/``n64``) on the common/least-trending windows, above
  0.15;
* a same-family ITERModel ``t = 450`` audit: reproducibility evidence for the
  training reference, not an independent holdout.

The QI seed equilibrium (``quasilinear_vmec_qi_seed_linear_spectrum_scan``)
peaks at :math:`\gamma \approx 3.8\times10^{-3}` near ``ky = 0.143``, below the
0.02 nonlinear-launch threshold of ``scripts/artifacts/build_qi_branch_refinement_gate.py`` (retired; :ref:`retired-generators`),
so no nonlinear window exists for it.

Results
-------

.. list-table::
   :header-rows: 1
   :widths: 30 46 24

   * - Test
     - Result
     - Artifact
   * - One-constant transfer, Cyclone to Cyclone Miller
     - fitted ``C_sat = 3839.966``; Miller error far above 0.35;
       ``passed = false``
     - ``quasilinear_cyclone_miller_train_holdout_report.json``
   * - One-constant transfer, 12 cases
     - held-out mean relative error 6.49; worst holdout Solovev (predicted 53.2,
       observed 1.41)
     - ``quasilinear_stellarator_train_holdout_report.json``,
       ``quasilinear_holdout_gap_report.json``
   * - Saturation-rule sweep
     - linear weight 4.42, mixing length 6.49, absolute growth 6.85,
       training-mean null 1.80; no rule accepted
     - ``quasilinear_saturation_rule_sweep.json``
   * - Shape-aware power law
     - leave-one-geometry-out error 0.725 against linear weight 0.624 and
       training-mean null 0.170; ``promotion_gate.passed = false``
     - ``quasilinear_shape_aware_saturation.json``
   * - Candidate uncertainty
     - ``spectral_envelope_ridge`` 0.697, coverage 11/12; linear weight 1.320;
       null 1.171; ``linear_state_ridge`` 1.907; none accepted
     - ``quasilinear_candidate_uncertainty.json``
   * - Ridge-penalty sweep
     - best ``lambda = 0.5``: error 0.689, held-out 0.764, coverage 11/12; no
       penalty passes 0.35
     - ``quasilinear_candidate_regularization_sweep.json``
   * - Rank screening
     - ``spectral_envelope_ridge`` Spearman 0.636 (held-out 0.624), pairwise
       order 0.697 (held-out 0.689), held-out error 0.777; gates 0.75; no model
       passes
     - ``quasilinear_screening_skill.json``
   * - Residual anatomy
     - external axisymmetric VMEC cases carry 83% of the
       ``spectral_envelope_ridge`` residual; HSX and W7-X mean error 0.31
     - ``quasilinear_error_anatomy.json``
   * - Model selection
     - blockers ``dataset_sufficiency_passed``,
       ``candidate_uncertainty_passed``, ``required_candidate_accepted``,
       ``required_candidate_transport_error``
     - ``quasilinear_model_selection_status.json``

All artifacts are under ``docs/_static/``. Every model-development report
carries an ``input_validation`` block built from the nonlinear summary gates, so
it can only be regenerated from windows that passed those gates.

HSX and W7-X
^^^^^^^^^^^^

Every scanned HSX and W7-X branch in the tracked short electrostatic
adiabatic-electron spectra is stable, so with ``gamma_floor = 0`` the
mixing-length estimate is zero while the nonlinear windows are finite (relative
error one by construction). The raw linear-weight fit overpredicts both by
roughly a factor of four; ``spectral_envelope_ridge`` is closer
(``docs/_static/quasilinear_stellarator_usefulness.json``).

.. image:: _static/quasilinear_stellarator_usefulness.png
   :alt: Stellarator quasilinear usefulness summary
   :width: 100%

Stellarator saturated states involve subdominant and stable eigenmodes
[Pueschel16]_, energy transfer to damped modes [Hegna18]_ and zonal-flow
dynamics [Tiwari25]_, and linear growth rates are a weak proxy for saturated
heat flux across quasi-symmetric configurations [McKinney19]_. In GKX,
quasilinear metrics are used for screening, differentiable-optimization
research and model development, not for absolute stellarator heat flux.

The HSX input is the Nuhrenberg-Zille QHS deck generated by ``vmex``; the W7-X
command uses the shipped QI deck. For the benchmark equilibria, point
``[geometry].vmec_file`` at the machine-specific WOUT, which is not in Git.

.. code-block:: bash

   cd examples/vmec
   vmex input.NuhrenbergZille_1988_QHS
   vmex input.nfp3_QI_fixed_resolution_final
   cd ../..
   gkx scan \
     --config examples/02_linear_stellarator/case_full.toml \
     --ky-values 0.047619047619047616,0.09523809523809523,0.14285714285714285,0.19047619047619047,0.23809523809523808,0.2857142857142857 \
     --Nl 4 --Nm 8 --solver time --dt 0.005 --steps 400 \
     --quasilinear \
     --out docs/_static/quasilinear_hsx_spectrum_scan \
     --no-progress
   gkx scan \
     --config benchmarks/cases/w7x_linear_quasilinear_vmec.toml \
     --ky-values 0.047619047619047616,0.09523809523809523,0.14285714285714285,0.19047619047619047,0.23809523809523808,0.2857142857142857 \
     --Nl 4 --Nm 8 --solver time --dt 0.005 --steps 400 \
     --quasilinear \
     --out docs/_static/quasilinear_w7x_spectrum_scan \
     --no-progress

Each holdout is added to the Cyclone/Miller report with, for W7-X:

.. code-block:: bash

   python scripts/artifacts/plot_quasilinear_calibration.py report \
     --points docs/_static/quasilinear_cyclone_miller_train_holdout_points.json \
     --spectrum docs/_static/quasilinear_w7x_spectrum_scan.quasilinear_spectrum.csv \
     --nonlinear-summary docs/_static/nonlinear_w7x_gate_summary.json \
     --split holdout \
     --case w7x_nonlinear_window \
     --geometry w7x \
     --electron-model adiabatic \
     --fit-train-scale \
     --out docs/_static/quasilinear_w7x_train_holdout_report.json

Spectrum-shape gates
--------------------

A shape gate compares only the normalized ``ky`` distribution of the linear
heat-flux weight with the resolved nonlinear ``Diagnostics/HeatFlux_kyst``
spectrum; it does not test the absolute level. Pass criteria are total
variation ``TV <= 0.2`` and cosine similarity ``>= 0.95``.

.. list-table::
   :header-rows: 1

   * - Case
     - TV
     - cosine
     - Result
   * - W7-X
     - 0.056
     - 0.992
     - pass
   * - HSX
     - 0.11
     - 0.97
     - pass
   * - Cyclone Miller
     - 0.094
     - 0.983
     - pass
   * - Cyclone s-alpha (long window)
     - 0.215
     - 0.896
     - fail; mismatch in the low- and high-``ky`` tails

The values are in ``docs/_static/quasilinear_<case>_spectrum_shape_gate.json``.
KBM has no shape gate because its nonlinear lane is electromagnetic.

.. code-block:: bash

   # retired generator; restore it first: git show f005418bf575:scripts/artifacts/plot_quasilinear_diagnostics.py > scripts/artifacts/plot_quasilinear_diagnostics.py
   python scripts/artifacts/plot_quasilinear_diagnostics.py shape-gate \
     --spectrum docs/_static/quasilinear_hsx_spectrum_scan.quasilinear_spectrum.csv \
     --nonlinear tools_out/final_nonlinear_audit/hsx_nonlinear_t50.out.nc \
     --out tools_out/quasilinear_hsx_spectrum_shape_gate.png \
     --ql-column heat_flux_weight_total \
     --nonlinear-variable Diagnostics/HeatFlux_kyst \
     --time-max 49.2 \
     --tv-gate 0.2 \
     --cosine-gate 0.95

The nonlinear NetCDF inputs are local run outputs and are not tracked.

Convergence evidence
--------------------

Admission follows the practice of nonlinear benchmark papers: a saturated
heat-flux window is used only if it is robust to resolution and window choice
[Dimits00]_ [GX]_ [GonzalezJerez22]_. The 0.15 grid-shift threshold is of the
order of the heat-flux convergence tolerances reported for Laguerre-Hermite
calculations in [GX]_. Stellarator flux-tube domain choices change results
[Sanchez21]_, which is why external-VMEC cases sit behind explicit gates.
Turbulent flux traces are autocorrelated and need uncertainty and stopping
checks [Oberparleiter16]_, low velocity resolution can move the saturated heat
flux [Hoffmann23]_, and W7-X heat-flux time series show the same statistical
structure [Papadopoulos23]_.
