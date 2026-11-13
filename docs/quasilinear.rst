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

Scope
-----

Quasilinear outputs are linear diagnostics and model-dependent optimization
proxies. Absolute flux requires nonlinear calibration and independent holdout
validation in the intended regime. Assess uncertainty coverage and ranking
on held-out geometries before using a reduced model for optimization.

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

Compare serial and worker runs with the same points and verify ordered
identity of both linear weights and saturated estimates.

The Miller companion spectrum uses ``ky`` values resolved by the nonlinear
run's ``Ny = 64`` grid:

.. code-block:: bash

   gkx scan \
     --config benchmarks/cases/cyclone_miller_quasilinear.toml \
     --ky-values 0.1,0.2,0.3,0.4,0.5 \
     --quasilinear \
     --out tools_out/miller_spectrum_scan

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
     - :mod:`scripts.checks._gates.quasilinear_calibration`
     - train/holdout/audit schemas, spectrum integration, nonlinear-window
       ingestion, scale fitting and scoring
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

The research path of ``examples/08_quasilinear/run.py`` applies it to
a small Cyclone linear-RHS fixture. The differentiated observable is
:math:`[\gamma, \omega, k_{\perp,\mathrm{eff}}^2, \widehat{Q}_i,
Q_i^{(\mathrm{ML})}]`, where :math:`Q_i^{(\mathrm{ML})}` is the uncalibrated
mixing-length heat-flux proxy, with respect to ``[a/L_n, a/L_Ti]``. Compare with central finite differences that follow the nearest isolated
branch, and inspect the eigenvalue gap before interpreting the derivative.

.. code-block:: bash

   python examples/08_quasilinear/run.py

Tests
-----

The fast suite covers:

* TOML and command-line plumbing for ``[quasilinear]``;
* phase and amplitude invariance of the linear weights;
* rejection of electromagnetic channels;
* summary and species artifact serialization, and ordered worker scans;
* AD-versus-finite-difference checks of ``quasilinear_feature_objective`` and
  of isolated-eigenvalue and implicit eigenpair sensitivities on small dense
  GKX linear-RHS fixtures.

Calibration reports
-------------------

Calibration artifacts use :mod:`scripts.checks._gates.quasilinear_calibration`, so
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

Window metadata comes from ``scripts.checks._gates.transport_windows`` and records the
transient cutoff, late-window mean and standard deviation, running-mean drift,
block-bootstrap SEM, sample counts and source provenance. Replicated windows are
combined with ``scripts.checks._gates.transport_windows.nonlinear_window_ensemble_report``,
which requires every input window to be promotion-ready and checks the relative
spread of the late-window means and the combined SEM.

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

Keep generated calibration reports and spectra outside version control.
Separate training, holdout and audit points and record the selected saturation
rule, spectrum integration weights and nonlinear diagnostic convention.
A fitted multiplier does not establish the physical validity of a saturation
rule. Require resolution checks and independent geometry holdouts before
reporting absolute flux accuracy.
