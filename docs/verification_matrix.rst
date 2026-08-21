Verification Matrix
===================

Purpose
-------

This page is the research-facing index of what GKX treats as verified,
validated, exploratory, or deferred. It is meant to answer four questions for
each lane:

1. what physical model is being exercised,
2. what observable is compared,
3. what the reference is,
4. what acceptance gate applies.

Literature Baselines Reviewed
-----------------------------

The current matrix is anchored on these published baselines:

- Tronko et al., *Verification of Gyrokinetic codes: theoretical background and applications*:
  verification methodology, observed-order checks, and benchmark-observable
  framing.
- Mandell et al., *GX: a GPU-native gyrokinetic turbulence code for tokamak and stellarator design*:
  CBC, W7-X, KBM, nonlinear transport, velocity-space convergence, and
  performance figure conventions.
- González-Jerez et al., *Electrostatic gyrokinetic simulations in W7-X geometry*:
  W7-X ITG/TEM scans, zonal-flow response, and nonlinear ITG heat flux.
- Nevins et al., *Characterizing electron temperature gradient turbulence*:
  ETG operating-point conventions.
- Monreal et al., *Residual zonal flows in tokamaks and stellarators at arbitrary wavelengths*:
  residual-zonal-flow metrics and damping interpretation.
- Merlo et al., *Linear multispecies gyrokinetic flux tube benchmarks in shaped
  tokamak plasmas*:
  shaping scans, ballooning-angle handling, Rosenbluth-Hinton residuals, and
  GAM damping.
- González-Jerez et al., *Electrostatic microturbulence in W7-X: comparison of
  local gyrokinetic simulations with Doppler reflectometry measurements*:
  fluctuation amplitudes, frequency spectra, and zonal-flow spectral content.
- Maurer et al., *Global electromagnetic turbulence simulations of W7-X-like
  plasmas with GENE-3D*:
  heavy-electron electromagnetic verification before realistic-mass stellarator
  production runs.

Status Legend
-------------

- ``Closed``: benchmark lane is accepted for research claims.
- ``Open``: lane is active and expected to close.
- ``Exploratory``: useful for development, not yet a paper claim.
- ``Deferred``: intentionally out of scope for the current paper/release.

Tokamak Linear
--------------

.. list-table::
   :header-rows: 1

   * - Lane
     - Observable
     - Reference
     - Status
     - Baseline gate
   * - Cyclone ITG
     - ``gamma(k_y)``, ``omega(k_y)``, eigenfunction overlap
     - GX + CBC literature
     - Closed
     - ``rtol <= 1e-2`` except documented low-``k_y`` / near-marginal cases
   * - ETG
     - ``gamma(k_y)``, ``omega(k_y)``
     - GX + ETG benchmark literature
     - Closed
     - ``rtol <= 1e-2`` on the tracked branch
   * - KBM
     - ``gamma(k_y)``, ``omega(k_y)``, branch continuity vs ``beta``
     - GX
     - Closed
     - ``rtol <= 1e-2`` on the accepted branch
   * - KAW
     - branch-followed linear response
     - GX
     - Deferred
     - close branch identity before publication use
   * - TEM
     - ``gamma(k_y)``, ``omega(k_y)``
     - GX / literature
     - Open
     - close branch-following and reference selection first
   * - Shaped multispecies tokamak
     - ``gamma``, ``omega``, eigenfunction shape
     - Sauter benchmark set
     - Open
     - literature-backed operating point and overlap gate required
   * - Shaped tokamak zonal-flow / GAM
     - residual level, damping rate, GAM envelope
     - Merlo et al. + analytical Rosenbluth-Hinton estimates where applicable
     - Open
     - residual and damping must match literature/code-backed references before publication use; signed ``Phi_zonal_mode_kxt`` is now available. The current artifact is ``docs/_static/miller_zonal_response_pilot.png`` from ``tools/artifacts/build_zonal_flow_artifacts.py miller-panel`` using Merlo Case-III Table-III parameters, an initial density perturbation, a common pre-recurrence fit window ``t≈30``, separate positive/negative-extrema damping fits, and a Hilbert-phase frequency extraction on that same window. It gives ``residual≈0.192`` against a paper-scale target of about ``0.19``, ``ω_GAM R0 / v_i≈2.20`` against a figure read-off near ``2.24``, and ``γ_GAM R0 / v_i≈-0.176`` against a figure read-off near ``-0.17``. The remaining explicit follow-up item is the later finite-moment recurrence rather than the benchmark-scale Merlo gate

Frozen artifact paths for the currently closed tokamak linear lanes:

- ``docs/_static/cyclone_comparison.png``
- ``docs/_static/etg_comparison.png``
- ``docs/_static/kbm_comparison.png``
- ``docs/_static/kbm_eigenfunction_overlap_summary.png``
- ``docs/_static/comparison/reference_modes/kbm_linear_reference_ky0p3000.npz``
- ``docs/_static/benchmark_core_linear_atlas.png``

Closed raw-overlay diagnostic artifacts for the KBM lane:

- ``docs/_static/reference_modes/kbm_linear_gkx_ky0p3000.csv``
- ``docs/_static/kbm_eigenfunction_reference_overlay_ky0p3000.png``
- ``docs/_static/reference_modes/kbm_eigenfunction_reference_overlay_ky0p3000.json``
- ``tools/artifacts/generate_linear_reference_overlays.py kbm``

The refreshed bounded-cost extraction produces normalized overlap
``0.999985`` and relative ``L^2`` mismatch ``0.00721`` against the frozen GX
raw mode at ``k_y \approx 0.3`` when run with the exact KBM grid contract, the
selected growth-fit window, and a late-time eigenfunction tail window. The
generator writes a machine-readable gate report with ``overlap >= 0.95`` and
``relative L^2 <= 0.25`` as the acceptance policy, and this raw-overlay
artifact now passes.

Branch-followed scan tables should use the same gate-report convention:
observed-order gates for resolution or velocity-space convergence, and
branch-continuity gates for adjacent ``gamma``/``omega`` jumps and successive
eigenfunction overlap when overlap data are available. The tracked KBM
candidate table now has a no-rerun summary path through
``tools/artifacts/build_linear_validation_artifacts.py kbm-branch`` and
``docs/_static/kbm_branch_gate_summary.json``. That summary now uses the
continuity-first selected branch and passes the strict checks:
``max_rel_gamma_jump ~= 0.388``, ``max_rel_omega_jump ~= 0.320``, and no
successive-overlap deficit.
The candidate table's ``selected`` rows are also the authoritative source for
the public fixed-beta comparison plot and mismatch table. This prevents older
pointwise-best extraction and low-``k_y`` checkpoint snapshots from being
mixed with the continuity-followed branch in release-facing artifacts.

Observed-order convergence tables should also gate both the asymptotic finest
refinement and the full set of pairwise refinement orders. The generic
``tools/artifacts/build_linear_validation_artifacts.py observed-order`` path now records this policy in JSON.
The tracked Cyclone velocity-space convergence artifact
``docs/_static/cyclone_resolution_observed_order.json`` is closed on an
office/GPU ``ky=0.30`` time-path sweep with all pairwise orders positive,
final-pair order above ``4.8``, and finest-grid relative growth-rate error
about ``1.1e-3``.

The current materialized gate reports are indexed by
``tools/release/check_validation_coverage_manifest.py gate-index`` in
``docs/_static/validation_gate_index.json`` and
``docs/_static/validation_gate_index.png``. Exploratory diagnostics can set
``gate_index_include=false`` so they remain documented but do not count as
release blockers. The current release-gate index has ``17/18`` tracked reports
passing. The sole open row is the quasilinear model-selection status, which is
intentionally fail-closed because the current spectral-envelope candidate is a
scoped model-development diagnostic rather than a promoted absolute-flux
predictor.

Stellarator Linear
------------------

.. list-table::
   :header-rows: 1

   * - Lane
     - Observable
     - Reference
     - Status
     - Baseline gate
   * - W7-X ITG flux tube
     - ``gamma(k_y)``, ``omega(k_y)``
     - stella/GENE benchmark paper + GX
     - Closed
     - ``rtol <= 1e-2`` on closed adiabatic-electron ITG branches. This row
       does not close W7-X TEM or kinetic-electron validation; those remain
       blocked by the explicit TEM branch audit below.
   * - W7-X TEM / kinetic-electron extension
     - ``gamma(k_y)``, ``omega(k_y)``, multi-alpha/multi-surface windows
     - stella/GENE benchmark paper + W7-X TEM literature
     - Open
     - ``docs/_static/tem_branch_parity_audit.json`` is outside the publication
       parity envelope and ``docs/_static/w7x_tem_extension_status.json`` keeps
       multi-alpha, multi-surface, and kinetic-electron nonlinear windows open.
       Do not use the closed W7-X ITG row as a TEM claim.
   * - W7-X zonal flow
     - residual level, damping envelope
     - stella/GENE benchmark paper + zonal-flow literature
     - Open; time coverage closed, residual and late-envelope gates open
     - a case-specific runtime/tool path exists through ``benchmarks/runtime_w7x_zonal_response_vmec.toml`` and ``tools/artifacts/build_w7x_zonal_validation_artifacts.py response-panel``. The runtime now supports the paper-facing ``init_field="phi"`` Gaussian potential initializer and writes both the older volume-weighted ``Phi_zonal_mode_kxt`` diagnostic and the W7-X line-average ``Phi_zonal_line_kxt`` observable. The frozen VMEC-backed artifact now lives at ``docs/_static/w7x_zonal_response_panel.png`` with metadata in ``docs/_static/w7x_zonal_response_panel.json`` and replayable traces in ``docs/_static/w7x_zonal_response_panel.traces.csv``; it uses line-first normalization, following the paper text. The stella/GENE Fig. 11 reference traces and inset residuals are digitized by ``tools/artifacts/build_w7x_zonal_reference_artifacts.py digitize`` into ``docs/_static/w7x_zonal_reference_digitized.csv`` and ``docs/_static/w7x_zonal_reference_digitized_residuals.csv``. ``tools/artifacts/build_w7x_zonal_reference_artifacts.py compare`` writes the residual/time-coverage/envelope artifact ``docs/_static/w7x_zonal_reference_compare.json`` and can now replay the comparison from the tracked combined trace CSV. ``tools/artifacts/build_w7x_zonal_validation_artifacts.py contract`` writes ``docs/_static/w7x_zonal_contract_audit.png`` as a publication-facing open-lane diagnostic, and ``tools/artifacts/build_w7x_zonal_validation_artifacts.py state-convention`` writes ``docs/_static/w7x_zonal_state_convention_audit.png`` to close the initializer/observable convention layer. The current GKX artifact enforces the intended test-4 ``k_x rho_i`` values ``[0.05, 0.07, 0.10, 0.30]`` with a periodic radial box for the ``k_y=0`` zonal run and reaches ``t≈3460`` for ``k_x rho_i=0.05`` and ``t≈1980`` for the other three wavelengths. Under the paper-facing normalization, residuals fail at ``k_x rho_i=0.07``, ``0.10``, and ``0.30`` and late envelopes fail for all wavelengths. The state-level audit closes the convention question with Gaussian-profile relative ``L2`` error ``1.85e-6`` and helper/manual observable agreement near ``2e-16``. The bounded recurrence sweep shows that ``Nl=12,Nm=48`` has the best no-closure trace error on ``t v_t/a <= 100`` and that constant-source closure suppresses the final Hermite tail but worsens the trace error. The remaining closure step is therefore a physical velocity-space recurrence / damping fix rather than a documentation or normalization change
   * - W7-X fluctuation spectra
     - resolved ``k_y`` spectra, ``k_x``-``k_y`` fluctuation power, and temporal spectra
     - W7-X nonlinear gate plus Doppler-reflectometry comparison conventions
     - Initial simulation diagnostic closed; experimental transfer-function validation deferred
     - ``tools/artifacts/plot_w7x_fluctuation_spectrum_panel.py`` writes ``docs/_static/w7x_fluctuation_spectrum_panel.png`` with CSV/JSON/PDF companions from the gated W7-X ``t≈200`` nonlinear NetCDF output. The script refuses failed nonlinear gate summaries by default and marks the JSON with ``claim_level = "validated_nonlinear_simulation_spectrum_not_experimental_validation"`` and ``gate_index_include = false``. It therefore closes the reproducible spectrum-estimator layer while leaving density/zonal-frequency comparison through a Doppler-reflectometry transfer function as a future manuscript extension
   * - HSX
     - ``gamma(k_y)``, ``omega(k_y)``
     - GX / internal frozen references
     - Closed
     - near-marginal deviations documented explicitly
   * - Electromagnetic stellarator verification
     - heavy-electron linear/nonlinear EM response
     - GENE-3D verification conventions
     - Open
     - close heavy-electron EM lane before realistic-mass claims

Frozen artifact paths for the currently closed stellarator linear lanes:

- ``docs/_static/w7x_linear_t2_scan.csv``
- ``docs/_static/hsx_linear_t2_scan.csv``
- ``docs/_static/w7x_linear_t2_lastvalue.csv``
- ``docs/_static/hsx_linear_t2_lastvalue.csv``
- ``docs/_static/comparison/reference_modes/w7x_linear_reference_ky0p3000.npz``
- ``docs/_static/reference_modes/w7x_linear_gkx_ky0p3000.csv``
- ``docs/_static/w7x_eigenfunction_reference_overlay_ky0p3000.png``
- ``docs/_static/reference_modes/w7x_eigenfunction_reference_overlay_ky0p3000.json``
- ``docs/_static/benchmark_core_linear_atlas.png``

For W7-X, the whole-window scan and the late-time last-value reduction tell the
same story. For HSX, the whole-window ``mean_rel_gamma`` metric is kept as an
honest near-marginal stress signal, but the late-time closure should be read
from ``docs/_static/hsx_linear_t2_lastvalue.csv`` because the final
``(gamma, omega)`` values are much tighter than the whole-window average.

The W7-X raw eigenfunction overlay is now closed at ``k_y rho_i = 0.3`` using
``tools/artifacts/generate_linear_reference_overlays.py w7x``. The frozen GX bundle was refreshed
from the finite ``t≈2`` raw field history because the older bundle source
contained non-finite late-time fields. The matched imported-geometry
GKX extraction uses the validated ``z_index`` diagnostic contract and
gives normalized overlap ``0.9999999994`` and relative ``L^2`` mismatch
``3.33e-5`` against the frozen GX raw mode.

.. image:: _static/w7x_eigenfunction_reference_overlay_ky0p3000.png
   :alt: W7-X raw eigenfunction overlay against the frozen GX reference
   :width: 100%

Reference-Code Linear Parity Matrix
-----------------------------------

The lanes above each close one physics family against its own reference. This
section is the cross-cutting view: one instrument, one convention, several
cases run as a single campaign, so the numbers are comparable with each other
and not only with their own histories.

The instrument is ``tools/comparison/build_gx_parity_matrix.py``, driven by
``tools/gx_parity_matrix_manifest.toml``. For each case it

- reads the GX spectrum with GX's own late-window convention, the mean over the
  second half of its diagnostic trace;
- runs the batched GKX linear ``k_y`` scan at the velocity resolution
  ``(N_l, N_m)`` of the GX input deck, importing the GX run's own geometry so
  that both codes integrate identical geometric coefficients;
- integrates, and fits ``gamma`` and ``omega`` over the last thirty percent of
  that trajectory; and
- repeats the scan over half the integration time and records how far the
  answer moved.

The four electrostatic cases are integrated for exactly the time their GX deck
integrated, ``t=150`` for Cyclone and ``t=200`` for the two stellarators. The
two electromagnetic cases are not: their stability-limited step is ``2e-4``
against ``2e-3`` for the electrostatic cases, so they are integrated to ``t=20``
against the deck's ``t=40``. Every integration time is recorded per case in the
``resolution`` block of ``docs/_static/gkx_gx_linear_parity_matrix.json``.

The last step is what makes the table readable. A near-marginal ``k_y`` needs
far more e-foldings than a peak-``k_y`` mode before either code has an
eigenvalue rather than a transient, so a raw relative difference at the edge of
a spectrum mixes model disagreement with unfinished convergence. A wavenumber
whose growth rate moves by more than five percent between the half-length and
the full-length integration is marked unsettled in
``docs/_static/gkx_gx_linear_parity_matrix.csv``, and the headline column below
reports the largest difference over the settled wavenumbers only.

.. list-table:: Linear parity against GX, matched resolution and geometry
   :header-rows: 1
   :widths: 20 12 10 14 14 14 16

   * - Case
     - geometry
     - ``(N_l,N_m)``
     - settled ``k_y``
     - max ``|d gamma|``
     - ``d gamma`` at peak
     - ``d omega`` at peak
   * - Cyclone ITG, s-alpha
     - s-alpha analytic
     - ``(16,48)``
     - 10 of 11
     - ``0.6%``
     - ``+0.02%``
     - ``+7.7e-05``
   * - Cyclone ITG, Miller
     - Miller
     - ``(16,48)``
     - 15 of 15
     - ``2.4%``
     - ``-5.6e-05``
     - ``-1.6e-05``
   * - Cyclone ITG, Miller, kinetic electrons
     - Miller
     - ``(16,48)``
     - 1 of 7
     - ``10.8%``
     - ``-2.13%``
     - ``-1.32%``
   * - KBM, Miller
     - Miller
     - ``(16,48)``
     - 3 of 5
     - ``1.3%``
     - ``-0.91%``
     - ``+0.44%``
   * - W7-X ITG, VMEC
     - VMEC flux tube
     - ``(8,16)``
     - 22 of 27
     - ``3.2%``
     - ``+3.3e-05``
     - ``-2.2e-06``
   * - HSX QHS ITG, VMEC
     - VMEC flux tube
     - ``(8,16)``
     - 9 of 11
     - ``0.9%``
     - ``+1.3e-05``
     - ``-8.7e-06``

.. image:: _static/gkx_gx_linear_parity_matrix.png
   :alt: Growth rate and frequency parity against the GX comparison code
   :width: 100%

Two results in that table are worth stating in words.

First, the integration length, not the model, sets the Cyclone difference.
Reproducing the tracked ``t=10`` Cyclone protocol at ``(N_l,N_m)=(16,48)``
gives ``+2.24%`` in ``gamma`` and ``+5.41%`` in ``omega`` at the peak
``k_y rho_i = 0.3``. Integrating the same configuration for the ``t=150`` the
GX deck itself runs moves that to ``+0.02%`` and ``+0.01%``, and the whole
resolved band ``0.15 <= k_y rho_i <= 0.55`` falls inside ``0.21%`` in ``gamma``
and ``0.07%`` in ``omega``. The older Cyclone mismatch figures are therefore a
statement about a short fitting window, not about the operators.

Second, where both codes are converged and discretized identically, they agree
to far better than any gate in this repository. At the W7-X peak
``k_y rho_i = 1.6`` the growth-rate difference is ``3.3e-5`` and the frequency
difference is ``2.2e-6``; at the HSX peak ``k_y rho_i = 1.1`` both are below
``5e-5``. This is the expected behaviour when the geometry, the velocity
resolution and the parallel grid are shared: the two codes are converging on
the same discrete eigenvalue, so what remains is time-integration and fitting
error rather than a model difference.

The unsettled rows are all low-``k_y`` near-marginal modes, and they are
reported rather than dropped. The one genuinely unresolved row is the
kinetic-electron Cyclone case at ``k_y rho_i = 0.1``, where the fitted GKX
frequency is over a hundred times the GX value: that is a different branch, the
electrostatic shear-Alfven response that the GX deck's own comments warn about
at small but finite ``beta``, and not a growth-rate error on the ITG branch.

One case in that table has a coarser instrument than the others, and the table
does not show it. Rebuilding GX at the same commit with ``-prec-sqrt=true``
added -- IEEE-correct single-precision ``sqrtf``, the setting upstream's own
maintained desktop makefile uses, and one that only 4 of upstream's 20
makefiles carry -- moves **KBM, Miller** by ``0.16%`` to ``0.20%`` in ``gamma``
and ``omega``. It moves nothing else: the converged Cyclone and W7-X modes come
back bit-identical, and the rest of the matrix moves by at most ``0.07%``. The
run-to-run noise floor underneath that measurement is exactly zero, because
rerunning one binary reproduces its ``omega_kxkyt`` bitwise, so the whole
``0.2%`` is the compiler flag. KBM is the electromagnetic branch at finite
``beta`` with kinetic electrons and the shortest integration in the matrix,
``t=20`` against ``t=150``, which is where single-precision rounding has the
most room to accumulate.

So the KBM row is only readable down to a few tenths of a percent, and a gate
on it tighter than that would be gating ``nvcc``, not physics. That resolution
is recorded as ``build_reproducibility_floor = 0.003`` on the ``kbm_miller``
case in ``tools/gx_parity_matrix_manifest.toml``;
``tools/comparison/build_gx_parity_matrix.py`` writes it into the artifact per
case and marks each row that falls inside it, so the next regeneration carries
the resolution of the instrument beside the reading, and
``tests/release/test_release_gates.py`` keeps the number from being tightened
back below what was measured. Regenerating the KBM reference under one
canonical build would not remove the floor: upstream is not uniform about the
flag, so the same spread would reappear against any reference taken on the
other setting. No other case in the matrix has a measured floor, which is a
statement about what has been measured and not a claim that theirs is zero.
The underlying two-build measurement is in ``plan/notes/gx_precsqrt.md``.

Cost was measured on the same RTX A4000 for the three cases where both codes
were re-run under ``/usr/bin/time -v``. Both codes evaluate the whole ``k_y``
spectrum in a single invocation, so the wall times are per-spectrum, and they
are normalized by the simulated time each run covered because the two codes
were not always integrated for the same interval. The comparison is not
like-for-like in arithmetic: GX is compiled in single precision, while these
GKX runs are float64 under ``JAX_ENABLE_X64``.

.. list-table:: Cost per unit simulated time, same device, whole-spectrum runs
   :header-rows: 1
   :widths: 24 19 19 19 19

   * - Case
     - GKX s per unit ``t``
     - GX s per unit ``t``
     - GKX peak host / device
     - GX peak host / device
   * - Cyclone ITG, s-alpha
     - ``3.58``
     - ``2.44``
     - ``1340`` / ``178`` MB
     - ``645`` / ``1547`` MB
   * - W7-X ITG, VMEC
     - ``1.58``
     - ``2.20``
     - ``1954`` / ``636`` MB
     - ``2838`` / ``1547`` MB
   * - HSX QHS ITG, VMEC
     - ``0.70``
     - ``0.51``
     - ``1548`` / ``269`` MB
     - ``636`` / ``1499`` MB

The two codes are therefore within about a factor of ``1.5`` of each other in
either direction on this hardware, with GKX ahead on the W7-X flux tube and
behind on the two smaller-``N_z`` cases, and GKX consistently resident in far
less device memory. The GX device figures are the peak of a two-second
``nvidia-smi`` sample stream with the idle baseline of the shared card
subtracted, so they are coarser than the GKX figures, which come from the JAX
allocator's own peak counter.

Coverage of this campaign, stated as scope rather than implied by the table: it
covers electrostatic and electromagnetic cases, adiabatic and kinetic
electrons, tokamak and stellarator geometry, and three geometry models, all
linear. It does not cover ETG or TEM, because GX ships no deck for either, and
it covers no nonlinear case. The KAW lane stays deferred for a reason upstream
of GKX: a freshly built GX does not reproduce its own shipped KAW reference, so
there is nothing stable to compare against yet.

Provenance differs across the rows and the distinction matters. The Cyclone and
KBM references are the spectra that ship with GX, reproduced by the local
build. The W7-X and HSX references were generated by this campaign with that
build, so comparing against them is a cross-code check, not a check against a
published result. The HSX case is new: no upstream deck exists, the input deck
was written for this campaign, and its VMEC flux tube had to be generated
through a locally patched copy of the GX geometry module, which fails on a
vacuum equilibrium because two drift arrays keep two extra length-one axes. The
patch only squeezes those axes. Two independent checks confirm it changed no
values: the generated tube satisfies the vacuum identity ``cvdrift == gbdrift``
to ``2.8e-17``, and GKX's own VMEC flux-tube construction from the same
equilibrium reproduces the tube's ``|B|`` to a relative ``1.2e-6``, with
``q`` and ``s_hat`` agreeing to ``2.2e-7`` and ``1.6e-5``.

Nonlinear Validation
--------------------

.. list-table::
   :header-rows: 1

   * - Lane
     - Observable
     - Reference
     - Status
     - Baseline gate
   * - Cyclone ITG
     - heat-flux window mean/std/RMS, ``Wphi``, ``Wg``
     - GX
     - Closed
     - current release gate ``<= 1e-1``; mature target ``<= 5e-2`` after further tightening
   * - Cyclone Miller
     - same as above
     - GX
     - Closed
     - allow documented low-amplitude / overlap-only adjustments
   * - KBM
     - ``Wg``, ``Wphi``, ``Wapar``, heat flux
     - GX
     - Closed
     - mature lane
   * - W7-X
     - heat-flux windows, saturation trend
     - GX + W7-X benchmark conventions
     - Closed
     - release gate ``<= 1e-1``; manuscript target tighter where feasible. The exact-state convention audit at ``docs/_static/w7x_exact_state_audit.png`` separately closes startup state, late geometry/field arrays, and scalar diagnostic reconstruction against GX dumps with maximum finite pointwise relative error ``4.62e-5`` under a ``1e-4`` gate and scalar diagnostics below ``1.8e-7``.
   * - HSX
     - heat-flux windows, saturation trend
     - GX / internal frozen references
     - Closed
     - near-threshold behavior documented
   * - ETG full-GK pilot
     - short-window nonlinear transport
     - GX + ETG operating-point convention
     - Exploratory
     - manuscript use only if the pilot is explicitly framed as such
   * - kinetic-electron Cyclone
     - electromagnetic nonlinear transport
     - GX
     - Deferred
     - keep out of the paper until branch identity and runtime cost are closed

Frozen artifact paths for the currently closed nonlinear lanes:

- ``docs/_static/nonlinear_cyclone_diag_compare_t400.png``
- ``docs/_static/nonlinear_cyclone_miller_diag_compare_t122.png``
- ``docs/_static/nonlinear_kbm_diag_compare_t400_stats.png``
- ``docs/_static/nonlinear_w7x_diag_compare_t200.png``
- ``docs/_static/hsx_nonlinear_compare_t50_true.png``
- ``docs/_static/benchmark_core_nonlinear_atlas.png``

Machine-readable nonlinear window gates are now tracked for the first refreshed
subset:

.. image:: _static/nonlinear_window_statistics.png
   :alt: Windowed nonlinear diagnostic agreement against GX
   :width: 100%

The summary panel above is generated by
``tools/artifacts/build_nonlinear_validation_panels.py window-statistics`` from the frozen gate-summary JSON
files. It plots the gate statistic (windowed mean relative mismatch) and the
maximum relative mismatch for each diagnostic, excluding exploratory summaries
with ``gate_index_include=false``.

- ``docs/_static/nonlinear_cyclone_miller_gate_summary.json``: passed at the
  tightened case gate ``0.095``.
- ``docs/_static/nonlinear_kbm_gate_summary.json``: passed at the tightened
  case gate ``0.02``.
- ``docs/_static/nonlinear_hsx_gate_summary.json``: passed at the tightened
  case gate ``0.05``.
- ``docs/_static/nonlinear_w7x_gate_summary.json``: passed at the current
  ``0.10`` mean-relative release gate after the corrected adaptive state
  continuation and comparison-compatible ``Phi2`` artifact refresh.
- ``docs/_static/nonlinear_cyclone_gate_summary.json``: passed on the mature
  Cyclone ``t=100..400`` transport window at the current ``0.10``
  mean-relative release gate.
- ``docs/_static/nonlinear_cyclone_short_gate_summary.json``: retained only as
  an exploratory ``t=5`` startup/resolved-spectrum audit and excluded from the
  release-gate index.

Equilibrium Flow Shear
----------------------

.. list-table::
   :header-rows: 1

   * - Layer
     - Observable
     - Reference
     - Status
     - Acceptance gate
   * - Shearing coordinates and remap
     - :math:`k_x^*(t)`, retained norm, de-alias mask, Hermitian residual
     - analytic shearing wave and corrected-remap formulation
     - Closed
     - analytic trajectory, inverse-remap identity, and machine-zero physical-
       field symmetry away from the de-alias boundary
   * - Periodic and linked linear operators
     - complete cache, linked-neighbor spacing, linear RHS
     - zero-shear static operator plus centered finite differences
     - Closed
     - zero-shear identity, invariant linked topology, and JAX tangent/finite-
       difference agreement
   * - Explicit RK2/RK3 integration
     - state, fields, heat flux, observed order
     - established unsheared trajectory and timestep refinement
     - Closed as a research API
     - zero-shear trajectory identity and designed physical-time order
   * - Fixed-step IMEX integration
     - endpoint state, fields, heat flux, JVP, VJP
     - static linked IMEX trajectory and centered finite differences
     - Closed as a research API
     - zero-shear identity, first-order convergence, endpoint diagnostic
       identity, and finite-difference derivative agreement
   * - Full-resolution fixed-step response
     - independently converged baseline/treatment late-window heat flux
     - matched ``64x64x24``, ``Nl=4``, ``Nm=8`` weak-shear campaign
     - Rejected for promotion
     - each ``t=[240,300]`` window must pass finite-sample, stationarity,
       terminal-mean, block-count, and SEM checks before the treatment effect is
       evaluated; the internal windows failed stationarity and gave a 4.82%
       increase, while stationary fixed-RK4 comparison windows gave a 24.82%
       increase, so the 5% reduction and two-combined-SEM gates failed
   * - Input-file and executable support
     - reproducible user-configured flow-shear run
     - completed equation, integrator, transport, and comparison gates
     - Deferred after the full-resolution fixed-step response failed
     - no TOML key or executable claim while the response row is rejected

The earlier adaptive ``t=300`` internal campaign passed its independent
stationarity gates and showed a 6.10% reduction, while the external adaptive
campaign showed no statistically resolved reduction. A source audit found that
the two paths advance the shearing basis at different times within the adaptive
RK stages. The final fixed-step audit does not recover that suppression. The
internal IMEX pair is still drifting over ``t=[240,300]`` and gives means
``15.4508 +/- 0.2628`` and ``16.1948 +/- 0.1602``. The fixed-RK4 comparison pair
passes both independent window gates and gives ``11.7154 +/- 0.2157`` and
``14.6236 +/- 0.1407``: a 24.82% increase separated by 11.29 combined SEM. This
is negative model-promotion evidence, not a threshold to retune. Startup traces
and short strong-shear suppression pilots cannot override it.

Quasilinear Diagnostics and Model Selection
-------------------------------------------

The quasilinear verification surface is deliberately split between validated
linear-state diagnostics, rejected absolute-flux calibration attempts, and one
scoped model-selection result. A closed model-selection status must not be read
as a promoted runtime predictor.

.. list-table::
   :header-rows: 1

   * - Lane
     - Observable
     - Reference or artifact
     - Status
     - Baseline gate
   * - Electrostatic quasilinear weights and spectra
     - heat/particle weights, growth/frequency spectra, and channel metadata
     - ``docs/_static/quasilinear_*_spectrum.*`` and
       ``docs/_static/quasilinear_validated_calibration_inputs.json``
     - Closed as diagnostics
     - electrostatic channel validation and reproducible spectrum generation;
       this is not calibrated absolute-flux prediction
   * - One-constant and simple saturation-rule absolute-flux models
     - train/holdout heat-flux prediction error
     - ``docs/_static/quasilinear_stellarator_train_holdout_report.json`` and
       ``docs/_static/quasilinear_saturation_rule_sweep.json``
     - Rejected / unpromoted
     - current one-constant and simple-rule reports fail the held-out
       absolute-flux gate and must not be exposed as a user-facing saturation
       law
   * - ``spectral_envelope_ridge`` model selection
     - leave-one-geometry-out error and interval coverage
     - ``docs/_static/quasilinear_candidate_uncertainty.json`` and
       ``docs/_static/quasilinear_candidate_regularization_sweep.json`` and
       ``docs/_static/quasilinear_model_selection_status.json``
     - Closed as scoped model-development result / failed promotion gate
     - the expanded 11-case uncertainty/model-selection gate fails because the
       best candidate has mean relative error about ``0.400 > 0.35`` and
       prediction-interval coverage ``9/11``; after adding the QP holdout it
       still fails the strict rank/correlation screening gates, so it
       does not promote a runtime/TOML absolute-flux predictor, universal
       nonlinear transport model, or shipped saturation option
   * - Future absolute-flux promotion
     - calibrated heat-flux prediction on nonlinear holdouts
     - future late-window convergence metadata and promotion JSON
     - Open
     - every holdout needs finite passed post-transient convergence metadata:
       transient cutoff, running-mean drift, block/bootstrap uncertainty,
       finite sample count, and source provenance

These gates do not change the deferred W7-X lanes: W7-X zonal long-window
recurrence/damping and W7-X TEM / kinetic-electron validation remain outside
the current manuscript/release scope. They also do not promote a universal
absolute-flux model. Production nonlinear optimization remains unpromoted:
the selected optimized-equilibrium summaries do not carry auditable raw-source,
stationarity, uncertainty, resolution, and spectral evidence.

Autodiff Validation
-------------------

.. list-table::
   :header-rows: 1

   * - Workflow
     - Observable
     - Validation type
     - Status
   * - Sensitivity analysis
     - ``gamma``, ``omega``, transport windows
     - finite-difference / complex-step / tangent consistency
     - Closed for fixed-step equation-level observables; open for converged
       long-window turbulent transport
   * - Nonlinear IMEX implicit VJP
     - electrostatic endpoint ion heat flux versus ``a/L_Ti``
     - centered finite difference + Krylov-tolerance stability
     - Closed as a bounded derivative gate; not a saturated-transport claim
   * - Two-mode inverse problem
     - planted parameter recovery
     - gradient check + covariance estimate
     - Closed
   * - UQ / Laplace example
     - posterior covariance and propagated uncertainty
     - Hessian/Jacobian validation
     - Open
   * - Stellarator optimization prototype
     - low-dimensional objective reduction
     - gradient consistency + constrained solve behavior
     - Closed for reduced objective plumbing; open for production nonlinear
       heat-flux optimization

The single-mode inverse figure is intentionally a sensitivity and
non-identifiability demonstration. The two-mode figure is the closed
parameter-recovery validation. Both examples now write finite-difference
Jacobian checks, Jacobian rank/condition number, covariance, standard
deviations, correlations, and one-sigma UQ ellipse area into their summary
JSON files. Those metadata are part of the validation gate: differentiated
observables are not promoted to inverse-design or UQ claims unless the
derivative check is conditioned and the inverse problem is identifiable.

Differentiable Geometry and Stellarator Objectives
--------------------------------------------------

The VMEC/Boozer objective lane is split into three claim levels. The first two
are currently release/manuscript scoped; the third remains a future promotion
gate.

.. list-table::
   :header-rows: 1

   * - Workflow
     - Observable
     - Reference or artifact
     - Status
     - Baseline gate
   * - VMEC/Boozer equal-arc geometry parity
     - ``bmag``, ``bgrad``, ``gradpar``, zero-beta metric profiles, drift
       profiles, ``q``, ``s_hat``, and solver Jacobian
     - ``docs/_static/vmec_boozer_parity_matrix.json``
     - Closed for artifact-passing rows
     - ``mboz,nboz >= 21`` and row-level parity tolerances pass; QI scope is
       limited to the fixed-resolution row plus evaluated robustness variants
   * - Solver-ready objective gradients
     - linear eigenfrequency, growth, ``<k_perp^2>``, electrostatic heat-flux
       weight, and ``gamma Q_i/k_perp^2``
     - ``docs/_static/solver_objective_gradient_gate.json`` and
       ``docs/_static/vmec_boozer_gradient_holdout_matrix.json``
     - Closed for reduced QH/Li383 gates
     - implicit AD/finite-difference mismatch remains within the tracked gate;
       current combined maximum relative mismatch is about ``2.7e-2`` after
       adding reduced nonlinear-window estimator rows
   * - VMEC/Boozer aggregate optimization promotion
     - aggregate objective decrease plus surface/field-line generalization
     - ``docs/_static/vmec_boozer_aggregate_objective_gate.json``,
       ``docs/_static/vmec_boozer_multi_point_objective_gate.json``,
       ``docs/_static/vmec_boozer_reduced_portfolio_guard.json``,
       ``docs/_static/vmec_boozer_aggregate_line_search_gate.json``,
       ``docs/_static/vmec_boozer_aggregate_line_search_comparison.json``,
       ``docs/_static/vmec_boozer_aggregate_alpha_holdout_gate.json``,
       ``docs/_static/vmec_boozer_aggregate_surface_holdout_gate.json``,
       ``docs/_static/vmec_boozer_second_equilibrium_aggregate_gate.json``,
       ``docs/_static/vmec_boozer_aggregate_holdout_promotion_gate.json``, and
       ``tools/release/check_vmec_boozer_gates.py aggregate-holdout``
     - Closed for VMEC/Boozer held-out plumbing; open for broad production
       nonlinear optimization claims
     - aggregate finite-difference and line-search artifacts pass on the same
       training sample set, and the independent QH production-scope validation
       artifact covers a held-out field-line ``alpha`` with a passed replicated
       nonlinear-window ensemble. Production nonlinear optimization still
       requires auditable optimized-equilibrium ensembles and matched audits.
       One converged held-out VMEC/Boozer
       transport window cannot by itself support a broad optimized-equilibrium
       heat-flux claim.
       The multi-alpha finite-difference artifact passes and the growth-vs-QL
       comparison shows objective-dependent descent directions. The
       reduced-portfolio guard now verifies that the multi-alpha rows have real
       VMEC/Boozer provenance, multi-alpha/multi-``k_y`` metadata, finite
       aggregate FD fields, finite growth/QL AD/FD diagnostics, and no
       production nonlinear claim. The alpha-heldout and surface-heldout
       splits pass as reduced generalization evidence, Li383 passes as a
       second-equilibrium aggregate finite-difference/line-search check, and
       the QH held-out transport artifact closes the VMEC/Boozer aggregate
       promotion gate. The separate production nonlinear optimization guard now
       includes long-window D-shaped, circular, and QH VMEC/Boozer holdouts plus
       historical optimized-equilibrium and matched summaries. None includes
       the required immutable raw-source/time-window manifest or all
       stationarity, uncertainty, resolution, and spectral gates; the scoped
       production guard therefore remains unpromoted.
   * - Reduced stellarator ITG optimization and UQ
     - objective reduction history, AD/finite-difference derivative parity,
       local covariance, and projected uncertainty
     - ``docs/_static/stellarator_itg_optimization_comparison.json`` and
       ``docs/_static/stellarator_itg_optimization_uq.json``
     - Closed as reduced optimization plumbing
     - objective/UQ metadata pass for the tracked QA control vector; the
       nonlinear entry is a smooth reduced window estimator
   * - VMEC/Boozer nonlinear startup FD audit
     - compact startup-window heat-flux response to geometry perturbation
     - ``docs/_static/vmec_boozer_nonlinear_window_fd_audit.json``
     - Exploratory plumbing gate
     - finite-output and finite-difference-response checks pass, but
       ``transport_average_gate = false``
   * - VMEC-state nonlinear-gradient launch runbook
     - QL-admitted VMEC-state controls and required state-to-input mapping
       readiness
     - ``docs/_static/nonlinear_gradient_ql_seed_screen.json`` and
       ``docs/_static/nonlinear_gradient_state_control_runbook.json``;
       launch-plan and measured-response artifacts are tracked in
       ``docs/_static/nonlinear_gradient_state_to_input_mapping_campaign.json``
       and
       ``docs/_static/nonlinear_gradient_state_to_input_mapping_response.json``,
       plus the symmetry-compatible
       ``docs/_static/nonlinear_gradient_asymmetric_state_to_input_mapping_campaign.json``
       and
       ``docs/_static/nonlinear_gradient_asymmetric_state_to_input_mapping_response.json``
     - Closed for checked short-bracket launch mapping; nonlinear evidence still open
     - the QL seed screen admits ``Rsin_mid_surface_m1`` and
       ``Zcos_mid_surface_m1`` as state controls. The measured ``RBC/ZBS``
       response matrix has rank ``0`` and relative target residual ``1`` for
       both controls, but the measured ``LASYM=true`` ``RBS/ZBC`` response has
       rank ``2`` with condition number about ``1.02`` and residuals near
       machine precision. The runbook now emits explicit least-squares input
       directions for checked short-bracket launches; converged long-window
       nonlinear-gradient campaigns remain the next evidence gate
   * - VMEC-state short-bracket launch status
     - mapped state-control VMEC launch decks and bounded nonlinear campaign
       manifests
     - ``docs/_static/nonlinear_gradient_state_control_short_bracket_launch.json``
       and
       ``docs/_static/nonlinear_gradient_state_control_short_bracket_launch_status.json``
     - Closed as launch readiness; nonlinear-gradient evidence still open
     - the two mapped state controls are converted to explicit ``LASYM=true``
       ``RBS/ZBC`` input directions with absolute ``1e-3`` scalar steps. All
       six baseline/plus/minus VMEC solves terminate normally, and two bounded
       ``t=150`` nonlinear campaign manifests are prepared. This is not a
       transport-gradient claim until those nonlinear campaigns pass runtime,
       replicated-window, central-FD, and evidence gates
   * - VMEC-state short-bracket nonlinear audit
     - bounded nonlinear runtime outputs, replicated windows, and central-FD
       promotion blockers for mapped state controls
     - ``docs/_static/nonlinear_gradient_state_control_short_bracket_nonlinear_audit_status.json``
       plus the two ``qa_lowres_state_control_short_bracket_*_central_fd_gradient_gate.json``
       artifacts and their replicated-window sidecars
     - Closed as negative short-bracket evidence; production gradient still open
     - all ``18`` office GPU nonlinear runs complete. The six corrected
       runtime-output groups and six replicated-window ensemble groups pass,
       with heat-flux means near ``10``. Both central-FD gates fail closed:
       the ``Rsin`` response fraction is about ``0.0045`` with asymmetry
       about ``9.5`` and uncertainty about ``7.7``; the ``Zcos`` response
       fraction is about ``0.0015`` with asymmetry about ``45`` and
       uncertainty about ``23``. A bracket-amplitude sweep or longer/lower-noise
       window is required before promotion
   * - VMEC-state bracket-amplitude sweep
     - bounded nonlinear outputs, replicated windows, and central-FD promotion
       blockers for mapped controls at larger single-control amplitudes
     - ``docs/_static/nonlinear_gradient_state_control_bracket_sweep_status.json``
       plus the four ``qa_lowres_alpha0p*_state_control_short_bracket_*_central_fd_gradient_gate.json``
       artifacts and their replicated-window sidecars
     - Closed as negative single-control bracket evidence; production gradient
       still open
     - all ``36`` office GPU nonlinear runs at ``alpha_delta=3e-3`` and
       ``1e-2`` complete with no runtime failures. Output and replicated-window
       gates remain stable, but all four central-FD gates fail. The best
       response fraction is about ``0.0045`` against the ``0.03`` gate, so the
       next test must reduce variance or change the observable/control basis
   * - Selected optimized-equilibrium nonlinear transport audit
     - optimized-equilibrium post-transient heat-flux average with uncertainty
       and nonlinear audit bars
     - ``docs/_static/optimized_equilibrium_replicates/optimized_equilibrium_replicate_t700_ensemble_gate.json``
       and ``docs/_static/production_nonlinear_optimization_guard.json``
     - Historical aggregate; blocked from promotion
     - the selected QA summary passes its old ``t=[350,700]`` aggregate gate
       but lacks the new raw-source, per-trace stationarity, resolution, and
       spectral evidence contract
   * - QA/ESS ``ZBS(1,0)`` 7.5% nonlinear-gradient follow-up
     - matched baseline/plus/minus post-transient heat-flux windows and central
       finite-difference response
     - ``docs/_static/qa_ess_zbs10_rel7p5_nonlinear_gradient_zbs_1_0_central_fd_gradient_gate.json``
       plus the three replicated ensemble sidecar directories and
       ``docs/_static/qa_ess_zbs10_rel7p5_variance_reduction_plan.json``
     - Closed as negative bounded-gradient evidence; production gradient still open
     - all twelve ``t=900`` office-GPU outputs pass the ``t=[450,900]`` output
       gates, and the central finite-difference response is resolved and local
       (``response_fraction = 0.0319``, ``fd_asymmetry_rel = 0.044``). The
       plus ensemble fails its spread gate (``0.196 > 0.15``), and propagated
       uncertainty remains too large (``gradient_uncertainty_rel = 1.81``), so
       the result is not promoted. The paired-seed runbook also fails closed:
       common-label plus-minus differences have relative uncertainty about
       ``0.984``. A midpoint common-mode control variate lowers apparent
       residual uncertainty to ``0.238`` but is blocked because the control
       mean is not independently known, motivating an independent control-mean
       estimate or a better-conditioned observable. The pre-run launch
       contract estimates ``21`` independent control-mean pairs, ``42`` new
       nonlinear runs, and projected combined uncertainty ``0.480``. The
       follow-up office campaign now completes those ``21`` matched pairs and
       passes the strict ``t=[600,1100]`` control-mean gate with combined
       response uncertainty ``0.311 < 0.5``, plus spread ``0.1268``, minus
       spread ``0.1193``, and no failed per-seed window rows.

Use this section as the verification boundary for README figures: the
VMEC/Boozer parity, gradient-holdout, and reduced optimization/UQ panels can be
cited as reduced objective evidence. Startup-window finite-difference panels
and reduced nonlinear-window estimators must not be cited as saturated
transport-gradient validation. The optimized-equilibrium replicate panel may be
cited as a post-transient transport-window audit for the selected candidate,
not as a universal quasilinear absolute-flux model.

Parallelization Validation
--------------------------

Independent ``k_y`` and ensemble parallelization is accepted only when a
serial numerical-identity gate accompanies the timing data. The current closed
artifact is ``docs/_static/parallel_ky_scan_gate.png`` with metadata in
``docs/_static/parallel_ky_scan_gate.json``. It runs the real Cyclone linear
solver with ``ky_batch=1`` and a fixed-shape batched scan, then requires
``max_gamma_rel_error <= 1e-8`` and ``max_omega_abs_error <= 1e-8``. The
observed speedup is reported as an engineering metric, not as the gate itself.
The larger CPU/GPU strong-scaling artifacts
``docs/_static/independent_ky_scan_scaling_large.json`` and
``docs/_static/quasilinear_uq_ensemble_scaling_large.json`` are the current
release references for production independent-work scaling. Their split CPU
and GPU companions must keep per-row identity, timing, and worker/profile
metadata synchronized with the performance and validation manifests.
``docs/_static/parallelization_completion_status.json`` is the release ledger
that turns those artifacts into a scoped production-closure claim while keeping
nonlinear domain-decomposition speedup out of scope.
Fixed-step nonlinear full-state sharding has a small control-flow artifact at
``docs/_static/nonlinear_sharding_profile.json`` generated by
``tools/profiling/profile_nonlinear_sharding.py``. The older bounded two-GPU
smoke artifact, ``docs/_static/nonlinear_sharding_profile_office_gpu.json``, is
identity-clean only on a tiny grid. The matched benchmark-grid artifact,
``docs/_static/nonlinear_sharding_profile_office_gpu_benchmark_grid.json``,
fails final-state identity and is slower than serial. It is the controlling
production-candidate result. The clean-revision interacting-multimode refresh,
``docs/_static/nonlinear_sharding_profile_office_gpu_physical.json``, confirms
that both ``ky`` and ``kx`` whole-state placement fail physical identity and
are slower than serial, so whole-state sharding remains blocked. The large combined sweep
``docs/_static/nonlinear_sharding_strong_scaling_large.json`` is a
profiler/identity artifact, not a production nonlinear speedup claim. The same
diagnostic keeps ``z``-axis FFT sharding out of the release claim until it has
a dedicated communication/layout design and a passing identity gate. True
nonlinear domain decomposition with halo/FFT communication, conservation
checks, and benchmark-size speedup remains outside the release claim.

Notes
-----

- A lane should not move from ``Open`` to ``Closed`` without an owning script,
  frozen artifact path, and literature/reference statement.
- README figures should use only ``Closed`` lanes unless a panel is explicitly
  marked exploratory.
- Raw eigenfunction overlays for manuscript use should be rendered only from
  frozen reference bundles checked into
  ``docs/_static/comparison/reference_modes/``; companion GKX traces and
  JSON gate reports stay in ``docs/_static/reference_modes/``.
  Do not build publication figures from transient external files or ad hoc
  office-machine outputs.
- Experimental-facing figures such as W7-X fluctuation spectra should remain
  scoped as simulation diagnostics unless the diagnostic transfer function and
  access model are encoded directly in the repo.
- Electromagnetic stellarator claims should be split explicitly into
  heavy-electron verification and realistic-electron research runs.
