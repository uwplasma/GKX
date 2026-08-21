Geometry
========

S-alpha flux-tube model
-----------------------

We start with a simple analytic geometry based on the s-alpha model, which is
widely used for Cyclone base case benchmarks. The field-aligned perpendicular
wave number is

.. math::

   k_x(\theta) = k_{x0} - \left(\hat{s}\,\theta - \alpha \sin\theta \right) k_y,

and the metric coefficients are

.. math::

   g_{ds2} = 1 + \left(\hat{s}\,\theta - \alpha \sin\theta\right)^2,\quad
   g_{ds21} = -\hat{s}\left(\hat{s}\,\theta - \alpha \sin\theta\right),\quad
   g_{ds22} = \hat{s}^2.

The perpendicular wave number is then

.. math::

   k_\perp^2(\theta) =
   k_y \left(k_y g_{ds2} + 2 k_x g_{ds21} \right) + k_x^2 g_{ds22},

with an additional :math:`B^{-2}` factor from the s-alpha magnetic field
strength,

.. math::

   B(\theta) = \frac{1}{1 + \epsilon \cos\theta}.

Parameters
----------

The geometry is specified by:

- ``q``: safety factor
- ``s_hat``: magnetic shear
- ``epsilon``: inverse aspect ratio
- ``R0``: reference major radius
- ``B0``: reference magnetic field
- ``drift_scale``: drift normalization (``1.0`` is the tracked default; ``2.0`` selects the alternate doubled-drift convention)

Field-aligned grid parameters
-----------------------------

For direct comparison with published Cyclone base case benchmarks,
``GridConfig`` exposes field-aligned grid inputs:

- ``y0`` sets the minimum binormal wave number via :math:`k_y \rho = 1/y_0`.
  Internally this maps to ``Ly = 2\pi y0`` so that the FFT grid spacing matches.
- ``ntheta`` and ``nperiod`` (or ``zp``) control the parallel grid. We set
  :math:`Z_p = 2\,nperiod-1` and choose ``Nz = ntheta * Zp``, which spans
  :math:`[-\pi Z_p, \pi Z_p)`.

Curvature and grad-B drift
--------------------------

The magnetic drift frequency used in the linear operator follows the standard
s-alpha form

.. math::

   \omega_d(\theta) = k_y \left(\mathcal{C}_v + \mathcal{C}_g\right)
   + k_x \left(\mathcal{C}_v^0 + \mathcal{C}_g^0\right),

with

.. math::

   \mathcal{C}_v = \mathcal{C}_g =
   \frac{\cos\theta + (\hat{s}\,\theta - \alpha \sin\theta)\sin\theta}{R_0},
   \qquad
   \mathcal{C}_v^0 = \mathcal{C}_g^0 =
   -\frac{\hat{s}\sin\theta}{R_0}.

These parameters will be extended to VMEC/DESC geometry once the linear solver
is validated against Cyclone benchmarks.

Slab Model
----------

GKX exposes a slab flux-tube geometry contract directly with
``geometry.model = "slab"``. This is the correct backend for slab secondary
and collisional-ETG benchmark families; it is not an ``s-alpha``
approximation.

The slab overrides are:

- ``bmag = 1`` and ``bgrad = 0``
- ``cvdrift = gbdrift = cvdrift0 = gbdrift0 = 0``
- ``gradpar = 1`` by default, or ``1/z0`` when ``geometry.z0 > 0``
- the metric still uses the supplied ``s_hat`` unless ``geometry.zero_shat = true``
- with ``zero_shat = true``, the slab metric becomes ``gds2 = 1``,
  ``gds21 = 0``, ``gds22 = 1`` and the effective solver shear is zero

That contract is now locked by unit tests so future secondary or slab
full-GK work is built on stable geometry semantics. The retired reduced
collisional-ETG path is no longer part of the maintained runtime.

Geometry Data Contract
----------------------

The linear cache now accepts either:

- the analytic ``SAlphaGeometry`` model, or
- a sampled ``FluxTubeGeometryData`` contract.

``FluxTubeGeometryData`` stores the solver-ready profiles on a specific
``theta`` grid:

- ``bmag`` and ``bgrad``,
- ``gradpar``,
- metric coefficients ``(gds2, gds21, gds22)``,
- curvature / grad-B drift coefficients ``(cv, gb, cv0, gb0)``,
- geometry metadata such as ``q``, ``s_hat``, ``R0``, and the
  ``kperp2_bmag`` / ``bessel_bmag_power`` switches.

This is the insertion point for future VMEC/DESC or imported field-line
geometry. The helper ``sample_flux_tube_geometry`` converts the analytic
s-alpha model into the same contract, and ``ensure_flux_tube_geometry_data``
normalizes analytic and sampled inputs onto one solver-facing representation.

The sampled geometry contract is now a JAX pytree and is accepted by the
linear cache, runtime initial-condition builder, RHS assembly entry points,
nonlinear config runner, and reference-compatible volume-weight diagnostics.
That means upcoming VMEC or imported field-line geometry can be threaded into
more of the codebase without rebuilding solver-specific side paths.

The contract also preserves explicit ``jacobian`` and ``grho`` profiles when
they are available from imported geometry. The helper
``load_imported_geometry_netcdf`` reads grouped NetCDF files with
``Geometry``/``Grids`` groups and root-level ``*.eik.nc`` field-line geometry
files from VMEC/DESC-style workflows. That is the intended short path for
imported stellarator examples: import the sampled field-line geometry first,
prove solver/diagnostic parity on that contract, and only then add a native
VMEC path that generates the same contract inside GKX.

Runtime and executable paths can now construct that bridge directly from config with
``geometry.model = "imported-netcdf"`` and
``geometry.geometry_file = "external_geometry.nc"``. Analytic s-alpha remains
the default with ``geometry.model = "s-alpha"``. For slab cases use
``geometry.model = "slab"`` with optional ``geometry.z0`` and
``geometry.zero_shat`` controls. In practice an imported geometry file
can be either a grouped diagnostic ``*.out.nc`` file or a VMEC/DESC-generated
``*.eik.nc`` file such as the W7-X examples used in benchmark comparisons.
Root-level ``*.eik.nc``
files are no longer assumed to be closed-interval grids: the importer now
infers whether the terminal theta point is present from the periodic endpoint
content of the geometry profiles, so both VMEC-style closed grids and
already-open Miller ``*.eiknc.nc`` grids are mapped onto the correct solver
contract. For imported geometry, the
runtime now also adopts the file's ``theta`` extent, twist-shift
``jtwist/x0`` defaults for both ``linked`` and ``fix aspect`` boundaries, and
``kxfac`` metadata so the flux-tube grid is built from the same field-line
domain encoded in the file.
The same importer is also exposed under the aliases
``geometry.model = "imported-eik"``, ``"vmec-eik"``, and ``"desc-eik"`` so configs
can reflect the provenance of a root-level ``*.eik.nc`` file without changing
the solver-facing geometry contract.
The linear runtime uses the same geometry builder for KBM and ordinary cases.
Regression coverage exercises imported sampled geometry explicitly for both
``"vmec-eik"`` and ``"desc-eik"`` aliases, so imported W7-X-style geometry is
checked through the promoted runtime entry point. The test suite locks both
root-level contracts explicitly:

- imported VMEC/DESC closed-interval ``*.eik.nc`` files must preserve
  ``theta_scale``/``nfp`` metadata and trim the terminal theta point
  consistently when mapped onto the solver's open field-line grid, and
- imported GX Miller ``*.eiknc.nc`` files must stay on their already-open theta
  grid without a spurious terminal-point trim.

With the corrected imported-VMEC contract, that imported-geometry bridge now
also reproduces the GX W7-X linear ITG ``t=2`` reference on the same sampled
field line over the tracked ``ky`` range. The refreshed scan in
``docs/_static/w7x_linear_t2_scan.csv`` shows mean relative ``gamma`` errors of
about ``2.3%`` to ``3.5%`` and mean relative ``omega`` errors of about
``0.02%`` to ``0.27%`` across ``ky = 0.1`` through ``0.8``.

That same imported contract now has a first-class nonlinear runtime workflow:
``examples/nonlinear/non-axisymmetric/w7x_nonlinear_imported_geometry.py`` and
``examples/nonlinear/non-axisymmetric/runtime_w7x_nonlinear_imported_geometry.toml`` mirror the
GX nonlinear W7-X adiabatic-electron setup while keeping the geometry source
explicitly tied to a VMEC/DESC ``*.eik.nc`` field-line file.

GKX now also supports a direct VMEC runtime bridge with
``geometry.model = "vmec"``. This path uses the VMEC field-line helper
to produce an imported ``*.eik.nc`` file and then re-enters the same
imported-geometry contract described above. The bridge is cached by input
content and VMEC file timestamp when GKX chooses the output path itself.
If the user supplies an explicit ``geometry_file`` target, the runtime now
regenerates that file instead of silently reusing whatever stale ``*.eik.nc``
may already be present there.
That keeps the native JAX geometry contract centered on ``FluxTubeGeometryData``
while preserving reproducible imported-geometry workflows.
For VMEC ``fix aspect`` cases, the bridge now leaves ``x0`` unset when calling
the geometry helper so it chooses the cut from ``y0`` and the geometry itself.
GKX no longer back-solves ``x0 = Lx/(2 pi)`` into the helper input, which
was generating the wrong HSX/W7-X ``*.eik.nc`` files.
When ``booz_xform_jax`` is not installed into the active environment, point
GKX at it through ``BOOZ_XFORM_JAX_PATH`` or
``GKX_BOOZ_XFORM_JAX_PATH``. The internal backend is preferred. A
``booz_xform`` install is only needed as an automatic fallback reader for
older helper environments.
Differentiable VMEC/Boozer transport-gradient audits require a
``booz_xform_jax`` checkout at or after upstream commit ``1d5e8c``. That
revision replaces inactive zero-mode Fourier divisions by safe denominators in
the JAX Boozer transform so reverse-mode cotangents through ``w`` spectrum
reconstruction remain finite. Older checkouts can produce finite values but
non-finite gradients and must not be used for promoted transport-gradient
claims.
The first differentiable-geometry bridge is now explicit in
``gkx.geometry.differentiable``. Use
``discover_differentiable_geometry_backends()`` to audit optional
``vmex`` / ``booz_xform_jax`` availability and
``flux_tube_geometry_from_mapping(...)`` to validate an in-memory,
solver-ready field-line geometry bundle before passing it into the existing
``FluxTubeGeometryData`` contract. This is a real contract boundary, not a
proxy equilibrium: the upstream differentiable pipeline must still supply the
sampled ``theta``, ``bmag``, ``gradpar``, metric, drift, Jacobian, and
``grho`` arrays. The bridge is tracer-safe: finite-value checks are kept on
host inputs, while JAX-traced arrays can flow through
``flux_tube_geometry_from_mapping(..., validate_finite=False)`` so geometry
observables, inverse-design objectives, and covariance estimates can be
differentiated.

The release validation artifact is generated by:

.. code-block:: bash

   JAX_ENABLE_X64=1 PYTHONPATH=src \
     python examples/theory_and_demos/differentiable_geometry_bridge.py

It writes ``docs/_static/differentiable_geometry_bridge.png`` and
``docs/_static/differentiable_geometry_bridge.json``. The JSON records
``vmex`` and ``booz_xform_jax`` API availability, autodiff-vs-finite
difference sensitivity errors, inverse-design convergence, local UQ covariance
diagnostics, and seven optional real-backend derivative gates: a ``vmex``
boundary-aspect check, a ``vmex`` metric-tensor check through
``vmex.geom.eval_geom``, a stellarator VMEC field-line tensor check through
``vmex.geom`` plus ``vmex.vmec_bcovar``, a direct VMEC
tensor-derived flux-tube mapping check, a tiny ``booz_xform_jax``
Boozer-spectrum check, a bounded Boozer-spectrum-to-flux-tube mapping check, and a real
``vmex`` ``VMECState`` to ``booz_xform_jax`` to GKX field-line
geometry check. The metric-tensor gate currently has max absolute
AD-vs-finite-difference error about ``5.9e-8`` and max relative error about
``1.3e-7``. The field-line tensor gate uses the non-axisymmetric
``nfp4_QH_warm_start`` fixture and checks ``|B|`` ripple plus sampled VMEC
metric observables before any reduced GKX metric/drift closure is
applied; its current max absolute AD-vs-finite-difference error is about
``2.1e-3`` and max relative error is about ``2.4e-5``. The direct VMEC
flux-tube gate inverts the sampled VMEC metric tensor, derives ``gds*``,
``gradpar``, Jacobian, ``grho``, and a local grad-:math:`B` drift closure, and
checks the resulting solver-ready geometry observables; the current max
relative AD-vs-finite-difference error is about ``1.3e-4`` on the
``nfp4_QH_warm_start`` fixture. The same artifact now also records a bounded
VMEC/EIK array-parity audit for that direct tensor path. That audit currently
keeps the full production gate open because the direct tensor path still uses
a VMEC-coordinate/equal-theta sampling and local grad-:math:`B` closure.
The same report now also runs a JAX-native ``vmex -> booz_xform_jax``
Boozer equal-arc core audit. On the tracked ``nfp4_QH_warm_start`` fixture,
that audit matches the imported convention for ``bmag``, the solver Jacobian,
``gradpar``, ``q``, and ``s_hat`` with worst normalized/scalar errors
``4.5e-3`` and ``2.4e-3``; the derivative-like ``bgrad`` check is recorded
separately and is ``2.3e-2``. The same JAX-native path now reconstructs the
zero-beta Boozer metric profiles ``gds2``, ``gds21``, ``gds22``, and ``grho``
with worst normalized mismatch ``3.45e-2`` and the loaded-convention zero-beta
drift profiles ``cvdrift``, ``gbdrift``, ``cvdrift0``, and ``gbdrift0`` with
worst normalized mismatch ``3.50e-2``. The remaining promotion gap is
finite-beta and broader production-runtime drift parity beyond the tracked
zero-beta equal-arc fixtures, not the Boozer equal-arc field-line or zero-beta
metric/drift normalization on the tracked fixture.
The Boozer gates evaluate
the JAX-native Boozer ``|B|``
spectrum along a field line, build the ``FluxTubeGeometryData`` input mapping,
and compare geometry-observable sensitivities against central finite
differences. In the current artifact the VMEC-state path has max absolute
AD-vs-finite-difference error about ``5.8e-7`` and max relative error about
``1.4e-8`` for the tracked geometry observables.

The reusable API entry point for this workflow is
``geometry_inverse_design_report(mapping_fn, initial_params, target_observables, ...)``:
it runs a bounded Gauss-Newton inverse design on selected solver-ready
geometry observables, checks the final sensitivity Jacobian against central
finite differences, and records local covariance diagnostics. High-fidelity
``vmex`` / ``booz_xform_jax`` optimization examples should use the same
contract once their in-memory field-line mapping is available.

The gradient-report API accepts ``jacobian_chunk_size``. ``None`` evaluates
all forward directions in one ``vmap``; an integer bounds the simultaneous
directions; and ``"auto"`` delegates the memory policy to SOLVAX. The standard
VMEC/Boozer geometry sensitivity report uses ``"auto"`` and records that
choice in its JSON output. Chunked and unchunked Jacobians are required to
agree before a report can support an optimization claim.

``jacobian_mode`` separately controls derivative direction. ``"forward"`` is
appropriate for few design variables, ``"reverse"`` for few observables, and
``"auto"`` selects between them from the input/output dimensions. Requesting
chunking forces forward mode because the chunk size budgets simultaneous JVP
directions. Reports record the resolved mode, and both explicit modes must
agree with central finite differences before either is used in optimization.

The bridge validates more than array shapes. Host-side mappings must contain
finite scalar metadata such as ``q``, ``R0``, ``B0``, and ``theta_scale``,
must provide at least one ``theta`` sample, and must use a positive integer
``nfp``. JAX-traced mappings can still be passed with
``validate_finite=False`` so autodiff transforms do not attempt host NumPy
checks during tracing. The finite-difference utilities used by these gates
also reject non-positive step sizes, and the inverse-design covariance block
records rank and conditioning before any optimization result is promoted from
local sensitivity evidence to a transport-design claim.
Each geometry AD/finite-difference gate now also records a compact
``conditioning`` block alongside the raw Jacobians. That block includes finite
flags for the AD and finite-difference Jacobians, singular values, numerical
rank, condition number, AD row/column norms, per-parameter finite-difference
step scaling, and the observable/parameter location of the worst absolute and
relative AD/FD mismatch. This metadata is intentionally separate from the pass
tolerance: a derivative can agree with finite differences and still be a poor
optimization direction if the sensitivity map is nearly rank deficient or if
the finite-difference step is not well scaled to the chosen VMEC coefficient.
Research artifacts should quote both the derivative error and this conditioning
metadata before treating a VMEC/Boozer bridge row as optimization-ready.

Growth-rate transport-gradient audits also need an eigenbranch-locality check.
The public helper
``solver_linear_operator_matrix_from_geometry(geometry, ...)`` materializes the
same GKX linear operator used by
``solver_growth_rate_from_geometry(...)``. The report
``vmex_transport_growth_branch_locality_report_from_states(base, plus, minus, ...)``
then compares the dominant-growth finite-difference slope against the slope of
the eigenvalue nearest to the base dominant eigenvalue for every configured
surface, field line, and ``k_y`` sample. If the independently selected
max-growth branch switches, or if the base branch is under-isolated, the report
fails closed and labels the row before any transport-gradient optimization
claim is admitted. Current VMEC-JAX exposes the equilibrium implicit derivative through its public
optimization API. GKX growth campaigns select that policy with
``--jacobian implicit`` and retain the branch-locality report as a package-level
physics gate. Historical frozen-axis boundary-chain artifacts remain in
``docs/_static`` as conditioning evidence, but their private-tape executable was
removed when VMEC-JAX retired that optimizer generation. New derivative claims
require the current VMEC-JAX turbulence tangent tests, GKX eigenbranch
locality checks, and an independent finite-difference comparison on the exact
objective used by the campaign.

The reusable low-level entry point is
``observable_gradient_validation_report(observable_fn, params, ...)``. It
flattens arbitrary geometry or objective observables, compares JAX AD
Jacobians with central finite differences, records absolute and relative error
tables, checks a tangent direction, adds finite flags, and applies an explicit
rank/condition-number gate. Its payload is strict JSON compatible: nonfinite
diagnostic numbers are written as ``null`` while the corresponding finite flag
and failure reason remain explicit. ``geometry_sensitivity_report`` is a thin
``FluxTubeGeometryData`` wrapper around the same helper.

For ``vmex`` and ``booz_xform_jax`` this remains a bridge contract, not a
claim that GKX has run a full optimization. The upstream JAX pipeline
must first produce the sampled solver-ready field-line arrays accepted by
``flux_tube_geometry_from_mapping``. Passing the reusable AD/finite-difference
gate proves local differentiability and conditioning of the supplied
observables; production stellarator optimization still requires the VMEC/Boozer
array parity, solver-objective gradient, and nonlinear transport gates
described below.

.. figure:: _static/differentiable_geometry_bridge.png
   :width: 95%
   :align: center
   :alt: Differentiable geometry bridge validation

   Differentiable geometry bridge validation. The panel checks boundary-control
   sensitivities, geometry-observable Jacobians, a two-parameter inverse design,
   and local UQ covariance at the in-memory flux-tube contract boundary. When
   ``vmex`` is available, the panel/JSON also includes a real VMEC
   boundary-aspect derivative check and sampled VMEC metric-tensor derivative
   check, plus a real VMEC field-line tensor check for a non-axisymmetric
   fixture, a direct VMEC tensor-derived flux-tube mapping check, and a
   Boozer equal-arc core/metric parity check against the imported VMEC/EIK
   geometry; when
   ``booz_xform_jax`` is available, it runs a bounded JAX-native
   Boozer spectral transform, samples that spectrum onto a field-line
   flux-tube mapping, checks both autodiff derivative paths against central
   finite differences, and, when both optional backends are available, starts
   from a real ``vmex`` ``VMECState`` before converting through
   ``booz_xform_jax`` into the GKX field-line contract.

Multi-Equilibrium Boozer Parity Matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The single-fixture bridge artifact is complemented by a replayable
multi-equilibrium matrix tracked at
``docs/_static/vmec_boozer_parity_matrix.{png,pdf,json,csv}``.
The builder enforces ``mboz,nboz >= 21`` before calling the real optional
backend path, because the QI drift gate is under-resolved at lower Boozer mode
counts. The JSON now includes a ``sample_set_provenance`` block and the CSV
includes a ``sample_set_id`` column so each bounded row is identified by
``case_name``, ``ntheta``, ``mboz``, and ``nboz``. That provenance block also
records that the builder does not launch external VMEC solves; input-only QI
variants remain explicitly artifact-limited until a bundled ``wout`` reference
exists. The tracked matrix covers the ``nfp4_QH_warm_start``,
``nfp3_QI_fixed_resolution_final``, and ``shaped_tokamak_pressure`` examples.
At ``mboz=nboz=21`` the current regenerated artifact passes all matrix rows.
The fixed-resolution QI case passes the loaded-convention drift subgate with
mismatch about ``7.13e-2`` against an ``8e-2`` release tolerance after fixing
the Boozer half-mesh radial-index convention. The evaluated QI robustness
variants at ``ntheta=8`` and ``ntheta=16`` also pass. The broader QI seed
campaign is still artifact-limited because three input-only QI seeds have no
bundled ``wout`` references, and none of this is broad random-seed nonlinear
QI transport validation or QI optimization. The release guard now requires the
finite-beta/pressure ``shaped_tokamak_pressure`` equal-arc row, so the current
claim cannot silently regress to zero-beta-only parity evidence. Finite-beta
solver-objective geometry gradients, broader production-runtime
pressure-correction drift audits, and nonlinear transport optimization remain
explicitly scoped as follow-up work.

.. figure:: _static/vmec_boozer_parity_matrix.png
   :width: 95%
   :align: center
   :alt: VMEC/Boozer equal-arc parity matrix

   VMEC/Boozer equal-arc parity matrix. Each cell reports the absolute
   mismatch for one subgate, while the color shows mismatch divided by the
   relevant tolerance. The matrix is generated from the actual optional
   ``vmex`` and ``booz_xform_jax`` bridge path and rejects Boozer mode
   counts below 21.

The next implementation step is to extend the same equal-arc path to
finite-beta/production-runtime curvature and drift reconstruction, then replace
the reduced estimator-gradient checks with converged transport-gradient and
broader optimized-equilibrium audits beyond the selected QA candidate.

In-memory differentiable geometry API
-------------------------------------

Differentiable stellarator optimization must stay on the in-memory path:

.. code-block:: python

   from gkx import flux_tube_geometry_from_vmec_boozer_state

   geom = flux_tube_geometry_from_vmec_boozer_state(
       state,
       static,
       indata,
       wout,
       surface_index=surface_index,
       alpha=0.0,
       ntheta=32,
       mboz=21,
       nboz=21,
   )

This public wrapper converts a solved ``vmex`` state through
``booz_xform_jax`` and returns the existing GKX
``FluxTubeGeometryData`` solver contract. The path is
``VMECState -> BoozXformInputs -> Boozer coefficients -> FluxTubeGeometryData``
and does not write or reload ``*.eik.nc`` files. The file-backed VMEC/EIK route
remains the right runtime import path for ordinary examples, but it is not the
path to use for end-to-end differentiable optimization.

The current wrapper is a production API boundary, not a new physics claim. It
inherits the same ``mboz,nboz >= 21`` and equal-arc parity requirements as the
VMEC/Boozer gates. Full stellarator-optimization claims still require
multi-surface/multi-field-line objective gates and nonlinear heat-flux audits
of optimized equilibria.

The lightweight readiness tests mirror that claim boundary. The parity-matrix
tests reject ``mboz,nboz < 21`` and assert that a passed equal-arc matrix is
still tagged as ``not_full_transport_gradient_claim``. The gradient-holdout
tests require the ``mode21_vmec_boozer_state`` source scope, ``mboz,nboz >= 21``,
and explicitly track the nonlinear-window estimator objectives as a reduced
differentiability gate rather than a production nonlinear-optimization gate.
The release guard
``docs/_static/vmec_boozer_differentiability_claim_guard.json`` now checks
those contents directly: it requires the equal-arc parity matrix, the QH/Li383
mode-21 frequency/quasilinear/nonlinear-window gradient holdouts, explicit
``diagnostic_open`` status for the direct VMEC tensor-vs-imported-EIK
convention gap, a passing finite-beta/pressure equal-arc parity row, and a
startup-only label for the nonlinear finite-difference audit. It now also
requires the shaped-pressure finite-beta eigenfrequency-gradient gate in
``docs/_static/vmec_boozer_shaped_pressure_solver_frequency_gradient_gate.json``
and the shaped-pressure finite-beta quasilinear-gradient gate in
``docs/_static/vmec_boozer_shaped_pressure_quasilinear_gradient_gate.json``.
It also requires the shaped-pressure finite-beta reduced nonlinear-window
estimator-gradient gate in
``docs/_static/vmec_boozer_shaped_pressure_nonlinear_window_gradient_gate.json``.
A tagged release must fail if these artifacts try to promote a direct
tensor-parity failure or a startup nonlinear-window response into a converged
nonlinear transport-gradient claim.
The same guard also checks the solver-objective content of each QH/Li383
gradient row: frequency rows must carry ``gamma`` and ``omega``; quasilinear
rows must additionally carry ``kperp_eff2``, ``linear_heat_flux_weight``, and
``mixing_length_heat_flux_proxy``; nonlinear-window estimator rows must also
carry the window mean, coefficient of variation, and trend metrics. The
release thresholds are ``5e-2`` for frequency rows, ``2e-2`` for quasilinear
rows, and ``7.5e-2`` for reduced nonlinear-window estimator rows. These are
AD/finite-difference consistency gates, not nonlinear turbulence-gradient
accuracy claims.

For release claims, the differentiable-geometry lane is closed only for
artifact-passing equal-arc parity rows, reduced QH/Li383
AD/finite-difference objectives, and the shaped-pressure finite-beta
eigenfrequency/quasilinear/reduced nonlinear-window estimator-gradient gates.
The fixed-resolution QI row and evaluated QI ``ntheta`` variants now pass, but
production nonlinear heat-flux optimization and finite-beta converged
transport-gradient gates are still open.
The active publication wording must keep these levels separate: the current
bridge starts at real ``vmex`` state coefficients and reaches GKX
solver observables, but it has not yet validated converged nonlinear
turbulence gradients, broad QI transport behavior, or nonlinear audits of
optimized equilibria.
The VMEC bridge now also expands environment variables in ``geometry.vmec_file``.
The shipped portable runtime TOMLs now point to relative ``wout_*.nc`` paths
under ``examples/vmec``. Generate those WOUT files locally from the bundled
``vmex`` input decks with ``examples/vmec/generate_wouts.sh`` or a single
``vmex input.<case>`` command. Environment variables and ``--vmec-file``
remain useful for machine-specific validation equilibria, but they are no
longer required for the bundled demos.
For future validation-lane selection, the external ``vmex`` example-data
portfolio can be inventoried without copying those VMEC files into this
repository. The tracked inventory artifact
``docs/_static/vmex_equilibrium_inventory.{png,pdf,json}`` is an
equilibrium-selection aid only. It excludes VMEC files with
degenerate reference-scale metadata from the recommended follow-up list, but it
still does not validate quasilinear transport until each selected VMEC
equilibrium also has matched linear and nonlinear GKX runs and physics
gates. The first bounded smoke checks have finite stable linear branches for
Li383, nfp4 QH, CTH-like, and shaped-tokamak fixtures from
``vmex/examples/data``; those checks only validate the runtime geometry and
quasilinear-feature plumbing, not nonlinear transport.
The nonlinear W7-X and HSX startup audits now confirm that this VMEC runtime
path reproduces GX startup ``g_state`` and ``phi`` to roundoff when the
generated ``*.eik.nc`` is rebuilt from the same VMEC input.
The late-time W7-X diagnostic-state audit now also matches GX on the exact
dumped nonlinear state once the comparison tool reconstructs the compressed
real-FFT positive-``ky`` dump grid directly from ``diag_state_ky_t*.bin``. The
tracked exact-state panel ``docs/_static/w7x_exact_state_audit.png`` records a
maximum finite pointwise relative error of ``4.62e-5`` under the explicit
``1e-4`` convention gate, with late scalar diagnostics below ``1.8e-7``. That
closes the remaining imported-geometry diagnostic-contract gap for nonlinear
VMEC cases: startup, ``phi``, ``kperp2``, ``fluxfac``, ``Wg``, ``Wphi``, and
heat flux all agree on the same GX state.
The follow-on exact-state linear audit on that same W7-X dump now also matches
GX to roundoff. The remaining operator-level fixes were:

- treat ``boundary = "fix aspect"`` and ``"continuous drifts"`` as GX linked
  twist-and-shift boundaries in the linear cache, and
- include the GX collision-conservation correction on top of the
  Lenard-Bernstein damping term.

With those in place, the imported VMEC/eik bridge, the late-time linear RHS,
and the late-time nonlinear E x B diagnostics all agree with GX on the same
dumped stellarator state. The final nonlinear W7-X free-run mismatch then
collapsed once the runtime de-alias mask matched GX exactly: the two-thirds
cutoff must be strict (``< 1/3``), not inclusive. With that correction, the
tracked stock-GX W7-X ``t = 200`` VMEC runtime rerun also passes the native
late-window comparison, so the shipped nonlinear W7-X example is now closed at
startup, exact-state, and long-horizon levels.

Tokamak Miller geometry now follows the same imported-geometry bridge pattern.
With ``geometry.model = "miller"``, the in-package geometry backend
constructs the Miller surface, straight-field-line and equal-arc grids, metric
and drift coefficients, writes a root-level ``*.eiknc.nc`` file, and then
re-enters the same imported-geometry contract described above. An existing
generated target is reused unless ``gkx geometry miller --force`` is
requested; this makes repeated runs cheap while keeping explicit regeneration
available after changing an input deck.
On the tracked Cyclone Miller parameters, the generated ``*.eiknc.nc`` file
matches the clean GX grouped ``Geometry`` arrays to roundoff in the main
metric and drift profiles. With the root-level open/closed theta inference
corrected, the clean-mainline Cyclone Miller late-state audit also now closes
on the exact dumped GX state: ``kperp2``, ``fluxfac``, ``phi``, ``Wg``,
``Wphi``, and heat flux all match to roundoff on the same nonlinear state.

Two user-facing entry points now exercise that bridge:

- ``gkx geometry vmec --config ...`` generates a compatible
  ``*.eik.nc`` file from a GKX runtime TOML.
- ``gkx geometry miller --config ...`` generates a compatible
  Miller ``*.eiknc.nc`` file from a GKX runtime TOML, or reuses the
  existing target when its path is already populated.
- ``examples/nonlinear/non-axisymmetric/hsx_nonlinear_vmec_geometry.py`` and
  ``examples/nonlinear/non-axisymmetric/runtime_hsx_nonlinear_vmec_geometry.toml`` run a nonlinear
  adiabatic-electron ITG case on the bundled QHS VMEC input deck after its
  ``wout_NuhrenbergZille_1988_QHS.nc`` file is generated with ``vmex``.
  They still accept ``--vmec-file`` for exact HSX validation WOUTs while
  letting GKX generate and reuse the field-line geometry automatically.

VMEC and Miller runtime examples
--------------------------------

VMEC-driven stellarator runs:

.. code-block:: bash

   cd examples/vmec
   vmex input.nfp3_QI_fixed_resolution_final
   cd ../..
   gkx run-runtime-nonlinear \
     --config examples/nonlinear/non-axisymmetric/runtime_w7x_nonlinear_vmec_geometry.toml \
     --steps 200 \
     --out tools_out/w7x_vmec.out.nc

   cd examples/vmec
   vmex input.NuhrenbergZille_1988_QHS
   cd ../..
   gkx run-runtime-nonlinear \
     --config examples/nonlinear/non-axisymmetric/runtime_hsx_nonlinear_vmec_geometry.toml \
     --steps 200 \
     --out tools_out/hsx_vmec.out.nc

Miller geometry runs:

.. code-block:: bash

   gkx run-runtime-nonlinear \
     --config examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear_miller.toml \
     --steps 200 \
     --out tools_out/cyclone_miller.out.nc

Imported geometry currently bypasses analytic twist-shift reconstruction and
uses the provided grid as-is. That keeps the GX-import bridge honest while the
native VMEC path is still being generalized.
