Input Files and Executable
==========================

GKX supports lightweight TOML inputs that map directly onto the
``GridConfig``, ``TimeConfig``, ``GeometryConfig``, and ``ModelConfig`` dataclasses.
You can use these inputs from the executable or from a Python driver.

Unified Runtime Schema
----------------------

In addition to benchmark-case TOMLs, GKX supports a **case-agnostic**
runtime schema (``RuntimeConfig``) with explicit species and physics toggles.
This allows Cyclone/ETG/KBM to run through the same solver path without
changing solver internals.

Minimal runtime TOML example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: toml

   [[species]]
   name = "ion"
   charge = 1.0
   mass = 1.0
   density = 1.0
   temperature = 1.0
   tprim = 2.49
   fprim = 0.8
   kinetic = true

   [physics]
   linear = true
   nonlinear = false
   electrostatic = true
   electromagnetic = false
   adiabatic_electrons = true
   tau_e = 1.0

   [normalization]
   contract = "cyclone"
   diagnostic_norm = "rho_star"

   [terms]
   streaming = 1.0
   mirror = 1.0
   curvature = 1.0
   gradb = 1.0
   diamagnetic = 1.0
   collisions = 1.0
   hypercollisions = 1.0
   end_damping = 1.0
   apar = 0.0
   bpar = 0.0
   nonlinear = 0.0

   [run]
   ky = 0.3
   Nl = 24
   Nm = 12
   solver = "auto"
   fit_signal = "auto"

Quasilinear diagnostics
^^^^^^^^^^^^^^^^^^^^^^^

Linear runtime runs can compute electrostatic quasilinear transport weights
directly from the final linear state or Krylov eigenvector:

.. code-block:: toml

   [quasilinear]
   enabled = true
   mode = "weights"                    # weights | saturated
   saturation_rule = "none"            # none | mixing_length | lapillonne_2011
   amplitude_normalization = "phi_rms" # phi_rms | phi_midplane | field_energy
   kperp_average = "phi_weighted"
   csat = 1.0
   gamma_floor = 0.0
   include_stable_modes = false
   channels = ["es"]

The current validated output level is **linear weights** plus optional
uncalibrated electrostatic saturation rules. Electromagnetic quasilinear
channels are intentionally rejected until their channel normalization and
nonlinear calibration gates are added. The runtime writes
``*.quasilinear.summary.json`` and ``*.quasilinear_species.csv`` when
``[output].path`` or ``--out`` is set. Serial ``scan-runtime-linear`` runs also
write ``*.quasilinear_spectrum.csv``; batched scans are intentionally disabled
for quasilinear output until per-ky state extraction has a numerical identity
gate. In scan spectra, the ``ky`` column is the requested scan coordinate and
``mode_ky`` records the signed selected grid-mode coordinate; they can differ
for linked-boundary grids, so publication plots should use ``ky`` while audits
can inspect ``mode_ky``.

Equivalent executable flags are available for single-point runtime runs:

.. code-block:: bash

   gkx run-runtime-linear \
     --config examples/linear/axisymmetric/cyclone.toml \
     --quasilinear \
     --ql-mode saturated \
     --ql-saturation-rule mixing_length \
     --ql-csat 1.0 \
     --out tools_out/cyclone_quasilinear

Minimal TOML example
--------------------

.. code-block:: toml

   case = "cyclone"
   reference_alignment = true

   [grid]
   Nx = 1
   Ny = 24
   Nz = 96
   Lx = 62.8
   Ly = 62.8
   boundary = "linked"
   y0 = 20.0
   ntheta = 32
   nperiod = 2

   [time]
   t_max = 10.0
   dt = 0.002
   use_diffrax = true
   diffrax_solver = "Dopri8"
   diffrax_adaptive = true
   diffrax_rtol = 1.0e-6
   diffrax_atol = 1.0e-8
   state_sharding = "auto"
   compressed_real_fft = true

   [run]
   ky = 0.3
   Nl = 24
   Nm = 12
   solver = "auto"
   method = "imex2"

   [fit]
   auto_window = true
   window_method = "loglinear"
   fit_signal = "auto"

The ``[time]`` section also accepts ``compressed_real_fft`` (default ``true``) to
select the compressed real-FFT nonlinear bracket. Set ``compressed_real_fft = false`` to
use a full complex FFT for the nonlinear term. Diagnostics output can be
decimated with ``sample_stride`` (record every ``N`` steps) and
``diagnostics_stride`` (compute streaming diagnostics every ``N`` steps). Set
``diagnostics = false`` in ``[time]`` (or ``--no-diagnostics`` on the executable) to
disable diagnostics entirely for speed. For CFL-controlled timestep control, use
``fixed_dt = false`` along with ``cfl`` and optional ``cfl_fac`` /
``dt_min`` / ``dt_max`` limits. When ``cfl_fac`` is omitted, GKX uses
the benchmark-locked method default instead of a universal constant:
``rk3``/``sspx3`` use ``1.73``, ``rk4`` uses ``2.82``, and other methods keep
``1.0``. When adaptive timestepping is enabled, diagnostics include
``dt_t`` (per-sample timestep history) and ``dt_mean`` (average effective dt)
to quantify CFL-driven savings. In benchmark-locked nonlinear runs the adaptive
``dt`` estimate combines the linear frequency cap with the instantaneous
nonlinear cap, matching the tracked comparison CFL update instead of using the nonlinear
bracket alone. To control the Laguerre handling in nonlinear
brackets, set ``laguerre_nonlinear_mode = "grid"`` (reference quadrature,
default) or ``laguerre_nonlinear_mode = "spectral"`` (use spectral ``Jl``
without the quadrature transform).
Use ``nonlinear_dealias = false`` to disable nonlinear dealias masking for
reference/debug runs where you want to preserve all configured base modes.
When ``nonlinear_dealias = true``, nonlinear runtime mode selection is
dealias-aware: if the requested ``ky`` is filtered out by the 2/3 mask, the
runner automatically picks the nearest retained ``ky``. The executable prints the
effective ``ky_sel``/``kx_sel`` used by diagnostics.
For benchmark-locked runs, leaving ``dt_max`` unset keeps ``dt_max = dt``.
Set ``state_sharding = "auto"`` (or ``"ky"``) to enable distributed
parallelization of the packed state array over multiple JAX devices. This is
honored by the sharding-aware integration paths, including the fixed-step RK2
nonlinear identity/profiler lane; unsupported solver paths or one-device runs
fall back to single-device execution. Other valid values are ``"kx"``, ``"z"``,
``"l"``, ``"m"``, and ``"species"``. Treat this as a correctness-gated
parallelization option unless the run also has a matched scaling artifact.
Increase ``dt_max`` explicitly only when you intentionally trade strict
comparison matching for throughput.

Diagonal nonlinear hypercollision splitting is enabled with
``collision_split = true``. The conserving collision operator remains in the
RHS because its field-particle correction is not diagonal. The
``collision_scheme`` key selects the hypercollision update:
``implicit`` (backward-Euler), ``exp`` (exact diagonal exponential), and
``sts``/``rkc`` aliases (treated as stabilized explicit/exponential updates for
diagonal operators).

The ``[geometry]`` section supports ``drift_scale`` to switch between benchmark-compatible
(``drift_scale = 1.0``) and the alternate doubled-drift convention (``drift_scale = 2.0``). The default configuration in GKX uses the tracked benchmark value.
The physical meaning of the runtime terms and geometry coefficients is detailed
in :doc:`theory` and :doc:`operators`; the TOML layer here documents how those
implemented models are selected and parameterized.
For slab benchmarks, set ``model = "slab"``. Optional slab-specific keys are
``z0`` (sets ``gradpar = 1/z0`` when positive, matching the reference slab domain
normalization) and ``zero_shat = true`` (forces the zero-shear slab metric
``gds2 = 1, gds21 = 0, gds22 = 1``).
It also accepts ``model = "imported-netcdf"`` with
``geometry_file = "external_geometry.nc"`` to run from imported sampled
field-line geometry instead of the analytic ``s-alpha`` model. The imported
file can be a tracked benchmark ``*.out.nc`` or a root-level ``*.eik.nc`` geometry
file produced by the VMEC workflow. When that imported geometry is used with a
linked boundary, GKX now follows the file's own ``theta`` range,
``jtwist/x0`` geometry factor, and ``kxfac`` metadata instead of forcing the
analytic s-alpha grid defaults.
For direct VMEC workflows, the runtime also accepts ``model = "vmec"``.
In that mode GKX calls the VMEC geometry helper to generate a
matching ``*.eik.nc`` file on demand, then immediately reuses the same imported
geometry path as the VMEC examples. Set ``vmec_file`` plus the flux-tube keys
``torflux``, ``npol`` and optionally ``alpha``. ``geometry_file`` can be used
as an explicit output path for the generated ``*.eik.nc`` file, and
``geometry_helper_repo`` can point to a non-default helper checkout if needed.
The preferred VMEC path is the internal ``booz_xform_jax`` backend, discovered from
``BOOZ_XFORM_JAX_PATH`` or ``GKX_BOOZ_XFORM_JAX_PATH`` when it is not
installed into the active Python environment. This is now the recommended
imported-geometry route for new stellarator cases. The shipped VMEC TOMLs
point to locally generated files under ``examples/vmec``; generate those files
with ``vmex input.<case>`` before running the examples.
``vmec_file`` supports ``$ENV_VAR`` expansion, and relative paths are resolved
against the TOML directory first. Command-line overrides are resolved from the
shell working directory.
Use ``geometry_helper_repo`` and ``geometry_helper_python`` when an imported
geometry helper checkout or interpreter must be selected explicitly.
When ``geometry_file`` is set for ``model = "vmec"``, GKX regenerates
that target instead of reusing a stale file from an older VMEC conversion.
For VMEC ``fix aspect`` runs, GKX follows the helper default contract and
does not inject ``x0`` from the runtime ``Lx``. That keeps the generated
``*.eik.nc`` file aligned with the imported W7-X/HSX geometry output.

For Miller tokamak workflows, the runtime also accepts ``model = "miller"``.
In that mode GKX uses its in-package Miller backend to generate a matching
root-level ``*.eiknc.nc`` file, then immediately re-enters the same imported
geometry path used for VMEC ``eik.nc`` files.
Set the Miller inputs directly in ``[geometry]``:
``rhoc``, ``q``, ``s_hat``, ``R0``, optional ``R_geo``, ``shift``,
``akappa``, ``akappri``, ``tri``, ``tripri``, and ``betaprim``.
The generated Miller surface has inverse aspect ratio ``rhoc / R0``. Set that
ratio explicitly when reproducing a benchmark; the separate ``epsilon`` field
belongs to analytic geometry metadata and does not replace the surface radius.
``geometry_file`` can be used as an explicit output path for the generated
Miller ``*.eiknc.nc`` file. Existing output is reused by default; pass
``--force`` to ``gkx geometry miller`` to regenerate it. The
``geometry_helper_python`` and ``geometry_helper_repo`` settings apply to VMEC
helper discovery, not to this in-package Miller backend.

Executable path overrides
^^^^^^^^^^^^^^^^^^^^^^^^^

The ``gkx`` and ``gkx`` executables accept path overrides for
runtime-configured runs:

.. code-block:: bash

   gkx run --config case.toml --vmec-file examples/vmec/wout_circular_tokamak.nc
   gkx run --config case.toml --geometry-file external_geometry.eik.nc
   gkx run-runtime-nonlinear --config case.toml --init-file ~/restart.nc

These override paths expand ``~`` and environment variables and are resolved
against the shell's current working directory. TOML paths still resolve against
the config file directory. ``--vmec-file`` replaces ``[geometry].vmec_file`` for
VMEC-backed configs. ``--geometry-file`` replaces ``[geometry].geometry_file``
only; it does not change ``[geometry].model``. For imported-geometry configs,
that file is the imported EIK/NetCDF input. For ``model = "vmec"``, it remains
the generated EIK target/cache path.

Solver and fit-signal keys
--------------------------

The ``[run]`` and ``[scan]`` sections accept ``solver`` and ``fit_signal`` keys:

* ``solver = "auto"`` (default): choose time vs Krylov and fall back if needed
* ``solver = "time"``: always use time integration
* ``solver = "explicit_time"``: force the explicit single-mode time integrator
  used by controlled explicit-time comparisons
* ``solver = "krylov"``: always use the matrix-free eigen solver

* ``fit_signal = "auto"`` (default): pick ``phi`` vs density based on fit quality
* ``fit_signal = "phi"``: use the electrostatic potential time trace
* ``fit_signal = "density"``: use the density moment time trace

For large ky scans, ``scan-runtime-linear --batch-ky`` integrates all ky values
in a single time integration pass (time integrator only) and then extracts the
growth rates from the per-ky traces.

For quasilinear spectra, use ``scan-runtime-linear --workers N`` instead of
``--batch-ky``. This runs independent per-``ky`` solves, computes the
quasilinear state extraction for each mode, preserves serial spectrum ordering,
and records the worker identity contract in the scan summary JSON. Combined
``--batch-ky`` quasilinear artifacts remain disabled until the batched
state-extraction identity gate is separately closed.

Executable usage
----------------

.. code-block:: bash

  cd examples/linear/axisymmetric && gkx cyclone.toml
  gkx scan-runtime-linear --config examples/linear/axisymmetric/runtime_etg.toml --out docs/_static/runtime_etg_scan
  gkx --plot docs/_static/runtime_etg_scan.scan.csv
  gkx run-runtime-linear --config examples/linear/axisymmetric/cyclone.toml --out tools_out/cyclone_runtime
   gkx scan-runtime-linear --config examples/linear/axisymmetric/runtime_etg.toml --batch-ky
   gkx run-runtime-nonlinear --config examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml --sample-stride 5 --out docs/_static/nonlinear_cyclone_diag.csv

For ``run-runtime-nonlinear``, omit ``--steps`` when ``fixed_dt = false`` unless
you explicitly want a capped step count. The executable now preserves ``steps = None``
for adaptive nonlinear runs so the runtime can keep integrating in chunks until
it reaches the requested ``t_max`` instead of silently reverting to the old
``round(t_max / dt)`` ceiling.

Diagnosed nonlinear runs stop at saturation by default (``[time] run_to =
"saturation"``): the runtime integrates in chunks and, after each chunk, tests
the accumulated heat-flux trace with spin-up excluded. It stops once the
autocorrelation-corrected relative SEM of the windowed mean is at or below
``saturation_rel_sem``, the window is at least ``saturation_min_window`` long,
and the first and second halves of the window agree within twice their combined
SEM. The same stationarity test guards the electrostatic field energy ``Wphi``
and gyrokinetic free energy ``Wg``. ``t_max`` remains the hard cap, so a run
that never saturates behaves exactly as before.
The chosen window and its ``mean +/- SEM`` are printed in the run summary and
recorded under ``saturation`` in the summary JSON. Pass
``--no-until-saturated`` (or set ``run_to = "t_max"``) for a fixed horizon.

For single-point runtime commands, artifact output can be requested either with
``--out`` or directly in the runtime TOML:

.. code-block:: toml

   [output]
   path = "tools_out/runtime_case"

``--out`` takes precedence over ``[output].path`` when both are provided.
Linear runs write a JSON summary and, when a fitted signal is available, a
``*.timeseries.csv`` sidecar with ``t,signal_real,signal_imag,signal_abs``.
Nonlinear runs write a JSON summary plus a diagnostics CSV. When the requested
path already ends in ``.csv``, that exact filename is used for the diagnostics
table and the JSON summary is written next to it as ``*.summary.json``.

If the nonlinear output path ends in ``.out.nc`` (recommended) or another
``.nc`` suffix, the runtime switches to NetCDF restart/diagnostic artifacts
instead of the lightweight JSON/CSV pair. In that mode GKX writes:

* ``*.out.nc``: diagnostic history together with ``Grids``, ``Geometry``, and
  ``Inputs`` groups.
* ``*.big.nc``: final fields and moments in spectral and real-space layouts.
* ``*.restart.nc``: restart state for continuation runs.

See :doc:`outputs` for the detailed variable inventory.

The nonlinear diagnostics CSV base columns are
``t,dt,gamma,omega,Wg,Wphi,Wapar,energy,heat_flux,particle_flux`` and
species-resolved columns are appended when available:
``heat_flux_s{i}``, ``particle_flux_s{i}`` for species index ``i``.
When turbulent-heating diagnostics are present, the CSV also includes
``turbulent_heating`` and ``turbulent_heating_s{i}``.

Python driver
-------------

.. code-block:: bash

  python examples/utilities/runtime_from_toml.py --config examples/linear/axisymmetric/etg.toml

The canonical KBM TOML is currently a controlled comparison input rather than
a promoted standalone solve: its experimental shift-invert branch raises when
the physical eigenpair-residual gate is not met. Reproduce the reviewed result
with ``python benchmarks/kbm_linear_comparison.py``.

TOML sections
-------------

Supported sections include:

* ``[grid]`` (``GridConfig``)
* ``[time]`` (``TimeConfig``)
* ``[geometry]`` (``GeometryConfig``)
* ``[model]`` (case-specific model config)
* ``[init]`` (``InitializationConfig``)
* ``[run]`` (single-ky run settings)
* ``[scan]`` (ky scan settings)
* ``[fit]`` (growth-rate windowing options)
* ``reference_alignment`` (top-level flag or ``[reference_alignment] enabled = true`` to
  enforce the tracked comparison defaults)
* ``[terms]`` (toggle linear terms)
* ``[krylov]`` (Krylov solver settings)

Runtime sections
^^^^^^^^^^^^^^^^

For runtime-configured inputs (``load_runtime_from_toml``), supported sections
are:

* ``[[species]]`` (kinetic species definitions)
* ``[physics]`` (electrostatic/electromagnetic, adiabatic/kinetic, linear/nonlinear)
* ``[collisions]`` (collision and hypercollision controls)
* ``[normalization]`` (contract key + optional overrides)
* ``[terms]`` (term toggles used by modular RHS assembly)
* ``[expert]`` (advanced fixed-mode controls for specialized workflows)
* ``[output]`` (artifact path for single-point runtime commands)
* ``[parallel]`` (parallelization policy for independent scans and future
  sharded paths; defaults to serial)
* ``[run]`` / ``[scan]`` / ``[fit]`` (driver controls)

Notable runtime-only keys:

* ``[collisions] damp_ends_amp`` / ``damp_ends_widthfrac``: reference-compatible end
  damping defaults are ``0.1`` and ``0.125``.
* ``[physics] reduced_model``: physics-family selector for runtime inputs.
  The maintained runtime supports full gyrokinetics via ``"gyrokinetic"``
  and its full-GK aliases. Non-promoted reduced-model values fail closed with
  ``NotImplementedError`` instead of silently routing through the wrong equations.
* ``[collisions] damp_ends_scale_by_dt``: opt-in per-step damping-rate scaling
  for controlled reproduction studies. The reference-compatible default is ``false`` because
  ``damp_ends_amp`` is already a per-unit-time damping rate.
* ``[collisions] hypercollisions_const`` / ``hypercollisions_kz``: defaults are
  the reference-compatible ``0.0`` / ``1.0`` (kz-proportional hypercollisions enabled by
  default, constant hypercollisions off).
* ``[collisions] p_hyper_m``: when omitted, the runtime path uses the
  resolution-aware default ``min(20, Nm/2)`` instead of a fixed exponent across
  Hermite resolutions.
* ``[collisions] nu_hermite`` / ``nu_laguerre``: coefficients entering the
  Lenard-Bernstein collision eigenvalue used by the modular collision kernel.
* ``[collisions] nu_hyper`` / ``p_hyper``: isotropic hypercollision amplitude
  and exponent.
* ``[collisions] nu_hyper_l`` / ``nu_hyper_m`` / ``nu_hyper_lm`` and
  ``p_hyper_l`` / ``p_hyper_m`` / ``p_hyper_lm``: Laguerre-only,
  Hermite-only, and mixed hypercollision channels.
* ``[collisions] D_hyper`` / ``p_hyper_kperp``: perpendicular hyperdiffusion
  amplitude and exponent.
* ``[normalization] flux_scale``: multiplicative factor applied to
  heat/particle flux diagnostics (tracked comparison default ``1.0``).
* ``[normalization] wphi_scale``: multiplicative factor applied to ``Wphi``
  diagnostics (the Cyclone nonlinear comparison config uses ``1.155``).
* ``[normalization] rho_star``: rescales the operator wavenumbers before
  building the drift/Bessel terms.
* ``[normalization] omega_d_scale`` / ``omega_star_scale``: multiplicative
  normalization factors for magnetic-drift and diamagnetic-drive terms.
* ``[init] init_single`` with ``gaussian_init = false`` and ``init_single = false``:
  initialize a random perturbation across the configured startup-loop bounds in
  ``(ky,kx)``.
* ``[init] init_single`` with ``gaussian_init = true`` and ``init_single = false``:
  initialize a Gaussian envelope across the same startup-loop bounds.
* ``[init] init_single = true``:
  initialize only the selected ``(ky,kx)`` mode. When combined with
  ``init_field = "phi"`` and ``gaussian_init = true``, the selected
  electrostatic-potential mode is initialized with a Gaussian profile along the
  flux tube; this is the contract used by the W7-X zonal-flow response
  benchmark.
* ``[init] init_field = "all"``: the runtime/TOML path uses Hermite/Laguerre
  moment-normalized scaling for this initializer, with reduced amplitudes for ``tpar``
  (``1/sqrt(2)``) and ``qpar`` (``1/sqrt(6)``).
* ``[init] init_field = "phi"``: initialize a requested electrostatic
  potential profile by inverting the same quasineutrality solve used during
  time advance. This is the appropriate contract for zonal-response
  literature tests that prescribe ``phi(t=0)`` rather than a density-moment
  perturbation; the masked ``ky=0, kx=0`` gauge mode remains unavailable.
* ``[init] init_electrons_only``: if ``true`` in multispecies runs, initialize
  only electron species. If ``false`` (default), initialize all kinetic
  species.
* ``[init] random_seed``: RNG seed used for reference-compatible random initial
  conditions. The runtime follows a Linux ``glibc`` ``rand()`` sequence with
  positive-``kx``-major loop order and exact startup loop bounds, so random
  multi-mode perturbations are host-platform independent and reproduce the
  same seeded pattern on macOS and Linux.
* ``[init] init_file``: load a saved complex state from either the full-``ky``
  GKX layout or a packed positive-``ky`` interchange layout.
* ``[init] init_file_scale`` / ``init_file_mode``: scale a loaded restart state
  and either ``replace`` the analytic seed (default) or ``add`` it to the
  fresh perturbation. This is the general runtime equivalent of
  restart-scaling and restart-with-perturb workflows.
* ``[expert] fixed_mode`` with ``iky_fixed`` / ``ikx_fixed``: keep one Fourier
  mode exactly frozen during nonlinear evolution; this is the ``eqfix``-style
  contract used by the ``secondary`` benchmark.
* ``[expert] source`` / ``phi_ext``: runtime-only benchmark hooks for
  external electrostatic forcing. ``source = "phiext_full"`` with a small
  ``phi_ext`` injects the source into the solved ``phi`` field before the RHS is
  assembled, so it changes the actual evolution rather than only the saved
  diagnostics.  The Merlo Case-III Rosenbluth-Hinton/GAM example uses the
  literature protocol instead: ``source = "default"`` plus an initial ion
  density perturbation.
* ``[time] run_to``: nonlinear stop policy. ``"saturation"`` (the default)
  ends a diagnosed nonlinear run as soon as the post-spin-up heat-flux window
  has converged, with ``t_max`` kept as the hard cap; ``"t_max"`` always
  integrates the full horizon and reproduces the pre-auto-stop behavior
  exactly. Linear runs and nonlinear runs with ``diagnostics = false`` ignore
  the key, because the stop test needs the streamed heat-flux trace. The
  executable overrides are ``--until-saturated`` and ``--no-until-saturated``.
  Set ``"t_max"`` explicitly on any deck whose heat flux is not the observable
  it exists to produce -- zonal-flow and secondary-instability responses,
  linear-response diagnostics -- because a horizon chosen by a quantity that is
  zero by construction is not a horizon. Which shipped decks do that, and the
  measurement clearing the rest to run under the default, are held by the
  ``run_to`` audit in ``tests/release/test_release_gates.py``.
* ``[time] saturation_rel_sem``: target relative standard error of the mean on
  the windowed heat flux (default ``0.05``, i.e. 5%). The SEM is corrected for
  autocorrelation via the Sokal integrated autocorrelation time, so it is the
  error of the physical time average rather than of independent samples.
* ``[time] saturation_min_window``: minimum averaging-window span in time
  units before a run may stop. When omitted, the requirement is derived from
  the trace itself as ten integrated autocorrelation times.
* ``[time] nstep_restart``: when writing a nonlinear NetCDF bundle,
  checkpoint every ``nstep_restart`` steps instead of waiting for the end of
  the run. This is useful for long adaptive runs and batch jobs.
* ``[time] method = "sspx3"``: use the SSPx3 explicit scheme directly. This is
  the relevant explicit method for ``secondary`` and collisional-ETG benchmark
  families. Plain ``rk3`` now follows the three-stage Heun-style timestepper;
  ``rk3_classic`` keeps the older classical RK3 update if you need it for
  controlled comparisons, and ``rk3_heun`` remains as an explicit alias.
* ``[terms]``: each key is a pure multiplicative operator weight:
  ``streaming``, ``mirror``, ``curvature``, ``gradb``, ``diamagnetic``,
  ``collisions``, ``hypercollisions``, ``hyperdiffusion``, ``end_damping``,
  ``apar``, ``bpar``, ``nonlinear``.

Runtime parallelization controls
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``[parallel]`` section is parsed by ``RuntimeParallelConfig`` and is
serial by default:

.. code-block:: toml

   [parallel]
   strategy = "serial"
   axis = "ky"
   batch_size = 2
   num_devices = 2
   strict_identity = true
   profile = false
   backend = "auto"

Current accepted strategies are ``"serial"``, ``"batch"``,
``"combined_ky"``, ``"device_batch"``, ``"pmap"``, ``"pjit"``,
``"shard_map"``, ``"state"``, and ``"velocity"``. Strategy
``"batch-ky"`` is accepted as an alias for ``"combined_ky"`` and selects the
existing combined-``k_y`` time-integration scan path. Quasilinear scan
artifacts still require serial per-``k_y`` evaluation until the per-mode state
extraction has its own numerical-identity gate.

For runtime ``k_y`` scans, ``strategy = "batch"`` with ``axis = "ky"`` selects
the production independent-worker path. ``num_devices`` is interpreted as the
requested worker count when the executable call does not explicitly pass
``workers``; ``backend = "thread"`` or ``"process"`` selects the executor, and
``"auto"`` keeps the executable default. The runtime output records whether
the worker policy came from the TOML file or from explicit executable
arguments, together with the requested/effective worker counts and the
ordering-preservation identity contract.

The only velocity-space RHS route exposed at this stage is deliberately
diagnostic: ``strategy = "velocity"``, ``axis = "hermite"``, and
``backend = "streaming_only"``, ``backend = "streaming_electrostatic"``, or
``backend = "electrostatic_linear_slices"``. ``backend = "auto"`` selects the
electrostatic-slices route when the active terms satisfy that gate; otherwise
the runtime raises instead of silently falling back to an unvalidated path.
These backends are accepted only by ``gkx.linear_rhs_parallel_cached``.
The first two require all non-streaming linear terms disabled. The
electrostatic-slices backend allows only streaming, mirror, curvature, grad-B,
and diamagnetic-drive weights; collisions, linked boundaries, electromagnetic
terms, and nonlinear terms remain disabled until their own identity gates are
added. Current velocity RHS routes are limited to single-species periodic 5D
electrostatic states.

For full runtime TOML files this velocity-space route is exposed only through
the fixed-step linear executable path with ``fit_signal = "phi"``. Diffrax
linear runs and density-assisted automatic fitting stay serial until they have
their own identity gates, so requesting velocity parallelization there raises a
clear error.

For independent scan, sensitivity, and UQ workloads, use
``gkx.batch_map`` for JAX-array maps and ``gkx.independent_map``
for file-backed Python tasks such as calibration rows or leave-one-out UQ
holdouts. Require a serial identity artifact before using timing results in a
publication claim. The helpers preserve ordering, so diagnostics such as growth
rates, frequencies, quasilinear weights, and covariance summaries can be
carried through one parallel map.

Runtime output and restart controls
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``[output]`` section controls runtime artifact layout and restart behavior:

* ``path``: artifact target. Use a plain prefix such as
  ``tools_out/runtime_case`` for JSON/CSV sidecars, or ``*.out.nc`` for a
  nonlinear NetCDF restart bundle.
* ``restart``: force loading from ``restart_from_file`` or from the derived
  sibling ``*.restart.nc`` next to ``path``. Raise an error if the restart
  file is missing.
* ``restart_if_exists``: opportunistically resume from an existing restart file
  without requiring one to be present.
* ``save_for_restart``: write the ``*.restart.nc`` checkpoint when a nonlinear
  NetCDF bundle is requested.
* ``restart_to_file`` / ``restart_from_file``: explicit checkpoint paths when
  the default sibling naming is not desired.
* ``restart_with_perturb``: combine the loaded restart state with a fresh
  analytic seed instead of fully replacing it. Internally this maps onto
  ``init_file_mode = "add"``.
* ``restart_scale``: multiplicative scale applied to the loaded restart state.
* ``append_on_restart``: append continued diagnostic history to the existing
  ``*.out.nc`` file instead of replacing it.
* ``warm_start``: carry a converged state from one point of a *repeated*
  workload to the next instead of restarting every point cold. It is opt-in
  (default ``false``) and has no effect on a single run, which has no
  predecessor. See `Warm-started scans`_ below.
* ``resolved_diagnostics``: materialize mode-resolved spectra in nonlinear
  diagnostic history. It defaults to ``true`` for publication and restart
  artifacts. Set it to ``false`` when only scalar time traces such as heat and
  particle fluxes, free energy, field energy, growth rate, and frequency are
  required. The compact path preserves those scalar channels while avoiding
  the larger spectral-history arrays.
* ``nsave``: checkpoint cadence fallback, in steps, for nonlinear NetCDF
  bundles when ``time.nstep_restart`` is not set.
* ``plots``: draw the run's figures beside its output once it finishes. It
  defaults to ``true``; set it to ``false`` (or pass ``--no-plots``) on batch
  surfaces that only want the data. See :doc:`outputs` for the figure set each
  run kind writes.

For direct restart control outside the ``[output]`` helper path, the generic
``[init] init_file`` / ``init_file_scale`` / ``init_file_mode`` keys remain the
lower-level mechanism.

Warm-started scans
^^^^^^^^^^^^^^^^^^

``[output] restart`` reuses a state written to disk by an earlier run.
``[output] warm_start`` is its in-memory sibling for work that repeats inside
one invocation: a converged eigenmode at one ``ky`` is an excellent initial
guess at a neighbouring ``ky``, so the next point can start from the answer
instead of rediscovering it from the analytic seed. How much that is worth
depends entirely on the solver, and :doc:`performance` measures both cases
before you decide.

``gkx scan-runtime-linear`` therefore walks the requested ``ky`` list in
monotone order -- so that neighbours follow neighbours regardless of the order
you asked for -- and seeds each point from the previous point's final state.
On the eigensolver paths that state is the converged eigenvector and becomes
the Krylov starting vector; on the time-integration paths it is the final
distribution function. Results are written back into the order you requested,
so the reported arrays are unchanged in shape, ordering, and meaning.

The carried state is rescaled by an exact power of two before reuse. Both the
eigensolve and the growth-rate fit are invariant to the overall scale of the
initial condition, so this loses nothing, and it stops a state that has grown
by ``exp(gamma T)`` from overflowing. A state that comes back non-finite or
zero is refused and the next point starts cold.

Turn it on with ``--warm-start``, with ``[scan] warm_start = true``, or with
``[output] warm_start = true``; the flag wins over ``[scan]``, which wins over
``[output]``. It is opt-in rather than the default because it is not free and
not always identical -- see :doc:`performance` for the measurements. GKX also
declines to warm start on its own when carrying state would break a contract
you asked for:

* ``--workers`` greater than one, or ``[parallel] strategy = "batch"``, because
  independent workers must stay independent;
* ``solver = "explicit_time"``, which cannot return a final state;
* ``--batch-ky`` and ``[parallel] strategy = "combined_ky"``, which already
  integrate every ``ky`` in one pass and so have nothing to carry.

**Two things to know before switching it on.** First, on a *fixed* horizon a
warm-seeded time integration reports a different growth rate than a cold one,
because it has removed a startup transient the cold run still contains; the two
agree only once the horizon is long enough that both are converged, which is
also why the GX-parity decks are left cold. Second, an eigensolver started
inside one branch can converge to that branch. On a case whose branch structure
you do not already know, run the scan once with ``--no-warm-start`` and compare
point by point before trusting the warm numbers.

``run_runtime_parameter_scan`` takes the same idea as an opt-in
``warm_start=True``: the previous point's state is reused only while the
parameter moves by no more than ``warm_start_max_step`` (relative, default
0.25) from the point before it. That is weaker than ``continuation=True``,
which follows one branch and requires every point to return a state.

For the explicit equations attached to these controls, see :doc:`operators`.
