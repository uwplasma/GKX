Input Files and Executable
==========================

Every GKX input is one runtime TOML deck, loaded by
``gkx.load_runtime_from_toml`` into a ``RuntimeConfig``. The ``[grid]``,
``[time]``, ``[geometry]`` and ``[init]`` tables merge into ``GridConfig``,
``TimeConfig``, ``GeometryConfig`` and ``InitializationConfig``; species,
physics toggles, collisions, normalization, operator terms, output, quasilinear
and parallel policy have their own tables (listed under `Runtime sections`_).
The same deck drives the executable and the Python API, for Cyclone, ETG, KBM
or any other case, through one solver path.

Unified Runtime Schema
----------------------

Resolution and time-step defaults
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Velocity resolution.** When neither the call nor the deck names
  ``Nl`` (Laguerre) and ``Nm`` (Hermite), the fallback is
  ``(Nl, Nm) = (12, 24)`` for linear runs and ``(4, 8)`` for nonlinear runs;
  a deck with kinetic electrons gets per-species counts instead (see
  :ref:`per-species-moments`). Set both explicitly for any result you report.
- **ky layout.** ``[grid] ky_layout = "half"`` (default) stores the ``1 + Ny // 2``
  non-negative rows; ``"full"`` stores the two-sided ``ky`` axis of length
  ``Ny`` (``gkx.core_ky_layout``). Results do not depend on the choice: they
  agree to round-off, and ``"half"`` is faster and smaller, in the nonlinear
  step and in the heat-flux window gradient.
- **Time stepping.** A deck that sets ``[time] dt`` runs ``rk2`` at that fixed
  step unless it names another ``method``. A deck without ``dt`` gets
  ``method = "rk4"`` with ``fixed_dt = false``: the CFL controller chooses the
  step with ``cfl = 0.9``. An explicit ``method`` or ``fixed_dt`` in the deck
  always wins.

Runtime precision
^^^^^^^^^^^^^^^^^

The default is **float32** (``complex64`` states).
``JAX_ENABLE_X64=true``, set before JAX is imported, selects double precision.
The runtime's own initial state is then built in ``complex128``, so the Krylov
basis, the eigenvector and the certification gate are all float64. This is what
the shipped decks mean when their header says to run them under
``JAX_ENABLE_X64`` for parity reproduction.

What x64 buys is the **gate**, not necessarily the eigenvalue.
``certifiable_residual_tolerance`` floors the eigenpair residual gate at
``1e3 * eps(dtype)``, so a float32 run certifies against 1.19e-4 while a float64
run certifies against 1e-9. Use float64 when the number you are quoting has to
carry a float64-scale residual.

A float64 run on a deck too coarse to support a certifiable eigenpair raises
instead of returning an uncertified number. If that happens, raise the
velocity resolution (``Nl``, ``Nm``).

``GKX_X64`` is a **test-harness** variable, not a runtime switch: it only selects
the precision banner in ``tests/conftest.py``. Setting it without
``JAX_ENABLE_X64`` does not change any run's precision.

``run_runtime_linear(cfg, initial_state=G128, ...)`` accepts an explicit
seed and preserves its dtype. It requires shape
``(kinetic_species, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size)`` on the
selected-ky runtime grid. Record the seed dtype and hash; casting a rounded f32
seed does not recover lost input precision. There is no TOML dtype option, and
prepared linear ``solve(initial_state=...)`` raises ``NotImplementedError``.

Minimal runtime TOML example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: toml

   schema_version = 1

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
   Nl = 16
   Nm = 48
   solver = "auto"
   fit_signal = "auto"

``Nl = 16, Nm = 48`` is the velocity resolution of the shipped Cyclone deck
``examples/01_linear_tokamak/case_full.toml``.

Schema compatibility
^^^^^^^^^^^^^^^^^^^^

Maintained runtime decks declare ``schema_version = 1`` at the document root.
Versionless GKX 1.8.2 decks remain readable as legacy schema 0 and are not
rewritten in place. Resolved decks written by the equilibrium shorthand are
schema 1. A non-integer or unsupported future version fails before numerical
work and points to this migration contract; upgrade GKX before reading a newer
schema. The removed Diffrax keys (``use_diffrax``, ``diffrax_solver``,
``diffrax_adaptive``, ``diffrax_rtol``, ``diffrax_atol``,
``diffrax_max_steps``) in ``[time]`` are refused with the native replacement
named in the error.
Keys that no code read are removed from the schema (``geometry_helper_python``,
``geometry_helper_repo``, ``tau_fac``, ``z_ion``, ``dealias_kz``, ``save_state``,
``[parallel] profile`` and ``[quasilinear] species``, ``delta_ky``,
``write_spectrum``). A deck from an earlier version that carries one at the
value that version wrote still loads; any other value is refused with the key
named.

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

The validated output level is **linear weights** plus optional
uncalibrated electrostatic saturation rules. Electromagnetic quasilinear
channels are rejected. The runtime writes
``*.quasilinear.summary.json`` and ``*.quasilinear_species.csv`` when
``[output].path`` or ``--out`` is set. Serial ``scan-runtime-linear`` runs also
write ``*.quasilinear_spectrum.csv``; combined-``ky`` batched scans refuse
quasilinear output. In scan spectra, the ``ky`` column is the requested scan coordinate and
``mode_ky`` records the signed selected grid-mode coordinate; they can differ
for linked-boundary grids, so publication plots should use ``ky`` while audits
can inspect ``mode_ky``.

Equivalent executable flags are available for single-point runtime runs:

.. code-block:: bash

   gkx run-runtime-linear \
     --config examples/01_linear_tokamak/case_full.toml \
     --quasilinear \
     --ql-mode saturated \
     --ql-saturation-rule mixing_length \
     --ql-csat 1.0 \
     --out tools_out/cyclone_quasilinear

Time-integration controls
-------------------------

Keys in ``[time]`` besides ``t_max``, ``dt`` and ``method``:

- ``compressed_real_fft`` (default ``true``): compressed real-FFT nonlinear
  bracket; ``false`` uses a full complex FFT and requires two-sided
  ``ky_layout = "full"`` storage. A half-spectrum state is refused.
- ``checkpoint`` (default ``false``): rematerialize fixed-step explicit
  nonlinear scans during reverse-mode differentiation. This reduces retained
  intermediates at the cost of recomputation, including diagnosed and prepared
  runs.
- ``sample_stride`` and ``diagnostics_stride``: step-based sampling controls.
  Nonlinear explicit diagnostics use the larger stride for fresh retained
  rows. ``diagnostics = false`` (or ``--no-diagnostics``) disables diagnostics.
- ``chunk_steps``: optional positive integer controlling the nonlinear runtime
  chunk length. When omitted, explicit chunks use 128 steps and adaptive
  chain-IMEX chunks use 32. A chunk boundary is a monitoring and host-handoff
  boundary; it does not request a new diagnostic timestamp or a restart file.
  It also enables chunking for diagnosed fixed-step runs to ``t_max``. These
  controls are nonlinear-only; unsupported execution routes raise.
- ``diagnostics_interval``: optional positive finite interval in simulation
  time for retained nonlinear diagnostics. It takes precedence over both
  step-stride controls. Samples use actual accepted-step times, with the first
  step at or after each interval threshold retained. Several thresholds crossed
  in one step produce one row. The final accepted state is retained once even
  when it falls between thresholds. See :doc:`outputs` for sampling semantics.
- ``fixed_dt = false`` with ``cfl`` (default ``0.9``) and optional
  ``cfl_fac`` / ``dt_min`` / ``dt_max``: CFL-controlled steps. When ``cfl_fac``
  is omitted, ``rk3``/``sspx3`` use ``1.73``, ``rk4`` uses ``2.82``, and other
  methods ``1.0``. Nonlinear runs use ``2.1`` for ``rk3`` and ``2.7`` for
  ``rk4`` up to ``Nl * Nm = 32``, and those values times ``(32 / (Nl * Nm))**0.6``
  beyond: about 60% of the smallest largest-stable factor measured (``rk3`` 3.5,
  ``rk4`` 4.5 at ``Nl * Nm <= 32``; 2.1 and 3.2 at 128, where the explicit
  bound underestimates the spectral radius). A nonlinear chunk that ends non-finite
  or whose free energy grows more than 10^4-fold is redone at half the factor
  with a warning, and the smaller factor is kept (three halvings at most,
  adaptive runs only). Adaptive runs record ``dt_t`` (per-sample step history) and
  ``dt_mean`` (average effective step). The nonlinear adaptive ``dt`` estimate
  combines the linear frequency cap with the instantaneous nonlinear cap.
  Leaving ``dt_max`` unset keeps ``dt_max = dt``; raise it explicitly only
  when you trade strict comparison matching for throughput.
- ``method = "auto"`` (the default of a nonlinear deck with no ``dt``): the run
  takes ``lawson-rk4-pi`` where the radial and binormal drift frequencies are at
  least 56% of the linear frequency bound (drift plus streaming) and the step cap
  (``dt_max``, else ``dt``) is at least 2.6 times the ``rk4-pi`` bound step, and
  ``rk4-pi`` elsewhere. It prints the choice, the share and the ratio
  (``stepper auto: ...``). The sharded
  routes (``[parallel]``, ``state_sharding``) and decks with per-species moment
  counts take ``rk4-pi``. Kinetic electrons at :math:`\beta \le 3\times10^{-4}`
  default to ``imex-ark3`` instead.
- ``method = "rk4-pi"`` (nonlinear, ``fixed_dt = false``): RK4 whose step is
  chosen from an embedded error estimate and capped at twice the CFL step; see
  :doc:`solvers`. ``cfl_fac`` defaults to ``2.82`` as for ``rk4``.
- ``method = "lawson-rk3"``, ``"lawson-rk4"`` or ``"lawson-rk4-pi"`` (nonlinear,
  explicit diagnostics): the integrating-factor forms of ``rk3``, ``rk4`` and
  ``rk4-pi``, which advance the curvature and grad-B drift exactly
  (:ref:`lawson-steppers`). ``auto`` selects ``lawson-rk4-pi``; name a ``lawson-*``
  method to force it. The CFL bound drops the radial and binormal drift
  frequencies and keeps streaming and the :math:`E\times B` rate. ``cfl_fac``
  defaults to ``2.5`` for ``lawson-rk3`` and ``lawson-rk4`` (scaled with
  ``Nl * Nm`` like the other explicit nonlinear methods) and ``2.82`` for
  ``lawson-rk4-pi``. They need one ``(Nl, Nm)`` for all species and the
  Laguerre-coefficient basis, and raise otherwise.
- ``laguerre_nonlinear_mode``: ``"grid"`` (reference quadrature, default) or
  ``"spectral"`` (spectral ``Jl`` without the quadrature transform) for the
  Laguerre handling in nonlinear brackets.
- ``nonlinear_dealias`` (default ``true``): 2/3 dealias masking. With masking
  on, nonlinear mode selection is dealias-aware: a requested ``ky`` removed by
  the mask is replaced by the nearest retained ``ky``, and the executable prints
  the effective ``ky_sel``/``kx_sel``. ``false`` keeps all configured modes for
  reference/debug runs.
- ``state_sharding``: ``"auto"`` or ``"ky"`` shards the packed state over
  multiple JAX devices along ``ky``; ``"kx"``, ``"z"``, ``"l"``, ``"m"`` and
  ``"species"`` shard along that axis. Sharding-aware integration paths honor
  it; unsupported solver paths and one-device runs fall back to single-device
  execution. Treat it as a correctness-gated option unless the run also has a
  matched scaling artifact.

Diagonal nonlinear hypercollision splitting is enabled with
``collision_split = true``. The conserving collision operator remains in the
RHS because its field-particle correction is not diagonal. The
``collision_scheme`` key selects the hypercollision update:
``implicit`` (backward-Euler), ``exp`` (exact diagonal exponential), and
``sts``/``rkc`` aliases (treated as stabilized explicit/exponential updates for
diagonal operators).
With ``method = "imex-ars3"`` or ``"imex-ark3"`` and the default Landau
operator, ``collision_scheme = "implicit"`` (the default) advances the full
conserving collision operator, field-particle part included, by a local
backward-Euler sub-step after each step (density, momentum and energy are
conserved to rounding), but only when the explicit collision bound would bind:
the spectral radius :math:`\rho(C)` of the collision matrix, :math:`8.7\,\nu_e`
at :math:`(N_l, N_m) = (4, 8)`, satisfies :math:`\rho(C)\,\mathrm{d}t > 2`.
Otherwise collisions stay in the explicit RHS, as for any other value of
``collision_scheme``. With ``fixed_dt = false`` the step is also kept below
:math:`5/\rho(C)` when the sub-step is on (:doc:`guidance`).

Geometry controls
-----------------

``[geometry] drift_scale`` selects the benchmark-compatible drift convention
(``1.0``, the default) or the alternate doubled-drift convention (``2.0``).
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
linked boundary, GKX follows the file's own ``theta`` range,
``jtwist/x0`` geometry factor, and ``kxfac`` metadata instead of forcing the
analytic s-alpha grid defaults.
For direct VMEC workflows, the runtime also accepts ``model = "vmec"``.
In that mode GKX calls the VMEC geometry helper to generate a
matching ``*.eik.nc`` file on demand, then immediately reuses the same imported
geometry path as the VMEC examples. Set ``vmec_file`` plus the flux-tube keys
``torflux``, ``npol`` and optionally ``alpha``. ``geometry_file`` can be used
as an explicit output path for the generated ``*.eik.nc`` file.
The VMEC path uses the ``booz_xform_jax`` backend, which is installed with
GKX; ``GKX_BOOZ_XFORM_JAX_PATH`` or ``BOOZ_XFORM_JAX_PATH`` point to a source
checkout instead. This is the recommended imported-geometry route for new
stellarator cases. The shipped VMEC TOMLs
point to locally generated files under ``examples/vmec``; generate those files
with ``vmex input.<case>`` before running the examples.
``vmec_file`` supports ``$ENV_VAR`` expansion, and relative paths are resolved
against the TOML directory first. Command-line overrides are resolved from the
shell working directory.
When ``geometry_file`` is set for ``model = "vmec"``, GKX regenerates
that target instead of reusing a stale file from an older VMEC conversion.
For VMEC ``fix aspect`` runs, GKX follows the helper default contract and
does not inject ``x0`` from the runtime ``Lx``. That keeps the generated
``*.eik.nc`` file aligned with the imported W7-X/HSX geometry output.

For Miller tokamak workflows, the runtime also accepts ``model = "miller"``.
In that mode GKX uses its in-package Miller backend to generate a matching
root-level ``*.eiknc.nc`` file, then re-enters the same imported
geometry path used for VMEC ``eik.nc`` files.
Set the Miller inputs directly in ``[geometry]``:
``rhoc``, ``q``, ``s_hat``, ``R0``, optional ``R_geo``, ``shift``,
``akappa``, ``akappri``, ``tri``, ``tripri``, and ``betaprim``.
The generated Miller surface has inverse aspect ratio ``rhoc / R0``. Set that
ratio explicitly when reproducing a benchmark; the separate ``epsilon`` field
belongs to analytic geometry metadata and does not replace the surface radius.
``geometry_file`` can be used as an explicit output path for the generated
Miller ``*.eiknc.nc`` file. Existing output is reused by default; pass
``--force`` to ``gkx geometry miller`` to regenerate it.

Executable path overrides
^^^^^^^^^^^^^^^^^^^^^^^^^

The ``gkx`` executable accepts path overrides:

.. code-block:: bash

   gkx run --config case.toml --vmec-file examples/vmec/wout_circular_tokamak.nc  # built by generate_wouts.sh
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
* ``solver = "krylov"``: always use the eigensolver. Its default route is
  ``KrylovConfig(method="auto")``: the Hermite-banded direct shift-invert
  (``banded``) when its block arrays fit in memory, else ``adaptive``. Either
  returns an eigenpair only when its residual passes the certification gate (see
  `Runtime precision`_). Other ``KrylovConfig.method`` values are ``adaptive``,
  ``power``, ``propagator``, ``arnoldi``, ``shift_invert`` (matrix-free GMRES
  inner solves) and ``sparse_shift_invert`` (assembles the sparse operator and
  factors it once on the host).

* ``fit_signal = "auto"`` (default): pick ``phi`` vs density based on fit quality
* ``fit_signal = "phi"``: use the electrostatic potential time trace
* ``fit_signal = "density"``: use the density moment time trace

For large ky scans, ``scan-runtime-linear --batch-ky`` integrates all ky values
in a single time integration pass (time integrator only) and then extracts the
growth rates from the per-ky traces.

For quasilinear spectra, use ``scan-runtime-linear --workers N`` instead of
``--batch-ky``. This runs independent per-``ky`` solves, computes the
quasilinear state extraction for each mode, preserves serial spectrum ordering,
and records the worker identity contract in the scan summary JSON.
``--batch-ky`` with ``[quasilinear] enabled = true`` raises ``ValueError``;
quasilinear scans require per-``ky`` evaluation.

Executable usage
----------------

.. code-block:: bash

   gkx examples/01_linear_tokamak/case_full.toml    # same as: gkx run --config ...
   gkx scan-runtime-linear --config benchmarks/cases/etg_linear_scan.toml --out tools_out/runtime_etg_scan
   gkx plot tools_out/runtime_etg_scan.scan.csv
   gkx run-runtime-linear --config examples/01_linear_tokamak/case_full.toml --out tools_out/cyclone_runtime
   gkx scan-runtime-linear --config benchmarks/cases/etg_linear_scan.toml --batch-ky
   gkx run-runtime-nonlinear --config examples/03_nonlinear_tokamak/case_full.toml --sample-stride 5 --out tools_out/nonlinear_cyclone_diag.csv

``examples/01_linear_tokamak/case_full.toml`` is a production-length run (32168 ``rk4`` steps at
``Nl = 16``, ``Nm = 48``); see :doc:`quickstart` for a ten-second demo.
``gkx CASE.toml`` dispatches to ``gkx run --config CASE.toml``, which picks the
linear or nonlinear command from ``[physics]``.

For ``run-runtime-nonlinear`` with ``fixed_dt = false``, omit ``--steps`` unless
you want a capped step count: the runtime then integrates in chunks until it
reaches ``t_max`` (or saturates, below). With ``fixed_dt = true`` and no
``--steps``, the step count is ``round(t_max / dt)``.

Diagnosed nonlinear runs stop at saturation by default (``[time] run_to =
"saturation"``): the runtime integrates in chunks and, from ``saturation_t_min``
on, assesses the accumulated per-species heat-flux trace in physical time every
``max(25, 5%)`` of the elapsed time. The transient is cut with MSER, the mean
error is a batch-means SEM, and the window must be free of drift for every
species and for the total (:ref:`saturation-stop`). The verdict is reported as a
status string: ``saturated``, ``growing``, ``quiescent``, ``drifting``,
``intermittent``, ``unconfirmed``, ``too_short`` or ``cap_reached:<status>``.
``t_max`` remains the hard cap, so a run that never saturates integrates the
full horizon.
Nonfinite or non-increasing retained timestamps cannot produce a saturation
verdict. Valid adaptive timestamps remain supported, but the sample-based
uncertainty estimator is not qualified for irregular sampling.
The chosen window and its ``mean +/- SEM`` are printed in the run summary and
recorded under ``saturation`` in the summary JSON. Pass
``--no-until-saturated`` (or set ``run_to = "t_max"``) for a fixed horizon.

For single-point runtime commands, artifact output can be requested either with
``--out`` or directly in the runtime TOML:

.. code-block:: toml

   [output]
   path = "tools_out/runtime_case"

``--out`` takes precedence over ``[output].path`` when both are provided.
A plain prefix writes a JSON summary plus CSV sidecars; a nonlinear target
ending in ``.out.nc`` (or another ``.nc`` suffix) writes the NetCDF bundle
``*.out.nc`` / ``*.big.nc`` / ``*.restart.nc`` instead. See :doc:`outputs` for
every file, CSV column and NetCDF variable.

Python driver
-------------

.. code-block:: bash

  # point CONFIG at the top of the script at benchmarks/cases/etg_linear_scan.toml
  python examples/12_restart_and_analysis/run.py

The canonical KBM TOML is a controlled comparison input. Inspect the
physical eigenpair residual and compare solver routes before using a result.

Parameter reference
-------------------

Every key a deck accepts, by table. ``unset`` is an absent value that the
loader or the run resolves, as the meaning says. A key outside these lists
raises an error that names the nearest valid spelling.
``examples/reference_inputs`` holds decks that set every key of a run kind at
its default. ``examples/common_input.toml`` lists only the keys that differ from
the defaults; the decks of ``examples/`` start from it, and pinned end-damping
rates and departures from the stepper carry a comment (:ref:`common-deck`).

``[[species]]``
^^^^^^^^^^^^^^^

.. csv-table::
   :header: "Key", "Type", "Default", "Meaning"
   :widths: 18, 9, 14, 59

   "``name``","str","``""ion""``","Species label."
   "``charge``","float","``1.0``","Charge in units of the proton charge."
   "``mass``","float","``1.0``","Mass over the reference ion mass."
   "``density``","float","``1.0``","Density over the reference density."
   "``temperature``","float","``1.0``","Temperature over the reference temperature."
   "``tprim``","float","``2.49``","Temperature gradient ``a/L_T``."
   "``fprim``","float","``0.8``","Density gradient ``a/L_n``."
   "``nu``","float","``0.0``","Collision frequency."
   "``kinetic``","bool","``true``","Evolve the species kinetically; ``false`` leaves it adiabatic."
   "``Nl``","int","unset","Laguerre moments this species keeps, at most the state's ``Nl``."
   "``Nm``","int","unset","Hermite moments this species keeps, at most the state's ``Nm``."
   "``hermite_closure``","str","``""hypercollision""``","End of the Hermite ladder: ``hypercollision`` or ``reflectionless``."

``[grid]``
^^^^^^^^^^

.. csv-table::
   :header: "Key", "Type", "Default", "Meaning"
   :widths: 18, 9, 14, 59

   "``Nx``","int","``48``","Radial (x) grid points, so the number of kx modes."
   "``Ny``","int","``48``","Binormal (y) grid points; the half layout stores ``1 + Ny//2`` rows."
   "``Nz``","int","``64``","Points along the field line when ``ntheta`` is unset."
   "``Lx``","float","``62.8``","Radial box length in rho_i; sets the kx spacing ``2 pi / Lx``."
   "``Ly``","float","``62.8``","Binormal box length in rho_i; used for the ky spacing when ``y0`` is unset."
   "``boundary``","str","``""periodic""``","Parallel boundary: ``periodic``, ``linked`` (twist and shift), or for VMEC ``fix aspect``, ``exact periodic``, ``continuous drifts``."
   "``jtwist``","int","unset","Twist-and-shift integer linking kx modes; taken from the geometry when unset."
   "``non_twist``","bool","``false``","Use the non-twisting flux-tube boundary with local-shear drift terms."
   "``kxfac``","float","``1.0``","Factor on the twist-shift kx shift, ``2 pi zp kxfac s_hat``."
   "``z_min``","float","``-3.141592653589793``","Lower end of the field-line coordinate when ``ntheta`` is unset."
   "``z_max``","float","``3.141592653589793``","Upper end of the field-line coordinate when ``ntheta`` is unset."
   "``y0``","float","unset","Binormal box parameter; the ky spacing is ``1/y0``. Overrides ``Ly``."
   "``ntheta``","int","unset","Points per poloidal turn; sets ``Nz = ntheta * zp`` and the span ``+-pi zp``."
   "``nperiod``","int","unset","Poloidal turns kept on each side of the centre; gives ``zp = 2 nperiod - 1``."
   "``zp``","int","unset","Number of ``2 pi`` segments along the field line; derived from ``nperiod`` when unset."
   "``ky_layout``","str","``""half""``","``half`` stores the non-negative ky rows, ``full`` the two-sided axis; results agree."

``[time]``
^^^^^^^^^^

.. csv-table::
   :header: "Key", "Type", "Default", "Meaning"
   :widths: 18, 9, 14, 59

   "``t_max``","float","``100.0``","End time in ``a/v_ti``; the hard cap when ``run_to = """"saturation""""``."
   "``dt``","float","``0.1``","Fixed step, or the starting step of the CFL controller; setting it selects ``rk2`` at that step."
   "``method``","str","``rk4``, ``auto`` if nonlinear","Integrator: ``rk2``, ``rk3``, ``sspx3``, ``rk4``, ``rk4-pi``, ``lawson-*``, ``imex``, ``imex2``, ``imex-ars3``, ``imex-ark3``, or ``auto``."
   "``fixed_dt``","bool","``false``","Keep ``dt`` fixed; ``false`` lets the CFL controller choose the step."
   "``sample_stride``","int","``1``","Steps between stored time samples."
   "``diagnostics_stride``","int","``1``","Steps between nonlinear diagnostic rows."
   "``chunk_steps``","int","unset","Nonlinear steps per chunk; 128 explicit, 32 adaptive IMEX when unset."
   "``diagnostics_interval``","float","unset","Simulation-time interval between nonlinear diagnostic rows; overrides the strides."
   "``diagnostics``","bool","``true``","Compute nonlinear diagnostics (``--no-diagnostics`` turns them off)."
   "``checkpoint``","bool","``false``","Rematerialize fixed-step nonlinear scans in reverse mode to save memory."
   "``implicit_restart``","int","``20``","GMRES restart length of implicit solves."
   "``implicit_preconditioner``","str","unset","Preconditioner of implicit solves; the solver default when unset."
   "``state_sharding``","str","unset","Shard the state over devices along ``auto``, ``ky``, ``kx``, ``z``, ``l``, ``m`` or ``species``."
   "``progress_bar``","bool","``false``","Print progress with elapsed time and ETA."
   "``dt_min``","float","``1e-07``","Smallest step the CFL controller may take."
   "``dt_max``","float","unset","Largest step the controller may take; ``dt`` when unset."
   "``cfl``","float","``0.9``","CFL safety factor of the step controller."
   "``cfl_fac``","float","unset","Stability prefactor of the CFL bound; per-method, moment-scaled default."
   "``nstep_restart``","int","unset","Write a restart checkpoint every this many steps."
   "``collision_split``","bool","``false``","Advance the diagonal hypercollisions in a split sub-step."
   "``collision_scheme``","str","``""implicit""``","Split update: ``implicit``, ``exp``, ``sts`` or ``rkc``."
   "``collision_operator``","str","``""landau""``","Collision model: ``landau``, ``none``, ``lenard_bernstein``, ``sugama``, ``improved_sugama``, ``coulomb``, ``coulomb_finite_kperp``."
   "``compressed_real_fft``","bool","``true``","Evaluate the nonlinear bracket on the half spectrum with paired real FFTs."
   "``nonlinear_dealias``","bool","``true``","Apply the 2/3 dealiasing mask to the nonlinear term."
   "``laguerre_nonlinear_mode``","str","``""grid""``","Laguerre treatment of the bracket: ``grid`` or ``spectral``."
   "``run_to``","str","``""saturation""``","Nonlinear stop: ``saturation`` ends the run at a stationary heat flux, ``t_max`` runs the full horizon."
   "``saturation_rel_sem``","float","``0.1``","Largest relative standard error of the windowed heat-flux mean that counts as saturated."
   "``saturation_t_min``","float","``800.0``","Run time before any saturated verdict."
   "``saturation_drift_rel``","float","``0.1``","Tolerated relative drift of the window mean."
   "``damp_ends_rate``","float","unset","Field-line end-absorber rate in ``v_ti/a``; the 30-transit rule when unset, 0 turns it off."

``[geometry]``
^^^^^^^^^^^^^^

.. csv-table::
   :header: "Key", "Type", "Default", "Meaning"
   :widths: 18, 9, 14, 59

   "``model``","str","``""s-alpha""``","Geometry: ``s-alpha``, ``miller``, ``slab``, ``vmec``, ``imported-netcdf``."
   "``geometry_backend``","str","``""auto""``","Backend that builds the Miller or VMEC field-line geometry; ``auto`` picks one."
   "``geometry_file``","str","unset","Imported ``*.eik.nc`` file, or the output path of a generated one."
   "``vmec_file``","str","unset","VMEC or VMEX ``wout`` file for ``model = """"vmec""""``."
   "``rhoc``","float","``0.5``","Miller flux-surface minor radius over ``a``."
   "``R_geo``","float","unset","Miller reference major radius; ``R0`` when unset."
   "``shift``","float","``0.0``","Miller Shafranov shift derivative."
   "``akappa``","float","``1.0``","Miller elongation."
   "``akappri``","float","``0.0``","Miller elongation radial derivative."
   "``tri``","float","``0.0``","Miller triangularity."
   "``tripri``","float","``0.0``","Miller triangularity radial derivative."
   "``torflux``","float","unset","Normalized toroidal flux of the VMEC surface."
   "``npol``","float","unset","Poloidal turns of the VMEC flux tube."
   "``npol_min``","float","unset","Smallest ``theta/pi`` accepted for the VMEC boundary-cut crossing."
   "``isaxisym``","bool","``false``","Treat the VMEC equilibrium as axisymmetric."
   "``which_crossing``","int","unset","Index of the boundary-cut crossing used by a VMEC cut."
   "``include_shear_variation``","bool","``false``","Apply the local-equilibrium shear-variation correction to VMEC geometry."
   "``include_pressure_variation``","bool","``false``","Apply the local-equilibrium pressure-variation correction to VMEC geometry."
   "``betaprim``","float","unset","Pressure gradient ``d beta/d rho`` of the local equilibrium."
   "``q``","float","``1.4``","Safety factor of the analytic geometry."
   "``s_hat``","float","``0.8``","Magnetic shear of the analytic geometry."
   "``z0``","float","unset","Slab length; sets ``gradpar = 1/z0``."
   "``zero_shat``","bool","``false``","Slab with zero shear."
   "``epsilon``","float","``0.18``","Inverse aspect ratio of the ``s-alpha`` model."
   "``R0``","float","``1.0``","Major radius over ``a``."
   "``B0``","float","``1.0``","Magnetic field on axis."
   "``alpha``","float","``0.0``","Field-line label of the VMEC flux tube."
   "``drift_scale``","float","``1.0``","Magnetic-drift convention: 1 standard, 2 doubled."
   "``kperp2_bmag``","bool","``true``","Include the ``1/B^2`` factor in ``k_perp^2``."
   "``bessel_bmag_power``","float","``0.0``","Power of ``1/B`` in the Bessel argument."

``[init]``
^^^^^^^^^^

.. csv-table::
   :header: "Key", "Type", "Default", "Meaning"
   :widths: 18, 9, 14, 59

   "``init_field``","str","``""density""``","Seeded field: ``density``, ``upar``, ``tpar``, ``tperp``, ``qpar``, ``qperp``, ``phi`` or ``all``."
   "``init_amp``","float","``1e-05``","Initial amplitude."
   "``init_single``","bool","``true``","Seed only the selected ``(ky, kx)`` mode; ``false`` seeds every retained mode."
   "``random_seed``","int","``22``","Seed of the random initial condition."
   "``gaussian_init``","bool","``false``","Gaussian envelope along the field line instead of random phases."
   "``gaussian_width``","float","``0.5``","Width of the Gaussian envelope."
   "``gaussian_envelope_constant``","float","``1.0``","Constant part of the Gaussian envelope."
   "``gaussian_envelope_sine``","float","``0.0``","Sine part of the Gaussian envelope."
   "``kpar_init``","float","``0.0``","Parallel wavenumber of the seed."
   "``init_file``","str","unset","Saved state to start from."
   "``init_file_scale``","float","``1.0``","Factor applied to the loaded state."
   "``init_file_mode``","str","``""replace""``","``replace`` the analytic seed by the loaded state, or ``add`` it."
   "``init_electrons_only``","bool","``false``","Seed only the electron species."

``[physics]``
^^^^^^^^^^^^^

.. csv-table::
   :header: "Key", "Type", "Default", "Meaning"
   :widths: 18, 9, 14, 59

   "``reduced_model``","str","``""gyrokinetic""``","Physics family; ``gyrokinetic`` is the only one supported."
   "``linear``","bool","``true``","Linear run."
   "``nonlinear``","bool","``false``","Nonlinear run; exclusive with ``linear``."
   "``electrostatic``","bool","``true``","Solve for ``phi``."
   "``electromagnetic``","bool","``false``","Solve for the electromagnetic fields selected by ``use_apar`` and ``use_bpar``."
   "``use_apar``","bool","``false``","Keep ``A_parallel`` when ``electromagnetic`` and ``beta > 0``."
   "``use_bpar``","bool","``false``","Keep ``B_parallel`` when ``electromagnetic``."
   "``adiabatic_electrons``","bool","``true``","Boltzmann electron response."
   "``adiabatic_ions``","bool","``false``","Boltzmann ion response; no kinetic ion species."
   "``tau_e``","float","``1.0``","Adiabatic-species temperature ratio ``T_i/T_e``."
   "``beta``","float","``0.0``","Reference plasma beta."
   "``collisions``","bool","``true``","Enable the collision operator."
   "``hypercollisions``","bool","``true``","Enable the hypercollision damping of high moments."

``[collisions]``
^^^^^^^^^^^^^^^^

.. csv-table::
   :header: "Key", "Type", "Default", "Meaning"
   :widths: 18, 9, 14, 59

   "``nu_hermite``","float","``1.0``","Hermite coefficient of the Lenard-Bernstein eigenvalue."
   "``nu_laguerre``","float","``2.0``","Laguerre coefficient of the Lenard-Bernstein eigenvalue."
   "``nu_hyper``","float","``0.0``","Isotropic hypercollision amplitude."
   "``p_hyper``","float","``4.0``","Isotropic hypercollision exponent."
   "``nu_hyper_l``","float","``0.0``","Laguerre hypercollision amplitude (constant branch only)."
   "``nu_hyper_m``","float","``1.0``","Hermite hypercollision amplitude."
   "``nu_hyper_m_const``","float","unset","Hermite amplitude of the constant branch; ``nu_hyper_m`` when unset."
   "``nu_hyper_lm``","float","``0.0``","Mixed Laguerre-Hermite hypercollision amplitude (constant branch only)."
   "``p_hyper_l``","float","``6.0``","Laguerre hypercollision exponent."
   "``p_hyper_m``","float","unset","Hermite hypercollision exponent; ``min(20, Nm/2)`` when unset."
   "``p_hyper_lm``","float","``6.0``","Mixed hypercollision exponent."
   "``D_hyper``","float","``0.0``","Perpendicular hyperdiffusion amplitude."
   "``p_hyper_kperp``","float","``2.0``","Perpendicular hyperdiffusion exponent."
   "``hypercollisions_const``","float","``0.0``","Weight of the constant-coefficient hypercollision branch."
   "``hypercollisions_kz``","float","``1.0``","Weight of the ``|k_z|`` hypercollision branch."
   "``damp_ends_amp``","float","unset","Removed spelling of the end-damping strength; refused unless 0."
   "``damp_ends_widthfrac``","float","``0.125``","End-damping layer width as a fraction of the chain."
   "``damp_ends_scale_by_dt``","bool","``false``","Removed; refused when true."
   "``electron_ion``","bool","``false``","Scoped beta = 0 electron-ion collision terms with fixed ion flow."

``[normalization]``
^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header: "Key", "Type", "Default", "Meaning"
   :widths: 18, 9, 14, 59

   "``contract``","str","``""cyclone""``","Normalization contract: ``cyclone``, ``etg``, ``kinetic``, ``tem`` or ``kbm``."
   "``rho_star``","float","unset","Rescales the operator wavenumbers."
   "``omega_d_scale``","float","unset","Factor on the magnetic-drift frequency."
   "``omega_star_scale``","float","unset","Factor on the diamagnetic drive."
   "``diagnostic_norm``","str","``""rho_star""``","Growth-rate and frequency normalization of the diagnostics: ``rho_star`` or ``none``."
   "``flux_scale``","float","``1.0``","Factor on heat and particle flux diagnostics."
   "``wphi_scale``","float","``1.0``","Factor on the ``Wphi`` diagnostic."

``[terms]``
^^^^^^^^^^^

.. csv-table::
   :header: "Key", "Type", "Default", "Meaning"
   :widths: 18, 9, 14, 59

   "``streaming``","float","``1.0``","Weight of the parallel streaming term."
   "``mirror``","float","``1.0``","Weight of the mirror term."
   "``curvature``","float","``1.0``","Weight of the curvature drift."
   "``gradb``","float","``1.0``","Weight of the grad-B drift."
   "``diamagnetic``","float","``1.0``","Weight of the diamagnetic drive."
   "``collisions``","float","``1.0``","Weight of the collision operator."
   "``hypercollisions``","float","``1.0``","Weight of the hypercollisions."
   "``hyperdiffusion``","float","``0.0``","Weight of the perpendicular hyperdiffusion."
   "``end_damping``","float","``1.0``","Weight of the end damping."
   "``apar``","float","``1.0``","Weight of ``A_parallel``."
   "``bpar``","float","``1.0``","Weight of ``B_parallel``."
   "``nonlinear``","float","``0.0``","Weight of the nonlinear term."

``[expert]``
^^^^^^^^^^^^

.. csv-table::
   :header: "Key", "Type", "Default", "Meaning"
   :widths: 18, 9, 14, 59

   "``fixed_mode``","bool","``false``","Hold one Fourier mode frozen during nonlinear evolution."
   "``iky_fixed``","int","unset","ky index of the frozen mode."
   "``ikx_fixed``","int","unset","kx index of the frozen mode."
   "``source``","str","``""default""``","External forcing: ``default`` or ``phiext_full``."
   "``phi_ext``","float","``0.0``","Amplitude of the external ``phi`` source."

``[output]``
^^^^^^^^^^^^

.. csv-table::
   :header: "Key", "Type", "Default", "Meaning"
   :widths: 18, 9, 14, 59

   "``path``","str","unset","Output prefix, or ``*.out.nc`` for a nonlinear NetCDF bundle."
   "``restart``","bool","``false``","Require a restart file and resume from it."
   "``restart_if_exists``","bool","``false``","Resume from a restart file when one exists."
   "``save_for_restart``","bool","``true``","Write ``*.restart.nc`` with a nonlinear bundle."
   "``restart_to_file``","str","unset","Explicit restart output path."
   "``restart_from_file``","str","unset","Explicit restart input path."
   "``restart_with_perturb``","bool","``false``","Add a fresh analytic seed to the loaded state."
   "``append_on_restart``","bool","``true``","Append the continued history to the existing ``*.out.nc``."
   "``warm_start``","bool","``false``","Seed each point of a scan from the previous one."
   "``resolved_diagnostics``","bool","``true``","Keep mode-resolved spectra in the nonlinear history."
   "``plots``","bool","``true``","Draw the run's figures beside its output."
   "``restart_scale``","float","``1.0``","Factor applied to the loaded restart state."
   "``nsave``","int","``10000``","Checkpoint cadence in steps when ``nstep_restart`` is unset."
   "``time_interval``","float","unset","Simulation-time interval between rows written to CSV and NetCDF history."

``[quasilinear]``
^^^^^^^^^^^^^^^^^

.. csv-table::
   :header: "Key", "Type", "Default", "Meaning"
   :widths: 18, 9, 14, 59

   "``enabled``","bool","``false``","Compute quasilinear transport weights."
   "``mode``","str","``""weights""``","``weights`` or ``saturated``."
   "``saturation_rule``","str","``""none""``","``none``, ``mixing_length`` or ``lapillonne_2011``."
   "``amplitude_normalization``","str","``""phi_rms""``","Mode amplitude: ``phi_rms``, ``phi_midplane`` or ``field_energy``."
   "``kperp_average``","str","``""phi_weighted""``","Weighting of ``k_perp^2``; only ``phi_weighted``."
   "``csat``","float","``1.0``","Saturation constant."
   "``gamma_floor``","float","``0.0``","Growth rate below which a mode is excluded."
   "``include_stable_modes``","bool","``false``","Keep stable modes in the weights."
   "``channels``","list of str","``[""es""]``","Transport channels; only ``es``."
   "``output_path``","str","unset","Prefix of the quasilinear output files; ``[output] path`` when unset."

``[parallel]``
^^^^^^^^^^^^^^

.. csv-table::
   :header: "Key", "Type", "Default", "Meaning"
   :widths: 18, 9, 14, 59

   "``strategy``","str","``""serial""``","``serial``, ``batch``, ``combined_ky``, ``device_batch``, ``pmap``, ``pjit``, ``shard_map``, ``state`` or ``velocity``."
   "``axis``","str","``""ky""``","Decomposed axis: ``ky`` or ``species_hermite``."
   "``batch_size``","int","unset","Points per batch."
   "``num_devices``","int","unset","Workers or devices to use."
   "``strict_identity``","bool","``true``","Check the parallel result against the serial one."
   "``backend``","str","``""auto""``","Worker kind (``thread``, ``process``) or velocity route; ``auto`` picks."
   "``auto``","bool","``false``","Resolve to ``shard_map`` over ``species_hermite`` from the visible devices."

``[run]``
^^^^^^^^^

.. csv-table::
   :header: "Key", "Type", "Default", "Meaning"
   :widths: 18, 9, 14, 59

   "``Nl``","int","unset","Laguerre moments; the resolution-guided fallback when unset."
   "``Nm``","int","unset","Hermite moments; the resolution-guided fallback when unset."
   "``diagnostics``","bool","unset","Compute diagnostics of a nonlinear run."
   "``dt``","float","unset","Step of the run; overrides ``[time] dt``."
   "``fit_signal``","str","unset","Signal the growth rate is fitted to: ``auto``, ``phi``, ``density``."
   "``kx``","float","unset","Radial wavenumber of the fitted mode; 0 when unset."
   "``ky``","float","unset","Binormal wavenumber of a single linear run; 0.3 when unset."
   "``laguerre_mode``","str","unset","Laguerre treatment of the nonlinear bracket for this run."
   "``method``","str","unset","Integrator of the run; overrides ``[time] method``."
   "``mu_vperp_max``","float","unset","Nodal magnetic-moment grid with this velocity cutoff instead of Laguerre."
   "``sample_stride``","int","unset","Steps between stored samples; overrides ``[time] sample_stride``."
   "``solver``","str","unset","Linear solver: ``auto``, ``time``, ``krylov``, ``explicit_time``."
   "``steps``","int","unset","Number of steps; with ``dt`` sets ``t_max``."

``[scan]``
^^^^^^^^^^

.. csv-table::
   :header: "Key", "Type", "Default", "Meaning"
   :widths: 18, 9, 14, 59

   "``Nl``","int","unset","Laguerre moments of the scan."
   "``Nm``","int","unset","Hermite moments of the scan."
   "``dt``","float","unset","Step of the scan."
   "``fit_signal``","str","unset","Signal the growth rate is fitted to."
   "``ky``","float","unset","List of ky values to scan."
   "``method``","str","unset","Integrator of the scan."
   "``sample_stride``","int","unset","Steps between stored samples."
   "``solver``","str","unset","Linear solver of the scan."
   "``steps``","int","unset","Steps per scan point."
   "``warm_start``","bool","unset","Seed each ky from the previous one."

``[fit]``
^^^^^^^^^

.. csv-table::
   :header: "Key", "Type", "Default", "Meaning"
   :widths: 18, 9, 14, 59

   "``auto_window``","bool","``true``","Choose the fit window from the signal."
   "``fit_signal``","str","``""auto""``","Signal the growth rate is fitted to."
   "``growth_weight``","float","``0.2``","Weight of growth over stationarity in the window score."
   "``min_amp_fraction``","float","``0.0``","Smallest amplitude, relative to the peak, the window may start at."
   "``min_points``","int","``40``","Fewest samples in the window."
   "``mode_method``","str","``""project""``","Mode extraction: ``project``, ``svd``, ``z_index`` or ``max``."
   "``require_positive``","bool","``true``","Reject a window with no positive growth."
   "``start_fraction``","float","``0.2``","Fraction of the signal skipped before the window."
   "``tmax``","float","unset","End of a manual fit window."
   "``tmin``","float","unset","Start of a manual fit window."
   "``window_fraction``","float","``0.4``","Fraction of the signal in the automatic window."
   "``window_method``","str","``""stationary""``","Window selection: ``stationary``, ``loglinear`` or ``fixed``."

``[damping_reference]``
^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header: "Key", "Type", "Default", "Meaning"
   :widths: 18, 9, 14, 59

   "``source``","str","required","Path of the legacy deck that was converted."
   "``sha256``","str","required","SHA-256 of that deck."
   "``route``","str","required","``fixed_linear`` or ``timestep_free``."
   "``amplitude``","float","required","Legacy ``damp_ends_amp``."
   "``dt_input``","float","required","Legacy input timestep."
   "``dt_step``","float","required","Actual reference step of a fixed linear route; 0 for ``timestep_free``."
   "``scale_by_dt``","bool","required","Legacy ``damp_ends_scale_by_dt`` flag."

Runtime sections
----------------

``load_runtime_from_toml`` reads these tables. An unknown key in ``[grid]``,
``[time]``, ``[geometry]`` or ``[init]`` raises ``ValueError`` (so do the
removed Diffrax ``[time]`` keys). An unknown key in ``[[species]]``,
``[physics]``, ``[collisions]``, ``[normalization]``, ``[terms]``, ``[expert]``,
``[output]``, ``[quasilinear]`` or ``[parallel]`` raises ``TypeError``. ``[run]``, ``[scan]``
and ``[fit]`` are read by the commands that use them.

* ``schema_version`` (document root; see `Schema compatibility`_)
* ``[grid]`` (``GridConfig``), ``[time]`` (``TimeConfig``),
  ``[geometry]`` (``GeometryConfig``), ``[init]`` (``InitializationConfig``)
* ``[[species]]`` (kinetic species definitions); the per-species moment
  counts are described under `Per-species velocity resolution`_.
* ``[physics]`` (electrostatic/electromagnetic, adiabatic/kinetic, linear/nonlinear)
* ``[collisions]`` (collision and hypercollision controls)
* ``[normalization]`` (contract key + optional overrides)
* ``[terms]`` (term toggles used by modular RHS assembly)
* ``[expert]`` (advanced fixed-mode controls for specialized workflows)
* ``[output]`` (artifact path and restart controls)
* ``[quasilinear]`` (quasilinear transport diagnostics)
* ``[parallel]`` (parallelization policy; defaults to serial)
* ``[damping_reference]`` (provenance written by
  ``migrate_end_damping_reference``; see `Explicit reference migration`_)
* ``[run]`` / ``[scan]`` / ``[fit]`` (driver controls: ``ky``, ``Nl``, ``Nm``,
  ``mu_vperp_max`` (nodal magnetic-moment grid, see :doc:`numerics`),
  ``solver``, ``method``, ``dt``, ``steps``, the scan ``ky`` list, and the
  growth-rate window options ``auto_window``, ``tmin``, ``tmax``,
  ``window_fraction``, ``min_points``, ``start_fraction``, ``growth_weight``,
  ``require_positive``, ``min_amp_fraction``, ``window_method``,
  ``mode_method``, ``fit_signal``)

Notable runtime-only keys:

* ``[time] damp_ends_rate``: the field-line end-absorber rate
  :math:`\nu_{end}` in inverse simulation-time units, finite and nonnegative.
  Every route (time, Krylov, eigen, shift-invert, nonlinear) applies the same
  rate; no timestep enters. Omit it for the default, 30 thermal transits of the
  fastest kinetic species across the end layer. ``0`` turns the absorber off.
  See :ref:`end-damping`.
* ``[collisions] damp_ends_widthfrac``: layer width as a fraction of the chain
  (default ``0.125``).
* ``[physics] reduced_model``: physics-family selector for runtime inputs.
  The maintained runtime supports full gyrokinetics via ``"gyrokinetic"``
  and its full-GK aliases. Non-promoted reduced-model values fail closed with
  ``NotImplementedError`` instead of silently routing through the wrong equations.
* ``[collisions] damp_ends_amp`` (removed): a positive value without
  ``damp_ends_rate`` on a non-periodic deck is refused, because GKX < 2.6 read
  it as ``damp_ends_amp/dt`` on the time route and as a rate elsewhere.
  ``damp_ends_amp = 0`` still means no damping. ``damp_ends_scale_by_dt = true``
  is refused too.
* ``[collisions] hypercollisions_const`` / ``hypercollisions_kz``: defaults are
  ``0.0`` / ``1.0`` (kz-proportional hypercollisions enabled by
  default, constant hypercollisions off).
  The parallel multiplier is spectral:
  :math:`|\partial_z|G=\mathcal{F}^{-1}(|k_z|\mathcal{F}G)`,
  on each periodic line or linked chain. It is not pointwise multiplication
  of physical-z samples by the wavenumber array. GKX releases before 2.1.0
  applied a pointwise multiplier on periodic lines: rerun periodic transport
  results from those versions that used nonzero kz damping and Hermite orders
  above two. Linked runs are unaffected.

* ``[collisions] p_hyper_m``: when omitted, the runtime path uses the
  resolution-aware default ``min(20, Nm/2)`` instead of a fixed exponent across
  Hermite resolutions.
* ``[collisions] nu_hermite`` / ``nu_laguerre``: coefficients entering the
  Lenard-Bernstein collision eigenvalue used by the modular collision kernel.
* ``[collisions] electron_ion`` (default ``false``): scoped GX electrostatic
  electron-ion increment with fixed ion flow at ``beta = 0``. Select
  ``[time] collision_operator = "lenard_bernstein"`` and standard coefficients
  ``nu_hermite = 1``, ``nu_laguerre = 2``. It requires one quasineutral kinetic
  ion/electron pair, uniform Laguerre counts (``Nl >= 2``, ``Nm >= 3``) and
  serial execution; :doc:`operators` gives the approximation and refusals.
* ``[collisions] nu_hyper`` / ``p_hyper``: isotropic hypercollision amplitude
  and exponent.
* ``[collisions] nu_hyper_l`` / ``nu_hyper_m`` / ``nu_hyper_lm`` and
  ``p_hyper_l`` / ``p_hyper_m`` / ``p_hyper_lm``: Laguerre-only,
  Hermite-only, and mixed hypercollision channels.
  ``nu_hyper_l`` and ``nu_hyper_lm`` act **only** in the constant-coefficient
  branch -- the :math:`|k_z|` kernel carries no Laguerre index -- so a deck
  that declares either while ``hypercollisions_const = 0.0`` (the default) is
  refused rather than run without the sink it asked for. Case validation also
  refuses either declaration when ``[physics] hypercollisions = false`` or
  ``[terms] hypercollisions = 0.0`` disables the whole operator.
* ``[collisions] nu_hyper_m_const``: the constant branch's Hermite rate.
  Omitted (the default) it reuses ``nu_hyper_m``, which damps Hermite in
  *both* branches when both are on. Set it to
  ``0.0`` alongside ``hypercollisions_const = 1.0`` to declare a pure Laguerre
  sink while the :math:`|k_z|` Hermite hypercollisions stay exactly as the
  reference code sets them. See :ref:`velocity-regularization`.
* ``[[species]] hermite_closure``: ``"hypercollision"`` (default) or
  ``"reflectionless"``; any other string raises ``ValueError``. It selects how
  that species ends its Hermite ladder, so each species is set separately and
  an omitted key keeps the existing behavior bit for bit. ``"reflectionless"``
  removes the species' :math:`|k_z|` hypercollision band
  (``hypercollisions_kz``, ``nu_hyper_m``, ``p_hyper_m``) and applies the
  outgoing-wave sink of Kanekar et al. on the last Hermite moment instead
  (:ref:`hermite-closure`). The constant Laguerre/Hermite hypercollision branch,
  ``nu_hyper``, collisions and end damping are unchanged, and the sink is
  switched off with ``[physics] hypercollisions = false``. It has no tunable
  parameter and needs no ``hypercollisions_kz``. Meant for kinetic electrons at
  modest ``Nm`` (see the guidance in :ref:`hermite-closure`); the mixed
  species-Hermite electrostatic parallel route refuses it.
* ``[collisions] D_hyper`` / ``p_hyper_kperp``: perpendicular hyperdiffusion
  amplitude and exponent.
* ``[normalization] flux_scale``: multiplicative factor applied to
  heat/particle flux diagnostics (tracked comparison default ``1.0``).
* ``[normalization] wphi_scale``: multiplicative factor applied to ``Wphi``
  diagnostics (default ``1.0``; no shipped configuration overrides it).
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
  integrates the full horizon. Linear runs and nonlinear runs with ``diagnostics = false`` ignore
  the key, because the stop test needs the streamed heat-flux trace. The
  executable overrides are ``--until-saturated`` and ``--no-until-saturated``.
  Set ``"t_max"`` explicitly on any deck whose heat flux is not the observable
  it exists to produce -- zonal-flow and secondary-instability responses,
  linear-response diagnostics -- because a horizon chosen by a quantity that is
  zero by construction is not a horizon. Which shipped decks do that, and the
  measurement clearing the rest to run under the default, are held by the
  ``run_to`` audit in ``tests/release/test_release_gates.py``.
* ``[time] saturation_rel_sem``: largest relative standard error of the
  windowed heat-flux mean, from batch means, that still counts as saturated
  (default ``0.10``).
* ``[time] saturation_t_min``: run time in ``a/v_t`` before any ``saturated``
  verdict (default ``800``; the Landreman 2025 GX database, CGYRO and GENE
  averaging windows are 500-1000). Capped at ``t_max`` for shorter runs.
* ``[time] saturation_drift_rel``: tolerated change of the window mean, as a
  fraction of its scale, across the averaging window (default ``0.10``). A trend
  must also be two standard errors from zero to count.
  These are finite-sample safeguards, not a calibrated sequential confidence
  guarantee; see :ref:`saturation-stop`.
* ``[time] nstep_restart``: when writing a nonlinear NetCDF bundle,
  checkpoint every ``nstep_restart`` steps instead of waiting for the end of
  the run. This is useful for long adaptive runs and batch jobs.
* ``[time] method = "sspx3"``: use the SSPx3 explicit scheme directly. This is
  the relevant explicit method for ``secondary`` and collisional-ETG benchmark
  families. Plain ``rk3`` is the three-stage Heun-style scheme.
  The names ``euler``, ``k10``, ``rk3_heun`` and ``rk3_classic`` were retired
  and raise an error naming the replacement (``rk2``, ``rk4``, ``rk3``,
  ``rk3``). Linear integration also accepts ``rk2``, ``rk4``,
  and the diagonal IMEX schemes ``imex`` and ``imex2`` (ARS(2,2,2), second
  order).
* ``[terms]``: each key is a pure multiplicative operator weight:
  ``streaming``, ``mirror``, ``curvature``, ``gradb``, ``diamagnetic``,
  ``collisions``, ``hypercollisions``, ``hyperdiffusion``, ``end_damping``,
  ``apar``, ``bpar``, ``nonlinear``.

.. _per-species-moments:

Per-species velocity resolution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``[run] Nl`` and ``Nm`` size the state and are the default for every species.
A ``[[species]]`` table may keep fewer moments for its species with its own
``Nl`` (Laguerre) and ``Nm`` (Hermite); a value left out keeps the ``[run]``
value, and a value above it raises ``ValueError``.

.. code-block:: toml

   [run]
   Nl = 12
   Nm = 48

   [[species]]
   name = "ion"
   Nm = 32          # ions: (12, 32)

   [[species]]
   name = "electron"
   Nl = 6           # electrons: (6, 48)

*Input.* ``[[species]] Nl`` / ``Nm`` (``RuntimeSpeciesConfig.Nl`` / ``Nm``,
positive integers or absent); from Python ``LinearParams.moments``, one
``(Nl, Nm, p_hyper_m)`` per species with ``None`` for "the state's". The same
resolved numbers are written to the run's ``gkx.toml`` copy, and the run summary
prints ``species moments: ion 12 x 32, electron 6 x 48`` beside ``moments``.

*Fallback.* A deck with two kinetic species whose thermal speeds differ by
a factor of 4 or more (kinetic electrons) that names no ``Nl`` or ``Nm``, in
``[run]``, on the call or in a species table, takes per-species counts: linear
ions (12, 32) and electrons (6, 48); nonlinear (4, 16) for both. The rule keys on the resolved pair being the generic fallback, so
naming exactly (12, 24) (linear) or (4, 8) (nonlinear) is read as naming none;
name any other pair, or any species count, to turn it off. The evidence is in
:doc:`guidance`.

*What a count changes.* A species keeps its moments below its counts; the ones
above stay zero. Its hypercollisions are normalized to its own counts (the
default ``p_hyper_m`` is ``min(20, Nm/2)`` of its own ``Nm``), its gyroaveraged
fields carry only its own ``Nl`` Laguerre terms, and the end-damping rate stays
the one set by the fastest species. The right-hand side, the nonlinear bracket and the
linear eigensolvers run each group of species at its own size, so the cost is
the sum over species of ``Nl * Nm``. A run with every species at the state's
counts is the uniform run, bit for bit.

*Outputs.* The state and restart arrays keep the ``[run]`` extents with zeros
above a species' counts, and a restart written at one set of counts reads back
at another by zeroing what the species no longer keeps.

*When to change it.* Raise the electron ``Nm`` first for a trapped-electron or
EM mode, the ion ``Nl`` for large ``ky rho_i``; lower the ion ``Nm`` for a case
dominated by electron physics. Check the growth rate or the heat flux against
the next larger counts of that species, as for ``[run]`` (:doc:`guidance`).

*Not supported.* Hermite-sharded routes, a custom collision operator and the
per-term diagnostic RHS raise ``NotImplementedError`` with unequal counts;
``mu_vperp_max`` refuses a species with fewer Laguerre nodes. ``imex-ars3`` and
``imex-ark3`` factor and solve each species group at its own counts.

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
   backend = "auto"

Accepted strategies are ``"serial"``, ``"batch"``,
``"combined_ky"``, ``"device_batch"``, ``"pmap"``, ``"pjit"``,
``"shard_map"``, ``"state"``, and ``"velocity"``. ``"batch-ky"`` is an alias
for ``"combined_ky"`` (the combined-``k_y`` time-integration scan path), and
``"none"``/``"off"`` are aliases for ``"serial"``. Quasilinear scan artifacts
require per-``k_y`` evaluation. ``auto = true`` resolves to
``strategy = "shard_map"`` with ``axis = "species_hermite"``.

For runtime ``k_y`` scans, ``strategy = "batch"`` with ``axis = "ky"`` selects
the production independent-worker path. ``num_devices`` is interpreted as the
requested worker count when the executable call does not explicitly pass
``workers``; ``backend = "thread"`` or ``"process"`` selects the executor, and
``"auto"`` keeps the executable default. The runtime output records whether
the worker policy came from the TOML file or from explicit executable
arguments, together with the requested/effective worker counts and the
ordering-preservation identity contract.

A ``k_y`` scan reads ``strategy`` at two levels, and each strategy belongs to
exactly one of them. ``"batch"``, ``"combined_ky"``, ``"device_batch"``,
``"pmap"`` and ``"pjit"`` are *scan-level*: they say how the ``k_y`` points are
dispatched, the scan honours them itself, and each worker then runs an ordinary
single-``k_y`` solve with ``strategy = "serial"``. Their reported growth rates
are bit-identical to a serial scan. ``"velocity"``, ``"state"`` and
``"shard_map"`` are *solver-level*: they say how one solve is decomposed, so
they reach the per-``k_y`` solve unchanged and are honoured or refused there on
the solver's own terms. Only ``"velocity"`` has a linear RHS route today, so a
``k_y`` scan requesting ``"state"`` or ``"shard_map"`` raises rather than
quietly running unsharded.

The velocity-space RHS route is diagnostic: ``strategy = "velocity"``, ``axis = "hermite"``, and
``backend = "streaming_only"``, ``backend = "streaming_electrostatic"``, or
``backend = "electrostatic_linear_slices"``. ``backend = "auto"`` selects the
electrostatic-slices route when the active terms satisfy that gate; otherwise
the runtime raises instead of silently falling back to an unvalidated path.
These backends are accepted only by ``gkx.linear_rhs_parallel_cached``.
The first two require all non-streaming linear terms disabled. The
electrostatic-slices backend allows only streaming, mirror, curvature, grad-B,
and diamagnetic-drive weights; collisions, linked boundaries, electromagnetic
terms, and nonlinear terms are refused. Velocity RHS routes are limited to
single-species periodic 5D electrostatic states.

For full runtime TOML files this velocity-space route is exposed only through
the fixed-step linear executable path with ``fit_signal = "phi"``.
Density-assisted automatic fitting is serial only, so requesting velocity
parallelization there raises an error.

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
* ``time_interval``: optional positive finite interval in simulation time for
  rows written to nonlinear CSV and NetCDF history. This selects among the
  diagnostics already retained by the solver; it does not change returned
  diagnostics, the summary, the final state or checkpoint cadence. See
  :doc:`outputs` for its relation to ``time.diagnostics_interval``.
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

Explicit reference migration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a **resolved legacy deck** with a known execution route:

.. code-block:: python

   from gkx.workflows.runtime.toml import migrate_end_damping_reference

   case = migrate_end_damping_reference(
       "legacy.toml", route="fixed_linear", dt_step=0.02,
   )  # actual reference step, not an assumed input default
   case.to_toml("fixed_rate.toml")

Use ``route="timestep_free"`` without ``dt_step`` for a reference RHS that
received no solver timestep. The helper does not run a simulation or overwrite
the source. It records ``[damping_reference]``: source path/SHA-256, route,
amplitude, input timestep, actual reference step and old scaling flag. The
record survives ``gkx.load`` / ``case.to_toml``; no source file is needed to
reload it. A changed rate with stale provenance is rejected at runtime: clear
``damping_reference`` when intentionally defining a new model.
Migration is host-side bookkeeping; differentiate the resulting
``LinearParams.damp_ends_rate``, not the file-reading helper.

Let :math:`s=1` for the old true scaling flag and :math:`s=0` otherwise:

.. math::

   \nu_{ref} = \frac{A}{\Delta t_{input}^{s}}
   \begin{cases}
     1/\Delta t_{step}, & \text{fixed linear route},\\
     1, & \text{timestep-free RHS}.
   \end{cases}

This preserves the isolated RHS and RK stage map **at the reference step**.
Subsequent timestep refinement holds :math:`\nu_{ref}` fixed; it does not
preserve the legacy per-step model at every timestep. Adaptive linear sources,
conflicting ``run.dt`` overrides and already-converted decks are rejected.
Resolve any command-line timestep override into the source deck first.
This is provenance and algebraic equivalence, **not benchmark recertification**.
