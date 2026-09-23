Examples
========

The ``examples`` directory is a numbered gallery, one directory per workflow
(``examples/README.md`` lists each group with its purpose and runtime). Each
group holds a ``run.py`` and, where the workflow is deck-driven, a short
tutorial ``case.toml`` plus the literature-resolution ``case_full.toml``.
Validation-only decks live under ``benchmarks/cases/``.

Config-backed runtime cases
---------------------------

These scripts are the closest match to the production benchmark workflows.
They load the checked-in runtime TOMLs and expose only the most useful runtime
overrides at the command line.

Tokamak cases
^^^^^^^^^^^^^

.. code-block:: bash

   python examples/01_linear_tokamak/run.py
   python examples/03_nonlinear_tokamak/run.py
   python examples/06_electromagnetic/run.py
   gkx benchmarks/cases/etg_linear_scan.toml
   gkx benchmarks/cases/kaw_linear.toml
   gkx benchmarks/cases/kbm_nonlinear.toml
   gkx benchmarks/cases/cyclone_nonlinear_miller.toml

VMEC-backed tokamak and stellarator cases
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install vmec-jax
   cd examples/vmec
   ./generate_wouts.sh
   cd ../..

   gkx run --config benchmarks/cases/circular_vmec_linear.toml
   gkx run --config benchmarks/cases/circular_vmec_nonlinear.toml

   gkx run --config examples/02_linear_stellarator/case_full.toml
   gkx run --config benchmarks/cases/w7x_linear_quasilinear_vmec.toml
   gkx benchmarks/cases/w7x_nonlinear_vmec_geometry.toml
   python examples/04_nonlinear_stellarator/run.py

For the VMEC-backed stellarator examples, keep ``STEPS = None`` in the script
when you want the default adaptive horizon. Set ``STEPS`` (e.g. ``STEPS = 200``)
only when you intentionally want a short profiling or diagnostic window. For longer W7-X nonlinear runs, keep
adaptive timesteps enabled (the default for the examples) or reduce ``dt`` if
you need a fixed-step stability study.

The bundled VMEC decks are self-contained examples. Exact HSX or W7-X
validation should use the same TOMLs with ``--vmec-file`` pointing to the
machine-specific benchmark WOUT. If you only need one local WOUT, run
``vmex input.NAME`` in ``examples/vmec`` instead of the full
``generate_wouts.sh`` helper.

The shipped nonlinear stellarator runtime TOMLs now also emit artifact bundles
under ``tools_out/`` by default:

- ``tools_out/w7x_nonlinear_vmec_runtime.diagnostics.csv``
- ``tools_out/hsx_nonlinear_vmec_runtime.diagnostics.csv``
- ``tools_out/w7x_nonlinear_imported_runtime.diagnostics.csv``

Those diagnostics and their matching ``*.summary.json`` files are the intended
inputs for the parity helpers under ``tools/``.
The direct Python runtime wrappers now route through the same artifact-aware
nonlinear path as the executable, so long adaptive runs update that bundle as each
chunk completes.

Runtime TOML entry points
-------------------------

When you want the full config surface instead of the thin case wrappers, use
the executable or the generic example drivers directly. These runtime utilities are
best treated as solver-smoke and exploration entry points; the benchmark
examples remain the audited parity surface for ETG and the other validation
lanes:

.. code-block:: bash

   # point CASE at the top of the script at any runtime TOML
   python examples/12_restart_and_analysis/run.py
   python benchmarks/etg_linear_benchmark.py --outdir tools_out/etg
   python benchmarks/kbm_linear_comparison.py --output tools_out/kbm_linear_comparison.png

   gkx run-runtime-linear \
     --config examples/08_quasilinear/case_full.toml \
     --out tools_out/cyclone_quasilinear

   gkx run-runtime-linear \
     --config benchmarks/cases/w7x_linear_imported_geometry.toml

   gkx examples/01_linear_tokamak/case_full.toml

``examples/06_electromagnetic/case_full.toml`` is retained as the canonical operator input for controlled
comparison studies. Its experimental shift-invert path fails closed while the
full-resolution physical residual exceeds the documented acceptance gate; use
the reviewed comparison driver above for the promoted KBM result.

For a bounded runtime-configured independent ``k_y`` scan that uses
``[parallel] strategy = "batch"`` without changing the single-``k_y`` solver
layout, run:

.. code-block:: bash

   python examples/11_parallel_scan/run.py

The companion
``examples/11_parallel_scan/case.toml`` selects two thread
workers through ``[parallel].num_devices``. The runtime still dispatches normal
single-``k_y`` solver calls and gathers results in input order; it does not opt
into the combined-``k_y`` solver path.

Scaling utilities
-----------------

For production parallelization of independent scans and UQ ensembles, prefer
the package helpers:

.. code-block:: python

   import jax.numpy as jnp
   import gkx as sgk

   ky = jnp.asarray([0.1, 0.2, 0.3, 0.4])
   chunks = sgk.ky_scan_batches(ky, n_batches=2)
   values = sgk.batch_map(
       lambda x: {"gamma": x, "ql_weight": x**2},
       ky,
       batch_size=2,
   )

For file-backed calibration and uncertainty workflows that are independent but
not JAX-array ``vmap`` workloads, use ``sgk.independent_map``:

.. code-block:: python

   rows = sgk.independent_map(
       lambda case: {"case": case, "score": len(case)},
       ["cyclone", "hsx", "w7x"],
       workers=2,
   )

These helpers preserve serial ordering and fall back to a one-device ``vmap``
path on laptops. Multi-device runs should still be checked against the serial
result before publication speedups are claimed.

Autodiff validation reports also accept ``workers`` for thread-parallel
central finite-difference columns; the acceptance criterion is numerical
identity with the serial report.

For a solver-backed identity gate, run the Cyclone ``k_y``-batch scan artifact:

.. code-block:: bash

   python tools/artifacts/generate_parallel_identity_gate.py ky-scan

**Generated figure.** Real Cyclone linear solver comparison between serial and
fixed-shape
``k_y``-batched scans. The figure verifies that ``gamma`` and ``omega``
are identical while reporting the observed batch speedup separately.


For a logical-CPU API gate that exercises ``RuntimeParallelConfig`` and pytree
outputs, run:

.. code-block:: bash

   python tools/artifacts/generate_parallel_identity_gate.py logical-cpu --logical-devices 2

**Generated figure.** Independent-scan interface gate for structured outputs. This
validates the
parallel API used by UQ and sensitivity ensembles; it is not a nonlinear
performance claim.


The first lower-level communication gate for velocity-space decomposition is
the Hermite ghost exchange:

.. code-block:: bash

   python tools/artifacts/generate_velocity_parallel_gates.py hermite-exchange --logical-devices 2

**Generated figure.** ``shard_map`` nearest-neighbor exchange for Hermite moments. This
validates
the communication primitive that a future nonlinear velocity-space sharding
path needs before field reductions and full-RHS identity gates are added.


The paired field-reduction gate is:

.. code-block:: bash

   python tools/artifacts/generate_velocity_parallel_gates.py field-reduce --logical-devices 2

**Generated figure.** ``shard_map`` reduction/broadcast over a Hermite mesh. This
establishes the
field-solve communication primitive before streaming-ladder and nonlinear
RHS identity gates are attempted.


The first production-field-solve reduction gate is:

.. code-block:: bash

   python tools/artifacts/generate_electrostatic_parallel_gates.py field-reduce --logical-devices 2

**Generated figure.** Hermite-sharded ``m=0`` density reduction for the electrostatic
quasineutrality solve, compared against the production field solve.


The Hermite streaming-ladder coefficient gate is:

.. code-block:: bash

   python tools/artifacts/generate_velocity_parallel_gates.py hermite-ladder --logical-devices 2

**Generated figure.** ``shard_map`` Hermite exchange plus the ``sqrt(m+1)`` /
``sqrt(m)``
streaming-ladder coefficients. This is still a communication/coefficient
gate; full linear streaming also needs the parallel derivative identity
gate before production runtime wiring.


The first electrostatic drift-slice gate is:

.. code-block:: bash

   python tools/artifacts/generate_electrostatic_parallel_gates.py drift --logical-devices 2

**Generated figure.** Hermite-sharded mirror and curvature/grad-B drift slices,
including
offset-1 and offset-2 Hermite exchanges, compared against the production
linear RHS with only those terms enabled.


The matching electrostatic diamagnetic-drive gate is:

.. code-block:: bash

   python tools/artifacts/generate_electrostatic_parallel_gates.py diamagnetic --logical-devices 2

**Generated figure.** Hermite-sharded electrostatic diamagnetic drive. The sharded route
first
uses the electrostatic field-reduction gate, then applies the local
``m=0`` and ``m=2`` drive masks on each Hermite shard. This closes the
diamagnetic slice for the opt-in electrostatic linear-slices backend.


The periodic streaming microkernel gate adds that field-line derivative:

.. code-block:: bash

   python tools/artifacts/generate_velocity_parallel_gates.py periodic-streaming --logical-devices 2

**Generated figure.** Periodic spectral parallel derivative plus Hermite streaming
ladder through
the ``shard_map`` path, compared directly against the production streaming
operator.


The next gate places that same sharded streaming kernel under the production
linear-RHS call graph with every non-streaming contribution disabled:

.. code-block:: bash

   python tools/artifacts/generate_linear_rhs_parallel_gates.py streaming --logical-devices 2

**Generated figure.** Streaming-only ``linear_rhs_cached`` comparison against the
velocity-sharded
periodic streaming path. This closes the first full-RHS call-graph identity
gate for the streaming term only; it is not yet a full linear scan or
nonlinear speedup claim.


With a nonzero electrostatic response, use:

.. code-block:: bash

   python tools/artifacts/generate_linear_rhs_parallel_gates.py streaming-electrostatic --logical-devices 2

**Generated figure.** Streaming plus electrostatic ``phi`` call-graph comparison. The
field solve
uses the Hermite-sharded electrostatic reduction gate; this validates the
next velocity-sharded streaming slice before drift, diamagnetic-drive, and
nonlinear terms are introduced.


For the composed electrostatic linear-slices backend, use:

.. code-block:: bash

   python tools/artifacts/generate_linear_rhs_parallel_gates.py electrostatic-slices --logical-devices 2

**Generated figure.** Full opt-in electrostatic linear-slices call-graph comparison for
streaming, mirror, curvature, grad-B, and diamagnetic drive. This is an
opt-in electrostatic linear-RHS identity artifact for the single-species
periodic electrostatic RHS path; collisions, linked boundaries,
electromagnetic terms, and nonlinear brackets remain separate gates.


For the current opt-in Hermite-sharded electrostatic linear RHS path, use the
engineering sweep helper:

.. code-block:: bash

   python tools/profiling/profile_linear_rhs_parallel_slices.py sweep \
     --platform cpu --devices 1,2,4,8 --nms 64,128 \
     --nl 4 --ny 32 --nz 128 --rtol 1e-5

**Generated figure.** Device-count and Hermite-resolution sweep for the opt-in
electrostatic
linear-slices backend. The right panel is the identity gate; the left panel
is engineering timing only and should not be promoted as a nonlinear or
publication speedup claim.


Plotting outputs
----------------

To visualize nonlinear diagnostic histories from ``*.out.nc`` files:

.. code-block:: bash

   # runs a short case, saves it, reloads the bundle and plots it
   python examples/12_restart_and_analysis/run.py
   gkx --plot <output_file>

Geometry examples
-----------------

VMEC and Miller geometry usage examples are documented in :doc:`geometry`.

Nonlinear restart and continuation
----------------------------------

The tracked nonlinear runtime path supports a NetCDF ``out/big/restart``
bundle together with continuation from the saved restart state.

One-shot nonlinear bundle write:

.. code-block:: bash

   gkx run-runtime-nonlinear \
     --config examples/03_nonlinear_tokamak/case_full.toml \
     --steps 200 \
     --out tools_out/cyclone_release.out.nc

For the short Cyclone comparison replay (``t_max = 5``, no collisions), use
``benchmarks/cases/cyclone_nonlinear_short.toml``.
That file pins the short-run dissipation contract explicitly
(``p_hyper = 2``, ``damp_ends_amp = 0``) instead of relying on the longer
production defaults.

Restart-aware TOML snippet:

.. code-block:: toml

   [time]
   nstep_restart = 100

   [output]
   path = "tools_out/cyclone_release.out.nc"
   restart_if_exists = true
   save_for_restart = true
   append_on_restart = true
   restart_with_perturb = false

With that configuration, rerunning the same nonlinear command resumes from
``tools_out/cyclone_release.restart.nc`` when it already exists and appends the
continued history to ``tools_out/cyclone_release.out.nc``. This is the
recommended user-facing workflow for long nonlinear turbulence jobs.

Lightweight turbulence movies
-----------------------------

First persist the production state and its saturation verdict:

.. code-block:: bash

   python tools/campaigns/nonlinear_saturated_state.py \
     --toml CASE.toml --state-out saturated_state.npz \
     --output saturation.json

Then run a short production-policy continuation and render it off-device:

.. code-block:: bash

   python tools/artifacts/build_turbulence_movie.py CASE.toml \
     --initial-state saturated_state.npz --frames 60 \
     --steps-per-frame 40 --snapshots movie_cuts.npz
   python tools/artifacts/build_turbulence_movie.py \
     --render-from movie_cuts.npz --output turbulence.mp4 --fps 10

Each chunk uses the deck's explicit method and timestep policy,
:math:`\Delta t_n=P_{\rm CFL}(\phi_n;\Delta t_{\min},\Delta t_{\max},c_{\rm CFL})`.
Frame times remain absolute,
:math:`t_k=t_{\rm state}+\sum_{n=1}^{kN_f}\Delta t_n`.
The snapshot stores only

.. math::

   \phi(x,y,z_{\rm mid}),\qquad \phi(x_{\rm mid},y,z),

so its per-frame payload is :math:`N_xN_y+N_yN_z`, rather than
:math:`N_xN_yN_z`. At ``96x96x48`` this is a 32-fold reduction. Schema 3 also
records the source path and saturation flag, method, fixed/adaptive policy,
resolution, VMEC field-line coordinates, and physical perpendicular extent.
Imported-geometry movies require finite ``R(z)``, ``Z(z)``, and
``zeta(z)`` profiles and fail instead of drawing an analytic torus when they
are absent. A finite stellarator flux-tube segment is generally open in
three-dimensional space: its endpoints need not coincide. The linked boundary
identifies the gyrokinetic fields through twist-and-shift, not by closing the
centreline in Cartesian coordinates.
Rendering never holds a GPU allocation. A seed-only movie is allowed but is
labelled ``seeded continuation`` and is not saturation evidence.

Geometry generation workflows
-----------------------------

The runtime geometry path generates imported geometry files from VMEC or
Miller inputs. VMEC uses the documented optional geometry bridge; Miller uses
the in-package backend and needs no external helper:

.. code-block:: bash

   cd examples/vmec
   vmex input.NuhrenbergZille_1988_QHS
   cd ../..
   export GKX_BOOZ_XFORM_JAX_PATH=/absolute/or/relative/booz_xform_jax
   gkx geometry vmec \
     --config examples/04_nonlinear_stellarator/case_full.toml

   gkx geometry miller \
     --config benchmarks/cases/cyclone_nonlinear_miller.toml

Benchmark and scan helpers
--------------------------

These scripts produce the scan-level plots and tables used in the benchmark
discussion:

.. code-block:: bash

   python benchmarks/cyclone_linear_benchmark.py
   python benchmarks/etg_linear_benchmark.py
   python benchmarks/kbm_linear_comparison.py
   python benchmarks/kinetic_linear_benchmark.py
   python benchmarks/tem_linear_benchmark.py

``kbm_linear_comparison.py`` plots the reviewed fixed-beta ``ky`` table. The
matched rerun and branch-continuity analysis live in
``tools/comparison/compare_gx_kbm.py``.

The kinetic-electron script loads
``examples/05_kinetic_electrons/case_full.toml``. The same input
can be run directly with
``gkx examples/05_kinetic_electrons/case_full.toml``.
It runs on the native RK4 owner.

The TEM script loads ``benchmarks/cases/tem_linear.toml``; users
can run the same single-mode case directly with
``gkx benchmarks/cases/tem_linear.toml``.
It runs on the native RK2 owner.

Foundational demos
------------------

These smaller examples are useful for understanding the numerical building
blocks without running a full benchmark case:

.. code-block:: bash

   python benchmarks/basis_orthonormality.py
   python examples/09_autodiff/run.py
   python examples/08_quasilinear/implicit_sensitivity.py

Differentiable optimization examples
------------------------------------

The public optimization example is a VMEX QA stellarator workflow with one
physical GKX heat-flux tuple appended to the objective list:

.. code-block:: bash

   python examples/10_vmex_optimization/run.py

``run.py`` (formerly ``QA_optimization.py``) mirrors VMEX's boundary-mode ladder, preserves its
``A=6`` and mean-``iota=0.42`` targets, and adds the checkpointed nonlinear
heat-flux derivative. The equilibrium is vacuum; finite ``a/L_T`` and ``a/L_n``
drive GKX. Edit the top-level constants to set resolution and run length.

The optimizer window starts from a detached saturated state and supplies a
local design derivative. Historical matched runs give a preliminary 12.26%
nominal reduction (conditional 95% CI 10.64--13.88%). This is not statistically
resolved: individual drift failures and missing resolved spectra block
promotion. These traces predate the periodic hypercollision correction and must
be regenerated before evaluating current-operator transport.
See :doc:`stellarator_optimization` for the audit, resolution
ladder, and CSV data.

The reduced synthetic stellarator-ITG diagnostic scripts (a deprecated
reduced model) were retired from ``examples/`` after GKX 2.3.0; they are
recoverable from commit ``f9485f044``. Their tracked JSON sidecars under
``docs/_static`` remain as historical records only.

The production bridge now exposes the same portfolio layout for real
``vmex -> booz_xform_jax -> GKX`` rows:
``stellarator_itg_vmec_boozer_sample_objective_table_from_state`` returns a
``(surface, alpha, ky, objective)`` table and
``stellarator_itg_vmec_boozer_portfolio_objective_from_state`` reduces it with
the same weights as the cheap gate. Promotion still requires held-out
surface/field-line artifacts and matched baseline/optimized long
post-transient nonlinear windows, not startup traces or reduced-window
estimators.

``examples/09_autodiff/run.py`` writes a summary JSON alongside the figure.
A single observed mode does not identify both gradients; the two-mode
construction does, which is why it is the gallery example.


.. figure:: _static/autodiff_inverse_twomode.png
   :width: 90%
   :align: center

   Two-mode inverse validation. The goal is to recover the planted gradients
   from two independent mode observables and verify that the autodiff Jacobian
   stays consistent with finite differences. The shipped result reaches the
   target to numerical precision and is the reviewer-facing parameter-recovery
   validation.

Secondary slab workflow
-----------------------

.. code-block:: bash

   python -m gkx.cli run-runtime-linear \
     --config benchmarks/runtime_secondary_slab.toml

   python benchmarks/secondary_slab_workflow.py

The staged helper runs the linear seed, writes a restart state in the runtime
binary layout, and then launches the nonlinear follow-up with the matching
restart and fixed-mode controls used in the tracked secondary benchmark.

Full-GK ETG nonlinear pilot
---------------------------

.. code-block:: bash

   JAX_ENABLE_X64=1 gkx benchmarks/cases/etg_nonlinear.toml --steps 200

This is the full-GK two-species ETG nonlinear pilot lane. The shipped
contract now matches the audited short-window startup path: ``Lx = 1.25`` for the linked ETG box and
``gaussian_init = true`` with ``init_single = false`` because GX reads
``init_single`` from its ``[Expert]`` section, not from ``[Initialization]``.
