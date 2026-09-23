Parallelization policy
======================

GKX parallelization claims are separated by workload class and by the
identity gates that exist for them. This page owns the parallel evidence:
what each route does, which gate it passes, which artifact records it, and
how to regenerate that artifact. Single-device performance is in
:doc:`performance`; runnable examples are in :doc:`examples`.

For release notes and manuscripts, read this page together with
:doc:`release_scope`. Independent scans and ensembles are the production path.
Whole-state nonlinear sharding and nonlinear domain or velocity-space
decomposition remain diagnostic correctness/profiler paths until they pass
workload-specific identity, conservation, transport-window, and matched
profiler gates.

Strategy registry
-----------------

The metadata API exposes a JSON-friendly strategy table. Release-ready
independent-work rows are ordered first: ``independent_ky_scan``, then
``uq_ensemble``.

.. list-table::
   :header-rows: 1
   :widths: 28 18 18 24

   * - ``name``
     - ``readiness``
     - ``independent_work``
     - ``changes_solver_layout``
   * - ``independent_ky_scan``
     - ``release_ready``
     - ``true``
     - ``false``
   * - ``uq_ensemble``
     - ``release_ready``
     - ``true``
     - ``false``
   * - ``whole_state_kx_ky``
     - ``diagnostic``
     - ``false``
     - ``true``
   * - ``velocity_species_hermite``
     - ``diagnostic``
     - ``false``
     - ``true``
   * - ``fft_axis_domain``
     - ``diagnostic``
     - ``false``
     - ``true``

Production path: independent work
---------------------------------

Production-ready parallelism is scoped to independent solver calls:

- independent ``k_y`` scans;
- quasilinear calibration grids;
- finite-difference and sensitivity batches;
- UQ and ensemble workloads.

Use ``gkx.ky_scan_batches`` and ``gkx.batch_map`` for JAX-array
workloads, and ``gkx.independent_map`` for file-backed Python tasks.
These helpers preserve serial ordering and restrict communication to result
aggregation. On one device they reduce to batched ``vmap`` execution; on
several they use JAX device batching and trim padded edge samples
deterministically. Any timing claim from this path must be paired with a serial
numerical-identity gate for the reported observables, such as ``gamma``,
``omega``, quasilinear weights, or covariance summaries.

For UQ and optimization portfolios, ``gkx.independent_ensemble_provenance_gate``
is the compact production-readiness check. It runs the same member function
serially and through ``independent_map``, verifies numerical identity and result
ordering, checks that oversubscribed worker requests clip to the ensemble size,
reconstructs the deterministic independent-work decomposition, and probes
``IndependentMapExecutionError`` metadata for worker failures. This is a
provenance and identity gate only; it does not make a nonlinear
domain-decomposition speedup claim.

Runtime ``k_y`` scans can request the same independent-worker policy directly
from TOML. This is a scan orchestration path, not a solver-layout sharding path:

.. code-block:: toml

   [parallel]
   strategy = "batch"
   axis = "ky"
   num_devices = 4      # or batch_size = 4
   backend = "auto"     # "thread" or "process" are explicit alternatives

When command-line scan workers are not set explicitly, ``strategy = "batch"``
with ``axis = "ky"`` resolves to independent per-``k_y`` solver calls and
records the resolved worker policy in runtime scan artifacts.

Scaling evidence
~~~~~~~~~~~~~~~~

The large tracked artifacts use real solver work rather than synthetic sleeps,
and every multi-worker row is compared against the one-worker result:

.. list-table::
   :header-rows: 1
   :widths: 30 34 36

   * - Artifact
     - Workload
     - Strong scaling (identity passes on every row)
   * - ``docs/_static/independent_ky_scan_scaling_large.json``
     - Cyclone linear scan, 64 ``k_y`` values, ``Ny=128``, ``Nz=96``,
       ``Nl=4``, ``Nm=8``, 240 RK2 steps per mode
     - CPU ``1.94x`` / ``3.78x`` / ``7.18x`` on 2 / 4 / 8 workers; two RTX
       A4000 GPUs ``1.88x`` (94 per cent efficiency). ``gamma`` and ``omega``
       mismatch is zero.
   * - ``docs/_static/quasilinear_uq_ensemble_scaling_large.json``
     - six Cyclone ITG gradient samples x five ``k_y``, ``Ny=96``, ``Nz=64``,
       ``Nl=3``, ``Nm=6``, 2000 RK2 steps per mode, late-time growth fits and
       a reduced mixing-length feature
     - CPU ``1.70x`` / ``2.75x`` / ``5.41x`` on 2 / 4 / 8 requested workers
       (six ensemble chunks); two RTX A4000 GPUs ``1.71x`` (86 per cent)

These are the figures to cite for parallelization speedup claims. The UQ
feature observable is parallelization and UQ plumbing; it is not a promoted
absolute nonlinear heat-flux predictor. The combined artifacts cite their split
CPU and GPU companions (``*_cpu_large`` and ``*_gpu_large``), which are
regenerated with:

.. code-block:: bash

   KY=$(python -c "print(','.join(f'{0.04 + 0.0125*i:.3f}' for i in range(64)))")

   python scripts/profiling/profile_parallel_workloads.py independent-ky \
     --backend cpu --devices 1,2,4,8 --ky "$KY" \
     --ny 128 --nz 96 --nl 4 --nm 8 --steps 240 \
     --out-prefix docs/_static/independent_ky_scan_scaling_cpu_large
   python scripts/profiling/profile_parallel_workloads.py independent-ky \
     --backend gpu --devices 1,2 --ky "$KY" \
     --ny 128 --nz 96 --nl 4 --nm 8 --steps 240 \
     --out-prefix docs/_static/independent_ky_scan_scaling_gpu_large
   python scripts/artifacts/plot_scaling_panels.py independent-ky

   python scripts/profiling/profile_parallel_workloads.py quasilinear-uq \
     --backend cpu --devices 1,2,4,8 \
     --out-prefix docs/_static/quasilinear_uq_ensemble_scaling_cpu_large
   python scripts/profiling/profile_parallel_workloads.py quasilinear-uq \
     --backend gpu --devices 1,2 \
     --out-prefix docs/_static/quasilinear_uq_ensemble_scaling_gpu_large
   python scripts/artifacts/plot_quasilinear_diagnostics.py uq-ensemble-scaling

Two smaller identity gates cover the same policy. The solver-backed Cyclone
``k_y``-scan gate runs the linear solver serially and with fixed-shape ``k_y``
batching and checks ``gamma`` and ``omega`` identity
(``docs/_static/parallel_ky_scan_gate.json``, zero mismatch). The logical-CPU
gate exercises ``RuntimeParallelConfig`` and ``batch_map`` on a structured
pytree output (``docs/_static/logical_cpu_parallel_scan_gate.json``,
``max_gamma_rel_error=6.7e-8``, ``max_ql_rel_error=1.0e-7``, zero ``omega``
error); it verifies the parallel API, not gyrokinetic physics. Their timings are
engineering metadata only.

.. code-block:: bash

   python scripts/artifacts/generate_parallel_identity_gate.py ky-scan
   python scripts/artifacts/generate_parallel_identity_gate.py logical-cpu --logical-devices 2

Production closure status
-------------------------

``docs/_static/parallelization_completion_status.json`` combines the production
scaling evidence and the diagnostic decomposition gates into one
machine-readable claim boundary. It reports the release
production-completion percentage and the status of each lane: production
independent-work parallelization is closed for independent ``k_y`` scans and
quasilinear/UQ ensembles, with the best speedups quoted above, and it embeds the
independent UQ/optimization provenance gate for serial-vs-parallel ordering,
worker clipping, exception metadata, and deterministic reconstruction.
Whole-state nonlinear sharding and FFT-axis decomposition remain diagnostic,
not production nonlinear speedup claims.

The lower-level decomposition-contract status checks deterministic shard
assignment, serial reconstruction identity, and claim-level separation without
rerunning large profiles. It passes for production independent ``k_y`` and UQ
portfolios and for a diagnostic nonlinear state-domain partition. Passing the
diagnostic row does not imply runtime nonlinear domain decomposition: it only
proves that the metadata split/reassemble contract is internally consistent and
correctly scoped as non-production.

Nonlinear runs: the species x Hermite mesh
------------------------------------------

The nonlinear decomposition GKX routes production runs through is a 2-D
``(species, hermite)`` device mesh under ``jax.shard_map``, species factored
first and Hermite second, with ``(ky, kx)``, the Laguerre axis and ``z``
replicated on every device. Every FFT the RHS performs -- both bracket
transforms, the periodic ``z`` derivative and the twist-shift linked chains --
runs over an axis this mesh never splits, so a shard is a slab the *serial*
kernels already evaluate and there is no distributed transpose anywhere in the
operator. The route is
``gkx.parallel.integrators.integrate_nonlinear_species_hermite``; it accepts the
explicit methods (``euler``, ``rk2``, ``rk3`` and its variants, ``rk4``,
``sspx3``).

Factoring rule (``gkx.parallel.velocity_plan.build_species_hermite_mesh_plan``):
the species block count is the largest factor of the device count that divides
``Ns``, the Hermite block count is the remainder, ``Nm`` must be **exactly**
divisible by it, and each Hermite block must hold at least the width-2 halo. A
padded Hermite axis would put a shard's ghost rows outside its neighbour, so an
indivisible request fails closed and the runtime error names the device counts
that do work. Two devices with two species therefore give a mesh ``(2, 1)``:
halo-free, one collective.

Collectives, and nothing else:

.. list-table::
   :header-rows: 1
   :widths: 12 20 68

   * - Tag
     - Primitive
     - What it carries
   * - C1
     - ``psum``
     - The masked global ``m = 0..3`` moment head, then the species sum inside
       the field solve. Delivers those rows to every shard in one collective
       rather than a reduce followed by a broadcast; adding exact zeros from the
       non-owning Hermite blocks leaves the owner's contribution bit-for-bit
       unchanged.
   * - C2
     - ``ppermute`` x2
     - One width-2 Hermite halo, one exchange per direction, covering the
       ``m +- 1`` ladder of streaming and mirror and the ``m +- 2`` reach of
       curvature together. Emitted only when the Hermite axis is actually split.
   * - C5
     - ``psum``
     - The four scalar traces, accumulated inside the integration scan carry.

``tests/unit/parallel/test_parallel_linear_velocity.py`` pins that structure on
the compiled HLO: the route emits no ``all-to-all`` (one would mean a
perpendicular axis had been split), it does emit an ``all-reduce``, and the
halo is exactly two ``collective-permute`` operations.

Shard-local kernels are the unmodified serial kernels. That is possible because
a ``HermiteWindow`` carries the *global* moment coordinates of the slab into the
operators whose coefficient is a function of the global Hermite index -- the
streaming field drives at ``m = 0, 1, 2``, the diamagnetic drive at
``m = 0..3``, the reflectionless closure at the last moment, ``build_H``, and the
hypercollision normalization. Without it every shard would drive its own local
row zero and close the hierarchy at its own last moment, which is a wrong answer
rather than a slow one. Conserving collisions are the one term family a split
Hermite axis does not serve yet: they read the ``m = 0, 1, 2`` moments of the
local slab, so a run with nonzero collisions and more than one Hermite block
raises ``NotImplementedError``. A species-only mesh is unaffected.

**Placement is staged from host.** Handing ``device_put`` an array already
committed to one device asks the runtime to reshard it across the mesh, and on
the two-GPU box that path silently returns a wrong answer (maximum relative
error 1.0 against serial, while a one-device mesh on the same GPU is exact).
``gkx.parallel.integrators.stage_from_host`` round-trips through host memory
once per run and is a no-op when the buffer is already correctly placed.

Identity is gated in the same test file: the sharded RHS matches the serial
production nonlinear RHS to ``1e-6`` relative on one-, two- and four-device
meshes, and a fixed-step trajectory with its scan-fused scalar traces matches
serial to ``5e-6`` absolute.

Speedup status: **not a production speedup claim.** The nonlinear bracket is
kept in its own kernel (``optimization_barrier``) because XLA otherwise folds
the whole linear operator into the elementwise ``add`` that joins it to the
bracket, and that fusion is several times slower on a one-species shard. The
test that pins the split records the measurement behind it on 2 x RTX A4000
(jax 0.11.1, grid ``(Ns, Nl, Nm, Nky, Nkx, Nz) = (2, 8, 16, 64, 64, 32)``):
two-device scaling of ``1.30x`` with the fusion and ``2.14x`` without, against
a ``1.90x`` scaling gate. No profiler artifact for that workload is tracked
under ``docs/_static``, so the number stays a recorded engineering measurement
until one is. Logical CPU devices share the same cores, so a CPU run of this
route is an identity check, not a timing.

Runtime routing for nonlinear runs
----------------------------------

``[parallel]`` is read by the nonlinear runtime path as well as the linear
one, and every request it cannot honour raises rather than silently running
serially. One sharded nonlinear run is requested with:

.. code-block:: toml

   [parallel]
   strategy = "shard_map"
   axis = "species_hermite"
   num_devices = 2
   strict_identity = true

or, letting the mesh follow the visible devices:

.. code-block:: toml

   [parallel]
   auto = true

``auto`` resolves to ``strategy = "shard_map"`` and
``axis = "species_hermite"`` over every visible device, with
``strict_identity = true``, and the resolved plan is printed before the run
starts, for example::

   routing nonlinear run through shard_map nonlinear route on a auto-selected
   species x Hermite mesh (2 species x 1 Hermite) across 2 devices, shard
   (1, 4, 16, 8, 8, 8), collectives: field psum, no halo (strict_identity=on)

``auto`` never silently overrules an explicit request: a conflicting
``strategy`` or ``axis`` is a configuration error. The accepted axis aliases for
the production mesh are ``species_hermite``, ``velocity``, ``s_m``, ``species``,
``m`` and ``hermite``. ``axis = "ky"`` is also accepted, as a routing
diagnostic: ``num_devices`` must divide the ``k_y`` extent the state actually
stores, and the refusal names the layout that extent came from.

For either axis the runtime places the whole nonlinear state on the mesh and
runs the ordinary production integrator on it, so the operator is the
production nonlinear RHS rather than a reduced stand-in. The audited
``shard_map`` route with the named collectives above is the one the fixed-step
trajectory gate covers; the runtime lane does not yet call it. This is a
*routing* claim only, and no speedup is claimed for either runtime axis.

Routing is fail-closed on numerical identity. With ``strict_identity = true``
the run is also executed serially and the two answers must agree on the final
state and on the ``Wg``, ``Wphi``, heat-flux, and particle-flux traces, with the
same allclose-style tolerance convention (``atol = 5e-6``,
``rtol = 1e-4``). A violation raises ``NonlinearParallelIdentityError`` and the
sharded result is discarded; it is never returned in place of the serial answer.
Setting ``strict_identity = false`` skips the serial reference and is only
appropriate once the gate is known to pass for that workload. The sharded route
needs at least two JAX devices; on CPU they can be forced with
``XLA_FLAGS=--xla_force_host_platform_device_count=N`` for testing, which splits
one thread pool and is slower than serial (:doc:`performance`).

Every other combination raises ``NonlinearParallelRoutingError`` naming the
supported set. In particular ``axis = "z"`` is rejected, for two separate
reasons:

- the production parallel-streaming derivative is a spectral FFT along ``z``, so
  a whole-state ``z`` shard does not survive SPMD partitioning; and
- the retired device-z pencil route (``gkx.operators.nonlinear.device_z``,
  removed in ARCH-A) evaluated a reduced diagnostic bracket operator with a
  model field solve and no streaming, mirror, curvature, collision, or species
  terms. Its serial-vs-sharded identity was real, but it was identity for a
  different operator than the one a runtime nonlinear run integrates, so it
  could not stand in for the production RHS.

The independent-work strategies (``batch``, ``combined_ky``) are also rejected
on this path: they orchestrate separate solver calls for ``k_y`` scans and
ensembles and cannot shard a single nonlinear run.

Diagnostic path: whole-state nonlinear sharding
-----------------------------------------------

Fixed-step whole-state nonlinear sharding is diagnostic-only. The
``integrate_nonlinear_sharded`` / ``TimeConfig.state_sharding`` path
(``"auto"`` or a state axis such as ``"ky"`` or ``"kx"``) uses a ``pjit`` scan
that preserves the serial Runge-Kutta update. It is useful for control-flow
validation, state-axis identity gates, profiler localization, and testing
candidate layouts. It is not a production nonlinear domain decomposition or
multi-GPU speedup claim. Do not use it as evidence for a whole-state nonlinear
sharding speedup; it has no scoped speedup claim until separate identity gates
and matched profiler artifacts exist for that exact workload.

Whole-state sharding does not close the communication problem for nonlinear
FFTs, halo exchange, conservation checks, or benchmark-size transport runs.
``z``-axis FFT sharding is not release-gated because the JAX/XLA FFT layout does
not pass the multi-device identity gate. On forced multi-device CPU backends the
whole-state ``pjit`` route can abort inside XLA's CPU FFT layout code, so the
profiler skips active multi-device CPU whole-state sharding by default and
records ``cpu_whole_state_pjit_sharding_unsafe_for_fft_layout`` as a fail-closed
blocker; ``--allow-unsafe-cpu-state-sharding`` is for bounded debugging only.

The artifacts that control the conclusion:

- ``docs/_static/nonlinear_sharding_profile_office_gpu_physical.json``: three
  interacting spectral perturbations on a ``(4,8,32,32,64)`` state for 100 RK2
  steps with a nonzero initial nonlinear RHS. On two RTX A4000 GPUs serial
  execution takes 3.77 s, while ``ky`` and ``kx`` whole-state placement take
  9.19 s and 7.22 s, and both fail final-state and RHS identity.
- ``docs/_static/nonlinear_sharding_profile_office_gpu_benchmark_grid.json``:
  ``(4,8,64,192,24)``, 20 RK2 steps. Active ``kx`` sharding runs at ``0.21x``
  of serial (0.893 s serial against 4.22 s sharded) and fails trajectory
  identity (``max_abs_state_error=20.0``, ``max_abs_rhs_error=1279``) with a
  zero potential difference, which is why a potential-only check is not a
  sufficient gate.
- ``docs/_static/nonlinear_sharding_strong_scaling_large.json``: the final state
  is identity-correct at every tracked point, but logical-CPU speedup saturates
  near ``1.39x`` and the two-RTX-A4000 ``auto`` route is slower than one GPU on
  the larger ``Nx=48, Ny=96, Nz=128, Nl=4, Nm=8`` fixed-step case (``0.586x``
  strong scaling). The combined artifact is fail-closed: ``identity_passed`` may
  be true while ``speedup_passed`` is false, with explicit ``speedup_blockers``
  naming the backend/device row that regressed.
- ``docs/_static/nonlinear_sharding_production_speedup_gate.json``: the only
  artifact that may promote whole-state nonlinear sharding wording beyond
  diagnostic/profiler evidence, and only for the exact workload it gates. It
  fails closed as ``diagnostic_only`` unless the CPU and GPU rows both pass
  serial identity, use active state sharding, and meet the speedup and
  parallel-efficiency thresholds. Its ``backend_blocker_report`` keeps
  identity-evidence blockers separate from speedup/efficiency blockers, so an
  identity-complete slowdown remains diagnostic.

For final-state-only profiling the explicit scan accepts ``return_fields=False``,
which skips the post-step RHS evaluation that exists only to materialize the
final field history. The artifacts are regenerated with isolated subprocesses
so each device count gets a clean JAX runtime:

.. code-block:: bash

   python scripts/profiling/profile_nonlinear_sharding.py \
     --sharding auto --sharding-options auto,kx \
     --out-json docs/_static/nonlinear_sharding_profile.json

   python scripts/profiling/profile_nonlinear_sharding.py sweep \
     --backend cpu --devices 1,2,4,8 \
     --nx 24 --ny 48 --nz 96 --nl 4 --nm 8 --steps 8 \
     --out-prefix docs/_static/nonlinear_sharding_strong_scaling_cpu_large
   python scripts/profiling/profile_nonlinear_sharding.py sweep \
     --backend gpu --devices 1,2 \
     --nx 48 --ny 96 --nz 128 --nl 4 --nm 8 --steps 12 \
     --out-prefix docs/_static/nonlinear_sharding_strong_scaling_gpu_xlarge
   # Equivalent two-GPU preset with JAX traces enabled.
   python scripts/profiling/profile_nonlinear_sharding.py sweep --office-gpu-xlarge

   python scripts/artifacts/plot_scaling_panels.py nonlinear-sharding
   python scripts/artifacts/generate_nonlinear_sharding_production_gate.py

The profiler JSON records device count, requested sharding axis, warm
serial/sharded timings, profiler-trace status, final-state and final-field/RHS
errors, the fastest identity-preserving candidate, and a versioned source
contract (command, argv, backend, device count, warmup/repeat policy and
software versions). The small checked-in ``nonlinear_sharding_profile.json`` and
``nonlinear_sharding_profile_office_gpu.json`` are control-flow smoke tests on
states with a zero or tiny nonlinear bracket and are superseded for physical
identity decisions. The fast checker
``scripts/checks/check_parallel_scaling_artifacts.py`` validates the gate, its
CSV sidecar, its CPU/GPU source rows, its required-backend blockers, and the
per-backend blocker report without rerunning any profiler.

.. _device-z-route-overhead:

Diagnostic path: nonlinear domain and device-z decomposition
------------------------------------------------------------

These prototypes decompose the perpendicular spectral plane or the field-line
axis. None of them runs the production nonlinear RHS, and none is a production
nonlinear speedup claim. Their source modules (``gkx.operators.nonlinear``
``parallel``, ``spectral_core``, ``spectral_identity_*``,
``domain_decomposition``, ``parallel_contracts_*`` and ``device_z``) and their
profilers left the package in ARCH-A (2026-09-23) because no run, CLI command or
solver path reached them; the JSON artifacts below are kept as frozen evidence
of what was measured.

- **Local state-domain gate.**
  ``docs/_static/nonlinear_domain_parallel_identity_gate.json`` checks a
  deterministic local nonlinear state update with one-cell halo chunks against
  the serial update, and its embedded ``nonlinear_domain_transport_window_identity``
  sub-gate advances a short fixed-step window comparing final state, boundary,
  mass, free-energy-proxy and boundary-flux-proxy traces. Any recorded blocker
  (noncanonical axes, incomplete chunk coverage, shape mismatch) disables the
  prototype path even if the arrays compare equal. It does not validate
  distributed FFTs, field solves, runtime routing or speedup.
- **Spectral communication gate.**
  ``docs/_static/nonlinear_spectral_communication_identity_gate.json`` checks, on
  ``(N_l,N_m,N_y,N_x,N_z)`` spectral data, the split/reassemble and transpose
  operations a distributed FFT would need, a tile-reassembled nonlinear RHS
  ``-\{\phi,g\}``, a short fixed-step micro-integration, and a device-z pencil
  fused-bracket route over a short transport window. Passing it is what made
  ``fft_axis_domain`` diagnostic rather than blocked; logical tiles were
  reconstructed for identity validation, not executed as a distributed FFT.
- **Routing work model.**
  ``docs/_static/nonlinear_spectral_domain_routing_profile.json`` on a
  ``(2,4,32,32,4)`` four-tile profile: the global-reconstruction route has a
  communication/owned-work ratio of ``6.375`` and a parallel-efficiency ceiling
  of ``0.136``, and times at ``1.08x`` of serial; the pencil route's model gives
  ``0.075`` and ``0.930``, but its recorded timing (``0.75x``) predates the
  fused local bracket below and has not been regenerated.
- **Device-z pencil bracket.** The ``z``-sharded route keeps both transform
  axes local to each device, so the local bracket is computed with the same
  fused perpendicular transform the serial route uses and sharding ``z``
  involves no collectives in the bracket. The current result is
  ``docs/_static/nonlinear_device_z_pencil_scaling_decomposition_gpu2_fused_profile.json``:
  on two RTX A4000 GPUs, timed one grid per process with the
  ``--isolate-shapes`` flag of the device-z profiler (retired with the route in
  ARCH-A; the JSON is frozen evidence), the
  single-device route overhead is ``0.988`` to ``1.005`` of the fused serial route,
  one-to-two-device scaling is ``1.92x`` to ``2.01x``, and the net speedup is
  ``1.95x`` to ``2.01x`` over five grids from ``(4,8,96,96,48)`` to
  ``(4,16,128,128,64)``, with serial-versus-sharded final-state error exactly
  ``0.0``. Isolation is a correctness requirement: grids timed in one process
  share compiled executables and a warm allocator and have been measured to
  under-report the route overhead.

  This is the bracket micro-route -- a five-dimensional state, a model field
  solve ``phi = n/(1+kperp^2)`` and the ExB bracket alone, with no streaming,
  drift, collision, dissipation, electromagnetic or species terms -- so it is a
  localization result, not a production nonlinear speedup claim, and the
  runtime rejects ``axis = "z"`` for the reasons above.

  The scalar diagnostic path is a separate and much larger cost:
  ``docs/_static/nonlinear_device_z_pencil_transport_gpu2_observable_split_profile.json``
  records a host-gathered observable gate costing ``42.6`` times the sharded
  compute median on the auto-chunked ``(4,16,96,96,64)`` two-GPU diagnostic.
  ``--observable-repeats`` adds that split to a profile, and
  ``--observable-mode sharded_reduce`` computes the observables through device
  z reductions for identity debugging, recomputing the bracket to do so. The
  earlier transport-window, RHS and granularity profiles
  (``nonlinear_device_z_pencil_{rhs,transport}_{cpu4,gpu2}_profile.json``,
  ``nonlinear_device_z_pencil_transport_gpu2_granularity_profile.json`` and
  ``nonlinear_device_z_pencil_scaling_decomposition_gpu2_profile.json``) were
  measured with the axis-staged bracket the fused one replaced and remain as
  identity evidence only. ``--z-chunk-size`` and ``--auto-z-chunk-size`` are a
  feasibility control for cuFFT batched-plan failures on large grids (with
  ``XLA_PYTHON_CLIENT_PREALLOCATE=false``), not a performance control.

Before nonlinear domain decomposition can be promoted beyond this diagnostic
state, the runtime route must pass all of the following gates on the same
workload family that appears in the speedup figure:

- full nonlinear RHS identity for ``dG``, ``phi``, the nonlinear bracket,
  density/field-solve layout, Hermitian projection, and dealiasing;
- fixed-step serial-vs-decomposed integration identity for final state,
  final fields, final RHS, and per-step scalar traces;
- boundary/interface identity for owned and halo cells, not only a global norm;
- conservation agreement for density/mass, a free-energy-like diagnostic,
  zonal response, and heat-flux proxies;
- post-transient transport-window agreement for Cyclone, KBM, and at least one
  stellarator smoke case;
- CPU serial, CPU decomposed, one-GPU serial, and two-GPU decomposed parity
  under the same observable contract;
- matched profiler artifacts for the exact backend, device count, software
  stack, grid, warmups/repeats, and identity tolerance being claimed.

Until those gates exist, nonlinear decomposition work can be documented as
diagnostic engineering evidence only, even if a new profile shows positive
timing on one machine.

Velocity-space routes for linear runs
-------------------------------------

Velocity-space decomposition of the linear operator is gated from the bottom
up. ``gkx.build_velocity_sharding_plan`` records a species-first,
Hermite-second layout, including which axes need Hermite ghost exchange and
which need field-solve reductions and broadcasts. Every route is opt-in through
``gkx.linear_rhs_parallel_cached`` with
``RuntimeParallelConfig(strategy="velocity", ...)``, and each rejects the terms
it has no identity gate for rather than dropping them.

The communication and call-graph layers, each with its own identity artifact
(two logical CPU devices unless stated):

.. list-table::
   :header-rows: 1
   :widths: 34 38 28

   * - Layer
     - Artifact (``docs/_static``)
     - Recorded error
   * - Hermite nearest-neighbour ghost exchange
     - ``hermite_exchange_gate.json``
     - zero
   * - Hermite-sharded field reduction (``lax.psum``)
     - ``velocity_field_reduce_gate.json``
     - ``2.8e-9`` relative
   * - Hermite streaming-ladder coefficients
     - ``hermite_streaming_ladder_gate.json``
     - zero ladder error; reduction ``1.9e-6``
   * - Electrostatic field reduction (``m=0`` density to ``phi``)
     - ``electrostatic_field_reduce_gate.json``
     - zero
   * - Periodic streaming microkernel against ``streaming_ladder_term``
     - ``periodic_streaming_microkernel_gate.json``
     - zero
   * - Streaming-only linear RHS (``backend="streaming_only"``)
     - ``linear_rhs_streaming_gate.json``
     - ``5.6e-7`` relative
   * - Streaming with an electrostatic field (``backend="streaming_electrostatic"``)
     - ``linear_rhs_streaming_electrostatic_gate.json``
     - ``4.0e-7`` relative; ``phi`` ``1.9e-9``
   * - Mirror and curvature/grad-B drift slices
     - ``electrostatic_drift_gate.json``
     - zero
   * - Diamagnetic drive
     - ``electrostatic_diamagnetic_gate.json``
     - zero
   * - Composed single-species periodic electrostatic RHS
       (``backend="electrostatic_linear_slices"``)
     - ``linear_rhs_electrostatic_slices_gate.json``
     - ``3.7e-7`` relative; ``phi`` ``4.2e-9``

They are regenerated by ``scripts/artifacts/generate_velocity_parallel_gates.py``
(``hermite-exchange``, ``field-reduce``, ``hermite-ladder``,
``periodic-streaming``), ``scripts/artifacts/generate_electrostatic_parallel_gates.py``
(``field-reduce``, ``drift``, ``diamagnetic``) and
``scripts/artifacts/generate_linear_rhs_parallel_gates.py`` (``streaming``,
``streaming-electrostatic``, ``electrostatic-slices``), each with
``--logical-devices 2``.

**Species route.** ``RuntimeParallelConfig(strategy="velocity", axis="species",
num_devices=2)`` evaluates the complete electrostatic linear-slice RHS with one
species per device: the shared quasineutrality collective first, then
streaming, mirror, curvature, grad-B and diamagnetic terms on local species
shards. The enclosing explicit species ``pmap`` integrator stages
species-dependent state and cache arrays from host once, supports the built-in
conserving collision operator, hypercollisions and the electromagnetic field
equations (density, parallel-current, polarization and perpendicular-pressure
moments reduced with ``lax.psum``), and preserves reverse-mode differentiation
through the compiled loop; IMEX stays fail-closed. The standalone ``shard_map``
RHS keeps collisions fail-closed.

**Mixed species--Hermite route.** ``backend="electrostatic_species_hermite"``
with ``axis="species_hermite"`` and four devices evaluates the periodic,
collision-free electrostatic two-species RHS on a ``(species, m) = (2, 2)`` mesh.
Quasineutrality reduces density over both mesh axes and polarization over
species only; the Hermite ladder exchanges boundary moments within each species
row; global Hermite and Laguerre indices place the diamagnetic drive, close its
Laguerre sum with the analytic truncation coefficient
:math:`\mathcal J_{L}=-\mathcal J_{L-1}(b/2)/L`, and normalize constant and
:math:`|k_z|` hypercollisions as serial does. Linked flux-tube boundaries use the
production chain FFT on each shard because ``ky``, ``kx`` and ``z`` stay local,
and a nontrivial linked case with conserving collisions passes state/field
identity on four logical CPU devices. Mixed-mesh electromagnetic fields, other
integrators and all GPU claims remain fail-closed; the two-GPU office host
cannot test a four-device mixed mesh.

Tracked engineering profiles for these routes (``docs/_static``):

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Artifact
     - Result
   * - ``linear_rhs_parallel_slices_profile.json``
     - Hermite-heavy slices route, eight logical CPU devices: ``1.40x`` warm
       RHS, ``5.96e-6`` relative error.
   * - ``linear_rhs_parallel_slices_sweep.json``
     - Hermite resolution x logical device count: identity passes at every
       point; overhead-limited at one and two devices, best ``1.57x`` at
       ``Nm=128`` on four. A regime map, not a scaling claim.
   * - ``linear_rhs_parallel_slices_profile_gpu.json``
     - Two RTX A4000 GPUs: identity passes, ``0.03x`` of the single-GPU serial
       route. No GPU Hermite-sharding claim.
   * - ``linear_rhs_species_profile_gpu.json``
     - Species route, ``2x8x32x128x1x128`` state, two RTX A4000 GPUs:
       ``5.3e-8`` relative error, 8.21 ms against 7.11 ms, a scoped ``1.16x``
       warm-RHS speedup; smaller states are slower on two GPUs.
   * - ``linear_rhs_species_profile_cpu.json``
     - Species route, two logical CPU devices: ``3.41x`` isolated RHS, exact
       100-step Euler state/field histories, ``0.96x`` end to end.
   * - ``linear_rhs_species_hermite_profile_cpu.json``
     - Mixed ``(2, 2)`` mesh, four logical CPU devices: ``3.11x`` warm RHS,
       exact 100-step state/field histories, ``0.97x`` end to end.

They are produced by ``scripts/profiling/profile_linear_rhs_parallel_slices.py``
(``--axis species`` or ``--axis species_hermite`` for the species routes,
``--integration-steps`` for the trajectory rows, ``sweep`` for the regime map),
for example:

.. code-block:: bash

   python scripts/profiling/profile_linear_rhs_parallel_slices.py sweep \
     --platform cpu --devices 1,2,4,8 --nms 64,128 \
     --nl 4 --ny 32 --nz 128 --rtol 1e-5

   python scripts/profiling/profile_linear_rhs_parallel_slices.py \
     --axis species_hermite --platform cpu --logical-devices 4 \
     --nl 4 --nm 16 --ny 64 --nz 64 --warmups 2 --repeats 7 \
     --integration-steps 100 --integration-repeats 5 \
     --integration-dt 1e-7 --integration-sample-stride 1 \
     --out-prefix tools_out/linear_rhs_species_hermite_profile_cpu

These gates validate communication and numerical identity for the stated
bounded linear paths. They do not validate mixed-mesh electromagnetic fields,
multi-species nonlinear field solves, nonlinear brackets, or nonlinear
transport speedup.

Diagnostic gate: differentiable species initial states
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Species-parallel linear solves preserve traced initial states through input
placement. Concrete inputs retain explicit placement; traced inputs never
round-trip through host NumPy. For fixed parameters and linear evolution,

.. math::

   J(s)=\|G_T(sG_0)\|^2+\|\phi_T(sG_0)\|^2,
   \qquad J'(1)=2J(1).

The electromagnetic trajectory test checks this identity with active
:math:`A_\parallel,B_\parallel`, serial versus species ``pmap``, and eager versus
outer-jitted reverse differentiation. Run in separate precision processes:

.. code-block:: bash

   JAX_ENABLE_X64=true XLA_FLAGS=--xla_force_host_platform_device_count=2 \
     python -m pytest tests/unit/parallel/test_parallel_linear_velocity.py \
     -k species_pmap_electromagnetic_trajectory_matches_serial

Repeat with ``JAX_ENABLE_X64=false`` for float32. This three-step identity is
an AD/routing contract, not electromagnetic model validation, long-time
turbulent sensitivity accuracy or a parallel speedup measurement.

Claim rules
-----------

Use the following rules when writing docs, release notes, or papers:

- Call independent ``k_y``/UQ/ensemble batching the production-ready
  parallelization path when the serial identity gate is current.
- For runtime scan TOMLs, use ``[parallel] strategy = "batch"`` with
  ``axis = "ky"`` only for independent ``k_y`` scan orchestration.
- Call the nonlinear species x Hermite route production-routed and
  identity-gated; do not claim speedup for it until a matched profiler artifact
  is tracked.
- Call whole-state nonlinear sharding a diagnostic correctness/profiler gate,
  not production nonlinear parallelism unless the exact workload has passed its
  identity and profiler promotion gates.
- Call the electrostatic two-species linear route production-routed and
  identity-gated, but do not claim speedup until its matched workload profile
  passes. Other velocity-space ``shard_map`` work remains communication-gated
  and opt-in.
- Call the mixed species--Hermite linear backend scoped and identity-gated
  for periodic Euler/RK2 integration. Quote speedup only for its exact tracked
  four-logical-CPU workload.
- Do not claim nonlinear speedup from sharding, velocity decomposition, spectral
  toggles, or linear-slice profiles without passing identity gates and fresh
  profiler artifacts for the exact workload, backend, device count, software
  stack, and identity tolerance being claimed.
- Keep speedup plots separate from identity gates: identity establishes
  correctness; profiler artifacts establish only the scoped timing claim they
  measure.

Large-run scaling acceptance checklist
--------------------------------------

A CPU/GPU strong-scaling result is release-ready only when the tracked
artifacts satisfy all of the following:

- the combined ``*_large`` JSON/CSV/PNG/PDF files point back to split CPU and
  GPU source artifacts for the same workload family;
- every split artifact records the actual problem size, backend, requested
  device counts, warmup/repeat policy, and positive per-worker or per-profile
  timing samples;
- every row has ``identity_gate_pass = true`` and compares against the
  one-worker or one-device serial reference for the observable being claimed;
- nonlinear whole-state sharding rows embed the per-device profiler/profile
  payload, including trace-request status, serial timing stats, sharded timing
  stats, selected axis, and final-state error metrics;
- any speedup wording names the exact backend, device count, workload, grid,
  software stack, identity tolerance, and artifact files that produced it.

If any item is missing, the result can be kept as local engineering evidence
only. In particular, whole-state nonlinear sharding remains not a production
nonlinear speedup claim, even when the embedded profile reports a positive
engineering timing ratio. Promoting that lane requires fresh profiler artifacts
for the exact workload plus full nonlinear identity, conservation, field-solve,
FFT/bracket communication, and transport-window gates.

Fast artifact contract check
----------------------------

Before editing scaling docs or manifests, run the checked-in artifact contract:

.. code-block:: bash

   python scripts/checks/check_parallel_scaling_artifacts.py

This command does not rerun large profiles and does not enforce any minimum
speedup. It validates that the tracked JSON/CSV/PNG/PDF sidecars exist, the
``parallel_scaling`` manifest lists them, split CPU/GPU source artifacts are
attached where required, numerical identity gates pass, error fields are finite,
and timing/profiler payloads are positive and scoped to their documented claim.

Release artifact policy
-----------------------

The release-gated parallelization artifacts are grouped by what they are
allowed to support:

.. list-table::
   :header-rows: 1
   :widths: 28 24 24 24

   * - Artifact family
     - Primary files
     - Claim allowed
     - Claim not allowed
   * - Independent ``k_y`` scans
     - ``independent_ky_scan_scaling_large.{json,csv,png,pdf}``
     - Production parallelization for independent linear scans when
       ``gamma``/``omega`` identity is current.
     - Nonlinear domain decomposition or nonlinear transport speedup.
   * - Quasilinear/UQ ensembles
     - ``quasilinear_uq_ensemble_scaling_large.{json,csv,png,pdf}``
     - Production batching for independent reduced-feature and UQ workloads.
     - Promoted absolute nonlinear heat-flux prediction.
   * - Whole-state nonlinear sharding
     - ``nonlinear_sharding_strong_scaling_large.{json,csv,png,pdf}``
     - Correctness and profiler-direction evidence for the ``pjit``
       state-axis layout.
     - Production nonlinear multi-GPU speedup.
   * - Prototype nonlinear state-domain gate
     - ``nonlinear_domain_parallel_identity_gate.{json,png}``
     - Fail-closed serial-vs-halo-decomposed identity evidence for one bounded
       local stencil, including the embedded transport-window proxy traces.
     - Distributed FFT, field-solve, production conservation, transport-runtime,
       or speedup claims.
   * - Prototype nonlinear spectral communication gate
     - ``nonlinear_spectral_communication_identity_gate.{json,png}``
     - Fail-closed split/reassemble identity evidence for FFT round trip,
       pseudo-spectral bracket, and spectral field-solve layout.
     - Runtime distributed FFT routing, nonlinear conservation,
       transport-window, or speedup claims.
   * - Velocity-space linear slices
     - ``linear_rhs_parallel_slices_sweep.{json,png,pdf}``
     - Bounded engineering evidence for opt-in electrostatic linear RHS slices.
     - Electromagnetic, linked-boundary, collision, or nonlinear speedup.

Both ``tools/performance_optimization_manifest.toml`` and
``tools/validation_coverage_manifest.toml`` list these artifacts explicitly.
The tests require the manifests, files, and claim scopes to stay synchronized,
so deleting or silently reinterpreting a scaling artifact fails the fast
parallelization gate.
