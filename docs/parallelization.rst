Parallel execution
==================

GKX supports independent solver batches and an opt-in species--Hermite mesh
for a single nonlinear solve. Parallel execution defaults to serial. Device
placement, numerical identity and speedup are separate properties: measure
execution time only after checking the observables against a serial run.

Independent scans and ensembles
-------------------------------

Use ``gkx.ky_scan_batches`` and ``gkx.batch_map`` for JAX-array workloads,
and ``gkx.independent_map`` for Python tasks. Independent tasks retain serial
result ordering. A runtime linear scan can request workers through TOML:

.. code-block:: toml

   [parallel]
   strategy = "batch"
   axis = "ky"
   num_devices = 2
   backend = "auto"

``backend`` also accepts ``"thread"`` and ``"process"``. This policy runs
separate solver calls; it does not partition one distribution array. Choose
worker counts that fit both device memory and host resources.

Nonlinear species--Hermite mesh
-------------------------------

Request the nonlinear mesh explicitly:

.. code-block:: toml

   [parallel]
   strategy = "shard_map"
   axis = "species_hermite"
   num_devices = 2
   strict_identity = true

Alternatively, ``[parallel] auto = true`` selects the species--Hermite mesh
from visible devices. The default ``auto`` is false. Conflicting explicit
strategy or axis settings raise a configuration error.

The mesh factors devices across species first, then Hermite moments. The
Hermite block count must divide the Hermite extent exactly, and each block
must accommodate the width-two halo. Auto mode selects the largest compatible
subset within the device budget; an explicit incompatible device count raises.
The perpendicular plane, Laguerre moments and field-line coordinate remain replicated. Consequently, the FFTs
operate on local complete axes. Field moments use collective reductions;
Hermite neighbours use halo exchanges when that axis is split.

``gkx.parallel.integrators.integrate_nonlinear_species_hermite`` provides
the named-collective integration path. Named collectives support ``rk2``,
``rk3``, ``rk4``, ``sspx3`` and adaptive ``rk4-pi`` with production diagnostics, adaptive chunks, progress and saturation
checks. Fixed-step diagnostics-off runs use a final-state scan when supported.
Diagnosed preparation uses the same backend with ``strict_identity = false``.
Its fixed-step array interface supports matched model differentiation under the
prepared topology constraints described in :doc:`api`.

Active physical collisions, sources, fixed-mode restrictions, checkpointing,
additional state sharding and reflectionless closures use the ordinary solver
with whole-state placement. IMEX policies also
use that fallback. Status messages identify this placement; it does not establish
distributed RHS work. Unsupported strategies or axes still raise.
Adaptive chain IMEX and generic ``imex`` preparation are refused. Changing a
method or collision model changes the numerical problem and requires its own
validation.

.. _unequal-moment-mesh:

Species with unequal moment counts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Species that keep different :math:`(N_l, N_m)` (:ref:`per-species-moments`) form
groups of different shape, and one ``shard_map`` program cannot give each device
a different shape. A padded state split over species or over the state's own
``Nm`` would leave the device that holds the smaller species idle on zeros.
Instead the mesh is Hermite-only, :math:`(1, D)`, and each group's Hermite axis
is cut into :math:`D` blocks of *its own* :math:`N_m`. A device then advances
:math:`\sum_g n_{s,g} N_{l,g} N_{m,g} / D` entries, whatever the counts, and
each group runs on its own exact-size cache. The device blocks are packed in one
array, so the stepper and the field reduction are those of the uniform mesh.

Every group's :math:`N_m` must be divisible by :math:`D` and leave blocks of at
least two moments; a count that does not divide raises and names the device
counts that do. The route serves the fixed-step final-state scan: ``rk2``,
``rk3``, ``rk4`` and ``sspx3`` with ``fixed_dt = true``, ``run_to = "t_max"``,
no diagnostics, and no progress output, and the library entry point
``integrate_nonlinear_species_hermite`` with its fused scalar traces. Diagnosed,
adaptive and ``rk4-pi`` runs, IMEX methods and the linear routes keep the
whole-state placement; the diagnosed route would repack the state on every
right-hand side. The run status line reports the groups and the entries per
device; ``examples/14_multi_device_species`` prints the split.

``strict_identity`` defaults to true. It compares the parallel result with a
serial execution, including the final state and energy and transport traces.
Its runtime tolerances are ``atol = 5e-6`` and ``rtol = 1e-4``. This additional
execution has a cost; keep it enabled while qualifying a new workload.

Limits and verification
-----------------------

The ``ky`` nonlinear axis is a routing diagnostic. It does not establish
distributed perpendicular FFT execution. A whole-state ``z`` split is
unsupported because the production streaming operator uses a spectral FFT
along that coordinate. Diagnostic split/reassemble and linear-slice helpers
do not establish nonlinear transport accuracy or multi-device speedup.

For each new mesh, compare the same initial state, timestep policy, precision,
collision model and output sampling against serial execution. Check final
states, fields, conserved quantities and transport windows. Logical CPU
devices are useful for communication and differentiation checks, but share
physical CPU resources and cannot establish hardware scaling.

Multiple processes
------------------

Initialize JAX distributed execution externally and enter the solve on every
rank. The selected mesh must contain devices from every initialized process.
Auto selection balances the participating device count across processes before
factoring the species--Hermite mesh. The ``ky`` placement diagnostic accepts
only one process.

A runtime call with ``return_state = true`` gathers the full global state onto
every host, so each host needs memory for that complete array. Strict identity
checks compare addressable shards and exchange error scalars rather than
gathering the state solely for validation. Plan state-return and host-memory
costs separately from the distributed device allocation.

Benchmark warm runs after compilation, wait for device completion and report
the workload, backend, device count, precision and numerical tolerances with
the timing. Reproduce results with the checked-in tests and benchmark drivers;
keep generated measurements outside version control. See :doc:`performance`
for timing practice and :doc:`nonlinear_autodiff` for derivative limits.
