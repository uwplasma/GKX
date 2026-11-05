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
must accommodate the width-two halo. The perpendicular plane, Laguerre
moments and field-line coordinate remain replicated. Consequently, the FFTs
operate on local complete axes. Field moments use collective reductions;
Hermite neighbours use halo exchanges when that axis is split.

``gkx.parallel.integrators.integrate_nonlinear_species_hermite`` provides
the named-collective integration path. The runtime and prepared APIs validate
method, timestep policy and operator support before execution. An unsupported
combination raises an error; changing the method or collision model to make a
mesh fit changes the numerical problem and requires its own validation.

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

Benchmark warm runs after compilation, wait for device completion and report
the workload, backend, device count, precision and numerical tolerances with
the timing. Reproduce results with the checked-in tests and benchmark drivers;
keep generated measurements outside version control. See :doc:`performance`
for timing practice and :doc:`nonlinear_autodiff` for derivative limits.
