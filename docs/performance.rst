Performance
===========

GKX compiles JAX array kernels and reuses operator caches across time steps.
Compilation cost depends on array shapes, method and static operator choices.
Changing those choices can trigger recompilation. Keep compilation and steady
execution time separate when comparing configurations.

Measuring execution
-------------------

Use the same input, precision, hardware allocation and output cadence for
both measurements. Warm the exact callable before timing and wait for device
completion: JAX dispatch is asynchronous. Report the number of warmups and
repetitions together with the spread of the measured times.

``gkx.prepare`` returns a reusable simulation. Its ``warmup`` method executes
the prepared nonlinear closure to trigger compilation; it therefore costs a
simulation run. ``estimate_memory`` reports an estimate for the case topology,
not a measured peak allocation. See :doc:`api` for the prepared interface.

The runtime caches geometry, species coefficients, spectral masks and basis
indices so RHS evaluations can reuse them. Reuse a prepared case when topology
is fixed. Batch independent parameter points only when the resulting live
states fit device memory; increasing the worker count can increase peak memory.

Chunk size and retained diagnostics
-----------------------------------

``[time] chunk_steps`` controls the number of steps between nonlinear runtime
handoffs. Larger chunks amortize host synchronization but delay monitoring and
status updates and can retain more diagnostic data at once. Adaptive chain
IMEX also retunes its factor between chunks, so check numerical tolerances
when changing its chunk length.

``[time] diagnostics_interval`` changes the retained history in physical time.
Adaptive host-side thinning can reduce returned and saved data without
reducing diagnostic kernel evaluations. Measure the actual compute and memory
costs; a smaller output file does not establish faster integration.

CPU and GPU runs
----------------

CPU thread pools, process workers and device batching compete for host
resources. Measure worker counts on the target machine. For GPU runs, record
the accelerator model, software versions, precision and whether other jobs
share the device. Transfer and output costs belong in the timing when
they are part of the intended workflow.

Profiler and benchmark drivers write generated summaries, traces and plots to
an output directory selected by the caller. Keep those outputs outside version
control. A reproducible result needs its input deck, command, numerical
acceptance criteria and software versions as well as elapsed time.

Advanced routes
---------------

Kernel selection depends on backend, grid and operator support. A faster
operator evaluation must agree with the reference path for the stated
precision before it supports a performance claim. A small-kernel timing does
not determine full-run speed, where compilation, field solves, FFTs,
diagnostics and I/O also contribute.

Independent scans and the species--Hermite nonlinear mesh have different
communication costs. Neither a successful placement nor an identity test
establishes speedup. See :doc:`parallelization` for configuration, mesh
constraints and serial comparison checks. Matrix-free eigensolvers reduce
storage requirements but still require residual and branch certification;
see :doc:`differentiable_eigensolver` and :doc:`solvers`.
