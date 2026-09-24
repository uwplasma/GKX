# 2026-09-22 ky-state-sharding: `[time] state_sharding` on CPU

Follow-up to SHARD-PAD (#287). Contents:

- `xla_cpu_fft_layout_repro.py`: a standalone JAX reproducer of the XLA:CPU FFT layout RET_CHECK.
  It fails on jax 0.10.2 and 0.11.1.
- `XLA_ISSUE_DRAFT.md`: upstream issue text (not filed) with the root cause in
  `cpu_layout_assignment.cc`.
- `state_sharding_identity.py`: serial vs `state_sharding` on a live deck, reading results back.
- `identity_after.jsonl`: CPU (fake devices), 10 runs with the fix: ky on the full and half
  layouts, 2 and 4 devices, plus kx on 4 devices, in float32 and x64. Every run matches serial
  bitwise (`max_abs_diff = 0`, `phi_max_abs_diff = 0`), with a state change of 4.7e-4 and
  |phi|max of 3.8e-3. Before the fix, the full-layout runs failed with the RET_CHECK and the
  half-layout runs raised `IndivisibleError`.
- `identity_gpu.jsonl`: 2 x RTX A4000, jax 0.10.2 CUDA 12. It covers three versions of the code:
  - The pre-#292 partitioner route: full ky and kx are bitwise equal to serial on GPU; half ky
    raises `IndivisibleError`. The RET_CHECK is CPU-only: XLA:GPU feeds the FFT through a bitcast
    from the default layout, and `xla_cpu_fft_layout_repro.py` runs correctly on GPU.
  - The first #292 draft, with one gather region per RHS call and per projector call: full ky is
    silently **wrong** (a max difference of 5e-4, about the size of the state change). The error
    needs the projector on and a ky split, and it sits on the conjugate rows 1, 2, 6 and 7, which
    cross devices. Each region was exact when run alone (the projector, and the RHS with and
    without the nonlinear term). This was not minimized further; it points at an XLA:GPU
    simplification across consecutive gather regions.
  - The current code, with one gather region per step: every case is bitwise equal to serial,
    including x64.
- `step_probe.py`: the one-step split-vs-unsplit probe used to localize that failure.
