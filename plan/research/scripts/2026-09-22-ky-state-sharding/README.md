# 2026-09-22 ky-state-sharding: `[time] state_sharding` on CPU

Follow-up to SHARD-PAD (#287). Contents:

- `xla_cpu_fft_layout_repro.py`: a standalone JAX reproducer of the XLA:CPU FFT layout RET_CHECK.
  It fails on jax 0.10.2 and 0.11.1.
- `XLA_ISSUE_DRAFT.md`: upstream issue text (not filed) with the root cause in
  `cpu_layout_assignment.cc`.
- `state_sharding_identity.py`: serial vs `state_sharding` on a live deck, reading results back.
- `identity_after.jsonl`: 10 runs with the fix: ky on the full and half layouts, 2 and 4 devices,
  plus kx on 4 devices, in float32 and x64. Every run matches serial bitwise
  (`max_abs_diff = 0`, `phi_max_abs_diff = 0`), with a state change of 4.7e-4 and |phi|max of 3.8e-3.
  Before the fix, the full-layout runs failed with the RET_CHECK and the half-layout runs raised
  `IndivisibleError`.
