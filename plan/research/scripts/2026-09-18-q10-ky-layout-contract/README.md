# Q10 stage 1 — the ky >= 0 layout contract (plan 5.3 N3)

Evidence for a refactor that must not move a number. Every gate is an A/B
between an unmodified `origin/main` tree and the branch tree, run in separate
processes from the same interpreter. `run_gates.sh` only sequences Q9's harness
scripts (`../2026-09-14-q9-batched-chain-fft/`), which are reused verbatim.

## Commands

    # export origin/main into REF_TREE, then, from a clean environment
    # (JAX_ENABLE_X64 and GKX_X64 unset -- exporting them contaminates the
    # "f32" arm through `env`, which inherits the caller's environment):
    PY=<venv python> bash run_gates.sh "$REF_TREE" "$NEW_TREE" "$OUT"

    # determinism control for the one f32 output that moved
    XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
      python ../2026-09-14-q9-batched-chain-fft/rhs_identity.py out.npz

    # the graph and input probes used to localize that difference
    python probe_argrhs.py argrhs.txt     # optimized HLO, cache/params as arguments
    python probe_leaves.py leaves.json    # dtype/shape/sha256 of every grid,
                                          # cache and params leaf

## Contents

- `ledgers/{ref,new}_{diagnostics,runtime}_{32,64}.json` — optimized-HLO op
  ledgers, `profile_runtime_kernels.py nonlinear-step-hlo`. `ledger_table.txt`
  is the ref -> new table: every count and every byte total is unchanged.
- `identity/cmp_rhs_identity_{f32,x64}.json` — every named RHS term, the total,
  the full nonlinear RHS and the VJPs, four cases.
- `identity/cmp_rhs_identity_f32_singlethread_fft.txt` — the same f32 A/B with
  the multithreaded CPU FFT off: 58/58 bitwise.
- `identity/cmp_gate_traj_{f32,x64}.json` — 100-step trajectories over the
  integrators, routes and checkpointed window: 65/65 bitwise in both.

## The one f32 difference, and why it is not this branch

With the default flags the f32 A/B reported `nl32/nonlinear_rhs` at
`||d||/||ref|| = 1.4e-10` (and `nl64/nonlinear_rhs` at 2.8e-10 in a later
pair). It is not a change:

- the optimized HLO of that graph is identical modulo metadata and SSA
  numbering, both with cache and params captured and with them passed as
  arguments (`probe_argrhs.py`);
- every grid, cache and params leaf has the same dtype, shape and SHA-256
  (`probe_leaves.py`);
- **two runs of unmodified `main`, minutes apart, differ on the same key by
  1.4e-10**, while two back-to-back runs of either tree agree bitwise;
- with `--xla_cpu_multi_thread_eigen=false` every pair is 58/58 bitwise,
  including main against the branch.

So the float32 nonlinear RHS at 32x32x24 and 64x64x24 is not bit-reproducible
run to run under the multithreaded XLA:CPU FFT thunk, on `main` as much as
here. Q9 measured the mechanism: the thunk splits a batch of lines across the
intra-op pool and DUCC's SIMD lanes round differently from its scalar
remainder. Any future f32 bitwise gate on these graphs has to pin that flag.
