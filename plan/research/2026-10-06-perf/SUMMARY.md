# PERF lane 2026-10-06: step, window adjoint and imex-ars3 profile

Source pinned at `ba3d87d` (2.5.1), clean tree. One A4000 (GPU1) on the office
host, jax/jaxlib 0.10.2, float32, `XLA_PYTHON_CLIENT_PREALLOCATE=false`. The
GPU was shared with other jobs (both GPUs at 65-100% utilization from other
users, host load 37-40 on 36 threads), so absolute times scatter; every A/B
below alternates arms in one process (A,B,B,A,...) and reports the spread.
Drivers: `plan/research/2026-09-22-perf-lit/profile_perf_lit.py` (`forward`,
`window`; deck path updated to `examples/03_nonlinear_tokamak/case_full.toml`,
inner-remat toggle mapped to `adjoint_memory_budget_bytes`) and a `gkx.solve`
bench. Raw traces stay on the office host; only these numbers are kept.

## 1. Profile

Device-kernel time of one RK3 step (Cyclone, compressed real FFT, grid
Laguerre), from the xprof trace, by kernel kind:

| grid (Nl,Nm) | step ms | elementwise fusions | concatenate | slice fusions | FFT | transpose | cuBLAS scal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 32x32x24 (4,8) | 8.3 | 29% | 8% | 10% | 12% | 10% | 29% |
| 64x64x24 (4,8) | 23.6 | 16% | 30% | 12% | 12% | 10% | 16% |
| 64x64x24 (8,16) | 93.3 | 25% | 24% | 18% | 11% | 11% | 7% |

Peak device memory per step graph: 110 MB, 437 MB, 1.36 GB. The four largest
fusions at 64x64x24 (8,16) (`loop_slice_fusion`, two `input_concatenate_fusion`,
`loop_negate_real_fusion`; 33 ms of 88) each re-contain the same
~60-op Laguerre grid->moment contraction (`core_velocity._laguerre_contract_axis1`)
that XLA duplicates into every consumer, plus the Hermite ladder slices and
the linked-FFT scatter. The field solve stays at 1.4-4.7 ms. At small sizes
cuBLAS `scal` kernels (beta=0 initialization of the degenerate batched
matmuls) are a quarter of the step.

Window value+gradient (`nonlinear_heat_flux_window`, RK3, 256 steps, tprim
scale + nine geometry arrays):

| case | value s | v+g block (shipped 4 GiB budget) | v+g nested remat | v+g unchecked |
|---|---:|---:|---:|---:|
| 16x16x16 (4,8) | 0.167 | 0.997 (0.94-1.01), temp 270 MB | 1.143 (1.14-1.17), temp 61 MB | 0.577, temp 4.4 GB |
| 32x32x24 (4,8) | 1.010 | 4.35 (4.34-4.37), temp 1.6 GB | - | - |

Gradient-graph compile 12-13 s. The tprim gradient of the block schedule is
bitwise equal to the unchecked one; the geometry-gradient norm differs by
1.8e-7 relative (float32 reordering).

`gkx.solve` at 32x32x24 (4,8), kinetic-electron deck, 64 fixed steps, warm
(second call in the same process):

| method | warm wall s | backend recompiles per call | recompile s |
|---|---:|---:|---:|
| rk3 | 6.8 | 1 (`run_raw`) | 5.1-5.4 |
| imex-ars3 | 32.0 | 27 (factor probes `take`, `take_u`, `make_all`, ...) | 24.5-25.3 |

## 2. Window adjoint: already on main

Both requested changes are already in 2.5.x: the inner per-step remat is
dropped under `ADJOINT_MEMORY_BUDGET_BYTES` (#279, 1.15x faster than the
nested schedule here, 1.73x over unchecked instead of 1.98x), and the
saturated state is a jit operand of `_nonlinear_heat_flux_window_total`
(#264). Measured: a second saturated state triggers 0 backend compiles
(window v+g 3.13 s vs 3.08 s warm at 32x32x24, 64 steps). This PR pins that
with `test_window_adjoint_takes_the_saturated_state_as_an_operand`.

## 3. Hotspot experiment (negative)

An `optimization_barrier` on `_laguerre_to_spectral` (materialize the
contraction once instead of letting XLA duplicate it into four fusions),
interleaved in one process, 8 rounds x 5 calls, RK3 step:

| grid (Nl,Nm) | A ms (base) | B ms (barrier) | bits |
|---|---:|---:|---|
| 64x64x24 (8,16) | 150.3 | 151.1 | identical |
| 64x64x24 (4,8) | 33.2 | 33.6 | identical |
| 32x32x24 (4,8) | 8.2 | 9.2 | identical |

No win; not kept. Pallas was not tried: no single elementwise/concatenate
chain dominates (the top fusion is 17% of the step), and the field solve is
5% of the step, so neither qualifies under the lane's rule.

## Ranked next targets

1. Per-call recompilation in `gkx.solve`: 75-80% of a warm short run
   (imex-ars3 factor probes 24 s, rk3 `run_raw` 5 s). The probe closures in
   `solvers_nonlinear_imex._probe_columns` and `run_raw` capture the cache as
   constants; a module-level jit with arrays as operands (as #264 did for the
   window) removes it. This is what a VMEX objective that re-saturates per
   evaluation pays.
2. Linear-RHS assembly fusions (Hermite ladder slices + linked-FFT scatter
   concatenates), 40-45% of the step at 64x64x24.
3. cuBLAS `scal` at small grids: find the remaining batched dot_generals.
