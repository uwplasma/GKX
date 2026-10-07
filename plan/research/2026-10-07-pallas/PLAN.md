# Pallas GPU plan (2026-10-07, DRAFT, paused)

Source pinned at `7f5e151` (main). One A4000 (GPU1, office), jax/jaxlib 0.10.2, complex64. Both GPUs were 95-100% busy with other users' jobs throughout: absolute times are inflated and only trustworthy as ratios inside one row. Spread = min-max of 7 warm calls (forward) or 3 (window). Raw records stay on the office host (lane scratch `pg/`; the CPU pass `pc/` was killed at the pause).

## Measured GPU profile (device-kernel time by category, from xprof traces)

| case | warm | compile | peak dev | elementwise | concat | slice/pad | FFT | IFFT-norm `scal` | transpose/copy | memcpy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RK3 step 32x32x24 (4,8) adiabatic | 8.40 ms [8.26-10.69] | 3.7 s | 116 MB | 36.6% | 9.9% | 13.7% | 14.8% | 8.5% | 11.7% | 4.3% |
| RK3 step 64x64x24 (4,8) | 36.1 ms [35.9-38.5] | 3.9 s | 462 MB | 19.3% | 29.0% | 15.8% | 15.6% | 4.4% | 13.8% | 2.0% |
| RK3 step 64x64x24 (8,16) | 159.9 ms [159.5-161.9] | 4.0 s | 1.47 GB | 27.9% | 23.2% | 15.8% | 14.7% | 9.0% | 7.6% | 1.8% |
| KE rk3 16x16x16 (2,4), per step | 1.76 ms | 6.0 s/call recompile | 35 MB | 10.4% | 12.1% | 0.3% | 16.5% | 5.5% | 19.5% | 34.2% |
| KE rk3 32x32x24 (4,8), per step | 21.7 ms | 8.3 s/call recompile | 208 MB | 38.4% | 6.1% | 18.7% | 7.3% | 3.6% | 21.9% | 4.0% |
| KE imex-ars3 16x16x16 (2,4), per step | 5.7 ms | 15.7 s/call (27 recompiles) | 91 MB | 29.2% | 4.0% | 2.2% | 4.4% | 1.1% | 2.2% | 28.4% (+matmul 14.9%) |
| KE imex-ars3 32x32x24 (4,8), per step | 67.1 ms | 36.5 s/call (37 recompiles) | 1.70 GB | 69.7% (one `loop_add_fusion`, 60% alone) | 3.7% | 0.3% | 3.9% | 1.9% | 1.1% | 6.9% |
| window v+g 16^3, 256 steps | 1.59 s (value 0.30 s) | 11.2 s (+12.9 lower) | 273 MB | - | - | - | - | - | - | - |
| window v+g 32x32x24, 256 steps | 9.78 s (value 2.15 s) | 13.4 s (+17.3 lower) | 1.63 GB | - | - | - | - | - | - | - |
| linear eigen value, Nz 48 (4,8) | 1.39 s | 0.9 s | 153 MB | - | - | - | - | - | - | - |
| linear eigen value, Nz 96 (4,8) | 3.06 s | 1.0 s | 378 MB | - | - | - | - | - | - | - |

KE per-step times are (wall(120 steps) - wall(40 steps)) / 80 on warm calls; the per-call recompiles on main are what PR #347 removes. Window traces were not captured (trace dir not written by the window driver).

## Source review (summary; URLs in the PR body)

- Pallas on Ampere (sm_86) is Triton only; Mosaic GPU is Hopper+. Triton was deprecated in JAX 0.11.0 (best-effort maintenance). No complex dtypes in kernels: split real/imag f32 planes. No FFT inside kernels. Power-of-two blocks (pad + mask Nz=24, Nm, kx). `custom_vjp` needed per kernel for the adjoint; CPU needs the pure-JAX fallback (interpret mode only for tests).
- GX hand-writes ~170 CUDA kernels: fused linear RHS (`rhs_linear`, `streaming_rhs`), bracket, field solves, linked-chain copy kernels, `add_scaled` time-stepper fusion; it folds i kz, |kz| into cuFFT callbacks in `grad_parallel_linked.cu`, does the Laguerre transform as a real f32 strided-batched GEMM, reductions via cuTENSOR. The callback fusion is not available to XLA or Pallas.
- CGYRO/GENE-X: bottlenecks are FFT, transposes and bandwidth; remedies batching and fusion. arXiv:2606.29702 (Pallas stencil kernels for a gyrokinetic case beating XLA's gather-built velocity stencil) is the closest analogue: read before prototyping.

## Ranked candidates (bound = measured share of the step; gain <= bound)

1. **imex-ars3 stage `loop_add_fusion`** (60% of the imex step at 32x32x24): first identify its source (block-Thomas solve sweep / Woodbury apply in `solvers_nonlinear_imex._thomas_solve`). Likely a pure-JAX fix (matvec layout, `m`-major contiguous sweep) before Pallas; Pallas candidate = batched block-tridiagonal sweep with the inverses in shared memory, real/imag split. Bound 60% of the imex step. Effort M. CPU fallback = current code.
2. **Fused Hermite-Laguerre linear RHS (streaming + mirror + drift + hyper) in z-local blocks**: the concatenate + slice + elementwise share is 59-68% of the RK3 step at 64x64x24. A Pallas kernel computing the ladder couplings in registers and writing the RHS once could remove the duplicated producers XLA fuses into each consumer. Bound ~40% (data-movement part). Effort L (complex split, custom_vjp, power-of-two padding). Triton deprecation makes it a jax-0.10 dead end unless we move to Hopper or official Triton + jax_triton.
3. **Inverse-FFT normalisation `scal`** (4.4-9.0%): pure JAX, no Pallas: fold 1/N into the neighbouring multiply (`fft(conj(x))` form or `norm=` choice). Not bitwise; quantify. Effort S.
4. **Laguerre grid<->moment contraction**: already fused broadcast-MAC; GX uses an f32 GEMM. Part of the elementwise share; not isolated. Low priority.
5. **Bracket pointwise product in real space**: part of elementwise share, small; low priority.
6. **Small-grid latency (KE 16^3: memcpy 34%, ~36k kernels per 120 steps)**: launch-bound; command buffers / `scan(unroll)` before any kernel work.

Not yet prototyped. The rule was "prototype only if the bound is >= 20% of the step": candidates 1 and 2 qualify; 1 should be identified first because it may be a pure-JAX fix.
