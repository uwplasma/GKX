# PERF-LIT report (2026-09-22)

Lane PERF-LIT, `plan.md` section G.2. This file collects the lane's
deliverables: the measured profile and ranked implementation list (below, as
published in PR #284's description when the lane paused) and three literature
surveys in this folder:

- [Survey A — gyrokinetic codes, adjoints and preconditioners](SURVEY_A_codes_adjoints_preconditioners.md)
- [Survey B — sparse direct solvers and implicit adjoints from JAX](SURVEY_B_sparse_direct_and_implicit_adjoints.md)
- [Survey C — nonlinear-trajectory derivatives, precision and GPU mechanics](SURVEY_C_trajectory_derivatives_precision_gpu.md)

Raw records and scripts are in this folder; `summarize.py` rebuilds the tables.

---

Draft, docs/research only (plan.md G.2, lane PERF-LIT). Paused by the maintainer mid-run; the Handoff at the end says exactly what is left.

Adds `plan/research/2026-09-22-perf-lit/`: a one-process profiler (`profile_perf_lit.py`: `forward`, `fftfloor`, `window`, `eigen`, `eigen-adaptive`), run scripts, `summarize.py`, and raw JSON records. No `src/` change. The written report (REPORT.md) and the three survey appendices are not yet committed; their content is summarized below (see Handoff).

## Measured profile (one RTX A4000, complex64, JAX 0.10.2, baseline f9485f044)

Deck: shipped linked Cyclone nonlinear TOML (rk3, full two-sided ky, compressed real FFT), grid and Nl/Nm overridden, random state masked to the linked cover. Timings and memory are properties of the compiled graph, not a turbulence result. Compile separated from warm execution (median of 3-7 synchronized calls); memory from `memory_analysis()` and device `peak_bytes_in_use`; HLO op-name counts with `materialized_bytes` (the #259 correction); device-time split from a Perfetto trace of 5 RK3 steps.

**Forward**

| Grid, Nl/Nm | State MB | Field solve ms | Nonlinear RHS ms | RK3 step ms | Step temp MB | Materialized per step | GPU time: fft / concatenate / transpose / elementwise fusions / cuBLAS |
|---|---:|---:|---:|---:|---:|---:|---|
| 32x32x24, 4/8 | 6.3 | 1.17 | 2.31 | 5.26 | 40 | 130 MB (21x state) | 16 / 10 / 12 / 48 / 9 % |
| 32x32x24, 8/16 | 25.2 | 1.19 | 9.94 | 18.0 | 180 | 1.06 GB (42x) | 14 / 21 / 10 / 43 / 8 % |
| 64x64x24, 4/8 | 25.2 | 1.28 | 9.54 | 15.6 | 160 | 0.65 GB (26x) | 15 / 25 / 11 / 35 / 9 % |
| 64x64x24, 8/16 | 100.7 | 1.32 | 36.3 | 69.4 | 700 | 4.23 GB (42x) | 12 / 25 / 9 / 42 / 8 % |

On the GPU the step is memory-traffic bound (two-sided ky concatenates plus elementwise fusions), not FFT bound. The field solve has a flat ~1.2 ms latency floor across a 16x range of state size (half of one RHS at 32x32x24 Nl4/Nm8).

**Window derivative** (`gkx.nonlinear_heat_flux_window`, gradient w.r.t. tprim scale and nine geometry arrays; gds21/gds22 are refused under a trace on linked decks)

| Grid, Nl/Nm, steps | Value s | Value+grad, block ckpt s (x value) | Value+grad, no ckpt s | Temp MB block / none | Compile s (lower+compile) |
|---|---:|---:|---:|---:|---:|
| 16x16x16, 4/8, 256 | 0.148 | 0.941 (6.35x) | 0.435 (2.94x) | 61 / 4,425 | 6.4 + 13.6 |
| 16x16x16, 4/8, 1024 | 0.529 | 3.72 (7.03x) | OOM (16.5 GiB) | 103 / — | 7.0 + 13.3 |
| 32x32x24, 4/8, 256 | 0.969 | 5.21 (5.38x) | not run | 359 / — | 7.5 + 18.5 |

`checkpointed_explicit_scan` checkpoints each block and also each step inside the block, so the forward is recomputed twice: 2.16x the unchecked gradient at 256 steps. Residual per step ≈ 17 MB at 16x16x16 (4,425 MB / 256); storing one block's residuals instead would cost ≈ 0.3 GB at 256 steps, ≈ 0.55 GB at 1024 (derived), ≈ 3.3 GB at 32x32x24/1024 (derived).

Minutes per stage (derived, not measured end to end): at 32x32x24 Nl4/Nm8 a 1024-step window gradient ≈ 21 s warm + one ≈ 26 s compile, while re-saturating a tube for the example's fixed 8,000 steps ≈ 42 s per evaluation. Re-saturation dominates each objective evaluation.

**Linear eigen derivative.** Dense route (`solver_growth_rate_from_geometry`) value+grad: 0.033 s at n=144 (VMEX default), 0.99 s at n=1,536, 2.83 s at n=3,072 (382 MB temp). Adaptive matrix-free route at n=144 on a laptop at load ~100 (indicative only, not recorded): 4.0 s value, 34 s value+grad warm. Shift-invert `pr3-cm` at Q28 `d96` (n=3,072, committed Q28 records): 13,996 inner iterations, preconditioner apply 3.1-4.4 matvecs, 57k-79k matvec-equivalents vs 33,916 for `adaptive`; 75-81% of inner time is the preconditioner apply (derived). Today's office-CPU `adaptive` control reproduces 33,915 operator applications (30.5 s single-thread, 1.29 GB RSS).

## Bottlenecks, ranked by cost to the VMEX objective

B1 re-saturation per evaluation (derived, largest); B2 double forward recompute in the window adjoint (measured 2.16x); B3 layout traffic (concatenate 10-25%, elementwise 35-48%, 21-42x state per step); B4 latency floor (field solve ~1.2 ms; 16x16x16 launch-bound); B5 preconditioner apply in shift-invert (75-81%); B6 chaotic divergence past ~1024 steps (quality); B7 compile 20-26 s per gradient graph. FFTs are 12-16% of GPU time: minor.

## Literature, tied to the bottlenecks (primary sources; UNVERIFIED marked)

- Checkpointing (B2): Revolve optimum p = tN − C(s+t, t−1) (Griewank & Walther, ACM TOMS 26, 2000, doi:10.1145/347837.347846); GKX's two-level scheme is at t≈3 cost for t=2 memory because of the inner per-step remat. JAX names/offload policies (docs.jax.dev gradient-checkpointing, memory-spaces); Diffrax online checkpointing adds nothing for fixed N. Pitfalls: jax#27748, jax#23869.
- Layout (B3, B4): ky ≥ 0 storage as GX/stella/GS2 (plan §5.3 N3; #266 step 0.54-0.60x). Two-for-one FFT packing halves inverse FFTs but needs ky=0 Hermitian averaging (gyaradax, arXiv:2604.06085); ceiling here ~6-8% (derived). cuFFT is a fusion barrier; XLA plans one batched cuFFT per op on innermost axes (openxla fft_thunk.cc). CUDA-graph command buffers and scan unroll for launch-bound small grids (UNVERIFIED for GKX).
- Chaotic sensitivity (B6): finite windows are biased; iGENE (arXiv:2605.03086) reports window gradients at 15-50% of finite differences near N≈512 that still drove flux matching. Ensemble adjoints (Lea, Allen & Haine, Tellus 2000; Eyink et al., Nonlinearity 2004) converge slowly (power-law tails). Shadowing (LSS/NILSS/FD-NILSS/NILSAS; Wang 2014, Ni & Wang JCP 2017, Ni & Talnikar arXiv:1801.08674) costs ~(unstable exponents + 1) runs; 1,200-1,500 estimated for channel DNS (Blonigan et al. arXiv:1702.06809); no published Lyapunov spectrum for flux-tube gyrokinetics.
- Linear solves and eigen derivatives (B5): GENE/GKW matrix-free SLEPc; CGYRO precomputes dense per-point implicit matrices with fields folded in (Candy et al. JCP 2016); GX's unpublished `pk/imex_full` branch pairs z-Fourier streaming inversion with a MAGMA batched banded LU of the z-local (l,m) mirror+drift block — the closest analogue of `pr3-cm`, used as an ARK stage (code only, validation unknown). ADI as a kinetic preconditioner: ~100x over none (Gasteiger et al., JPP 2017, arXiv:1611.02114). yancc (arXiv:2607.20861) uses line smoothers in multigrid and rejects sparse LU on GPU. Eigenvalue gradients need only the left eigenvector, obtainable from the same shift-invert factor (conjugate-transpose solve): zero extra solves. SFINCS discrete adjoint reuses its preconditioner LU (Paul et al. JPP 2019); Acton et al. JPP 2024 is the only gyrokinetic adjoint with published cost.
- JAX routes to sparse direct: SOLVAX 0.25 has MUMPS/SuperLU factors (N/T/H solves) but eager only; `custom_linear_solve` traces its solves, so a host factor needs `pure_callback`/FFI with `trans="T"` for reverse mode. spineax/cuDSS: CUDA 13, no transpose solve (second factor per backward). `jax.experimental.sparse.linalg.spsolve`: deprecated cuSolverSp QR, refactors each call. klujax forces CPU and x64 at import; pardiso-mkl-jax is real-only.
- Precision: GKX already runs complex64 (gyaradax's 2.47x mixed-precision gain is taken). f32 dots on Ampere default to TF32; A4000 FP64 is 1/64 FP32 on GA10x (bandwidth-bound kernels lose ~2x, UNVERIFIED here).

## Ranked implementation list (top 8) with acceptance tests

1. **Window adjoint: drop the per-step remat inside checkpoint blocks under a memory budget** (GKX; B2). Expected 1.5-2x on the gradient (derived from the measured 2.16x). Accept: value and gradient equal to the current schedule (f32 ≤1e-6 rel, f64 ≤1e-12) at 256 and 1024 steps; `temp_size_in_bytes` under a declared budget with fallback to the current schedule; ≥15% warm speedup A/B/A/B on the A4000 at 16x16x16/1024 and 32x32x24/256.
2. **Warm-started re-saturation plus batched tubes** (GKX, F.5 steps 2-4; B1). Accept: saturation gate passes from the warm start; wall time per objective+gradient per tube recorded on CPU and A4000; vmapped tubes reuse one compiled window.
3. **ky ≥ 0 layout to default** (GKX, #266 split + ADJ-HALF + SHARD-PAD; B3). Accept: §5.3 N3 gates (identity ≤1e-13 f64, zero stage concatenates, VJP parity) and a window gradient no slower than the full layout on the A4000.
4. **Latency floor** (GKX; B4): identify and fuse the ~1.2 ms field-solve chain; try command buffers and `scan(unroll=2..4)` at 16x16x16. Accept: identical outputs, field solve < 0.5 ms at 32x32x24 or a written reason.
5. **Left-eigenvector gradient from one shift-invert factor as a traced SOLVAX primitive** (SOLVAX, then a GKX consumer): `pure_callback` + `custom_linear_solve` around the MUMPS factor (trans T/H), multi-RHS, refactor reusing analysis; dense stays the default at VMEX sizes. Accept: growth-rate gradient equal to dense to 1e-8 (x64) at n=3,072 and faster than the `adaptive` gradient at n ≥ 3,072.
6. **Device apply for `pr3-cm` / batched banded z-local blocks** (SOLVAX kernel, GKX operator; B5, reused later by §5.4 implicit streaming; GX-IMEX pattern). Accept: apply agrees with host block-Thomas < 1e-13 (x64); reproducible cost reduction at `d96` under the §5.1 gate with unchanged certification.
7. **Ensemble-averaged window gradients plus a Lyapunov/autocorrelation measurement** (GKX; B6, quality). Accept: leading exponent and heat-flux autocorrelation time recorded for the example tube; K-window median and spread; cosine vs FD/SPSA reported. No shadowing unless the unstable dimension is small.
8. **Precision audit** (GKX): `precision=HIGHEST` on invariant-carrying contractions, TF32 on/off energy check, complex64 vs complex128 window gradients at 256 steps; FFT packing only after item 3. Accept: energy-budget residual unchanged, gradient cosine > 0.99 inside the window.

Where things belong: checkpoint policy, layout, field-solve fusion, FFT packing, ensemble windows → GKX. Traced host sparse factor (N/T/H, multi-RHS, refactor), eigenvalue-gradient helper, batched banded/block-tridiagonal device kernels → SOLVAX (GKX supplies operator ordering and the field border).

## Handoff

**Goal and lane.** plan.md G.2 PERF-LIT: measured profile of GKX forward/derivative paths plus a literature and software survey, ranked. Baseline `f9485f044` (2.3.0), clean `src/`. Branch `research/perf-lit-20260922`.

**Done and verified.**
- GPU base phase (one A4000, complex64): forward rows at four sizes, isolated-FFT floors, window value/gradient at 16x16x16/256 (block and none), 16x16x16/1024 (block; none OOMs at 16.5 GiB), 32x32x24/256 (block), dense eigen value/gradient at n = 144, 1,536, 3,072. Records: `plan/research/2026-09-22-perf-lit/records/gpu_a4000/*.json`; print with `python plan/research/2026-09-22-perf-lit/summarize.py plan/research/2026-09-22-perf-lit/records/gpu_a4000`.
- Office CPU `adaptive` control at `d96` (single thread, x64): `records/cpu_pr3/A1_adaptive.txt`.
- Literature survey complete in three sections (gyrokinetic codes/adjoints/preconditioners, 37 refs; sparse direct and implicit adjoints from JAX incl. a read-only SOLVAX 0.25.0 inventory, 32 refs; trajectory derivatives/precision/GPU mechanics, 61 refs). Condensed above.

**Not done.**
- `REPORT.md` and the three survey appendices are not committed in the folder yet (the write was blocked in this session); the text above is the report's content. Commit it as `plan/research/2026-09-22-perf-lit/REPORT.md`.
- `PHASE=extra` of `run_profile.sh`: inner-remat-off window variants at 16x16x16/256, 16x16x16/1024, 32x32x24/256 (tests item 1 directly); complex128 window at 256 steps; complex128 forward; forward traces with kernel-to-source mapping (identifies the field-solve chain and the top elementwise fusions).
- The `pr3-cm` `d96` arm (stopped at the pause; its partial output was not kept).
- CPU forward/window rows on an idle host; the `eigen-adaptive` row on an idle host.

**Known failures / caveats.** The random state is not saturated turbulence: objective values are not physics. Office host load was 13-23 on 36 cores during GPU runs (device times are less sensitive than compile times). The laptop was at load 70-114, so no laptop timing is quoted as a record.

**Raw records.** In-repo: `plan/research/2026-09-22-perf-lit/records/`. Office host: GPU traces under `perflit_gpu/trace_*` in the home directory (not committed, large); a pinned detached worktree at `f9485f044` named `gkx-perf-lit` in the home directory, venv `venvs/gkx-nl` (JAX 0.10.2 CUDA 12, SOLVAX 0.22.0).

**Next steps, in order.**
1. On the office host, from the pinned worktree, check `nvidia-smi` and use only a free GPU: `CUDA_VISIBLE_DEVICES=<free> PHASE=extra bash plan/research/2026-09-22-perf-lit/run_profile.sh ~/perflit_gpu ~/venvs/gkx-nl/bin/python gpu` (copy the updated script from this branch first). Confirm the inner-remat-off rows give identical value/gradient to `win16_256.json`/`win16_1024.json`/`win32_256.json` and record the speedup and temp.
2. `bash plan/research/2026-09-22-perf-lit/run_pr3.sh ~/perflit_pr3 ~/venvs/gkx-nl/bin/python` (single-threaded, ~15 min).
3. CPU rows: `bash plan/research/2026-09-22-perf-lit/run_profile.sh ~/perflit_cpu ~/venvs/gkx-nl/bin/python cpu` on an idle host; `eigen-adaptive --nz 24 --nl 2 --nm 3 --precision 64`.
4. Commit the new records, `REPORT.md` and survey appendices; update the tables above.
5. Open implementation lanes from the ranked list, starting with item 1 (PERF-ADJ can absorb it) and item 2 (OPT-VMEX-NL).
