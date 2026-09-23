## Survey C: nonlinear-trajectory derivatives, precision and GPU mechanics

Scope: what the literature says about (1) checkpointing schedules, (2) derivatives of time-averaged quantities in chaotic systems, (3) precision, (4) JAX/XLA mechanics for a scan-over-time adjoint, (5) cuFFT through XLA. It is read against GKX's setup: a window of N = 256..8192 RK steps, O(sqrt N) block checkpointing (7.8 GB to 148 MB at 1024 steps, 1.77-1.92x runtime), gradients that diverge beyond about 1024 steps, and a 16 GB RTX A4000 as the target. Every claim is taken from the cited primary source unless it is marked **UNVERIFIED** (not checked against a primary source) or **INFERENCE** (my own reasoning or arithmetic).

### Bottom line

- **Checkpointing is no longer the bottleneck.** The sqrt(N) scheme already sits at Griewank–Walther's optimum for "about 2 forward sweeps", and the measured 1.77–1.92x runtime matches that. Revolve would save memory only if GKX accepts about 2.8–2.9 forward sweeps. Chaos is what limits the window, not memory.
- **Plain window adjoints of chaotic turbulence are biased, and the bias does not shrink as the window grows.** The divergence past about 1024 steps is expected. The only other differentiable gyrokinetic study (iGENE [20]) sees divergence at N ≳ 512. Within the window, its gradients are 15–50% of the finite-difference values, yet they still drove flux-matching optimisations to convergence.
- **Shadowing methods cost about (number of unstable Lyapunov exponents) × (forward cost).** Wall-bounded DNS has an estimated n+ of about 1.2–1.5×10³ [15]. Nobody has measured n+ for gyrokinetic (GK) turbulence. These methods are unproven for GK and probably unaffordable for "minutes per stage".
- **On precision:** the gyaradax f32-FFT/f64-rest split gave 2.47x over all-f64. GKX already defaults to complex64, so that gain is already taken. On Ampere the risk is TF32 in f32 contractions (JAX's DEFAULT precision). A second risk is complex128 on a GA10x-class card, where FP64 hardware runs at 1/64 of the FP32 rate.
- **Inside XLA, FFTs are opaque cuFFT calls and act as fusion barriers.** They run on innermost axes only, so any other axis needs a transpose copy. The only way past the barrier is custom cuFFT LTO callbacks, which have no autodiff unless GKX writes its own VJPs.

---

### 1. Checkpointing schedules

**Binomial / Revolve** (Griewank & Walther, ACM TOMS 26(1):19–45, 2000 [1]; Griewank 1992 [2]). Proposition 2.1 as restated in [3]: with m steps and s checkpoints, the minimum number of forward-step recomputations is

  p(m, s) = t·m − C(s+t, t−1), where t (the "repetition number") is the unique integer with C(s+t−1, t−1) < m ≤ C(s+t, t).

Check: m = 10, s = 3 gives t = 2 and p = 15 [3]. With unlimited memory, p = m − 1, so p counts the initial sweep as well. Memory and recompute factor both grow like log m [1].

**INFERENCE (arithmetic), for GKX:**

| N | s | t | p/N |
|---|---|---|---|
| 1024 | 64 (≈ memory of the 32×32 sqrt scheme) | 2 | 1.94 |
| 1024 | 44 | 2 | 1.96 |
| 1024 | 17 | 3 | 2.81 |
| 1024 | 10 | 5 | 3.67 |
| 8192 | 181 (≈ 2·sqrt N) | 2 | 1.98 |
| 8192 | 127 | 2 | 1.98 |
| 8192 | 35 | 3 | 2.91 |
| 8192 | 20 | 4 | 3.75 |

So the sqrt scheme is already at the t = 2 optimum. Revolve's win is either about 30% fewer states at the same cost, or about 4–5x fewer states for about 1.45x more forward work.

**Multistage and hierarchical storage.**
- Stumm & Walther [4] add disk to RAM and show that putting some checkpoints on disk cuts recomputation.
- Aupy, Herrmann, Hovland & Robert [5] give the optimal two-level (memory + disk) algorithm with read/write costs.
- H-Revolve (Herrmann & Pallez [6]) is optimal for several storage levels, with a Python library.
- Online checkpointing, where N is unknown in advance: Stumm & Walther 2010 [7] and Wang, Moin & Iaccarino 2009 [8].
- Zhang & Constantinescu [3] also checkpoint the RK stage values (CAMS). They report "up to two times the speedup" over classical Revolve for multistage schemes, and the method is used in PETSc TSAdjoint. This is relevant because each GKX step is an RK step.
- Dedalus's automated adjoint (Skene & Burns [9]) uses the `checkpoint_schedules` library. One test uses H-Revolve with 400 checkpoints in RAM and 50 on disk.

**JAX.** The `jax.checkpoint` docs [10] list these policies:
- `dots_with_no_batch_dims_saveable`
- `save_only_these_names` / `save_any_names_but_these`, used with `checkpoint_name`
- `offload_dot_with_no_batch_dims`
- `save_and_offload_only_these_names`, which keeps some names on device, offloads others to `pinned_host`, and recomputes the rest

The docs show that recursive (nested) checkpointing over D functions gives O(log₂ D) memory for O(log₂ D) times the FLOPs. They also advise putting `jax.checkpoint` on the body passed to `lax.scan`, because XLA's own rematerialisation across control flow is "not as thorough".

The memory-spaces doc [11] shows offloading cutting temporary memory from 17.25 MB to 6.50 MB in a scanned MLP. It notes the pattern "depends on `jax.lax.scan()`".

Known failures:
- JAX #23869 [12] (0.4.33, H100): offloading a scan-carried residual had no effect, "0.0 GB host". The issue is closed, with a cross-reference to XLA #17541.
- JAX #27748 [13]: remat around scan worked in 0.4.13 and ran out of memory in 0.5.3. The XLA log says it "Can't reduce memory use below 27.95GiB".

**INFERENCE for GKX:** offloading is worth trying only for grids where the state times about 2·sqrt(N) does not fit in 16 GB. The A4000 uses PCIe 4.0 x16, so each offloaded state costs about its size divided by 25 GB/s (**UNVERIFIED** effective bandwidth).

**Diffrax / Equinox.**
- `RecursiveCheckpointAdjoint` [14] uses *online* checkpointing (Stumm–Walther 2010; Wang et al. 2009). By default `checkpoints = log(max_steps)`. The docs state backprop takes "O(n log n) time in the number of steps", with memory about checkpoints × size(y0). It cannot be forward-mode differentiated.
- It is built on `equinox.internal.checkpointed_while_loop` [14b]. That docstring says it matches offline treeverse when N ≤ (s+1)(s+2)/2, and otherwise "may make extra steps". It also implements only Algorithm I of Stumm–Walther.
- **INFERENCE:** with N fixed and known, GKX gains nothing from online schemes. A static two-level scan (which GKX already has) or a static Revolve schedule is at least as good, and XLA can see its shapes at compile time.

### 2. Chaotic sensitivity

**Why window adjoints diverge.**
- Lea, Allen & Haine (Tellus 52A:523–532, 2000 [16]) showed that naive adjoint sensitivity of long-time averages is "of limited utility". In Lorenz-63, averaging adjoint gradients over an ensemble of intermediate-length windows gives an estimate accurate to about O(10%).
- Eyink, Haine & Lea (Nonlinearity 17:1867, 2004 [17]) showed that this ensemble adjoint computes Ruelle's linear-response formula. However, "because of a power-law tail in the histogram of adjoint gradients", the ensemble sum is a Lévy flight, the CLT fails, and convergence is "very slow".
- Chandramoorthy et al. (AIAA J 57:4514, 2019; arXiv:1811.08567 [18]) estimated upper bounds on the convergence rate for turbulent flows. They found ensemble sensitivity "computationally intractable in each of the numerical examples considered", even at the optimistic bound.
- Metz et al. [19] tie the same failure in differentiable simulators to the Jacobian spectrum.

**Plasma evidence.** This is the closest analogue to GKX.
- iGENE (Artigues, Merlo & Jenko, arXiv:2605.03086 [20]) is a TensorFlow flux-tube GK code. On Cyclone Base Case (CBC) ion-temperature-gradient (ITG) turbulence, its window gradients of Q_i approach the finite-difference value and then "diverge for N ≳ 512". This matches the heat-flux autocorrelation time of 500–1000 steps.
- At N ≈ 512, the gradients with respect to ω_T, ω_n and q reach "15%–34%" of the finite-difference values, and ε reaches about 50%. Adam-based flux matching still "converge[s] stably" in single-point, 7-point and 6-point kinetic-electron tests, confirmed by independent validation runs.
- They are capped at about 16,000 unrolled steps per GPU because of memory. They list ensemble adjoints as an open question.
- Kim et al. (JPP 2024, arXiv:2310.18842 [21]) optimised stellarators with nonlinear GX + DESC using SPSA instead of gradients, because the flux traces are noisy.
- **INFERENCE:** GKX's cutoff near 1024 steps is consistent with iGENE's. GKX should expect gradients that are directionally useful but biased low in magnitude, not accurate ones.

**Shadowing family.** These methods target the derivative of the infinite-time average.
- LSS: Wang, Hu & Blonigan (JCP 267, 2014, doi:10.1016/j.jcp.2014.03.002 [22]), with a convergence proof in Wang (SINUM 52:156–170, 2014 [23]).
- NILSS: Ni & Wang (JCP 347:56–77, 2017 [24]) restricts the problem to the unstable subspace, so cost scales with the number of unstable exponents n+. It was shown on a 12,000-cell backward-facing step.
- FD-NILSS: Ni, Wang, Fernandez & Talnikar (JCP 394:615–631, 2019 [25]). For 3-D cylinder flow at Re = 525 (3.7×10⁵ cells), it measured about 17 unstable covariant Lyapunov vectors (CLVs) and used M = 40 homogeneous tangents. That means 42 primal runs of 400 segments × 200 steps, costing about as much as the primal simulations used for the comparison.
- NILSAS: Ni & Talnikar (arXiv:1801.08674 [26]) is the adjoint version, with cost independent of the number of parameters and requiring M ≥ n+ + 1 adjoint solutions. It tabulates n+:
  - 13 of 4×10⁴ for a 2-D step
  - fewer than 5 for a NACA 0012 airfoil
  - about 20 of 1.9×10⁶ for the 3-D cylinder
  - about 1.5×10³ of 2.2×10⁶ for Re_τ = 180 channel flow
- Blonigan et al. (CTR 2016, arXiv:1702.06809 [15]) extrapolate n+ ≈ 1200–1500 for that channel. They conclude wall-bounded flows need "at least O(1000) simulations" with NILSS, and that n+ scales with domain size for periodic boxes.
- Multiple-shooting shadowing (Blonigan & Wang [27]) avoids computing n+ up front but still scales with the unstable dimension.
- **UNVERIFIED / unknown:** no published Lyapunov spectrum for flux-tube GK turbulence was found. A flux tube is a periodic box in x and y, so by [15]'s domain-size argument n+ may well be O(10²–10³). **Measuring the leading Lyapunov exponents of a GKX run is a precondition before investing in any shadowing method.**

**Linear response / S3 / fast response.**
- S3 (Chandramoorthy & Wang [28]; SIADS 2022 [29]) splits Ruelle's formula into well-conditioned ergodic averages. The published validations are low-dimensional (one-dimensional unstable manifolds).
- Ni's "fast adjoint response" (SIADS 22:2792, 2023; arXiv:2111.07692 [30]) runs with cost independent of the number of parameters, but it is shown for hyperbolic systems of modest dimension.
- No application of either to high-dimensional turbulence was found.

**Other routes.**
- Garai & Murman (AIAA J 59(6), 2021 [31]) stabilise the adjoint using its energy budget. They call it "computationally cheap" and say the result is approximate; only the abstract was checked.
- Wang & Zaki (arXiv:2606.25399, June 2026 [32]) close the *ensemble-averaged* adjoint with an eddy viscosity for wall turbulence.
- OGF (Hickling et al., arXiv:2507.05149 [33]) propagates gradients online with finite-difference estimates and was demonstrated on forced compressible homogeneous isotropic turbulence (HIT) DNS.
- Schnell & Thuerey (ICLR 2024 [34]) modify backpropagation through time (BPTT) to tame exploding gradients while keeping the minima unchanged.
- Huhn & Magri (JFM 882:A24, 2020 [35]) used covariant Lyapunov analysis to optimise a low-dimensional thermoacoustic system.

**Honest status.** No work reports accurate shadowing-quality gradients for high-Reynolds 3-D DNS or for plasma turbulence at a cost near one forward run. The only DNS-scale shadowing success is the weakly turbulent cylinder, with n+ ≈ 17–20. The plasma success, iGENE, relies on biased window gradients being good enough for descent.

**INFERENCE for GKX:** a cheap, defensible upgrade is Lea–Allen–Haine averaging. Run K windows of length about one autocorrelation time from decorrelated starts and average the gradients. At 16³ the GPU is under-used, so vmap over K windows may be nearly free (**INFERENCE**, must be measured). The price is heavy-tailed variance [17]. Report a median or trimmed mean with a spread, not a single number.

### 3. Precision

- **gyaradax** (Galletti, Volkmann & Brandstetter, arXiv:2604.06085 [36]) runs the 2-D FFTs, derivatives and inverse FFTs of the Poisson bracket in Float32, and keeps the linear terms, field solver and final forward FFT in Float64. On a B300, JAX went from 12.49 to 30.89 steps/s, a **2.47x** gain, with memory unchanged at 9.4 GB. Custom cuFFT LTO callbacks lift this to 60.54 steps/s.
  - They found a Hermitian-symmetry defect in the gyroaveraged φ at k_x = 0 (up to 14% relative error). R2C inverse transforms hide it, and C2C packing exposed it. Explicit Hermitian averaging fixed it.
  - **INFERENCE:** GKX's default is already complex64 throughout, so this gain is already banked. The lesson that transfers is to enforce Hermitian symmetry explicitly whenever the real-FFT path is bypassed.
- **Single-precision DNS** (Karp et al., arXiv:2506.05150 [37]): across four solvers, including channel flow at Re_τ = 550 and a cylinder at Re_D = 3900, IEEE fp32 for the whole simulation showed "no significant discrepancies" from fp64 in the statistics. Averaging uncertainty dominated. The study is forward-only, with no adjoints.
- **Adjoints at reduced precision.**
  - Hatfield et al. (MWR 148:1541, 2020 [38]) ran the 4D-Var tangent-linear and adjoint in single precision without loss of quality.
  - McRae & Palmer (arXiv:2003.08972 [39]) got reasonable MITgcm adjoint results with as few as 10 significand bits.
  - Klöwer et al. [40]: raw Float16/BFloat16 degrades a chaotic shallow-water model, but works with rescaling and compensated summation.
  - **INFERENCE:** for a chaotic window adjoint, rounding error is amplified by the same Lyapunov growth that dominates beyond about one autocorrelation time. Within the usable window, fp32 is very likely not the limiting error, and complex128 buys little. Test this by comparing c64 and c128 gradients at N = 256 and 512.
- **Loss scaling** is a gradient-underflow fix for fp16/bf16 training. It does not apply to an fp32/fp64 adjoint.
- **TF32.** `jax.lax.Precision.DEFAULT` on GPU "uses tensorfloat32 if available (e.g. on A100 and H100 GPUs)". HIGHEST "uses float32" [41]. `jax_default_matmul_precision` accepts 'highest', 'float32', 'F32_F32_F32', 'TF32_TF32_F32_X3' and others [42]. TF32 keeps 10 mantissa bits [43], so even an identity matmul can change values.
  - The A4000 has 192 third-generation Tensor Cores [44], so TF32 applies to it.
  - **UNVERIFIED:** whether XLA routes *complex64* dot/einsum through TF32 cuBLAS paths. Hermite/Laguerre contractions and field solves should set `precision=HIGHEST`, or the global flag, and check energy and free-energy conservation with it on and off.
  - FFTs are not matmuls and are unaffected.
- **FP64 on the A4000.** The GA102 whitepaper gives two FP64 units per SM and an FP64 rate of 1/64 of FP32 [45]. **INFERENCE:** the A4000 (GA104) shares the GA10x SM design. FFT-heavy code is bandwidth-bound, so complex128 will cost somewhere between the 2x bytes ratio and much worse on the arithmetic-bound kernels. Measure it; do not assume 64x.

### 4. JAX/XLA mechanics for a scan-over-time adjoint

- **Donation** [46] reuses an input buffer for an output with the same shape and dtype, and warns "Some donated buffers were not usable". It works only at the jit boundary. Scan carries are already updated in place by XLA. **INFERENCE:** donate the state and optimiser buffers on the outer jitted `value_and_grad` call. Donation does nothing inside the scan.
- **`lax.scan(unroll=k)`** [47] puts k iterations inside each loop body. `unroll=True` or `0` unrolls fully. **INFERENCE:** unrolling helps small-grid, launch-bound bodies by allowing cross-step fusion. It multiplies compile time and conflicts with the checkpoint boundary, so sweep k from 1 to 4 on the inner scan only.
- **Memory reporting.** XLA's `CompiledMemoryStats` [48] documents on-device need as at least code + argument + output − alias + temp. It also exposes `peak_memory_in_bytes`, host_* fields and a serialised buffer assignment. Use `jitted.lower(...).compile().memory_analysis()`.
  - These are static buffer-assignment numbers. **INFERENCE (from XLA source [49]):** cuFFT plans get their scratch at run time through a scratch allocator, so it is not counted in `temp_size_in_bytes`.
  - The BFC allocator preallocates 75% of GPU memory by default (`XLA_PYTHON_CLIENT_PREALLOCATE`, `_MEM_FRACTION`) [50], so `nvidia-smi` is meaningless for peak memory.
  - `jax.profiler.save_device_memory_profile` shows live buffers at one moment, not the peak, and treats jit functions as opaque [51].
- **Traces.** Use `jax.profiler.trace(..., create_perfetto_link=True)`, or `start_trace`/`stop_trace` with XProf/TensorBoard. GPU tracing needs CUPTI [52].
- **HLO dumps.** `XLA_FLAGS=--xla_dump_to=DIR` dumps the HLO before and after optimisation as text. `--xla_dump_hlo_pass_re=.*` dumps after every pass [53]. Use this to count FFT custom-calls and transposes per step, and to confirm remat placement.
- **Compilation cache.** Set `jax_compilation_cache_dir`. Only entries that took longer than `jax_persistent_cache_min_compile_time_secs` (default 1.0 s) to compile are stored. `jax_persistent_cache_enable_xla_caches` can also persist the per-fusion autotune cache [54]. This matters for "minutes per stage" if each VMEX stage recompiles.
- **Known blowups.** JAX #27748 (remat around scan regressed between 0.4.13 and 0.5.3) [13], #23869 (offloading a scan carry ignored) [12], #26639 (memory growing linearly with outer loops over a stacked-output scan; open) [55]. **Practical:** pin the jax/jaxlib version in the benchmark, and assert `memory_analysis()` numbers in CI.
- **GPU flags** [56]: the latency-hiding scheduler and `--xla_gpu_enable_command_buffer` (CUDA graphs) exist. **INFERENCE:** command buffers target launch overhead, which is the likely bottleneck at 16³ with about 23 FFTs × RK stages per step. This is unmeasured.

### 5. GPU FFT specifics

- **XLA FFT semantics** [57]: FFT, IFFT, RFFT and IRFFT on "up to 3 axes", always the *innermost* ones. RFFT shrinks the last axis to `fft_length[-1]//2+1`, and multi-dimensional real transforms do the real transform on the innermost axis first. On GPU, "GPU FFT uses cuFFT".
- **jnp.fft** [58] calls `jnp.moveaxis` to bring non-trailing axes to the end ("XLA only supports FFTs over the innermost axes"). **INFERENCE:** that becomes a physical transpose copy unless layout assignment absorbs it. Check in the HLO dump. GKX should store arrays as (…, ky, kx) with x and y last.
- **XLA's GPU FFT thunk** [49] builds one cuFFT batched plan per thunk, lazily and cached per device. The batch is the product of all leading dimensions, the stride is 1 and the transform is out-of-place. **INFERENCE:** so species, Hermite, Laguerre and z should be leading batch dimensions of a single call, not a Python loop or vmap-split calls. JAX #9591 reports a crash for batched plans larger than 4 GB [59].
- **cuFFT** [60]:
  - R2C output has ⌊N/2⌋+1 complex values along the last dimension.
  - It favours contiguous, batched transforms.
  - Sizes 2^a 3^b 5^c 7^d are optimised, with powers of 2 fastest.
  - Out-of-place C2R "will always overwrite input buffer".
  - Each plan gets its own workspace.
  - LTO callbacks (`cufftXtSetJITCallback`) need nvJitLink/NVRTC. Legacy callbacks need static linking on Linux.
- **Fusion barrier.** gyaradax [36] states that cuFFT is "a hard fusion barrier" in XLA. They recovered about 2x with LTO load/store callbacks. One of those callbacks skips writing the 59% of modes that dealiasing discards. They also used two-for-one C2C packing of the real fields (Z = A + iB), cutting the inverse FFTs from 4 to 2 per bracket.
  - **INFERENCE:** packing is expressible in pure JAX and is differentiable, so it is the portable part of their win. It could cut GKX's roughly 23 FFTs per RHS substantially. The Hermitian-symmetry caveat above applies.
  - Callbacks require a custom call plus a hand-written VJP.
- **2/3 dealiasing cost (INFERENCE, arithmetic):** padding N to 3N/2 in both x and y gives 2.25x the points, and slightly more than 2.25x FFT work from the log factor. Choose the padded sizes to be 2^a 3^b.
- **VkFFT** (Tolmachev, IEEE Access 11:12039, 2023 [61]): an MIT-licensed, cross-backend FFT library claiming "comparable or better performance" than vendor libraries. **UNVERIFIED:** no maintained JAX custom-call binding was found. Using it would mean writing FFI glue plus VJPs.
- **A search-result caution:** a search summary claimed that GANDALF (a JAX KRMHD code, arXiv:2511.21891) validated AD gradients of time-averaged spectra to 2%. The v1 paper says AD is "unused in current GANDALF implementation". Do not cite it for AD.

### Implications for GKX (ranked)

1. Measure the leading Lyapunov exponent(s) and the Q autocorrelation time. Set N to about one autocorrelation time, and report gradient magnitude as biased (iGENE: 15–50%).
2. Average K decorrelated windows (Lea–Allen–Haine) via vmap and report a robust spread. Do not pursue NILSS/NILSAS until n+ is known.
3. Keep the sqrt(N) checkpointing. Add a static t = 3 Revolve or pinned-host offload only for grids that do not fit in 16 GB.
4. Set `precision=HIGHEST` on contractions. Validate c64 against c128 gradients inside the usable window. Keep c64 as the default on the A4000.
5. Audit the HLO for FFT count, transposes and batch layout. Try two-for-one packing, `unroll` 2–4 and command buffers. Enable the persistent cache with its XLA caches.

### References

1. Griewank, A. & Walther, A. (2000). Algorithm 799: revolve. ACM TOMS 26(1):19–45. https://doi.org/10.1145/347837.347846
2. Griewank, A. (1992). Achieving logarithmic growth of temporal and spatial complexity in reverse AD. ANL P228. https://ftp.mcs.anl.gov/pub/tech_reports/reports/P228.pdf
3. Zhang, H. & Constantinescu, E. Optimal checkpointing for adjoint multistage time-stepping schemes. arXiv:2106.13879. https://arxiv.org/abs/2106.13879
4. Stumm, P. & Walther, A. (2009). MultiStage approaches for optimal offline checkpointing. SISC 31(3):1946–1967. https://doi.org/10.1137/080718036
5. Aupy, G., Herrmann, J., Hovland, P. & Robert, Y. (2016). Optimal multistage algorithm for adjoint computation. SISC 38(3):C232–C255. https://doi.org/10.1137/15M1019222
6. Herrmann, J. & Pallez, G. (2020). H-Revolve. ACM TOMS. https://doi.org/10.1145/3378672 (https://inria.hal.science/hal-02080706/)
7. Stumm, P. & Walther, A. (2010). New algorithms for optimal online checkpointing. SISC 32(2):836–854. https://doi.org/10.1137/080742439
8. Wang, Q., Moin, P. & Iaccarino, G. (2009). Minimal repetition dynamic checkpointing. SISC 31(4):2549–2567. https://doi.org/10.1137/080727890
9. Skene, C. S. & Burns, K. J. Fast automated adjoints for spectral PDE solvers. arXiv:2506.14792. https://arxiv.org/abs/2506.14792
10. JAX docs, Gradient checkpointing. https://docs.jax.dev/en/latest/gradient-checkpointing.html
11. JAX docs, Memory spaces / host offloading. https://docs.jax.dev/en/latest/201/memory-spaces.html
12. JAX issue #23869. https://github.com/jax-ml/jax/issues/23869
13. JAX issue #27748. https://github.com/jax-ml/jax/issues/27748
14. Diffrax adjoints docs. https://docs.kidger.site/diffrax/api/adjoints/ ; [14b] Equinox `checkpointed.py`: https://github.com/patrick-kidger/equinox/blob/main/equinox/internal/_loop/checkpointed.py
15. Blonigan, P. J. et al. (2016). Toward a chaotic adjoint for LES. CTR Summer Program. arXiv:1702.06809. https://arxiv.org/abs/1702.06809
16. Lea, D. J., Allen, M. R. & Haine, T. W. N. (2000). Sensitivity analysis of the climate of a chaotic system. Tellus A 52:523–532. https://doi.org/10.1034/j.1600-0870.2000.01137.x
17. Eyink, G. L., Haine, T. W. N. & Lea, D. J. (2004). Ruelle's linear response formula, ensemble adjoint schemes and Lévy flights. Nonlinearity 17:1867. https://doi.org/10.1088/0951-7715/17/5/016
18. Chandramoorthy, N., Fernandez, P., Talnikar, C. & Wang, Q. (2019). Feasibility analysis of ensemble sensitivity computation in turbulent flows. AIAA J 57(10):4514–4526. https://arxiv.org/abs/1811.08567
19. Metz, L., Freeman, C. D., Schoenholz, S. S. & Kachman, T. (2021). Gradients are not all you need. arXiv:2111.05803. https://arxiv.org/abs/2111.05803
20. Artigues, V., Merlo, G. & Jenko, F. (2026). iGENE: a differentiable flux-tube gyrokinetic code in TensorFlow. arXiv:2605.03086 (also listed as Phys. Plasmas 33:083901). https://arxiv.org/abs/2605.03086
21. Kim, P. et al. (2024). Optimization of nonlinear turbulence in stellarators. JPP 90. arXiv:2310.18842. https://arxiv.org/abs/2310.18842
22. Wang, Q., Hu, R. & Blonigan, P. (2014). Least squares shadowing sensitivity analysis of chaotic limit cycle oscillations. JCP 267. https://doi.org/10.1016/j.jcp.2014.03.002
23. Wang, Q. (2014). Convergence of the least squares shadowing method. SINUM 52:156–170. https://arxiv.org/abs/1304.3635
24. Ni, A. & Wang, Q. (2017). NILSS. JCP 347:56–77. https://arxiv.org/abs/1611.00880
25. Ni, A., Wang, Q., Fernandez, P. & Talnikar, C. (2019). FD-NILSS. JCP 394:615–631. https://arxiv.org/abs/1711.06633
26. Ni, A. & Talnikar, C. NILSAS. arXiv:1801.08674 (JCP). https://arxiv.org/abs/1801.08674
27. Blonigan, P. J. & Wang, Q. Multiple shooting shadowing. arXiv:1704.02047. https://arxiv.org/abs/1704.02047 ; preconditioned MSS: https://arxiv.org/abs/1810.12222
28. Chandramoorthy, N. & Wang, Q. (2019). Sensitivity computation of statistically stationary quantities in turbulent flows (S3). arXiv:1905.09362. https://arxiv.org/abs/1905.09362
29. Chandramoorthy, N. & Wang, Q. (2022). Efficient computation of linear response of chaotic attractors with one-dimensional unstable manifolds. SIADS. https://doi.org/10.1137/21M1405599
30. Ni, A. (2023). Fast adjoint algorithm for linear responses of hyperbolic chaos. SIADS 22:2792. arXiv:2111.07692. https://arxiv.org/abs/2111.07692
31. Garai, A. & Murman, S. M. (2021). Stabilization of the adjoint for turbulent flows. AIAA J 59(6). https://doi.org/10.2514/1.J059998
32. Wang, Q. & Zaki, T. A. (2026). Mitigating adjoint chaos in wall turbulence. arXiv:2606.25399. https://arxiv.org/abs/2606.25399
33. Hickling, T., MacArt, J. F., Sirignano, J. & Waidmann, D. OGF. arXiv:2507.05149. https://arxiv.org/abs/2507.05149
34. Schnell, P. & Thuerey, N. (2024). Stabilizing backpropagation through time to learn complex physics. ICLR. https://arxiv.org/abs/2405.02041
35. Huhn, F. & Magri, L. (2020). Stability, sensitivity and optimisation of chaotic acoustic oscillations. JFM 882:A24. https://arxiv.org/abs/1909.12979
36. Galletti, G., Volkmann, E. & Brandstetter, J. (2026). gyaradax: local gyrokinetics JAX code. arXiv:2604.06085. https://arxiv.org/abs/2604.06085
37. Karp, M. et al. Effects of lower floating-point precision on scale-resolving numerical simulations of turbulence. arXiv:2506.05150. https://arxiv.org/abs/2506.05150
38. Hatfield, S., McRae, A., Palmer, T. & Düben, P. (2020). Single-precision in the tangent-linear and adjoint models of incremental 4D-Var. MWR 148:1541–1552. https://doi.org/10.1175/MWR-D-19-0291.1
39. McRae, A. T. T. & Palmer, T. N. Using reduced-precision arithmetic in the adjoint model of MITgcm. arXiv:2003.08972. https://arxiv.org/abs/2003.08972
40. Klöwer, M., Düben, P. D. & Palmer, T. N. (2020). Number formats, error mitigation, and scope for 16-bit arithmetics… JAMES. https://doi.org/10.1029/2020MS002246
41. JAX docs, `jax.lax.Precision`. https://docs.jax.dev/en/latest/jax.lax.html
42. JAX docs, configuration options. https://docs.jax.dev/en/latest/config_options.html
43. PyTorch docs, numerical accuracy (TF32). https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html ; NVIDIA TF32 blog: https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/
44. NVIDIA RTX A4000 datasheet. https://www.nvidia.com/content/dam/en-zz/Solutions/gtcs21/rtx-a4000/nvidia-rtx-a4000-datasheet.pdf
45. NVIDIA Ampere GA102 architecture whitepaper. https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf
46. JAX docs, buffer donation. https://docs.jax.dev/en/latest/buffer_donation.html
47. JAX docs, `jax.lax.scan`. https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html
48. XLA `compiled_memory_stats.h`. https://github.com/openxla/xla/blob/main/xla/pjrt/compiled_memory_stats.h
49. XLA GPU `fft_thunk.cc`. https://github.com/openxla/xla/blob/main/xla/backends/gpu/runtime/fft_thunk.cc
50. JAX docs, GPU memory allocation. https://docs.jax.dev/en/latest/gpu_memory_allocation.html
51. JAX docs, device memory profiling. https://docs.jax.dev/en/latest/device_memory_profiling.html
52. JAX docs, profiling. https://docs.jax.dev/en/latest/profiling.html
53. OpenXLA, dumping HLO. https://openxla.org/xla/hlo_dumps
54. JAX docs, persistent compilation cache. https://docs.jax.dev/en/latest/persistent_compilation_cache.html
55. JAX issue #26639. https://github.com/jax-ml/jax/issues/26639
56. JAX docs, GPU performance tips. https://docs.jax.dev/en/latest/gpu_performance_tips.html
57. OpenXLA operation semantics, Fft. https://openxla.org/xla/operation_semantics
58. JAX source `jax/_src/numpy/fft.py`. https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/fft.py
59. JAX issue #9591. https://github.com/jax-ml/jax/issues/9591
60. NVIDIA cuFFT documentation. https://docs.nvidia.com/cuda/cufft/index.html
61. Tolmachev, D. (2023). VkFFT: a performant, cross-platform and open-source GPU FFT library. IEEE Access 11:12039–12058. https://doi.org/10.1109/ACCESS.2023.3242240
