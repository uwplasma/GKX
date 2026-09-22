## Survey A: gyrokinetic codes, adjoints, and preconditioners

Scope and provenance. I read each claim below from a primary source: the paper's full text, official documentation, or the code repository. The source is cited with [n]. **[code]** means I read it from a public repository or its commit messages; it is not a published result. **UNVERIFIED** means I could not confirm it from a primary source. The earlier review already covers stella, GS2, GENE's Krylov–Schur and Jacobi–Davidson solvers, GX's explicit baseline and Kim's 2024 CSGF abstract, Freitag–Spence, GCRO-DR, inexact Krylov, and MGK. Those appear here only as cross-references.

### 1. CGYRO

- **Stiffness split.** CGYRO uses a mixed implicit–explicit scheme. All collisionless terms, including parallel streaming, drifts, drive and the nonlinearity, advance explicitly with RK4. In the docs this is `DELTA_T_METHOD=0`; adaptive Cash–Karp, Bogacki–Shampine and Verner options are also available, with `ERROR_TOL=1e-4` [1,2]. Collisions and the trapping (mirror, pitch-angle-derivative) term advance implicitly with second-order Crank–Nicolson. The two advances are operator-split, and the paper states the scheme is second order overall [1, §3.9].
- **How the implicit step is solved.** The fields are expressed as velocity sums of H. This folds φ, A∥ and B∥ into the collision step, so the step becomes one matrix equation of rank N_s·N_ξ·N_u for all species at each configuration point [1, eq. 130]. The fields are therefore eliminated inside a dense velocity-space matrix; the paper does not use a separate Schur or response-matrix solve. In practice the matrix is precomputed and stored as the "collisional constant tensor" `cmat`, with local size nv × nv × nc_loc. For benchmark nl03c, `cmat` is 10× the size of all other buffers combined. The authors describe this as trading memory for an "order of magnitude" compute speedup in the collision step [3].
- **Streaming and upwind dissipation.** Streaming in θ is explicit, and the authors state that explicit advection "must always include grid-scale dissipation" [1]. The scheme uses 3rd- or 5th-order upwind stencils (`NUP_THETA`). The dissipation acts on g = h − z δA∥ v∥/… rather than on h, which the paper finds more robust at finite β. It is weighted by |v∥|. The gyrocenter-density perturbation that the dissipation creates is projected out ("conservative upwind"), which fixes Ampère-cancellation-type errors at k⊥ρ_s < 0.1. The estimated numerical damping is γ_upwind ≈ 20 (a/qR)(v/c_s) λ³/N_θ⁵ [1, eqs. 83–88]. A spectral analogue of upwind dissipation is also applied to the radial drift, which enforces outgoing-wave behaviour in ballooning space [1, §3.5].
- **Why streaming stays explicit.** The authors write that implicit advection "is useful for small linear case, but prohibitively expensive in terms of storage and speed, for multiscale cases" (N_r > 1024, N_α > 256). The explicit Δt is set by the fastest numerical Alfvén wave or ω_H mode [1, §3.9].
- **Eigen and linear solvers.** None is documented. Linear runs are initial-value runs that stop at the frequency tolerance `FREQ_TOL=1e-3` [2].
- **Relevance to GKX.** Two things carry over. First, precomputing a factorization of every small z-local block, with the fields eliminated inside it, is the CGYRO pattern and would be cheap for GKX's (l,m) blocks. Second, their conservative-upwind idea is relevant if GKX adds z-dissipation to streaming. CGYRO provides no preconditioner to copy.

### 2. GENE (and GKV, GENE-X)

- **Time integration.** GENE uses explicit RK3, RK4, or an optimized 6-stage 4th-order RK. SLEPc computes the extreme eigenvalues of the linear operator and uses them to set the maximum stable Δt [4, §3.1]. Doerk & Jenko split collisions off with first-order Runge–Kutta–Chebyshev substeps and use SLEPc spectra to choose schemes, reporting up to 3× total speedup [5]. Crandall et al. confirm that GENE's collision step is still split-explicit RKC. They also report that collisional spatial diffusion at k_yρ_s ~ 10 can cut Δt by about 10×, which makes high-k_y edge runs "impractical until alternative implicit time-stepping schemes can be employed" [6, §3.3].
- **PETSc and SLEPc usage.** SLEPc serves both the eigenvalue mode and the Δt estimate [4,5]. A separate PETSc-based neoclassical solver exists, but its cost limited benchmarks to ν_c ≲ 0.3 [6]. I found no primary source for an implicit or IMEX streaming option in flux-tube GENE; the GENE manual is not public (**UNVERIFIED** either way). The eigen-preconditioners (Merz et al. 2012) are covered in the earlier review.
- **GKV** (cited by [6], documented in [7,8]). GKV splits collisions off and solves them implicitly with Crank–Nicolson, using a Krylov solver (Bi-CGSTAB per the GKV manual). Data is transposed so that the iterative solve needs no inter-node communication [7,8]. No preconditioner is described.
- **GENE-X.** GENE-X is a full-f edge code (FCI, finite differences) [9]. I could not confirm its time integrator or elliptic solver details from the full text (**UNVERIFIED**). It has little bearing on flux-tube spectral preconditioning.

### 3. GKW

- **Time integration.** The recommended scheme is explicit (RK4, modified midpoint, or a 2nd-order hyperbolic scheme). An implicit scheme (UMFPACK/AMD sparse LU) "is also implemented, but is still under development and not yet recommended for general use" [10, §8.6; 11].
- **Eigen solver.** GKW's eigen solver (`METHOD='EIV'`) wraps SLEPc and is matrix-free: the shell operator is either the RHS or one explicit time step, (1+dtN)^n. The field solve is kept implicit rather than forming −BD⁻¹C, because forming it would destroy sparsity [10, §6.2]. Extraction can be harmonic or Ritz, with a target growth rate and frequency. The manual reports that finding a third (stable) eigenvalue "takes about 16–18 times longer" than the leading ones [10, §6.1]. No preconditioner is used.
- **Relevance to GKX.** GKW is essentially the naive baseline: unpreconditioned, matrix-free, Krylov–Schur with harmonic extraction. It shows why interior or subdominant modes are expensive without shift-invert.

### 4. GX

- **Formulation** [12]. Mandell et al. (2018) project the gyrokinetic equation onto a Laguerre–Hermite basis. The fields involve only the lowest Hermite moments, with Laguerre sums weighted by Bessel-function coefficients. Streaming couples m±1; the toroidal drifts couple m±2 and l±1 [13, §6].
- **GX paper** [13]. The released solvers are explicit only (RK3 default, RK4, SSPx3, Ketcheson K10,4) [13, §5; 14]. Δt comes from a grid-frequency estimate that includes ω_A,max and ω_H, not from eigenvalues. The authors name IMEX as future work, citing the "sparse banded structure" of the system and the fact that the fields depend only on low moments [13, §5].
- **Implicit work since** [15]. The Kim, Mandell, Abel and Dorland Sherwood 2024 abstract describes third-order additive RK (Conde et al. 2017) for electron streaming in slab geometry. It claims a 60× larger stable Δt "with minimal increase in computational cost per time-step," for linear ES/EM and nonlinear ES runs.
- **Public GX IMEX branches** [code, 16]. The public Bitbucket repo has several branches:
  - `imex`: Mandell, 2022, "IMEX scheme for ESGK … it might be working?".
  - `pk/imex` and `pk/imex_bounce`: Kim, 2024. Includes "k's method with twist and shift" and LU re-solve for "bounce matrices".
  - `pk/imex_full`: active as of 2026-08-29.
  - `sundials-imex`: Reynolds and Amihere, 2026-05.

  Reading the source of `pk/imex_full`:
  - `ts_imex_full.cu` is a 3-stage ARK scheme with implicit electron solves of the form G = (I − γΔtB)⁻¹G.
  - Streaming is inverted in z-Fourier space (`zft_streaming_invert_full[_em]`).
  - `mirror_drifts_matrix.cu` assembles the z-local mirror+drift operator as a banded matrix: N = N_l·N_m, KL = KU = N_m, batch = N_z·N_s·N_ky·N_kx. It factors with MAGMA `cgbtrf_batched` and solves with `cgbtrs_batched`.
  - `cusolve.cu` uses cuSOLVER `Xgetrf`/`Xgetrs` on dense (N_l·N_m)² "bounce" matrices per z.
  - `green.cu` builds a Green's-function (response-matrix) field solve.

  This is structurally the closest existing analogue to GKX's pr3-cm factors: a per-k_z Hermite streaming solve plus an exact z-local (l,m) drift+mirror block. GX uses it as the implicit stage of a time integrator, however, not as a preconditioner. It is unpublished and its validation status is unknown; one 2024 commit message reads "crashes though."
- **GX `adjoint` branch** [code, 16]. One commit (2024-05-09): "Starting point for implementation of Acton–Barnes adjoint equations."

### 5. Adjoints and derivatives for optimization

| Work | Code | Derivative method | Cost per gradient (as reported) |
|---|---|---|---|
| Paul, Abel, Landreman & Dorland 2019 [17] | SFINCS (neoclassical DKE) | Continuous and discrete adjoint. The discrete adjoint reuses the forward preconditioner's LU factors for the transpose solve. The preconditioner is the full matrix minus cross-species and speed coupling. | Two linear solves regardless of N_params. Up to 200× cheaper than finite differences for O(10²) parameters. |
| Acton, Barnes, Newton & Thienpondt 2024 [18] | stella (linear, electromagnetic, collisional) | Adjoint gyrokinetic equation for dγ/dp. The adjoint is time-stepped to steady state using stella's operator split: SSP-RK3 explicit terms, semi-Lagrange mirror, and implicit streaming via a tridiagonal (Thomas) solve. Used inside Levenberg–Marquardt. | About 3.17 CPU-h (one forward plus one adjoint run), flat in N. Finite differences cost 4.75, 7.92 and 12.67 CPU-h for N = 2, 4, 7. |
| Jorge et al. 2024 [19] | GS2 (linear) + SIMSOPT/VMEC | Finite differences, with MPI-concurrent evaluations and Levenberg–Marquardt on a quasilinear-flux objective. | N+1 linear GS2 scans per gradient. |
| Kim et al. 2024 [20] | GX (nonlinear) + DESC | SPSA: 2 objective evaluations per iteration, a random-direction finite difference, chosen because the flux traces are noisy. | About 3 min per GX run and about 10 min per optimization iteration on one V100. About 20 h per optimization. |
| iGENE (Artigues, Merlo & Jenko 2026) [21] | TensorFlow GENE-like, RK4 | Reverse-mode AD over the last N RK4 steps from a saturated state. | Linear gradients match finite differences. Nonlinear gradients diverge for N ≳ 512 (Lyapunov growth), so only a truncated window is usable. At most 16,000 steps fit in one GPU's memory. |
| gyaradax (Galletti, Volkmann & Brandstetter 2026) [22] | JAX port of GKW, RK4 | Reverse-mode AD through time stepping. γ(k_y) Jacobian via `jacrev` in N_ky passes. | No gradient cost reported (**UNVERIFIED**). Forward speed is 1.8–10.5× GKW; a B300 GPU is compared against 64–128 CPU cores. |

What these show: the only gyrokinetic adjoint with published cost numbers (Acton) is time-stepped. None of these works differentiates through an eigen-solve via left eigenvectors. The discrete-adjoint trick of reusing the preconditioner factorization for the transpose system [17] transfers directly to GKX's shift-invert: the transpose of an ADI product is the reversed product of transposes, and block-Thomas has an exact transpose solve.

### 6. Preconditioners for Hermite, Laguerre and Legendre hierarchies and kinetic equations

- **Implicit Hermite-spectral Vlasov (LANL SPS line).** Delzanno 2015 introduced fully implicit Crank–Nicolson Fourier–Hermite Vlasov–Maxwell solved by JFNK [23]. Across this line of work the Krylov solve is mostly unpreconditioned:
  - Camporeale et al. use "non-preconditioned restarted GMRES" [24].
  - Pagliantini et al. (implicit Hermite-DG) use unpreconditioned JFNK/GMRES and explicitly leave "efficient preconditioning strategies" to future work [25].
  - The SPS review (Roytershteyn & Delzanno) describes PETSc JFNK without detailing a preconditioner [26].

  Conclusion: I found no published Hermite-hierarchy preconditioner of the pr3-cm type. The physics-based preconditioning that exists in this community is fluid/moment-based: the HOLO review [27], fluid preconditioning for implicit PIC [28], and the JFNK review [29]. The analogue for GKX would be a low-moment (gyrofluid) coarse space, not a Hermite-line solve.
- **Legendre and speed coupling (neoclassical DKE codes).**
  - SFINCS preconditions GMRES with a sparse LU of the matrix minus cross-species and speed coupling [17].
  - MONKES exploits the block-tridiagonal Legendre structure (the Legendre analogue of Hermite m±1 streaming) with exact block Gaussian elimination at O(N_ξ N_fs³). It gets low-collisionality coefficients in about 1 min on one core [30].
  - yancc (JAX, GPU) rejects sparse-LU preconditioners: they need hundreds of GB, degrade at high collisionality when terms are dropped, and "see minimal speedup on GPU". It uses multigrid-preconditioned Krylov with block-diagonal smoothers that each keep full coupling along one coordinate, applied in sequence. That is line relaxation, a close cousin of ADI. yancc reports about 10× speedup and about 10× less memory than SFINCS [31].
- **ADI and Peaceman–Rachford for kinetic equations.** Gasteiger, Einkemmer, Ostermann & Tskhakaya applied an ADI (x/v splitting) preconditioner to GMRES and Richardson for the steady inhomogeneous Vlasov equation. They report cost proportional to the number of grid points and close to a 100× speedup over the unpreconditioned solver [32]. This is the nearest published precedent for using ADI as a preconditioner, as opposed to a time integrator, in kinetic problems.
- **Recycling and deflation for shifted sequences.** Soodhalter, Szyld & Xue prove that no ideal method can recycle one augmented subspace for all shifts with storage independent of the shift count, and they propose practical compromises [33]. The recycling survey covers slowly changing sequences such as parameter scans and optimization iterates [34]. GCRO-DR is covered in the earlier review.
- **Response-matrix and Schur field solves in GK.** stella and GS2 use response matrices (earlier review). GS2 collisions use Sherman–Morrison to reduce to banded solves [6, citing GS2]. CGYRO folds the fields into a dense per-point velocity matrix [1]. The GX IMEX branch uses a Green's-function field response [code, 16].
- **Hermite–Laguerre moment codes.** GYACOMO (Hermite–Laguerre gyromoments) is explicit only (EE, RK2–4, SSP-RK3, DOPRI5); its collision matrices are precomputed with COSOLVER [code, 35].

### 7. GPU sparse-direct and batched banded or tridiagonal solvers

- **cuDSS or cuSOLVER sparse-direct.** I found no flux-tube GK publication using cuDSS or cuSOLVER sparse-direct solvers.
- **The only flux-tube GK instance found is code, not a paper.** The GX `pk/imex_full` branch uses MAGMA batched banded LU (`cgbtrf/cgbtrs_batched`) over N_z·N_s·N_ky·N_kx systems of size N_l·N_m with bandwidth N_m. It also uses cuSOLVER dense `Xgetrf/Xgetrs` for per-z bounce matrices [code, 16].
- **Adjacent GPU evidence (collisions, not streaming):**
  - XGC (PIC) collisions: Ginkgo batched BiCGSTAB cut linear-solve time by about 90% versus LAPACK band LU on the DIII-D case [36].
  - Adams et al.: PETSc and Kokkos batched TFQMR with Jacobi preconditioning for the Landau operator. They report "over 20x faster" than an aggregated solver in 2V on MI250X and about 2–3.5× in other settings. Matrix construction, not the solve, dominates in 3V [37].
  - CGYRO: precomputed dense per-point `cmat` [3].
- **Pattern.** When blocks are small and uniform, GPU practice is either precomputed or batched direct factorization (CGYRO, GX-IMEX) or batched preconditioned Krylov per system (XGC, PETSc). Nobody reports global sparse-direct solves on the GPU; yancc explicitly argues against them [31].

### Bottom line for GKX (sober)

1. **The closest literature analogue of pr3-cm is unpublished GX IMEX code.** It combines per-k_z Hermite streaming inversion with a batched banded z-local (l,m) mirror+drift factorization, but uses it as an ARK time integrator, not as a shift-invert preconditioner. The published Hermite-implicit work (the SPS line) mostly runs without preconditioning. GKX's preconditioner is therefore not duplicated in the literature. For the same reason there are no published iteration-count baselines to compare against.
2. **ADI as a kinetic preconditioner has one published success** (about 100× over no preconditioner [32]). Coordinate-wise line smoothing inside multigrid (yancc [31]) is the main alternative if the ADI sweep count stops paying off.
3. **Derivatives.** The cheapest proven route is the discrete adjoint with factorization reuse (Paul 2019 [17]); Acton's time-stepped adjoint is the only gyrokinetic one with published costs [18]. Nonlinear AD gradients are only trustworthy over short windows [21].

### References

1. J. Candy, E. A. Belli, R. V. Bravenec, "A high-accuracy Eulerian gyrokinetic solver for collisional plasmas," J. Comput. Phys. 324, 73 (2016). https://doi.org/10.1016/j.jcp.2016.07.039 (full text: https://www.osti.gov/pages/biblio/1512805)
2. CGYRO input documentation (DELTA_T_METHOD, NUP_THETA, FREQ_TOL). https://gafusion.github.io/doc/cgyro/cgyro_list.html
3. I. Sfiligoi et al., "Minimizing CGYRO HPC Communication Costs in Ensembles with XGYRO by Sharing the Collisional Constant Tensor Structure," ICPP Workshops 2025, pp. 210–212. https://doi.org/10.1145/3750720.3757303 ; https://arxiv.org/abs/2507.22245
4. T. Görler et al., "The global version of the gyrokinetic turbulence code GENE," J. Comput. Phys. 230, 7053 (2011). https://doi.org/10.1016/j.jcp.2011.05.034
5. H. Doerk, F. Jenko, "Towards optimal explicit time-stepping schemes for the gyrokinetic equations," Comput. Phys. Commun. (2014). https://doi.org/10.1016/j.cpc.2014.03.024 ; https://arxiv.org/abs/1403.7402
6. P. Crandall et al., "Multi-species collisions for delta-f gyrokinetic simulations: Implementation and verification with GENE," Comput. Phys. Commun. 255, 107360 (2020). https://doi.org/10.1016/j.cpc.2020.107360 ; https://www.osti.gov/pages/servlets/purl/1852090
7. S. Maeyama et al., "Implementation of a gyrokinetic collision operator with an implicit time integration scheme and its computational performance," Comput. Phys. Commun. 235, 9 (2019). https://doi.org/10.1016/j.cpc.2018.07.015
8. S. Maeyama, GKV manual (2018), §3.2.2. https://www.p.phys.nagoya-u.ac.jp/gkv/_src/3147/gkv_manual_20180315.pdf
9. D. Michels et al., "GENE-X: A full-f gyrokinetic turbulence code based on the flux-coordinate independent approach," Comput. Phys. Commun. 264, 107986 (2021). https://www.sciencedirect.com/science/article/abs/pii/S0010465521000989
10. A. G. Peeters et al., "GKW how and why" (GKW manual 0.4-b1). https://bitbucket.org/gkw/gkw/downloads/GKW_manual_0.4-b1.pdf
11. A. G. Peeters et al., "The nonlinear gyro-kinetic flux tube code GKW," Comput. Phys. Commun. 180, 2650 (2009). https://doi.org/10.1016/j.cpc.2009.07.001
12. N. R. Mandell, W. Dorland, M. Landreman, "Laguerre–Hermite pseudo-spectral velocity formulation of gyrokinetics," J. Plasma Phys. 84, 905840108 (2018). https://doi.org/10.1017/S0022377818000041
13. N. R. Mandell et al., "GX: a GPU-native gyrokinetic turbulence code for tokamak and stellarator design," J. Plasma Phys. 90, 905900402 (2024). https://doi.org/10.1017/S0022377824000631 ; https://arxiv.org/abs/2209.06731
14. GX input documentation ([Time] scheme). https://gx.readthedocs.io/en/latest/Inputs.html
15. P. Kim, N. Mandell, I. Abel, W. Dorland, "Towards Semi-Implicit Methods for the GX Gyrokinetics Code," Sherwood 2024 abstract. https://sherwoodtheory.org/sw2024/uploads/90/sherwood_2024_abstract.pdf
16. GX repository, branches `imex`, `pk/imex`, `pk/imex_bounce`, `pk/imex_full` (commit dcdd7aedd3: `src/ts_imex_full.cu`, `src/mirror_drifts_matrix.cu`, `src/cusolve.cu`, `src/green.cu`), `sundials-imex`, `adjoint`. https://bitbucket.org/gyrokinetics/gx
17. E. J. Paul, I. G. Abel, M. Landreman, W. Dorland, "An adjoint method for neoclassical stellarator optimization," J. Plasma Phys. 85, 795850501 (2019). https://doi.org/10.1017/S0022377819000527 ; https://arxiv.org/abs/1904.06430
18. G. Acton, M. Barnes, S. Newton, H. Thienpondt, "Optimisation of gyrokinetic microstability using adjoint methods," J. Plasma Phys. 90, 905900406 (2024). https://doi.org/10.1017/S0022377824000709 ; https://arxiv.org/abs/2403.12621
19. R. Jorge et al., "Direct microstability optimization of stellarator devices," Phys. Rev. E 110, 035201 (2024). https://doi.org/10.1103/PhysRevE.110.035201 ; https://arxiv.org/abs/2301.09356
20. P. Kim et al., "Optimization of nonlinear turbulence in stellarators," J. Plasma Phys. 90, 905900210 (2024). https://doi.org/10.1017/S0022377824000369 ; https://arxiv.org/abs/2310.18842
21. V. Artigues, G. Merlo, F. Jenko, "iGENE: A differentiable flux-tube gyrokinetic code in TensorFlow," Phys. Plasmas 33, 083901 (2026). https://doi.org/10.1063/5.0339939 ; https://arxiv.org/abs/2605.03086
22. G. Galletti, E. Volkmann, J. Brandstetter, "gyaradax: Local Gyrokinetics JAX Code," arXiv:2604.06085 (2026). https://arxiv.org/abs/2604.06085
23. G. L. Delzanno, "Multi-dimensional, fully-implicit, spectral method for the Vlasov–Maxwell equations with exact conservation laws in discrete form," J. Comput. Phys. 301, 338 (2015). https://doi.org/10.1016/j.jcp.2015.07.028
24. E. Camporeale, G. L. Delzanno, B. K. Bergen, J. D. Moulton, "On the velocity space discretization for the Vlasov–Poisson system: comparison between implicit Hermite spectral and Particle-in-Cell methods," Comput. Phys. Commun. 198, 47 (2016). https://doi.org/10.1016/j.cpc.2015.09.002 ; https://arxiv.org/abs/1312.4991
25. C. Pagliantini, G. Manzini, O. Koshkarov, G. L. Delzanno, V. Roytershteyn, "Energy-conserving explicit and implicit time integration methods for the multi-dimensional Hermite-DG discretization of the Vlasov–Maxwell equations," Comput. Phys. Commun. 284, 108604 (2023). https://doi.org/10.1016/j.cpc.2022.108604 ; https://arxiv.org/abs/2110.11511
26. V. Roytershteyn, G. L. Delzanno, "Spectral approach to plasma kinetic simulations based on Hermite decomposition in the velocity space," Front. Astron. Space Sci. 5, 27 (2018). https://doi.org/10.3389/fspas.2018.00027
27. L. Chacón et al., "Multiscale high-order/low-order (HOLO) algorithms and applications," J. Comput. Phys. 330, 21 (2017). https://doi.org/10.1016/j.jcp.2016.10.069
28. G. Chen, L. Chacón, C. A. Leibs, D. A. Knoll, W. Taitano, "Fluid preconditioning for Newton–Krylov-based, fully implicit, electrostatic particle-in-cell simulations," J. Comput. Phys. 258, 555 (2014). https://doi.org/10.1016/j.jcp.2013.10.052 ; https://arxiv.org/abs/1309.6243
29. D. A. Knoll, D. E. Keyes, "Jacobian-free Newton–Krylov methods: a survey of approaches and applications," J. Comput. Phys. 193, 357 (2004). https://doi.org/10.1016/j.jcp.2003.08.010
30. F. J. Escoto, J. L. Velasco, I. Calvo, M. Landreman, F. I. Parra, "MONKES: a fast neoclassical code for the evaluation of monoenergetic transport coefficients," Nucl. Fusion 64, 076030 (2024). https://doi.org/10.1088/1741-4326/ad3fc9 ; https://arxiv.org/abs/2312.12248
31. R. Conlin, M. Landreman, "yancc: A GPU-accelerated, differentiable solver for neoclassical transport in tokamaks and stellarators," arXiv:2607.20861 (2026). https://arxiv.org/abs/2607.20861
32. M. Gasteiger, L. Einkemmer, A. Ostermann, D. Tskhakaya, "Alternating direction implicit type preconditioners for the steady state inhomogeneous Vlasov equation," J. Plasma Phys. 83, 705830107 (2017). https://doi.org/10.1017/S0022377817000101 ; https://arxiv.org/abs/1611.02114
33. K. M. Soodhalter, D. B. Szyld, F. Xue, "Krylov subspace recycling for sequences of shifted linear systems," Appl. Numer. Math. 81, 105 (2014). https://doi.org/10.1016/j.apnum.2014.02.006 ; https://arxiv.org/abs/1301.2650
34. K. M. Soodhalter, E. de Sturler, M. E. Kilmer, "A survey of subspace recycling iterative methods," GAMM-Mitt. 43 (2020). https://doi.org/10.1002/gamm.202000016 ; https://arxiv.org/abs/2001.10347
35. GYACOMO repository (`src/time_integration_mod.f90`; README on COSOLVER matrices). https://github.com/Antoinehoff/gyacomo
36. A. Kashi, P. Nayak, D. Kulkarni, A. Scheinberg, P. Lin, H. Anzt, "Integrating batched sparse iterative solvers for the collision operator in fusion plasma simulations on GPUs," J. Parallel Distrib. Comput. 178, 69 (2023). https://doi.org/10.1016/j.jpdc.2023.03.012
37. M. F. Adams, P. Wang, J. Merson, K. Huck, M. G. Knepley, "A performance portable, fully implicit Landau collision operator with batched linear solvers," arXiv:2209.03228. https://arxiv.org/abs/2209.03228

UNVERIFIED items: whether GENE has any implicit streaming option (manual not public); GENE-X's integrator and field solver; gyaradax's gradient cost; the algorithm of Taitano et al. (J. Comput. Phys. 297, 357, 2015, https://doi.org/10.1016/j.jcp.2015.05.025), omitted because I could not read the abstract or full text; and the validation status of the GX IMEX branches.
