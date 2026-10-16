# Literature notes: stiff time integration and eigen-solvers for KE/EM gyrokinetics (for GKX)

How sure each claim is:
- **[V]** I checked it against the primary text (PDF read, or the official docs/abstract).
- **[M]** It comes from memory or a secondary source. Check it before citing.
- **[!]** This contradicts an assumption in the brief.

---

## 1. stella (Barnes, Parra, Landreman, JCP 391 (2019) 365; arXiv:1806.02162)

Ref: https://arxiv.org/abs/1806.02162, DOI 10.1016/j.jcp.2019.01.025 [V]

- **Splitting [V].** The gyrokinetic equation (GKE) is split as dg/dt = A[g] + B g + C g:
  - A: magnetic drifts, ω* drive and E×B nonlinearity, all explicit.
  - B: mirror force, i.e. advection in v∥.
  - C: parallel streaming, including the ∂z(J0 φ) term.
  - The splitting is first-order Lie. It runs in "flip-flop" order: A,B,C on one step, then C,B,A on the next. This is second order when A is linear, but the effective step doubles. Only first order is guaranteed with the nonlinearity.
  - The paper measures linear frequency error scaling as ~Δt² for flip-flop versus ~Δt^{5/4} for plain Lie. Overall accuracy is at most second order.
- **Explicit part [V].** Options are RK4, SSP-RK2 and SSP-RK3; SSP-RK3 is the default (Gottlieb–Shu–Tadmor, SIAM Rev. 43 (2001) 89). A higher-order explicit scheme is offered for its larger stability region, not for accuracy.
- **Mirror term [V].** Semi-Lagrangian interpolation in v∥. It imposes no stability or accuracy restriction on Δt.
- **Implicit streaming (GS2-style) [V].**
  - Compact two-point stencil in z, with spatial upwinding u_z: u_z=0 is centred and 2nd order, u_z=1 is fully upwind and 1st order.
  - Time centring u_t: u_t=0 is Crank–Nicolson, u_t=1 is backward Euler.
  - The result is a bidiagonal system in z for each (v∥, μ, s). It is coupled densely through φ^{n+1}, because quasineutrality integrates over velocity and sums over species.
- **Response matrix (Green's function) [V].**
  1. Write g^{n+1} = g_inh + g_h.
  2. g_inh solves the step with φ^{n+1}=0. g_h is the homogeneous response to φ^{n+1}.
  3. Build δg_h/δφ_p by applying a unit impulse in φ at each of the N_z+ = N_z·N_seg points of the extended (twist-and-shift connected) z domain.
  4. Substitute into quasineutrality. This gives an N_z+ × N_z+ dense system (I − Q·δg/δφ) φ^{n+1} = Q g_inh. It is LU-factored once.
  5. Each step costs one bidiagonal sweep per velocity point, one LU back-substitution of size N_z+ per (k_x chain, k_y), and a final g update.
  - Zonal modes (k_y=0, periodic in z) are solved twice: once with zero incoming BC and once with a unit incoming BC. The two are combined to enforce periodicity.
  - **Cost model.** Setup is N_z+ implicit sweeps per chain plus an O(N_z+³) LU. The matrix must be rebuilt whenever Δt changes.
- **Measured gain [V].** For kinetic electrons at k_yρ_i≈1, the explicit stable Δt is about 100× smaller than the IMEX Δt, and the gap widens at long wavelength.
  - Explicit CFL from the electrostatic shear-Alfvén / ω_H wave scales as k∥/k⊥.
  - Uncentred ∂zφ gives a worse 1/k⊥² limit (their App. A). **This bears on GKX: keep the ∂z φ coupling centred.**
  - With IMEX, the remaining Δt limit comes from the explicit magnetic drifts: ∝1/k_y at high k_y, and also at long wavelength through drift×φ coupling.
- **Electromagnetic stella [V, thesis; I found no journal paper].** Source: R. Davies, PhD thesis, Univ. York (2022), ch. 6 and App. B, https://etheses.whiterose.ac.uk/32068/1/Davies_thesis.pdf
  - A∥ and B∥ are added linearly, with both implicit and explicit options, and benchmarked against GS2 in β scans.
  - Formulation: evolve ḡ = g + (Z/T) v∥ ⟨A∥⟩-type combination, which removes ∂A∥/∂t. Sources are written in h (the "mixed ḡ/h formulation").
  - The implicit streaming response matrix now contains **all three fields** ("similar to GS2").
  - The mirror term also needs a response-matrix treatment in EM, because it acts on h and so needs updated fields.
  - Drifts stay explicit. Their CFL cost Δt about 10× going from nperiod=2 to nperiod=4, because k⊥ grows along z.
- **Recent extension [V].** Stroteich, Acton, Barnes, Xanthopoulos, Schneider, "Numerical and diagnostic extensions to the gyrokinetic δf code stella", arXiv:2610.02958 (Oct 2026), https://arxiv.org/abs/2610.02958
  - Adds explicit integrators SSP32/42/52, SSP43 (C=20/11) and Doerk's RK64.
  - Reports an "efficiency proxy" = imaginary-axis extent at Re(λΔt) = −0.003, divided by stage count: RK4 0.71, SSP33 0.58, SSP43 0.56, RK64 0.82, SSP22 0.20.
  - Best observed: SSP22 for the linear test, SSP33 for the nonlinear test. Tests used adiabatic electrons. **EM is listed as future work there.**
  - stella's automatic CFL-based Δt adaptation exists; they disabled it for their tests.
- **stella Δt adaptation [M].** I believe it checks the explicit-term CFL every step against cfl_cushion bounds. When violated it shrinks Δt by a fixed factor and **recomputes response matrices**; it grows Δt when well below the limit. I did not read the source.
- **"Acton et al. stella EM paper" [!]** I could not find a 2024/25 stella EM journal paper. The only verified stella EM description is the Davies thesis. Acton appears as co-author on the 2026 numerics paper above, which is electrostatic.

## 2. GS2 (Kotschenreuther, Rewoldt, Tang, CPC 88 (1995) 128)

Ref: DOI 10.1016/0010-4655(95)00035-E [M, DOI from memory]; record: https://collaborate.princeton.edu/en/publications/comparison-of-initial-value-and-eigenvalue-codes-for-kinetic-toro/ [V]

- **Algorithm [V].** Source: Hammett, GS2 algorithm notes, https://w3.pppl.gov/~hammett/work/GS2/docs/algorithm.txt
  - All linear terms are implicit: streaming, drifts, drive, mirror (implicit through (ε, λ) coordinates) and fields. Only the nonlinearity is explicit.
  - θ-scheme: (f^{n+1}−f^n)/Δt = −L(δ f^{n+1} + (1−δ) f^n), with δ = 1 − (fexpr + i fexpi). fexpr=0.5 is time-centred; ~0.4–0.48 is used in practice. Spatial upwind/off-centring uses bakdif.
  - Stability: δ ≥ 0.5 is stable for all Δt. With δ=0.5, |A|=1 for real frequency and strongly damped modes → A ≈ −1 (no damping; ringing).
  - **Caveat for eigen-hunting.** Implicitness can under-represent a fast-growing mode with large Δt·Re ω, so a slower mode looks dominant.
- **Response matrix [V].** The full system size is (N_E N_ξ N_s + N_fields) N_θ. KRT splits it into an implicit field solve and an implicit f solve, with no approximation.
  - Setup cost ~ (N_fields N_θ)³: unit impulse per field per θ point of the extended ballooning domain, then dense inversion.
  - Per-step cost ~ C1 (N_fields N_θ)² + C2 N_E N_ξ N_θ N_s.
  - N_fields = 3 for (φ, A∥, B∥) [V]. Pedagogical derivation: E. Belli PhD thesis (Princeton 2006) [M].
- **Nonlinear Δt [M].** GS2 checks an E×B CFL each step (cfl factor in the input). If violated, Δt is reduced by delt_adj (≈2) and **all response matrices are recomputed**; Δt is increased back toward the user's delt when safe. delt is a hard upper bound in nonlinear runs [V, CodeRunner wiki: https://bitbucket.org/gyrokinetics/gs2/wiki/CodeRunner_Test_Page].
- **Eigen-solver [V].** GS2 eigval_knobs, https://bitbucket.org/gyrokinetics/gs2/wiki/Namelists/eigval_knobs
  - SLEPc is applied to the **time-advance operator** (a time-stepper eigenproblem).
  - Defaults: solver_option = Krylov-Schur. Also available: JD, Arnoldi, CISS, FEAST, etc. extraction_option includes harmonic and refined variants.
  - Target set by targ_re/targ_im; the wiki suggests Re≈0, Im≈10.
  - nadv = number of GS2 steps per SLEPc "mat-vec", used to separate clustered eigenvalues.
- **ADI warning, directly relevant to GKX [V].** Belli & Hammett, "A numerical instability in an ADI algorithm for gyrokinetics", CPC 172 (2005) 119, DOI 10.1016/j.cpc.2005.06.007
  - An ADI scheme that separates the field solve from the f advance has a severe Δt restriction in gyrokinetics.
  - So does partially implicit Adams–Bashforth with implicit parallel motion and explicit fields.
  - Fully explicit can beat both at long wavelength.
  - **Lesson: implicit streaming without implicit field response is unstable/ineffective. Field response must be in the implicit block, as GKX already does with Woodbury.**

## 3. GENE

- **Time stepping [V].** Explicit RK4 by default (Roman et al. 2010 below). SLEPc computes the largest-|λ| eigenvalue of L so that Δt can be set exactly from the RK stability region.
  - Doerk & Jenko, "Towards optimal explicit time-stepping schemes for the gyrokinetic equations", CPC 185 (2014) 1938, arXiv:1403.7402 [V content; M page/DOI], https://arxiv.org/abs/1403.7402
  - Spectrum of the collisionless L is stretched along the imaginary axis. The most restrictive eigenvalue is at the smallest or largest k_y.
  - Among 4-stage methods, RK4 (β_imag = 2√2) is optimal per stage. RK4M (6-stage) is 18% below the bound β_imag ≤ s−1.
  - RKC (Runge–Kutta–Chebyshev) is used for collisions (negative-real spectrum, β_real ≈ 1.93 s²) through Strang splitting e^{CΔt/2} e^{(N+L)Δt} e^{CΔt/2}.
  - Nonlinear Δt comes from an estimate of how E×B shifts eigenvalues along the imaginary axis. Up to 3× speedups.
- **ω_H / KSAW limit [V].** Same Doerk & Jenko paper. Dispersion relation ω²_KSA = k∥² c_s² / (μ_e (k⊥ρ_s)² + β_e).
  - With k∥L ~ 10, k⊥ρ_s ~ 0.05, μ_e = 1/3600 and β_e = 0, ω ~ 4000 c_s/L, about 10⁴× typical γ.
  - "Even small values of β_e prevent the divergence." So run KE electrostatic cases at small finite β; GX does the same with β_ref = 1e-5 [V, GX paper].
  - For β_e/μ_e ≳ (k⊥ρ_s)², it becomes the ordinary Alfvén wave v_A k∥. The electron streaming CFL k∥ v_te is then usually more restrictive than the Alfvén limit.
  - Upwind or dissipative discretisations can strongly distort the spectrum. Dissipation on low-k_y φ̄ creates large negative eigenvalues; avoid it.
- **Eigen-solvers [V].**
  - (a) Roman, Kammerer, Merz, Jenko, "Fast eigenvalue calculations in a massively parallel plasma turbulence code", Parallel Comput. 36 (2010) 339. PDF: https://genecode.org/PAPERS_2/parco2010.pdf
    - Matrix-free. Rightmost eigenvalues behave like *interior* eigenvalues because the spectrum is elongated along the imaginary axis.
    - Compared inexact shift-invert and Cayley (GMRES/BiCGStab/TFQMR; FGMRES with an inner GMRES whose tolerance starts at 0.9 and decreases) against **Krylov–Schur with harmonic Ritz extraction** (target κ, no linear solves).
    - Harmonic projection was "always at least five times faster" and more reliable. Cayley gave no advantage over SI in their tests.
    - SI with a far shift converges faster but less accurately. A near shift makes the inner solves blow up.
  - (b) Romero & Roman, "Computing subdominant unstable modes of turbulent plasma with a parallel Jacobi–Davidson eigensolver", CCPE 23 (2011) 2179, DOI 10.1002/cpe.1740, https://riunet.upv.es/handle/10251/34280
  - (c) Merz, Kowitz, Romero, Roman, Jenko, "Multi-dimensional gyrokinetic parameter studies based on eigenvalue computations", CPC 183 (2012) 922, DOI 10.1016/j.cpc.2011.12.018. PDF: https://riunet.upv.es/bitstreams/e53dd022-06ed-493f-ad5b-7eb1477ec408/download
    - Jacobi–Davidson (JD) correction equation solved by BiCGStab(2), with only ~5 iterations needed.
    - **Preconditioner K⁻¹ ≈ (L_g − σI)⁻¹, where L_g is the part of L acting directly on g, with the field (χ) terms dropped.** L_g is an explicit sparse matrix with only 9 diagonals versus ~4000 for L. It keeps parallel electron dynamics, the dominant term that sets the spectral radius.
    - Factorisation is ASM+ILU (overlap δ=2) or pARMS, which is Schur-complement multilevel ILU.
    - L_g is block-diagonal in species s and μ, so block-Jacobi over (s, μ) is effective.
    - The shift is fixed at σ=3, somewhat larger than the target τ=1, so the factorisation is not rebuilt per iteration.
    - Result: JD+ASM was 7× faster than harmonic Krylov–Schur on a 125-point scan. They also used subspace recycling across scan points.
- **GENE implicit time stepping [!].** I found no evidence of an implicit time integrator in flux-tube GENE. Treat it as explicit-only plus RKC for collisions.

## 4. CGYRO (Candy, Belli, Bravenec, JCP 324 (2016) 73)

Ref: DOI 10.1016/j.jcp.2016.07.039 [V citation; I did not read the full text]

- **[!] Streaming is explicit in CGYRO, not implicit.** The official input docs say "CGYRO uses an explicit time-integration scheme for collisionless terms", so Δt must be smaller than in GYRO [V]. https://gafusion.github.io/doc/cgyro/cgyro_list.html
- **Time integration [V].** DELTA_T_METHOD (from the same docs page):
  - 0 = RK4 (default, fixed step)
  - 1 = Cash–Karp 6:5(4), adaptive
  - 2 = Bogacki–Shampine 7:5(4), adaptive
  - 3 = Verner 10:7(6), adaptive
  - ERROR_TOL default 1e-4. Embedded-pair error control applies to the explicit collisionless part.
- **θ discretisation [V].** Conservative upwind, 1st/3rd/5th order (NUP_THETA, default 5th order). The scheme projects out density and current perturbations created by the grid-scale dissipation.
- **Collisions [V; mechanism partly M].**
  - Operator-split implicit collision step using a precomputed constant tensor "cmat" of size nv × nv × nc × nt. nv = velocity×species, nc = configuration (radial×θ), nt = toroidal modes. So it is a dense velocity-space matrix per (configuration point, toroidal mode), applied as a matrix–vector product each step.
  - cmat dominates memory (10× all other buffers for the nl03c benchmark). Ref: Sfiligoi et al., XGYRO, arXiv:2507.22245, https://arxiv.org/abs/2507.22245
  - **[M]** I recall the step is Crank–Nicolson-type, cmat = (I − Δt/2 C)⁻¹(I + Δt/2 C), possibly including the field-particle/field response so the step is self-consistent. Not verified.
- **Velocity grid [V].** Uses the NEO (ξ, energy) grid with bounded energy. GX notes CGYRO's bounding of v∥,max as a way to ease the streaming CFL (GX paper §7). **GKX analogue:** the largest Hermite collocation v∥ grows ~√(2 Nm), which sets the explicit streaming CFL.
- **Phases [V].** Each step has three communication phases: str (needs full nc), nl (full nt) and coll (full nv).

## 5. GX (Mandell et al., arXiv:2209.06731; J. Plasma Phys. 2024)

Ref: https://arxiv.org/abs/2209.06731 [V]. JPP citation 90(4) 905900402, DOI 10.1017/S0022377824000631 [M]

- **Integrators [V].** Explicit only: standard RK, SSP-RK, and Ketcheson's 10-stage 4th-order SSP low-storage scheme (Ketcheson, SISC 30 (2008) 2113), chosen for GPU memory. All benchmarks used RK3. IMEX for electrons is listed as future work. They note the banded Hermite structure and the fact that fields need only the lowest moments both favour IMEX.
- **Δt control [V].** No eigenvalue computation. They estimate ω_x,max, ω_y,max (drift and E×B) and ω_z,max = max[(b·∇z) k_z,max v∥,max v_t,max, ω_A,max].
  - ω_A,max = (b·∇z) k_z,max v_te / sqrt(β_ref n_e τ_e m_i/(2 m_e)... + k⊥,min² ρ_s²). This is the shear-Alfvén frequency; it becomes ω_H as β → 0.
  - Δt (ω_x + ω_y + ω_z) ≤ C s, with s = 1.73 for RK3 and 2.82 for RK4 (imaginary-axis extents √3 and 2√2), and C ≈ 0.5–0.9.
  - v∥,max comes from the largest Gauss–Hermite node for Nm, and (μB)max from the largest Gauss–Laguerre node for Nl.
- **EM [V].** φ, A∥, B∥, benchmarked in β scans against GS2. KE electrostatic runs use β_ref = 1e-5 to tame ω_H.
- **Dissipation [V].**
  - Hyperviscosity −D (k⊥/k⊥,max)^{2n}, with D=0.05, n=2.
  - Hypercollisions −ν_hyp m^p G_{ℓ,m} for m > 2, which conserves n, u and T. ν_hyp is chosen to remove the same energy at large m as parallel phase mixing would (App. B, uses the Hermite-space advection velocity √(2m) k∥ v_t).
  - No upwinding in z, because z is spectral.
- **Note.** The hypercollision rate scales like k∥ v_t m^{p−1/2}. Treated explicitly, it adds a negative-real eigenvalue ∝ m^p, which can set Δt at large p. It is a natural candidate for the implicit (diagonal) part of an IMEX scheme.

## 6. GKW (Peeters et al., CPC 180 (2009) 2650)

Ref: DOI 10.1016/j.cpc.2009.07.001 [V]. Manual v0.4-b1: https://bitbucket.org/gkw/gkw/downloads/GKW_manual_0.4-b1.pdf [V]

- **Time integration [V, manual §8.6].** Explicit modified midpoint, RK4 (recommended), or a 2nd-order scheme stable for hyperbolic equations. Also an implicit scheme (METHOD='IMP', UMFPACK sparse LU) described as experimental and not recommended.
- **Eigen-solver [V, manual ch. 6].** METHOD='EIV' with SLEPc.
  - **The SLEPc operator is the explicit time-step propagator (1 + Δt N)^n with n = naverage** (or the RHS directly, mat_vec_routine=2). It is a shell matrix: fields are eliminated implicitly per mat-vec, never formed as D⁻¹C.
  - Eigenvalue map: λ_gkw = ln(λ_slepc)/(n Δt).
  - Extraction is Ritz (default) or harmonic (needs a target; "a kind of shift-and-invert"). type_solver='krylovschur'.
  - FAQ: Δt cannot exceed the explicit limit. Very small Δt slows convergence. Too tight a tolerance can prevent finding the right modes.
- **Lesson.** GKW (and GS2) find dominant growth rates by Arnoldi on the time-stepper, a Tuckerman–Barkley-style approach, not by shift-invert of L.

## 7. IMEX Runge–Kutta literature

- **ARS [V citation; M details].** Ascher, Ruuth, Spiteri, "Implicit-explicit Runge–Kutta methods for time-dependent PDEs", Appl. Numer. Math. 25 (1997) 151, DOI 10.1016/S0168-9274(97)00056-1. Scanned copy: https://es.scribd.com/document/354602919/AscherANM97
  - ARS(2,3,2): implicit SDIRK with γ = 1 − 1/√2, L-stable and stiffly accurate. Explicit part has δ = −2√2/3.
  - ARS(3,4,3): implicit γ ≈ 0.4358665, L-stable. Explicit part's imaginary-axis extent ≈ 2.82 [V, listed in arXiv:2506.09519 Table].
- **ARS(2,3,2) explicit part, my own algebra [M; check it].**
  - Its stability polynomial is 1 + z + z²/2 + γ²(1−δ) z³, and γ²(1−δ) ≈ 0.0858 × 1.9428 ≈ 1/6. So the explicit polynomial is essentially RK3's, with imaginary-axis extent ≈ √3.
  - **So GKX's ARS(2,3,2) instability is probably not a pure explicit-part stability problem.** Likely causes:
    - (a) coupled, non-commuting stability of implicit streaming plus explicit drifts. Both are oscillatory; this is "H-stability";
    - (b) order reduction or the stiff-limit behaviour of the field-coupled implicit part;
    - (c) the explicit drift/drive eigenvalues having positive real parts near the imaginary axis.
- **Coupled oscillatory IMEX stability (H-stability) [V].**
  - Weller, Lock, Wood, "Runge–Kutta IMEX schemes for the horizontally explicit/vertically implicit (HEVI) solution of wave equations", JCP 252 (2013) 365, DOI 10.1016/j.jcp.2013.06.025
  - Lock, Wood, Weller, QJRMS 140 (2014) 1654, DOI 10.1002/qj.2246 [M pages]
  - Steyer, Vogl, Taylor, Guba, "Efficient IMEX RK methods for nonhydrostatic dynamics", arXiv:1906.07219, https://arxiv.org/abs/1906.07219
  - Test equation: u̇ = −i k_x N u − i k_z S u, with N and S **non-commuting**. Both the explicit and implicit operators are skew (imaginary spectra). The stability matrix is R_H(x,z) = I − i(bᵀ⊗xN + b̂ᵀ⊗zS)(I + A⊗ixN + Â⊗izS)⁻¹(1⊗I).
  - The scalar "IMEX stability region" analysis is insufficient for non-commuting operators. This is GKX's case: streaming (z, Hermite) and drifts (k⊥, Laguerre/Hermite energy) do not commute.
  - In HOMME-NH, the max stable step (MUS) was ARS343 > ARK324 > ARS232 (e.g. 275 vs 125 at the ×1 planet), and new IMKG2/3 methods were designed for large H-stability regions.
  - **Recommendation:** compute GKX's R_H numerically on the actual per-chain (streaming+field) ⊕ (drift) linear operators to screen tableaux.
- **Imaginary-axis bound [V].** An r-stage explicit RK method has imaginary-axis extent ≤ r−1 (Kinnmark & Gray 1984; Theorem 2.1 in Steyer et al.).
- **ARK with embedded pairs, Kennedy & Carpenter [M details].**
  - "Additive Runge–Kutta schemes for convection–diffusion–reaction equations", Appl. Numer. Math. 44 (2003) 139, DOI 10.1016/S0168-9274(02)00138-1. Defines ARK3(2)4L[2]SA (γ = 0.4358665215, same γ as ARS343), ARK4(3)6L[2]SA (γ = 1/4) and ARK5(4)8L[2]SA (γ = 41/200).
  - Kennedy & Carpenter, "Higher-order additive Runge–Kutta schemes for ODEs", Appl. Numer. Math. 136 (2019) 183, DOI 10.1016/j.apnum.2018.10.007. Defines ARK4(3)7L[2]SA and ARK5(4)8L[2]SAb.
  - All are ESDIRK: constant diagonal γ, so **one factorisation of (I − γΔt L_I) per Δt serves every stage**. That suits the per-chain block-Thomas factorisation. The embedded pair gives a free error estimate.
- **SUNDIALS ARKODE [V].**
  - Butcher tables: https://sundials.readthedocs.io/en/latest/arkode/Butcher_link.html. IMEX pairs: ARK2 (GKC21), ARS(2,2,2), SSP2(3,3,2)-LSPUM, ARK324L2SA_ERK_4_2_3 (default 3rd order), ARK436L2SA_ERK_6_3_4, ARK437L2SA_ERK_7_3_4 (default 4th order), ARK548L2SA(b)_ERK_8_4_5.
  - Math: https://sundials.readthedocs.io/en/latest/arkode/Mathematics_link.html
    - Error test ‖β(y−ỹ)‖_WRMS ≤ 1 with bias β = 1.5.
    - Growth cap 20 per step. After two failures h'/h is capped at 0.3, then 0.1.
    - **Matrix (I − γJ) is reused until |γ/γ̃ − 1| > 0.2** or 20 steps pass. This is the template for GKX: only refactor the chain blocks when Δt drifts more than 20%, or quantise Δt to a ladder of cached factorisations.
- **SSP-IMEX [M].** Pareschi & Russo, "Implicit–explicit Runge–Kutta schemes and applications to hyperbolic systems with relaxation", J. Sci. Comput. 25 (2005) 129, DOI 10.1007/s10915-004-4636-4. Their SSP2(3,3,2) / SSP3(4,3,3) are designed for stiff relaxation (negative-real implicit part), not oscillatory implicit parts.
- **Stiff-limit caution [M].** For implicit terms with huge imaginary eigenvalues, L-stability (R(∞) = 0) damps unresolved electron/Alfvén oscillations. That is good for robustness but changes high-frequency physics. A-stable but not L-stable choices (CN-like) give |R| = 1 on the imaginary axis, with ringing and no damping (see the GS2 notes in §2). ARS and KC ESDIRKs are L-stable.

## 8. Exponential integrators

- **References.**
  - Cox & Matthews, "Exponential time differencing for stiff systems", JCP 176 (2002) 430, DOI 10.1006/jcph.2002.6995 [M]
  - Kassam & Trefethen, "Fourth-order time-stepping for stiff PDEs", SISC 26 (2005) 1214, DOI 10.1137/S1064827502410633 [M]. ETDRK4 with φ-functions evaluated by contour integrals, practical when L is diagonal or small-block.
  - Hochbruck & Ostermann, "Exponential integrators", Acta Numerica 19 (2010) 209, DOI 10.1017/S0962492910000048 [M]. A review of exponential RK, Rosenbrock, multistep and Lawson methods, including stiff-order conditions.
- **Krylov φ-functions.**
  - Sidje, Expokit, ACM TOMS 24 (1998) 130, DOI 10.1145/285861.285868 [M]
  - Niesen & Wright, phipm, ACM TOMS 38 (2012) art. 22, arXiv:0907.4631 [V]: adaptive in both time-substep and Krylov dimension.
  - Gaudreault, Rainwater, Tokman, KIOPS, JCP 372 (2018) 236, DOI 10.1016/j.jcp.2018.06.026, arXiv:1804.05126 [V]: incomplete orthogonalisation, single exponential of an augmented matrix, faster than phipm.
- **EPIRK / exponential Rosenbrock [M].** Tokman, JCP 213 (2006) 748 and JCP 230 (2011) 8762. Hochbruck, Ostermann, Schweitzer, SINUM 47 (2009) 786.
- **Plasma applications.**
  - Crouseilles, Einkemmer, Prugger, "An exponential integrator for the drift-kinetic model", CPC 224 (2018) 144, DOI 10.1016/j.cpc.2017.11.003, arXiv:1705.09923 [V]. Removes the CFL condition from the linear part; nonlinearity via Arakawa explicit.
  - **Crouseilles, Einkemmer, Massot, "Exponential methods for solving hyperbolic problems with application to collisionless kinetic equations", JCP 420 (2020) 109688, arXiv:1910.12720 [V]:**
    - Classic exponential integrators (ETD-RK and similar) behave erratically in allowed Δt when the explicit remainder has imaginary spectrum, and show "serious deficiencies in stability".
    - **Lawson (integrating-factor) methods do not**, and are recommended. Tested on Vlasov–Poisson and drift-kinetic ITG.
- **Relevance to GKX.**
  - The stiff linear part (streaming + field response per ballooning chain, block-tridiagonal in Hermite) is small and dense per chain.
  - e^{τ L_chain} could be precomputed per (k_y, chain) once per Δt: an eigendecomposition or Padé of the block. Or apply it through a few Krylov steps (expv).
  - **Lawson-RK = RK on v = e^{−tL} g** inherits the explicit RK's imaginary-axis stability for the remainder. That is attractive since drifts are oscillatory.
  - Caveat: Lawson has stiff order reduction, and e^{τL} must be accurate for the field-coupled modes.

## 9. Preconditioners

- **Physics-based preconditioning [M].**
  - Chacón, Knoll, Finn, "An implicit, nonlinear reduced resistive MHD solver", JCP 178 (2002) 15, DOI 10.1006/jcph.2002.7015
  - Chacón & Knoll, "A 2D high-β Hall MHD implicit nonlinear solver", JCP 188 (2003) 573
  - Chacón, "An optimal, parallel, fully implicit Newton–Krylov solver for three-dimensional viscoresistive MHD", Phys. Plasmas 15 (2008) 056103
  - Knoll & Keyes, "Jacobian-free Newton–Krylov methods: a survey", JCP 193 (2004) 357, DOI 10.1016/j.jcp.2003.08.010
  - Core idea: precondition the stiff wave (Alfvén/whistler) coupling by a parabolised Schur complement. The hyperbolic block system [[I, Δt D], [Δt E, I]] is reduced to (I − Δt² E D) on one variable, which is approximated and solved by multigrid.
  - **GKX analogue:** the Schur complement of the distribution block onto the fields (φ, A∥, B∥) per chain is exactly the response-matrix / Woodbury reduction GS2 and stella use.
- **Moment (fluid) preconditioning of kinetic systems [V abstract].** Chen, Chacón, Barnes, "Fluid preconditioning for Newton–Krylov-based, fully implicit, electrostatic particle-in-cell simulations", arXiv:1309.6243, https://arxiv.org/abs/1309.6243
  - Linearised moment equations closed with kinetic moments are used as the preconditioner. Performance is insensitive to the mass ratio.
  - **For a Hermite–Laguerre basis this is natural:** low (ℓ,m) moments are the fluid moments, and fields couple only to m ≤ 1 (A∥ to m=1, φ/B∥ to m=0 and Laguerre ℓ). So a preconditioner could be an exact solve on the low-moment + field Schur block, plus a cheap (diagonal/Thomas) solve on high m.
- **Hermite implicit spectral codes.**
  - Delzanno, "Multi-dimensional, fully-implicit, spectral method for the Vlasov–Maxwell equations with exact conservation laws in discrete form", JCP 301 (2015) 338, DOI 10.1016/j.jcp.2015.07.028 [V citation; OSTI 1247042]. JFNK. Abstracts say preconditioning cut Krylov iterations ~10×; I could not see the preconditioner details.
  - Camporeale, Delzanno, Bergen, Moulton, arXiv:1312.4991 (1D1V Fourier–Hermite JFNK) [V].
  - Roytershteyn & Delzanno, Front. Astron. Space Sci. 5:27 (2018), DOI 10.3389/fspas.2018.00027 [V]: SpectralPlasmaSolver review.
  - Vencels et al., "SpectralPlasmaSolver", J. Phys. Conf. Ser. 719 (2016) 012022 [M].
  - Koshkarov et al., "The multi-dimensional Hermite-discontinuous Galerkin method for the Vlasov–Maxwell equations", CPC 264 (2021) 107866 [M]. I found no published Hermite-specific preconditioner beyond the moment/JFNK ideas.
- **Implicit gyrokinetics with Krylov solvers.**
  - GT5D (Idomura, full-f Eulerian) [V via secondary]: ASIRK 2nd-order additive semi-implicit RK with implicit fast kinetic-electron terms. The implicit solve is >80% of runtime.
  - ILU preconditioning *hurt* convergence there because the diagonal is small relative to off-diagonals, the same pathology as pure streaming/Alfvén operators. Later work used communication-avoiding GMRES with an FP16 preconditioner.
  - Idomura, JCP 313 (2016) 511 [V citation]; SC20 Idomura et al. https://sc20.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap508.html [V]
  - XGC fully implicit EM (Sturdevant et al., based on Chen & Chacón): residual in field variables only; Picard, Anderson or JFNK [V talk abstract]: https://cnls.lanl.gov/External/ICNSP/Sturdevant.pdf
  - COGENT edge IMEX: JFNK with an operator-split multiphysics preconditioner [V abstract]: https://focm2023.pages.math.cnrs.fr/files/confirmed/I1-04580-Ghosh.pdf
- **Block-Jacobi per species [V, GENE CPC 2012].** L_g (no fields) is block-diagonal in species and μ. Block-Jacobi over (s, μ) with ILU of the z–v∥ band was effective. **This directly addresses GKX's single-shift ADI failure:** factor (L_g,s − σI) separately per species, which handles ion and electron streaming scales exactly, rather than one ADI parameter for both.

## 10. Eigen-solvers for non-normal stiff operators

- **Shift-invert / Cayley.**
  - Meerbergen, Spence, Roose, "Shift-invert and Cayley transforms for detection of rightmost eigenvalues of nonsymmetric matrices", BIT 34 (1994) 409 [M]
  - Lehoucq & Salinger, IJNMF 36 (2001) 309 [M]: Cayley transform for large CFD stability.
  - Roman et al. 2010 [V] found Cayley ≡ SI within Krylov (T_C = I + (σ−s)T_SI) and no advantage in GENE.
- **Inexact SI with preconditioned inner solves.**
  - Simoncini, "Variable accuracy of matrix–vector products in projection methods for eigencomputation", SINUM 43 (2005) 1155 [M]: inner tolerance can be *relaxed* as the outer residual decreases.
  - Freitag & Spence, "Shift-invert Arnoldi's method with preconditioned iterative solves", SIMAX 31 (2009) 942, DOI 10.1137/080716281 [M]: the **tuned preconditioner** P_t, with P_t x_k = A x_k on the current Ritz vector, keeps inner iteration counts bounded as the outer iteration converges.
- **Jacobi–Davidson [M].** Sleijpen & van der Vorst, "A Jacobi–Davidson iteration method for linear eigenvalue problems", SIMAX 17 (1996) 401, DOI 10.1137/S0895479894270427.
  - Only needs approximate correction-equation solves with a preconditioner for (A − θI). GENE uses it with K ≈ (L_g − σI)⁻¹, BiCGStab(2) and 5 iterations [V].
  - Tolerant of a preconditioner built with a fixed σ ≠ θ.
- **Time-stepper Arnoldi [M].** Tuckerman & Barkley, "Bifurcation analysis for timesteppers", in Numerical Methods for Bifurcation Problems and Large-Scale Dynamical Systems, IMA Vol. 119, Springer (2000) 453.
  - Arnoldi on the propagator e^{τL}, or on an implicit/IMEX step map (I − ΔtL_I)⁻¹(I + ΔtL_E).
  - Stiff fast modes map near 0 (L-stable implicit) or inside the unit circle. Rightmost eigenvalues of L become the largest |μ|. No inner linear solves are needed beyond what the time stepper already does.
  - This is what GS2 (nadv) and GKW (naverage, ln λ/(nΔt)) do [V].
  - **Caveat:** implicit-step distortion. Refine with a few Rayleigh-quotient or JD steps on the exact L.
- **Rational Krylov [M].** Ruhe, "Rational Krylov: a practical algorithm for large sparse nonsymmetric matrix pencils", SISC 19 (1998) 1535. Multiple shifts in one subspace, so separate shifts near ion and electron branches can coexist.
- **Harmonic Ritz / harmonic Krylov–Schur [V, Roman 2010; GKW; GS2].** Target κ, no linear solves. At least 5× faster than SI in GENE tests, and the default "first try" in GKW/GS2 eigen namelists.
- **How codes get dominant γ, summary.**

  | Code | Initial value | Eigen |
  |---|---|---|
  | GENE | RK4 | SLEPc harmonic Krylov–Schur, or JD + ASM/ILU on (L_g − σI) [V] |
  | GS2 | implicit | SLEPc on the time-advance map with nadv steps; Krylov–Schur default [V] |
  | GKW | RK4 | SLEPc on the (1+ΔtN)^n map, Ritz or harmonic [V] |
  | stella | IMEX | no eigensolver found [M: I found none] |
  | CGYRO | RK4/adaptive | no SLEPc eigensolver found [M]; I believe a separate linear eigenvalue mode exists, not verified |

## Synthesis for GKX (my inferences, not from the sources)

1. **The ADI preconditioner fails on KE for an expected reason.** A single shift cannot match two streaming scales; separating field and f solves is also unstable (Belli & Hammett 2005).
   - Replacement: reuse the IMEX per-chain block-Thomas + Woodbury solve of (σ − L_stream − L_field) as the preconditioner. It is exact for streaming + fields for all species at once, with the species block structure handled by the Thomas recursion over Hermite.
   - Treat drifts/drive as the unpreconditioned remainder in GMRES/BiCGStab, inside SI or JD.
   - This mirrors GENE's K ≈ (L_g − σI)⁻¹ with σ fixed, but better, since GKX's block includes the field response.
2. **Cheapest robust alternative: time-stepper Arnoldi** on the IMEX ARS(3,4,3) step map, GS2/GKW style, followed by JD refinement on L. Or use harmonic Krylov–Schur with no solves.
3. **For adaptive Δt in IMEX,** use an ESDIRK-based ARK pair (KC ARK3(2)4L[2]SA shares γ with ARS343). Refactor chain blocks only when γΔt changes > 20% (ARKODE rule), or quantise Δt.
4. **Screen tableaux with the coupled (H-stability) amplification matrix** on real GKX operators. Scalar regions do not explain ARS(2,3,2): its explicit part has RK3-like imaginary-axis extent √3 by my algebra.
5. **Treat Hermite hypercollisions (diagonal) in the implicit part.** At small β, keep a finite β (≈1e-5, as GX and GENE do) or include A∥ implicitly so ω_H is bounded.
6. **Lawson-RK with e^{τ L_chain}** is the exponential option least prone to imaginary-axis instability (Crouseilles et al. 2020).

## Could not verify
- A stella EM journal paper (2024/25) attributed to Acton. Only the Davies 2022 thesis is verified.
- CGYRO 2016 paper internals: whether cmat includes the field response, and the exact CN form.
- GS2 and stella Δt-adaptation internals (delt_adj, cfl_cushion).
- Exact page or DOI for: Doerk & Jenko CPC 2014; GX JPP; KRT 1995; Lock et al. 2014; Freitag & Spence; Simoncini; Ruhe; Tuckerman & Barkley.
- Imaginary-axis extent of the KC ARK324 ERK part.
- Whether GENE has any implicit time integrator.
