# Solver, velocity-convergence and throughput review (2026-09-13)

Independent review of the plan state at #227 (`3fb7d5c35`, stacked on #226
`1e11faf7a`; main `52b8dd693`). Read-only on the repository: no source, test,
default or reference changed. Evidence scripts and their logs are in
[scripts/2026-09-13](scripts/2026-09-13/); every number below comes from one of
them or from a cited source. Environment: fresh venv Python 3.11.14, JAX/jaxlib
0.10.2, NumPy 2.4.6, SciPy 1.17.1, SOLVAX 0.20.0; Apple M3 Max CPU, complex128,
one XLA thread, `nice 10`. The machine carried load 8–60 from unrelated jobs,
so **wall-clock times are indicative only**; iteration counts, residuals,
sparsity, factor fill, HLO op counts and spectra are load-independent. Office
GPUs were not used; one read-only NetCDF re-read ran on the office CPU.

## 1. Verdicts

1. **The shift-invert rejections in the log are explained**, on the exact
   assembled operator of the log's own Ny16/ky=+.3 pilot (n=4096):
   budget below need with restart loss; a preconditioner that averages the
   drift along z; 37% of the unknowns exactly decoupled and undamped; and the
   exact sparse route that already exists never run as a control (§3.1–3.3).
2. **Preconditioning and recycling have large measured headroom** (§3.4), but
   the current Hermite-line preconditioner degrades with Nz·nLinks, Nl and Nm
   faster than the drift alone explains (§3.5): at (Nz,Nl,Nm)=(48,8,16) it no
   longer reaches 1e-5 in 400 unrestarted iterations.
3. **The Nl24→32 growth change is a property of the truncated eigenmode, not a
   fit or transient artifact** (§4): γ(t) is stationary from t≈100 in both GX
   runs, φ(θ) overlap between the runs is 0.991, and the Laguerre free-energy
   spectrum is a stationary plateau (≈1–2% per index from ℓ≈6 to the cutoff)
   with no sink. Large-b truncation of the gyroaverage is refuted for this deck
   (b_max=12.7; Nl=16 already captures Γ0 to 4e-6).
4. **The nonlinear step's largest lever is structural** (§5): the state carries
   both ky halves and rebuilds the negative half in the bracket and again after
   every stage although the RHS output is already Hermitian-complete
   (idempotence error exactly 0). The unrolled per-chain-class linked FFTs
   issue 45 FFT ops per RHS at 32×32×24 (9 chain classes at 96×96×48).
5. **Process**: the evidence discipline is good, but fixed-budget pass/fail
   pilots on one configuration consumed days that an exact-matrix analysis at
   n=4096 settles in minutes. Cheapest decisive instrument first.

## 2. Assessment of the collaborators' conclusions

| Claim or decision (plan/log) | Assessment |
|---|---|
| Periodic kz hypercollisions must be FFT‖kz‖IFFT; sharded route too (#226) | Agree; correct and tested. #226 is CLEAN. |
| Precision is not the sole cause | Agree, but it was never a plausible sole cause of a 1e-4 vs 1e-5 miss. |
| Restart loss matters; no default promotion | Restart 20 is below the Krylov dimension this system needs (90); a fixed 60-iteration cap cannot distinguish "slow" from "stagnating". Keep defaults until §6 L4–L5, but stop treating fixed-budget rejection as preconditioner evidence. |
| Next: streaming-only vs full-RHS preconditioner defect | Answered (§3.2). |
| `batched`/`incremental`/`flexible` are one algorithm; inverse factory discards convergence | Agree (verified); same in `_implicit_gmres_step`, which also passes `implicit_maxiter` as `max_restarts`. |
| ky .3 on Ny12 selects −.3 (Nyquist) | Agree; the Nyquist row also hosts the neutral decoupled band (§3.3). |
| GX shares the Nl24→32 change | Agree. |
| Investigate large-b Laguerre truncation / absorber | Refuted for this deck (§4.1); absorber possible but secondary. |
| Nonlinear targets: FFT pipeline, scatter removal in linked paths | Right direction, wrong order: completion/layout first (§5). |
| Sharding species×Hermite only; whole-state parked | Agree. |

## 3. Linear eigensolver evidence

Case (all of §3): the log's Ny16 signed-mode pilot — linked Cyclone deck,
Nx8/Ny16/Nz16, Nl4/Nm8, ntheta16/nperiod1/jtwist1, rate .1, ky=+.3,
σ=.09302951−.28199404j, normalized complex128 seed; n=4096.

### 3.1 Exact operator (`d1_preconditioner_defect.py`, `d1b_neutral_modes.py`)

SOLVAX `sparse_operator_matrix` probing (64 columns per batch) gives
nnz=133,598; SuperLU of A−σI: fill 603,873, factor ≈0.08 s. Certified target
**λ0 = +0.115621−0.241179j** (residual 1.3e-15); next eigenvalues ±0.285794j,
±0.277311j, … with **Re=0**; |λ0−σ|=0.0467, |λ1−σ|=0.0931 (separation 2.0).

The linked chains cover kx rows {0,1,2,6,7}; rows {3,4,5} (1536 of 4096
unknowns) have **zero coupling** to the covered block in either direction,
zero real diagonal and only drift couplings; every Re=0 eigenvalue above has
weight 1.000 on them, the target 1e-31. The Hermite-line linked solve returns
zeros on those rows, so inner solves never excite them, but the outer
Rayleigh quotient, masks and any other preconditioner see them.

### 3.2 Which omitted physics costs (unrestarted GMRES, right preconditioning, iterations to 1e-5)

| Operator terms | none | damping | hermite-line | field-corrected |
|---|---:|---:|---:|---:|
| full | — (1.7e-2 @200) | — (1.65e-1 @200) | **90** (1e-8 @110) | 89 |
| no end damping | — | — | 93 | 92 |
| no mirror | — | — | 83 | 82 |
| no diamagnetic drive | — | — | 85 | 84 |
| no mirror/drive/end damping | — | — | 80 | 79 |
| same, drifts off | 135 | — | **7** | **1** |

The `damping` preconditioner is worse than none. `field-corrected` buys one
iteration over Hermite-line at several times the apply cost unless drifts are
absent. The z-averaged drift is the dominant defect at this size; the
principal-part implementation is correct.

### 3.3 Budget and restart (`d5_bakeoff.py`, log restart control)

Hermite-line reaches 1e-5 in 90 unrestarted iterations and 1e-8 in 110;
GMRES(20) does not reach 1e-5 in 300. The log's restart-60 residual (.007999
at 60 iterations) matches the unrestarted curve (9.7e-3 at 60). The pilot's
60-iteration cap could not converge.

### 3.4 Headroom (`d5_bakeoff.py`; host SciPy instruments, not production code)

| Right preconditioner | its to 1e-5 | GMRES(20) | factor nnz |
|---|---:|---:|---:|
| none | — (6.7e-5 @300) | — | 0 |
| hermite-line (GKX) | 90 | — | — |
| exact LU per Laguerre index | 33 | 57 | 177,080 |
| exact LU per Hermite index | — | — | 56,489 |
| ILU(τ=1e-2, fill 3) of the exact operator | **6** | 6 | 325,491 |
| ILU(τ=1e-4, fill 10) | 2 | 2 | 547,810 |
| exact LU | 1 | 1 | 603,873 |

Twelve outer shift-invert RHSs (exact-LU Arnoldi vectors), Hermite-line,
rtol 1e-5, 400-iteration cap: SOLVAX `gmres(restart=20)` cold converges
**2/12** (4798 iterations); `gcrot(m=20,k=10,recycle_strategy="fifo")` 7/12
(4344); **`gcrot(...,"harmonic")` 12/12 in 2233** — SOLVAX already has
GCRO-DR deflated restarting.

### 3.5 Size ladder (`d6_ladder.py`; Nx=1 single chain unless noted; same σ)

| (Nz, Nl, Nm) | n | nnz/row | LU fill ratio | LU (s, loaded) | hermite-line its to 1e-5 | drifts off |
|---|---:|---:|---:|---:|---:|---:|
| (16,4,8) | 512 | 48 | 5.2 | 0.01 | 26 | 22 |
| (32,4,8) | 1024 | 92 | 5.4 | 0.08 | 24 | 21 |
| (32,8,16) | 4096 | 105 | 10.5 | 1.4 | 159 | 104 |
| (48,8,16) nperiod2 | 6144 | 156 | 9.8 | 3.5 | — (1.6e-1 @400) | 216 |
| (64,8,32) | 16384 | 200 | 11.7 | 19 | — (5.7e-3 @400) | 225 |

Reading: nnz/row grows ≈ Nz·nLinks (spectral streaming is dense along the
chain), LU fill ratio ≈10–12 and factor time ≈n^1.7; at a production chain
(Nl16·Nm48·Nz96·3 ≈ 2e5 unknowns, ≈300 nnz/row) an exact factor is ≈1e9
entries — not viable without replacing spectral streaming by a banded
approximation inside the preconditioner. The drift is dominant at the pilot
size, but with Nl=8 the mirror (l↔m) and field couplings the preconditioner
omits also matter (drifts-off counts 104–225). Note the fixed σ is no longer
near the target at larger rungs (nearest eigenvalues have Re≈−0.01), so these
counts are conservative for a well-placed shift.

The sixth rung (Nx4, multi-link chains) did not finish within the script's
2400 s alarm on the loaded machine and is not reported.

### 3.6 What other codes do (fetched 2026-09-13)

- **stella** (arXiv 1806.02162 §5.3): implicit streaming with a compact
  two-point stencil; fields by Green's functions — one unit impulse of φ per
  point of the extended z grid gives the response δg/δφ, then
  (I − Q Σ δg/δφ) φ = φ_inh is solved "via LU decomposition and
  back-substitution"; the zonal chain needs two solves. Setup per chain:
  N_{z+} bidiagonal sweeps over velocity space plus an O(N_{z+}³) LU; per
  step one bidiagonal solve, one back-substitution, one bidiagonal solve.
  Max stable Δt ≈100× explicit at kyρ≈1.
- **GS2**: Kotschenreuther's implicit algorithm with response matrices
  obtained from delta-function sources, N_θ×N_θ inversions per (kx,ky),
  "velocity-space integrals in the field solve … the bottleneck"; the SLEPc
  eigensolver acts on the implicit time-step operator (no inner solves).
- **MGK** (arXiv 2608.17418): the same structure as an eigenproblem —
  per-orbit blocks factored independently, a field Schur complement, then
  shift-invert Arnoldi; 0.01–0.1 s per mode; spectral (dense) in θ only
  because the blocks are ≤241 wide.
- **GX**: streaming spectral in z, explicit RK; IMEX "future work"; a 2024
  CSGF abstract (Kim) reports a third-order ARK with implicit electron
  streaming as a Hermite-tridiagonal system per kz for uniform B, giving up
  to 60× larger stable steps; inhomogeneous B "preliminary"; no public
  branch. GENE and CGYRO keep streaming explicit.
- **GENE eigensolvers** (Roman, Kammerer, Merz & Jenko, ParCo 2010): matrix-
  free Krylov–Schur; inexact shift-invert with unpreconditioned inner GMRES
  (restart 12 best, TFQMR fastest, 667–1181 s), residual on A growing
  linearly with σ; **harmonic Krylov–Schur "at least five times faster"** with
  no inner solves. Merz et al. CPC 2012 add Jacobi–Davidson with a
  preconditioner from an explicit sparse low-order matrix of the g-part and
  subspace recycling across scans. Freitag–Spence: with a fixed
  preconditioner the inner cost grows as the eigenvector converges; the tuned
  update P_i = P + (A−P)X_iX_iᴴ keeps it flat; inner tolerance ∝ ‖r_i‖
  recovers the direct-solve rate. GCRO-DR (Parks et al. 2006) is what SOLVAX's
  `gcrot("harmonic")` implements.
- **Sparse direct on the spectral-streaming block**: each (ℓ,m) row is a
  z-clique, so LU fills the (m,z) block per ℓ: ≈N_ℓ(N_m N_{z+})³ flops and
  tens of GB per production chain (arithmetic, not measured) — consistent
  with the ladder's fill growth; cuDSS does not change the fill. With FD
  streaming of half-bandwidth p the per-ℓ block is block-tridiagonal in z
  with N_m×N_m tridiagonal blocks: block-Thomas O(N_z N_m³) per ℓ per chain,
  and the field response reuses the factor. Classical FD preconditioning of
  spectral operators: Orszag JCP 37 (1980); Canuto & Quarteroni JCP 60
  (1985). Caution: a centred-FD symbol for a first derivative mismatches the
  Fourier symbol near the Nyquist kz (kh/sin kh), so use upwind/compact or
  accept the mismatch at the highest kz.
- **Half-spectrum storage**: GX `Nyc = 1 + Ny/2`, "k_y ≥ 0 Fourier modes,
  with the k_y < 0 modes determined by the reality condition"; GS2 and
  stella "use the reality condition to limit the simulated k-domain so that
  k_y ≥ 0"; gyaradax, GANDALF, jax-cfd and exponax all use real-to-complex
  transforms with the last axis N/2+1. GKX is the outlier.
- **Cheapest first candidate for L4** (from the above, not yet measured): an
  operator-split preconditioner P = (I−τS)(I−τD) with S the existing exact
  kz/Hermite-line streaming solve and D a z-local block solve of the exact
  ω_d(z) drift with its m±2/ℓ±1 couplings plus damping (dense (N_ℓN_m)²
  blocks per z, batched LU once per shift). Splitting error O(τ²[S,D]) is
  irrelevant for a preconditioner; it keeps spectral streaming and touches no
  layout. Only if it falls short does the FD/response-matrix structure (which
  also yields the IMEX electron step) become necessary.

### 3.7 Source facts (at `3fb7d5c35`)

`_shift_invert_apply_factory` (`solvers_linear_krylov_algorithms.py:350–400`)
returns only `x`; `gmres_solve_method` is validated and static but unused;
defaults `shift_tol=1e-4, shift_maxiter=50, shift_restart=20`
(`solvers_linear_krylov.py:60–62`); the outer loop restarts from one Ritz
vector (`:302–347, :796–860`). Preconditioner data
(`solvers_linear_implicit.py:135–193, 249–253, 290–327`): damping plus the
diagonal (2m+1),(2ℓ+1) drift, z-averaged; no m±2/ℓ±1 drift couplings, mirror,
drive, end damping or fields (field-corrected adds a dense capacitance
correction). `_sparse_shift_invert_branch` (`solvers_linear_krylov.py:486–566`)
is the exact route. `_implicit_gmres_step` (`solvers_linear_implicit.py:781–819`)
passes `implicit_maxiter` (200) as `max_restarts` × restart 20. Pins disagree:
`requirements.txt:7 solvax>=0.7.3,<0.8` vs `pyproject.toml:33 solvax>=0.12.0`.

## 4. Velocity convergence evidence

### 4.1 Gyroaverage truncation refuted for the GX control (`d3_bmap.py`)

GX deck (office `full96.in`, read-only): s-α, ε=.18, q=1.4, ŝ=.8, ntheta32,
nperiod2, **nkx=1**, y0=1.81818 (ky=.55), Nm96, vnewk=0, `hypercollisions=true`,
absorber .1/widthfrac .125. GKX's cache for the same geometry: b(θ)=0.42 at 0,
11.1 at 2π, **b_max=12.73** at θ=−2.31π. 1−Σ_{ℓ<N}J_ℓ²/Γ0 along the chain:
N=8 0.21, **N=16 3.7e-6**, N=24 3e-14. Without the bmag factor b_max≈17.5 and
N=16 still captures Γ0 to 1.1e-3. GX pristine `bc2fe552` applies `nu_hyper_l`
only in the const-coefficient kernel (`linear.cu`, hypercollisions block);
the kz branch uses `nu_hyper_m` only; both branches read the same
`nu_hyper_m`, so a matched Laguerre-sink test must zero the const branch's
Hermite coefficient. GKX mirrors this (`cache_arrays.py:154–169`).

### 4.2 Re-read of the existing GX outputs (`v1_reread.py`, office CPU, no new run)

| | Nl24 | Nl32 | Nl32, Nz192 |
|---|---:|---:|---:|
| γ fit [100,200] / [200,300] / [210,300] | .032847 / .033009 / .033009 | .024910 / .024858 / .024854 | .025001 / .024944 / .024940 |
| instantaneous γ at t=100…300 | .03301 … .03301 | .02363 → .02487 | .02375 → .02496 |
| upper-quarter Laguerre fraction, t=50…300 | .076 → .0796 (flat from t≈100) | .082 → .0817 | .082 → .0813 |
| P(ℓ=Nl−1, m≤2)/total; P(ℓ=Nl−1, all m)/total | 5.5e-4; 2.1e-3 | 3.3e-4; 1.3e-3 | 3.2e-4; 1.3e-3 |
| corner/00 | 3.2e-7 | 3.0e-7 | 2.3e-7 |
| upper-quarter Hermite fraction | .022 | .022 | .021 |
| ‖φ‖²(θ)/max at ±π, ±2π, ±3π | 1.1e-2, 4.5e-4, 1.3e-9 | 8e-3, 3.4e-4, 9e-10 | 8e-3, 3.4e-4, 3e-12 |

Final W(ℓ)/ΣW, Nl24: `.34 .14 .08 .05 .05 .05 .03 .02 .02 .02 .02 .02 .02 .02 .01 .01 .01 .01 .02 .02 .02 .01 .01 .00`;
Nl32: `.31 .13 .07 .05 .05 .04 .02 .02 .02 .03 .03 .02 .01 .01 .01 .01 .02 .02 .02 .01 .01 .01 .00 .01 .01 .01 .02 .02 .01 .01 .00 .00`.
Complex overlap |⟨φ24|φ32⟩| at t=300: **0.991**; parity |⟨φ(θ)|φ(−θ)⟩| 0.99.

Reading: same mode (φ overlap 0.99, ω 0.500 vs 0.505), stationary growth
(no finite-time bias), z-converged (Nz192 changes γ by +0.4%), and a Laguerre
spectrum that does not decay with ℓ — a plateau of ≈1–2% per index modulated
with period ≈ 6–8 up to the cutoff, in both runs. The truncation therefore
removes O(10%) of the free energy and the eigenvalue depends on where it is
cut. This is the "plateau then sharp drop" Laguerre shape Hoffmann, Frei &
Ricci 2023 report, at an amplitude that is not small.

### 4.3 Mechanism from the literature

Mandell, Dorland & Landreman 2018 (App. C, eq. C3): the curvature drift
couples m↔m±2, the **∇B drift couples ℓ↔ℓ±1 with coefficients ∝ ℓ** —
multiplication by μB, i.e. phase mixing in μ, the Laguerre analogue of
Hermite phase mixing by streaming. Frei, Hoffmann & Ricci 2022: "magnetic
drifts broaden the gyromoment spectrum significantly in the absence of
collisions". GX's "Laguerre hypercollisions unnecessary" was argued at Nℓ≈4
for nonlinear CBC; its *linear* CBC benchmark (GX App. G.1) used Dougherty
ν_ii=1e-2 without hypercollisions, so a collisionless Nℓ 24–32 regime was
never validated by GX. Barnes, Dorland & Tatsuno 2010 (PoP 17, 032106):
collisionless runs develop unresolved velocity structure and damping is
"artificially terminated"; ν≪γ regularizes it. A resonant factor
1/(ω−ω_d μB) has Laguerre amplitudes e-folding at ℓ≈2a², a≈ω_d/γ, so the
required Nℓ scales like (ω_d/γ)² (estimate, not a published result); with
γ≈0.03 and ω_d growing toward |θ|≈3π, Nℓ>32 is plausible. Truncation turns
the drift-resonance branch cut (Kuroda et al. 1998; Rodríguez & Zocco 2024,
arXiv 2405.19235) into discrete eigenvalues that densify with Nℓ.
Remedies in use: Laguerre hypercollisions ν_ℓ(ℓ/Nℓ)^6 (GX const branch);
−η_v(p+2j) with η_v=1e-3–1e-2 (Hoffmann, Frei & Ricci 2023, arXiv 2308.01016);
small Dougherty ν with ν→0 extrapolation.

Checked against current upstream GX (Bitbucket `gyrokinetics/gx`, branch
`gx`, head `3865a5377886`, 2026-08-11; `next` differs only by a shared-memory
attribute): `src/linear.cu:214–232` and `src/device_funcs.cu:3367` are
unchanged from `bc2fe552` in these paths — the `hypercollisions_kz` kernel
has no Laguerre index at all (`if (m>2) res = -nu m^p g`), only the const
kernel carries `nl·nu_hyper_l (l/nl)^p_hyper_l` for `m>2 || l>1`; defaults
`nu_hyper_l=0.0, nu_hyper_m=1.0, p_hyper_l=6, p_hyper_m=min(20,nm/2)`,
`hypercollisions=true` ⇒ kz branch. GX's docs (`docs/Inputs.rst`,
readthedocs) still list `nu_hyper_l=0.5` and describe Laguerre
hypercollisions "at grid scales in μB" — stale relative to the code. The
shipped benchmark deck `itg_salpha_adiabatic_electrons.in` is collisionless
(`vnewk=0`, `hypercollisions=true`, Nl16/Nm48), unlike the paper's Fig. 1
(ν_ii=1e-2, no hypercollisions); its reference output gives γ=0.0346,
ω=0.498 v_ti/a at kyρ=0.55 (γ peaks at 0.093 near ky=0.3), so the Nl
sequence 16/24/32 → 0.0346/0.0330/0.0249 is monotone and accelerating, and
γ≈0.033 v_ti/a is the expected magnitude (the "CBC γ≈0.2–0.3" figure is in
c_s/R; ×2.78 here). GYACOMO's code comments call GX-like Hermite/Laguerre
hypercollisions "unadvised" and its CBC decks use Dougherty ν=0.05 (paper
linear scans ν=1e-3, J=P/2) — small physical collisions rather than
Laguerre hyperdiffusion. Frei, Hoffmann & Ricci 2023: "FOW and FLR effects
require a large number of Laguerre GMs"; "magnetic gradient drifts broaden
the GM spectrum (both Hermite and Laguerre moments)"; agreement with GENE at
(P,J)≳(32,16) with ν_ii=1e-4.

## 5. Nonlinear step evidence (`d7_hlo.py`, load-independent)

Cyclone nonlinear deck at 32×32×24, Nl2/Nm4, rk3, compressed real FFT,
Laguerre grid mode; state 1.6 MB complex64:

| jitted function | fft | concatenate | gather | transpose | copy | while | bytes written by concatenate/copy ops |
|---|---:|---:|---:|---:|---:|---:|---:|
| nonlinear RHS | **45** | 9 | 31 | 78 | 55 | 0 | 88 MB (55× state) |
| one RK3 step, `return_fields=False` | 135 | 31 | 113 | 252 | 178 | 2949 | 297 MB (185× state) |

Projector idempotence on the RHS output: ‖P(rhs(PG))−rhs(PG)‖/‖rhs‖ = **0**
and 0 on G+dt·dG — Hermitian completion after every stage
(`projection.py:169–172`, `solvers_nonlinear_explicit.py:52–72,353`) is exact
redundant work. Linked chain classes: 5 at 32×32×24, 7 at 64×64×24, 9 at
96×96×48 (`(nChains,nLinks)` from (1452,1) to (3,13)); each class issues its
own gather/FFT/IFFT/scatter in streaming and again in hypercollisions, which
is where 45 FFT ops per RHS come from. The repository's XLA profile
(`docs/performance.rst`) already attributes 41.9% of step time to four
`_complete_hermitian_ky` concatenations and ~39% to FFTs. GX stores only
`Nyc=1+Ny/2` rows (`grids.cu:12`). The `integrate_nonlinear_from_config`
route (runtime "diagnostics disabled") pays a fourth full RHS per RK3 step for
fields it only needs at the end (`solvers_nonlinear_explicit.py:367–378`,
`workflows/runtime/results.py`); the chunked run-to-saturation route forces
per-step diagnostics (`workflows/nonlinear.py:394–395`) and re-concatenates
all previous chunks for the stop test (`workflows/runtime/chunks.py:295–301`).

Literature and runtime facts (fetched 2026-09-13):

- gyaradax (arXiv 2604.06085, Table 2, one B300): JAX double precision 12.49
  steps/s, JAX mixed precision 30.89 (2.5×; FFTs, derivatives and IFFTs in
  f32, linear terms/field solve/final FFT in f64), custom CUDA 60.54; packing
  both derivatives of a field into one complex transform (Z2Z) halves inverse
  FFTs but exposed a ky=0 Hermitian-symmetry defect up to 14% from the
  gyroaverage, cured by averaging each (kx,−kx) pair at ky=0 (error 1e-2 →
  1e-6 in f32); cuFFT "acts as a hard fusion barrier". Derived cost: ≈7 ns
  (f64) and ≈3 ns (mixed) per element-step on a B300, ≈15.6 ns for GKW on 64
  EPYC cores, versus GKX's 196 ns on 3 M3 cores; hardware differs, so these
  calibrate the shape of the gap, not a target.
- `jnp.fft` over non-trailing axes inserts a `moveaxis` before and after
  `lax.fft` ("XLA only supports FFTs over the innermost axes",
  `jax/_src/numpy/fft.py`); GKX's `(…, Ny, Nx, Nz)` layout with transforms
  on `(Ny, Nx)` pays two transposes per perpendicular transform — consistent
  with the 78 `transpose` ops per RHS above.
- XLA:CPU FFTs are DUCC `FftThunk`s that receive the intra-op thread pool
  only when `xla_cpu_multi_thread_eigen` is true (`fft_thunk.cc`); the thunk
  runtime dropped that pool in jaxlib 0.4.32 (jax#25808, 3–4× `fftn`
  regression, fixed by openxla/xla#21900, first in jax 0.5.1). GKX's floor
  jax≥0.10.1 includes the fix; the review's own runs disabled the pool on
  purpose for reproducibility, so the profiled "3 of 14 cores" figure in
  `docs/performance.rst` should be re-measured with the pool on (N0).
- `concatenate` always materializes on both backends; `dynamic-update-slice`
  on a dead operand can be in place (`CanShareOperandBufferWithUser`), so
  `.at[].set` is preferable where a write cannot be avoided.
- `jnp.fft` has no SPMD partitioner; sharding a transform axis all-gathers
  (`spmd_partitioner.cc`), `custom_partitioning` still lacks differentiation
  (jax#29954), and jaxDecomp publishes no PCIe results — FFT axes must never
  be sharded on the two-A4000 box.
- GENE's GPU nonlinearity spends ≈30% in transposes and dealias copies
  (arXiv 1310.1485): movement is expected in a pseudo-spectral step, but it
  is minimized by layout, not by scatter tweaks.
- Single precision "can be used effectively for the entirety" of DNS in four
  codes (arXiv 2506.05150); Dedalus adjoints use H-Revolve schedules
  (arXiv 2506.14792); nested `jax.checkpoint` gives O(log D) memory for
  O(log D)× recompute — the binomial policy to compare against GKX's block
  checkpointing (1.77–1.92× runtime) for 10⁴-step windows.

## 6. Revised plan (adopted into plan.md §0.5, §5.1, §5.3)

Principle: cheapest decisive instrument first; every experiment reports a
convergence curve or an exact reference, never only a fixed-budget pass/fail;
one numerical change per PR.

**L — linear eigensolver (CPU, ~2 weeks).**
L1 instrumentation: return (true residual, iterations, converged) from every
inner solve and log per outer step; wire or remove `shift_solve_method`;
make `implicit_maxiter` mean iterations. No numerics change.
L2 exact reference ladder with `method="sparse_shift_invert"` (probe count,
nnz, fill, factor/solve time, memory, certified pairs ≤1e-10) on single-ky
atlas decks up to the size where it stops being the fastest certified route.
L3 restrict Krylov vectors, Rayleigh quotients and masks to linked-covered
rows (or project the decoupled rows out); gate λ0 to 1e-12 and no Re=0
uncovered mode selectable; decide whether time integration should evolve
those rows at all.
L4 preconditioner bake-off on the exact-operator harness: (a) current
Hermite-line; (a′) the operator-split P=(I−τS)(I−τD) of §3.6 — exact
streaming line solve times a z-local exact-drift block solve — as the
cheapest candidate that keeps spectral streaming; (b) low-order physics
preconditioner assembled per chain —
banded FD streaming in z, full z-dependent drift with m±2/ℓ±1 couplings,
mirror, hypercollisions, end damping, fields by a small per-(kx,z) Schur
complement — factored by banded LU or ILU(τ) (GENE 2012 / stella pattern);
(c) exact per-ℓ block (33) and (d) ILU of the probed operator (6) as
ceilings. Metrics: iterations to 1e-5/1e-8 unrestarted and at restart 20,
setup matvec-equivalents, apply cost, memory, trend on the L2 ladder. Adopt at
≥3× fewer total matvec-equivalents to a certified pair at the largest rung,
setup included, CPU and GPU.
L5 solver structure: inner `gcrot` harmonic recycling carried across outer
RHSs plus a tolerance schedule (tight for the first Krylov block, then
ε/‖r_outer‖); outer thick-restart/Krylov–Schur with harmonic extraction
(implement in SOLVAX). Compare exact-LU shift-invert, propagator/exponential
Krylov (no inner solves; GS2's approach) and matrix-free shift-invert with
the L4 winner by time-to-certified-pair; choose defaults per size regime.
L6 shift policy: report |λ1−σ|/|λ0−σ| (2.0 here); place σ beyond the target
on the growth side; reuse factorizations across nearby σ. The same L4(b)
factorization is the candidate implicit-streaming operator for kinetic
electrons (stella's response-matrix pattern), to be tested only after L4.

**V — velocity convergence (parallel with L; CPU first).**
V1 done (§4.2). V2 ablations, eigen-based as soon as L2 makes them
affordable, else short IVP: (i) `gradb=0` then `curvature=0` at Nℓ 24/32 —
μ-phase-mixing predicts the shift vanishes with ∇B off and persists with
curvature off; (ii) eigen-solve at Nℓ 16/24/32/48 — the eigenvalue itself
moving with an ℓ-plateau in its vector confirms truncation; (iii) Laguerre
sink `nu_hyper_l ∈ {0.01, 0.1, 0.5}`, `p_hyper_l=6`, const-branch Hermite
coefficient zeroed — truncation predicts γ(Nℓ) collapses with a plateau in
ν_ℓ; (iv) Dougherty ν ∈ {1e-3, 3e-3, 1e-2} with ν→0 extrapolation (ν·b≈0.13
at b=12.7 exceeds γ, so 1e-2 is not adopted blindly); (v) raise R/L_T:
sensitivity should fall like (γ/ω_d)². V3 only then the three-rung ladder
with the chosen regularization, both codes, one ledger row. If the mechanism
holds, atlas rows for collisionless linear modes must declare their
regularization and report the ν→0 extrapolation.

**N — nonlinear throughput (profile now; GPU when free).**
N0 load-independent ledger per RK stage (HLO counts and bytes of fft,
concatenate, gather, scatter, transpose, copy) plus A/B/A/B timings on a
pinned SHA on an idle machine; gate for every N change.
N1 complete once per step, not per stage, and let the bracket return its
positive half to a single completion (idempotence is exact); target most of
the 41.9%; gate RHS/trajectory identity ≤1e-13 (f64) over 100 steps and VJP
parity.
N2 batch the linked-chain classes (pad chains to a common length or group
classes) so streaming and hypercollisions issue O(1) FFT launches per RHS
instead of 2×classes; gate identity plus FFT-op count.
N3 half-spectrum state (Nyc rows) behind a layout adapter; start after N1
quantifies the residual completion cost.
N4 fewer transforms: pack ∂x/∂y of each operand into one complex transform,
batch G and χ, compute ∇(J0χ) once without the Hermite index; gate transform
count halves, ky=0 symmetry defect ≤1e-6 in f32.
N5 mixed precision bracket (f32 FFT/products, f64 linear/fields/accumulation);
gate time-averaged flux within replicate spread, saturated-window gradient
cosine >0.99 vs f64.
N6 remove avoidable work: field solve instead of a full RHS in the final-state
runtime route; stride diagnostics inside the scan; incremental stop buffers;
fuse the curvature/∇B kernels.
N7 species×Hermite sharding re-measured only after N1–N4: zero
all-gather/all-to-all on 6-D arrays in HLO; adopt at ≤0.8× single-GPU step
time at 96×96×48.

**Housekeeping.** Merge #226 on its green head; retarget #227; resolve the
SOLVAX pin contradiction; keep log entries to decision, command, result,
limitation.

## 7. Limits of this review

One σ and one underresolved pilot for §3.1–3.4; §3.5 uses the pilot's σ at
sizes where it is no longer near the target. ILU/block-LU are host SciPy
instruments on an assembled matrix; assembly and fill at production size are
extrapolated (§3.5), not measured. §4.2 reads GX diagnostics only; GKX's own
W(ℓ) for the same runs should be compared before V2. §5 counts HLO ops at a
small grid on CPU; GPU op counts and the share of time per op are not
measured here.

## 8. Reproduction

From a checkout at `3fb7d5c35` with the environment above:
`env PYTHONPATH=$PWD/src:$PWD MPLBACKEND=Agg JAX_PLATFORMS=cpu JAX_ENABLE_X64=true GKX_X64=1 python plan/research/scripts/2026-09-13/<script>.py`
(`d7_hlo.py` without x64; `v1_reread.py` on the office host with the NetCDF
paths in the file). Logs of the runs used here sit beside the scripts as `*.txt` (the repository ignores `*.log`).
