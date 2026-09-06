# How gyrokinetic codes verify, and what GKX can adopt

Researched 2026-09-06 for the plan synthesis. Every row was checked against the
linked source; items that could not be fetched are marked UNVERIFIED.

## Adoptable benchmark cases with published reference numbers

These need no other code to be run: the reference is in the paper or shipped
with GX's source.

| Case | Source | Quantity | Reference numbers | Geometry / parameters |
|---|---|---|---|---|
| Cyclone nonlinear χᵢ vs R/L_T, Dimits shift | [Dimits et al. 2000](https://w3.pppl.gov/~hammett/ark/1999/cyclone-b26.pdf) §2, §5 | χᵢ(R/L_T), nonlinear threshold | χᵢL_n/(ρᵢ²v_ti) ≃ 15.4[1−6.0 L_T/R]; R/L_T,eff ≃ 6.0 vs linear R/L_T,crit ≃ 4.0 | s-α, q=1.4, ŝ≈0.78–0.80, ε=0.18, R/L_T=6.92, ηᵢ=3.114, adiabatic e⁻ |
| Rosenbluth–Hinton residual | Dimits 2000 §4; [Mandell 2018](https://arxiv.org/abs/1708.04029) eq. 5.2; GS2 CI test | Φ(∞)/Φ(0) | 1/(1+1.6q²/√ε); GS2 variant ε=0.02, q=1.3, k⊥ρᵢ=0.002 expects 0.04659 (tol 0.08) ([GS2 f90](https://gitlab.com/gyrokinetics/gs2/-/raw/master/tests/linear_tests/zonal_flow_residual/zonal_flow_residual.f90)) | analytic |
| Cyclone Miller nonlinear heat flux by (Nl,Nm) | [GX paper](https://arxiv.org/abs/2209.06731) §6.1, Tables 1–2, App. G | Qᵢ/Q_GB | Boltzmann e⁻: (16,32) 7.8±0.8, (8,16) 7.3±0.9, (4,8) 7.2±0.8, (4,6) 7.0±0.9, (4,4) 5.9±0.7, (3,8) 9.1±0.8. Kinetic e⁻: (16,32) Qᵢ=27.8±4.0, Qₑ=8.4±1.2; (4,16) 23.6±3.2 / 7.3±0.9 | Miller R₀/a=2.78, r/a=0.5, q=1.4, ŝ=0.8; a/L_n=0.8, a/L_T=2.49; Nx=192, Ny=64, Nz=24; ν_ii=1e-2; D=0.05, n=4; hypercollisions f=1, p=Nm/2 |
| ITG→KBM β transition | GX §6.1.1 | γ,ω vs β at k_yρᵢ=0.3 | transition ≈ β_ref=1.3% (GX vs GS2); published scan omits δB∥ | Miller CBC, kinetic e⁻ |
| W7-X bean flux tube | [González-Jerez et al. 2022](https://arxiv.org/abs/2107.06060) Tables 1–3 | γ(k_y), ω(k_y); zonal response; Qᵢ(t) | peak γ at k_yρᵢ=2.1; Qᵢ/Q_gB = 2.26 (stella) vs 2.47 (GENE); GX reproduces at (8,16) and (4,8) | high-mirror VMEC, s=0.64, α₀=0; a/L_Ti=3, a/L_n=1; VMEC input in App. A |
| ETG | [Nevins et al. 2007](https://www.osti.gov/servlets/purl/1564634) | χₑ | ⟨χₑ⟩ ≈ 3.0±0.13 (ρₑ/L_T)ρₑv_te; codes 2.4–3.2 | CBC roles swapped, ŝ=0.1, adiabatic ions; L_x=100ρₑ, L_y=64ρₑ |
| Slab-limit ITG vs gyrofluid | Mandell 2018 §5.1 | γ(k_y) | graphical only | q=2, R₀/a=5, a/L_n=1, a/L_T=1.5/2/3, ν=0.01 |
| KAW slab, KBM Miller, Cyclone s-α/Miller adiabatic and kinetic, W7-X | GX source `benchmarks/linear/*` (local checkout bc2fe552) | ω, γ | stored in `*_correct.out.nc`; GX's own gate is γ rel. diff < 1e-3, ω < 5e-3 ([check.py](https://api.bitbucket.org/2.0/repositories/gyrokinetics/gx/src/3865a53778862e1686f414bf6f416339e24887c9/benchmarks/linear/ITG_cyclone/check.py)) | decks shipped with GX |
| Bravenec 2011/2013, Görler 2016 | paywalled | linear ω, nonlinear fluxes, global EM | UNVERIFIED | DIII-D mid-radius |

[Merlo et al. 2025](https://arxiv.org/abs/2508.06116) is GENE-only (W7-X
20181016.037, ρ_tor=0.4, 1536×480×80×48×12; Qᵢ/Q_GB≈0.28, Qₑ/Q_GB≈0.66) and is a
target, not a cheap parity case.

## The open GX stellarator dataset

[Landreman et al. 2025](https://arxiv.org/abs/2502.11657) (JPP 91 E120), data at
[Zenodo 10.5281/zenodo.14867777](https://zenodo.org/records/14867777) (CC-BY 4.0,
30.4 GB; a copy is on the local machine). Verified from the HDF5 and Appendix B:

- 100,705 flux tubes from 23,577 DESC equilibria in five classes: random
  Fourier boundaries (51,075), rotating ellipses on a circle (12,795) and on a
  curve with torsion (12,791), QUASR vacuum (8,235), QUASR with pressure (15,809).
- Two GX nonlinear ES adiabatic-electron simulations per tube: fixed gradients
  a/L_T=3, a/L_n=0.9, and randomly varied gradients. GX commit b88d763 (an
  ancestor of the local bc2fe552). Q_avg, Q_std (time std over t>150), Q(z),
  zonal φ² amplitude.
- Resolution: nx=ny=64, nz=96, nhermite=8, nlaguerre=4, x0=y0=10, periodic in
  all three coordinates, rk3 at 0.9 CFL, t_max=800, ν_ii a/v_i=0.01,
  hypercollisions on, D_hyper=0.05, T_i/T_e=1. About 8 A100-minutes per run;
  2×10⁵ runs in <28,000 GPU-hours.
- Their convergence evidence: 100 random tubes rerun with every resolution
  parameter changed by ×2 or ×10 → R²=0.993 (unstable cases), 0.995 including
  stable; stability classification accuracy 0.994. Periodic vs twist-and-shift on
  100 tubes at matched k_x,max: R²=0.97 on ln Q, stability accuracy 0.95.
- Geometry stored as the seven eik functions on a uniform arclength grid z ∈
  [−37.7, 36.9], dz=0.7854 (96 points, last point excluded): bmag, gbdrift,
  cvdrift, gbdrift0/ŝ, gds2, gds21/ŝ, gds22/ŝ². The stored flux-surface average
  ⟨|∇x|⟩ is reproduced to 1e-6 by the periodic mean-ratio Σ(|∇x|/B)/Σ(1/B).
  gradpar is constant (B·∇z = B), jacobian = 1/B, grho = |∇x| from gds22/ŝ²;
  cvdrift0 is absent (equal to gbdrift0 in vacuum classes, an approximation for
  the finite-pressure classes). GX computes its parallel B-gradient as a
  periodic spectral derivative (`Geometry::calculate_bgrad`), as GKX does.
- GX's launch-dimension clamp is only in `grad_parallel_linked.cu:173`; periodic
  launches use uncapped dimensions, so these references are not affected by it.

Measured GKX cost at this resolution on an Apple M3 Max CPU, main at a99dac89,
runtime nonlinear cyclone deck edited to 64×64×96 with (Nl,Nm)=(4,8), linked
boundary, dt=0.01: 30 steps in 51.6 s and 330 steps in 392.2 s, i.e. 1.14 s per
step after compilation (90 ns per element-step). At a GX-like timestep the
dataset's t=800 horizon is 5–12 CPU-hours per tube; an RTX A4000 should be
10–15× faster, so a 100-tube sample is 40–120 GPU-hours.

## How each code gates its regression suite

| Code | In-repo references | Gate | CI |
|---|---|---|---|
| GX | yes, `benchmarks/{linear,nonlinear}/*_correct.out.nc` + `check.py` | γ < 1e-3, ω < 5e-3 relative | none seen (UNVERIFIED) |
| stella | yes, `AUTOMATIC_TESTS/numerical_tests/test_1…8` EXPECTED_OUTPUT netCDF | `np.allclose(rtol=0, atol=1e-15)` on fields/fluxes, 1e-12 on ω ([test_5a](https://raw.githubusercontent.com/stellaGK/stella/master/AUTOMATIC_TESTS/numerical_tests/test_5_diagnostics/test_5a_diagnostics.py)) | every push/PR, macOS+Ubuntu, 1 and 4 ranks |
| GS2 | golden numbers in Fortran: CBC γ=0.17027768751030398 tol 1e-3; nonlinear CBC Q=90±10 | physics tolerance, fixed random seed | GitLab CI, three compilers, coverage |
| CGYRO | yes, `reg01…reg22` with `out.cgyro.prec` | relative error < 1e-6 on the precision file | expects exactly 21 PASS per push, CPU/OpenACC/OpenMP |
| Gkeyll | regression binaries | built, not compared | build only |
| GENE / GENE-X | not public; GENE-X uses MMS | UNVERIFIED | — |

## pyrokinetics

GX is supported: TOML read/write and `.out.nc` reader for fields, fluxes,
eigenvalues ([gx.py](https://raw.githubusercontent.com/pyro-kinetics/pyrokinetics/main/src/pyrokinetics/gk_code/gx.py)),
Miller only; VMEC raises "not implemented". Golden answers exist for GX, CGYRO,
GENE, GKW, GS2, stella, TGLF. A GKX plugin would inherit input conversion and a
shared linear-output comparison harness; adding VMEC to its `LocalGeometry`
would make it useful for stellarators for every code. IMAS `gyrokinetics_local`
tooling: [imas-gk](https://gitlab.com/gkdb/imas-gk). MGKDB
([arXiv 2609.03132](https://arxiv.org/abs/2609.03132)) holds GENE/CGYRO/TGLF only.

## The standard verification appendix

Every code paper carries γ(k_y), ω(k_y) for CBC against an older code; a
Rosenbluth–Hinton residual; a nonlinear heat-flux time trace; a resolution
appendix listing every N; and a convergence table with a stated criterion (GX:
"within 15%"; GX App. C for the boundary absorber in AΔt).

## Implications used in the plan

1. Two-tier gating: bitwise regression on stored outputs (stella/CGYRO) for
   refactors; physics tolerances (GX's 1e-3/5e-3) for algorithm changes; fixed
   seeds so goldens are deterministic.
2. Ship GX's decks verbatim for the six shipped linear cases and gate against
   GX's own `*_correct.out.nc` at GX's own tolerance.
3. Reproduce GX Table 1 as a CI artifact and assert within the quoted ±σ.
4. Analytic tier: RH residual, Dimits offset-linear fit, KAW slab dispersion,
   Landau damping (already in the README).
5. Publish an Appendix-G equivalent with normalization conversions.
6. Write the pyrokinetics plugin with VMEC support.
7. Statistical, not pointwise, nonlinear parity across many configurations
   (gyaradax's KS/Pearson on spectra; the Landreman dataset for GKX).
8. Keep the per-push physics tier at CGYRO's scale; nightly for W7-X nonlinear.
