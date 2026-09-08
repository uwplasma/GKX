# Differentiable gyrokinetics and turbulence-aware stellarator optimization, 2022–2026

Researched 2026-09-06. Every row was fetched from arXiv or the publisher unless
marked UNVERIFIED. The purpose is to locate what GKX can claim that nobody has.

## Landscape

| Code / paper | Year | Method | Demonstrated | Key numbers | URL |
|---|---|---|---|---|---|
| iGENE (Artigues, Merlo, Jenko) | 2026, Phys. Plasmas 33 083901 | TensorFlow AD of local EM nonlinear GK; v∥/μ grid 32×8; s-α/circular/Miller only | windowed reverse-mode gradients of time-averaged Q; Adam flux matching at 1 and 7 radial points | gradients approach FD then diverge at N≳512 steps; at N≈512 gradients for ω_T, ω_n, q reach 15–34% of FD, ε ≈50%; Q autocorrelation 500–1000 steps; 7-point run 1.6×10⁶ steps on 2 GPUs per point; linear gradients match FD except ε, ŝ (30–50% off) | https://arxiv.org/abs/2605.03086 |
| gyaradax (Galletti et al.) | 2026 | JAX/CUDA port of GKW; electrostatic, collisionless; s-α/circular only | linear inverse problem (recover R/L_T from φ); ∂γ/∂(R/L_T); no nonlinear AD gradient | R/L_T=6.908 recovered vs 6.9; 5.4×/10.5× over GKW; no EM, collisions, Miller or VMEC | https://arxiv.org/abs/2604.06085 |
| GANDALF | 2025 | JAX spectral kinetic RMHD, Hermite basis | not GK; AD unused | — | https://arxiv.org/abs/2511.21891 |
| JAX-in-Cell | 2025 | JAX 1D3V PIC | AD demos; no GK | — | https://arxiv.org/abs/2512.12160 |
| Review (Joglekar et al.) | 2026, invited Phys. Plasmas | surrogates / neural operators / PINNs / differentiable simulation | cites ADEPT, DESC, FOCUSADD; no gyrokinetic content | open problems: reverse-mode memory, discrete-vs-continuous gradients | https://arxiv.org/abs/2603.11231 |
| yancc | 2026 | JAX drift-kinetic neoclassical, differentiable | adjoint sensitivities for neoclassics | 1% vs MONKES/SFINCS | https://arxiv.org/abs/2607.20861 |
| DESC bounce averaging (Unalmis et al.) | 2026, JPP 92 E72 | reverse-mode ε_eff, energetic particles, turbulence proxies | first finite-β ε_eff reverse-mode optimization | — | https://arxiv.org/abs/2412.01724 |
| Acton, Barnes et al. | 2024 | adjoint of linear δf GK in stella; Miller, 11 shaping parameters | minimize single-k_y ITG γ | adjoint ≈ two linear solves vs N_p+1; linear only; no stellarator | https://arxiv.org/abs/2403.12621 |
| Gaur et al. | 2025, PPCF 67 125015 | reverse-mode ideal ballooning in DESC | future work: linear adjoint GK | — | https://eprints.whiterose.ac.uk/id/eprint/237029/ |
| Thakur & Nadarajah | 2025 | stabilized-march adjoint shadowing | Lorenz-63, KS only; cost O(n_u²) | — | https://arxiv.org/abs/2505.00838 |
| Wang & Zaki | 2026 | ensemble-averaged adjoint with closure, wall turbulence | individual adjoints grow ∝ Lyapunov; mean adjoint decays | — | https://arxiv.org/abs/2606.25399 |
| Metz et al. | 2021 | "Gradients are not all you need" | chaos failure of unrolled AD | — | https://arxiv.org/abs/2111.05803 |
| Kim et al. | 2024, JPP 90 905900210 | GX+DESC, SPSA on nonlinear Q; QH A=8 nfp=4 | ~3× Q reduction (~2× fluid); +10% core T in T3D; critical gradient unchanged | 2 GX evals/iter, ~3 min each (64³, 8×4), ~20 h V100; ~50% Q variation across α | https://arxiv.org/html/2310.18842v2 |
| Jorge et al. | 2024, PRE 110 035201 | GS2 quasilinear f_Q=Σ γ/⟨k⊥²⟩ in SIMSOPT, LM with FD gradients | precise QH nfp=4, s=0.25, α=0, ten k_y | f_Q 0.192→0.112; f_QS 0.141→0.117; no nonlinear validation | https://arxiv.org/html/2301.09356 |
| Roberg-Clark et al. | 2022/2023/2026 | critical-gradient proxy | QH, compact QH reactor, 6-fp QI | Q ≤ W7-X high mirror | https://arxiv.org/abs/2210.16030 · https://arxiv.org/abs/2301.06773 · https://arxiv.org/abs/2506.22166 |
| Goodman et al. (SQuID) | 2024, PRX Energy 3 023010 | minimize |∇s| in bad curvature | GX-validated; lowest fluxes over the gradient range vs W7-X | — | https://arxiv.org/html/2405.19860 |
| Plunk et al. | 2025/26 | QI max-J with turbulent pinch | GK profile projections | — | https://arxiv.org/abs/2507.19319 |
| Mackenbach et al. | 2022 PRL / 2023 JPP | available energy of trapped electrons | GENE TEM fluxes: DIII-D, HSX, W7-X | Q ∝ AE^(1.5±0.1) | https://arxiv.org/abs/2109.01042 |
| Gerard et al. | 2024 | 563 QH configs | AE correlates only within shaping regimes | — | https://arxiv.org/abs/2404.07322 |
| Landreman et al. | 2025, JPP 91 E120 | 2×100,705 GX flux tubes; adiabatic e⁻; a/L_T=3, a/L_n=0.9 | CNN ensemble R²=0.989 (~4000×); best analytic proxy Spearman 0.788, R²=0.737; XGBoost 3-feature R²=0.887; Q spans >4 decades | data CC-BY 30 GB | https://arxiv.org/html/2502.11657 · https://zenodo.org/record/14867777 |
| GyroSwin / 5D surrogates / GKFieldFlow | 2025–26 | 5D neural surrogates (GKW, CGYRO), tokamak | ~10³× cost reduction | — | https://arxiv.org/abs/2510.07314 · https://arxiv.org/abs/2502.07469 · https://arxiv.org/abs/2601.02614 |
| Wei et al. (GTC) | 2026 | latent space of QH geometries for global GK surrogates | zonal residue vs axis excursion | — | https://arxiv.org/abs/2603.17366 |
| GENE-KNOSOS-Tango | 2025 | flux-driven W7-X validation | 4 scenarios | — | https://arxiv.org/abs/2503.08943 |
| T3D+GX(+KNOSOS) | 2022–26 | stellarator profile prediction | APS abstracts only; paywalled paper UNVERIFIED | — | https://ui.adsabs.harvard.edu/abs/2022APS..DPPBO3006Q/abstract |
| Tesseract (Coughlin et al.) | 2025 | wrap non-differentiable Gkeyll in JAX | Z-pinch | — | https://arxiv.org/abs/2511.13262 |

No 2025–26 nonlinear GK adjoint, and no shadowing/NILSS application to GK, was
found. Every differentiable GK code is tokamak-local.

## Genuinely unclaimed, with the evidence bar and a compute estimate

1. **End-to-end ∂(γ, f_Q)/∂(boundary modes) through VMEX → geometry → GKX in
   stellarator geometry.** Bar: AD–FD agreement on the Jorge 2024 setup (precise
   QH nfp=4, s=0.25, ten k_y) and reproducing f_Q 0.192→0.112 with
   N_p-independent gradient cost; add the α-averaging Kim flagged. Under one
   GPU-day.
2. **Finite-window gradient law in stellarator geometry versus iGENE's 512-step
   tokamak result.** Bar: replicate Kim 2024 (A=8, nfp=4 QH, 64³, 8×4, ~3 min per
   evaluation) and reach the same ~3× reduction in fewer evaluations than "2 per
   iteration, ~20 h V100", with FD checks at three or more window lengths.
   3–6 GPU-days.
3. **Physics-based differentiable surrogate scored on the open Landreman
   dataset.** Compute GKX linear and quasilinear f_Q (and a windowed nonlinear
   estimate) on a 10⁴ subsample and report Spearman/R² against the nonlinear Q
   beside 0.788 (analytic proxy) and 0.989 (CNN). 1–3 GPU-days.
4. **Gradient of the nonlinear threshold.** Kim 2024 saw no change in the
   critical gradient and lists it as future work; Acton notes linear thresholds
   miss the Dimits shift. Windowed AD of Q at 3–4 a/L_T near threshold gives the
   derivative of the Q-versus-gradient curve. 5–10 GPU-days.
5. **Differentiable stellarator profile prediction** (iGENE did a 7-point
   tokamak flux match on 14 GPUs; a quasilinear GKX+VMEX stellarator version is
   unclaimed). Under 2 GPU-days.

## Do not chase

- Long-horizon nonlinear adjoints or shadowing for time-averaged Q: divergence
  beyond ~512 steps (iGENE); stabilized march costs O(n_u²) in the unstable
  dimension; Wang–Zaki needs ensemble-averaged adjoints with closures.
- Out-predicting CNN surrogates (R²=0.989) or 5D field surrogates;
  differentiability earns its place as a physics-consistent, extrapolating
  objective, not as a faster regressor.
- Kinetic-electron EM nonlinear AD at scale (iGENE needed 24 nodes × 8 GPUs).
- Re-deriving QI critical-gradient designs (four IPP papers 2022–2026).

## Comparable targets

- Landreman–Paul precise QH nfp=4 with the Jorge 2024 numbers: claims 1 and 5.
- Kim 2024 QH (A=8, nfp=4; α=0 and πι/4): claims 2 and 4.
- The dataset convention (W7-X-like s, a/L_T=3, a/L_n=0.9, adiabatic
  electrons): claim 3, and the QUASR configurations in the dataset have public
  equilibria.
- Mackenbach TEM set (DIII-D, HSX, W7-X standard and high mirror; GENE) if TEM
  objectives are pursued.
