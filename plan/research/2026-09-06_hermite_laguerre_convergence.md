# Hermite–Laguerre velocity convergence: literature, diagnosis, protocol

Researched 2026-09-06. Motivation: GKX's linear Cyclone ITG growth rate moved
−7.0% from Nl 16→24 and −24.7% from Nl 24→32 (Nm 48–160), which no published
Hermite–Laguerre study shows. Sources were read from the arXiv PDFs; the
mis-cited ID 1708.09200 is not the Mandell paper, which is
[arXiv 1708.04029](https://arxiv.org/abs/1708.04029).

## Published converged resolutions

| Case | Converged (Nl,Nm) or (P,J) | Collisions / hypercollisions | Source |
|---|---|---|---|
| Local-limit linear ITG (Dong 1992 case) | (8,16) agrees with (16,32); spectrum P(7,15)~1e-5, P(15,31)~1e-10 | ν=0.01, truncation, no hypercollisions | [Mandell 2018 §5.1](https://arxiv.org/pdf/1708.04029) |
| Cyclone linear ITG (nonlocal) | needs (16,32); lower gives γ too large at high k_y; spectrum decays "much more gradually" | same | Mandell 2018 §5.2 |
| GX linear CBC, Boltzmann e⁻ | Nl=16, Nm=48, 3×2π segments, Nz=24 each | ν_ii=1e-2 Dougherty, **no hypercollisions** | [GX App. G.1](https://arxiv.org/pdf/2209.06731) |
| GX linear CBC, kinetic e⁻ | Nl=16, Nm=128; TEM branch still imperfect vs stella/GS2 | ν_ee=1e-2, ν_ii=1.65e-4, no hypercollisions | GX App. G.2 |
| GX nonlinear CBC, Boltzmann | (4,6) with hypercollisions; (6,12) without | ν_ii=1e-2; D=0.05, n=4; f_hyp=1, p=Nm/2 | GX Table 1, G.3 |
| GX nonlinear CBC, kinetic e⁻ | (4,16) within 15% of (16,32) | same | GX Table 2, G.4 |
| GX linear W7-X | (8,16), Nz=256 | — | GX G.5 |
| GYACOMO CBC linear ITG, collisionless | (P,J)≳(16,8); optimum P≈2J; ref (60,30); "exponential and non-monotonic" | velocity dissipation η_v | [Hoffmann 2023 §3.1](https://arxiv.org/pdf/2308.01016) |
| GYACOMO Dimits threshold | (P,J)≳(12,6) recovers κ_T≈4 | η_v=0.001 | Hoffmann 2023 §4.1 |
| GYACOMO nonlinear CBC flux | ~16 modes at η_v=0.001; η_v=0.01–0.05 converge faster but 30% off | — | Hoffmann 2023 §3.2 |
| GYACOMO collisional CBC (ν≈0.005) | (8,4)≈(16,8); operator choice immaterial | Dougherty/Sugama/Landau | Hoffmann 2023 §5 |
| GM flux-tube linear ITG adiabatic | (32,16) matches GENE (128,24); coincides with GX at all (P,J) | ν=1e-4 | [Frei 2023 §4.1](https://arxiv.org/pdf/2210.05799) |
| GM TEM at k_y=0.25 | (128,24); sharp trapped–passing gradients | collisionless | Frei 2023 §4.2 |
| GM KBM / MTM | (16,8) / ≳(32,16) | — | Frei 2023 §4.3–4.4 |
| GM collisional TEM scan | (16,8) sufficient; GAM needs (800,16) at small ν | Coulomb/Sugama | Frei 2023 §5.2, §4.5 |
| Z-pinch entropy mode | (4,2)–(20,10) linear; ν=0.01–0.1 shrinks need to (2,1)–(10,5) | Sugama | [Hoffmann 2023b](https://arxiv.org/pdf/2208.01346) |

2024–2026 gyromoment papers found: [2509.15329](https://arxiv.org/abs/2509.15329)
(hot-electron closure), [2603.13123](https://arxiv.org/abs/2603.13123) (LAPD
full-f), [2608.12205](https://arxiv.org/abs/2608.12205) (sheath boundaries). No
nonlinear electromagnetic kinetic-electron flux-tube gyromoment paper appeared
in the sweep (UNVERIFIED that none exists).

## Closures, hypercollisions, spectra

- Truncation closure is justified only where collisions dominate the top
  moments; at low ν it "will generally give poor results" (Mandell 2018 §3.6;
  GX §4.1).
- GX hypercollisions: ∂G/∂t = −ν_hyp m^p G for m>2, ν_hyp = 2.5 f_hyp
  (p+½)/m_max^{p+½} |k∥|v_t, p=Nm/2; none in Laguerre because "phase mixing in
  v∥ dominates" (GX eqs. 4.27–4.28, App. B).

  > **Correction, 2026-09-07, from pristine GX source at `bc2fe552`.** The
  > shipped code uses **2.3**, not 2.5 (`src/linear.cu:232`), normalizes by
  > `M = Nm_glob − 1`, and defaults `p_hyper_m = min(20, Nm/2)`
  > (`src/parameters.cu:185`), not `Nm/2`. GKX matches the first two exactly
  > (`cache_arrays.py:66,94-96`) and differs on the third (fixed 20.0). GX's
  > default branch is `hypercollisions_kz`, GKX's library default is
  > `hypercollisions_const`. See plan.md 0.5.6.
- Recurrence: truncation is a reflecting wall; damping must be smooth in m, e.g.
  −ν(m/Nm)^α ([Parker & Dellar 2015 §3.2.4](https://arxiv.org/pdf/1407.1932));
  T_R ≃ 2√(2P) qR₀/v_T (Frei 2023 eq. 3.3); hypercollisions are the most robust
  choice ([Issan 2024](https://arxiv.org/abs/2412.07073)).
- Spectra: constant-flux C_m ∝ m^{−1/2} with collisional cutoff
  exp[−(2√2/3)(ν/|k|)m^{3/2}], m_c ~ (|k|/ν)^{2/3} ([Kanekar 2015 eq. 4.21](https://arxiv.org/pdf/1403.6257));
  for a linear instability m_γ = (|k∥|v_th/2√2γ)² ([Loureiro 2016 §3](https://arxiv.org/pdf/1505.02649));
  m^{−3/2} and m^{−1/2}k^{−2} regimes ([Adkins 2018](https://arxiv.org/abs/1709.03203)).
  Hoffmann 2023 Fig. 3: Laguerre amplitude plateaus then drops sharply; Hermite
  decays with constant slope and oscillates from the (p±2) drift coupling.
- Kinetic electrons and TEM need many Hermite moments because of the sharp
  trapped–passing structure (Frei 2023 §4.2; GX §6.1.1).

## Why a −25% swing at Nl 24→32 is numerical

At Cyclone b ≲ 10, the Laguerre weights (b/2)^ℓ e^{−b/2}/ℓ! are negligible by
ℓ≈15; GX is accurate at Nl=4 nonlinearly and Frei coincides with GX at every
(P,J). Candidate causes, each with a test that isolates it:

1. ~~**Collisionless deck.**~~ **Refuted, 2026-09-07.** GX's own shipped deck
   `benchmarks/linear/ITG_cyclone/itg_salpha_adiabatic_electrons.in` sets
   `vnewk = 0.0` with `hypercollisions = true` and `closure_model = "none"` —
   collisionless, exactly like GKX's deck. The claim that App. G.1 uses ν=1e-2
   without hypercollisions does not describe the deck that produced the
   reference output, so it cannot be the discrepancy.
2. **Hypercollision defaults.** *Partly refuted, partly confirmed,
   2026-09-07.* The prefactor matches (both 2.3) and so does the normalization
   (`Nm−1`). Two real divergences remain: GKX's `p_hyper_m` is a fixed 20.0
   where GX uses `min(20, Nm/2)` — equal only for Nm ≥ 40, a factor 2.4 in
   top-moment damping at Nm=16 — and GKX's library defaults select the
   constant-coefficient branch where GX selects the kz branch. Neither is
   Nl-dependent at fixed Nm=48, so neither explains the Nl swing; both must be
   fixed before the Nm ladder means anything.
3. **Growth-rate extraction.** The driver notes γ "still 24% high at t=30";
   docs cite a t=7–10 window. Test: running γ(t) per Nl against the Krylov
   eigenvalue; require <1% agreement.
4. **Fixed dt.** dt=0.004663 RK4 regardless of Nl; GX's step shrinks with Nl,Nm.
   Test: dt/2 and dt/4 at Nl=32.
5. **Laguerre top-ℓ convention.** GKX sets J_Nl = −J_{Nl−1}(b/2)/Nl in the top-ℓ
   drive; GX zero-pads and keeps the truncated ΣJ_ℓ² for energy consistency
   (GX eq. 4.25 note). Test: toggle to zero-padding; compare nperiod 1 vs 2.
6. **End-damping strength.** GKX main uses amp=0.1 per unit time; GX uses
   A=0.1/Δt (≈20× stronger at this dt), z_width=L_z/8, converged for AΔt≳0.05.
   Weak absorbers reflect streaming energy back through high moments. Test:
   restore the per-step contract (#197), then scan amplitude and width at Nl=32.
7. **Default hypercollision channel.** `params.py` defaults
   `hypercollisions_const=1.0, kz=0.0`, with `mask_const=(m>2)|(ell>1)` and
   `l_norm=G.shape[1]`, which is Nl-dependent if active; docs claim the opposite
   default. Test: log the effective parameters; assert kz=1, const=0, and zero
   Laguerre hypercollisions.
8. **Top-moment pile-up.** Test: dump W(ℓ), W(m); a rising tail at ℓ→Nl instead
   of plateau-then-drop indicates closure failure or instability.

## Protocol

1. Baseline = GX App. G.1 configuration. Obtain γ two ways (late-window fit and
   eigensolver) and require agreement.
2. Fix dt (verified by halving), Nz and nperiod, end damping at GX's contract,
   the fit window, p=Nm/2 and f_hyp=1.
3. Scan one axis at a time: Nm ∈ {16,24,32,48,64,96} at Nl=16; Nl ∈
   {4,6,8,12,16,24,32} at Nm=48; then the P≈2J diagonal.
4. Decide with: |Δγ| per doubling <2%; corner power P(ℓ,m)/P(0,0) ≤ 1e-5;
   monotone Hermite decay with no tail upturn; ν and f_hyp insensitivity (<1–3%);
   T_R beyond the fit window if collisionless.
5. Accept when all hold on both axes and γ matches GS2/GX within their reported
   agreement. The −25% must vanish under (1)+(2)+(4)+(6), or the isolating test
   names the culprit.
