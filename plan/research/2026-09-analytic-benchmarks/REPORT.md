# Analytic benchmark specification for GKX (2026-09-28)

Status: research record, nothing promoted. No GKX runs were made. Reference
numbers come from `benchmarks_check.py` (numpy/scipy only). Its output is in
`benchmarks_check.out`. Figure readings are by eye from the printed figures.
Appendix A (`notes_A.md`: KBM and ballooning papers) and Appendix B
(`notes_B.md`: RH, Xiao-Catto, Sugama-Watanabe, Romanelli, BDR, KHD) keep the
per-paper transcriptions. They contain equations and paraphrase only. Where
this file and the notes differ, this file wins; §6 lists the corrections.

## 0. Conventions and the open sign question

- GKX/GX: $v_{ti}=\sqrt{T_i/m_i}$, $\rho_i=v_{ti}/\Omega_i$, `tprim` $=a/L_T$,
  `fprim` $=a/L_n$, `beta` $=8\pi n_{ref}T_{ref}/B_{ref}^2$ (one species' β, not
  the total), ω in $v_{ti}/a$. Input keys used below exist in current decks:
  `[geometry] model = "s-alpha"|"miller"`, `q`, `s_hat`, `epsilon`, `R0`,
  `alpha`, `shift`; `[physics] use_apar`, `use_bpar`, `beta`; `[grid] nperiod`.
- Papers using $v_t=\sqrt{2T/m}$: Sugama-Watanabe (SW), Romanelli, Biglari-Diamond-Rosenbluth
  (BDR), Tang-Connor-Hastie, and Aleynikova-Zocco (AZ). AZ's $\omega_{*i}=\tfrac12k_y\rho_iv_{th}/L_n$
  with $\rho_i=v_{th}/\Omega$ equals the GX $k_y\rho_iv_{ti}/L_n$.
  Papers using $\sqrt{T/m}$: Kim-Horton-Dong (KHD), Hastie-Hesketh ($a_i^2=T/m\omega_c^2$), RH ($a_i$),
  and GENE's $c_s$/$\rho_s$ at $T_i=T_e$.
  Conversion: $\omega[v_{ti}^{GX}/R]=\sqrt2\,\omega[v_{Ti}/R]$ and $k\rho^{GX}=k\rho^{(\sqrt2)}/\sqrt2$.
- τ differs: Romanelli, BDR and SW use $\tau=T_e/T_i$. KHD and GX `tau_fac` use $T_i/T_e$.
  Every benchmark below sets $T_i=T_e$ unless stated, so the difference is inert here. It is not inert in any τ scan.
- β differs: Hastie-Hesketh's β is the total local β. GENE/GS2 (Pueschel, AZ) use
  $\beta=\beta_e=\beta_i=\beta_{tot}/2$, which equals GX `beta` with $n_{ref}=n_e$, $T_{ref}=T_e$.
- **Frequency sign: check it before encoding any ω.**
  `docs/normalization.rst` fixes $s\sim e^{(\gamma-i\omega)t}$ but not the propagation direction.
  Appendix A assumes that positive GX ω is the electron direction. The in-repo goldens contradict this:
  `src/gkx/data/cyclone_reference_adiabatic.csv` (ITG) and `src/gkx/data/kbm_reference.csv` (KBM)
  both have ω>0, and both modes travel in the ion direction. Before encoding a sign, run one
  ion-direction case (CBC ITG) and one electron-direction case (TEM or ETG) in GKX against the goldens.
  Until then, compare |ω| only.

## 1. Ranked summary

Ranked by feasibility (cost × how cleanly GKX's model matches the paper's).

| # | Benchmark | Reference value | Evaluation | Tolerance | GKX coverage |
|---|---|---|---|---|---|
| 1 | RH residual (+ Xiao-Catto) | 0.1192 RH; 0.1018 XC circular (q=1.4, ε=0.18) | closed form | XC ±7%; RH ±15% | yes (ky=0, Boltzmann e with zonal correction) |
| 2 | SW GAM | ω_G=2.738 v_ti/R0, γ=−0.063 v_ti/R0 (q=1.4, τ=1, k_r→0); root of (2.7): 2.837, −0.041 | closed form + complex root | ω ±5%; γ ±15% (root) / ±35% (formula) | yes (ky=0, same run as #1) |
| 3 | PKJ08 CBC KBM β scan, ky=0.2 | β_crit^sub=1.14%, β^dom=1.26%, β_MHD=1.32% | figure/text + closed form | ±3% / ±4% / exact | yes (A∥; whether PKJ had B∥ on is unknown) |
| 4 | Strongly driven KBM (Tang80, AZ17) | ω_r=ω_*pi/2 | algebraic | ±5% (R/L_T≥35); (1.0,1.3)× at R/L_T=15 | needs ω_κ≠ω_∇B drift consistency and B∥ |
| 5 | Hastie-Hesketh s-α KBM | γ/ω_A≈0.33 (ka_i→0), 0.34 peak; stable by ka_i≈0.5 | figure + own ODE (0.3328) | ±10% at ka_i≤0.2 | yes, with small ε (no trapping) |
| 6 | CHT ideal ballooning α_crit(ŝ) | table in §2.6 | own ODE, θ cut-off converged to 1e-4 | onset α in [0.8,1.02]·α_crit | indirect (KBM ky→0 limit) |
| 7 | Romanelli / BDR ITG thresholds | η_ic=1+2.5(ε_n−0.2); R/L_T,c≈2.86 (BDR, read) | fit / figure | ±0.3 in η_ic; lower bound | yes, Boltzmann e; GKX keeps trapped ions |
| 8 | KHD EM-ITG β stabilization | β_ic=0.0109 (Eq. 24); ITG γ→0 near β_i≈0.0075 | closed form / figure | ±30%, trend | partial: α=0 at finite β, no B∥ or trapping in the paper |

Precision gates (≤5%): #1 (XC form), #2 (ω_G), #3 (β_crit) and #4 (ω_r ratio). Soft or trend gates: #5 to #8.

## 2. Specifications

### 2.1 Rosenbluth-Hinton residual and Xiao-Catto shaping

- Equations. RH, PRL 80, 724 (1998), Eqs. (15)/(22):
  $\dfrac{\phi_k(\infty)}{\phi_k(0)}=\dfrac{1}{1+1.6\,q^2/\sqrt\epsilon}$.
  Xiao & Catto, PoP 13, 082307 (2006), Eqs. (38)-(39):
  $\phi(\infty)/\phi(0)=1/(1+Sq^2/\sqrt\epsilon)$ with
  $S=\dfrac{1}{1+\kappa^2}\Big(3.27+\sqrt\epsilon+0.722\epsilon-1.443\delta-2.945\dfrac{\Delta}{\epsilon}+\dfrac{0.692\kappa^2-0.722}{q^2}\epsilon\Big)$,
  where Δ is the dimensionless Shafranov shift and δ is the triangularity.
- Reference values (closed form, exact):
  - RH(1.4, 0.18) = 0.1192. XC circular (1.4, 0.18) = 0.1018. RH(1.5, 0.1) = 0.0807.
  - XC Fig. 1 (q=2, ε=0.2, Δ=0.04): κ = 1, 1.8, 3 gives 0.0640, 0.1240, 0.2396. The figure reads 0.063 / – / 0.24 (±0.003).
  - Fig. 2, δ=0.4: 0.1461. Fig. 3, Δ = 0 and 0.1: 0.1075 and 0.1613.
- Normalization: none needed; the ratio is independent of the $v_t$ convention.
- GKX case:
  - ky=0 with a single kx, $k_x\rho_i=0.05$ (also check 0.025).
  - Circular Miller or s-α; `s_hat` is irrelevant at ky=0. q=1.4, ε=0.18.
  - Boltzmann electrons with the zonal (flux-surface-average) correction; collisionless.
  - Initial gyrocenter-density perturbation; nperiod ≥ 2.
  - Measure the mean of φ_zonal over the last 30%, after the GAM has decayed (t ≳ 100 a/v_ti).
- Comparison quantity: the ratio φ(∞)/φ(0).
- Tolerance:
  - XC ±7%. XC truncates at O(ε^{5/2}) (≈2% at ε=0.18), plus finite-k_x and averaging noise.
  - RH ±15%. RH drops O(ε) terms and sits 17% above XC here.
  - A q scan {1, 1.4, 2} must fit $1/(1+cq^2/\sqrt\epsilon)$ with c ∈ [1.6, 2.0].
  - A Miller κ scan (q=2, ε=0.2, s_κ=s_δ=0) should give residual(κ=3)/residual(κ=1) = 3.74 ±15%. The tolerance is loose because XC's equilibrium is global and Solov'ev-like.
- Coverage: full.

### 2.2 Sugama-Watanabe GAM (same run as 2.1)

- Equations. SW, J. Plasma Phys. 72, 825 (2006). Eq. (2.7) was re-checked against printed page 827:
  $$\frac1{K}=-i\hat\omega-\frac{iq^2}{2}\Big[2\hat\omega^3+3\hat\omega+(2\hat\omega^4+2\hat\omega^2+1)Z-\frac{\hat\omega}{2}\{2\hat\omega+(2\hat\omega^2+1)Z\}^2\{\tfrac{T_i}{T_e}+1+\hat\omega Z\}^{-1}
  +i\frac{\sqrt\pi}{2}\Big(\frac{k_rv_{Ti}q}{\Omega_i}\Big)^2e^{-\hat\omega_r^2/4}\Big\{\frac{\hat\omega_r^6}{64}+\Big(\frac{\hat\omega_r^4}{8}+\frac{3\hat\omega_r^2}{4}+3+\frac{6}{\hat\omega_r^2}\Big)\Big(1-\frac{3\hat\omega_r}{16}\{2\hat\omega_r+(2\hat\omega_r^2+1)Z(\hat\omega_r)\}\{\tfrac{T_i}{T_e}+1+\hat\omega_rZ_r(\hat\omega_r)\}^{-1}\Big)\Big\}\Big]$$
  Here $\hat\omega=R_0q\omega/v_{Ti}$ and $v_{Ti}=\sqrt{2T_i/m_i}$.
  - The notes' transcription of the nested bracket is correct.
  - "$Z_r$" is ambiguous on the page. The script takes Re Z, which affects only the finite-k_r term.
  - Asymptotic forms, Eqs. (2.9)-(2.10), with $\tau_e=T_e/T_i$:
    $\omega_G=\frac{\sqrt{7+4\tau_e}}{2}\frac{v_{Ti}}{R_0}\big[1+\frac{2(23+16\tau_e+4\tau_e^2)}{q^2(7+4\tau_e)^2}\big]^{1/2}$ and
    $\gamma=-\frac{\sqrt\pi}{2}q\frac{v_{Ti}}{R_0}\big[1+\ldots\big]^{-1}\big[e^{-\hat\omega_G^2}\{\hat\omega_G^4+(1+2\tau_e)\hat\omega_G^2\}+\tfrac14(k_rv_{Ti}q/\Omega_i)^2e^{-\hat\omega_G^2/4}\{\hat\omega_G^6/64+(1+\tfrac38\tau_e)(\hat\omega_G^4/8+3\hat\omega_G^2/4)\}\big]$.
- Reference evaluation: `sw_gam_formula` evaluates (2.9)-(2.10). `sw_gam_root` finds the secant root of 1/K=0, with Z continued analytically through `scipy.special.wofz`.
- Values in GX units ($v_{ti}/R_0$, τ=1):

  | case | (2.9)/(2.10): ω_G, γ | root of (2.7): ω_G, γ |
  |---|---|---|
  | q=1.4, k_r→0 | 2.7376, −0.0632 | 2.8367, −0.0409 |
  | q=1.5, k_r a_i=0.131 (paper Fig. 1) | 2.6902, −0.1394 | 2.7738, −0.1316 |
  | q=1.5, k_r→0 | 2.6902, −0.0377 | 2.7695, −0.0247 |

  For Cyclone (a/R0=0.36), (2.9) gives ω_G = 0.986 v_ti/a. The earlier script value of 1.802 is withdrawn: it used $\sqrt{7/4+\tau}$ with a different q correction.
- Normalization: multiply SW's $v_{Ti}/R_0$ values by √2. $k_rv_{Ti}q/\Omega_i=\sqrt2\,q\,k_r\rho_i^{GX}$.
- GKX case: as in 2.1, plus the Fig. 1 variant (ε=0.1, q=1.5, $k_x\rho_i=0.131$). Fit
  $\phi(t)=\phi_\infty+A\cos(\omega_Gt)e^{\gamma t}$ over t ∈ [2, 30] R0/v_ti.
- Tolerance:
  - ω_G: ±5% against the root. Formula and root differ by 3.5%, since $\hat\omega_G\approx2.7$ is not ≫1.
  - γ: ±15% against the root, ±35% against (2.10). The two differ by 35% at k_r→0.
  - SW Fig. 1 (GKV) shows about 16 periods in 40 R0/v_ti, a period of 2.5 (±10%). It is consistent with both.
- Coverage: full for passing-ion GAM physics. GKX keeps trapped ions. SW enters them only through the RH residual.

### 2.3 Pueschel-Kammerer-Jenko CBC KBM, ky=0.2 (primary code-to-code benchmark)

- Source: Pueschel, Kammerer, Jenko, PoP 15, 102310 (2008). Closed form from the text:
  $\beta_{crit,MHD}=0.6\hat s/[q^2(2\omega_n+\omega_{Ti}+\omega_{Te})]$ = **1.3206%**,
  with ŝ=0.786, q=1.4, ω_n=R/L_n=2.22 and ω_T=6.89.
- Reference values:
  - Subdominant KBM onset (eigenvalue solver, Fig. 2): **β_crit=1.14%** ±0.01. γ is linear in β there, with slope ≈2.9 (c_s/R)/%.
  - Dominant-mode onset: **1.26%**; extrapolated 1.21%. With consistent α: 1.26% and 1.22%. MHD limit: 1.32%.
  - α_MHD is 0.407 at β=1.14% and 0.450 at β=1.26%.
  - Fig. 1 KBM γ (c_s/R): 0.2 (1.3%), 0.5 (1.4%), 0.71 (1.5%), 0.87 (1.6%), 1.0 (1.7%); ±0.03.
  - KBM |ω| falls from 2.4 at 1.3% to 1.95 at 1.8%. GS2 is 0.1–0.25 lower.
  - ITG at β=0: γ=0.39, |ω|=0.52. ITG→TEM transition at 0.95%.
  - Fig. 5 β_crit^dom(ky): 1.37, 1.29, 1.26, 1.28, 1.34, 1.41, 1.52% for ky = 0.1 to 0.4 in steps of 0.05.
- Normalization: GENE β=β_e equals GX `beta`. At T_i=T_e, $k_y\rho_s$ equals GX `ky` and $c_s/R=v_{ti}/R$. For a-units (R/a=2.7775), multiply γ by R/a.
- GKX case:
  - `model="s-alpha"` with **alpha=0**, since PKJ set α_MHD=0.
  - q=1.4, s_hat=0.786, epsilon=0.18. tprim=6.89/2.7775=2.48 and fprim=0.80 for both species.
  - Kinetic electrons with m_e/m_i=5.669e-4 (hydrogen); collisionless; `use_apar=true`.
  - Run `use_bpar` both false and true: PKJ do not state whether B∥ was on.
  - nperiod ≥ 3 with an explicit `damp_ends_rate` (KBM-VEL: +7.5% from nperiod 2→3).
  - β scan 1.0–1.8%, in steps of 0.05% near onset.
- Comparison: the onset β where the linear fit of the KBM-branch γ(β) reaches zero, and the β where the KBM overtakes the TEM.
- Tolerance: β_crit^sub ±3% (1.11–1.17%); β^dom ±4%; γ(1.6%) ±10% (GENE–GS2 spread); |ω| ±12%.
- Coverage: yes. The subdominant onset needs an eigen/Krylov route, and KBM-VEL found that `solver="krylov"` does not converge on the EM KBM deck. The fallback is time runs above onset, extrapolated down.

### 2.4 Strongly driven KBM: ω_r = ω_*pi/2 (Tang-Connor-Hastie 1980; Aleynikova-Zocco 2017)

- Equations.
  - Tang et al., NF 20, 1439 (1980), §3: without trapped particles the marginal frequency is
    $\omega=\omega_{*i}(1+\eta_i)/2$, and $\omega^2_{MHD}\to\omega^2_{MHD}+\omega_{*pi}^2/4$.
  - Aleynikova & Zocco, PoP 24, 092106 (2017), Eq. (9), under the high-β ordering $\beta\sim b\sim\omega_d/\omega\sim\epsilon$:
    $$\frac{1}{\beta_i}\frac{v_{thi}^2}{\omega^2l_c^2}\partial_z\Big(bB\,\partial_z\phi\Big)=-\frac{2\omega_\kappa\omega_p}{\omega^2}\phi-b\Big[1-\frac{\omega_{*i}}{\omega}(1+\eta_i)\Big]\phi,$$
    with $\omega_p=\omega_{*i}(1+\eta_i)-\omega_{*e}(1+\eta_e)$.
  - Equilibrium consistency, Eq. (8): $\tfrac{\beta_i}{2}\omega_p^2/\omega^2=(\omega_\kappa-\omega_B)\omega_p/\omega^2$.
  - Code drifts, Eqs. (14)-(15): $\omega_\kappa=\omega_B+\frac{k_y\rho v}{2L}\frac{\beta}{2}[R/L_{Ti}+\tau R/L_{Te}+(1+\tau)R/L_n](1+\frac rR\cos z)^2$.
  - Variational form, Eq. (13): $\omega(\omega-\omega_{*pi})\int b|\phi|^2/B=2\omega_p\int\omega_\kappa|\phi|^2/B-\frac{v_{thi}^2}{\beta_il_c^2}\int b|\partial_z\phi|^2/B$.
    This gives exactly $\omega=\omega_{*pi}/2\pm i\sqrt{\gamma^2_{MHD}-\omega_{*pi}^2/4}$, so Re ω = ω_*pi/2 is necessary for instability. The script checks the root (residual 0).
  - Appendix D, local limit: $\omega_r=\omega_{*i}(1+\eta_i)/2$ and $k_{y,max}=0$. The paper gives $\beta_{diam}\approx L_p/(qR)$ at small q and $L_p/(q^2R)$ at large q, marked "cum grano salis".
- Reference values: AZ Figs. 2-5 (GENE). Geometry is s-α CBC: q=1.4, ŝ=0.786, r/R=0.18, R/L_n=2.22, T_i=T_e, m_i/m_e=1836, β=β_i=β_e=1.5%, consistent α, ω_κ≠ω_B, B∥ on.
  - R/L_T = 35, 40, 45 (Fig. 2): ω_r lies on the ω_pi/2 line within reading (±5%). γ (c_s/R) is ≈7.1 / 7.1 / 7.0 at ky ρ_s=0.03 and falls to ≈4.0 / 3.6 / 3.2 at ky ρ_s≈0.4 (±0.2). The maximum γ is at ky→0.
  - R/L_T=15 (Fig. 3): γ≈5.0 at ky ρ_s=0.03 and 3.0 at 0.55. ω_r is slightly above ω_pi/2 (the Eq. 11 regime).
  - R/L_T=7.5 (Fig. 5): γ peaks at 1.55 near ky ρ_s=0.25 (0.75 at 0.03). ω_r>ω_pi for ky ρ_s<0.3, and ω_pi/2<ω_r<ω_pi above.
  - Appendix B, Fig. 6: forcing ω_κ=ω_B lowers γ strongly. Appendix C: the regime boundary is at R/L_T between 10 and 15.
- Normalization: AZ's $\omega_{*i}$ equals GX `ky*fprim` (in $v_{ti}/a$). In GX units, $\omega_{*pi}/2=k_y(\mathrm{fprim}+\mathrm{tprim})/2$:
  ky·6.70, ky·7.60 and ky·8.50 $v_{ti}/a$ for R/L_T = 35, 40, 45 (R/a=2.7775). The consistent
  α is $q^2\beta\sum_s(\mathrm{fprim}+\mathrm{tprim})_s(R/a)$: 2.19, 2.48 and 2.78.
- GKX case:
  - `model="s-alpha"`, q=1.4, s_hat=0.786, epsilon=0.18, R0=2.7775.
  - fprim=0.80 and tprim=(35|40|45)/2.7775 for both species; beta=0.015; **alpha consistent (values above)**.
  - `use_apar=true`, `use_bpar=true`, kinetic hydrogen electrons.
  - ky ∈ {0.05, 0.1, 0.2, 0.3}; nperiod ≥ 3.
- Comparison: $\omega_r/(k_y(\mathrm{fprim}+\mathrm{tprim})/2)$, using |ω| until §0 is settled.
- Tolerance: 1 ± 0.05 at R/L_T ≥ 35 and ky ≤ 0.3. At R/L_T=15 the ratio must be in (1.0, 1.3). γ within ±10% of the AZ GENE readings at ky ρ_s ≤ 0.1.
- Coverage: **must be verified.** GKX's s-α drift must include the β′ correction ω_κ≠ω_∇B of AZ Eq. (15); GS2 had to be patched for it. If GKX uses ω_κ=ω_B, this benchmark tests the wrong equation (Appendix B, Fig. 6). B∥ is required.

### 2.5 Hastie-Hesketh kinetic modification of s-α ballooning

- Equations. Hastie & Hesketh, NF 21, 651 (1981).
  - Eq. (2.1): $\partial_\theta[f\partial_\theta F]+\alpha gF+\frac{\omega^2}{\omega_A^2}fF=0$, with $f=1+(S\theta-\alpha\sin\theta)^2$,
    $g=\cos\theta+(S\theta-\alpha\sin\theta)\sin\theta$ and $\omega_A=V_A/(Rq)$.
  - Eq. (2.6), the b≪1 form: it adds $-\frac{\beta_iq^2}{2}[7\Omega_{2i}+4\tau\Omega_{1i}^2/\Omega_{0e}]g^2F$
    and multiplies the inertia by $\Omega_{1i}=1-\omega_{*i}(1+\eta_i)/\omega$. For ω_*/ω→0 it reduces to MHD with adiabatic index $(7/4+\tau)/(1+\tau)$.
  - Eq. (2.3) with H from Eq. (2.4) is the full FLR plus Landau-resonance equation (curve a in Figs. 2-4).
  - Relation given in the text: $\beta=(r_n/Rq^2)\alpha(1+\tau)/[1+\eta_i+\tau(1+\eta_e)]$.
- Reference values.
  - Fig. 2 (S=0.6, α=0.8, q=√2, η_e=η_i=τ=1, r_n/R=0.1, β=0.02), γ/ω_A: ≈0.33 at ka_i→0, 0.34 at 0.1, 0.33 at 0.2, 0.29 at 0.3, 0.21 at 0.4 and 0.05 at 0.5. A residual of about 0.02 remains beyond 0.6 (±0.02).
  - Fig. 3: Re ω/ω_*i is slightly above 1, in the ion direction.
  - **Own check:** the ka_i→0 limit of Eq. (2.6), with $\beta_iq^2(7+4\tau)/2=0.11$, gives **0.3328** and matches the figure. The incompressible Eq. (2.1) gives 0.4102.
  - Fig. 4 (S=0.4, α=0.54, β=0.0135): the curve starts near 0.1 and peaks near 0.25 at ka_i≈0.22. Appendix A read the peak as 0.30; this is corrected here. Eq. (2.6) at ka_i→0 gives 0.134.
- Normalization:
  - $ka_i=k_\theta\rho_i^{GX}$, because HH's $a_i$ uses $\sqrt{T/m}$. So ka_i equals GX `ky` at θ=0.
  - GX `beta`=0.01, i.e. $\beta_i$ is half the total of 0.02.
  - With lengths in R and fprim=tprim=10 for both species, α = q²β Σ(fprim+tprim) = 0.800 (script).
  - $\omega_A=\sqrt{2/\beta_i}/q=10.0\,v_{ti}/R$ and $\omega_{*i}=10k_y\,v_{ti}/R$, so $\omega_{*i}/\omega_A=k_y$. Fig. 3 confirms this mapping.
- GKX case:
  - `model="s-alpha"`, R0=1 (lengths in R), q=1.41421, s_hat=0.6, alpha=0.8.
  - epsilon=0.01, because HH neglects trapping and the variation of ∇B.
  - fprim=tprim=10 for both species; beta=0.01; kinetic hydrogen electrons.
  - `use_apar` and `use_bpar` true, since Tang's equation keeps δB∥.
  - ky ∈ {0.05, 0.1, 0.2, 0.3, 0.4, 0.5}; nperiod ≥ 4, because the ka_i→0 mode is extended.
- Comparison: γ/ω_A = γ[v_ti/R]/10, and Re ω/ω_*i.
- Tolerance: ±10% at ka_i ≤ 0.2 (reading ±0.02, plus the electron FLR/Landau physics that HH neglect). γ/ω_A must fall below 0.1 by ka_i=0.5. Re ω/ω_*i ∈ [1.0, 1.3].
- Coverage: yes, except trapping, which GKX cannot turn off; small ε suppresses it.

### 2.6 Connor-Hastie-Taylor ideal ballooning boundary

- Equation. Connor, Hastie & Taylor, PRL 40, 396 (1978), Eq. (10):
  $\frac{d}{d\eta}\big[1+(s\eta-\alpha\sin\eta)^2\big]\frac{dF}{d\eta}+\alpha[\cos\eta+\sin\eta(s\eta-\alpha\sin\eta)]F=0$,
  with $\alpha=-q^2R\,d\beta/dr$ (β total).
- Reference evaluation: `cht_alpha_crit_converged`.
  - Newcomb criterion: an even solution with a zero on (0, θ_max] means unstable. α_crit is bisected to 1e-7.
  - The truncated domain approaches the infinite-domain value from above, with error O(1/θ_max) (each doubling halves the change). θ_max is doubled from 50 until α_crit moves by < 1e-4.
  - The Richardson value 2v_n − v_{n−1} is also printed.
  - The earlier script stopped at θ_max=60 and was up to 0.012 high.

  | ŝ | 0.2 | 0.4 | 0.6 | 0.786 | 0.8 | 1.0 | 1.5 |
  |---|---|---|---|---|---|---|---|
  | α_crit (converged) | 0.3151 | 0.3545 | 0.4246 | 0.5052 | 0.5117 | 0.6087 | 0.8776 |
  | final θ_max | 12800 | 6400 | 3200 | 3200 | 3200 | 3200 | 6400 |

  The last doubling moved each value by 5e-5 to 9e-5. At ŝ=1 the value, 0.609, agrees with the
  familiar "α_c ≈ 0.6 at ŝ=1". At the CBC shear (0.786) the ODE gives α_crit=0.505, which is 7% above
  the 0.6ŝ=0.472 rule of thumb that PKJ use for β_MHD. The PKJ benchmark (§2.3) keeps their closed form.

  - The CHT Fig. 1 curve reads 0.36 (ŝ=0.2), 0.44 (0.4), 0.52 (0.6), 0.60 (0.8), 0.67 (1.0), ±0.03 (Appendix A).
  - The printed curve sits above the converged ODE at ŝ ≥ 0.4. **The ODE value is the reference; the figure is a ±0.1 sanity check.**
- GKX use: not a direct run. It checks the ky→0 KBM onset. At fixed gradients, the α at the GKX KBM onset should approach the ideal α_crit from below as ω_*→0. The GKX runs need B∥, consistent drifts, low ky and large nperiod. PKJ found the KBM onset 14% below β_MHD.
- Tolerance: the extrapolated ky→0 GKX onset α must lie in [0.8, 1.02]·α_crit(ŝ).
- Coverage: indirect only; GKX is not an MHD code.

### 2.7 ITG thresholds: Romanelli 1989 and Biglari-Diamond-Rosenbluth 1989

- Romanelli, Phys. Fluids B 1, 1018 (1989).
  - Local kinetic dispersion, Eq. (12): $(1+1/\tau)-\int d^3vF_MJ_0^2\frac{\omega-\omega_{*T}}{\omega-\omega_D}=0$.
  - ∇B-model threshold, Eq. (26): $\eta_{ic}=2(1+1/\tau)\epsilon_n$, or $\frac43(1+1/\tau)\epsilon_n$ with isotropic averaging. At flat density and τ=1 this is $R/L_{T,c}=4$ or 2.667.
  - Full-kinetic fit, Eq. (27), for ŝ=0.5, q=1.5, τ=1, b_i=0.1: $\eta_{ic}=1$ for $\epsilon_n<0.2$, else $1+2.5(\epsilon_n-0.2)$. This gives 1.0, 1.125, 1.75 and 3.0 at ε_n = 0.1, 0.25, 0.5 and 1.
  - Fig. 3's full-kinetic dotted line reads 4.3 at ε_n=1 (±0.2), so the fit and the line disagree there.
  - The fluid Eq. (22) holds only at large ε_n (3.125 at ε_n=1).
- BDR, Phys. Fluids B 1, 109 (1989).
  - Fluid flat-density stability needs $L_T/R>4/(7+\tau)$ (Eq. 32), i.e. $R/L_{T,c}=2.0$.
  - The resonant (full D) boundary in Fig. 2 reads $L_T/R\approx0.35$, i.e. $R/L_{T,c}\approx2.86$ (±0.25).
  - Necessary condition for instability: η_i>2/3.
- Normalization: Romanelli's $b_i=(k\rho_i^{GX})^2$, so b_i=0.1 gives ky≈0.316. $\epsilon_n=L_n/R=(a/R)/\mathrm{fprim}$ and η_i = tprim/fprim.
  $\Omega=\omega/\omega_{*e}$. τ=T_e/T_i is the inverse of GX `tau_fac`.
- GKX case (Romanelli):
  - `model="s-alpha"`, alpha=0, q=1.5, s_hat=0.5, epsilon=0.05 (small).
  - Boltzmann electrons with τ=1; ky=0.3.
  - Scan tprim at fixed fprim for ε_n ∈ {0.25, 0.5, 1.0}. Take η_ic where γ extrapolates linearly to zero.
- GKX case (BDR): fprim=0, ky ∈ {0.1, 0.2}, scan R/L_T over [2, 5].
- Tolerance: ±0.3 in η_ic, trend only; the paper neglects trapped ions and the variation of v_∥. BDR is a lower bound: GKX R/L_T,c must be ≥ 2.6.
- Coverage: yes. GKX keeps trapped ions; small ε limits them.

### 2.8 Kim-Horton-Dong EM-ITG β stabilization

- KHD, Phys. Fluids B 5, 4030 (1993).
  - Local ballooning threshold, Eq. (24):
    $\beta_{ic}=\dfrac{2k_\parallel^2k_\perp^2}{\omega_{Di}(\omega_{*pi}-\omega_{*pe})-(\omega_{*pi}k_\perp/2)^2}$.
    In the long-wavelength limit, $\beta_{ic}\simeq k_\parallel^2/\{\epsilon_n[1+\eta_i+(1+\eta_e)T_e/T_i]\}$.
  - Base case: k_∥L_n=0.1, ε_n=0.2, η_i=2.5, η_e=2, τ=1, k_yρ_i=0.5. Eq. (24) gives **0.0109**. The long-wavelength form gives 0.0077.
  - The local-fluid ballooning onset in Fig. 1 reads 0.0115, within 5% of Eq. (24).
  - Local-kinetic ITG γ falls from 0.20 to 0 by β_i≈0.0075. The kinetic ballooning onset is ≈0.0065 (±0.0005).
  - Nonlocal Fig. 6a (s=0.6, q scan): β=0.004 cuts γ from ≈0.17 to ≈0.02 at q=1.5 (±0.01).
- Normalization: KHD's $v_t=\sqrt{T/m}$ matches GX. β_i equals GX `beta`. γ is in $v_{ti}/L_n$; multiply by fprim for a-units.
  The base case corresponds to R/L_n=5, R/L_Ti=12.5 and R/L_Te=10.
- GKX case:
  - `model="s-alpha"` with **alpha=0** at finite β, as in the paper.
  - q=1.5, s_hat=0.6, small ε; kinetic electrons.
  - `use_apar=true`, `use_bpar=false`; ky=0.5; β scan over [0, 0.01].
- Tolerance: ±30%, trend only. KHD's electrons are a first-order expansion and have no trapping.
- Coverage: partial. Use it as a KBM-branch sanity check, not as a gate.

## 3. Not recommended as precision tests

- ETG/ITG species swap: exact to solver tolerance. It is a code-symmetry check, not an analytic one. Keep it as a unit test.
- Tsai-Chen 1993 (energetic-particle KBM): without energetic particles only the window $\omega_{*pi}/2<\omega_r<\omega_{*pi}$ applies.
- Aleynikova et al. JPP 2018 (W7-X KBM): the equilibrium must be consistent with β, so soft targets only (Appendix A §6).
- Slab ITG, TEM, microtearing and TAE have no closed form accurate to ≤5%.

## 4. Papers

**Fetched and read:** Aleynikova & Zocco, PoP 24, 092106 (2017), doi:10.1063/1.5000052. It has been integrated in §2.4 and is no longer missing.

**DOIs verified from the downloaded PDFs:**
- 10.1063/1.863121 (Antonsen-Lane)
- 10.1017/S0022377806004958 (SW)
- 10.1063/1.859206 (BDR)
- 10.1063/1.859023 (Romanelli)
- 10.1063/1.860623 (KHD; the earlier 1.860622 was wrong)
- 10.1063/1.860624 (Tsai-Chen; the earlier 1.860625 was wrong)
- 10.1063/1.2266892 (Xiao-Catto; the earlier 1.2245215 was wrong)
- 10.1063/1.3005380 (PKJ)
- 10.1063/1.5000052 (AZ)

**From publisher metadata only** (no DOI in the text layer): CHT 10.1103/PhysRevLett.40.396, RH 10.1103/PhysRevLett.80.724, Tang 10.1088/0029-5515/20/11/011.

**Not printed, so unverified:** Hastie-Hesketh (NF 21, 651) and Aleynikova et al. JPP 84 (2018).

**Still wanted.** DOIs are taken from the papers' reference lists and stay unverified until downloaded:
1. Pueschel & Jenko, PoP 17, 062307 (2010), 10.1063/1.3435280. Settles whether B∥ was on in PKJ08.
2. Hinton & Rosenbluth, PPCF 41, A653 (1999), 10.1088/0741-3335/41/3A/059. Finite-k_r residual.
3. Sugama & Watanabe, PoP 13, 012501 (2006), 10.1063/1.2149311. Long GAM/zonal-flow paper; helical residual.
4. Xiao, Catto & Dorland, PoP 14, 055910 (2007), 10.1063/1.2718519. Finite-ρ_pol residual.
5. Gao, Itoh, Sanuki & Dong, PoP 13, 100702 (2006), 10.1063/1.2359722. GAM eigenmodes.
6. Zocco, Helander & Connor, PPCF 57, 085003 (2015), 10.1088/0741-3335/57/8/085003. EM-ITG (AZ Ref. 22).
7. Dong, Horton & Kim, Phys. Fluids B 4, 1867 (1992), 10.1063/1.860040. KHD's nonlocal η_i code.
8. Connor, Hastie & Taylor, Proc. R. Soc. A 365, 1 (1979), 10.1098/rspa.1979.0001. Full ballooning theory.
9. Dimits et al., PoP 7, 969 (2000), 10.1063/1.873896. Cyclone residual and threshold values.
10. Kotschenreuther, Dorland, Beer & Hammett, PoP 2, 2381 (1995), 10.1063/1.871261. R/L_T,crit fit.

## 5. Next steps (owners)

1. Settle the ω sign (§0) with one ITG run and one TEM or ETG run against the goldens.
2. Check whether GKX's s-α drift includes the β′ term (ω_κ≠ω_∇B). §2.4 and §2.5 depend on it.
3. Encode §2.1 and §2.2 as one ky=0 test (office GPU), then run the §2.3 β scan.

## 6. Corrections to the earlier survey and the notes

- GAM ω_G: 1.802 was wrong. The correct values are 2.738 v_ti/R0 (formula) and 2.837 (root) at q=1.4.
- CHT α_crit at θ_max=60 was up to 0.012 high. The converged table replaces it.
- The peak in HH Fig. 4 is ≈0.25, not 0.30.
- The GX sign assumption in the notes conflicts with the in-repo goldens (§0).
- Three DOIs were corrected (§4), and Aleynikova & Zocco 2017 has been read.
