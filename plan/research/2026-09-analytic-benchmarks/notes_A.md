# Notes A — classic KBM / ballooning theory papers (read in full, 2026-09-28)

Appendix A of REPORT.md. Scope: seven PDFs (maintainer's subscription copies, not committed). Equations are transcribed from the page images (Gaussian units as printed).
Figure readings are by eye from the scans. Each one carries its reading uncertainty.

GKX/GX conventions assumed for the mapping: lengths are in `a` (or `R` when `geo=s-alpha` with `a=R` style inputs), velocities are in
`v_ti = sqrt(T_i/m_i)` (GX uses `v_t = sqrt(T/m)`, **no factor 2**), `rho_i = v_ti/Omega_i`, `beta_ref = 8 pi n_ref T_ref / B_ref^2`,
and `omega` is in `v_ti/a`. In GX, positive real frequency means the **electron** diamagnetic direction for `ky>0` (GX sign); GENE
and most papers below report the ion direction as positive, so **flip the sign of omega when comparing**. Check this against
existing GKX CBC goldens before trusting it. Conversions: `c_s/R` (GENE, T_i=T_e) equals `v_ti/R` when `T_e=T_i`
because GENE's c_s = sqrt(T_e/m_i). For `k_y rho_s` (GENE, rho_s = c_s/Omega_i), `k_y rho_s = k_y rho_i(GX)` when `T_e=T_i`.
GENE's `v_th = sqrt(2T/m)`, so GENE `rho_i` (as in Pueschel 2008, "units of rho_i") may carry a sqrt2. Pueschel defines rho_i via
c_s ("(v_ti rho_i^2 ...)" in the flux units). Treat `k_y rho_i` there as `k_y rho_s` with rho_s = c_s/Omega_i. This matches GX when T_e=T_i.

---

## 1. Connor, Hastie & Taylor, PRL 40, 396 (1978)

- **Citation:** J. W. Connor, R. J. Hastie, J. B. Taylor, "Shear, Periodicity, and Plasma Ballooning Modes", Phys. Rev. Lett. **40**, 396–399 (6 Feb 1978). DOI 10.1103/PhysRevLett.40.396.
- **Model:** ideal MHD δW, high-n limit, axisymmetric. Introduces the ballooning transformation, which gives an infinite-domain ODE per surface; the |y|→∞ behaviour is tied to Mercier.
- **Key equations:**
  - Coordinates (ψ,χ,ζ), \(\mathbf B=\nabla\psi\times\nabla\zeta+f(\psi)\nabla\zeta\), \(\nu=fJ/R^2\), \(\oint\nu\,d\chi=2\pi q\).
  - Eikonal (1): \(\varphi(\psi,\chi)\exp\{in[\zeta-\int^\chi\nu\,d\chi]\}\).
  - (3) \(\varphi(\theta,x)=\sum_m e^{-im\theta}\int_{-\infty}^{\infty}e^{im\eta}\hat\varphi(\eta,x)\,d\eta\).
  - (5) \(\delta W_0=\pi\int J\,d\chi\,d\psi\Big[\frac{B^2}{R^2B_p^2}|k_\parallel X|^2+R^2B_p^2\big|\frac1n\frac{\partial}{\partial\psi}k_\parallel X\big|^2-2\frac{dp}{d\psi}\big(\frac{\kappa_n}{RB_p}|X|^2-\frac{ifB_p}{B^2}\kappa_s\frac{X}{n}\frac{\partial X^*}{\partial\psi}\big)\Big]\), with \(ik_\parallel\equiv(JB)^{-1}(\partial/\partial\chi+in\nu)\) and \(X=RB_p\xi_\psi\).
  - (7) transform; (8) quasimode \(\hat X=F(\psi,y)\exp(-in\int^y\nu\,dy)\).
  - (9) \(\frac1J\frac{d}{dy}\Big\{\frac{1}{JR^2B_p^2}\Big[1+\Big(\frac{R^2B_p^2}{B}\int^y\frac{\partial\nu}{\partial\psi}dy\Big)^2\Big]\frac{dF}{dy}\Big\}+\frac{2}{RB_p}\frac{dp}{d\psi}\Big(\kappa_n-\frac{fRB_p^2}{B^2}\kappa_s\int^y\frac{\partial\nu}{\partial\psi}dy\Big)F=0\).
  - Asymptotics: \(\frac{d}{dy}y^2\frac{dF_0}{dy}+DF_0=0\), \(F\sim Ay^{\alpha_1}+By^{\alpha_2}\), \(\alpha_{1,2}=-\tfrac12\pm(\tfrac14-D)^{1/2}\) (Mercier: D<1/4).
  - **(10) s–α model:** \(\frac{d}{d\eta}[1+(s\eta-\alpha\sin\eta)^2]\frac{dF}{d\eta}+\alpha[\cos\eta+\sin\eta(s\eta-\alpha\sin\eta)]F=0\), with \(s\equiv d(\ln q)/d(\ln r)\) and \(\alpha\equiv-(2Rq^2/B^2)\,dp/dr\) (Gaussian units, B²/8π pressure. In SI-like β units \(\alpha=-q^2R\,d\beta/dr\)).
- **Normalizations to GKX:** \(\alpha = q^2 R\,\beta_{ref}\sum_s (n_s T_s)(a/L_{n_s}+a/L_{T_s})/a\) (per unit n_ref T_ref), i.e. for GX `s-alpha` geometry `alpha = q^2 * beta * sum_s dens*temp*(fprim+tprim) * (R/a)` when gradients are in 1/a. GX's `shat`=s. GX `s-alpha` input takes `shift` = α directly (check sign: GX uses α>0 destabilizing).
- **Benchmarks (Fig. 1, maximum stable α vs s, the solid curve = infinite domain; dotted = Dobrott F(±π)=0):** read values (±0.03 in α):
  s=0 → α≈0.30 (curve starts vertical near α≈0.3 at s≲0.1); s=0.2 → α≈0.36; s=0.4 → α≈0.44; s=0.6 → α≈0.52; s=0.8 → α≈0.60; s=1.0 → α≈0.67; s=1.2 → α≈0.76; s=1.4 → α≈0.84; s≈1.6 → α≈0.9. Only the first-stability boundary is shown (the second stability branch is not plotted; see Hastie–Hesketh Fig. 1). Text: "roughly dp/dr ~ 0.25 B0²/Rq²" ⇒ α≈0.5 over most of the range; Dobrott's constraint overestimates it by ~20%.
  Fig. 2: marginal eigenfunctions F(η) for s=0.1 (curve A, extends to η≈50, secondary maxima near η≈2π,4π) and s=0.7 (curve B, decays by η≈10).
- **Supports:** B-MHD-1 (ideal ballooning limit of the GK code: KBM γ→0 as k_y→0 should cross zero near the s–α boundary). Use the modern ideal-ballooning solver value, not the figure reading: the canonical s=0.8 threshold α≈0.6 (this figure). Tolerance: ±5% on α_crit from an ODE shooter (the benchmark should be our own ODE, validated against this figure at ±0.05).
- **Refs to fetch:** Connor, Hastie & Taylor, Proc. R. Soc. Lond. A **365**, 1 (1979), DOI 10.1098/rspa.1979.0001 (full theory, higher order). Dobrott et al., PRL **39**, 943 (1977), DOI 10.1103/PhysRevLett.39.943. Coppi, PRL **39**, 938 (1977), DOI 10.1103/PhysRevLett.39.938. Mercier, Nucl. Fusion **1**, 47 (1960) [printed "(1969)"], DOI 10.1088/0029-5515/1/1/004.

---

## 2. Tang, Connor & Hastie, Nucl. Fusion 20, 1439 (1980)

- **Citation:** W. M. Tang, J. W. Connor, R. J. Hastie, "Kinetic-ballooning-mode theory in general geometry", Nucl. Fusion **20**(11), 1439–1453 (1980). DOI 10.1088/0029-5515/20/11/011.
- **Model:** collisionless linear gyrokinetics (Vlasov–Maxwell) in general axisymmetric geometry, ballooning representation. Ordering ε ~ ρ_i/L ~ ω/Ω_i, k⊥ρ_i ~ 1, k∥ρ_i ~ ε. Fields Φ, A∥, δB∥ in Coulomb gauge. Three coupled integral equations. Two limits: low frequency (ω < ω_bi, ω_ti) and intermediate (ω_bi, ω_ti < ω < ω_be, ω_te).
- **Key equations (as printed):**
  - (2.7) \(\Phi(\psi,\chi,\zeta)=\sum_p\hat\Phi(\psi,\chi-2\pi p,\zeta)\). (2.11) \(S=[\zeta-\int_0^\chi d\chi' IJ/R^2+\int^\psi k(\psi)d\psi]\). (2.13) \(k_\psi=-nRB_\chi[\int_0^\chi d\chi'\frac{\partial}{\partial\psi}(IJ/R^2)-k(\psi)]\), (2.14) \(k_b=nB/RB_\chi\).
  - (2.18) GK equation: \(\frac{v_\parallel}{JB}\frac{\partial\hat h}{\partial\chi}-i\hat h(\omega-\vec k_\perp\cdot\vec v_D)=-\frac{ie}{T}F_m(\omega-\omega_*^T)\Big[J_0(\alpha)\big(\hat\phi-\frac{v_\parallel}{c}\hat A_\parallel\big)+J_1(\alpha)\frac{v_\perp}{k_\perp}\frac{\delta\hat B_\parallel}{c}\Big]\), with (2.21) \(\omega_*^T=\omega_*[1+\eta(\frac{mE}{T}-\frac32)]\), (2.22) \(\omega_*=\frac{ncT}{e}\frac{d}{d\psi}\ln n_0\), \(\alpha=k_\perp v_\perp/\Omega\), \(\eta=d\ln T/d\ln n_0\).
  - (2.31) quasineutrality, (2.32) \(k_\perp^2\hat A_\parallel=\frac{4\pi}{c}\hat j_\parallel\), (2.34) \(\delta\hat B_\parallel=-\frac{1}{k_\perp}\frac{4\pi}{c}\sum 2\pi e\int dE\,d\mu\frac{B}{|v_\parallel|}v_\perp(\hat h_++\hat h_-)J_1\).
  - MHD reference (3.1)–(3.2) and the low-β, (ω/ω_s)²≪1 limit (3.5): \(\frac1J\frac{\partial}{\partial\chi}\big(\frac{1}{JB^2}|\nabla S|^2\frac{\partial\hat\psi}{\partial\chi}\big)+\frac{n_0M_i\omega^2}{B^2}|\nabla S|^2\hat\psi+2\kappa_w\frac{1}{B^2}(\nabla P\times\vec B\cdot\nabla S)\hat\psi=0\). The (ω/ω_s)²≫1 Alfvén limit (3.6) adds \(-4\gamma P(\kappa_w)^2\hat\psi\).
  - Low-frequency result (3.13), with \(\omega_{*p}\equiv\omega_{*i}(1+\eta_i)-\omega_{*e}(1+\eta_e)\), \(\bar\omega_{Di}=2\omega_\kappa(1-\lambda B)+\lambda B\omega_B\). (3.15) \((\beta_i/2)\omega_{*p}=\omega_\kappa-\omega_B\) (the equilibrium consistency; relevant to GKX's `beta_prime` handling of ∇B drift).
  - (3.24) single low-frequency KBM equation:
    \(\frac{L_c^2}{JB^2}\frac{\partial}{\partial\chi}\big(\frac bJ\frac{\partial\hat\phi}{\partial\chi}\big)+\big(\frac{\omega}{\omega_A}\big)^2\Big\{\frac{2\omega_{*p}\omega_\kappa}{\omega^2}\hat\phi+b\big[1-\frac{\omega_{*i}}{\omega}(1+\eta_i)\big]\hat\phi-\sum_j\frac{T_j}{T_i}\big[1-\frac{\omega_*}{\omega}(1+2\eta)\big]_j\frac{15}{8}\int_{Tr}d\lambda B(1-\lambda B)^{-1/2}\big\langle\hat\phi\frac{\omega_\kappa}{\omega}\big\rangle\frac{\omega_\kappa}{\omega}\Big\}=0\), with \(b=k_\perp^2\rho_i^2/2\), \(\omega_A^2=v_A^2/L_c^2\).
  - Marginal frequency without trapped particles: **\(\omega=\omega_{*i}(1+\eta_i)/2\)**. Conclusion: \(\omega^2_{MHD}\to\omega^2_{MHD}+\omega^2_{*pi}/4\).
  - (3.25) \(n_{crit}\simeq a^2/(R\rho_iq^2)\) (typ. 10–20).
  - Intermediate regime: (3.27)–(3.29) Q, Q′. (3.33) R. (3.35)–(3.37) \(\delta\tilde B_\parallel\). (3.38) \(\frac{L_c^2}{JB^2}\frac{\partial}{\partial\chi}\big(\frac bJ\frac{\partial\hat\psi_\parallel}{\partial\chi}\big)=\frac{\omega^2}{\omega_A^2}\hat\psi_\parallel K\) with K = (3.39) (identical in form to Aleynikova Eq. 2 / Hastie–Hesketh H), \(\alpha_{\ell j}=[1-\frac{\omega_{*j}}{\omega}(1+\ell\eta_j)]\).
  - (3.40) β≪1, b and |ω_D/ω| small: \(\frac{L_c^2}{JB^2}\partial(\frac bJ\partial\hat\phi)=-(\frac{\omega}{\omega_A})^2\hat\phi\big[b\alpha_{1i}+2\frac{\omega_\kappa\omega_{*p}}{\omega^2}-(\frac{\omega_\kappa}{\omega})^2(7\alpha_{2i}+4\tau\alpha_{1i}\alpha_{1e}/\alpha_{0e})\big]\). This reduces to MHD (3.6) with (3.41) \(\gamma=[(7/4)T_i+T_e]/(T_e+T_i)\). (3.42) adds the trapped-particle term.
- **Normalizations to GKX:** ω_* here uses d/dψ and physical n, so \(\omega_{*i}=k_y\rho_i v_{ti}/L_n\) (GX: `ky*fprim` in v_ti/a units, per species with charge sign). \(\omega_{*pi}=\omega_{*i}(1+\eta_i)\) = `ky*(fprim+tprim)` for ions. β_i=8πn_0T_i/B² = `beta_ref*dens_i*temp_i`. b=k⊥²ρ_i²/2 uses ρ_i with v_ti=√(2T/m)?? Printed as b ≡ k⊥²ρ_i²/2, consistent with GX's b = k⊥²ρ_i² (√(T/m)) only if their ρ_i uses √(2T/m); assume Tang's ρ_i=√(2T/m)/Ω, so **b_Tang = b_GX = (k⊥ρ_{i,GX})²**. Verify before use.
- **Benchmarks:** analytic only. (a) Near-marginal real frequency ω_r = ω_{*pi}/2 in the MHD-like strongly driven, low-k regime, with trapped particles neglected. (b) The ideal-MHD limit ω_*/ω→0 with adiabatic index γ=(7/4+τ)/(1+τ). Tolerance: ω_r/ω_{*pi} = 0.5 ± 0.05 in the strongly driven regime (cf. Aleynikova Fig. 5).
- **Refs to fetch:** Rosenbluth & Sloan, Phys. Fluids **14**, 1725 (1971), DOI 10.1063/1.1693670. Chu, Chu, Guest, Hsu, Ohkawa, PRL **41**, 247 (1978), DOI 10.1103/PhysRevLett.41.247. Frieman, Rewoldt, Tang, Glasser, Phys. Fluids **23**, 1750 (1980), DOI 10.1063/1.863201. Tang et al. NF 16, 191 (1976), DOI 10.1088/0029-5515/16/2/002. Rutherford, Chen, Rosenbluth PPPL-1418 (1978).

---

## 3. Hastie & Hesketh, Nucl. Fusion 21, 651 (1981)

- **Citation:** R. J. Hastie, K. W. Hesketh, "Kinetic modifications to the MHD ballooning mode", Nucl. Fusion **21**(6), 651–656 (1981). DOI 10.1088/0029-5515/21/6/001 (not printed on the PDF; verify).
- **Model:** large-aspect-ratio circular s–α, β~ε², collisionless. Numerical shooting on Tang et al.'s intermediate-frequency equation (2.3). Even modes, F′(0)=0, F(θ→∞)→0. Second-order RK. Complex ω/ω_{*i} iterated.
- **Key equations:**
  - (2.1) \(\frac{\partial}{\partial\theta}[1+(S\theta-\alpha\sin\theta)^2]\frac{\partial F}{\partial\theta}+\alpha F[\cos\theta+(S\theta-\alpha\sin\theta)\sin\theta]+\frac{\omega^2}{\omega_A^2}F[1+(S\theta-\alpha\sin\theta)^2]=0\), \(S=rq'/q\), \(\alpha=-2p'Rq^2/B^2\), \(\omega_A^2=V_A^2/R^2q^2\).
  - (2.2) low-frequency version with \(\Omega_{mj}=1-\frac{\omega_{*j}}{\omega}(1+m\eta_j)\), \(\omega_{*j}=\frac{nT_j}{e_jBr_n}\), \(\frac1{r_n}=\frac1n\frac{dn}{dr}\), \(\beta_i=2p_i/B^2\), and a trapped term \(-\frac{\beta_iq^2}{2}(\sum_j\frac{T_j}{T_i}\Omega_{2j})\frac{15}{8}[\ldots]\).
  - (2.3) \(\frac{\partial}{\partial\theta}[1+(S\theta-\alpha\sin\theta)^2]\frac{\partial F}{\partial\theta}=\frac{\omega^2}{\omega_A^2b_0}HF\). (2.4) H as printed (Q, Q′, R, Ω_{0e}, Ω_{1e}, β_i, β_e, τ). \(\omega_{*i}^T=\omega_{*i}(1-\frac32\eta_i+\frac{mv^2}{2T_i}\eta_i)\). \(\omega_{Di}=\frac{mv_\parallel^2}{T}\omega_\kappa+\frac{mv_\perp^2}{2T}\omega_B\). \(\omega_\kappa=\omega_{*i}\frac{r_n}{R}[\cos\theta+(S\theta-\alpha\sin\theta)\sin\theta]\). \(\omega_B=\omega_\kappa+\omega_{*i}\frac{r_n}{R}\frac{\alpha}{2q^2}\). \(b=b_0[1+(S\theta-\alpha\sin\theta)^2]\), \(b_0=\frac{T}{m\omega_c^2}(\frac{nq}{r})^2\), \(z=(b\,mv_\perp^2/T)^{1/2}\).
  - (2.5) same with Q,Q′,R expanded to O((ω_D/ω)²) (no Landau resonance). (2.6) b≪1 form: \(\ldots+\frac{\omega^2}{\omega_A^2}\Omega_{1i}F[1+(\ldots)^2]+\alpha F[\ldots]-\frac{\beta_iq^2}{2}[7\Omega_{2i}+4\tau\frac{\Omega_{1i}^2}{\Omega_{0e}}]F[\cos\theta+(S\theta-\alpha\sin\theta)\sin\theta]^2=0\). The ω_*/ω→0 limit gives MHD with γ=(7/4+τ)/(1+τ).
  - **β–α relation (text, p. 654):** \(\beta=(r_n/Rq^2)\,\alpha\,(1+\tau)[1+\eta_i+\tau(1+\eta_e)]^{-1}\). Checked: S=0.6, α=0.8 → β=0.020, and S=0.4, α=0.54 → β=0.0135, both as printed.
  - k a_i ≡ b_0^{1/2}, k = nq/r. Note that the \(\omega_B\neq\omega_\kappa\) split is the consistent-drift choice (∇B drift includes the α/2q² correction), the same as GX `beta_prime` in s-α.
- **Normalizations to GKX:** Here \(ka_i=k_\theta\rho_i\) with \(a_i^2=T/(m\omega_c^2)\), i.e. **ρ_i with v_ti=√(T/m)**, which is exactly GX `ky` (at θ=0). r_n/R = L_n/R, so GX: R/L_n = 10 → with `geo s-alpha`, R/a normalization choose a=L_n... Simplest: set `Rmaj/a`… use GX lengths in R: fprim = R/L_n = 10, tprim = η·fprim = 10 (both species), q=√2, shat=0.6, shift(α)=0.8, beta (per β_i=2p_i/B²) = 0.02 in Hastie's β. Note β here is "local β" with p=p_i+p_e. **Map β_Hastie → GX beta_ref:** the α relation uses total pressure: α = q²R β′ with β=2p/B². β_total/(1+τ) = β_i = GX beta_ref (n_ref=n_i, T_ref=T_i). Resolve which β is meant by recomputing α from GX inputs: α = q² beta_ref Σ(fprim+tprim) = 2×0.01×(20+20)=0.8 ✓. So **GX beta_ref = 0.01** for Fig. 2, and 0.00675 for Fig. 4. Consistent with β_Hastie = total β.
  Frequency: ω_A = V_A/(Rq). In GX units (v_ti/R): ω_A = v_ti/(q√(beta_ref/2)·...) → ω_A/(v_ti/R) = √(2/beta_ref)/q × √(m_i... ) i.e. V_A/v_ti = √(2/β_i) for v_ti=√(T/m). So ω_A = (1/q)√(2/β_i) v_ti/R = (1/1.414)·√200 = 10.0 v_ti/R for Fig. 2 (β_i=0.01). For Fig. 4 (β_i=0.00675), ω_A = 12.17 v_ti/R. And ω_{*i}=k_y ρ_i v_ti/L_n = 10 k_y in v_ti/R.
- **Benchmarks (reading ± uncertainties):**
  - **Fig. 1 ideal-MHD (S,α) boundary** (first & second stability): first boundary rises from α≈0.35 at S≈0.05 to α≈0.6 at S=1 (±0.05). The second-stability boundary goes through (α≈1, S≈0.2), (α≈2, S≈0.5), (α≈3, S≈1.1). Unstable region between them.
  - **Fig. 2 (S=0.6, α=0.8, q=1.414, η_e=η_i=τ=1, r_n/R=0.1, β=0.02), γ/ω_A vs ka_i:** curve (a) full Eq. (2.3): ka_i=0 → ≈0.33. 0.1 → 0.34. 0.2 → 0.33. 0.3 → 0.29. 0.4 → 0.21. 0.45 → 0.13. 0.5 → ≈0.05. 0.6 → ≈0.03. 0.8 → ≈0.02 (residual Landau-driven mode). Curves (b) and (c) → 0 near ka_i≈0.5–0.55. Uncertainty ±0.02 in γ/ω_A, ±0.02 in ka_i.
  - **Fig. 3 same parameters, Re(ω/ω_A) vs ka_i:** (a) ≈0.1 at 0.05. 0.2 at 0.1. 0.4 at 0.25. 0.55 at 0.4. 0.75 at 0.55. 1.0 at 0.8 (±0.03). Text: Re(ω/ω_{*i}) "slightly exceeds unity" throughout, in the ion diamagnetic direction. Consistency: ω_{*i}/ω_A = ka_i·(q/r_n... ) → with numbers above ω_{*i}=10ka_i v_ti/R and ω_A=10 v_ti/R ⇒ ω_{*i}/ω_A = ka_i. So Re(ω)/ω_A ≈ 1.0–1.25 × ka_i, which is consistent with the figure ✓ (this also confirms the ω_A mapping).
  - **Fig. 4 (S=0.4, α=0.54, β=0.0135, weakly MHD unstable), γ/ω_A vs ka_i:** 0 at ka_i→0 (≈0.03). Rises to ≈0.30 at ka_i≈0.25. ≈0.22 at 0.4. ≈0.1 at 0.5. ≈0.03 at 0.6 (±0.02). This shows FLR destabilization at small ka_i.
  - **Fig. 5 (Eq. 2.5, no Landau): (S,α) stability boundaries for ka_i=0.1,0.2,0.3,0.4.** The unstable region recedes with ka_i. The ka_i=0.1 boundary is slightly outside the MHD boundary. The ka_i=0.4 nose is near α≈0.62, S≈0.55 (±0.05).
- **Supports:** **B-KBM-HH (primary tokamak KBM linear benchmark).** GKX run: s-α, q=√2, shat=0.6, α(shift)=0.8, R/L_n=10, R/L_T=10 both species, T_i=T_e, beta_ref=0.01, collisionless, kinetic electrons (large m_i/m_e, e.g. hydrogen). Include δB∥ (Tang retains it). The paper neglects trapped particles (ε→0), so use GX s-α with ε (`eps`) small (e.g. 0.01–0.05) or set trapping off. Tolerance: γ/ω_A within ±15% for ka_i≤0.3 (figure reading plus model differences: GK includes electron FLR/trapping/electron Landau). ω_r/ω_{*i} ∈ [1.0,1.3]. Qualitative: strong stabilization by ka_i≈0.5 with a small residual γ≲0.05 ω_A.
- **Refs to fetch:** Chu et al. PRL 41, 247 (1978) DOI 10.1103/PhysRevLett.41.247. Seyler & Friedberg Phys. Fluids 23, 331 (1980) DOI 10.1063/1.862975. Lortz & Nührenberg Phys. Lett. A 68, 49 (1978) DOI 10.1016/0375-9601(78)90752-0. Mercier IAEA 1978 vol. 1, 701. Rutherford, Chen, Rosenbluth PPPL-1418.

---

## 4. Antonsen & Lane, Phys. Fluids 23, 1205 (1980)

- **Citation:** T. M. Antonsen Jr., B. Lane, "Kinetic equations for low frequency instabilities in inhomogeneous plasmas", Phys. Fluids **23**(6), 1205–1214 (1980). DOI 10.1063/1.863121.
- **Model:** general (including nonaxisymmetric) geometry, all orders of k⊥ρ, ω/Ω ~ ρ/a ≪ 1. Fields: \(\hat\phi\), \(\hat\psi\) (A∥-like), \(\hat\sigma\) (compressional δB∥). Gauge ∇·A=0. Derives the gyrokinetic equation (24) and a variational form (32). Sec. V treats fluid-like ions and adiabatic electrons (\(k_\parallel v_{ti}\ll\omega\ll k_\parallel v_{te}\)). Eqs. (39a–c) and quadratic form (40). Appendix A covers stellarator ray/EBK quantization.
- **Key equations:** (1) \(\mathbf B_0=-\psi'(V)\nabla V\times\nabla[\zeta-q(V)\theta]\). (13) \(z=\hat z e^{iS}\). (15) \(S=n^0(\zeta-\int^\theta q_l d\theta')+\bar S(\psi)\). (16) \(z=\sum_l\hat z(\psi,\theta-2\pi l,\zeta)\). (18) \(\tilde{\mathbf A}=\mathbf b\hat\psi-i\hat\sigma\mathbf b\times\nabla S+\hat l\nabla S\). (19) \(\hat{\mathbf B}\cong-i\hat\psi\mathbf b\times\nabla S+\mathbf b|\nabla S|^2\hat\sigma\).
  (23) \(\hat g=\hat h-\frac1{B_0}\frac{\partial F_0}{\partial\mu}[J_0(\frac{v_\perp|\nabla S|}{\Omega})(q\hat\phi-\frac{v_\parallel}{c}q\hat\psi)+q\hat\sigma\frac{v_\perp|\nabla S|}{c}J_1(\ldots)]\). (24) \(-i(\omega-\omega_d+iv_\parallel\mathbf b\cdot\nabla')\hat h=\int\frac{d\xi}{2\pi}e^{-iL}\mathrm{st}(\hat f_0)+i\omega(\frac{\partial F_0}{\partial\epsilon}-\frac{\mathbf B_0\times\nabla S\cdot\nabla' F_0}{B_0 m\Omega\omega})[J_0(\ldots)(q\hat\phi-\frac{v_\parallel}{c}q\hat\psi)+\frac{q\hat\sigma|\nabla S|v_\perp}{c}J_1(\ldots)]\). (25) \(\omega_d=\nabla S\cdot\mathbf v_d\).
  (29)–(31) field equations. (32) variational principle. (34) \(\hat\psi=-i(c/\omega)\mathbf b\cdot\nabla\hat\chi\). (36) trapped-electron response. (39a–c) three-field system. (40) quadratic form. (44) pitch-angle operator \(\mathrm{st}_a=2\frac{\nu(\epsilon)}{B}(1-yB)^{1/2}\frac{\partial}{\partial y}y(1-yB)^{1/2}\frac{\partial}{\partial y}\). App. A (A7) ray equations, (A8) \(\oint S_\alpha d\alpha+S_VdV=2\pi n+m\pi/2\).
- **Normalizations:** Gaussian. \(\omega_*^e=c T_e\mathbf b\times\nabla S\cdot\nabla' n_0/(eB_0n_0)\). No numerical results.
- **Benchmarks:** none numeric. It supports the **structure** tests: (i) GK→ideal-MHD energy principle in the \(v_\perp|\nabla S|/\Omega,\ \omega_*/\omega,\ \omega_d/\omega\to0\) limit (Kruskal–Oberman electrons, CGL-like ions, ξ∥=0). (ii) The compressional field σ must be kept for the correct KBM limit (same conclusion as Aleynikova §3.2). (iii) Ballooning in nonaxisymmetric geometry: the local eigenvalue ω_l(α,V,S_α,S_V), relevant to the GKX stellarator flux tube (dependence on α and θ₀).
- **Refs to fetch:** Rutherford & Frieman Phys. Fluids 11, 569 (1968) DOI 10.1063/1.1691954. Taylor & Hastie Plasma Phys. 10, 479 (1968) DOI 10.1088/0032-1028/10/5/301. Hastie, Hesketh & Taylor NF 19, 1223 (1979) DOI 10.1088/0029-5515/19/9/006. Strauss Phys. Fluids 22, 1079 (1979). W. M. Tang NF 18, 1089 (1978) DOI 10.1088/0029-5515/18/8/006.

---

## 5. Tsai & Chen, Phys. Fluids B 5, 3284 (1993)

- **Citation:** S.-T. Tsai, L. Chen, "Theory of kinetic ballooning modes excited by energetic particles in tokamaks", Phys. Fluids B **5**(9), 3284–3290 (1993). DOI 10.1063/1.860624.
- **Model:** large aspect ratio, CHT s–α, core (C) plus energetic (E) species. Ordering β_C~ε, β_E~ε², T_C/T_E~ε², n_E/n_C~ε³, k_θρ_C~ε^{3/2}, k_θρ_E~ε^{1/2}, ω/ω_A~ε. Asymptotic matching of inertial (θ~ε^{-1}) and ideal (θ~1) regions gives a fishbone-like dispersion relation. δB∥ from pressure balance (1) \(B\delta B_\parallel\simeq4\pi n_ie\omega_{*p}\delta\phi/\omega\).
- **Key equations:** (2) \(\big(\frac{d}{d\theta}f\frac{d}{d\theta}+\Omega^2f+\alpha g\big)\delta\phi-\sum_j\eta_j\langle g\Omega_{dj}J_0(k_\perp\rho)\delta G\rangle_j=0\), \(\Omega^2\equiv\omega(\omega-\omega_{*pi})/\omega_A^2\), \(f=1+(s\theta-\alpha\sin\theta)^2\), \(g=\cos\theta+(s\theta-\alpha\sin\theta)\sin\theta\), \(\alpha=q^2R_0\beta'\), \(\eta_j=4\pi e_jq^2R_0^2\omega/c^2\), \(\omega_A=V_A/qR_0\).
  (4) \(\big(\frac{d^2}{d\theta^2}+\Omega^2+\frac{\alpha\cos\theta}{f}-\frac{(s-\alpha\cos\theta)^2}{f^2}\big)\delta\psi-\ldots=0\), δψ=f^{1/2}δφ. (6) \(\delta\psi_0=e^{i\Omega|\theta|}\). (10) \(-i\Omega+\delta W_f+\delta W_K=0\). (11) \(\delta W_f=\frac12\int d\theta[|\frac{d\delta\psi_I}{d\theta}|^2+(\frac{(s-\alpha\cos\theta)^2}{f^2}-\frac{\alpha\cos\theta}{f})|\delta\psi_I|^2]\).
  Gap mode unstable for (24) \(\omega_{*pi}>\omega_r>\omega_{*pi}/2\). Growth (23) \(\gamma\simeq\frac{2\delta W_{Ki}(\delta W_f+\delta W_{Kr})}{\omega_{*pi}-2\omega_r}\omega_A^2\). Continuum mode threshold (19) \(\delta W_{Ki}=\Omega_r\).
  Slowing-down beam (39). Circulating: (42) \(\omega_r\simeq\{\omega_{*pi}+[\omega_{*pi}^2-4\omega_A^2(\delta W_r^2-\delta W_{Ki}^2)]^{1/2}\}/2\). (45) \(\omega_r\simeq0.834\omega_{tm}\). (46) \(\alpha_{EC}\simeq0.57s\omega_{tm}^2/\omega_A\omega_r\). Trapped: (52) ω_r≈0.78ω̄_dm, (53) α_EC≈0.14ω̄_dm/ω_A, (59) ω_r≈0.834ω_bm, (60) α_EC≈0.4(s/θ_T)ω_bm²/(ω_rω_A).
- **Normalizations:** α=q²R₀β′ (β total, core); ω_A=V_A/(qR₀). The GKX mapping is as for Hastie–Hesketh.
- **Benchmarks:** the analytic gap window \(\omega_{*pi}/2<\omega_r<\omega_{*pi}\) for the MHD-unstable KBM (no energetic particles). This is the **ω_r ≥ ω_{*pi}/2 check** for B-KBM tests. The EP results are outside current GKX scope (they need a slowing-down distribution). Numerical constants (0.834, 0.57, 0.14, 0.4) are only relevant if an EP species is ever added. Tolerance: qualitative/window.
- **Refs to fetch:** Chen, White, Rosenbluth PRL 52, 1122 (1984) DOI 10.1103/PhysRevLett.52.1122. Biglari & Chen PRL 67, 3681 (1991) DOI 10.1103/PhysRevLett.67.3681. Fu & Cheng Phys. Fluids B 4, 3722 (1992) DOI 10.1063/1.860328. Chen & Tsai Plasma Phys. 25, 349 (1983) DOI 10.1088/0032-1028/25/4/001. Hastie, Chen, Ke, Tsai, Chen Chin. Phys. Lett. 4, 561 (1987).

---

## 6. "Aleynikova_Kinetic.pdf": **not** the PoP 24, 092106 paper

- **What the PDF actually is:** K. Aleynikova, A. Zocco, P. Xanthopoulos, P. Helander, C. Nührenberg, "Kinetic ballooning modes in tokamaks and stellarators" (preprint; published J. Plasma Phys. **84**, 745840602 (2018), DOI 10.1017/S0022377818001186; verify). The PDF has no DOI printed. **Aleynikova & Zocco, PoP 24, 092106 (2017) is its Ref. [17]** and was fetched later; DOI 10.1063/1.5000052 (verified from the PDF). It is summarized in REPORT.md §B2.
- **Model:** GENE linear EM flux tube with δB∥. Hydrogen, m_i/m_e=1836, T_i=T_e. Tokamak: r/R=0.26, q=1.3, ŝ=0.745. W7-X: r/R≈0.095, q=1.1, ŝ≈−0.1 (KJM standard, TEH high-mirror). β_GENE=β_{i,e}=β_total/2=8πn_{i0}T_ref/B_ref². The equilibrium is varied consistently with β, and pressure gradients are consistent. Lengths in a, frequencies in c_s/a, k in ρ_s.
- **Key equations:** (1) \(\frac{1}{\beta_iB}\frac{v_{ths}^2/l_c^2}{\omega^2}\frac{\partial}{\partial z}bB\frac{\partial\phi}{\partial z}=K\phi\). (2) K (as Tang (3.39)) with \(\alpha_{n,j}=1-(\omega_{*i}/\omega)(1+n\eta_j)\), \(\omega_{*i,e}=\frac12k_y\rho_{i,e}v_{th}/L_n\), \(\omega_B=(\mathbf k_\perp\rho_s/2)\cdot v_{ths}\hat{\mathbf b}\times\nabla B/B\), \(\omega_\kappa=(\mathbf k_\perp\rho_s/2)\cdot v_{ths}\hat{\mathbf b}\times(\hat{\mathbf b}\cdot\nabla\hat{\mathbf b})\), \(v_{thi}=\sqrt{2T_i/m_i}\), \(b=k_\perp^2v_{thi}^2/2\Omega_i^2B\).
  (3) \(\frac1{\beta_i}\frac{v_{thi}^2}{\omega^2l_c^2}\frac{\partial}{\partial z}bB\frac{\partial\phi}{\partial z}=-\frac{2\omega_\kappa\omega_p}{\omega^2}\phi-b[1-\frac{\omega_{*i}}{\omega}(1+\eta_i)]\phi\), \(\omega_p=\omega_{*i}(1+\eta_i)-\omega_{*e}(1+\eta_e)\). ω_κ≠ω_B at finite β is required. (4) Boozer field-aligned form with \(b=-\frac12\frac{\rho_i^2}{a^2}\frac{B_a^2}{B^2}\frac1{\sqrt{g_B}}\sum_{i,j}\partial_i\sqrt{g_B}g^{ij}\partial_j\).
  The necessary condition for instability in the high-∇T regime is **Re ω = ω_{pi}/2**.
- **Benchmarks (read ±5–10%):**
  - Fig. 5 (W7-X KJM, β=3%, a/L_T=7, a/L_n=0): γ(a/c_s) ≈3.75 at k_yρ_s→0. 3.6 at 0.3. 3.0 at 0.6. 2.4 at 0.8. 1.4 at 1.0. Drops to ≈0.6 at 1.1 (switch to TEM). ω ≈ ω_pi/2 = (k_yρ_s)(a/L_T)/2… the line reaches 3.5 at k_yρ_s=1.0?? Read: the blue line passes ~(1.0, 3.5)... ω_pi/2 in c_s/a with GENE ω_* = ½k_yρ v_th/L ⇒ ω_{pi} = k_yρ_s(a/L_n+a/L_T)·(c_s/a)·(1/√2·√2)… keep it empirical: slope ≈ 3.5 per unit k_yρ_s (±0.2) for a/L_T=7 ⇒ ω_{pi}/2 ≈ (a/L_T)k_yρ_s/2 ✓ (7/2=3.5). So in GX units ω_r = (ky·tprim)/2 (sign flipped for GX).
  - Fig. 6a (a/L_T=5): γ≈2.2 at k_y→0, 1.6 at 0.6, 0.5 at 1.0. ω≈ω_pi/2 (slope 2.5). Fig. 6b (a/L_T=3): γ≈0.28 at 0, min 0.15 at 0.3, ω deviates above ω_pi/2.
  - Fig. 1/2: without δB∥ the KBM γ is lower by up to a factor ~6. Tokamak Fig. 2b (a/L_T=7, β=3%): γ≈1.95 at k_y→0, ≈1.5 at 0.4, ≈0.3 at 0.9. With δB∥, ω stays at ω_pi/2 only at low k_y.
  - Fig. 7 (a/L_T=2, a/L_n=0, max over k_yρ_s∈[0.05,0.8]): KBM critical β ≈0.65% (TEH), ≈1.9% (KJM bean), ≈2.2% (KJM triangular). ω_KBM ≈0.05 c_s/a = ω_pi/2 at k_yρ_s=0.05 ✓.
  - Fig. 8: KBM β_crit vs k_yρ_s. KJM: 1.9 (0.05), 2.05 (0.1), 2.4 (0.2), 2.85 (0.3), 2.95 (0.4) %. TEH: 0.65, 0.97, 1.4, 2.05, 2.35 % (at 0.05…0.4) and a second point ≈0.15–0.3% at 0.4. iMHD lines: KJM ≈2.7%, TEH ≈0.07%.
  - Fig. 9 (CBC s–α GENE, a/L_T=2.2/2.48/2.70 [a units]): the KBM onset β ≈1.5%/1.45%/1.4%.
- **Supports:** B-KBM-ωpi2 (strongly driven frequency law ω_r=ω_{pi}/2, flat density). Geometry-agnostic, so it is ideal for GKX tokamak **and** VMEC stellarator runs. Tolerance: |ω_r/(ω_{pi}/2)−1|<10% for k_yρ≤0.5 at a/L_T=7, β=3%. γ(k_y→0) within 15% of 3.75 (W7-X) only if geometry/equilibrium consistency is matched (hard; treat as a soft target). Also B-dBpar: γ ratio with/without δB∥ >2.
- **Refs to fetch:** Aleynikova & Zocco PoP 24, 092106 (2017), DOI 10.1063/1.5000052 (now read; see REPORT.md). Zocco, Helander & Connor PPCF 57, 085003 (2015) DOI 10.1088/0741-3335/57/8/085003. Pueschel & Jenko PoP 17, 062307 (2010) DOI 10.1063/1.3435280. Ishizawa et al. JPP 81, 435810203 (2015) DOI 10.1017/S0022377815000276. Roberts & Taylor PRL 8, 197 (1962) DOI 10.1103/PhysRevLett.8.197. Belli & Candy PoP 17, 112314 (2010) DOI 10.1063/1.3495976. Nührenberg NF 56, 076010 (2016) DOI 10.1088/0029-5515/56/7/076010.

---

## 7. Pueschel, Kammerer & Jenko, Phys. Plasmas 15, 102310 (2008)

- **Citation:** M. J. Pueschel, M. Kammerer, F. Jenko, "Gyrokinetic turbulence simulations at high plasma beta", Phys. Plasmas **15**, 102310 (2008). DOI 10.1063/1.3005380.
- **Model:** GENE (local), collisionless, ŝ–α with **α=0** (α_MHD set to zero), CBC: R/L_T=6.89 (both), R/L_n=2.22, T_i=T_e, ŝ=0.786, q=1.4, ε=r/R=0.18, m_e/m_i=5.669×10⁻⁴ (hydrogen). (1) \(\beta\equiv\beta_e=8\pi n_{e0}T_{ref}/B_{ref}^2\). Linear: L_y=2π/k_y, N_x=24, N_z=24, N_v∥=96, N_μ=16, v∥max=3, μmax=9. Units c_s/R and ρ_i (= ρ_s since T_e=T_i). δB∥ presumably off (not stated).
- **Key equations:** \(\beta_{crit,MHD}=0.6\hat s/[q_0^2(2\omega_n+\omega_{Ti}+\omega_{Te})]=1.32\%\) (I evaluated it: 1.3206%, α_MHD=0.472). (2) \(\omega_s=d^2\Phi_{zon}/dx^2\). (3) \(\tilde s=q_0\frac{R}{B_{ref}}\frac{dB_y}{dx}\). (4)–(5) Rechester–Rosenbluth \(Q_e^{em}=\langle\tilde q_{e\parallel}\tilde B_x\rangle/B_{ref}\). (6) \(\chi_e^{em}=\chi_{e\parallel}\langle(\tilde B_x/B_{ref})^2\rangle\). (7) \(\chi_{e\parallel}\approx\frac1{k_\parallel}(T_e/m_e)^{1/2}\sim q_0R(T_e/m_e)^{1/2}\).
- **GKX mapping:** directly GX CBC inputs: `tprim=6.89` (per R: GX normalized by a with R/a=… use a=R convention or tprim=2.49,fprim=0.8 with R/a=2.77), `shat=0.786`, `q=1.4`, `eps=0.18`, `shift=0` (α=0!), kinetic electrons `mass=5.669e-4`, **GX beta = β_e = 0.0127 etc.** (n_ref=n_e, T_ref=T_e; same definition). k_y ρ_i(GENE, c_s) = GX `ky` (T_i=T_e). γ in c_s/R = v_ti/R. ω sign: GENE positive = ion diamagnetic; GX flips it.
- **Benchmarks (k_yρ=0.2):**
  - Fig. 1 (GENE △ / GS2 □, dominant mode): γ(c_s/R): β=0 → 0.39. 0.4% → 0.33. 0.8% → 0.24. 0.95% → ITG→TEM transition. TEM γ≈0.17 flat (1.0–1.25%). KBM: 1.3% → 0.2. 1.4% → 0.5. 1.5% → 0.71. 1.6% → 0.87. 1.7% → 1.0. 1.8% → 1.13 (±0.03). ω(c_s/R): ITG 0.52→0.65 (β 0→0.9%). TEM ≈ −0.35. KBM ≈2.4 at 1.3% decreasing to ≈1.95 at 1.8% (GENE). GS2 KBM ω ≈0.1–0.25 lower (2.15 at 1.3%).
  - Text: ITG→TEM at β=0.95%. KBM dominant at β=1.27%.
  - **Fig. 2 subdominant KBM (eigenvalue solver): β_crit=1.14%** (γ linear in β: ≈−0.03 at 1.125%, 0 at 1.14%, 0.11 at 1.185%; slope ≈2.9 (c_s/R)/%). 14% below β_MHD=1.32%.
  - β_crit^dom=1.26%, β_crit^extrapol=1.21% (k_y=0.2). With consistent α_MHD: 1.26% / 1.22%.
  - Fig. 5, β_crit(k_yρ): dom: 1.37 (0.1), 1.29 (0.15), 1.26 (0.2), 1.28 (0.25), 1.34 (0.3), 1.41 (0.35), 1.52 (0.4). extrapol: 1.34, 1.25, 1.21, 1.21, 1.27, 1.32, 1.41 % (±0.01).
  - Fig. 3 (ω_n=3, ω_Ti=4, ω_Te=6): marginally stable tail. γ≈0 near β≈1.51–1.52%, 0.02 at 1.55, 0.07 at 1.6, 0.135 at 1.65, 0.2 at 1.7.
  - Fig. 6 (k_y=0.2) α_crit^extrap: vs ω_n (0.75–4): 0.420→0.439. vs ω_Ti (3–11): 0.56,0.472(5),0.43(7),0.41(9),0.403(11). vs ω_Te (3–11): 0.404→0.453. β_crit: ω_n 1.40(0.75)→1.03(4). ω_Ti 1.98(3)→0.93(11). ω_Te 1.45(3)→1.05(11) %. The k_y→0 check: β_crit=1.87% (α_crit=0.526) for low ω_Ti.
  - Nonlinear: Q_i^es vs β (Fig. 8) ≈165 (β=0.1%), 175 (0.1–0.2), 140 (0.4), ~105 (0.6), ~30 (1.0), ~8 (1.2) in v_tiρ_i²p_0/R². Transition at β≈1.26%. Nonlinear: N_x=192, N_ky=24, N_z=48 (β≥0.7%), L_y=125.66, L_x=101.78. Q_e^em ∝ β² nonlinear, ∝ β linear. η_ITG=0.625±0.026.
  - Minor: ŝ=0.636 → ω=0.627. 0.786 → 0.656. 0.936 → 0.694 (ITG, β presumably 0.8%, k_y=0.2).
- **Supports:** **B-KBM-CBC (primary GK code-to-code benchmark):** GX/GKX CBC β-scan at k_y=0.2 (α=0). Target KBM γ=0.5±0.05 at β=1.4%, 0.87±0.07 at 1.6%. KBM ω≈2.1–2.4 (codes differ by ~10%, so use ±12%). Dominant-mode transition β ∈ [1.2,1.3]%. Subdominant β_crit=1.14±0.03% (needs an eigen/Krylov solver; GKX Krylov path). Also ITG γ(β=0)=0.39±0.03 and ω=0.52 (sign flipped in GX). Nonlinear Q is a soft ±30% target (known GENE/GYRO spread).
- **Refs to fetch:** Candy PoP 12, 072307 (2005) DOI 10.1063/1.1954138. Kammerer, Merz, Jenko PoP 15, 052102 (2008) DOI 10.1063/1.2909618. Pueschel & Jenko PoP 17, 062307 (2010) DOI 10.1063/1.3435280. Dimits et al. PoP 7, 969 (2000) DOI 10.1063/1.873896. Zonca, Chen, Dong, Santoro PoP 6, 1917 (1999) DOI 10.1063/1.873449. Dong, Chen, Zonca NF 39, 1041 (1999) DOI 10.1088/0029-5515/39/8/310. Chen, Parker et al. NF 43, 1121 (2003). Snyder PhD (1999). Parker et al. PoP 11, 2594 (2004) DOI 10.1063/1.1690302.

---

## Cross-paper summary of proposed benchmarks

| ID | Source | Observable | Target | Tol. |
|---|---|---|---|---|
| B-MHD-1 | CHT78 Fig.1, HH81 Fig.1 | s–α ideal ballooning α_crit(s) | α_crit(0.8)≈0.60, (0.4)≈0.44 | ±0.05 (figure). ±2% vs own ODE |
| B-KBM-HH | HH81 Figs 2–4 | γ/ω_A, ω/ω_A vs ka_i (GX beta_ref=0.01, α=0.8, s=0.6, q=√2, R/L_n=R/L_T=10) | γ/ω_A≈0.33 @0.1–0.2, stable ≲0.5 | ±15% |
| B-KBM-ωpi2 | Tang80, TC93, Aleynikova Figs 5–7 | ω_r/ω_{*pi} in strongly-driven low-k KBM | 0.5 (window 0.5–1 for gap mode) | ±10% |
| B-KBM-CBC | PKJ08 Figs 1,2,5 | CBC β-scan k_y=0.2, α=0 | β_crit^sub=1.14%, dom=1.26%, γ(1.6%)=0.87 c_s/R | ±3%/±5%/±10% |
| B-dBpar | Aleynikova Figs 1–2, Antonsen–Lane | KBM γ with/without δB∥ | ratio up to ~6 | qualitative (>2) |
| B-MHD-β | PKJ08 | β_MHD = 0.6ŝ/[q²(2ω_n+ω_Ti+ω_Te)] | 1.32% | closed-form check |

Open items: (1) verify the Hastie–Hesketh DOI and the JPP 2018 DOI for the Aleynikova PDF. (2) Confirm the GX sign convention for ω against an existing CBC golden before encoding the sign. (3) Pueschel 2008 does not state whether δB∥ was on. Check against Pueschel & Jenko 2010.
