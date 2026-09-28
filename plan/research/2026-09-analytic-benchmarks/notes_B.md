Appendix B of REPORT.md.

# notes_B — zonal-flow residual, GAM, ITG thresholds, EM-ITG (six papers, read in full)

GKX/GX convention used for mapping throughout:
v_ti^GX = sqrt(T_i/m_i), rho_i^GX = v_ti^GX/Omega_i, lengths in a (or R, per deck), t in a/v_ti,
tprim = a/L_T, fprim = a/L_n, eps = r/R, beta_ref = 8 pi n T / B^2, tau_fac = T_i/T_e (GX "tau_fac" = T_i/T_e for Boltzmann electrons).
WARNING: three of these papers use v_t = sqrt(2T/m) (Romanelli, Biglari et al., Sugama-Watanabe eqs.), two use sqrt(T/m) (Kim-Horton-Dong; Sugama-Watanabe Fig. 1 axis). Convert explicitly.

Closed-form evaluations below were done with numpy (script inline in this session; trivial to reproduce).

---------------------------------------------------------------------------------------------------
## 1. Rosenbluth & Hinton 1998

Citation: M. N. Rosenbluth and F. L. Hinton, "Poloidal Flow Driven by Ion-Temperature-Gradient Turbulence in Tokamaks", Phys. Rev. Lett. 80, 724–727 (26 Jan 1998). DOI 10.1103/PhysRevLett.80.724 (printed code S0031-9007(97)05109-0).

Model/assumptions: linear collisionless electrostatic gyrokinetics (Frieman-Chen) for the n=0 (axisymmetric) response to a source S_k; Maxwellian F_0; eikonal S=S(psi), k_perp = grad S; times >> bounce time; electrons adiabatic (no source); ion source O(k_perp^2 rho^2); expansion in small gyroradius and banana width, e^{iQ}J_0 ~ 1 + iKv_par/B - (Kv_par/B)^2/2 - (k_perp rho)^2/4; large-aspect-ratio circular geometry; deep banana regime (collisions neglected); low-energy resonant particles neglected. Result: n=0 flows NOT damped by collisionless processes — only shielded by classical + neoclassical polarization.

Key equations (numbers as printed):
- (1) $\frac{\partial g_k}{\partial t}+v_\parallel\hat b\cdot\nabla g_k+i\omega_D g_k=\frac{e}{T}F_0J_0\frac{\partial\phi_k}{\partial t}+S_kF_0$
- (2) $\omega_D=(\vec v_d\cdot\nabla\psi)S'(\psi)=Kv_\parallel\hat b\cdot\nabla(v_\parallel/B)$, $K=(mcI/e)S'(\psi)$, $I\equiv RB_\phi$
- (3) $-\frac{e}{T_i}n_0\phi_k+\int d^3v\,J_0g_{ik}=\frac{e}{T_e}n_0\phi_k+\int d^3v\,g_{ek}$
- (7) $g_k=e^{-iQ}\left[\frac{e}{T}\overline{(e^{iQ}J_0\phi_k)}+\overline{(e^{iQ}R_k)}\right]F_0$, $Q=Kv_\parallel/B$, $R_k=\int dt\,S_k$
- (13) $\frac{e}{T_i}\phi_k=\frac{1}{\mathcal D}\oint\frac{dl_p}{B_p}\int d^3v\,F_{0i}\{R_{\rm even}+iK[v_\parallel/B-\overline{(v_\parallel/B)}]R_{\rm odd}\}$
- (14) $\mathcal D=\oint\frac{dl_p}{B_p}\int d^3v\,F_{0i}\{K^2[\overline{(v_\parallel/B)^2}-\overline{v_\parallel/B}^2]+\overline{(k_\perp^2\rho^2)}/2\}$  [neoclassical + classical polarization]
- (15) $\frac{e\phi_k}{T_i}=(1+1.6q^2/\epsilon^{1/2})^{-1}\int dt\,S_{ik}/(k_\perp^2a_i^2)$, $a_i^2=(T_i/m_i)/\Omega_i^2$, $\epsilon=r/R$, $q=\epsilon B/B_p$
- (18) $\langle|\phi_k|^2\rangle\simeq2\tau_c\langle|S_k|^2\rangle|\mathcal K|^2t$ (linear growth of mean-square potential, no linear damping)
- (19) model: $\partial_t|\phi_k|^2=A_k|S|^2-B_k|\phi_k|^2-C_k|\phi_k|^2|S|^2-D_k\nu_{ii}|\phi_k|^2$
- (20) initial value: $\delta f_k(0)=[\delta n_k(0)/n_0+(m_i/T_i)v_\parallel u_{\parallel k}(0)]F_{i0}$
- (21) $\frac{e\phi_k}{T_i}=(1+1.6q^2/\epsilon^{1/2})^{-1}\left[\frac{e\phi_k(0)}{T_i}+(1.6\epsilon^{3/2}+0.8\lambda\epsilon)\left(\frac{ik_\perp\alpha B/\Omega_{ip}}{k_\perp^2a_i^2}\right)\right]$, $u_{\parallel k}(0)=\alpha B(1+\lambda\cos\theta)$, $\Omega_{ip}=eB_p/(m_ic)$
- (22) $u_p=(1+1.6q^2/\epsilon^{1/2})^{-1}u_p(0)$

THE RESIDUAL (benchmark form, with "phi(0)" = potential after classical polarization shielding established, i.e. from an initial density/gyrocenter perturbation):
$$\frac{\phi_k(\infty)}{\phi_k(0)}=\frac{1}{1+1.6\,q^2/\sqrt{\epsilon}}$$
Coefficient 1.6 (1.635 in Xiao-Catto's more precise expansion, eq. 36 there). Valid: k_r rho_i -> 0 but k_r rho_pol << 1 as well; eps << 1; q>1 regime quoted; collisionless; adiabatic electrons with zonal (flux-surface-avg) response excluded — n=0 electron response via Q=0 & J_0=1 (i.e. electrons' flux-surface-averaged phi is NOT adiabatic; this matches GX "iphi00=2" / zonal-adiabatic option).
Normalization: independent of rho_i / v_ti choice; a_i = rho_i^GX exactly (sqrt(T/m)/Omega). No figures/tables in the paper.

Evaluations: q=1.4, eps=0.18 -> 0.11917; q=2, eps=0.2 -> 0.06531; q=1.5, eps=0.1 -> 0.08075.
Proposed benchmark: GKX linear, ky=0, single kx (k_x rho_i ≈ 0.05), circular s-alpha (or Miller circular, shat irrelevant at ky=0), initial density (gyrocenter) perturbation, adiabatic electrons with zonal correction, collisionless; time-average phi_zf over last 30% after GAM decay (t ≳ 100 a/v_ti for Cyclone). Tolerance: ±15% relative vs RH (RH itself is O(eps^{1/2}) and O(k_r rho_pol) accurate; known GK codes give ~0.10–0.11 at q=1.4 eps=0.18 due to finite-eps); tighten to ±5% against Xiao-Catto (below). Also check residual scales as q^2: q-scan {1,1.4,2} with eps=0.18 to confirm 1/(1+c q^2/sqrt eps) shape with fitted c in [1.4,1.9].

---------------------------------------------------------------------------------------------------
## 2. Xiao & Catto 2006

Citation: Y. Xiao and P. J. Catto, "Plasma shaping effects on the collisionless residual zonal flow level", Phys. Plasmas 13, 082307 (2006). DOI 10.1063/1.2266892.

Model: RH procedure (their eq. 1) evaluated in a simple global Grad-Shafranov equilibrium (constant dp/dpsi and I dI/dpsi; limit of Zheng-Wootton-Solano / Helander-Sigmar), inverse-aspect-ratio expansion to O(eps^{5/2}); elongation kappa, triangularity delta, Shafranov shift Delta; radial variation of shaping parameters, q, p ignored; long-wavelength (k_perp rho_pol -> 0); only passing particles contribute to <∫F0 (v_par/B)bar^2>.

Key equations:
- (1) $\phi_k(\infty)=\phi_k(0)\Big/\left\{1+\frac{m_i(IS')^2}{n_0T_i\langle k_\perp^2/B^2\rangle}\left\langle\int d^3v\,F_0\frac{v_\parallel}{B}\left[\frac{v_\parallel}{B}-\overline{\left(\frac{v_\parallel}{B}\right)}\right]\right\rangle\right\}$
- (2) $\phi_k(\infty)=\phi_k(0)/(1+1.6q^2/\sqrt\varepsilon)$
- (3) $\psi(R,Z)=\frac{\psi_0}{R_0^4}\left[(R^2-R_0^2)^2+\frac{Z^2}{E^2}(R^2-R_x^2)-\tau R_0^2\left(R^2\ln\frac{R^2}{R_0^2}-(R^2-R_0^2)-\frac{(R^2-R_0^2)^2}{2R_0^2}\right)\right]$
- (8)-(9) $x=\varepsilon\cos\theta-\Delta\cos^2\theta-\frac{\kappa^2}{4E^2}\varepsilon^2\sin^2\theta$, $z=\kappa\varepsilon\sin\theta$
- (10) $\Delta=(1+\tau/3)\varepsilon^2/2$; (11) $\kappa\equiv 2E/\sqrt{1-R_x^2/R_0^2}$; (12) $\delta=[\kappa^2/(4E^2)-\Delta/\varepsilon^2]\varepsilon=\varepsilon/(1-R_x^2/R_0^2)-\Delta/\varepsilon$
- (14) $\tilde A\equiv\Delta/\varepsilon^2$, (15) $\tilde B\equiv\Delta/\varepsilon^2+\delta/\varepsilon$; (16) $R=R_0(1+\varepsilon\cos\theta-\tilde A\varepsilon^2\cos^2\theta-\tilde B\varepsilon^2\sin^2\theta)$, $Z=R_0\kappa\varepsilon\sin\theta$
- (20) $q=\frac{\kappa R_0I}{8\psi_0}[1+\tfrac12(1+3\tilde A+\tilde B)\varepsilon^2+O(\varepsilon^4)]$; (21) $B_p=\frac{I}{R_0}\frac{\varepsilon}{q}\sqrt{\kappa^2\cos^2\theta+\sin^2\theta}(1+O(\varepsilon))$
- (25) trapped-passing boundary $\lambda_c=1-\varepsilon-\varepsilon^2(\tilde A+\kappa^2/2q^2)$
- (28) $\langle k_\perp^2/B^2\rangle=(IS'\varepsilon/q)^2\,\pi(1+\kappa^2)/\oint(dl_p/B_p)\,[1+O(\varepsilon^2)]$
- (36) $\langle\int d^3vF_0\overline{(v_\parallel/B)}^2\rangle=\frac{n_0T_i}{m_iB_0^2}\frac{2\pi}{\oint dl_p/B_p}\{1-1.635\varepsilon^{3/2}-[\frac{1+\kappa^2}{2q^2}+(\frac32(3\tilde A+\tilde B)-1)]\varepsilon^2-\frac12(-0.722+1.502\tilde A+1.443\tilde B+\frac{0.722-0.692\kappa^2}{q^2})\varepsilon^{5/2}\}$
- (37) difference = $\frac{n_0T_i}{m_iB_0^2}\frac{2\pi}{\oint}[1.635\varepsilon^{3/2}+\frac12\varepsilon^2-\frac12(-0.722+1.502\tilde A+1.443\tilde B+\frac{0.722-0.692\kappa^2}{q^2})\varepsilon^{5/2}]$
- (38) $\phi_k(\infty)=\phi_k(0)/(1+Sq^2/\sqrt\varepsilon)$
- (39) $S=\frac{1}{1+\kappa^2}\left(3.27+\sqrt\varepsilon+0.722\varepsilon-1.443\delta-2.945\frac{\Delta}{\varepsilon}+\frac{0.692\kappa^2-0.722}{q^2}\varepsilon\right)$
  Circular limit kappa=1, delta=Delta=0, q->inf: S -> (3.27+sqrt eps+0.722 eps)/2 -> 1.635 + higher-order eps corrections that REDUCE residual.
Note Δ here is dimensionless Shafranov shift (Δ/R0), δ is triangularity, both "one order smaller in ε".

Plotted reference values (I verified eq. 39 reproduces all three figures to reading precision):
- Fig. 1: residual vs kappa ∈[1,3], q=2, eps=0.2, Δ=0.04, δ=0. Eq.39: κ=1 -> 0.0640, κ=1.8 -> 0.1240, κ=3 -> 0.2396 (fig reads ≈0.063, —, ≈0.24; ±0.003).
- Fig. 2: vs δ∈[0,0.4], q=2, eps=0.2, Δ=0.04, κ=1.8: δ=0 -> 0.1240, δ=0.4 -> 0.1461 (fig 0.124 / 0.146, ±0.001).
- Fig. 3: vs Δ∈[0,0.1], q=2, eps=0.2, κ=1.8, δ=0: Δ=0 -> 0.1075, Δ=0.1 -> 0.1613 (fig 0.107/0.161, ±0.001).
- Cyclone-like circular q=1.4, eps=0.18: S=1.9107 -> residual 0.1018 (vs RH 0.1192; −15%). This is the better target for a circular GK code.
Paper notes Belli's GS2 study finds similar κ trend, stronger δ dependence at large κ (Miller eq includes gradients of shaping — so tolerance for Miller runs vs eq.39 should be loose on δ, s_κ, s_δ).

Proposed benchmark: (a) circular Miller/s-alpha q=1.4 eps=0.18 residual target 0.102 ±7%; (b) Miller κ-scan {1,1.5,2,2.5,3} with q=2, eps=0.2, δ=0, shift chosen so dR/dr≈−Δ-consistent (Miller shift param shift = dR0/dr; Δ=0.04 at eps=0.2 corresponds to d(Δ R0)/dr ≈ 2Δ/eps = 0.4 — derive carefully), s_κ=s_δ=0: check monotone rise and ratio residual(κ=3)/residual(κ=1) = 3.74 ±15%. Tolerance loose because Miller is a local equilibrium, not this global one.

---------------------------------------------------------------------------------------------------
## 3. Sugama & Watanabe 2006 (JPP)

Citation: H. Sugama and T.-H. Watanabe, "Collisionless damping of geodesic acoustic modes", J. Plasma Phys. 72(6), 825–828 (2006). doi:10.1017/S0022377806004958.
(Full-length companion with corrections: Sugama & Watanabe, J. Plasma Phys. 72, 825 is the short one; the long paper is Phys. Plasmas 13, 012501 (2006) + erratum — see refs.)

Model: electrostatic collisionless GK for zonal component k_perp = k_r ∇r; large aspect ratio circular, B=B0(1−ε cosθ); passing ions only (mirror force neglected); finite drift-orbit width retained via δ̂ = (ε/Ω_P)[v_par + μB0/(m v_par)], Ω_P=eB_P/(mc); zero-gyroradius limit k_perp ρ -> 0 in (2.3); k_r a_i << 1 for (2.4); electrons Boltzmann for m≠0, zero for m=0; initial Maxwellian gyrocenter density perturbation; slow RH trapped-particle physics added afterwards via (2.8).

Key equations:
- (2.1) $\left(\frac{\partial}{\partial t}+v_\parallel\mathbf b\cdot\nabla+i\omega_D\right)\delta f_{\mathbf k_\perp}=-(v_\parallel\mathbf b\cdot\nabla+i\omega_D)\left(F_0J_0(k_\perp\rho)\frac{e\phi_{\mathbf k_\perp}}{T}\right)$, $\omega_D\equiv k_rv_{dr}$
- (2.2) $\left(\frac{\partial}{\partial t}+\frac{v_\parallel}{R_0q}\frac{\partial}{\partial\theta}\right)(e^{ik_r\hat\delta\cos\theta}\delta\hat f_{\mathbf k_\perp})=-\frac{v_\parallel}{R_0q}\frac{\partial}{\partial\theta}\left(e^{ik_r\hat\delta\cos\theta}J_0\frac{e\phi_{\mathbf k_\perp}}{T}\right)$
- (2.3) $\delta\hat f_{k_r,m}(\omega)=\sum_{l,l'}i^{l'-l}J_l(k_r\hat\delta)J_{l'}(k_r\hat\delta)\left(\frac{(m+l)(v_\parallel/R_0q)}{\omega-(m+l)(v_\parallel/R_0q)}\right)\left(\frac{e\phi_{k_r,m+l-l'}(\omega)}{T}\right)+\delta\hat I_{k_r,m}(\omega)$
- (2.4) $\int d^3vF_0\delta\hat f_{ik_rm}(\omega)=n_0e\phi_{k_rm}(\omega)/T_e$ (m≠0)
- (2.5) $n_0(k_ra_i)^2\frac{e}{T_i}[-i\omega\phi_{k_r0}(\omega)-\phi_{k_r0}(t=0)]=\int d^3vF_0\frac{k_r}{2R_0\Omega_i}\left(v_\parallel^2+\frac{\mu B_0}{m_i}\right)\left[\delta\hat f_{ik_r-1}+\frac{e\phi_{k_r-1}}{T_i}-\delta\hat f_{ik_r1}-\frac{e\phi_{k_r1}}{T_i}\right]$, $a_i=(T_i/m_i)^{1/2}/\Omega_i$
- (2.6) m=1 response with resonances ω=v_par/R0q and ω=2v_par/R0q (latter weighted by (k_r δ̂/2)^2).
- (2.7) $\frac{1}{K(\omega)}=-i\hat\omega-i\frac{q^2}{2}\Big[2\hat\omega^3+3\hat\omega+(2\hat\omega^4+2\hat\omega^2+1)Z(\hat\omega)-\frac{\hat\omega}{2}\{2\hat\omega+(2\hat\omega^2+1)Z(\hat\omega)\}^2\{\frac{T_i}{T_e}+1+\hat\omega Z(\hat\omega)\}^{-1}+i\frac{\sqrt\pi}{2}\left(\frac{k_rv_{T_i}q}{\Omega_i}\right)^2e^{-\hat\omega_r^2/4}\Big\{\frac{\hat\omega_r^6}{64}+\left(\frac{\hat\omega_r^4}{8}+\frac{3\hat\omega_r^2}{4}+3+\frac{6}{\hat\omega_r^2}\right)\left(1-\frac{3\hat\omega_r}{16}\{2\hat\omega_r+(2\hat\omega_r^2+1)Z(\hat\omega_r)\}\{\frac{T_i}{T_e}+1+\hat\omega_rZ_r(\hat\omega_r)\}^{-1}\right)\Big\}\Big]$
  with $\hat\omega\equiv R_0q\omega/v_{T_i}$, $v_{T_i}\equiv\sqrt{2T_i/m_i}$, Z = plasma dispersion function. [Nesting re-checked against the printed page 827: the FOW group is { w^6/64 + (w^4/8 + 3w^2/4 + 3 + 6/w^2)(1 - (3w/16){...}{...}^{-1}) }, as transcribed.]
- (2.8) $\phi_{k_r0}(t)=\phi_{k_r0}(\infty)+[\phi_{k_r0}(0)-\phi_{k_r0}(\infty)]\cos(\omega_Gt)\exp(\gamma t)$, $\phi(\infty)=\phi(0)/(1+1.6q^2/\epsilon^{1/2})$
- (2.9) $\omega_G=\frac{\sqrt{7+4\tau_e}}{2}q\left(\frac{v_{T_i}}{R_0q}\right)\left[1+\frac{2(23+16\tau_e+4\tau_e^2)}{q^2(7+4\tau_e)^2}\right]^{1/2}$
- (2.10) $\gamma=-\frac{\sqrt\pi}{2}q^2\left(\frac{v_{T_i}}{R_0q}\right)\left[1+\frac{2(23/4+4\tau_e+\tau_e^2)}{q^2(7/2+2\tau_e)^2}\right]^{-1}\Big[\exp(-\hat\omega_G^2)\{\hat\omega_G^4+(1+2\tau_e)\hat\omega_G^2\}+\frac14\left(\frac{k_rv_{T_i}q}{\Omega_i}\right)^2\exp(-\hat\omega_G^2/4)\Big\{\frac{\hat\omega_G^6}{64}+\left(1+\frac38\tau_e\right)\left(\frac{\hat\omega_G^4}{8}+\frac{3\hat\omega_G^2}{4}\right)\Big\}\Big]$
  valid for $\hat\omega_G^2\equiv(R_0q\omega_G/v_{T_i})^2\gg1$; $\tau_e\equiv T_e/T_i$.
  Note (2.10)'s first bracket is identical to (2.9)'s bracket written with 23/4 etc. (algebraically equal: 2(23+16τ+4τ²)/(7+4τ)² = 2(23/4+4τ+τ²)/(7/2+2τ)²·... check: yes equal since numerator and denominator both /4).
  q-dependent corrections: ω_G ∝ [1 + 2(23+16τ+4τ²)/(q²(7+4τ)²)]^{1/2} (for τ=1: 1+ 86/(121 q²)); γ has the q² prefactor, the (1+...)^{-1} correction, e^{-ω̂²} with ω̂_G ∝ q, and the finite-orbit-width term ∝ (k_r v_T q/Ω_i)² e^{-ω̂²/4} which dominates at large q.

Normalization mapping: k_r v_{T_i} q/Ω_i = √2 q (k_r ρ_i^GX). ω in v_{T_i}/R0 = √2 · (ω in v_ti^GX/R0). To GX a-units multiply by (a/R0).

Evaluations (τ_e=1):
- q=1.4, ε irrelevant, k_r→0: ω_G = 1.936 v_Ti/R0 = 2.738 v_ti/R0 (GX); γ = −0.0447 v_Ti/R0 = −0.0632 v_ti/R0 (GX). For Cyclone (a/R0=0.36): ω_G ≈ 0.986 v_ti/a, γ ≈ −0.0228 v_ti/a (k_r→0; finite k_r raises |γ|).
- q=1.5, k_r a_i=0.131 (Fig. 1 case, ε=0.1): ω_G=1.902 v_Ti/R0 = 2.690 v_ti/R0 (period ≈2.34 R0/v_ti), γ=−0.0986 v_Ti/R0 = −0.139 v_ti/R0; without FOW term γ=−0.0377 v_ti/R0 (the "dotted curve").
- Residual for Fig.1: RH(q=1.5, ε=0.1)=0.0807 (figure asymptote ≈0.08–0.1, reading ±0.02).
Fig. 1 (only figure): <φ_k(t)>/<φ_k(0)> vs t in R0/v_ti (v_ti=(T_i/m_i)^{1/2}), 0–40; GKV simulation (open circles), thin solid = numerical root of 1/K=0, thick = (2.9)-(2.10), dotted = no FOW. Envelope decays to near-residual by t≈30; ~16 oscillations in 40 (period ≈2.4–2.6, consistent with 2.34 within reading ±10%).

Proposed benchmark: GKX ky=0, kx ρ_i such that k_r a_i = 0.131, circular ε=0.1, q=1.5, τ=1, adiabatic electrons (zonal-corrected): fit φ(t) to (2.8) over t∈[2,30] R0/v_ti. Targets: ω_G=2.69 v_ti/R0 ±5%; γ=−0.139 v_ti/R0 ±25% (formula asymptotic in ω̂_G ≫1, which is only ω̂_G≈2.9; compare also to numerical root of (2.7) — that is the ±10% target). Cyclone variant q=1.4: ω_G 0.986 v_ti/a ±5%.

---------------------------------------------------------------------------------------------------
## 4. Romanelli 1989

Citation: F. Romanelli, "Ion temperature-gradient-driven modes and anomalous ion transport in tokamaks", Phys. Fluids B 1(5), 1018–1025 (May 1989). DOI 10.1063/1.859023.

Model: electrostatic, low-β circular s-α (α=0) ballooning; adiabatic electrons; trapped ions neglected; kinetic ions without ω_D/ω expansion; v_par, v_perp constant along orbit (O(ε) error). Normalizations: τ=T_e/T_i (NOTE: opposite of GX tau_fac), φ̂=eφ/T_i, α=(2b_i)^{1/2}v_⊥, v=v/v_ti, 2b_i=k²v_ti²/Ω_i², v_ti=(2T_i/m_i)^{1/2} ⇒ b_i = (k ρ_i^GX)², b_s=τ b_i. ε_n = L_n/R, L_n=|∇n/n|^{-1}, η_i = d ln T_i/d ln n, ŝ=rq'/q.

Key equations:
- (1) $(1/\tau+1)\hat\phi(\chi)=\int d^3v\,J_0(\alpha)h$
- (2) $i\frac{v_\parallel}{qR}\frac{\partial h}{\partial\chi}+(\omega-\omega_D)h=(\omega-\omega_{*T})J_0(\alpha)F_M\hat\phi(\chi)$
- (3) $\omega_D=\hat\omega_D(v_\perp^2/2+v_\parallel^2)$, $\omega_{*T}=\omega_{*i}[1+\eta_i(v_\perp^2+v_\parallel^2-\frac32)]$, $\hat\omega_D=2\epsilon_n\omega_{*i}(\cos\chi+\hat s\chi\sin\chi)$, $\omega_{*i}=-k_\theta cT_i/eB|L_n|$
- (5)-(8) Fredholm integral eq. and kernel; (9) slab/no-curvature limit
- (11) fluid limit; (12) local kinetic: $\left[\left(1+\frac1\tau\right)-\int d^3vF_MJ_0^2\frac{\omega-\omega_{*T}}{\omega-\omega_D}\right]\hat\phi=0$
- (13) Weber eigenvalue (strong coupling), (14) 2σ; (15) flat density $b\bar\Omega\hat s^2+(2\hat s-1)+bq^2\bar\Omega(b\bar\Omega+2+\tau\bar\Omega^2)=0$; (16) slab-like root; (17) $\bar\Omega=\pm(-2/\tau\pm i\hat s/q\tau)^{1/2}$
- (18) b_s ≪ ε_T^{1/2}: $\Omega=\frac12\left(1-i\epsilon_n\frac{\hat s(1+2q^2)^{1/2}}{q}\right)-\frac12\left[\left(1-i\epsilon_n\frac{\hat s(1+2q^2)^{1/2}}{q}\right)^2-4i\epsilon_n\frac{\hat s(1+2q^2)^{1/2}}{q\tau}(1+\eta_i)\right]^{1/2}$
- (19) ∇B-model local kinetic, b_s=0: $D(z)\equiv1+\frac1\tau-\frac{\eta_i}{2\epsilon_n}+\left[z\left(1-\frac{\eta_i}{2\epsilon_n}\right)+\frac{1-\eta_i}{2\epsilon_n}\right]e^zE_1(z)=0$, $z=\Omega\tau/2\epsilon_n$ (Ω=ω/ω_{*e})
- (21) fluid roots; (22) fluid threshold $1+\eta_i=\epsilon_n[4+(\tau/2)(1-1/2\epsilon_n)^2]$; large ε_n: η_ic=(4+τ/2)ε_n
- (23) resonance correction δz = iπτ³e^{−τ}(η_i/2ε_n −1)
- (24)-(25) marginal: D_i=π[z(1−η_i/2ε_n)+(1−η_i)/2ε_n]e^z, z=(1−η_i)/(η_i−2ε_n)
- (26) KINETIC THRESHOLD (∇B model, flat-density branch): $\eta_i=(1+1/\tau)2\epsilon_n$, valid for Re z<0 i.e. η_i>1 or 2ε_n>(1+1/τ)^{-1}; for ε_n<0.5(1+1/τ)^{-1}≈0.25 fluid result applies; minimum near ε_n≈0.25. With v⊥²/2+v∥² → (2/3)(v⊥²+v∥²): η_ic = (4/3)(1+1/τ)ε_n for large ε_n.
- (27) FIT (full kinetic, ŝ=0.5, q=1.5, τ=1, b_i=0.1): $\eta_{ic}=\begin{cases}1,&\epsilon_n<0.2\\1+2.5(\epsilon_n-0.2),&\epsilon_n>0.2\end{cases}$
- (28)-(31) QL flux; (31) $\chi_i=5(v_{ti}\rho_i^2/L_n)\epsilon_n^{1/2}[\eta_i-\eta_{ic}(\epsilon_n)]^{1/2}$; (32) marginal T_i profile.

Figures / targets (reading uncertainty):
- Fig.1 fluid, b_s=ε_n=0.1, ŝ=τ=1; q=1 solid, q=10 dashed: Im Ω(η_i=10) ≈ 1.40 (q=1), 1.30 (q=10); Ω in ω_{*e}; ±0.05. Re Ω ≈ −0.35 at η_i=10 (q=1).
- Fig.2 local kinetic (12), b_i=0.1, τ=1; ε_n=0.5 and 0.1: Im Ω(η_i=10, ε_n=0.1) ≈ 1.05; ε_n=0.5 ≈1.2 at η_i≈8; threshold η_ic ≈ 1.3–1.5 (ε_n=0.1) ±0.2. Re Ω ≈ −0.3 at η_i=10 (ε_n=0.1).
- Fig.3 η_ic vs ε_n (ŝ=0.5, q=1.5 for full kinetic): local-kinetic minimum ≈0.9 at ε_n≈0.2–0.25; at ε_n=1: local kinetic ≈3.0, full kinetic ≈4.3(dotted), fit(27) = 3.0. ±0.2.
- Fig.4 ε_n=0.25, b_s=0.1, q=ŝ=τ=1: thresholds fluid≈1.0, local kinetic≈1.2–1.5, full kinetic≈1.6; ±0.2. Im Ω at η_i=10: full kinetic ≈0.65.
- Fig.5 QL flux vs η_i, τ=1, b_s=0.1: thresholds ≈1.2 (ε_n=0.1), ≈3.4 (ε_n=1).
Mapping: ε_n = L_n/R = (R/a · fprim)^{-1}·... i.e. ε_n = (a/R)/fprim; η_i = tprim/fprim; Ω=ω/ω_{*e}, ω_{*e} = k_y ρ_i v_ti^GX/L_n (with τ=1, sign convention electron-direction positive); b_i=0.1 ⇒ k_y ρ_i^GX ≈ 0.316.
Proposed benchmark: GKX adiabatic-electron flux-tube, circular s-α (α=0), q=1.5, ŝ=0.5, τ=1, ky ρ=0.3, scan η_i at fixed ε_n∈{0.1,0.25,0.5,1.0}; find η_ic by extrapolating γ→0. Target Eq.(27) ±0.3 in η_ic (Romanelli neglects trapped ions and v_∥ variation; GK includes both). Most robust single point: ε_n=1 ⇒ η_ic≈3.0 (fit) vs 4.3 (full kinetic line at ŝ=0.5,q=1.5) — ambiguous, so prefer the flat-density (26) limit scaled: R/L_Tc = η_ic/ε_n.

---------------------------------------------------------------------------------------------------
## 5. Kim, Horton & Dong 1993

Citation: J. Y. Kim, W. Horton, and J. Q. Dong, "Electromagnetic effect on the toroidal ion temperature gradient mode", Phys. Fluids B 5(11), 4030–4039 (Nov 1993). DOI 10.1063/1.860623.

Model: low-β s-α ballooning, EM (φ, A_∥; no δB_∥), GK ions full kinetic (ω_D and transit resonances, FLR), no trapped particles, v_∥ modulation neglected; electrons: expansion ω ~ ω_Di ~ ω_De ≪ |k_∥ v_te| (first order). Nonlocal integral-equation code (extension of Dong-Horton-Kim 1992) and local kinetic code. v_tj = (T_j/m_j)^{1/2} (= GX v_ti), ρ_i = v_i/ω_ci (= GX ρ_i), τ = T_i/T_e (= GX tau_fac!), β_i = 8πNT_i/B², α=−(2Rq²/B²)dp/dr, ε_n=L_n/R.

Key equations:
- (1) quasineutrality; (2) $-\frac{4\pi}{c}J_\parallel=\nabla_\perp^2A_\parallel=-k_\perp^2A_\parallel=\frac{4\pi e}{c}(\int v_\parallel f_ed^3v-\int v_\parallel f_id^3v)$
- (4) $\left(i\frac{v_\parallel}{Rq}\frac{\partial}{\partial\theta}+\omega-\omega_{Dj}\right)h_j=\frac{q_jF_{Mj}}{T_j}(\omega-\omega_{*j}^T)J_0(\delta_j)\left(\phi(\theta)-\frac{v_\parallel}{c}A_\parallel(\theta)\right)$
  $\omega_{Dj}=\hat\omega_{Dj}(v_\perp^2/2+v_\parallel^2)/2v_{tj}^2$, $\omega^T_{*j}=\omega_{*j}[1+\eta_j(v^2/2v_{tj}^2-3/2)]$, $\hat\omega_{Dj}=2\epsilon_n\omega_{*j}[\cos\theta+\sin\theta(s\theta-\alpha\sin\theta)]$, $k_\perp^2=k_y^2[1+(s\theta-\alpha\sin\theta)^2]$
- (9)-(10) coupled integral equations; (12)-(16) kernels; $K_e^1\simeq\frac{qRT_i}{2v_{ti}T_e}(\omega-\omega_{*e})\mathrm{sgn}(\theta-\theta')$ etc.
- (17) $\omega(1+\tau-P_0)\phi=[\tau(\omega-\omega_{*e})-k_\parallel P_1]\psi$
- (18) $\frac{2k_\parallel^2k_\perp^2}{\beta_i}\psi=\omega[k_\parallel P_1-\tau(\omega-\omega_{*e})]\phi-\{k_\parallel^2P_2-\tau[\omega(\omega-\omega_{*e})-\omega_{De}(\omega-\omega_{*pe})]\}\psi$, $\psi=(\omega v_{ti}/ck_\parallel)A_\parallel$
- (19) $P_m=\int(v_\parallel)^m\frac{\omega-\omega_{*i}[1+\eta_i(v^2/2-3/2)]}{\omega-\epsilon_n\omega_{*i}(v_\perp^2/2+v_\parallel^2)-k_\parallel v_\parallel}J_0^2(k_\perp v_\perp)e^{-v^2/2}v_\perp\frac{dv_\perp dv_\parallel}{\sqrt{2\pi}}$
  normalized: k_⊥,k_∥ to ρ_i and L_n; ω, ω_* to v_i/L_n; ω_{*e}=k_y, ω_{*i}=−k_y, ω_{*pe}=ω_{*e}(1+η_e).
- (20)-(21) local fluid; (22) ES fluid dispersion $\tau(1-\frac{\omega_{*e}}{\omega})-\frac{k_\parallel^2}{\omega^2}(1-\frac{\omega_{*pi}}{\omega})=-(k_\perp^2-\frac{\omega_{Di}}{\omega})(1-\frac{\omega_{*pi}}{\omega})$
- (23) ideal ballooning; (24) $\beta_{ic}=\frac{2k_\parallel^2k_\perp^2}{\omega_{Di}(\omega_{*pi}-\omega_{*pe})-(\omega_{*pi}k_\perp/2)^2}$, long wavelength $\beta_{ic}\simeq k_\parallel^2/\epsilon_n[1+\eta_i+(1+\eta_e)T_e/T_i]\sim\epsilon_n/q^2$
- (25) EM-modified dispersion; (26) coefficient of $\omega_D\omega_{*pi}$: $1-\frac{\tau\omega(\omega-\omega_{*e})-k_\parallel^2(1-\omega_{*pi}/\omega)}{(2k_\parallel^2k_\perp^2/\beta_i)-\omega_{Di}(\omega-\omega_{*pe})}$
- (27) ITG stabilization β: $\beta_i\simeq\beta_{it}=\frac{2k_\parallel^2k_\perp^2}{\omega_{Di}(\omega_{*i}-\omega_{*pe})+(k_\parallel^2\omega_{*pi}/\omega_{Di})}$ (∝η_i^{-1}... paper: β_it ∝ k_∥², ~independent of k_y, "inversely proportional to η_i" per text)

Figures / targets (base: η_i=2.5, η_e=2, k_yρ_i=0.5, k_∥L_n=0.1 (local), ε_n=0.2, τ=1; γ, ω in v_i/L_n):
- Fig.1 (local): LK η_i-mode γ: 0.20 at β_i=0 → 0 at β_i≈0.0075; LF: 0.49 → 0 at β_i≈0.0115; LK ballooning onset β_c≈0.0065, LF ballooning onset ≈0.0115. ω_r(LK η_i) ≈ −0.12→−0.2 (ion dir.); ±0.01 in γ, ±0.0005 in β.
- Fig.2 γ vs k_∥L_n (LK): β_i=0 γ_max≈0.15 at k_∥L_n≈0; β=0.001 peak ≈0.13 at k_∥≈0.08; 0.004 peak≈0.07 at ≈0.12; 0.006 peak≈0.035 at ≈0.14; all cut off at k_∥L_n≈0.18. ±0.01.
- Fig.3 γ vs k_yρ_i: β=0 peak ≈0.19 at k_yρ≈0.5; 0.002→0.13; 0.004→0.08; 0.006→0.035; unstable ≈0.1–1.5. ±0.01.
- Fig.4 γ vs η_i ∈[0,10]: β=0 γ(10)≈0.58; 0.003→0.3; 0.006→0.08; ballooning(0.006) ≈0.0. ±0.02.
- Fig.5 (nonlocal) eigenfunction η_i=2.5, η_e=2, kyρ=0.5, ε_n=0.2, τ=1, s=0.6, q=1.5, β_i=0.004: φ even, A_∥ odd, |Â_∥/φ|≪1.
- Fig.6a (nonlocal) γ vs q∈[0.5,2.5], s=0.6: β=0: 0.07 (q=0.5)→0.18 (q=2.5); β=0.002 peak ≈0.135 at q≈1.1; β=0.004 peak ≈0.11 at q≈0.9, →0 at q≈1.55. s=2: β=0 ≈0.10 at q=2.5; β=0.004 ≈0.095 at q≈2.1.
- Fig.6b γ vs s∈[0.4,2]: q=1 β=0: 0.155 at s=0.4; q=1 β=0.004: 0.12 at s=0.6; q=2 β=0.004 rises from 0 at s≈0.8 to ≈0.10 at s=2.
Mapping to GKX: β_i = beta_ref (for n_i=n_e, T_i=T_ref); v_i/L_n = (v_ti/a)·fprim; k_yρ_i direct; ε_n=0.2 ⇒ R/L_n=5; η_i=2.5 ⇒ R/L_Ti=12.5; η_e=2 ⇒ R/L_Te=10. α (MHD) = 0 in the curves (s-α with α set to 0 despite finite β — must set alpha=0 / no β' in GKX for parity). Needs kinetic electrons (EM).
Proposed benchmark: nonlocal case Fig.6a: s-α, q=1.5, s=0.6, ε_n=0.2, η_i=2.5, η_e=2, τ=1, k_yρ=0.5, ε=r/R small (no trapped particles: use GKX without trapping? — GK code will include trapped electrons; paper neglects them, so ± large). Qualitative targets: γ(β=0.004)/γ(β=0) ≈ 0.55 at q=1.5 (read ≈0.02/0.17 at q=1.5… i.e. strong suppression), crossover to KBM near β≈0.006–0.008. Tolerance: β_crit ±30%, trend only. Use as KBM-branch sanity check, not quantitative gate.

---------------------------------------------------------------------------------------------------
## 6. Biglari, Diamond & Rosenbluth 1989

Citation: H. Biglari, P. H. Diamond, and M. N. Rosenbluth, "Toroidal ion-pressure-gradient-driven drift instabilities and transport revisited", Phys. Fluids B 1(1), 109–118 (Jan 1989). DOI 10.1063/1.859206.

Model: ES, kinetic circulating ions with drift resonances, leading-order FLR (b_⊥=(k_⊥ρ_i)²/2, ρ=v_t/Ω, v_t=(2T/m)^{1/2} ⇒ b_⊥ = (k_⊥ρ_i^GX)²), parallel compressibility; adiabatic electrons; ion trapping ignored for toroidal branch; ordering ω_be,ω_te ≫ ω_* > |ω| ~ ω_di > ω_bi, ω_ti. ω_{di}=R/2L_{Ti}? — printed: Ω=ω/ω̄_di, Ω^T_{*i}=ω^T_{*i}/ω̄_di, ω̄_di... "ω_di = R/2L_Ti"(sic, as a ratio). τ=T_e/T_i (as in Romanelli).

Key equations:
- (2) $\delta h_{ic}\simeq\frac{e\delta\phi}{T_i}\left(1-b_\perp\bar v_\perp^2+\frac{(\omega_{ti}/\bar\omega_{di})^2\bar v_\parallel^2}{[\Omega-(\bar v_\parallel^2+\bar v_\perp^2/2)]^2}\right)\frac{\Omega-\Omega^T_{*i}(\bar v_\parallel^2+\bar v_\perp^2+\eta_i^{-1}-\frac32)}{\Omega-(\bar v_\parallel^2+\bar v_\perp^2/2)}F_{Mi}$
- (3) $D(\Omega)=D_0+D_{FLR}+D_S$; $D_0=-(1+\tau^{-1})+Y^2+\Omega^T_{*i}\{[(1-\eta_i^{-1})/\Omega-2]Y^2+2Y\}$; $D_{FLR}/b_\perp=-Y^2-2\Omega(Y-1)^2+\Omega^T_{*i}(2(\eta_i^{-1}+2\Omega)(Y-1)^2+Y^2/\Omega\eta_i)$; $D_S/(\omega_{ti}/\bar\omega_{di})^2=[\Omega^T_{*i}(\eta_i^{-1}-\frac32)-\Omega][1-(1+\frac{1}{2\Omega})Y+\frac{Y^2}{\Omega}]+\Omega^T_{*i}[\Omega-(\Omega+\frac32)Y+(2-\frac{1}{2\Omega})Y^2]$; $Y=\Omega^{1/2}\exp(-\Omega)\int_{-\infty}^{\Omega}dz\,z^{-1/2}\exp z$
- Nyquist: instability needs (i) 2η_i^{-1}−3<0 ⇒ η_i>2/3 (or η_i<0), (ii) Re D(Ω_E)>0.
- (4) local nonresonant: $1+\frac{\bar\omega_{de}-\omega_{*e}}{\omega}-\frac{[7\bar\omega_{de}/4-\omega_{*e}(1+\eta_i)]\bar\omega_{de}}{\tau\omega^2}+\left(b_\perp-\frac{k_\parallel^2v_{ti}^2}{2\omega^2}\right)\frac{\omega_{*e}(1+\eta_i)}{\omega}=0$
- (5) $2\omega_{tor}=[1-b_\perp(1+\eta_i)]\omega_{*e}-\bar\omega_{de}\pm\{[1-b_\perp(1+\eta_i)]^2\omega_{*e}^2+(1+7\tau^{-1})\bar\omega_{de}^2-2[1+2\tau^{-1}(1+\eta_i)]\omega_{*e}\bar\omega_{de}\}^{1/2}$
- (6) $\gamma_{tor}\simeq\{[\frac12+\tau^{-1}(1+\eta_i)]\omega_{*e}\bar\omega_{de}\}^{1/2}$
- (7)-(10) Weber eigenmode; (8) nonlocal ω_tor with factor (1+iŝ/√2 q); (9) $\omega_{tor}\simeq i\{[\frac12+\tau^{-1}(1+\eta_i)(1+i\hat s/\sqrt2q)]\omega_{*e}\bar\omega_{de}\}^{1/2}$; (10) $(k_r\rho_i)^2=\sqrt2(\hat s/q)(\bar\omega_{de}/|\omega_{tor}|)$
- (15)-(22) mixing-length transport; (22) $\chi_i\sim\frac{q}{\hat s}\omega_{*e}\rho_s^2\frac{1+\eta_i}{\tau}$
- (26)-(29) trapped-ion branch; (27) $\gamma_{tr}\simeq(\sqrt{2\epsilon}\{[1+2(1+\eta_i)/\tau]/(1+\tau^{-1})\}\omega_{*e}\bar\omega_{de})^{1/2}$; (29) $\frac{2(1+\tau^{-1})}{\sqrt{2\epsilon}}\frac{1+\eta_i}{\tau}>\frac{\omega_{*e}}{\bar\omega_{de}}$; trapped branch unstable for η_i>2/3 and ν_i<ε^{5/4}ω_{*e}ω̄_{de}; (31) χ_i trapped.
- (32) flat density: $2\omega/\tau\omega^T_{*i}\simeq b_\perp+\epsilon_{Ti}\pm[(1+7\tau^{-1})\epsilon_{Ti}^2-4\tau^{-1}\epsilon_{Ti}]^{1/2}$, ε_Ti = L_Ti/R; fluid stability for ε_Ti>4/(7+τ).
- FLAT-DENSITY THRESHOLD (full resonant D, b_⊥=k_∥=0, from Fig.2): L_Ti/R ≃ 0.35 ⇒ R/L_Ti,crit ≈ 2.86 (τ not stated explicitly in text; assume τ=1). Fig.2 stability diagram (L_n/R vs L_Ti/R): η_i=2/3 line near origin; unstable boundary at L_Ti/R≈0.35–0.4 for |L_n/R|≳0.5, vertical boundary ~0.4 at L_n/R≈2; reading ±0.03.
- (33)-(34) energetic trapped-particle generalization.
- Table I: scaling compendium (slab/toroidal/trapped-ion) for k_θρ_i, γ/ω_{*e}, Δx/ρ_s, δn/n, χ_i/(ω_{*e}ρ_s²), χ_e.
- Appendix (A1)-(A4): F_{α,β} integrals; Y(Ω)=Ω^{1/2}e^{−Ω}(−iπ^{1/2}+∫_0^Ω dz z^{−1/2}e^z).
Proposed benchmark: GKX adiabatic-electron flat-density (fprim=0 or tiny) circular, ky ρ small (0.1–0.2), q large/ŝ moderate: scan R/L_Ti; target R/L_Ti,crit ≈ 2.9 (±0.5; GK codes with FLR/transit + trapped ions typically give ~4 for Cyclone-like with ŝ=0.8, q=1.4 — so use as lower bound / consistency, not tight gate). Also check η_i>2/3 necessary condition at weak gradient (qualitative).

---------------------------------------------------------------------------------------------------
## Summary of numeric targets
| Quantity | Params | Value | Tol |
|---|---|---|---|
| RH residual | q=1.4, ε=0.18 | 0.1192 | ±15% |
| Xiao-Catto circ. residual | q=1.4, ε=0.18 | 0.1018 | ±7% |
| XC Fig1 κ=1/1.8/3 | q=2, ε=0.2, Δ=0.04 | 0.064/0.124/0.240 | ±10% (Miller vs global eq) |
| GAM ω_G (SW 2.9) | q=1.4, τ=1, k_r→0 | 2.738 v_ti/R0 (0.986 v_ti/a at a/R=0.36) | ±5% |
| GAM γ (SW 2.10) | q=1.4, k_r→0 | −0.0632 v_ti/R0 | ±30% |
| GAM ω_G,γ | q=1.5, ε=0.1, k_r a_i=0.131 | 2.690, −0.139 v_ti/R0 | ±5%, ±25% |
| Romanelli η_ic | ŝ=0.5, q=1.5, τ=1, b=0.1 | 1 (ε_n<0.2), 1+2.5(ε_n−0.2) | ±0.3 |
| BDR flat-density | b=k_∥=0 | R/L_Ti ≈ 2.86 | ±0.5 (lower bound) |
| KHD β_crit η_i mode | local LK Fig1 | β_i≈0.0075; KBM ≈0.0065 | ±30% trend |

## References worth fetching next
- F. L. Hinton and M. N. Rosenbluth, "Dynamics of axisymmetric (E×B) and poloidal flows in tokamaks", Plasma Phys. Control. Fusion 41, A653 (1999). DOI 10.1088/0741-3335/41/3A/059 (finite-k_r residual; RH time dependence).
- H. Sugama and T.-H. Watanabe, "Collisionless damping of zonal flows in helical systems", Phys. Plasmas 13, 012501 (2006), DOI 10.1063/1.2149311; and PRL 94, 115001 (2005), DOI 10.1103/PhysRevLett.94.115001 (stellarator residual — relevant for GKX stellarator decks).
- Y. Xiao, P. J. Catto, W. Dorland, "Effects of finite poloidal gyroradius, shaping, and collisions on the zonal flow residual in a plasma", Phys. Plasmas 14, 055910 (2007), DOI 10.1063/1.2718519; and Xiao & Catto Phys. Plasmas 13, 102311 (2006), DOI 10.1063/1.2358497 (short-wavelength residual formula).
- Z. Gao, K. Itoh, H. Sanuki, J. Q. Dong, "Multiple eigenmodes of geodesic acoustic mode in collisionless plasmas", Phys. Plasmas 13, 100702 (2006), DOI 10.1063/1.2359722; Gao et al. PoP 15, 072511 (2008) (GAM with elongation).
- E. Belli, PhD thesis Princeton 2006 (GS2 shaping residual fit).
- J. Q. Dong, W. Horton, J. Y. Kim, "Toroidal kinetic η_i-mode study in high-temperature plasmas", Phys. Fluids B 4, 1867 (1992), DOI 10.1063/1.860040.
- R. R. Dominguez and R. E. Waltz, Phys. Fluids 31, 3147 (1988), DOI 10.1063/1.866970 (flat-density ITG).
- M. Kotschenreuther, W. Dorland, M. A. Beer, G. W. Hammett, "Quantitative predictions of tokamak energy confinement from first-principles simulations with kinetic effects", Phys. Plasmas 2, 2381 (1995), DOI 10.1063/1.871261 (R/L_T,crit fit).
- A. M. Dimits et al., Phys. Plasmas 7, 969 (2000), DOI 10.1063/1.873896 (Cyclone; GK residual values).
- W. M. Tang, G. Rewoldt, L. Chen, Phys. Fluids 29, 3715 (1986), DOI 10.1063/1.865803.
- Z. Lin, T. S. Hahm, W. W. Lee, W. M. Tang, P. H. Diamond, PRL 83, 3645 (1999), DOI 10.1103/PhysRevLett.83.3645.
