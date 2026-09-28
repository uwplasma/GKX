# Analytic benchmarks for GKX: KBM and beyond (2026-09-28)

Scope: literature survey plus light numpy/scipy checks (`benchmarks_check.py`,
output in `benchmarks_check.out`). No GKX runs. Sources read in full: open-access
arXiv 2505.10153 (resonant KBM theory, general geometry). Other formulas below
are standard textbook forms and are marked "to verify against PDF" where the
original paper was not accessible.

## Ranked candidates

| # | Benchmark | Formula / problem | Regime | Reference value | GKX feasibility | Effort |
|---|-----------|-------------------|--------|-----------------|-----------------|--------|
| 1 | Ideal s-alpha ballooning boundary (Connor-Hastie-Taylor 1978) | Eq. (1), 1-D ODE, Newcomb zero-crossing | infinite-n, ideal MHD | alpha_c(s): 0.33 (s=0.2), 0.36 (0.4), 0.43 (0.6), 0.52 (0.8), 0.61 (1.0), 0.88 (1.5) — this script | Compare GKX KBM beta_crit at a/L_T -> consistent alpha, k_y rho -> small, with A_par+B_par; KBM beta_crit must sit at or below the ideal limit (with B_par) | low |
| 2 | Diamagnetic-modified MHD KBM (Tang-Connor-Hastie 1980; Aleynikova-Zocco 2017) | Eq. (2): omega(omega - omega_*pi) = -gamma_MHD^2 | strong drive, eta_e = 0, non-resonant ions | omega_r = omega_*pi/2, gamma^2 = gamma_MHD^2 - omega_*pi^2/4 | Direct: measure omega_r/omega_*pi -> 0.5 as beta grows at ky rho 0.05-0.1 (arXiv 2505.10153 confirms in GK) | low |
| 3 | Rosenbluth-Hinton residual | Eq. (4) | collisionless, large aspect ratio, k_r rho -> 0 | q=1.4, eps=0.18: 0.1192 | already planned; Xiao-Catto shaping correction adds a second check | low |
| 4 | GAM frequency/damping (Sugama-Watanabe 2006, Gao 2006-08) | Eq. (5) + Landau damping exp(-q^2 ...) | large q, circular | omega R/v_ti = 1.802 (q=1.4, tau=1) | time trace of zonal phi; circular Miller | low |
| 5 | Resonant KBM threshold, CBC variant (arXiv 2505.10153) | Eq. (3) with Z-function ion response | R/L=10 all, s=0.8 / 0.1, ky rho_s=0.1 | beta_crit ~0.5% (s=0.8), 0.05-0.1% (s=0.1); peak gamma at beta 1.5-2% | same regime, needs nperiod>=3 (see KBM-VEL log) | medium |
| 6 | Toroidal ITG local dispersion (Kim-Horton-Dong / Biglari-Diamond-Rosenbluth) and Romanelli eta_i^crit | local 0-D root with Z / drift-resonance integrals | adiabatic electrons, k_par fixed | Romanelli: (R/L_T)_crit ~ 4/3 (1+tau)(1+2s/q) at flat density (to verify) | via strongly localized mode; compare trend not value (~5%) | medium |
| 7 | Slab ITG (Kadomtsev/ Hahm-Tang) threshold eta_i^crit ~ 2 (b->0) | 0-D Z-function dispersion | slab, adiabatic electrons | eta_c ~ 2 (b->0) | needs a slab-geometry option (zero curvature) | medium |
| 8 | Collisionless TEM local theory (Kadomtsev-Pogutse; Connor/Hastie) | bounce-averaged e-response | trapped fraction ~sqrt(2 eps) | trend only | kinetic electrons; weak analytic precision | high |
| 9 | ETG = ITG with species swap | isomorphism | adiabatic ions | identical normalized gamma | trivial swap test, exact to solver tolerance | low |
| 10 | Microtearing / TAE gap / Dimits shift | no clean <=1% closed form | - | TAE gap omega = v_A/(2qR) only | not recommended as precision tests | - |

Priority for <=1% tests: 1, 2, 3, 4, 9. Items 5-7 are trend (5-10%) tests.

## Equations

(1) s-alpha ideal ballooning (Connor, Hastie, Taylor 1978):
$$\frac{d}{d\theta}\Big[(1+h^2)\frac{dF}{d\theta}\Big] + \alpha(\cos\theta + h\sin\theta)F = 0,\qquad h=\hat s\theta-\alpha\sin\theta,$$
$\alpha = -q^2R\,d\beta/dr$. Marginal: $F(0)=1,F'(0)=0$ with no zero on $[0,\infty)$. The script
truncates at $\theta_{\max}=60$; values are upper-side estimates to ~1e-2 and agree with the
familiar diagram ($\alpha_c\approx0.6$ at $\hat s=1$).

(2) Diamagnetic MHD / strongly driven KBM: replace $\Omega^2\to\omega(\omega-\omega_{*pi})$,
$\omega_{*pi}=\omega_{*i}(1+\eta_i)$:
$$\omega = \tfrac12\omega_{*pi} + i\sqrt{\gamma_{MHD}^2-\omega_{*pi}^2/4},\qquad
\beta_{crit}^{KBM}>\beta_{crit}^{MHD}\ \text{(stabilizing shift)}.$$

(3) General-geometry KBM equation (arXiv 2505.10153, dimensionless):
$$\frac{1}{\hat J\hat B}\partial_\theta\Big(\frac{g^{yy}}{\hat J\hat B^3}\partial_\theta\Phi\Big)
=\Omega^2\frac{\beta_i\tau^2}{2\epsilon_n^2}\Big\{\frac{Q-\alpha_{0,i}}{1+\tau-\tau Q}\alpha_{0,e}-\alpha_{1,e}\frac{2\Omega_\kappa}{\Omega}\Big\}\Phi,$$
$\alpha_{m,j}=1-\frac{\omega_{*j}}{\omega}(1+m\eta_j)$, and resonant ion response
$\tau Q=[(\tau\Omega+1-\eta_i/2)\Gamma_0+\eta_i b(\Gamma_1-\Gamma_0)]D_0+\eta_i\Gamma_0D_1$,
$D_0=-Z(\xi)/(4\Omega_\kappa\xi)$, $D_1=-[1+\xi Z(\xi)]/(4\Omega_\kappa)$, $\xi^2=\Omega/4\Omega_\kappa$.
Non-resonant limit: $Q\approx\alpha_{0,i}+((\omega_{\nabla B}+\omega_\kappa)/\omega-b)\alpha_{1,i}$.
This 1-D nonlinear eigenproblem is the recommended high-accuracy reference solver for GKX
(same geometry arrays, independent physics path).

(4) Rosenbluth-Hinton: $\phi(\infty)/\phi(0)=1/(1+1.6q^2/\sqrt\epsilon)$; Xiao-Catto adds
shaping/finite-$\epsilon$ corrections.

(5) GAM (Sugama-Watanabe): $\omega_G^2=\frac{v_{ti}^2}{R^2}\big(\tfrac74+\tau\big)\Big[1+\frac{46+32\tau+8\tau^2}{(7+4\tau)^2}\frac{1}{2q^2}\Big]$, $v_{ti}=\sqrt{2T_i/m_i}$ (to verify the
$q^{-2}$ coefficient against the PDF before promoting).

## GKX regime match
- A_par + B_par are required for (1)/(2): without B_par the KBM threshold is known to shift
  down strongly (KBM-VEL log: +27% gamma from B_par).
- s-alpha geometry must use alpha consistent with the gradients (Aleynikova-Zocco: drifts
  consistent with the pressure gradient are what make (2) match GK).
- Parallel domain: nperiod>=3 and explicit `damp_ends_rate` (KBM-VEL findings).

## Papers needed (paywalled / not retrieved)
1. J.W. Connor, R.J. Hastie, J.B. Taylor, PRL 40, 396 (1978), doi:10.1103/PhysRevLett.40.396
2. W.M. Tang, J.W. Connor, R.J. Hastie, Nucl. Fusion 20, 1439 (1980), doi:10.1088/0029-5515/20/11/011
3. K. Aleynikova, A. Zocco, Phys. Plasmas 24, 092106 (2017), doi:10.1063/1.5000052
4. R.J. Hastie, T.C. Hesketh, Nucl. Fusion 21, 651 (1981), doi:10.1088/0029-5515/21/6/002
5. T.M. Antonsen, B. Lane, Phys. Fluids 23, 1205 (1980), doi:10.1063/1.863121
6. S.-T. Tsai, L. Chen, Phys. Fluids B 5, 3284 (1993), doi:10.1063/1.860625
7. M.N. Rosenbluth, F.L. Hinton, PRL 80, 724 (1998), doi:10.1103/PhysRevLett.80.724
8. Y. Xiao, P.J. Catto, Phys. Plasmas 13, 082307 (2006), doi:10.1063/1.2245215
9. H. Sugama, T.-H. Watanabe, J. Plasma Phys. 72, 825 (2006), doi:10.1017/S0022377806004958
10. F. Romanelli, Phys. Fluids B 1, 1018 (1989), doi:10.1063/1.859023
11. J.Y. Kim, W. Horton, J.Q. Dong, Phys. Fluids B 5, 4030 (1993), doi:10.1063/1.860622
12. H. Biglari, P.H. Diamond, M.N. Rosenbluth, Phys. Fluids B 1, 109 (1989), doi:10.1063/1.859206
13. M.J. Pueschel, M. Kammerer, F. Jenko, Phys. Plasmas 15, 102310 (2008), doi:10.1063/1.3005380
14. A. Zocco et al., KBM in stellarators, J. Plasma Phys. (2018) — confirm DOI.
DOIs were written from memory; confirm each on the publisher page before download.

Source read: https://arxiv.org/html/2505.10153
