# Collision operators and hypercollisions: review for GKX (2026-10-15)

Scope: what the reference codes use, which hypercollision forms vanish with
velocity resolution, and how moment codes build field-particle terms. GX facts
were checked against the GX source; the rest is from the cited papers and is
marked where a detail was not re-verified.

## Reference codes

| code | default collision model | reference |
|---|---|---|
| GX | conserving Dougherty in Laguerre-Hermite form; Hermite hypercollisions | Mandell, Dorland & Landreman, JPP 84, 905840108 (2018), [doi:10.1017/S0022377818000041](https://doi.org/10.1017/S0022377818000041); Mandell et al., JPP 90, 905900402 (2024), [doi:10.1017/S0022377824000631](https://doi.org/10.1017/S0022377824000631) |
| GS2 | model operator: pitch-angle scattering + energy diffusion, momentum and energy restoring terms, gyroaveraged k_perp terms | Abel et al., PoP 15, 122509 (2008), [doi:10.1063/1.3046067](https://doi.org/10.1063/1.3046067); Barnes et al., PoP 16, 072107 (2009), [doi:10.1063/1.3155085](https://doi.org/10.1063/1.3155085) |
| stella | Dougherty-type; a linearized Fokker-Planck option exists in the source (`coll_fokkerplanck`); no paper describing it was found | Barnes, Parra & Landreman, JCP 391, 365 (2019), [doi:10.1016/j.jcp.2019.01.025](https://doi.org/10.1016/j.jcp.2019.01.025); [stella source](https://github.com/stellaGK/stella) |
| GENE | linearized Landau / Sugama, multi-species, finite k_perp | Crandall et al., CPC 255, 107360 (2020), [doi:10.1016/j.cpc.2020.107360](https://doi.org/10.1016/j.cpc.2020.107360) |
| GyacoMo (moments) | Coulomb, Sugama, Dougherty as precomputed gyro-moment matrices | Frei et al., JPP 87, 905870501 (2021), [arXiv:2104.11480](https://arxiv.org/abs/2104.11480); Frei, Ernst & Ricci, PoP 29, 093902 (2022), [arXiv:2202.06293](https://arxiv.org/abs/2202.06293); Jorge, Frei & Ricci, JPP 85, 905850604 (2019), [arXiv:1906.03252](https://arxiv.org/abs/1906.03252) |
| CGYRO / NEO | Sugama or full linearized FP; collision step implicit with a precomputed inverse per (k, theta) | Candy, Belli & Bravenec, JCP 324, 73 (2016), [doi:10.1016/j.jcp.2016.07.039](https://doi.org/10.1016/j.jcp.2016.07.039); Belli & Candy, PPCF 54, 015015 (2012), [doi:10.1088/0741-3335/54/1/015015](https://doi.org/10.1088/0741-3335/54/1/015015) |
| XGC | fully nonlinear Fokker-Planck-Landau, conservative finite volume | Hager et al., JCP 315, 644 (2016), [doi:10.1016/j.jcp.2016.03.064](https://doi.org/10.1016/j.jcp.2016.03.064) |
| GKW | pitch-angle + energy scattering with momentum and energy restoring terms | Peeters et al., CPC 180, 2650 (2009), [doi:10.1016/j.cpc.2009.07.001](https://doi.org/10.1016/j.cpc.2009.07.001) |

Citations checked against arXiv and Crossref records on 2026-10-15.

## GX hypercollisions (source-checked)

- Default `p_hyper_m = min(20, Nm/2)` (integer division), `nu_hyper_m = 1`, `nu_hyper_l = 0`.
- The `|k_z|` form: `-nu_hyper_m 2.3 (p+1/2) M^{-1/2} (m/M)^p v_t |k_par| g_{l,m}` for `m > 2`, `M = Nm - 1`. GKX matches it.
- The constant form adds `v_t [Nl nu_l (l/Nl)^{p_l} + Nm nu_m (m/Nm)^{p_m}] + nu_lm ((2l+m)/(2Nl+Nm))^{p_lm}` for `m > 2` or `l > 1`.

## Hypercollisions that vanish with resolution

- A sink `nu (m/M)^p` at fixed `m` falls as `M^{-p}`, so a ladder `M, 2M, 4M` is a convergence test once `p >= ~4`. GX's `(p+1/2)/sqrt(M)` prefactor keeps the last-mode rate at about `sqrt(M) k_par v_t`, the streaming coupling there, which absorbs the outgoing phase-space flux without reflection (Hammett & Perkins, PRL 64, 3019 (1990), [doi:10.1103/PhysRevLett.64.3019](https://doi.org/10.1103/PhysRevLett.64.3019); Parker & Dellar, JPP 81, 305810203 (2015), [doi:10.1017/S0022377814001287](https://doi.org/10.1017/S0022377814001287)).
- The Laguerre direction has no streaming flux to match. The GX constant-branch prefactor `Nl (l/Nl)^{p_l}` also falls at fixed `l` as `Nl^{1-p_l}`.
- In turbulence phase mixing is partly cancelled by the plasma echo, so modest `M` with hypercollisions can suffice (Parker et al., PoP 23, 070703 (2016), [doi:10.1063/1.4958954](https://doi.org/10.1063/1.4958954); Kanekar et al., JPP 81, 305810104 (2015), [doi:10.1017/S0022377814000622](https://doi.org/10.1017/S0022377814000622)).

## Field-particle terms in moment codes

- The like-species linearized operator is rotationally invariant, so it is block diagonal in the Legendre index `l` of the Burnett (Legendre-Sonine) basis (Ji & Held, PoP 13, 102103 (2006), [doi:10.1063/1.2356320](https://doi.org/10.1063/1.2356320)). Hermite-Laguerre codes use an exact basis map to Legendre-Sonine and precompute the matrix once (Jorge et al. 2019; Frei et al. 2021).
- Conservation is an exact null space (density, parallel momentum, energy). The H-theorem is a symmetric negative-semidefinite matrix in the orthonormal basis. Both are cheap tests.
- Spitzer-Harm conductivity ratio to the Lorentz gas, `gamma_E(Z)`: 0.5816 (Z=1), 0.6833 (2), 0.7849 (4), 0.9225 (16) (Spitzer & Harm, Phys. Rev. 89, 977 (1953), [doi:10.1103/PhysRev.89.977](https://doi.org/10.1103/PhysRev.89.977); the values are the standard table, not re-read from the paper).

## What GKX does (this lane)

- New default `collision_operator = "landau"`: the linearized Landau operator, drift-kinetic limit, any `(Nl, Nm)`, any species. Its Legendre-Sonine blocks come from the weak form of the Rosenbluth diffusion (test part) and the multipole Rosenbluth potentials (field part), integrated by composite Gauss quadrature. An exact orthogonal map takes them to the Hermite-Laguerre box. This is the same structure as the moment codes above, but evaluated numerically in float64 rather than from closed-form multiprecision coefficients.
- Measurements and their commands are in `results/` and in the PR body.
