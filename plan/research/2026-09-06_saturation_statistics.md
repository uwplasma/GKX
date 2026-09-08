# Statistics of time-averaged turbulent fluxes: estimators, stopping, reporting

Researched 2026-09-06. Sources fetched unless marked UNVERIFIED.

## Estimators and rules

| Item | Formula / rule | Source |
|---|---|---|
| Integrated autocorrelation time | τ = 1+2Σ_k ρ_k; Var[X̄] = σ²τ/n; n_eff = n/τ | [Parker et al. 2018](https://arxiv.org/pdf/1807.04779) eqs 9–12 |
| Sokal windowing (emcee) | τ̂(M) with smallest M ≥ Cτ̂(M), C=5; trust τ̂ only when n ≳ 50τ | [emcee autocorr](https://emcee.readthedocs.io/en/stable/tutorials/autocorr/) |
| Batch-means τ | τ ≈ m·s²_m/s², n^{1/3} batches of size n^{2/3} | Parker 2018 App. A |
| Sub-interval averaging (GK community) | Var[X̄] ≈ σ²_Y/N over N sub-windows; largest N with lag-1 correlation insignificant | [Vaezi & Holland 2019](https://arxiv.org/pdf/1902.10879) §II–III |
| BMBC (Russo–Luchini) | lag-1 corrected batch means; batch-size bias | [Rezaeiravesh et al. 2023](https://arxiv.org/pdf/2310.08676) eq 13 |
| Continuous-signal SE | SE = σ/√(t_ave/t_auto) | [St-Onge et al. 2022](https://arxiv.org/pdf/2201.01506); DNS form [arXiv 1802.01056](https://arxiv.org/pdf/1802.01056) |
| AR-model sampling error, Bayesian Richardson | sampling error from an AR fit; discretization error via Richardson with sampling error propagated | [Oliver et al. 2014](https://arxiv.org/abs/1311.0828); used by Lee–Moser |
| MSER-5 transient truncation | d* = argmin_d (n−d)⁻²Σ_{i>d}(X_i−X̄_{n,d})² on batch-of-5 means; d* in the second half ⇒ invalid, run longer | [Hoad & Robinson 2011](https://www.informs-sim.org/wsc11papers/044.pdf) |
| Geweke | z = (mean first 10% − mean last 50%)/SE_spectral; |z|<2 | [coda geweke.diag](https://www.rdocumentation.org/packages/coda/versions/0.19-4.1/topics/geweke.diag) |
| Heidelberger–Welch | Cramér–von Mises stationarity discarding 10…50%; then half-width/mean ≤ 0.1 | [coda heidel.diag](https://search.r-project.org/CRAN/refmans/coda/html/heidel.diag.html) |
| Relative fixed-width stopping | 2z_{δ/2}σ̂_n/√n + p(n) ≤ ε|θ̂_n|; p(n)=I(n≤n*)+n⁻¹; batch means with b_n=⌊√n⌋ | [Flegal & Gong](https://arxiv.org/pdf/1303.0238) §2, §3.1 |
| Multivariate ESS | ESS = n(|Λ|/|Σ|)^{1/p}; stop when ESS ≥ 2^{2/p}π χ²_{1−α,p}/[(pΓ(p/2))^{2/p}ε²]; check at 10% increments | [Vats, Flegal & Jones](https://arxiv.org/pdf/1512.07713) eqs 7–9; [mcmcse minESS](https://rdrr.io/cran/mcmcse/man/minESS.html) |
| Validity of sequential stopping | needs an FCLT and a consistent variance estimator; early stopping on a bad σ̂_n undercovers | [Glynn & Whitt 1992](https://projecteuclid.org/euclid.aoap/1177005777); review [arXiv 2510.22688](https://arxiv.org/html/2510.22688) |
| Climate analogue | control runs ≥ 500 yr after spin-up; residual drift documented | [Eyring et al. 2016](https://gmd.copernicus.org/articles/9/1937/2016/gmd-9-1937-2016.pdf) |

What the GK literature actually does: GX runs CBC to t=1000 (Boltzmann) or 600
(kinetic) and its Table 1 "±" is undefined; stella/GENE W7-X average over
t∈[1500,1900] with no error method; Hoffmann's error bars are fluctuation
amplitude, not standard error; Gysela near marginality averages "over several
tens of correlation times". Vaezi–Holland: the CGYRO mean converges after ~500
a/c_s at experimental gradients, ~1000 at 0.8×, not by 2500 at 0.5×; 5%
fractional SE needs ~2000 a/c_s and >5000 near marginality; skewness grows
toward marginality. Parker: GENE flux τ ≈ 11.7 samples × 0.7 R₀/v_Ti with
right-tailed bursts. [Papadopoulos et al.](https://arxiv.org/pdf/2212.14219):
W7-X ITG flux has Hurst H>0.5, avalanches.

UNVERIFIED or not found: "Oberparleiter 2016 statistical uncertainty" (only
PoP 23, 042509 on neoclassical–ITG interplay exists); "Vaezi & Holland 2018
Fundamental limits" (PoP 25, 102309 is on input-parameter uncertainty); Nevins
2005 is about PIC noise; T3D/GX saturation detection is not documented.

## A protocol a CI script can implement

1. Pre-register in the input: ε_rel, α=0.05, N_τ, T_min, T_max, checkpoint
   spacing. Never tune after seeing Q(t).
2. Save Q_s, Γ_s, Π_s per species at Δt_s ≈ 0.2 τ_int.
3. Transient: MSER-5 on batch-of-5 means; accept d* only in the first half, else
   extend 10%. Cross-check Geweke |z|<2 and Heidelberger–Welch.
4. Post-transient τ_int by Sokal window (C=5) and by batch means; flag if they
   differ by more than 2×.
5. Require T_avg ≥ 50 τ_int; T_min = 300 a/c_s; the τ check only after n*.
6. Batch length b = max(⌊√n⌋, 5τ_int) samples with ≥20 batches; choose b as the
   largest count with |ρ_Y(1)| below its standard error; apply the BMBC lag-1
   correction.
7. Report each flux as mean ± t_{K−1,0.975}·SE with ESS = n/τ_int and T_avg/τ_int.
8. Stationarity: split-half means differ by < 2 pooled SE; trend slope × T_avg
   < SE.
9. Multivariate stop: mBM on (Q_i, Q_e, Γ, …); mESS ≥ minESS(p, ε=0.3, α=0.05)
   (p=1: 171; p=2: 209).
10. Precision stop: 95% half-width ≤ ε_rel·|mean|, ε_rel=0.05.
11. Check rules only at 10% increments of n and not before n*.
12. Persistence: criteria hold at two consecutive checkpoints (a local rule, not
    from the literature).
13. Near marginality (skewness>1 or H>0.5): fit a log-link ARMA and forecast the
    T needed; report "unconverged" if orders disagree.
14. Emit JSON with transient cut, both τ_int, ESS, mESS, CI, tests passed, and
    whether T_max was hit. Never a bare mean.

## Comparisons and ladders

Report Δ = Q̄_A − Q̄_B with CI ±1.96√(SE_A²+SE_B²); "converged" means the CI
contains 0 **and** its half-width < ε_rel·Q̄. A wide CI is inconclusive, not
agreement. Propagate sampling SE into Richardson extrapolation (Oliver).

## Pitfalls

Fluctuation std as an error bar; treating samples as independent (7× variance
underestimate at φ₁=0.75); kernel-sensitive integral τ; batch-size bias that
cannot be automated in situ; single-gradient comparisons inside the Dimits shift;
MSER minimum near the series end; early stopping on a noisy σ̂_n; discretization
error rivaling sampling error; PIC noise faking saturation.
