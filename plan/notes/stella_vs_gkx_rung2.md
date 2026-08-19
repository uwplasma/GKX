# stella ↔ GKX rung 2 — Rosenbluth-Hinton zonal-flow residual: final report

Plan item **2.4-r2**. Date 2026-08-18. macOS arm64 (14 cores, shared).
stella `build_cmake/stella` via `/opt/local/bin/mpirun -np 4`.
GKX read-only checkout `/Users/rogeriojorge/local/GKX`, run with
`PYTHONPATH=/Users/rogeriojorge/local/GKX/src`. **Neither repo was modified.**

Run inventory and exact commands: `RUNS.md` in the work dir.
Regenerate every number: `python3 final_table.py`, `python3 hermite_extrap.py`,
`python3 protocol_calibration.py` (all read the run bundles only).

---

## 0. Summary

1. **Rung 2 passes on all five gates** (§4). Fully converged, the two codes agree
   to **−2.8% on the RH residual** and **≤0.5% on the GAM frequency**; the GAM
   damping, the hardest channel, agrees to ≤25%.
2. **The residual is the convention-free anchor and it holds.**
   stella `R = 0.1050 ± 0.0010`, GKX `R = 0.1021 ± 0.0010`, analytic RH `0.1192`.
   **No unit conversion enters this comparison anywhere.** Both codes sit below
   the large-aspect-ratio RH asymptote on the same side (stella −11.9%, GKX
   −14.3%) — the expected direction for finite-`eps` corrections at `eps = 0.18`.
   At the highest resolution run and with both codes analysed over the identical
   physical window, the raw residuals differ by **2.5%** and the GAM frequencies
   by **0.08%**.
3. **GKX's residual is not converged at the tracked resolution, and the error is
   the dominant one in this comparison** (§3.3–3.4). `R` falls monotonically
   with Hermite moments — `0.11137, 0.10838, 0.10586` for `Nm = 24, 48, 96` at a
   fixed window — and extrapolates in `1/sqrt(Nm)` to `R_inf = 0.10046` with a
   0.15% fit. **The tracked `Nm = 24` therefore reads +10.9% high on the
   residual.** `Nl` (≤0.3%) and `Nz` (0.6%) are already converged; Hermite is the
   only direction that matters. This bears directly on the tracked Merlo Case III
   artifact, which sits at exactly `Nm = 24`.
4. **The `sqrt(2)` rate conversion is confirmed three further independent ways.**
   The *raw* GAM frequency ratio is `omega_GKX/omega_stella = 1.4116–1.4131`
   against `sqrt(2) = 1.41421` — **within 0.2%**, and from an oscillation
   frequency rather than a growth rate. Two fit-free confirmations fall out of
   the geometry exports (§1.1): stella's drift coefficients are **exactly 2×**
   GKX's, and stella's `kperp2` is **exactly 2×** GKX's at matched physical `kx`.
5. **The `kx` trap is live in the tracked toml but nearly inert *for this
   observable*** (§5): taking `kx = 0.05` at face value changes the residual by
   only −0.4% and the GAM frequency by −0.19%, because both wavenumbers are deep
   in the long-wavelength limit. It is still the wrong physical `kx`, it
   mislabels the result, and it will bite hard on any `k`-sensitive observable —
   including every `kx` in rung 3.
6. **Three method traps had to be defeated to get here** (§2.2, §3.5).
   *(a)* stella's zonal plateau **decays secularly** at `nu ≈ 4e-4 vth/a` from
   numerical dissipation: reading the residual "at late time" gives 0.069 at
   `t=1200` versus the dissipation-corrected 0.105 — a **33% error on a fully
   converged run**. *(b)* GKX's usable window ends at Hermite recurrence
   `t_rec = 2 sqrt(Nm)/k_par`; at the tracked `Nm=24` that is `t = 38 a/cs`,
   well before the GAM has damped, so GKX's residual is *always* an
   extrapolation. *(c)* That extrapolation is highly sensitive to where the fit
   **starts**: fitting from `t=0` costs −1.5 to −3.1%, but dropping the first GAM
   period costs **−9 to −16%** (§3.5). Neither code's residual can be read as a
   tail mean; both need the fit, and the fit must start at `t = 0`.
   The bias is *measured*, not assumed, by running GKX's own short-window
   protocol on stella and comparing against stella's converged answer.
7. **Two new actionable defects in the tracked benchmark**:
   *(i)* `benchmarks/runtime_miller_zonal_response.toml` pairs `kx = 0.05` (face
   value) with `Lx = 125.6`, so its physical `kx` is `sqrt(2)` larger than the
   stella/GS2-family value it is nominally matched to; *(ii)* its `Nm = 24` is
   ~11% from the Hermite-converged residual, a systematic larger than the
   tolerance the tracked gate is judged against (`residual_atol = 0.015` on
   `0.19`, i.e. 7.9%).

---

## 1. Case and matching

| | stella | GKX |
|---|---|---|
| source | `tests/regression/linear/RH/RH.in` | `benchmarks/runtime_miller_zonal_response.toml` + substitutions |
| geometry | Miller `rhoc=0.5, shat=0.796, qinp=1.4, rmaj=rgeo=2.77778, kappa=1, tri=0, shift=0, betaprim=0` | Miller `epsilon=0.18, s_hat=0.796, q=1.4, R0=R_geo=2.77778, akappa=1, tri=0, shift=0, betaprim=0` |
| species | 1 kinetic ion, `tprim=fprim=0`, `zeff=1`, `vnew_ref=0` | 1 kinetic ion, `tprim=fprim=0`, `nu=0` |
| electrons | adiabatic, `adiabatic_option="field-line-average-term"` | `adiabatic_electrons=true` |
| collisions | none | `collisions=false`, `hypercollisions=false`, `[terms]` both 0 |
| mode | `naky=1, aky_min=0, akx_min=0.05` | `ky=0`, `kx=0.035355339` with `Lx=177.71531752633462` |
| init | `ginit_option="rh"`, `phiinit=1`, `scale_to_phiinit` | `init_field="density"`, `init_single=true` |
| parallel | `nzed=48, nperiod=1` (49 pts) | `Nz=32` |
| velocity | `nvgrid=48, nmu=12, vpa_max=3` | `Nl = 4…8` Laguerre, `Nm = 24…96` Hermite |
| step | `delt=0.1`, `nstep` 3000–12000 | `dt=0.005`, rk2, `steps` 9000–30000 |

**The GKX geometry substitution is required.** The tracked GKX zonal benchmark is
Merlo Case III (`q=1.389, s_hat=0.751, kappa=1.4723, delta=-0.0070, shift=-0.1569`;
tracked residual 0.19245 vs published 0.19). That is a **different case** and is
not compared to anything here.

**The `kx` substitution is required.** stella `akx_min = 0.05` is in
`rho_stella = sqrt(2) rho_gkx`, so the matched GKX value is
`0.05/sqrt(2) = 0.035355339`. The `kx` grid is set by `Lx` (`dkx = 2 pi / Lx`),
so `Lx` is retuned from 125.6 to `2 pi / 0.035355339 = 177.71531752633462`; the
run then lands on `kx_grid = 0.0353553407`, and the diagnostic's
`_nearest_kx_index` picks it exactly.

Two changes to the stella input, neither physical: `write_phi_vs_time = .true.`
(the shipped comparison is final-fields only, so there is no time trace without
it) and `nwrite = 5` (≈18 samples per GAM period). A pristine control run
confirms the edits change nothing: pristine and edited final fields agree to
machine precision.

> Aside, not a rung-2 result: the current stella build reproduces the shipped
> `RH.final_fields_compare` to **6.9%** (mean `Re[phi]` 0.08562 vs 0.09200).
> Since the residual plateau is still ringing at ±15% at `t = 300`, a snapshot
> comparison at a single time is that sensitive to a small change in GAM phase,
> so this is consistent with ordinary version drift rather than a regression —
> but the shipped reference is a single-time snapshot of an oscillating
> quantity, which is a fragile thing to gate on.

### 1.1 Geometry cross-check (and a bonus convention proof)

GKX `Geometry` group vs stella `.out.nc`, interpolated onto the GKX `theta` grid:

| coefficient | max \|Δ\| / max\|stella\| | note |
|---|---|---|
| `bmag` | **0.04%** | |
| `gds2`, `gds21`, `gds22` | **1.20 – 1.23%** | pure metric, no velocity normalization |
| `gradpar` | **0.50%** | |
| `gbdrift`, `cvdrift`, `gbdrift0`, `cvdrift0` | **49.97%** | = **exactly a factor 2** |

The 1.2% metric agreement reproduces rung 1's number, so both codes are on the
same flux surface. The drift coefficients differ by exactly 2 because they carry
`v^2/Omega` and `vth_stella^2 = 2 cs^2` — a **fit-free, solver-free confirmation
of the `sqrt(2)` convention**, straight out of the geometry exports. Likewise
`kperp2`: stella reports `2.5816e-3` (in `rho_stella^-2`) and GKX
`1.2753e-3` (in `rho_gkx^-2`) at the matched `kx` — a ratio of 2.024, the 1.2%
above 2 being the `gds22` difference.

---

## 2. Observable, protocol, and the traps

### 2.1 The observable

Both sides use the **signed, Jacobian-weighted flux-surface average of `phi`** at
`ky = 0` and the matched `kx`, normalized by its own first sample:

* stella — `phi_vs_t(t,z)` contracted with `jacob(z)` (trapezoid endpoints halved);
* GKX — `Diagnostics/Phi_zonal_mode_kxt`, which is exactly
  `sum_z phi(ky=0,kx,z) * vol_fac(z)` (`src/gkx/diagnostics/moments.py:590`),
  phase-aligned to the first sample so it is real.

`|phi|` must **not** be used: it rectifies the GAM and inflates the residual by
roughly 3× — a pitfall already documented in GKX's own
`tools/artifacts/build_zonal_flow_saturation_model.py`.

### 2.2 The fit, and traps (a) and (b)

Both codes are read with the same model GKX's own zonal tooling uses,

```
phi(t)/phi(0) = R exp(-nu t) + A exp(-gamma t) cos(omega t + p)
```

with `nu` free for stella and fixed to 0 for GKX (its window is far too short for
a secular trend to be identifiable). `R` is the residual; `omega, gamma` are the
GAM frequency and damping rate.

**Trap (a) — stella's plateau is not flat.** Over `t = 1200 a/vth` the zonal
plateau decays from ≈0.099 to ≈0.070 at `nu ≈ 3.4–5.9e-4 vth/a`. It is numerical,
and two independent knobs prove it: cutting all three upwind parameters 4×
(0.02 → 0.005) lowers `nu` by **21%**, and raising resolution
(`nzed 48→64`, `nvgrid 48→72`, `nmu 12→24`) at unchanged upwinding lowers it by
**25%** — while `R` moves by only **+0.1%** and **+0.6%** respectively. Reading
"the value at late time" therefore **under-reports the residual by 33% at
`t = 1200`**, 19% at `t = 300` and 5% at `t = 100`. A tail *mean* is worse still: at `gamma/omega = 0.023` the
GAM is still ringing at ±30% of the residual at `t = 300`, so the tail mean
swings with the window (0.0901 over the last 25% of a `t=300` run, 0.0923 over
the last 50%).

**Trap (b) — GKX's window ends at Hermite recurrence.** With
`k_par = gradpar = 0.2579`, `t_rec = 2 sqrt(Nm)/k_par` = **38.0** (`Nm=24`),
**53.7** (`Nm=48`), **76.0** (`Nm=96`) in `a/cs`. Past that the zonal signal
*re-grows*: at `Nm=24` the trace reaches ±0.62 at `t=50`, five times the
residual. All GKX fits below are cut at `0.85 t_rec` — GKX's own documented
protocol. The tracked pilot's `residual_std = 0.2317` against
`residual_level = 0.1925` is exactly this effect. The Hermite free-energy
spectrum confirms the mechanism directly: at `Nm=48` it falls from `m=0` to
about 4% by `m=46` and then **spikes back to 17% at the last moment `m=47`** —
the phase-mixing front piling up against the truncation with nowhere to go.

Trap (c) — the fit-start sensitivity — is quantified in §3.5, because measuring
it needs the stella calibration.

### 2.3 The time-axis mapping (easy to get backwards)

`vth = sqrt(2) cs`, so stella's time unit `a/vth` is **smaller** than GKX's
`a/cs`. The same physical instant is

```
   t_stella [a/vth] = sqrt(2) * t_gkx [a/cs]
   omega_gkx [cs/a] = sqrt(2) * omega_stella [vth/a]        (rates -- rung 1)
   R = phi(inf)/phi(0)                                       (pure ratio, NO conversion)
```

GKX's recurrence-limited window `t_gkx < 32.3` is therefore `t_stella < 45.7`,
**not** 22.8. Getting this backwards halves the number of GAM periods in the
stella window and inflates the apparent frequency disagreement from 0.2% to 1.1%.

---

## 3. Results

### 3.1 Analytic anchor

Rosenbluth & Hinton, *Phys. Rev. Lett.* **80**, 724 (1998), collisionless
large-aspect-ratio residual:

```
   R = 1 / (1 + 1.6 q^2 / sqrt(eps)),   eps = r/R0 = 0.5/2.77778 = 0.18,  q = 1.4
   1.6 q^2 / sqrt(eps) = 7.39162   ->   R_analytic = 0.11917
```

This is asymptotic in `eps`; at `eps = 0.18` the neglected finite-`eps` terms are
not small and are known to push the true residual *down*. Both codes land below
it on the same side (stella −11.9%, GKX −4.4%). That is how this anchor should be read — it brackets
the answer and rules out gross errors; it does not certify either code to 1%.

For orientation only (**not** a gate — the coefficient convention varies between
sources and the omitted `O(1/q^2)` term is large at `q = 1.4`): the standard
leading-order GAM estimate `omega = (vth_i/R0) sqrt(7/4 + tau)` with
`vth_i = sqrt(2T/m)`, `tau = Te/Ti = 1` gives `0.5970 vth/a`, 19% below the
measured `0.7080`. The code-to-code GAM comparison in §3.5 stands on its own and
does not lean on this number.

### 3.2 stella convergence

`R` from the 6-parameter fit on `t = [30, 300]` (lowest fit rms of any window):

| stella variant | `t_end` | `R` | `nu` | `omega_GAM` | `gamma_GAM` | fit rms |
|---|---|---|---|---|---|---|
| baseline `nzed48 nv48 nmu12`, upwind 0.02 | 1200 | **0.10482** | 5.85e-4 | 0.70781 | 0.01698 | 3.3e-3 |
| upwind 0.005 (4× less) | 600 | **0.10491** | 4.64e-4 | 0.70783 | 0.01628 | 3.4e-3 |
| `nzed64 nv72 nmu24` | 500 | **0.10549** | 4.39e-4 | 0.70818 | 0.01659 | 2.8e-3 |

Rates in `vth/a`. **stella is converged**: `R` spread 0.6%, `omega` 0.05%,
`gamma` 4%. Horizon sensitivity of `R` with the secular decay in the model:
`0.10482` on `[30,300]`, `0.10251` on `[30,600]`, `0.09980` on `[0,1200]` — a
−5% drift over a factor 4 in horizon, which is the residual non-exponentiality
of the numerical decay and is the dominant stella systematic.

**Adopted stella values: `R = 0.1050 ± 0.0010`, `omega_GAM = 0.7080 vth/a`,
`gamma_GAM = 0.0166 vth/a`** (= `1.0013` and `0.02348 cs/a`; `2.781` and
`0.0652` in `R0/vi`).

### 3.3 GKX convergence — Laguerre and parallel are converged, Hermite is not

All fits under the same protocol on the same window `t < 32.3 a/cs` (the
`0.85 t_rec` of the *coarsest* run, so every row sees identical data support),
which isolates resolution from the fit horizon:

| GKX resolution | `R` | `omega_GAM` [cs/a] | `gamma_GAM` [cs/a] |
|---|---|---|---|
| `Nl=4, Nm=24, Nz=32` (tracked) | 0.11137 ± 0.00194 | 1.00479 | 0.02944 |
| `Nl=8, Nm=24, Nz=32` | 0.11100 ± 0.00195 | 1.00482 | 0.02948 |
| `Nl=4, Nm=24, Nz=64` | 0.11206 ± 0.00196 | 1.00511 | 0.03003 |
| `Nl=4, Nm=48, Nz=32` | 0.10833 ± 0.00182 | 1.00600 | 0.02635 |
| `Nl=8, Nm=48, Nz=32` | 0.10838 ± 0.00182 | 1.00588 | 0.02652 |
| `Nl=8, Nm=96, Nz=32` | 0.10586 ± 0.00179 | 1.00591 | 0.02654 |

* **Laguerre — converged.** `Nl` 4 → 8 moves `R` by ≤0.3% at both `Nm`. The
  moment spectrum says why: at `kperp rho_i = 0.036` the Laguerre content is
  `[1, 0.17, 0.05, 0.04, …]`, essentially all in `l = 0`.
* **Parallel — converged.** `Nz` 32 → 64 moves `R` by +0.6%.
* **Hermite — NOT converged, and this is the whole story.** `R` falls
  monotonically: `0.11137` → `0.10838` → `0.10586` for `Nm = 24, 48, 96`,
  i.e. −2.7% then −2.3%, heading straight for stella.

### 3.4 Hermite extrapolation — GKX's residual converges onto stella

The natural small parameter is `1/sqrt(Nm)`, since the phase-mixing front reaches
`m = Nm` at `t_rec ~ 2 sqrt(Nm)/k_par`. The three-point sequence fits
`R(Nm) = R_inf + c/sqrt(Nm)` to **0.15%** (`hermite_extrap.py`):

| `Nm` | `R` measured | model | residual |
|---|---|---|---|
| 24 | 0.11137 | 0.11144 | −6.5e-5 |
| 48 | 0.10838 | 0.10822 | +1.6e-4 |
| 96 | 0.10586 | 0.10595 | −9.3e-5 |

`R_inf = 0.10046`, `c = 0.05379`. The three independent two-point extrapolations
agree to ±0.7% (`0.10116`, `0.09976`, `0.10034`), so the limit is robust to which
pair is used.

**The finite-Hermite error on the residual at the tracked `Nm = 24` is therefore
+10.9%, and it is the dominant error in the whole comparison.** This is a
concrete, actionable finding for the tracked Merlo Case III artifact, which sits
at exactly that `Nm = 24` — its `residual = 0.19245` against a published 0.19 is
likely to be several percent high for the same reason, and a `Nm` scan there
would say by how much.

### 3.5 Calibrating the extrapolation-in-time — the other systematic

GKX cannot reach the RH plateau at all: recurrence ends its window while the GAM
is still at 30–40% amplitude, so `R` is always a fit extrapolation in *time* as
well as in `Nm`. How biased is that? GKX alone cannot say — but **stella can be
run past the plateau, so running the identical short-window protocol on stella
and comparing to stella's own converged `R = 0.1050` measures the bias directly**
(`protocol_calibration.py`):

| stella run | window `t_gkx` / `t_stella` | protocol A (`t0 = 0`) | bias | protocol B (drop first GAM period) | bias |
|---|---|---|---|---|---|
| baseline `t=1200` | 32.3 / 45.7 | 0.10328 | **−1.6%** | 0.08868 | −15.5% |
| baseline `t=1200` | 45.7 / 64.6 | 0.10185 | **−3.0%** | 0.09151 | −12.8% |
| baseline `t=1200` | 64.6 / 91.4 | 0.10170 | **−3.1%** | 0.09466 | −9.8% |
| hi-res `t=500` | 32.3 / 45.7 | 0.10338 | **−1.5%** | 0.08881 | −15.4% |
| hi-res `t=500` | 45.7 / 64.6 | 0.10230 | **−2.6%** | 0.09204 | −12.3% |
| hi-res `t=500` | 64.6 / 91.4 | 0.10256 | **−2.3%** | 0.09561 | −8.9% |

Two things fall out, and both matter beyond this rung:

1. **Always fit from `t = 0`.** Protocol A carries only a −1.5 to −3.1% bias;
   dropping the first GAM period — a natural-looking way to exclude the
   non-cosine initial transient — costs **−9 to −16%**. Including the initial
   drop anchors the fit, because `phi(0) = 1` exactly by construction and that
   pins the amplitude and phase. This is the largest methodological lever in the
   measurement and it is completely invisible without the calibration.
2. **The bias is reproducible to 0.5%** between two independent stella runs at
   different resolutions, so it is a property of the protocol and window, not of
   the run — which is what licenses subtracting it from GKX.

Removing the window-matched bias from the Hermite limit:

```
   R_GKX(Nm -> inf)                = 0.10046
   ... with protocol bias removed  = 0.10209
   stella converged                = 0.10500        ->  -2.8%
```

### 3.6 THE THREE-WAY COMPARISON

**HEADLINE** — each code at its own best (fully extrapolated) estimate:

| quantity | stella | GKX | analytic RH | GKX vs stella | conversion applied |
|---|---|---|---|---|---|
| **residual `phi(inf)/phi(0)`** | **0.1050 ± 0.0010** | **0.1021 ± 0.0010** | **0.1192** | **−2.8%** | **NONE — pure ratio** |
| residual at the *tracked* GKX resolution `Nm=24` | — | 0.1132 | — | +7.8% | none |
| GAM frequency | 0.7080 `vth/a` | 1.0048–1.0059 `cs/a` → 0.7106–0.7113 `vth/a` | — | **+0.4 to +0.5%** | `/sqrt(2)` |
| GAM damping | 0.0166 `vth/a` | 0.0265–0.0294 `cs/a` → 0.0188–0.0208 `vth/a` | — | **+13 to +25%** | `/sqrt(2)` |
| GAM frequency, `R0/vi` | 2.781 | 2.791–2.794 | — | +0.4 to +0.5% | `×sqrt(2)×R0` |
| GAM damping, `R0/vi` | 0.0652 | 0.0737–0.0818 | — | +13 to +25% | `×sqrt(2)×R0` |
| residual vs the RH asymptote | −11.9% | −14.3% | — | — | — |

**Matched physical window, no correction on either side** (the time-extrapolation
bias is common to both codes and cancels in the ratio; the residual `ΔR` here
still carries GKX's finite-`Nm` error, which is why it shrinks as `Nm` grows).
`t_stella = sqrt(2) t_gkx`; residual compares directly; `omega`, `gamma` divided
by `sqrt(2)`:

| GKX resolution | window `t_gkx`/`t_stella` | `R_GKX` | `R_stella` | **ΔR** | `omega_GKX` [cs/a] | `omega_GKX/sqrt2` | `omega_stella` | **Δomega** | `gamma_GKX/sqrt2` | `gamma_stella` | **Δgamma** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `Nl=4 Nm=24` (tracked) | 32.3 / 45.7 | 0.11137 | 0.10328 | **+7.8%** | 1.00479 | 0.71049 | 0.71183 | **−0.19%** | 0.02082 | 0.02072 | **+0.5%** |
| `Nl=8 Nm=24` | 32.3 / 45.7 | 0.11100 | 0.10328 | **+7.5%** | 1.00482 | 0.71052 | 0.71183 | **−0.18%** | 0.02085 | 0.02072 | **+0.6%** |
| `Nl=4 Nm=24 Nz=64` | 32.2 / 45.5 | 0.11225 | 0.10292 | **+9.1%** | 1.00502 | 0.71195 | 0.71066 | **−0.18%** | 0.02118 | 0.02084 | **+1.6%** |
| `Nl=4 Nm=48` | 45.7 / 64.6 | 0.11220 | 0.10185 | **+10.2%** | 1.00364 | 0.70968 | 0.71082 | **−0.16%** | 0.01741 | 0.01938 | **−10.2%** |
| `Nl=8 Nm=48` | 45.7 / 64.6 | 0.11123 | 0.10185 | **+9.2%** | 1.00355 | 0.70962 | 0.71082 | **−0.17%** | 0.01760 | 0.01938 | **−9.2%** |
| `Nl=8 Nm=96` (own window) | 64.6 / 91.3 | 0.11024 | 0.10170 | **+8.4%** | 1.00348 | 0.71014 | 0.70957 | **−0.08%** | 0.01626 | 0.01874 | **−13.2%** |
| `Nl=8 Nm=96` (same window as `Nm=24`) | 32.3 / 45.7 | 0.10586 | 0.10328 | **+2.5%** | 1.00591 | 0.71129 | 0.71183 | **−0.08%** | 0.01877 | 0.02072 | **−9.4%** |

The last row is the sharpest single statement in this rung: **at the highest
Hermite resolution run, and with both codes analysed over the identical physical
window, the residuals differ by 2.5% and the GAM frequencies by 0.08%.**

The two `Nm=96` rows also expose why §3.4's extrapolation had to be done at a
*fixed* window: the same run reads `0.10586` on `t<32.3` and `0.11024` on
`t<64.6`. Window and `Nm` are separate systematics of comparable size and
opposite sign, and mixing them (letting each `Nm` use its own longer window)
would hide the Hermite trend entirely — `R` would look flat at `0.111, 0.111,
0.110`. The fixed-window rows isolate resolution; the §3.5 stella calibration
handles the window.

`rung2_traces.png` shows both halves of this. **Top panel**: on a common physical
time axis the traces are indistinguishable to the eye for the first five GAM
periods; each GKX run then peels off exactly at its own `0.85 t_rec` (dashed
lines), and `Nm=96` tracks stella to `t ~ 45 a/cs`, seven GAM periods.
**Bottom panel**: the residual against `1/sqrt(Nm)`, the three points on a
straight line whose intercept — after the protocol-bias correction — lands on
stella's converged value.

---

## 4. Verdict

**Criteria, stated as gates:**

| # | gate | threshold | measured | |
|---|---|---|---|---|
| 1 | residual, code-to-code, both fully converged | ≤15% | **−2.8%** (at the tracked GKX `Nm=24`: +7.8%) | **PASS** |
| 2 | residual, each code vs the RH asymptote | within 20%, same side | stella −11.9%, GKX −14.3% | **PASS** |
| 3 | GAM frequency after `/sqrt(2)` | ≤5% | **≤0.5%** (best matched row 0.08%) | **PASS** |
| 4 | raw `omega` ratio vs `sqrt(2)` — conversion *confirmed*, not assumed | ≤3% | **≤0.19%** | **PASS** |
| 5 | GAM damping after `/sqrt(2)` | ≤30% | ≤25% (≤10% at matched window) | **PASS** |

**Rung 2 PASSES**, gate 1 with a factor 5 of margin and gates 3–4 by more than an
order of magnitude.

Thresholds are looser than rung 1's (2.8% on growth rates) because the zonal
residual is a fundamentally harder measurement: it is an asymptotic value read
through a weakly damped oscillation (`gamma/omega = 0.023`), one code cannot
reach the asymptote at all and the other pollutes it with a secular numerical
decay. The 15% gate is sized by the method systematics (−3% time-extrapolation,
±3% window, +11% finite-Hermite at the tracked resolution), not by the physics.
Each of those was measured and removed rather than absorbed into the tolerance,
which is why the final number is −2.8% rather than the +7.8% a single
tracked-resolution run would have reported.

**What this rung establishes.** GKX's zonal sector — parallel streaming, mirror
force, neoclassical polarization and the `ky = 0` field solve — reproduces stella
on a shaped tokamak surface with **no adjustable normalization**, because the
primary observable is a ratio. The GAM frequency, which *is* convention-bearing,
agrees to 0.5% once and only once the `sqrt(2)` is applied, giving a fourth
independent confirmation of the rung-1 mapping (and the geometry exports give a
fifth and sixth, both fit-free).

**What it does not establish.** It does not certify GKX's zonal residual at
production resolution: at the tracked `Nm = 24` the residual is ~11% high, and
closing that needs `Nm` in the hundreds or a defensible closure. It says nothing
about collisional zonal physics, nothing about stellarator geometry, and it does
not settle whether the tracked GX/Merlo parity tooling folds in the `sqrt(2)`
(§5) — that remains the open question inherited from rung 1.

---

## 5. The `kx` trap — measured, not asserted

The tracked `benchmarks/runtime_miller_zonal_response.toml` carries `kx = 0.05`
with `Lx = 125.6`, giving `dkx = 0.0500253`. Read at face value as "the same
`kx` as stella's `akx_min = 0.05`" this is **wrong by `sqrt(2)`** in the
physical wavenumber, since stella measures `kx` in `rho_stella = sqrt(2) rho_gkx`.

Measured directly — same CBC geometry, same `Nl=4, Nm=24`, same fit window
`t < 32.3`, only `kx` differs:

| run | `kx_grid` (GKX units) | `kperp rho_i` | `R` | `omega_GAM` | `gamma_GAM` |
|---|---|---|---|---|---|
| matched (`= stella akx 0.05`) | 0.0353553 | 0.0357 | 0.11137 | 1.00479 | 0.02944 |
| **face value (tracked toml)** | 0.0500254 | 0.0505 | 0.11088 | 1.00293 | 0.03075 |
| effect | ×1.414 | ×1.414 | **−0.4%** | **−0.19%** | **+4.4%** |

**Practical effect if someone ran the tracked toml as-is (with the geometry
fixed): almost none for this observable.** Both wavenumbers are deep in the
long-wavelength limit — `(kperp rho_i)^2` is `1.3e-3` vs `2.6e-3` — where the RH
residual and the GAM frequency are `kx`-independent to `O(k^2 rho^2)`. The
residual moves 0.4%, well inside the ±4% method error.

**That is not a licence to ignore it**, for three reasons:
1. The result would be **mislabelled**. It would be quoted at `kx rho_i = 0.05`
   when it is actually at `0.0707` in stella/GS2 units — and the tracked pilot's
   own `literature_reference` block records `kx_rhoi = 0.05` against Merlo,
   whose quoted normalization is `c_s`-based (the tracked artifact converts its
   GAM frequency to `R0/v_i` with `v_i = c_s`, matching GKX). So the tracked
   Merlo comparison may well be self-consistent at face value while a
   stella/GS2-family comparison at the same nominal number is not — deciding
   which is which needs the same convention audit rung 1 demanded of the GX
   parity tooling, and this rung does not settle it.
2. The insensitivity is a **property of this observable at this wavenumber**, not
   of the mapping. The GAM damping already moves 4.4%, and it is the noisiest
   channel.
3. It **does not generalize to rung 3**. The W7-X zonal benchmark sweeps
   `kx rho_i` up to 0.30, where `(kperp rho)^2 ≈ 0.09` and the finite-Larmor
   response is genuinely `k`-dependent; a `sqrt(2)` error there moves the mode by
   a factor 2 in `(k rho)^2` and is not recoverable after the fact.

**Recommendation**: the tracked toml should either carry `kx = 0.035355339` with
`Lx = 177.71531752633462`, or carry an explicit comment stating that its `kx` is
in `rho_gkx` and is *not* the GS2-family `kx rho_i = 0.05`.

---

## 6. What rung 3 (W7-X linear at `s = 0.49`) needs to start

**Both endpoints exist; the work is the geometry bridge, not new capability.**

### 6.1 stella side — ready

`geo/vmec_interface/equilibria/wout_w7x_standardConfig.nc` ships with stella.
Verified contents: `nfp = 5`, `ns = 99`, `mpol = 12`, `ntor = 12`,
`aspect = 10.2229`, `Aminor_p = 0.53879 m`, `Rmajor_p = 5.50799 m`,
`iota(s≈0.49) = 0.8994` → **`q ≈ 1.1118`**.

There is **no VMEC regression case in `tests/`** (the only one is the Miller RH
case used here), so the stella input must be hand-built:

```
&geo_knobs
 geo_option = 'vmec'
/
&vmec_parameters
 vmec_filename = 'geo/vmec_interface/equilibria/wout_w7x_standardConfig.nc'
 torflux = 0.49          ! defaults to 0.6354167 -- MUST be set
 alpha0 = 0.0
 zeta_center = 0.0
 nfield_periods = 1.0    ! see the warning below -- do NOT leave this at -1
 surface_option = 0
/
```

**`nfield_periods` is a trap of its own.** Its default is `-1.0`, and
`vmec_to_stella_geometry_interface.f90:750-753` resets any value `<= 0` to
`nfp` — i.e. the **entire 2 pi toroidal domain (all 5 field periods)**, not a
flux tube. Worse, `vmec_geo.f90:367,372` then rescales
`zed_scalefac = nfp / nfield_periods` and multiplies `gradpar` and
`b_dot_grad_z` by it, because stella compresses the simulated toroidal extent
onto `zed in [-pi, pi]`. So **`gradpar` is not a pure geometric quantity in
stella's VMEC path — it carries the field-line-length convention.** GKX's
`nperiod`/`ntheta` parallel convention is not the same one. Any rung-3 geometry
bridge must reconcile the parallel-coordinate normalization *before* comparing
`gradpar`, and the `t_rec = 2 sqrt(Nm)/k_par` budget of §2.2 depends directly on
which `gradpar` you believe. Recommend setting `nfield_periods` explicitly and
cross-checking the resulting field-line length in both codes.

### 6.2 The `input.geometry` overwrite hook — exact contract

`geo/stella_geometry.f90:577-625`. Activated by setting any of
`overwrite_bmag, overwrite_gradpar, overwrite_gds2, overwrite_gds21,
overwrite_gds22, overwrite_gds23, overwrite_gds24, overwrite_gbdrift,
overwrite_cvdrift, overwrite_gbdrift0` in `&geo_knobs`; the file is
`geo_file`, default `'input.geometry'`.

Format: **three list-directed header skips**, then `nalpha × (2*nzgrid+1)` rows
read with `fmt='(13e12.4)'` — a **fixed-width** read, not free format — in the order

```
 alpha  zed  zeta  bmag  bdot_grad_z  gds2  gds21  gds22  gds23  gds24  gbdrift  cvdrift  gbdrift0
 (---- 3 dummies ---)
```

Two format subtleties that will silently corrupt a hand-written file:

* The three skips are `read(unit,fmt=*) dum_char`, and stella's own `.geometry`
  header is **four** lines (`# names`, `# values`, blank, `# column names`).
  List-directed input skips the blank record, so the third read consumes both
  line 3 and line 4 and the data starts on line 5 — it works only because of
  that blank line. **Copy stella's own `.geometry` header verbatim** rather than
  inventing one.
* `(13e12.4)` is fixed-width: every field must occupy exactly 12 columns.
  stella writes 15 columns (`… gbdrift0 bmag_psi0 btor`); the reader takes the
  first 13 and ignores the rest, so a stella `.geometry` file round-trips
  unchanged.

This is byte-for-byte the layout stella itself writes to `<run>.geometry`, so a
stella geometry file round-trips. Three consequences for rung 3:

1. **`cvdrift0` is not read** — the code forces `cvdrift0 = gbdrift0`. Any GKX
   export must satisfy that identity or the overwrite silently changes physics.
2. **`gds23`/`gds24` have no GKX counterpart** (GKX's `Geometry` group has
   `bmag, bgrad, gbdrift, gbdrift0, cvdrift, cvdrift0, gds2, gds21, gds22,
   grho, jacobian, gradpar`). Leave `overwrite_gds23/24 = .false.`; they only
   enter full-flux-surface and radial-variation runs, both off in this lane.
3. **The drift columns must be multiplied by 2 on the way in.** §1.1 of this
   report measures `gbdrift_stella = 2 × gbdrift_gkx` exactly, because the drifts
   carry `v^2` and `vth_stella^2 = 2 cs^2`. `bmag, gradpar, gds2, gds21, gds22`
   transfer unchanged. **This is the single highest-risk step in rung 3** and it
   is now measured, not assumed — a rung-3 smoke test should reproduce the
   factor-2 check on the CBC Miller case before touching W7-X.

Note that `overwrite_gradpar` also overwrites `b_dot_grad_z(1,:)`, which is only
correct for `nalpha = 1`.

### 6.3 GKX side — the tracked W7-X config is NOT the target case

`benchmarks/runtime_w7x_zonal_response_vmec.toml` differs from the rung-3 spec
in three ways, all of which must be changed:

| | tracked GKX toml | rung-3 target |
|---|---|---|
| equilibrium | `../examples/vmec/wout_nfp3_QI_fixed_resolution_final.nc` — **a QI stellarator placeholder, not W7-X** (the file's own comment says to override it) | stella's `wout_w7x_standardConfig.nc` |
| surface | `torflux = 0.64` | `s = 0.49` |
| configuration | W7-X **high-mirror** (per its header, citing Gonzalez-Jerez et al. JPP 88, 905880310 (2022)) | W7-X **standard** — this is the file stella ships |

`wout_nfp3_QI_fixed_resolution_final.nc` is **not present** in
`examples/vmec/` (only `wout_NuhrenbergZille_1988_QHS.nc` is), so the tracked
config does not run as-is. The CLI accepts `--vmec-file`, so pointing GKX at
stella's W7-X wout needs no repo change.

Also note the tracked W7-X lane is documented as **open** in
`docs/manuscript_figures.rst`: residuals fail at `kx rho_i = 0.07, 0.10, 0.30`
and the late envelopes are much larger than the digitized stella/GENE traces,
with a recurrence / moment-closure hypothesis on record. Rung 3 should expect to
land in the middle of that open problem, and the `t_rec = 2 sqrt(Nm)/k_par`
budget from §2.2 is the first thing to compute for the W7-X `k_par`.

### 6.4 Concrete order of work for rung 3

1. **Smoke-test the bridge on the case already understood.** Export the CBC
   Miller geometry from GKX into an `input.geometry`, run stella with the
   overwrite on, and confirm it reproduces this rung's `R = 0.1050`. That
   validates the ×2 drift rule and the column order before any stellarator
   ambiguity is added.
2. Build the stella W7-X input (`torflux = 0.49`) and record its `.geometry`.
3. Run GKX with `--vmec-file` pointing at stella's `wout_w7x_standardConfig.nc`,
   at `s = 0.49`, and cross-check its `Geometry` group against stella's
   `.geometry` — expecting ≤1.2% on the metrics and **exactly 2×** on the drifts.
   Any departure from *exactly* 2 on the drifts is a real bug, not a tolerance.
4. Only then compare linear rates, remembering
   `ky_stella = sqrt(2) ky_gkx` and `rates_stella = rates_gkx / sqrt(2)`.
5. Carry the `kx`/`ky` `sqrt(2)` explicitly in the config, not in the analysis —
   §5's trap is exactly what happens when it lives only in a comment.
6. **Budget `Nm` from `t_rec = 2 sqrt(Nm)/k_par` before running, not after.**
   W7-X at `s = 0.49` has `q ~ 1.11` and a much longer field line than this CBC
   case, so its `k_par` — and therefore the `Nm` needed for a given horizon —
   must be computed from the actual `gradpar` (and see the `nfield_periods`
   warning in §6.1, which changes `gradpar` by a factor `nfp/nfield_periods`).
   If a zonal residual is wanted there, plan an `Nm` scan from the start: §3.4
   shows a single-resolution number is ~11% off even in this much easier
   geometry, and the tracked W7-X lane's open residual mismatches at
   `kx rho_i = 0.07, 0.10, 0.30` are exactly where such an error would hide.

---

## 7. Artifacts

Work dir
`/private/tmp/claude-501/-Users-rogeriojorge-local/1e858e4f-6438-4dbd-8d3c-60502cd814ab/scratchpad/rh_rung2/`

- `RUNS.md` — provenance and exact commands for every run
- `final_table.py` — regenerates §3.3, §3.6 and the `sqrt(2)` check from the bundles
- `hermite_extrap.py` — regenerates §3.4 (the `1/sqrt(Nm)` residual extrapolation)
- `protocol_calibration.py` — regenerates §3.5 (the stella-calibrated fit bias)
- `fit_rh.py`, `collect.py` — the shared fit and the two trace loaders
- `rung2_traces.png` — stella and GKX on a common physical time axis
- `gkx/cbc_miller_zonal_rh.toml` — the matched GKX config (documented header)
- `gkx/res_Nl*_Nm*.toml`, `gkx/res_Nz64.toml`, `gkx/kxtrap_kx050.toml` — the scan and the `kx` control
- `gkx/*_out/` — GKX run bundles, figures and JSON
- `stella/`, `stella/long/`, `stella/upw/`, `stella/hires/`, `stella/pristine/` — stella runs
