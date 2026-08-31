# Merlo Case-III zonal artifact: resolution audit (plan item 2.8)

**Date:** 2026-08-19
**Target:** `docs/_static/miller_zonal_response_pilot.json` — a currently-PASSING row of the
GKX validation gate index.
**Question:** does that row pass only because it is under-resolved?
**Repo:** `/Users/rogeriojorge/local/GKX` @ `7cf5e6d1` (read-only; nothing was written there).
**Scratch:** `/private/tmp/claude-501/-Users-rogeriojorge-local/1e858e4f-6438-4dbd-8d3c-60502cd814ab/scratchpad/merlo/`

---

## 0. Verdict summary

| Gated quantity | Ref | atol | Tracked (Nm=24) | Converged | \|err\| | Verdict |
|---|---|---|---|---|---|---|
| `residual_level` | 0.19 | 0.015 | 0.19245 (PASS) | 0.208 ± 0.006 | 0.018 | **FAIL** (1.2x atol) |
| `gam_frequency_R0_over_vi` | 2.24 | 0.10 | 2.20329 (PASS) | 2.38 ± 0.05 | 0.14 | **FAIL** (1.4x atol) |
| `gam_growth_rate_R0_over_vi` | -0.17 | 0.03 | -0.17551 (PASS) | -0.166 / -0.204 | 0.004 / 0.034 | **INCONCLUSIVE** |

**Two of the three gates pass only because the run is under-resolved.** `omega_GAM` passes
at Nm=24 and at no other resolution on the ladder. `residual_level` passes at Nm=24 and 48
(both recurrence-dominated) and fails from Nm=96 up. `gamma_GAM` survives resolution
refinement but its converged value is set by the diagnostic output cadence, not by physics,
so its PASS carries no evidential weight.

**The mechanism is Hermite recurrence, not smooth truncation error.** At Nm=24 the residual
window [42,60] sits entirely past the onset of velocity-space recurrence, and the scatter
inside the window (0.231) exceeds the number being gated (0.193).

**kx trap: NOT CONFIRMED — falsified by the source paper's own text.** Merlo et al. define
`rho_i = v_i/Omega_i` with `v_i = sqrt(T_i/m_i)`, the same one-T convention GKX uses.
`kx = 0.05` with `Lx = 125.6` is the correct physical wavenumber. See Sec. 6.

---

## 1. Protocol as shipped

`benchmarks/runtime_miller_zonal_response.toml`

* Miller geometry, Merlo Table III Case III: `q=1.389`, `s_hat=0.751`, `epsilon=0.18`,
  `R0=R_geo=2.77778`, `shift=-0.1569`, `akappa=1.4723`, `akappri=-0.0728`,
  `tri=-0.0070`, `tripri=-0.0140`, `betaprim=0.0`.
* Grid: `Nx=6`, `Ny=8`, `Nz=32`, `Lx=125.6`, `Ly=62.8`; `[run] Nl=4`, `Nm=24`.
* Physics: electrostatic, adiabatic electrons, zero gradients, all collision /
  hypercollision / hyperdiffusion weights zero, `nonlinear=0`.
* Drive: `init_field="density"`, `init_amp=1e-6`, `init_single=true`.
* Time: `dt=0.005`, `steps=12000` (t_max = 60 in code units), `sample_stride=10`
  (1201 samples, Δt_sample = 0.05).
* `[run] kx = 0.05`, `ky = 0.0`.

`tools/artifacts/build_zonal_flow_artifacts.py miller-panel` then:

1. runs the case through `run_runtime_nonlinear_with_artifacts`;
2. picks the kx grid point nearest 0.05. With `Lx=125.6` the retained (post-dealias)
   kx grid is `[-0.050025, 0, +0.050025]`, so `kx_index=2`, `kx_selected=0.0500254`;
3. loads `Diagnostics/Phi_zonal_mode_kxt`, real part, phase-aligned;
4. calls `zonal_flow_response_metrics` with `tail_fraction=0.3`,
   `initial_policy="first_abs"`, `peak_fit_max_peaks=4`,
   `damping_fit_mode="branchwise_extrema"`, `frequency_fit_mode="hilbert_phase"`,
   `fit_window_tmax=30.0`, `hilbert_trim_fraction=0.2`.

Extraction definitions (`src/gkx/diagnostics/zonal_validation.py`):

* `initial_level = |phi(0)|` (first sample, RH convention);
* `residual_level = mean(phi over the last 30% of the trace) / |phi(0)|`, i.e. the
  **signed arithmetic mean over t in [42.05, 60.00]** code units;
* `residual_std = std(phi) / |phi(0)|` over the same window;
* `gamma_GAM`: log-linear fit through up to 4 positive and 4 negative extrema of the
  residual-subtracted trace inside t <= 30;
* `omega_GAM`: Hilbert instantaneous phase over the same t <= 30 window, 20% trimmed;
* both multiplied by `R0 = 2.77778` to report in `R0/v_i`.

### Units of the time axis

GKX normalizes lengths to `L_ref = a = 1` (`epsilon = r/R0 = 0.18`, `r = 0.5`,
`r/a = 0.5`) and time to `a/v_i`. So the shipped run reaches

* **t_max = 60 a/v_i = 21.6 R0/v_i**, and
* the residual window is **[15.1, 21.6] R0/v_i**.

Merlo et al. state the residual "is computed after the GAM oscillation is completely
damped. Simulations are run well beyond this limit, **typically up to 150 R0/v_i**, to
ensure a true stationary state **and check that the recurrence problem is not affecting
the results**." The shipped run is 7x shorter than that.

That alone is not fatal: with `gamma R0/v_i = -0.17` the physical GAM envelope has decayed
to `exp(-0.17 * 15.1) = 0.077` of its initial amplitude by the start of the residual
window and to `0.025` by the end. The GAM *is* damped. The problem is what refills the
window.

---

## 2. Reproduction of the tracked artifact

Re-ran the shipped config unchanged (outputs redirected to scratch):

| quantity | tracked JSON | this run | Δ |
|---|---|---|---|
| `residual_level` | 0.192452 | 0.193287 | +0.43% |
| `gam_frequency_R0_over_vi` | 2.203287 | 2.204211 | +0.04% |
| `gam_growth_rate_R0_over_vi` | -0.175513 | -0.174475 | -0.59% |
| `residual_std` | 0.231714 | 0.230968 | — |
| `kx_selected` | 0.0500254 | 0.0500254 | 0 |
| trace tmax | 59.9116 | 60.0017 | +2 samples |

**The artifact reproduces.** The residual difference is fully explained by the tracked
trace ending ~2 samples earlier than the current one (a time-axis bookkeeping change);
two samples out of ~360 in the averaging window move the gated number by 0.4%, which is
itself the first sign of how unstable that mean is.

---

## 3. What actually happens in the residual window

The three Nm runs are **pointwise identical** until the Hermite resolution runs out:
Nm=24/48/96 agree to <1e-3 up to t≈22, and Nm=48/96 agree up to t≈30. After that they
diverge, and the divergence is Hermite recurrence (velocity-space aliasing), not physics.

Peak-to-peak swing of the normalized trace **inside the gate window** [42, 60] (= [15.1,
21.6] R0/v_i), compared with the physically expected damped-GAM remnant there (~0.03):

| Nm | min | max | peak-to-peak | `residual_std` | `residual_std / residual` | quietest-window centre |
|---|---|---|---|---|---|---|
| 24 | -0.180 | +0.600 | 0.780 | 0.2310 | **1.19** | t ≈ 26.0 |
| 48 | -0.020 | +0.430 | 0.449 | 0.1202 | 0.61 | t ≈ 38.5 |
| 96 | +0.153 | +0.263 | 0.109 | 0.0307 | 0.15 | t ≈ 55.5 |

Two things follow.

1. **At the tracked resolution the scatter inside the averaging window is 119% of the
   number being gated.** The "residual" is not a plateau level; it is the arithmetic mean
   of an oscillation ~25x larger than the physical GAM remnant that should be there.

2. The quietest point of the trace — the gap between the damped primary GAM and the
   regrowing recurrence — moves as **t_quiet ≈ 5.5 sqrt(Nm)** (26.0 / 38.5 / 55.5 for
   Nm = 24 / 48 / 96; ratios 1.48 and 1.44 vs sqrt(2) = 1.414). This is the textbook
   Hermite recurrence scaling. Requiring the whole gate window (t <= 60) to sit before
   recurrence gives

   > **Nm >~ 120** for the shipped t_max = 60, dt = 0.005 protocol.

   The tracked run has t_quiet ≈ 26, so **the entire residual window sits 1.6x to 2.3x
   past the onset of recurrence**. Merlo et al. explicitly instruct the opposite.

The docs already record this trade-off. `docs/testing.rst` says raising the resolution to
Nm=28 lowers the recurrence ratio and moves omega_GAM onto the read-off but pushes gamma
to about -0.192, and concludes that the shipped artifact therefore stays on the Nm=24,
Nl=4 baseline. `docs/manuscript_figures.rst` says the same: a higher-moment audit lowers
the recurrence ratio but over-damps the GAM, "so the frozen Merlo artifact remains on the
current Nm=24 baseline". **The project already knew the answer moves with resolution and
pinned the artifact at the resolution where it agrees.**

---

## 4. The Hermite resolution ladder

`Nl = 4` and `Nz = 32` held fixed at the tracked values throughout (both verified
converged, Sec. 5d). Everything else is the shipped config; only `Nm` changes.

### 4a. dt = 0.005 (the tracked timestep)

| Nm | `residual_level` | `residual_std` | std/residual | `omega R0/vi` | `gamma R0/vi` |
|---|---|---|---|---|---|
| 24 (tracked) | 0.19329 | 0.2310 | 1.19 | 2.20421 | -0.17448 |
| 48 | 0.19555 | 0.1202 | 0.61 | 2.28286 | -0.17258 |
| 96 | 0.20441 | 0.0307 | 0.15 | 2.33429 | -0.16406 |
| 144 | **run aborts** | — | — | — | — |

`Nm = 144` at `dt = 0.005` dies with non-finite diagnostics in `Wg_t` at t = 46.46 —
the Hermite streaming CFL scales as sqrt(Nm) and the shipped timestep is not stable past
Nm ≈ 96. **The tracked configuration cannot be resolution-refined without also refining
dt**, which is itself worth recording. The ladder was therefore rerun at `dt = 0.0025`.

### 4b. dt = 0.0025 (the ladder that actually converges)

| Nm | `residual_level` | `residual_std` | std/residual | `omega R0/vi` | `gamma R0/vi` | t_quiet |
|---|---|---|---|---|---|---|
| 24 | 0.19318 | 0.2312 | 1.20 | 2.20318 | -0.26452 | 26.0 |
| 48 | 0.19554 | 0.1202 | 0.61 | 2.28602 | -0.21958 | 38.8 |
| 96 | 0.20448 | 0.0307 | 0.15 | 2.33983 | -0.20157 | 55.5 |
| 144 | 0.20590 | 0.0295 | 0.14 | 2.34507 | -0.20280 | >60 |
| 192 | 0.20820 | 0.0239 | 0.11 | 2.35341 | -0.20618 | >60 |

(`t_quiet` for Nm = 144 and 192 is reported as ">60" because the trace never re-grows
inside the run — which is exactly the condition the gate window needs.)

Residual and omega agree between the two ladders to <0.3% at every common Nm, so they are
timestep-converged; `gamma` differs between the ladders for a reason that turns out not to
be dt at all (Sec. 5).

### 4c. Extrapolation

Fitting `y = a + b * Nm^(-p)` on the `dt = 0.0025` ladder:

| quantity | p = 1/2 | R^2 | p = 1 | R^2 | direct Nm=192 |
|---|---|---|---|---|---|
| `residual_level` | **0.2159** | 0.922 | **0.2084** | 0.854 | 0.2082 |
| `omega R0/vi` | **2.4457** | 0.979 | **2.3766** | 0.996 | 2.3534 |
| `gamma R0/vi` (Δt_s=0.025) | -0.1630 | 0.884 | -0.1898 | 0.950 | -0.2062 |

**Fit quality is poor for the residual and the extrapolation is not trustworthy** — nothing
like the 0.15% fit the prior Rosenbluth-Hinton study reported. That is expected here: the
Nm = 24 and 48 points are not on a convergence curve at all, they are recurrence-dominated
window means whose value depends on the accidental phase of a spurious oscillation. Fitting
through them mixes two unrelated behaviours. Restricting the fit to the clean branch
(Nm >= 96) gives residual 0.2112 (p=1) to 0.2164 (p=1/2), consistent with the direct value.

**Therefore the primary estimate here is the direct high-Nm value, not the extrapolation.**
Taking the highest resolution, its drift toward Nm -> infinity, and the analysis-window
systematic (Sec. 5b) together:

* `residual_level` = **0.208 ± 0.006**
* `omega_GAM R0/vi` = **2.38 ± 0.05**
* `gamma_GAM R0/vi` = **-0.166 ± 0.004** at the shipped diagnostic cadence, or
  **-0.204 ± 0.004** at twice that cadence (see Sec. 5c — this is the whole problem)

---

## 5. Independent problem: the estimators are not stable at the gate tolerance

While setting up the dt control I found that two of the three gated numbers move by more
than their own gate tolerance under changes that are *not physics*.

### 5a. gamma_GAM depends on the diagnostic output cadence

Halving `dt` at fixed `Nm=24` changes the trace itself by at most 4.2e-3 (relative to
|phi(0)|) — the residual moves 0.06% and omega moves 0.05%. But `gamma` moves from
**-0.17448 to -0.26452 (+52%)**, flipping that gate from PASS (|err| = 0.0055) to
FAIL (|err| = 0.0945 = 3.1x atol).

The cause is the diagnostic sample spacing, not dt. Subsampling the fine-dt trace back to
the coarse cadence restores the tracked answer:

| trace | Δt_sample | residual | omega | gamma |
|---|---|---|---|---|
| dt=0.005, native | 0.050 | 0.19329 | 2.20421 | -0.17448 |
| dt=0.0025, native | 0.025 | 0.19318 | 2.20318 | **-0.26452** |
| dt=0.0025, subsampled x2 | 0.050 | 0.19302 | 2.20214 | -0.17476 |
| dt=0.005, interpolated to 0.025 | 0.025 | 0.19315 | 2.20325 | -0.17460 |

The mechanism is a single extremum. The branchwise fit takes the first four negative
extrema below t=30; on the finely sampled trace the fourth one is a genuine but tiny
wiggle at t=24.98 with value -0.0092, while at coarse cadence the fourth is at t=25.98
with value -0.0407. Since the fit is log-linear in |value|, that one point changes the
slope by 52%.

`gamma_GAM` is therefore not resolved to the 0.03 tolerance it is gated at — it is
resolved to roughly +/-0.09, set by whether a near-zero-crossing wiggle happens to be
sampled.

### 5b. omega_GAM and residual_level depend on the analysis window

Post-processing sweeps on the *same* tracked trace (Nm=24, dt=0.005), varying only the
tool's own default knobs:

| knob | values swept | `residual` range | `omega` range | `gamma` range |
|---|---|---|---|---|
| `tail_fraction` 0.20 -> 0.40 | 5 | **0.1666 – 0.2122** | — | — |
| `fit_window_tmax` 22 -> 35 | 6 | — | **2.204 – 2.529** | **-0.147 – -0.217** |
| `peak_fit_max_peaks` 3 -> 6 | 4 | — | — | -0.147 – -0.174 |

Compare against the gate tolerances: residual +/-0.015, omega +/-0.10, gamma +/-0.03.

* residual spread from the window choice alone is 0.046 = **3.0x atol**;
* omega spread from the fit-window choice alone is 0.325 = **3.3x atol**;
* gamma spread from the fit-window choice alone is 0.070 = **2.3x atol**.

And in every case the shipped default lands at or next to the best-agreeing value:
`fit_window_tmax=30.0` gives omega = 2.204, the *closest* point in the entire sweep to
the 2.24 reference, and every other window in the sweep is further away. At Nm=96 the same
sweep gives 2.31–2.53.

This is a stronger statement than the resolution finding: even at fixed resolution, the
three gated numbers are not determined to their own tolerance by the physics. The gate is
measuring the analysis defaults as much as the solver.


### 5c. The cadence effect is not an Nm=24 curiosity

Repeating the subsampling test at converged resolution isolates it completely. Taking the
`dt = 0.0025` traces and decimating them to the shipped diagnostic cadence:

| run | Δt_sample | residual | omega | gamma |
|---|---|---|---|---|
| Nm=96, dt=0.0025 | 0.025 | 0.20448 | 2.33983 | **-0.20157** |
| Nm=96, dt=0.0025, decimated | 0.050 | 0.20446 | 2.34137 | **-0.16405** |
| Nm=144, dt=0.0025 | 0.025 | 0.20590 | 2.34507 | **-0.20280** |
| Nm=144, dt=0.0025, decimated | 0.050 | 0.20585 | 2.34648 | **-0.16377** |

`residual_level` and `omega_GAM` are cadence-independent to 5 significant figures.
`gamma_GAM` is a function of the cadence and essentially nothing else:

* `sample_stride` giving Δt = 0.05 -> gamma ≈ **-0.164** -> |err| = 0.006 -> **PASS**
* `sample_stride` giving Δt = 0.025 -> gamma ≈ **-0.202** -> |err| = 0.032 -> **FAIL**

The gated damping rate flips its own verdict on the value of `sample_stride`, at converged
Nm and converged dt. It is not a measurement of the physics.

### 5d. Nl and dt controls

* **Nl is converged**, confirming the prior study: `Nm=96, dt=0.0025`, `Nl=4` vs `Nl=8`
  moves residual by -0.04%, omega by -0.15%, gamma by -1.6%. (It does cut the tail scatter
  `residual_std` by 44%, 0.0307 -> 0.0173, so some of the late-time ripple is Laguerre
  recurrence — but it does not move the gated values.)
* **dt is converged for residual and omega**, not an issue: halving dt moves residual by
  <0.1% and omega by <0.3% at every Nm tested.
* **dt=0.005 is not stable above Nm≈96.** `Nm=144` at the shipped `dt=0.005` aborts with
  non-finite diagnostics at t=46.5 (Hermite streaming CFL scales as sqrt(Nm)). The whole
  high-Nm ladder therefore had to be rerun at `dt=0.0025`, which is why both ladders are
  reported.
---

## 6. The kx trap: checked, and NOT confirmed

The suspicion was that `kx = 0.05` with `Lx = 125.6` is sqrt(2) too large physically,
because the reference might quote `kx rho_i` in the GS2/stella convention
`rho_i = sqrt(2T/m)/Omega` while GKX uses `rho_i = sqrt(T/m)/Omega`.

**GKX side (confirmed).** `src/gkx/operators/linear/params.py:265` sets
`rho = sqrt(temperature * mass) / |charge|`, and
`src/gkx/operators/linear/cache_builder.py:526` forms the FLR argument as
`b = rho^2 * kperp2` (no factor 1/2); the Laguerre gyroaverage uses
`alpha^2 = 2 * x * b`, whose Maxwellian average is `exp(-b) I_0(b)`. So GKX's `rho_i` is
`v_i / Omega_i` with `v_i = sqrt(T_i/m_i)` — the GENE-family "one-T" convention, not the
GS2 one. A GS2-convention `k rho = 0.05` would correspond to GKX `k rho = 0.0354`.

**Reference side (decisive).** The reference is Merlo et al., *Linear multispecies
gyrokinetic flux tube benchmarks in shaped tokamak plasmas*, Phys. Plasmas 23, 032104
(2016) — a GENE / GKW / GS2 cross-benchmark. I pulled the accepted manuscript
(UKAEA preprint CCFE-PR(15)88,
https://scientific-publications.ukaea.uk/wp-content/uploads/Preprints/CCFE-PR1588.pdf)
and extracted the text. Two statements settle it:

* Sec. IV: `v_j = sqrt(T_j/m_j)` is defined as the thermal velocity of species j, and the
  paper notes the three codes use different internal normalizations (Appendices A3, B2,
  C2) — i.e. the *paper* reports in one common convention, and that convention is the
  one-T one.
* Sec. IV, on plotting linear spectra: wavenumbers are normalized to the ion Larmor
  radius **rho_i = v_i / Omega_i**, with the same `v_i`. Frequencies and growth rates are
  reported in units of `R0 / v_i`, again with the same `v_i`.
* Sec. V (the RH/GAM section) states the simulations evolve an ion density perturbation
  associated with the mode **kx rho_i = 0.05, ky = 0**, with no hyperdiffusion and zero
  density/temperature gradients.

So the paper's `rho_i` and GKX's `rho_i` are the *same* quantity. `kx = 0.05` in the TOML
is the correct physical wavenumber, `Lx = 125.6` gives `2*pi/Lx = 0.0500254` which is what
the tool selects, and there is **no sqrt(2) error in the Merlo case**.

Two corroborating checks:

* the reference `omega_GAM R0/v_i = 2.24` is only consistent with the one-T convention.
  The circular large-aspect-ratio estimate `omega_GAM = sqrt(7/4 + tau) * sqrt(2T/m) / R`
  is 2.35 in `R0 / sqrt(T/m)` units and 1.66 in `R0 / sqrt(2T/m)` units; the paper's
  Fig. 16(b) axis runs 2.2–2.7 across the five cases. A GS2-unit reading would put those
  numbers near 1.6.
* the paper's Fig. 13 time axis is labelled `time v_i / R0` and its Fig. 14/16 ordinates
  are `omega_GAM R0/v_i` and `gamma_GAM R0/v_i`, matching the artifact's field names.

**Caveat on where the suspicion probably belongs.** The tracked artifact's `references`
list also names a *W7-X stella/GENE benchmark* for "zonal-flow observable conventions",
and the sibling W7-X lane in the same tool
(`tools/artifacts/build_w7x_zonal_validation_artifacts.py`, `kx_rhoi_values = [0.05, 0.07,
0.10, 0.30]`) is transcribed from stella-family work — and that lane's residuals are
documented as *failing* at kx = 0.07, 0.10 and 0.30. If a sqrt(2) convention error exists
anywhere in this family it is far more likely to be there than in the Merlo case. That is
a separate check and was not performed here.

### Reference read-off accuracy (bonus)

While in the paper: Case III values in the artifact are figure read-offs from Figs. 12/14/16.
From Fig. 16 the Case III points sit at roughly residual 0.185–0.19,
`omega_GAM R0/v_i` ~ 2.25–2.30 and `gamma_GAM R0/v_i` ~ -0.17. The artifact's 0.19 / 2.24 /
-0.17 are defensible read-offs, though `omega` may be read ~0.03 low. Note also that
Fig. 15 is normalized to `Rgeom(r)/v_i`, not `R0/v_i`, so it must not be mixed with Fig. 16.

### Geometry transcription (bonus, unverified impact)

Merlo Table III Case III is
`q_s=1.389, s=0.751, kappa=1.4723, delta=-0.0070, zeta=2.83e-3, Delta=-0.0139,
alpha_MHD=0.5425, dRgeom/dr=-0.1569, dkappa/dr=-0.0728, ddelta/dr=-0.0140, dzeta/dr=0.003`.

The TOML transcribes q, s_hat, kappa, delta, dRgeom/dr (as `shift`), dkappa/dr and
ddelta/dr correctly, and its header comment mislabels `Delta` as "D". Two Table III
entries are dropped:

* squareness `zeta = 2.83e-3` and `dzeta/dr = 0.003` — GKX's Miller model has no
  squareness; both are tiny, so this is almost certainly negligible.
* **`alpha_MHD = 0.5425` is dropped**: the TOML sets `betaprim = 0.0`. In GKX,
  `analytic.py:128` maps `dpdrho = 0.5 * betaprim`, and `alpha_MHD = -q^2 R0 dbeta/dr`
  implies `betaprim ≈ -0.101` for this surface. Setting it to zero removes the
  equilibrium pressure-gradient contribution to the Miller local shear and drifts. The
  RH test zeroes the *kinetic* gradients, but `alpha_MHD` is an equilibrium property of
  the CHEASE surface and should survive that. **Not measured here** — flagged as a
  separate fidelity item.

---

## 7. Verdicts

### `residual_level` — **FAIL**

| | value | \|err\| vs 0.19 | atol 0.015 |
|---|---|---|---|
| tracked, Nm=24 | 0.19245 | 0.0025 | PASS (0.16x atol) |
| converged, Nm=192 | 0.20820 | 0.0182 | **FAIL (1.21x atol)** |
| converged estimate | 0.208 ± 0.006 | 0.018 ± 0.006 | **FAIL** |
| 1/sqrt(Nm) extrapolation | 0.2159 | 0.0259 | **FAIL (1.7x atol)** |
| 1/Nm extrapolation | 0.2084 | 0.0184 | **FAIL (1.2x atol)** |

Every converged estimate fails. The residual is monotonically **increasing** with Hermite
resolution over the clean branch (0.2045 -> 0.2059 -> 0.2082 at Nm = 96/144/192) and had not
fully flattened at Nm = 192, so the true value is >= 0.208 — i.e. the gap grows, not shrinks,
with further refinement. The failure margin is not large (1.2x atol at Nm=192) and the low
end of the uncertainty band grazes the tolerance, so this is a **marginal but consistent
FAIL**, not a blowout.

Note the direction is *opposite* to the prior study's suspicion: the residual rises with
Nm here rather than falling, so the "Nm=24 reads ~11% high, converged ~0.173" scenario is
not what happens. The tracked value is ~7% *low* relative to converged, and it agrees with
the paper only because a recurrence-dominated window mean happened to land near 0.19.

### `gam_frequency_R0_over_vi` — **FAIL**

| | value | \|err\| vs 2.24 | atol 0.10 |
|---|---|---|---|
| tracked, Nm=24 | 2.20329 | 0.0367 | PASS (0.37x atol) |
| converged, Nm=192 | 2.35341 | 0.1134 | **FAIL (1.13x atol)** |
| converged estimate | 2.38 ± 0.05 | 0.14 ± 0.05 | **FAIL** |
| 1/Nm extrapolation (R^2=0.996) | 2.3766 | 0.1366 | **FAIL (1.4x atol)** |
| 1/sqrt(Nm) extrapolation | 2.4457 | 0.2057 | **FAIL (2.1x atol)** |

This is the cleanest and most damning of the three: omega rises monotonically and smoothly
with Nm (2.203 -> 2.286 -> 2.340 -> 2.345 -> 2.353) with an excellent 1/Nm fit, and it walks
straight out of the gate band. **Nm = 24 is the only resolution on the ladder at which this
gate passes.** It is under-resolution that produces the agreement.

### `gam_growth_rate_R0_over_vi` — **INCONCLUSIVE**

Two clean, separately-converging ladders, differing only in the diagnostic output cadence:

| Nm | Δt_sample = 0.050 (shipped) | Δt_sample = 0.025 |
|---|---|---|
| 24 | -0.17476 | -0.26452 |
| 48 | -0.17271 | -0.21958 |
| 96 | -0.16405 | -0.20157 |
| 144 | -0.16377 | -0.20280 |
| 192 | -0.16826 | -0.20618 |
| converged | **-0.166 -> \|err\| 0.004 -> PASS** | **-0.204 -> \|err\| 0.034 -> FAIL** |

This gate *does* survive resolution refinement at the shipped `sample_stride` — that is an
honest result and it should be recorded as such. But it survives for the wrong reason: the
converged answer is a function of how often the diagnostic is written out, and doubling
that rate (a change with no physics content whatsoever) moves it by 23% and flips the
verdict. The quantity is not determined to its own 0.03 tolerance, so the PASS carries no
evidential weight either way. Verdict: **INCONCLUSIVE**.

---

## 8. Answer to the question that was asked

**Does this row pass only because of under-resolution? Yes for two of the three gates.**

* `residual_level` and `gam_frequency_R0_over_vi` both pass at Nm = 24 and both fail at
  every converged resolution. `Nm = 24` is the only point on the ladder where the frequency
  gate passes at all.
* `gam_growth_rate` passes at converged resolution too, but only at the shipped output
  cadence; it is not resolved to its tolerance.
* The mechanism is Hermite recurrence, not a smooth truncation error. At `Nm = 24` the
  quiet point of the trace is t ≈ 26 while the residual window is [42, 60], so the entire
  averaging window is 1.6x–2.3x past recurrence onset and the scatter inside it (0.231) is
  larger than the number being gated (0.193). Merlo et al. explicitly require the opposite:
  run to ~150 R0/v_i and verify recurrence is not affecting the result. The shipped run
  reaches 21.6 R0/v_i and does not verify this.
* Making the gate window legitimate needs **Nm >~ 120** at t_max = 60 (from
  t_quiet ≈ 5.5 sqrt(Nm)), and `Nm >= 144` additionally needs `dt <= 0.0025` for stability.

**The sqrt(2) kx trap is NOT present in this case** — the Merlo paper defines
`rho_i = v_i/Omega_i` with `v_i = sqrt(T_i/m_i)`, identical to GKX's convention, so
`kx = 0.05` / `Lx = 125.6` is correct. That suspicion should be redirected to the W7-X
zonal lane, which is transcribed from stella-family work and is already failing.

## 9. Suggested follow-ups (not performed)

1. Re-gate the Merlo row at `Nm >= 144`, `dt <= 0.0025`, and either widen the tolerances to
   what the protocol can actually resolve or fix the protocol.
2. Make `gamma_GAM` cadence-independent: fit a parametric damped sinusoid to the whole
   pre-recurrence window instead of a log-linear fit through 4 hand-picked extrema, or at
   minimum interpolate extrema sub-sample and reject near-zero-crossing candidates.
3. Pin `fit_window_tmax` to a physically defined point (e.g. a fixed number of GAM periods,
   or t_quiet) rather than the hard-coded 30.0 that happens to optimize agreement at Nm=24.
4. Audit the W7-X zonal lane for the sqrt(2) rho_i convention against its stella source.
5. Decide whether `alpha_MHD = 0.5425` (Merlo Table III) should be carried into
   `betaprim` instead of the current 0.0, and measure the effect.
6. The paper's 150 R0/v_i recurrence check is unreachable for a collisionless Hermite
   closure at any feasible Nm (it would need Nm ~ 5000+). Either state that limitation
   explicitly next to the artifact, or introduce a documented closure and show the residual
   is insensitive to it.

---

## Appendix: reproduction

```
export PYTHONPATH=/Users/rogeriojorge/local/GKX/src JAX_ENABLE_X64=1
python3 /Users/rogeriojorge/local/GKX/tools/artifacts/build_zonal_flow_artifacts.py \
    miller-panel --config <scratch>/cfg/merlo_Nm<N>[_dthalf].toml \
    --out-bundle <scratch>/out/merlo_Nm<N>.out.nc \
    --out-png <scratch>/out/merlo_Nm<N>.png
```

Configs are the shipped TOML with only `[run] Nm` (and, for the `_dthalf` variants,
`dt: 0.005 -> 0.0025` and `steps: 12000 -> 24000`) changed. Analysis:
`<scratch>/analyze.py`, `<scratch>/plateau.py`.
Runtimes on this machine (macOS arm64, CPU): Nm=24 ~71 s, 48 ~141 s, 96 ~277 s at
dt=0.005; roughly 2x that at dt=0.0025 (Nm=192 took ~977 s).

**The GKX repo was not modified.** All outputs went to the scratch directory.


---

# Follow-through: items 2.8b / 2.9 / 2.10 (2026-08-19, agent/merlo)

The audit above is the diagnosis. This section is what was changed, measured, and
shipped, with the owner's sign-off to raise the baselines to converged values.

## 0. A fourth defect, found while re-running: the benchmark truncates silently

Regenerating the *tracked* artifact from its own TOML on current `main` produces a
trace that stops at **t = 7.66** instead of 60, with `residual = 0.293` and
`gamma = NaN`. `[time] run_to` defaults to `"saturation"`; for a zero-gradient
relaxation run the heat flux is identically ~0, the saturation stop declares the
flux window converged inside the first chunk, and the loop breaks. Nothing raises.
The residual is then read off "the last 30% of the trace" -- of a trace one sixth
of a GAM period long.

The audit's own reproduction predates this behaviour, which is why it did not see it.
Adding `run_to = "t_max"` to `benchmarks/runtime_miller_zonal_response.toml` restores
the audit numbers exactly: `residual 0.193287`, `omega 2.204211`,
`gamma -0.174475` at Nm=24, dt=0.005, against the audit's 0.19329 / 2.20421 /
-0.17448. **The other two runtime TOMLs (`runtime_w7x_zonal_response_vmec.toml`,
`runtime_secondary_slab.toml`) also do not set `run_to` and were not checked.**

## 1. Item 2.9 -- gamma_GAM now has an estimator that one sample cannot move

New damping mode `period_rms_envelope` in `src/gkx/diagnostics/zonal_validation.py`.
For `y(t) = C(t) + A exp(-gamma t) cos(omega t + phi)` with `C` slowly varying, the
RMS of `y` about its own running one-period mean is `A exp(-gamma t)` times a
`t`-independent factor. So: sliding one-period mean, sliding one-period RMS of the
deviation, then a log-linear fit weighted by `envelope^2` (the inverse-variance
weight for a log fit) over a window stated in **GAM periods**, not samples and not a
hard-coded absolute time. Both convolutions are `mode="valid"`, so no output sample
ever sees zero padding.

Cadence independence, measured three ways on the converged Nm=144 trace:

| test | period_rms_envelope | branchwise_extrema (retired) |
|---|---|---|
| 3 independent runs, only `sample_stride` changed (5/10/20) | spread **0.00070** | 0.00293 |
| same trace, decimation x1/x2/x4/x8 + sampling-phase offsets | spread **0.00030** | **0.03790** |
| `fit_window_tmax` 22 -> 35 (moves omega by 0.24) | spread **0.0015** | 0.018 |
| `damping_fit_periods` 2 -> 4 | spread 0.0026 | n/a |
| `damping_fit_start_periods` 0.5 -> 1.5 | spread 0.0045 | n/a |
| `tail_fraction` 0.20 -> 0.40 | spread 0.0004 | 0.0065 |

Against the gate tolerance 0.03 the decimation spread is **1.0% vs 126%**.
On the Nm=24 trace pair that produced the audit's headline 52% swing
(-0.17448 at dt_sample=0.05 vs -0.26452 at 0.025), the new estimator gives
**-0.11924 vs -0.11937**, a 0.1% move.

Correctness, not only stability: on synthetic `0.2 + exp(-gamma t) cos(omega t)` the
estimator returns gamma to within 1-2% for gamma = 0.03 / 0.06 / 0.10 and
omega = 0.8 / 1.2, with a cadence spread below 1e-4 across a 16x span. The 1-2% low
bias is deterministic (one window of a *decaying* signal is not one window of a
stationary one) and is inside the quoted uncertainty. Pinned by three tests in
`tests/validation/physics_gates/test_validation_gates.py`.

With the fixed estimator, gamma is also **resolution- and timestep-converged**, which
the old one could not show: Nm = 96/144/192 give -0.1834 / -0.1841 / -0.1856, Nl=8
moves it by 1.0%, and dt=0.005 vs 0.0025 at Nm=96 moves it by 0.1%. The apparent dt
sensitivity in the audit was the cadence artifact all along -- halving dt doubled the
output cadence.

## 2. Item 2.10 -- alpha_MHD: FALSIFIED, and structurally so

`alpha_MHD = -q^2 R0 dbeta/dr` with Merlo Table III's 0.5425 gives
`betaprim = -0.5425 / (1.389^2 * 2.77778) = -0.1012272`. Rerun at Nm=144, dt=0.0025:
the trace is **bit-identical** to the baseline, max `|delta phi| = 0.0` over 2401
samples; residual, omega and gamma agree to every printed digit.

That is not a null result, it is a structural one. Diffing the two generated
`*.eiknc.nc` files: `betaprim` changes `gds2` (by 9.6%), `gds21` (8.1%),
`gbdrift` (32%), `cvdrift` (22%) and `aprime` (9.8%), and leaves `gds22`,
`gbdrift0`, `cvdrift0`, `bmag`, `gradpar`, `grho`, `jacob`, `drhodpsi` and `kxfac`
**exactly** unchanged. At `ky = 0`, `kperp^2 = gds2 ky^2 + 2 gds21 kx ky + gds22 kx^2`
collapses to `gds22 kx^2`, and the drift keeps only its `kx` partners. Every
coefficient alpha_MHD touches is multiplied by `ky`. **The dropped alpha_MHD cannot
explain a residual or frequency offset for a purely radial zonal mode**, so path (a)
is closed. (It would matter for any finite-ky Case-III comparison, and the TOML
should still carry it if this deck is ever reused at `ky != 0`.)

## 3. Item 2.8b -- the new baseline, and what the gate asserts

`benchmarks/runtime_miller_zonal_response.toml`: `Nm 24 -> 144`, `dt 0.005 -> 0.0025`,
`steps 12000 -> 24000`, plus `run_to = "t_max"`.

| quantity | GKX converged | uncertainty | Merlo read-off | gap | paper tolerance |
|---|---|---|---|---|---|
| `residual_level` | **0.2059** | +/- 0.006 | 0.19 | 0.0159 | 0.015 -> **FAIL 1.06x** |
| `omega_GAM R0/v_i` | **2.345** | +/- 0.05 | 2.24 | 0.1051 | 0.10 -> **FAIL 1.05x** |
| `gamma_GAM R0/v_i` | **-0.184** | +/- 0.010 | -0.17 | 0.0141 | 0.03 -> PASS |

Uncertainties are the Hermite drift over Nm = 96/144/192 (residual 0.2045/0.2059/0.2082,
omega 2.340/2.345/2.353, gamma -0.1834/-0.1841/-0.1856) combined with the estimator-knob
systematics measured above. Both failures grow monotonically with Nm, so refinement
widens them.

**Path taken: (b).** The tolerances were not touched. The artifact carries two reports:

* `gate_report` -- **asserted**. GKX's own converged residual / omega / gamma at the
  tolerances above, plus three conditions that make them measurements rather than
  window artifacts: `residual_scatter_ratio <= 0.25` (measures 0.143; the retired
  Nm=24 baseline sat at 1.20, i.e. the scatter exceeded the gated number),
  `analysis_window_past_recurrence == 0` against `t_quiet = 5.5 sqrt(Nm) = 66.0`, and
  `trace_completeness == 1` against the configured horizon.
* `literature_comparison` -- **reported, never asserted**. The same three observables
  against Merlo at the published tolerances, `passed: false`, with
  `paper_scale_gate_passed: false` at the top level. The failing metrics are named in
  the artifact and the gap is quantified in the artifact's own `notes`.

## 4. What is still open

1. The residual/omega gap itself. Candidates left, in order of plausibility:
   the Fig. 16 read-off (Sec. 6 above already notes omega may be read ~0.03 low, and
   0.03 is a third of the gap); the dropped squareness `zeta = 2.83e-3`,
   `dzeta/dr = 0.003`, which GKX's Miller cannot express; and the horizon -- the paper
   runs to 150 R0/v_i and GKX reaches 21.6 with recurrence-free windows only to ~24.
2. `run_to` in the other two runtime TOMLs (item 0 above).
3. The W7-X zonal lane's sqrt(2) convention question (item 2.11), untouched here.
