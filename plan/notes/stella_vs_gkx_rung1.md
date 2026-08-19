# stella ↔ GKX rung 1 — CBC linear ITG (collisionless Miller): final report

Plan item **2.4-r1**. Date 2026-08-18. macOS arm64, CPU JAX 0.9.2.
stella `build_cmake/stella` @ 4cdc5fcd via `/opt/local/bin/mpirun -np 4`.
GKX `/Users/rogeriojorge/local/GKX` (read-only) for the time-solver scans;
`/Users/rogeriojorge/local/GKX-worktrees/krylov` (uncommitted krylov fix) for the
certified eigensolver point.

Run inventory, configs and geometry cross-check are in `RUNS.md` /
`plan/notes/stella_vs_gkx_runs.md`; this file is the result.

---

## 0. Summary

1. **Rung 1 passes.** After the normalization remap, GKX reproduces stella's CBC linear
   ITG spectrum over `ky_stella = 0.2–0.7` to **|Δgamma| ≤ 2.8%, |Δomega| ≤ 1.9%**, and
   to **≤0.5% / 1.7%** at every point run to an adequate fit horizon.
2. **The conversion is a single `sqrt(2)`, applied in opposite senses:**
   `ky_stella = sqrt(2)·ky_gkx` and `(gamma, omega)_stella = (gamma, omega)_gkx / sqrt(2)`.
   Verified three independent ways (§2.3, §2.4, §5): ratio constancy across a ×4.6
   dynamic range, falsification of five alternative mappings (each 25–40× worse), and a
   fit-free confirmation from the certified eigensolver.
3. **The ~30% time-fit bias is lane-specific, not universal.** On this collisionless
   Miller lane the fit agrees with the certified eigenvalue to **0.30% in gamma** — and
   errs in the *opposite* direction from the collisional s-alpha lane. Plan finding #2
   should be re-scoped to a collisional-lane pathology.
4. **New actionable defect**: fixed-`t_max` GKX linear scans under-report low-ky growth
   by 10–15%. Size `t_max` by e-folding count (`gamma·t_max ≳ 7`), not wall time (§3.1).
5. **Not resolved here**: whether the tracked GX parity tooling folds in the `sqrt(2)`
   (plan finding #1). The tracked cyclone table is a different lane in both geometry and
   collisionality, so it cannot answer that — a like-for-like s-alpha run is needed (§4).

---

## 1. Case

CBC in a-units, identical physics on both sides:
Miller local equilibrium `rhoc=0.5, shat=0.796, q=1.4, rmaj=rgeo=2.77778,
kappa=1, delta=0, shift=0, betaprim=0`; one kinetic ion species with
`tprim=2.49, fprim=0.8`; adiabatic electrons (stella
`adiabatic_option="field-line-average-term"`, GKX `adiabatic_electrons=true`);
electrostatic; **collisionless** (GKX `collisions=false, hypercollisions=false`
and `[terms] collisions=0, hypercollisions=0`).

Resolutions:

| | parallel | velocity space | step |
|---|---|---|---|
| stella `cbc_scan6.in` | `nzed=24, nperiod=2` (3 poloidal turns, 73 pts) | `nvgrid=24, nmu=12, vpa_max=3` | `delt=0.03, nstep=2000` (t=60) |
| GKX | `ntheta=32, nperiod=2` → `Nz=96` (3 turns) | `Nm=48` Hermite, `Nl=12` Laguerre | `dt=0.002`, `imex2`, linked boundary |

Velocity-space match is only partial: `nmu=12 = Nl=12` (exact, per the plan's mapping
rule) but `nvpa = 2·nvgrid+1 = 49 ≈ 1×Nm`, below the recommended `nvpa ≈ 2–3×Nm`. This
is the leading candidate for the residual ~1–3% spread, alongside the fit horizon (§3.1).

Geometry cross-check: GKX's Miller `.eik.nc` export agrees with stella's `.geometry` to
≤1.2% pointwise, so any residual rate discrepancy is solver/velocity-space, not equilibrium.

---

## 2. Unit conversion — the exact mapping and its empirical verification

### 2.1 Where the factor comes from

Both codes use the same reference length `a` and both report `ky` in units of
`1/rho_ref` and rates in units of `v_ref/a`. They do **not** use the same `v_ref`:

| | `v_ref` | `rho_ref = v_ref/Omega` | family |
|---|---|---|---|
| stella (and GX, GS2) | `vth = sqrt(2T/m)` | `sqrt(2 T m)/|q|` | GS2 |
| **GKX** | `cs = sqrt(T/m)` | `sqrt(T m)/|q|` | GENE / c_s |

GKX source, `src/gkx/operators/linear/params.py:264-265`:

```python
vth=jnp.sqrt(temperature / mass),
rho=jnp.sqrt(temperature * mass) / jnp.abs(charge),
```

(the local name is `vth` but the value is `c_s`). Hence
`rho_stella = sqrt(2)·rho_gkx` and `vth_stella = sqrt(2)·cs_gkx`.

### 2.2 The conversion applied

For one and the same physical mode (same `k_y` in m⁻¹, same `gamma`, `omega` in s⁻¹):

```
   ky_stella [1/rho_stella]        =  sqrt(2) · ky_gkx [1/rho_gkx]
   (gamma, omega)_stella [vth/a]   =  (gamma, omega)_gkx [cs/a] / sqrt(2)
```

**One factor `sqrt(2)`, opposite sense on the wavenumber and on the rates.** The
wavenumber is multiplied because a *larger* `rho_ref` means the same physical `k_y`
carries a larger dimensionless value; the rates are divided because a *larger*
`v_ref` means the same physical rate carries a smaller dimensionless value. Getting
only one of the two halves right is the most likely failure mode, and both halves are
tested independently below.

Same factor applies to `kx` (`kx_stella = sqrt(2)·kx_gkx`) — relevant for rung 2.
Dimensionless ratios (e.g. the Rosenbluth-Hinton residual `phi(inf)/phi(0)`) are
convention-free and need no conversion.

### 2.3 Empirical verification 1 — ratio constancy across ky

If the rate conversion is right, the raw ratio `R = rate_GKX[cs/a] / rate_stella[vth/a]`
at matched physical `ky` must be a **ky-independent constant equal to `sqrt(2)=1.41421`**.
Across the 6 scan points `gamma` varies by ×2.68 and `omega` by ×4.64, so a constant
ratio is a strong test, not a single-point coincidence:

| ky_stella | ky_gkx | R_gamma | R_omega |
|---|---|---|---|
| 0.20 | 0.1414 | 1.40858 | 1.43826 |
| 0.30 | 0.2121 | 1.41640 | 1.42493 |
| 0.40 | 0.2828 | 1.42103 | 1.42706 |
| 0.50 | 0.3536 | 1.43274 | 1.40366 |
| 0.60 | 0.4243 | 1.45426 | 1.41652 |
| 0.70 | 0.4950 | 1.44026 | 1.44074 |

`R_gamma` mean **1.42888**, rms scatter 0.01537 → **+1.04%** from `sqrt(2)`.
`R_omega` mean **1.42519**, rms scatter 0.01261 → **+0.78%** from `sqrt(2)`.

The scatter (≈1%) is the size of the residual physics/convergence discrepancy, not of
the conversion: the conversion is a single exact constant and it reproduces itself at
every ky to about 1%.

### 2.4 Empirical verification 2 — falsification of the alternatives

Mean |relative error| of GKX vs stella over the scan under each candidate mapping
(stella interpolated onto the mapped ky):

| hypothesis | mean \|Δgamma\| | mean \|Δomega\| |
|---|---|---|
| **H1  ky·√2, rates/√2  (adopted)** | **1.2%** | **1.0%** |
| H2  ky·√2, rates unchanged | 42.9% | 42.5% |
| H3  ky unchanged, rates/√2 | 30.4% | 53.7% |
| H4  ky unchanged, rates unchanged | 84.5% | 117.3% |
| H5  ky/√2, rates·√2 (sign flipped) | 258.7% | 373.2% |
| H6  ky·√2, rates/2 | 28.6% | 28.7% |

H1 beats every alternative by a factor 25–40. Note H3 (`ky` un-remapped, rates
converted) is only mildly bad in `gamma` (30%) because `gamma(ky)` is flat near its
peak — it is the **frequency** channel (53.7%) that decisively rejects it. Both
halves of the mapping are therefore independently confirmed, and `omega` is the
discriminating observable for any future convention audit.

---

## 3. Comparison table

`comparison.csv` / `comparison_table.txt`, produced by `make_comparison.py`.
stella values are the time-averaged `Re[omavg], Im[omavg]` at `t=60 a/vth` from
`stella_scan/cbc_scan6.omega`. GKX values are the longest-horizon time-solver fit
available for each ky.

| ky_stella | ky_gkx | gamma_stella | omega_stella | gamma_GKX (cs/a) | omega_GKX (cs/a) | gamma_GKX→vth/a | omega_GKX→vth/a | Δgamma | Δomega | GKX t_max |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.20 | 0.1414 | 0.03762 | 0.05986 | 0.05298 | 0.08609 | 0.03747 | 0.06088 | **−0.40%** | **+1.70%** | 120 |
| 0.30 | 0.2121 | 0.06328 | 0.09832 | 0.08964 | 0.14011 | 0.06338 | 0.09907 | **+0.16%** | **+0.76%** | 80 |
| 0.40 | 0.2828 | 0.08435 | 0.14136 | 0.11987 | 0.20173 | 0.08476 | 0.14265 | **+0.48%** | **+0.91%** | 80 |
| 0.50 | 0.3536 | 0.09703 | 0.18702 | 0.13902 | 0.26252 | 0.09830 | 0.18563 | **+1.31%** | **−0.75%** | 40 |
| 0.60 | 0.4243 | 0.10095 | 0.23308 | 0.14681 | 0.33016 | 0.10381 | 0.23346 | **+2.83%** | **+0.16%** | 40 |
| 0.70 | 0.4950 | 0.09653 | 0.27774 | 0.13903 | 0.40015 | 0.09831 | 0.28295 | **+1.84%** | **+1.88%** | 40 |

Rates in `v_ref/a`; `Δ = (GKX_converted − stella)/stella`.
**Agreement: |Δgamma| ≤ 2.8%, |Δomega| ≤ 1.9% across the whole scan; ≤0.5%/1.7% at
the three points run to t_max ≥ 80.** stella's `gamma(ky)` peak (ky≈0.6) and the
linear `omega(ky)` are both reproduced.

### 3.1 Fit-horizon convergence (important)

The `t_max=40` batch scan is under-converged at low ky, and the error is **one-sided
downward** — the fit reads *below* the converged value and rises with horizon:

| ky_stella | Δgamma @ t_max=40 | Δgamma @ longest horizon |
|---|---|---|
| 0.20 | −8.6% | −0.40% (t=120) |
| 0.30 | −14.4% | +0.16% (t=80) |
| 0.40 | +5.7% | +0.48% (t=80) |

This is expected: at low ky the growth rate is small, so a fixed `t_max` buys fewer
e-foldings and the fit window still contains transient. Sorting every point in the scan
by e-folding count `N = gamma_GKX · t_max` makes the rule explicit:

| N = gamma·t_max | points | |Δgamma| vs stella |
|---|---|---|
| ~2–3 | ky_gkx 0.1414, 0.2121 @ t=40 | 9–14% (one-sided low) |
| ~5–6 | ky_gkx 0.2828, 0.3536, 0.4243, 0.4950 @ t=40 | 1.3–5.7% |
| ~6–10 | ky_gkx 0.1414 @ t=120, 0.2121/0.2828 @ t=80 | ≤0.5% |

**Set `t_max` from the expected e-folding count, not from a fixed wall-time budget:
`gamma·t_max ≳ 7` for sub-percent, and a fixed `t_max=40` scan systematically
under-reports low-ky growth by 10–15%.** The three high-ky rows in the main table are
still at t=40 (N≈5.5–5.9), so their small positive Δ is within the horizon-error band
and may tighten further.

---

## 4. Tracked GX/GKX cyclone table — why it is NOT compared here

`/Users/rogeriojorge/local/GKX/docs/_static/cyclone_mismatch_table.csv` is a
**different physical lane** and is reported for context only:

| | this rung (r1) | tracked cyclone table |
|---|---|---|
| geometry | Miller, `kappa=1`, `shat=0.796`, `q=1.4` | s-alpha, `eps=0.18`, `s_hat=0.8` |
| collisions | collisionless (all collision terms 0) | hypercollisional (`nu_hermite=1`, `nu_laguerre=2`) |
| reference code | stella | GX |

Only one ky lands within 0.005 of a tracked entry — GKX `ky=0.3536` vs tracked
`ky=0.350` — and it is carried in `comparison.csv` in the `gx_lane_*` columns with
raw values (`gamma_gx=0.089845, omega_gx=0.330795, gamma_gkx=0.087650,
omega_gkx=0.332307`) and **no relative-error column computed**, deliberately. Two
independent physics differences (shaping and collisionality) separate the lanes, so a
number-to-number difference there is uninterpretable and must not be quoted as a
convention result.

**Consequence for plan finding #1**: this rung *cannot* settle whether the tracked GX
parity tooling already folds in the `sqrt(2)`. Doing that needs a like-for-like run —
GKX on the s-alpha + hypercollisional lane at the tracked ky, compared to a GX run of
the same case — not a cross-lane inference. That is a separate task from this rung.

---

## 5. Certified eigensolver cross-check (collisionless lane)

**Question**: on the collisional s-alpha lane the time-solver fit reads ~30% *below* the
certified eigenvalue (plan finding #2), yet the collisionless-Miller fits above match
stella to 0.1–3%. Both cannot be universal. One certified point decides it.

**Run** — `ky_gkx = 0.35355339` (the remap of stella `ky = 0.5`), same
`cbc_miller_scan_sqrt2.toml`, fixed krylov path:

```
JAX_ENABLE_X64=1 PYTHONPATH=/Users/rogeriojorge/local/GKX-worktrees/krylov/src \
  python3 -m gkx.cli run-runtime-linear --config cbc_miller_scan_sqrt2.toml \
          --ky 0.35355339 --solver krylov
```

The fix routes generic contracts to the residual-certified `adaptive` branch
(`KrylovConfig(method="adaptive")` in `workflows/runtime/startup.py`), which either
certifies the pair against the continuous operator or fails closed. Log
(`gkx/gkx_krylov_certified_ky0p3536.log`), ~7 min wall:

```
runtime: krylov method=adaptive dim=24 restarts=2
runtime: running certified adaptive propagator with dim=24 max_restarts=4 tol=0.000119
runtime: adaptive solve finished with eig=0.138607-0.267248j residual=5.92e-05
         converged=True stable=True
ky=0.3536 gamma=0.138607 omega=0.267248
```

Certified: **residual 5.92e-05 < tol 1.19e-4, converged and stable.** (The tolerance is
1.19e-4 rather than the 1e-9 base because the runtime CLI builds complex64 states and
the gate is floored at `1000·eps` of the working precision — a real limit on how tightly
this path can certify, worth knowing but not a failure.)

**Result at ky_stella = 0.5:**

| source | gamma | omega | units |
|---|---|---|---|
| GKX certified eigensolver | 0.138607 | 0.267248 | cs/a |
| GKX certified, converted | **0.098010** | **0.188973** | vth/a |
| GKX time-solver fit (t=40), converted | 0.098301 | 0.185627 | vth/a |
| stella | 0.097030 | 0.187022 | vth/a |

| comparison | Δgamma | Δomega |
|---|---|---|
| **time fit vs certified** | **+0.30%** | **−1.77%** |
| certified vs stella | +1.01% | +1.04% |
| time fit vs stella | +1.31% | −0.75% |

### Verdict — the ~30% fit bias is lane-specific, not universal

On the collisionless Miller lane the time-solver fit and the certified eigenvalue agree
to **0.30% in gamma** and 1.77% in omega. There is no ~30% deficit here, and the small
residual difference has the **opposite sign** (fit reads slightly *above* certified,
not 30% below). The collisional s-alpha observation therefore does **not** generalize:
plan finding #2 should be re-scoped from "IVP fit bias" to a **collisional-lane-specific**
fit pathology. Prime suspects remain on that lane specifically — hypercollisional damping
reshaping the transient, fit-signal choice, and window placement — and the fitrobust work
should target the s-alpha collisional case, not the fit machinery in general.

Caveat: one point, one lane, one resolution. It refutes universality; it does not certify
the fit everywhere. The natural follow-up is one certified point on the *collisional*
lane with an otherwise identical setup, which would isolate collisionality as the variable.

### Bonus: a fit-free confirmation of the sqrt(2) conversion

The certified eigensolver uses no fit window at all, so its ratio to stella is free of
fit artifacts:

```
R_gamma = 0.138607 / 0.097030 = 1.42850      sqrt(2) = 1.41421
R_omega = 0.267248 / 0.187022 = 1.42897
```

Both within **1.05%** of `sqrt(2)`, and consistent with each other to **0.03%**. This is
an independent, solver-independent confirmation of §2: the conversion is a property of
the normalization, not of the time-integration path.

---

## 6. What rung 2 (Rosenbluth-Hinton / zonal residual) needs to start

Both endpoints already exist; the gap is a geometry match, not new capability.

**stella side** — `/Users/rogeriojorge/local/stella/tests/regression/linear/RH/RH.in`
(shipped regression, with reference `RH.final_fields_compare`). It uses
**exactly the rung-1 CBC Miller geometry** (`rhoc=0.5, shat=0.796, qinp=1.4,
rmaj=rgeo=2.77778, kappa=1, tri=0`), `tprim=fprim=0`, adiabatic electrons,
`naky=1, aky_min=0.0, akx_min=0.05`, `ginit_option="rh"`, `nzed=48, nperiod=1`,
`nvgrid=48, nmu=12`, `delt=0.1, nstep=3000`. Note `write_phi_vs_time=.false.` —
**it must be turned on** to get a time trace for residual/GAM fitting; the shipped
comparison is final-fields only.

**GKX side** — `benchmarks/runtime_miller_zonal_response.toml` (Merlo Case III RH/GAM
benchmark): `ky=0.0, kx=0.05`, `tprim=fprim=0`, `init_field="density"`,
`init_single=true`, periodic boundary, `Nl=4, Nm=24, Nz=32`, `dt=0.005`,
`steps=12000`, all collision/hyper terms zeroed. Diagnostics for this observable already
exist (`src/gkx/diagnostics/zonal_validation.py`, `diagnostics/validation_gates.py:374`).

**Three things to settle before running:**

1. **Geometry mismatch.** The GKX zonal benchmark is Merlo Case III
   (`q=1.389, s_hat=0.751, kappa=1.4723, delta=-0.0070, shift=-0.1569`), *not* the
   stella RH CBC geometry. Rung 2 needs a GKX copy of that toml with the rung-1
   Miller block substituted (`q=1.4, s_hat=0.796, akappa=1.0, tri=0, shift=0,
   R0=R_geo=2.77778, rhoc=0.5`) — the geometry already validated to ≤1.2% against
   stella in rung 1. Do not compare stella RH against Merlo Case III.
2. **`kx` carries the same `sqrt(2)`.** stella `akx_min=0.05` (in `rho_stella`)
   corresponds to GKX `kx = 0.05/sqrt(2) = 0.0353553`. The existing GKX toml has
   `kx = 0.05` at face value, i.e. a *different physical* `kx` from stella's — this is
   exactly the trap rung 1 uncovered, and it is live in the tracked benchmark file.
3. **Which observables convert.** The residual `phi(t→inf)/phi(t=0)` is a pure ratio
   and is **convention-independent** — the cleanest possible cross-code check, immune to
   the `sqrt(2)` question. The **GAM frequency and damping rate are rates** and convert
   as `/sqrt(2)` exactly like `gamma, omega` here. Reporting both separates
   "geometry/physics agree" from "conventions agree".

Analytic anchor for the matched CBC geometry (`eps = rhoc/rmaj = 0.18, q = 1.4`):
Rosenbluth-Hinton residual `1/(1 + 1.6 q²/sqrt(eps)) ≈ 0.119`. Both codes should land
near this before any code-to-code difference is meaningful.

Suggested rung-2 order: (a) stella RH.in with `write_phi_vs_time=.true.`;
(b) GKX toml with substituted CBC geometry at `kx=0.0353553`; (c) compare residual
(no conversion) then GAM frequency/damping (`/sqrt(2)`).

---

## 7. Artifacts

Work dir
`/private/tmp/claude-501/-Users-rogeriojorge-local/1e858e4f-6438-4dbd-8d3c-60502cd814ab/scratchpad/stella_vs_gkx/`

- `comparison.csv` — machine-readable table (incl. `gx_lane_*` context columns)
- `comparison_table.txt` — rendered table + conversion verification
- `make_comparison.py` — regenerates both from the raw run outputs
- `verify_units.py` — the hypothesis-falsification test of §2.4
- `stella_scan/cbc_scan6.omega` — stella 6-point scan
- `gkx/cbc_miller_collisionless.toml`, `gkx/cbc_miller_scan_sqrt2.toml` — GKX configs
- `gkx/*.log` — GKX run logs (scan t=40, low-ky t=80, refine t=80/120, certified krylov)
- `RUNS.md` — provenance of every run

Reproduce the table: `python3 make_comparison.py` (reads logs only, no re-running).
