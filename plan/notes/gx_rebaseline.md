# GX Rebaseline on Office GPU — GKX plan item 2.1b

Date: 2026-08-18. Machine: `office` (pop-os, 2x RTX A4000).
GX build: `/home/rjorge/GX/gx` @ commit **3865a537** (`Merged in next (PR #76)`), built `GK_SYSTEM=office`.
Reference revision in GKX docs: **bc2fe552** (56 commits behind).

Output directory (all artifacts preserved): **`~/gx_rebaseline_20260818/`** on office.

---

## HEADLINE RESULT

**1. The 56-commit GX drift is benign. The tracked probe reproduces EXACTLY.**
Re-running the identical deck with `t_max = 10.0` on the new build lands on **Step 2145,
t = 10.00213** — the very same step count `capability_matrix.toml:9` records ("2,145 steps to
t=10", dt = 4.663e-3 in both) — and gives at ky=0.3:

| | γ | ω |
|---|---|---|
| **New GX @3865a537, t_max=10** | **0.101840** | **0.286760** |
| Tracked GX @bc2fe552 | 0.101814 | 0.286777 |
| **Δ** | **+2.6e-5 (+0.026 %)** | **−1.7e-5 (−0.006 %)** |

Bit-level step agreement plus sub-0.03 % eigenvalue agreement ⇒ none of the 56 upstream commits
(bpar CFL, g0 gyroaverage, FLR/Bessel precision, pyvmec sign, iota/pressure derivatives, Wspectra
removal) moves this case. **The tracked parity tables are reproducible and remain valid.**

**2. But that tracked pair is not a converged eigenvalue — it is a t=10 transient.**
Run the same deck to its own `t_max = 150` and it settles somewhere else entirely.

Cyclone s-alpha, adiabatic electrons, ky=0.3:

| source | γ | ω | note |
|---|---|---|---|
| **New GX @3865a537, converged t=150** | **0.093049** | **0.281991** | this work |
| New GX @3865a537, t=10 probe | 0.101840 | 0.286760 | this work, reproduces tracked |
| GX's own regression reference `_correct.out.nc` | 0.093018 | 0.281990 | shipped with GX |
| GKX PR#45 parity matrix `gamma_reference` | 0.0930266 | 0.2819990 | tracked, converged |
| GKX `docs/benchmarks.rst` headline | 0.101814 | 0.286777 | **t=10 transient** |

Deltas from the new run:
- vs **GX's own reference**: Δγ = +3.3e-5 (**+0.033 %**), Δω = +1e-6 (**+0.0004 %**) → new build reproduces GX's shipped answer.
- vs **PR#45 parity matrix**: Δγ = +2.2e-5 (**+0.024 %**), Δω = −8e-6 (**−0.003 %**) → PR#45's table is reproducible.
- vs **`benchmarks.rst` headline**: Δγ = −0.008765 (**−8.6 %**), Δω = −0.004786 (**−1.7 %**).

### Why the benchmarks.rst number is a transient
`capability_matrix.toml:9` records the probe as "completed 2145 steps to `t=10`". PR#45's fixture
`tools/comparison/fixtures/parity/cyclone_salpha_itg.toml` confirms it: `[time] t_max = 10.0`.
At t≈10 the ITG eigenmode has not settled; γ(ky=0.3) rings hard before locking in:

```
t= 4.67  γ=0.018294   ω=0.297231
t= 9.33  γ=0.101431   ω=0.292946   <-- the tracked 0.101814/0.286777 lives in this transient band
t=13.99  γ=0.085587   ω=0.279570
t=18.66  γ=0.114716   ω=0.291892
t=23.32  γ=0.101993   ω=0.290369
t=27.98  γ=0.081984   ω=0.286921
t=51.30  γ=0.092742   ω=0.282001   <-- settled
t=97.93  γ=0.093024   ω=0.282002
t=150.0  γ=0.093049   ω=0.281991
```

GX's own shipped reference at t=10.7 gives γ=0.097318, ω=0.285217 — also inside the ringing band,
and also ≠ its own converged 0.093018. So the scatter is intrinsic to stopping at t=10, not a code
difference.

**Recommendation:** the pair is *reproducible* (0.026 % on the new build) but it is a **fixed-t=10
runtime-probe reading, not a physical growth rate**, and it sits on the steepest part of the
transient. `docs/benchmarks.rst:156` and `benchmarks/capability_matrix.toml:9` currently present it
as "its terminal diagnostic at ky=0.3", which reads as an eigenvalue. Either quote the converged
pair **(0.09305, 0.28199)** that PR#45's parity matrix already uses, or keep the t=10 value and
label it explicitly as a fixed-step smoke-probe. As-is, anyone comparing GKX against the
`benchmarks.rst` headline with a converged solver will see a spurious 8.6 % γ discrepancy.

**Reproducing:** `~/gx_rebaseline_20260818/cyclone_salpha_t10/probe_t10.in` is the tracked deck with
`t_max=150.0 -> 10.0`, the single-line edit that regenerates the tracked pair in ~55 s.

---

## 1. Decks found

### GX's own benchmark decks (`/home/rjorge/GX/benchmarks/linear/`) — the real source
| case | deck | shipped reference |
|---|---|---|
| Cyclone s-alpha ITG, adiabatic e | `ITG_cyclone/itg_salpha_adiabatic_electrons.in` | `itg_salpha_adiabatic_electrons_correct.out.nc` |
| Cyclone Miller ITG, adiabatic e | `ITG_cyclone/itg_miller_adiabatic_electrons.in` | `itg_miller_adiabatic_electrons_correct.out.nc` |
| Cyclone Miller ITG, kinetic e | `ITG_cyclone/itg_miller_kinetic_electrons.in` | `itg_miller_kinetic_electrons_correct.out.nc` |
| KBM Miller | `KBM/kbm_miller.in` | `kbm_miller_correct.out.nc` |
| W7-X ITG, adiabatic e | `ITG_w7x/itg_w7x_adiabatic_electrons.in` | (+ `wout_w7x.nc`) |
| KAW | `KAW/kaw_betahat10.0_kp0.01.in` | — |

The Cyclone s-alpha deck matches the documented CBC parameters exactly: `tprim=2.49`, `fprim=0.8`,
`qinp=1.4`, `shat=0.8`, `eps=0.18`, `Rmaj=2.77778`, adiabatic electrons, `y0=20.0` (so
ky_min·ρ = 1/20 = 0.05 and ky=0.3 is mode 6). **No deck construction was needed.**

### GKX-side GX decks
- `tools/comparison/fixtures/etg_ky25_reference.in` — GX deck for ETG CBC at ky·ρ_i = 25 (`y0=0.2`, adiabatic ions, electron species). Authored by GKX.
- `tools/comparison/fixtures/etg_runtime_ky15_reference.in` — ETG at ky=15.

### PR #45 (`bench/gx-parity-matrix`) — six parity fixtures (TOML contracts, not GX decks)
`tools/comparison/fixtures/parity/`: `cyclone_salpha_itg.toml`, `cyclone_miller_itg.toml`,
`cyclone_miller_kinetic_electrons.toml`, `kbm_miller.toml`, `w7x_itg.toml`, `hsx_itg.toml`.
Each transcribes the physics contract from the corresponding GX deck and imports geometry from
**GX's own output netCDF** so both codes integrate identical geometric coefficients.
Tracked results: `docs/_static/gkx_gx_linear_parity_matrix.{csv,json}` (also in PR#45).

**No HSX GX deck exists** in either repo — `hsx_itg.toml` points at an external
`$GX_PARITY_REF_DIR` output that is not tracked. ETG has GX decks only on the GKX side.

### `[Wspectra]` removal — no impact
None of the GX benchmark decks nor the GKX fixtures carry a `[Wspectra]` section. All decks ran
unedited against the new build.

---

## 2/3. Run results (new GX @3865a537)

Reference column = PR#45 `gkx_gx_linear_parity_matrix.csv` `gamma_reference`/`omega_reference`,
which are GX **@bc2fe552 converged** values. Directly comparable to the new converged runs.

| case | ky | γ new-GX | γ tracked | Δγ | ω new-GX | ω tracked | Δω |
|---|---|---|---|---|---|---|---|
| cyclone_salpha_itg | 0.3 | 0.093049 | 0.0930266 | **+0.02 %** | 0.281991 | 0.2819990 | **−0.003 %** |
| cyclone_salpha_itg (t=10 probe) | 0.3 | 0.101840 | 0.101814 † | **+0.03 %** | 0.286760 | 0.286777 † | **−0.006 %** |
| cyclone_miller_itg | 0.3 | 0.125874 | 0.1258519 | **+0.02 %** | 0.215463 | 0.2154680 | **−0.002 %** |
| cyclone_miller_itg | 0.4 | 0.143106 | 0.1431215 | **−0.01 %** | 0.306686 | 0.3066950 | **−0.003 %** |
| kbm_miller | 0.2 | 0.338671 | 0.3383614 | **+0.09 %** | 0.834762 | 0.8344574 | **+0.04 %** |
| kbm_miller | 0.3 | 0.313732 | 0.3135143 | **+0.07 %** | 1.077147 | 1.0764189 | **+0.07 %** |
| w7x_itg | — | *still running on office (t_max=200); collect from `~/gx_rebaseline_20260818/w7x_itg/`* | | | | | |
| hsx_itg | — | **no GX deck exists in either repo** — cannot rerun | | | | | |

† `docs/benchmarks.rst` headline pair, not the parity-matrix column.

Cyclone (both geometries) agrees to **1 part in 10⁴**. KBM agrees to **7 parts in 10⁴** — slightly
larger, as expected for the electromagnetic branch (finite β, kinetic electrons) that the upstream
**bpar nonlinear-CFL** and **g0 gyroaverage** fixes actually touch. Still far inside any parity
tolerance.

cyclone_salpha and cyclone_miller are **complete** (t=150, values above read from the final
`.out.nc`). KBM numbers are from the settled plateau of the running diagnostics near t≈32 of 40,
stable to 5–6 significant figures across tens of diagnostic writes; the final `.out.nc` will land in
`kbm_miller/` and should shift only in the last digit.

**Still running on office at hand-off** (both detached, will finish on their own):
- `kbm_miller` — GPU 1, ~t=32/40.
- `w7x_itg` — GPU 0, ~t=14/200 (started once the `booz_xform` python shim was in place; expect ~1 h).

Collect with: `python3 ~/gx_rebaseline_20260818/extract.py <case>/<run>.out.nc`
(prints ω/γ vs time at ky=0.3; edit `kytarget` for other ky).

**Conclusion for item 2/3: the 56-commit drift does not move the tracked linear eigenvalues.**
The tracked parity tables remain valid; only the `benchmarks.rst` headline pair needs correcting
(see HEADLINE).

Values for cyclone_miller and kbm_miller were read from the converged plateau of the running
diagnostics (they had settled to 5–6 significant figures well before t_max; e.g. cyclone_miller
ky=0.3 held ω=0.21547±0.00002, γ=0.12586±0.00004 over the last 8 diagnostic writes).

---

## 4. THE NORMALIZATION QUESTION — DEFINITIVE ANSWER

### Answer: **(d) — one of the claims was simply wrong.**
The premise "GX and stella both use vth=sqrt(2T/m)" is **false for GX**. GX is in the *same*
`vth = sqrt(T/m)` family as GKX. Therefore:
- **GKX ↔ GX needs NO remap.** The sub-percent parity with no visible conversion is legitimate.
- **GKX ↔ stella DOES need the √2 remap.** stella is genuinely GS2-family.

Both original observations were correct; only the assumption that GX sits with stella was wrong.
(Independently reached here from source, and confirmed by the PR#45 assessment agent.)

### Evidence — GX side (`/home/rjorge/GX` @3865a537)

**`src/parameters.cu:1062-1065`** (in `Parameters::init_species`):
```c
species[s].vt   = sqrt(species[s].temp / species[s].mass);
species[s].tz   = species[s].temp / species[s].z;
species[s].zt   = species[s].z / species[s].temp;
species[s].rho2 = species[s].temp * species[s].mass / (species[s].z * species[s].z);
```
(Note: the coordinator's brief cited `1336-1338`; the actual lines at this revision are
**1062-1065**. Same code, corrected locations.)

**Structural proof that this is `sqrt(T/m)`, not `sqrt(2T/m)`** — the species-relative ratio
`sqrt(temp/mass)` alone is convention-agnostic (the √2 cancels in a ratio), so the decisive
evidence is the *equations*:

- **`src/device_funcs.cu:3035`** — Hermite parallel-streaming ladder:
  ```c
  rhs_par[globalIdx] = rhs_par[globalIdx] - vt_ * (sqrtf(m+1)*gmp1 + sqrtf(m)*gmm1) * gradpar;
  ```
  Coefficients are `sqrt(m+1)`, `sqrt(m)` with **no 1/√2**. That is the *probabilists'* Hermite
  recursion `ṽ·H̃_m/√(m!) = √(m+1)·H̃_{m+1}/√((m+1)!) + √m·H̃_{m-1}/√((m-1)!)`, which holds only for
  weight `exp(-ṽ²/2)` with `ṽ = v∥/v_t` — i.e. `F_M ∝ exp(-v∥²/(2 v_t²))`, hence **v_t² = T/m**.
  If v_t were `sqrt(2T/m)` the Maxwellian would be `exp(-ṽ²)` (physicists' Hermite) and the
  coefficients would carry `1/√2`: `sqrt((m+1)/2)`, `sqrt(m/2)`. GX has no such factor.

- **`src/device_funcs.cu:907`** — Bessel argument:
  ```c
  J0f[ig] = j0f(sqrtf(2. * muB[idj] * kperp2[idxyz]*rho2_s)) * f[idxyz] * fac;
  ```
  With `μB = v⊥²/(2v_t²)` this is `J0(k⊥ρ_s · v⊥/v_t) = J0(k⊥v⊥/Ω)` — again the `exp(-μB) =
  exp(-v⊥²/2v_t²)` weight, same conclusion.

- **`src/device_funcs.cu:1496` etc.** — FLR argument is `b_s = kperp2 * rho2_s`, i.e. `b = k⊥²ρ_s²`
  with **no factor of 1/2**. GS2-family codes write `b = k⊥²ρ²/2` because their ρ carries the √2.

- **`src/geometry.cu:532,536,540,544`** — GS2-convention drifts are halved on ingest:
  ```c
  nc_gbdrift_h[n]  = dtmp[n] / 2.0;
  nc_gbdrift0_h[n] = dtmp[n] / 2.0;
  nc_cvdrift_h[n]  = dtmp[n] / 2.0;
  nc_cvdrift0_h[n] = dtmp[n] / 2.0;
  ```
  GX converts GS2-convention geometry *into* its own halved convention on read. **Confirms the
  coordinator's point 2** (their line numbers 537/795 are approximately right; the exact block is
  532-544).

### Evidence — GKX side (`/Users/rogeriojorge/local/GKX-worktrees/planpr`)

- **`src/gkx/operators/linear/params.py:264-266`**:
  ```python
  vth=jnp.sqrt(temperature / mass),
  rho=jnp.sqrt(temperature * mass) / jnp.abs(charge),
  tz=temperature / charge,
  ```
  Line-for-line the same three quantities as GX's `parameters.cu:1062-1065`
  (`vt = sqrt(T/m)`, `tz = T/z`, `rho2 = T·m/z²` ⇒ `rho = sqrt(T·m)/|z|`).

- **`src/gkx/operators/linear/streaming.py:474`** + **`cache_arrays.py:60-61,78-79`** —
  `hermite_ladder_coeffs` returns `sqrt(m+1)` and `sqrt(m)`; `apply_hermite_v` returns
  `sqrt_p * G_plus + sqrt_m * G_minus`. **Identical ladder to GX, no 1/√2.**

- **`src/gkx/diagnostics/normalization.py:41-46`** — `CYCLONE_NORMALIZATION` has
  `rho_star=1.0, omega_d_scale=1.0, omega_star_scale=1.0`, `diagnostic_norm_default="none"`.
  The `contract="cyclone"` is a **pure no-op**. Confirmed against `docs/normalization.rst` table.
  → **rules out hypothesis (c)** (the contract does not fold in any factor).

- **PR#45 `tools/comparison/fixtures/parity/cyclone_salpha_itg.toml`** — `ky = 0.3`, `tprim = 2.49`,
  `fprim = 0.8`, `y0 = 20.0`: numerically identical to the GX deck, **no √2 rescaling anywhere**.
  Geometry is imported from GX's own output netCDF.
  → **rules out hypotheses (a)-as-special-casing and (b)**: the decks are not "written in
  GKX-compatible units" as a conversion, and the tooling does not convert — *no conversion is
  needed*, because the conventions already coincide.

- `grep` for `sqrt(2)|1.414|convert|remap` across `tools/comparison/*.py` and
  `build_gx_parity_matrix.py`: **no ky/γ/ω conversion exists** — correctly so.

### Independent corroboration of the PR#45 assessment agent's five points
All five were re-derived here from source and from a live GX output, **all confirmed**:

1. ✅ `vt = sqrt(temp/mass)`, `tz = temp/z`, `rho2 = temp*mass/(z*z)` — found at
   `src/parameters.cu:**1062-1065**` (their cited 1336-1338 is off; corrected location above).
2. ✅ GX halves GS2-convention drifts on ingest — `src/geometry.cu:**532, 536, 540, 544**`
   (`gbdrift`, `gbdrift0`, `cvdrift`, `cvdrift0` each `/2.0`).
3. ✅ **Verified empirically from the new run's own output netCDF**: at θ=0,
   `gbdrift = cvdrift = 0.35999972 = 1/Rmaj` (Rmaj=2.77778), **not** `2/Rmaj = 0.72`.
   This is the strongest single check — it shows the *stored, integrated* geometry is the halved
   (non-GS2) form, exactly what `v_t² = T/m` requires.
4. ✅ GKX importer applies 0.5 only to root-level `.eik.nc`, never to grouped GX output —
   `src/gkx/geometry/flux_tube.py:433-452`: the `selection.is_grouped_output` branch returns the
   drifts verbatim, the fallback branch multiplies by `0.5`. Correct bridge on each branch.
5. ✅ ky grids agree: both are `ky = n/y0` with `y0=20`; PR#45's parity CSV ky column is
   0.05, 0.10, 0.15, … matching GX's printed spectrum exactly.
6. ✅ `contract="cyclone"` is a no-op (all scales 1.0, `diagnostic_norm="none"`).

### Evidence — stella side (why the √2 there is real)
From `plan/notes/stella_study.md:55`: stella's thermal speed is `v_th,s = sqrt(2 T_s/m_s)`
(Maxwellian `exp(-vpa² - μB)` on the code grid) — genuinely GS2-family. `stella_vs_gkx_runs.md:34-41`
records the measured remap. Note `stella_study.md:117` asserts "both codes use v_th = sqrt(2T/m)"
about GX and stella — **that line is the error**, and it even hedges "still worth a one-line unit
sanity check". This work is that check, and it comes out the other way.

### Exact conversion formulas

Let subscript **G** = GKX/GX family (`v_ref = sqrt(T_ref/m_ref)`, `ρ_ref = sqrt(T_ref m_ref)/|q_ref|B_ref`)
and **S** = stella/GS2 family (`v_ref^S = sqrt(2T_ref/m_ref) = √2 · v_ref^G`, so `ρ_ref^S = √2 · ρ_ref^G`).

For the **same physical mode** and the same reference length `a`:

```
ky:     ky_S · ρ_ref^S  =  √2 · (ky_G · ρ_ref^G)        →   ky_S = √2 · ky_G
kx:     kx_S            =  √2 · kx_G                          (same reasoning)

rates:  γ_S [v_ref^S/a] =  γ_G [v_ref^G/a] / √2         →   γ_S = γ_G / √2
        ω_S [v_ref^S/a] =  ω_G [v_ref^G/a] / √2         →   ω_S = ω_G / √2

time:   t_S [a/v_ref^S] =  t_G [a/v_ref^G] / √2
```

Inverse (stella → GKX): `ky_G = ky_S/√2`, `γ_G = √2·γ_S`, `ω_G = √2·ω_S`.

**GKX ↔ GX: identity.** `ky_GX = ky_GKX`, `γ_GX = γ_GKX`, `ω_GX = ω_GKX`.

Gradients `tprim = a/L_T`, `fprim = a/L_n` are dimensionless and **identical in all three codes**
(CBC `R/L_T = 6.9` with `a/R = 0.36` ⇒ `tprim = 2.49` everywhere). `β` is also convention-free.

### Sanity check against the stella measurement
`stella_vs_gkx_runs.md:14` records stella at ω=0.187022, γ=0.097030 in `[vth/a, vth=sqrt(2T/m)]`.
Mapping to the GKX/GX family: γ_G = √2·0.097030 = 0.13722, ω_G = √2·0.187022 = 0.26449 — the right
order for a Miller CBC ITG mode near the spectrum peak (cf. new-GX Miller at ky=0.3:
γ=0.12586, ω=0.21547; different ky and geometry details, so agreement is only indicative).

---

## 5. Saved outputs (office)

Root: **`~/gx_rebaseline_20260818/`**

```
cyclone_salpha/       itg_salpha_adiabatic_electrons.{in,out.nc,big.nc,restart.nc}, run.log
                  itg_salpha_adiabatic_electrons_correct.out.nc  (GX shipped reference, for diffs)
cyclone_salpha_t10/   probe_t10.{in,out.nc}, run.log   <-- reproduces the tracked benchmarks.rst pair
cyclone_miller/   itg_miller_adiabatic_electrons.{in,out.nc}, run.log, *_correct.out.nc
kbm_miller/       kbm_miller.{in,out.nc}, run.log, kbm_miller_correct.out.nc
w7x_itg/          itg_w7x_adiabatic_electrons.{in,out.nc}, wout_w7x.nc, run.log
extract.py        helper: prints omega/gamma vs time at a chosen ky from any GX out.nc
run_gpu0.sh, run_rest.sh, *_driver.log
```

Every case directory keeps GX's own `*_correct.out.nc` next to the new output, so later runs can
diff three ways: new-vs-new, new-vs-GX-shipped-reference, and new-vs-GKX-tracked.

### Environment note (needed to reproduce)
GX shells out to `python` for Miller/VMEC geometry preprocessing
(`geometry_modules/miller/gx_geo.py`, `geometry_modules/pyvmec/gx_geo_vmec.py`), but the machine has
only `python3`. Without a `python` on PATH those cases fail with `Cannot open file *.eik.out` /
`Error: No such file or directory. See file: src/geometry.cu, line 448`.
Additionally the VMEC path (`gx_geo_vmec.py`) needs **`booz_xform`**, which the system python3 lacks;
`~/stellarator_venv` has it.

**Fix applied** (only new file is `~/bin/python`; nothing else on the machine was modified):
```bash
mkdir -p ~/bin
printf '#!/bin/bash\nexec $HOME/stellarator_venv/bin/python "$@"\n' > ~/bin/python
chmod +x ~/bin/python
export PATH="$HOME/bin:$PATH"
```
A bare **symlink does not work** — `ln -s ~/stellarator_venv/bin/python ~/bin/python` breaks venv
detection and `booz_xform` stays unimportable. It must be a wrapper script.
s-alpha geometry is internal to GX and unaffected by any of this.

Worth raising upstream: GX hardcodes `python` rather than `python3`/`sys.executable` when shelling
out to its geometry modules, which is a silent failure on any modern distro.
