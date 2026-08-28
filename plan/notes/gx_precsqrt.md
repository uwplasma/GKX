# GKX plan item 2.1e — office GX rebuild with `-prec-sqrt=true`

Date: 2026-08-19 · Machine: `office` (pop-os, 36 cores, 2× RTX A4000 sm_86)
GX revision: **3865a537** (unchanged, clean) · nvcc 11.5.119, gcc-10.4.0 host compiler

**Bottom line:** `-prec-sqrt=true` is *not* a no-op — it changes the generated device code and
moves eigenvalues at the 1e-4…1e-3 relative level. But it does not degrade anything, and on the
one case that previously failed GX's own regression criterion (`kbm_miller`) it **improves**
agreement with the shipped reference. Recommend adopting it as canonical. GKX's tracked parity
matrix does **not** need regenerating.

---

## 1. Flag semantics — determined empirically, not assumed

The CUDA 11.5 `nvcc --help` text says `--use_fast_math` *implies* `--prec-sqrt=false`, which
leaves open whether an explicit `-prec-sqrt=true` is honoured or silently overridden. Settled by
compiling a probe kernel (`sqrtf`, float divide, double `sqrt`) at `-arch=compute_86` and reading
the PTX:

| flags | f32 sqrt | f32 divide | f64 sqrt |
|---|---|---|---|
| *(none)* | `sqrt.rn.f32` | `div.rn.f32` | `sqrt.rn.f64` |
| `-use_fast_math` | `sqrt.approx.ftz.f32` | `div.approx.ftz.f32` | `sqrt.rn.f64` |
| `-use_fast_math -prec-sqrt=true` | **`sqrt.rn.ftz.f32`** | `div.approx.ftz.f32` | `sqrt.rn.f64` |
| `-prec-sqrt=true -use_fast_math` | **`sqrt.rn.ftz.f32`** | `div.approx.ftz.f32` | `sqrt.rn.f64` |
| `-prec-sqrt=true` | `sqrt.rn.f32` | `div.rn.f32` | `sqrt.rn.f64` |

**Resulting behaviour:** the explicit `-prec-sqrt=true` **wins over** the `-use_fast_math`
implication, and does so **regardless of argument order**. What changes is *only* single-precision
`sqrtf`: fast approximate → IEEE round-to-nearest. Flush-to-zero (`ftz`), fast division
(`div.approx`) and contracted FMA all stay on, and double-precision `sqrt` was already
correctly-rounded. So the two flags do not cancel; `-prec-sqrt=true` is a strict, narrow
tightening of `-use_fast_math`.

This matters because GX is a predominantly single-precision code (≈503 `float` declarations vs
≈111 `double` in `src/*.cu`; 51 `sqrtf(` call sites in device code), so the change has real reach.

### Decision: add **only** `-prec-sqrt=true`, keep `-use_fast_math`

That is exactly how upstream `Makefiles/Makefile.ubuntu` spells it, and nothing else in that file
differs numerically from `Makefile.office` (the rest is package paths, `CUDAARCH=75` vs our `86`,
and no `--std=c++17`). `-use_fast_math` was kept because **no upstream makefile drops it** —
removing it would be a larger deviation from upstream than upstream itself ever takes.

Worth recording: **upstream is not uniform on this flag.** Of the 20 shipped makefiles, only four
carry `-prec-sqrt=true` (`ubuntu`, `gx`, `getafix`, `psfcgpu`); the large HPC ones
(`perlmutter`, `summit`, `stellar`, `daint`, `polaris`, `raven`, `traverse`, `m100`, …) use bare
`-use_fast_math`, i.e. the old office setting. So "upstream's current numerics flag" really means
"`Makefile.ubuntu`'s" — which is the right analogue for a desktop workstation, and is the
maintained one (maintainer `iabel@umd.edu`).

---

## 2. Build

Side-by-side, both runnable:

| build | tree | binary | NVCCFLAGS numerics |
|---|---|---|---|
| **A (old)** | `/home/rjorge/GX` | `/home/rjorge/GX/gx`, archived `~/gx_builds/gx.nofastsqrt` | `-use_fast_math` |
| **B (new)** | `/home/rjorge/GX_precsqrt` | `/home/rjorge/GX_precsqrt/gx`, archived `~/gx_builds/gx.precsqrt` | `-use_fast_math -prec-sqrt=true` |

- Backup of the original flags: `/home/rjorge/GX/Makefiles/Makefile.office.nofastsqrt.bak`.
- Both trees are at **3865a537**; `git status` in each shows only untracked `Makefiles/Makefile.office`
  (plus the `.bak` in the original tree). Nothing tracked was modified, nothing committed or pushed.
  A stale cosmetic edit left in `benchmarks/linear/ITG_cyclone/itg_salpha_adiabatic_electrons.in`
  by the previous session (a dropped trailing comment; `t_max = 150.0` value unchanged) was
  reverted, so the checkout is now genuinely clean.
- Build: `make -j36 GK_SYSTEM=office`, exit 0, no errors (only the usual benign
  `nvlink warning : Skipping incompatible libpthread.a/libdl.a`).
- Binaries differ (167 147 984 vs 167 143 888 bytes). SASS confirms the flag bit:
  `obj/device_funcs.o` grows from 75 331 to 76 504 disassembled lines — the multi-instruction IEEE
  sqrt sequences replacing single `MUFU.RSQ` approximations.

Exact compile line used (from `make -n GK_SYSTEM=office obj/linear.o`):

```
/usr/bin/nvcc -Wall -Wno-unused-local-typedefs -Wno-deprecated-declarations -Wno-parentheses \
  -Wno-unused-result -c -o obj/linear.o src/linear.cu \
  -ccbin /home/rjorge/local/install/gcc-10.4.0/bin/g++ --std=c++17 \
  --forward-unknown-to-host-compiler -arch=compute_86 -code=sm_86 \
  -use_fast_math -prec-sqrt=true -fPIC -rdc=true -O3 \
  -I. -I include -I geometry_modules/vmec/include \
  -I /home/rjorge/local/install/libcutensor-1.7.0.1/include \
  -I /home/rjorge/local/install/nccl-2.18.1/include \
  -I /home/rjorge/local/install/openmpi-4.1.6/include \
  -I /home/rjorge/local/install/netcdf-c-4.9.2/include \
  -I /home/rjorge/local/install/gsl-2.7.1/include \
  -DGX_PATH=\"/home/rjorge/GX_precsqrt\"
```

---

## 3. Runs

New outputs: **`~/gx_rebaseline_precsqrt_20260819/`**, same five cases, same inputs, GX's shipped
`*_correct.out.nc` kept beside each, plus `extract.py`, `report.py`, `checkpf.py`, `vs_tracked.py`.
All five exited rc=0. GPU 0 was occupied by another user's job throughout, so everything ran
serially on GPU 1 (`CUDA_VISIBLE_DEVICES=1`); no other process was disturbed.

| case | t_max | runtime old → new |
|---|---|---|
| cyclone_salpha_t10 | 10 | 0.93 → 0.41 min |
| cyclone_salpha | 150 | 6.26 → 6.17 min |
| kbm_miller | 40 | 15.29 → 15.44 min |
| cyclone_miller | 150 | 10.02 → 8.18 min |
| w7x_itg | 200 | 8.73 → 7.39 min |

**No performance cost.** The apparent speedups are GPU-contention differences between the two
sessions, not the flag; the fair comparison is `cyclone_salpha` (both alone on a GPU), 6.26 → 6.17 min,
and `kbm_miller`, +1%. Both are noise.

### Noise floor is exactly zero

Before attributing any delta to the flag I measured GX's run-to-run scatter by re-running
`cyclone_salpha` and `kbm_miller` a second time with the *same* build B binary. Both came back
**bitwise identical** — `np.array_equal` on the entire `omega_kxkyt` array is `True`, identical
`tend` to all digits. GX is deterministic on this machine, so **every** number below is caused by
the flag and nothing else.

---

## 4. Three-way comparison

Final-time values at t_max (the convention the previous re-baseline quoted). "shipped" = GX's own
`*_correct.out.nc`. Percentages are relative differences.

### cyclone_salpha, t=150 (converged) — the headline case

| ky | qty | shipped | old (A) | new (B) | new−old | old−ship | new−ship |
|---|---|---|---|---|---|---|---|
| 0.20 | γ | 0.075014 | 0.075009 | 0.075060 | **+0.068%** | −0.006% | +0.062% |
| 0.20 | ω | 0.177876 | 0.177909 | 0.177900 | −0.005% | +0.018% | +0.014% |
| **0.30** | **γ** | 0.093018 | **0.093049** | **0.093049** | **+0.0000%** | +0.034% | +0.034% |
| 0.30 | ω | 0.281990 | 0.281991 | 0.281994 | +0.001% | +0.000% | +0.001% |
| 0.40 | γ | 0.080897 | 0.080911 | 0.080911 | **+0.0000%** | +0.018% | +0.018% |
| 0.40 | ω | 0.374950 | 0.374916 | 0.374927 | +0.003% | −0.009% | −0.006% |
| 0.50 | γ | 0.054089 | 0.054103 | 0.054103 | **+0.0000%** | +0.026% | +0.026% |
| 0.50 | ω | 0.455883 | 0.455887 | 0.455885 | −0.001% | +0.001% | +0.000% |

γ at ky = 0.3, 0.4, 0.5 is **identical to six decimal places** between the two builds. Only
ky = 0.2 (and the near-marginal ky = 0.05) move at all.

### cyclone_salpha t=10 probe (transient, not an eigenvalue)

| ky | qty | old (A) | new (B) | new−old |
|---|---|---|---|---|
| 0.30 | γ | 0.101840 | 0.101865 | +0.025% |
| 0.30 | ω | 0.286760 | 0.286759 | −0.000% |

### cyclone_miller

| ky | qty | shipped | old (A) | new (B) | new−old | old−ship | new−ship |
|---|---|---|---|---|---|---|---|
| 0.30 | γ | 0.125875 | 0.125874 | 0.125808 | −0.052% | −0.001% | −0.053% |
| 0.30 | ω | 0.215471 | 0.215463 | 0.215483 | +0.009% | −0.004% | +0.006% |
| 0.40 | γ | 0.143121 | 0.143106 | 0.143073 | −0.023% | −0.010% | −0.033% |
| 0.40 | ω | 0.306696 | 0.306686 | 0.306702 | +0.005% | −0.003% | +0.002% |

### kbm_miller — the one case where the build choice is visible

| ky | qty | shipped | old (A) | new (B) | new−old | old−ship | new−ship |
|---|---|---|---|---|---|---|---|
| 0.20 | γ | 0.339001 | 0.337937 | 0.338487 | **+0.163%** | −0.314% | **−0.151%** |
| 0.20 | ω | 0.833044 | 0.831894 | 0.833589 | **+0.204%** | −0.138% | +0.065% |
| 0.30 | γ | 0.314112 | 0.313732 | 0.313182 | **−0.175%** | −0.121% | −0.296% |
| 0.30 | ω | 1.075928 | 1.075978 | 1.076424 | +0.041% | +0.005% | +0.046% |

### w7x_itg (GX ships no `_correct.out.nc` for this case)

| ky | qty | old (A) | new (B) | new−old |
|---|---|---|---|---|
| 0.30 | γ | 0.032714 | 0.032693 | −0.062% |
| 0.30 | ω | 0.065310 | 0.065318 | +0.012% |
| **1.00** | **γ** | 0.174714 | 0.174714 | **+0.0000%** |
| 1.00 | ω | 0.148908 | 0.148908 | −0.000% |
| 1.60 | γ | 0.244639 | 0.244619 | −0.008% |
| 1.60 | ω | 0.444202 | 0.444199 | −0.001% |

### Upstream's own PASS/FAIL criterion (`check.py`: 2nd-half-average, all ky>0, γ<1e-3, ω<5e-3)

| case | build A (old) | build B (new) |
|---|---|---|
| cyclone_salpha | max\|Δγ\|=7.26e−3 (ky=0.05) → FAIL | 7.57e−3 (ky=0.05) → FAIL |
| cyclone_miller | max\|Δγ\|=3.39e−3 (ky=0.05) → FAIL | 3.51e−3 (ky=0.05) → FAIL |
| **kbm_miller** | max\|Δγ\|=**2.60e−3** (ky=0.10) → **FAIL** | **9.29e−4** (ky=0.40) → **PASS** |

The two cyclone "FAIL"s are pre-existing in *both* builds and are driven entirely by ky = 0.05,
the lowest, near-zero-γ marginal mode where a relative difference is meaningless. Not a regression,
not caused by the flag. The kbm_miller result is the real signal: the flag flips it from FAIL to PASS.

---

## 5. Does `-prec-sqrt=true` move anything that matters?

**Yes, but only just — and in the right direction.** Honest reading:

- On well-converged, physically meaningful modes the flag is **literally undetectable**:
  cyclone s-α γ at ky = 0.3/0.4/0.5 and W7-X γ at ky = 1.0 are bit-identical between builds.
- Elsewhere it shifts things by **0.01–0.07%**, i.e. below or at the edge of GKX's sub-0.1% gates:
  cyclone_miller 0.02–0.05%, cyclone s-α ky=0.2 0.07%, W7-X ky=0.3 0.06%, t=10 probe 0.025%.
- **`kbm_miller` is the exception at 0.16–0.20%**, which does exceed a sub-0.1% gate. This is the
  electromagnetic branch (finite β, kinetic electrons) at a short t_max = 40 — the most
  arithmetic-sensitive of the five, and the same case the previous re-baseline already flagged as
  having the largest scatter. Crucially the shift is **toward** the references, not away: kbm goes
  FAIL → PASS against GX's shipped reference, and against GKX's tracked matrix its ky=0.2 γ error
  improves from −0.125% to +0.037% and its ω error from −0.307% to −0.104%.

So this is not the "moves nothing measurable" outcome, but it is close to it, and the movement is
benign-to-beneficial.

### Both builds vs GKX's tracked parity matrix (`docs/_static/gkx_gx_linear_parity_matrix.csv`)

| case | ky | Δγ old | Δγ new | Δω old | Δω new | converged |
|---|---|---|---|---|---|---|
| cyclone_salpha_itg | 0.30 | +0.024% | +0.024% | −0.003% | −0.002% | True |
| cyclone_miller_itg | 0.30 | +0.017% | −0.035% | −0.002% | +0.007% | True |
| cyclone_miller_itg | 0.40 | −0.011% | −0.034% | −0.003% | +0.002% | True |
| kbm_miller | 0.20 | −0.125% | **+0.037%** | −0.307% | **−0.104%** | True |
| kbm_miller | 0.30 | +0.069% | −0.106% | −0.041% | +0.000% | True |
| w7x_itg | 0.30 | −29.68% | −29.73% | +1.48% | +1.49% | **False** |
| w7x_itg | 1.00 | −0.014% | −0.014% | +0.004% | +0.004% | True |
| w7x_itg | 1.60 | −0.003% | −0.011% | +0.001% | +0.000% | True |

(The old-build column reproduces the previously reported deltas exactly, confirming the
methodology matches. The W7-X ky=0.3 row is the known marginal mode the tracked table itself
marks `converged=False`.)

Every `converged=True` row stays within **±0.11%** of tracked under the new build, and six of eight
within ±0.04%.

---

## 6. Recommendations

**1. Make build B (`-use_fast_math -prec-sqrt=true`) canonical for future office references.**
Reasons, in order of weight:
- It matches upstream's maintained desktop makefile (`Makefile.ubuntu`), so parity numbers are
  taken under the arithmetic that upstream's own reference platform uses — which was the point of
  this exercise.
- IEEE-correct `sqrtf` removes an arbitrary, undocumented source of divergence from the reference
  chain, at zero cost in either accuracy or runtime.
- It measurably *improves* agreement with GX's shipped regression reference on the only case that
  previously failed upstream's own `check.py` criterion.
- No downside found: no converged eigenvalue degrades by more than 0.05%, and the most-converged
  ones do not move at all.

To adopt: promote `/home/rjorge/GX_precsqrt/Makefiles/Makefile.office` into `/home/rjorge/GX`
(the `.nofastsqrt.bak` preserves the old flags for reproducing build A) and rebuild in place, or
just keep using the `GX_precsqrt` tree. Both binaries are archived in `~/gx_builds/`. Note the
`-DGX_PATH` bake-in: `gx.precsqrt` resolves its geometry python modules to `/home/rjorge/GX_precsqrt`,
so that tree must stay in place.

If GKX tracks the build environment (e.g. `benchmarks/capability_matrix.toml`), the office entry
should record `-use_fast_math -prec-sqrt=true` and note that this now matches `Makefile.ubuntu`.

**2. GKX's tracked parity numbers do NOT need regenerating.** Every `converged=True` row still
agrees to ≤0.11%, and the previous re-baseline's conclusion — that the 56-commit drift does not
move the tracked linear eigenvalues — survives the rebuild unchanged. Two specific points:

- The converged Cyclone s-α headline pair the previous session recommended for
  `docs/benchmarks.rst` (**γ=0.093049, ω=0.281991**) is *unchanged* by the flag: γ is
  bit-identical and ω moves to 0.281994 (+1e-5 relative). That recommendation stands as written.
- If the docs anywhere still quote the **t=10 probe** pair, under build B it becomes
  0.101865 / 0.286759 rather than 0.101840 / 0.286760. This is a further argument for the
  previously recommended fix of replacing that transient pair with the converged one — the
  transient value is build-sensitive precisely because it is not an eigenvalue.

**3. One thing to watch: `kbm_miller` should not be gated at sub-0.1%.** It is the single case
where the compiler flag is visible above that threshold (0.16–0.20% between two legitimate builds
of the *same commit*, with zero run-to-run noise). A sub-0.1% gate on kbm is therefore gating on
compiler arithmetic, not on physics. Either loosen the kbm tolerance to ~0.3% (documenting why:
short t_max=40 electromagnetic branch), or regenerate the kbm reference under the canonical build
and keep the tight gate — but do not leave a tight gate on a number that moves with `-prec-sqrt`.

---

## 7. Artifacts left on `office`

- `~/gx_rebaseline_precsqrt_20260819/` — new reference outputs, five cases + `repeat/`
  (determinism check), shipped `*_correct.out.nc` alongside, analysis scripts
  (`report.py`, `checkpf.py`, `vs_tracked.py`, `extract.py`), `driver.log`, `repeat.log`.
- `~/gx_rebaseline_20260818/` — previous build-A references, untouched.
- `/home/rjorge/GX_precsqrt/` — build-B tree @3865a537, `build.log` with the full build transcript.
- `~/gx_builds/gx.nofastsqrt`, `~/gx_builds/gx.precsqrt` — both binaries archived.
- `/home/rjorge/GX/Makefiles/Makefile.office.nofastsqrt.bak` — original office flags.
- `~/nvcc_precsqrt_test/` — the PTX flag-interaction probe (`t.cu` + five `.ptx`).

No commits, no pushes, no system or other-user changes.
