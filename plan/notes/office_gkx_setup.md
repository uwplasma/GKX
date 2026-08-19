# GKX on the office GPU box — setup, CPU/GPU parity gate, timings

Date: 2026-08-18. Host: `ssh office` (pop-os, 36 cores, 2× RTX A4000 16 GB, driver 580.119.02).
Code under test: `origin/main` @ **fb559974**, plus `fix/krylov-certified-default` @ **9385069e** (PR #52).

---

## 1. Recon

- **GPUs idle** before and after all work; only the user's own `stellarator_venv` webapp
  (~700–775 MiB per GPU, 0 % util) present. Left in that state.
- **Existing GKX checkouts on office**:
  - `/home/rjorge/GKX` — on `feat/bounded-memory-nonlinear-adjoint` @ 8b59d806,
    **DIRTY**: 275 modified/deleted paths (mostly `docs/_static/*.json` deletions) + 4 untracked.
    *Not touched* — no reset, no checkout, no stash.
  - `/home/rjorge/local/scratch_gkx_cpugpu_char/GKX` — separate clean clone on `main` @ fb55997.
- **Pre-existing env**: system `python3` 3.10.12 with jax **0.6.2** (CUDA, 2 devices).
  Only venv was `~/venvs/vmex`. No jax ≥ 0.10 anywhere.
- System nvcc 11.5; nvidia-smi reports CUDA 13.0 capability from the driver.

## 2. Branches fetched (into `~/GKX`, no local state disturbed)

```
git -C ~/GKX fetch origin main \
  refs/heads/fix/phase0-robustness:refs/remotes/origin/fix/phase0-robustness \
  refs/heads/feat/wout-cli:refs/remotes/origin/feat/wout-cli \
  refs/heads/fix/krylov-certified-default:refs/remotes/origin/fix/krylov-certified-default
```
| ref | sha |
|---|---|
| origin/main | fb559974 |
| origin/fix/phase0-robustness (#50) | 6d92db70 |
| origin/feat/wout-cli (#51) | 648ae975 |
| origin/fix/krylov-certified-default (#52) | 9385069e |

Worktrees created so the dirty primary checkout stays put:
`~/gkx-wt/main` (fb559974, detached), `~/gkx-wt/krylov` (9385069e, detached).

## 3. Environment recipe (exact, reproducible)

**Blocker found:** the box only has Python **3.10**, and PyPI caps jax at **0.6.2** for
py310 — below the project floor of 0.10.1. Fixed *without touching system packages* by
bootstrapping `uv` in a throwaway venv and letting it fetch a standalone CPython 3.12
into the user's home.

```bash
# one-time bootstrap (no system packages modified)
python3 -m venv ~/.venvs/bootstrap
~/.venvs/bootstrap/bin/pip install -U pip uv
export PATH=$HOME/.venvs/bootstrap/bin:$PATH
uv python install 3.12                      # -> CPython 3.12.13 in ~/.local/share/uv

# main-branch env
uv venv --python 3.12 ~/.venvs/gkx-gpu
VIRTUAL_ENV=$HOME/.venvs/gkx-gpu uv pip install "jax[cuda12]>=0.10.1" pytest
cd ~/gkx-wt/main && VIRTUAL_ENV=$HOME/.venvs/gkx-gpu uv pip install -e .

# second env for the krylov PR (uv cache makes this ~1 min, hardlinked)
uv venv --python 3.12 ~/.venvs/gkx-gpu-krylov
VIRTUAL_ENV=$HOME/.venvs/gkx-gpu-krylov uv pip install "jax[cuda12]>=0.10.1" pytest
cd ~/gkx-wt/krylov && VIRTUAL_ENV=$HOME/.venvs/gkx-gpu-krylov uv pip install -e .
```

Two venvs (rather than one + `PYTHONPATH`) because uv's editable install uses a
metapath finder that **wins over `PYTHONPATH`**, so a path override cannot redirect
`gkx` to another worktree. (On the laptop's `~/.venvs/gkx-jax-latest` the older-style
editable install *does* respect `PYTHONPATH` — that asymmetry is worth knowing.)

**Result:** jax/jaxlib **0.11.1**, gkx 1.7.1 editable.

```
$ ~/.venvs/gkx-gpu/bin/python -c "import jax; print(jax.devices())"
[CudaDevice(id=0), CudaDevice(id=1)]
[(0,'NVIDIA RTX A4000','gpu'), (1,'NVIDIA RTX A4000','gpu')]
```

**cuda12 wheels work fine** despite system nvcc being 11.5 — the wheels ship their own
CUDA 12.9 runtime and driver 580 is new enough. No cu11 fallback needed.
Note: JAX preallocates ~75 % of GPU memory by default; use
`XLA_PYTHON_CLIENT_MEM_FRACTION=.4` when sharing the box.

## 4. Correctness / parity gate

### 4a. Default linear demo — `python -m gkx.cli`

| | γ | ω |
|---|---|---|
| Laptop CPU reference | 0.089982 | 0.289838 |
| **Office GPU** | **0.089982** (0.08998227206171222) | **0.289838** (0.2898382438280345) |

**Exact agreement to all printed digits. PASS.** Wall 51.6 s (cold, incl. compile).

### 4b. Cyclone nonlinear short (100 steps, 64×64×24)

Three-way, all default (f32) precision:

| quantity | laptop CPU (jax 0.10.2) | office CPU (jax 0.11.1) | **office GPU** | GPU vs laptop CPU (rel) |
|---|---|---|---|---|
| Wg | 4.066938126925379e-4 | 4.066938126925379e-4 | 4.066947731189430e-4 | **2.4e-6** |
| Wphi | 8.416007403866388e-6 | 8.416006494371686e-6 | 8.416030141233932e-6 | **2.7e-6** |
| heat_flux(t=5) ×1e5 | 3.4246590076 | 3.4246586438 | 3.4246717405 | **3.7e-6** |
| omega_last | 0.048158060759306 | 0.048158276826143 | 0.048158809542656 | 1.6e-5 |
| gamma_last | -0.0042812637984753 | -0.0042812637984753 | -0.0042836484499276 | 5.6e-4 * |

\* `gamma_last` is a single instantaneous log-derivative of a ~1e-3-relative quantity;
its absolute error is 2.4e-6, i.e. the same f32 noise as everything else.

**All deltas 2–4e-6 relative = textbook single-precision agreement. PASS.**

#### Resolved anomaly: the plan's `Wg` reference is stale, not a GPU discrepancy

The plan's reference `Wg=0.000406441` disagrees with the GPU value by **6.2e-4** relative
— far above f32 noise. Root-caused as **code version, not hardware**:

- office GPU  → 0.000406695
- office CPU (same jax) → 0.000406694
- **laptop CPU, main's nonlinear code (via `GKX-worktrees/krylovpr`, jax 0.10.2) → 0.000406694**

All three agree. The 0.000406441 reference must come from the laptop's
`feat/bounded-memory-nonlinear-adjoint` @ 7cf5e6d1 checkout, whose nonlinear path differs
from `main` @ fb559974. (`Wphi` and `heat_flux` references *do* reproduce on main to
~7e-7, so only `Wg` moved between the two code states.)
**Action for the plan: re-baseline `Wg` to 0.00040669 against main, or note which branch
the 0.000406441 figure belongs to.** The krylov PR touches only
`solvers/linear/{krylov,adaptive_propagator}.py` and `workflows/runtime/startup.py`, so its
nonlinear path is byte-identical to main — that is what made this isolation possible.

### 4c. ky scan — `scan-runtime-linear --config .../cyclone.toml --solver time`

| ky | plan reference γ | **office GPU** γ | office CPU γ | GPU vs CPU rel |
|---|---|---|---|---|
| 0.1 | 0.0168 | 0.016811420 | 0.016810950 | 2.8e-5 |
| 0.2 | 0.0362 | 0.036193841 | 0.036193881 | 1.1e-6 |
| 0.3 | 0.0632 | 0.063224408 | 0.063224459 | 8.0e-7 |
| 0.4 | 0.0575 | 0.057530505 | 0.057529965 | 9.4e-6 |
| 0.5 | 0.0244 | 0.024371486 | 0.024371537 | 2.1e-6 |

GPU ω = [0.079899, 0.197997, 0.300885, 0.392742, 0.472466].
Every GPU γ rounds to the reference at the reference's 4-decimal precision. **PASS.**
Wall: GPU **25.0 s**, office CPU 103.5 s (**4.1×**).

## 5. PR #52 `fix/krylov-certified-default` on GPU — **BLOCKER FOUND**

`scan-runtime-linear --config .../cyclone.toml --ky-values 0.3 --no-progress`

| config | result | wall |
|---|---|---|
| laptop CPU (reference in task) | γ=0.088930, ω=0.280209 | — |
| laptop CPU, today, jax 0.10.2 | γ=0.088931955, ω=0.280219764 | 77.5 s |
| **office GPU, default matmul precision** | **RuntimeError — hard fail** | **1148 s (19 min)** |
| **office GPU, `JAX_DEFAULT_MATMUL_PRECISION=highest`** | **γ=0.088932239, ω=0.280219972** | **28.7 s** |

Failure text:
```
RuntimeError: certified adaptive eigensolve rejected the dominant eigenpair:
stable=True converged=False residual=0.00337443 tolerance=0.000119209;
refusing to report an uncertified growth rate
```

**Root cause: TF32.** The A4000 is Ampere, so JAX's default f32 matmul precision on GPU is
tensorfloat32 (~10-bit mantissa). That inflates the eigenpair residual to 3.4e-3, **28×
above** the certification floor `certifiable_residual_tolerance = 1000*eps(complex64) =
1.192e-4` (`src/gkx/solvers/linear/adaptive_propagator.py:26`, gate applied at
`krylov.py:577`). The gate is correct; the arithmetic feeding it is not. Forcing full f32
matmuls makes it converge on the first restart — hence also **40× faster**, because the
default run burns all its restarts failing.

GPU (highest) vs laptop CPU: γ rel **3.2e-6**, ω rel **7.4e-7** — f32 parity. **The fix
itself is correct on GPU; it just needs its matmul precision pinned.**

**Recommendation for PR #52:** pin the precision on the Krylov/Arnoldi contractions the
same way PR #43 pinned the conserved-quantity contractions (`jax.lax.dot_general(...,
precision=HIGHEST)` or a `jax.default_matmul_precision("highest")` context around the
certified branch). Without it, the certified default hard-fails on every Ampere-or-newer
GPU — i.e. exactly the machines the plan targets. This also makes PR #44's tf32 audit a
hard dependency of #52, not an independent item.

### Bonus finding: main's *uncertified* krylov is wrong on this config (CPU **and** GPU)

Same command on `main` @ fb559974 (cyclone.toml's own `solver = "krylov"` default):

| | γ | ω | wall |
|---|---|---|---|
| office GPU, default precision | **−0.126120** | 0.227927 | 12.3 s |
| office GPU, `highest` precision | **−0.126120** | 0.227927 | 12.6 s |
| office CPU | **−0.115960** | 0.272345 | 9.1 s |
| correct (certified / time-solver) | **+0.08893** | 0.28022 | — |

main silently returns a **stable** mode (wrong sign) for a genuinely unstable ITG case,
and returns *different* wrong answers on CPU vs GPU. Not a tf32 artifact — `highest`
changes nothing. This is a direct, quantitative justification for PR #52 and worth
putting in the plan as the motivating number.

## 6. Timing snapshot (Cyclone nonlinear, 64×64×24, Nl=4 Nm=8, rk3, dt=0.05)

Warm step cost from the 400-step minus 100-step integrator wall, /300:

| host | 100 steps | 400 steps | **warm ms/step** | cold compile | speedup vs GPU |
|---|---|---|---|---|---|
| **office GPU** (1× A4000) | 24.96 s | 31.04 s | **20.3** | ~22.9 s | 1× |
| office CPU (36 cores) | 57.39 s | 178.31 s | **403.1** | ~17.1 s | 19.9× slower |
| laptop CPU (arm64, jax 0.10.2) | 48.49 s | 151.04 s | **341.8** | ~14.3 s | 16.9× slower |

- End-to-end 100-step wall: **GPU 33.8 s** (first run, incl. import+compile) /
  **27.6 s** on a warmer FS cache, vs laptop CPU 52.6 s and office CPU 59.7 s.
  The plan's "65 s laptop CPU" figure is a bit pessimistic vs today's 52.6 s.
- **Compile dominates short runs on GPU**: 22.9 s of the 25 s integrator wall for 100
  steps is XLA compilation. GPU only wins end-to-end past ~70 steps. Enabling JAX's
  persistent compilation cache would be the single biggest UX win for the one-command
  goal in the plan.
- Module import is only 1.12 s.
- Linear ky scan (5 points, time solver): GPU 25.0 s vs office CPU 103.5 s = **4.1×**.
- Only **one GPU** is used; nothing in these paths shards across the two A4000s.

## 7. Blockers / notes for the plan

1. **PR #52 hard-fails on Ampere GPUs** under default matmul precision (§5). Must pin
   precision before the certified krylov can be the default. Highest-severity item found.
2. **main's krylov returns a wrong-sign growth rate** on cyclone.toml on both CPU and GPU
   (§5 bonus) — the certified branch is not just hygiene, it is fixing a real wrong answer.
3. **Plan's `Wg=0.000406441` reference is stale** vs main; re-baseline to 0.00040669 (§4b).
4. **Python 3.10 on office cannot reach the jax floor.** Any office recipe must bootstrap
   a newer interpreter; the uv route in §3 does it without root or system changes.
5. **GPU compile time (~23 s) dominates short jobs**; persistent compilation cache is the
   obvious follow-up.
6. Not benchmarked (out of scope, fetched and available on office): `fix/phase0-robustness`
   (#50), `feat/wout-cli` (#51).

## 8. Artifacts left on office

- `~/.venvs/{bootstrap,gkx-gpu,gkx-gpu-krylov}` — new, mine.
- `~/gkx-wt/{main,krylov}` — new worktrees off `~/GKX`, clean.
- `~/gkx-runs/{demo,nl,nlcpu,scan,scancpu,krylov,krylov_hp,mainkry,timing,timingcpu}` — outputs.
- `~/GKX` primary checkout: **untouched** (still dirty on its feature branch, as found).
- No commits, no pushes, no system packages, no other users' files touched.
