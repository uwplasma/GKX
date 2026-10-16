# Bug hunt 2026-10-15

Office host, shared RTX A4000 (other jobs at 100% utilisation), `gkx-latest` venv (jax 0.11.2, SOLVAX 0.28.2), float32 unless noted. Raw logs are on office under `gkx-evidence/bug-hunt/`.

## Warm imex wall time, before and after #363

KE tutorial deck (`examples/05_kinetic_electrons/case.toml`), 32x32, (Nl,Nm)=(4,8), imex-ars3, fixed dt, 20 steps. Each line is three solves in one process, runs A/B/A/B.

| src | cold s | warm s |
|---|---|---|
| main df29530 | 71.4, 74.4 | 8.93, 9.45, 9.71, 9.48 |
| #363 | 74.7, 66.9 | 3.13, 2.81, 3.09, 2.91 |

Warm solves are 3.2x faster. The final Wg differs at the 1e-11 level between repeats of the same source too, so this deck is not bitwise-repeatable on the GPU.

## Shift-invert under the new policy (auto = pr3-cm, restart 150, maxiter 600, adaptive seed)

| deck | adaptive | shift-invert |
|---|---|---|
| Cyclone AE case_full (8,24) | 0.098776-0.276905j, res 3.8e-6, 36 s | 0.098774-0.276932j, outer res 1.6e-4 > gate 1.2e-4, inner 48/48 unconverged (0.873), 2246 s |
| KBM case_full (8,24) | 0.183+0.189j, res 4.4e-4, **not certified**, 1936 s (wrong branch: reference omega 1.08) | not run |
| ETG etg_linear (8,8) ky=15 | did not finish in 2400 s | not run |

On main, shift-invert failed every deck with `residual=nan`. With the new policy it finds the right mode, but costs 60x the adaptive solve and still misses the gate. KBM and ETG therefore default to adaptive. It fails closed instead of returning NaN, but at production resolution it does not yet certify KBM (8,24), and ETG did not finish. The fast route for those decks is still open (time solver, or a working preconditioned shift-invert).

## Review findings fixed in this PR

| # | where | bug | test |
|---|---|---|---|
| 1 | workflows/linear.py | `propagator_steps`, `shift_precond_alpha`, `shift_precond_block_solve` dropped between KrylovConfig and the solver | suite |
| 2 | solvers_linear_precond_pr3.py | complex128 requested under float32 (truncation warnings) | suite |
| 3 | runtime/commands.py, nonlinear.py | a `--dt` override left the fixed-dt step count at t_max/deck dt, so the run stopped early | test_cli step inference |
| 4 | diagnostics/growth_rates.py | a start_fraction too late for any window silently returned the earliest window | test_analysis |
| 5 | growth_rates.py | r2_phase = -inf for a zero-frequency mode in the reporting path | test_analysis |
| 6 | growth_rates.py | the auto fit returned (0,0,0,0) for an all-NaN trace, so it read as a neutral mode | test_analysis |
| 7 | runtime/startup.py | KBM/ETG runtime defaults were shift-invert, which never certified | test_cli |

## Reported, not fixed

- `[physics] tau_fac`, `z_ion`, `[expert] dealias_kz`, and the `KineticElectronModelConfig` fields `nu_e` and `nu_i` are parsed but read nowhere.
- With `window_method="stationary"` (the default), `start_fraction`, `end_fraction`, `require_positive`, `min_amp_fraction` and `min_r2` are ignored.
- Adaptive dt CFL: for a ky subset, `ky_max` is taken by index into the subset (`solvers_time_explicit_cfl.py:263`), which can underestimate it about 10x.

## Policy runs at the PR head (2026-10-16, office, gkx-latest, float32, CUDA 1 shared)

Command for each row: `prof_lin.py --toml <deck> --opts '{"solver":..., "Nl":..,"Nm":..,"ky_target":..}'` (from the 10-13 profiler, with status output added).

| deck | (Nl,Nm) | route | eigenvalue (gamma - i omega) | residual / gate | wall s |
|---|---|---|---|---|---|
| Cyclone AE case_full, ky 0.3 | 8,24 | adaptive | 0.0987759-0.276905j | 3.8e-6 / 1.2e-4 | 42.9 |
| | | shift-invert (pr3-cm auto) | 0.0988028-0.276852j | 3.6e-4, 48/48 inner unconverged | ~2250 |
| | | time, 4000 steps | gamma 0.0962, omega 0.2880 (transient) | - | 22.9 |
| Miller KE (cyclone_miller_quasilinear, kinetic e), ky 0.3 | 4,8 | adaptive | 0.296078-0.261071j | 2.3e-4, refused | - |
| | | shift-invert | stopped after 40 min | - | - |
| | | time, 40000 steps | non-finite fields at the deck dt | - | - |
| KBM case_full, ky 0.3 | 4,8 | adaptive | 0.155051-1.20777j | 1.15e-4 / 1.2e-4 | 409.3 |
| | | shift-invert | -0.0356+0.0481j (wrong branch) | 0.999, 48/48 inner unconverged | - |
| | | time, 8000 steps | gamma 0.842, omega 1.152 (t = 8, transient) | - | 18.3 |
| ETG etg_linear, ky 15 | 4,8 | adaptive | did not finish in 40 min | - | - |

Policy and literature: docs/algorithms.rst, "Eigensolver policy".
