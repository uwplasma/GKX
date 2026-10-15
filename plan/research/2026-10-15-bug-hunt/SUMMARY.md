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
| KBM case_full (8,24) | see PR | not run (stopped: same cost class) |
| ETG etg_linear (8,8) ky=15 | see PR | not run |

On main, shift-invert failed every deck with `residual=nan`. With the new policy it finds the right mode, but costs 60x the adaptive solve and still misses the gate. KBM and ETG therefore default to adaptive.

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
