# IMEX lane 2026-10-16: electron mirror in the implicit block

Branch `perf/imex-mirror-implicit`. Office A4000 (shared, other jobs at 100% GPU), venv `gkx-latest`, complex64 unless noted. Raw logs and scripts: office `~/gkx-evidence/imex/records/`. Deck: `examples/05_kinetic_electrons/case.toml` (Cyclone KE, beta 1e-4, Nz 16).

## Choice of mirror treatment (dense one-step map, KE Miller ky 0.3, (4, 8), x64)

Stable dt of ARS343 and dominant-eigenvalue error (gamma, omega); gamma = 0.1275.

| implicit part | stable dt | error at stable dt | error at dt 0.2 |
|---|---|---|---|
| S0 (main) | 0.051 | 1e-7 | - |
| S0 + mirror, monolithic | 1.16 | 2e-3 | 7e-6 |
| S0 + electron mirror, monolithic | 0.82 | 9e-4 | - |
| S0 + Laguerre-diagonal mirror only | 0.145 | 3e-6 | - |
| S0, mirror Strang-split (exact or CN half steps) | 0.82 | 6e-2 (omega) | 4e-2 |
| approximate factorisation + GMRES | - | - | 62 iterations to 1e-5 |

Strang splitting costs 4e-3 in omega even at dt 0.05. AF-preconditioned GMRES needs 13-240 iterations, Laguerre-diagonal-preconditioned 8-16. Taken: electron mirror inside the per-chain block Thomas (electron blocks `Nl N` square); ions keep `N x N` blocks.

## Factor residual (complex64), 32x32x16 (4, 8)

| dt | residual | after one refinement |
|---|---|---|
| 0.05 | 8.5e-4 | 5.1e-7 |
| 0.2 | 2.9e-2 | 1.2e-5 |
| 0.4 | 0.13 | 2.2e-4 |
| 0.8 | 0.81 | 4.5e-3 |

Refinement is switched on at build only when the check residual exceeds 5e-3.

## Linear gamma (nonlinear driver, bracket off, 16x16x16, (4, 8), fit of Wphi over t in [10, 15])

| deck | rk4 + CFL | imex 0.05 | imex 0.2 | imex 0.4 |
|---|---|---|---|---|
| KE (05) | 0.686881 | 0.687039 (2.3e-4) | 0.687128 (3.6e-4) | 0.690249 (4.9e-3) |
| KBM (06, beta 0.015) | 0.643915 | 0.643903 (1.9e-5) | 0.643869 (7.1e-5) | 0.643712 (3.2e-4) |

KE differences at 0.05 equal those at 0.2 to 1e-4: the fit (several modes, coarse samples) limits that column, not the step.

## Nonlinear, fixed dt, to t = 150 (32x32) or 60 (64x64), (4, 8). Wall includes setup and compile

| grid | code | dt | finite | wall s | Q_i, Q_e (late half, mean +- std) |
|---|---|---|---|---|---|
| 32 | main | 0.05 | yes | 218, 192 | 490 +- 72, 669; 481 +- 62, 665 |
| 32 | branch | 0.1 | yes | 221, 191 | 491 +- 61, 680; 460 +- 67, 648 |
| 32 | branch, refinement forced | 0.1 | yes | 280 | 501 +- 67, 683 |
| 32 | branch | 0.2 | no, t = 24.2 | | |
| 32 | branch | 0.4 | no, t = 18.4 | | |
| 64 | main | 0.035 | yes | 292 | 235 +- 86, 290 |
| 64 | main | 0.05 | no, t = 15.05 | | |
| 64 | branch | 0.07 | yes | 315 | 258 +- 76, 310 |
| 64 | branch | 0.1 | no, t = 15.6 | | |

Interleaved A/B on one GPU (main, branch, main, branch).

## Memory, 64x64x16, (8, 16), dt 0.1

main: OOM while factoring (`_thomas_factor`). Branch: factor 7.1 GB, peak 10.0 GB (build plus check). A full run still does not start: the scan captures the factor as 7.09 GB of compile-time constants and the process dies after lowering, with no Python error. Passing the factor as a scan operand is the fix; not done here. Changes that got there: probed blocks kept on the host, probe batch sized by the probed state, the mirror applied from its probe in the S probes, no stacked copy of the Schur inverses.

## Reading

The mirror limit is gone. dt doubles, and the bracket now sets it: 0.1 at 32x32 and 0.07 at 64x64 against 0.05 and 0.035. The electron blocks cost about as much per step as the doubled dt saves, so time to solution at fixed dt is unchanged (191-221 s against 192-218 s). The gain has to come from adaptive dt: in the linear phase the step map allows dt ~ 1, but a fixed factor cannot follow the bracket CFL once turbulence saturates. That is the next step, and it is not in this branch.

## Adaptive dt (branch `perf/imex-adaptive`, PR #373), 2026-10-18

Office A4000 (one GPU, other jobs idle), `gkx-latest`, complex64, Cyclone KE deck, (4, 8), `fixed_dt = false`. Records: office `~/gkx-evidence/imex/` (`q6.log`, `q7.log`, `q8.log`, `nl.py`). Wall includes setup and compile. Q = late-half mean +- std (ion, electron).

| grid, t_end | code | wall s | steps | Q_i | Q_e |
|---|---|---|---|---|---|
| 32, 150 | RK4 + CFL | 1181 | 10622 | 484 +- 73 | 667 +- 93 |
| 32, 150 | #370, ARS3 fixed 0.1 | 287 | 301 | 494 +- 66 | 689 +- 102 |
| 32, 150 | adaptive ARK3(2) | 204 | 316 | 485 +- 62 | 674 +- 97 |
| 64, 100 | RK4 + CFL | 2080 | 8328 | 257 +- 99 | 314 +- 91 |
| 64, 100 | #370, ARS3 fixed 0.07 | 291 | 287 | 255 +- 65 | 316 +- 70 |
| 64, 100 | adaptive ARK3(2) | 447 | 280 | 260 +- 61 | 314 +- 56 |

Speed-up over RK4 + CFL: 5.8x at 32, 4.65x at 64 (target 5x). At 64 the adaptive run is 1.5x slower than #370 at the same mean dt (0.071): the extra 150 s is control setup (Arnoldi, probe compile) and factor rebuilds, not steps.

Control: dt = min(5 cfl / (w_nl/1.7 + rho/2.5), dt_err, dt_max). With the CFL constant 1 the bracket (w_nl ~ 80 in saturation) held dt to 0.019 (665 s at 32); the constant 5 is the measured saturated limit (fixed 0.1 stable, 0.2 not). Embedded-error tolerance 1e-2 per step.

64x64x16, (8, 16): the factor is now a scan operand (no 7 GB constants). RK4 + CFL to t = 30: 3144 s. Adaptive ARK3: setup 400 s, then 22 s per 32-step chunk, but a factor rebuild goes out of memory on the 16 GB card (BFC fragmentation near the 10 GB build peak; the old factor is freed first, in use stays 8.0-8.4 GB). With the platform allocator the run did not finish in 57 min. Open.
