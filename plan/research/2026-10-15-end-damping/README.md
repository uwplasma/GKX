# End-damping contract (2026-10-15)

Raw records: office `~/gkx-evidence/end-damping/` (results_main.txt, logs/).
`scan.csv`: `scan.py <deck> <ky> <Nl> <Nm> <C|rate> <nperiod> <solver>` on
office GPU, venv gkx-ke (jax 0.11.2, solvax 0.27), tree = origin/main
df2953026 with explicit `[time] damp_ends_rate = C * omega_L`,
`omega_L = max_s v_ts * gradpar / (0.125 * (2 nperiod - 1) * 2 pi)`.
Decks: cyc = examples/01 (adiabatic, s-alpha), ke = examples/05 case_full,
kbm = examples/06 case_full (sparse-direct eigen, `sparse_eig.py`),
tem = benchmarks/cases/tem_linear.toml with the 0.125 layer, kem = ke on the
circular Miller surface.

`gate.txt`: `gate.py` (laptop CPU) reruns the end-mode physics gate at
explicit rates. Nm 24, dt 0.002 runs away for rates <= 30 and decays for
>= 50; Nm 12 (dt inside CFL) decays at every rate; Nm 48 at dt 5e-4 decays at
50-400. `gate_eig.py`: Nm 24, rate 10 dominant eigenvalue -0.0200 - 13.77i
(stable), so the runaway is an over-CFL step (bound 0.0014), not the operator.

Findings
- Adiabatic Cyclone ky 0.3 is insensitive: gamma 0.09825-0.09831 for C 0-300
  at nperiod 2-4. At nperiod 1 it falls with C (0.0923 at 10, 0.0804 at 1000).
- KE ky 0.3 (8,16), nperiod 2: plateau 0.16786-0.16807 for C 3-100; C 0 gives
  1.59, C 1 gives 0.174. nperiod 3 gives ~0.29: not domain-converged at (8,16).
- KBM s-alpha (4,8): plateau C 10-3000, 0.2382-0.2441 (nperiod 2),
  0.2492-0.2502 (nperiod 3). Weak damping (C < 3) is off the plateau.
- Default chosen C = 30: inside every plateau above, nu*dt < 0.3 at the decks'
  CFL steps.
