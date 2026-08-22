# GKX research-grade roadmap

This branch is the living plan and audit log. It is intentionally not part of
`main`. The previous long-form log remains in Git history; this file records the
current evidence, open defects, and next decisions.

Last reconciled: 2026-08-22 against `main` at `0ff569c3` (merged PR #81)
and 29 open PRs: #74, #82--#105, and #107--#110. PR #106 was closed as a
duplicate. The post-merge `main` workflow passes.

## Rules

- Do not merge this PR. Update `plan/research-grade-roadmap` as evidence changes.
- One physics claim requires an equation, a reproducible command, raw data, and a
  quantitative gate. A plot alone is not validation.
- Performance claims require cold and warm wall time, peak memory, hardware,
  precision, JAX version, resolution, and an accuracy-matched comparator.
- All commits made from Rogerio's machines use
  `Rogerio Jorge <rogerio.jorge@ist.utl.pt>` with no AI co-author trailer.
- Keep source changes small. Source files, lines, runtime, and peak memory should
  decrease unless a measured capability requires otherwise.

## Current verdict

GKX has a broad JAX gyrokinetic implementation and a useful finite-window
nonlinear derivative. It is not yet ready to claim statistically resolved
stellarator transport optimization. The immediate blockers are output truth,
saturation-window selection, perpendicular resolution, physical rendering, and
independent replicated validation.

| Area | Implemented | Independently established | Public status |
| --- | --- | --- | --- |
| Hermite--Laguerre electrostatic GK | streaming, drifts, mirror, drive, fields, collisions, nonlinear bracket | many unit identities; selected linear/collision benchmarks | usable, validation matrix incomplete |
| Electromagnetic GK | `A_parallel`, `B_parallel`, finite-beta paths | selected derivative/device gates | experimental outside tested lanes |
| VMEC/Boozer geometry | imported and differentiable paths | selected finite-beta and state-control checks | main plot is synthetic; physical renderer is in PR #86 |
| Linear derivatives | implicit eigenpair VJP with residual gates | AD/FD tests and selected GX comparisons | promoted on certified simple branches |
| Nonlinear derivative | checkpointed discrete adjoint of a finite post-saturation window | reduced AD/FD and device-parity tests | local finite-window derivative only |
| QA transport reduction | optimization scripts and preliminary paired runs | source-pinned QA Ny=96/128/160 seed-31 rung shows a resolution-dependent stop; only Ny=160 clears the cutoff screen, and it fails persistence | 12.26% is preliminary, not statistically resolved |
| Saturation stopping | IAT-corrected SEM and stationarity guards | implementation tests | production policy has defects below |
| CPU/GPU | JAX CPU/GPU and experimental sharding | selected kernels/cases | no general scaling or GX-competitive claim |

## Model and numerical contract

The evolved gyrocentre perturbation is

\[
 \delta f_s=F_{Ms}\sum_{\ell=0}^{N_\ell-1}
 \sum_{m=0}^{N_m-1}G^{\ell m}_s
 L_\ell(\mu B/T_s)\psi_m(v_\parallel/v_{ts}),
\]

stored as `G[species, laguerre, hermite, ky, kx, z]`. The abstract discrete
system is

\[
 \dot G = (L_{\parallel}+L_d+L_\mu+L_*+C)G
          + \mathcal N(G,\phi,A_\parallel,B_\parallel),
 \qquad \mathcal F(G,\phi,A_\parallel,B_\parallel)=0.
\]

Hermite streaming is nearest-neighbour in `m`; magnetic and FLR terms are sparse
or structured in `(m,ell)`; the nonlinear `E x B` bracket is pseudo-spectral in
`(kx,ky)`. GKX already uses SOLVAX's batched tridiagonal solve in opt-in
Hermite-line and linked/coarse preconditioners for implicit matrix-free GMRES;
the default `auto` preconditioner remains diagonal. That structured solve
approximates selected linear terms and does not invert the nonlinear Fourier
convolution. Promotion requires residual- and accuracy-matched iteration,
wall-time, memory, and VJP evidence on CPU and GPU.

The drive inputs are

\[
  \mathrm{tprim}=a/L_T=-a\,\partial_r\ln T,\qquad
  \mathrm{fprim}=a/L_n=-a\,\partial_r\ln n.
\]

`adiabatic_electrons=true` closes the electron density with a Boltzmann response;
otherwise a kinetic electron species must be evolved.

### One nonlinear derivative

Production exposes one method: a checkpointed discrete adjoint of a finite
window started from a separately saturated state,

\[
 G_{k+1}=\Phi_k(G_k,p),\qquad
 J(p)=N_w^{-1}\sum_{k\in W}Q(G_k,p),\qquad
 G_0=\operatorname{stop\_gradient}(G_{sat}).
\]

The reverse recursion is

\[
 \lambda_k=\Phi_{G,k}^{T}\lambda_{k+1}+N_w^{-1}Q_{G,k}^{T},
 \quad
 \frac{dJ}{dp}=N_w^{-1}\sum_{k\in W}Q_{p,k}
 +\sum_k\lambda_{k+1}^{T}\Phi_{p,k}.
\]

Block rematerialization with block length near `sqrt(N)` targets
`O(sqrt(N))` state storage. This differentiates the executed finite discrete
trajectory. It is not an infinite-time turbulent sensitivity. Root adjoints are
not applicable to a chaotic saturated trajectory; the discarded shadowing
prototypes did not pass the existing sign/conditioning tests.

## P0 defects found in the 2026-08-21 audit

### R1 — adaptive chunks violate output time truth

`run_adaptive_runtime_chunk_loop` advances a full 1,024 accepted steps before
checking either saturation or `t_max`. It then truncates the diagnostic arrays
without rolling back the state. The inspected QA run advanced its fourth chunk
toward about `t=238`, wrote diagnostics only through `t=200.620`, and recorded
the saturation window through `t=238.025`. The restart/final field therefore can
represent a later state than its label.

Required gate:

\[
 t_{state}=t_{field}=t_{restart}=t_{diag,last},\qquad
 t_{window,max}\le t_{diag,last}\le t_{max}+\epsilon_{step}.
\]

PR #84 implements the contract: it caps the terminal accepted step, keeps the
terminal diagnostic under output striding, rejects horizon overshoot, and checks
every 128 steps. The interval remains provisional pending the PERF-1 campaign.

### R2 — the present saturation interval includes nonlinear overshoot

For the inspected QA case:

- reported window: `29.834 <= t <= 238.025`, while saved data end at `200.620`;
- `Q = 12.06 +/- 1.22`, relative SEM 10.1%, so the run did not saturate;
- `t=50--55` contains only two stored samples and gives `Q ~= 8.18`;
- `mean(Q,50:200) ~= 10.81`, `mean(Q,100:200) ~= 11.10`;
- `Wphi` is temporarily flat near 50, but `Wg` is still evolving.

Therefore `t=50--55` is not enough. `Wphi` is electrostatic field energy and is
not the transport objective. `Wg` is gyrocentre/distribution free energy; its
drift is a useful stationarity guard but not a substitute for heat-flux
uncertainty.

The production estimator must choose a stationary suffix and report

\[
 \bar Q=N^{-1}\sum Q_i,\quad
 \tau_{int}=\Delta t\left(\tfrac12+\sum_{k=1}^{K}\rho_k\right),\quad
 N_{eff}=T/(2\tau_{int}),\quad
 \operatorname{SEM}_{corr}=s_Q/\sqrt{N_{eff}}.
\]

Acceptance requires a minimum number of independent samples, bounded trend in
`Q`, agreement of adjacent batches, and drift gates for `Wphi` and `Wg`. Compare
IAT and five-correlation-time batch means on the same traces. Scan at most 32
candidate suffixes so selection remains `O(N log N)` or better.

The first held-out audit rejects a heat-flux-only shortcut. The current GKX
rule stops 54/208 traces, but 29 accepted means differ by more than 5% and 7 by
more than 10% from the mean of the final 400 time units. A direct
Oberparleiter-style rule stops earlier and increases the tail error. Conservative
Q-only variants remove the >10% errors only by saving about 13% median runtime.
Fresh runs therefore record both energy guards and spectra before SAT-1 changes
the production default.

The first fresh VMEC pilots confirm that the guards and resolution gate cannot
be collapsed into one scalar. At `32x32x48`, the means over successive
50-time-unit intervals rose from `Q=0.147` to `2.45` and `7.25`; `Wg` rose from
`3.62` to `68.97` and `271.08`. At `48x48x48`, the current production decision
over `t=80.3--250` still reports 17.7% corrected relative SEM. `Wphi` and `Wg`
pass the same half-window stationarity check, but that does not override the
flux uncertainty. Both runs used shared GPUs and are not performance baselines.

The first QHS sizing run shows the converse failure mode. At `64x96x48` and
`t=250`, the median-crossing window begins at `t=33.05`, retains the overshoot,
and fails with `Q=9.108`, 9.64% corrected relative SEM. The direct `t=75--250`
suffix gives `Q=8.352`, 2.38% corrected relative SEM and passes the Q, Wphi,
and Wg half-window checks. This does not promote that hand-selected interval;
it shows why SAT-1 must select and validate a stationary suffix instead of
averaging every point after the first median crossing.

The matched QHS `64x128x48` rung reached `t=250` in 2,192.3 s. Its fixed
`t=150--250` suffix has `Q=6.476` and 2.96% corrected relative SEM, but Q,
Wphi, and Wg all fail the half-window stationarity test; the `Ny=96` value was
`Q=8.382`. The new spectral endpoint passes the necessary screen narrowly
(heat-flux cutoff/peak 9.91%, last-three-bin mass 1.69%), but a 22.7% moving
mean is not a resolution result.

The exact continuations now reach `t=750`. Over `t=500--750`,
`Q=6.113 +/- 0.076` with 1.24% corrected relative SEM; Q, Wphi, and Wg pass
their half-window gates. The independent late suffixes `t=600--750` and
`t=650--750` also pass all three gates. The transient is nevertheless long:
the earlier `t=350--500` suffix changes from `Q=6.580` to `5.994` and Wg from
145.2 to 134.3, so both fail. The `t=600--750` heat-flux cutoff/peak is 9.43%
and the last three positive-ky bins carry 1.78% of magnitude. This establishes
a late Ny=128 state; it does not validate the causal stop rule or perpendicular
convergence. Ny=160 is the controlled next QHS rung.

A causal stationary-suffix shortcut also fails on this trace. Scanning at most
32 suffixes and requiring 5% corrected SEM, `10 tau_ac`, half-window agreement,
and bounded Q/Wphi/Wg regression drift would first stop near `t=56` at
`Q=9.96`; the later level is about 6.5. A 100-time-unit floor delays that false
confidence but is not transferable: on 16 existing VMEC traces with all four
diagnostics, the current rule evaluated sequentially with both energy guards
accepts seven, four differ by more than 5% from the final tail, and one differs
by 10.4%. SAT-1 therefore needs a held-out, causal shadow-stop campaign with
change-point/persistence tests; selecting a favorable suffix after seeing the
endpoint is not an admissible production algorithm.

A bounded replay gives SAT-1 a concrete next hypothesis. At each checkpoint,
test only the trailing 100 time units, require the Q/Wphi/Wg gates above, and
require the complete decision to remain true for 20 additional time units. On
the two traces used to formulate it, this would stop QA at `t=149.7` with
`Q=10.935` (2.7% above the fixed `t=150--250` mean) and QHS at `t=299.2` with
`Q=6.189` (1.3% above the later `t=500--750` mean). Those apparent 40--60%
savings are post-hoc design evidence, not validation. Freeze the rule and score
it without retuning on QA seed 31, QHS Ny=160, QI, and the 16 legacy VMEC
traces before considering a default change.

That prescribed legacy score rejects the rule: it accepts 6/16 traces, with
four errors above 5%, one at 12.5%, and 9.2% median absolute error against each
run's final 100-time-unit mean. A training-only grid over the 16 legacy traces
plus QA seed 22 and QHS seed 22 selects a more conservative second hypothesis:
a 75-time-unit trailing window and 60-time-unit persistence. It accepts only
3/18 training traces, with 4.2% worst error and 29% median saved runtime; the
QA and QHS savings are 22% and 56%. This selection is explicitly exposed to
training bias. It was frozen before QA seed 31, QHS Ny=160, and QI finished.
The source-pinned QA, QHS, and QI scores are recorded below. Only QA Ny=128
stops, while that resolution fails the frozen spectral screen and Ny=160 does
not persist. The rule therefore remains a shadow policy, without retuning.

The first nominal holdout, QHS `64x160x48` seed 22, reached
`t=250.320` in 3,219.4 s. The frozen `75+60` rule makes no stop, so it avoids a
false acceptance but saves no time on this case. Over the declared
`t=150--250` audit window,

\[
 Q=5.6619\pm0.1263\;(2.23\%),\qquad
 \bar Q_1=5.8460,\quad \bar Q_2=5.4793,
\]

and Q fails the two-SEM half-window gate. `Wphi` passes, while `Wg` falls from
130.37 to 124.16 between halves and fails. The heat-flux `ky` cutoff/peak is
4.88%, the last three positive-`ky` bins carry 0.97%, and the corresponding
`Phi2` values are 1.93% and 0.45%; the outer six `kx` bins carry 3.54% of heat
flux. Thus Ny=160 clears the necessary perpendicular-tail screen, but the
drifting transport cannot yet be compared with Ny=128. An exact-state
continuation to absolute `t=500` was then launched; the environment audit below
excludes this nominal campaign from acceptance. The compact trace SHA-256 is
`09d67e8679713aa4d872ed76a0ecabcd8001c8a79f05e46f24e70e98bc16f81e`.

The second nominal holdout, QA `96x160x48` seed 31, reached `t=250.186` in
5,315.5 s. Its `t=150--250` window passes Q/Wphi/Wg stationarity with
`Q=10.9623 +/- 0.2817` (2.57%). Seed 22 gives `10.6374 +/- 0.2761` under the
same calculation: the 3.01% spread is 0.82 combined standard errors, and the
conservative replicated result is `Q=10.7999 +/- 0.2817` (2.61%). The frozen
rule stops at `t=215.947`, saves 13.7% of the fixed horizon, and reports
`Q=10.8424`, 1.09% below the seed-31 final-window mean. Numerically, the first
two nominal scores contain no false stop: one useful QA stop and one QHS
non-stop. QI was not scored before its source-pinned rerun, and no retuning is
allowed.

The seed-31 heat-flux cutoff/peak is 8.05%, its last-three-bin mass is 1.48%,
and the corresponding `Phi2` values are 8.80% and 1.37%; outer-six-`kx` masses
are 0.12% and 0.38%. Both independent Ny=160 seeds pass the necessary spatial
screen and agree statistically. At this stage a matched Ny=128 seed-31 rung was
still needed; the source-pinned pair below supersedes the nominal result. Trace
SHA-256:
`eb8541d337504d94579f60bcc4f3288ee6c9ff7c7fdec5297d14ce9526bfdc6f`.

A subsequent environment audit found that every office campaign above used
the PR #91 tool checkout but imported GKX from the older editable worktree at
`c749abfa`. The intended branch differs in exact-horizon plumbing in three
nonlinear solver modules; its operators and geometry are unchanged, but a
research-grade holdout cannot rely on that inference. These runs remain useful
for sizing and hypothesis rejection, not acceptance. PR #91 commit `f7da8c49`
now fails closed unless the campaign imports its own checkout and embeds the
source path, Git commit, and dirty state in NPZ/state/JSON outputs. QHS Ny=160
and QI restarted from zero with a logged clean `f7da8c49` source; QA seed 31
and its Ny=128 match must also be repeated. The frozen rule is not retuned.

The first source-pinned rerun, QHS `64x160x48` seed 22, reached exactly
`t=250` in 2,906.9 s from the clean `f7da8c49` tree. On `t=150--250`,

\[
 Q=6.5083\pm0.0937\;(1.44\%),\qquad
 \bar Q_1=6.6380,\quad \bar Q_2=6.3795,
\]

and Q, Wphi, and Wg pass the two-SEM half-window gates. The heat-flux
cutoff/peak is 4.79% and the last three positive-`ky` bins carry 0.93% of
spectral magnitude; the Phi2 values are 0.010% and 0.030%. The frozen `75+60`
rule does not stop: admissible windows persist for only 11.7, 6.6, and 5.0
time units, never the required 60. Continue the exact state before comparing
the mean with the late Ny=128 result. Trace SHA-256:
`928289d9e14d585a0dcb70b0b57939d556bb4030f71df6c2098fd4d5d6363911`.
The current production selector instead retains `t=31.84--250`, including the
overshoot, and reports `Q=8.591 +/- 1.481` (17.2%) plus a failing Wg guard.
This is direct evidence that burn-in selection, not an excessively long
stationary average, causes unnecessary continuation on this case; it is not
evidence that an arbitrary short late window is safe.

The exact-state continuation now reaches absolute `t=500` on the same clean
source. The frozen rule still makes no stop. Its `t=245.04--284.85` pass island
lasts 39.81 time units; later islands last 6.28, 22.34, and 7.47, with no pass
after `t=367.76`. Although `t=400--500` has `Q=6.2927 +/- 0.1493` (2.37%) and
Q/Wphi/Wg all pass their half-window gates, the terminal `t=450--500` window
fails all three. The complete-history shipped selector still includes the
overshoot (`t=31.43--500`) and fails its SEM gate at 12.9%. This is direct
low-frequency-modulation evidence against both the current burn-in and an
unpersisted late-window shortcut. Perpendicular resolution remains adequate:
over `t=400--500`, heat-flux cutoff/peak is 5.28%, last-three-bin mass is 0.94%,
and the Phi2 values are 0.0094%/0.028%. Continue to `t=750` without retuning.
Continuation-trace SHA-256:
`4c757301c2b3aa289e73f82a18d956ee977101017c2667a2d4ebeb5a6edc5407`.

The same clean state now reaches exact `t=750` after another 2,841.2 s. The
combined 1,955-sample trace still gives the frozen `75+60` rule no stop. Every
new pass island is short; the longest after `t=500` is 14.28 time units, far
below the predeclared 60. The complete-history production selector also fails:
it retains `t=31.43--750`, including the overshoot, and reports
`Q=6.9979 +/- 0.6603` (9.44%). Conversely, the segment-local report declares
the continuation saturated by averaging `t=501.22--750`; that label has not
evaluated the prior history and is not a causal whole-run decision.

The terminal `t=650--750` audit is stationary:

\[
 Q=6.1947\pm0.0768,\qquad
 W_\phi=1.9000\pm0.0216,\qquad
 W_g=136.16\pm1.70,
\]

and all three half-window gates pass. The immediately shorter `t=675--750`
and `t=700--750` windows fail again, so this endpoint is evidence of resolved
low-frequency modulation, not permission to shorten the averaging rule. The
same `t=650--750` heat-flux cutoff/peak is 6.46%, last-three-bin mass 0.94%,
and outer-six-`kx` mass 3.23%; the Phi2 values are 2.70%, 0.46%, and 0.63%.
Ny=160 therefore remains spectrally adequate. A clean-source Ny=128 `t=750`
match is evaluated below. Trace SHA-256:
`e3d1152ce4b679cdb32119a7c59ad7452a51dc024cd2fab9f68391850e05d2c9`.

The matched source-pinned QHS `64x128x48`, seed-22 run reached exact `t=750`
in 6,224.6 s with 1,740 samples. The frozen rule again makes no stop: 14 pass
islands last at most 48.39 time units. The terminal `t=675.38--750` decision
passes only at the final sample, so its persistence is 0.044 rather than the
fixed 60. Over the matched `t=650--750` suffix,

\[
 Q_{128}=6.1130\pm0.1260,\qquad Q_{160}=6.1947\pm0.0768.
\]

The Ny=128 value is 1.32% lower, only 0.55 combined SEM, but Q, Wphi, and Wg
all fail its half-window stationarity gates. Its heat-flux cutoff/peak,
last-three-`ky` mass, and outer-six-`kx` mass are 10.57%, 1.78%, and 3.20%,
versus 6.46%, 0.94%, and 3.23% at Ny=160. Thus the means are statistically
compatible while Ny=128 narrowly fails the necessary spectral screen and is
not a valid acceptance rung. Ny=160 remains the minimum accepted QHS spatial
resolution; no causal saturated QHS transport value is promoted. SHA-256:
NPZ `8b32d886d7ae75d01fa4dc290ac2410d5d15eb4ed109547428ae2f8d615f0785`,
JSON `a698489ae24df0114b228a090758b850cf0d9ccd1e76cbaed7537834c720fd29`,
log `db3870bcce370ad9ce8e99249a98bbc44a4936030fdc085fd8de73fbe75af92d`.

The untouched Ny=160 seed-31 replicate then reached exact `t=750` in
8,469.3 s. The frozen rule again makes no stop: 11 pass islands last at most
28.55 time units. On the matched `t=650--750` suffix,

\[
 Q_{22}=6.1947\pm0.0768,\qquad Q_{31}=5.2766\pm0.0714.
\]

Seed 31 is 14.82% lower, separated by 8.76 combined SEM. Its heat-flux
cutoff/peak, last-three-`ky` mass, and outer-six-`kx` mass pass at 7.10%,
1.02%, and 3.77%; this is not a perpendicular-resolution failure. Initial
`Wphi` and `Wg` agree between seeds within 0.23%, while the late zonal-`Phi2`
fractions differ, 94.76% versus 88.42%. The terminal 75-unit seed-31 window
passes the three half-window checks but covers only 7.82 of its longest guard
autocorrelation time, below the frozen ten-IAT gate. QHS is therefore
spectrally resolved at Ny=160 but not seed-robust or causally saturated. Do not
average the two seeds into a promoted value; extend independent states and
resolve the slow zonal-regime dependence first. SHA-256: NPZ
`64d51d485da74e96949f07a2efdb2b165d8b42d5ca0bcd0bc9c7d742ff057176`,
JSON `742f2c01883c3a27ac8d9080c23d37cc8f2337df6f2adef382794d24ecc36996`,
log `910b6a4b407adf09de086ade7e0c812690e55c35fe032a34860a72a0211dc579`,
state `914048a6201ba20d7b097c4b2af9839fa6e733b56a96c48c2b46fbf8f176d6ce`,
and bound replay `852d2da7547f93b631013715cb787765162d2e459a53cadcf5bc7ddda2734f65`.

The untouched source-pinned QI `96x96x48` seed-22 holdout reached exactly
`t=250` in 3,462.3 s from the same clean solver tree. Over `t=150--250`,
`Q=4.1641 +/- 0.0922` (2.21%) and Q/Wphi/Wg pass the half-window gates, but
the shorter `t=200--250` suffix fails Q and Wg stationarity. The frozen rule
makes no stop: its longest admissible interval is `t=187.88--246.72`, only
58.85 time units, after which drift returns. This is a narrowly correct
non-stop under the predeclared 60-unit hold, not permission to shorten the
threshold. Ny=96 also fails the necessary spatial screen: heat-flux
cutoff/peak is 15.50% and the last three bins carry 3.71% of magnitude, while
Phi2 is much better resolved at 0.076%/0.241%. Continue the exact state and
add a matched Ny refinement. Trace SHA-256:
`d7b511db5065e405f2a7511e0f335a8bc9b3cc12dcdf47af2fa9c4166e34ea55`.

The matched source-pinned QI `96x128x48`, seed-22 rung reached exact `t=250`
in 4,959.3 s. The frozen rule again makes no stop: its only late pass island is
`t=235.03--250`, 14.97 time units. Over `t=175--250`, both resolutions pass
Q/Wphi/Wg half-window stationarity, but Ny=96 and Ny=128 give
`Q=4.1379 +/- 0.1213` and `3.5087 +/- 0.0446`. The 15.21% reduction is 4.87
combined SEM, so Ny=96 transport was not converged. Raising `ky*rho_i|max`
from 1.476 to 2.0 lowers heat-flux cutoff/peak from 14.80% to 4.20% and
last-three-ky mass from 3.71% to 0.95%; Phi2 is already small at both
resolutions. Ny=128 clears the necessary spectral screen, but `t=200--250`
fails every Ny=128 half-window guard. SHA-256: NPZ
`9d8f3062389780536ce4df7497b03905b2038ad2f64d75895f7891afaedbc52b`, JSON
`1724347485b1996cc4d0ec1f66e68473d1f6c601cebf0e3bfe6ba4289bd60933`, log
`730c377f3b57fdf57ad96c45e3d98ef57487e44173d72f36696b617dab298168`.

The exact Ny=128 continuation now reaches absolute `t=500` after another
4,902.5 s. The combined 1,925-sample history still makes no frozen stop: its
six pass islands last 11.51, 19.45, 14.13, 0.52, 27.48, and 8.19 time units.
The terminal `t=425.09--500` window passes Q/Wphi/Wg with
`Q=3.4484 +/- 0.0224`, but has not persisted for the predeclared 60 units.
Over the matched `t=400--500` suffix, Ny=96 and Ny=128 give
`Q=4.1869 +/- 0.0822` and `3.4194 +/- 0.0333`: Ny=96 is 18.33% higher, or
8.65 combined SEM. Ny=128 retains its spectral pass (4.38% cutoff/peak, 0.97%
last-three-ky mass, and 3.19% outer-six-kx mass). QI therefore rejects Ny=96
for transport while withholding a causal saturation claim. Continuation
SHA-256: NPZ
`46627a0d9e5256119a80debd0827864481d033c844ecc64b173f3d188689285b`, JSON
`ef89e389591c1e286754bd4708707eb46dca408571d6c19523a9cd24f5d5fa9e`, log
`01f66c52ae2c5e97bb5732570489a4d387c9f4af001f057fbc4f71ff30d9438b`.

The earlier environment-contaminated QA `96x128x48` rung completed at `t=250`
in 3,648.2 s. On the
fixed audit suffix `t=150--250`, `Q=11.006` with 2.21% corrected relative SEM;
Q, Wphi, and Wg pass the half-window checks. The matched `Ny=96` value is
`Q=10.860`, a 1.35% change. This apparent compatibility is not a spectral
pass: 16.3% of the `Ny=128` heat-flux spectral magnitude lies above
the old cutoff and the new cutoff remains 18.2% of the peak. The matched
`96x160x48` seed-22 rung completed in 5,327.5 s. Its fixed `t=150--250` suffix
has `Q=10.643 +/- 0.278` (2.61% corrected relative SEM), with Q/Wphi/Wg all
stationary. The change from Ny=128 is -3.30%, only 0.98 combined SEM. Its
heat-flux cutoff/peak falls to 7.59%, the last three bins carry 1.50%, and the
outer six signed kx shells carry 0.12%. The completed independent seed-31 rung
agrees statistically and passes the same tail screen. Because the imported
source was not the recorded checkout, all values in this paragraph are sizing
evidence only; the source-pinned seed-31 pair below supersedes them.

That clean-source QA `96x128x48`, seed-31 rung now reaches exact `t=250` from
commit `f7da8c49` in 3,397.1 s. The fixed `t=150--250` result is
`Q=10.8522 +/- 0.3675` (3.39% corrected SEM); Q/Wphi/Wg pass their half-window
gates. The frozen `75+60` rule ignores a first 49.02-time-unit pass island, then
stops at the first sampled checkpoint `t=230.99`. Its trailing-window
`Q=10.9089` is 0.52% above the fixed final-window mean and saves 7.6% of the
horizon. The current selector instead keeps `t=26.00--250` and narrowly fails
at 5.61% relative SEM. This is a held-out temporal success without threshold
retuning, not a resolution pass: over `t=150--250`, heat-flux cutoff/peak is
15.69% and last-three-bin mass is 3.11%; Phi2 is better at 1.13%/1.45%, and
outer-six-kx heat mass is 0.11%. Trace SHA-256:
`91f6768849caa8a315594d7f3bac256cc46a9203829ae0affae65d20ab6b5c45`.

The matched source-pinned Ny=160 seed-31 rung reached exact `t=250` in
5,108.6 s. The frozen rule still makes no stop: its terminal pass island is
only `t=220.54--249.67`, not the required 60 time units. On the Ny=128 stop
window `t=155.844--230.844`, Ny=128 and Ny=160 give
`Q=10.9140 +/- 0.4563` and `11.0353 +/- 0.2539`; the 1.11% difference is only
0.23 combined SEM and Q/Wphi/Wg pass at both resolutions. The terminal
`t=175--250` comparison differs by 5.51% (1.14 combined SEM), so this is
compatibility, not convergence. Ny=160 lowers heat-flux cutoff/peak from
13.87% to 7.55% and last-three-ky mass from 3.04% to 1.45%; outer-six-kx mass
remains 0.11%. It clears the frozen cutoff screen, while time persistence and
an independent seed remain open. Ny=160 SHA-256: NPZ
`4564fd5f9d50f441623040c9329bf22e498f4d7fbadf4ef25abbf195842ad185`, JSON
`05af33be9bf3f4645750c42e32a32db0ab754d3a0c161537c396066c8f6784e5`, log
`c1d0fca85be22b465b1d20e41f4cb1cdb0e06ce6152155e19ef98940f88ce49b`.

The exact seed-31 state now continues to absolute `t=350` in 2,100.3 s. The
joined 1,066-sample history still makes no frozen stop: seven pass islands last
at most 29.13 time units, and the terminal island lasts only 6.31. Over
`t=250--350`, `Q=10.9902 +/- 0.3072` and Q/Wphi/Wg pass their half-window
checks; heat cutoff/peak is 9.98%, last-three-`ky` mass 1.51%, and outer-six-
`kx` mass 0.12%. The shipped full-history selector now passes by averaging
`t=24.95--350`, with `Q=11.3449 +/- 0.3128`; this explains current runtime
behavior but does not satisfy the frozen causal-persistence hold. Continuation
SHA-256: NPZ
`d16582748f84080d3a86137657d00fb9846f479103956896bfee0032c1058c62`, JSON
`45917775003d9d7de3ae107333e85f55378a842cb5bc45a303480dce69f646ee`, log
`bf70889c3da4865114a31e7e49bb7d12c085ff9a02926e2e2ea42d14aaa1f5c1`.

The independent source-pinned Ny=160 seed-22 run reached exact `t=250` in
5,007.6 s. Its frozen rule also makes no stop: three pass islands last 42.23,
18.60, and 24.57 time units, while terminal Wphi and Wg drift. On the matched
`t=150--250` window, seed 22 and seed 31 give

\[
 Q_{22}=10.4833\pm0.1734,\qquad Q_{31}=11.2606\pm0.4042.
\]

The -6.90% difference is 1.77 combined SEM, so the means are not statistically
separated at two SEM. The spatial gate is less robust: heat cutoff/peak is
12.24% for seed 22 versus 9.46% for seed 31, crossing the fixed 10% screen;
last-three-`ky` mass is 1.52% versus 1.49%. Ny=160 is therefore not yet a
seed-robust accepted QA rung. SHA-256: NPZ
`0f31911b8723c6f006d0c3cc921f6958507fed6f9e4df04986071e45c73d06ea`, JSON
`14715807b6faae2c122f1d17632f5a744bea2d81b4342a26a2b01099140e3d5a`, log
`6747c8f51827b4454d932037956354ea3d5514df8f7cbff14568ecb855d29807`.

Its exact state now continues to absolute `t=350` in another 2,065.1 s. The
frozen rule stops causally at `t=312.194`: the `t=237.436--312.194` window has
`Q=10.5986 +/- 0.2300`, all Q/Wphi/Wg gates pass, and the pass island has
persisted for 60.16 time units. That island lasts until `t=345.839`, but the
terminal window fails again by `t=350`; a stop decision cannot see that future
failure. Seed 31 still makes no stop through the same horizon, so this held-out
success does not establish a seed-robust fast rule.

On the matched `t=250--350` suffix, seed 22 and seed 31 give
`Q=10.7420 +/- 0.2200` and `10.9902 +/- 0.3072`, a -2.26% difference or only
0.66 combined SEM. Both spatial screens now pass: heat cutoff/peak is 8.77%
and 9.98%, last-three-`ky` mass 1.55% and 1.51%, and outer-six-`kx` mass is
0.12% for both. The two seeds therefore support a common long-window Ny=160
mean, but not a common causal stop. The Ny=192 seed-31 resolution rung and the
Ny=160 seed-31 `cfl=0.5` control were then run independently on separate GPUs.
Continuation SHA-256: NPZ
`4954bac97ed8cad9f93093d6c7b13c11fc1635734b795058169303adbdb82385`, JSON
`29b96bf8d1fc0d9e9b6a6fdbe42da0cf2c5b3763ff7ab28f2883d511d4167f42`, log
`b9f56670afdf3f05e89509bba93d415063c9dae0332b5ed4f50bba491bc29ed9`.

The untouched Ny=160 seed-33 holdout then reached exact `t=350` in 7,106.8 s
with 1,055 samples. The standard `75+60` replay makes no stop; its longest pass
island is 48.59 time units. The preregistered `125+30` hypothesis would stop at
`t=184.186` on `t=59.330--184.186`, with

\[
 Q_{stop}=10.8984\pm0.2445,
\]

and passing Q/Wphi/Wg autocorrelation and half-window gates. Its heat-tail ratio
is 9.924%, just below the fixed 10% limit; last-three-`ky` and outer-six-`kx`
heat masses are 1.531% and 0.118%. The unseen terminal `t=250.181--350` mean is

\[
 Q_{33}=10.7489\pm0.4246.
\]

The stop differs by only 1.39%, or 0.31 combined SEM, and the terminal heat
spectrum passes at 9.237% tail, 1.540% last-three-`ky` mass, and 0.117%
outer-six-`kx` mass. Criterion 5 nevertheless fails: terminal Wphi falls from
1.6345 to 1.5021 between halves and Wg from 235.05 to 212.11, so both are
nonstationary even though Q passes. Seed 33 agrees with seed 22 to 0.015
combined SEM and seed 31 to 0.46 SEM on the same terminal window. Thus three
seeds support the long-window transport mean, but the held-out shortcut is
rejected exactly as preregistered and is not retuned. NPZ/JSON/log SHA-256:
`57a9736b744280b869fd642af945a1e18e9baadd0889729b6225baea7a2d2436`,
`e891acdcc69a566d942c0305a3e74d6af8043d78e4824f5b01723cee596d0e86`,
`35d5c0509bbab0266cd0c607788f27140dbe08b4f9c058aea5c01c27fe5dfb97`;
preregistered replay `c2fb94a9`, standard replay `f152057c`.

The concurrent QI audit exposed a separate orchestration fault: a second
same-name campaign could start before the first wrote its final NPZ/JSON. The
younger process truncated the shared log and would later race every output. It
was terminated by exact PID; the older trajectory may size physics, but its log
and wall time are rejected and a clean repeat is required for acceptance. PR
#91 commit `eebff63b` now takes nonblocking advisory locks on every requested
summary, trace, and state path before importing or running GKX. Stale lock files
are harmless because the kernel releases the lock on crash/exit. The 43 focused
campaign/chunk/gradient tests, Ruff, and diff checks pass.

The matched Ny=192 seed-31 rung reached exact `t=250` in 6,594.6 s with 843
samples. Its frozen rule stops at `t=218.783` after 60.17 time units of
persistence, while Ny=160 at the same seed still makes no stop. The causal
decision therefore remains resolution-sensitive. On the predeclared
`t=150--250` audit window,

\[
 Q_{160}=11.2606\pm0.4042,\qquad Q_{192}=11.1314\pm0.2702.
\]

The -1.15% change is only 0.27 combined SEM. Q and Wphi pass the half-window
gate at Ny=192, but Wg does not over the full 100 units; all three pass only on
the terminal 75-unit window. Spatial evidence improves monotonically:
heat-flux cutoff/peak falls from 9.46% to 6.12%, last-three-`ky` mass from
1.49% to 1.06%, Phi2 cutoff/peak from 11.61% to 7.37%, and outer-six-`kx` heat
mass stays small at 0.12%. Ny=160 and Ny=192 thus support a compatible
long-window transport mean and a resolved Ny=192 spectrum, not a promoted
temporal stop or observed-order convergence claim. SHA-256: NPZ
`289c2f30076e4aa4c4bbec0496b1a058db9834e91f649b17cf17c372f95fff73`, JSON
`645b323c7f0df02a161e35cd20abadabb0538887161109be2a2c0cf3cace4904`, log
`e0ce2d906f1517ab1f824fb13f954cc3be3d8479f1c17a3e517fc3a26e44d769`.

The matched seed-31 Ny=160 `cfl=0.5` control reached exact `t=250` in
9,997.4 s with 1,505 samples, versus 5,108.6 s and 756 samples at `cfl=1`.
The median adaptive step falls from 0.03255 to 0.01636 and wall time rises by
1.96x. On the predeclared `t=150--250` window,

\[
 Q_{cfl=1}=11.2606\pm0.4042,\qquad
 Q_{cfl=0.5}=10.6370\pm0.2368.
\]

The -5.54% change is 1.33 combined SEM, so the means are not separated by the
two-SEM gate. Both controls pass Q/Wphi/Wg half-window stationarity. Heat-flux
cutoff/peak is 9.46% versus 9.89%, last-three-`ky` mass 1.49% versus 1.57%,
Phi2 cutoff/peak 11.61% versus 10.77%, and outer-six-`kx` heat mass 0.114%
versus 0.119%. The refined run also makes no frozen stop: its terminal pass
island is `t=190.565--250`, lasting only 59.43 of the required 60 time units.
Timestep refinement therefore supports the late-time mean and spatial screens,
but neither promotes the stop nor justifies paying about twice the runtime by
default. SHA-256: NPZ
`bc3b7b47f39615ee9a5d8c6597805e58ca02ae205fc6080c57171bf98d4021d3`, JSON
`d33d01e845d66b237fb1a240cc841983ac80306cc5e02907d6624bac90c4df50`, log
`fe3de89cf3a3db34f27d0d45d887ab1e9fc52848f2d9e92acf0ca7f299e02024`.
The untouched seed-33 QA holdout is complete. The independent QHS Ny=160
seed-31 horizon run and the original QI Ny=160 seed-22 sizing run are active on
the two GPUs. Because the QI log was collided, its lock-protected clean repeat
is queued from PR #91 commit `eebff63b`; the original trace cannot be acceptance
evidence.

The source-pinned QA `96x96x48`, seed-31 rung then reached exact `t=250` in
2,200.7 s. It also makes no frozen stop; the longest pass island is
`t=193.23--243.55`, only 50.33 time units. On the Ny=128 stop window
`t=155.844--230.844`, Ny=96/128/160 give `Q=11.6622`, `10.9140`, and
`11.0353`; Ny96 differs from Ny128 by 6.42% (1.50 combined SEM) and from Ny160
by 5.38% (1.94 combined SEM). More decisively, its heat-flux cutoff/peak is
41.78% and last-three-ky mass 11.53%, versus 13.87%/3.04% at Ny128 and
7.55%/1.45% at Ny160. The frozen stop is therefore not resolution-stable, and
Ny96 is rejected even where its mean looks statistically compatible. SHA-256:
NPZ `41b8e641896d5830cee03c45103d52851e8d83e35a68c4340d3a7aa902d647cc`,
JSON `6daf27302fd872a3364a6f3d6bbdae27c61dc25be6d0b573cc7cdd8dd3a8df9a`,
log `f2c054a10b053be97eb315d26e284bc0bf9392049d1c3a92f06af7e038548f8f`.

The frozen decision is now reproducible from the compact traces, not an ad-hoc
notebook calculation. PR #91 commit `067788f0` keeps simulation and replay in
one campaign tool:

```bash
PYTHONPATH=src python tools/campaigns/nonlinear_saturated_state.py \
  --replay-trace TRACE.npz --replay-summary SUMMARY.json \
  --replay-window 75 --replay-persistence 60 \
  --replay-rel-sem 0.05 --replay-min-tau-multiples 10 --output REPORT.json
```

The report records ordered trace, summary, and implementation SHA-256 digests,
every causal pass island, and the first persistence-qualified stop. Legacy
multi-segment traces fail closed without one bound summary per segment. The
source-pinned replays give no stop for QA Ny=96, QA Ny=160 seed 31 through
`t=350`, QA Ny=160 seed 22 through `t=250`, the `cfl=0.5` control, either QI
rung, or either QHS rung. QA Ny=128 stops at `t=230.987`, QA Ny=192 at
`t=218.783`, and QA Ny=160 seed 22 only after continuation to `t=312.194`.
These mutually inconsistent decisions are why the rule remains unpromoted.
Report SHA-256:

| trace | replay report |
| --- | --- |
| QA Ny=96 | `0b7957abc291bcd3dac55f32c58535c627097dae7158d29dc31c43c6f623d9ef` |
| QA Ny=128 | `f9db9ee5b0c2a39db2fb46025884f812e938016b8c10bb943b06b4d2af3001a7` |
| QA Ny=160 | `6b5263302b60c2cd4a42e278f94a03866b72e7788b41095af0abf98f739892c8` |
| QA Ny=160 seed 31, t=350 | `5542c67c774afb349393f2682c4c98ac38cb45dace4e56be36f0bf21486e9a06` |
| QA Ny=160 seed 22, t=250 | `78164cafda9f90f17541172b9658776677f9a179647aecf3cee4ca21bacfe5e0` |
| QA Ny=160 seed 22, t=350 | `607b327130c436f037f44fa7761c83f861ec41e2ef6bace42189a44a2d4042b4` |
| QA Ny=192 | `2996d20f0934990fb31de6f8863ebda6189ec0d51fc337e64d3f9d445be76ab5` |
| QA Ny=160 seed 31, `cfl=0.5` | `6ece6ecf61102060b637c13f34a55a7667072e9df025f4cf95e82e39b3b95f53` |
| QI Ny=96 | `3cd1d3e7f1debddfadfbbed6b14ffb2912e465bd73025c6ba59eb022d14996ea` |
| QI Ny=128 | `c8133f3c1de07c430bbef20c4a19629253fe447ca0031d18f9627f5dcbff4ff3` |
| QHS Ny=160 | `8159e162b4cda45fd0d82c16f1c42a89a87326946aaf88d4502cd9aa5b257d5e` |
| QHS Ny=128 | `eb660d5618c3b1298fb78260cd62caad342e76481c43258097ea8c019bc57240` |

These are policy-replay records, not solver outputs or a claim that the shadow
rule is ready to become the default.

Independent review found that the campaign accepted timestep/CFL overrides but
did not put the resolved policy in its machine-readable identity. PR #91 commit
`35aa7d29` introduces schema v2 with `fixed_dt`, requested `dt`, `dt_max`,
`cfl`, and method in JSON, trace, and restart state. A real VMEC write/restart
smoke reaches absolute `t=0.2` and replays both segments; changing only CFL
rejects the restart before integration. The active source-pinned controls stay
on frozen schema-v1 source and retain their exact command and log hashes.

PR #89 applies the Oberparleiter final-drift gate to each raw trajectory rather
than to a signed ensemble mean. The nominal campaign has 44 stationary traces
out of 48; the `dt=0.04`, perpendicular 24, and velocity `(6,12)` refinements
have 13/16, 6/8, and 30/32. No compact campaign output contains resolved spectra,
so the published 12.26% reduction is preliminary. PR #90 makes this fail-closed
in the release contract: promotion requires hashed raw sources, observed time
limits and averaging windows, and explicit stationarity, autocorrelation,
timestep, perpendicular, parallel, velocity, and spectral gates.

The accepted vacuum baseline/candidate boundaries have now been solved
independently with VMEX commit `0362f701` at `ns=101`, `ftol=1e-10`,
and `niter_max=15000`. Both return `ier_flag=0`: baseline/candidate need
1,224/1,209 iterations, have aspect 5.00641/5.00698, and WOUT SHA-256
`323dd3ef`/`58da1b89`. A compact TOML manifest binds the boundary-input
hashes, VMEX commit, residuals, WOUTs, and figure generator. Regenerating the
3D-LCFS/Boozer panel gives SHA-256 `f4cd87ab`, byte-identical to the tracked
documentation asset. The matched `96x160x48`, `t=350`, seed-31 baseline reached
exact `t=350` in 7,429.7 s. The production median-crossing interval starts at
`t=22.67`, includes the transient, and is correctly rejected at 9.97% corrected
relative SEM even though `Wphi` and `Wg` pass their half-window gates. The fixed
terminal 75-unit audit is stationary instead:

\[
 Q_{base}=8.0466\pm0.1739,\qquad \tau_{int}=2.532,
\]

with `Wphi=3.1432 +/- 0.0855`, `Wg=191.22 +/- 2.94`, and all three half-window
gates passing. Its heat-flux cutoff/peak, last-three-`ky` mass, and outer-six-
`kx` mass are 1.96%, 0.375%, and 2.19%; the corresponding `Phi2` `ky` values
are 1.28% and 0.288%. The frozen `75+60` shadow rule would stop causally at
`t=244.341` with `Q=8.4230 +/- 0.2759`; its terminal-75 mean is 4.47% lower.
This one baseline therefore passes the fixed terminal-window and spatial
screens but does not validate the shadow rule across configurations.

An isolated `t=50--55` window from this source-pinned baseline is not an
acceptable shortcut: it has 18 samples over 4.78 time units, shorter than
`10 tau_int=7.88`, with 8.68% corrected relative SEM, and its Q, Wphi, and Wg
half-window gates all fail. Its Q/Wphi/Wg means exceed the stationary
terminal-75 means by 4.46%, 66.5%, and 31.0%. On the full history available at
`t=55`, the production window still has 22.1% relative SEM. Wphi alone cannot
certify a transport mean.

Remote/local SHA-256 agree: JSON `49ca6d7e`, NPZ `205dafd0`, state `050cadea`,
and log `6e399265`; the source-pinned replay is `754f5d85`. The matched candidate
is now active on GPU 1. Its first waiter failed because `pgrep` matched the
waiter's own command line; output locks prevented a duplicate, and the one
candidate was started only after an anchored Python-process check. This remains
a one-seed sizing pair, not a transport-reduction claim.

### R3 — the plotted stellarator tube is a synthetic torus

`gkx.flux_tube_3d` uses

\[
 R=R_0+a\cos\theta,\quad Z=a\sin\theta,\quad
 \zeta=q\theta/N_{fp},
\]

and never evaluates the VMEC surface. `NFP=2` is present in the inspected file;
the image looks like `NFP=1` because the renderer is generic. `aminor=0` also
falls through to a hard-coded aspect ratio. The apparent discontinuity is not
evidence of a solver boundary defect.

Fix: artifact generation, outside the JAX solver PyTree, must read the resolved
VMEC/EIK `Rplot`, `Zplot`, and toroidal coordinate, then map the saved field onto
that centreline. Tests must use a non-axisymmetric boundary and verify
field-line/grid alignment and twist numerically before visual comparison. A
finite linked flux tube need not close in Cartesian space; its spectral
twist-and-shift identification is not a centreline-closure condition.

### R4 — perpendicular spectrum is unresolved

The heat-flux spectrum rises sharply at the largest saved `ky*rho`; the `phi^2`
spectrum is better but also turns up/noises at the tail. Treat this as a failed
resolution gate, not a physical result. Increasing `Ny` increases `ky_max` for
fixed `Ly`, but nonlinear triads and dealiasing require joint `(Nx,Ny)` tests.

PR #87 adds a visible, fail-loud precheck to every ``Q(ky)`` and ``Phi^2(ky)``
plot,

\[
 R_{tail}=\frac{\max_{k_y\in\text{top 10%}}|S(k_y)|}
 {\max_{k_y>0}|S(k_y)|}.
\]

``R_tail >= 0.1`` warns that the cutoff is unresolved. This is a necessary
screen, not an acceptance gate. On the supplied QA bundle over ``t=[100,200]``,
``R_tail=0.65`` for heat flux and ``0.36`` for potential: both fail despite the
smoother-looking potential plot. Acceptance still requires convergence of
integrated ``Q`` and its spectrum over paired ``Nx``/``Ny`` refinements; the
warning threshold must be calibrated against those refinements.

The fresh `48x48x48` pilot extends the positive retained range from
`ky*rho=0.476` to `0.714`, but the last four modes still carry 49.7% of the
late positive-`ky` heat flux and the cutoff mode carries 13.1%. The `32x32x32`
cutoff mode carries 36.5%. At `64x64x48`, the `t=250` run still has 9.68%
corrected relative SEM and the last three retained `ky` bins carry 51.1% of the
absolute positive-`ky` flux. At `96x96x48`, the physical `t=150--250` spectrum
peaks at `ky*rho=1.333`; its `ky*rho=1.476` cutoff is 54.2% of the peak and the
last three bins carry 14.0%. The outer three `kx` shells carry only about 0.1%,
so the controlled next QA rung holds `Nx=96`, `Nz=48`, seed, and horizon fixed
and raises only `Ny`. At `Ny=128`, the late mean changes by 1.35%, but the
new `ky*rho=2.0` endpoint is still 18.2% of the heat-flux peak and the last
three bins carry 2.96% of summed absolute flux. At `Ny=160`, the endpoint is
7.59%, the last three bins carry 1.50%, and the late mean differs from Ny=128
by -3.30% (0.98 combined SEM). Seed 31 is the independent replication gate.
At Ny=160, seed 31 gives `Q=10.962 +/- 0.282`; its 3.01% difference from seed
22 is 0.82 combined standard errors. Its heat-flux cutoff/last-three values are
8.05%/1.48% and its Phi2 values are 8.80%/1.37%. The environment audit makes
these runs sizing evidence only. In the superseding source-pinned seed-31
triple, Ny=96/128/160 give `Q=11.6622/10.9140/11.0353` on the frozen stop
window. Ny128/160 agree within 1.11% (0.23 combined SEM), while heat-flux
cutoff/peak falls from 41.78% to 13.87% to 7.55%. Only Ny160 passes the screen,
and it does not pass persistence; time continuation and another seed remain
required before convergence.

For the first QHS sizing rung (`64x96x48`, `t=150--250`), heat flux peaks at
`ky*rho=0.333`; the `ky*rho=1.476` cutoff is 15.6% of the peak but the last
three bins carry only 3.18% of summed absolute positive-`ky` flux. The Phi2
cutoff is 10.4% and its last three bins carry 2.32%; the outer six signed `kx`
bins carry 2.14% of heat-flux magnitude. This is close to, but does not pass,
the conservative cutoff screen. The controlled `64x128x48` late state passes
narrowly at 9.43% cutoff/peak and 1.78% last-three-bin mass. The
`64x160x48`, `t=150--250` spectrum passes more clearly at 4.88% and 0.97%, but
its Q and Wg windows still drift. Continue the same state before comparing
transport across Ny; if the late value agrees, refine Nx jointly because the
outer-six-`kx` mass rose to 3.54%.

### R5 — CI and review governance

Main's only observed source failure was a mypy error introduced by #80. PR #81
fixed it with a one-line source-neutral change. The PR #97 nonlinear shard also
showed a runner-budget failure after all tests printed 100%: its clean rerun
passed in 14m51s, nine seconds below the old 15-minute limit. PR #81 therefore
gives only `nonlinear-core` 20 minutes while every other quick shard retains 15.
PR #81's own nonlinear shard then passed in 14m22s under that targeted budget;
the extra five minutes cover runner cleanup variance without relaxing any test.
All 41 PR checks and the resulting `main` push passed, but #81 was merged
externally on 2026-08-22 with no approving review. Branch protection nominally
asks for one review, but every merged PR reports `REVIEW_REQUIRED`, no checks are
required, force pushes are allowed, and administrator bypass was used. Require
the aggregate CI check and one non-author approval after the recovery rewrite.

Eighteen open PRs still target #81's `fix/main-ci-mypy` branch: #74, #82--#89,
#93, #98, #100, #101, #105, and #107--#110. That base and `main` have the same
tree, but their merge base is pre-#81 `5f3ab32e` because #81 was squash-merged.
A base-only retarget would therefore make GitHub's three-dot review diff include
#81 again. Keep the old base alive until coordinated replay. For each direct
head, record its old SHA/tree/patch, rebase its own commits from `d910ac56` onto
`0ff569c3`, require byte-identical head trees and intended two-dot/stable patches,
push with `--force-with-lease`, retarget to `main`, and rerun all checks. Replay
stacked PRs in topological order. Do this once during cutover, not as an extra
round of branch churn now.

## Repository recovery and slimming

The user's sub-10-MB target is tracked in two distinct units. The hard clone
gate is the complete Git object database for an ordinary full clone, including
all live heads and tags; that private rehearsal is below 10,000,000 bytes. The
expanded checkout is a separate target and is **not** yet below 10 MB: the
post-#104 stacked tree is 18,904,270 tracked bytes. The installed `src/gkx`
tree is about 3.35 MB. Reaching a sub-10-MB expanded checkout therefore requires
moving more reproducible documentation/evidence payloads to hash-addressed
release assets, not deleting physics, tests, or provenance blindly.

Measured before rewrite:

| Quantity | Value |
| --- | ---: |
| ordinary main+tags clone pack | 135.04 MiB |
| full mirror pack | 273.29 MiB |
| current tracked tree | 45.98 MB |
| current `git archive` gzip | 29.59 MB |
| historical `docs/_static` packed blobs | 264.79 MB |
| historical PNG/PDF/NetCDF blobs | about 262.8 MB |

Removing only `docs/_static` and NetCDF history yields 11.64 MiB but prunes 32
commits under default `git-filter-repo` behaviour. The verified source-complete
dry run instead:

1. retains all 3,358 original main/tag commits and topology with
   `--prune-empty never --prune-degenerate never`;
2. retains complete `src/` history;
3. removes historical blobs for `docs`, `tests`, `tools`, `examples`,
   `benchmarks`, `scripts`, `plan`, and `*.nc`;
4. adds one current snapshot of those auxiliary paths, excluding generated
   `_static` assets and NetCDF data;
5. stores large reproducible artifacts in a release with hashes and a fetch
   manifest; and
6. deletes merged remote topic branches after preserving the original refs in
   an offline `git bundle --all`.

Dry-run result after aggressive garbage collection: 3,359 commits including
the snapshot, `git fsck --full --strict` clean, a 5.93 MiB pack, and a 2.27 MB
compressed current tree. This meets the
sub-10-MiB clone target while preserving every commit and all core source
history. Keeping every historical generated plot/test/tool blob is incompatible
with that target.

The source-complete public-ref rehearsal also passes. The refreshed candidate
maps `main` plus every open PR head through #105, retains all 28 remote tags,
passes strict `fsck`, and contains no AI attribution marker. PR #102 fixes an
asymmetric artifact writer: comparison JSON already summarized deterministic
LCFS/Boozer grids, while each single-result JSON retained the full arrays after
rendering. A follow-up audit also found that all five reduced-ITG scripts
resolved `examples/` as the repository root, so clean-checkout imports and
default output paths were wrong, and that the tracked nonlinear grid/backend
provenance was stale. The fix resolves the real root, regenerates current
72-by-72 metadata, and replaces a second copy of the nonlinear trace with a
checked reference. The nonlinear sidecar falls from 423,062 to 87,377 bytes;
the comparison sidecar falls from 273,704 to 247,507 bytes. Histories, scans,
retained traces, figures, numerical values, and physics results are unchanged.

The rewrite replaces only Git blob `45a40017c94bee2b6cca8e5a2b35573457c2db55`
(423,062 bytes) with its faithful compact representation; older 38-kB and
placeholder revisions remain untouched. PR #82 is hosted as one
exact-current-tree roadmap snapshot whose
parent is the rewritten CI-fix head. Its full incremental history remains in
the verified recovery bundle and old-to-new ref map. This avoids making a
living log consume most of the size margin while preserving its recoverability.
PR #103 then palette-encodes the retained full-resolution Landau PNG through
its generator (`371,750 -> 151,626` bytes); the six physics tests and a default
end-to-end regeneration preserve the published roots. The rewrite substitutes
only Git blob `99a3ff82ac7c9e15e66635e1bb054380decb81ad`.

PR #104 centralizes that deterministic palette path and applies it to three
more retained README figures. Runtime/memory, linear parity, and eigensolver
reach fall from 194,469/174,972/93,532 to 87,930/81,064/36,279 bytes. Their
same-canvas PSNR is 72.15/56.06/62.38 dB; every parent pixel coordinate and
dimension remains fixed. CI caught an initial 16-line source-budget regression;
the correction leaves `src/gkx` two lines below its frozen baseline and adds a
four-preview palette gate. PR #105 then removes 52 source lines by sharing
linear sampling/cache policy and the donated/nondonated JIT-wrapper body. It
retains module-local cache-builder and trace-time implementation injection.

PR #93 is the first geometry-owner cut. The differentiable facade delegates
its in-memory VMEC/Boozer bridge to the canonical core implementation while
retaining the facade's monkeypatch hooks. An independent follow-up removed an
unused parity constant and gates the public signature and `__wrapped__` owner,
so future core API changes cannot silently leave the facade stale. At
`b309a0af` the branch has 96,436 installed Python lines (-29 from the frozen
baseline) and removes 23 Python lines across the complete patch. Fifty-five
direct geometry tests, 117 release gates, Ruff, changed-module mypy, and the
architecture gate pass locally; all 41 required GitHub checks pass.
A separately opened #106 was closed once this existing owner was discovered;
it is not a second implementation or a promotion candidate.

PR #107 makes the runtime diagnostic fit policy the single frozen owner for
the eleven shared window/mode fields. The one-point request and ky-scan options
inherit that host-side record, and dispatch forwards only its canonical
mapping. No public signature, default, traced argument, JAX kernel, fit rule,
or result schema changes. The patch removes 19 installed-source lines
(`96,465 -> 96,446`) without adding a file. All 239 runtime helper/runner/CLI
tests, 117 release tests, four exact public-signature comparisons, Ruff,
changed-module mypy, and the architecture and repository-size gates pass
locally; all 41 required GitHub checks pass. The PR is draft and must not be
merged here.

PR #108 removes a duplicate resolved-diagnostic hot path. Heat and particle
ES/Apar/Bpar kernels were each evaluated twice per sampled state: once for
totals and again for channel spectra. The channel evaluation is now the sole
owner, and aligned reductions form the totals in the unchanged 58-field
schema. The patch removes ten installed-source lines and two dependency slots.
At `(Ns,Nl,Nm,Ny,Nx,Nz)=(2,4,8,32,32,24)`, the warmed CPU diagnostic kernel is
18.8% faster (0.742 to 0.603 ms), with 7.6% fewer XLA-estimated FLOPs, 4.5%
fewer accessed bytes, and 46.4% fewer JAXPR equations. Float64 differs from the
old totals by at most `6.34e-16` relatively; float32's worst max-scaled error is
`2.52e-6` from reduction reassociation. This is a diagnostic-kernel result, not
an end-to-end saturation-speedup claim. All 102 owned nonlinear diagnostic
tests and 117 release tests pass locally. The two broader float32-only failures
also reproduce on the untouched base; both pass on office JAX 0.11.1 under the
actual CI `JAX_ENABLE_X64=true` contract. All 41 required GitHub checks now
pass; only the noninterfering GPU benchmark remains open.

PR #109 makes `diffrax_core` the single owner of velocity-shape inference,
state sharding, and packed complex-state placement for both linear and
nonlinear Diffrax integration. Public signatures, the invalid-shape error,
packed layout, solver/cache/term policy, differentiation, and traced arithmetic
are unchanged. The same two-step linear/nonlinear base and head runs, including
single-device sharding, have byte-identical final-state and field-history
hashes. The patch removes 19 installed-source lines (`96,465 -> 96,446`) and
adds no file. All 14 focused Diffrax tests, 117 release tests, Ruff, mypy,
architecture, repository-size, and diff gates pass locally. All 41 required
GitHub checks pass.

PR #110 removes two more exact objective-policy copies without adding a file.
`stellarator_reduced` now solely owns the float32/x64 finite-difference
tolerances already consumed by the table layer, and `vmec_transport_admission`
solely owns finite scalar parsing for the VMEC transport reports. Public
signatures, defaults, schemas, thresholds, and results are unchanged. The patch
removes 14 installed-source lines (`96,465 -> 96,451`). Base/head digests of
both policies and every public signature match in float32 and x64. All 122
owning tests pass locally and under office JAX 0.11.1; 117 release tests, Ruff,
architecture, size, and diff gates pass locally. CI is active.

The rewrite maps only the three exact old PNG blobs to those final compact
blobs. PR #104's generator/source/image/physics-test blobs and aggregate text
patch are exact; its large release-gate file differs only in the intentional
rewrite comment that points to the branch-only roadmap. PR #105 has exact
public/private aggregate patch ID `3992710b`. Immediately before this record,
a fresh clone has 27 heads plus `origin/HEAD`, 28 tags, 3,466 commits, and
17,728 objects: pack 8,912,462 bytes, pack plus index 9,409,918 bytes, and
complete `.git` 9,744,233 bytes. Strict `fsck`, exact roadmap payload, no
alternates, zero reachability for the superseded blobs, and zero AI-attribution
hits pass.

After replaying the matched-CFL roadmap result, a fresh no-local clone of the
candidate through roadmap commit `7afdcc3d` has 25 heads, 28 tags, 3,443
commits, and 17,569 objects. Its transfer pack is 8,771,451 bytes; pack plus
index is 9,264,455 bytes; and the complete `.git` file sum is 9,597,923 bytes,
402,077 bytes below the strict decimal 10-MB gate. Strict `fsck`, zero
alternates, and zero reachable AI-attribution matches pass. The candidate head
names are exactly `main` plus all 24 open public PR branches. Closed topic heads
still require a frozen delete/retain map, and no remote history has moved.

PR #84 commit `55d41c09` adds the real-bug regression
`t_window,max=t_diag,last` under output striding. All 41 CI checks pass. PR #91
merge `14442da2` carries that contract into the reproducible saturation replay;
commit `eebff63b` adds atomic output locks after a real duplicate-writer event.
Private replays `bf8d5429`, `0b1d6ced`, and `eab1327d` have exact stable patch
IDs; the complete public/candidate PR #91 patch shares ID `4c2b67d8`. A new
ordinary no-local clone after exact plan replay `fc4c8d5a` has 25 heads, 28
tags, 3,453 commits, and 17,626 objects: pack 8,771,159 bytes, pack plus index
9,265,759 bytes, and complete `.git` file sum 9,599,487 bytes. Strict `fsck`,
no alternates, and zero reachable AI-attribution matches pass, leaving 400,513
bytes below the strict decimal gate. This is the measured pre-record candidate;
the next roadmap record must be replayed and remeasured rather than silently
assuming identical compression. No public history moved.

The PR #102 follow-up is replayed privately as `eed7eee3`; its stable patch ID
exactly matches public commit `51b55741`. Roadmap replay `8349f25d` has the
same patch ID and byte-identical `plan/` plus `plan.md` payload as public commit
`f38b4c69`. Immediately before this record, a fresh ordinary clone advertises
25 heads plus `origin/HEAD` and 28 tags, with 3,456 commits and 17,656 objects.
Its pack is 8,960,103 bytes, pack plus index is 9,455,543 bytes, and complete
`.git` file sum is 9,789,391 bytes. Strict `fsck`, no alternates, and zero
reachable AI-attribution matches pass; the decimal margin is 210,609 bytes.

The verified backup, exact retention contract, identity policy, blockers, and
coordinated cutover sequence are in `plan/history_rewrite.md`.

PR #88 is the first non-destructive tree cut. After CI exposed one missed
tool/test dependency, a repository-wide reachability audit restored 44 consumed
artifacts. The corrected cut deletes 153 generated assets: 7,696,596 bytes and
5,882 text lines. It reduces the tracked tree from 45.98 to 38.28 MB; the
affected 88 tests and all 41 current CI checks pass. PR #95 then retains only
the 12 README/core-physics visuals while making optional renders explicitly
regenerable. The remaining machine-readable evidence must migrate one schema
family at a time to hash-addressed release assets; deleting it blindly would
break documented provenance and tests.

The installed package baseline is 206 Python files and 96,465 lines:

| Package area | Files | Lines |
| --- | ---: | ---: |
| diagnostics | 21 | 15,077 |
| solvers | 30 | 14,794 |
| objectives | 26 | 13,329 |
| operators | 34 | 13,079 |
| geometry | 26 | 12,905 |
| workflows | 23 | 9,916 |
| artifacts | 13 | 6,242 |
| parallel, terms, core and other | 33 | 11,123 |

Twenty-seven modules exceed 900 lines and 88 exceed 500; combining files to
reduce their count would only hide complexity behind the 1,000-line gate. Slim
in this order: remove duplicated mathematics; keep campaign/report generation
outside hot solver paths; replace accidental facades with direct owners; then
move offline evidence builders out of the installed wheel behind compatibility
imports where they are public. Each cut must preserve the API manifest, CLI
snapshots, numerical gates, wheel smoke test, and accuracy-matched CPU/GPU
profiles. The first cut is in PR #91: one shared first-zero autocorrelation
estimator replaces runtime/post-hoc duplication and removes 21 source lines.
The same campaign now enables GKX's existing persistent JAX cache before the
runtime import. An identical two-process `4x4x4` CPU smoke fell from 7.11 s
cold to 2.61 s warm; this compile-dominated 63% reduction is a workflow check,
not a production-resolution timing claim. Commit `067788f0` also folds frozen
policy replay into that campaign owner and deletes the temporary standalone
tool, a net 45-line reduction from the first reproducible implementation. The
running QA/QHS/QI jobs remain pinned to the earlier solver tree.
The first package target is at most 190 files and 90,000 lines; lower targets
follow only after import/coverage evidence identifies another coherent cut.
PR #98 is the next measured cut: artifact I/O now imports the canonical
dealiased spectral layout instead of carrying three private copies. It removes
17 installed lines (`96,465 -> 96,448`) with 45 restart/layout tests, Ruff,
mypy, and the architecture gate passing locally. PR #99 builds on #98 and
removes the duplicate growth-fit validation/window/least-squares path. Exact
legacy parity holds for complex64/complex128, nonfinite filtering, and all four
windows; 52 diagnostic tests, 201 benchmark/runtime tests, Ruff, mypy, and the
architecture gate pass. The stacked source is 96,391 lines, 74 below baseline,
and a 10,000-point fit microbenchmark is unchanged (478 us versus 476 us).

Before the coordinated force push:

- publish the immutable bundle and SHA-256 checksum;
- enumerate every surviving branch/tag and map old to new commit IDs;
- make docs build without local `_static` results, using generated or fetched
  assets with verified hashes;
- normalize Rogerio's old Wisc/lowercase identities to the IST identity;
- remove known Claude co-author trailers and relabel three Codex-labelled
  commits; keep the other human authors unchanged;
- rehearse clone, install, docs, tests, tags, open PR rebases, and branch rules;
- announce that every existing clone must be recloned or hard-rebased.

Do not force-push until those checks and the recovery bundle are reviewable.

PR #95 keeps the six-second README loop as a 720-pixel, 24-frame, 4-fps WebP:
`828,066 -> 224,766` bytes, with 36.49 dB PSNR against its rendered source
frames. Its generator also palette-encodes the 1,561-by-1,189 QA initial/final
LCFS and Boozer panel to 74,390 bytes at 48.24 dB PSNR. PR #103 applies the
same generator-owned rule to the retained Landau panel (`371,750 -> 151,626`
bytes, 49.86 dB PSNR) while preserving its physics gates. The latest all-live-
head rehearsal metrics are recorded above. Strict `fsck`, exact live-head
parity, and the attribution scan pass; this remains evidence, not force-push
authorization.

## Validation campaign

Run cheap pilots only after R1--R4 are fixed. Escalate a case when the cheaper
level fails a gate.

| Axis | Pilot | Production | Refinement |
| --- | --- | --- | --- |
| perpendicular | 32x32 | 64x64 | 96x96, then Ny=128/160 while tails fail |
| parallel | Nz=24 | Nz=32 | Nz=48/64 |
| velocity | `(Nl,Nm)=(2,4)` | `(4,8)` | `(6,12)` and `(8,16)` as needed |
| timestep | nominal adaptive | independent `dt_max`/CFL | half-step matched run |
| randomness | 2 pilot seeds | at least 4 independent seeds | add until CI resolves sign |
| window | stationary suffix | >=10 IAT and >=4 batches | double duration |

For each stellarator rung, retain zonal/nonzonal `Phi2` and the zonal-flow
frequency/residual with `Q`, `Wphi`, and `Wg`. Recent global gyrokinetic studies
find substantially different nonlinear zonal suppression in QA, QH, and QI
configurations; linear growth and a stationary total flux do not establish the
same nonlinear state.

Equilibria: circular tokamak control; Landreman--Paul QA vacuum and beta 2.5%;
Landreman--Paul QH; Nuhrenberg--Zille QHS/HSX; one QI; one non-stellarator VMEC
control. Start with positive finite `tprim` and `fprim` so every case has a
transport signal.

For baseline/candidate seed pairs `d_i=Q_i^candidate-Q_i^baseline`, report

\[
 \bar d,\quad s_d/\sqrt n,\quad
 CI_{95\%}(\bar d),\quad P(d<0),
\]

with within-run autocorrelation and between-seed variation separated. A QA
transport-reduction claim requires the paired 95% interval below zero at two
overlapping resolutions, stable sign under timestep refinement, and acceptable
spectral tails. The optimization derivative is evaluated on training windows;
the final claim uses independent seeds and windows.

Every promotable run record must include

```text
path, sha256, seed, dt/CFL, Nx, Ny, Nz, Nl, Nm,
t_max, window_tmin, window_tmax, tau_int, batch_length,
Q_mean, Q_sem_corr, drift_Q, drift_Wphi, drift_Wg, spectral_tail
```

No ensemble average may cancel a failing member. Require all individual
stationarity gates, at least four non-overlapping batches of length
`5*tau_c`, a final-window drift below 20% of the mean, and a reported 5--10%
autocorrelation-corrected relative SEM target. Failed traces remain in the
artifact and are not silently discarded.

## Performance campaign

- Profile compile, geometry/cache setup, RHS, diagnostics, host transfer, I/O,
  and rendering separately.
- Scan `sample_stride=diagnostics_stride` at 10/25/50 on the same trajectory;
  compare diagnostic wall/storage cost and stability of IAT/SEM. The user's
  stride-50 file has median diagnostic spacing 2.31 and only two samples in
  `t=50--55`, so that interval cannot support an uncertainty estimate.
- Compare chunk sizes 32/64/128/256 on CPU and each GPU; select the smallest
  interval with <=5% warm-throughput penalty and bounded stop latency.
- Reuse prepared scans across chunks and optimization evaluations; changing only
  state must not retrace.
- Measure state bytes from
  `Ns*Nl*Nm*Nky*Nkx*Nz*sizeof(dtype)` and verify device memory is bounded by one
  chunk plus solver workspace.
- Sharding claims require speedup >1 on both CPU and GPU at a problem size that
  passes the same physics gates. Otherwise keep serial as the default.
- Compare existing implicit preconditioners `auto`, `pas`, `hermite-line`, and
  `hermite-line-coarse` on resolved linear and nonlinear IMEX cases. Report
  Krylov iterations/residual, compile and warm wall time, peak memory, and VJP
  parity before changing the diagonal default.

## Output and user experience

At nonlinear startup print three compact lines defining the run:

1. species and closure: `tprim=a/LT`, `fprim=a/Ln`, kinetic/adiabatic electrons;
2. diagnostics: `Q` heat flux, `Wg` gyrocentre free energy, `Wphi`
   electrostatic field energy;
3. mode trace: `phi ~ exp[(gamma-i omega)t]`, so `gamma` is growth/decay and
   `omega` is the signed frequency.

PR #85 implements these definitions. Independent review removed its unrelated
attempt to make `run_to = "saturation"` the shipped default: the source-pinned
QA/QHS/QI evidence above does not support that promotion. Fixed `t_max` remains
the fail-safe default until SAT-1 is seed-, timestep-, and resolution-robust.

The old low-level `[gkx] step/t/progress/eta` line was chunk-local and reset
every 128 or 1,024 steps. The trajectory was continuous, but campaign logs
looked restarted and their inner ETA was not the total-run ETA. PR #91 commit
`cd2a30f5` labels compiled updates `[gkx:segment]` and forwards the existing
host-side cumulative chunk status in the campaign. It adds no traced argument,
retrace, or diagnostic-cadence change. A two-chunk test locks monotone total
time and one wall clock. All 41 required checks pass; the nonlinear shard took
16m13s, directly exercising #81's targeted 20-minute budget.

Movies must reuse lightweight decimated runtime cuts rather than rerun the
physics. Store only an `x-y` cut and a `(y,z)` tube skin for about 60 frames;
encode a small WebP/MP4. Both the still and movie must use physical VMEC
coordinates and show the selected averaging interval only inside saved data.
PR #96 removes two of the three defects: snapshots retain only the rendered
`phi(x,y,z_mid)` and `phi(x_mid,y,z)` cuts (32x fewer values per `96x96x48`
frame), persist VMEC `R`, `Z`, and toroidal angle, and share one physical tube
renderer. It also fixes the README hero's import failure. PR #97 implements the
explicit restart alternative: load the saturation campaign's exact state,
continue with the deck's production explicit method and CFL policy, retain
absolute time and source verdict, then render off-device. Existing
scalar/spectral NetCDF output still cannot reconstruct field phases; a movie
without a saved state remains impossible rather than fabricated. An RTX A4000
continued the exact QA `96x160x48`, `(Nl,Nm)=(4,8)`, `t=250` state for 30 frames
in 49.1 s, but that first command imported an older installed GKX and omitted
the VMEC coordinate profiles. It is solver/timing evidence, not physical-movie
acceptance. PR #97 now reads `Nl` and `Nm` from `[run]` and fails closed instead
of drawing its analytic fallback when imported geometry is absent. A CPU smoke
then caught the EIK's closed 49-point coordinate interval being paired with the
solver's open 48-point field grid; PR #97 applies the same terminal trim as
production NetCDF output.

The clean `PYTHONPATH=src` GPU rerun now passes that acceptance gate. Its
2.5 MB schema-3 snapshot contains finite `phi_xy[30,96,160]`,
`phi_yz[30,160,48]`, and 48-point VMEC `R`, `Z`, and toroidal-angle profiles;
it records `nfp=2`, `(Nl,Nm)=(4,8)`, absolute times `250.150--253.949`,
`production_runtime_continuation`, and `source_saturated=true`. The off-device
H.264 render is 218,799 bytes, 900x472, 10 fps, and 30 frames. Inspection of
the first, middle, and final frames shows the physical two-field-period open
flux tube with evolving turbulence and no synthetic-torus or disconnected-end
artifact. Snapshot SHA-256:
`b5bd4e0a61ecd29885757c69cb882ff053cafa5aaf9793025bc7a99e615712a2`;
MP4 SHA-256:
`628b1cff237b5b5f77816579463b78585ab94eb82448598b0757cc41bb53e7a7`.

Independent re-review narrows that result: schema 3 records the source-state
path and copied `saturated` flag, but not the source-state content hash or the
campaign's input/VMEC/source identity. The hashes above bind the rendered cuts
and movie, not the restart that produced them. Treat this as physical-rendering
and continuation acceptance only. Before citing a movie as transport evidence,
compose PR #97 with PR #91's state-identity contract, require exact identity
agreement, and persist the source-state hash; legacy states must be labelled
unverified rather than silently promoted.

## Work queue

| ID | Priority | Status | Deliverable | Gate |
| --- | --- | --- | --- | --- |
| CI-1 | P0 | closed | PR #81 mypy fix + nonlinear-only 20-minute budget | all 41 PR checks and post-merge `main` CI pass |
| GOV-1 | P0 | review | PR #83 removes plan from main; PR #82 stays open | plan absent from main, branch recoverable |
| GOV-2 | P0 | cutover | rebase 18 direct #81-base heads onto rewritten `main`, then retarget | exact old/new head-tree and patch map; force-with-lease; fresh CI |
| RUN-1 | P0 | review | PR #84 exact horizon and 128-step checks | demonstrated on the supplied QA artifact; all 41 checks green |
| SAT-1 | P0 | active | PR #91 fixed-horizon replay, Q/Wphi/Wg gates, and output locks | QHS seed-31 makes no stop and differs from seed 22 by 8.76 combined SEM; QI clean repeat active |
| GEO-1 | P0 | review | PR #86 physical VMEC tube coordinates | NetCDF round trip + Cartesian coordinate test; all 41 checks green |
| RES-1 | P0 | active/review | PR #87 spectrum-tail warnings + QA/QHS/QI Ny scan | QHS Ny160 clears spectral tails but fails seed robustness; collided QI sizing is excluded and its clean locked repeat is active |
| VAL-0 | P0 | review | PR #89 per-trace QA audit; PR #90 promotion evidence contract | all 41 checks green on each; promotion remains false |
| UX-1 | P1 | review | PR #85 startup glossary | definitions pass; fixed horizon retained until SAT-1 passes |
| MOV-1 | P1 | active/review | PR #96 physical cuts + PR #97 production-state continuation | rendering passes; hash-bind source state and PR #91 identity before evidence use |
| VAL-1 | P1 | active | QA, QHS, and QI fixed-horizon campaign | paired CI + resolution + zonal gates |
| AD-1 | P1 | active/review | PR #100 narrows the finite-window adjoint claim | repeat source-pinned AD/FD knee, CPU/GPU, and optimized-equilibrium gates |
| SLIM-1 | P1 | active/review | PRs #88/#95/#102--#105/#107--#110 remove redundant renders/grids/traces/policies/setup; latest pre-record rehearsal `.git` is below 9.75 MB | freeze closed-head map, publish recovery records, then real network-clone gate |
| OUT-1 | P1 | review | PR #94 fails closed on rejected plot windows, including the one-page summary | supplied QA replot, focused tests, and all 41 CI checks pass |
| PR-1 | P1 | audited | every merged PR | dispositions in `plan/pr_audit.md`; named debt stays open |
| PERF-1 | P2 | active/review | existing SOLVAX line preconditioners + pure-JAX packed-FFT prototype | matched residual/forward/VJP/wall/memory comparison before any default change |

## Reproducibility records

- PR inventory and audit findings: `plan/pr_audit.md`
- literature and code survey: `plan/references.md`
- chronological decisions and measurements: `plan/log.md`
- exact recovery and force-push protocol: `plan/history_rewrite.md`
- detailed legacy investigations retained pending consolidation: `plan/notes/`
