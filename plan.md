# GKX research-grade roadmap

This branch is the living plan and audit log. It is intentionally not part of
`main`. The previous long-form log remains in Git history; this file records the
current evidence, open defects, and next decisions.

Last reconciled: 2026-08-22 against `main` at `5f3ab32e` (merged PR #80)
and open PRs #74 and #81--#101.

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
| QA transport reduction | optimization scripts and preliminary paired runs | 44/48 nominal traces pass the per-trace drift gate; spectra absent | 12.26% is preliminary, not statistically resolved |
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
The first two fresh scores are now recorded below; QI decides the remaining
holdout without further threshold tuning.

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
continuation to absolute `t=500` is running. The compact trace SHA-256 is
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
screen and agree statistically. A matched Ny=128 seed-31 rung is still needed
before calling the Ny convergence seed-independent. Trace SHA-256:
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

The earlier environment-contaminated QA `96x128x48` rung completed at `t=250`
in 3,648.2 s. On the
fixed audit suffix `t=150--250`, `Q=11.006` with 2.21% corrected relative SEM;
Q, Wphi, and Wg pass the half-window checks. The matched `Ny=96` value is
`Q=10.860`, a 1.35% change. This is transport convergence evidence, not yet a
spectral pass: 16.3% of the `Ny=128` heat-flux spectral magnitude lies above
the old cutoff and the new cutoff remains 18.2% of the peak. The matched
`96x160x48` seed-22 rung completed in 5,327.5 s. Its fixed `t=150--250` suffix
has `Q=10.643 +/- 0.278` (2.61% corrected relative SEM), with Q/Wphi/Wg all
stationary. The change from Ny=128 is -3.30%, only 0.98 combined SEM. Its
heat-flux cutoff/peak falls to 7.59%, the last three bins carry 1.50%, and the
outer six signed kx shells carry 0.12%. The completed independent seed-31 rung
agrees statistically and passes the same tail screen, as recorded above. A
matched Ny=128 seed-31 rung remains before promoting seed-independent Ny
convergence.

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
outer-six-kx heat mass is 0.11%. The source-pinned Ny=160 seed-31 rung is queued
behind QHS `t=750`; a source-pinned Ny=96 match is still required. Trace
SHA-256:
`91f6768849caa8a315594d7f3bac256cc46a9203829ae0affae65d20ab6b5c45`.

PR #89 applies the Oberparleiter final-drift gate to each raw trajectory rather
than to a signed ensemble mean. The nominal campaign has 44 stationary traces
out of 48; the `dt=0.04`, perpendicular 24, and velocity `(6,12)` refinements
have 13/16, 6/8, and 30/32. No compact campaign output contains resolved spectra,
so the published 12.26% reduction is preliminary. PR #90 makes this fail-closed
in the release contract: promotion requires hashed raw sources, observed time
limits and averaging windows, and explicit stationarity, autocorrelation,
timestep, perpendicular, parallel, velocity, and spectral gates.

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
8.05%/1.48% and its Phi2 values are 8.80%/1.37%. This resolves the seed spread
at Ny=160, while a Ny=128 seed-31 rung remains the clean seed-independent
resolution check.

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
fixes it with a one-line source-neutral change. The PR #97 nonlinear shard also
showed a runner-budget failure after all tests printed 100%: its clean rerun
passed in 14m51s, nine seconds below the old 15-minute limit. PR #81 therefore
gives only `nonlinear-core` 20 minutes while every other quick shard retains 15.
PR #81's own nonlinear shard then passed in 14m22s under that targeted budget;
the extra five minutes cover runner cleanup variance without relaxing any test.
Branch protection nominally
asks for one review, but every merged PR reports `REVIEW_REQUIRED`, no checks are
required, force pushes are allowed, and administrator bypass was used. Require
the aggregate CI check and one non-author approval after the recovery rewrite.

## Repository recovery and slimming

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
maps `main` plus every open PR head through #101, retains all 28 remote tags,
passes
strict `fsck`, and contains no AI attribution marker. Closed topic heads still
require a frozen delete/retain map, and no remote history has moved.

The verified backup, exact retention contract, identity policy, blockers, and
coordinated cutover sequence are in `plan/history_rewrite.md`.

PR #88 is the first non-destructive tree cut. After CI exposed one missed
tool/test dependency, a repository-wide reachability audit restored 44 consumed
artifacts. The corrected cut deletes 153 generated assets: 7,696,596 bytes and
5,882 text lines. It reduces the tracked tree from 45.98 to 38.28 MB; the
affected 88 tests pass and full CI is rerunning. The remaining 27.24 MB
`docs/_static` tree must become generated previews plus hash-addressed release
artifacts; deleting it blindly would break documented evidence and tests.

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
not a production-resolution timing claim. The running QA/QHS jobs predate it.
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

PR #95 now re-encodes the six-second README loop from its primary release MP4:
`828,066 -> 346,234` bytes at 900 px and 5 fps. Immediately before this
measurement-recording commit, the
refreshed 21-head rehearsal (`main` plus all 20 open PRs through #99) retains
all 28 remote tags and has a 9,112,197-byte pack, 9,601,953 bytes including its
index, and a 5,363,108-byte current-tree archive. It has 3,438 commits and
17,453 reachable objects. Strict
`fsck`, stable patch IDs for all 19 recent replayed commits, exact live-head
parity, and the AI-attribution scan pass. This is still not force-push
authorization; recovery publication and the coordinated cutover gates remain.

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

## Work queue

| ID | Priority | Status | Deliverable | Gate |
| --- | --- | --- | --- | --- |
| CI-1 | P0 | ready for review | PR #81 mypy fix + nonlinear-only 20-minute budget | full CI green; no LOC regression |
| GOV-1 | P0 | review | PR #83 removes plan from main; PR #82 stays open | plan absent from main, branch recoverable |
| RUN-1 | P0 | review | PR #84 exact horizon and 128-step checks | R1 equations above; CI and review pending |
| SAT-1 | P0 | active | stationary suffix + Q/Wphi/Wg gates; PR #91 fixed-horizon trace capture | synthetic + held-out long traces |
| GEO-1 | P0 | review | PR #86 physical VMEC tube coordinates | NetCDF round trip + Cartesian coordinate test; source budget restored |
| RES-1 | P0 | active/review | PR #87 spectrum-tail warnings + QA/QHS/QI Ny scan | source-pinned QA seed replication, QHS continuation, and QI refinement pending |
| VAL-0 | P0 | review | PR #89 per-trace QA audit; PR #90 promotion evidence contract | local gates green; GitHub CI/review pending; promotion false |
| UX-1 | P1 | review | PR #85 startup glossary | CLI snapshots and definitions |
| MOV-1 | P1 | review | PR #96 physical cuts + PR #97 production-state continuation | physical QA GPU artifact, metadata, hashes, and visual inspection pass |
| VAL-1 | P1 | active | QA, QHS, and QI fixed-horizon campaign | paired CI + resolution + zonal gates |
| AD-1 | P1 | active/review | PR #100 narrows the finite-window adjoint claim | repeat source-pinned AD/FD knee, CPU/GPU, and optimized-equilibrium gates |
| SLIM-1 | P1 | active/review | corrected PR #88 removes 7.70 MB; public-ref rewrite rehearsal passes | freeze closed-head map, publish recovery records, then network-clone gate |
| PR-1 | P1 | audited | every merged PR | dispositions in `plan/pr_audit.md`; named debt stays open |
| PERF-1 | P2 | active/review | existing SOLVAX line preconditioners + pure-JAX packed-FFT prototype | matched residual/forward/VJP/wall/memory comparison before any default change |

## Reproducibility records

- PR inventory and audit findings: `plan/pr_audit.md`
- literature and code survey: `plan/references.md`
- chronological decisions and measurements: `plan/log.md`
- exact recovery and force-push protocol: `plan/history_rewrite.md`
- detailed legacy investigations retained pending consolidation: `plan/notes/`
