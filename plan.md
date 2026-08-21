# GKX research-grade roadmap

This branch is the living plan and audit log. It is intentionally not part of
`main`. The previous long-form log remains in Git history; this file records the
current evidence, open defects, and next decisions.

Last reconciled: 2026-08-21 against `main` at `5f3ab32e` (merged PR #80).

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
| VMEC/Boozer geometry | imported and differentiable paths | selected finite-beta and state-control checks | geometry-to-plot path is wrong |
| Linear derivatives | implicit eigenpair VJP with residual gates | AD/FD tests and selected GX comparisons | promoted on certified simple branches |
| Nonlinear derivative | checkpointed discrete adjoint of a finite post-saturation window | reduced AD/FD and device-parity tests | local finite-window derivative only |
| QA transport reduction | optimization scripts and preliminary paired runs | resolution/seed evidence is incomplete | do not claim statistically resolved reduction |
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
`(kx,ky)`. These structures are candidates for block/banded solves only after
profiling shows the dense or matrix-free operation is limiting. SOLVAX's block
tridiagonal ideas are relevant to implicit moment solves, not automatically to
the nonlinear Fourier convolution.

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

Fix in a dedicated PR: cap the terminal accepted step, retain the terminal
diagnostic independent of output stride, reduce the check interval from 1,024
to a measured value (start at 128), and replace the regression test that
currently blesses state/diagnostic mismatch.

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
that centreline. Tests must use a non-axisymmetric boundary and verify field-line
closure/twist numerically before visual comparison.

### R4 — perpendicular spectrum is unresolved

The heat-flux spectrum rises sharply at the largest saved `ky*rho`; the `phi^2`
spectrum is better but also turns up/noises at the tail. Treat this as a failed
resolution gate, not a physical result. Increasing `Ny` increases `ky_max` for
fixed `Ly`, but nonlinear triads and dealiasing require joint `(Nx,Ny)` tests.

For each retained spectrum require, for example,

\[
 f_{tail}=\frac{\sum_{k\in\text{top 20%}}|Q_k|}
 {\sum_k|Q_k|}<f_{max},\qquad
 \partial_{k_y}\log |Q_{k_y}|<0
\]

over the resolved tail, plus convergence of integrated `Q`. The thresholds are
to be calibrated against 96/128-point refinements, not chosen after seeing the
answer.

### R5 — CI and review governance

Main's only observed CI failure was a mypy error introduced by #80. PR #81
fixes it with a one-line source-neutral change. Branch protection nominally
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

Dry-run result: 3,359 commits including the snapshot, `git fsck --full` clean,
7.72 MiB pack, and a 2.2 MiB compressed current source tree. This meets the
sub-10-MiB clone target while preserving every commit and all core source
history. Keeping every historical generated plot/test/tool blob is incompatible
with that target.

Before the coordinated force push:

- publish the immutable bundle and SHA-256 checksum;
- enumerate every surviving branch/tag and map old to new commit IDs;
- make docs build without local `_static` results, using generated or fetched
  assets with verified hashes;
- normalize Rogerio's old Wisc/lowercase identities to the IST identity;
- remove five Claude co-author trailers and relabel three `codex/...` merge
  messages; keep the other human authors unchanged;
- rehearse clone, install, docs, tests, tags, open PR rebases, and branch rules;
- announce that every existing clone must be recloned or hard-rebased.

Do not force-push until those checks and the recovery bundle are reviewable.

## Validation campaign

Run cheap pilots only after R1--R4 are fixed. Escalate a case when the cheaper
level fails a gate.

| Axis | Pilot | Production | Refinement |
| --- | --- | --- | --- |
| perpendicular | 32x32 | 64x64 | 96x96, then 128 only if tail fails |
| parallel | Nz=24 | Nz=32 | Nz=48/64 |
| velocity | `(Nl,Nm)=(2,4)` | `(4,8)` | `(6,12)` and `(8,16)` as needed |
| timestep | nominal adaptive | independent `dt_max`/CFL | half-step matched run |
| randomness | 2 pilot seeds | at least 4 independent seeds | add until CI resolves sign |
| window | stationary suffix | >=10 IAT and >=4 batches | double duration |

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

## Performance campaign

- Profile compile, geometry/cache setup, RHS, diagnostics, host transfer, I/O,
  and rendering separately.
- Compare chunk sizes 32/64/128/256 on CPU and each GPU; select the smallest
  interval with <=5% warm-throughput penalty and bounded stop latency.
- Reuse prepared scans across chunks and optimization evaluations; changing only
  state must not retrace.
- Measure state bytes from
  `Ns*Nl*Nm*Nky*Nkx*Nz*sizeof(dtype)` and verify device memory is bounded by one
  chunk plus solver workspace.
- Sharding claims require speedup >1 on both CPU and GPU at a problem size that
  passes the same physics gates. Otherwise keep serial as the default.

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

## Work queue

| ID | Priority | Deliverable | Gate |
| --- | --- | --- | --- |
| CI-1 | P0 | PR #81 mypy fix | all CI green; no LOC regression |
| GOV-1 | P0 | remove plan from main; keep replacement plan PR open | plan absent from main, branch recoverable |
| RUN-1 | P0 | exact state/diagnostic horizon and short check chunks | R1 equations above |
| SAT-1 | P0 | stationary suffix + Q/Wphi/Wg gates | synthetic + held-out long traces |
| GEO-1 | P0 | physical VMEC tube coordinates | non-axisymmetric numerical closure test |
| RES-1 | P0 | spectrum-tail warnings and convergence protocol | known resolved/unresolved fixtures |
| UX-1 | P1 | startup glossary | CLI snapshots and definitions |
| MOV-1 | P1 | decimated x-y and 3-D movies | no physics rerun, size/time budgets |
| VAL-1 | P1 | multi-equilibrium replicated campaign | paired CI + resolution gates |
| AD-1 | P1 | re-audit nonlinear adjoint evidence/claims | AD/FD, Lyapunov-window, CPU/GPU |
| SLIM-1 | P1 | asset migration and rewrite rehearsal | 7.72-MiB-class pack, fsck/install/docs/tests |
| PR-1 | P1 | independent audit of every merged PR | ledger in `plan/pr_audit.md` |
| PERF-1 | P2 | CPU/GPU chunk/cache/sharding campaign | accuracy-matched wall/memory results |

## Reproducibility records

- PR inventory and audit findings: `plan/pr_audit.md`
- literature and code survey: `plan/references.md`
- chronological decisions and measurements: `plan/log.md`
- detailed legacy investigations retained pending consolidation: `plan/notes/`
