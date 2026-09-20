# Q28 — `pr3-cm` in `src/`, and Q21's two levers re-measured there

Evidence for the 2026-09-20 `plan/log.md` entry (plan §5.1 L4/L5).

## Scripts

`diagnose.py`
: separates Q26's budget cap. `--mode budget` runs
  `dominant_eigenpair(method="shift_invert")` through the shipped path over a
  ladder of `shift_maxiter` and records the `EigenSolveStatus`. `--mode inner`
  takes the first right-hand side the outer Arnoldi generates and reports, as
  relative residuals against `||b||`, the trivial guess, the shipped
  `x0 = M^-1 b`, the shipped restarted solve from each, and an unrestarted
  ladder. `r_x0` is what decides between "the preconditioner is weak" and "the
  initial guess is wrong"; the ladder decides between "the preconditioner is
  weak" and "the restart is too short".

`measure.py`
: one arm per process. `--arm adaptive` is the control (the runtime default,
  with `adaptive_propagator_eigenpair` wrapped so its own
  `operator_applications` counter is recorded); `--arm si` is
  `KrylovConfig(method="shift_invert")` with `--preconditioner` and
  `--block-solve`. Matvec-equivalents follow Q21's registered accounting in
  `../2026-09-19-inner-solve-cost/PREDICTIONS.txt`.

`run_arms.sh`
: one fresh, cold, single-threaded process per arm, run serially, with the host
  load recorded at every boundary. Same process conditions as Q7's
  `run_production.sh` and Q21's `run_arms.sh`.

Q7's `bakeoff.py` is imported unchanged for the case construction, so the
operator, the deck and the seed are the ones Q7 and Q21 measured. Two rungs are
added to its `CASES` table: `d96` = `(1, 24, 96, 32, 2, 4, 8)`, the rung Q26
recorded, and `d96l8` = `(1, 24, 96, 32, 2, 8, 24)`, which is Q21's `r96`.

## Output directories

| directory | what it holds |
|---|---|
| `out/` | the budget-cap ladder and the inner-solve separation at `d96` |
| `d96/` | first `d96` pass: `adaptive` control, `pr3-cm` both block solves, `hermite-line`, the shipped budget |
| `d96b/` | the configuration search that found a certifying `pr3-cm` setting |
| `ab/` | the alternated `d96` measurement the log's first table reports |
| `r96/` | `d96l8` with the shift `shift_source="propagator"` derives |
| `r96s/` | `d96l8` with Q7's protocol shift (target + 0.05 in growth), which certifies |

Each arm writes `<label>.txt` ending in one `RESULT {json}` line and an
`exit N` line; `supervisor.txt` records the start time, load, exit status and
end time of every arm.

## Reproducing

From the repository root, with the environment in the log entry:

```
bash plan/research/scripts/2026-09-19-pr3cm-src/run_arms.sh OUTDIR \
  "A1_adaptive::--arm adaptive --case d96" \
  "B1_pr3_bt::--arm si --case d96 --krylov-dim 48 --restarts 2 --restart 600 \
     --maxiter 600 --inner-rtol 1e-6 --preconditioner pr3-cm --block-solve block-thomas"
```

`SCRIPT=diagnose.py` selects the diagnostic instead.

## Host

Contended throughout by four concurrent lanes: 1-minute load 7-12 during the
`d96` block and 15-65 during the `d96l8` block. No wall time in these files is
reproducible, and the log entry quotes none as a result. The load-independent
quantities -- matvec-equivalents, inner-iteration counts, factor bytes, the
certified residual and the returned pair -- are what the verdict rests on.
