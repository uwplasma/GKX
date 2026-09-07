# Contributing to GKX

GKX is a research code: its output is numbers people put in papers. That shapes
how changes are reviewed here. A patch is judged less on whether the tests go
green than on whether it makes the code's claims more checkable.

## Set up an environment that will not lie to you

Use a fresh virtual environment. GKX requires **jax >= 0.10.1**, because
`gkx.objectives.core` opts into `eig(..., enable_eigvec_derivs=True)`, which
older jax rejects.

```bash
python -m venv env
env/bin/pip install -e ".[dev]"
```

Running against an older jax does not fail with a version message. It fails
with a `TypeError` deep inside the objective vector and four Hermite-hierarchy
physics gates going red, which reads as "the physics is broken". A measurement
was recorded in the project plan on that basis and had to be withdrawn.
`tests/unit/api/test_public_types.py` now checks the floor and says what to do.

Run tests the way the nightly job does:

```bash
MPLBACKEND=Agg JAX_ENABLE_X64=true GKX_X64=1 pytest
```

Both variables matter. `JAX_ENABLE_X64` selects float64; `GKX_X64` is GKX's own
switch. Some gates are precision-specific and run in subprocesses so one
backend crash cannot take the suite with it.

## Know what the test runner is actually doing

Configuration lives in `pytest.ini`, not `pyproject.toml`:

```
addopts = -q --maxfail=1 --disable-warnings -m "not slow"
```

Two consequences that have each cost real time:

- **`--maxfail=1` stops at the first failure.** A run that reports one failure
  has not measured the others. Pass `--maxfail=0` when you want a blast radius.
- **`slow` tests never run in CI.** No job executes them: the coverage matrix
  clears `addopts` and re-adds `-m "not slow"`, and the nightly full suite
  inherits the same filter. Marking a new gate `slow` makes it dead. If a gate
  is worth having, make it fast enough to run, or say in review why it should
  exist unrun.

## What a change needs

One correctness change or one bounded measurement per pull request. The body
should say:

- scope and non-goals;
- the equation or API contract that changed;
- the tests, and **what failure they were verified against** — a test that has
  never failed has not been shown to test anything. Break the thing on purpose,
  record that the test catches it and that nothing else does;
- measured accuracy, runtime, memory and size deltas;
- which evidence-ledger rows it touches;
- adopt, reject or unresolved, and the next question.

### Numbers live in the ledger

`tools/evidence_ledger.toml` holds one row per public number: the artifact it is
computed from, the generator that rebuilds that artifact, the reference it is
compared against, and that reference's rank — analytic, published,
reference-code golden, open dataset, or self-run. `tests/release/` recomputes
every README number from its artifact.

Publishing a number means adding a row, not editing a test. A number in the
README or docs without a ledger row is a defect.

Prefer the highest-ranked reference available. A self-run comparison is rank 5
and is never described as published. If a reference is known to be flawed, the
row stays `provisional` and says why: a defective reference makes a comparison
uninformative, it does not make our side right.

### Budgets move deliberately

`tools/package_architecture_manifest.toml` caps file counts and line counts. The
checkers fail when a count regresses. Raising a baseline is allowed and normal,
but the manifest asks for a written reason next to the new number explaining why
this file or these lines had to exist and why no existing home would do. Read
the neighbouring comments for the standard.

### Claim wording is pinned

Some sentences in `README.md` and the docs are asserted verbatim by
`tests/release/test_release_gates.py` — the quasilinear scope, the deferred
lanes, the promotion criteria. Do not paraphrase them to read better. If a claim
should change, change the evidence first, then the sentence, then the gate.

## Before you push

```bash
ruff check . && ruff format --check .
python -m sphinx -W -b html docs docs/_build/html
pytest tests/release
for c in tools/release/check_*.py; do python "$c" || echo "FAILED $c"; done
```

## Running things

Never benchmark a shared checkout. Use `git worktree` so a concurrent session
cannot move the source under a running measurement, pin the commit, verify the
tree is clean, and record host, environment and command in the log.

Watch machine load before starting a sweep; the physics suites are CPU-hungry
and will happily oversubscribe a laptop.

## Reporting a problem

Open an issue with the deck or script, the exact command, the environment
(`python -VV`, `jax.__version__`, `jaxlib.__version__`, platform), and what you
expected versus what you got. If it is a numerical disagreement, say what the
reference is and where it came from; that is usually the whole question.

## Provenance and authorship

Commits are authored by the contributor. Do not add AI co-author trailers, and
do not remove upstream credit from vendored or adapted code. Pin companion
versions (VMEX, ESSOS, SOLVAX, JAX) when a result depends on them.

## Where the work is decided

`plan.md` is the single active roadmap: phases, exits, and what is deliberately
deferred with the trigger that would reopen it. `plan/log.md` is append-only —
one entry per step, including the ones that failed. Work that contradicts the
plan changes the plan in the same pull request rather than working around it.
