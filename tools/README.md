# Tools

This directory holds no Python. It keeps the repository manifests read by the
`scripts/` commands and the tests, and the GX parity fixtures in
`comparison/fixtures/`. The developer code lives under `scripts/`, one command
per purpose (archived plan §21.4):

| Command | Modules | Owns |
| --- | --- | --- |
| `scripts/check.py` | `scripts/checks/` | repository gates: size, architecture, performance manifest, release readiness, validation coverage, test gates |
| `scripts/benchmark.py` | `scripts/benchmarks/` | literature benchmark drivers, runtime/memory and integrator benchmarks |
| `scripts/profile.py` | `scripts/profiling/` | runtime, startup, RHS-term, parallel and sharding profiles |
| `scripts/validate.py` | `scripts/campaigns/`, `scripts/artifacts/` | validation campaigns and the builders of the tracked gate JSONs and figures |
| `scripts/compare.py` | `scripts/comparison/` | GX comparison protocols, parity matrix, reference panels |

Each command runs one module with the remaining arguments, exactly as if the
module had been executed directly; `--list` prints the modules. Still to come:
`scripts/figures.py` (README figures) and `scripts/release.py`. The inventory is
`python scripts/check.py architecture inventory`.

The file-by-file map of the move, who uses each file, and what is left to
contract is `plan/research/2026-09-22-slim-tools/MAP.md`.

## What remains here

- `*.toml`: manifests read by the `scripts/check.py` gates, the comparison and
  benchmark modules, and the tests.
- `comparison/fixtures/`: GX parity decks and the GX golden-file notes
  (destination: `benchmarks/cases/`, together with the manifests that name
  them).

Nothing new goes here.

## Inventory and refactor gate

```bash
python scripts/check.py architecture inventory \
  --json-out tools_out/repository_inventory.json \
  --summary-json-out tools_out/repository_inventory_summary.json
```

`tools/package_architecture_manifest.toml` tracks the Python file and line
baselines and the final targets. The default check fails if a count regresses
upward; the final consolidation release additionally runs:

```bash
python scripts/check.py architecture --require-topology-targets
```
