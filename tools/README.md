# Tools

This directory holds no Python. It keeps the repository manifests read by the
`scripts/` commands and the tests, and the GX parity fixtures in
`comparison/fixtures/`. The developer code lives under `scripts/`, one command
per purpose (archived plan §21.4):

`scripts/check.py` runs the repository gates (size, architecture, performance
manifest, release readiness, validation coverage, test gates) from
`scripts/checks/`; `--list` prints them. The benchmark, profiling, campaign,
comparison and figure drivers were removed on 2026-10-15; `plan/log.md` lists
each one and the commit to recover it from (`git show df2953026:<path>`).

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
