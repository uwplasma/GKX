# Tools

This directory holds repository-maintenance code that has not yet been folded
into the `scripts/` developer commands. It is not solver library code and not
end-user examples. The target (archived plan §21.4) is zero Python here and at
most eight commands under `scripts/`:

| Command | Owns |
| --- | --- |
| `scripts/check.py` | repository gates: size, architecture, performance manifest, release readiness, validation coverage, test gates |
| `scripts/inventory.py` | files, lines, imports, cycles, API, duplicates |
| `scripts/benchmark.py` | representative CPU/GPU benchmark matrix |
| `scripts/profile.py` | configurable JAX trace/profile runner |
| `scripts/validate.py` | scientific validation matrix |
| `scripts/figures.py` | reviewed documentation/README figures |
| `scripts/compare.py` | local external-code comparison protocol |
| `scripts/release.py` | build, smoke install, metadata, tag preflight |

The former `tools/release/` gates now live in `scripts/checks/` and run through
`python scripts/check.py <subcommand>`; `python scripts/check.py --list` prints
the subcommands. The file-by-file map of what remains here, who uses each file,
and where it goes next is `plan/research/2026-09-22-slim-tools/MAP.md`.

## What remains

- `artifacts/`: builders for reviewed README/docs figures, tables and JSON/CSV
  summaries (destination: `scripts/figures.py` or `scripts/validate.py`).
- `profiling/`: CPU/GPU runtime, memory and hot-path profilers (destination:
  `scripts/profile.py`).
- `comparison/`: external-code comparison and parity utilities, and their
  parity fixtures (destination: `scripts/compare.py`, fixtures under
  `benchmarks/cases/`).
- `campaigns/`: long-run launch and postprocess helpers still imported by tests
  or examples (destination: `scripts/validate.py`, or deletion once the tests
  that pin them are rewritten).
- `*.toml`: manifests read by the `scripts/check.py` gates and by the tests.

Nothing new goes here. A script that nothing in CI, tests, docs or examples
uses is deleted; it stays recoverable from Git history.

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
