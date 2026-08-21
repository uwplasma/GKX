# Tools

This directory contains temporary repository-maintenance entry points, not
solver library code and not end-user examples. The GKX 2.0 target removes this
tree in favor of at most 12 conventional commands under `scripts/`.

## Ownership Rules

Keep a script in `tools/` only when it has a current owner and one of these
roles:

- `release`: bounded CI/release gates and repository-policy checks.
- `artifacts`: builders for reviewed README/docs figures, tables, and JSON/CSV
  summaries.
- `profiling`: CPU/GPU runtime, memory, and hot-path profiling reproducers.
- `comparison`: explicit external-code comparison and parity utilities.
- `campaigns`: documented long-run launch or postprocess helpers that are still
  part of an active validation or optimization lane.

Top-level `tools/*.py` is closed. New scripts must be added to a purpose
folder below. The former flat scripts were classified
as follows:

| Current kind | Destination |
| --- | --- |
| `generate_*` artifact/gate refreshers | `tools/artifacts/` if they write reviewed docs/readme artifacts; `tools/release/` only if CI/release calls them as gates. |
| `benchmark_*` reproducibility drivers | `benchmarks/` when user-facing and small; `tools/profiling/` when they are engineering profilers. |
| `build_linear_validation_artifacts.py`, `make_tables.py`, `make_benchmark_atlas.py` | moved to `tools/artifacts/`. |
| `digitize_*`, `derive_*` reference helpers | moved to `tools/artifacts/`. |
| diagnostic probes such as `ky_diagnostics.py` | active comparison diagnostics moved to `tools/comparison/`; unowned probes removed from `main`. |

Preview compression now belongs to the repository-hygiene command as
`tools/release/check_repository_size_manifest.py compress-previews`. This keeps
mutation opt-in while size policy, release-asset provenance, and preview
selection share one manifest owner.

Move or delete scripts that are only local probes, historical audits, blocked
campaign launchers, or one-off debugging helpers. If a removed script may be
useful later, keep it outside `main` in a draft PR or experiment branch rather
than shipping it as part of the maintained repository.

## What Does Not Belong Here

- Solver, geometry, diagnostic, objective, or artifact-library functionality.
  Put that in `src/gkx` only if it is promoted and reusable.
- User tutorials. Put those in `examples/`.
- Lightweight benchmark entry points and manifests. Put those in `benchmarks/`.
- Test-only helpers. Put those under `tests/` fixtures or test packages.
- Raw generated outputs, NetCDF files, logs, scratch directories, or profiler
  traces. Keep those ignored or attach them to releases when needed.

## Inventory

Use the maintained inventory tool before large moves or deletions:

```bash
python tools/release/check_package_architecture_manifest.py inventory \
  --json-out tools_out/repository_inventory.json \
  --summary-json-out tools_out/repository_inventory_summary.json
```

The inventory classifies tracked files by role and recommended action. It is a
planning aid, not a substitute for checking imports, docs references, tests, and
release manifests before deleting or moving files.

## Refactor Gate

`tools/package_architecture_manifest.toml` tracks the current Python-file
baseline and the final target. The default architecture check fails if the tool
count regresses upward. The final consolidation release should additionally run:

```bash
python tools/release/check_package_architecture_manifest.py --require-topology-targets
```

That strict mode fails until `tools/` is removed and the retained behavior is
owned by at most 12 commands in `scripts/`.
