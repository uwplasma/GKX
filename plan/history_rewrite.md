# History rewrite and recovery

Target: an ordinary full clone whose Git pack is below 10 MiB, while preserving
the complete pre-rewrite repository in a verified recovery bundle and retaining
all core-source history in the hosted repository.

## Evidence from the rehearsal

| Object | Result |
| --- | ---: |
| complete backup bundle | 285,415,042 bytes; 178 refs |
| bundle SHA-256 | `d08b073f34886914512d9c6d459c08d3404a49f26b4b0df5388e73acfa837205` |
| selected original main/tag commits | 3,358 |
| rewritten commits, including current auxiliary snapshot | 3,359 |
| rewritten refs | `main` + 28 tags |
| rewritten objects | 15,925 in one pack |
| rewritten pack | 5.93 MiB |
| rewritten current-tree archive | 2,269,659 bytes |
| integrity | `git fsck --full --strict` clean |
| candidate + current JSON/CSV evidence + PR #92 | 7.36 MiB pack; 3.5 MiB archive; 121 release tests pass |

`git bundle verify` reports that the bundle records a complete SHA-1 history.
It is the lossless recovery source; the hosted rewrite deliberately drops old
generated and auxiliary blobs that cannot coexist with a sub-10-MiB clone.

## Retention contract

- Preserve every pre-rewrite ref and object in the immutable bundle.
- Preserve the topology of every selected `main`/tag commit, including commits
  made empty by filtering; never use the default empty-commit pruning.
- Preserve complete `src/` history in the rewritten hosted repository.
- Preserve the current text tree with one provenance-labelled snapshot after
  tests and documentation stop requiring generated artifacts.
- Keep all 28 existing tags. Rewrite their targets and publish an old-to-new
  tag map; do not silently retarget a release without the map.
- Rebase the fourteen open PR heads (#74 and #81--#93) after the final rewrite.
  PR #82 remains open and unmerged as the living roadmap.
- Delete merged topic heads only after their exact old tips appear in the
  published ref map and complete bundle.

Historical `docs`, `tests`, `tools`, `examples`, `benchmarks`, `scripts`,
`plan`, and NetCDF blobs are removed from hosted history. Their current compact
text is restored by the snapshot. Large reproducible results move to a release
asset addressed by SHA-256; small deterministic fixtures remain in Git.

## Identity contract

- Normalize Rogerio's historical Wisc and case variants to
  `Rogerio Jorge <rogerio.jorge@ist.utl.pt>`.
- Remove Claude/Codex co-author trailers and generated-by footers.
- Relabel AI-authored commits created on Rogerio's machine to Rogerio while
  retaining their original timestamp, parent topology, and message content
  except AI branch labels such as `codex/` or `[codex]`.
- Leave every other human author and committer unchanged.
- Gate the candidate with a case-insensitive scan of author, committer, subject,
  and body for `Claude`, `Codex`, `Co-Authored-By`, and known AI-generated
  footers. Ordinary scientific prose containing “generated with” is not an
  authorship marker.

The full candidate audit is explicit. The selected history contains seven
objects with AI attribution markers: four objects (two duplicated logical
changes) with a Claude co-author trailer and three merge objects with `codex/`
or `[codex]` labels. Their original IDs are

```text
49ce7a8e b3c77752 a451089f b2959b06
f0e5ff99 5f84b0c4 4c56ba28
```

The rehearsal removes the four trailers, rewrites the three labels to
`rogeriojorge/`, and maps every Rogerio Wisc identity to the IST identity. Its
reachable authors are 3,355 Rogerio commits, two Eduardo Lascas Neto commits,
and two Raheem Hashmani identities; no Claude/Codex/co-author/generated-by
marker remains. The other human identities are unchanged.

## Required cutover sequence

1. Freeze merges and export the old branch, tag, protection, and open-PR maps.
2. Recreate `git bundle create GKX-pre-rewrite.bundle --all`; verify its hash,
   `git bundle verify`, strict `fsck`, and a restore clone on a second path.
3. Merge only reviewed source fixes. Keep PR #82 open. Replace tests that read
   `docs/_static` results with generated tiny fixtures or hash-verified fetched
   release data.
4. Rebuild the rewrite from the frozen mirror with explicit path, mailmap, and
   commit-message callbacks and `--prune-empty never --prune-degenerate never`.
5. In fresh local clones, run install/import, Ruff, mypy, unit/integration/
   validation/release tests, strict Sphinx, examples, and CPU/GPU smoke gates.
6. Measure a network-equivalent full clone with no alternates. Require both
   `.git/objects` and the current-tree archive below 10 MiB and a clean `fsck`.
7. Publish the bundle, checksum, old-to-new ref map, artifact manifest, and
   re-clone instructions before moving any public ref.
8. Temporarily relax only the rules needed for the coordinated force push;
   update `main` and all tags from exact candidate SHAs, then rebase the fourteen
   open PR heads with `--force-with-lease`.
9. Verify GitHub Actions on the rewritten refs, then delete only the enumerated
   merged heads and restore protection: required aggregate CI, one non-author
   approval, and no force pushes.

## Current blocker

The initial 5.93-MiB asset-free candidate is a proof, not a publishable
repository. Its release-gate file gave 110 passes and seven failures in three
dependency classes:

1. a test reads four generated quasilinear train/holdout reports from
   `docs/_static`;
2. the performance manifest requires local rendered result files; and
3. the validation-coverage manifest requires local rendered/result files.

PR #92 keeps machine-readable JSON/CSV/TOML evidence fail-closed while treating
PNG/PDF/SVG/WebP/GIF/MP4 files as reproducible renders. Restoring only the
current JSON/CSV snapshot to the rehearsal makes all 121 release tests pass;
after aggressive packing the complete candidate is 7.36 MiB and its source
archive is 3.5 MiB. PR #88 safely removes 153 unreferenced assets after
restoring 44 files consumed outside the documentation tree.

The remaining blockers are strict documentation and the scientific selection
of the small rendered set: keep only concise README/docs figures, verify every
reference, then run install/import, full tests, Sphinx, examples, CPU/GPU,
strict fsck, and network-equivalent clone gates before any force push.
