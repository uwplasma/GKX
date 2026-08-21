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
- Rebase the twelve open PR heads (#74 and #81--#91) after the final rewrite.
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
   update `main` and all tags from exact candidate SHAs, then rebase the twelve
   open PR heads with `--force-with-lease`.
9. Verify GitHub Actions on the rewritten refs, then delete only the enumerated
   merged heads and restore protection: required aggregate CI, one non-author
   approval, and no force pushes.

## Current blocker

The 5.93-MiB candidate is a proof, not a publishable repository. Its release
suite fails because
`tests/release/test_release_gates.py` reads
`docs/_static/quasilinear_cyclone_miller_train_holdout_report.json`, which the
rewrite correctly excludes. All such documentation/artifact dependencies must
be removed before the candidate can pass the cutover gates. PR #88 safely
removes 153 unreferenced assets after restoring 44 files consumed outside the
documentation tree, but deliberately retains this referenced file; it is tree
slimming, not yet dependency decoupling.
