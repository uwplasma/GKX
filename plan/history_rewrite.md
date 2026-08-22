# History rewrite and recovery

Target: an ordinary full clone whose Git pack is below 10 MiB, while preserving
the complete pre-rewrite repository in a verified recovery bundle and retaining
all core-source history in the hosted repository.

## Why the clone is large

The audited local clone has 44,217 packed objects in 160.60 MiB; a fresh
network clone reports 133.70 MiB transferred. Summing unique historical blob
payloads before compression attributes 324 MB to `docs/`, 247 MB to root files
(mostly successive large plan snapshots), 169 MB to `tools/`, 115 MB to
`tests/`, and 101 MB to `src/`. The largest individual blobs are a 3.92-MB
historical HSX WOUT, 2.88/2.47-MB versions of a generated benchmark panel,
1.87/1.61-MB nonlinear-atlas renders, and a 1.73-MB synthetic optimization
plot. Repeated PNG/PDF/JSON/plan revisions, not the current solver source, are
the dominant removable history.

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
- Rebase the twenty open PR heads (#74 and #81--#99) after the final rewrite.
  PR #82 remains open and unmerged as the living roadmap.
- Delete merged topic heads only after their exact old tips appear in the
  published ref map and complete bundle.

Historical `docs`, `tests`, `tools`, `examples`, `benchmarks`, `scripts`,
`plan`, and NetCDF blobs are removed from hosted history. Their current compact
text is restored by the snapshot. Large reproducible results move to a release
asset addressed by SHA-256; small deterministic fixtures remain in Git.

## Source slimming contract

The pre-slimming installed tree has 206 Python files and 96,465 lines:

| Package group | Files | Lines |
| --- | ---: | ---: |
| diagnostics | 21 | 15,077 |
| solvers | 30 | 14,794 |
| objectives | 26 | 13,329 |
| operators | 34 | 13,079 |
| geometry | 26 | 12,905 |
| workflows | 23 | 9,916 |
| artifacts | 13 | 6,242 |
| package root and remaining groups | 33 | 11,123 |

The first enforced milestone is at most 190 files and 90,000 lines. Reach it
through dead-path removal, one-owner contracts, and proven delegation such as
PR #93; do not concatenate unrelated modules or abstract compiled hot kernels
merely to lower counts. Each source cut must preserve public signatures,
JIT/AD semantics, CPU/GPU numerics, runtime, and peak memory through focused
tests plus the architecture gate. The manifest's 45-file/45,000-line values
remain a long-range aspiration, not permission for a mega-refactor.

An AST inventory plus Pylint's eight-line clone scan gives the next bounded
cuts. Order them by risk, and merge none into the roadmap PR:

| Cut | One owner | Required gate |
| --- | --- | --- |
| Remove duplicate `_dealiased_*` helpers from `artifacts/io.py` | `artifacts/spectral_layout.py` | odd/even/singleton restart round trips and NetCDF layout identity |
| Share linear/nonlinear Diffrax shape packing and solve options | `solvers/time/diffrax_core.py` | eager/JIT, donation, checkpoint, adaptive, CPU/GPU identity and compile count |
| Remove the second growth-fit input/least-squares path | `diagnostics/growth_windows.py` + one public wrapper | all fit-window modes, uncertainty, nonfinite and monkeypatch contracts |
| Share scalar/resolved heat, particle and heating reductions | `diagnostics/moments.py` for kernels; `diagnostics/transport.py` for public totals | per-species/channel sums, dealiased spectra, JIT/VJP and normalization identity |
| Share explicit/IMEX nonlinear diagnostic setup | `solvers/nonlinear/diagnostics.py` | fixed/adaptive CFL, restart, stride, exact horizon, memory and wall-time non-regression |
| Collapse repeated runtime fit-option records/forwarding | one frozen options record under `workflows/runtime` | CLI/TOML signatures, scan semantics and no extra traced arguments |

Do not start by merging `io.py`, `nonlinear_netcdf.py`, or `workflows/linear.py`
into larger files: all three already exceed the module budget. First remove
their duplicate serializers/forwarders, then lower each exception below 1,000
lines. Likewise, moving report code from `src` to `tools` does not count as a
line reduction; functionality must be deleted, generated, or delegated.

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
   update `main` and all tags from exact candidate SHAs, then rebase the twenty
   open PR heads with `--force-with-lease`.
9. Verify GitHub Actions on the rewritten refs, then delete only the enumerated
   merged heads and restore protection: required aggregate CI, one non-author
   approval, and no force pushes.

## Resolved preparation and remaining cutover gates

The initial 5.93-MiB asset-free candidate was a proof, not a publishable
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

A literal documentation scan finds 118 embedded local renders totalling
10,861,085 bytes; the README alone embeds 12 totalling 2,354,028 bytes. They
cannot all be restored on top of the 7.36-MiB numeric-evidence pack. Keep a
small physics/algorithm/QA figure set whose network-clone pack still passes the
10-MiB gate, re-encode oversized animations, and replace the remaining local
directives only with real generator instructions or published release links --
never suppress broken-image warnings.

PR #95 implements that selection: retain exactly the 12 README/core-physics
visuals (2,354,028 bytes), remove 226 reproducible renders (18,170,355 bytes),
and replace 121 optional RST directives with explicit generated-figure notes.
Missing optional renders now carry an explicit `regenerate_on_demand` manifest
action; numeric sidecars stay mandatory and a present regenerated file is still
hash-checked. Strict Sphinx passes without warning suppression. A fresh
single-branch clone of the combined rehearsal has one 9,464,119-byte pack
(9.46 MB, 9.03 MiB) and a 5,883,089-byte source archive (5.88 MB, 5.61 MiB).

Repository hygiene, strict Sphinx, sdist/wheel build, installed-wheel import,
CLI startup, strict `fsck`, and all 2,554 collected x64 tests pass in that fresh
clone. GitHub currently advertises 68 branch heads but only 20 open PR heads;
closed heads cannot remain attached to their old object graph under the 10-MB
contract. A no-alternates rehearsal now maps all 20 open heads: four reviewed
slimming heads already represented in the candidate point at slim `main`, and
the other sixteen are replayed from their true PR bases. Aggregate patch IDs
match for every textual patch; PR #90 alone omits a generated PNG that PR #95
intentionally removes. No replayed head reaches the old `main` object graph.

Immediately before the present measurement-recording commit, the refreshed
ordinary clone advertises 21 heads (`main` plus the 20 PR heads through #99)
and all 28 remote tags, with 3,450 reachable commits and 17,522 objects. Its
one pack is 9,128,917 bytes; pack plus index is 9,620,605 bytes, the complete
`.git` directory is 9,953,481 logical bytes, and the compressed current-tree
archive is 5,363,108 bytes. Strict `fsck` passes. All 30 recent replayed commits
have exact stable patch-ID matches, the clone has no object alternate, and the
reachable commit metadata has no AI attribution marker. This closes the local
public-ref size gate. The unpacked checkout remains 19,558,568 logical bytes;
that is a separate source-slimming target, not transferred Git history. The
user's 2026-08-21 request authorizes a
coordinated forced rewrite, but authorization does not replace the remaining
safety gates: land only reviewed slimming prerequisites in the intended main
tree, freeze and publish the remote ref map, publish the recovery material, run
representative CPU/GPU examples, and verify a real GitHub network clone before
restoring branch protection. Until those prerequisite PRs are reviewed, moving
the rehearsal's `main` ref would merge their contents by force push and is not
an admissible substitute for review.
The margin forbids another tracked render without removing equivalent payload.

PR #95 subsequently replaced the README loop from the primary 120-frame MP4,
keeping the same six-second interval as 30 WebP frames at 900 px: 828,066 bytes
became 346,234. Applying that exact blob substitution to the rehearsal removes
the old blob completely and produces a 9,030,044-byte pack, 473,852-byte index,
5,363,108-byte source archive, and strict `fsck`. The final replay must still
refresh every live head; these figures establish margin, not cutover identity.
