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
- Rebase the twenty-nine open PR heads (#74, #82--#105, and #107--#110) after
  the final rewrite.
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

The post-#95 slim tree still has 553 `docs/_static` JSON files (5,094,324
bytes). A direct reference scan finds 248 leaf-looking reports (1,470,629
bytes), but a transitive scan finds every one reachable through another
tracked report: for example, the release-facing horizon audit points to the
423-kB reduced-optimization trace. Do not delete these as allegedly unused or
replace them with one opaque aggregate JSON. First version the evidence schema
so a provenance edge may be a SHA-256-addressed release asset, teach consumers
to verify/fetch it explicitly, and migrate one evidence family per PR. Keep
small verdict/threshold summaries in Git; move raw objective histories,
replicate reports, and profiles behind the hash. This reduces JSON count while
preserving auditable provenance instead of leaving dangling filenames.

The saturation campaign also duplicates every scalar sample in its summary
JSON when a compressed NPZ trace is requested. The clean 662-sample QHS run
produced a 98,879-byte JSON plus a 984,418-byte spectral NPZ. PR #91 commit
`4ef05fb9` keeps the inline trace only when no trace artifact was requested;
otherwise it writes the NPZ path, SHA-256, byte count, schema, and report. The
same summary becomes 2,128 bytes (97.8% smaller) without weakening standalone
use or provenance.

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
   update `main` and all tags from exact candidate SHAs, then rebase the
   twenty-nine open PR heads with `--force-with-lease`. Do not merely retarget
   the eighteen direct #81-base PRs: the squash merge left equal base trees but
   divergent ancestry, so each head must be replayed onto rewritten `main` and
   pass exact tree/patch checks before retargeting.
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
clone. At the 20-head rehearsal, GitHub advertised 68 branch heads but only 20
open PR heads; closed heads cannot remain attached to their old object graph
under the 10-MB contract. That no-alternates rehearsal mapped all 20 open heads:
four reviewed slimming heads already represented in the candidate pointed at
slim `main`, and the other sixteen were replayed from their true PR bases.
Aggregate patch IDs matched for every textual patch; PR #90 alone omitted a
generated PNG that PR #95 intentionally removes. No replayed head reached the
old `main` object graph. The current 24-head inventory is recorded below.

The refreshed ordinary clone now advertises 25 heads (`main` plus every open PR
head through #103) and all 28 remote tags, with 3,424 reachable commits and
17,446 objects. PR #82 is represented by one source-neutral plan snapshot; its
incremental roadmap history remains in the recovery bundle and ref map. PR #102
also corrects the single-result artifact writer so deterministic rendered grids
are summarized like the comparison sidecar, validates and references the
duplicate nonlinear trace, and fixes clean-checkout root resolution in all five
reduced-ITG scripts. The regenerated nonlinear optimization JSON is 87,377
bytes instead of 423,062 bytes; the comparison JSON is 247,507 instead of
273,704 bytes. The replay must substitute the exact compacted blobs while
leaving earlier compact/placeholder revisions unchanged.

PR #103 adds a generator-owned palette encoding for the retained Landau
validation panel. It preserves the 3,028-by-822 pixel grid and every curve,
annotation, and numerical result while changing 371,750 bytes to 151,626.
Pixel comparison against the previous RGBA render gives 49.86 dB PSNR and
0.066 mean absolute channel error. The rewrite replaces only Git blob
`99a3ff82ac7c9e15e66635e1bb054380decb81ad`; full default regeneration retains
the two exact-root comparisons and `1.954e-14` collisionless spectral residual.

PR #104 adds the same shared deterministic palette path to the retained
runtime/memory, linear-parity, and eigensolver-reach figures. The rewrite maps
only these exact old-to-new blob pairs:

| Artifact | Old blob | New blob |
|---|---|---|
| runtime/memory | `686c0aaaa7834f7780cd62aebec4780371d93f69` | `da5e5148fcf40dea6dd3ab1f146901066d59344c` |
| linear parity | `a10c72fa33c593671c34800f55720842df99d54d` | `3fce1247198d095e91f55fe5411862341b4abdf1` |
| eigensolver reach | `7b09d5f4c427c671d5f6dbde4cfa3369fabf806b` | `2cce16e1776cd14a2b59fa427d76b7fe162ca8fb` |

The public/private aggregate text patch has stable ID `72e21a1c`. All
generator/source/image/physics-test head blobs are exact; the large release
gate differs only at its intentional rewritten roadmap-reference comment. The
binary deltas disappear privately because each parent already contains the
mapped compact blobs; the dimension-preservation commit is retained empty via
`--prune-empty never`. PR #105's exact public/private aggregate stable patch ID
is `3992710b`. Immediately before this record, a fresh 27-head/28-tag clone has
3,466 commits, 17,728 objects, an 8,912,462-byte pack, 9,409,918 bytes for pack
plus index, and a 9,744,233-byte complete `.git` file sum. Strict `fsck`, no
alternates, zero reachable original/intermediate blobs, exact roadmap payload,
and the attribution scan pass.

PR #95 keeps the six-second turbulence interval as a 720-pixel, 24-frame WebP
and palette-encodes the initial/final QA equilibrium panel in their owning
generators. The final rewrite maps both loop blobs
`0dda88abe0e8472a10d0cec388ec255774b76983` and
`e972c776a9e0289090dea8bc4c6fc1ab589cd72e` to
`bb9529af3db1004643a9720b2269d486c1f25b24`, and maps equilibrium-panel blob
`ff4510d2b990665400ae1b01046c4983e7e5b832` to
`8be00988b25d0ed1218dd14e6cc56e3a004e16a9`. No obsolete rendering remains
reachable in the rehearsal.

Immediately before this record, a fresh no-alternates clone of all 25 heads and
28 tags has 3,427 commits and 17,469 objects: an 8,723,042-byte
network-equivalent pack, a 490,204-byte index, 9,213,246 bytes for pack plus
index, a 9,546,342-byte complete `.git` file sum, and the unchanged
4,888,710-byte current-tree archive. Strict
`fsck` passes, the clone has no object alternate, and reachable commit metadata
has no AI attribution marker. This closes the local public-ref size gate with
1,276,958 bytes of transfer-pack margin and 453,658 bytes under the stricter
complete-`.git` decimal gate. The user's 2026-08-21 request authorizes a
coordinated forced rewrite, but authorization does not replace the remaining
safety gates: land only reviewed slimming prerequisites in the intended main
tree, freeze and publish the remote ref map, publish the recovery material, run
representative CPU/GPU examples, and verify a real GitHub network clone before
restoring branch protection. Until those prerequisite PRs are reviewed, moving
the rehearsal's `main` ref would merge their contents by force push and is not
an admissible substitute for review.

The current rehearsal additionally replays PR #84 commit `55d41c09` as
`bf8d5429`; both aggregate patches have stable ID
`d35a7a70430902d9408caba26dd53d6463b8d0a6`. PR #91 merge `14442da2` and
output-lock commit `eebff63b` are replayed as `0b1d6ced` and `eab1327d`; the
complete public/candidate PR #91 patches share stable ID
`4c2b67d88380fdd8805deea2832bd9020f78a19e`. A fresh ordinary `--no-local`
clone after exact plan replay `fc4c8d5a` advertises 25 heads and 28 tags, with
3,453 commits and 17,626 objects. Its pack is 8,771,159 bytes, pack plus index
is 9,265,759, and the complete `.git` file sum is 9,599,487 bytes. Strict
`fsck`, no object alternate, and the reachable-attribution scan all pass; the
strict decimal margin is 400,513 bytes. This is a commit-pinned pre-record
measurement, not an assumption that the following log commit compresses to the
same size. The following commit must be replayed and measured independently.

PR #102 follow-up commit `51b55741` is replayed as `eed7eee3` with exact stable
patch ID `e6f87004`; roadmap commit `f38b4c69` is replayed as `8349f25d` with
exact stable patch ID `3526b11c` and byte-identical roadmap payload. Immediately
before this record, a fresh no-local clone has 25 heads plus `origin/HEAD`, 28
tags, 3,456 commits, and 17,656 objects. Pack, pack-plus-index, and complete
`.git` file sums are 8,960,103, 9,455,543, and 9,789,391 bytes. Strict `fsck`,
zero alternates, and the reachable-attribution scan pass, leaving 210,609 bytes
of strict decimal margin. The following roadmap record again requires replay
and independent measurement before any coordinated cutover.

PR #93 follow-up `b309a0af` is replayed as `1fe08146`. Its complete aggregate
patch has the same public/private stable ID `f214864b`, and the three affected
head blobs are byte-identical. A garbage-collected, ordinary no-local clone of
the current candidate has 28 remote refs, 28 tags, 3,471 commits, and 17,758
objects. Its pack is 8,736,180 bytes, pack plus index is 9,234,476 bytes, and
the complete `.git` file sum is 9,568,911 bytes. Strict `fsck`, no alternates,
and zero reachable AI-attribution matches pass, leaving 431,089 bytes of
strict decimal margin. The redundant draft #106 is closed and is not retained
as a live rewrite head. No public history moved.

Draft PR #107 commit `96911c3b` is replayed as `168dc834`. The exact
public/private stable patch ID is `df901451`, and all four affected head blobs
are byte-identical. A garbage-collected ordinary clone now advertises 29 remote
refs and 28 tags, with 3,473 commits and 17,777 objects. Its pack is 8,687,046
bytes, pack plus index is 9,185,874 bytes, and complete `.git` file sum is
9,520,481 bytes. Strict `fsck`, no alternates, and zero reachable
AI-attribution matches pass, leaving 479,519 bytes of strict decimal margin.
No public history moved.

Draft PR #108 commit `f7634a03` is replayed as `8f2444d8`. The exact
public/private stable patch ID is `62fef1c4`, and all four affected head blobs
are byte-identical. A garbage-collected ordinary clone now advertises 30 remote
refs and 28 tags, with 3,475 commits and 17,798 objects. Its pack is 8,690,625
bytes, pack plus index is 9,190,041 bytes, and complete `.git` file sum is
9,524,832 bytes. Strict `fsck`, no alternates, and zero reachable
AI-attribution matches pass, leaving 475,168 bytes of strict decimal margin.
No public history moved.

Draft PR #109 commit `0561e5dc` is replayed as `90656d0c`. The exact
public/private stable patch ID is `b094d3f5`, and all three affected head blobs
are byte-identical. A garbage-collected ordinary clone now advertises 31 remote
refs and 28 tags, with 3,478 commits and 17,818 objects. Its pack is 8,696,888
bytes, pack plus index is 9,196,864 bytes, and complete `.git` file sum is
9,531,837 bytes. Strict `fsck`, no alternates, and zero reachable
AI-attribution matches pass, leaving 468,163 bytes of strict decimal margin.
No public history moved.

Public PR #81 was squash-merged externally as `0ff569c3` on 2026-08-22 after
all 41 PR checks passed but without an approving review. Its public tree equals
topic head `d910ac56`; the private candidate already carried the one-line
run-summary typing repair on `main`, so only the reviewed CI workflow delta was
missing. Private commit `00fb4dae` preserves the public squash subject, body,
timestamp, one-parent topology, Rogerio author, and GitHub committer while using
the exact private topic tree `b1c356ee`; its parent is the prior private main
`2f8521ab`. A garbage-collected ordinary no-local clone now advertises 32
remote refs and 28 tags, with 3,486 commits and 17,871 objects. Its pack is
8,524,088 bytes, pack plus index is 9,025,548 bytes, and complete `.git` file
sum is 9,360,868 bytes. Strict `fsck`, no alternates, and zero reachable
AI-attribution matches pass, leaving 639,132 bytes of strict decimal margin.
No public history was force-pushed.
