# GKX history rewrite: analysis and plan (2026-10-15)

Status: proposal. Nothing was rewritten, force-pushed or deleted on the remote.
The owner decides each item in "Decisions" below.

## 1. Where the 202 MB comes from

| Measurement (2026-10-15, `main` = `df2953026`) | Size |
| --- | --- |
| Local `.git` of the working checkout (`count-objects`, size-pack) | 202 MB in 10 packs |
| `git clone --mirror` (includes 360 `refs/pull/*`) | 282 MB |
| `git clone --bare` from GitHub (heads + tags, what users get) | **19.1 MB** pack, 30,149 objects |
| Same, after `git gc --aggressive` | 15.9 MB |
| `--depth 1` clone of `main` (HEAD only) | 7.8 MB |
| Tracked tree of `main` (working files) | 23.9 MB, 2,173 files |

The 202 MB is local. It is one 138 MB pack plus older packs held by 181 local
branches, `refs/review/*`, the stash and reflogs on the shared checkout,
including objects from before the 2026-08-23 rewrite. A fresh clone is about
19 MB of pack plus about 24 MB of checkout. Re-clone, or prune the shared
checkout once no session needs its local-only branches: 574 local commits are
not on the remote.

History beyond HEAD is about 11 MB of pack (19.1 - 7.8). By extension, all
history (packed): `.py` 6.8 MB, `.md` 3.1 MB (mostly `plan/log.md`
revisions), `.png` 2.0 MB, `.json` 1.0 MB, `.rst` 0.85 MB, `.webp` 0.83 MB,
`.mp4` 0.82 MB, `.csv` 0.63 MB, `.txt` 0.59 MB. The biggest deleted path is
`src/spectraxgk/benchmarks.py` (20 MB unpacked across revisions, 0.19 MB
packed). No single blob is large. The 2026-08-23 rewrite already removed
the heavy artifacts. `git filter-repo --analyze` reports are in the raw
records (Section 7).

## 2. Options and measured sizes

Each dry run was done on a bare clone of the GitHub heads and tags, followed by
`reflog expire` and `gc --aggressive`. Baseline on the same footing: 15.9 MB.

| Option | What it does | Pack | Tags kept with identical trees | Commits on `main` |
| --- | --- | --- | --- | --- |
| 0 | nothing (Part A only) | 15.9 MB (19.1 MB as GitHub serves it) | all 41 | 4,102 |
| 1a | strip historical media/data blobs (png, webp, mp4, gif, jpg, pdf, npz, npy, nc, h5, csv, json, jsonl, txt, out, log) that are in no tag tree and not in HEAD: 537 blobs | 14.6 MB | all 41, verified tree-by-tree | 4,102 |
| 1b | same, but also strip blobs that only old tags hold: 588 blobs | 14.5 MB | none: every tag tree changes | 4,102 |
| 2 | 1b + squash everything before `v2.0.0` into one root commit (graft), dropping the 32 tags and 1 branch that do not contain it | 10.9 MB | 9 (v2.0.0 to v2.5.1) | 840 |
| 2' | same with the cut at `v1.8.2` | 10.9 MB | 10 | 945 |
| 3 | orphan: one commit with the HEAD tree | 7.8 MB | none | 1 |

A cut at `v2.5.0` does not work as a single graft: side branches merged after
it fork before it, so 4,035 commits survive. A late cut needs a graft on every
such merge parent.

Part A (this PR) adds a commit that deletes about 46k lines. It does not
shrink history. Options 1–3 decide whether the deleted `scripts/` code stays
in history. The recommended options keep it: code history is the useful part
of history, and it is small.

## 3. Recommendation

- **Do not rewrite for size.** Option 1a saves 1.3 MB of 15.9 (8%). Option 2
  saves 5 MB but breaks 32 tags, including releases v0.0.1 to v1.8.2 that PyPI
  and papers cite, and every open PR has to be force-updated. Neither fixes
  what the owner sees, which is the local 202 MB.
- If the owner still wants a rewrite: take **1a**. Release tags keep their exact
  trees. Only intermediate commits lose superseded figures and JSON.
- Separately, and with no rewrite: re-clone the working checkout, or prune
  merged local branches and run `git gc --prune=now`. That takes 202 MB to
  about 20 MB.

## 4. Refs affected by a rewrite

`git filter-repo` rewrites every commit hash from the first changed commit
onward. Under 1a and 1b that is almost every commit, because media blobs
change in early history.

- **Branches:** all 95 remote heads get new hashes (75 are already merged into
  `main`, so they could be deleted instead of rewritten; the owner decides).
- **Tags:** under 1a, all 41 tags are recreated on new commits with identical
  trees, and annotated tag objects are rewritten. Under 1b their trees change.
  Under 2, 32 tags are deleted (archive/pre-gkx2-condescending-buck, the
  baselines, v0.0.1 to v1.8.2).
- **`refs/pull/*`:** read-only on GitHub. They keep pointing at the old
  objects, so GitHub's stored size does not shrink without a GitHub Support
  request for a gc. Normal clones do not fetch them.
- **Open PRs** (heads at 2026-10-15): #361 `research/physical-free-energy`,
  #360 `validation/tem-rate-check`, #359 `research/pallas-plan`, #358
  `docs/examples-and-docs-1007`, #357 `fix/end-damping-timestep-free`, #323
  `docs/portable-artifact-references`, #275
  `draft/physical-cadence-handoff` (base `fix/saturation-retained-window-floor`),
  #274 `draft/em-multimode-handoff` (base `fix/nonzero-flr-cache-test`), plus
  this PR (`slim/scripts-and-media`). None of them touches a removed media
  path. Because filter-repo rewrites their branches in the same run, pushing
  the rewritten branches keeps each PR's diff unchanged. The PR timelines show
  a force-push.
- **Other clones and worktrees** (laptop, office `~/src/GKX`, the lane
  worktrees) must re-clone or `git fetch && git reset --hard origin/<branch>`.
  Any commit made on old hashes must be rebased with
  `git rebase --onto <new-base> <old-base> <branch>`. Map old to new hashes
  with `filter-repo/commit-map` in the rewritten repo.

## 5. Exact commands (option 1a)

Run in a fresh mirror, never in a working checkout:

```bash
git clone --mirror https://github.com/uwplasma/GKX.git gkx.git && cd gkx.git
git bundle create ../GKX-pre-rewrite-$(date +%F).bundle --all    # backup
# keep: every blob in HEAD of main or in any tag tree
( git ls-tree -r main | awk '{print $3}'
  for t in $(git tag); do git ls-tree -r "$t^{tree}" | awk '{print $3}'; done ) | sort -u > keep.txt
git rev-list --objects --all | awk 'NF==2' \
  | grep -E '\.(png|webp|mp4|gif|jpg|jpeg|pdf|npz|npy|nc|h5|hdf5|csv|json|jsonl|txt|out|log)$' \
  | awk '{print $1}' | sort -u > media.txt
comm -23 media.txt keep.txt > strip-ids.txt          # 537 ids on 2026-10-15
git filter-repo --strip-blobs-with-ids strip-ids.txt \
  --refs $(git for-each-ref --format='%(refname)' refs/heads refs/tags)
git reflog expire --expire=now --all && git gc --prune=now --aggressive
```

For option 2, add before `filter-repo`: delete the refs that do not contain
the cut, then `git replace --graft $(git rev-parse v2.0.0^{commit})`.
filter-repo makes the graft permanent.

Push only after the owner confirms, with branch protection relaxed for the
window. Push heads and tags; `--mirror` would also try the read-only `refs/pull/*`:

```bash
git push --force origin 'refs/heads/*:refs/heads/*' && git push --force origin --tags
```

## 6. Verification (performed on the dry runs)

- `git fsck --strict`: clean on 1a and on 2 (v2.0.0 cut).
- `main^{tree}` is identical before and after on 1a, 1b and 2
  (`87bdc5fa…`), so the tests on the rewritten HEAD are the tests on today's
  HEAD.
- Tag diff, comparing `git rev-parse <tag>^{tree}` before and after: 1a has 0
  of 41 tags changed; 1b changes 41 of 41.
- Still to run on the real rewrite: CI on the rewritten `main`, and
  `pip install .` from one rewritten release tag.

## 7. Raw records

Dry-run clones, `filter-repo --analyze` output and the strip lists are local
under `<local>/GKX-history-rewrite/` (`fresh.git`, `sim1.git`, `sim1t.git`,
`sim2-v2.0.0.git`, `sim2-v1.8.2.git`, `strip-ids.txt`,
`strip-ids-tagsafe.txt`, scripts `sim1.sh` and `sim2.sh`).

## Decisions for the owner

1. Rewrite at all? Recommended: no. Fix the local clone instead.
2. If yes: option 1a (tag-safe), 1b, 2 (cut v2.0.0 or v1.8.2) or 3.
3. Delete the 75 remote branches already merged into `main` before the rewrite?
4. Ask GitHub Support to gc `refs/pull/*` after the rewrite? Without it the
   server keeps the old objects, though clones do not fetch them.
5. Pick a window and freeze the open PRs (#357–#361, #323, #274, #275, this
   PR) while the rewrite is pushed.
