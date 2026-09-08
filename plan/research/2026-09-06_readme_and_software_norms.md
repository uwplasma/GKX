# README exemplars and research-software norms

Researched 2026-09-06. Each README was fetched raw and measured; "prose" is the
share of non-code, non-heading, non-badge, non-table lines with four or more
words (±5 points).

## Fetched READMEs

| Name | Lines | Prose | First screen | Where validation lives | Citing / CITATION.cff |
|---|---|---|---|---|---|
| GKX (main) | 504 | 36% | 7 badges, one-sentence pitch, turbulence webp, `pip install gkx` at line 22 | two README tables plus docs | no / no |
| GX (bitbucket `gx`) | 124 | 37% | title, 4-sentence pitch, "rapid development", dependency list | `benchmarks/` + `check.py`; docs page is 43 words | yes / n/a |
| stella | 284 | 35% | CI badge, 4-line pitch, TOC, dependencies | README prose on 8 automated test groups; no numbers | no / yes |
| GS2 | 155 | 41% | DOI+pipeline+coverage badges, 6-line pitch, clone | none | yes / — |
| CGYRO/gacode | 24 | 25% | best practices, per-code citation table | none | yes / no |
| Gkeyll | 383 | 37% | About, code-structure layers, long install | none | no / UNVERIFIED |
| DESC | 138 | 32% | logo, 8 badges, 2-line pitch, "Please cite" + 4 papers, quick start; zero code lines | docs notebooks and JPP Part I | yes / yes |
| SIMSOPT | 86 | 56% | 3 badges, coils figure, one-sentence pitch | testing stated as principle | yes / no |
| ESSOS (uwplasma) | 290 | 26% | logo, tagline, 6 badges, TOC | one mention | no / no |
| VMEX (uwplasma) | 574 | 37% | 6 badges, pitch, showcase webp, install, Python example | 26-line convergence-parity section | no / yes |
| gyaradax | 164 | 21% | logo, 3-line pitch, whitepaper link, torus.gif of nonlinear ITG, install | prose; numbers in the preprint | yes / no |
| JAX-in-Cell (uwplasma) | 334 | 14% | logo, tagline, badges, TOC | none | no / no |
| Dedalus | 131 | 12% | title, ~14 badges, links | none | no / yes |
| JAX-MD | 181 | 32% | logo, tagline, nav bar, Colab notebooks | none | yes / no |
| Diffrax | 83 | 28% | one-line pitch, 7 feature bullets, install, example at line 20 | none | yes / no |
| PyTorch Geometric | 491 | 43% | logo, badges, nav, quick tour at line 67 | linked | yes / yes |
| jax-cfd | 152 | 42% | banner, "reproduce results from our PNAS paper" Colab | reproduction notebooks | yes / no |
| PhiFlow | 278 | 22% | logo, badges, image-table example gallery | benchmarks section | yes / no |
| Warp (NVIDIA) | 308 | 20% | badges, header image, "one million particles in 20 lines" at line 28 | none | yes / yes |

Best in class (Warp, Diffrax, JAX-MD, gyaradax, DESC): pitch → visual →
install → runnable example inside the first 30 lines → capability list →
docs/citing/license, ≤ ~310 lines, first code fence by line ~25. Anti-patterns
seen: architecture up front, badge walls without an example, version history,
dependency lists first, and (GKX/VMEX) performance prose, physics essays,
claim-scope governance, figure-regeneration tables.

## Norms

- [JOSS review checklist](https://joss.readthedocs.io/en/latest/review_checklist.html):
  statement of need; installation; example usage on real problems; API docs;
  automated tests; community guidelines; OSI license; performance claims
  confirmed. GKX lacks CONTRIBUTING.md and a citation file.
- [JOSS paper](https://joss.readthedocs.io/en/latest/paper.html): Summary,
  Statement of need, State of the field, Software design, Research impact, AI
  usage disclosure; ≤ ~1750 words. "How GKX compares" belongs in State of the
  field, not the README.
- [FAIR4RS](https://doi.org/10.15497/RDA00068): per-version persistent
  identifier (Zenodo), rich metadata (CITATION.cff, codemeta), license,
  provenance, domain standards (VMEC wout, netCDF).
- [Ten simple rules for documenting scientific software](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006561):
  examples, quickstart, README as the only documentation users read, how to cite.
- [CITATION.cff](https://citation-file-format.github.io/): GitHub renders "Cite
  this repository". stella, DESC, VMEX, Dedalus, PyG and Warp ship one.
- Code-paper split: SIMSOPT's JOSS paper has no validation section; GVEC's has a
  separate "Verification and validation" against VMEC; DESC's validation is in
  its JPP Part I.

## What the GKX README lacks, and the evidence each needs

1. Citing section, CITATION.cff and a Zenodo DOI badge; evidence: a Zenodo
   archive of the release.
2. A five-line VMEC → growth-rate example with the expected number printed;
   evidence: a CI test asserting that number against the tracked reference.
3. A gradient example with a plot (AD vs finite difference on one geometry
   parameter); evidence: the existing figure generator with a tolerance
   assertion in CI.
4. A nonlinear saturation figure with a flux number and a reference overlaid;
   evidence: a tracked nonlinear reference window compared in CI.
5. A "try it without installing" notebook badge; evidence: a notebook executed
   in CI.

No plasma or scientific repository fetched auto-generates README numbers from
CI; GKX's recomputed parity table is already ahead of peers. Shrink and move the
figure-recipe table; do not drop it.
