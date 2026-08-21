## Baseline Manifest (Step 4)

- Timestamp (UTC): `2026-02-24T05:42:57Z`
- Git commit (short): `fca003d`
- Git commit (full): `fca003d29e7f98b6674f2686f769b3510acf1589`

### Scope

Step 4 delivered:

- normalization/sign contract documentation updates
- full post-change artifact regeneration (tables + figures)
- full validation gates (tests, type-check, docs build)

### Commands run

```bash
python tools/artifacts/make_tables.py --case all --no-progress
python tools/artifacts/build_linear_validation_artifacts.py figures --case all --no-progress
pytest -q --maxfail=1 --disable-warnings
mypy src/gkx
python -m sphinx -W -b html docs docs/_build/html
```

### Artifact checksums (SHA-256)

```text
9951603e2ea12e85fde9c10d9d3b97369906e8f3d11cb4ca2b4adbd8f4319db5  kbm_mismatch_table.csv
ed11e63c8cca5dd34e07a402e611eb0840b6f90a9049098bde0078c9cee40fe2  etg_mismatch_table.csv
fba2a63b4b82f5e69fbca9ec44fd3b13e0f849b538f1b2180d44550d08fd9501  kinetic_mismatch_table.csv
2baf3b30b4c7ae5147a93f1574ebfb7235830ce1e0acb4f0793df11190b4329c  linear_summary.png
3904d7dc98023f1f7cba82ee6cd25e90ba6d066ae7f543ce4e18ea8e25ce5bf4  cyclone_reference.pdf
468106d761d6c7bcf0cdc0ee1b52dace75878c773c5f9fc834a2474150ae6319  linear_summary.pdf
3a65c024a008c83be7a5f846465b33f9fff7961f50a3ac707e3f9e98a0ff5e24  tem_mismatch_table.csv
44706d8880b4d75b6509ef7b3fa5a586c51fa9c333d9126f03b359f27b6718bc  cyclone_mismatch_table.csv
665186718c420bfb0da1597070f98bfc11c5481a4b2219f4e73ee450a7db807a  etg_comparison.png
6f27772e1250aeb5185d9b5db6b51983f5519fc38900ebf720c25ade9c68d6c5  etg_comparison.pdf
c70a6e0202899af37c32a6acc9b440af9f940b9e3811c581f232cbf599bf7ba0  cyclone_comparison.pdf
d337ecbafddfa852536d0f701374d90a0cd981a6d491338c27761aeb074caf79  cyclone_comparison.png
35cf7ba3daf9d8c8067a41922516bceb69da940274a762f4efc08940e69df276  etg_trend_table.csv
```
