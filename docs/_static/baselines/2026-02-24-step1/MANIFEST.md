## Baseline Manifest (Step 1)

- Timestamp (UTC): `2026-02-24T04:06:23Z`
- Git commit (short): `111edf8`
- Git commit (full): `111edf8c6ee03256da2e116a3cc12a072c7bd4a3`

### Commands run

```bash
pytest -q --maxfail=1 --disable-warnings
mypy src/gkx
python -m sphinx -W -b html docs docs/_build/html
python tools/artifacts/make_tables.py --case cyclone --no-progress
python tools/artifacts/make_tables.py --case etg --no-progress
python tools/artifacts/build_linear_validation_artifacts.py figures --case cyclone --no-progress
python tools/artifacts/build_linear_validation_artifacts.py figures --case etg --no-progress
```

### Artifact checksums (SHA-256)

```text
2baf3b30b4c7ae5147a93f1574ebfb7235830ce1e0acb4f0793df11190b4329c  linear_summary.png
362143681942973c2dba31784a659b29efe2341f7333400fa961c2e6de20e153  cyclone_full_operator_scan_table.csv
8b3d507b12aabd06d0da15e38987cfe248b4835c92874dd34f3be3db27f3aa4c  etg_trend_table.csv
43a393aa499d14d9000a0491333d55938aeefd7c3300daffc014286732a1b328  cyclone_scan_convergence.csv
4b6ffa0ad6b9949edf4d29eaa8c6f21d886f3b64617b6dc07c903699cc6c0701  cyclone_comparison.png
4e6a00af6dd4f43138ab3ebb2290d814c970b6d51f7fa4ccdfe629a0472c8478  cyclone_comparison.pdf
44706d8880b4d75b6509ef7b3fa5a586c51fa9c333d9126f03b359f27b6718bc  cyclone_mismatch_table.csv
5f628f5cedd6b9d90cb681a95a6af8ac869e3de2e11d36ace69ccef9373c85bb  etg_comparison.pdf
665186718c420bfb0da1597070f98bfc11c5481a4b2219f4e73ee450a7db807a  etg_comparison.png
83761d30a582ca7c7a08560d6b82a0b22756712d0b644f5697fb9626d76bab76  cyclone_scan_table_highres.csv
83761d30a582ca7c7a08560d6b82a0b22756712d0b644f5697fb9626d76bab76  cyclone_scan_table_lowres.csv
d90530fca658535e95ac18f1422ae19485e7575f91601495103cbce553f3a521  etg_mismatch_table.csv
c041b981af494a6688a668f0b53f94c77efdd76e49621430cad6a8f743a462a7  cyclone_reference.pdf
```
