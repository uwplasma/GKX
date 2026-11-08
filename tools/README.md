# Tools

Repository manifests describe architecture, validation and benchmark policies.
`comparison/fixtures/` contains comparison inputs and reference conventions.

Run repository checks from the repository root:

```bash
python scripts/check.py --list
```

Keep generated inventories, summaries, traces and simulation outputs outside
version control. Preserve input decks, physics tables and analysis drivers
needed to reproduce checks.
