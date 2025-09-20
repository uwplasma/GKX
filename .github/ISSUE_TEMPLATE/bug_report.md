---
name: 🐞 Bug report
about: Something isn’t working as expected
labels: bug
---

## Description
A clear and concise description of the bug.

## Repro Steps
1. Command(s) you ran (paste exact input):
```

spectraxgk --input examples/<file>.toml

```
2. What happened vs what you expected?

## Logs / Traceback
<details>
<summary>Expand to show full logs</summary>

```

# paste the full error here

````
</details>

## Environment
- OS:
- Python version: `python --version`
- spectraxgk version: `python -c "import spectraxgk, importlib.metadata as m; print(m.version('spectraxgk'))"`
- JAX version: `python -c "import jax; print(jax.__version__)"`

## Config snippet
If applicable, paste relevant TOML sections:

```toml
# minimal input that reproduces the bug
````

## Additional context

Anything else we should know?
