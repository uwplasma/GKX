<p align="center">
  <img src="https://raw.githubusercontent.com/uwplasma/SPECTRAX-GK/refs/heads/main/docs/images/spectraxgk_logo.svg" alt="SPECTRAX-GK" width="100">
</p>

# SPECTRAX-GK

[![PyPI](https://img.shields.io/pypi/v/spectraxgk.svg)](https://pypi.org/project/spectraxgk/)
[![Python](https://img.shields.io/pypi/pyversions/spectraxgk.svg)](https://pypi.org/project/spectraxgk/)
[![CI](https://github.com/uwplasma/SPECTRAX-GK/actions/workflows/ci.yml/badge.svg)](https://github.com/uwplasma/SPECTRAX-GK/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# SPECTRAX-GK


Hermite–Laguerre, JAX-native linear gyrokinetics (slab, straight-B) with a Lenard–Bernstein collision operator. Designed for **modern JAX stacks** (Equinox, Diffrax), **JIT**, **CPU/GPU**, **CLI + Python drivers**, **simple compressed outputs**, and a **clean path to optimization** (Optax/JAXopt). Fourier and DG discretizations are both supported **architecturally**; the initial implementation ships with a 1D Fourier-in-\(z\) streaming + collisions model.


**Status**: minimal working linear slab demo (streaming + LB collisions), with scaffolding for quasi-neutrality and other physics. Intended as a pedagogical and extensible reference.


## Quick start


```bash
# 1) install (editable)
pip install -e .


# 2) run from CLI
spectraxgk --input examples/linear_slab.toml


# 3) load & plot in Python
python - <<
from spectraxgk.post import load_result, plot_energy
res = load_result("outputs/linear_slab_run.npz")
print(res)
plot_energy(res)
```

## Why this stack?
- **Equinox** (`eqx.Module`) makes physics operators **PyTrees**, so they JIT/vmap/pmap cleanly and can be optimized later.
- **Diffrax** gives high-quality ODE/SDE solvers with backprop/adjoints for differentiable simulation.
- **Optax/JAXopt** hooks let you do parameter inference, controller design, or closure calibration.


## Input format (TOML)
See `examples/linear_slab.toml`. You can run via CLI or import from Python.


## Outputs
We write a single compressed NumPy file (`.npz`) containing:
- `C` : complex Hermite–Laguerre coefficients, shape `(nt, Nn, Nm)`
- `t` : time grid
- `meta` : dict-like metadata (config + grid + git hash if available)
- `kpar`, `nu`, `vth`: key scalars copied for convenience


This keeps post-processing dead simple:
```python
import numpy as np
C = np.load("outputs/linear_slab_run.npz")
print(C.files) # ['C','t','meta','kpar','nu','vth']
```


## Roadmap
- Add electrostatic quasi-neutrality closure and J0/Gamma0 couplings (Laguerre algebra is already scaffolded)
- Batch multiple \(k_\parallel, k_\perp\) and species
- Optional Zarr/xarray writer
- DG in x/y with modal operators

## Citation guidance
This scaffold follows the Hermite–Laguerre moment approach used in the literature (e.g., Mandell 2018; Frei et al. 2021–2023) and draws architectural ideas from GPU-native GX. See references in the bottom of this file.


---


## Developer notes
- All key structs are `eqx.Module` PyTrees with **static fields** for non-array metadata.
- We separate: **grid** → **basis** → **operators** → **model** → **solver**.
- The RHS is vectorized and JIT’d; hooks exist for diagnostics & callbacks.
- CPU/GPU switch is automatic with JAX; enable x64 via env var or TOML.


---


## References
- Mandell, Dorland & Landreman, *Laguerre–Hermite pseudo-spectral velocity formulation of gyrokinetics*, JPP (2018).
- Frei et al., *Advanced linearized GK collision operators via a moment approach*, JPP (2021).
- Frei et al., *Moment-based approach to flux-tube linear GK*, arXiv:2210.05799 (2022); JPP (2023).
- GX code docs: Fourier–Hermite–Laguerre spectral GK.
- Diffrax and Equinox documentation.


---

## 🧑‍💻 Contributing

We welcome issues and pull requests! Please:

1. Fork the repo
2. Install dev deps: `pip install -e ".[dev]"`
3. Run checks: `ruff check . && pytest && mypy .`
4. Submit a PR

---

## 📜 License

MIT License © 2025 [UWPlasma](https://github.com/uwplasma)
