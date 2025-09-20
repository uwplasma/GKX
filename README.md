# SPECTRAX-GK

[![PyPI](https://img.shields.io/pypi/v/spectraxgk.svg)](https://pypi.org/project/spectraxgk/)
[![Python](https://img.shields.io/pypi/pyversions/spectraxgk.svg)](https://pypi.org/project/spectraxgk/)
[![CI](https://github.com/uwplasma/SPECTRAX-GK/actions/workflows/ci.yml/badge.svg)](https://github.com/uwplasma/SPECTRAX-GK/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**SPECTRAX-GK** is a modern, differentiable solver for the multispecies **Vlasov–Poisson** system in 1D–1V, implemented with [JAX](https://github.com/google/jax).
It supports both **Fourier–Hermite** and **Discontinuous Galerkin (DG)** discretizations, runs on CPUs/GPUs/TPUs, and is designed for **plasma physics research, reproducibility, and education**.

---

## 🚀 Features

- **Two discretizations**
  - Fourier–Hermite pseudo-spectral solver
  - DG-in-x + Hermite-in-v solver
- **Linear and nonlinear physics**
- **Multi-species support** (electrons, ions, arbitrary charge & mass)
- **Units-aware input**
  - time in plasma periods ![wp](https://latex.codecogs.com/svg.image?1/\omega_p&bg=transparent)
  - length in Debye lengths ![ld](https://latex.codecogs.com/svg.image?\lambda_D&bg=transparent)
  - temperature in eV
  - drift velocity in fractions of ![c](https://latex.codecogs.com/svg.image?c&bg=transparent)
- **Differentiable & JIT-able**: compatible with JAX AD for optimization and ML workflows
- **Built-in diagnostics**: field & kinetic energy, electric field evolution, distribution functions
- **Publication-quality plots and animations**
- **Modern dev workflow**: Ruff (lint/format), MyPy (types), pytest (tests), pre-commit hooks, CI/CD

---

## 📖 Background

We solve the **Vlasov–Poisson equations** in one spatial and one velocity dimension (1D1V):

![Vlasov](https://latex.codecogs.com/svg.image?\frac{\partial%20f_s}{\partial%20t}%20+%20v\,\frac{\partial%20f_s}{\partial%20x}%20+%20\frac{q_s}{m_s}E(x,t)\,\frac{\partial%20f_s}{\partial%20v}%20=%200&bg=transparent)

with self-consistent electrostatics:

![Poisson](https://latex.codecogs.com/svg.image?\frac{\partial%20E}{\partial%20x}%20=%20\frac{1}{\epsilon_0}\sum_s%20q_s%20\int_{-\infty}^{\infty}%20f_s\,dv&bg=transparent)

where
- ![fs](https://latex.codecogs.com/svg.image?f_s(x,v,t)&bg=transparent): distribution function of species *s*
- ![qs](https://latex.codecogs.com/svg.image?q_s&bg=transparent), ![ms](https://latex.codecogs.com/svg.image?m_s&bg=transparent): species charge and mass
- ![E](https://latex.codecogs.com/svg.image?E(x,t)&bg=transparent): electric field

### Key plasma scales

Plasma frequency:

![wp](https://latex.codecogs.com/svg.image?\omega_p%20=%20\sqrt{\frac{n_0%20q^2}{\epsilon_0%20m}}&bg=transparent)

Debye length:

![ld](https://latex.codecogs.com/svg.image?\lambda_D%20=%20\sqrt{\frac{\epsilon_0%20k_B%20T}{n_0%20q^2}}&bg=transparent)

### Discretizations

- **Fourier–Hermite**: expand in Fourier (x) and Hermite (v); efficient for Landau damping, two-stream, bump-on-tail instabilities.
- **DG–Hermite**: discontinuous Galerkin in x + Hermite in v; more robust for nonlinear dynamics.

---

## 📦 Installation

### From PyPI (recommended)

```bash
pip install spectraxgk
````

### From source

```bash
git clone https://github.com/uwplasma/SPECTRAX-GK.git
cd SPECTRAX-GK
pip install -e ".[dev]"
```

The `dev` extras include `pytest`, `ruff`, and `mypy`.

---

## ⚡ Quick Start

Run a simulation from an example `.toml` config:

```bash
spectraxgk --input examples/two_stream.toml
```

This produces:

* Energy traces (kinetic + field)
* Electric field evolution
* Distribution function snapshots/animations

---

## ⚙️ Input Configuration

Inputs are given in `.toml` files. Example: **Two-stream instability**

```toml
[sim]
mode = "dg"              # "fourier" or "dg"
backend = "diffrax"      # "eig" or "diffrax"
tmax = 10.0              # simulation length in units of 1/ω_p
nt = 200
nonlinear = true

[grid]
L_lambdaD = 64           # box length in multiples of Debye length
Nx = 32

[hermite]
N = 24

[bc]
kind = "periodic"

[[species]]
name = "e_plus"
q = -1.0
n0 = 0.5*1e19
mass_base = "electron"
temperature_eV = 1.0
drift_c = +0.1

[[species]]
name = "e_minus"
q = -1.0
n0 = 0.5*1e19
mass_base = "electron"
temperature_eV = 1.0
drift_c = -0.1

[[species]]
name = "ions"
q = +1.0
n0 = 1.0*1e19
mass_base = "proton"
temperature_eV = 1.0
drift_c = 0.0
```

### Notes

* **Units**:

  * `tmax`: multiples of ![wp](https://latex.codecogs.com/svg.image?1/\omega_p\&bg=transparent)
  * `L_lambdaD`: multiples of ![ld](https://latex.codecogs.com/svg.image?\lambda_D\&bg=transparent)
  * `temperature_eV`: in eV
  * `drift_c`: fraction of ![c](https://latex.codecogs.com/svg.image?c\&bg=transparent)
* **Species**:

  * `mass_base = "electron"` or `"proton"` (scaled by `mass_multiple`)
  * densities given in SI (m⁻³)

---

## 📊 Output & Diagnostics

Diagnostics automatically produced:

* **Energies**: kinetic + field energy

  ![Wkin](https://latex.codecogs.com/svg.image?W_{\mathrm{kin},s}\(t\)%20=%20\frac{n_{0,s},m_s,v_{\mathrm{th},s}^2}{4\sqrt{2}}\int_0^{L}!\big\(C_{0,s}\(x,t\)%20+%20\sqrt{2},C_{2,s}\(x,t\)\big\),dx\&bg=transparent)

  ![Wfield](https://latex.codecogs.com/svg.image?W_{\mathrm{field}}\(t\)%20=%20\int_0^L\frac{E\(x,t\)^2}{2,\epsilon_0},dx\&bg=transparent)

* **Electric field** evolution in space & time

* **Distribution functions** per species

Plots are configurable under `[plot]` in the `.toml` file:

```toml
[plot]
nv = 257
vmin_c = -0.3
vmax_c = 0.3
fig_width = 10.0
fig_row_height = 2.2
fps = 30
dpi = 150
```

---

## 🧪 Development Workflow

### Code style

We use [Ruff](https://github.com/astral-sh/ruff) as **formatter and linter**:

```bash
ruff check .
ruff format .
```

### Type checking

We use [MyPy](https://mypy.readthedocs.io):

```bash
mypy .
```

### Tests

Run unit and regression tests with [pytest](https://docs.pytest.org):

```bash
pytest -v
```

Tests cover:

* Fourier/DG solver shapes
* Poisson operator assembly
* Unit conversions (Debye length, plasma frequency)
* Example configs (smoke tests)

### Pre-commit hooks

Install pre-commit once:

```bash
pre-commit install
```

Then checks (ruff, mypy, pytest) run automatically on commit.

---

## 🤖 Continuous Integration

* **GitHub Actions** run on every push/PR:

  * `ruff check` + `ruff format --check`
  * `mypy`
  * `pytest`
* **PyPI release**: pushing a git tag (`vX.Y.Z`) triggers an automatic build + upload.

---

## 📚 References

* Landau, L. D. *On the vibration of the electronic plasma*. J. Phys. USSR, 1946.
* Klimontovich, Y. L., Silin, V. P. *The Spectra of Systems of Interacting Particles*. JETP, 1960.
* Cheng, C. Z., Knorr, G. *The Integration of the Vlasov Equation in Configuration Space*. J. Comput. Phys., 1976.
* Boyd, J. P. *Chebyshev and Fourier Spectral Methods*. Dover, 2001.
* Shu, C.-W. *Discontinuous Galerkin Methods*. Springer, 2016.

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
