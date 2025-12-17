# SPECTRAX-GK

**SPECTRAX-GK** is a **JAX**-accelerated, differentiable **multispecies electrostatic gyrokinetic (GK)** solver using a **Fourier–Laguerre–Hermite (FLH)** moment representation. It is designed for:
- fast linear/nonlinear prototyping (slab / periodic boxes),
- multispecies kinetics (different `q_s, T_s, n0_s, rho_s, vth_s, nu_s, Upar_s`),
- stable pseudo-spectral nonlinear terms,
- optional conserving Lenard–Bernstein collisions,
- optional simple `∇∥ ln B` couplings in a 1D slab `B(z)` model.

The implementation is intentionally “research-grade readable”: the physics operators are explicit, modular, and JIT-friendly.

---

## Features

- **Multispecies FLH moments**: state `G_{s,ℓ,m}(k)` (complex Fourier coefficients; `ℓ` Laguerre, `m` Hermite).
- **Electrostatic quasineutrality** solved in Fourier space with **polarization**:
  \[
  \text{den}(k)\,\phi_k
  = \sum_s q_s n_{0s}\sum_\ell J_\ell(b_s)\,G_{s,\ell,m=0}(k),
  \]
  \[
  \text{den}(k)=\sum_s \frac{q_s^2 n_{0s}}{T_s}\left(1-\Gamma_0(b_s)\right) + \lambda_D^2 k^2.
  \]
  with \(\Gamma_0(b)=e^{-b}I_0(b)\) evaluated stably via `i0e(b)=e^{-b}I0(b)`.
- **Streaming operator** using the standard Hermite ladder structure (and optional `Upar` advection).
- **Optional `∇∥ ln B` couplings** via a simple slab profile \(B(z)=1+B_\epsilon\cos(kz)\) stored in real space and applied via z-only FFTs.
- **Nonlinear E×B** pseudo-spectral term with **2/3 de-aliasing**.
- **Conserving Lenard–Bernstein collisions** in the LH basis (sparse form), applied to `H` (see below).
- **Diffrax** time integration (`Tsit5` by default), with either fixed or adaptive steps.
- **Diagnostics** that avoid complex leaves when saving (Diffrax-friendly).

---

## Mathematical model (what the code is doing)

### 1) Representations: `G` and `H`

The code evolves a moment field `Gk`:
- `Gk` has shape `(Ns, Nl, Nh, Ny, Nx, Nz)` in *fftshifted* Fourier ordering.
- `Ns` species; `Nl` Laguerre; `Nh` Hermite; `(Ny,Nx,Nz)` Fourier grid.

A second field `Hk` is used inside operators and collisions:
\[
H_{s,\ell,m} = G_{s,\ell,m} + \frac{q_s}{T_s} J_\ell(b_s)\,\phi\,\delta_{m0}.
\]
This “\(h=g+(q/T)\phi\)” mapping is standard in gyrokinetics; in the code:
- `phi_k = solve_phi_quasineutrality_multispecies(Gk, params)`
- `Hk = build_Hk_from_Gk_phi(Gk, phi_k, params)`

### 2) Gyroaveraging and polarization

Define \(b_s(k)=k_\perp^2\rho_s^2\). For each species we compute:
- \(J_\ell(b_s)\) in Laguerre space (code: `Jl_s`)
- \(\Gamma_0(b_s)=e^{-b_s}I_0(b_s)\) (code: `Gamma0_s`)

Quasineutrality is solved in Fourier space with:
\[
\phi_k = \frac{\sum_s q_s n_{0s}\sum_\ell J_\ell(b_s)\,G_{s,\ell,0}(k)}{
\sum_s \frac{q_s^2 n_{0s}}{T_s}(1-\Gamma_0(b_s)) + \lambda_D^2 k^2}.
\]
A gauge fix enforces \(\phi_{k=0}=0\).

### 3) Streaming in Hermite moments (and optional `Upar`)

The Hermite ladder part uses:
\[
\mathcal{L}_m[H] = \sqrt{m+1}\,H_{m+1} + \sqrt{m}\,H_{m-1}.
\]
The RHS contribution uses the Fourier derivative convention \(\partial_z \to i k_z\).
In the code this is applied as:
- `kz_term = (-1j*kz)` for RHS sign convention,
- `stream_base = kz_term * vth_s * ladder`,
- plus a convective term `stream_U = kz_term * Upar_s * Hk`.

### 4) Optional `∇∥ ln B` coupling

When enabled, the code applies a model `gradlnB(z)` that depends only on `z`. The coupling mixes Laguerre and Hermite neighbors in the bracketed combination (implemented literally in `_with_gradB`), and multiplies by `gradlnB(z)` by transforming only along `z`:
1. `Ak(kz)` → `Az(z)` via `ifftz`,
2. multiply by `gradlnB(z)`,
3. transform back via `fftz`.

This avoids full 3D convolutions.

### 5) Nonlinear E×B (pseudo-spectral)

The code forms a species+Laguerre-dependent gyroaveraged potential:
\[
\phi_{s\ell}(k) = J_\ell(b_s)\,\phi_k,
\]
and constructs \(v_E\) in Fourier space:
\[
v_{Ex} = -\partial_y\phi, \quad v_{Ey} = \partial_x\phi,
\]
then computes the advection \(v_E\cdot\nabla G\) in real space and transforms back.
Dealiasing uses the 2/3 mask.

### 6) Collisions: conserving Lenard–Bernstein in LH moments

When `enable_collisions=True`, collisions are computed on `Hk`:
- `C = collision_lenard_bernstein_conserving_multispecies(Hk, params)`
- species-wise frequency `nu_s` scales the operator.

The implementation uses a conserving form requiring certain low-order velocity moments (computed from LH coefficients).

---

## Repository structure

- `spectraxgk/_initialization_multispecies.py`
  - parses user parameters,
  - builds k-grids, de-alias mask, species arrays,
  - computes gyroaverage factors (`Jl_s`, `Gamma0_s`),
  - builds quasineutrality denominators (`den_qn`, `den_h`),
  - builds IC `Gk_0` and `Hk_0`,
  - stores only **JAX-safe** items in returned params.
- `spectraxgk/_model_multispecies.py`
  - operators: conjugate symmetry, `solve_phi(...)`,
  - mapping `G ↔ H`,
  - nonlinear term, collisions,
  - `rhs_gk_multispecies(Gk, params, Nh, Nl) -> (dGk, phi_k)`.
- `spectraxgk/_simulation_multispecies.py`
  - packs complex state into a real vector for Diffrax,
  - runs time integration,
  - saves diagnostics and probes in real form.
- `spectraxgk/_hl_basis.py`
  - k-grids, `twothirds_mask`, Laguerre product tensor `alpha_tensor`,
  - `J_l_all`, and fftshift conjugate-index utilities.

---

## Installation

### 1) From source (recommended for development)

```bash
git clone https://github.com/uwplasma/SPECTRAX-GK.git
cd SPECTRAX-GK
python -m pip install -U pip
pip install -e .
```

### 2) Optional extras
```bash
pip install rich matplotlib pytest mypy
```

### 3) Enable x64 (recommended for accuracy / stability studies)

You can enable x64 either via environment variables or code:
```bash
SPECTRAX_X64=1 (your package switch)

JAX_ENABLE_X64=1 (standard JAX switch)
```

Example:

```bash
export SPECTRAX_X64=1
export JAX_ENABLE_X64=1
python -c "import jax; print(jax.config.read('jax_enable_x64'))"
```

--- 

## Quickstart: run an example

Create a script like examples/two_stream_1d.py (or use your existing example):

```bash
python examples/two_stream_1d.py
```

You can control:

grid: Nx,Ny,Nz

moments: Nl,Nh

physics toggles: enable_streaming, enable_nonlinear, enable_collisions, enable_gradB_parallel

species list (multispecies):

q, T, n0, rho, vth, nu, Upar

## Inputs

All inputs live in the input_parameters dict passed to:

out = simulation_multispecies(input_parameters=..., Nx=..., Ny=..., Nz=..., Nl=..., Nh=...)


Key fields:

Geometry/time:

Lx, Ly, Lz

t_max, dt, timesteps

Physics toggles:

enable_streaming

enable_nonlinear

enable_collisions

enforce_reality

enable_gradB_parallel

IC settings:

nx0, ny0, nz0, pert_amp

perturb_species (names/indices/weights; stripped out before returning params)

Debye regularization:

lambda_D (helps k_perp=0 solvability)

Grad-B slab profile:

B_eps, B_mode

Species list:

each dict may include: name, q, T, n0, rho, vth, nu, Upar

Outputs & diagnostics

If save="diagnostics", the returned dict contains:

time

phi_rms, max_abs_phi, max_abs_G

free-energy-like diagnostics:

W_free, W_field, per-species contributions W_s

spectral diagnostics:

Hermite spectrum E_m (shape (Ns, Nh))

It also returns params fields for reproducibility (kx_grid, Jl_s, Gamma0_s, etc.), plus your original species metadata.

Testing

Create and run tests:

pytest -q


Recommended tests:

linear collisionless: W_free conserved (with nonlinear off).

collisions on: W_free decays monotonically (or at least decreases).

enforce_reality: conjugate symmetry preserved.

perturb_species mask works (only intended species get initial perturbation).

Contributing

Contributions are welcome. Please:

Open an issue describing the physics/feature/bug.

Add or update tests under tests/.

Keep code JIT-safe: avoid strings or Python lists in parameter trees passed into JIT.

See CONTRIBUTING.md for the full guide.

Roadmap (suggested)

Electromagnetic extension (A_∥) and Ampère solve

More collision models (Dougherty, pitch-angle scattering, model Landau)

Better linear benchmarks (ion acoustic, ETG/ITG slab, Landau damping fits)

GPU-first performance tuning and profiling recipes

Citation

If you use this code in academic work, please cite:

The SPECTRAX-GK repository

Standard GK references (see docs/physics.md for a curated list)