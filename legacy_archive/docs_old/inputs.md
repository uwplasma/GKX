# Inputs (TOML)

All simulations in **SPECTRAX-GK** are configured through a single **TOML file**.  
This keeps runs reproducible, sharable, and easy to document in publications.

---

## 🔹 Example: Two-stream instability

```toml
[sim]
mode = "dg"              # "fourier" or "dg"
backend = "diffrax"      # "eig" (linear eigenmodes) or "diffrax" (time integration)
tmax = 10.0              # simulation time (in units of 1/ω_p)
nt = 200                 # number of time steps
nonlinear = true         # false = linearized, true = full nonlinear

[grid]
L_lambdaD = 64           # box length in multiples of Debye length
Nx = 32                  # number of grid points (DG: cells; Fourier: modes)
debye_species = "e_plus" # which species sets λ_D, ω_p scaling (default: first electron)

[hermite]
N = 24                   # number of Hermite modes in velocity space

[bc]
kind = "periodic"        # boundary conditions in x

[[species]]
name = "e_plus"
q = -1.0                 # charge number (e.g. -1 = electron)
n0 = 0.5*1e19            # density in m⁻³ (safe math allowed, e.g. 0.5*1e19)
mass_base = "electron"   # "electron" or "proton"
mass_multiple = 1.0      # e.g. 1836 for proton as multiple of electron mass
temperature_eV = 1.0     # species temperature in eV
drift_c = +0.1           # drift velocity as fraction of c
```

---

## 🔹 Sections Explained

### `[sim]` – Simulation controls

* **mode**:

  * `"fourier"` → pseudo-spectral Fourier–Hermite solver (efficient, best for linear instabilities).
  * `"dg"` → DG-in-x + Hermite-in-v solver (robust for nonlinear turbulence).
* **backend**:

  * `"eig"` → compute linear eigenmodes only.
  * `"diffrax"` → time-evolve distribution & fields.
* **tmax**: runtime in **plasma periods** (multiples of $1/\omega_p$). Internally converted to seconds.
* **nt**: number of time steps (time resolution).
* **nonlinear**: switch between linearized vs full nonlinear dynamics.

---

### `[grid]` – Physical box & resolution

* **L\_lambdaD**: domain size in **Debye lengths** of a chosen reference species. Converted to meters.
* **Nx**: number of grid cells or modes in x.
* **debye\_species**: optional selector (string name or index) for which species defines $\lambda_D$ and $\omega_p$. Defaults to first negative-charge species (electrons).

---

### `[hermite]` – Velocity-space resolution

* **N**: number of Hermite functions used. Larger N → more velocity-space fine structure, but costlier.

---

### `[bc]` – Boundary conditions

* `"periodic"`: standard for plasma instabilities.
* `"dirichlet"` / `"neumann"`: supported for testing.

---

### `[[species]]` – Define each species

You can add as many as you want.

* **name**: label (e.g. "electrons", "ions").

* **q**: charge number (e.g. -1 for electrons, +1 for singly ionized protons).

* **n0**: density in **m⁻³**.

* **mass\_base**: `"electron"` or `"proton"`.

* **mass\_multiple**: scale factor. (e.g. `1.0` for electrons, `1836` for protons).

* **temperature\_eV**: thermal energy in eV. Internally → thermal speed via

  ![vth](https://latex.codecogs.com/svg.image?v_{\text{th}}=\sqrt{\frac{2k_BT}{m}})

* **drift\_c**: drift speed normalized to c.

* Extra knobs: perturbation amplitude, seed, initial mode $k$, collisions.

---

## 🔹 Units & Conversions

* **tmax**: given in multiples of $1/\omega_p$, converted to seconds.
* **L\_lambdaD**: given in multiples of $\lambda_D$, converted to meters.
* **temperature\_eV**: converted to Joules → $v_{\text{th}}$.
* **drift\_c**: multiplied by $c$.
* **n0**: always in SI (m⁻³).

✅ You can safely use arithmetic like `0.5*1e19` or `2*pi`. The loader evaluates these.

---

## 📝 Tips

* Small $N$ (\~12) and $Nx$ (\~16) are enough for **linear tests**.
* For nonlinear turbulence, increase to $N \sim 48-64$, $Nx \sim 64-128$.
* Always check **L/λ\_D** and **tmax·ω\_p** in the printed summary.
* Make species consistent: equal densities for quasineutrality, unless testing instability.
