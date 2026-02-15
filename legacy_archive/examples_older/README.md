
# Examples

The `examples/` folder contains ready-to-run input files (`.toml`) showcasing standard plasma physics problems.
Run any example with:

```bash
spectraxgk --input examples/<file>.toml
````

---

## 📂 Example Cases

### `two_stream.toml`

* **Physics**: Two-stream instability with counter-propagating electron beams.
* **Discretization**: DG-in-x, Hermite-in-v.
* **Expected behavior**: Exponential growth of electric field energy, followed by nonlinear saturation.

### `landau_damping.toml`

* **Physics**: Linear Landau damping of Langmuir waves.
* **Discretization**: Fourier–Hermite.
* **Expected behavior**: Electric field decays exponentially at the Landau rate.

### `bump_on_tail.toml`

* **Physics**: Bump-on-tail instability (electron distribution with fast tail).
* **Discretization**: Fourier–Hermite.
* **Expected behavior**: Wave growth due to resonant particles, eventual plateau formation in velocity space.

### `ion_acoustic.toml`

* **Physics**: Ion acoustic waves with electrons + ions.
* **Discretization**: DG-in-x.
* **Expected behavior**: Ion acoustic oscillations with phase velocity ≈ electron thermal speed.

---

## ⚙️ Modifying Examples

You can:

* Change `sim.tmax` (in units of $1/\omega_p$)
* Change `grid.L_lambdaD` (box length in Debye lengths)
* Adjust species parameters (`temperature_eV`, `drift_c`, `n0`, etc.)
* Modify plotting options under `[plot]`

---

## 📊 Outputs

Each run produces:

* Diagnostic plots (energies, electric field vs. time)
* Distribution function snapshots
* Optional animations (`.mp4` / `.gif`)

---

## 🧩 Adding New Cases

To add a new example:

1. Copy an existing `.toml` file.
2. Modify parameters.
3. Document the case in this README.
4. Run it and check the diagnostics.

---

## ✅ Tips

* Keep `Nx` (grid points) large enough for resolution.
* Hermite modes `N` should capture the velocity space tail.
* For nonlinear runs, monitor both **energies** and **phase-space structures**.

---

These examples reproduce **classic kinetic plasma benchmarks** and serve as templates for new physics studies.
