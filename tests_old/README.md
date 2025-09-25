# Tests in SPECTRAX-GK

The `tests/` folder contains **unit tests**, **regression tests**, and **physics benchmark tests**.
They ensure that the solvers remain stable, correct, and scientifically reliable.

---

## 🧩 Test Types

### 1. Unit Tests
- Check **small, isolated pieces of code** (e.g., Fourier transforms, Hermite recursion).
- Example: `test_util_kgrid.py` checks that the k-grid is built consistently.

### 2. Regression Tests
- Run **short versions of example problems** and check key outputs against references.
- Example: `test_fourier_linear_smoke.py` verifies that the Fourier solver produces fields and coefficients of the correct shape.

### 3. Physics Benchmarks
- Validate the code against **classic plasma physics problems**:
  - **Landau damping**: Verify exponential decay of electric field energy.
  - **Two-stream instability**: Verify exponential growth of perturbation energy.
  - **Ion acoustic waves**: Verify correct dispersion relation.

---

## ⚙️ Running Tests

Run all tests:
```bash
pytest -v
````

Run specific test files:

```bash
pytest tests/test_util_kgrid.py
```

Run with coverage:

```bash
pytest --cov=spectraxgk --cov-report=term-missing
```

---

## 📈 Continuous Integration

On each pull request, GitHub Actions will:

1. Install the package and dependencies.
2. Run **Ruff** (lint + format).
3. Run **MyPy** (static type checking).
4. Run **pytest**.

A PR must pass all checks before merging.

---

## 🧪 Adding New Tests

1. Create a file `tests/test_<topic>.py`.
2. Use [pytest](https://docs.pytest.org) conventions:

   * Functions prefixed with `test_`.
   * Use `assert` for conditions.
3. Keep tests short (<1s runtime if possible).
4. For benchmarks:

   * Use simplified physics parameters (small grid, short runtime).
   * Assert qualitative trends (e.g., `E_field` decreases).

---

## 🔑 Philosophy

* **Unit tests**: fast, precise, deterministic.
* **Regression tests**: catch breaking changes in APIs and outputs.
* **Physics benchmarks**: ensure the solver continues to reproduce well-known plasma physics.

---

> ✅ If you add a new solver or diagnostic, please add at least one test!
