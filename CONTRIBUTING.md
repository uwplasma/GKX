# Contributing to SPECTRAX-GK

We welcome contributions from the community!
This document explains how to set up your development environment, run tests, and submit changes.

---

## 🛠 Development Setup

1. **Fork and clone** the repository:
   ```bash
   git clone https://github.com/<your-username>/SPECTRAX-GK.git
   cd SPECTRAX-GK
````

2. **Install in editable mode with dev dependencies**:

   ```bash
   pip install -e ".[dev]"
   ```

3. **Enable pre-commit hooks**:

   ```bash
   pre-commit install
   ```

---

## 🧪 Running Tests

We use [pytest](https://docs.pytest.org) for unit and regression tests.

Run all tests:

```bash
pytest -v
```

Run with coverage:

```bash
pytest --cov=spectraxgk --cov-report=term-missing
```

---

## 📏 Code Style & Quality

* **Linting & formatting**: [Ruff](https://github.com/astral-sh/ruff)

  ```bash
  ruff check .
  ruff format .
  ```

* **Type checking**: [MyPy](https://mypy.readthedocs.io)

  ```bash
  mypy .
  ```

* **Pre-commit hooks** automatically enforce these checks before commit.

---

## 🔄 Pull Requests

1. Create a feature branch:

   ```bash
   git checkout -b feature/my-new-feature
   ```

2. Commit changes with clear messages:

   ```bash
   git commit -m "feat: add DG solver for bump-on-tail"
   ```

3. Push and open a Pull Request:

   ```bash
   git push origin feature/my-new-feature
   ```

PRs are automatically tested via GitHub Actions.

---

## 🧑‍🔬 What to Contribute

* Bug fixes
* New physics modules (e.g., collisional operators)
* Additional diagnostics or visualization tools
* Documentation improvements
* Tests for uncovered parts of the code

---

## 📚 Resources

* [JAX docs](https://jax.readthedocs.io)
* [Diffrax](https://docs.kidger.site/diffrax/)
* [pytest](https://docs.pytest.org)
* [Ruff](https://docs.astral.sh/ruff/)

---

Thank you for contributing to **SPECTRAX-GK**!
