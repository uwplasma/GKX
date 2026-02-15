# Developer Guide

This guide explains how to extend SPECTRAX-GK while keeping it JAX- and JIT-friendly.

## Core principles

1. **Keep params JIT-safe**
   - Do not store strings or lists-of-strings in `params` used inside `jit`.
   - Convert user-facing specs (like `perturb_species`) into numeric arrays up front.

2. **Avoid complex leaves in Diffrax args**
   - Pack the complex state into a real vector.
   - Keep `args=params_solver` real-only (strip complex caches).

3. **Dealias consistently**
   - Apply 2/3 mask to inputs and outputs of nonlinear operators.

4. **Optional conjugate symmetry**
   - For “real fields”, enforce `Ak(k)=conj(Ak(-k))` in fftshifted ordering.

## Adding a new physics term

### Example: add a linear drive term

1. Implement a function `drive_term(Gk, params)` in `_model_multispecies.py`.
2. Add a toggle like `enable_drive` in initialization.
3. In `rhs_gk_multispecies`, add:

```python
Dk = lax.cond(enable_drive, lambda _: drive_term(Gk, params), lambda _: zeros_like(Gk), operand=None)
dGk = dGk + Dk
