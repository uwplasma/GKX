# SPECTRAX-GK

SPECTRAX-GK is a **JAX**-accelerated, differentiable solver for **multispecies electrostatic gyrokinetics** using a **Fourier–Laguerre–Hermite (FLH)** representation.

It is built to be:
- **fast** (JIT + vectorized species operators),
- **transparent** (operators match the math),
- **extensible** (new physics can be added without rewriting the whole code),
- **reproducible** (all parameters stored in a JAX-friendly tree).

## What you can do with it

- Linear collisionless dynamics (streaming-only, nonlinear off) with free-energy conservation checks.
- Add **conserving model collisions** (Lenard–Bernstein form).
- Add a simple **grad-B parallel coupling** using a slab `B(z)` profile.
- Run multispecies systems with different drifts `Upar_s` (e.g., two-stream setups).

## Where to start

- **Install**: see [Install](install.md)
- **Run your first example**: see [Quickstart](quickstart.md)
- **Understand the equations**: see [Physics](physics.md)
- **Configure runs**: see [Inputs](inputs.md) and [Outputs & Diagnostics](outputs.md)

## Design philosophy

- Keep the parameter tree **JIT-safe**: no strings inside `params` used in compiled functions.
- Keep complex arrays out of Diffrax `args` and saved diagnostics (pack/unpack state as real).
- Make each physics operator locally testable.
