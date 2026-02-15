# Physics

This page documents the operators implemented in `spectraxgk/_model_multispecies.py`
and how they connect to the FLH moment representation.

> **Scope**: electrostatic, periodic box, Fourier in space, Laguerre–Hermite in velocity,
> with optional pseudo-spectral nonlinearity, model collisions, and a simple grad-B parallel coupling.

---

## State variables

The evolved state is:

- `G_{s,ℓ,m}(k)` (stored as `Gk`), complex Fourier coefficient in fftshifted ordering
- Species index `s=1..Ns`
- Laguerre index `ℓ=0..Nl-1`
- Hermite index `m=0..Nh-1`

We also form:

\[
H_{s,\ell,m} = G_{s,\ell,m} + \frac{q_s}{T_s} J_\ell(b_s)\,\phi\,\delta_{m0}.
\]

This mapping is implemented by `build_Hk_from_Gk_phi(...)`.

---

## Gyroaveraging and \(b_s\)

Define \(b_s(k)=k_\perp^2\rho_s^2\).
The gyroaverage factors \(J_\ell(b_s)\) (Laguerre-space) and \(\Gamma_0(b_s)=e^{-b_s}I_0(b_s)\)
enter quasineutrality and polarization.

---

## Quasineutrality (electrostatic closure)

The code solves in Fourier space:

\[
\text{den}(k)\,\phi_k
= \sum_s q_s n_{0s}\sum_{\ell} J_\ell(b_s)\,G_{s,\ell,0}(k),
\]

\[
\text{den}(k)=\sum_s \frac{q_s^2 n_{0s}}{T_s}\left(1-\Gamma_0(b_s)\right)+\lambda_D^2 k^2.
\]

Gauge: \(\phi_{k=0}=0\).  
Dealias: \(\phi_k\) is zeroed outside the 2/3 mask.

---

## Parallel streaming (Hermite ladder)

Let the Hermite ladder operator be:

\[
\mathcal{L}_m[H] = \sqrt{m+1}\,H_{m+1} + \sqrt{m}\,H_{m-1}.
\]

Streaming contributes (up to conventions/signs):

\[
\partial_t G \sim - v_{th,s}\,\partial_z \mathcal{L}_m[H] \;-\; U_{\|,s}\,\partial_z H \;+\; \cdots
\]

In Fourier, \(\partial_z \to i k_z\). The implementation uses `(-i k_z)` for the RHS convention.

---

## Optional \( \nabla_\parallel \ln B \) coupling

When enabled, a slab profile is used:

\[
B(z) = 1 + B_\epsilon \cos\left(\frac{2\pi B_{\text{mode}}}{L_z}z\right),
\qquad
\nabla_\parallel \ln B = \frac{B'(z)}{B(z)}.
\]

The operator mixes neighboring Laguerre/Hermite coefficients in the bracketed combination implemented in code,
then multiplies by \(\nabla_\parallel \ln B(z)\) via z-only FFTs.

---

## Nonlinear E×B (pseudo-spectral)

A gyroaveraged potential per species and Laguerre index is formed:

\[
\phi_{s\ell}(k)=J_\ell(b_s)\phi_k.
\]

Then:

\[
v_{Ex}=-\partial_y \phi_{s\ell}, \quad v_{Ey}=\partial_x \phi_{s\ell},
\qquad
\mathcal{N}[G]= v_E \cdot \nabla G.
\]

The Laguerre triple-product tensor `alpha_kln` couples Laguerre indices in the nonlinear term.

---

## Collisions: conserving Lenard–Bernstein model

When enabled, collisions are computed as:

\[
C_s = \nu_s \,\mathcal{C}[H_s],
\]

where \(\mathcal{C}\) is a conserving Lenard–Bernstein operator implemented in LH moments.
The conserving form requires computing low-order velocity moments from the LH coefficients,
then injecting correction terms to enforce conservation properties.

---

## Suggested validation suite

1. **Collisionless, nonlinear off**: `W_free` should be nearly conserved (numerically).
2. **Collisions on**: `W_free` should decay relative to collisionless run.
3. **Reality enforcement**: conjugate symmetry in Fourier is preserved.
4. **Species masking**: `perturb_species` only excites intended species.