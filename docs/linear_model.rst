Linear Model And Derivations
============================

This page records the exact linear model currently benchmarked in GKX.
It is intended as the technical baseline for the nonlinear extension.

Scope
-----

Current linear validation scope:

- Cyclone base case (adiabatic electrons)
- ETG (two-species kinetic ions/electrons, electrostatic limit)
- KBM beta scan (electromagnetic, :math:`A_\parallel` enabled, :math:`B_\parallel` disabled)

Normalized equation
-------------------

For species :math:`s`, we evolve Laguerre-Hermite moments
:math:`G^{(s)}_{\ell m}(k_x, k_y, z, t)` in a flux tube:

.. math::

   \partial_t G_{\ell m}
   + \mathcal{S}_{\ell m}[H]
   + \mathcal{M}_{\ell m}[H]
   + \mathcal{C}_{\ell m}[H]
   + \mathcal{G}_{\ell m}[H]
   = \mathcal{D}_{\ell m}[\phi]
   + \mathcal{K}_{\ell m}[G],

where:

- :math:`\mathcal{S}` is parallel streaming (Hermite ladder)
- :math:`\mathcal{M}` is mirror coupling (:math:`b'(z)` geometry factor)
- :math:`\mathcal{C}` and :math:`\mathcal{G}` are curvature and grad-:math:`B` drifts
- :math:`\mathcal{D}` is diamagnetic drive from :math:`a/L_n, a/L_T`
- :math:`\mathcal{K}` contains collisions/hypercollisions/end damping

The field-coupled variable is

.. math::

   H_{\ell m}
   =
   G_{\ell m}
   + \frac{Z_s}{T_s} J_\ell \phi\,\delta_{m0}
   - \frac{Z_s v_{th,s}}{T_s} J_\ell A_\parallel\,\delta_{m1}
   + J_\ell^B B_\parallel\,\delta_{m0},

with :math:`J_\ell^B = J_\ell + J_{\ell-1}`.

Hermite-Laguerre projection
---------------------------

The reduced distribution is expanded as

.. math::

   g_s =
   \sum_{\ell=0}^{N_\ell-1}\sum_{m=0}^{N_m-1}
   G^{(s)}_{\ell m}\, \mathcal{L}_\ell(\mu)\,\mathcal{H}_m(v_\parallel),

with orthogonality

.. math::

   \langle \mathcal{H}_m \mathcal{H}_n \rangle = \delta_{mn},
   \qquad
   \langle \mathcal{L}_\ell \mathcal{L}_j \rangle = \delta_{\ell j}.

This gives sparse couplings:

- streaming: :math:`m \leftrightarrow m\pm1`
- curvature: :math:`m \leftrightarrow m, m\pm2`
- grad-:math:`B`: :math:`\ell \leftrightarrow \ell, \ell\pm1`
- mirror: mixed :math:`(\ell\pm1, m\pm1)` stencil

These sparsity patterns are implemented as fused tensor kernels in the RHS.

Gyroaverage and :math:`k_\perp`
-------------------------------

GKX uses the Laguerre gyroaverage

.. math::

   J_\ell(b) = \frac{1}{\ell!}\left(-\frac{b}{2}\right)^\ell e^{-b/2},
   \qquad
   b = k_\perp^2 \rho_s^2.

For s-alpha geometry, the metric coefficients :math:`gds2, gds21, gds22`
define

.. math::

   k_\perp^2
   =
   \left[
     k_y^2 gds2
     + 2 k_x k_y \hat{s}^{-1} gds21
     + (k_x \hat{s}^{-1})^2 gds22
   \right] B^{-2}(z),

including linked/NTFT corrections when enabled.

Field equations
---------------

Electrostatic limit:

.. math::

   \mathcal{Q}\,\phi
   =
   \sum_s Z_s n_s \sum_\ell J_\ell G_{\ell 0}^{(s)}.

Electromagnetic linear closure:

- coupled solve for :math:`(\phi, B_\parallel)` from quasineutrality and perpendicular Ampere
- :math:`A_\parallel` from parallel Ampere

The current KBM benchmark uses :math:`A_\parallel` on and :math:`B_\parallel` off.

Growth-rate/frequency extraction
--------------------------------

Two production paths are used:

1. Log-linear fit on complex mode signal :math:`s(t)`:

   .. math::

      s(t) \approx s_0 e^{(\gamma - i\omega)t}
      \Rightarrow
      \log |s| = \gamma t + c,\quad
      \arg s = -\omega t + c_\phi.

2. Instantaneous ratio diagnostic (for consistency checks):

   .. math::

      r_n = \frac{s_{n+1}}{s_n},
      \quad
      \gamma_n = \frac{\log |r_n|}{\Delta t},
      \quad
      \omega_n = -\frac{\arg(r_n)}{\Delta t}.

Windows are selected from intervals with sustained log-linear behavior and
finite amplitude support.

Numerical realization
---------------------

- perpendicular Fourier representation in :math:`(k_x, k_y)`
- field-aligned :math:`z` grid with linked boundary support
- JAX-fused RHS with cache-backed geometry/gyroaverage tensors
- diffrax and custom fixed-step integrators
- matrix-free Krylov/shift-invert for eigenvalue-focused scans

Benchmark contract
------------------

For all linear benchmark plots/tables:

- report :math:`\gamma,\omega` with explicit normalization
- publish mismatch tables by scan coordinate (:math:`k_y` or :math:`\beta_{ref}`)
- include per-case parameter tables (geometry, gradients, species, toggles, grid, resolution)
- use GX (s-alpha geometry) as the electromagnetic cross-code baseline for KBM

The nonlinear roadmap builds directly on this operator decomposition and
normalization contract.
