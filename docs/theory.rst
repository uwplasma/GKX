Theory
======

For a full operator-level derivation tied to the implemented code paths
(normalization, Hermite-Laguerre projection, field equations, and growth-rate
diagnostics), see :doc:`linear_model`. For the explicit implemented operator
set, collisions, hypercollisions, nonlinear brackets, and parameter-to-source
mapping, see :doc:`operators`.

Gyrokinetic ordering
--------------------

GKX targets the low-frequency, strongly magnetized regime where the
characteristic fluctuation frequency is small compared to the ion cyclotron
frequency. In this limit, the phase-space dynamics can be reduced to a
five-dimensional gyrokinetic system for the non-adiabatic part of the
distribution function. Classic derivations of the gyrokinetic equation can be
found in Frieman & Chen (1982) and Antonsen & Lane (1980). [FC82]_ [AL80]_

Flux-tube model
---------------

We employ a field-aligned, local flux-tube model in which the perpendicular
spatial dependence is represented spectrally and the parallel coordinate is
resolved along a field line. This approximation underlies the Cyclone base case
benchmark commonly used in gyrokinetic validation studies. [Dimits00]_

The default boundary condition is a linked (twist-and-shift) flux tube, so the
parallel derivative couples Fourier modes across adjacent :math:`k_x` indices.
``GridConfig(non_twist=True)`` selects the non-twisting flux tube (NTFT): the
linear cache shifts :math:`k_x` along the field line by
:math:`\delta k_x = k_y f_{twist}(z) + \rho_* m_0/x_0`, with
:math:`f_{twist} = \hat{s}\,g_{ds21}/g_{ds22}` and :math:`m_0` the nearest-integer
link index, in both :math:`k_\perp` and the drifts
(``gkx.operators.linear.cache_builder._build_ntft_kperp_and_drift_arrays``).

Hermite-Laguerre velocity space
-------------------------------

The perturbed distribution is expanded in a Hermite (parallel velocity) and
Laguerre (magnetic moment) basis. For a single species, the expansion is

.. math::

   \frac{g(\mathbf{k}, \theta, v_\parallel, \mu)}{F_M} =
   \sum_{\ell=0}^{N_\ell-1} \sum_{m=0}^{N_m-1}
   G_{\ell m}(\mathbf{k}, \theta)
   \mathcal{L}_\ell(\mu B/T)\mathcal{H}_m(v_\parallel/v_{th}),

where the normalized velocity basis includes the chosen Laguerre sign and
Hermite normalization. Its **projected gyroaverage coefficient**, not the
Laguerre polynomial evaluated at the wavenumber, is

.. math::

   J_\ell(b) = e^{-b/2}\frac{(-b/2)^\ell}{\ell!},

where :math:`b = k_\perp^2 \rho^2`. This Laguerre-Hermite formulation is detailed
by Mandell, Dorland & Landreman (2017). [MDL17]_

Field solve and gyrokinetic variable
------------------------------------

GKX supports electrostatic and electromagnetic linear closures. For
electrostatic runs, quasineutrality is solved in Fourier space for
:math:`\phi`, with an optional adiabatic response controlled by
:math:`\tau_e = T_i/T_e`:

.. math::

   \left(\tau_e + \sum_s \frac{Z_s^2 n_s}{T_s}\left[1-\sum_{\ell} J_{\ell}^2\right]\right) \phi
   = \sum_s Z_s n_s \sum_{\ell} J_{\ell} G_{\ell, m=0}.

Electromagnetic runs solve the coupled quasineutrality/perpendicular-Ampere
system for :math:`(\phi, B_\parallel)` and then obtain :math:`A_\parallel` from
parallel Ampere’s law. The gyrokinetic variable is

.. math::

   H_{\ell m} = G_{\ell m}
   + \frac{Z_s}{T_s}\,J_\ell \phi \, \delta_{m0}
   - \frac{Z_s v_{th,s}}{T_s}\,J_\ell A_\parallel \, \delta_{m1}
   + J_{\ell}^{B}\,B_\parallel \, \delta_{m0},

with :math:`J_{\ell}^{B} = J_{\ell} + J_{\ell-1}`. These relations match the
Laguerre-Hermite pseudo-spectral form used in the gyrokinetic literature.

The nonzonal field-system test in ``tests/unit/operators/test_terms_fields.py``
assembles a separate dense :math:`3\times3` system from
`GX (arXiv v3), (32)--(34) <https://arxiv.org/html/2209.06731v3>`_ at :math:`B=1`.
It uses two kinetic species with unequal temperatures/masses, finite FLR, and
separate density/current/perpendicular-moment excitations; float32/64 values
and normalized residuals must agree. Variable-:math:`B` cases test GKX's
additional :math:`B^{-2}` perpendicular-Ampere convention only: its physical
normalization remains unqualified by those paper equations alone.  The
independent physical-energy identification below supplies the additional field
normalization check.  Zonal/gauge tests remain separate.

The companion geometry-to-field test checks the constant-:math:`B`, nonzonal quadratic implied by
`GX (arXiv v3), (12), (16), and (32)--(34)
<https://arxiv.org/html/2209.06731v3>`_.  For the retained basis it independently
forms

.. math::

   S_\phi=\sum_{s\ell}n_sZ_sJ_\ell G_{s\ell0},\quad
   S_A=\sum_{s\ell}n_sZ_sv_{th,s}J_\ell G_{s\ell1},\quad
   S_B=\sum_{s\ell}n_sT_sJ_\ell^B G_{s\ell0},

and

.. math::

   F_{\rm src}=\frac12\sum_{s\ell m}n_sT_s|G_{s\ell m}|^2
   +\frac12\operatorname{Re}(\phi^*S_\phi-A_\parallel^*S_A+B_\parallel^*S_B).

`Howes et al. (2006), (B19)--(B20)
<https://arxiv.org/pdf/astro-ph/0511812>`_ identify the physical fluctuation
energy as particle entropy plus magnetic energy.  Applying quasineutrality and
the corrected local-:math:`B` field equations in `published GX, (2.12) and
(4.22)--(4.25)
<https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/gx-a-gpunative-gyrokinetic-turbulence-code-for-tokamak-and-stellarator-design/2C4BB81955E7E749B95B8B8141E997FA>`_
gives, mode by mode,

.. math::

   F_{\rm phys}=\frac12\sum_{s\ell m}n_sT_s|H_{s\ell m}|^2
   -\frac12\sum_s\frac{n_sZ_s^2}{T_s}|\phi|^2
   +\frac{k_{\perp,N}^2|A_\parallel|^2+B^2|\mathtt{bpar}|^2}
          {\beta_{\rm ref}}.

The fields here are dimensionless.  GX Appendix A, Tables 4--5 normalize
:math:`k_\perp` by :math:`\rho_{\rm ref}^{-1}`, :math:`A_\parallel` by
:math:`\rho_*\rho_{\rm ref}B_N`, and :math:`\delta B_\parallel` by
:math:`\rho_*B_N`; GKX stores
:math:`\mathtt{bpar}=\delta B_\parallel/(\rho_*B(z))`.  Hence
:math:`\delta B_\perp/(\rho_*B_N)=k_{\perp,N}A_\parallel` and
:math:`\delta B_\parallel/(\rho_*B_N)=B\,\mathtt{bpar}`.  Together with
:math:`\beta_{\rm ref}=8\pi n_{\rm ref}T_{\rm ref}/B_N^2` and
:math:`v_{th}=\sqrt{T/m}`, this gives the two :math:`1/\beta_{\rm ref}` magnetic
terms above (and the :math:`\beta_{\rm ref}/2` in Ampere's law), with no hidden
:math:`\sqrt{2}` factor.

For the :math:`B^{-2}` cache convention tested at varying :math:`B`,
:math:`k_{\perp,N}^2=\mathtt{kperp2}\,B^2`.  The test constructs this expression
independently and checks :math:`F_{\rm src}=F_{\rm phys}` first at constant
:math:`B=1`, then with the varying-:math:`B` volume measure; dropping either
magnetic :math:`B^2` factor must fail.  The constant-:math:`B` check exercises
both cache conventions but cannot distinguish them.  This is a per-mode
finite-FLR, nonzonal, periodic identity; zonal/gauge and general multi-mode
Hermitian weighting remain open.
Within that envelope the componentwise JAX convention is
:math:`\operatorname{conj}(\nabla_G F_{\rm src})=n_sT_sH_s`.
The symmetric truncated Hermite ladder and skew-adjoint periodic derivative
also give the tested streaming-only identity

.. math::

   \operatorname{Re}\sum_{s\ell mz}n_sT_sH_{s\ell m}^*
   (\partial_tG_{s\ell m})_{\rm stream}=0.

This excludes curved-geometry, drive, collision, boundary-damping and nonlinear
terms. The runtime ``Wg + Wphi + Wapar`` diagnostic is a
distinct monitoring quantity and contains no :math:`B_\parallel` contribution.

For varying :math:`B`, the companion term-level gate follows `Mandell et al.
(2018), (4.4)--(4.6)
<https://doi.org/10.1017/S0022377818000041>`_: it contracts streaming plus
mirror forcing with :math:`n_sT_sH_s^*` and the field-line volume weight
:math:`J(z)\propto 1/(|\mathrm{gradpar}|B)`.  The periodic boundary contribution
vanishes, and the finite-resolution defect converges once aliased products are
resolved.  Differentiating the reduced source quadratic with its self-consistent
fields gives the same weighted :math:`n_sT_sH_s^*` contraction.  With equilibrium
gradients, collisions, hyper-dissipation, and end damping disabled, the gate
also contracts the complete assembled linear right-hand side: streaming,
mirror, curvature, and grad-:math:`B`.  Every retained term is nonzero, while
wrong volume/sign and missing-imaginary-unit controls produce nonzero exchange.
The physical identification and variational identity do not establish a full
budget: nonlinear transfer, heat-flux normalization, sources, sinks, and
time-discretization remain outside this instantaneous gate.

Linear gyrokinetic operator
---------------------------

In the linear model, the Hermite-Laguerre moments evolve according to a
drift/mirror operator,

.. math::

   \frac{\partial G_{\ell m}}{\partial t}
   + v_{\mathrm{th}}\,\mathcal{L}_m[H]
   + v_{\mathrm{th}}\,b^\prime(\theta)\,\mathcal{M}_{\ell m}[H]
   = -i Z/T\,\bigl(c_v \mathcal{C}_m[H] + g_b \mathcal{G}_\ell[H]\bigr)
   + i k_y \phi \,\mathcal{D}_{\ell m},

where :math:`\mathcal{L}_m` is the Hermite streaming ladder and
:math:`b^\prime(\theta)` is the parallel magnetic field gradient used in the
mirror force. The curvature (``cv``) and grad-:math:`B` (``gb``) drift couplings
are encoded in :math:`\mathcal{C}_m` and :math:`\mathcal{G}_\ell`. Explicitly,

.. math::

   \mathcal{C}_m[H] =
   \sqrt{(m+1)(m+2)} H_{\ell, m+2}
   + (2m+1) H_{\ell m}
   + \sqrt{m(m-1)} H_{\ell, m-2},

.. math::

   \mathcal{G}_\ell[H] =
   (\ell+1) H_{\ell+1, m}
   + (2\ell+1) H_{\ell m}
   + \ell H_{\ell-1, m},

.. math::

   \mathcal{M}_{\ell m}[H] =
   -\sqrt{m+1}\,(\ell+1) H_{\ell, m+1}
   -\sqrt{m+1}\,\ell H_{\ell-1, m+1}
   +\sqrt{m}\,\ell H_{\ell, m-1}
   +\sqrt{m}\,(\ell+1) H_{\ell+1, m-1}.

The diamagnetic drive term :math:`\mathcal{D}_{\ell m}` follows a Laguerre
formulation with explicit :math:`a/L_n` and :math:`a/L_T` dependence,
including a separate coupling in :math:`m=2` for temperature-gradient drive.

Streaming acts on :math:`H`: the parallel derivative is applied to
:math:`H_{\ell m}` (field terms included at :math:`m = 0, 1`) before the
Hermite ladder, periodically or along the linked twist-and-shift chains
(``gkx.terms.linear_terms.streaming_contribution``).

Nonlinear E×B and flutter terms
-------------------------------

The nonlinear gyrokinetic equation adds the :math:`E\times B` bracket and the
electromagnetic flutter coupling. In GKX the nonlinear contribution is

.. math::

   \left(\frac{\partial g}{\partial t}\right)_\mathrm{NL}
   = -\left\{ \langle \chi \rangle, g \right\}
   - v_{th}\,\left(\sqrt{m+1}\,\{\langle A_\parallel \rangle, g\}_{m+1}
   + \sqrt{m}\,\{\langle A_\parallel \rangle, g\}_{m-1}\right),

with the Poisson bracket

.. math::

   \{g, \chi\} = \frac{\partial g}{\partial x}\frac{\partial \chi}{\partial y}
   - \frac{\partial g}{\partial y}\frac{\partial \chi}{\partial x}.

The gyrokinetic potential includes the perpendicular magnetic perturbation
through

.. math::

   \chi = J_\ell \phi + J_\ell^B B_\parallel,

so the nonlinear operator naturally splits into :math:`E\times B`,
:math:`B_\parallel`, and flutter contributions. The implementation in
:mod:`gkx.terms.nonlinear` supports standard gyrokinetic normalization:
gradients are computed with FFTs in :math:`x,y`, the bracket is evaluated in
real space, and the result is filtered by the de-alias mask before returning to
spectral space.
