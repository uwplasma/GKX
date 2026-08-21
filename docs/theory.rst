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
For non-twisting flux tubes (NTFT), GKX employs an ``m0`` and
``deltaKx`` formulation compatible with GX, which modifies the effective
:math:`k_\perp` and drift terms using the same twist factor and linking indices.

Hermite-Laguerre velocity space
-------------------------------

The perturbed distribution is expanded in a Hermite (parallel velocity) and
Laguerre (magnetic moment) basis. For a single species, the expansion is

.. math::

   g(\mathbf{k}, \theta, v_\parallel, \mu) =
   \sum_{\ell=0}^{N_\ell-1} \sum_{m=0}^{N_m-1}
   G_{\ell m}(\mathbf{k}, \theta)
   L_\ell(b) H_m(v_\parallel),

with the gyroaverage factor

.. math::

   J_\ell(b) = e^{-b/2} L_\ell(b),

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

Field-aligned streaming representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To maintain compatibility with audited reference benchmarks, GKX supports
applying the parallel derivative to a gyrokinetic variable that includes the
explicit field terms but omits the full :math:`H_{\ell m}` correction at
``m>1``. This is achieved by defining

.. math::

   \tilde{G}_{\ell m} = G_{\ell m}
   + \frac{Z_s}{T_s} J_\ell \phi\,\delta_{m0}
   - \frac{Z_s v_{th}}{T_s} J_\ell A_\parallel\,\delta_{m1}
   + J_\ell^B B_\parallel\,\delta_{m0},

and then applying the parallel derivative to :math:`\tilde{G}` before the
Hermite ladder. This preserves the validated linked-boundary operator contract
used by the imported-geometry and benchmark lanes.

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
