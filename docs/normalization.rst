Normalization
=============

This section documents the normalization conventions used in GKX and
the calibration parameters that scale drift/drive terms for benchmark
comparisons.

Canonical normalization contract
--------------------------------

Benchmark-family normalization values live in
``gkx.diagnostics.normalization`` as
:class:`gkx.diagnostics.normalization.NormalizationContract` records, the single
source of case defaults:

.. list-table:: Canonical per-case normalization contracts
   :header-rows: 1

   * - Case key
     - ``rho_star``
     - ``omega_d_scale``
     - ``omega_star_scale``
     - ``diagnostic_norm_default``
   * - ``cyclone``
     - ``1.0``
     - ``1.0``
     - ``1.0``
     - ``none``
   * - ``etg``
     - ``1.0``
     - ``1.0``
     - ``1.0``
     - ``none``
   * - ``kinetic`` (``kinetic_itg`` alias)
     - ``1.0``
     - ``1.0``
     - ``1.0``
     - ``none``
   * - ``tem``
     - ``1.0``
     - ``1.0``
     - ``1.0``
     - ``none``
   * - ``kbm``
     - ``1.0``
     - ``1.0``
     - ``1.0``
     - ``none``

Every shipped case contract uses unit scale factors: no non-unity
``omega_d_scale`` or ``omega_star_scale`` calibration factors ship with GKX
(see ``src/gkx/diagnostics/normalization.py``).

The benchmark constants of the script API (``CYCLONE_OMEGA_D_SCALE`` and the
others in ``gkx.benchmarking_shared``) read these contracts, so a calibration
lives in one module.

Dimensionless units
-------------------

We evolve a dimensionless gyrokinetic system normalized to ion thermal
quantities. The parallel streaming term uses the normalized thermal velocity
:math:`v_{th}` and the flux-tube coordinate :math:`z` (theta-like). The
perpendicular wave numbers are normalized by the ion gyro-radius
:math:`\rho_i`, while the distribution function and potential use the standard
gyrokinetic scaling:

.. math::

   \tilde{\phi} = \frac{e \phi}{T_i}, \qquad
   \tilde{\omega} = \frac{\omega}{v_{th}/R_0}.

Here :math:`v_{th} = \sqrt{T/m}`, not :math:`\sqrt{2T/m}`, and correspondingly
:math:`\rho = \sqrt{T m}/|q|` (``gkx.operators.linear.params``). GX uses the
same definition, so GKX and GX results compare directly with no conversion.
Codes in the GS2 family, including stella, define :math:`v_{th} = \sqrt{2T/m}`;
against those, wave numbers and rates convert as

.. math::

   k_{y,\mathrm{GS2}} = \sqrt{2}\, k_{y,\mathrm{GKX}}, \qquad
   \gamma_{\mathrm{GS2}} = \gamma_{\mathrm{GKX}}/\sqrt{2}, \qquad
   \omega_{\mathrm{GS2}} = \omega_{\mathrm{GKX}}/\sqrt{2},

with the same factor on the time axis and a factor of two on the drift
coefficients. Gradient inputs such as ``tprim`` and ``fprim``, and
:math:`\beta`, are convention-free. Frequency is the channel that exposes a
mistaken convention: omitting the wave-number conversion costs about 30 per
cent in a growth rate near its peak, where the curve is flat, but 54 per cent
in the frequency.

The stored compressional field is the local fractional perturbation

.. math::

   \mathtt{bpar}=\frac{\delta B_\parallel}{\rho_* B(z)},\qquad
   \frac{\delta B_\parallel}{\rho_* B_N}=\mathtt{bmag}(z)\,\mathtt{bpar}.

Thus perpendicular Ampere's law uses the local beta
:math:`\beta_{\rm ref}/\mathtt{bmag}^2`, while the magnetic Hamiltonian and
fluxes use ``bpar`` directly because
:math:`\mu\delta B_\parallel=(\mu B)(\delta B_\parallel/B)`.  `GX equations
(15)--(16) <https://arxiv.org/html/2209.06731v3>`_ provide the magnetic
Hamiltonian background, but the paper's reference-field normalization does not
by itself disambiguate the stored field for variable :math:`B`.  The local
convention is stated explicitly in the GX implementation correction
`2e417afe62f4ad730fae005fb8927337e1cbefa3
<https://bitbucket.org/gyrokinetics/gx/commits/2e417afe62f4ad730fae005fb8927337e1cbefa3>`_.
GKX checks this convention for compressional pressure balance, the magnetic
Hamiltonian, and the particle-flux field factor.  The particle-flux check
reuses the production spatial quadrature weights, so it is not an independent
validation of that quadrature.  Heat flux and the complete electromagnetic
free-energy budget remain open.

For the Cyclone base case we take the reference length :math:`L_{ref}=a` so
that the input gradients are expressed as
:math:`a/L_T` and :math:`a/L_n`. With :math:`R_0 = R/a`, this means
:math:`R/L_T = (R_0)\,(a/L_T)` and similarly for :math:`R/L_n`.

Kinetic species conventions
---------------------------

GKX supports multi-species kinetic systems. Species-dependent arrays
carry the charge, mass, density, temperature, and gradient inputs:

- ``charge_sign``: :math:`Z_s` (e.g., :math:`+1` for ions, :math:`-1` for electrons).
- ``temp`` / ``mass`` / ``density``: normalized to the reference species.
- ``tz``: :math:`Z_s / T_s` coupling used in the field terms.
- ``tprim`` / ``fprim``: normalized gradients for each species,
  :math:`a/L_T` and :math:`a/L_n`. These are the only gradient units the
  operator consumes. The legacy names ``R_over_LTi`` / ``R_over_Ln`` are
  accepted as aliases with a ``DeprecationWarning``. Use the :math:`R/L = R_0 \, (a/L)` conversion above
  when quoting a result against literature values.

For adiabatic closures, ``tau_e`` provides the ratio between the kinetic
species temperature and the Boltzmann species temperature.

Field-aligned grid parameters
-----------------------------

For the Cyclone base case we use a field-aligned grid with:

.. math::

   y_0 = 20,\qquad n_\theta = 32,\qquad n_{period} = 2.

In GKX these map to ``GridConfig(y0=20, ntheta=32, nperiod=2)``, which
sets:

.. math::

   L_y = 2 \pi y_0,\qquad
   z \in [-\pi Z_p, \pi Z_p),\qquad
   Z_p = 2 n_{period} - 1.

The reduced scan tables and regression tests use ``Nx=1, Ny=24, Nz=96`` on this
grid to match the discrete ky set used in the reference CSV.

Spectral grids
--------------

The spectral grid uses the compressed Fourier conventions of the tracked
reference data. The perpendicular wave numbers are defined as

.. math::

   k_x = \frac{n_x}{x_0}, \qquad k_y = \frac{n_y}{y_0},

with ``x0 = Lx / (2π)`` and ``y0 = Ly / (2π)``. The parallel wave number is

.. math::

   k_z = \frac{n_z}{Z_p},

where :math:`Z_p` sets the field-line length
(:math:`z \in [-\pi Z_p, \pi Z_p)`), and :math:`k_z` is defined *without* the
``gradpar`` factor. These definitions are implemented by
``gkx.core_grid.build_spectral_grid`` and are consistent with the
audited reference-grid convention.

The midplane index used by the reference growth-rate diagnostic corresponds to
``z_index = Nz//2 + 1``, matching the audited benchmark kernel logic when ``Nz > 1``.

Perpendicular normalization
---------------------------

The reference diagnostic contract defines the perpendicular metric as
:math:`k_\perp^2/B^2` before the Laguerre gyroaverage. GKX matches it with:

- ``kperp2_bmag = True`` (include the :math:`B^{-2}` factor in :math:`k_\perp^2`)
- ``bessel_bmag_power = 0`` (no extra :math:`B` scaling inside the Bessel argument)

The Cyclone base case defaults use this setting, and
``scripts/comparison/compare_gx_rhs_terms.py compare`` assumes it.

Sign conventions
----------------

The growth-rate fitting in :func:`gkx.diagnostics.growth_rates.fit_growth_rate` assumes

.. math::

   s(t) \sim \exp((\gamma - i \omega)\, t),

so:

- ``gamma > 0`` indicates instability.
- ``omega`` is obtained from the negative phase slope.

This is consistent across time-integration and Krylov post-processing paths.
For scan tables and figures, values are reported in the same sign convention as
the solver output unless an explicit diagnostic normalization is requested.

Normalization parameters
------------------------

The linear operator exposes three normalization parameters that influence the
drift/drive terms:

- ``rho_star``: scales :math:`k_x` and :math:`k_y` in the drift and drive
  terms.
- ``omega_d_scale``: scales curvature/grad-:math:`B`/mirror couplings.
- ``omega_star_scale``: scales the diamagnetic drive.

In code, ``rho_star`` multiplies the Fourier grids inside
:func:`gkx.operators.linear.cache_builder.build_linear_cache`, while ``omega_d_scale`` and
``omega_star_scale`` enter directly in :func:`gkx.operators.linear.rhs.linear_rhs_cached`.

Diagnostic normalization mode
-----------------------------

Benchmark runners expose ``diagnostic_norm`` and route it through
``gkx.diagnostics.normalization.apply_diagnostic_normalization``:

- ``none``: return raw solver ``(gamma, omega)``.
- ``rho_star``: multiply reported ``(gamma, omega)`` by ``rho_star``.

This affects reporting only; it does not alter the RHS/operator.

The unified runtime schema defaults to ``diagnostic_norm = "rho_star"`` so that
out-of-the-box reports match the tracked benchmark normalization. Set
``diagnostic_norm = "none"`` in the TOML or runtime config to recover raw
solver outputs.

Diagnostic scaling
------------------

The reference diagnostics apply fixed factors in a few places that depend on
the storage convention (for example real-FFT Nyquist handling or per-unit-time
damping). The runtime schema exposes two diagnostic scale factors:

- ``flux_scale``: multiplicative factor applied to the reported heat/particle
  fluxes (default ``1.0`` for the tracked comparison convention).
- ``wphi_scale``: multiplicative factor applied to ``Wphi`` (default ``1.0``;
  no shipped configuration overrides it).

Both affect reporting only; they do not alter the RHS/operator. They record
the exact comparison settings behind benchmark plots.

The reference end-damping defaults are ``damp_ends_amp = 0.1`` and
``damp_ends_widthfrac = 0.125``. The damping kernel interprets
``damp_ends_amp`` as a per-step strength when the linear caller supplies a
timestep; otherwise it remains a rate, including on nonlinear routes.
Only an isolated Euler damping update removes exactly that local fraction;
other RK schemes apply their stability polynomial. See :doc:`operators` for
the equations and for why a linear timestep scan does not refine one fixed
operator.

Defaults (model parameters):

- ``rho_star = 1.0`` (model default)
- ``omega_d_scale = 1.0`` (model default)
- ``omega_star_scale = 1.0`` (model default)

The regression tables record all three, so every tabulated result carries
the normalization it was computed with.

Programmatic usage
------------------

.. code-block:: python

   from gkx.diagnostics.normalization import get_normalization_contract

   contract = get_normalization_contract("etg")
   # contract.omega_d_scale == 1.0
   # contract.omega_star_scale == 1.0
