Normalization
=============

This section documents the normalization conventions used in GKX and
the calibration parameters that scale drift/drive terms for benchmark
comparisons.

Canonical normalization contract
--------------------------------

GKX now centralizes benchmark-family normalization values in
``gkx.diagnostics.normalization`` via :class:`gkx.diagnostics.normalization.NormalizationContract`.
This is the single source of truth for case defaults:

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
     - ``0.4``
     - ``0.8``
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
     - ``0.8``
     - ``none``

These contracts are consumed by benchmark constants for the stable script API
(``CYCLONE_OMEGA_D_SCALE``, etc.), so existing scripts keep working while all
new calibration updates flow through one module.

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
  operator consumes; the fields were called ``R_over_LTi`` / ``R_over_Ln``
  until the names were corrected, and the old names still work with a
  ``DeprecationWarning``. Use the :math:`R/L = R_0 \, (a/L)` conversion above
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
-------------------------

GKX’s explicit benchmark integrator uses the compressed Fourier conventions required by the tracked reference data.
The perpendicular wave numbers are defined as

.. math::

   k_x = \frac{n_x}{x_0}, \qquad k_y = \frac{n_y}{y_0},

with ``x0 = Lx / (2π)`` and ``y0 = Ly / (2π)``. The parallel wave number is

.. math::

   k_z = \frac{n_z}{Z_p},

where :math:`Z_p` sets the field-line length
(:math:`z \in [-\pi Z_p, \pi Z_p)`), and :math:`k_z` is defined *without* the
``gradpar`` factor. These definitions are implemented by
``gkx.core.grid.build_spectral_grid`` and are consistent with the
audited reference-grid convention.

The midplane index used by the reference growth-rate diagnostic corresponds to
``z_index = Nz//2 + 1``, matching the audited benchmark kernel logic when ``Nz > 1``.

Perpendicular normalization
--------------------------------------

The reference diagnostic contract defines the perpendicular metric as :math:`k_\perp^2/B^2` before applying
the Laguerre gyroaverage. To match that convention in GKX:

- ``kperp2_bmag = True`` (include the :math:`B^{-2}` factor in :math:`k_\perp^2`)
- ``bessel_bmag_power = 0`` (no extra :math:`B` scaling inside the Bessel argument)

The Cyclone base case defaults follow this benchmark setting, and the
``compare_gx_rhs_terms.py compare`` assumes the same normalization.

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
-----------------------------

The reference diagnostics apply fixed factors in a few places that depend on the storage
convention (e.g. real-FFT nyquist handling or per-unit-time damping). The
runtime schema therefore exposes light-weight diagnostic scale factors:

- ``flux_scale``: multiplicative factor applied to the reported heat/particle
  fluxes (default ``1.0`` for the tracked comparison convention).
- ``wphi_scale``: multiplicative factor applied to ``Wphi`` (default ``1.0``;
  the Cyclone nonlinear comparison config uses ``1.155``).

These are reporting-only knobs; they do not alter the RHS/operator. They are
intended to document exact comparison settings used for benchmark plots.

The reference end-damping defaults are ``damp_ends_amp = 0.1`` and
``damp_ends_widthfrac = 0.125``. The damping kernel interprets
``damp_ends_amp`` directly as a rate; the runtime no longer rescales it by the
timestep.

Defaults (model parameters):

- ``rho_star = 1.0`` (model default)
- ``omega_d_scale = 1.0`` (model default)
- ``omega_star_scale = 1.0`` (model default)

These parameters are surfaced in the regression tables so that future
normalization refinements can be tracked in a reproducible way.

Programmatic usage
------------------

.. code-block:: python

   from gkx.diagnostics.normalization import get_normalization_contract

   contract = get_normalization_contract("etg")
   # contract.omega_d_scale == 1.0
   # contract.omega_star_scale == 1.0
