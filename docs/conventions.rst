Conventions and conversion to GX, GS2 and stella
================================================

This page states GKX's units and sign conventions and shows how to translate
inputs and outputs to GX, GS2 and stella. :doc:`normalization` covers the
GKX-internal details (calibration contracts, diagnostic scaling, the stored
``bpar`` field).

Each entry below is marked with how it was checked:

- **code**: read from the GKX source (``src/gkx``) or from the other code's
  source or official documentation;
- **run**: confirmed by running the codes on the same case (the 2026-09-27
  cross-code record, ``plan/research/2026-09-27-xcode/REPORT.md``);
- **derived**: follows from the unit definitions but was not checked by a run;
- **unverified**: stated from the literature without a check here.

Reference quantities
--------------------

All four codes are local flux-tube codes. Each normalizes to a reference
species (mass :math:`m_r`, temperature :math:`T_r`, density :math:`n_r`), a
reference length :math:`L_r` and a reference field :math:`B_r`. They differ in
the reference speed:

.. math::

   v_r^{\mathrm{GKX,GX}} = \sqrt{T_r/m_r}, \qquad
   v_r^{\mathrm{GS2,stella}} = \sqrt{2T_r/m_r} = \sqrt{2}\, v_r^{\mathrm{GKX}}.

GKX sets ``vth = sqrt(T/m)`` and ``rho = sqrt(T m)/|Z|`` per species in
``gkx.operators.linear.params`` (**code**). GX uses the same definition. GS2
8.2.1 and stella v1.0 use :math:`\sqrt{2T/m}` (GS2 user manual,
*Normalisations*; **run** for both: the cross-code runs agree only with the
:math:`\sqrt{2}` below applied). The older ``norm_option`` switch described in
the GS2 manual does not exist in the 8.2.1 source.

The gyroradius and time unit follow from :math:`v_r`:

.. math::

   \rho_r = \frac{v_r}{\Omega_r},\quad \Omega_r = \frac{Z_r e B_r}{m_r c},
   \qquad t_{\rm unit} = \frac{L_r}{v_r},

so :math:`\rho_r^{\mathrm{GS2}} = \sqrt{2}\,\rho_r^{\mathrm{GKX}}` and
:math:`t_{\rm unit}^{\mathrm{GS2}} = t_{\rm unit}^{\mathrm{GKX}}/\sqrt{2}`.

**Reference length.** The shipped tokamak decks use :math:`L_r = a`, the
minor radius, and give the major radius as ``R0 = R/a = 2.77778`` (**code**).
GX's Cyclone decks do the same (``Rmaj = 2.77778``). GS2 normalizes all lengths
to :math:`a` (manual), and in its s-alpha model enters :math:`a/R` through
``epsl = 2a/R`` and ``pk = 2a/(qR)``. stella's Miller namelist takes ``rmaj``
in units of :math:`a`. Nothing forces :math:`L_r = a`: a deck with
``R0 = 1`` measures lengths in :math:`R`, and then gradients and rates are
per :math:`R`. Always check which length a published number uses.

**Reference field.** GX and GKX Miller geometry use :math:`B_r = B_t(R_{\rm geo})`
(the ``R_geo`` key); GS2 uses the toroidal field at :math:`R_a`, the centre of
the flux surface. The two agree when ``R_geo`` equals ``Rmaj``, as in every
Cyclone deck here (**code**, GX and GS2 docs).

Conversion formulas
-------------------

Write :math:`\hat X` for a normalized quantity. From the definitions above,
with the same :math:`L_r`, :math:`B_r` and reference species in both codes:

.. list-table::
   :header-rows: 1
   :widths: 28 42 30

   * - Quantity
     - GKX (= GX) from GS2 / stella
     - Status
   * - binormal wave number
     - :math:`\hat k_y^{\rm GKX} = \hat k_y^{\rm GS2}/\sqrt{2}`
     - run
   * - radial wave number
     - :math:`\hat k_x^{\rm GKX} = \hat k_x^{\rm GS2}/\sqrt{2}`
     - derived
   * - ballooning angle
     - :math:`\theta_0` unchanged
     - code
   * - frequency, growth rate
     - :math:`\hat\omega^{\rm GKX} = \sqrt{2}\,\hat\omega^{\rm GS2}`,
       :math:`\hat\gamma^{\rm GKX} = \sqrt{2}\,\hat\gamma^{\rm GS2}`
     - run
   * - time
     - :math:`\hat t^{\rm GKX} = \hat t^{\rm GS2}/\sqrt{2}`
     - derived
   * - collision frequency
     - :math:`\hat\nu^{\rm GKX} = \sqrt{2}\,\mathtt{vnewk}^{\rm GS2}`
     - derived
   * - heat and particle flux
     - :math:`\hat Q^{\rm GKX} = 2\sqrt{2}\,\hat Q^{\rm GS2}`,
       :math:`\hat\Gamma^{\rm GKX} = 2\sqrt{2}\,\hat\Gamma^{\rm GS2}`
     - derived
   * - velocity-grid extent
     - :math:`v/v_t^{\rm GKX} = \sqrt{2}\, v/v_r^{\rm stella}`
       (``vpa_max = 3`` in stella is :math:`3\sqrt{2}\,v_t`)
     - derived
   * - :math:`\beta`, ``tprim``, ``fprim``, :math:`q`, :math:`\hat s`
     - unchanged
     - code

The flux factor comes from the gyro-Bohm unit
:math:`Q_{gB} = n_r T_r v_r (\rho_r/L_r)^2`, which carries :math:`v_r^3`, so
:math:`Q_{gB}^{\rm GS2} = 2\sqrt{2}\,Q_{gB}^{\rm GKX}`. The collision
frequency is a rate, so it converts like :math:`\gamma`. GS2's manual defines
``vnewk`` with :math:`v_{thr} = \sqrt{2T/m}`; GX reads ``vnewk`` into its
``nu_ss`` with no rescaling (``parameters.cu``), so GX's ``vnewk`` is already
in GKX units.

GX needs no conversion: GKX was built to reproduce GX, and the certified GKX
eigenvalues agree with GX to 0.3% or better on the Cyclone decks (README,
*Benchmarks*).

Gradients and geometry parameters
---------------------------------

``tprim`` and ``fprim`` are :math:`L_r/L_T = -L_r\, d\ln T/dr` and
:math:`L_r/L_n`, positive for profiles that peak on axis. The names and signs
are the same in all four codes (**code**). Convert to :math:`R/L_T` with
:math:`R/L_T = (R/a)(a/L_T)`: Cyclone's ``tprim = 2.49`` is
:math:`R/L_T = 6.92`.

Safety factor and shear are :math:`q` and :math:`\hat s = (r/q)\,dq/dr`, with
:math:`r` the half-diameter (GS2 ``irho = 2``, its default). GKX's keys are
``q`` and ``s_hat``; GX, GS2 and stella Miller use ``qinp`` and ``shat``.

In the circular s-alpha model GKX builds (``gkx.geometry.analytic``,
**code**)

.. math::

   B = \frac{1}{1+\epsilon\cos\theta},\qquad
   \nabla_\parallel = \frac{1}{qR_0},\qquad
   \omega_{\rm cv} = \omega_{\nabla B} =
   \frac{\cos\theta + (\hat s\theta - \alpha\sin\theta)\sin\theta}{R_0},

and :math:`k_x(\theta) = k_{x0} - (\hat s\theta - \alpha\sin\theta)k_y`.
GX's s-alpha branch (``geometry.cu``) has the same expressions, with
:math:`\alpha` entered as ``shift``; GKX calls it ``alpha``. GS2's
``gbdrift``/``cvdrift`` are twice GKX's for the same geometry, because of the
:math:`\sqrt{2}` in :math:`\rho_r` (**run**, through the matched growth
rates). The sign of GS2's s-alpha ``shift`` relative to :math:`\alpha` was not
checked (**unverified**).

The Miller keys follow GX: ``rhoc``, ``R0`` (GX ``Rmaj``), ``R_geo``,
``shift``, ``akappa``, ``akappri``, ``tri``, ``tripri``, ``betaprim``. stella
spells two of them ``kappa`` and ``kapprim``, and ``triprim``; GS2 uses
``rmaj`` and ``r_geo``.

Parallel coordinate
-------------------

GKX uses :math:`z \in [-\pi Z_p, \pi Z_p)` with :math:`Z_p = 2n_{\rm period}-1`
and ``ntheta`` points per :math:`2\pi` (``GridConfig``). GS2's ``theta`` and
stella's ``zed`` span the same :math:`2n_{\rm period}-1` poloidal turns for the
same ``nperiod``. The linked boundary condition is ``boundary = "linked"`` in
GKX and ``boundary_option = "linked"`` in GS2 and stella.

The ballooning angle is :math:`\theta_0 = k_x/(\hat s k_y)` in GKX (from the
:math:`k_x(\theta)` above), GS2 (``kt_grids_box.f90``) and stella
(``geometry_miller.f90``). It is a ratio of wave numbers, so the
:math:`\sqrt{2}` cancels.

Perpendicular grid
------------------

:math:`k_y = n/y_0` and :math:`k_x = n/x_0` with :math:`L_y = 2\pi y_0` and
:math:`L_x = 2\pi x_0` in units of :math:`\rho_r` (``gkx.core_grid``, GX
documentation). GKX's ``ky`` output is :math:`k_y\rho_i` with
:math:`\rho_i = \sqrt{T_i m_i}/(eB)`; GS2's ``aky`` is :math:`k_y\rho_i` with
the :math:`\sqrt{2}` gyroradius. Cyclone at ``ky = 0.3`` in GKX is
``aky = 0.4243`` in GS2 and stella.

The ``ky``-resolved spectra in GKX's NetCDF output hold one row of each
:math:`(k_y,-k_y)` pair; GX writes the pair sum, twice these values for
:math:`k_y>0` (:doc:`outputs`, **code**).

Fourier and frequency signs
---------------------------

Fields are expanded as :math:`\phi = \sum_{\mathbf k} \hat\phi_{\mathbf k}
e^{i(k_x x + k_y y)}` (``jnp.fft.fftfreq`` ordering), and growth rates are
fitted to :math:`\hat\phi \propto e^{(\gamma - i\omega)t}`
(``fit_growth_rate``, **code**). With this time convention the Cyclone ITG mode
has :math:`\omega > 0` in all four codes: GKX and GX (tracked reference CSV),
GS2 (:math:`\hat\omega = 0.153` at ``aky = 0.4243``, Miller) and stella
(:math:`\hat\omega = 0.178`) (**run**). Positive :math:`\omega` is the
ion-diamagnetic direction in these runs. Electron modes (TEM, ETG) have
:math:`\omega < 0`. No sign flip is needed between the codes for ``omega``.

Fields
------

GKX evolves :math:`\hat\phi = (e\phi/T_r)(L_r/\rho_r)`, the standard
gyrokinetic ordering. The GS2 and stella fields carry their own
:math:`\rho_r`, so :math:`\hat\phi^{\rm GKX} = \sqrt{2}\,\hat\phi^{\rm GS2}`
at equal physical amplitude (**derived**). Linear amplitudes are arbitrary, so
this only matters for nonlinear spectra. The normalizations of
:math:`A_\parallel` and :math:`\delta B_\parallel` in GS2 and stella were not
compared here (**unverified**); GKX's ``bpar`` convention is in
:doc:`normalization`.

:math:`\beta_r = 8\pi n_r T_r/B_r^2` in GKX (``[physics] beta``), GX
(``beta``, Inputs documentation), GS2 and stella (``beta``). It contains no
velocity, so it needs no conversion.

Velocity space
--------------

GKX and GX expand the distribution in Hermite polynomials of
:math:`v_\parallel/v_t` and Laguerre polynomials of :math:`\mu B/T`, with
:math:`v_t = \sqrt{T/m}`, truncated at ``Nm`` and ``Nl`` (:doc:`theory`). GS2
uses energy and pitch-angle grids (``negrid``, ``ngauss``); stella uses
:math:`(v_\parallel, \mu)` grids (``nvgrid``, ``nmu``) with extents
``vpa_max`` and ``vperp_max`` in units of :math:`\sqrt{2T/m}`. There is no
one-to-one map between these resolutions; converge each code separately.

GKX's model collision operator takes ``nu`` per species plus the
``[collisions]`` Hermite/Laguerre coefficients, following GX. The collision
model itself differs between codes (GS2 and stella default to their own
operators), so matching :math:`\hat\nu` does not guarantee matching damping.

Input keys
----------

.. list-table::
   :header-rows: 1

   * - Quantity
     - GKX TOML
     - GX TOML
     - GS2 namelist
     - stella namelist
   * - binormal wave number
     - ``[run] ky``
     - ``y0``, ``ny``
     - ``aky``
     - ``aky_min``/``aky_max``
   * - T, n gradients
     - ``[[species]] tprim``, ``fprim``
     - ``tprim``, ``fprim``
     - ``tprim``, ``fprim``
     - ``tprim``, ``fprim``
   * - collision frequency
     - ``[[species]] nu``
     - ``vnewk``
     - ``vnewk``
     - ``vnewk``
   * - safety factor, shear
     - ``[geometry] q``, ``s_hat``
     - ``qinp``, ``shat``
     - ``qinp``, ``shat``
     - ``qinp``, ``shat``
   * - inverse aspect ratio
     - ``[geometry] epsilon``
     - ``eps``
     - ``eps``
     - via ``rhoc``/``rmaj``
   * - reference beta
     - ``[physics] beta``
     - ``beta``
     - ``beta``
     - ``beta``
   * - T_i/T_e (adiabatic e)
     - ``[physics] tau_e``
     - ``tau_fac``
     - ``tite`` (``&knobs``)
     - ``tite``
   * - parallel extent
     - ``[grid] ntheta``, ``nperiod``
     - ``ntheta``, ``nperiod``
     - ``ntheta``, ``nperiod``
     - ``nzed``, ``nperiod``

GX keys are those of the GX *Inputs* documentation. ``tau_e`` in GKX is the
temperature ratio used by the Boltzmann closure (:doc:`normalization`); check
the direction of the ratio in each code before copying a value other than 1.

Worked example: Cyclone base case
---------------------------------

Circular geometry, adiabatic electrons, :math:`L_r = a`, one mode at
:math:`k_y\rho_i = 0.3` in GKX units.

GKX (``examples/01_linear_tokamak/case_full.toml``, abridged):

.. code-block:: toml

   [[species]]
   tprim = 2.49
   fprim = 0.8
   nu = 0.0

   [geometry]
   q = 1.4
   s_hat = 0.8
   epsilon = 0.18
   R0 = 2.77778

   [grid]
   ntheta = 32
   nperiod = 2
   boundary = "linked"

   [run]
   ky = 0.3

GS2 s-alpha (``aky`` is :math:`\sqrt{2}\times 0.3`):

.. code-block:: fortran

   &kt_grids_single_parameters  aky = 0.4242640687  theta0 = 0.0 /
   &theta_grid_parameters  eps = 0.18  epsl = 0.72  shat = 0.8
                           pk = 0.5142857  ntheta = 32  nperiod = 2 /
   &theta_grid_knobs  equilibrium_option = "s-alpha" /
   &species_parameters_1  z = 1.0  mass = 1.0  dens = 1.0  temp = 1.0
                          tprim = 2.49  fprim = 0.8  vnewk = 0.0 /

stella circular Miller:

.. code-block:: fortran

   &geometry_miller  rhoc = 0.5  qinp = 1.4  shat = 0.8
                     rmaj = 2.77778  rgeo = 2.77778  shift = 0.0
                     kappa = 1.0  tri = 0.0 /
   &kxky_grid_range  naky = 1  aky_min = 0.4242640687
                     aky_max = 0.4242640687 /
   &species_parameters_1  tprim = 2.49  fprim = 0.8 /

GX uses the GKX numbers unchanged (``tprim = 2.49``, ``qinp = 1.4``,
``shat = 0.8``, ``eps = 0.18``, ``Rmaj = 2.77778``, :math:`k_y` from ``y0``).

Measured growth rates, circular Miller, :math:`k_y\rho_i = 0.3`
(cross-code record, section 2):

.. list-table::
   :header-rows: 1

   * - Code
     - :math:`\hat\gamma` (own units)
     - :math:`\hat\gamma` (GKX units)
   * - GKX, certified eigenpair
     - 0.1259
     - 0.1259
   * - GS2 8.2.1
     - 0.0887
     - 0.1255
   * - stella v1.0
     - 0.1236
     - 0.1748

GS2 agrees with GKX to 0.3% after the :math:`\sqrt{2}`. stella is 1.4 times
higher on this case; the record traces the gap to stella's mirror term, not
to a unit conversion.
