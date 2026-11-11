Validation
==========

Validation separates operator identities, numerical convergence and physical
comparison. Passing one layer does not establish the others. Run checks on
the source revision, precision and backend used for the intended calculation.

Operator and integration checks
-------------------------------

The test suite covers spectral derivatives, basis and quadrature identities,
field solves, collision invariants and dissipation, explicit and implicit
time integration, restart continuity and diagnostic normalization. Tests for
optimized or parallel routes compare them with the serial reference operator.

Run focused tests from the repository root, for example:

.. code-block:: bash

   python -m pytest tests/validation/physics_gates/test_collision_physics.py
   python -m pytest tests/unit/objectives/test_autodiff_solver_objectives.py

These commands execute bounded checks. They do not rerun a complete physical
convergence campaign. See :doc:`testing` for the test groups and coverage policy.

Linear physical comparisons
----------------------------

Landau damping, wave dispersion, instability growth and zonal response each
need a declared analytical or independent numerical reference. State the
normalization, collision model and observable used by the reference. In a
finite Hermite expansion, phase mixing can recur; distinguish a transient
initial-value fit from an eigenmode of the regularized finite operator.

For initial-value fits, vary the analysis window and diagnostic cadence and
check competing modes. For eigenmodes, inspect the full-operator residual,
branch identity and conditioning. Increase velocity and spatial resolution,
and vary timestep, domain and numerical regularization independently.
Residual certification is an algebraic check on the chosen discretization.

Zonal-flow response
-------------------

``benchmarks/cases/collisional_zonal_response.toml`` is the Rosenbluth-Hinton
protocol [RH98]_: a linear run at :math:`k_y=0`, :math:`k_x\rho_i=0.05`, with
adiabatic electrons, :math:`q=1.4`, :math:`\hat s=0.5`, :math:`\epsilon=0.1`
and an initial ion density perturbation uniform along the line. The observable
is the flux-surface average :math:`\langle\phi\rangle(t)/\langle\phi\rangle(0)`,
the Jacobian-weighted sum over :math:`z` of ``RuntimeLinearResult.field_history``
built by :func:`gkx.diagnostics.modes.zonal_response`, which also supplies the
:math:`t=0` value. The fit ``signal`` of the result is the midplane value of
one point of the line and is not this quantity.

Collisionless residual. The formula :math:`1/(1+1.6\,q^2/\sqrt\epsilon)` gives
0.0916. The GAM-averaged residual over :math:`t\in[100,300]`, which averages
whole GAM periods (:math:`\omega_{\rm GAM}=1.007\,v_t/a`, undamped at this
:math:`q`), converges from above as the velocity space is resolved:

.. list-table::
   :header-rows: 1

   * - :math:`(N_l,N_m)`
     - (4, 8)
     - (6, 16)
     - (8, 24)
     - (12, 36)
     - formula
   * - residual
     - 0.117
     - 0.100
     - 0.095
     - 0.093
     - 0.0916

At :math:`(8,24)` the residual at :math:`\epsilon=0.05, 0.1, 0.2` is 0.0757,
0.0953, 0.1234 against 0.0666, 0.0916, 0.1248, and at :math:`k_x\rho_i=0.025`
and 0.05 it is 0.0947 and 0.0953. The formula is the :math:`\epsilon\to0`
limit at :math:`k_x\rho_\theta\ll1`; the comparison at other :math:`\epsilon`
is at fixed velocity resolution, which is coarser for the narrower trapped region.

Collisional damping. All four operators of :doc:`operators` act on the same
initial state and relax :math:`\langle\phi\rangle` to a common level of about
0.006 of its initial value, sixteen times below the collisionless residual and
independent of :math:`\nu` and of the operator. An exponential-plus-constant
fit over :math:`t\ge50` gives the decay rate: 0.0059, 0.0064 and 0.0134 at
:math:`\nu=0.005` for ``landau``, ``sugama`` (``improved_sugama`` equal) and
``lenard_bernstein``, and 0.0202, 0.0222 and 0.0332 at :math:`\nu=0.02`. The
block operators decay at about :math:`\nu` to :math:`1.3\,\nu`; the
Lenard-Bernstein operator, whose rate does not depend on the energy, faster.
GAM-averaged :math:`\langle\phi\rangle/\langle\phi\rangle(0)` at
:math:`(N_l,N_m)=(6,16)`, :math:`\epsilon=0.1`:

.. list-table::
   :header-rows: 1

   * - operator
     - :math:`\nu`
     - [50, 100]
     - [100, 150]
     - [150, 200]
     - [200, 300]
     - [300, 400]
   * - landau
     - 0.005
     - 0.0613
     - 0.0462
     - 0.0346
     - 0.0251
     - 0.0169
   * - sugama
     - 0.005
     - 0.0589
     - 0.0432
     - 0.0315
     - 0.0222
     - 0.0146
   * - improved_sugama
     - 0.005
     - 0.0589
     - 0.0432
     - 0.0315
     - 0.0222
     - 0.0147
   * - lenard_bernstein
     - 0.005
     - 0.0376
     - 0.0222
     - 0.0145
     - 0.0095
     - 0.0071
   * - landau
     - 0.02
     - 0.0221
     - 0.0116
     - 0.0082
     - 0.0067
     - 0.0061
   * - sugama
     - 0.02
     - 0.0201
     - 0.0104
     - 0.0075
     - 0.0064
     - 0.0059
   * - lenard_bernstein
     - 0.02
     - 0.0116
     - 0.0076
     - 0.0066
     - 0.0061
     - 0.0057
   * - landau
     - 0.05
     - 0.0104
     - 0.0071
     - 0.0064
     - 0.0062
     - 0.0060
   * - lenard_bernstein
     - 0.05
     - 0.0097
     - 0.0074
     - 0.0065
     - 0.0058
     - 0.0052

The level at :math:`\nu=0.02` is 0.0100, 0.0067, 0.0064 at :math:`(N_l,N_m)=(4,8),
(6,16), (8,24)`.

Comparison with Hinton-Rosenbluth and Xiao-Catto-Molvig. The primary papers
[HR99]_ and [XCM07]_ could not be retrieved: the publisher, repository and
mirror pages refused automated access. The formulas are read from two open
papers that quote them, [XCD07]_ Eqs. (24)-(26) and [Frei21]_ Sec. 5.2
Eqs. (5.1)-(5.2); their PDF text is partly garbled and the two agree. With :math:`\beta=\epsilon^2/q^2`,
:math:`\Theta=1.635\epsilon^{3/2}+\epsilon^2/2+0.36\epsilon^{5/2}` and
:math:`\Gamma=0.585\epsilon^{1/2}+0.529\epsilon`,

.. math::

   \frac{\langle\phi\rangle(t)}{\langle\phi\rangle(0)}
   = \frac{\beta}{1+\beta}\Big[1+\frac{1-\Theta}{\Theta+\beta}
   \exp\!\Big(-\frac{(1+\beta)\Gamma}{\Theta+\beta}\,\nu_{ii}t\Big)\Big],

which starts from the collisionless residual :math:`1/(1+q^2\Theta/\epsilon^2)`
(0.0811 for :math:`q=1.4`, :math:`\epsilon=0.1`; the GKX residual is 0.093 to
0.095) and relaxes to :math:`\beta/(1+\beta)=0.0051` independent of
:math:`\nu_{ii}`, at the rate :math:`3.80\,\nu_{ii}`. GKX's level, 0.0051 to
0.006 at :math:`t=300`--400, and its independence of :math:`\nu` and of the
operator agree with that. The rate depends on what :math:`\nu_{ii}` is.
The extracted text of the source defines :math:`1/\tau_{ii}` as
:math:`4\pi e^4 n\ln\Lambda/(m^{1/2}(2T)^{3/2})`, which is GKX's ``nu``
(:math:`C=4.443`, the first convention in the warning of :doc:`operators`). [Frei21]_ use
:math:`C=4\sqrt\pi/3=2.363` (Braginskii), 0.532 times that, and agree with GENE
and with the formula. The first convention gives a decay twice as fast as GKX
at every resolution (mean :math:`|\ln(\mathrm{GKX}/\mathrm{formula})|` 0.61 at
:math:`(24,64)`, :math:`\nu=0.005`); the second converges to it with velocity
resolution:

.. list-table::
   :header-rows: 1

   * - :math:`\nu=0.005`, :math:`(N_l,N_m)`
     - (6, 16)
     - (12, 36)
     - (16, 48)
     - (24, 64)
   * - mean :math:`|\ln(\mathrm{GKX}/\mathrm{formula})|`, :math:`C=2.363`, :math:`40\le t\le350`
     - 0.65
     - 0.19
     - 0.13
     - 0.09
   * - :math:`\langle\phi\rangle/\langle\phi\rangle(0)` at :math:`t=50, 100, 200, 300`
     -
     -
     - 0.051, 0.028, 0.018, 0.010
     - 0.042, 0.029, 0.015, 0.010

and the formula (:math:`C=2.363`) gives 0.051, 0.033, 0.015, 0.0087 at the same
times (three-GAM-period moving average of the GKX trace). At :math:`\nu=0.002`,
:math:`(8,24)`, the same mean deviation is 0.23. The neoclassical collisionality
:math:`\nu_*=qR\nu_{ii}/(\epsilon^{3/2}\sqrt{2T/m})` is 0.23, 0.9 and 2.3 at
:math:`\nu=0.005, 0.02, 0.05` (Braginskii :math:`\nu_{ii}`). At :math:`\nu=0.02`
the deviation is 0.51 at :math:`(12,36)` and 0.48 at :math:`(24,64)`: converged in
velocity space and 2--3 times slower than the formula (0.0359 at :math:`t=40`
against 0.0202, 0.0108 at :math:`t=100` against 0.0064). At :math:`\nu=0.05`,
:math:`(12,36)`, the trace is 0.0445 at :math:`t=20` against 0.0151, and 0.0086
at :math:`t=60` against 0.0053; there the decay time is close to the GAM period
and the formula, a single pole started at the residual, has no GAM transient to
compare with. [Frei21]_ report agreement with the formula and with GENE at
:math:`\nu_*=3.1`, which GKX does not reproduce at this setup. The agreement is
therefore established only for :math:`\nu_*\lesssim0.3`, to about 10% at the
finest resolution, and rests on reading :math:`\nu_{ii}`
of the formula as the Braginskii one, which only the primary paper settles.
The decay is not a single exponential; a fit of the moving average (level fixed
at 0.0051) gives 1.14, 1.63, 1.75, 1.88 :math:`\nu` at the four resolutions,
against :math:`3.80\times0.532=2.02\,\nu` for the formula.

``lenard_bernstein`` takes its damping from ``nu_hermite`` and ``nu_laguerre``,
which this deck sets to zero for the block operators. Without them its
conservation terms alone have eigenvalues of :math:`+\nu` and :math:`+2\nu` and
:math:`\langle\phi\rangle` grows exponentially (228-fold over 300 time units at
:math:`\nu=0.02`); the case is refused when a run combines that operator,
nonzero ``nu`` and both coefficients at zero. Set ``nu_hermite = 1`` and
``nu_laguerre = 2``.

Collisional trapped-electron mode
---------------------------------

``benchmarks/cases/tem_collisional.toml`` is a kinetic-electron mode on the
Cyclone Miller surface (:math:`q=1.4`, :math:`\hat s=0.8`, :math:`r/a=0.5`,
:math:`R/a=2.77778`) with :math:`T_e=T_i`, :math:`m_i/m_e=3670`,
:math:`a/L_{Te}=2.49`, :math:`a/L_n=2` and no ion temperature gradient. Its
fastest mode has :math:`\omega<0` (electron direction). Species ``nu`` is the
self-collision frequency in GKX units, :math:`\nu_i=\nu_e\sqrt{m_e/m_i}`; the
scan variable is :math:`\nu\equiv\nu_e`.

The three codes use different linear collision models:

- GS2 (8.2.1, ``collision_model = "default"``): Lorentz pitch-angle scattering
  plus energy diffusion with the energy-dependent Coulomb frequencies, number,
  momentum and energy restoring terms, and electron-ion scattering through
  ``zeff = 1``. ``vnewk`` is in units of :math:`\sqrt{2T/m}/a`, so
  ``vnewk`` :math:`=\nu/\sqrt2` for the electrons and the same factor for the ions.
- stella (master, ``collision_model = "dougherty"``): energy-independent
  Dougherty operator in :math:`(v_\parallel,\mu)` with momentum and energy
  conservation, self-species only. ``vnew_ref`` is the ion ``vnewk``;
  electrons get :math:`\mathtt{vnew\_ref}/\sqrt{m_e}`.
- GKX: ``landau`` and ``sugama`` models and ``lenard_bernstein`` of
  :doc:`operators`. ``lenard_bernstein`` is the model of the stella operator;
  ``landau`` of the GS2 one, which is a Lorentz plus diffusion approximation of it.

Units follow the Cyclone four-code convention: :math:`k_y^{\rm GKX}=k_y^{\rm GS2}/\sqrt2`,
:math:`\gamma^{\rm GKX}=\sqrt2\,\gamma^{\rm GS2}`, :math:`\nu=\sqrt2\,\mathtt{vnewk}`.

Collisionless growth rate at :math:`k_y\rho_i=0.3` (the first column is the
collisionless value at the resolution that column's values are reported at).
GS2 is converged in its grids (``ntheta = 64``, ``nperiod = 4``, ``negrid = 24``,
``ngauss = 12`` agrees with the production grid to 0.03%); stella and GKX are not:

.. list-table::
   :header-rows: 1

   * - code, resolution
     - :math:`\gamma`
   * - GS2
     - 0.3625
   * - stella, ``nzed`` 32 / 48 / 64 (``nperiod`` 2 / 3 / 4), ``nvgrid`` 16 / 24 / 32, ``nmu`` 8 / 12 / 16
     - 0.2997 / 0.3290 / 0.3405
   * - GKX :math:`(N_l,N_m)=(4,8),(6,16),(8,24),(12,32),(16,48),(24,64)`
     - 0.2739, 0.2875, 0.3059, 0.3208, 0.3344, 0.3404

Growth rate with collisions, :math:`k_y\rho_i=0.3`, in parentheses the ratio to
the collisionless rate of the same code and resolution:

.. list-table::
   :header-rows: 1

   * - :math:`\nu`
     - GS2
     - stella
     - GKX landau (4,8)
     - (6,16)
     - (8,24)
     - (12,32)
     - GKX lenard_bernstein (4,8)
   * - 0.01
     - 0.3217 (0.888)
     - 0.2879 (0.961)
     - 0.2656 (0.970)
     -
     - 0.2909 (0.951)
     -
     - 0.2673 (0.976)
   * - 0.05
     - 0.2666 (0.736)
     - 0.2498 (0.833)
     - 0.2388 (0.872)
     - 0.2423 (0.843)
     - 0.2495 (0.816)
     - 0.2548 (0.794)
     - 0.2389 (0.872)
   * - 0.1
     - 0.2304 (0.636)
     - 0.2132 (0.711)
     - 0.2130 (0.778)
     - 0.2098 (0.730)
     -
     -
     - 0.2056 (0.751)

At :math:`(4,8)` the Sugama and improved Sugama models give 0.2651, 0.2371 and
0.2098 at :math:`\nu=0.01, 0.05, 0.1` (improved Sugama identical to four
digits), the Landau operator 0.2656, 0.2388, 0.2130. At :math:`k_y\rho_i=0.5`
(collisionless 0.5208 GS2, 0.4605 stella, 0.4269 GKX at :math:`(4,8)`) the ratios at
:math:`\nu=0.01, 0.05, 0.1` are 0.927, 0.811, 0.730 (GS2), 0.975, 0.888, 0.803 (stella) and 0.975, 0.897, 0.825
(GKX landau, :math:`(4,8)`; lenard_bernstein 0.980, 0.908, 0.828).

Convergence of the GKX ``landau`` rates, :math:`k_y\rho_i=0.3`. Beyond
:math:`(12,32)` the shift-invert Krylov solve does not fit in memory (it reached
52 GB resident at :math:`(16,48)`), so these runs integrate in time (``rk4``,
:math:`\Delta t` 0.0005 to 0.0004, :math:`t_{\max}=40`); at :math:`(4,8)` the two
methods agree to :math:`10^{-4}` relative.

.. list-table::
   :header-rows: 1

   * - :math:`(N_l,N_m)`
     - (4, 8)
     - (6, 16)
     - (8, 24)
     - (12, 32)
     - (16, 48)
     - (24, 64)
   * - :math:`\gamma(\nu=0)`
     - 0.2739
     - 0.2875
     - 0.3059
     - 0.3208
     - 0.3344
     - 0.3404
   * - :math:`\gamma(\nu=0.05)`
     - 0.2388
     - 0.2423
     - 0.2495
     - 0.2548
     - 0.2574
     - 0.2582
   * - ratio
     - 0.872
     - 0.843
     - 0.816
     - 0.794
     - 0.770
     - 0.759
   * - :math:`\gamma(\nu=0.1)`
     - 0.2130
     - 0.2098
     -
     -
     - 0.2197
     -
   * - :math:`\gamma(\nu=0.01)`
     - 0.2656
     -
     - 0.2909
     -
     - 0.3095
     -

The collisional rate converges: its increments are 0.0053, 0.0026, 0.0008 over
the last three doublings of :math:`N_lN_m` (a power law :math:`N^{-1.7}`), which
puts the limit at :math:`0.2587\pm0.0006` for :math:`\nu=0.05`, 3% below GS2's
0.2666. At the other collision frequencies the high-order points are 0.2197 at
:math:`\nu=0.1` (GS2 0.2304) and 0.3095 at :math:`\nu=0.01` (GS2 0.3217), the latter
still rising by 0.019 from :math:`(8,24)`: the weaker the collisions, the less
they regularise, and :math:`\nu=0.01` is not converged. The collisionless rate does not converge on the same scale: its
increments over the last three doublings are 0.0149, 0.0136, 0.0060. Taking the
last ratio, 0.44, as the common ratio of the remaining increments gives 0.345
(0.349 for a ratio of 0.6), so the best estimate is :math:`0.347\pm0.005`,
4% below GS2's 0.3625 with a gap that these runs do not close. The limit of the
ratio is :math:`0.2587/0.347=0.745\pm0.01`, against GS2's 0.736. The drift of
the ratio with resolution (0.872 to 0.759) is therefore the collisionless rate
converging, not the collisional reduction changing: collisions regularise the
velocity-space structure that the collisionless mode develops, and the
collisional rate is the better-conditioned observable.

Two checks on the cause. The low-energy pitch-angle frequency
:math:`\nu_D(x)=\nu\,[\mathrm{erf}(x)-G(x)]/x^3`, which diverges as :math:`x\to0`,
was capped below :math:`x_0` in all four species pairs. It changes the rate at
:math:`\nu=0.05` by -0.1%, -0.4%, -0.3% for :math:`x_0=0.3, 0.6, 1.0` at
:math:`(4,8)` and by -0.2%, -0.2% for :math:`x_0=0.3, 0.6` at :math:`(12,32)`:
the hypothesis that the singular low-energy frequency limits the convergence is
not supported. The closure matters at the percent level: the collisionless rate
at :math:`(12,32)` is 0.3159, 0.3208, 0.3201 for :math:`k_z`-scaled
hypercollision coefficients 0.5, 1, 2, and without hypercollisions 0.2701 and
0.2430 at :math:`(4,8)` and :math:`(6,16)` (Krylov), where the reflected
Hermite ladder is not damped.

stella's Fokker-Planck model (``collision_model = "fokker-planck"``, test
particle part, and with the field part at ``lmax = jmax = 1``) was run on the
collisional deck at :math:`\nu=0.05` on the 32/16/8 grid, collisionless 0.2997:
0.2437 (0.813) for the test-particle part, 0.2469 (0.824) with the field part,
against 0.2498 (0.833) for its Dougherty operator. On the 48/24/12 grid
(collisionless 0.3290) the test-particle part gives 0.2575 (0.783), within 0.5%
of the converged GKX ``landau`` value 0.2587 and 3% below GS2. Its field part with
``exact_conservation_tp`` aborts at start-up on two ranks in this build (``free(): invalid
pointer``); the field part alone runs.

What the comparison supports. Collisions lower the growth rate in all three
codes by the same order, and with the Landau-like models the absolute
collisional rate agrees across codes to 3% (GKX 0.2587, stella's Fokker-Planck
test part 0.2575 at its second grid, GS2 0.2666); the collisionless rates differ by 5% at
the best resolution run (GKX 0.347, stella 0.341 at its finest and still rising,
GS2 0.3625). The reduction depends on the model: GKX ``lenard_bernstein`` and
stella's Dougherty operator are the same model and agree to 5% at :math:`(4,8)`
and :math:`\nu=0.05`, and GS2 with ``zeff = 0`` gives 0.2999 (0.828) there; the
electron-ion scattering of GS2's ``zeff = 1`` and of the Landau operator makes
the reduction larger.

Adiabatic-electron Cyclone ITG (:math:`k_y\rho_i=0.3`, ion collisions only) is
the single-species check of the frequency conversion and of the Dougherty
operator. The growth rate over its collisionless value at :math:`\nu=0.1, 0.2,
0.5` is 0.873, 0.731, 0.453 for GKX ``lenard_bernstein`` (Krylov,
:math:`(8,24)`, equal to :math:`(4,8)` to 0.01) and 0.867, 0.660, 0.175 for
stella; the gap persists at :math:`k_y\rho_i=0.15` (stella 0.103, GKX 0.263 at
:math:`\nu=0.5`). GS2's default model raises the ITG growth rate at small
:math:`\nu` as GKX ``landau`` does: 1.049 and 1.037 at :math:`\nu=0.1, 0.2`
against 1.052 and 1.066 at :math:`(6,16)`.

The gap is not the frequency convention, the velocity resolution or the
damping; it sits in the finite-:math:`k_\perp` restoring terms. Both operators
relax with eigenvalue :math:`-\nu(2\ell+m)`: stella's ``vnew_ref`` multiplies
:math:`\partial_{v_\parallel}(\tfrac12\partial_{v_\parallel}+v_\parallel)` and
:math:`2\partial_{\hat\mu}(\hat\mu\partial_{\hat\mu}+\hat\mu)` in two
successive implicit steps, each adding the gyrodiffusion
:math:`\tfrac14\nu k_\perp^2\rho^2` (with :math:`\rho` based on
:math:`\sqrt{2T/m}`), and GKX adds :math:`\nu b` once, the same total. Each restoring term was switched off in both codes at
:math:`\nu=0.5`, GKX at :math:`(4,8)` by removing it from the operator, stella
(rungs ``nzed``/``nvgrid``/``nmu`` = 48/24/12 and 64/32/16) with
``momentum_conservation`` and ``energy_conservation``; the growth rate
:math:`\gamma` in GKX units is

.. list-table::
   :header-rows: 1

   * - restoring terms kept
     - GKX
     - stella 48/24/12
     - stella 64/32/16
   * - none, :math:`k_y\rho_i=0.3`
     - -0.0862
     - -0.0858
     - -0.0864
   * - temperature only
     - 0.0144
     - 0.0220
     - 0.0180
   * - flow only (parallel and perpendicular)
     - -0.0109
     - -0.0667
     -
   * - all
     - 0.0583
     - 0.0247
     - 0.0219
   * - none, :math:`k_y\rho_i=0.15`
     - -0.0491
     - -0.0494
     -
   * - temperature only
     - 0.0387
     - 0.0395
     -
   * - flow only
     - -0.0597
     - -0.0830
     -
   * - all
     - 0.0153
     - 0.0059
     -

Without restoring terms the two codes agree to 0.5%, so the collision
frequency, the :math:`2\ell+m` damping and the :math:`b` term are the same.
The temperature term agrees at :math:`k_y\rho_i=0.15` (2%) and approaches GKX
as stella's grid is refined at 0.3 (0.0220 to 0.0180 against 0.0144); stella's
own result for the full operator moves 11% between its two rungs. The
perpendicular-flow term is where they differ: removing it from GKX gives 0.0059
at :math:`k_y\rho_i=0.15`, stella's 0.0059, but -0.0229 against stella's 0.0247
at 0.3, so stella's term is neither absent nor the same.

GKX's operator is the exact gyroaveraged Dougherty model. With
:math:`\delta f=h\,e^{-i\mathbf k\cdot\boldsymbol\rho}` the base operator
gives :math:`-\nu(2\ell+m+b)` and the flow and temperature of :math:`\delta f`
at fixed position give restoring functions :math:`v_\parallel J_0`,
:math:`v_\perp J_1` and :math:`(v^2-3/2)J_0`, each multiplying the same moment
of :math:`h`. The leading eigenvalues of that operator, built by quadrature
independently of the Hermite-Laguerre transcription, equal GKX's to
:math:`10^{-8}` at :math:`b=0, 0.5, 2` (a gated test), and the slowest decay is
:math:`-0.0051` at :math:`b=0.1`, the first-order conservation of perpendicular
momentum. stella's perpendicular-flow restoring term weights
:math:`(\hat v_\perp^2-\tfrac12)J_1(k_\perp\rho\,v_\perp)` and multiplies
:math:`k_\perp^2\rho^2 v_\perp^2J_1`
(``collisions_dougherty.f90``, ``get_uperp`` and ``advance_mudiff_implicit``),
which is second order in :math:`b` where the exact term is first order; with
that form the same quadrature gives a slowest eigenvalue of -0.0718 at
:math:`b=0.1`, close to the -0.0856 without the term. This is a reading of the
source with the constants of the commented alternative in ``get_uperp`` left
open, not a run of a modified stella. No change to GKX follows.

Cross-code protocol
-------------------

Match geometry profiles and coordinates, normalization, species, kinetic or
adiabatic responses, electromagnetic terms, grid and boundary conditions,
initialization, precision, timestep policy, collisions and regularization.
Check raw fields and diagnostics before fitting rates. Compare complex-field
phase conventions as well as amplitudes; a frequency sign or reference-speed
conversion can otherwise look like a physical discrepancy.

Use the formulas in :doc:`conventions` and :doc:`normalization`. GKX, GX, GS2
and stella have different discretizations and collision-model options. A
matched finite-resolution result supports that specific case; it does not
establish a collisionless limit, a converged nonlinear transport prediction,
or agreement across all kinetic-electron and electromagnetic regimes.
Report unresolved discrepancies with the configuration and convergence tests
needed to distinguish their causes.

Physical reference cases include electron-temperature-gradient turbulence
[Dorland00]_ [Jenko00]_, stellarator calculations [Sanchez21]_
[Papadopoulos23]_, quasilinear stellarator turbulence [Pueschel16]_ and stellarator
transport and zonal-flow studies [Hegna18]_ [McKinney19]_ [Tiwari25]_. Match
the published model and analysis protocol before interpreting agreement.

Comparison with GX, GS2 and stella
----------------------------------

The three comparison figures in the README plot GKX, GX, GS2 and stella on the
same flux-tube problems.
``docs/figures/four_code_comparison.json`` holds every plotted value, and
``python docs/figures/four_code_comparison.py`` redraws the figures from it.
The raw runs are not scripted in the repository. The GKX decks are in
``benchmarks/cases`` and ``examples``; the GX, GS2 and stella columns need
those codes installed and the settings listed here.

**Problems and units.** Circular Miller Cyclone (``a/L_T = 2.49``,
``a/L_n = 0.8``, ``q = 1.4``, ``s_hat = 0.8``) with adiabatic or kinetic
electrons (``T_e = T_i``, ``m_i/m_e = 3670``), and the W7-X standard VMEC
configuration (torflux 0.64, adiabatic electrons), all in GX units. GS2 and
stella use ``sqrt(2T/m)`` as the reference speed; their ``gamma``, ``omega``,
``k_y``, time and heat flux are converted with ``gamma_GX = sqrt(2) gamma_GS2``,
``k_y,GX = k_y,GS2/sqrt(2)``, ``t_GX = t_GS2/sqrt(2)`` and
``Q_GX = 2 sqrt(2) Q_GS2``.

**Linear growth rates.** Each code runs at its own converged resolution: the
cheapest whose growth rate stays within 1% of the code's highest-resolution
value over ``k_y rho_i = 0.2`` to ``0.5``.

* GKX and GX: ``(Nl, Nm) = (12, 32)``, ``Nz = 144``.
* GS2: 48 ``theta`` per ``2 pi`` x 3, 16 energies, 41 pitch angles.
* stella: ``nzed = 48`` x 3, ``nvgrid = 24``, ``nmu = 12``.

The adiabatic Cyclone growth rate at ``k_y rho_i = 0.3`` is 0.1255 (GKX, GX),
0.1254 (GS2) and 0.1252 (stella). GKX and GX share the Hermite-Laguerre model
and agree to 0.1% for ``k_y rho_i >= 0.2``. GS2 and stella lie 1% below them at
``k_y rho_i = 0.4`` and 3.5% at 0.6, and with kinetic electrons they stay
within 6%. In the ``A_parallel`` scan GS2 is within 1% up to ``beta_ref = 0.01``
and 5-6% lower on the ballooning branch above it. On W7-X, stella is 2-4% below
for ``k_y rho_i >= 1``; for ``k_y rho_i <= 0.6`` near-degenerate modes make the
fitted branch depend on the code.

**Cost.** The linear time is one ``k_y`` to ``t v_ti/a = 60`` (GS2 and stella:
the same physical time), taken from the initial-value route for GKX and GX. The
certified eigensolver takes 43 s at ``(12, 32)`` on the same GPU
(:doc:`guidance`). GKX and GX ran on one RTX A4000, GS2 and stella on MPI ranks
of an 18-core Xeon W-2295 (4 ranks for the linear timings, 8 for the nonlinear
runs). The GPU and the CPU were shared with other jobs throughout.

.. list-table::
   :header-rows: 1
   :widths: 9 17 12 17 20 15 10

   * - Code
     - Hardware
     - DOF per ``k_y``
     - Linear time per ``k_y``
     - Nonlinear time to ``t = 400``
     - Nonlinear ``Q_i``
     - ``gamma``, ``k_y rho_i = 0.3``
   * - GKX
     - RTX A4000
     - 55,296
     - 4.2 s (10 s with compilation)
     - 523 s
     - 6.9 +- 0.2
     - 0.1255
   * - GX
     - RTX A4000
     - 55,296
     - 47 s
     - 1610 s (from the step rate)
     - 6.6 +- 0.3
     - 0.1255
   * - GS2
     - 4 cores linear, 8 cores nonlinear
     - 190,240
     - 24 s
     - about 33 h (264 core-h); run to ``t = 100``
     - not reached
     - 0.1254
   * - stella
     - 4 cores linear, 8 cores nonlinear
     - 83,520
     - 29 s
     - about 16 h (128 core-h); run to ``t = 140``
     - 6.1 +- 0.8 over ``t = 104 .. 140``, not converged
     - 0.1252

**Nonlinear Cyclone.** The deck of the GX paper uses
``(Nx, Ny, Nz) = (192, 64, 24)``, ``y0 = 28.2`` and adiabatic electrons, with
127 x 22 Fourier modes.

* GKX and GX use ``(Nl, Nm) = (4, 8)``, 2.1 M degrees of freedom, in single
  precision.
* GS2 uses 21 pitch angles x 8 energies x 25 ``theta`` (23.5 M; its growth rate
  is 1.2% below the converged value at ``k_y rho_i = 0.3``).
* stella uses 24 ``v_par`` x 6 ``mu`` x 25 ``z`` (10.1 M; 3% below).
* GS2 and stella have no end damping or hypercollisions. All four codes use
  hyperdiffusion with ``D = 0.05``.
* GS2 halves its time step to 0.01 (code units) at the first peak under its
  default CFL control; stella keeps 0.02.

GKX reaches ``t v_ti/a = 400`` in 523 s (``Q_i = 6.9 +- 0.2``) and GX in 1610 s
(``6.6 +- 0.3``; the paper gives 7.2 +- 0.8). The GKX run pinned the end damping
at 0.1; the default rate gives ``6.8 +- 0.2`` over two seeds
(:ref:`common-deck`). GS2 and stella take 2.3 and 2.0 s
per step on 8 ranks. In 5.9 h (47 core-hours each) they reach ``t = 100`` and
``140``; at that step rate and time step they need about 33 h and 16 h (264 and
128 core-hours) for ``t = 400``. These full-length times are extrapolations,
not measurements. Both codes reproduce the first peak and the relaxation that
follows. The stella mean over ``t = 104 .. 140`` is 6.1 +- 0.8 (GKX 7.3, GX 6.7
over the same interval) but spans five autocorrelation times, and GS2 has not
left the relaxation. The heat flux uses the same estimator for every code.

**Speed benchmark deck.** ``benchmarks/cases/cyclone_nonlinear_t400.toml``
(64x64x24, ``(Nl, Nm) = (4, 8)``) measures speed; it is not a converged
calculation. The saturated heat flux is ``Q = 7.3`` at ``Nx = 64`` and 5.2 at
``Nx = 192`` (9.1 and 5.0 with the end damping pinned at 0.1), and ``(4, 8)``
overestimates the linear growth rate by 9 to 86% at ``k_y = 0.3`` to 0.5.

Nonlinear transport
-------------------

Specify the saturation criterion, averaging interval and sampling cadence.
Check stationarity independently for every trace, account for autocorrelation
when estimating uncertainty, and repeat seeds. Save energy and resolved
field/flux spectra so spatial and velocity-space convergence can be assessed.
Compare smaller timesteps and longer windows as well as higher resolutions.
See :doc:`stellarator_optimization` for the paired-seed uncertainty estimator.

Derivative and performance checks
---------------------------------

Compare derivatives with finite differences across perturbation sizes and
window lengths. A finite-window adjoint supports a local trajectory
sensitivity; assess a design change with independent long simulations.
See :doc:`nonlinear_autodiff` and :doc:`differentiable_eigensolver`.

Measure performance only after checking numerical agreement. Separate
compilation from warm execution, wait for device completion and report the
hardware, software versions, allocation, workload and tolerances. See
:doc:`performance` and :doc:`parallelization`.

Reproducing results
-------------------

Keep input decks, analysis commands and source revisions with a shared result.
Write generated simulation, benchmark and test outputs outside version control.
Source physics tables and comparison input decks remain required inputs.
A published figure or summary cannot substitute for executable setup and
analysis instructions. The supported feature scope is in :doc:`release_scope`.
