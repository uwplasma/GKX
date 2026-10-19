Resolution, Step and Solver Guidance
====================================

What to set, in what order, and how to check it. Values are starting points
for Cyclone-like cases; the check at the end of each section is the test.
Measurements are in :doc:`numerics`, :doc:`solvers` and :doc:`examples`.

Choose the time stepper
-----------------------

GKX advances :math:`\partial_t g = L g + N(g)`. Write :math:`L = S + R`, where
:math:`S` holds the stiff terms (parallel streaming with its dissipation, and
the field response) and :math:`R` the drifts, mirror, drive and collisions.
:math:`N` is the :math:`E\times B` bracket.

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Scheme
     - Use it for
     - Limit
   * - ``rk4`` + CFL (default without ``dt``)
     - any deck; the controller sets the step from the stiffest explicit term
     - step follows electron streaming when electrons are kinetic
   * - ``rk2``/``rk3`` at fixed ``dt``
     - reproducing a published step, parity runs
     - warns when ``dt`` exceeds the CFL estimate
   * - ``imex``, ``imex2``
     - linear runs with stiff collisions or hypercollisions
     - diagonal damping only; streaming stays explicit
   * - ``imex-ars3``
     - nonlinear kinetic electrons at 32x32 and up
     - fixed ``dt``, chosen by hand; factor memory and build time
   * - ``solver = "krylov"``/``"adaptive"``
     - linear growth rates: no time step at all
     - one mode per solve

**Why IMEX helps.** With kinetic electrons the explicit step is set by
electron streaming, :math:`\Delta t \lesssim 1/(v_{te}k_{z,\max})`, about
:math:`\sqrt{m_i/m_e}\approx 60` times shorter than the ion scale. In
``imex-ars3`` the operator :math:`S` is inverted exactly per twist-shift chain
(block Thomas plus a Woodbury correction for the fields), so
:math:`\Delta t` is set by :math:`R` and :math:`N`. On Cyclone with kinetic
electrons at 64x64x24 this gave 10x shorter time to solution than RK3 at the
same heat flux. At a 16x16 tutorial grid or a small QA deck it does not pay:
the factor build dominates.

**Choosing the IMEX step.** ``dt`` is not found by a controller. 0.05 is
stable at 32x32x16 and non-finite at 64x64x24, where 0.035 and 0.025 work.
Rerun at half the step and compare the heat flux. Set
``XLA_PYTHON_CLIENT_PREALLOCATE=false`` on a GPU.

Resolution
----------

- **Velocity space.** Linear runs default to :math:`(N_l, N_m) = (12, 24)`,
  nonlinear runs to (4, 8). Vary :math:`N_m` first. Kinetic electrons and TEM
  need more: TEM growth rates still change at (16, 48).
- **Perpendicular.** Nonlinear production runs use 64-96 points in
  :math:`x` and :math:`y`; ``gkx --estimate`` prints the grid the geometry
  asks for, with the reason.
- **Parallel.** :math:`N_z` = 24-64 per field-period segment, and
  ``nperiod >= 2`` for any linear run with linked boundaries.
- **Check.** The heat flux or growth rate changes less than your tolerance
  when :math:`N_m` doubles.

Kinetic electrons
-----------------

- Leave ``fixed_dt = false`` with an explicit scheme; read the bound in
  ``diagnostics.cfl_scales``.
- Use a small finite :math:`\beta` with :math:`A_\parallel`
  (:math:`\beta = 10^{-4}`). At :math:`\beta = 0` the electrostatic
  :math:`\omega_H` mode cuts the step a further 6 times.
- For growth rates use ``solver = "krylov"`` or ``"adaptive"``.
- Time: allow :math:`10/\gamma` for the linear phase, saturation by
  :math:`t\approx 60`-100, then 100-200 for the average, or
  ``run_to = "saturation"``.
- **Check.** Rerun with ``damp_ends_rate`` 3x up and down and with
  ``nperiod + 1`` (below).

Electromagnetic and KBM
-----------------------

- :math:`\beta` in the deck sets :math:`A_\parallel`; check the mode against
  the next :math:`N_m`.
- The linear KBM branch moves with the end-damping rate and with ``nperiod``.
  At ``nperiod = 1`` the layer is a quarter of the line: use 2 or more.
- Explicit steps must keep :math:`\nu_{end}\Delta t < 0.3`. An ``nperiod = 1``
  KBM deck (rate 598) blew up at :math:`\Delta t = 10^{-3}` and ran at
  :math:`5\times10^{-4}`.
- Eigenvalues: ``auto`` (banded, else ``adaptive``) certifies KBM; a shift-invert
  with a poor shift can certify a non-dominant branch.

End damping
-----------

One rate, ``[time] damp_ends_rate``, applies on every route and no time step
enters it. The default is 30 e-folds per thermal transit of the fastest kinetic
species across the end layer, and every run prints it. Below about 1 e-fold
the kinetic-electron and KBM rates are not converged. Rerun at 3x above and
below the printed rate and at ``nperiod + 1``; a converged rate moves by less
than its tolerance in all three (:ref:`end-damping`).

Collisions
----------

The default operator is the linearized Landau operator (:doc:`operators`). It
needs the velocity resolution of the case; a factor is built once (0.5 s at
(4, 8), tens of seconds at (16, 48)) and cached. Set ``nu`` per species;
``lenard_bernstein`` is cheaper and not accurate at high :math:`\nu`
(21% low on a Cyclone scan at :math:`\nu_i = 0.1`).

Eigensolver
-----------

``auto`` (default) runs the Hermite-banded direct shift-invert where its blocks
fit in memory and ``adaptive`` otherwise; ``adaptive`` is Arnoldi on the
time-advance map, as GS2 does with SLEPc. Both certify the pair against the full
operator. Use matrix-free shift-invert only for a chosen interior branch
(:doc:`algorithms`).
