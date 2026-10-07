Tutorial
========

Twelve steps, from a first run to a stellarator optimization. Step *N* is the
directory ``examples/NN_*``: each has one ``run.py`` (imports, the parameters
at the top, a run that prints its progress, then printed, saved and plotted
results under ``./outputs/NN_*``) and, where deck-driven, a tutorial
``case.toml`` that runs in seconds to a minute on a laptop CPU plus a
literature-resolution ``case_full.toml``. Tutorial decks are deliberately
coarse: their numbers show the workflow, not converged physics.

What GKX solves
---------------

GKX evolves the gyrokinetic distribution of each species in a flux tube, a
thin tube that follows one magnetic field line. The perturbed distribution is
:math:`\delta f_s = -Z_s\phi F_{Ms}/T_s + h_s`; GKX advances
:math:`g_s = h_s - Z_s\langle\phi\rangle F_{Ms}/T_s` (electrostatic form)

.. math::

   \frac{\partial g_s}{\partial t}
   + v_\parallel\,\hat{\mathbf b}\cdot\nabla h_s
   + i\omega_{ds} h_s
   + \{\langle\chi\rangle, h_s\}
   = i\omega_{*s}^T\langle\chi\rangle F_{Ms} + C(h_s),

with streaming along the line, the magnetic (curvature and :math:`\nabla B`)
drift :math:`\omega_d`, the nonlinear :math:`E\times B` bracket, the drive
:math:`\omega_*^T \propto k_y\rho\,[a/L_n + a/L_T(v^2/v_t^2 - 3/2)]` from the
density and temperature gradients, and collisions :math:`C`. The potential
:math:`\chi = \phi - v_\parallel A_\parallel + \ldots` closes the system through
quasineutrality (and Ampère's law when electromagnetic); :doc:`theory` writes
both. Velocity space is expanded in Laguerre (:math:`\mu`, index
:math:`\ell<N_\ell`) and Hermite (:math:`v_\parallel`, index :math:`m<N_m`)
polynomials, so ``Nl`` and ``Nm`` are the velocity resolution; ``Nx``, ``Ny``
are Fourier modes across the line and ``Nz`` points along it.

**Units** (:doc:`normalization`): lengths in the minor radius :math:`a`,
perpendicular scales in the ion gyroradius :math:`\rho_i`, velocities in
:math:`v_{ti}`, time in :math:`a/v_{ti}`. ``tprim`` and ``fprim`` are
:math:`a/L_T` and :math:`a/L_n`. Growth rate :math:`\gamma` and frequency
:math:`\omega` are in :math:`v_{ti}/a`; heat flux :math:`Q_s` is in gyro-Bohm
units :math:`Q_{GB} = n_i T_i v_{ti}\rho_i^2/a^2`.

1. First run: Python and the command line
-----------------------------------------

.. code-block:: bash

   pip install gkx
   gkx examples/01_linear_tokamak/case.toml          # s-alpha Cyclone, one k_y
   gkx examples/01_linear_tokamak/case_miller.toml   # same physics, Miller geometry
   python examples/01_linear_tokamak/run.py          # k_y scan + eigenfunction

*Physics.* The Cyclone base case: ion temperature gradient (ITG) drive
(:math:`a/L_T = 2.49`, :math:`a/L_n = 0.8`) with kinetic ions and Boltzmann
(adiabatic) electrons, :math:`\delta n_e/n_e = \tau_e(\phi-\langle\phi\rangle)`.
The run is *linear* (no bracket), so each :math:`k_y` grows as
:math:`\phi\propto e^{(\gamma - i\omega)t}`; GKX integrates in time and fits
:math:`\gamma,\omega` over a window. ``s-alpha`` is the circular large-aspect
model set by :math:`q`, :math:`\hat s` and :math:`\epsilon`; ``miller`` builds
a shaped local equilibrium (``akappa``, ``tri`` in the ``[geometry]`` block).

*Output.* The CLI prints ``gamma`` and ``omega`` with an error bar from the fit
window and warns when the fit is not settled (fewer than two e-foldings in the
window: extend ``t_max``). ``run.py`` plots :math:`\gamma(k_y)` (the ITG peaks
near :math:`k_y\rho_i\approx0.3`), :math:`\omega(k_y)` (positive for the ITG in GKX's sign convention,
:doc:`conventions`) and the eigenfunction :math:`\phi(\theta)`, a ballooning
mode peaked at the outboard midplane :math:`\theta=0` where curvature is bad.
Edit ``KY`` or the deck and rerun; :doc:`inputs` lists every key.

2. Stellarator geometry (VMEC)
------------------------------

.. code-block:: bash

   python examples/02_linear_stellarator/run.py
   gkx wout_circular_tokamak.nc --estimate   # any wout: print the chosen grid
   gkx wout_circular_tokamak.nc              # run it to saturation
   gkx plot wout_circular_tokamak/gkx.out.nc # replot the saved bundle

The same ITG physics in an HSX-like quasi-helically symmetric equilibrium.
The flux tube is cut from a VMEC ``wout`` (built once with ``vmex`` if missing)
and the geometry coefficients :math:`B(z)`, :math:`|\nabla\alpha|^2`,
:math:`\omega_d(z)` now vary along the line with the field periods. The plot
shows :math:`\gamma,\omega` versus :math:`k_y` and :math:`|\phi|` along the
field-line coordinate: modes localize in the bad-curvature wells, which is
what shaping and optimization act on. :doc:`geometry` covers VMEC, Miller and
imported files. The ``gkx wout_*.nc`` shorthand builds a nonlinear deck for
any equilibrium (``examples/vmec/generate_wouts.sh`` makes the bundled
``wout`` files) and writes the resolved deck for ``gkx <deck>.toml`` reruns.

3. Nonlinear turbulence (tokamak)
---------------------------------

.. code-block:: bash

   python examples/03_nonlinear_tokamak/run.py

Turning on the :math:`\{\langle\chi\rangle, h\}` bracket couples all
:math:`(k_x,k_y)` modes: linear instabilities grow, drive zonal flows and
saturate into turbulence. The figure from ``gkx.plot`` shows the field
amplitude and energies (exponential growth, then a plateau) and the heat flux
:math:`Q_i/Q_{GB}`. A transport number is the time average over the saturated
part only; the tutorial horizon stops before saturation.

4. Nonlinear turbulence (stellarator)
-------------------------------------

.. code-block:: bash

   python examples/04_nonlinear_stellarator/run.py

Step 3 in the VMEC flux tube of step 2. Read the plot the same way; compare
the saturated flux between geometries only at production resolution
(``case_full.toml``).

5. Kinetic electrons
--------------------

.. code-block:: bash

   python examples/05_kinetic_electrons/run.py

Electrons become a second kinetic species (:math:`m_e/m_i = 2.7\times10^{-4}`,
small :math:`\beta = 10^{-4}` with :math:`A_\parallel`), so trapped-electron
and electron-drift physics enter and both species carry heat flux. Electrons
stream :math:`\sqrt{m_i/m_e}\approx60` times faster than ions, so the CFL
controller's step is set by electron streaming; the script plots
:math:`Q_i, Q_e` and :math:`\Delta t(t)` and runs the tokamak once more with
the opt-in implicit ``imex-ars3`` step, which removes that limit. The decks
state ``[time] damp_ends_rate`` (the end-of-tube absorber rate): kinetic-electron
growth rates depend on it, and without it the time and Krylov routes damp
differently. :doc:`examples` has the dt and ``t_max`` guidance.

6. Electromagnetic: kinetic ballooning mode
-------------------------------------------

.. code-block:: bash

   python examples/06_electromagnetic/run.py

At :math:`\beta = 0.015` the parallel vector potential :math:`A_\parallel`
(Ampère's law, :math:`k_\perp^2 A_\parallel \propto \beta\,j_\parallel`) bends
field lines and the pressure gradient drives the kinetic ballooning mode. The
plot shows the potential trace and eigenfunction. The tutorial deck only
exercises the path; ``case_full.toml`` resolves the KBM.

7. Collisions
-------------

.. code-block:: bash

   python examples/07_collisions/run.py

Solves the step-1 ITG with each shipped collision operator, from the diagonal
Lenard-Bernstein/Dougherty model to the gyrokinetic Coulomb operator with
finite-:math:`k_\perp` effects, at collisionality ``NU`` (:math:`\nu a/v_{ti}`).
The bars compare growth rates; with ``NU_SCAN = True`` the plot shows each
model's collisional damping and its error against the finite-:math:`k_\perp`
Coulomb reference.

8. Quasilinear transport
------------------------

.. code-block:: bash

   python examples/08_quasilinear/run.py
   python examples/08_quasilinear/implicit_sensitivity.py

A linear scan gives at each :math:`k_y` the flux per unit amplitude
:math:`Q/|\phi|^2` from the eigenfunction; a mixing-length rule
:math:`|\phi|^2\sim\gamma/\langle k_\perp^2\rangle` turns it into a saturated
estimate. The plot shows :math:`\gamma`, the weight and the estimate versus
:math:`k_y`. The rule is uncalibrated: it ranks spectra and designs, it does
not predict absolute flux. The second script differentiates these
observables through the eigenpair and checks the derivatives against finite
differences (points on the diagonal of the parity plot). :doc:`quasilinear`.

9. Automatic differentiation
----------------------------

.. code-block:: bash

   python examples/09_autodiff/run.py
   python examples/09_autodiff/geometry_bridge.py

The whole time integration is a JAX program, so
:math:`\partial(\gamma,\omega)/\partial(a/L_T, a/L_n)` comes exactly from
autodiff. ``run.py`` plants gradients, observes two modes and recovers the
gradients by Gauss-Newton from a wrong guess: the plots show the loss falling,
the parameter path reaching the planted point, the autodiff-versus-finite
difference Jacobian on the diagonal, and growth-rate sweeps.
``geometry_bridge.py`` differentiates through VMEC/Boozer geometry.
:doc:`differentiable_eigensolver`, :doc:`nonlinear_autodiff`.

10. Stellarator optimization with VMEX
--------------------------------------

.. code-block:: bash

   pip install vmex
   VMEX_EXAMPLES_CI=1 python examples/10_vmex_optimization/run.py   # wiring run

Adds the nonlinear GKX heat flux, differentiated through a post-saturation
window, to a VMEX quasi-axisymmetry objective (aspect ratio 6, mean
:math:`\iota = 0.42`) and optimizes the boundary in stages. It prints each
stage's objective terms and writes ``summary.json``, the optimized VMEC input,
its ``wout`` and the vmex equilibrium plots. The default ladder is a GPU
research run; :doc:`stellarator_optimization` states the scope of the result.

11. Parallel scans
------------------

.. code-block:: bash

   python examples/11_parallel_scan/run.py

Independent :math:`k_y` points run in parallel workers (``[parallel]
strategy = "batch"``) and are gathered in input order, so the result is
identical to the serial scan; the script checks this and plots the spectrum.
:doc:`parallelization`.

12. Restart and analysis
------------------------

.. code-block:: bash

   python examples/12_restart_and_analysis/run.py

Long nonlinear runs are split into legs: the first leg writes its state, the
second starts from it (``[init] init_file``). The plot overlays the joined run
on an uninterrupted one (they agree to single precision) and marks the window
over which the heat flux is averaged. On the command line the same is
``gkx deck.toml`` with ``[output]`` restart keys; :doc:`outputs` lists every
stored variable, and ``gkx plot <run>.out.nc`` replots a saved bundle.
