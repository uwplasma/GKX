Tutorial
========

Twelve steps, from a first run to a stellarator optimization. Step *N* is
the directory ``examples/NN_*``. Each has one ``run.py``: imports, parameters
at the top, a run that prints its progress, then printed, saved and plotted
results under ``./outputs/NN_*``. Deck-driven steps also have a tutorial
``case.toml`` (seconds to a minute on a laptop CPU) and, where it applies, a
literature-resolution ``case_full.toml``. Tutorial decks are coarse: the
numbers show the workflow, not converged physics.

What GKX solves
---------------

GKX advances the gyrokinetic distribution of each species in a flux tube, a
thin tube along one field line. Schematically, with :math:`h_s` the
non-adiabatic part:

.. math::

   \frac{\partial g_s}{\partial t}
   + v_\parallel\,\hat{\mathbf b}\cdot\nabla h_s
   + i\omega_{ds} h_s
   + \{\langle\chi\rangle, h_s\}
   = i\omega_{*s}^T\langle\chi\rangle F_{Ms} + C[h_s].

The terms are parallel streaming, the curvature and :math:`\nabla B` drift
:math:`\omega_d`, the :math:`E\times B` bracket, the gradient drive
:math:`\omega_*^T\propto k_y\rho\,[a/L_n + a/L_T(v^2/v_t^2-3/2)]`, and
collisions. The potentials :math:`\chi` follow from quasineutrality and, when
electromagnetic, Ampère's law (:doc:`theory`, :doc:`operators`).

Velocity space is a Laguerre (:math:`\mu`, index :math:`\ell<N_l`) and Hermite
(:math:`v_\parallel`, index :math:`m<N_m`) expansion, so ``Nl`` and ``Nm`` set
the velocity resolution. ``Nx``, ``Ny`` count Fourier modes across the tube and
``Nz`` points along it.

Units (:doc:`normalization`): lengths in :math:`a`, perpendicular scales in
:math:`\rho_i`, velocities in :math:`v_{ti}=\sqrt{T_i/m_i}`, time in
:math:`a/v_{ti}`. ``tprim`` and ``fprim`` are :math:`a/L_T` and
:math:`a/L_n`. Heat flux is in gyro-Bohm units.

1. First run
------------

.. code-block:: bash

   pip install gkx
   gkx examples/01_linear_tokamak/case.toml   # one k_y
   python examples/01_linear_tokamak/run.py   # k_y scan and eigenfunction

The Cyclone base case: ion temperature gradient (ITG) drive,
:math:`a/L_T = 2.49`, :math:`a/L_n = 0.8`, kinetic ions and Boltzmann
electrons, :math:`\delta n_e/n_e=\tau_e(\phi-\langle\phi\rangle)`. A linear
mode grows as :math:`\phi\propto e^{(\gamma-i\omega)t}`; GKX fits
:math:`\gamma` and :math:`\omega` over a window and warns when the fit is not
settled (extend ``t_max``).

The plot shows :math:`\gamma(k_y)`, which peaks near
:math:`k_y\rho_i\approx0.3`, :math:`\omega(k_y)` (positive for ITG in GKX's
sign convention, :doc:`conventions`) and the eigenfunction, a ballooning mode
centred on the outboard midplane where curvature is unfavourable. To use
Miller instead of s-alpha geometry, set ``[geometry] model = "miller"``
(:doc:`geometry`).

2. Stellarator geometry
-----------------------

.. code-block:: bash

   python examples/02_linear_stellarator/run.py
   gkx wout_circular_tokamak.nc --estimate    # print the chosen grid
   gkx wout_circular_tokamak.nc               # run to saturation
   gkx plot wout_circular_tokamak/gkx.out.nc  # replot the saved bundle

The same ITG physics in an HSX-like quasi-helically symmetric equilibrium. The
flux tube is cut from a VMEC ``wout``, and :math:`B`, :math:`|\nabla\alpha|^2`
and :math:`\omega_d` now vary along the line. The plot shows the rates against
:math:`k_y` and :math:`|\phi|` along the field line.

The script solves the ``wout`` once with ``vmex`` if it is missing (minutes on
a CPU; ``pip install vmex``) and stops with a message if ``vmex`` is absent.
``examples/vmec/generate_wouts.sh`` builds all bundled ones. ``gkx wout_*.nc``
builds a nonlinear deck for any equilibrium and writes the resolved deck so the
run can be repeated with ``gkx <deck>.toml``.

3. Nonlinear turbulence, tokamak
--------------------------------

.. code-block:: bash

   python examples/03_nonlinear_tokamak/run.py

The :math:`\{\langle\chi\rangle,h\}` bracket couples all modes: linear
instabilities grow, drive zonal flows and saturate into turbulence. The figure
shows field amplitude and energies (growth, then a plateau) and the heat flux
:math:`Q_i/Q_{GB}`. A transport number is the time average over the saturated
part; the tutorial horizon stops before saturation.

4. Nonlinear turbulence, stellarator
------------------------------------

.. code-block:: bash

   python examples/04_nonlinear_stellarator/run.py

Step 3 in the flux tube of step 2. Compare fluxes between geometries only at
production resolution (``case_full.toml``).

5. Kinetic electrons
--------------------

.. code-block:: bash

   python examples/05_kinetic_electrons/run.py

Electrons become a second kinetic species (:math:`m_e/m_i=2.7\times10^{-4}`,
:math:`\beta=10^{-4}` with :math:`A_\parallel`). Trapped-electron physics
enters and both species carry heat flux. Electrons stream
:math:`\sqrt{m_i/m_e}\approx60` times faster than ions, so the CFL step is set
by electron streaming. The script plots :math:`Q_i`, :math:`Q_e` and
:math:`\Delta t(t)`, and runs the tokamak again with the implicit ``imex-ars3``
step, which removes that limit. :doc:`guidance` says when to use which, and how
to check the end-damping rate.

6. Electromagnetic
------------------

.. code-block:: bash

   python examples/06_electromagnetic/run.py

At :math:`\beta=0.015` the parallel vector potential, from Ampère's law
:math:`k_\perp^2A_\parallel\propto\beta\,j_\parallel`, bends field lines and
modifies the ITG. The tutorial deck uses adiabatic electrons and settles in a
few seconds. The kinetic ballooning mode (KBM) needs kinetic electrons and
:math:`(N_l,N_m)=(16,48)`: ``case_full.toml`` (Nz 96; a GPU run) is the deck for
any KBM number, and :doc:`guidance` lists what to check.

7. Collisions
-------------

.. code-block:: bash

   python examples/07_collisions/run.py

Solves the step 1 ITG with each collision operator at collisionality ``NU``
(:math:`\nu a/v_{ti}`). The default is the linearized Landau operator;
Lenard-Bernstein/Dougherty, two Sugama models and the finite-:math:`k_\perp`
Coulomb operator are the alternatives. The bars compare growth rates. Equations
and verification: :doc:`operators`.

8. Quasilinear transport
------------------------

.. code-block:: bash

   python examples/08_quasilinear/run.py
   python examples/08_quasilinear/implicit_sensitivity.py

A linear scan gives the flux per unit amplitude :math:`Q/|\phi|^2` at each
:math:`k_y`. A mixing-length rule :math:`|\phi|^2\sim\gamma/\langle k_\perp^2\rangle`
turns it into a saturated estimate. The rule is uncalibrated: it ranks spectra
and designs, and does not predict absolute flux. The second script
differentiates these observables through the eigenpair and checks them against
finite differences (:doc:`quasilinear`).

9. Automatic differentiation
----------------------------

.. code-block:: bash

   python examples/09_autodiff/run.py

The solve is a JAX program, so :math:`\partial(\gamma,\omega)/\partial(a/L_T,a/L_n)`
is exact. The script plants gradients, observes two modes, and recovers the
gradients by Gauss-Newton from a wrong guess. The plots show the loss, the
parameter path, and autodiff against finite-difference Jacobians
(:doc:`differentiable_eigensolver`, :doc:`nonlinear_autodiff`).

10. Stellarator optimization with VMEX
--------------------------------------

.. code-block:: bash

   pip install vmex
   VMEX_EXAMPLES_CI=1 python examples/10_vmex_optimization/run.py   # wiring run

Adds the nonlinear heat flux, differentiated through a post-saturation window,
to a VMEX quasi-axisymmetry objective and optimizes the boundary in stages. It
prints each stage's terms and writes ``summary.json``, the optimized input and
``wout``. The default ladder is a GPU research run; :doc:`stellarator_optimization`
states the scope of the result.

11. Parallel scans
------------------

.. code-block:: bash

   python examples/11_parallel_scan/run.py

Independent :math:`k_y` points run in parallel workers
(``[parallel] strategy = "batch"``) and are gathered in input order, so the
result equals the serial scan; the script checks this (:doc:`parallelization`).

12. Restart and analysis
------------------------

.. code-block:: bash

   python examples/12_restart_and_analysis/run.py

A long nonlinear run is split into legs: the first writes its state and the
second starts from it (``[init] init_file``). The plot overlays the joined run
on an uninterrupted one and marks the averaging window. :doc:`outputs` lists
every stored variable.
