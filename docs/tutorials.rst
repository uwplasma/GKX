Tutorials
=========

Short paths through the code. Each step runs on a laptop CPU.

First linear run
----------------

.. code-block:: bash

   pip install gkx
   gkx

This runs a linear Cyclone case and prints ``gamma`` and ``omega``. Next, the
gallery version scans ``k_y`` and plots the eigenfunction:

.. code-block:: bash

   python examples/01_linear_tokamak/run.py

Edit the ``KY`` list at the top of ``run.py``, or the deck ``case.toml``, and run again.

From an equilibrium to turbulence
---------------------------------

.. code-block:: bash

   gkx wout_circular_tokamak.nc --estimate
   gkx wout_circular_tokamak.nc
   gkx plot wout_circular_tokamak/gkx.out.nc

The first command prints the grid and the reason for each entry; the second
runs to saturation; the third replots the saved bundle. ``gkx`` also writes the
resolved deck, so the run can be repeated with ``gkx <deck>.toml``. The
``wout`` files are built by ``examples/vmec/generate_wouts.sh``.

Restart and analyse
-------------------

.. code-block:: bash

   python examples/12_restart_and_analysis/run.py

Continues a nonlinear run from its saved state and averages the joined heat
flux trace. :doc:`outputs` lists every stored variable.

Take a derivative
-----------------

.. code-block:: bash

   python examples/09_autodiff/run.py

Recovers :math:`a/L_{T_i}` and :math:`a/L_n` from two linear modes by
Gauss-Newton with exact derivatives. :doc:`differentiable_eigensolver` explains
the derivative; :doc:`nonlinear_autodiff` extends it to the nonlinear heat
flux.

Optimize a stellarator
----------------------

See :doc:`stellarator_optimization` for the VMEX scripts that put a GKX
growth rate, quasilinear flux or nonlinear heat flux into a QA shape
optimization.
