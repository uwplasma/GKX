Quickstart
==========

Install
-------

.. code-block:: bash

   pip install gkx

or install the development checkout:

.. code-block:: bash

   git clone https://github.com/uwplasma/GKX
   cd GKX
   pip install -e .

First run
---------

.. code-block:: bash

   gkx

Running ``gkx`` with no TOML launches a short Cyclone initial-value
linear demo, prints live progress with elapsed time and ETA, and writes the
artifacts in the current directory:

- ``gkx_default_linear.toml``: the input file that reproduces the run
- ``gkx_default_linear.summary.json``
- ``gkx_default_linear.timeseries.csv``
- ``gkx_default_linear.eigenfunction.csv``
- ``gkx_default_linear.png``

The demo is the Cyclone s-alpha ITG case at ``ky = 0.3`` with
``Nl = 7`` Laguerre and ``Nm = 14`` Hermite moments, integrated with ``rk4`` at
``dt = 0.02`` for 4000 steps (``t = 80``); it takes about ten seconds on a
laptop CPU. The plot is the standard two-panel linear view: the log-scale
``|\phi|^2`` time history with fitted ``(\gamma, \omega)`` on the left, and
the normalized real/imaginary eigenfunction on the right. To rerun the same
numerical case explicitly:

.. code-block:: bash

   gkx gkx_default_linear.toml --progress

A short nonlinear run (50 steps of the Cyclone nonlinear deck) and a re-plot
of either output:

.. code-block:: bash

   gkx run-runtime-nonlinear --config examples/03_nonlinear_tokamak/case_full.toml --steps 50 --out tools_out/cyclone_nonlinear.out.nc
   gkx plot tools_out/cyclone_nonlinear.out.nc
   gkx plot gkx_default_linear.summary.json

The nonlinear command already draws its figures beside the output;
``gkx plot`` redraws them from the saved files.

Full Cyclone benchmark deck
---------------------------

``examples/01_linear_tokamak/case_full.toml`` is a production-length linear
run, not a demo: ``t_max = 150`` at ``dt = 0.004663`` (32168 ``rk4`` steps)
with ``Nl = 16``, ``Nm = 48``. On a CI runner it takes about 16 minutes. Both
commands below run that deck:

.. code-block:: bash

   gkx examples/01_linear_tokamak/case_full.toml
   gkx run-runtime-linear --config examples/01_linear_tokamak/case_full.toml --out cyclone_runtime

Progress and output files
-------------------------

When progress output is enabled (on a TTY, or with ``--progress``), the
executable prints step/time progress, wall elapsed time, and an estimated
wall-clock time remaining. Adaptive nonlinear runs also report chunk-level
elapsed/ETA updates.

When ``--out`` is a plain prefix, single-point runs write a JSON summary plus
CSV sidecars under that prefix. If the nonlinear target ends in ``.out.nc`` or
another ``.nc`` suffix, the runtime writes a restartable NetCDF bundle instead:

- ``*.out.nc``: diagnostic history, geometry, and input metadata
- ``*.big.nc``: final fields and moments in spectral and real-space layouts
- ``*.restart.nc``: restart state for continuation runs

The same artifact prefix can be stored in the runtime TOML itself:

.. code-block:: toml

   [output]
   path = "tools_out/cyclone_runtime"

To make the run restart-aware, add the restart controls directly to the TOML:

.. code-block:: toml

   [time]
   nstep_restart = 100

   [output]
   path = "tools_out/cyclone_runtime.out.nc"
   restart_if_exists = true
   save_for_restart = true
   append_on_restart = true

Rerunning the same nonlinear command then resumes from the saved
``*.restart.nc`` checkpoint and appends the continued history to ``*.out.nc``.
See :doc:`outputs` for the variables in each file.

Precision
---------

Runs use JAX's default float32 (``complex64`` states). Set
``JAX_ENABLE_X64=true`` before Python starts to run in float64, and check
precision and resolution convergence for the observable you report. See
:doc:`inputs` for what float64 changes in the eigensolver certification.

Self-contained VMEC geometry
----------------------------

The VMEC-backed examples are prefilled with relative ``wout_*.nc`` paths. The
repository ships small ``vmex`` input decks, not large generated WOUT
files. Generate the needed equilibria locally, then run the TOMLs directly:

.. code-block:: bash

   pip install vmec-jax
   cd examples/vmec
   vmex input.circular_tokamak
   vmex input.NuhrenbergZille_1988_QHS
   vmex input.nfp3_QI_fixed_resolution_final
   cd ../..

   gkx run --config benchmarks/cases/circular_vmec_linear.toml
   gkx run --config examples/02_linear_stellarator/case_full.toml
   gkx run --config benchmarks/cases/w7x_linear_quasilinear_vmec.toml

The bundled circular, QHS and QI equilibria are self-contained
demonstrators. Exact machine-specific HSX or W7-X validation uses the same TOMLs
with ``--vmec-file`` pointing to the corresponding benchmark ``wout_*.nc``.

A ``wout`` file can also be run directly: ``gkx wout_XXX.nc`` resolves a default
deck, writes the resolved input and all outputs under ``./<wout-stem>/``, and
``gkx my_input.toml wout_XXX.nc`` runs an edited deck against that equilibrium.

Geometry path overrides
-----------------------

The executable can override geometry paths without editing the TOML.
These command-line paths are resolved from the shell's current working
directory, while paths written in the TOML remain resolved from the TOML
location. Use ``--vmec-file`` when the runtime config already uses a
VMEC-backed geometry model:

.. code-block:: bash

   gkx run \
     --config examples/04_nonlinear_stellarator/case_full.toml \
     --vmec-file /absolute/or/relative/wout_machine_specific.nc \
     --out tools_out/hsx_vmec_run

Use ``--geometry-file`` only for advanced imported-geometry configs that
already use ``model = "vmec-eik"``, ``model = "imported-eik"``, or
``model = "imported-netcdf"``. This is not needed for the shipped VMEC examples:

.. code-block:: bash

   gkx run \
     --config external_imported_geometry_case.toml \
     --geometry-file /absolute/or/relative/external_geometry.eik.nc \
     --out tools_out/imported_run

``--geometry-file`` only replaces ``[geometry].geometry_file``; it does not
switch ``model = "vmec"`` into imported-geometry mode. For ``model = "vmec"``,
``geometry_file`` remains the generated ``*.eik.nc`` target/cache path.

Python demo
-----------

.. code-block:: python

   from gkx import load_runtime_from_toml, run_runtime_linear

   # gkx_default_linear.toml is written by the no-argument ``gkx`` run above.
   config, _ = load_runtime_from_toml("gkx_default_linear.toml")
   result = run_runtime_linear(
       config, ky_target=0.3, Nl=7, Nm=14, solver="time", fit_signal="phi"
   )

   print(result.gamma, result.omega)  # gamma about 0.103

``run_runtime_linear`` does not read the deck's ``[run]`` table; pass ``Nl``,
``Nm`` and ``solver`` explicitly. With no ``Nl``/``Nm`` it falls back to
``(Nl, Nm) = (12, 24)``.

Tracked comparison tables are available through :mod:`gkx.benchmarking_shared`.

Run from TOML
-------------

.. code-block:: bash

   gkx examples/01_linear_tokamak/case.toml
   python examples/01_linear_tokamak/run.py   # the same case as a scripted k_y scan

Both finish in seconds; ``case_full.toml`` beside them is the production-length
literature deck.

Figure generation
-----------------

.. code-block:: bash

   python scripts/artifacts/make_benchmark_atlas.py
