Saved Outputs and Restart Files
===============================

GKX supports two saved-output modes:

- lightweight prefix-based JSON/CSV sidecars for quick solver runs,
- nonlinear NetCDF restart bundles for parity, post-processing, and restart.

Lightweight saved outputs
-------------------------

When ``[output].path`` or ``--out`` is a plain prefix such as
``tools_out/runtime_case``, the runtime writes small sidecar files:

- linear runtime: ``*.summary.json`` and, when available,
  ``*.timeseries.csv``
- nonlinear runtime: ``*.summary.json``, ``*.diagnostics.csv``, and
  ``*.state.bin`` when the final state is requested

The nonlinear diagnostics CSV contains the base columns
``t,dt,gamma,omega,Wg,Wphi,Wapar,energy,heat_flux,particle_flux`` plus any
available species-resolved columns:

- ``heat_flux_s{i}``
- ``particle_flux_s{i}``
- ``turbulent_heating``
- ``turbulent_heating_s{i}``

Nonlinear NetCDF bundle
-----------------------

When the nonlinear output target ends in ``.out.nc`` (recommended) or another
``.nc`` suffix, GKX writes three coordinated files:

- ``case.out.nc``
- ``case.big.nc``
- ``case.restart.nc``

This is the release-facing format for nonlinear parity, restart, and external
post-processing workflows.

``*.out.nc``
^^^^^^^^^^^^

The main nonlinear history file contains:

- ``Grids``: time history and active spectral ``kx/ky/theta`` coordinates
- ``Geometry``: flux-tube metric arrays and geometry scalars
- ``Inputs``: imported runtime metadata needed by the comparison tooling
- ``Diagnostics``: scalar, species-resolved, and resolved nonlinear outputs

The diagnostic group includes the main history series:

- ``Phi2_t``
- ``Wg_st``
- ``Wphi_st``
- ``Wapar_st``
- ``HeatFlux_st``
- ``ParticleFlux_st``
- ``TurbulentHeating_st`` when available

It also carries resolved reductions used by the parity tooling, including
``*_kxst``, ``*_kyst``, ``*_kxkyst``, ``*_zst``, and ``Wg_lmst``.

``*.big.nc``
^^^^^^^^^^^^

The large-field sidecar stores the final state in forms convenient for
inspection and comparison:

- spectral ``Phi``, ``Apar``, ``Bpar``
- real-space ``PhiXY``, ``AparXY``, ``BparXY``
- basis moments such as ``Density``, ``Upar``, ``Tpar``, ``Tperp``
- particle moments such as ``ParticleDensity``, ``ParticleUpar``,
  ``ParticleUperp``, ``ParticleTemp``

``*.restart.nc``
^^^^^^^^^^^^^^^^

The restart sidecar stores the nonlinear Hermite-Laguerre state in the packed
restart layout together with the final time. GKX can reload this file
directly through either:

- the explicit ``[init] init_file`` path, or
- the higher-level ``[output] restart*`` controls.

Restart workflow
----------------

Recommended continuation configuration:

.. code-block:: toml

   [time]
   nstep_restart = 100

   [output]
   path = "tools_out/cyclone_release.out.nc"
   restart_if_exists = true
   save_for_restart = true
   append_on_restart = true
   restart_with_perturb = false

Behavior of the main restart controls:

- ``restart``: require and load a restart file
- ``restart_if_exists``: resume only if the restart file already exists
- ``restart_to_file`` / ``restart_from_file``: override the default sibling
  ``*.restart.nc`` path
- ``restart_scale``: scale the loaded state
- ``restart_with_perturb``: add a new analytic perturbation on top of the
  loaded state instead of replacing it
- ``append_on_restart``: keep prior ``*.out.nc`` history and append new samples
- ``save_for_restart``: emit the checkpoint file during nonlinear runs
- ``time.nstep_restart`` or ``output.nsave``: choose checkpoint cadence in steps

For long adaptive jobs, the usual user-facing pattern is simply to rerun the
same nonlinear command. If ``restart_if_exists = true`` and the checkpoint is
present, the runtime resumes from ``*.restart.nc`` and keeps growing the
history in ``*.out.nc``.

Plotting diagnostics
--------------------

Use the plotting helper to visualize nonlinear diagnostic histories from
``*.out.nc`` files:

.. code-block:: bash

   python examples/utilities/plot_runtime_outputs.py tools_out/cyclone_release.out.nc \
     --out tools_out/cyclone_release_diagnostics.png

The script reads ``Diagnostics/t`` together with ``Phi2_t``, ``Wg_st``,
``Wphi_st``, and ``HeatFlux_st`` (when present) and produces a 2x2 panel.
