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

It is also what the bare equilibrium shorthand writes: ``gkx wout_XXX.nc``
groups its artifacts under ``./<wout-stem>/`` with ``[output] path`` set to
``<wout-stem>/gkx.out.nc``. A plain prefix there would have produced CSV
sidecars, and with them no spectra, no final fields, and no restart file --
which is most of what the run's own figure set is drawn from. Naming
``--out`` or an ``[output] path`` yourself overrides the default unchanged.

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

Every run that writes an output prefix also draws its own figures beside that
output, so a finished run leaves both the data and the pictures of it. The
nonlinear figure set is:

- ``<base>.summary.png`` -- the one-page summary: ``Q(t)`` and ``Gamma(t)``
  with the measured window shaded and ``<Q> +/- SEM`` annotated, the ``Q(ky)``
  and ``Phi^2(ky)`` spectra, the real-space potential at the outboard
  midplane, and a text panel naming the equilibrium, the resolution, the input
  deck, the averaging window and where it came from, and the stop time. It is
  always written, and any panel the output form cannot supply says so in
  place rather than costing the page.
- ``<base>.flux_time.png`` -- ``Q(t)`` and ``Gamma(t)`` at full size.
- ``<base>.flux_spectra.png`` and ``<base>.phi2_spectra.png`` -- the
  ``ky``/``kx`` heat-flux spectra and the four-panel ``Phi^2`` summary. Both
  need the k-resolved spectra, which only the NetCDF bundle carries. The
  high-``ky`` check shown on the plot is

  .. math::

     R_{\rm tail} =
     \frac{\max_{k_y\,\mathrm{in\ top\ }10\%}|S(k_y)|}
          {\max_{k_y>0}|S(k_y)|}.

  ``R_tail >= 0.1`` prints a warning: increase ``Ny`` at fixed ``Ly``, then
  repeat a matched ``Nx``/``Ny`` scan. A small tail is necessary but does not
  prove convergence; trust ``Q(ky)`` even when ``Phi^2(ky)`` looks converged.
- ``<base>.snapshot_xy.png`` and ``<base>.flux_tube_3d.png`` -- the final
  potential as an ``x``-``y`` cut and on the field-aligned flux tube itself,
  which is the one figure showing the geometry the run was performed on. Both
  read the ``*.big.nc`` final-field companion and are skipped when the run did
  not write one.

A CSV diagnostics sidecar carries time traces only, so it gets the summary and
the flux traces. A linear point and a ``ky`` scan each write
``<base>.plot.png``. Pass ``--no-plots``, or set ``[output] plots = false``,
to skip them. Plotting never affects a run's exit status: a failure prints a
warning and leaves the saved simulation untouched.

Alongside a NetCDF bundle the runtime also writes ``<base>.summary.json``,
which is where the measured saturation window and ``<Q> +/- SEM`` are recorded
in machine-readable form. ``gkx --plot`` reads it back, so a re-plot shades the
same window the run reported rather than falling back to the second half of
the trace.

To re-render a saved bundle later, or to look at a GX run with the same
command:

.. code-block:: bash

   gkx --plot tools_out/cyclone_release.out.nc
   gkx --plot gx_run.out.nc --out gx_run_panel.png

``--plot`` recognizes GX output as well as GKX's own and draws whatever the
file carries -- ``Phi^2(t)``, the fluxes, and the ``ky`` spectrum when present
-- with a title that names GX, so a panel lifted into a slide cannot be
mistaken for GKX data. On GKX's own nonlinear bundle it rebuilds the whole
figure set listed above, not only the single panel.

Use the plotting helper to visualize nonlinear diagnostic histories from
``*.out.nc`` files:

.. code-block:: bash

   python examples/utilities/plot_runtime_outputs.py tools_out/cyclone_release.out.nc \
     --out tools_out/cyclone_release_diagnostics.png

The script reads ``Diagnostics/t`` together with ``Phi2_t``, ``Wg_st``,
``Wphi_st``, and ``HeatFlux_st`` (when present) and produces a 2x2 panel.
