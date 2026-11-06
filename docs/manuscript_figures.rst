:orphan:

Publication figures
===================

Generate figures from reproducible solver and analysis commands, with outputs
outside version control. Keep the input deck and analysis procedure available
alongside each published result.

A caption should state the geometry and species model, resolution, precision,
collision and regularization settings, timestep policy, diagnostic and analysis
window. For comparisons, identify the reference and acceptance criterion.
Report convergence and uncertainty for the quantity being claimed.

A speedup figure requires matched numerical results and timed warm execution
on the stated hardware. A nonlinear transport figure requires a declared
stationarity window and uncertainty estimate. A derivative figure requires a
finite-difference check and a clear statement of the differentiated window.
See :doc:`release_scope`, :doc:`parallelization` and :doc:`nonlinear_autodiff`.
