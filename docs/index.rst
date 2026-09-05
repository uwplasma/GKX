GKX
===========

GKX is a JAX-native gyrokinetic solver using Hermite-Laguerre velocity
space, Fourier perpendicular coordinates, and field-aligned flux-tube geometry.

Install with ``pip install gkx`` and run ``gkx`` in a terminal to
launch the default linear Cyclone demo. The same executable also accepts
checked-in TOMLs directly and can plot saved runtime outputs with
``gkx --plot <artifact>``.

Start with :doc:`quickstart` for a first result or :doc:`examples` for a task.
For the equations, read :doc:`theory`, :doc:`normalization` and :doc:`numerics`.
Before using a result in research, check :doc:`research_grade_plan` and
:doc:`release_scope` for its validation boundary and known limitations.

.. toctree::
   :maxdepth: 2
   :caption: Learn and run

   quickstart
   examples
   inputs
   outputs
   stellarator_optimization
   parallelization

.. toctree::
   :maxdepth: 2
   :caption: Physics and algorithms

   theory
   normalization
   linear_model
   operators
   geometry
   numerics
   algorithms
   solvers
   differentiable_eigensolver
   nonlinear_autodiff
   quasilinear

.. toctree::
   :maxdepth: 2
   :caption: Research evidence and reference

   research_grade_plan
   release_scope
   benchmarks
   verification_matrix
   codes
   performance
   validation_strategy
   references
   api

.. toctree::
   :maxdepth: 1
   :caption: Development

   research_grade_program
   architecture
   testing
   code_structure
   solvax_defaults
   manuscript_figures
