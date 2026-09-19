GKX
===

JAX gyrokinetics with Hermite–Laguerre velocity moments and field-aligned
flux tubes. Start with a runnable case, then its model and evidence.
:doc:`algorithms` is the overview of the methods and the decisions behind them.
Read :doc:`research_grade_plan` before using experimental physics or making
transport-optimization claims.

.. toctree::
   :maxdepth: 1
   :caption: Start and run

   quickstart
   examples
   inputs
   outputs

.. toctree::
   :maxdepth: 1
   :caption: Physics and numerical contracts

   algorithms
   theory
   normalization
   geometry
   linear_model
   operators
   numerics
   solvers

.. toctree::
   :maxdepth: 1
   :caption: Sensitivities and optimization

   differentiable_eigensolver
   nonlinear_autodiff
   quasilinear
   stellarator_optimization

.. toctree::
   :maxdepth: 1
   :caption: Evidence and research status

   research_grade_plan
   research_grade_program
   benchmarks
   verification_matrix
   validation_strategy
   parallelization
   performance
   codes
   manuscript_figures
   references

.. toctree::
   :maxdepth: 1
   :caption: Development and API

   architecture
   code_structure
   testing
   solvax_defaults
   release_scope
   api
