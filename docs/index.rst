GKX
===

JAX gyrokinetics with Hermite–Laguerre velocity moments and field-aligned
flux tubes. Start with a runnable case, then its model and evidence.
:doc:`algorithms` is the overview of the methods and the decisions behind them.
:doc:`release_scope` and :doc:`verification_matrix` state what is and is not
claimed; read them before citing a result.

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

   benchmarks
   verification_matrix
   validation_strategy
   parallelization
   performance
   codes
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
