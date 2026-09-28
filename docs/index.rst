GKX
===

JAX gyrokinetics for tokamak and stellarator flux tubes, with a
Hermite--Laguerre velocity basis, certified eigenvalues, nonlinear turbulence
and exact derivatives. Start with :doc:`quickstart` and :doc:`tutorials`;
:doc:`features` lists what the code can do and :doc:`design_decisions` explains
why it is built this way. :doc:`release_scope` and :doc:`verification_matrix`
state what is and is not claimed.

.. figure:: _static/readme/readme_linear.png
   :alt: Linear growth rates, frequencies and an eigenfunction against GX
   :width: 100%

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   quickstart
   tutorials
   features

.. toctree::
   :maxdepth: 1
   :caption: Inputs and outputs

   inputs
   outputs

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples
   stellarator_optimization

.. toctree::
   :maxdepth: 1
   :caption: Physics and models

   theory
   normalization
   conventions
   geometry
   linear_model
   operators
   quasilinear

.. toctree::
   :maxdepth: 1
   :caption: Numerics and algorithms

   numerics
   solvers
   algorithms
   differentiable_eigensolver
   nonlinear_autodiff
   parallelization

.. toctree::
   :maxdepth: 1
   :caption: How the code works

   design_decisions
   architecture
   code_structure
   solvax_defaults

.. toctree::
   :maxdepth: 1
   :caption: Benchmarks and validation

   benchmarks
   verification_matrix
   validation_strategy
   performance
   codes
   release_scope

.. toctree::
   :maxdepth: 1
   :caption: Development

   testing
   api
   references
