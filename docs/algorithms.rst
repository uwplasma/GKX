Algorithms
==========

Linear operator workflow
------------------------

For each time step, the linear operator proceeds as follows:

1. Compute :math:`b = k_\perp^2 \rho^2` from the s-alpha geometry.
2. Evaluate the Laguerre gyroaverage coefficients :math:`J_\ell(b)`.
3. Solve the field equations for :math:`(\phi, B_\parallel, A_\parallel)`.
4. Construct :math:`H_{\ell m}` including field couplings.
5. Apply the Hermite ladder streaming operator to :math:`H`.

This sequence is designed to be JIT-compilable, differentiable, and efficient
under JAX.

Linear operator decomposition
-----------------------------

The linear operator is decomposed into physically motivated pieces that act on
:math:`H_{\ell m}`:

.. math::

   \frac{\partial G_{\ell m}}{\partial t}
   = - v_{\mathrm{th}}\,\mathcal{L}_m[H]
   - v_{\mathrm{th}}\,b^\prime(\theta)\,\mathcal{M}_{\ell m}[H]
   - i\,c_v\,\mathcal{C}_m[H]
   - i\,g_b\,\mathcal{G}_\ell[H]
   + i k_y \phi \,\mathcal{D}_{\ell m}
   + \mathcal{D}_{\mathrm{coll}}.

Here:

- :math:`\mathcal{L}_m` is the Hermite ladder streaming operator.
- :math:`\mathcal{M}_{\ell m}` is the mirror coupling involving
  :math:`b^\prime(\theta)`.
- :math:`\mathcal{C}_m` and :math:`\mathcal{G}_\ell` encode curvature and
  grad-:math:`B` couplings, respectively.
- :math:`\mathcal{D}_{\ell m}` is the energy-weighted diamagnetic drive, with
  additional :math:`A_\parallel` and :math:`B_\parallel` couplings when
  electromagnetic terms are enabled.
- :math:`\mathcal{D}_{\mathrm{coll}}` represents optional Lenard-Bernstein and
  hyper-diffusion damping.

Operator splitting summary
--------------------------

Time integration is handled with explicit Runge-Kutta schemes or IMEX/implicit
updates. The operator splitting used in ``imex`` mode treats
``\mathcal{D}_{\mathrm{coll}}`` implicitly while keeping the streaming and
drift/drive terms explicit. The fully implicit option performs a backward-Euler
solve using GMRES with a diagonal preconditioner that includes damping and
drift/mirror diagonals.

Data layout and memory
----------------------

We store Laguerre and Hermite indices in the leading axes and use FFT-friendly
ordering for the perpendicular Fourier grid. The layout is optimized for vector
operations and for the current JAX parallelization path that partitions the
packed state across device meshes when runtime parallelization is enabled via
``state_sharding``.
