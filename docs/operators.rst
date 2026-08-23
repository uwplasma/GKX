Operators And Terms
===================

This page documents the implemented operator set in GKX and ties each
term to its runtime parameters and source files.

State And Coupled Variable
--------------------------

For each species :math:`s`, GKX evolves Laguerre-Hermite moments
:math:`G^{(s)}_{\ell m}(k_x,k_y,z,t)`. The field-coupled variable used by the
linear operator is

.. math::

   H_{\ell m}^{(s)}
   =
   G_{\ell m}^{(s)}
   + \frac{Z_s}{T_s} J_\ell \phi \,\delta_{m0}
   - \frac{Z_s v_{th,s}}{T_s} J_\ell A_\parallel \,\delta_{m1}
   + J_\ell^B B_\parallel \,\delta_{m0},

with :math:`J_\ell^B = J_\ell + J_{\ell-1}`.

In the explicit-time reference-compatible path, streaming is applied to the
field-coupled streamed variable built from the same field terms before the
Hermite ladder is taken.

Source mapping:

- ``src/gkx/linear.py``
- ``src/gkx/terms/fields.py``
- ``src/gkx/terms/assembly.py``

Implemented Linear Operator
---------------------------

The assembled RHS is

.. math::

   \partial_t G
   =
   \mathcal{R}_{stream}
   + \mathcal{R}_{mirror}
   + \mathcal{R}_{curv}
   + \mathcal{R}_{gradB}
   + \mathcal{R}_{dia}
   + \mathcal{R}_{coll}
   + \mathcal{R}_{hyper}
   + \mathcal{R}_{k_\perp\text{-hyper}}
   + \mathcal{R}_{end}.

Every term has a matching multiplicative weight in ``TermConfig`` and
``RuntimeTermsConfig``.

Gyroaverage And Bessel Factors
------------------------------

The Laguerre gyroaverage coefficients are

.. math::

   J_\ell(b) = \frac{1}{\ell!}\left(-\frac{b}{2}\right)^\ell e^{-b/2},
   \qquad
   b = k_\perp^2 \rho_s^2.

Nonlinear electromagnetic terms additionally use :math:`J_0(\alpha)` and
:math:`J_1(\alpha)` on the quadrature grid.

Source mapping:

- ``src/gkx/core/velocity.py``
- ``src/gkx/terms/nonlinear.py``

Streaming
---------

The Hermite ladder streaming term is

.. math::

   \mathcal{R}_{stream}
   =
   -w_{stream}\,k_\parallel v_{th,s}
   \left(\sqrt{m+1}\,X_{\ell,m+1} + \sqrt{m}\,X_{\ell,m-1}\right),

where :math:`X` denotes either :math:`H` or the benchmark-compatible streamed
variable, depending on the solver path.

Controls:

- ``LinearParams.kpar_scale``
- ``RuntimeTermsConfig.streaming``
- boundary/link metadata from the geometry/grid

Mirror
------

The mirror term uses :math:`b'(z)` and couples both Laguerre and Hermite
indices:

.. math::

   \mathcal{R}_{mirror}
   =
   -w_{mirror}\,v_{th,s}\,b'(z)\,
   \Big[
   -\sqrt{m+1}(\ell+1)H_{\ell,m+1}
   -\sqrt{m+1}\ell H_{\ell-1,m+1}
   +\sqrt{m}\ell H_{\ell,m-1}
   +\sqrt{m}(\ell+1)H_{\ell+1,m-1}
   \Big].

Curvature And Grad-B
--------------------

The drift terms are

.. math::

   \mathcal{R}_{curv}
   =
   - i\,w_{curv}\,\tau_z\,\omega_d\,c_v(z)
   \Big[
   \sqrt{(m+1)(m+2)}H_{\ell,m+2}
   + (2m+1)H_{\ell m}
   + \sqrt{m(m-1)}H_{\ell,m-2}
   \Big],

.. math::

   \mathcal{R}_{gradB}
   =
   - i\,w_{gradB}\,\tau_z\,\omega_d\,g_b(z)
   \Big[
   (\ell+1)H_{\ell+1,m}
   + (2\ell+1)H_{\ell m}
   + \ell H_{\ell-1,m}
   \Big].

Controls:

- ``LinearParams.omega_d_scale``
- ``RuntimeTermsConfig.curvature``
- ``RuntimeTermsConfig.gradb``

Diamagnetic Drive
-----------------

The diamagnetic drive acts through density and temperature-gradient couplings
in the low Hermite moments. In code it drives:

- ``m=0`` through density and perpendicular-energy combinations,
- ``m=2`` through temperature-gradient coupling,
- ``m=1`` and ``m=3`` for electromagnetic ``A_parallel`` terms when enabled.

Controls:

- ``LinearParams.omega_star_scale``
- ``LinearParams.fprim``
- ``LinearParams.tprim``
- ``RuntimeTermsConfig.diamagnetic``

Collisions
----------

GKX ships five collision models, selected with the ``collision_operator`` key in
the ``[time]`` section of a TOML input or through
:func:`gkx.operators.linear.params.collision_operator_from_config`:

.. list-table::
   :header-rows: 1
   :widths: 22 40 38

   * - ``collision_operator``
     - Model
     - Reference
   * - ``none``, ``lenard_bernstein``
     - Conserving diagonal Lenard-Bernstein/Dougherty relaxation
     - built in, described below
   * - ``sugama``
     - Drift-kinetic Sugama, conservative by construction
     - Frei, Ernst & Ricci (2022), Eqs. (C6a)--(C6f)
   * - ``improved_sugama``
     - Improved Sugama, corrected Pfirsch-Schlüter friction
     - Sugama et al. (2019); Frei, Ernst & Ricci (2022)
   * - ``coulomb``
     - Drift-kinetic linearized Coulomb (Landau)
     - Frei, Ernst & Ricci (2022), Eqs. (C9a)--(C9f)
   * - ``coulomb_finite_kperp``
     - Gyrokinetic Coulomb retaining finite :math:`k_\perp`
     - Frei, Ball, Hoffmann, Jorge, Ricci & Stenger (2021), Eqs. (3.47)--(3.50)

``none`` and ``lenard_bernstein`` both keep the built-in diagonal term described
in this section. The four moment operators *replace* that term with a dense
Hermite-Laguerre matrix; the solver disables the diagonal contribution exactly
when a moment operator is active, so collisions are never counted twice.

Two constraints follow from the tabulated coefficients. The run's moment count
``Nl*Nm`` must match a shipped table (8 or 18), and the operators run on the
fixed-step cached integrator, so ``use_diffrax = false``. The diffrax, sharded,
and Krylov eigenvalue paths raise rather than silently substituting the diagonal
term. The Coulomb tables are generated at unit mass and temperature ratio, so a
multispecies request is refused rather than extrapolated.

Collision frequency
^^^^^^^^^^^^^^^^^^^

The tabulated matrices carry only the dimensionless pair scaling
:math:`\nu_{ab} = n_b / (\sqrt{m_a}\,T_a^{3/2})`. The common collisionality
prefactor is applied from the species ``nu``, so every model sits on the same
collisionality axis as the built-in Lenard-Bernstein term.

.. warning::

   Three mutually incompatible normalizations of the collision frequency appear
   in this literature. Writing
   :math:`\nu = C\,n q^4 \ln\Lambda / (m^{1/2} T^{3/2})`, Sugama (2009, 2019)
   uses :math:`C = 4.4429`, Frei et al. (2021) and Frei, Ernst & Ricci (2022)
   use :math:`C = 2.3633`, and Frei, Hoffmann & Ricci (2022) use
   :math:`C = 0.7523`. Any comparison against a published figure must fix the
   convention first; it is the most common source of apparent agreement.

Baseline Lenard-Bernstein model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The built-in collisional model is a Lenard-Bernstein-style diagonal damping
plus conservation-restoring low-order moment corrections.

Base damping:

.. math::

   \mathcal{R}_{coll}^{base}
   =
   - w_{coll}\,\nu_s\,\Lambda_{\ell m}\,H_{\ell m},

where ``lb_lam`` is the cached Hermite/Laguerre collision eigenvalue.

The code then reconstructs low moments:

.. math::

   \bar{u}_\perp = \sqrt{b}\sum_\ell J_\ell^B H_{\ell,0},
   \qquad
   \bar{u}_\parallel = \sum_\ell J_\ell H_{\ell,1},

and a temperature-like correction :math:`\bar{T}` from ``m=0`` and ``m=2``.
These are added back only into the ``m=0,1,2`` channels.

Structural verification of the moment operators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Every shipped matrix is asserted against the published closed forms, not only
against internal consistency. All twelve Appendix-C coefficients of Frei, Ernst
& Ricci (2022) reproduce, for example

.. math::

   C^{(1,1)}_{(1,1)} = -\frac{28}{15}\sqrt{\frac{2}{\pi}},
   \qquad
   C^{(3,0)}_{(1,1)} = -\frac{8}{5}\sqrt{\frac{1}{3\pi}} .

GKX stores the opposite Laguerre sign convention to the paper, so a published
entry maps onto the stored one through :math:`(-1)^{j+j'}`; that flips exactly
the couplings between different Laguerre parities and leaves same-parity entries
alone, which makes the comparison a check on the convention as well as on the
values.

The properties a linearized collision operator must satisfy are gated
numerically in ``tests/validation/physics_gates/test_collision_conservation.py``:

.. list-table::
   :header-rows: 1
   :widths: 46 54

   * - Property
     - Result
   * - Density, parallel-momentum, energy conservation
     - machine precision (:math:`\le 2.2\times10^{-16}`)
   * - H-theorem, negative semidefinite symmetrized operator
     - holds for every model
   * - Onsager self-adjointness
     - exact (:math:`\le 3.4\times10^{-17}`)
   * - Published Appendix-C coefficients
     - reproduce to :math:`10^{-12}`
   * - Finite-Larmor :math:`b\to0` limit
     - reduces to the drift-kinetic operator exactly
   * - Finite-Larmor conservation defect
     - :math:`B^{1.96}`--:math:`B^{1.99}`, first order in :math:`b`

The invariants are the left null vectors of the moment matrix, since the
production of a moment functional :math:`v` is :math:`v^{T} C N`. With
Hermite-major index :math:`p(J+1)+j` they are :math:`e_0` (density), :math:`e_2`
(parallel momentum), and :math:`e_1 + e_4/\sqrt{2}` (energy), the last read off
the exact null space rather than assumed.

The finite-Larmor operator acts on *gyrocenter* moments, whose conservation and
plain self-adjointness are modified by gyroaveraging, so the ordering is the
test: both defects must vanish at :math:`b=0` and enter at first order in
:math:`b = B^2/2`. A kernel assembled at the wrong order would show
:math:`B^{1}` or :math:`B^{4}` instead.

Tabulated resolutions
^^^^^^^^^^^^^^^^^^^^^

Finite-Larmor tables ship at 8 and 18 Hermite-Laguerre moments, generated in
60-digit arithmetic on a 14-point Bessel-argument grid
:math:`B = k_\perp v_{\mathrm{th}}/\Omega \in [0, 4]` and stored as
checksummed float64. The runtime interpolates at :math:`B=\sqrt{2b}` from the
cached :math:`b`, so one table covers every perpendicular wavenumber, and it
selects the table matching the run's ``Nl*Nm`` automatically.

Cost and the resolution ceiling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The finite-Larmor operator interpolates its tables at every grid point, so the
compiler sees a distinct moment matrix per :math:`(k_y, k_x, z)`. That storage
grows as :math:`n^2` in the moment count while the state grows as :math:`n`.
Measured temporary storage in the compiled linear RHS, relative to the built-in
diagonal operator on a 16x8x32 grid:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - moments
     - diagonal operator
     - finite-Larmor Coulomb
   * - 8 ``(4, 2)``
     - 1.8 MB
     - 10.6 MB (5.9x)
   * - 18 ``(6, 3)``
     - 3.9 MB
     - 47.3 MB (12.1x)

This is comfortable at the resolutions GKX ships, and it is the mechanism that
bounds the reachable resolution. Extrapolating the per-point matrices to a
32x64x32 grid gives 0.07 GB at 8 moments, 0.34 GB at 18, but **17 GB at**
``(16, 8)`` and 275 GB at ``(32, 16)`` -- so the published convergence
resolutions are not reachable in this form on a 16 GB card.

A rank-:math:`R` separable factorization in the Bessel argument is the obvious
candidate, and the tables do support it: the :math:`B` dependence is smooth, and
:math:`R=10` reconstructs both shipped resolutions to about :math:`10^{-6}`.
A direct micro-benchmark of the naive form was not an unambiguous win, however
-- it traded roughly four times the arithmetic for the lower memory traffic, and
the compiler already fuses part of the interpolation -- so this is recorded as a
measured constraint rather than an implemented optimization.
``tests/validation/physics_gates/test_collision_operator_cost.py`` bounds the
current cost so a structural regression is caught.

Published convergence studies ask for considerably more than eight moments --
(16,8) for linear Cyclone-base-case ITG and (32,16) for the converged case -- so
resolution should be scanned rather than assumed. Generate further resolutions
with::

    python tools/artifacts/build_finite_wavelength_coulomb_data.py \
        --hermite 7 --laguerre 3 --digits 60 --workers 24 --check

Claim boundary and extension plan
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is a conserving Lenard--Bernstein/Dougherty-like model, not a complete
linearized gyrokinetic Landau operator. The low-order field-particle correction
is important: the operator cannot be represented only by a diagonal damping
array. The implementation contract for collision extensions therefore has two
paths. Both receive a post-field ``CollisionContext`` so finite-Larmor-radius
models can distinguish the evolved distribution :math:`G` from the
Hamiltonian response :math:`H`:

- ``apply(context)`` for the complete unit-weight RHS,
  including low-rank or dense field-particle terms;
- ``SplitCollisionOperator.split_step(context, dt)`` as an
  optional contract only when the model supplies a mathematically valid exact
  or implicit finite-time update. The runtime does not automatically route this
  method yet. Diagonal hypercollision splitting must not be reused for a
  non-diagonal conserving operator.

The first path is available from Python through
``nonlinear_rhs_cached(..., collision_operator=operator)``. The callback must
return a JAX array with the state shape. GKX removes the built-in
collision contribution before adding ``terms.collisions * operator.apply(...)``;
hypercollisions remain independent:

.. code-block:: python

   class CollisionModel:
       def apply(self, context):
           return collision_rhs(
               context.distribution,
               context.hamiltonian,
               context.cache,
               context.parameters,
           )

   rhs, fields = nonlinear_rhs_cached(
       state, cache, parameters, terms,
       collision_operator=CollisionModel(),
   )

The callback is evaluated after the field solve and traced by JAX, so its array
operations remain differentiable. ``context.fields`` carries ``phi``, ``apar``,
and ``bpar``; ``context.hamiltonian`` uses the same enabled-field policy as the
gyrokinetic RHS. This avoids silently replacing a finite-:math:`b`
field-particle model by a :math:`G`-only approximation.
This contract is also used by the generated equal-species finite-wavelength
Coulomb and Sugama validation operators described below. Those operators are
research validation paths, not input-file options: TOML selection and split
integration remain disabled until the full zonal-response acceptance gate
passes.

The built-in ``collision_split`` policy consequently splits only diagonal
hypercollisions. The conserving collision term remains in the Runge--Kutta or
IMEX RHS, including its field-particle corrections. Earlier implementations
removed the complete collision RHS and advanced only its diagonal part; that
violated the stated conservation model and is no longer supported.

``collision_invariant_rates`` returns the discrete long-wavelength density,
parallel-momentum, and thermal-energy rates of a state-shaped contribution.
``collision_quadratic_rate`` evaluates
:math:`\operatorname{Re}\langle H,C[H]\rangle` with optional species/spatial
weights. Release tests use these functions to verify a local-Maxwellian null
space, all three fluid invariants, and dissipative non-fluid response at
:math:`b=0`.

``multispecies_collision_invariant_rates`` supplies the stricter acceptance
contract for a species-coupled model. For species-normalized coefficients it
returns each particle-density rate and the physically weighted sums

.. math::

   \dot P_\parallel = \sum_s n_s\sqrt{m_s T_s}\,\dot N_s^{10},
   \qquad
   \dot E = \sum_s n_s T_s
   \left(\sqrt{2}\,\dot N_s^{20}+2\dot N_s^{01}\right).

The model is promotable only when every particle rate and both summed rates
vanish to discretization tolerance. This diagnostic is implemented and
autodiff-tested; it does not by itself promote a multispecies collision model.

Reduced drift-kinetic Sugama equation gate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``drift_kinetic_sugama_six_moment_contribution`` and
``drift_kinetic_coulomb_six_moment_contribution`` implement the complete
like-species six-gyromoment matrices reported in Appendix C, equations
(C6a)--(C6f) and (C9a)--(C9f), of the `improved Sugama moment formulation
<https://arxiv.org/abs/2202.06293>`_.  In GKX ordering, the nontrivial
moment vector is

.. math::

   \boldsymbol{N}=(N^{20},-N^{01},N^{30},-N^{11})^T,

where the signs on Laguerre moments account for the code's polynomial
convention.  The operator is :math:`\nu_s M\boldsymbol{N}` with two symmetric
blocks.  The thermal block is

.. math::

   M_T=\frac{1}{45\sqrt{\pi}}
   \begin{pmatrix}
   -64\sqrt{2} & 64\\
   64 & -32\sqrt{2}
   \end{pmatrix},

and the heat-flux block is

.. math::

   M_q=\frac{1}{525\sqrt{\pi}}
   \begin{pmatrix}
   -1083\sqrt{2} & 624/\sqrt{3}\\
   624/\sqrt{3} & -1187\sqrt{2}
   \end{pmatrix}.

For the exact linearized Coulomb operator, the corresponding blocks are

.. math::

   M_T^L=\frac{1}{15\sqrt{\pi}}
   \begin{pmatrix}
   -16\sqrt{2} & 16\\
   16 & -8\sqrt{2}
   \end{pmatrix},
   \qquad
   M_q^L=\frac{1}{15\sqrt{\pi}}
   \begin{pmatrix}
   -24\sqrt{2} & 24/\sqrt{3}\\
   24/\sqrt{3} & -28\sqrt{2}
   \end{pmatrix}.

Tests evaluate every matrix entry, symmetry, non-positive eigenvalues, the
Maxwellian thermal null direction, density/momentum/energy invariants, and a
collision-frequency JVP against centered finite differences.  This is a real
high-collisionality reduced operator and an equation-level acceptance gate for
the future coefficient generator.  It intentionally returns zero outside the
six-moment projection and is therefore not selected by TOML or presented as a
full Hermite--Laguerre hierarchy, finite-Larmor-radius, multispecies, or
production Sugama/Coulomb implementation.

The same C6/C9 coefficients also exercise the production table boundary. The
maintainer command

.. code-block:: console

   python tools/artifacts/build_linear_validation_artifacts.py collision-table

evaluates the analytic coefficients with 80-decimal-digit ``mpmath``
arithmetic, writes a deterministic ``float64`` array, and records its SHA-256,
mode ordering, polynomial convention, source equations, and claim scope in a
JSON sidecar. ``load_collision_moment_matrix(model)`` verifies the package-data
checksum before returning a host array. ``apply_collision_moment_matrix`` then
packs ``(ell,m)`` into the paper's Hermite-major ordering, applies either one
shared or one matrix per species in JAX, and restores the state layout.
``interpolate_collision_moment_matrix`` provides the finite-:math:`b` runtime
boundary: it linearly interpolates a strictly increasing :math:`k_\perp` table
on device, clamps only outside the generated range, and accepts either one
shared table, an explicit table per species, or an ordered target/source pair
table with shape ``(target, source, kperp, moment, moment)``. Pair tables use
the target species' :math:`k_\perp` field, matching the species-dependent
gyroradius convention, and return the spatial matrix layout consumed directly
by ``apply_multispecies_collision_moment_matrix``. The resulting matrix may
vary at every perpendicular/parallel grid point without leaving traced JAX
execution.
``TabulatedMultispeciesCollisionOperator`` exposes this finite-wavelength
boundary through the standard collision protocol. It is a JAX pytree containing the coefficient
grid and fully assembled, collision-frequency-weighted pair table; its
``apply`` method obtains :math:`k_\perp\rho_s=\sqrt{b_s}` from the solver cache,
interpolates each target/source block, and acts on the post-field Hamiltonian.
In contrast, ``DriftKineticMomentCollisionOperator`` implements the
drift-kinetic convention of Frei, Ernst & Ricci (2022), Eq. (73): its dense
matrix acts directly on evolved gyrocenter moments because :math:`f\simeq g`
in that limit. A full-RHS gate checks both conventions independently.
Generated tables, rather than the runtime
operator, own directed-frequency normalization and coefficient provenance.
Tests require node and endpoint identity, generated-table/direct-equation
identity for both models, species-local spatial application, and JVP/finite-
difference agreement through state amplitude, collision frequency, and an
interior :math:`k_\perp` target. The ordered-pair gate additionally checks JIT
application, pair-block identity, zero-:math:`b` multispecies invariants, and
target-species-leading spatial interpolation. A separate held-out gate constructs matrices
from the implemented Mandell--Dorland--Landreman finite-Larmor-radius collision
equations, never from the interpolator, and recovers the expected second-order
table-spacing convergence against direct operator evaluations.

This table boundary now supports two deliberately distinct levels. The small
package-data table contains only the drift-kinetic six-moment matrices and is
used for fast equation and API tests; repeating it at higher resolution would
not create finite-:math:`b` physics. The offline generator separately builds
the complete equal-species diagonal Coulomb test/field matrices and
polarization vectors at arbitrary retained Hermite--Laguerre order. Original-
and improved-Sugama field matrices are derived from that exact Coulomb test
table. These larger archives stay outside the package and are accepted only
through the slab-ITG and collisional-zonal runtime gates.

The first exact primitive used by that generated hierarchy is
``bessel_laguerre_kernels(bessel_argument, n_max)`` in ``core.velocity``. It
implements Frei et al. (2021), equation (2.13),

.. math::

   K_n(B) = e^{-B^2/4}\frac{(B^2/4)^n}{n!},

where :math:`B=k_\perp v_{\mathrm{th}}/\Omega`, using the stable recurrence
:math:`K_{n+1}=K_n B^2/[4(n+1)]`. These coefficients expand
:math:`J_0(B\sqrt{x})` in Laguerre polynomials and decay factorially,
motivating the paper's finite-sum rule :math:`N>B^2/4`. The validation suite
compares them with an independent 96-point Gauss--Laguerre projection,
verifies the reported sub-0.1% tail at :math:`B=1,N=3`, and checks
JIT and JVP/finite-difference agreement. This validates one generator building
block; it does not supply the collision-specific coupling coefficients.

``associated_bessel_laguerre_coefficients`` implements the complete
equation-(2.12) prefactor for arbitrary non-negative Bessel order :math:`m`,

.. math::

   A_n^m(b)=\frac{n!}{(n+m)!}\left(\frac{b}{2}\right)^m K_n(b).

Direct reconstructions of :math:`J_m(b\sqrt{x})` for :math:`m=0,1,2` agree
with an independent special-function implementation over the tested
wavenumber and velocity domain. This closes the Bessel-expansion layer used by
the collision sums, but not the speed-function or test-/field-particle
contractions themselves.

The offline generator also evaluates the Coulomb speed integrals
:math:`e_{ab}^k` and :math:`E_{ab}^k` from Appendix A, equations (A8a)--(A8b),
with 80-digit arithmetic. Orders zero through five at three unequal thermal-
speed ratios agree with direct improper quadrature of their defining
integrals. ``coulomb_speed_moments`` now composes those integrals into the
velocity-integrated test and field functions in equations (A5) and (A13).
Six cases spanning :math:`0.25\leq m_a/m_b\leq4`,
:math:`0.5\leq T_a/T_b\leq3`, and spherical order zero through three agree
with direct three-dimensional Maxwellian quadrature of equations (A2) and
(A10). The equal-species density moment vanishes and the momentum test/field
pieces cancel. These remain generator internals because the complete matrix
contractions that consume them are not yet implemented.

The same generator evaluates equation (3.10)'s monomial coefficients for
:math:`L_j^{p+1/2}(x)`. Independent generalized-Laguerre evaluations verify
tensor orders :math:`p=0,1,4` through polynomial order :math:`j=8`. The next
basis-transform coefficients are cancellation-sensitive. Their provenance is
now closed: the base transform is Appendix A, equation (A4), of
`Jorge, Ricci & Loureiro (2017) <https://arxiv.org/abs/1709.01411>`_, and the
finite-:math:`m` transform and inverse are Appendix B, equations (B5)--(B6), of
`Jorge, Frei & Ricci (2019) <https://arxiv.org/abs/1906.03252>`_. Both formulas
were audited against their defining basis identity rather than accepted as
printed.

The isotropic base transform and inverse are now implemented in the offline
generator from equations (A4) and (A3), respectively. Selected coefficients
agree with independent 80-point Gauss--Hermite/Gauss--Laguerre velocity
projections, including the hand identities :math:`cP_1=H_1/2` and
:math:`c^2P_2=H_2/4+L_1/2`. Forward/inverse products close through total degree
12 with maximum error ``8.73e-15`` even though that shell's condition number is
``1.93e8``. All nested sums remain multiprecision until the final table cast.

The finite-:math:`m` forward transform is now generated as a complete lower-
triangular parity block. Under SciPy's associated-Legendre convention, a
literal equation-(B5) transcription gives half the independently projected
:math:`m=0` coefficients and the opposite sign for odd :math:`m`; the required
factor :math:`2(-1)^m` is fixed independently by the :math:`m=0` endpoint,
eight velocity-space projections, and pointwise reconstruction. Unlike the
isotropic map, every lower reduced-degree shell of the same parity is retained.
Even and odd blocks through reduced degree six reconstruct the physical basis
for :math:`m=0,1,2,3`. Literal equation (B6) fails the finite-:math:`m` inverse
identity. Equation (3.33) of Frei et al. (2021), which includes the weighted
Laguerre-product contraction omitted from that direct normalization, matches
every entry of independently inverted degree-six blocks for
:math:`m=0,1,2,3`. Complete 80-digit block inversion remains the independent
oracle; equation (3.33) supplies the scalar inverse used by collision-matrix
assembly. This closes coefficient generation, not the test-/field-particle
contractions or their transport validation.

The next algebraic layer is also generated and independently checked.
``laguerre_product_expansion_coefficient`` implements both the unweighted
product in equations (3.44)--(3.45) and the :math:`x^m`-weighted product in
equations (3.36)--(3.37) of Frei et al. (2021). Pointwise polynomial
reconstruction covers :math:`m=0,1,2`. Combining that product with the
finite-:math:`m` transform and :math:`K_n(b)` yields
``gyroaveraged_spherical_moment_coefficient``, one coefficient of equation
(3.35). Six coefficients through :math:`m=3` and :math:`b=1.3` agree with
independent Bessel-weighted velocity projection; 20- and 32-term Bessel sums
also agree. This validates the gyro-moment-to-spherical-moment map consumed by
the Coulomb contractions.

The offline generator now contracts equations (3.48)--(3.49) into complete
finite-:math:`b` test- and field-particle matrices with explicit Hermite,
Laguerre, spherical-harmonic, and Bessel truncations. Unlike-species generation
keeps :math:`b_a=k_\perp\rho_a` in the test and outer gyroaverage factors and
:math:`b_b=k_\perp\rho_b` in the field-particle source moments; a regression
holds :math:`b_a` fixed and verifies that only the field block changes with
:math:`b_b`. At :math:`b=0`, every
published nonzero six-moment Coulomb entry is recovered. The larger generated
block is symmetric to ``8.33e-17``, negative semidefinite, and preserves
density, parallel momentum, and thermal energy within ``3.3e-16``. Equation
(3.41)'s :math:`\Pi^{pjm}` agrees with five independent
:math:`J_0J_m`-weighted velocity projections to about :math:`10^{-13}`.
Equation (3.50) remains four separate vectors multiplying
:math:`q_a\phi/T_a` and :math:`q_b\phi/T_b`, because quasineutrality couples
species; like-species test/field polarization cancels. This closes the offline
algebra, not runtime multispecies assembly or transport promotion.

The reproducible algebra/convergence artifact is generated with

.. code-block:: console

   python tools/artifacts/build_linear_validation_artifacts.py collision-verification

.. figure:: _static/collision_operator_verification.png
   :alt: Coulomb collision operator convergence, projection, entropy, and matrix gates
   :width: 100%

   **Offline Coulomb-operator closure.** Panel (a) shows Bessel--Laguerre
   convergence of a finite-:math:`b` polarization coefficient to a 24-term
   reference, Bessel convergence of an assembled 4-by-4 collision block, and
   the independent spherical/radial hierarchy convergence of that block;
   panel (b) compares five generated coefficients with independent
   80-by-80 Gauss--Hermite/Laguerre velocity projection; panel (c) shows five
   dissipative modes and the three density, parallel-momentum, and thermal-
   energy null modes, while its inset verifies leading :math:`O(b^2)` classical
   gyro-diffusion away from the drift-kinetic limit; panel (d) exposes the
   complete retained drift-kinetic moment block. The machine-readable gate is
   stored beside the figure in ``collision_operator_verification.json``.

The largest direct-projection relative error is :math:`3.2\times10^{-13}`;
the published-coefficient, symmetry, and invariant residuals are at or below
:math:`1.2\times10^{-16}`. At :math:`b=0.8`, the assembled 4-by-4 block changes
by :math:`4.50\times10^{-7}` between Bessel orders four and six at the admitted
:math:`(p_{\max},j_{\max})=(8,4)` spherical cutoff. The former
default spherical cutoff :math:`(p_{\max},j_{\max})=(3,1)` is rejected because
it differs by 29% from the converged :math:`(9,4)` reference. The admitted
:math:`(8,4)` block reduces that error to :math:`8.68\times10^{-7}`. These checks
implement the conservation,
Maxwellian-null, adjointness, and H-theorem requirements emphasized by
`Abel et al. (2008) <https://arxiv.org/abs/0808.1300>`_ and the
finite-:math:`b` moment algebra of
`Frei et al. (2021) <https://arxiv.org/abs/2104.11480>`_. The quadrature check
is a deterministic manufactured-projection test: it verifies the generated
operator against its continuous velocity-space definition, independently of
the symbolic contraction path.

Physical promotion is deliberately stricter. After multispecies
quasineutrality is assembled, the operator must reproduce Spitzer--Härm/
Braginskii transport, the weakly collisional Hermite--Laguerre convergence and
finite-:math:`b` ITG scans, and the separately defined collisionless
Rosenbluth--Hinton residual and Hinton--Rosenbluth collisional damping traces.
The target resolutions, observables, and figure protocols follow Figures
4--9 of Frei et al. (2021) and the conductivity study of
`Frei, Ernst & Ricci (2022) <https://arxiv.org/abs/2202.06293>`_. Until those
runtime gates pass, the panel supports operator algebra and numerical closure,
not a production Landau-transport claim.

The runtime research boundary now mirrors the same decomposition.
``FiniteWavelengthCoulombOperator`` stores test, field, and four polarization
tables with independent target/source Bessel-argument axes
:math:`B_a=k_\perp v_{Ta}/\Omega_a`. The gyrokinetic cache stores
:math:`b_a=k_\perp^2T_am_a/(q_aB)^2`, so a bilinear JAX interpolator evaluates
the table at :math:`B_a=\sqrt{2b_a}` and :math:`B_b=\sqrt{2b_b}`; the resolved
kernel applies equations (3.48)--(3.49) to gyrocenter moments :math:`G_a` and
:math:`G_b`, then adds equation (3.50) using the solved potential and distinct
:math:`q_a/T_a` and :math:`q_b/T_b` factors. It intentionally does not apply
the matrices to ``build_H`` because that would double-count the pullback
polarization.

For a single like-species plasma,
``EqualSpeciesFiniteWavelengthCoulombOperator`` stores only the physical
diagonal :math:`B_a=B_b` of those tables. This is not a reduced collision
model: it retains the same test, field, and four polarization terms, while
avoiding coefficient generation on target/source wavelength pairs that cannot
occur in the one-ion-species problem. Its one-dimensional interpolation is
JIT compatible and differentiable with respect to the local Bessel argument.
Multispecies input is rejected rather than silently applying the diagonal
assumption.

Independent Python pair loops, JIT execution, JVP/finite-difference checks,
like-species polarization cancellation, generated-coefficient application,
full-pair/diagonal identity, and the complete cached linear-RHS seam pass. This
closes runtime algebra and differentiable interpolation. It does not yet close Hermite/Laguerre
truncation or any transport benchmark; therefore this class remains a Python
research API and has no input-file selector.

Finite-:math:`b` conservation must be stated carefully. Collisions are local at
the particle position, whereas the evolved moments are defined at the
gyrocenter. Consequently, the gyrocenter density row is not a null row at
finite :math:`k_\perp\rho`; it represents classical gyro-diffusion. The tracked
verification artifact recovers the density null at :math:`b=0`, obtains a
nonzero finite-:math:`b` row, and measures the expected leading
:math:`O(b^2)` scaling separately for the test, field, and combined rows.
Equation (3.35) maps a gyrocenter distribution to particle moments; it is not
an inverse of the gyrophase average in equation (3.5). Local particle-space
conservation therefore cannot be inferred by applying that moment map to the
already gyroaveraged collision matrix. A future direct particle-coordinate
implementation must test equations (3.2)--(3.4) before gyroaveraging instead.

The multiprecision generator uses exact integer combinatorics for polynomial
binomial factors, memoizes repeated inverse-basis contractions within one
assembly, and retains arbitrary precision where gamma functions and
non-integer coefficients require it. These changes preserve the generated
blocks bit for bit while reducing a representative assembly from 34.1 to 3.46
seconds. Applying equation (3.35)'s :math:`M^{000}` map to the gyroaveraged
collision matrix leaves a nonzero residual, as equation (3.5) predicts; that
quantity is retained only as a rejected diagnostic, not a conservation or
resolution gate.

The lowest-order multispecies drift-kinetic boundary is also implemented
without assuming equal species. For an ordered pair :math:`(a,b)`,
``drift_kinetic_sugama_pair_matrices`` evaluates Appendix C, equations
(C4)--(C5), of Frei, Ernst & Ricci (2022) as

.. math::

   \mathcal C_a = \sum_b \nu_{ab}
   \left(T_{ab}\,N_a + F_{ab}\,N_b\right),
   \qquad
   \sigma=\frac{m_a}{m_b},\quad \tau=\frac{T_a}{T_b}.

Here :math:`T_{ab}` and :math:`F_{ab}` are the published test- and
field-particle matrices on the eight-mode ``Nl=2, Nm=4`` space; coefficients
outside the six active moments are exactly zero. The helper returns matrices
normalized by the directed frequency

.. math::

   \nu_{ab}\propto \frac{n_b}{\sqrt{m_a}\,T_a^{3/2}},

so callers retain explicit ownership of normalization. The separate
``apply_multispecies_collision_moment_matrix`` contract stores target species
first and source species second, applies all source blocks in one JAX
contraction, and supports pointwise spatial matrices. At equal mass and
temperature, :math:`T_{aa}+F_{aa}` reproduces the independent 80-digit C6
table. ``assemble_drift_kinetic_sugama_matrix`` vectorizes all ordered pairs
and adds each test-particle block to its target-species diagonal. An unequal
ion-pair gate checks published coefficients directly; a physical
directed-frequency gate conserves each species' particles and total momentum
and thermal energy, produces a negative weighted quadratic rate, and matches
finite differences through :math:`\sigma` and :math:`\tau`. An independent
matrix-exponential trajectory preserves those invariants through unequal-
species relaxation and reduces the collision residual by more than five
orders of magnitude.

For Python solver experiments,
``DriftKineticMomentCollisionOperator.from_species`` wraps that matrix in the
standard collision protocol. Its ``apply`` method uses
``CollisionContext.distribution`` as required by the drift-kinetic limit. A
collision-only two-species linear-RHS gate verifies a
nonzero response and the same physical invariants. The operator is a JAX
pytree, preserving differentiation when species parameters are constructed
inside an objective.

This is the original Sugama model's real low-order drift-kinetic projection.
It is useful for reduced-model verification but is not an arbitrary-moment
hierarchy or a finite-:math:`b` multispecies runtime model.

Lowest-order improved-Sugama correction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``drift_kinetic_improved_sugama_pair_matrices`` adds the complete low-order
test- and field-particle corrections in Appendix C, equations (101)--(102), to
the original-Sugama ordered pair. The paper labels the driven moment with a
superscript and the response with a subscript, so the published coefficient
array is transposed once into the runtime row/column application convention.
For equal species the test correction vanishes and the field correction
reproduces equations (103a)--(103c) from an independently generated 80-digit
table. Unequal-mass and unequal-temperature pair coefficients are checked
directly, and their assembled matrix conserves particle number, total parallel
momentum, and total thermal energy. At equal species the correction reduces
the heat-flow-block Frobenius distance to the Coulomb matrix from about
``0.521`` to ``0.205``; the equal-temperature multispecies weighted symmetric
operator is non-positive over the complete reduced moment space.

``assemble_drift_kinetic_improved_sugama_matrix`` and
``DriftKineticMomentCollisionOperator.from_improved_species`` expose this equation slice
through the same vectorized JAX and collision-protocol paths. This is a
friction-flow matrix validation, not a parallel-conductivity claim. The
published conductivity comparison retains more moments and reports that the
original operator can underpredict current by at least 10%, while the improved
operator approaches Coulomb within 1%; GKX therefore keeps
conductivity promotion blocked until the arbitrary-moment correction hierarchy
and its driven steady-state gate are implemented.

Driven parallel-current response
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The response algorithm needed by that promotion is now an explicit JAX
contract. For a Maxwellian electron background, equation (81) of Frei, Ernst
& Ricci (2022) linearizes to

.. math::

   \frac{dN_e^{pj}}{dt}
   = \sum_{p'j'} C_{pj,p'j'}N_e^{p'j'} + s_E^{pj},
   \qquad
   s_E^{pj}=-\sqrt{2}\,\widehat E\,\delta_{p1}\delta_{j0},

where :math:`\widehat E=eE/(v_{Te}m_e)` and the electron current follows from
:math:`u_e=N_e^{10}v_{Te}/\sqrt{2}`. ``parallel_electric_field_source``
constructs this source in Hermite-major ordering, and
``solve_driven_collision_response`` solves
:math:`C N_e+s_E=0` after explicitly removing invariant or intentionally
truncated modes. The solve uses ``jax.numpy.linalg.solve`` and passes JIT,
steady-time-limit, and AD/finite-difference gates.

The lowest-order Appendix-C original and improved Sugama blocks provide a
useful equation-level boundary: at :math:`Z=1` the improved block carries over
10% more current than the original, while their difference is below 1% by
:math:`Z=100`, where pitch-angle scattering dominates. This reproduces the
published qualitative ordering, but it is not a Spitzer-conductivity result.
The absolute-conductivity gate still requires arbitrary-order original and
improved matrices at :math:`(P,J)=(20,5)` alongside the now-generated Coulomb
hierarchy, matched collision-frequency normalization, saturation under
:math:`eE/(\sqrt{m_eT_e}\nu_{ee})=10^{-3}`, and the three-model Figure-16 scan
over ion charge. No input-file collision selector is enabled by this response
utility.

The direct Coulomb contraction now resolves that response through
:math:`(P,J)=(20,5)` without evaluating the finite-wavelength Bessel hierarchy.
At the final point, the largest current change from :math:`(15,5)` is
:math:`1.66\times10^{-4}` over :math:`Z=1,2,5,10,100`; the invariant,
self-adjointness, non-positive-spectrum, and driven-solve residuals all pass a
:math:`2\times10^{-12}` algebra gate. The figure is regenerated from equations
(3.53)--(3.56), not from stored coefficient arrays:

**Generated figure.** Drift-kinetic Coulomb and Sugama response hierarchies. Panel (a)
compares
Coulomb with the arbitrary-order original and improved models, panel (b) is
the nested velocity-space error, panel (c) records the conservation and
dissipation gates, and panel (d) reproduces stationary-current saturation at
the paper's applied-field normalization.


The machine-readable JSON and CSV use the same prospectively fixed 0.5%
current and :math:`2\times10^{-12}` algebra thresholds. At equal temperature,
the arbitrary-order original-Sugama test matrix equals the Coulomb test matrix;
its field matrix is the self-adjoint low-rank restoration of momentum and
thermal energy. This construction reproduces every published C6 coefficient
at low order and, at :math:`(P,J)=(20,5)`, yields 11.29% less current than
Coulomb at :math:`Z=1` and only 0.61% less at :math:`Z=100`. Prospectively
fixed gates require at least an 8% low-charge deficit and no more than a 2%
high-charge difference, reflecting the Figure-16 ordering. The improved field
correction is formed
from the Coulomb Braginskii :math:`N` matrix Schur complement and the exact
drift-kinetic transforms in equations (79)--(81) of Frei, Ernst & Ricci (2022).
It reproduces every C103 coefficient at :math:`K=1`; the shipped hierarchy
retains complete total-degree shells through :math:`K=5`, checks the final
:math:`K=4\rightarrow5` response change, and requires agreement with Coulomb
within 1% at every scanned charge. The measured maximum changes are 0.439% in
the final correction-order step and 0.0237% from :math:`(P,J)=(15,5)` to
:math:`(20,5)`; the largest improved-to-Coulomb difference is 0.307%.
Regenerate all three formats with

.. code-block:: bash

   python tools/artifacts/build_linear_validation_artifacts.py collision-response

For the absolute normalization, define
:math:`\widehat E=eE/(m_ev_{Te}\nu_{ee})`. The plotted response
:math:`(u_e/v_{Te})/\widehat E` is then
:math:`\sigma_\parallel/[n_e e^2/(m_e\nu_{ee})]`. The black curve in panel (a)
is the Spitzer high-charge limit
:math:`64/[3\,2^{3/2}\pi Z]`; the :math:`Z=100` Coulomb point differs by 7.453%,
inside the fixed 8% gate. Panel (d) uses the equivalent paper convention
:math:`eE/(\sqrt{m_eT_e}\nu_{ee})=10^{-3}`. Its three exact matrix-exponential
traces saturate by :math:`t\nu_{ee}=50`; independent steady solves over a 100x
field range close the linear-response gate. These results promote the
unmagnetized equal-temperature conductivity problem, not finite-:math:`b`
collisional transport.

A deterministic Cyclone ITG probe also records the finite-wavelength failure
boundary rather than hiding it. At :math:`k_y\rho\simeq0.63`, increasing the
normalized collision weight from zero to three damps the fitted growth rate;
at :math:`k_y\rho\simeq0.94`, the same drift-kinetic model instead excites the
short-wave branch. This is the behavior identified when collisional FLR terms
are omitted in the local collisional-ITG study. The regression test therefore
requires both observations and keeps configuration-file selection fail-closed.
Only a finite-:math:`b` operator may promote the short-wavelength lane.

The finite-:math:`b` runtime path now has a shared multiprecision pair-table
builder that amortizes wavelength-independent basis algebra and converts the
paper's Laguerre signs to the runtime convention. This is an implementation
prerequisite, not the integrated ITG gate. A low-order :math:`(P,J)=(3,1)`
development probe still excites the short-wave branch even with the complete
finite-:math:`b` contribution. Consistent with the paper's convergence study,
the result is rejected as velocity-space under-resolution. Increasing from
:math:`(3,1)` to :math:`(5,2)` lowers its fitted collisional growth from 0.773
to 0.520, but does not suppress it. Promotion requires the independently
converged :math:`(18,6)` endpoint and matching :math:`k_\perp`/collisionality
scans; no reduced probe or favorable trend substitutes for it.

Offline generation also factors the equation-(B5) associated-basis overlap
exactly into a Gamma-weighted radial Laguerre contraction and a differentiated
angular Legendre contraction, then reuses that overlap across each parity
shell. At :math:`(P,J)=(5,2)`, with two wavelengths, spherical/radial cutoffs
9/4, Bessel--Laguerre cutoff 6, and 32 decimal digits, this reduced wall time
from 184.01 s to 26.41 s (7.0x) without changing the table checksum. This is a
table-generation result, not a simulation or transport speedup claim.
A higher :math:`(P,J)=(7,3)` two-wavelength table now completes serially in
411.22 s inside the external campaign bound. Its zero-wavelength test/field
blocks agree with the independently generated drift-kinetic equations to
relative errors :math:`1.08\times10^{-31}` and :math:`1.92\times10^{-32}`.
This is validated hierarchy evidence, but remains below the required
:math:`(18,6)` ITG endpoint.

The equal-species original-Sugama finite-wavelength slice is now separate from
that Coulomb field model. At equal mass and temperature, equations (3.72)--
(3.76) of `Frei et al. (2021) <https://arxiv.org/abs/2104.11480>`_ make its test
component exactly the Coulomb test component. Equations (3.79) and
(3.65), (3.68)--(3.69), and (3.80) make the field component the sum of three
explicit rank-one parallel-flow, perpendicular-flow, and energy responses.
The response vectors are projected directly with product velocity quadrature;
they are not inferred by forcing finite-wavelength gyrocenter moments into the
null space. The offline conversion reuses the expensive Coulomb test table::

   python tools/artifacts/build_linear_validation_artifacts.py \
     collision-original-sugama-table \
     --coulomb-table finite_b_zonal_P24_J10_diagonal.npz \
     --out finite_b_zonal_P24_J10_original_sugama.npz \
     --quadrature-order 80

The archive conversion requires complete angular coverage, the signed runtime
Laguerre convention, finite coefficients, and agreement between 80- and
96-node projections. Its equation test reproduces the published drift-kinetic
C6 construction at :math:`B=0` to roundoff and limits the finite-wavelength
field block to a symmetric rank-three matrix. At finite :math:`B`, the flow
channels have the nonzero gyro-diffusive action required by the gyroaveraged
operator, while the symmetric part of the complete retained collision matrix
remains strictly dissipative in the equation-level gate.
``EqualSpeciesFiniteWavelengthSugamaOperator``
interpolates the resulting table in JAX and acts on the post-field
nonadiabatic response :math:`H`; JIT/JVP and finite-difference interpolation
gates pass, but a paper-facing zonal trace is still required for promotion.

The finite-wavelength improved-Sugama field correction is generated from the
same Coulomb archive. At equal mass and temperature the test correction
vanishes, while equations (61)--(69) of
`Frei, Ernst & Ricci (2022) <https://arxiv.org/abs/2202.06293>`_ add the
Braginskii field correction through parallel and perpendicular generalized
flow projections. The implementation evaluates those velocity integrals
independently with product Gauss--Hermite/Laguerre quadrature and accepts the
higher-order result only when an additional 16 nodes change the correction by
less than :math:`10^{-11}` relative or :math:`10^{-12}` absolute::

   python tools/artifacts/build_linear_validation_artifacts.py \
     collision-improved-sugama-table \
     --coulomb-table finite_b_zonal_P24_J10_diagonal.npz \
     --out finite_b_zonal_P24_J10_improved_sugama.npz \
     --correction-order 5 --quadrature-order 80 --digits 80

At :math:`B=0`, correction orders :math:`K=1,2,3` reproduce the independent
drift-kinetic analytical implementation with relative matrix errors from
:math:`3.4\times10^{-15}` to :math:`1.1\times10^{-14}`. At the production
P24/J10, :math:`K=5` resolution, nested 80/96-node checks over
:math:`B=0.12,0.16,0.24,0.32` differ by at most :math:`7.1\times10^{-14}`
relative. These are equation and quadrature gates. The Figure-13/14 zonal
ordering and velocity-section comparison is the independent physical gate; it
now passes at P24/J10 through :math:`t\nu=30`. At
:math:`k_x\rho_i=0.1`, the late responses are 0.00242 (original), 0.00255
(improved), and 0.00288 (Coulomb); at :math:`k_x\rho_i=0.2`, they are
:math:`9.66\times10^{-5}`, :math:`1.15\times10^{-4}`, and
:math:`1.92\times10^{-4}` in the same order. The improved early-window RMS
error relative to Coulomb is lower than the original-model error at both
wavenumbers.
The common finite-wavelength runner now accepts provenance-matched Coulomb,
original-Sugama, and improved-Sugama archives. For the :math:`k_x=0.2` trace it
can checkpoint the physical :math:`t\nu=5` state, reconstruct the equation-(52)
parallel and perpendicular gyrocenter-distribution cuts at the outboard
midplane, and continue the same integration to :math:`t\nu=30`.

As a separate full-distribution reference utility,
``conservative_full_f_dougherty_cross_moments``. For directed collision rates
:math:`\nu_{sr}`, it evaluates the pairwise primitive moments

.. math::

   u_{sr} =
   \frac{m_s n_s \nu_{sr} u_s + m_r n_r \nu_{rs} u_r}
        {m_s n_s \nu_{sr} + m_r n_r \nu_{rs}},

.. math::

   (n_s\nu_{sr}+n_r\nu_{rs})m_s v_{t,sr}^2
   =m_s n_s\nu_{sr}v_{t,s}^2+m_r n_r\nu_{rs}v_{t,r}^2
   +\frac{m_s n_s\nu_{sr}m_r n_r\nu_{rs}}
          {m_s n_s\nu_{sr}+m_r n_r\nu_{rs}}
    \frac{(u_s-u_r)^2}{d_v}.

These are equations (2.11)--(2.12) of the
`improved multispecies Dougherty derivation <https://doi.org/10.1017/S0022377822000289>`_.
Francisquez et al. derive a nonlinear full-:math:`f` Fokker--Planck model,
whereas GKX evolves a linearized delta-:math:`f` gyrokinetic state. The
JAX implementation accepts arbitrary mass ratios and directed rates, keeps
zero-rate and self pairs unchanged, and is equation-gated for pairwise momentum
and energy conservation, the equal-species limit, positive target temperature,
and AD/finite-difference agreement. A three-species, multi-sample gate also
checks every interacting pair for :math:`d_v=1,2,3` and verifies Galilean
invariance of the target flow and thermal speed. It supports derivation checks
and future full-:math:`f` work; it must not be inserted directly into the
shipped linearized field-particle restoration.

The next runtime model tier extends this verified low-order normalization to
the published linearized gyrokinetic Sugama/Coulomb operators in the full
Hermite--Laguerre moment basis. A linearized
multispecies Dougherty variant is admissible only after its delta-:math:`f`
projection is derived explicitly; the full-:math:`f` primitive targets are not
used as a shortcut. Promotion requires discrete Maxwellian
null-space, particle conservation per species, total momentum and energy
conservation, adjointness, non-positive entropy production, velocity-resolution
convergence, collisional ITG, conductivity, and zonal-flow damping gates.

Relevant derivations and verification targets include the
`Laguerre--Hermite pseudo-spectral formulation <https://doi.org/10.1017/S0022377818000041>`_,
the `advanced linearized gyrokinetic moment operators <https://arxiv.org/abs/2104.11480>`_,
the `improved Sugama moment implementation <https://arxiv.org/abs/2202.06293>`_,
and the `local collisional ITG study <https://arxiv.org/abs/2201.02860>`_.
The independent GYACOMO source implementation loads full Sugama/Landau test and
field matrices generated offline by COSOlver, interpolates them in
:math:`k_\perp`, and applies the dense moment coupling at runtime. That audit
supports the same separation here: generate cancellation-sensitive
coefficients in high precision, store provenance and checksums, then keep the
JAX runtime to validated table interpolation and matrix application.

Controls:

- ``RuntimePhysicsConfig.collisions``
- ``RuntimeTermsConfig.collisions``
- ``RuntimeSpeciesConfig.nu``
- ``RuntimeCollisionConfig.nu_hermite``
- ``RuntimeCollisionConfig.nu_laguerre``

For two kinetic species, the explicit species-parallel integrator evaluates
this complete collision contribution locally on each device after the shared
field reduction. Nonzero, unequal ion/electron rates are identity-gated against
serial RHS evolution on logical CPUs and two office GPUs. The direct
species-sharded RHS helper remains collision-free; use
``integrate_linear(..., parallel=RuntimeParallelConfig(strategy="velocity",
axis="species", num_devices=2))`` for the validated collisional route.

The guiding-centre invariant gate is explicitly long-wavelength. At
:math:`k_\perp\rho=0`, a five-step gate starts from populated high moments,
requires a nonzero collision response, and preserves each species' density,
parallel-momentum, and temperature-like moments in both serial and decomposed
integration. At finite :math:`k_\perp\rho`, guiding-centre density, momentum,
and energy are not locally conserved; collisions are local in real space, and
the gyrocentre change is nonlocal. This is physical behavior of the published
model, not a residual to tune away. A direct finite-:math:`b` gate checks every
term of equations (3.38)--(3.42), including parallel/perpendicular flow and
temperature restoration. Promotion to Landau/Sugama remains blocked because
those operators contain different velocity-dependent test-particle and
species-coupled field-particle physics, not because this model should be forced
to conserve finite-:math:`b` guiding-centre moments.

The same enclosing route is gated for Hermite/Laguerre hypercollisions with
explicitly populated high moments and nonzero ``nu_hyper_l``/``nu_hyper_m``.
This verifies that the decomposed operator damps its intended high-order
subspace while retaining serial evolution identity; it is separate from the
low-order conserving collision correction above.

Electromagnetic species decomposition reuses the serial field equations rather
than maintaining a second approximation. Local density, parallel-current,
polarization, and perpendicular-pressure moments are summed over the local
species axis and then reduced across the named device axis. The resulting
``phi``, ``apar``, and ``bpar`` are shared by each local RHS assembly. A
two-species gate requires nonzero magnetic fields and serial/decomposed
trajectory identity; it does not yet cover mixed species--Hermite meshes.

Hypercollisions
---------------

GKX implements three Hermite/Laguerre hypercollision branches and an
optional :math:`|k_z|`-scaled branch:

.. math::

   \mathcal{R}_{hyper}^{const}
   =
   -w_{hyper}
   \Big[
      v_{th,s}\big(\tilde{\nu}_\ell r_\ell + \tilde{\nu}_m r_m\big)
      + \nu_{\ell m} r_{\ell m}
   \Big]G,

.. math::

   \mathcal{R}_{hyper}^{iso}
   =
   -w_{hyper}\,\nu_{hyper}\,r_{hyper}\,G,

.. math::

   \mathcal{R}_{hyper}^{|k_z|}
   \propto
   -w_{hyper}\,\nu_{k_z}\,|k_z|\,m^{p_m}\,G.

Controls:

- ``RuntimePhysicsConfig.hypercollisions``
- ``RuntimeTermsConfig.hypercollisions``
- ``RuntimeCollisionConfig.nu_hyper``
- ``RuntimeCollisionConfig.nu_hyper_l``
- ``RuntimeCollisionConfig.nu_hyper_m``
- ``RuntimeCollisionConfig.nu_hyper_lm``
- ``RuntimeCollisionConfig.p_hyper``
- ``RuntimeCollisionConfig.p_hyper_l``
- ``RuntimeCollisionConfig.p_hyper_m``
- ``RuntimeCollisionConfig.p_hyper_lm``
- ``RuntimeCollisionConfig.hypercollisions_const``
- ``RuntimeCollisionConfig.hypercollisions_kz``

Hyperdiffusion And End Damping
------------------------------

The perpendicular hyperdiffusion term is

.. math::

   \mathcal{R}_{k_\perp\text{-hyper}}
   =
   -w_{hyperdiff}\,D_{hyper}
   \left(\frac{k_\perp^2}{k_{\perp,\max}^2}\right)^{p_{hyper,k_\perp}} G,

masked by the dealias region.

The field-line end damping is

.. math::

   \mathcal{R}_{end} = -w_{end}\,A_{end}\,d(z)\,H.

Controls:

- ``RuntimeTermsConfig.hyperdiffusion``
- ``RuntimeCollisionConfig.D_hyper``
- ``RuntimeCollisionConfig.p_hyper_kperp``
- ``RuntimeCollisionConfig.damp_ends_amp``
- ``RuntimeCollisionConfig.damp_ends_widthfrac``
- ``RuntimeCollisionConfig.damp_ends_scale_by_dt``

Nonlinear :math:`E \\times B` And Flutter
-----------------------------------------

The nonlinear bracket is evaluated pseudospectrally:

.. math::

   \{f,g\} = \partial_x f\,\partial_y g - \partial_y f\,\partial_x g.

The electrostatic nonlinear term is

.. math::

   \mathcal{R}_{NL,E\times B} = -w_{nl}\,\{g,\langle \chi \rangle\},

and the electromagnetic flutter contribution couples adjacent Hermite moments:

.. math::

   \mathcal{R}_{NL,flutter}
   =
   -v_{th,s}
   \left(
   \sqrt{m}\,\{\langle A_\parallel \rangle,g\}_{m-1}
   +
   \sqrt{m+1}\,\{\langle A_\parallel \rangle,g\}_{m+1}
   \right).

Controls:

- ``TimeConfig.compressed_real_fft``
- ``TimeConfig.laguerre_nonlinear_mode``
- ``TimeConfig.nonlinear_dealias``
- ``RuntimeTermsConfig.nonlinear``

Source Mapping
--------------

- linear term kernels:
  ``src/gkx/terms/linear_terms.py``
- nonlinear term kernels:
  ``src/gkx/terms/nonlinear.py``
- assembly:
  ``src/gkx/terms/assembly.py``
- low-level parameter container:
  ``src/gkx/linear.py``
- runtime parameter surface:
  ``src/gkx/workflows/runtime/config.py``

Parameter Surface
-----------------

The primary parameter groups are:

- ``RuntimePhysicsConfig``
- ``RuntimeCollisionConfig``
- ``RuntimeNormalizationConfig``
- ``RuntimeTermsConfig``
- ``LinearParams``

For TOML syntax and all supported keys, see :doc:`inputs`.
