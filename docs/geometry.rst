Geometry
========

S-alpha flux-tube model
-----------------------

The analytic s-alpha model (``geometry.model = "s-alpha"``, the default) is
the geometry of the Cyclone base case benchmarks. With the shear function
:math:`\sigma(\theta) = \hat{s}\,\theta - \alpha \sin\theta`, the field-aligned
radial wave number is

.. math::

   k_x(\theta) = k_{x0} - \sigma(\theta)\, k_y,

and the metric coefficients are

.. math::

   g_{ds2} = 1 + \sigma^2,\quad
   g_{ds21} = -\hat{s}\,\sigma,\quad
   g_{ds22} = \hat{s}^2 .

The solver writes the perpendicular wave number with the radial wave number
divided by the shear, :math:`\hat{k}_x = k_{x0}/\hat{s}`:

.. math::

   k_\perp^2(\theta) =
   \left[k_y \left(k_y g_{ds2} + 2 \hat{k}_x g_{ds21} \right)
   + \hat{k}_x^2 g_{ds22}\right] B^{-2}(\theta),
   \qquad
   B(\theta) = \frac{1}{1 + \epsilon \cos\theta},

so the factors of :math:`\hat{s}` cancel at finite shear. The :math:`B^{-2}`
factor is applied when ``kperp2_bmag = True`` (see :doc:`normalization`). At
exactly zero shear the solver uses :math:`\hat{k}_x = k_{x0}` and
:math:`g_{ds22} = 1`, so radial modes keep their finite-Larmor-radius
dependence. The same zero-shear contract applies to ``SlabGeometry``
(``gkx.geometry.analytic``).

The metric coefficients select the zero-shear branch with ``jnp.where`` on
``s_hat``, so they stay traceable: at finite shear, ``jax.grad`` and
``jax.jit(jax.grad(...))`` of the s-alpha and slab metric coefficients with
respect to ``s_hat`` agree with central finite differences
(``tests/unit/geometry/test_geometry.py``). Derivatives across the zero-shear
switch are not claimed.

Parameters
----------

The geometry is specified by:

- ``q``: safety factor
- ``s_hat``: magnetic shear
- ``epsilon``: inverse aspect ratio
- ``R0``: reference major radius
- ``B0``: reference magnetic field
- ``alpha``: pressure-gradient parameter in :math:`\sigma(\theta)`
- ``drift_scale``: drift normalization (``1.0`` is the tracked default; ``2.0`` selects the alternate doubled-drift convention)

Field-aligned grid parameters
-----------------------------

For direct comparison with published Cyclone base case benchmarks,
``GridConfig`` exposes field-aligned grid inputs:

- ``y0`` sets the minimum binormal wave number via :math:`k_y \rho = 1/y_0`.
  Internally this maps to ``Ly = 2\pi y0`` so that the FFT grid spacing matches.
- ``ntheta`` and ``nperiod`` (or ``zp``) control the parallel grid. We set
  :math:`Z_p = 2\,nperiod-1` and choose ``Nz = ntheta * Zp``, which spans
  :math:`[-\pi Z_p, \pi Z_p)`.

Curvature and grad-B drift
--------------------------

The magnetic drift frequency used in the linear operator is

.. math::

   \omega_d(\theta) = k_y \left(\mathcal{C}_v + \mathcal{C}_g\right)
   + \hat{k}_x \left(\mathcal{C}_v^0 + \mathcal{C}_g^0\right),

with the same :math:`\hat{k}_x` as above and

.. math::

   \mathcal{C}_v = \mathcal{C}_g =
   d\,\frac{\cos\theta + \sigma(\theta)\sin\theta}{R_0},
   \qquad
   \mathcal{C}_v^0 = \mathcal{C}_g^0 =
   -d\,\frac{\hat{s}\sin\theta}{R_0},

where :math:`d` is ``drift_scale``.

Slab Model
----------

GKX exposes a slab flux-tube geometry contract directly with
``geometry.model = "slab"``. This is the correct backend for slab secondary
and collisional-ETG benchmark families; it is not an ``s-alpha``
approximation.

The slab overrides are:

- ``bmag = 1`` and ``bgrad = 0``
- ``cvdrift = gbdrift = cvdrift0 = gbdrift0 = 0``
- ``gradpar = 1`` by default, or ``1/z0`` when ``geometry.z0 > 0``
- the metric still uses the supplied ``s_hat`` unless ``geometry.zero_shat = true``
- with ``zero_shat = true``, the slab metric becomes ``gds2 = 1``,
  ``gds21 = 0``, ``gds22 = 1`` and the effective solver shear is zero

Unit tests lock this contract.

Geometry Data Contract
----------------------

The linear cache accepts either:

- the analytic ``SAlphaGeometry`` model, or
- a sampled ``FluxTubeGeometryData`` contract.

``FluxTubeGeometryData`` stores the solver-ready profiles on a specific
``theta`` grid:

- ``bmag`` and ``bgrad``,
- ``gradpar``,
- metric coefficients ``(gds2, gds21, gds22)``,
- curvature / grad-B drift coefficients ``(cv, gb, cv0, gb0)``,
- explicit ``jacobian`` and ``grho`` profiles when the source provides them,
- geometry metadata such as ``q``, ``s_hat``, ``R0``, and the
  ``kperp2_bmag`` / ``bessel_bmag_power`` switches.

``sample_flux_tube_geometry`` converts the analytic s-alpha model into the same
contract, and ``ensure_flux_tube_geometry_data`` normalizes analytic and
sampled inputs onto one solver-facing representation. The contract is a JAX
pytree and is accepted by the linear cache, the runtime initial-condition
builder, the RHS assembly entry points, the nonlinear config runner, and the
reference-compatible volume-weight diagnostics.

Imported field-line geometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``load_imported_geometry_netcdf`` reads grouped NetCDF files with
``Geometry``/``Grids`` groups and root-level ``*.eik.nc`` field-line geometry
files. A runtime deck selects it with ``geometry.model = "imported-netcdf"``
and ``geometry.geometry_file = "external_geometry.nc"``. The aliases
``"imported-eik"``, ``"vmec-eik"``, and ``"desc-eik"`` use the same importer,
so a deck can record the provenance of a ``*.eik.nc`` file without changing the
solver-facing contract. An imported file can be a grouped diagnostic
``*.out.nc`` file or a VMEC/DESC-generated ``*.eik.nc`` file.

The importer infers whether the terminal theta point is present from the
periodic endpoint content of the profiles, so VMEC-style closed grids and
already-open Miller ``*.eiknc.nc`` grids both map onto the solver's open grid.
Tests lock both cases: closed-interval VMEC/DESC files preserve
``theta_scale``/``nfp`` metadata and trim the terminal point, and GX Miller
``*.eiknc.nc`` files keep their open grid with no trim.

Imported geometry is used as sampled. GKX does not rebuild the twist-shift from
analytic formulas; the runtime takes the file's ``theta`` extent, its
``jtwist``/``x0`` defaults for ``linked`` and ``fix aspect`` boundaries, and
its ``kxfac`` metadata, so the flux-tube grid is built on the field-line domain
the file encodes. The linear cache treats ``"fix aspect"`` and
``"continuous drifts"`` as linked twist-and-shift boundaries.

On the W7-X linear ITG case, the imported geometry reproduces the GX ``t=2``
reference on the same sampled field line. The scan in
``docs/_static/w7x_linear_t2_scan.csv`` gives mean relative ``gamma`` errors of
2.3% to 3.4% and mean relative ``omega`` errors of 0.02% to 0.27% over
``ky = 0.1`` to ``0.8``.
``benchmarks/cases/w7x_nonlinear_imported_geometry.toml``
runs the GX nonlinear W7-X adiabatic-electron setup on a ``*.eik.nc`` file.

VMEC runtime bridge
~~~~~~~~~~~~~~~~~~~

With ``geometry.model = "vmec"``, the runtime generates a ``*.eik.nc`` file
from a VMEC WOUT (``gkx.geometry.vmec_eik``) and then re-enters the imported
contract above. When GKX chooses the output path, the file is cached by input
content and VMEC file timestamp. An explicit ``geometry_file`` is treated as an
output target and regenerated, never reused. For ``fix aspect`` cases the
bridge leaves ``x0`` unset, so the flux-tube cut is chosen from ``y0`` and the
geometry. ``geometry.vmec_file`` expands environment variables and ``~``.

The shipped runtime decks point to relative ``wout_*.nc`` paths under
``examples/vmec``. Generate them from the bundled ``vmex`` input decks with
``examples/vmec/generate_wouts.sh`` or a single ``vmex input.<case>`` command;
``--vmec-file`` overrides the path for a machine-specific equilibrium.

When ``booz_xform_jax`` is not installed into the active environment, point
GKX at it through ``BOOZ_XFORM_JAX_PATH`` or ``GKX_BOOZ_XFORM_JAX_PATH``. The
JAX backend is preferred; a ``booz_xform`` install is used only as an automatic
fallback reader. Differentiable VMEC/Boozer transport-gradient audits require a
``booz_xform_jax`` checkout at or after upstream commit ``1d5e8c``. That
revision replaces inactive zero-mode Fourier divisions by safe denominators in
the JAX Boozer transform so reverse-mode cotangents through ``w`` spectrum
reconstruction remain finite. Older checkouts can produce finite values but
non-finite gradients and must not be used for promoted transport-gradient
claims.

The W7-X exact-state audit ``docs/_static/w7x_exact_state_audit.json`` checks
the VMEC runtime path against GX on the same states: startup ``g_state`` and
``phi``, and, on the dumped late nonlinear state, ``phi``, ``kperp2``,
``fluxfac``, ``Wg``, ``Wphi``, and heat flux. Its maximum finite pointwise
relative error is ``4.62e-5`` under a ``1e-4`` gate, and the late scalar
diagnostics agree to better than ``1.8e-7``. The comparison reconstructs GX's
compressed real-FFT positive-``ky`` dump grid from ``diag_state_ky_t*.bin``.

Miller geometry
~~~~~~~~~~~~~~~

With ``geometry.model = "miller"``, the in-package backend constructs the
Miller surface, the straight-field-line and equal-arc grids, and the metric and
drift coefficients, writes a root-level ``*.eiknc.nc`` file, and re-enters the
imported contract. An existing generated target is reused unless
``gkx geometry miller --force`` is requested. The Cyclone Miller rows of
:doc:`verification_matrix` exercise this backend.

Command-line entry points:

- ``gkx geometry vmec --config ...`` generates a compatible ``*.eik.nc`` file
  from a GKX runtime TOML.
- ``gkx geometry miller --config ...`` generates a compatible Miller
  ``*.eiknc.nc`` file from a GKX runtime TOML, or reuses the existing target
  when its path is already populated.
- ``examples/04_nonlinear_stellarator/run.py`` and
  ``examples/04_nonlinear_stellarator/case_full.toml`` run a nonlinear
  adiabatic-electron ITG case on the bundled QHS VMEC input deck after its
  ``wout_NuhrenbergZille_1988_QHS.nc`` file is generated with ``vmex``.
  They accept ``--vmec-file`` for exact HSX validation WOUTs while GKX
  generates and reuses the field-line geometry.

In-memory geometry adapters
---------------------------

``gkx.geometry.differentiable`` is the in-memory boundary.
``discover_differentiable_geometry_backends()`` reports optional ``vmex`` /
``booz_xform_jax`` availability, and ``flux_tube_geometry_from_mapping(...)``
validates an in-memory, solver-ready field-line bundle before it enters
``FluxTubeGeometryData``. The upstream code must supply the sampled ``theta``,
``bmag``, ``gradpar``, metric, drift, Jacobian, and ``grho`` arrays; GKX invents
none of them. Finite-value checks run on host inputs, and JAX-traced arrays
pass through ``flux_tube_geometry_from_mapping(..., validate_finite=False)``,
so geometry observables, inverse-design objectives, and covariance estimates
can be differentiated.

For a solved VMEX state:

.. code-block:: python

   geometry = gkx.geometry.from_vmex(
       state, runtime, surface_index=surface_index, alpha=alpha
   )

The adapter calls only ``vmex.core.turbulence.gk_fieldline_geometry`` and then
the mapping validator above. It contains no VMEC spectral or Boozer
reconstruction and does not import VMEX until called. Use ``from_vmex`` for a
solved state, ``from_vmex_wout`` for a standard WOUT, and
``flux_tube_geometry_from_mapping`` when an upstream code already provides the
complete physical array contract. ``booz_xform_jax`` owns Boozer transforms and
spectra; GKX exposes a bounded spectral derivative check and a field-line
``|B|`` evaluator, and neither invents metric or drift coefficients.

Closed VMEX mirror geometry
---------------------------

Closed VMEX stellarator--mirror hybrids use the parallel adapter:

.. code-block:: python

   geometry = gkx.geometry.from_vmex_mirror(
       mirror_state,
       setup.discretization,
       setup.axis,
       axial_flux_derivative=0.02,
       current_derivative=0.0,
       ntheta=32,
   )

``vmex.mirror.turbulence.gk_closed_fieldline_geometry`` owns field-line
closure, the Clebsch label, Cartesian metric and drift projections,
normalization, and equal-arc remapping. GKX only validates the returned arrays.
The route accepts a line that closes after one periodic racetrack circuit and
sets ``s_hat=0`` so ``kx`` is the direct radial wavenumber. The emitted metric
is complete:

.. math::

   g_{yy}=L_{ref}^2s|\nabla\alpha|^2,\quad
   g_{xy}=\frac{L_{ref}^2}{2}\nabla\alpha\cdot\nabla s,\quad
   g_{xx}=\frac{L_{ref}^2}{4s}|\nabla s|^2,

and the GKX mirror coefficient is

.. math::

   bgrad=L_{ref}\frac{\boldsymbol B\cdot\nabla|B|}{|B|^2}
        =gradpar\,\partial_z\ln|B|.

This is a local-Maxwellian delta-f calculation on a **closed periodic mirror
hybrid**. It is not an open-mirror end-loss model. Joining VMEX's open end cuts
would omit absorbing/sheath boundaries, sources, ambipolar-potential
formation, collisions that replenish the loss cone, and the non-Maxwellian
background ordering those calculations require. The full derivation,
normalizations, validation ladder, and open-lane literature are maintained in
VMEX's ``mirror-gyrokinetics`` explanation page.

.. image:: _static/vmex_mirror_gkx_showcase.webp
   :alt: Closed VMEX mirror field line on a solved equilibrium, its
         magnetic-field profile centred on the field-strength minimum, and the
         GKX perpendicular metric
   :width: 100%

The shipped case: field-strength modulation, not a mirror ratio
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The case is evaluated on a solved fixed-boundary equilibrium. The builder runs
``solve_fixed_boundary(..., solve_lambda=True, require_convergence=True)``, as
VMEX's own ``examples/mirror/stellarator_mirror_hybrid.py`` does before it
plots, and hands the solved state to ``from_vmex_mirror`` and
``gk_closed_fieldline_geometry``. At the shipped resolution it converges in 979
iterations to a normalized strong-form MHD force residual of
:math:`4.84\times10^{-3}`, with ``variational.maximum``
:math:`8.67\times10^{-17}`, ``staggered_weak_force.maximum``
:math:`9.07\times10^{-17}` and ``normalized_divergence_rms``
:math:`1.40\times10^{-14}`, against a seed residual of 0.61. The record carries
all of these, together with ``converged``, ``iterations`` and the resolution
that achieved them, so the state can be checked rather than assumed.
``solve_lambda`` is not optional: with the default the solve reports converged
after four iterations at a residual of 0.55 and leaves the field-strength ratio
at 1.614.

The racetrack axis is planar: two straight legs of ``straight_length`` 4.0
joined by return bends of ``return_radius`` 2.0. Along the sampled field line
``bmag`` carries a maximum at each of the two return bends and a minimum on
each straight leg, so one circuit closes two wells rather than one. The profile
is smooth, with its second harmonic eight times its fundamental.

The recorded ``bmag_max_over_min`` is that modulation depth, 1.03801. It is
not a mirror ratio in the usual sense of a well minimum over a localized
throat maximum, and this page uses :math:`R_m` with that standard meaning in
the loss-cone boundary below, so the closed case does not reuse the symbol.
Nor does it track the rotating elliptical cross-section: the elongation
squared is 1.77778, and the identity between the two (1.77849) held only on the
seeded state. On solved equilibria the ratio moves with ``return_radius``, and
it labels one flux tube rather than the device. The loss-cone formula below
must not be applied to it.

The ``bmag`` and metric panels are drawn with the parallel origin at the
``bmag`` minimum. The equal-arc grid otherwise lands ``z=0`` on a return bend,
which draws a periodic circuit as a central barrier with half a well at each
edge; centering on the field-strength minimum shows the same data as the well
the periodicity implies. The shift is display only: the record uses the
unrolled geometry, and on the periodic parallel domain ``gamma``, ``omega``,
and the mixing-length proxy do not depend on where ``z=0`` falls.

The strong-form residual does not improve with resolution: on this racetrack
``force.normalized_rms`` sits between :math:`4.6\times10^{-3}` and
:math:`5.1\times10^{-3}` across ``ns`` 5--9, ``mpol`` 4--6 and
``coefficient_count`` 16--64, with no downward trend, while the weak-form
measures and :math:`\nabla\cdot\mathbf{B}` reach machine precision. VMEX's
closed lane asserts :math:`1.6\times10^{-4}` for its circular limit, which this
leg-and-bend axis does not reach. Whether that plateau is the curvature
junction or something undriven is asked upstream in `uwplasma/vmex#211
<https://github.com/uwplasma/vmex/issues/211>`__, so the residual is published
as a measurement and the shipped case is gated only against being an unsolved
state.

The :download:`poster snapshot <_static/vmex_mirror_gkx_snapshot.webp>`, the
:download:`animated WebP loop <_static/vmex_mirror_gkx_loop.webp>`, and the
:download:`rotating field-line movie <_static/vmex_mirror_gkx_rotation.mp4>`
show the same one-circuit-closing line. All media and their machine-readable
:download:`run record <_static/vmex_mirror_gkx_showcase.json>`, which stamps the
GKX and VMEX commits it was built with, are regenerated with:

.. code-block:: bash

   PYTHONPATH=src python scripts/artifacts/build_vmex_mirror_gkx_artifacts.py

with VMEX importable at the commit the record names.

The closed-field-line construction follows the straight-field-line mirror
coordinates of Ågren and Savenko and the paraxial rotating-ellipse fixtures in
Rodríguez, Helander, and Goodman. The GKX equal-arc convention follows the
stellarator flux-tube construction of Xanthopoulos *et al.* The separate open
lane is benchmarked against the kinetic open-mirror programs of Francisquez
*et al.* and Rosen *et al.*; these references motivate, rather than waive, the
missing open-boundary physics:

* `Xanthopoulos et al., Phys. Plasmas 16, 082303 (2009)
  <https://doi.org/10.1063/1.3187907>`__;
* `Rodríguez, Helander, and Goodman, J. Plasma Phys. 90, 905900212 (2024)
  <https://doi.org/10.1017/S0022377824000345>`__;
* `Francisquez et al., Phys. Plasmas 30, 103505 (2023)
  <https://doi.org/10.1063/5.0160289>`__;
* `Rosen et al., full-f kinetic mirror equilibria (2026)
  <https://arxiv.org/abs/2604.11684>`__.

Open-ended mirrors are out of scope
-----------------------------------

An open mirror is not a different endpoint option for the periodic flux-tube
solver. Particles cross physical end planes, the loss cone changes the
background distribution, sources replace lost particles, collisions scatter
particles between passing and trapped regions, and the electrostatic potential
adjusts the electron and ion loss rates. These pieces form one kinetic model.
Endpoint damping or a zero-incoming-value rule by itself is only a numerical
outflow experiment.

The minimum conservative one-dimensional model evolves a full distribution
``f_s(z, v_parallel, mu, t)``:

.. math::

   \partial_t(\mathcal J f_s)
   +\partial_z(\mathcal J v_\parallel f_s)
   +\partial_{v_\parallel}(\mathcal J a_{\parallel s}f_s)
   =\mathcal J\left(C_s[f]+S_s\right),

.. math::

   a_{\parallel s}
   =-\frac{\mu}{m_s}\partial_z B
    -\frac{q_s}{m_s}\partial_z\phi,
   \qquad
   \mathcal E_s=\frac{m_s v_\parallel^2}{2}+\mu B+q_s\phi.

Here :math:`\mathcal J` is the gyrocentre phase-space Jacobian. In a static
smooth field, the collisionless interior must preserve magnetic moment
:math:`\mu` and energy :math:`\mathcal E_s` to the scheme's order. With
midplane field :math:`B_0` and throat field :math:`B_m`, the zero-potential
loss-cone boundary is

.. math::

   \sin^2\vartheta_{lc}=\frac{B_0}{B_m}=\frac{1}{R_m}.

An electrostatic barrier shifts this boundary through :math:`\mathcal E_s`; it
must not be represented by changing the geometric mirror ratio. At an end
plane, outgoing characteristics leave the domain and incoming characteristics
need a declared absorbing, logical-sheath, or conducting-sheath rule, with the
reflection cutoff derived from the end potential and field solve. Sources must
state their particle and energy injection, and the collision operator must
state which invariants it preserves and how it repopulates the loss cone.

VMEX owns the open equilibrium, axis/surface geometry, ``B(z)``, metric and
drift arrays, physical end planes, and geometry differentiation. The
conservative phase-space fluxes, velocity-space boundaries, positivity,
sources, collisions, and field/sheath closure belong to an open-kinetic model,
and GKX's local-Maxwellian delta-f core is not one. GX, GS2, stella, and
gyaradax are periodic/twist-linked flux-tube references, and their endpoint
machinery is not evidence for an open model. The primary references are the
full-f conservative open-field-line formulation and sheath treatment of `Shi et
al. (2017) <https://doi.org/10.1017/S002237781700037X>`__, the high-field-mirror
study of `Francisquez et al. (2023) <https://arxiv.org/abs/2305.06372>`__, and
the review of classical end-loss processes by `Baldwin (1977)
<https://doi.org/10.1103/RevModPhys.49.317>`__.

GKX supports closed periodic mirror hybrids only and makes no open-mirror,
Pastukhov, sheath, or confinement-time claim.

VMEX WOUT geometry
------------------

For an existing VMEC-compatible WOUT, use the read-only companion without
reconstructing or re-solving an equilibrium:

.. code-block:: python

   geometry = gkx.geometry.from_vmex_wout(
       "wout_case.nc", surface_index=surface_index, alpha=alpha
   )

This route calls only
``vmex.core.turbulence.gk_fieldline_geometry_from_wout``; VMEX owns file
reading, spectral evaluation, normalization, metrics, and drifts, while GKX
owns only the generic contract and its consumption.

Differentiable geometry bridge validation
-----------------------------------------

The validation record is generated by:

.. code-block:: bash

   JAX_ENABLE_X64=1 PYTHONPATH=src \
     python examples/09_autodiff/geometry_bridge.py

It writes ``docs/_static/differentiable_geometry_bridge.json`` (the companion
panel is regenerated on demand and is not tracked). The JSON records ``vmex``
and ``booz_xform_jax`` API availability, autodiff-vs-finite-difference
sensitivity errors, inverse-design convergence, local UQ covariance
diagnostics, and seven optional real-backend derivative gates:

- a ``vmex`` boundary-aspect check;
- a ``vmex`` metric-tensor check through ``vmex.geom.eval_geom``, with maximum
  absolute and relative AD-vs-finite-difference errors of about ``5.9e-8`` and
  ``1.3e-7``;
- a stellarator VMEC field-line tensor check through ``vmex.geom`` plus
  ``vmex.vmec_bcovar`` on the non-axisymmetric ``nfp4_QH_warm_start`` fixture,
  which checks ``|B|`` ripple and sampled VMEC metric observables before any
  reduced GKX metric/drift closure, at about ``2.1e-3`` absolute and ``2.4e-5``
  relative;
- a direct VMEC tensor-derived flux-tube mapping check, which inverts the
  sampled metric tensor, derives ``gds*``, ``gradpar``, Jacobian, ``grho``, and
  a local grad-:math:`B` drift closure, at a maximum relative error of about
  ``1.1e-4`` on ``nfp4_QH_warm_start``;
- a small ``booz_xform_jax`` Boozer-spectrum check;
- a bounded Boozer-spectrum-to-flux-tube mapping check;
- a real ``vmex`` state to ``booz_xform_jax`` to GKX field-line geometry check,
  at about ``5.8e-7`` absolute and ``1.4e-8`` relative for the tracked geometry
  observables.

The same record carries a VMEC/EIK array-parity audit of the direct tensor
path, which stays ``diagnostic_open``: that path uses a
VMEC-coordinate/equal-theta sampling and a local grad-:math:`B` closure. The
JAX-native ``vmex -> booz_xform_jax`` Boozer equal-arc core does match the
imported convention on ``nfp4_QH_warm_start``: ``bmag``, the solver Jacobian,
``gradpar``, ``q``, and ``s_hat`` to worst normalized/scalar errors ``4.5e-3``
and ``2.4e-3``, the derivative-like ``bgrad`` to ``2.3e-2``, the zero-beta
metric profiles ``gds2``, ``gds21``, ``gds22``, and ``grho`` to ``3.45e-2``, and
the loaded-convention zero-beta drift profiles ``cvdrift``, ``gbdrift``,
``cvdrift0``, and ``gbdrift0`` to ``3.50e-2``.

The reusable API entry point for this workflow is
``geometry_inverse_design_report(mapping_fn, initial_params, target_observables, ...)``:
it runs a bounded Gauss-Newton inverse design on selected solver-ready
geometry observables, checks the final sensitivity Jacobian against central
finite differences, and records local covariance diagnostics.

The gradient-report API accepts ``jacobian_chunk_size``. ``None`` evaluates
all forward directions in one ``vmap``; an integer bounds the simultaneous
directions; and ``"auto"`` delegates the memory policy to SOLVAX. The standard
VMEC/Boozer geometry sensitivity report uses ``"auto"`` and records that
choice in its JSON output. Chunked and unchunked Jacobians are required to
agree before a report can support an optimization claim.

``jacobian_mode`` separately controls derivative direction. ``"forward"`` is
appropriate for few design variables, ``"reverse"`` for few observables, and
``"auto"`` selects between them from the input/output dimensions. A chunk size
budgets simultaneous JVP directions, so ``"auto"`` resolves to forward mode
when one is given and ``"reverse"`` with a chunk size is rejected. Reports
record the resolved mode, and both explicit modes must agree with central
finite differences before either is used in optimization.

The bridge validates more than array shapes. Host-side mappings must contain
finite scalar metadata such as ``q``, ``R0``, ``B0``, and ``theta_scale``,
must provide at least one ``theta`` sample, and must use a positive integer
``nfp``. The finite-difference utilities reject non-positive step sizes, and
the inverse-design covariance block records rank and conditioning. Each
geometry AD/finite-difference gate also records a ``conditioning`` block
alongside the raw Jacobians: finite flags for the AD and finite-difference
Jacobians, singular values, numerical rank, condition number, AD row/column
norms, per-parameter finite-difference step scaling, and the
observable/parameter location of the worst absolute and relative mismatch.
This metadata is separate from the pass tolerance: a derivative can agree with
finite differences and still be a poor optimization direction if the
sensitivity map is nearly rank deficient or the finite-difference step is
poorly scaled to the chosen VMEC coefficient. Quote both the derivative error
and the conditioning metadata before treating a VMEC/Boozer bridge row as
optimization-ready.

Growth-rate transport-gradient audits also need an eigenbranch-locality check.
The public helper
``solver_linear_operator_matrix_from_geometry(geometry, ...)`` materializes the
same GKX linear operator used by
``solver_growth_rate_from_geometry(...)``, and
``dominant_eigenvalue_branch_locality_report`` compares the dominant-growth
finite-difference slope against the slope of the eigenvalue nearest to the base
dominant eigenvalue. If the independently selected max-growth branch switches,
or if the base branch is under-isolated, the report fails closed. (The
per-surface VMEX wrapper that looped this over surfaces, field lines and
``k_y`` samples was a campaign report and left the package in ARCH-A.) A derivative claim requires VMEX's turbulence tangent tests,
this locality check, and an independent finite-difference comparison on the
exact objective. The ``vmex_boundary_chain_*.json`` records in
``docs/_static`` are conditioning evidence only.

The reusable low-level entry point is
``observable_gradient_validation_report(observable_fn, params, ...)``. It
flattens arbitrary geometry or objective observables, compares JAX AD
Jacobians with central finite differences, records absolute and relative error
tables, checks a tangent direction, adds finite flags, and applies an explicit
rank/condition-number gate. Its payload is strict JSON compatible: nonfinite
diagnostic numbers are written as ``null`` while the corresponding finite flag
and failure reason remain explicit. ``geometry_sensitivity_report`` is a thin
``FluxTubeGeometryData`` wrapper around the same helper.

Passing these gates proves local differentiability and conditioning of the
supplied observables. It is not a claim that GKX has run a full stellarator
optimization, which also needs the VMEC/Boozer array parity, solver-objective
gradient, and nonlinear transport gates below.

Multi-Equilibrium Boozer Parity Matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The replayable multi-equilibrium matrix is tracked at
``docs/_static/vmec_boozer_parity_matrix.json``; its ``{png,pdf,csv}``
companions are regenerated on demand by the builder and are not tracked. Each
cell of the figure reports the absolute mismatch for one subgate, coloured by
mismatch over tolerance. The builder enforces ``mboz,nboz >= 21`` before
calling the optional backend path, because the QI drift gate is under-resolved
at lower Boozer mode counts. A ``sample_set_provenance`` block and a
``sample_set_id`` CSV column identify each bounded row by ``case_name``,
``ntheta``, ``mboz``, and ``nboz``, and record that the builder launches no
external VMEC solves.

The matrix covers ``nfp4_QH_warm_start``, ``nfp3_QI_fixed_resolution_final``,
and the finite-beta ``shaped_tokamak_pressure`` example, and at
``mboz=nboz=21`` every row passes its equal-arc gates. The fixed-resolution QI
row passes the loaded-convention drift subgate at ``7.13e-2`` against an
``8e-2`` tolerance. The evaluated QI robustness variants at ``ntheta=8`` and
``ntheta=16`` also pass. Input-only QI seeds without a bundled ``wout``
reference are rejected with an ``artifact_reason`` rather than solved, so the
declared QI seed campaign is artifact-limited. None of this is broad
random-seed nonlinear QI transport validation or QI optimization. The release
guard requires the finite-beta ``shaped_tokamak_pressure`` row, so the claim
cannot regress to zero-beta-only parity evidence.

In-memory differentiable geometry API
-------------------------------------

Differentiable stellarator optimization must stay on the in-memory path:

.. code-block:: python

   from gkx import flux_tube_geometry_from_vmec_boozer_state

   geom = flux_tube_geometry_from_vmec_boozer_state(
       state,
       runtime,
       inp,
       wout,
       surface_index=surface_index,
       alpha=0.0,
       ntheta=32,
       mboz=21,
       nboz=21,
   )

This public wrapper converts a solved ``vmex`` state through
``booz_xform_jax`` and returns the GKX ``FluxTubeGeometryData`` solver
contract. The path is
``SpectralState -> Boozer tables -> booz_xform_jax -> FluxTubeGeometryData``
and does not write or reload ``*.eik.nc`` files. The file-backed VMEC/EIK route
is the runtime import path for ordinary examples, but not the path for
end-to-end differentiable optimization.

The wrapper is an API boundary, not a new physics claim. It inherits the
``mboz,nboz >= 21`` and equal-arc parity requirements of the VMEC/Boozer
gates. The parity-matrix tests reject ``mboz,nboz < 21`` and assert that a
passed equal-arc matrix is still tagged as
``not_full_transport_gradient_claim``. The gradient-holdout tests require the
``mode21_vmec_boozer_state`` source scope, ``mboz,nboz >= 21``, and track the
nonlinear-window estimator objectives as a reduced differentiability gate
rather than a production nonlinear-optimization gate.

The release guard
``docs/_static/vmec_boozer_differentiability_claim_guard.json`` checks those
contents directly. It requires the equal-arc parity matrix, the QH/Li383
mode-21 frequency/quasilinear/nonlinear-window gradient holdouts, explicit
``diagnostic_open`` status for the direct VMEC tensor-vs-imported-EIK
convention gap, a passing finite-beta/pressure equal-arc parity row, a
startup-only label for the nonlinear finite-difference audit, and three
shaped-pressure finite-beta gates:

- the eigenfrequency-gradient gate in
  ``docs/_static/vmec_boozer_shaped_pressure_solver_frequency_gradient_gate.json``;
- the quasilinear-gradient gate in
  ``docs/_static/vmec_boozer_shaped_pressure_quasilinear_gradient_gate.json``;
- the reduced nonlinear-window estimator-gradient gate in
  ``docs/_static/vmec_boozer_shaped_pressure_nonlinear_window_gradient_gate.json``.

A tagged release must fail if these artifacts try to promote a direct
tensor-parity failure or a startup nonlinear-window response into a converged
nonlinear transport-gradient claim. The guard also checks the solver-objective
content of each QH/Li383 gradient row: frequency rows must carry ``gamma`` and
``omega``; quasilinear rows must additionally carry ``kperp_eff2``,
``linear_heat_flux_weight``, and ``mixing_length_heat_flux_proxy``;
nonlinear-window estimator rows must also carry the window mean, coefficient
of variation, and trend metrics. The release thresholds are ``5e-2`` for
frequency rows, ``2e-2`` for quasilinear rows, and ``7.5e-2`` for reduced
nonlinear-window estimator rows. These are AD/finite-difference consistency
gates, not nonlinear turbulence-gradient accuracy claims.

For release claims, the differentiable-geometry lane is closed only for
artifact-passing equal-arc parity rows, reduced QH/Li383
AD/finite-difference objectives, and the shaped-pressure finite-beta
eigenfrequency/quasilinear/reduced nonlinear-window estimator-gradient gates.
The bridge starts at real ``vmex`` state coefficients and reaches GKX solver
observables. It has not validated converged nonlinear turbulence gradients,
broad QI transport behavior, nonlinear heat-flux optimization, or nonlinear
audits of optimized equilibria.

VMEC and Miller runtime examples
--------------------------------

VMEC-driven stellarator runs:

.. code-block:: bash

   cd examples/vmec
   vmex input.nfp3_QI_fixed_resolution_final
   cd ../..
   gkx run \
     --config benchmarks/cases/w7x_nonlinear_vmec_geometry.toml \
     --steps 200 \
     --out tools_out/w7x_vmec.out.nc

   cd examples/vmec
   vmex input.NuhrenbergZille_1988_QHS
   cd ../..
   gkx run \
     --config examples/04_nonlinear_stellarator/case_full.toml \
     --steps 200 \
     --out tools_out/hsx_vmec.out.nc

Miller geometry runs:

.. code-block:: bash

   gkx run \
     --config benchmarks/cases/cyclone_nonlinear_miller.toml \
     --steps 200 \
     --out tools_out/cyclone_miller.out.nc
