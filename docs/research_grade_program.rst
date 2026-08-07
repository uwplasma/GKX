Research-grade program
======================

Status: **plan**. Supersedes the ordering in ``docs/research_grade_plan.rst``
(on the eigensolver branch), which survives for its record of two corrections
worth keeping.

What the literature says the tests must be
------------------------------------------

Gyrokinetic code verification has a settled hierarchy, and GKX's position in it
is more advanced than an inventory of file names suggests.

**Method of manufactured solutions.** Used to verify the spectral Vlasov-Maxwell
implementation in GENE-X: construct a solution, insert it, and measure the
observed convergence order against the design order. GKX has an RK2 manufactured
case for the *time integrator* only; the spectral operator itself has no MMS
verification. This is the largest genuine numerics gap.

**Rosenbluth-Hinton residual.** The standard check that parallel streaming,
radial magnetic drifts and the :math:`k_y = 0` mode are right in the
:math:`k_\perp\rho_i \to 0` limit. **GKX already has this** and the tokamak case
passes against Merlo Case-III: residual 0.192 against 0.19, GAM damping
:math:`-0.176` against :math:`-0.17`, GAM frequency 2.20 against 2.24.

**Dimits shift.** The nonlinear upshift of the turbulence threshold above the
linear one. Absent from GKX, and it is the natural end-to-end nonlinear physics
gate because it tests saturation rather than growth.

**Collision-operator limits.** Already gated: conservation to 2.2e-16, H-theorem,
Onsager to 3.4e-17, published Appendix-C coefficients to 1e-12.

The single most informative open item
--------------------------------------

The W7-X zonal lane fails where the tokamak one passes. Residuals miss at
:math:`k_x\rho_i = 0.07, 0.10, 0.30` and the late envelopes are much larger than
digitized stella/GENE traces. The tracked hypothesis is **velocity-space
recurrence and moment closure**, and a closure ladder over constant-Hermite,
:math:`k_z`-weighted Hermite, mixed Laguerre-Hermite, Laguerre-only and isotropic
hypercollision families found **no family that improves trace error, late-envelope
recurrence and moment-tail metrics simultaneously**: the best was isotropic
:math:`\nu_{\rm hyper}=0.01` at mean trace error 0.2755 against baseline 0.2861,
but with a worse late-window standard-deviation ratio, 4.25 against 4.10.

This is not a side quest. It is the concrete, already-instrumented instance of the
recurrence problem, on the one benchmark where GKX visibly disagrees with two
established codes. Closing it would be the strongest single piece of evidence the
code could produce; failing to close it, with the reason stated, is the honest
alternative.

Ordered work
------------

Each item names its test class, because "add tests" is not a plan.

**A1 -- MMS for the spectral operator.** *Numerics.* Manufacture a solution of the
linear gyrokinetic operator in the Hermite-Laguerre basis, insert it as a source,
and measure the observed order in :math:`N_z`, :math:`N_l`, :math:`N_m`. Gate:
observed order within 10% of design order on at least three refinements.
Rationale: this is the only test that can distinguish a correct operator from one
that is merely self-consistent, and GKX currently verifies only its time
integrator this way.

**A2 -- Recurrence as a reported quantity.** *Physics + regression.* Free
streaming in a truncated Hermite basis recurs at
:math:`t_{\rm rec}\approx 2\sqrt{M}/(k_\parallel v_{\rm th})`. Emit
:math:`t_{\rm rec}` alongside :math:`\tau_{\rm ac}` and :math:`n_{\rm eff}` in
every window artifact, and gate that the analysed window is shorter than
:math:`t_{\rm rec}` unless a closure is declared. Gate: no promoted window is
longer than its own recurrence time without an explicit closure record.

**A3 -- W7-X zonal closure.** *Physics.* With A2 in place the closure ladder can
be re-read against a quantitative recurrence bound rather than an envelope
impression. Gate: either the digitized residuals are met at all three
:math:`k_x\rho_i`, or the artifact records which physical ingredient is missing
with the evidence that rules the others out.

**A4 -- Dimits shift.** *Physics.* Scan :math:`R/L_T` through the linear
threshold and locate the nonlinear onset. Gate: a nonlinear threshold measurably
above the linear one, with the offset reported in units of the correlation time
so the window statistics from P5 apply. This is also the first end-to-end use of
the saturated-state machinery.

**A5 -- Nonlinear autodiff.** *Algorithmic.* As previously scoped: production
heat flux inside the differentiated window, find the divergence knee, measure the
bias against finite differences, then choose windowed adjoint or NILSS on
evidence. Gate: a descent that lowers the flux, with bias **and** variance
reported.

**A6 -- Naming.** *Unit.* ``nonlinear_heat_flux_proxy`` returns a closure over
linear quantities. Rename so no public name asserts nonlinear gyrokinetics for a
linear-solve quantity. Gate: a test asserts the public surface contains no such
name.

**A7 -- Performance, measured not adopted.** *Benchmark.* GKX already uses
``jax.jit`` (12 files), ``lax.scan`` (13), ``shard_map`` (12), ``NamedSharding``
(7), ``jax.vmap`` (6), ``jax.checkpoint`` (5) and ``donate_argnums`` (4). The
primitives are in place, so this is not an adoption exercise. Measure peak memory
and runtime per production case, then test the three untried candidates --
``p_multigrid`` over the :math:`(N_l,N_m)` hierarchy, mixed precision with
iterative refinement, and ``chunked_jacrev`` -- each against a control that could
fail. Gate: nothing adopted without a measurement on the production operator.

Test taxonomy
-------------

Every item above declares which class it belongs to, and each class has a
different failure mode it must be able to catch:

.. list-table::
   :header-rows: 1

   * - class
     - catches
     - example in this program
   * - physics
     - right equations, wrong or missing terms
     - Rosenbluth-Hinton, Dimits, collision limits
   * - numerics
     - right equations, wrong discretisation
     - MMS observed order, conservation, recurrence bound
   * - algorithmic
     - right discretisation, wrong solver or gradient
     - AD against finite differences, eigensolver dense parity
   * - unit
     - right component, wrong wiring
     - naming, dtype control, gate composition
   * - regression
     - right yesterday, wrong today
     - tracked artifact gates, claim-scope pins

The rule this program adopts, from three failures this cycle: **a test that
cannot disagree with the implementation validates nothing.** The eigensolver
suite used normal matrices where the operator is non-normal; the recycling
benchmark ran unpreconditioned while claiming otherwise; the window statistics
agreed with their own estimator until validated against the empirical scatter of
independent realizations, which immediately found a 22% error at zero
correlation. Each new gate must state what result would falsify it.

Sequencing
----------

A6 is an afternoon and unblocks nothing; do it first to clear the naming debt.
A1 and A2 are independent and both feed A3. A4 needs A2 for its window
statistics. A5 is the long pole and should be its own PR series. A7 is
measurement and can run alongside anything.
