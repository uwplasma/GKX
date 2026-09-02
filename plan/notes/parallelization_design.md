# GKX Nonlinear Multi-Device Parallelization Design (Plan item 4.1)

Date: 2026-08-18. Author: design agent for feat/bounded-memory-nonlinear-adjoint.
Repo state read: /Users/rogeriojorge/local/GKX @ feat/bounded-memory-nonlinear-adjoint (read-only).
GX intel: office source dive 2026-08-18, GX @ 3865a537.

## 0. Decision

**Adopt a GX-parity velocity-space decomposition for production nonlinear runs:
a 2-D `(species, hermite)` device mesh under `jax.shard_map`, species factored
first, Hermite second, with the perpendicular plane `(ky, kx)`, the Laguerre
axis, and `z` fully replicated per device.** All bracket and parallel-streaming
FFTs stay device-local; communication reduces to (i) a field-solve `psum`,
(ii) a width-2 Hermite boundary `ppermute` (only when the Hermite axis is
actually split), (iii) collision-moment `psum`s, and (iv) scalar diagnostic
`psum`s accumulated inside the integration scan carry. Candidates ky-shard,
kx-shard, and full-operator z-shard are rejected below with the measured and
structural reasons.

This is the same decomposition GX ships in production (1 MPI rank = 1 GPU,
species first, then Hermite; spatial and Laguerre replicated;
`src/grids.cu:47-83`), realized with JAX collectives instead of NCCL calls.
It is also the only candidate for which GKX already has passing communication
identity gates on every required primitive.

---

## 1. Evidence base

### 1.1 GX (office dive, verified)
- 1 rank = 1 GPU (`cudaSetDevice(iproc%nGPUs)`, `src/main.cu:23`).
- Decomposed axes: SPECIES first, then HERMITE only; local `Nm = Nm/nprocs_m`
  must divide evenly; x, y, z, Laguerre always replicated (`src/grids.cu:47-83`).
- NCCL communicators: global, per-species across m-blocks, across species, and
  an m0-ranks communicator for the field solve (`src/grids.cu:157-184`).
- Hermite ghost-cell halo exchange for the m±1, m±2 neighbors used by parallel
  streaming (`src/moments.cu:~350-440`).
- Field solve sums gyro-densities across species with `ncclAllReduce` on the
  m0 communicator (`src/solver.cu:130`).

### 1.2 GKX measured landscape (docs/parallelization.rst + tracked artifacts)
| Lane | Result | Verdict |
|---|---|---|
| Independent ky / ensembles | 7.18x/8 CPU, 1.88x/2 GPU | production (scan orchestration only) |
| Whole-state pjit (ky) | 1.39x CPU, **0.59x** 2-GPU; CPU multi-device FFT-layout aborts | dead end |
| Hermite shard_map, no halo strategy | **0.03x** | pathological; full-axis collectives |
| Species-axis pmap, 2 GPU | 1.16x warm RHS (68 MiB state) | workload crossover shown |
| Mixed species×Hermite 2×2 logical CPU | **3.11x warm RHS**, exact 100-step identity, but 0.97x end-to-end | RHS decomposition works; observable path killed end-to-end |
| Device-z fused pencil bracket (PR #45) | route overhead 0.988–1.005 (parity), 2.00x parallel scaling, EXACT identity — but reduced operator only | micro-route; not the production RHS |
| Scalar diagnostics via separate gate path | **118x** compute (65x large; `sharded_reduce` 154x) | diagnostics must be fused into the scan, never recomputed |

Two transferable lessons drive this design:
1. **Route-overhead parity is the whole game.** The device-z lane missed its
   gate not on communication (efficiency 0.996–1.006, zero collectives in HLO)
   but on single-device route overhead (1.37–1.59). It reached exact identity
   and parity only when the shard-local code became *the same fused kernel*
   the serial route runs. The species×Hermite design below keeps every
   shard-local kernel byte-identical to the serial production kernels.
2. **Diagnostics dominate unless fused.** The 118x observable overhead came
   from recomputing the bracket outside the timed route. The production
   integrator already accumulates scalars in the scan carry
   (`src/gkx/solvers_nonlinear_state_integration.py:308-347`); the sharded
   route must keep that structure and add only scalar `psum`s.

---

## 2. Communication anatomy of the FULL nonlinear operator (from code)

State: `(Ns, Nl, Nm, Nky, Nkx, Nz)` complex (species axis optional at Ns=1).

| Term | Coupling touched | Source |
|---|---|---|
| ExB/EM bracket (pseudo-spectral) | `fft2`/`irfft2` over **(ky,kx)** only (`axes=(-3,-2)`/`(-2,-3)`); z, m, s untouched; per-(s,l or μ,m,z) plane | `operators/nonlinear/brackets.py:12-17,160,175` |
| Bracket, compressed real-FFT | Hermitian completion gathers conjugate **ky rows** (`_complete_hermitian_ky`) | `brackets.py:95-115` |
| Laguerre "grid" mode (production default `laguerre_mode="grid"`) | **dense Nl×Nl transform over the full Laguerre axis** both ways | `terms/nonlinear.py:411,424,450` |
| A∥ flutter | **m±1** shift of the apar bracket | `terms/nonlinear.py:331-362` |
| Parallel streaming | ladder **m±1** on H; field drives at **m=0,1,2**; z-derivative = **spectral FFT along z** (periodic) or **linked-chain FFT over nLinks×Nz** (twist-shift) | `operators/linear/streaming.py:563-574,43-58,190-216`; `terms/linear_terms.py:144-206` |
| Reflectionless Hermite closure | \|k_z\| **spectral along z** on the **last global m** only | `terms/linear_terms.py:290-342`; `streaming.py:371-387` |
| Mirror | **m±1 combined with l±1** | `terms/linear_terms.py:345-368` |
| Curvature / grad-B | **m±2** (v∥²) and **l±1** (μ) | `terms/linear_terms.py:371-401` |
| Diamagnetic drive | drives at global **m=0,2** (φ, B∥) and **m=1,3** (A∥); **l±1** gyroaverage neighbors | `terms/linear_terms.py:404-611` |
| Quasineutrality (φ) | **m=0 moment**, sum over **l**, sum over **species** | `operators/linear/moments.py:108-136`; `terms/fields.py:90-116` |
| A∥ solve | **m=1 moment**, sum over l and species | `terms/fields.py:243-258` |
| B∥/φ coupled solve | m=0 moments with JlB, sums over l and species | `terms/fields.py:153-220` |
| G→H map | φ,B∥ into **m=0**; A∥ into **m=1** (fields must reach every m-shard) | `operators/linear/moments.py:139-192` |
| Conserving collisions (LB/Dougherty) | **m=0,1,2** moments per species; l sums | `operators/linear/dissipation.py`, cross-moments in `operators/nonlinear/collisions.py` |
| \|k_z\| hypercollision | spectral along z on high m (global m index needed) | `operators/linear/dissipation.py` |
| Perp hyperdiffusion, end damping | pointwise / local in z | `operators/linear/dissipation.py` |
| Linked (twist-shift) boundary | chains live at **fixed ky**, span **multiple kx**, FFT over the **whole extended z chain** | `streaming.py:61-77,151-216`; `operators/linear/linked.py` |

**Hermite coupling bandwidth is exactly GX's: m±2 (curvature) is the maximum;
m±1 for streaming/mirror/flutter.** Laguerre coupling is l±1 for the linear
terms but the Laguerre-grid nonlinear path contracts the **full** l axis, so
Laguerre sharding would demand an allgather of the entire state per bracket —
this single fact removes l from the candidate mesh, independent of the l±1
halos. Species couple **only** through the field solves and (optionally)
cross-species collision moments — precisely the GX communicator structure.

Already in-repo and gated for this pattern (identity-passing):
- species-shard field solve with named `psum`: `terms/fields.py:44-49,328-390`
  (`_species_sum`, `solve_fields_species_shard`, `solve_electrostatic_phi_species_shard`);
- width-1/width-2 Hermite ghost exchange via `lax.ppermute` inside
  `jax.shard_map`: `parallel/velocity_hermite.py:51-201`;
- Hermite-sharded field reduction: `parallel/velocity_hermite.py:218-282`;
- a fused species×Hermite electrostatic linear RHS with global-Hermite-index
  bookkeeping (diamagnetic placement, hypercollision normalization, closure
  coefficient at the true last m): `solvers/linear/parallel_electrostatic.py`;
- species/Hermite planner, species-first factoring: `parallel/velocity_plan.py:87-176`;
- fail-closed nonlinear runtime routing surface:
  `workflows/runtime/parallel_nonlinear.py`.

---

## 3. Candidate analysis

### (a)/(d) Species×Hermite, perpendicular replicated — GX's choice. **SELECTED**
(a) and (d) are one family: (d) at mesh `(Ns, 1)`, (a) at mesh `(ns, nm)`.

- **Zero communication in the FFTs.** Both bracket FFTs (ky,kx) and both z
  FFT paths (periodic and linked-chain) act on axes that are never sharded.
  The linked twist-shift chains run unmodified per shard — already proven in
  the mixed-lane linked gate.
- **Bounded, known collectives** (inventory in §4.3): one field `psum`, one
  width-2 halo, collision-moment `psum`s, scalar `psum`s.
- **Route-overhead parity is structural**: the shard-local kernels are the
  serial kernels on a slab; no axis-staging, no extra transposes (the failure
  mode that cost the device-z lane 1.37–1.59x does not exist here).
- **Measured support**: 3.11x warm RHS on a 2×2 logical-CPU mesh with exact
  100-step identity for the linear slices; species pmap route already
  integrates end-to-end on 2 A4000s with reverse-mode AD gated to 1% FD.
- **Memory scaling**: per-device state shrinks by the mesh size (§5) —
  the only candidate that unlocks grids with no single-GPU baseline
  ((4,16,192,192,64) already OOMs one A4000).
- Cost: when the Hermite axis is actually split, the width-2 halo moves
  4/Nm_loc of the shard per RHS (§4.3). With species-first factoring the
  office 2-GPU box gets a **pure species mesh with zero halo**.

### (b) ky-shard (and kx-shard). **REJECTED for production**
The bracket consumes full (x,y) planes: `irfft2`/`fft2` over axes (ky,kx).
Sharding either perpendicular axis partitions a *transform* axis:
- **No perpendicular axis avoids the all-to-all.** With ky sharded, the kx
  transform is local but the y-transform needs a distributed transpose
  (all-to-all) — and vice versa. Per bracket call the transform sets touched
  are: 2-stacked grad(G), 2-stacked grad(χ) inverse transforms + 1 forward
  transform of the product, i.e. ≈5 plane-set transforms × Nl×Nm×Ns×Nz
  batch; EM runs multiply χ by up to 3 stacked fields. Estimated all-to-all
  volume ≈ 8–10× the shard size *per RHS* (≥8 GB/RHS at the §5 grid) versus
  ≈100 ms of compute — PCIe-hopeless, NVLink-marginal.
- The compressed real-FFT layout adds a second coupling: Hermitian completion
  `_complete_hermitian_ky` gathers conjugate ky rows (with a kx flip) across
  the ky axis — a cross-shard gather per bracket, and the same conjugate
  restoration appears in the linked-FFT output path (`streaming.py:100-133`).
- kx-sharding additionally breaks the twist-shift linked chains, which span
  multiple kx per ky (`_shift_kx_linked`, chain index maps).
- Measured: whole-state ky pjit 0.59x on 2 GPUs; logical spectral-domain route
  comm/work 6.375, efficiency ceiling 0.136. The routed `axis="ky"` lane stays
  what it is today: a fail-closed identity/routing diagnostic
  (`parallel_nonlinear.py`), not a speedup path.

### (c) z-shard fused bracket extended to the full operator. **REJECTED honestly**
The z-pencil micro-route is real (exact identity, 2.0x scaling, overhead
parity after fusing) — for an operator with **no z-derivative**. The full RHS
has four z-nonlocal operators:
1. production streaming derivative = spectral FFT along z (`grad_z_periodic`) —
   GKX's own routing docs already record that it "does not survive SPMD
   partitioning" (`parallel_nonlinear.py:14-23`);
2. twist-shift linked chains FFT over the *extended* chain axis (nLinks×Nz) —
   strictly more nonlocal than one z period;
3. reflectionless closure \|k_z\| on the last Hermite moment
   (`linear_terms.py:290-342`);
4. \|k_z\| hypercollisions.

Could a finite-difference z-stencil + halo replace (1)? The repo even ships a
2nd-order centered FD variant for twist-shift (`_grad_z_linked_fd`,
`streaming.py:79-97`). Honest evaluation: **no, not for production.**
- It changes the operator: O((k_z Δz)²) dispersion error versus spectral
  accuracy; Hermite-ladder recurrence physics (Landau damping, the closure's
  outgoing-wave condition) is sensitive to k_∥ fidelity, and every identity
  gate against the serial spectral answer fails *by construction* — the
  design's fail-closed contract (`strict_identity`) would have to be replaced
  by a physics re-acceptance campaign (Cyclone/KBM/stellarator benchmarks).
- (3) and (4) have **no finite-halo representation at any width**: \|k_z\| is a
  dense operator in z; an FD/recursive-filter surrogate is again a different
  operator.
- Even granting all that, the measured granularity study showed the two-GPU
  net for the *reduced* bracket saturating at ~1.47x before the fused-parity
  fix, and the full operator would add per-RHS z collectives the reduced one
  never paid.
Keep `device_z` exactly as scoped today: a CPU-microkernel diagnostic and the
bracket-fusion reference implementation.

### Candidate summary

| Candidate | FFT comm | Halo | Field solve | Identity to serial | Verdict |
|---|---|---|---|---|---|
| species×Hermite (GX) | none | m±2 `ppermute` (only if m split) | psum(s[,m]) | exact achievable (measured) | **production design** |
| ky-shard | all-to-all per bracket + Hermitian-row gather | — | local | tolerance-level | routing diagnostic only |
| kx-shard | all-to-all + broken linked chains | — | local | — | rejected |
| z-shard full RHS | z-FFT allgather or FD operator swap | m local | local | impossible (operator changes) or expensive | micro-diagnostic only |

---

## 4. Selected design: species-first × Hermite-second `shard_map`

### 4.1 Mesh and factoring
- Mesh: `jax.make_mesh((ns_chunks, nm_chunks), ("s", "m"))`.
- Factoring rule (reuse `parallel/velocity_plan.py::_axis_chunks`, which is
  already species-first): `ns_chunks = largest factor of num_devices that
  divides Ns`; `nm_chunks = num_devices // ns_chunks`, and **require
  `Nm % nm_chunks == 0` exactly** (GX parity: local Nm must divide evenly —
  no ceil-padding; fail closed with a routing error naming the divisible
  device counts).
- Consequences: office 2×A4000 + Ns=2 → mesh (2,1), **no halo, one field
  psum** — the minimal-communication production configuration. Ns=1
  (adiabatic electrons) on 2 devices → mesh (1,2), halo lane active. 4+
  devices → (2,2), (2,4), ...
- Nm_loc ≥ 2 required when nm_chunks > 1 (width-2 halo must come from the
  nearest neighbor only; matches GX's even-division constraint in practice).

### 4.2 Sharding specs
State `G (Ns, Nl, Nm, Nky, Nkx, Nz)`: `P("s", None, "m", None, None, None)`.

Cache/params placement (from `operators/linear/cache_arrays.py` shapes):
- species-indexed arrays — `Jl`, `JlB` `(Ns, Nl, ky, kx, z)`, `vth`, `tz`,
  `charge`, `density`, `tprim`, `fprim`, per-species collision rates:
  `P("s", ...)` on the species-carrying axis; replicated over "m".
- Hermite-indexed arrays — ladder coefficients `sqrt_m`, `sqrt_m_p1`/`sqrt_p`,
  hypercollision profiles, closure mask: shard on "m" **but generate them from
  the global m index** (`global_m = m_block_index * Nm_loc + local_m`), the
  bookkeeping the mixed lane already implements (diamagnetic drive placement
  at global m∈{0,1,2,3}, closure only on the true last moment, physical
  hypercollision normalization).
- perpendicular/geometry arrays — `kx_grid`, `ky_grid`, `dealias_mask`,
  `b`, `cv_d`, `gb_d`, `bgrad`, linked-chain index maps, Laguerre transform
  matrices: fully replicated.
- Fields `phi/apar/bpar (ky, kx, z)`: replicated (out_spec `P()` from the
  field-solve psum; every shard holds identical copies — GX broadcasts from
  the m0 communicator, we get the same effect from a full-mesh psum of
  masked partial moments, see below).

### 4.3 Collective inventory (per RHS evaluation; sizes at the §5 production grid, complex64)

| # | Collective | What | Axes | Size per RHS | When |
|---|---|---|---|---|---|
| C1 | `psum` | field-solve moments: density (m=0), parallel current (m=1), perp-pressure/B∥ moments — each shard contributes its locally-owned m rows (zeros otherwise), summed over l locally first | ("s","m") | 3–5 × field size = 3–5 × 8.4 MB ≈ 25–42 MB | every RHS |
| C2 | `ppermute` ×2 | width-2 Hermite halo of **H** (after `build_H`): last-2 planes up, first-2 planes down; covers streaming ladder m±1, mirror m±1, curvature m±2 in one exchange | ("m") | 2 dirs × 2 planes × s_loc·Nl·Nky·Nkx·Nz = 2×134 MB at mesh (2,2) | only if nm_chunks>1 |
| C3 | `ppermute` ×2 | width-1 halo of the **apar bracket** for flutter (bracket is computed per-m locally; flutter shifts it m±1) | ("m") | 2 × 67 MB | EM runs with nm_chunks>1 only |
| C4 | `psum` | conserving-collision moments m=0,1,2 (and cross-species Dougherty moments if enabled) | ("m") [+("s") for cross-species] | 3 × 8.4 MB per species row | collisions on |
| C5 | `psum` | scalar diagnostics (Wg, Wphi, heat flux, particle flux) accumulated in the scan carry | ("s","m") | O(bytes) | every step |

Notes:
- C1 replaces GX's {m0-communicator allreduce + broadcast} with a single
  masked full-mesh psum; algebraically identical, one collective instead of
  two, and every shard receives the fields it needs for `build_H` (φ,B∥→m=0
  row, A∥→m=1 row live only on the m-block that owns those global m's, but H
  is built locally from the replicated fields).
- C2 is the GX `src/moments.cu` ghost exchange, expressed as
  `lax.ppermute` with the edge shards masked (non-periodic: `shift_axis`
  zero-fills beyond the global Hermite boundary) — exactly the
  `hermite_shift_shard_map` kernel already gated in
  `parallel/velocity_hermite.py:125-201`, widened to depth 2.
- Halo fraction = 4/Nm_loc of the shard per RHS (50% at Nm=16, nm_chunks=2).
  On PCIe A4000s this is why species-first ordering matters; on NVLink boxes
  it amortizes (GX's production regime). Overlap: issue C2 first, compute the
  bracket (halo-free) while boundaries are in flight — natural in shard_map
  because the bracket does not consume halo data.
- The z-axis remains whole on every device, so `grad_z_periodic`,
  `abs_z_periodic`, linked-chain FFTs, \|k_z\| hypercollisions and end damping
  run the *serial production code paths unchanged*.
- The Laguerre-grid nonlinear transform (dense Nl×Nl) is local by
  construction (l replicated).

### 4.4 Field solve sharding
Use `solve_fields_species_shard` / `_species_sum` (`terms/fields.py:44-49,
328-359`) generalized to reduce over both mesh axes: each shard computes
`sum_l(Jl * G[s_loc, :, m_loc==global 0])` masked to the m rows it owns, then
one `lax.psum` over ("s","m") yields `nbar`; denominators are precomputed
(replicated). Same pattern for A∥ (m=1 current moment) and the coupled φ/B∥
2×2 solve (three moment fields). Adiabatic/field-line-average closures
(`_adiabatic_quasineutrality`) reduce over z, which is local. The custom_vjp
around `solve_fields` (`fields.py:393+`) transposes cleanly: psum ↔ identity
broadcast under shard_map AD.

### 4.5 Diagnostics fusion (the 118x lesson)
- The whole time loop stays inside one jitted `lax.scan` on the mesh
  (pattern of `state_integration.py:297-347` / `_integrate_nonlinear_sheared_scan`).
- Per-step scalars (Wg, Wphi, heat flux, particle flux) are computed from the
  `fields` and state already materialized by the RHS — never by re-running
  the bracket — reduced locally over the shard, then one scalar psum, and
  accumulated into the scan carry / stacked into scan ys on device. One host
  transfer at the end of the window.
- Reduction-order discipline: use the compensated-sum utilities
  (`device_z.py:152-181`, `_compensated_observable_sum`) or match the serial
  blocked-reduction structure, because XLA fuses `jnp.sum` differently inside
  a shard_map jit (measured §7: this — not the collectives — is the only
  source of non-bitwise residuals).
- Budget gate: streamed diagnostics ≤5% of step time (vs 118x today on the
  profiler's unfused path).

### 4.6 Adjoint compatibility (branch context: bounded-memory nonlinear adjoint)
Every collective in §4.3 is linear: `psum` transposes to broadcast/identity,
`ppermute` transposes to the reverse permutation. `jax.shard_map` with
`check_vma=True` differentiates these without defensive psums (current JAX
guidance), and the sanity check (§7) confirms `jax.grad` through the
halo+psum kernel is **bitwise-identical** to the serial gradient on jax
0.9.2. The checkpointed-scan adjoint (`checkpointed_explicit_scan`) composes
outside the shard_map unchanged. One known pin: the office JAX 0.6.2 note
about collision VMA annotations failing under standalone shard_map — retest
on 0.9.2/0.10.2 in the 4.2 trial (expected fixed; the VMA machinery is the
piece that matured).

### 4.7 Identity-gating plan (fail-closed, tiered)
- **Tier 0 — kernel gates (exact):** each shard-local term vs the serial term
  on the reconstructed state: streaming, mirror, curvature/grad-B,
  diamagnetic (global-m placement), flutter, closure-at-last-m,
  hypercollisions, collisions, field solve, bracket. Target: bitwise 0.0 in
  complex64 (achieved by the mixed lane and device-z fused route; achieved by
  §7 for halo/psum). Where a reduction is fused differently by XLA, fix the
  route (compensated sum / matching blocked structure) rather than widening
  the tolerance.
- **Tier 1 — trajectory gates:** fixed-step serial vs sharded integration
  (Euler, RK2; then IMEX): final state, final fields, final RHS, per-step
  Wg/Wphi/heat/particle traces. Gate: exact in complex64 where Tier 0 is
  exact; ≤1e-12 relative under JAX_ENABLE_X64 otherwise.
- **Tier 2 — runtime gate:** keep `strict_identity=true` semantics and the
  existing tolerance convention (atol 5e-6, rtol 1e-4,
  `parallel_nonlinear.py:52-53`) as the fail-closed production check; the
  sharded result is discarded on violation
  (`NonlinearParallelIdentityError`), never silently returned.
- **Tier 3 — physics gates** (promotion prerequisites, unchanged from
  docs/parallelization.rst:652-667): boundary/halo-cell identity (not just
  norms), conservation traces, post-transient transport windows for Cyclone,
  KBM, one stellarator smoke case; CPU-serial/CPU-sharded/1-GPU/2-GPU parity.
- **Adjoint gate:** grad of a windowed heat-flux objective w.r.t. R/LT
  serial-vs-sharded (extend the existing 1%-FD species-pmap derivative gate
  to the shard_map route; target: identical to serial gradient at Tier-1
  tolerance).

### 4.8 Runtime routing and auto-mesh UX
Extend `workflows/runtime/parallel_nonlinear.py` (keep its fail-closed shape):

```toml
[parallel]
auto = true            # detect devices, build the (s, m) mesh, strict identity on
```
resolves to
```toml
[parallel]
strategy = "shard_map"          # unchanged strategy vocabulary
axis = "species_hermite"        # new accepted axis (aliases: "velocity", "s_m")
num_devices = <len(jax.devices())>
strict_identity = true
```
- `SUPPORTED_NONLINEAR_AXES` grows to `("ky", "species_hermite", "species", "m")`
  — `"species"`/`"m"` force a 1-D mesh, `"species_hermite"` factors
  species-first via `build_velocity_sharding_plan`; `"ky"` keeps its current
  diagnostic routing; `"z"` keeps its two-reason rejection verbatim.
- Divisibility failures raise `NonlinearParallelRoutingError` listing the
  device counts that divide `(Ns, Nm)` (e.g. "Ns=2, Nm=16 supports 1, 2, 4,
  8, 16, 32 devices as (s×m) = (1×n)|(2×n)").
- `RuntimeParallelConfig` gains `auto: bool = False`; `auto=true` with any
  explicit strategy/axis conflict is an error, not a silent override.
- The resolved plan (mesh shape, chunks, halo depth, collective list — i.e. a
  `VelocityShardingPlan` dict) is recorded in the run artifacts, matching the
  scan-orchestration precedent.
- Host-stage the initial state and species caches once before `device_put`
  with the mesh sharding (the office resharding defect and the device-z gate
  both showed staging-from-host is the reliable path), then enter the jitted
  scan.

### 4.9 JAX version strategy (0.9.2 now, 0.10.x next)
- `jax.shard_map` is the stable public API on 0.9.2 (no experimental import);
  0.9.1 already made explicit-mode shard_map assert input PartitionSpec
  matching instead of silently resharding — treat any reshard-on-entry as a
  bug, which suits the fail-closed design.
- **jax 0.10.0 removed the C++ pmap infrastructure and `PmapSharding`.** The
  measured species route on the office stack is an enclosing `pmap`
  (`docs/parallelization.rst:713-733`); this design deliberately re-bases it
  on `shard_map` so the production lane is not standing on pmap when the pin
  moves. Nothing else in 0.10.0-0.10.2 (LAPACK batch-parallel CPU, scipy
  constructors) touches this problem; sanity-run the trial suite once in
  `~/.venvs/gkx-jax-latest` (0.10.2) to catch the `"cpu:0"` device-name
  change in any artifact bookkeeping.
- `check_vma=True` (default) is required — it is what makes reverse-mode AD
  of the collectives efficient and is the surface where the 0.6.2 collision
  VMA failure must be retested.
- Explicit sharding / sharding-in-types (`AxisType.Explicit`) is noted as the
  emerging middle ground, but this design stays on manual shard_map: the
  collectives are few, named, and audited, and identity gating wants exactly
  that determinism.

---

## 5. Memory budget at a production stellarator grid

Grid `(Ns, Nl, Nm, Nky, Nkx, Nz) = (2, 8, 16, 128, 128, 64)`;
elements = 268,435,456.

| Quantity | complex64 | complex128 |
|---|---|---|
| Full state | 2.147 GB | 4.295 GB |
| Shard, mesh (2,1) — office 2×A4000 | 1.074 GB | 2.147 GB |
| Shard, mesh (2,2) | 0.537 GB | 1.074 GB |
| Field (per field, replicated) | 8.4 MB | 16.8 MB |
| Jl/JlB cache (f32/f64, s-sharded on (2,·)) | 33.6 MB each | 67 MB each |

Working-set calibration from tracked measurements: the fused serial bracket
allocates ≈2.0× state of scratch (307 MB at a 151 MB state); the serial
route's largest single buffer observed is ≈4× state (the 4.83 GB failed
allocation at a 1.208 GB state that OOMed one 16 GB A4000). Budgeting peak ≈
4–6× resident state for RK2 + fused bracket + dealias scratch:

- **1 GPU serial (the baseline!): 8.6–12.9 GB** — marginal at best on a
  16 GB A4000 even with `XLA_PYTHON_CLIENT_PREALLOCATE=false`; the next grid
  up has *no* single-device baseline at all.
- **2 GPUs, mesh (2,1): 4.3–6.4 GB/device** — comfortable, plus 25–42 MB/RHS
  of field psum and zero halo.
- **4 devices, mesh (2,2): 2.1–3.2 GB/device** + 134 MB/direction/RHS halo.

So the decomposition is simultaneously the strong-scaling path and the
memory-capacity path — the same dual role it plays in GX.

---

## 6. Trial protocol for plan item 4.2

### 6.1 Benchmark matrix

| ID | Case | Grid (Ns,Nl,Nm,Ny,Nx,Nz) | Mesh(es) | Purpose |
|---|---|---|---|---|
| G1 | Cyclone, adiabatic electrons, ES, periodic | (1,4,16,96,96,32) | (1,2), (1,4) CPU; (1,2) GPU | halo lane isolated (species absent) |
| G2 | Cyclone, kinetic electrons, ES, twist-shift linked | (2,4,16,96,96,32) | (2,1) GPU; (2,2) CPU/4 | species psum + linked chains + halo |
| G3 | Stellarator (shipped W7-X/HSX family, 2-species) | (2,8,16,128,128,64) | (2,1) GPU | production size; memory headroom claim |
| G4 | G2 + LB/Dougherty collisions + \|kz\| hypercollisions + reflectionless closure + EM (A∥,B∥) | (2,4,16,96,96,32) | (2,2) CPU/4 | full-term coverage incl. C3/C4 collectives, VMA retest |

Steps: 4 (identity), 64 (trace drift), 256 (transport window, G2/G3 only).
Integrators: RK2 primary; Euler for Tier-0 debugging; IMEX recorded
fail-closed until separately gated. Dtypes: complex64 timed; JAX_ENABLE_X64
for identity sweeps.

### 6.2 Methodology (locked)
- **One grid per process** (`--isolate-shapes` methodology from PR #45): no
  shape reuse across timed configurations; fresh process per (grid, mesh,
  route).
- **Three-route decomposition per configuration** (the
  `profile_device_z_pencil_scaling_decomposition.py` pattern): serial fused
  route on 1 device; shard_map route on a **1-device mesh** (route overhead);
  shard_map route on the N-device mesh (parallel scaling).
  `net = scaling / overhead`, reported with all three factors.
- Warmup 3, repeats ≥10, report median + spread; compute-only row separate
  from the streamed-diagnostics end-to-end row (both reported).
- GPUs: `XLA_PYTHON_CLIENT_PREALLOCATE=false`, CUDA 12, record driver/JAX
  versions; **do not time while the second A4000 carries other users' work**
  (the standing office caveat) — record `nvidia-smi` occupancy in the
  artifact.
- CPU: `XLA_FLAGS=--xla_force_host_platform_device_count={2,4,8}`,
  `PYTHONPATH=src`.
- HLO audit per timed route: collective census must show only
  `collective-permute` (halo) and `all-reduce` (psums); any `all-to-all` is
  an automatic fail.
- Artifacts: JSON+CSV+PNG per row with grid, mesh, steps, versions, identity
  errors beside timings (house style of
  `nonlinear_device_z_pencil_*_profile.json`), checked by
  `tools/release/check_parallel_scaling_artifacts.py` conventions.

### 6.3 Success gates
| Gate | Threshold |
|---|---|
| Route overhead (1-device mesh vs serial) | ≤1.05 at every grid |
| Parallel scaling (N-device vs 1-device mesh) | ≥1.90x at 2 devices |
| **Net compute speedup, 2 GPU, G2/G3** | **≥1.8x** |
| End-to-end with streamed diagnostics, 2 GPU | ≥1.7x; diagnostics ≤5% of step |
| Identity, complex64 | exact (0.0) for state and fields at G1/G2 4-step; traces via compensated sums exact or ≤1e-7 |
| Identity, x64 | ≤1e-12 relative on state, fields, all four traces |
| Runtime gate | strict_identity pass at atol 5e-6 / rtol 1e-4 (unchanged convention) |
| Transport window (256 steps, G2/G3) | trace agreement within Tier-2 tolerances post-transient |
| Adjoint | sharded grad of windowed heat flux = serial grad ≤1e-12 (x64); ≤1% FD check (f32) |
| Memory | G3 runs on 2×A4000 with ≥25% headroom; record the 1-GPU baseline outcome (fit or OOM) either way |
| CPU strong scaling (secondary) | ≥1.6x/2, ≥2.8x/4 logical devices at G1/G2 |

Promotion wording stays governed by docs/parallelization.rst claim rules:
identity artifacts and profiler artifacts are separate; a passing identity
with a failing speedup remains diagnostic.

### 6.4 Office-machine run list (in order)
1. `nvidia-smi` occupancy check; abort timing if GPU1 is loaded.
2. G1 CPU: mesh (1,2) and (1,4), 4-step identity (c64 + x64), then 64-step
   timed; three-route decomposition.
3. G2 CPU: mesh (2,2)/4 logical devices, same ladder; linked-boundary case.
4. G4 CPU: full-term coverage, 4-step identity only (VMA/collision retest on
   0.9.2, repeat once under `~/.venvs/gkx-jax-latest` 0.10.2).
5. G2 GPU: mesh (2,1) — identity 4-step, timed 64-step, three-route
   decomposition, HLO census, Perfetto trace.
6. G1 GPU: mesh (1,2) — the halo-on-PCIe measurement; report halo overlap
   efficiency explicitly.
7. G3 GPU: mesh (2,1) — memory headroom + timed 64-step; attempt 1-GPU serial
   baseline and record fit/OOM.
8. G2 GPU 256-step transport window with streamed diagnostics (end-to-end
   row + diagnostics-fraction gate).
9. Adjoint gate on G2 (CPU x64, then GPU f32 FD check).
10. Regenerate `parallelization_completion_status` style artifact row set;
    run the fast artifact contract check.

---

## 7. Sanity check performed for this design (2 logical CPU devices, jax 0.9.2)

Script: `scratchpad/sanity_species_hermite.py` (state (2,3,8,4,4,4) c64,
mesh ("m",)=2, `XLA_FLAGS=--xla_force_host_platform_device_count=2`).

| Check | Result |
|---|---|
| width-2 Hermite halo via `lax.ppermute` in `jax.shard_map` (curvature-style m±2 ladder) vs serial shift reference | **bitwise exact (0.0)** |
| m=0 moment psum (masked owner-shard contribution) scalar vs serial | **exact (0.0)** |
| `jax.grad` through shard_map(halo + psum), check_vma default | **bitwise exact (0.0)** vs serial grad |
| combined state update in one fused jit | 1.9e-6 max-abs (few ULP, c64) — attributable to XLA fusing the moment reduction differently inside the sharded jit, the exact effect `device_z.py:173-181` documents and the compensated-sum/fused-parity fix removes |

This confirms on the pinned stack: the collective set is exact and
transposable; the only identity risk is reduction-fusion order, which is a
known, fixed-pattern problem — not a communication problem.

---

## 8. Risks and fallbacks
1. **PCIe halo cost at Ns=1** (mesh (1,2), 4/Nm_loc of shard per RHS): if G1
   GPU misses the gate, production 2-GPU support for single-species runs
   waits for overlap work (issue halo before bracket) or NVLink hardware;
   Ns≥2 runs are unaffected (halo-free at 2 GPUs). Do not widen the gate.
2. **Collision VMA under shard_map** (0.6.2 failure): if it reproduces on
   0.9.2, collisions run via the psum-moment formulation (C4) which avoids
   the conditional annotations; retest on 0.10.2.
3. **Reduction-fusion residuals** blocking exact c64 identity: apply the
   device-z fused-parity approach term by term; x64 ≤1e-12 remains the
   fallback gate, runtime gate unchanged.
4. **IMEX**: stays fail-closed (as in the species pmap lane) until its solver
   internals get their own sharded identity gate; explicit RK2 is the
   production nonlinear integrator for the promotion claim.
5. **Sheared/radial-phase runs** (`_integrate_nonlinear_sheared_scan`):
   radial phase multiplies inside the perpendicular FFT pair — local under
   this mesh; include one sheared smoke case in Tier 1 before claiming it.
