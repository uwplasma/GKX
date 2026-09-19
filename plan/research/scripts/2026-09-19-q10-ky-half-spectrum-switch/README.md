# Q10, the `ky >= 0` state-layout switch (plan 5.3 N3)

Stage 1 (#248) wrote the layout contract, split `_spectral_bracket_half_core`
out as the seam the switch would call, and left the evolved state two-sided, so
none of the profiled 41.9 per cent Hermitian completion was recovered.  Q24
(#254) gave the ky weight rule one owner and pinned the float32 FFT flag.

This row clears the blockers stage 1 listed and makes the half layout
*reachable and measured*.  It does **not** change any default: the runtime still
builds a two-sided grid, and the evidence below is in two halves for that
reason — an A/B against `origin/main` showing every shipped path bitwise, and a
full-vs-half ledger showing what the layout would buy and where it does not.

## The headline

The switch works, and the measurement says **do not adopt it as the default
yet**.  On the RHS graph it is what N3 predicted.  On the RK step graph of a
*linked* deck it costs more bytes than it saves, and the cause is not the
layout's arithmetic but the linked-chain gather losing its fused lowering.  A
chain-free deck shows the win on both graphs, which is what pins the cause.

## Scripts

| file | what it does |
|---|---|
| `run_identity.sh` | A/B of `origin/main` against this branch: every RHS term, the total, the nonlinear RHS, the VJPs, and 100-step trajectories across integrators, routes, sharding and a checkpointed window, in float32 and x64. Q9's `rhs_identity.py`, `gate_traj.py` and `compare_npz.py` are reused verbatim. |
| `run_ledgers.sh` | Optimized-HLO op counts and byte totals for `--ky-layout full` and `--ky-layout half`, both routes, rk3 and rk4, at 32x32x24 and 64x64x24. Both arms are the same binary: the layout is a grid argument, not a code difference. |
| `ledger_table.py` | Prints the full-vs-half table, and checks the `full` arm against #248's committed ledger. |

Neither driver waits for an idle host, unlike the 2026-09-18 stage-1 driver.
Every number here is a bitwise comparison or an operation count, and neither
depends on machine load; `nice -n 10` is kept so a co-tenant's job is not
slowed.  **No wall-clock claim is made from any of these runs.**

The float32 arms pin `--xla_cpu_multi_thread_eigen=false`, because #248
measured two runs of unmodified `main` differing by 1.4e-10 with no code change
between them under the multithreaded XLA:CPU FFT thunk.

## Identity: the default path is bitwise

`identity/cmp_*.json`, `origin/main` `254fcc7b7` against this branch, separate
processes, the FFT flag pinned.

| gate | cases | float32 | x64 |
|---|---:|---|---|
| RHS terms, total, nonlinear RHS, VJPs wrt G / tprim / nu_hyper_m | 58 | 58/58 bitwise | 58/58 bitwise |
| 100-step trajectories: 7 integrators, runtime rk3/rk4 x 3 modes, sharded, species-Hermite, checkpointed window value and d/dtprim | 65 | 65/65 bitwise | 65/65 bitwise |

`max_rel` is exactly 0 in all four comparisons.  The trajectory rows include
`heat_flux_t`, `Wg_t` and `Wphi_t`, so the flux and energy paths are covered.

That is the whole claim about shipped behaviour: nothing moves.  The one rule
that changed for *both* layouts — an even grid's Nyquist row is now a flux
representative on the two-sided axis too — is masked away by two-thirds
dealiasing in every one of these cases, which is why it is invisible here and
why it is argued on its own terms in the log entry rather than measured here.

## The two layouts, compared directly

`half` against `full` is not an A/B of two trees: it is one binary lowering the
same deck onto `Ny` and onto `Nyc = 1 + Ny//2` ky rows.  The half state
represents the same field; the negative rows are simply not stored.

Cyclone nonlinear deck, XLA:CPU, complex64.  The `full` arm reproduces #248's
committed ledger on all twelve graphs (`ledger_table.txt`, last section).

### The RHS graph — what N3 promised

| graph | fft | concatenate | copy | transpose | gather | reverse | bytes written |
|---|---:|---:|---:|---:|---:|---:|---:|
| RHS 32x32x24 Nl2/Nm4 | 13 -> 13 | 18 -> 14 | 19 -> 17 | 19 -> 17 | 14 -> 13 | 3 -> 0 | 24,991,260 -> 11,676,288 (**-53.3%**) |
| RHS 64x64x24 Nl4/Nm8 | 17 -> 17 | 22 -> 18 | 23 -> 21 | 23 -> 21 | 18 -> 17 | 3 -> 0 | 424,812,232 -> 201,977,856 (**-52.5%**) |

`reverse` 3 -> 0 is the Hermitian completion itself: `to_full` is a slice, a
reverse and a concatenate, and on a half state there is nothing to widen.  The
FFT count is unchanged because the bracket already transformed only the
non-negative rows — the compression was always in the kernel; what moves is
what the kernel hands back.

### The RK step graph — where it does not pay

Both routes give identical counts (Q18 made `--route runtime` lower the same
graph as `--route diagnostics`), so one table covers them.

| graph | fft | concatenate | copy | transpose | gather | reverse | bytes written |
|---|---:|---:|---:|---:|---:|---:|---:|
| rk3, 32 | 44 -> 44 | 38 -> 24 | 85 -> 146 | 66 -> 128 | 41 -> 33 | 14 -> 0 | 89,479,380 -> 127,670,560 (**+42.7%**) |
| rk4, 32 | 57 -> 57 | 49 -> 32 | 105 -> 163 | 86 -> 145 | 53 -> 44 | 17 -> 0 | 117,413,076 -> 139,236,640 (**+18.6%**) |
| rk3, 64 | 56 -> 56 | 44 -> 30 | 100 -> 170 | 81 -> 151 | 53 -> 45 | 14 -> 0 | 1,509,607,604 -> 2,369,675,584 (**+57.0%**) |
| rk4, 64 | 73 -> 73 | 57 -> 40 | 124 -> 191 | 105 -> 172 | 69 -> 60 | 17 -> 0 | 1,983,162,548 -> 2,570,830,144 (**+29.6%**) |

Every count that measures *work the layout removes* goes the right way:
concatenates down 30-37 per cent, gathers down 13-20 per cent, reverses to
zero.  Copies and transposes go up 54-94 per cent, and they carry the bytes.

### Why: the linked-chain gather stops fusing

Three things rule out an algorithm change.

1. The chain topology is identical in both layouts: the same five classes
   `(175,1) (16,2) (1,3) (4,4) (1,5)`, the same `linked_use_gather`, the same
   `linked_full_cover`.  The chains visit the same physical `(ky, kx)` modes; a
   test pins that.
2. The *logical* op counts that would show extra work all fall.  Gathers go
   **down**, 41 -> 33.  The growth is in `copy` and `transpose` only.
3. The materialized shapes name the path.  In the two-sided graph XLA copies a
   15-row slice, `c64[1,2,4,15,32,24]`, eight times.  In the half graph it
   copies and transposes the *whole* state, `c64[1,2,4,17,32,24]`, and the
   flattened chain view `c64[2,1,2,4,544,24]` where `544 = 17 * 32`, forty
   times — once per chain class instead of once for all of them.

That is the mechanism stage 1 already recorded, at `_reverse_from_one`:
*"written as `jnp.take` with that index vector, XLA lowers it as a general
gather and materialises the result: a transpose, a full-array copy, the gather
and a transpose back"*.  `_flatten_linked_fft_state` avoids it by keeping the
state's own row order and translating the chain maps with `_state_mode_rows`,
whose index pattern XLA was recognizing.  On the half layout the state's ky
extent stops being the grid's power of two (32 -> 17, 64 -> 33) and that
recognition is lost.

**The control that proves it.**  The same deck with `boundary = "periodic"` has
no linked chains at all.  There the half layout wins on both graphs:

| graph | concatenate | gather | reverse | bytes written |
|---|---:|---:|---:|---:|
| RHS 32 | 4 -> 3 | 1 -> 0 | 1 -> 0 | 10,272,768 -> 7,870,464 (**-23.4%**) |
| rk3, 32 | 16 -> 6 | 8 -> 0 | 8 -> 0 | 46,818,516 -> 41,903,392 (**-10.5%**) |

Same layout, same code, chains removed: the regression goes with them.

## What this means for the row

The layout is right and the contract holds; the cost is in one lowering detail
of the linked-chain gather, which is a separable problem.  Flipping the default
on this evidence would trade a measured 53 per cent byte saving on the RHS for
a measured 43-57 per cent byte regression on the step that a production deck
actually runs.  The follow-up is to build the chain maps directly in the
state's row order on a half grid, so the gather keeps the fused form, and then
re-measure this table.
