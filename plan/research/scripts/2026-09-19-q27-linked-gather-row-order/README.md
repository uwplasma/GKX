# Q27, the linked-chain gather on a half `ky` axis (plan 5.3 N3)

Row Q27 was written to fix something: #258 measured the half `ky` layout
saving 53 per cent of the RHS graph's bytes and *costing* 19 to 57 per cent on
a full RK step of a linked deck, attributed the cost to the linked-chain
gather losing a fused lowering once `Nyc` stops being the grid's power of two,
and held the default flip on that.

**There was nothing to fix.** The gather lowers identically on both layouts.
The regression was in the instrument: the ledger's `bytes_written` charges
every `concatenate` and `copy` in the optimized module text, fusion interiors
included, and XLA:CPU emits an instruction written inside a fusion body as
index arithmetic on the fused loop without allocating anything for it.  On the
corrected measure the half layout wins **all twelve graphs**, the step by 70
to 74 per cent.

Nothing in `src/` changes in this row.  `git diff origin/main -- src/` is
empty, and the compiled modules measured here are byte-for-byte the ones #258
measured -- `ledger_table.txt` checks all 24 arm/graph pairs against #258's
committed JSONs and they are the same.

## The mechanism, in the order it was found

1. **Where the bytes are.**  Attributing every `copy` in the rk3 32x32x24
   graph to its Python frame gives two sites that move:
   `streaming.py:_linked_fft_gather_output` 7 -> 40, and
   `brackets.py:_spectral_bracket_half_core` 3 -> 40.  Both are the *same*
   instruction pattern in both layouts -- a layout-only `transpose` of a
   `multiply`, then a `copy` to the default layout -- so nothing changed shape.

2. **Where those copies live.**  Splitting the module by computation:

   | rk3 32x32x24 | full n | full bytes | half n | half bytes |
   |---|---:|---:|---:|---:|
   | `concatenate`, fusion ROOT | 17 | 33,070,080 | 4 | 2,322,432 |
   | `concatenate`, fusion interior | 21 | 8,911,872 | 20 | 8,202,240 |
   | `copy`, entry computation | 19 | 210,100 | 18 | 111,872 |
   | `copy`, fusion ROOT | 11 | 8,835,072 | 11 | 8,736,768 |
   | `copy`, fusion interior | 55 | 38,452,256 | **117** | **108,297,248** |

   Every one of the 40 half-layout copies is a fusion *interior*.  In context
   the operand is a fusion `parameter`: the chain gather's own result is
   materialized once and handed in, and what each consumer fusion re-does is
   the mask multiply and the re-layout, as index arithmetic on the loop it is
   already running.  The gather itself is **not** duplicated -- 3 chain-output
   gathers in the rk3 graph on both layouts, and the module-wide `gather` count
   *falls*, 41 -> 33.

3. **The `Nyc` power-of-two reading is refuted directly.**  `gather_probe.py`
   compiles `grad_z_linked_fft` alone and sweeps `Ny` (`gather_probe.txt`).
   The lowering is flat across every `Ny`: `full` is always gather 6 / copy 7 /
   transpose 7 / concatenate 3 / reverse 2, `half` is always gather 6 / copy 6 /
   transpose 6 / concatenate 1 / reverse 0.  `Ny = 62` gives `Nyc = 32`, a
   power of two on the half axis and not on the full one, and behaves exactly
   like `Nyc = 17` and `Nyc = 33`.  The factorization of the `ky` extent does
   not enter.  On the same probe the half arm's scratch arena is 86 to 88 per
   cent smaller than the full arm's at every `Ny`, against a state that is only
   47 per cent smaller -- the chain gather is *better* per stored byte on the
   half axis, not worse.

## The instrument

`tools/profiling/profile_runtime_kernels.py` now reports three byte totals
where it reported one, and records XLA's own buffer assignment beside them.

| key | what it is |
|---|---|
| `bytes_written` | unchanged: every `concatenate` and `copy` in the module text. Kept at its old value so the whole committed ledger archive stays comparable, and documented as what it is. |
| `materialized_bytes` | the subset that owns an output buffer: the entry computation, a while/call body, and the ROOT of each fusion body. |
| `fused_interior_bytes` | the remainder. The two partition `bytes_written` exactly. |
| `memory_analysis` | `temp_size_in_bytes` and friends from XLA's buffer assignment for the same executable -- not a text count, so a fused index cannot inflate it. |

## Files

| file | what it does |
|---|---|
| `run_ledgers.sh` | the twelve graphs (both routes, rk3 and rk4, 32x32x24 and 64x64x24, `full` and `half`) plus the chain-free control. `GKX_JAX_CACHE=0`, so each arm compiles cold. |
| `cyclone_nonlinear_periodic_control.toml` | the shipped Cyclone deck with `boundary = "periodic"`, kept as a file so the `config` field of its ledger names a path in this repository. |
| `gather_probe.py`, `gather_probe.json`, `gather_probe.txt` | `grad_z_linked_fft` compiled alone, swept over `Ny`, both layouts. |
| `ledger_table.py`, `ledger_table.txt` | the tables below, and the check of both arms against #258's committed ledger. |
| `ledgers/` | the ledger JSONs. |
| `identity/` | the four A/B comparisons against `origin/main`. |

## The corrected ledger

`ledger_table.txt` has all of it; both routes give identical counts, as the
Q18 reference-route contract requires.

### Op counts (unchanged from #258)

| graph | fft | concatenate | copy | transpose | gather | reverse |
|---|---:|---:|---:|---:|---:|---:|
| RHS 32 | 13->13 | 18->14 | 19->17 | 19->17 | 14->13 | 3->0 |
| rk3 32 | 44->44 | 38->24 | 85->146 | 66->128 | 41->33 | 14->0 |
| rk4 32 | 57->57 | 49->32 | 105->163 | 86->145 | 53->44 | 17->0 |
| RHS 64 | 17->17 | 22->18 | 23->21 | 23->21 | 18->17 | 3->0 |
| rk3 64 | 56->56 | 44->30 | 100->170 | 81->151 | 53->45 | 14->0 |
| rk4 64 | 73->73 | 57->40 | 124->191 | 105->172 | 69->60 | 17->0 |

### Bytes

| graph | `bytes_written` (the old headline) | `materialized_bytes` | `temp_size_in_bytes` |
|---|---|---|---|
| RHS 32 | 24,991,260 -> 11,676,288 (**-53.3%**) | 11,278,236 -> 4,122,624 (**-63.4%**) | 14,369,264 -> 5,320,816 (**-63.0%**) |
| rk3 32 | 89,479,380 -> 127,670,560 (+42.7%) | 42,115,252 -> 11,171,072 (**-73.5%**) | 47,988,736 -> 24,129,536 (**-49.7%**) |
| rk4 32 | 117,413,076 -> 139,236,640 (+18.6%) | 53,392,564 -> 15,293,696 (**-71.4%**) | 73,564,160 -> 25,073,920 (**-65.9%**) |
| RHS 64 | 424,812,232 -> 201,977,856 (**-52.5%**) | 192,827,080 -> 72,327,168 (**-62.5%**) | 233,636,336 -> 103,267,824 (**-55.8%**) |
| rk3 64 | 1,509,607,604 -> 2,369,675,584 (+57.0%) | 707,127,348 -> 195,190,976 (**-72.4%**) | 770,768,896 -> 430,118,400 (**-44.2%**) |
| rk4 64 | 1,983,162,548 -> 2,570,830,144 (+29.6%) | 899,950,644 -> 267,518,144 (**-70.3%**) | 1,181,351,936 -> 443,516,928 (**-62.5%**) |

The state shrinks 46.9 per cent between the layouts, so a 70 to 74 per cent
fall in materialized `copy`/`concatenate` bytes is not the state getting
smaller: it is the Hermitian completion's work disappearing.  It is visible
one line down in the split -- `concatenate` at a fusion ROOT, 17 instructions
and 33.1 MB on the two-sided axis, 4 and 2.3 MB on the half one.

### The chain-free control

The control still wins on both graphs, as #258 reported.  What is new is that
the **linked** deck now wins by *more* on the step than the chain-free one
does (-73.5% against -59.1% at rk3 32), which is the opposite of a deck whose
chains cost it something.

| graph | `bytes_written` | `materialized_bytes` | `temp_size_in_bytes` |
|---|---|---|---|
| RHS | 10,272,768 -> 7,870,464 (-23.4%) | 4,448,256 -> 2,875,392 (-35.4%) | 8,483,184 -> 5,833,328 (-31.2%) |
| rk3 | 46,818,516 -> 41,903,392 (-10.5%) | 21,628,084 -> 8,848,640 (-59.1%) | 40,861,696 -> 24,673,280 (-39.6%) |
| rk4 | 59,204,820 -> 49,669,408 (-16.1%) | 26,076,340 -> 11,724,032 (-55.0%) | 61,767,680 -> 25,617,664 (-58.5%) |

## Identity

`src/` is untouched, so the shipped path cannot have moved.  The A/B is run
anyway, because it is this queue's registered gate and because "no diff" is a
claim about the tree rather than about the executable.  `origin/main`
`195205961`'s tree against this branch, separate processes, the float32 FFT
thread pool pinned off (`identity/cmp_*.json`).

| gate | cases | float32 | x64 |
|---|---:|---|---|
| RHS terms, total, nonlinear RHS, VJPs wrt G / tprim / nu_hyper_m | 58 | 58/58 bitwise | 58/58 bitwise |
| 100-step trajectories: 7 integrators, runtime rk3/rk4 x 3 modes, sharded, species-Hermite, checkpointed window value and d/dtprim | 65 | 65/65 bitwise | 65/65 bitwise |

`max_rel` is exactly 0 in all four comparisons.

## What is not claimed

`fused_interior_bytes` rises, +146% and +171% on the rk3 graphs.  Those
instructions allocate nothing, but they are not free: a fusion that reads its
parameter through a non-default layout reads it strided.  Whether that costs
wall-clock is a timing question, and this row makes **no timing claim** -- the
host carried a one-minute load average above 10 throughout.  Every number
here is an op count, a byte total derived from a compiled module, or XLA's
own buffer assignment, and none of them depends on machine load.

## Environment

macOS 14.4.1 arm64 (M3), python 3.11.14, jax/jaxlib 0.10.2, `JAX_PLATFORMS=cpu`,
`GKX_JAX_CACHE=0`, ruff 0.16.4, `nice -n 10`.  The identity arms additionally
pin `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false` for float32, and reuse
`../2026-09-19-q10-ky-half-spectrum-switch/run_identity.sh` verbatim.
