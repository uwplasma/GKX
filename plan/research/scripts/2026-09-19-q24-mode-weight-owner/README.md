# Q24 -- one owner for the ky weight rule (plan 5.3 N3 prerequisite)

Two questions, two harnesses. Both arms are separate processes from the same
interpreter, `origin/main` at `cf89dcc70` against the branch, and both pin
`--xla_cpu_multi_thread_eigen=false`.

## `weights_ab.py` -- does any weight move?

The rule is all this row changes, so the weight arrays themselves are the
proof. 640 cases: `Ny` 2..17 (even and odd), `Nx` in {1, 2, 3, 4, 8}, both
layouts, dealiased and not, both the Hermitian and the flux weight, compared as
raw bytes.

    python weights_ab.py out.json            # once per tree
    python weights_ab.py --diff ref.json new.json

`weights_ab.txt` is the result: **560 of 640 bitwise**, and the 80 that move are
exactly the undealiased half-axis weights of the eight even `Ny` -- the Nyquist
row, and only it. Every two-sided case, every dealiased case and every odd-`Ny`
case is unchanged.

## The RHS, VJP and trajectory gates -- does any run move?

Q9's harness, driven as in the 2026-09-18 Q10 stage-1 entry:

    python ../2026-09-14-q9-batched-chain-fft/rhs_identity.py out.npz
    python ../2026-09-14-q9-batched-chain-fft/gate_traj.py out.npz
    python ../2026-09-14-q9-batched-chain-fft/compare_npz.py ref.npz new.npz cmp.json

`cmp_rhs_{f32,x64}.json`: 58/58 bitwise, `max_rel` 0.
`cmp_traj_{f32,x64}.json`: 65/65 bitwise, `max_rel` 0. The trajectory rows
include `heat_flux_t`, `Wg_t` and `Wphi_t`, which are the diagnostics that read
the two weights.

With the flag pinned, the float32 arms are reproducible: #248 recorded a
1.4e-10 run-to-run move on `nl32/nonlinear_rhs` without it, on unmodified
`main`.
