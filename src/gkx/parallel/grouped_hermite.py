"""Hermite-sharded nonlinear route for species with unequal ``(Nl, Nm)``.

Species that keep different moment counts (``LinearParams.moments``) form groups
of unequal shape, and a ``shard_map`` runs one program on every device. The route
therefore cuts *each group's* Hermite axis into one block per device: a device
owns ``ns * nl * nm / D`` entries of every group, so the work per device is the
sum over groups of ``ns * nl * nm`` divided by ``D`` whatever the counts, and no
device idles on the padding of a smaller species.

A device's blocks are packed into one array ``(D * R, ky, kx, z)``, ``R`` being
the entries per device, so the stepper, the projector and the scan carry see the
single array they see on the uniform mesh. The field solve and the
magnetic-flutter halo are the uniform route's (:mod:`gkx.parallel.integrators`);
each group's right-hand side runs on its own exact-size cache.

:class:`GroupedLayout` is the same packed array as the carry of the diagnosed
(adaptive, chunked, progress) scan: the right-hand side, field solve and
diagnostics run on it with in-shard reductions, and the state is unpacked to
the ``(Ns, Nl, Nm)`` extent only for output.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from gkx.operators.linear.cache_arrays import moment_runs
from gkx.parallel import integrators as base
from gkx.parallel.velocity_hermite import (
    hermite_cache_partition_specs,
    hermite_head_terms,
    hermite_window_cache_arrays,
    hermite_window_indices,
)
from gkx.parallel.velocity_plan import build_grouped_hermite_mesh_plan


def group_shapes(params: Any, Nl: int, Nm: int) -> list[tuple[int, int, int]]:
    """``(ns, nl, nm)`` of each run of species keeping one ``(nl, nm)``."""

    return [(b - a, nl, nm) for a, b, nl, nm, _ in moment_runs(params.moments, Nl, Nm)]


def device_cost(shapes: list[tuple[int, int, int]], devices: int) -> float:
    """``Nl * Nm`` entries each of ``devices`` devices advances for these groups."""

    return sum(ns * nl * nm for ns, nl, nm in shapes) / devices


def _pack(G, runs, blocks):
    rest = G.shape[3:]
    parts = []
    for a, b, nl, nm, _ in runs:
        g = G[a:b, :nl, :nm].reshape(b - a, nl, blocks, nm // blocks, *rest)
        parts.append(jnp.moveaxis(g, 2, 0).reshape(blocks, -1, *rest))
    return jnp.concatenate(parts, axis=1).reshape(-1, *rest)


def _unpack(packed, runs, blocks, rest, *, shape=None):
    """One device's group slabs of ``packed``, or the whole array padded to ``shape``."""

    lead = blocks if shape is not None else 1
    rows = packed.reshape(lead, -1, *rest)
    out = None if shape is None else jnp.zeros(shape, packed.dtype)
    slabs, start = [], 0
    for a, b, nl, nm, _ in runs:
        size = (b - a) * nl * (nm // blocks)
        g = rows[:, start : start + size].reshape(lead, b - a, nl, nm // blocks, *rest)
        start += size
        if shape is None:
            slabs.append(g[0])
        else:
            g = jnp.moveaxis(g, 0, 2).reshape(b - a, nl, nm, *rest)
            out = out.at[a:b, :nl, :nm].set(g)
    return slabs if shape is None else out


@dataclass(frozen=True)
class _Stage:
    """Operands staged once on the Hermite mesh, and the shard-local right-hand side."""

    runs: list
    blocks: int
    mesh: Any
    resolved: Any
    sharded: tuple
    specs: tuple
    group_params: list
    local_rhs: Callable[..., Any]  # (local, *sharded) -> dG, fields, head, slabs


def _stage(
    G0,
    cache,
    params,
    *,
    term_cfg,
    plan,
    devices,
    num_devices,
    compressed_real_fft,
    laguerre_mode,
    head_rows,
):
    from gkx.parallel.velocity_plan import grouped_hermite_device_counts
    from gkx.solvers.linear.parallel_common import _resolve_parallel_devices

    Nl, Nm = G0.shape[1], G0.shape[2]
    runs = moment_runs(params.moments, Nl, Nm)
    shapes = group_shapes(params, Nl, Nm)
    device_list = list(
        devices
        if devices is not None
        else _resolve_parallel_devices(num_devices=num_devices)
    )
    try:
        resolved = plan or build_grouped_hermite_mesh_plan(
            shapes, num_devices=len(device_list)
        )
    except ValueError as exc:
        raise ValueError(
            f"{len(device_list)} devices cannot split species groups {shapes} "
            f"(ns, nl, nm): {exc}. Every group's Hermite count must divide by the "
            f"device count; this grid supports "
            f"{grouped_hermite_device_counts(shapes, limit=jax.device_count())} devices."
        ) from exc
    if resolved.chunks.get("s", 1) != 1:
        raise ValueError("unequal species counts shard Hermite only (species chunks 1)")
    blocks = int(resolved.chunks["m"])
    depth = int(resolved.hermite_ghost_depth)
    base._reject_unsharded_hermite_terms(term_cfg, params, resolved)
    mesh = Mesh(np.array(device_list[:blocks], dtype=object), ("m",))
    rest = tuple(G0.shape[3:])
    real_dtype = jnp.real(jnp.empty((), dtype=G0.dtype)).dtype
    linear_cfg = replace(term_cfg, nonlinear=0.0)
    rate = params.end_damping_rate(cache.dz * cache.damp_profile.shape[-1], real_dtype)
    group_params = [
        params.species_slice(a, b, damp_ends_rate=rate) for a, b, *_ in runs
    ]

    leaves, specs, counts = [], [], []
    for c, (_, _, nl, nm, _) in zip(cache.groups, runs, strict=True):
        index = hermite_window_indices(nm, chunks=blocks, ghost_depth=depth)
        windowed = hermite_window_cache_arrays(c, index, hermite_count=nm)
        window_specs = hermite_cache_partition_specs(windowed, "m")
        leaves += [np.asarray(index, np.int32), *windowed.values()]
        specs += [PartitionSpec("m"), *window_specs.values()]
        counts.append(tuple(windowed))
    sharded = tuple(
        base.stage_from_host(v, NamedSharding(mesh, s))
        for v, s in zip(leaves, specs, strict=True)
    )

    def local_rhs(local, *rest_leaves):
        slabs = _unpack(local, runs, blocks, rest)
        pos, heads, kwargs = 0, [], []
        for c, slab, (_, _, nl, nm, _), names, p in zip(
            cache.groups, slabs, runs, counts, group_params, strict=True
        ):
            index, *window = rest_leaves[pos : pos + 1 + len(names)]
            pos += 1 + len(names)
            head = base._local_head(
                slab, index, nm_chunks=blocks, ghost_depth=depth, head_rows=head_rows
            )
            heads.append(jnp.pad(head, [(0, 0), (0, Nl - nl)] + [(0, 0)] * 4))
            kwargs.append(
                dict(
                    cache=replace(c, **dict(zip(names, window, strict=True))),
                    params=p,
                    index=index,
                    hermite_count=nm,
                )
            )
        head = jnp.concatenate(heads, axis=0)
        fields = base._local_fields(head, cache, params, term_cfg, real_dtype, None)
        dG = [
            base._local_rhs(
                slab,
                fields,
                term_cfg=term_cfg,
                linear_cfg=linear_cfg,
                nm_chunks=blocks,
                ghost_depth=depth,
                real_dtype=real_dtype,
                compressed_real_fft=compressed_real_fft,
                laguerre_mode=laguerre_mode,
                **kw,
            ).reshape(-1, *rest)
            for slab, kw in zip(slabs, kwargs, strict=True)
        ]
        return jnp.concatenate(dG, axis=0), fields, head, slabs

    return _Stage(
        runs, blocks, mesh, resolved, sharded, tuple(specs), group_params, local_rhs
    )


def prepare_grouped_hermite(
    G0,
    cache,
    params,
    *,
    dt,
    steps,
    method_key,
    term_cfg,
    grid,
    vol_fac,
    flux_fac,
    plan,
    devices,
    num_devices,
    compressed_real_fft,
    laguerre_mode,
):
    """Stage the operands once; return the trajectory callable of ``base``."""

    record = vol_fac is not None and flux_fac is not None
    st = _stage(
        G0,
        cache,
        params,
        term_cfg=term_cfg,
        plan=plan,
        devices=devices,
        num_devices=num_devices,
        compressed_real_fft=compressed_real_fft,
        laguerre_mode=laguerre_mode,
        head_rows=base._DIAGNOSTIC_MOMENT_ROWS if record else 2,
    )

    def local_step(local, *leaves):
        dG, fields, head, slabs = st.local_rhs(local, *leaves)
        scalars = (
            base._fused_scalar_diagnostics(
                list(zip(slabs, st.group_params, strict=True)),
                head,
                fields,
                cache,
                params,
                grid,
                vol_fac=vol_fac,
                flux_fac=flux_fac,
                mesh_axes=("m",),
                global_species=int(G0.shape[0]),
            )
            if record
            else None
        )
        return dG, scalars

    compiled, sharding = base._scan_program(
        local_step,
        st.specs,
        st.mesh,
        PartitionSpec("m"),
        cache=cache,
        state_dtype=G0.dtype,
        dt=dt,
        steps=steps,
        method_key=method_key,
        compressed_real_fft=compressed_real_fft,
        record=record,
        axis_names={"m"},
    )
    runs, blocks, rest = st.runs, st.blocks, tuple(G0.shape[3:])

    def execute(operands, value):
        packed = _pack(jnp.asarray(value, dtype=G0.dtype), runs, blocks)
        final, stacked = compiled(base.stage_from_host(packed, sharding), *operands)
        traces = dict(zip(base._TRACE_NAMES, stacked, strict=True)) if record else {}
        state = _unpack(final, runs, blocks, rest, shape=tuple(G0.shape))
        return base.SpeciesHermiteRun(
            state=state, traces=traces, plan=st.resolved, mesh_shape=(1, blocks)
        )

    return jax.tree_util.Partial(execute, st.sharded)


class _View(NamedTuple):
    """What the diagnostic kernels receive for the packed state."""

    packed: Any  # the (D * R, ky, kx, z) carry
    head: Any  # its leading Hermite rows, complete on every device

    @property
    def shape(self):
        return self.head.shape

    @property
    def ndim(self):
        return self.head.ndim


class GroupedLayout:
    """Packed carry of the diagnosed scan, callable as its right-hand side.

    ``layout(P)`` returns ``(dP, fields)`` like the uniform named right-hand
    side; ``fields``, ``pack`` and ``unpack`` complete the interface. The
    diagnostics run in-shard on the packed array: the moment head (the rows
    heat flux, particle flux and heating read) and the free-energy spectra are
    reduced with one ``psum`` each, and every other term is a function of the
    replicated fields. No step gathers the state.
    """

    packed = True

    def __init__(self, G0, cache, params, *, term_cfg, plan, **options):
        st = _stage(
            G0,
            cache,
            params,
            term_cfg=term_cfg,
            plan=plan.mesh_plan,
            devices=plan.devices,
            num_devices=None,
            head_rows=2,
            **options,
        )
        self._st, self.operands = st, st.sharded
        self.shape = tuple(G0.shape)
        self.sharding = NamedSharding(st.mesh, PartitionSpec("m"))
        rest = self.rest = self.shape[3:]
        runs, blocks = st.runs, st.blocks
        real_dtype = jnp.real(jnp.empty((), dtype=G0.dtype)).dtype
        Nl, Nm = self.shape[1:3]

        def region(fn, ins, out):
            return jax.shard_map(
                fn,
                mesh=st.mesh,
                in_specs=ins,
                out_specs=out,
                axis_names={"m"},
                check_vma=base._CHECK_VMA,
            )

        def terms(local, rows):
            """Each group's share of the leading ``rows`` Hermite rows."""

            at = jax.lax.axis_index("m")
            return [
                hermite_head_terms(
                    s, at * (nm // blocks) + jnp.arange(nm // blocks), rows
                )
                for s, (*_, nm, _) in zip(_unpack(local, runs, blocks, rest), runs)
            ]

        def padded(parts):
            return jnp.concatenate(
                [
                    jnp.pad(h, [(0, 0), (0, Nl - nl)] + [(0, 0)] * 4)
                    for h, (_, _, nl, _, _) in zip(parts, runs)
                ]
            )

        def rhs(local, *leaves):
            dG, fields, *_ = st.local_rhs(local, *leaves)
            return dG, fields

        def fields(local):
            parts = jax.lax.psum(terms(local, 2), "m")
            return base._local_fields(
                padded(parts), cache, params, term_cfg, real_dtype, None
            )

        def heads(local, previous):
            new, old = jax.lax.psum((terms(local, 4), terms(previous, 2)), "m")
            return padded(new), padded(old)

        spec = PartitionSpec("m")
        self._rhs = region(rhs, (spec, *st.specs), (spec, PartitionSpec()))
        self._fields = region(fields, spec, PartitionSpec())
        self._heads = region(heads, (spec, spec), PartitionSpec())

        def free_energy(kernel, grid, vol_fac, use_dealias):
            def local(local):
                at = jax.lax.axis_index("m")
                out = [
                    kernel(s, grid, p, vol_fac, use_dealias=use_dealias)
                    for s, p in zip(_unpack(local, runs, blocks, rest), st.group_params)
                ]
                if not isinstance(out[0], tuple):
                    return jax.lax.psum(sum(out), "m")
                *spectra, lm = zip(*out)
                # (ns, Nm_local, nl) -> its place in the (ns, Nm, Nl) extent
                lm = [
                    jnp.pad(
                        jax.lax.dynamic_update_slice(
                            jnp.zeros((x.shape[0], nm, x.shape[2]), x.dtype),
                            x,
                            (0 * at, at * (nm // blocks), 0 * at),
                        ),
                        [(0, 0), (0, Nm - nm), (0, Nl - nl)],
                    )
                    for x, (_, _, nl, nm, _) in zip(lm, runs)
                ]
                return jax.lax.psum(
                    (*(jnp.concatenate(x) for x in spectra), jnp.concatenate(lm)), "m"
                )

            return lambda view: region(local, spec, PartitionSpec())(view.packed)

        self._free_energy = free_energy

    def __call__(self, G):
        return self._rhs(G, *self.operands)

    def fields(self, G, *_, **__):
        return self._fields(G)

    def pack(self, G):
        if G.ndim != len(self.shape):
            return G
        packed = _pack(G, self._st.runs, self._st.blocks)
        return jax.lax.with_sharding_constraint(packed, self.sharding)

    def unpack(self, G):
        if G.ndim == len(self.shape):
            return G
        return _unpack(G, self._st.runs, self._st.blocks, self.rest, shape=self.shape)

    def kernels(self, base_kernels):
        """The diagnostic kernels, reading the packed state through :class:`_View`."""

        def energy(kernel):
            return lambda G, grid, params, vol_fac, *, use_dealias: self._free_energy(
                kernel, grid, vol_fac, use_dealias
            )(G)

        def flux(kernel):  # reads the moment head
            return lambda G, *a, **k: kernel(G.head, *a, **k)

        def heating(kernel):  # reads the heads of both states
            return lambda G, old, *a, **k: kernel(G.head, old.head, *a, **k)

        wrappers = {
            **dict.fromkeys(
                ("distribution_free_energy", "distribution_free_energy_resolved"),
                energy,
            ),
            **dict.fromkeys(
                (
                    "heat_flux_species",
                    "heat_flux_channel_resolved_species",
                    "particle_flux_species",
                    "particle_flux_channel_resolved_species",
                ),
                flux,
            ),
            **dict.fromkeys(
                ("turbulent_heating_species", "turbulent_heating_resolved_species"),
                heating,
            ),
        }
        return replace(
            base_kernels,
            **{n: w(getattr(base_kernels, n)) for n, w in wrappers.items()},
        )

    def diagnostics(self, diagnostic):
        """``diagnostic`` (built on :meth:`kernels`) as a function of packed states."""

        def compute(G, fields, previous, previous_fields, dt):
            head, old = self._heads(G, previous)
            return diagnostic(
                _View(G, head), fields, _View(previous, old), previous_fields, dt
            )

        return compute
