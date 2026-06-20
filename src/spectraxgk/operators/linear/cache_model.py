"""Linear-operator cache data model."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field

import jax
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LinearCache:
    """Precomputed arrays for the linear operator."""

    Jl: jnp.ndarray
    b: jnp.ndarray
    kperp2: jnp.ndarray
    kperp2_bmag: bool
    bmag: jnp.ndarray
    omega_d: jnp.ndarray
    cv_d: jnp.ndarray
    gb_d: jnp.ndarray
    bgrad: jnp.ndarray
    jacobian: jnp.ndarray
    mask0: jnp.ndarray
    dz: jnp.ndarray
    kz: jnp.ndarray
    ky: jnp.ndarray
    kx: jnp.ndarray
    kx_grid: jnp.ndarray
    ky_grid: jnp.ndarray
    dealias_mask: jnp.ndarray
    kxfac: jnp.ndarray
    lb_lam: jnp.ndarray
    collision_lam: jnp.ndarray
    hyper_ratio: jnp.ndarray
    ratio_l: jnp.ndarray
    ratio_m: jnp.ndarray
    ratio_lm: jnp.ndarray
    mask_const: jnp.ndarray
    mask_kz: jnp.ndarray
    m_pow: jnp.ndarray
    m_norm_kz_factor: jnp.ndarray
    damp_profile: jnp.ndarray
    linked_damp_profile: jnp.ndarray
    l: jnp.ndarray  # noqa: E741 - public cache field for the Laguerre index.
    m: jnp.ndarray
    l4: jnp.ndarray
    sqrt_m: jnp.ndarray
    sqrt_m_p1: jnp.ndarray
    sqrt_p: jnp.ndarray
    sqrt_m_ladder: jnp.ndarray
    JlB: jnp.ndarray
    laguerre_to_grid: jnp.ndarray
    laguerre_to_spectral: jnp.ndarray
    laguerre_roots: jnp.ndarray
    laguerre_j0: jnp.ndarray
    laguerre_j1_over_alpha: jnp.ndarray
    kx_link_plus: jnp.ndarray
    kx_link_minus: jnp.ndarray
    kx_link_mask_plus: jnp.ndarray
    kx_link_mask_minus: jnp.ndarray
    linked_inverse_permutation: jnp.ndarray = dataclass_field(
        default_factory=lambda: jnp.asarray([], dtype=jnp.int32)
    )
    linked_gather_map: jnp.ndarray = dataclass_field(
        default_factory=lambda: jnp.asarray([], dtype=jnp.int32)
    )
    linked_gather_mask: jnp.ndarray = dataclass_field(
        default_factory=lambda: jnp.asarray([], dtype=bool)
    )
    linked_full_cover: bool = False
    linked_use_gather: bool = False
    linked_indices: tuple[jnp.ndarray, ...] = ()
    linked_kz: tuple[jnp.ndarray, ...] = ()
    use_twist_shift: bool = False
    jtwist: int = 0

    def tree_flatten(self):
        children = (
            self.Jl,
            self.b,
            self.kperp2,
            self.kperp2_bmag,
            self.bmag,
            self.omega_d,
            self.cv_d,
            self.gb_d,
            self.bgrad,
            self.jacobian,
            self.mask0,
            self.dz,
            self.kz,
            self.ky,
            self.kx,
            self.kx_grid,
            self.ky_grid,
            self.dealias_mask,
            self.kxfac,
            self.lb_lam,
            self.collision_lam,
            self.hyper_ratio,
            self.ratio_l,
            self.ratio_m,
            self.ratio_lm,
            self.mask_const,
            self.mask_kz,
            self.m_pow,
            self.m_norm_kz_factor,
            self.damp_profile,
            self.linked_damp_profile,
            self.l,
            self.m,
            self.l4,
            self.sqrt_m,
            self.sqrt_m_p1,
            self.sqrt_p,
            self.sqrt_m_ladder,
            self.JlB,
            self.laguerre_to_grid,
            self.laguerre_to_spectral,
            self.laguerre_roots,
            self.laguerre_j0,
            self.laguerre_j1_over_alpha,
            self.kx_link_plus,
            self.kx_link_minus,
            self.kx_link_mask_plus,
            self.kx_link_mask_minus,
            self.linked_inverse_permutation,
            self.linked_gather_map,
            self.linked_gather_mask,
        )
        linked_idx = self.linked_indices or ()
        linked_kz = self.linked_kz or ()
        children = children + tuple(linked_idx) + tuple(linked_kz)
        aux_data = (
            self.use_twist_shift,
            self.jtwist,
            len(linked_idx),
            len(linked_kz),
            self.linked_full_cover,
            self.linked_use_gather,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (
            use_twist_shift,
            jtwist,
            n_linked_idx,
            n_linked_kz,
            linked_full_cover,
            linked_use_gather,
        ) = aux_data
        base_count = 51
        base_children = children[:base_count]
        linked_idx = tuple(children[base_count : base_count + n_linked_idx])
        linked_kz = tuple(
            children[
                base_count + n_linked_idx : base_count + n_linked_idx + n_linked_kz
            ]
        )
        return cls(
            *base_children,
            linked_indices=linked_idx,
            linked_kz=linked_kz,
            use_twist_shift=use_twist_shift,
            jtwist=jtwist,
            linked_full_cover=linked_full_cover,
            linked_use_gather=linked_use_gather,
        )


