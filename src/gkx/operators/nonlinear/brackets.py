"""Pseudo-spectral bracket kernels for nonlinear gyrokinetic terms."""

from __future__ import annotations

import functools
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
from jax import lax

from gkx.core_grid import real_fft_mesh, real_fft_ordered_kx
from gkx.core_ky_layout import half_dealias_mask, is_half, to_full, to_half


def _fft2_xy(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.fft.fft2(x, axes=(-3, -2))


def _ifft2_xy(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.fft.ifft2(x, axes=(-3, -2))


def _ifft2_xy_with_radial_phase(
    x: jnp.ndarray, radial_phase: jnp.ndarray
) -> jnp.ndarray:
    """Transform a shearing wave after applying its fractional radial phase."""

    radial = jnp.fft.ifft(x, axis=-2)
    radial *= _broadcast_grid(radial_phase, radial.ndim)
    return jnp.fft.ifft(radial, axis=-3)


def _fft2_xy_remove_radial_phase(
    x: jnp.ndarray, radial_phase: jnp.ndarray
) -> jnp.ndarray:
    """Return a physical field to its nearest-cell shearing-wave coefficients."""

    binormal = jnp.fft.fft(x, axis=-3)
    binormal *= _broadcast_grid(jnp.conj(radial_phase), binormal.ndim)
    return jnp.fft.fft(binormal, axis=-2)


def _broadcast_mask(mask: jnp.ndarray, ndim: int) -> jnp.ndarray:
    shape = (1,) * (ndim - 3) + mask.shape + (1,)
    return jnp.reshape(mask, shape)


def _broadcast_grid(grid: jnp.ndarray, ndim: int) -> jnp.ndarray:
    shape = (1,) * (ndim - 3) + grid.shape + (1,)
    return jnp.reshape(grid, shape)


def _apply_mask_xy(field: jnp.ndarray, mask: jnp.ndarray | None) -> jnp.ndarray:
    if mask is None:
        return field
    real_dtype = jnp.real(jnp.empty((), dtype=field.dtype)).dtype
    mask_b = _broadcast_mask(jnp.asarray(mask, dtype=real_dtype), field.ndim)
    return field * mask_b


def _broadcast_to_G(x: jnp.ndarray, G: jnp.ndarray) -> jnp.ndarray:
    if x.ndim == G.ndim:
        return x
    if x.ndim == G.ndim - 1:
        return jnp.expand_dims(x, axis=-4)
    if x.ndim == 3:
        shape = (1,) * (G.ndim - 3) + x.shape
        return jnp.reshape(x, shape)
    if x.ndim < G.ndim:
        shape = (1,) * (G.ndim - x.ndim) + x.shape
        return jnp.reshape(x, shape)
    return x


def _stack_fields(G_hat: jnp.ndarray, fields: Sequence[jnp.ndarray]) -> jnp.ndarray:
    return jnp.stack(
        [_broadcast_to_G(jnp.asarray(field), G_hat) for field in fields], axis=0
    )


def _fft_scales(
    ky_grid: jnp.ndarray,
    *,
    real_dtype: jnp.dtype,
    fft_norm: float | None = None,
    ny_full: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return the ``(ifft, fft)`` normalizations for an ``Ny x Nx`` transform.

    The pair is fixed by the *physical* grid, ``Ny * Nx``, not by how many
    ``ky`` rows the operand stores: a half-spectrum operand is transformed with
    ``irfft2``/``rfft2`` over the same ``Ny`` real-space rows.
    """

    rows = int(ky_grid.shape[0]) if ny_full is None else int(ny_full)
    norm = float(rows * ky_grid.shape[1]) if fft_norm is None else float(fft_norm)
    return (
        jnp.asarray(norm, dtype=real_dtype),
        jnp.asarray(1.0 / norm, dtype=real_dtype),
    )


def _complete_hermitian_ky(
    positive_ky: jnp.ndarray,
    ny_full: int,
    nx: int,
    conjugate_kx: Any | None = None,
) -> jnp.ndarray:
    """Widen a ``ky >= 0`` block onto the two-sided axis.

    The rule lives in :mod:`gkx.core_ky_layout`; this wrapper keeps the
    positional signature the nonlinear kernels and their tests already use.
    """

    return to_full(
        positive_ky, ny_full=int(ny_full), nx=int(nx), conjugate_kx=conjugate_kx
    )


# Real-space bracket product. After the two ``irfft2`` calls the bracket is
# ``dG_x dchi_y - dG_y dchi_x`` per Hermite row, summed over the fields of a
# group. XLA fuses the scale, the broadcast over ``m`` and the field sum into
# loop fusions that run well off roofline, and writes them in a layout the
# forward transform then re-reads. ``fused_products`` reads each ``chi`` tile
# once per ``(s, l)`` program, reuses it for every ``m`` row, and writes the
# group sums in the layout ``rfft2`` consumes (Pallas/Triton, float32, NVIDIA).
# The jnp expression below stays the reference and the derivatives.


def _products_kernel(gx, gy, cx, cy, out, *, nm: int, groups, scale: float):
    s = jnp.asarray(scale, gx.dtype)
    cxs = [cx[f, 0, 0, :] * s for f in range(cx.shape[0])]
    cys = [cy[f, 0, 0, :] * s for f in range(cy.shape[0])]

    def row(m, _):
        a, b = gx[0, 0, m, :] * s, gy[0, 0, m, :] * s
        for g, fields in enumerate(groups):
            total = a * cys[fields[0]] - b * cxs[fields[0]]
            for f in fields[1:]:
                total = total + (a * cys[f] - b * cxs[f])
            out[g, 0, 0, m, :] = total
        return 0

    lax.fori_loop(0, nm, row, 0)


def _pallas_products(dgx, dgy, dcx, dcy, scale, groups, *, interpret=False):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu

    ns, nl, nm, ny, nx, nz = dgx.shape
    nf, n = dcx.shape[0], ny * nx * nz
    block = min(512, n & -n)
    flat = (ns, nl, nm, n)
    chi = (nf, ns, nl, n)
    g_spec = pl.BlockSpec((1, 1, nm, block), lambda s, l, p: (s, l, 0, p))
    c_spec = pl.BlockSpec((nf, 1, 1, block), lambda s, l, p: (0, s, l, p))
    o_spec = pl.BlockSpec(
        (len(groups), 1, 1, nm, block), lambda s, l, p: (0, s, l, 0, p)
    )
    out = pl.pallas_call(
        functools.partial(_products_kernel, nm=nm, groups=groups, scale=scale),
        out_shape=jax.ShapeDtypeStruct((len(groups),) + flat, dgx.dtype),
        grid=(ns, nl, n // block),
        in_specs=[g_spec, g_spec, c_spec, c_spec],
        out_specs=o_spec,
        interpret=interpret,
        compiler_params=plgpu.CompilerParams(num_warps=4),
    )(dgx.reshape(flat), dgy.reshape(flat), dcx.reshape(chi), dcy.reshape(chi))
    return out.reshape((len(groups),) + dgx.shape)


def _reference_products(dgx, dgy, dcx, dcy, scale, groups):
    dgx, dgy, dcx, dcy = (x * jnp.asarray(scale, x.dtype) for x in (dgx, dgy, dcx, dcy))
    prod = dgx[None] * dcy - dgy[None] * dcx

    def group_sum(fields):
        total = prod[fields[0]]
        for f in fields[1:]:
            total = total + prod[f]
        return total

    return jnp.stack([group_sum(fields) for fields in groups])


@functools.partial(jax.custom_jvp, nondiff_argnums=(4, 5))
def fused_products(dgx, dgy, dcx, dcy, scale, groups):
    """Group sums of ``dG_x dchi_y - dG_y dchi_x`` over the field stack.

    ``dgx``, ``dgy``: ``(Ns, Nl, Nm, Ny, Nx, Nz)``; ``dcx``, ``dcy``:
    ``(Nf, Ns, Nl, 1, Ny, Nx, Nz)``, all before the inverse-transform scale,
    which the kernel applies. Returns ``(len(groups), Ns, Nl, Nm, Ny, Nx, Nz)``.
    """

    return _pallas_products(dgx, dgy, dcx, dcy, scale, groups)


@fused_products.defjvp
def _fused_products_jvp(scale, groups, primals, tangents):
    tangents = tuple(
        jnp.zeros_like(p) if t is None else t for p, t in zip(primals, tangents)
    )
    ref = functools.partial(_reference_products, scale=scale, groups=groups)
    return fused_products(*primals, scale, groups), jax.jvp(ref, primals, tangents)[1]


def product_kernel_usable(g) -> bool:
    """Whether :func:`fused_products` applies: NVIDIA GPU, float32, 6-D state."""

    return (
        jax.default_backend() == "gpu"
        and "NVIDIA" in jax.devices()[0].device_kind
        and g.ndim == 6
        and g.dtype == jnp.float32
        and (g.shape[-3] * g.shape[-2] * g.shape[-1]) % 64 == 0
    )


def _spectral_bracket_half_core(
    G_hat: jnp.ndarray,
    chi_hat: jnp.ndarray,
    *,
    kx_grid: jnp.ndarray,
    ky_grid: jnp.ndarray,
    dealias_mask: jnp.ndarray,
    fft_norm: float | None = None,
    radial_phase: jnp.ndarray | None = None,
    ny_full: int | None = None,
    multiple_fields: bool,
    combine: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    fuse: tuple[tuple[tuple[int, ...], ...], Callable | None] | None = None,
) -> tuple[jnp.ndarray, int, int]:
    """Return the bracket on the ``ky >= 0`` rows, with ``(ny_full, nx)``.

    This is the whole pseudo-spectral kernel: both operands are read on their
    non-negative rows, transformed with ``irfft2``, multiplied in real space,
    and transformed back with ``rfft2``, which lands the product back on
    ``Nyc = 1 + Ny // 2`` rows.  Nothing in it needs the negative half, so it
    is the primitive the ``ky >= 0`` state layout (plan 5.3 N3) calls directly;
    :func:`_spectral_bracket_real_fft_core` is this function plus the widening
    that a two-sided state still requires.

    ``ny_full`` is the length of the two-sided ``ky`` axis and must be supplied
    whenever the operands are already in the half layout, because ``Nyc`` alone
    cannot say how long its own full axis is (:mod:`gkx.core_ky_layout`).
    Omitting it on a half operand is not a shape error -- the slice, the
    ``irfft2`` length and the mask all still compose -- it just computes the
    bracket of a different grid, which is why the argument is explicit rather
    than inferred.
    """

    complex_dtype = jnp.result_type(G_hat, chi_hat, jnp.complex64)
    real_dtype = jnp.real(jnp.empty((), dtype=complex_dtype)).dtype
    imag = jnp.asarray(1j, dtype=complex_dtype)
    G_hat = jnp.asarray(G_hat, dtype=complex_dtype)
    chi_hat = jnp.asarray(chi_hat, dtype=complex_dtype)
    mask = jnp.asarray(dealias_mask, dtype=real_dtype)
    kx = jnp.asarray(kx_grid, dtype=real_dtype)
    ky = jnp.asarray(ky_grid, dtype=real_dtype)
    if radial_phase is not None:
        phase = jnp.asarray(radial_phase, dtype=complex_dtype)
        if tuple(phase.shape) != tuple(kx.shape):
            raise ValueError("radial_phase must have shape (ky, x)")
        # Both bracket operands carry the same residual radial phase. It
        # cancels from their Poisson bracket, leaving the canonical shearing-
        # coordinate derivatives represented by the row-relative kx mesh.
        kx = kx - kx[:, :1]
    rows = int(ky.shape[0])
    ny = rows if ny_full is None else int(ny_full)
    ifft_scale, fft_scale = _fft_scales(
        ky_grid, real_dtype=real_dtype, fft_norm=fft_norm, ny_full=ny
    )

    if is_half(rows, ny_full):
        # The operands already carry the ``ky >= 0`` rows, so only the ``kx``
        # axis needs the real-FFT ordering: ``irfft2`` transforms ``ky``, which
        # leaves ``kx`` two-sided, and its Nyquist column's derivative
        # multiplier has to be taken with a positive sign there exactly as on a
        # full-axis operand.  ``real_fft_mesh`` cannot be reused because its
        # ``ky`` reduction would slice this axis a second time.
        kx_1d = real_fft_ordered_kx(kx)
        ky_nyc, kx_nyc = jnp.meshgrid(ky[:, 0], kx_1d, indexing="ij")
    else:
        _, _ky_half, kx_nyc, ky_nyc = real_fft_mesh(kx, ky)
    G_nyc = to_half(G_hat, ny_full=ny)
    chi_nyc = to_half(chi_hat, ny_full=ny)
    axes = (-2, -3)

    kx_b = _broadcast_grid(kx_nyc, G_nyc.ndim)
    ky_b = _broadcast_grid(ky_nyc, G_nyc.ndim)
    grad_G = jnp.stack([imag * kx_b * G_nyc, imag * ky_b * G_nyc], axis=0)
    kx_chi = _broadcast_grid(kx_nyc, chi_nyc.ndim)
    ky_chi = _broadcast_grid(ky_nyc, chi_nyc.ndim)
    grad_chi = jnp.stack([imag * kx_chi * chi_nyc, imag * ky_chi * chi_nyc], axis=0)
    shape = (kx.shape[1], ny)
    grad_G = jnp.fft.irfft2(grad_G, s=shape, axes=axes)
    grad_chi = jnp.fft.irfft2(grad_chi, s=shape, axes=axes)
    if fuse is not None and product_kernel_usable(grad_G[0]):
        groups, group_combine = fuse  # the kernel applies the scale
        norm = float(ny * kx.shape[1]) if fft_norm is None else float(fft_norm)
        chi_x, chi_y = grad_chi if multiple_fields else grad_chi[:, None]
        chi_shape = (chi_x.shape[0],) + grad_G[0].shape[:2] + chi_x.shape[-4:]
        chi_x, chi_y = (jnp.broadcast_to(c, chi_shape) for c in (chi_x, chi_y))
        bracket = fused_products(
            grad_G[0], grad_G[1], chi_x, chi_y, norm, groups
        )
        bracket = bracket[0] if group_combine is None else group_combine(bracket)
    else:
        dG_dx, dG_dy = grad_G * ifft_scale
        dchi_dx, dchi_dy = grad_chi * ifft_scale
        if multiple_fields:
            bracket = dG_dx[None, ...] * dchi_dy - dG_dy[None, ...] * dchi_dx
        else:
            bracket = dG_dx * dchi_dy - dG_dy * dchi_dx
        if combine is not None:  # linear in the real-space products: one forward FFT
            bracket = combine(bracket)
    positive_ky = jnp.fft.rfft2(bracket, axes=axes) * fft_scale
    positive_ky *= _broadcast_mask(
        half_dealias_mask(mask, ny_full=ny), positive_ky.ndim
    )
    return positive_ky, ny, int(kx.shape[1])


def _spectral_bracket_real_fft_core(
    G_hat: jnp.ndarray,
    chi_hat: jnp.ndarray,
    *,
    kxfac: jnp.ndarray,
    **kwargs: Any,
) -> jnp.ndarray:
    """Return the bracket in the ``ky`` layout its operands arrived in.

    The kernel is the same either way.  A two-sided operand pays the widening
    back onto ``Ny`` rows; a half-spectrum one does not, which is the whole of
    plan 5.3 N3 in the bracket -- the reality condition is a property of what
    is stored rather than something restored after the fact.
    """

    rows = int(jnp.asarray(kwargs["ky_grid"]).shape[0])
    half_operands = is_half(rows, kwargs.get("ny_full"))
    positive_ky, ny_full, nx = _spectral_bracket_half_core(G_hat, chi_hat, **kwargs)
    real_dtype = jnp.real(jnp.empty((), dtype=positive_ky.dtype)).dtype
    bracket_hat = (
        positive_ky
        if half_operands
        else _complete_hermitian_ky(positive_ky, ny_full, nx)
    )
    return jnp.asarray(kxfac, dtype=real_dtype) * bracket_hat


def _spectral_bracket_full_core(
    G_hat: jnp.ndarray,
    chi_hat: jnp.ndarray,
    *,
    kx_grid: jnp.ndarray,
    ky_grid: jnp.ndarray,
    dealias_mask: jnp.ndarray,
    kxfac: jnp.ndarray,
    fft_norm: float | None = None,
    radial_phase: jnp.ndarray | None = None,
    ny_full: int | None = None,
    multiple_fields: bool,
    combine: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    fuse: Any = None,  # the product kernel serves the half-spectrum route only
) -> jnp.ndarray:
    """Return the bracket by full complex transforms on a two-sided ``ky`` axis.

    This route multiplies the whole two-sided spectrum in real space, so it has
    no half-spectrum form: a state carrying only ``ky >= 0`` is refused here
    rather than transformed as if its rows were the full axis, which would
    silently compute the bracket of a different field.
    """

    if is_half(int(jnp.asarray(ky_grid).shape[0]), ny_full):
        raise ValueError(
            "the full-complex bracket needs the two-sided ky axis; this grid "
            "stores the ky >= 0 half. Use the compressed real-FFT bracket "
            "(compressed_real_fft=True) with a half-spectrum state."
        )
    complex_dtype = jnp.result_type(G_hat, chi_hat, jnp.complex64)
    real_dtype = jnp.real(jnp.empty((), dtype=complex_dtype)).dtype
    imag = jnp.asarray(1j, dtype=complex_dtype)
    G_hat = jnp.asarray(G_hat, dtype=complex_dtype)
    chi_hat = jnp.asarray(chi_hat, dtype=complex_dtype)
    mask = jnp.asarray(dealias_mask, dtype=real_dtype)
    kx = jnp.asarray(kx_grid, dtype=real_dtype)
    ky = jnp.asarray(ky_grid, dtype=real_dtype)
    ifft_scale, fft_scale = _fft_scales(
        ky_grid, real_dtype=real_dtype, fft_norm=fft_norm
    )
    phase = None
    if radial_phase is not None:
        phase = jnp.asarray(radial_phase, dtype=complex_dtype)
        if tuple(phase.shape) != tuple(kx.shape):
            raise ValueError("radial_phase must have shape (ky, x)")

    kx_b = _broadcast_grid(kx, G_hat.ndim)
    ky_b = _broadcast_grid(ky, G_hat.ndim)
    grad_G = jnp.stack([imag * kx_b * G_hat, imag * ky_b * G_hat], axis=0)
    if phase is None:
        dG_dx, dG_dy = _ifft2_xy(grad_G) * ifft_scale
    else:
        dG_dx, dG_dy = _ifft2_xy_with_radial_phase(grad_G, phase) * ifft_scale

    kx_chi = _broadcast_grid(kx, chi_hat.ndim)
    ky_chi = _broadcast_grid(ky, chi_hat.ndim)
    grad_chi = jnp.stack([imag * kx_chi * chi_hat, imag * ky_chi * chi_hat], axis=0)
    if phase is None:
        dchi_dx, dchi_dy = _ifft2_xy(grad_chi) * ifft_scale
    else:
        dchi_dx, dchi_dy = _ifft2_xy_with_radial_phase(grad_chi, phase) * ifft_scale

    if multiple_fields:
        bracket = dG_dx[None, ...] * dchi_dy - dG_dy[None, ...] * dchi_dx
    else:
        bracket = dG_dx * dchi_dy - dG_dy * dchi_dx
    if combine is not None:
        bracket = combine(bracket)
    bracket_hat = (
        _fft2_xy(bracket)
        if phase is None
        else _fft2_xy_remove_radial_phase(bracket, phase)
    ) * fft_scale
    bracket_hat *= _broadcast_mask(mask, bracket_hat.ndim)
    return jnp.asarray(kxfac, dtype=real_dtype) * bracket_hat


def _spectral_bracket_multi_real_fft(
    G_hat: jnp.ndarray,
    chi_hat_stack: jnp.ndarray,
    combine: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    fuse: tuple | None = None,
    **kwargs,
) -> jnp.ndarray:
    """Bracket of ``G`` with each field of the stack.

    ``combine`` reduces the field axis of the real-space products (linear, so
    one forward FFT serves every field); without it each field keeps its own.
    """

    # Shared singleton batch axes trigger XLA:CPU YNN's f32 VJP reduction fault.
    # Keep the original GPU layout: squeezing increases its VJP temporary memory.
    axes = tuple(
        i
        for i, n in enumerate(G_hat.shape[:-3])
        if n == 1 and chi_hat_stack.shape[i + 1] == 1
    )
    field_axes = tuple(i + 1 for i in axes)

    def run(g, chi, combine, fuse=None):
        return _spectral_bracket_real_fft_core(
            g, chi, multiple_fields=True, combine=combine, fuse=fuse, **kwargs
        )

    def reduced(g, chi):
        def squeezed(products):  # the caller's sum sees the full batch shape
            return jnp.squeeze(combine(jnp.expand_dims(products, field_axes)), axes)

        result = run(
            jnp.squeeze(g, axis=axes),
            jnp.squeeze(chi, axis=field_axes),
            None if combine is None else squeezed,
        )
        return jnp.expand_dims(result, axis=field_axes if combine is None else axes)

    return lax.platform_dependent(
        G_hat,
        chi_hat_stack,
        cpu=reduced,
        default=lambda g, chi: run(g, chi, combine, fuse),
    )


def _spectral_bracket_multi_full(
    G_hat: jnp.ndarray,
    chi_hat_stack: jnp.ndarray,
    combine: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    fuse: Any = None,
    **kwargs,
) -> jnp.ndarray:
    return _spectral_bracket_full_core(
        G_hat, chi_hat_stack, multiple_fields=True, combine=combine, **kwargs
    )


def _single_field(core: Callable[..., jnp.ndarray]) -> Callable[..., jnp.ndarray]:
    def bracket(G_hat: jnp.ndarray, chi_hat: jnp.ndarray, **kwargs) -> jnp.ndarray:
        chi = _broadcast_to_G(jnp.asarray(chi_hat), G_hat)
        return core(G_hat, chi, multiple_fields=False, **kwargs)

    return bracket


_spectral_bracket_real_fft = _single_field(_spectral_bracket_real_fft_core)
_spectral_bracket_full = _single_field(_spectral_bracket_full_core)


def _spectral_bracket(
    G_hat: jnp.ndarray,
    chi_hat: jnp.ndarray,
    *,
    compressed_real_fft: bool = True,
    **kwargs,
) -> jnp.ndarray:
    kernel = (
        _spectral_bracket_real_fft if compressed_real_fft else _spectral_bracket_full
    )
    return kernel(G_hat, chi_hat, **kwargs)
