"""Sampled and imported flux-tube geometry contracts.

``FluxTubeGeometryData`` is the solver-facing contract shared by analytic
geometry, imported NetCDF/eik files, and differentiable VMEC/Boozer bridges.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from spectraxgk.geometry.analytic import SAlphaGeometry, SlabGeometry

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class FluxTubeGeometryData:
    """Sampled flux-tube geometry contract for solver-ready metric profiles."""

    theta: jnp.ndarray
    gradpar_value: float
    bmag_profile: jnp.ndarray
    bgrad_profile: jnp.ndarray
    gds2_profile: jnp.ndarray
    gds21_profile: jnp.ndarray
    gds22_profile: jnp.ndarray
    cv_profile: jnp.ndarray
    gb_profile: jnp.ndarray
    cv0_profile: jnp.ndarray
    gb0_profile: jnp.ndarray
    jacobian_profile: jnp.ndarray
    grho_profile: jnp.ndarray
    q: float
    s_hat: float
    epsilon: float
    R0: float
    B0: float = 1.0
    alpha: float = 0.0
    drift_scale: float = 1.0
    kxfac: float = 1.0
    theta_scale: float = 1.0
    nfp: int = 1
    kperp2_bmag: bool = True
    bessel_bmag_power: float = 0.0
    source_model: str = "sampled"
    theta_closed_interval: bool = False

    def tree_flatten(self):
        children = (
            self.theta,
            self.gradpar_value,
            self.bmag_profile,
            self.bgrad_profile,
            self.gds2_profile,
            self.gds21_profile,
            self.gds22_profile,
            self.cv_profile,
            self.gb_profile,
            self.cv0_profile,
            self.gb0_profile,
            self.jacobian_profile,
            self.grho_profile,
            self.q,
            self.s_hat,
            self.epsilon,
            self.R0,
            self.B0,
            self.alpha,
            self.drift_scale,
            self.kxfac,
            self.theta_scale,
            self.nfp,
            self.kperp2_bmag,
            self.bessel_bmag_power,
        )
        return children, {
            "source_model": self.source_model,
            "theta_closed_interval": self.theta_closed_interval,
        }

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            *children,
            source_model=aux_data["source_model"],
            theta_closed_interval=aux_data["theta_closed_interval"],
        )

    def _theta_matches(self, theta: jnp.ndarray) -> jnp.ndarray:
        theta_arr = jnp.asarray(theta)
        if theta_arr.shape[-1] != self.theta.shape[0]:
            raise ValueError(
                "theta must have the same last dimension as the sampled geometry grid"
            )
        if isinstance(theta_arr, jax.core.Tracer):
            return theta_arr
        theta_line = (
            theta_arr
            if theta_arr.ndim == 1
            else theta_arr.reshape(-1, theta_arr.shape[-1])[0]
        )
        if isinstance(theta_line, jax.core.Tracer):
            return theta_arr
        if not np.allclose(
            np.asarray(theta_line), np.asarray(self.theta), rtol=1.0e-6, atol=1.0e-6
        ):
            raise ValueError("theta does not match the sampled geometry grid")
        return theta_arr

    def _broadcast_profile(
        self, theta: jnp.ndarray, profile: jnp.ndarray
    ) -> jnp.ndarray:
        theta_arr = self._theta_matches(theta)
        if theta_arr.ndim == 1:
            return profile
        shape = (1,) * (theta_arr.ndim - 1) + (profile.shape[0],)
        return jnp.broadcast_to(profile.reshape(shape), theta_arr.shape)

    def trim_terminal_theta_point(self) -> FluxTubeGeometryData:
        """Return a copy without the terminal theta sample.

        Imported `*.eik.nc` files commonly store a closed theta interval, while the
        spectral solver uses the matching open interval with the terminal point
        excluded. Trimming keeps the imported coefficients aligned with the
        runtime grid without changing the physical extent.
        """

        if self.theta.shape[0] < 2:
            raise ValueError(
                "Cannot trim the terminal point from a geometry grid with fewer than two samples"
            )
        return replace(
            self,
            theta=self.theta[:-1],
            bmag_profile=self.bmag_profile[:-1],
            bgrad_profile=self.bgrad_profile[:-1],
            gds2_profile=self.gds2_profile[:-1],
            gds21_profile=self.gds21_profile[:-1],
            gds22_profile=self.gds22_profile[:-1],
            cv_profile=self.cv_profile[:-1],
            gb_profile=self.gb_profile[:-1],
            cv0_profile=self.cv0_profile[:-1],
            gb0_profile=self.gb0_profile[:-1],
            jacobian_profile=self.jacobian_profile[:-1],
            grho_profile=self.grho_profile[:-1],
            theta_closed_interval=False,
        )

    def gradpar(self) -> float:
        return float(self.gradpar_value)

    def bmag(self, theta: jnp.ndarray) -> jnp.ndarray:
        return self._broadcast_profile(theta, self.bmag_profile)

    def bgrad(self, theta: jnp.ndarray) -> jnp.ndarray:
        return self._broadcast_profile(theta, self.bgrad_profile)

    def metric_coeffs(
        self, theta: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return (
            self._broadcast_profile(theta, self.gds2_profile),
            self._broadcast_profile(theta, self.gds21_profile),
            self._broadcast_profile(theta, self.gds22_profile),
        )

    def drift_coeffs(
        self,
        theta: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return (
            self._broadcast_profile(theta, self.cv_profile),
            self._broadcast_profile(theta, self.gb_profile),
            self._broadcast_profile(theta, self.cv0_profile),
            self._broadcast_profile(theta, self.gb0_profile),
        )

    def jacobian(self, theta: jnp.ndarray) -> jnp.ndarray:
        return self._broadcast_profile(theta, self.jacobian_profile)

    def grho(self, theta: jnp.ndarray) -> jnp.ndarray:
        return self._broadcast_profile(theta, self.grho_profile)

    def k_perp2(
        self, kx0: jnp.ndarray, ky: jnp.ndarray, theta: jnp.ndarray
    ) -> jnp.ndarray:
        gds2, gds21, gds22 = self.metric_coeffs(theta)
        s_hat = jnp.asarray(self.s_hat)
        s_hat_safe = jnp.where(s_hat == 0.0, 1.0, s_hat)
        kx_hat = kx0 / s_hat_safe
        kx_hat = jnp.where(s_hat == 0.0, kx0, kx_hat)
        kperp2 = ky * (ky * gds2 + 2.0 * kx_hat * gds21) + (kx_hat * kx_hat) * gds22
        if self.kperp2_bmag:
            bmag_inv = 1.0 / self.bmag(theta)
            return kperp2 * (bmag_inv * bmag_inv)
        return kperp2

    def drift_components(
        self,
        kx: jnp.ndarray,
        ky: jnp.ndarray,
        theta: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        kx0 = kx[None, :, None]
        ky0 = ky[:, None, None]
        theta0 = theta[None, None, :]
        cv, gb, cv0, gb0 = self.drift_coeffs(theta0)
        s_hat = jnp.asarray(self.s_hat)
        s_hat_safe = jnp.where(s_hat == 0.0, 1.0, s_hat)
        kx_hat = kx0 / s_hat_safe
        kx_hat = jnp.where(s_hat == 0.0, kx0, kx_hat)
        cv_d = ky0 * cv + kx_hat * cv0
        gb_d = ky0 * gb + kx_hat * gb0
        return cv_d, gb_d

    def omega_d(
        self, kx: jnp.ndarray, ky: jnp.ndarray, theta: jnp.ndarray
    ) -> jnp.ndarray:
        cv_d, gb_d = self.drift_components(kx, ky, theta)
        return cv_d + gb_d


def sample_flux_tube_geometry(
    geom: SAlphaGeometry | SlabGeometry, theta: jnp.ndarray
) -> FluxTubeGeometryData:
    """Sample an analytic geometry model onto a flux-tube theta grid."""

    theta_arr = jnp.asarray(theta)
    gds2, gds21, gds22 = geom.metric_coeffs(theta_arr)
    gds22_arr = gds22 if gds22.ndim else jnp.full_like(theta_arr, gds22)
    cv, gb, cv0, gb0 = geom.drift_coeffs(theta_arr)
    bmag = geom.bmag(theta_arr)
    gradpar = float(geom.gradpar())
    jacobian = 1.0 / (jnp.abs(jnp.asarray(gradpar)) * bmag)
    return FluxTubeGeometryData(
        theta=theta_arr,
        gradpar_value=gradpar,
        bmag_profile=bmag,
        bgrad_profile=geom.bgrad(theta_arr),
        gds2_profile=gds2,
        gds21_profile=gds21,
        gds22_profile=gds22_arr,
        cv_profile=cv,
        gb_profile=gb,
        cv0_profile=cv0,
        gb0_profile=gb0,
        jacobian_profile=jacobian,
        grho_profile=jnp.ones_like(theta_arr),
        q=float(geom.q),
        s_hat=float(geom.s_hat),
        epsilon=float(geom.epsilon),
        R0=float(geom.R0),
        B0=float(geom.B0),
        alpha=float(geom.alpha),
        drift_scale=float(geom.drift_scale),
        kxfac=1.0,
        theta_scale=1.0,
        nfp=1,
        kperp2_bmag=bool(geom.kperp2_bmag),
        bessel_bmag_power=float(geom.bessel_bmag_power),
        source_model="slab" if isinstance(geom, SlabGeometry) else "s-alpha",
        theta_closed_interval=False,
    )


def _periodic_spectral_derivative(values: np.ndarray, spacing: float) -> np.ndarray:
    """Return the periodic spectral derivative of a uniform 1D profile."""

    if values.ndim != 1:
        raise ValueError(
            "Periodic spectral derivatives require a one-dimensional profile"
        )
    if values.size < 2:
        return np.zeros_like(values)
    k = 2.0 * np.pi * np.fft.fftfreq(values.size, d=spacing)
    deriv_hat = 1j * k * np.fft.fft(values)
    return np.fft.ifft(deriv_hat).real.astype(values.dtype, copy=False)


def _bgrad_from_bmag(
    theta: np.ndarray, bmag: np.ndarray, gradpar_val: float, *, closed: bool
) -> np.ndarray:
    """Reconstruct the mirror term from ``bmag`` on the solver theta grid."""

    if theta.ndim != 1 or bmag.ndim != 1:
        raise ValueError(
            "Imported geometry bgrad reconstruction expects one-dimensional theta and bmag profiles"
        )
    if theta.shape != bmag.shape:
        raise ValueError(
            "theta and bmag must have the same shape for Imported geometry bgrad reconstruction"
        )
    if theta.size < 2:
        return np.zeros_like(bmag)
    if closed:
        work_theta = theta[:-1]
        work_bmag = bmag[:-1]
    else:
        work_theta = theta
        work_bmag = bmag
    spacing = float(work_theta[1] - work_theta[0])
    d_bmag = _periodic_spectral_derivative(work_bmag, spacing)
    bgrad = float(gradpar_val) * d_bmag / np.clip(work_bmag, 1.0e-30, None)
    if closed:
        return np.concatenate([bgrad, bgrad[:1]])
    return bgrad


def load_imported_geometry_netcdf(path: str | Path) -> FluxTubeGeometryData:
    """Load sampled flux-tube geometry from an imported NetCDF/eik file."""

    try:
        from netCDF4 import Dataset
    except ImportError as exc:  # pragma: no cover - optional import
        raise ImportError(
            "netCDF4 is required to load imported geometry NetCDF files"
        ) from exc

    def _read_scalar(variables, *names: str, default: float | None = None) -> float:
        for name in names:
            if name in variables:
                arr = np.asarray(variables[name][:], dtype=float)
                if arr.ndim == 0:
                    return float(arr)
                if arr.ndim == 1 and np.allclose(arr, arr[0]):
                    return float(arr[0])
                raise ValueError(
                    f"Imported geometry variable '{name}' must be scalar or constant on theta"
                )
        if default is None:
            raise KeyError(names[0])
        return float(default)

    def _read_profile(variables, *names: str) -> np.ndarray:
        for name in names:
            if name in variables:
                arr = np.asarray(variables[name][:], dtype=float)
                if arr.ndim != 1:
                    raise ValueError(
                        f"Imported geometry variable '{name}' must be one-dimensional on theta"
                    )
                return arr
        raise KeyError(names[0])

    def _infer_root_theta_closed_interval(theta: np.ndarray, variables) -> bool:
        """Infer whether a root-level ``*.eik.nc`` file includes a terminal theta endpoint.

        VMEC-style ``*.eik.nc`` files often include the periodic terminal point, while
        Miller helper writes an already-open theta grid. Root-level files therefore
        cannot be treated as closed intervals unconditionally.
        """

        if theta.ndim != 1 or theta.size < 2:
            return False
        profile_names = (
            "bmag",
            "gds2",
            "gds21",
            "gds22",
            "cvdrift",
            "gbdrift",
            "grho",
        )
        matches = 0
        checked = 0
        for name in profile_names:
            if name not in variables:
                continue
            arr = np.asarray(variables[name][:], dtype=float)
            if arr.ndim != 1 or arr.size != theta.size:
                continue
            checked += 1
            scale = max(float(np.nanmax(np.abs(arr))), 1.0)
            if abs(float(arr[-1] - arr[0])) <= max(1.0e-10, 1.0e-6 * scale):
                matches += 1
        if checked == 0:
            return False
        return matches >= max(1, checked // 2 + checked % 2)

    root = Dataset(Path(path), "r")
    try:
        is_grouped_output = "Geometry" in root.groups and "Grids" in root.groups
        if is_grouped_output:
            geom_vars = root.groups["Geometry"].variables
            grid_vars = root.groups["Grids"].variables
            theta = _read_profile(grid_vars, "theta")
            theta_closed_interval = False
        else:
            geom_vars = root.variables
            grid_vars = root.variables
            theta = _read_profile(root.variables, "theta")
            theta_closed_interval = _infer_root_theta_closed_interval(
                np.asarray(theta, dtype=float), root.variables
            )

        gradpar_val = _read_scalar(geom_vars, "gradpar")
        bmag = _read_profile(geom_vars, "bmag")
        drhodpsi = _read_scalar(geom_vars, "drhodpsi", default=1.0)
        if is_grouped_output and "bgrad" in geom_vars:
            bgrad = _read_profile(geom_vars, "bgrad")
        else:
            bgrad = _bgrad_from_bmag(
                np.asarray(theta, dtype=float),
                np.asarray(bmag, dtype=float),
                gradpar_val,
                closed=theta_closed_interval,
            )
        if is_grouped_output:
            cvdrift = _read_profile(geom_vars, "cvdrift")
            gbdrift = _read_profile(geom_vars, "gbdrift")
            cvdrift0 = _read_profile(geom_vars, "cvdrift0")
            gbdrift0 = _read_profile(geom_vars, "gbdrift0")
            jacobian = _read_profile(geom_vars, "jacobian", "jacob")
        else:
            # Root-level VMEC ``*.eik.nc`` files carry a pre-load drift
            # normalization and Jacobian that are converted at load time.
            cvdrift = 0.5 * _read_profile(geom_vars, "cvdrift")
            gbdrift = 0.5 * _read_profile(geom_vars, "gbdrift")
            cvdrift0 = 0.5 * _read_profile(geom_vars, "cvdrift0")
            gbdrift0 = 0.5 * _read_profile(geom_vars, "gbdrift0")
            jacobian = 1.0 / np.abs(
                float(drhodpsi) * float(gradpar_val) * np.asarray(bmag, dtype=float)
            )
        rmaj = _read_scalar(geom_vars, "rmaj", "Rmaj", default=1.0)
        aminor = _read_scalar(geom_vars, "aminor", default=0.0)
        epsilon = aminor / rmaj if abs(rmaj) > 0.0 else 0.0
        return FluxTubeGeometryData(
            theta=jnp.asarray(theta),
            gradpar_value=gradpar_val,
            bmag_profile=jnp.asarray(bmag),
            bgrad_profile=jnp.asarray(bgrad),
            gds2_profile=jnp.asarray(_read_profile(geom_vars, "gds2")),
            gds21_profile=jnp.asarray(_read_profile(geom_vars, "gds21")),
            gds22_profile=jnp.asarray(_read_profile(geom_vars, "gds22")),
            cv_profile=jnp.asarray(cvdrift),
            gb_profile=jnp.asarray(gbdrift),
            cv0_profile=jnp.asarray(cvdrift0),
            gb0_profile=jnp.asarray(gbdrift0),
            jacobian_profile=jnp.asarray(jacobian),
            grho_profile=jnp.asarray(_read_profile(geom_vars, "grho")),
            q=_read_scalar(geom_vars, "q", default=0.0),
            s_hat=_read_scalar(geom_vars, "shat", default=0.0),
            epsilon=float(epsilon),
            R0=float(rmaj),
            B0=1.0,
            alpha=_read_scalar(geom_vars, "alpha", default=0.0),
            drift_scale=1.0,
            kxfac=_read_scalar(geom_vars, "kxfac", default=1.0),
            theta_scale=_read_scalar(geom_vars, "theta_scale", "scale", default=1.0),
            nfp=int(round(_read_scalar(geom_vars, "nfp", default=1.0))),
            kperp2_bmag=True,
            bessel_bmag_power=0.0,
            source_model="imported-netcdf",
            theta_closed_interval=theta_closed_interval,
        )
    finally:
        root.close()

