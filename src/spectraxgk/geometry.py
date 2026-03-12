"""Analytic flux-tube geometry for the Cyclone base case."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from spectraxgk.config import GeometryConfig, GridConfig


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SAlphaGeometry:
    """Simple s-alpha geometry with circular concentric flux surfaces."""

    q: float
    s_hat: float
    epsilon: float
    R0: float = 1.0
    B0: float = 1.0
    alpha: float = 0.0
    drift_scale: float = 1.0
    kperp2_bmag: bool = True
    bessel_bmag_power: float = 0.0

    def tree_flatten(self):
        children = (
            self.q,
            self.s_hat,
            self.epsilon,
            self.R0,
            self.B0,
            self.alpha,
            self.drift_scale,
            self.kperp2_bmag,
            self.bessel_bmag_power,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @staticmethod
    def from_config(cfg: GeometryConfig) -> "SAlphaGeometry":
        return SAlphaGeometry(
            q=cfg.q,
            s_hat=cfg.s_hat,
            epsilon=cfg.epsilon,
            R0=cfg.R0,
            B0=cfg.B0,
            alpha=cfg.alpha,
            drift_scale=cfg.drift_scale,
            kperp2_bmag=cfg.kperp2_bmag,
            bessel_bmag_power=cfg.bessel_bmag_power,
        )

    def kx_effective(self, kx0: jnp.ndarray, ky: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        """Field-aligned kx(theta) with s-alpha shear shift."""

        shear = self.s_hat * theta - self.alpha * jnp.sin(theta)
        return kx0 - shear * ky
    def bmag(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Magnetic field strength for circular s-alpha geometry."""

        return 1.0 / (1.0 + self.epsilon * jnp.cos(theta))

    def gradpar(self) -> float:
        """Parallel gradient factor for s-alpha geometry (constant for equal-arc)."""

        return float(abs(1.0 / (self.q * self.R0)))

    def bgrad(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Magnetic field gradient term used in mirror force."""

        bmag = self.bmag(theta)
        return self.gradpar() * self.epsilon * jnp.sin(theta) * bmag

    def metric_coeffs(self, theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Metric coefficients (gds2, gds21, gds22) for s-alpha geometry."""

        shear = self.s_hat * theta - self.alpha * jnp.sin(theta)
        gds2 = 1.0 + shear * shear
        gds21 = -self.s_hat * shear
        gds22 = jnp.asarray(self.s_hat) * jnp.asarray(self.s_hat)
        return gds2, gds21, gds22

    def k_perp2(self, kx0: jnp.ndarray, ky: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        """Perpendicular wave-number squared for s-alpha geometry."""

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

    def drift_coeffs(
        self, theta: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Curvature and grad-B drift coefficients for s-alpha geometry."""

        shear = self.s_hat * theta - self.alpha * jnp.sin(theta)
        base = jnp.cos(theta) + shear * jnp.sin(theta)
        scale = jnp.asarray(self.drift_scale)
        cv = scale * base / self.R0
        gb = cv
        cv0 = scale * (-self.s_hat * jnp.sin(theta)) / self.R0
        gb0 = cv0
        return cv, gb, cv0, gb0

    def drift_components(
        self, kx: jnp.ndarray, ky: jnp.ndarray, theta: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return cv_d and gb_d drift components in (ky, kx, theta)."""

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

    def omega_d(self, kx: jnp.ndarray, ky: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        """Magnetic drift frequency for s-alpha geometry."""

        cv_d, gb_d = self.drift_components(kx, ky, theta)
        return cv_d + gb_d


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SlabGeometry:
    """GX slab geometry contract."""

    s_hat: float = 0.0
    z0: float | None = None
    q: float = 1.0
    epsilon: float = 0.0
    R0: float = 1.0
    B0: float = 1.0
    alpha: float = 0.0
    drift_scale: float = 0.0
    kperp2_bmag: bool = True
    bessel_bmag_power: float = 0.0
    zero_shat: bool = False

    def tree_flatten(self):
        children = (
            self.s_hat,
            self.q,
            self.epsilon,
            self.R0,
            self.B0,
            self.alpha,
            self.drift_scale,
            self.kperp2_bmag,
            self.bessel_bmag_power,
        )
        return children, {"z0": self.z0, "zero_shat": self.zero_shat}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, z0=aux_data["z0"], zero_shat=aux_data["zero_shat"])

    @staticmethod
    def from_config(cfg: GeometryConfig) -> "SlabGeometry":
        shat = 0.0 if bool(cfg.zero_shat) else float(cfg.s_hat)
        return SlabGeometry(
            s_hat=shat,
            z0=cfg.z0,
            q=1.0,
            epsilon=0.0,
            R0=cfg.R0,
            B0=cfg.B0,
            alpha=0.0,
            drift_scale=0.0,
            kperp2_bmag=cfg.kperp2_bmag,
            bessel_bmag_power=cfg.bessel_bmag_power,
            zero_shat=bool(cfg.zero_shat),
        )

    def kx_effective(self, kx0: jnp.ndarray, ky: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        shear = jnp.asarray(self.s_hat) * theta
        return kx0 - shear * ky

    def bmag(self, theta: jnp.ndarray) -> jnp.ndarray:
        return jnp.ones_like(jnp.asarray(theta))

    def gradpar(self) -> float:
        if self.z0 is not None and float(self.z0) > 0.0:
            return float(1.0 / float(self.z0))
        return 1.0

    def bgrad(self, theta: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(jnp.asarray(theta))

    def metric_coeffs(self, theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        theta_arr = jnp.asarray(theta)
        shear = jnp.asarray(self.s_hat) * theta_arr
        gds2 = 1.0 + shear * shear
        gds21 = -jnp.asarray(self.s_hat) * shear
        if float(self.s_hat) == 0.0:
            gds22 = jnp.ones_like(theta_arr)
        else:
            gds22 = jnp.full_like(theta_arr, float(self.s_hat) * float(self.s_hat))
        return gds2, gds21, gds22

    def k_perp2(self, kx0: jnp.ndarray, ky: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        gds2, gds21, gds22 = self.metric_coeffs(theta)
        s_hat = jnp.asarray(self.s_hat)
        s_hat_safe = jnp.where(s_hat == 0.0, 1.0, s_hat)
        kx_hat = kx0 / s_hat_safe
        kx_hat = jnp.where(s_hat == 0.0, kx0, kx_hat)
        kperp2 = ky * (ky * gds2 + 2.0 * kx_hat * gds21) + (kx_hat * kx_hat) * gds22
        return kperp2

    def drift_coeffs(
        self, theta: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        theta_arr = jnp.asarray(theta)
        zeros = jnp.zeros_like(theta_arr)
        return zeros, zeros, zeros, zeros

    def drift_components(
        self, kx: jnp.ndarray, ky: jnp.ndarray, theta: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        kx0 = jnp.asarray(kx)
        ky0 = jnp.asarray(ky)
        theta0 = jnp.asarray(theta)
        zeros = jnp.zeros((ky0.shape[0], kx0.shape[0], theta0.shape[0]), dtype=theta0.dtype)
        return zeros, zeros

    def omega_d(self, kx: jnp.ndarray, ky: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        kx0 = jnp.asarray(kx)
        ky0 = jnp.asarray(ky)
        theta0 = jnp.asarray(theta)
        return jnp.zeros((ky0.shape[0], kx0.shape[0], theta0.shape[0]), dtype=theta0.dtype)


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
            raise ValueError("theta must have the same last dimension as the sampled geometry grid")
        if isinstance(theta_arr, jax.core.Tracer):
            return theta_arr
        theta_line = theta_arr if theta_arr.ndim == 1 else theta_arr.reshape(-1, theta_arr.shape[-1])[0]
        if isinstance(theta_line, jax.core.Tracer):
            return theta_arr
        if not np.allclose(np.asarray(theta_line), np.asarray(self.theta), rtol=1.0e-6, atol=1.0e-6):
            raise ValueError("theta does not match the sampled geometry grid")
        return theta_arr

    def _broadcast_profile(self, theta: jnp.ndarray, profile: jnp.ndarray) -> jnp.ndarray:
        theta_arr = self._theta_matches(theta)
        if theta_arr.ndim == 1:
            return profile
        shape = (1,) * (theta_arr.ndim - 1) + (profile.shape[0],)
        return jnp.broadcast_to(profile.reshape(shape), theta_arr.shape)

    def trim_terminal_theta_point(self) -> FluxTubeGeometryData:
        """Return a copy without the terminal theta sample.

        GX `*.eik.nc` files commonly store a closed theta interval, while the
        spectral solver uses the matching open interval with the terminal point
        excluded. Trimming keeps the imported coefficients aligned with the
        runtime grid without changing the physical extent.
        """

        if self.theta.shape[0] < 2:
            raise ValueError("Cannot trim the terminal point from a geometry grid with fewer than two samples")
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

    def metric_coeffs(self, theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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

    def k_perp2(self, kx0: jnp.ndarray, ky: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
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

    def omega_d(self, kx: jnp.ndarray, ky: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        cv_d, gb_d = self.drift_components(kx, ky, theta)
        return cv_d + gb_d


def sample_flux_tube_geometry(geom: SAlphaGeometry | SlabGeometry, theta: jnp.ndarray) -> FluxTubeGeometryData:
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
        raise ValueError("Periodic spectral derivatives require a one-dimensional profile")
    if values.size < 2:
        return np.zeros_like(values)
    k = 2.0 * np.pi * np.fft.fftfreq(values.size, d=spacing)
    deriv_hat = 1j * k * np.fft.fft(values)
    return np.fft.ifft(deriv_hat).real.astype(values.dtype, copy=False)


def _gx_bgrad_from_bmag(theta: np.ndarray, bmag: np.ndarray, gradpar_val: float, *, closed: bool) -> np.ndarray:
    """Reconstruct GX's mirror term from ``bmag`` on the solver theta grid."""

    if theta.ndim != 1 or bmag.ndim != 1:
        raise ValueError("GX bgrad reconstruction expects one-dimensional theta and bmag profiles")
    if theta.shape != bmag.shape:
        raise ValueError("theta and bmag must have the same shape for GX bgrad reconstruction")
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


def load_gx_geometry_netcdf(path: str | Path) -> FluxTubeGeometryData:
    """Load sampled geometry from a GX-style NetCDF file."""

    try:
        from netCDF4 import Dataset
    except ImportError as exc:  # pragma: no cover - optional import
        raise ImportError("netCDF4 is required to load GX geometry NetCDF files") from exc

    def _read_scalar(variables, *names: str, default: float | None = None) -> float:
        for name in names:
            if name in variables:
                arr = np.asarray(variables[name][:], dtype=float)
                if arr.ndim == 0:
                    return float(arr)
                if arr.ndim == 1 and np.allclose(arr, arr[0]):
                    return float(arr[0])
                raise ValueError(f"GX geometry variable '{name}' must be scalar or constant on theta")
        if default is None:
            raise KeyError(names[0])
        return float(default)

    def _read_profile(variables, *names: str) -> np.ndarray:
        for name in names:
            if name in variables:
                arr = np.asarray(variables[name][:], dtype=float)
                if arr.ndim != 1:
                    raise ValueError(f"GX geometry variable '{name}' must be one-dimensional on theta")
                return arr
        raise KeyError(names[0])

    def _infer_root_theta_closed_interval(theta: np.ndarray, variables) -> bool:
        """Infer whether a root-level GX ``*.eik.nc`` file includes a terminal theta endpoint.

        VMEC-style ``*.eik.nc`` files often include the periodic terminal point, while
        GX's Miller helper writes an already-open theta grid. Root-level files therefore
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
        is_grouped_gx_output = "Geometry" in root.groups and "Grids" in root.groups
        if is_grouped_gx_output:
            geom_vars = root.groups["Geometry"].variables
            grid_vars = root.groups["Grids"].variables
            theta = _read_profile(grid_vars, "theta")
            theta_closed_interval = False
        else:
            geom_vars = root.variables
            grid_vars = root.variables
            theta = _read_profile(root.variables, "theta")
            theta_closed_interval = _infer_root_theta_closed_interval(np.asarray(theta, dtype=float), root.variables)

        gradpar_val = _read_scalar(geom_vars, "gradpar")
        bmag = _read_profile(geom_vars, "bmag")
        drhodpsi = _read_scalar(geom_vars, "drhodpsi", default=1.0)
        if is_grouped_gx_output and "bgrad" in geom_vars:
            bgrad = _read_profile(geom_vars, "bgrad")
        else:
            bgrad = _gx_bgrad_from_bmag(
                np.asarray(theta, dtype=float),
                np.asarray(bmag, dtype=float),
                gradpar_val,
                closed=theta_closed_interval,
            )
        if is_grouped_gx_output:
            cvdrift = _read_profile(geom_vars, "cvdrift")
            gbdrift = _read_profile(geom_vars, "gbdrift")
            cvdrift0 = _read_profile(geom_vars, "cvdrift0")
            gbdrift0 = _read_profile(geom_vars, "gbdrift0")
            jacobian = _read_profile(geom_vars, "jacobian", "jacob")
        else:
            # Root-level VMEC ``*.eik.nc`` files carry the pre-GX drift
            # normalization and a Jacobian that GX replaces at load time.
            cvdrift = 0.5 * _read_profile(geom_vars, "cvdrift")
            gbdrift = 0.5 * _read_profile(geom_vars, "gbdrift")
            cvdrift0 = 0.5 * _read_profile(geom_vars, "cvdrift0")
            gbdrift0 = 0.5 * _read_profile(geom_vars, "gbdrift0")
            jacobian = 1.0 / np.abs(float(drhodpsi) * float(gradpar_val) * np.asarray(bmag, dtype=float))
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
            source_model="gx-netcdf",
            theta_closed_interval=theta_closed_interval,
        )
    finally:
        root.close()


FluxTubeGeometryLike = SAlphaGeometry | SlabGeometry | FluxTubeGeometryData


def build_flux_tube_geometry(cfg: GeometryConfig) -> FluxTubeGeometryLike:
    """Build an analytic or imported flux-tube geometry from config."""

    model = str(cfg.model).strip().lower().replace("_", "-")
    if model in {"s-alpha", "salpha", "analytic"}:
        return SAlphaGeometry.from_config(cfg)
    if model in {"slab"}:
        return SlabGeometry.from_config(cfg)
    if model in {"gx-netcdf", "gx-nc", "netcdf", "nc", "gx-eik", "eik", "vmec-eik", "desc-eik"}:
        if cfg.geometry_file is None:
            raise ValueError("geometry.geometry_file must be set for imported NetCDF/eik geometry")
        return load_gx_geometry_netcdf(cfg.geometry_file)
    raise ValueError(
        "geometry.model must be one of "
        "{'s-alpha', 'slab', 'gx-netcdf', 'gx-eik', 'vmec-eik', 'desc-eik'}"
    )


def ensure_flux_tube_geometry_data(
    geom: FluxTubeGeometryLike,
    theta: jnp.ndarray,
) -> FluxTubeGeometryData:
    """Return sampled geometry data for analytic or pre-sampled inputs."""

    if isinstance(geom, FluxTubeGeometryData):
        try:
            geom._theta_matches(theta)
            return geom
        except ValueError as exc:
            theta_arr = jnp.asarray(theta)
            if geom.theta.shape[0] == theta_arr.shape[-1] + 1:
                trimmed = geom.trim_terminal_theta_point()
                trimmed._theta_matches(theta)
                return trimmed
            raise exc
    return sample_flux_tube_geometry(geom, theta)


def gx_twist_shift_params(
    geom: FluxTubeGeometryLike,
    grid: GridConfig,
) -> tuple[int, float]:
    """Return (jtwist, x0) following GX twist-and-shift defaults."""

    y0 = float(grid.y0) if grid.y0 is not None else float(grid.Ly) / (2.0 * jnp.pi)
    if isinstance(geom, FluxTubeGeometryData):
        gds21_val = float(np.asarray(geom.gds21_profile[0]))
        gds22_val = float(np.asarray(geom.gds22_profile[0]))
        shat = float(geom.s_hat)
    else:
        if grid.ntheta is not None:
            if grid.zp is not None:
                zp = int(grid.zp)
            elif grid.nperiod is not None:
                zp = 2 * int(grid.nperiod) - 1
            else:
                zp = 1
            theta_min = -jnp.pi * float(zp)
        else:
            theta_min = float(grid.z_min)
        _gds2, gds21, gds22 = geom.metric_coeffs(jnp.asarray([theta_min]))
        gds21_val = float(gds21[0])
        gds22_val = float(gds22)
        shat = float(geom.s_hat)
    twist_shift_geo_fac = 2.0 * shat * gds21_val / gds22_val if gds22_val != 0.0 else 0.0
    if grid.jtwist is None:
        jtwist = int(round(twist_shift_geo_fac))
        if jtwist == 0:
            jtwist = 1
    else:
        jtwist = int(grid.jtwist)
        if jtwist == 0:
            jtwist = 1
    if twist_shift_geo_fac == 0.0:
        x0 = y0
    else:
        x0 = y0 * abs(jtwist) / abs(twist_shift_geo_fac)
    return jtwist, x0


def apply_gx_geometry_grid_defaults(
    geom: FluxTubeGeometryLike,
    grid: GridConfig,
) -> GridConfig:
    """Apply GX-aligned grid defaults implied by the selected geometry."""

    grid_out = grid
    if isinstance(geom, FluxTubeGeometryData):
        theta = np.asarray(geom.theta, dtype=float)
        if theta.ndim != 1 or theta.size < 2:
            raise ValueError("Imported GX geometry theta grid must be one-dimensional with at least two points")
        if geom.theta_closed_interval:
            nz = int(theta.size - 1)
            z_min = float(theta[0])
            z_max = float(theta[-1])
        else:
            spacing = float(theta[1] - theta[0])
            nz = int(theta.size)
            z_min = float(theta[0])
            z_max = float(theta[-1] + spacing)
        grid_out = replace(
            grid_out,
            Nz=nz,
            z_min=z_min,
            z_max=z_max,
            ntheta=None,
            nperiod=None,
            zp=None,
        )
        if float(grid_out.kxfac) == 1.0:
            grid_out = replace(grid_out, kxfac=float(geom.kxfac))
    boundary = str(grid_out.boundary).lower()
    if boundary in {"linked", "fix aspect"} and not bool(grid_out.non_twist):
        jtwist, x0 = gx_twist_shift_params(geom, grid_out)
        grid_out = replace(grid_out, Lx=2.0 * np.pi * x0, jtwist=jtwist)
    return grid_out
