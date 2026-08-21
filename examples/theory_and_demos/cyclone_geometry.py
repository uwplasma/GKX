import jax.numpy as jnp

from gkx.config import CycloneBaseCase
from gkx.geometry import SAlphaGeometry
from gkx.core.grid import build_spectral_grid


def main(kx0: float = 0.0, ky: float = 0.3) -> None:
    cfg = CycloneBaseCase()
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)

    theta = grid.z
    kperp2 = geom.k_perp2(jnp.array(kx0), jnp.array(ky), theta)

    print("Cyclone base case k_perp^2")
    print(f"theta range: [{theta[0]:.3f}, {theta[-1]:.3f}]")
    print(f"min={kperp2.min():.6g} max={kperp2.max():.6g}")

    try:
        import matplotlib.pyplot as plt

        plt.plot(theta, kperp2)
        plt.xlabel("theta")
        plt.ylabel("k_perp^2")
        plt.title("Cyclone base case k_perp^2(\u03b8)")
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
