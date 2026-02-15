import jax.numpy as jnp
from spectraxgk_old.operators import StreamingOperator, LenardBernstein, ElectrostaticDrive


def test_streaming_couple_real_symmetry():
    Nn, Nm = 4, 2
    op = StreamingOperator(Nn=Nn, Nm=Nm, kpar=0.5, vth=1.0)
    # Put unit amplitude in (n=2, m=1); streaming couples to neighbors n=1 and n=3 (same m)
    C = jnp.zeros((Nn, Nm), dtype=jnp.complex64)
    C = C.at[2, 1].set(1.0 + 0.0j)

    Y = op(C)  # complex RHS: -i * a * (upper + lower)

    nz = jnp.abs(Y) > 0
    assert jnp.count_nonzero(nz) == 2
    assert nz[1, 1] and nz[3, 1]


def test_lenard_bernstein_diagonal_damping():
    Nn, Nm = 3, 2
    lb = LenardBernstein(Nn=Nn, Nm=Nm, nu=0.5)
    C = jnp.zeros((Nn, Nm), dtype=jnp.complex64)
    C = C.at[1, 0].set(1.0 + 0.0j)
    dC = lb(C)
    # With alpha=1, beta=2 default => lambda_{1,0}=1
    assert jnp.isclose(dC[1, 0].real, -0.5)
    assert jnp.isclose(dC[1, 0].imag, 0.0)


def test_electrostatic_drive_targets_n1_m0():
    Nn, Nm = 4, 3
    drv = ElectrostaticDrive(Nn=Nn, Nm=Nm, kpar=2.0, coef=3.0)
    C = jnp.zeros((Nn, Nm), dtype=jnp.complex64).at[0, 0].set(2.0 + 4.0j)
    dC = drv(C)

    expected_re = 3.0 * jnp.sqrt(2.0) * 2.0   # 6*sqrt(2)
    expected_im = -3.0 * jnp.sqrt(2.0) * 1.0  # -3*sqrt(2)

    assert jnp.isclose(dC[1, 0].real, expected_re)
    assert jnp.isclose(dC[1, 0].imag, expected_im)
    # Only (1,0) should be nonzero
    assert jnp.count_nonzero(jnp.abs(dC) > 0) == 1
