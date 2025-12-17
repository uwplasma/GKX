import jax.numpy as jnp
from spectraxgk_old.model import LinearGK
from spectraxgk_old.operators import StreamingOperator, LenardBernstein

def make_linear_model(Nn=4, Nm=3, kpar=0.0, vth=1.0, nu=0.2):
    stream = StreamingOperator(Nn=Nn, Nm=Nm, kpar=kpar, vth=vth)
    collide = LenardBernstein(Nn=Nn, Nm=Nm, nu=nu)
    # NEW: terms are the complex-space operators directly
    terms = (stream, collide)
    return LinearGK(stream=stream, collide=collide, terms=terms)

def test_real_split_matches_complex_rhs_zero_streaming():
    # With kpar=0, only collisions act; real split should match complex rhs
    model = make_linear_model(kpar=0.0)
    Nn, Nm = model.stream.Nn, model.stream.Nm

    # complex state with only (n=1,m=0)=1+2j
    C = jnp.zeros((Nn, Nm), dtype=jnp.complex64).at[1, 0].set(1.0 + 2.0j)

    # Complex RHS directly from operators: streaming=0 when kpar=0
    dC_complex = (model.stream(C)*0 + model.collide(C)).reshape(-1)

    # Real split rhs_real: pack -> call -> unpack
    Cr, Ci = jnp.real(C), jnp.imag(C)
    y_real = jnp.concatenate([Cr.reshape(-1), Ci.reshape(-1)], axis=0)
    d_real = model.rhs_real(0.0, y_real, None)
    M = Nn * Nm
    dCr = d_real[:M].reshape(Nn, Nm)
    dCi = d_real[M:].reshape(Nn, Nm)
    dC_from_real = dCr + 1j * dCi

    assert jnp.allclose(dC_from_real.reshape(-1), dC_complex.reshape(-1))
