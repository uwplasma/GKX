# from spectraxgk.diagnostics import field_energy_from_E, kinetic_energy_from_Cnt


# def test_energy_helpers_shapes():
#     nt = 10
#     C_nt = jnp.zeros((4, nt), dtype=jnp.complex128)  # N>=3
#     n0, m, vth = 1.0, 2.0, 3.0
#     Wk = kinetic_energy_from_Cnt(C_nt, n0=n0, m=m, vth=vth, L=1.0)
#     assert Wk.shape == (nt,)

#     E_xt = jnp.zeros((8, nt))
#     Wf = field_energy_from_E(E_xt, L=1.0)
#     assert Wf.shape == (nt,)
