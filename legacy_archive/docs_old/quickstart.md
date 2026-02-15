# Quickstart

This page shows the minimum workflow to run a multispecies simulation.

## Minimal run

```python
import numpy as np
from spectraxgk._simulation_multispecies import simulation_multispecies

species = [
    dict(name="ion", q=+1.0, T=1.0, n0=1.0, rho=0.0, vth=1.0, nu=0.0, Upar=0.0),
    dict(name="e-",  q=-1.0, T=1.0, n0=1.0, rho=0.0, vth=1.0, nu=0.0, Upar=0.0),
]

out = simulation_multispecies(
    input_parameters=dict(
        Lz=2*np.pi,
        t_max=10.0,
        enable_streaming=True,
        enable_nonlinear=False,
        enable_collisions=False,
        enforce_reality=True,
        nx0=0, ny0=0, nz0=1,
        pert_amp=1e-6,
        lambda_D=0.2,
        species=species,
    ),
    Nx=1, Ny=1, Nz=128,
    Nl=1, Nh=64,
    timesteps=400,
    dt=0.025,
    adaptive_time_step=False,
    save="diagnostics",
    save_every=1,
    progress=True,
)
print(out["phi_rms"][-1])
Turn on collisions
Set enable_collisions=True and provide nu per species:

python
Copy code
species = [
    dict(name="ion", q=+1.0, T=1.0, n0=1.0, rho=0.0, vth=1.0, nu=0.1, Upar=0.0),
    dict(name="e-",  q=-1.0, T=1.0, n0=1.0, rho=0.0, vth=1.0, nu=0.5, Upar=0.0),
]
You should see W_free decay compared to the collisionless run.

Control which species get the initial perturbation
Use perturb_species:

list of names: ["e-"]

list of indices: [1]

dict of weights: {"e-": 1.0, "ion": 0.0}

Example:

python
Copy code
input_parameters=dict(
    ...,
    perturb_species=["e-"],
)
Enable ∇∥ ln B coupling
python
Copy code
input_parameters=dict(
    ...,
    enable_gradB_parallel=True,
    B_eps=0.1,
    B_mode=1,
)
This uses the slab profile B(z)=1+B_eps*cos(2π*B_mode*z/Lz).

Tips
If you use k_perp=0 modes, consider lambda_D>0 to avoid degeneracy in quasineutrality.

Use x64 when checking conservation properties.