import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from diffrax import diffeqsolve, ODETerm, SaveAt, Dopri5, PIDController, NoProgressMeter

def initialize_simulation_parameters(user_parameters={}, timesteps=500, dt=0.01):
    # Default parameters
    default_parameters = {
        "t_max": timesteps * dt,  # Total simulation time
        "ode_tolerance": 1e-6,     # ODE solver tolerance
        "dt": dt,                   # Time step size
        "timesteps": timesteps,     # Number of time steps
        "time": jnp.linspace(0, timesteps * dt, timesteps),  # Time array
        "max_Hermite_index": 20,    # Maximum Hermite polynomial index
    }
    
    # Update default parameters with user-provided parameters
    parameters = {**default_parameters, **user_parameters}
    
    return parameters

def ode_system(t, f, args):
    def Hermite_Laguerre_system(f, k, nu, v_e, Nn, index, q):
        dn = jnp.where(index < Nn, 0, 1)
        n = index - Nn * dn
        dfm_dt = ((- 1j * k * jnp.sqrt(2) * (jnp.sqrt((n + 1) / 2) * f[index + 1] * jnp.sign(Nn - n - 1) +
                                            jnp.sqrt(n / 2) * f[index - 1] + v_e[dn] * f[index]) -
                (1j / k) * (q[0] ** 2 * f[0] + f[Nn] * q[1] ** 2) * jnp.where(n == 1, 1, 0))
                - nu * (n * (n - 1) * (n - 2)) / ((Nn - 1) * (Nn - 2) * (Nn - 3)) * f[index])
        return dfm_dt
    k, v_e, nu, m_max, q = args
    dfdt = vmap(Hermite_Laguerre_system, in_axes=(None, None, None, None, None, 0, None))(
        f, k, nu, v_e, (m_max + 1) // 2, jnp.arange(m_max), q
    )


@partial(jit, static_argnames=['timesteps'])
def simulation(input_parameters={}, timesteps=200, dt = 0.01, solver=Dopri5()):

    parameters = initialize_simulation_parameters(input_parameters, timesteps=timesteps, dt=dt)
    ode_system = ode_system()
    y0 = jnp.zeros(parameters["max_Hermite_index"], dtype=jnp.complex128).at[0].set(1)
    k = 0.3
    nu = 2
    v_e = jnp.array([1.0, 0.0])
    q = jnp.array([1, 0])
    args = (k, v_e, nu, len(y0), q)
    sol = diffeqsolve(
        ODETerm(ode_system),
        solver=solver,
        stepsize_controller=PIDController(rtol=parameters["ode_tolerance"], atol=parameters["ode_tolerance"]),
        # stepsize_controller=ConstantStepSize(),
        t0=0,
        t1=parameters["t_max"],
        dt0=dt,
        y0=y0,
        args=args,
        saveat=SaveAt(ts=parameters["time"]),
        max_steps=1000000,
        progress_meter=NoProgressMeter())
    solution = sol.ys
    time = sol.ts
    
    # Output results
    temporary_output = {"solution": solution, "time": time}
    output = {**temporary_output, **parameters}
    # diagnostics(output)
    return output