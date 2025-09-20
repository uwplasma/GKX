# requirements: pip install "jax[cpu]" qujax optax
import jax
import jax.numpy as jnp
import optax
import qujax


# --- Same FH pieces as above (copied here for self-containment) ---
def hermite_streaming_tridiag(k: float, N: int, vth: float) -> jnp.ndarray:
    n = jnp.arange(N-1)
    off = k * vth * jnp.sqrt((n + 1) / 2.0)
    H = jnp.zeros((N, N), dtype=jnp.float64)
    H = H.at[jnp.arange(N-1), jnp.arange(1, N)].set(off)
    H = H.at[jnp.arange(1, N), jnp.arange(N-1)].set(off)
    return H

def hermite_field_rank1(k: float, N: int, omega_p: float) -> jnp.ndarray:
    H = jnp.zeros((N, N), dtype=jnp.float64)
    def nz(H):
        val = (omega_p**2 / k) * jnp.sqrt(0.5)
        H = H.at[1,0].set(val)
        H = H.at[0,1].set(val)
        return H
    return jax.lax.cond(jnp.abs(k) > 0.0, nz, lambda H: H, H)

def build_parts(k: float, N: int, vth: float, omega_p: float):
    Hs = hermite_streaming_tridiag(k, N, vth)
    # normalized field piece so H = Hs + theta * (omega_p^2/k) * Hf
    Hf = hermite_field_rank1(k, N, omega_p) / ((omega_p**2)/k + 1e-30)
    return Hs, Hf

# --- Matrix exponential via eigen-decomposition (Hermitian H) ---
def expm_hermitian(H: jnp.ndarray, dt: float) -> jnp.ndarray:
    # H is real symmetric -> use eigh, then unitary = V diag(exp(-i λ dt)) V^T
    w, V = jnp.linalg.eigh(H.astype(jnp.float64))
    phase = jnp.exp((-1j) * w * dt).astype(jnp.complex128)
    return (V @ (phase[jnp.newaxis, :] * jnp.conj(V).T)).astype(jnp.complex128)

# --- Embed small-N unitary into 2^Q space as block-diag [U, I] ---
def pad_unitary(U_small: jnp.ndarray, Q: int) -> jnp.ndarray:
    dim_big = 2**Q
    N = U_small.shape[0]
    U = jnp.eye(dim_big, dtype=jnp.complex128)
    U = U.at[:N, :N].set(U_small)
    return U

# --- Build a single-step custom gate: parameters -> full-width unitary tensor ---
def make_step_gate(k, N, vth, omega_p, dt, Q):
    Hs, Hf = build_parts(k, N, vth, omega_p)
    def gate_func(params: jnp.ndarray) -> jnp.ndarray:
        # params is a length-1 vector: theta
        theta = params[0]
        H = Hs + theta * ((omega_p**2)/k) * Hf
        U_small = expm_hermitian(H, dt)            # (N,N)
        U = pad_unitary(U_small, Q)                # (2^Q, 2^Q)
        # qujax expects a unitary array in (2,2,...,2,2) tensor shape
        dim = U.shape[0]
        assert dim == 2**Q
        return U.reshape((2,)*Q*2)                 # (2,...,2)x(2,...,2)
    return gate_func

# --- Build a T-step circuit with one shared parameter theta across all steps ---
def build_qujax_time_evolution(N=64, k=0.5, vth=1.0, omega_p=1.0, T=200, dt=0.3):
    Q = int(jnp.ceil(jnp.log2(N)))
    # One step gate acting on all qubits:
    gate = make_step_gate(k, N, vth, omega_p, dt, Q)
    gate_seq = [gate] * T
    qubit_inds_seq = [list(range(Q))] * T
    # All gates share the same parameter index 0 -> one scalar theta
    param_inds_seq = [[0]] * T
    param_to_st = qujax.get_params_to_statetensor_func(gate_seq, qubit_inds_seq, param_inds_seq, n_qubits=Q)
    return param_to_st, Q

# --- Utilities to prepare/measure the FH state inside 2^Q space ---
def fh_state_to_statetensor(vec_fh: jnp.ndarray, Q: int) -> jnp.ndarray:
    v = vec_fh
    dim = 2**Q
    pad = jnp.zeros((dim - v.shape[0],), dtype=v.dtype)
    v_big = jnp.concatenate([v, pad], axis=0)
    return v_big.reshape((2,)*Q)

def statetensor_to_fh_coeff0(st: jnp.ndarray) -> jnp.complex128:
    # amplitude of basis |0...0> equals c_{n=0}
    return st.reshape((-1,))[0].astype(jnp.complex128)

# --- End-to-end loss: fit theta to observed E_k(t_j) ---
def build_loss_function(N=64, k=0.5, vth=1.0, omega_p=1.0, T=200, dt=0.3, c0_amp=1e-3):
    param_to_st, Q = build_qujax_time_evolution(N, k, vth, omega_p, T, dt)

    # initial FH state: c0=small, others 0
    c0 = jnp.zeros((N,), dtype=jnp.complex128).at[0].set(c0_amp)
    st_in = fh_state_to_statetensor(c0, Q)

    # Observable mapping c0 -> E_k
    alpha = 1j * (omega_p**2 / k)

    # times (for plotting/consistency only)
    ts = jnp.arange(T) * dt

    def forward(theta: jnp.ndarray):
        # single scalar parameter shared across T steps
        st_out = param_to_st(theta, statetensor_in=st_in)   # (2,)*Q
        c0_out = statetensor_to_fh_coeff0(st_out)
        E = alpha * c0_out
        return ts, E

    def loss(theta: jnp.ndarray, E_obs: jnp.ndarray):
        ts, E = forward(theta)
        return jnp.mean(jnp.abs(E - E_obs)**2)

    return loss, forward

# --- Demo: self-consistency where E_obs is generated at theta*=1 ---
if __name__ == "__main__":
    N, k, vth, omega_p, T, dt = 64, 0.5, 1.0, 1.0, 200, 0.3
    loss, forward = build_loss_function(N, k, vth, omega_p, T, dt)
    theta_star = jnp.array([1.0])
    ts, E_star = forward(theta_star)

    # Initialize theta and optimize with optax
    theta = jnp.array([0.8])
    opt = optax.adam(1e-2)
    opt_state = opt.init(theta)

    @jax.jit
    def step(theta, opt_state):
        val, grad = jax.value_and_grad(lambda th: loss(th, E_star))(theta)
        updates, opt_state = opt.update(grad, opt_state, theta)
        theta = optax.apply_updates(theta, updates)
        return theta, opt_state, val

    for it in range(200):
        theta, opt_state, val = step(theta, opt_state)
        if (it+1) % 50 == 0:
            print(f"iter {it+1:3d}  loss={val:.3e}  theta={float(theta):.4f}")

    print("Recovered theta ≈", float(theta))
