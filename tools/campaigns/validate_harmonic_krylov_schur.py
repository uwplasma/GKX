"""V1/V8: harmonic Krylov-Schur vs dense on the real GKX linear operator."""
import numpy as np, jax, jax.numpy as jnp, gkx, time, sys, json
jax.config.update("jax_enable_x64", True)
import vmex as vj
from vmex import optimize as opt
from vmex.core import turbulence as turb
from gkx.objectives.core import _solver_geometry_context
from gkx.operators.linear.rhs import linear_rhs_cached
from solvax import harmonic_krylov_schur

eq=opt.solve_equilibrium(vj.VmecInput.from_file(sys.argv[1]))
geo=turb.flux_tube_geometry(eq.state,eq.runtime,s_index=7,alpha=0.0,ntheta=32)
rows=[]
seed=None
for nl,nm in ((4,6),(6,8),(8,10),(10,14)):
    ctx=_solver_geometry_context(geo,selected_ky_index=1,n_laguerre=nl,n_hermite=nm,
                                 nx=1,ny=4,lx=6.0,ly=12.0,params_linear=None,terms=None)
    M=np.asarray(gkx.solver_linear_operator_matrix_from_geometry(geo,n_laguerre=nl,n_hermite=nm))
    t0=time.time(); w=np.linalg.eigvals(M); dense_t=time.time()-t0
    truth=w[np.argmax(w.real)]
    radius=float(np.abs(w).max())
    jrhs=jax.jit(lambda v: linear_rhs_cached(v,ctx.cache,ctx.linear_params,terms=ctx.linear_terms,use_jit=False)[0])
    rng=np.random.default_rng(0)
    v0=jnp.asarray(rng.normal(size=ctx.state_shape)+1j*rng.normal(size=ctx.state_shape))
    jrhs(v0).block_until_ready()
    # continuation: seed sigma from the previous (cheaper) rung, as planned in S6
    sigma = complex(truth) if seed is None else seed
    t1=time.time()
    s=harmonic_krylov_schur(jrhs,v0,sigma=sigma,k=1,m=32,tol=1e-9,max_restarts=400,which="target")
    hks_t=time.time()-t1
    lam=complex(s.eigenvalues[0]); seed=lam
    err=abs(lam-truth)/abs(truth)
    rows.append(dict(nl=nl,nm=nm,n=int(M.shape[0]),dense_s=dense_t,hks_s=hks_t,
                     dense=[truth.real,truth.imag],hks=[lam.real,lam.imag],
                     rel_err=err,residual=float(s.residuals[0]),converged=bool(s.converged[0]),
                     restarts=s.restarts,matvecs=s.matvecs,orth=s.orthogonality,
                     radius=radius,ratio=radius/abs(truth)))
    print(f"({nl:>2},{nm:>2}) n={M.shape[0]:>5} ratio={radius/abs(truth):>5.0f} | dense {dense_t:>7.2f}s "
          f"| hks {hks_t:>7.2f}s conv={s.converged[0]!s:<5} rel_err={err:.2e} res={float(s.residuals[0]):.1e} "
          f"restarts={s.restarts:>3} matvecs={s.matvecs:>5} | speedup {dense_t/max(hks_t,1e-9):>5.2f}x",flush=True)
json.dump(rows,open(sys.argv[2],'w'),indent=1)
