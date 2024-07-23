import jax 

def get_conjugate_solver(solver, max_iter, jit_=False, v_map=True):
    """Compute the gradient of the legendre transform of the function parameterized by state_u"""
    def conjugate_solver(y, x_init, state_u):
        predict_u = state_u.apply_fn
        params_u = state_u.params
        u_point = lambda x: predict_u({"params": params_u}, x)

        state = solver.solve(
            u_point,
            y,
            x_init=x_init
        )
        has_converged = 1 - (state.num_iter > max_iter - 1)
        return state.grad, has_converged
    if v_map:
        conjugate_solver = jax.vmap(conjugate_solver, (0, 0, None), 0)
    if jit_:
        conjugate_solver = jax.jit(conjugate_solver)
    return conjugate_solver