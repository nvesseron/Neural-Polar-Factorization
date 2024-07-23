import jax

def get_create_training_samples_i(conjugate_solver, apply_grad_conj):
    def create_training_samples_i_flow(input, F_input, state_u, state_conj_u):
        """Create batch to train the flow i. From data (x_i, F(x_i)), it constitutes the dataset (\nabla u^*(F(x_i)), x_i)"""
        input = input
        # compute \nabla u^*(F(input)) using a conjugate solver initialized with predictions of the conjugate NN
        params_conj_u = state_conj_u.params
        grad_conj_u_point = apply_grad_conj(state_conj_u.apply_fn)
        grad_conj_u = jax.vmap(lambda x: grad_conj_u_point({"params": params_conj_u}, x))
        init_conj_u = grad_conj_u(F_input)
        target, _ = conjugate_solver(F_input, init_conj_u, state_u)

        source_i = target
        target_i = input
        return source_i, target_i

    return create_training_samples_i_flow