import jax
import jax.numpy as jnp

def get_particles_update(grad_f, conjugate_solver, apply_grad_conj, transport_i, tau_f, tau_u, nb_steps_f=40, nb_steps_u=40, coef_mult_langevin_f=1.0, coef_mult_langevin_u=1.0, F_stochastic=False, num_warm_up=0):
    """Return a function that upddates the position of the particles according to the algorithm LMC-NFP"""
    @jax.jit
    def update_particles_m_LMCus_i(rng, particles_y, particles_x, state_u, state_i, state_conj_u):
        """Update the particles by applying \nabla u^* \circ F, then doing N LMC steps on u and applying the genralized inverse i"""
        rng, rng_i = jax.random.split(rng, 2)
        state_u = state_u
        state_i = state_i
        predict_u = state_u.apply_fn
        params_u = state_u.params
        grad_u_point = jax.grad(predict_u, argnums=1)
        grad_u = jax.vmap(lambda x: grad_u_point({"params": params_u}, x))

        # compute F(x_i)
        if F_stochastic:
            rng, rng_stoch = jax.random.split(rng, 2)
            grad_f_part = grad_f(particles_x, rng_stoch)
        else:
            grad_f_part = grad_f(particles_x)
        

        # compute \nabla u^*(F(x_i)) = \nabla u^*(\nabla f(x_i)) by initializing the conjugate solver with the predictions of the conjugate NN
        params_conj_u = state_conj_u.params
        grad_conj_u_point = apply_grad_conj(state_conj_u.apply_fn)
        grad_conj_u = jax.vmap(lambda x: grad_conj_u_point({"params": params_conj_u}, x))
        init_conj_u = grad_conj_u(grad_f_part)
        particles_y, _ = conjugate_solver(grad_f_part, init_conj_u, state_u)

        # update particles by doing several LMC steps on u
        def body_fun(i, val):
            rng, particles_y = val
            rng, rng_loop = jax.random.split(rng, 2)
            eps_noise_lmc = jax.random.normal(rng_loop, (particles_y.shape))
            particles_y = (particles_y- tau_u * coef_mult_langevin_u * grad_u(particles_y)) + jnp.sqrt(2 * tau_u) * eps_noise_lmc
            return (rng, particles_y)
        lower = 0
        upper = nb_steps_u
        init_val = (rng, particles_y)
        rng, particles_y = jax.lax.fori_loop(lower, upper, body_fun, init_val)
        
        # apply the flow i on the particles
        particles_x = transport_i(state_i, particles_y, rng_i)     
        return particles_y, particles_x


    @jax.jit
    def update_particles_langevin_f(rng, particles_x):
        """Update the particles by doing 1 LMC step on f"""
        rng, rng_eps_noise_lmc = jax.random.split(rng, 2)
        eps_noise_lmc = jax.random.normal(rng_eps_noise_lmc, (particles_x.shape))

        # compute F(x_i)
        if F_stochastic:
            rng, rng_stoch = jax.random.split(rng, 2)
            grad_f_part = grad_f(particles_x, rng_stoch)
        else:
            grad_f_part = grad_f(particles_x)

        # update particles by doing a LMC step on f
        particles_x = particles_x - tau_f * coef_mult_langevin_f * grad_f_part + jnp.sqrt(2 * tau_f) * eps_noise_lmc            
        return particles_x

    
    def update_particles_mixed_LMCf_m_LMCus_i(rng, particles_y, particles_x, state_u, state_i, state_conj_u, iteration):
        if iteration == 0:
            return particles_y, particles_x
        if iteration < num_warm_up:
            # Apply a regular LMC step on f
            particles_x = update_particles_langevin_f(rng, particles_x)
        elif iteration % nb_steps_f == 0:
            # Apply \nabla u^* \circ F - LMC steps on u - apply the generalized inverse i
            particles_y, particles_x = update_particles_m_LMCus_i(rng, particles_y, particles_x, state_u, state_i, state_conj_u)                    
        else:
            # Apply a regular LMC step on f
            particles_x = update_particles_langevin_f(rng, particles_x)
        return particles_y, particles_x

    return update_particles_mixed_LMCf_m_LMCus_i