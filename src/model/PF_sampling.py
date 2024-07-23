from typing import (
    Callable,
    Optional,
)
import jax
import jax.numpy as jnp
from src.model.PF_learning import PFTraining



class PFSampling:
    """Langevin Monte Carlo Neural Polar Factorization (LMC-NPF) algorithm

    It allows to sample from a density e^{-f} by adapting the LMC algorithm and learning the NPF of \nabla f.

    Args:
        f: Function associated to the density e^{-f} that we aim to sample from.
        grad_f: Gradient of the function f.
        rng: Random key used for seeding for updating the particles.
        nb_particles: Number of particles used in the sampling algorithm.
        warming_samples: Number of samples used to warm the NNs associated to the NPF of f.
        iter_warming: Iterator used for the warm up of the NNs.
        tau: Step size used in the LMC algorithms
        num_step: Total number of steps allowed for the sampling algorithm.
        particles_update_fn: Function used to update the particles after each step.
        pf_training: Training class that update the NPF state with the discovery of new particles.
        F_stochastic: Whether the gradient is a stochastic function
    """
    def __init__(self, 
                 grad_f: Callable, 
                 rng: jax.Array,
                 nb_particles: int = 128, 
                 warming_samples: jnp.ndarray = None, 
                 iter_warming: Optional[Callable] = None,
                 num_step: int = 10000,
                 particles_update_fn: Optional[Callable] = None,
                 pf_training: PFTraining = None,
                 F_stochastic: bool = False
                 ):
        
        self.grad_f = jax.jit(grad_f)
        self.nb_particles = nb_particles
        self.warming_samples = warming_samples
        self.iter_warming = iter_warming
        self.num_step = num_step
        self.particles_update_fn = particles_update_fn
        self.rng = rng
        self.pf_training = pf_training
        self.F_stochastic = F_stochastic

    def __call__(self, init_particles, num_warm_u_iter, num_warm_i_iter):
        """Warm up and Training"""

        # warmup with warming samples
        if self.warming_samples is not None:
            # create warming particles
            warming_particles, warming_particles_F = self.warming_samples
            
            # store them in an array that retains all particles that have been seen
            data = (jnp.array(warming_particles), jnp.array(warming_particles_F))
            # warm-up of u and i
            if num_warm_u_iter > 0:
                _ = self.pf_training.warm_up_conj_u(data_warm = data)
                _ = self.pf_training.train_u(data, num_warm_u_iter, out_iter=0)
            if num_warm_i_iter > 0:
                _ = self.pf_training.train_i_m(data, num_warm_i_iter, out_iter=0)

        # warm up with iterator
        elif self.iter_warming is not None:
            self.rng, rng_iter_init = jax.random.split(self.rng)
            rng = rng_iter_init
            _ = self.pf_training.warm_up_conj_u(iter = self.iter_warming, rng_init = rng_iter_init)
            # warm up of u
            for wi in range(num_warm_u_iter):
                rng, rng_b = jax.random.split(rng, 2)
                data = self.iter_warming.generate_samples(rng_b, self.pf_training.batch_size_u)
                _ = self.pf_training.train_u(data, 1, out_iter=wi)

            # warm up of i
            rng = rng_iter_init
            for wi in range(num_warm_i_iter):
                rng, rng_b = jax.random.split(rng, 2)
                data = self.iter_warming.generate_samples(rng_b, self.pf_training.batch_size_u)
                _ = self.pf_training.train_i_m(data, 1, out_iter=num_warm_u_iter+wi)

        # sampling and learning algorithm
        _ = self.sample_and_learn(init_particles)
        return self.pf_training.pf_state.state_u, self.pf_training.pf_state.state_conj_u, self.pf_training.pf_state.state_i

    def sample_and_learn(self, init_particles):
        """Sampling function that also learns the NPF"""
        #initialize particles 
        rng = self.rng
        # initialize the particles
        particles_x = init_particles    
        particles_y = particles_x
        
        for step in range(self.num_step):
            if step % 1000 == 0:
                print("Sampling ", step, self.num_step)
            # create rng keys
            (rng, rng_part) = jax.random.split(rng, 2)

            # update the position of the particles 
            particles_y, particles_x = self.particles_update_fn(rng_part, particles_y, particles_x, self.pf_training.pf_state.state_u, self.pf_training.pf_state.state_i, self.pf_training.pf_state.state_conj_u, step)

            # training of the networks
            if self.F_stochastic:
                rng, rng_stoch = jax.random.split(rng, 2)
                grad_f_part = self.grad_f(particles_x, rng_stoch)
            else:
                grad_f_part = self.grad_f(particles_x)
            training_input, training_target = particles_x, grad_f_part
            self.pf_training.train_step_pf((training_input, training_target), previous_iter=step * (self.pf_training.num_inner_iter_u+self.pf_training.num_inner_iter_i))

        self.rng = rng
        return 0

