from typing import (
    Callable,
    Optional,
)
import jax
import jax.numpy as jnp
from  src.model.PF_state import PFState
from src.model.training_steps import get_train_step_conj_u, get_step_fn_regression

class PFTraining: 
    """ Neural Polar Factorization (NPF) algorithm

    It estimates the two factors \nabla u and M of Brenier's polar factorization as well as the generalized inverse of M from samples (x_i, F(x_i))_i.

    Args:
        rng: Random key used for seeding for creating batch from the dataset.
        pf_state: Training state that gather the state of u, the state of the conjugate of u and the state of i.
        create_training_samples_i: Function that create the training batch for the flow i given the dataset (x_i, F(x_i)).
        num_inner_iter_u: Number of inner iterations for u.
        num_inner_iter_i: Number of inner iterations for i.
        num_optim_iter: Number of times the outer loop is run.
        num_warmup_conj_u: Number of warming step to warm the conjugate NN before the all training.
        train_step_u: Training function for u.
        train_step_i: Training function for i.
        batch_size_u: Batch size to train u.
        batch_size_i: Batch size to train i.
    """
    def __init__(self, 
                 rng: jax.Array, 
                 pf_state: PFState,
                 create_training_samples_i: Callable,
                 num_inner_iter_u: int = 1, 
                 num_inner_iter_i: int = 1, 
                 num_optim_iter: int = 10000,
                 num_warmup_u: int = 1000,
                 num_warmup_conj_u: int = 1000, 
                 train_step_u: Optional[Callable] = None, 
                 train_step_i: Optional[Callable] = None, 
                 batch_size_u: int = 1024,
                 batch_size_i: int = 1024,
                 pf_logging = None,
                 ):
        
        self.rng = rng
        self.pf_state = pf_state
        self.num_inner_iter_u = num_inner_iter_u
        self.num_inner_iter_i = num_inner_iter_i
        self.num_optim_iter = num_optim_iter
        self.num_warmup_u = num_warmup_u
        self.num_warmup_conj_u = num_warmup_conj_u
        self.train_step_u = train_step_u
        self.train_step_i = train_step_i
        self.batch_size_u = batch_size_u
        self.batch_size_i = batch_size_i
        self.create_training_samples_i = create_training_samples_i
        self.train_step_conj_u = get_train_step_conj_u()
        self.train_step_m = get_step_fn_regression()
        self.pf_logging = pf_logging


    def __call__(self, iter_optim):
        """Warmup and Training"""
        # Warm up conjugate NN
        _ = self.warm_up_conj_u(iter=iter_optim)

        # Learn NPF
        _ = self.train_pf(iter_optim)
        
        return self.pf_state.state_u, self.pf_state.state_conj_u, self.pf_state.state_i

    def train_u(self, data, nb_iter, out_iter=0, shuffle=True):
        """Training function for u"""
        input, target = data
        input, target = jnp.array(input), jnp.array(target)
        batch_u = {}
        self.rng, rng = jax.random.split(self.rng, 2)

        for step in range(nb_iter):
            rng, rng_batch = jax.random.split(rng, 2)

            # sample a new batch from the data
            if shuffle:
                data_choice = jax.random.choice(rng_batch, jnp.arange(len(input)), shape=(self.batch_size_u,), replace=True, axis=0)
            else:
                data_choice = jnp.arange(len(input))
            batch_u["input"], batch_u["target"] = input[data_choice], target[data_choice]

            # train u
            self.pf_state.state_u, self.pf_state.state_conj_u, data_u = self.train_step_u(self.pf_state.state_u, self.pf_state.state_conj_u, batch_u)
            
            # log data
            if self.pf_logging is not None: 
                self.pf_logging.log(self.pf_state, log_train_stat=data_u, data=(batch_u["input"], batch_u["target"]), step=out_iter+step)


    def train_i_m(self, data, nb_iter, out_iter=0, shuffle=True):
        """Training function for the generalized inverse i"""
        input, target = data
        input, target = jnp.array(input), jnp.array(target)
        batch = {}
        batch_m = {}
        self.rng, rng = jax.random.split(self.rng, 2)

        for step in range(nb_iter):
            rng, rng_batch = jax.random.split(rng, 2)
            # generate batch
            if shuffle:
                data_choice = jax.random.choice(rng_batch, jnp.arange(len(input)), shape=(self.batch_size_u,), replace=True, axis=0)
            else:
                data_choice = jnp.arange(len(input))            
            batch["input"], batch["target"] = self.create_training_samples_i(input[data_choice], target[data_choice], self.pf_state.state_u, self.pf_state.state_conj_u)
                        
            # train i
            self.pf_state.state_i, log_stat = self.train_step_i(rng_batch, self.pf_state.state_i, batch)

            # train m 
            if self.pf_state.state_m is not None:
                batch_m["input"], batch_m["target"] = batch["target"], batch["input"]
                self.pf_state.state_m, stat_m = self.train_step_m(self.pf_state.state_m, batch)
                log_stat.update(stat_m)

            # log data
            if self.pf_logging is not None:
                self.pf_logging.log(self.pf_state, log_train_stat=log_stat, data=(input[data_choice], target[data_choice]), step=out_iter+step)
    
    def train_step_pf(self, data, previous_iter=0):
        """train u and i in outer loop"""
        # train u
        self.train_u(data, self.num_inner_iter_u, out_iter=previous_iter)
        # train i and m when needed
        self.train_i_m(data, self.num_inner_iter_i, out_iter=previous_iter + self.num_inner_iter_u)

    def warm_up_conj_u(self, iter=None, data_warm=None, rng_init=None):
        """Training function for the conjugate NN"""

        if rng_init is not None:
            rng = rng_init
        else:
            self.rng, rng = jax.random.split(self.rng, 2)
                        
        for step in range(self.num_warmup_conj_u):
            rng_data, rng = jax.random.split(rng, 2)
            if iter is not None:
                # generate batch if an iterator is given
                data = iter.generate_samples(rng_data, self.batch_size_u)
                input, target = data
            else:
                # generate batch from data
                input_warm, target_warm  = data_warm
                data_choice = jax.random.choice(rng_data, jnp.arange(len(input_warm)), shape=(self.batch_size_u,), replace=True, axis=0)
                input, target = input_warm[data_choice], target_warm[data_choice]
            
            input, target= jnp.array(input), jnp.array(target)
            
            self.pf_state.state_conj_u, data_conj_u = self.train_step_conj_u(self.pf_state.state_u, self.pf_state.state_conj_u, input)
            
            # log data
            if self.pf_logging is not None: 
                self.pf_logging.log(self.pf_state, log_train_stat=data_conj_u, data=(input, target), step=step)

    
    def train_pf(self, iter):
        """Main Training loop"""
        self.rng, rng = jax.random.split(self.rng, 2)
        print("Training of u")
        for step in range(self.num_inner_iter_u):  
            if step % 1000 ==0:
                print("Step training u ", step, self.num_inner_iter_u)             
            rng_data, rng = jax.random.split(rng, 2)
            # generate data
            data = iter.generate_samples(rng_data, self.batch_size_u)
            # train
            self.train_u(data, nb_iter = 1, out_iter=step)

        print("Training of i")
        for step in range(self.num_inner_iter_i):   
            if step % 1000 ==0:
                print("Step training i ", step, self.num_inner_iter_i)            
            rng_data, rng = jax.random.split(rng, 2)
            # generate data
            data = iter.generate_samples(rng_data, self.batch_size_i)
            # train
            self.train_i_m(data, nb_iter = 1, out_iter=step)

                