from dataclasses import dataclass
from collections import namedtuple
from typing import Optional
import jax
from jax import lax
import jax.numpy as jnp
import functools
import optax


ConjStatus = namedtuple("ConjStatus", "val grad num_iter val_hist grad_norm")
@dataclass
class SolverAdam:
# taken from https://github.com/facebookresearch/w2ot/blob/main/w2ot/conjugate_solver.py
    max_iter: int
    gtol: float

    adam_kwargs: Optional[dict] = None
    lr_schedule_kwargs: Optional[dict] = None

    def __post_init__(self):
        if self.adam_kwargs is None:
            self.adam_kwargs = {'b1': 0.9, 'b2': 0.999}
        if self.lr_schedule_kwargs is None:
            self.lr_schedule_kwargs = {
                'init_value': 0.1,
                'decay_steps': self.max_iter,
                'alpha': 1e-4,
            }

    def conj_min_obj(self, x, f, y):
        # f^*(y) = -inf_x f(x) - y^T x
        return f(x) - x.dot(y)


    def solve(self, f, y, x_init=None, track_hist=False, return_grad_norm=False):
        assert y.ndim == 1

        if x_init is None:
            x_init = y

        conj_min_obj = functools.partial(self.conj_min_obj, y=y, f=f)

        lr_schedule = optax.cosine_decay_schedule(
            **self.lr_schedule_kwargs)
        adam = optax.adam(learning_rate=lr_schedule, **self.adam_kwargs)
        opt_state = adam.init(x_init)

        obj, grad = jax.value_and_grad(conj_min_obj)(x_init)
        if track_hist:
            f_hist = jnp.zeros((self.max_iter+1))
            f_hist = f_hist.at[0].set(obj)
        else:
            f_hist = None

        LoopState = namedtuple("LoopState", "i x grad opt_state obj f_hist")
        init_state = LoopState(0, x_init, grad, opt_state, obj, f_hist)

        def cond_fun(state):
            return (state.i < self.max_iter) & \
                (jnp.linalg.norm(state.grad, ord=jnp.inf) > self.gtol)

        def body_fun(state):
            updates, new_opt_state = adam.update(state.grad, state.opt_state, state.x)
            x_new = optax.apply_updates(state.x, updates)
            obj_new, grad_new = jax.value_and_grad(conj_min_obj)(x_new)
            if track_hist:
                next_f_hist = state.f_hist.at[state.i+1].set(obj_new)
            else:
                next_f_hist = None
            return LoopState(
                state.i+1, x_new, grad_new, new_opt_state, obj_new, next_f_hist)

        state = lax.while_loop(cond_fun, body_fun, init_state)

        obj = state.obj
        x = state.x
        n_iter = state.i
        val_hist = state.f_hist if track_hist else None
        grad_norm = jnp.linalg.norm(jax.grad(conj_min_obj)(x), ord=jnp.inf) if return_grad_norm else None

        return ConjStatus(
            val=-obj, grad=x, num_iter=n_iter,
            val_hist=val_hist, grad_norm=grad_norm,
        )
    
