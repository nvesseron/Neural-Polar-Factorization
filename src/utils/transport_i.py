import jax
import jax.numpy as jnp
import types
from typing import Any,Dict

import diffrax

def get_transport_i(sigma_diff, flow: bool = True, jit_=True):

    diffeqsolve_kwargs: Dict[str, Any] = types.MappingProxyType({})
    diffeqsolve_kwargs = dict(diffeqsolve_kwargs)
    
    def transport_flow(
      state_velocity_field,
      data: jnp.array,
      start_key,
      t0: float = 0.0,
      t1: float = 0.95,
      diffeqsolve_kwargs: Dict[str, Any] = types.MappingProxyType({})
  ) -> diffrax.Solution:
        """Transport data with the learnt map.

        This method pushes-forward the `source` by
        solving the neural ODE parameterized by the
        :attr:`~ott.neural.flows.OTFlowMatching.velocity_field`.

        Args:
        data: Initial condition of the ODE.
        condition: Condition of the input data.
        forward: If `True` integrates forward, otherwise backwards.
        t_0: Starting point of integration.
        t_1: End point of integration.
        diffeqsolve_kwargs: Keyword arguments for the ODE solver.

        Returns:
        The push-forward or pull-back distribution defined by the learnt
        transport plan.

        """
        diffeqsolve_kwargs = dict(diffeqsolve_kwargs)
        keys = jax.random.split(start_key, len(data))
        dim_data = data.shape[1]
        
        def solve_ode(input: jnp.ndarray, cond: jnp.ndarray, key):
            drift = lambda t, x, args: (state_velocity_field.
                    apply_fn({"params": state_velocity_field.params},
                            t=t,
                            x=x,
                            condition=cond) - x)/(1-t)
            diffusion = lambda t, x, args: sigma_diff * jax.numpy.identity(dim_data)
            brownian_motion = diffrax.VirtualBrownianTree(t0, t1, tol=1e-5, shape=(dim_data,), key=key)
            terms = diffrax.MultiTerm(diffrax.ODETerm(drift), diffrax.ControlTerm(diffusion, brownian_motion))      
            return diffrax.diffeqsolve(
                terms,
                diffeqsolve_kwargs.pop("solver", diffrax.Heun()), 
                t0=t0,
                t1=t1,
                dt0=0.001,
                y0=input,
                **diffeqsolve_kwargs,
            ).ys[0]

        return jax.vmap(solve_ode)(data, data, keys)
    if jit_:
        transport_flow = jax.jit(transport_flow)

    if flow:
        return transport_flow
    else:
        return None

