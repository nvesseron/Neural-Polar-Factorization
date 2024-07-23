from typing import Callable, Sequence
import jax
import jax.numpy as jnp
import flax.linen as nn
from ott.neural.solvers import neuraldual

class MLP(neuraldual.BaseW2NeuralDual):
  """A generic, not-convex MLP.

  Args:
    dim_hidden: sequence specifying size of hidden dimensions. The output
      dimension of the last layer is automatically set to 1 if
      :attr:`is_potential` is ``True``, or the dimension of the input otherwise
    is_potential: Model the potential if ``True``, otherwise
      model the gradient of the potential
    act_fn: Activation function
  """

  dim_hidden: Sequence[int]
  is_potential: bool = True
  act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.leaky_relu

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:  # noqa: D102
    squeeze = x.ndim == 1
    if squeeze:
      x = jnp.expand_dims(x, 0)
    assert x.ndim == 2, x.ndim
    n_input = x.shape[-1]

    z = x
    for n_hidden in self.dim_hidden:
      Wx = nn.Dense(n_hidden, use_bias=True)
      z = self.act_fn(Wx(z))

    if self.is_potential:
      Wx = nn.Dense(1, use_bias=True)
      z = Wx(z).squeeze(-1)

      quad_term = 0.5 * jax.vmap(jnp.dot)(x, x)
      z += quad_term
    else:
      Wx = nn.Dense(n_input, use_bias=True)
      z = Wx(z)

    return z.squeeze(0) if squeeze else z