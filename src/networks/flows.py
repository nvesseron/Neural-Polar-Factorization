from typing import Callable, Optional, Any
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training import train_state


class TimeEncoder(nn.Module):
  """A cyclical time encoder."""
  n_frequencies: int = 128

  @nn.compact
  def __call__(self, t: jnp.ndarray) -> jnp.ndarray: 
    freq = 2 * jnp.arange(self.n_frequencies) * jnp.pi
    t = freq * t
    return jnp.concatenate((jnp.cos(t), jnp.sin(t)), axis=-1)
  


class MLPBlock(nn.Module):
  """An MLP block."""
  dim: int = 128
  num_layers: int = 3
  act_fn: Any = nn.silu
  out_dim: int = 128

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Apply the MLP block.

    Args:
      x: Input data of shape (batch_size, dim).

    """
    for _ in range(self.num_layers):
      x = nn.Dense(self.dim)(x)
      x = self.act_fn(x)
    return nn.Dense(self.out_dim)(x)

class VelocityField(nn.Module):
  """Parameterized neural vector field.

  The `VelocityField` learns a map
  :math:`v: \\mathbb{R}\times \\mathbb{R}^d\rightarrow \\mathbb{R}^d` solving
  the ODE :math:`\frac{dx}{dt} = v(t, x)`. Given a source distribution at time
  :math:`t=0`, the `VelocityField` can be used to transport the source
  distribution given at :math:`t_0` to a target distribution given at
  :math:`t_1` by integrating :math:`v(t, x)` from :math:`t=t_0` to
  :math:`t=t_1`.

  Args:
    output_dim: Dimensionality of the neural vector field.
    latent_embed_dim: Dimensionality of the embedding of the data.
    condition_dim: Dimensionality of the conditioning vector.
    condition_embed_dim: Dimensionality of the embedding of the condition.
      If :obj:`None`, set to ``latent_embed_dim``.
    t_embed_dim: Dimensionality of the time embedding.
      If :obj:`None`, set to ``latent_embed_dim``.
    joint_hidden_dim: Dimensionality of the hidden layers of the joint network.
      If :obj:`None`, set to ``latent_embed_dim + condition_embed_dim +
      t_embed_dim``.
    num_layers_per_block: Number of layers per block.
    act_fn: Activation function.
    n_frequencies: Number of frequencies to use for the time embedding.

  """
  output_dim: int
  latent_embed_dim: int
  condition_dim: int = 0
  condition_embed_dim: Optional[int] = None
  t_embed_dim: Optional[int] = None
  joint_hidden_dim: Optional[int] = None
  num_layers_per_block: int = 3
  act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu
  n_frequencies: int = 128

  def __post_init__(self):

    # set embedded dim from latent embedded dim
    if self.condition_embed_dim is None:
      self.condition_embed_dim = self.latent_embed_dim
    if self.t_embed_dim is None:
      self.t_embed_dim = self.latent_embed_dim

    # set joint hidden dim from all embedded dim
    concat_embed_dim = (
        self.latent_embed_dim + self.condition_embed_dim + self.t_embed_dim
    )
    if self.joint_hidden_dim is not None:
      assert (self.joint_hidden_dim >= concat_embed_dim), (
          "joint_hidden_dim must be greater than or equal to the sum of "
          "all embedded dimensions. "
      )
      self.joint_hidden_dim = self.latent_embed_dim
    else:
      self.joint_hidden_dim = concat_embed_dim
    super().__post_init__()

  @nn.compact
  def __call__(
      self,
      t: jnp.ndarray,
      x: jnp.ndarray,
      condition: Optional[jnp.ndarray] = None,
      rng: Optional[jnp.ndarray] = None,
  ) -> jnp.ndarray:
    """Forward pass through the neural vector field.

    Args:
      t: Time of shape (batch_size, 1).
      x: Data of shape (batch_size, output_dim).
      condition: Conditioning vector.
      rng: Random number generator.

    Returns:
      Output of the neural vector field.
    """
    t = TimeEncoder(n_frequencies=self.n_frequencies)(t)
    t_layer = MLPBlock(
        dim=self.t_embed_dim,
        out_dim=self.t_embed_dim,
        num_layers=self.num_layers_per_block,
        act_fn=self.act_fn
    )
    t = t_layer(t)

    x_layer = MLPBlock(
        dim=self.latent_embed_dim,
        out_dim=self.latent_embed_dim,
        num_layers=self.num_layers_per_block,
        act_fn=self.act_fn
    )
    x = x_layer(x)

    if self.condition_dim > 0:
      condition_layer = MLPBlock(
          dim=self.condition_embed_dim,
          out_dim=self.condition_embed_dim,
          num_layers=self.num_layers_per_block,
          act_fn=self.act_fn
      )
      condition = condition_layer(condition)
      concatenated = jnp.concatenate((t, x, condition), axis=-1)
    else:
      concatenated = jnp.concatenate((t, x), axis=-1)

    out_layer = MLPBlock(
        dim=self.joint_hidden_dim,
        out_dim=self.joint_hidden_dim,
        num_layers=self.num_layers_per_block,
        act_fn=self.act_fn
    )
    out = out_layer(concatenated)
    return nn.Dense(self.output_dim, use_bias=True)(out)
  
  def create_train_state(
      self,
      rng: jax.Array,
      optimizer: optax.OptState,
      input_dim: int,
  ) -> train_state.TrainState:
    """Create the training state.

    Args:
      rng: Random number generator.
      optimizer: Optimizer.
      input_dim: Dimensionality of the input.

    Returns:
      Training state.
    """
    params = self.init(
        rng, jnp.ones((1, 1)), jnp.ones((1, input_dim)),
        jnp.ones((1, self.condition_dim))
    )["params"]
    return train_state.TrainState.create(
        apply_fn=self.apply, params=params, tx=optimizer
    )

