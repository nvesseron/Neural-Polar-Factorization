from collections import namedtuple
import jax.numpy as jnp
import jax
import optax 
import flax.linen as nn
import src.data.samplers as samplers
import src.utils.load_mnist as load_mnist



Objective = namedtuple(
    "Objective", ["name", "fn", "limits", "dim", "scale_samp", "offset"]
)

def objective1(input: jnp.array) -> float:
    x, y = input[0], input[1]
    return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2

def objective10(input: jnp.array) -> float:
    x, y = input[0], input[1]
    x = 6.28 / 10 * x - 1.57
    y = 3.14 / 10 * y
    return 3 * jnp.cos(x) * jnp.cos(y)

def objective11(input: jnp.array) -> float:
    x, y = input[0], input[1]
    x = 6.28 / 10 * x - 1.57
    y = 6.28 / 10 * y - 1.57
    return 3 * jnp.cos(x) * jnp.cos(y)

class LeNet(nn.Module):
  """A simple CNN model."""

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=1, kernel_size=(5, 5), padding="VALID")(x)
    x = nn.relu(x)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=1, kernel_size=(5, 5), padding="VALID")(x)
    x = nn.relu(x)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))
    x = nn.Dense(features=10)(x) 
    return x
  
class empirical_loss_MNIST():
    """Class for MNIST experiment"""
    def __init__(self, classifier, unravel, batch_size=128) -> None:
        self.key = jax.random.PRNGKey(0)
        self.classifier = classifier
        self.unravel = unravel
        train_images, train_labels, test_images, test_labels = load_mnist.mnist()
        self.train_images, self.test_images = jnp.array(train_images), jnp.array(test_images)
        self.train_labels, self.test_labels = jnp.array(train_labels), jnp.array(test_labels)
        self.batch_size = batch_size

    # stochastic g function to optimize
    def f_NN(self, w, key_s):
        indices = jax.random.choice(key_s, jnp.arange(len(self.train_images)), (self.batch_size,))
        x = self.train_images[indices]
        y = self.train_labels[indices]
        params = self.unravel(w)
        y_pred = self.classifier.apply(params, x)
        return optax.softmax_cross_entropy(logits=y_pred, labels=y).mean() 
    
    # accuracy of the weights w
    def acc_NN(self, w):
        x = self.test_images
        y = self.test_labels
        params = self.unravel(w)
        y_pred = self.classifier.apply(params, x)
        return (jnp.argmax(y_pred, axis=1) == jnp.argmax(y, axis=1)).mean() * 100
  
def get_LeNet(batch_size=128):
    """get LeNet objective for experiments 2,3,4"""
    classifier = LeNet()
    x_init = jnp.empty((5, 28, 28, 1))
    params = classifier.init(jax.random.PRNGKey(0), x_init) 
    flat_params, unravel = jax.flatten_util.ravel_pytree(params)
    mnist_object = empirical_loss_MNIST(classifier, unravel, batch_size=batch_size)
    return mnist_object.acc_NN, Objective("MNIST",mnist_object.f_NN, [-1, 1], flat_params.shape[0], 2, -1)



def get_objective(objective_config, F_sampler_name):
    objective_name = objective_config["name"]
    objective_dict = {
        "objective_1": Objective("1", objective1, [-5, 5], 2, 10.0, -5),
        "objective_10": Objective("10", objective10, [-5, 5], 2, 10.0, -5),
        "objective_11": Objective("11", objective11, [-5, 5], 2, 10.0, -5),
    }
    if objective_name in objective_dict:
        objective = objective_dict[objective_name]
        F = jax.vmap(jax.grad(objective.fn)) 
        F_sampler = getattr(samplers, F_sampler_name)(
            scale=objective.scale_samp, offset=jnp.array([objective.offset] * objective.dim), dim=objective.dim, F=F
        )
        return objective_name, objective.dim, F, F_sampler, None
    
    elif objective_name == "Topography":
        F_sampler = getattr(samplers, F_sampler_name)()
        return objective_name, 2, None, F_sampler, None
    
    elif objective_name == "LeNet_MNIST":
        accuracy_fn, objective = get_LeNet()
        F = jax.vmap(jax.grad(objective.fn), in_axes=(0, None))
        F_sampler = getattr(samplers, F_sampler_name)(
                scale=objective.scale_samp, offset=jnp.array([objective.offset] * objective.dim), dim=objective.dim, F=F
            )
        return objective_name, objective.dim, F, F_sampler, jax.jit(jax.vmap(accuracy_fn))



