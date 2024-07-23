from pathlib import Path
import jax
from jax import jit
from functools import partial
import numpy as np
import jax.numpy as jnp
import tifffile as tif
from scipy import ndimage

class uniform_F_sampler:
    """Sampler that generates samples (x_i, F(x_i)) where x_i has been uniformly sampled"""
    def __init__(self, dim: int=2, scale=1.0, offset=0.0, F=None):
        self.dim = dim
        self.scale = scale
        self.offset = offset
        self.F = F
        self.setup()

    def setup(self):
        @partial(jit, static_argnums=1)
        def generate_samples(key, num_samples):
            input = jax.random.uniform(key, shape=(num_samples, self.dim)) * self.scale + self.offset
            target = self.F(input)
            return input, target

        # define samples generator
        self.generate_samples = generate_samples

class topography_chamonix:
    """Sampler that generates Chamonix samples"""
    def __init__(self):
        self.setup()

    def setup(self):
        path_file = "./src/data/topography_data/Chamonix.tif"
        smoothness = 20
        image = tif.imread(path_file)

        # smooth the image
        image = image + np.random.random(size=(image.shape))
        self.f = jnp.array(ndimage.filters.gaussian_filter(image, smoothness, mode="nearest"))

        # compute gradient from the image
        gradx, grady = np.gradient(self.f)
        self.grad = np.concatenate((np.expand_dims(gradx, axis=2), np.expand_dims(grady, axis=2)), axis=2)
        self.grad = jnp.array(self.grad)

        # create input
        x, y = np.arange(self.grad.shape[1]), np.arange(self.grad.shape[0])
        xv, yv = np.meshgrid(x, y)

        # rescale input between -1 and 1
        xv = (xv.reshape((-1, 1)) / jnp.max(xv)) * 2 - 1
        yv = (yv.reshape((-1, 1)) / jnp.max(yv)) * 2 - 1

        self.source = jnp.concatenate((xv, yv), axis=1)
        self.target = jnp.reshape(self.grad, (-1, 2))
        
        # separate train from test
        self.index_test = jax.random.choice(jax.random.PRNGKey(0), jnp.arange(len(self.source)), shape=(int(15/100*len(self.source)),), replace=False)
        inter = np.arange(len(self.source))
        inter[[self.index_test]] = -2
        self.index_train = jnp.arange(len(self.source))[inter > -1]

        @partial(jit, static_argnums=1)
        def generate_samples(key, num_samples):
            index = jax.random.choice(key, self.index_train, shape=(num_samples,))
            target = self.target[index]
            input = self.source[index]
            
            return input, target

        # define samples generator
        self.generate_samples = generate_samples

        @partial(jit, static_argnums=1)
        def generate_samples_test(key, num_samples):
            index = jax.random.choice(key, self.index_test, shape=(num_samples,))
            target = self.target[index]
            input = self.source[index]
            
            return input, target

        # define samples generator
        self.generate_samples_test = generate_samples_test

class uniform_F_sampler_with_key:
    """Sampler that generates samples (x_i, F(x_i)) where x_i has been uniformly sampled and F is stochastic"""
    def __init__(self, dim: int=2, scale=1.0, offset=0.0, F=None):
        self.dim = dim
        self.scale = scale
        self.offset = offset
        self.F = F
        self.setup()

    def setup(self):
        @partial(jit, static_argnums=1)
        def generate_samples(key, num_samples):
            key_input, key_stoch = jax.random.split(key, 2)
            input = jax.random.uniform(key_input, shape=(num_samples, self.dim)) * self.scale + self.offset
            target = self.F(input, key_stoch)
            return input, target

        # define samples generator
        self.generate_samples = generate_samples