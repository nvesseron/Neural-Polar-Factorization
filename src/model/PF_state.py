import jax
import optax
import flax.linen as nn
from ott.neural.models import ICNN
from src.networks.flows import VelocityField

def create_train_state_(rng, neural, optimizer, dim_input):
    if neural is not None:
        state = neural.create_train_state(rng, optimizer, dim_input)
        return state
    else:
        return None   
    
class PFState:
    """ Class to store the training states of the convex NN u, the conjugate NN and the flow i

    Args:
        rng: Random key used for seeding for creating batch from the dataset.
        dim_data: Dimension of the data, 
        neural_u: network architecture for u, 
        neural_conj_u: network architecture for \nabla u^*,
        neural_i: neural vector field for i, 
        optimizer_u: optimizer function for potential u, 
        optimizer_conj_u: optimizer function for the conjugate, 
        optimizer_i: optimizer function for the flow i,
        neural_m: network architecture for m, 
        optimizer_i: optimizer function for m,
    """
    def __init__(self, 
                 rng, 
                 dim_data: int, 
                 neural_u: ICNN, 
                 neural_conj_u: nn.Module,
                 neural_i: VelocityField,
                 optimizer_u: optax.OptState, 
                 optimizer_conj_u: optax.OptState, 
                 optimizer_i: optax.OptState,
                 neural_m: nn.Module = None,
                 optimizer_m: optax.OptState = None,
                 ):
        self.dim_data = dim_data
        rn_state_u, rn_state_conj_u, rn_state_i, rn_state_m = jax.random.split(rng, 4)
        # create train states for the three NN
        self.state_u = create_train_state_(rn_state_u, neural_u, optimizer_u, dim_data)
        self.state_i = create_train_state_(rn_state_i, neural_i, optimizer_i, dim_data)
        self.state_conj_u = create_train_state_(rn_state_conj_u, neural_conj_u, optimizer_conj_u, dim_data)
        self.state_m = create_train_state_(rn_state_m, neural_m, optimizer_m, dim_data)
        self.apply_grad_conj = lambda apply_fn: apply_fn