import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb
import jax
import jax.numpy as jnp
import src.utils.misc as utils
from sklearn.manifold import TSNE
from ott.geometry import pointcloud
from ott.tools.sinkhorn_divergence import sinkhorn_divergence


class PFLogging:
    def __init__(self, rng, log_freq, log_freq_images, log_images_fn):
        self.rng = rng
        self.log_freq = log_freq
        self.log_freq_images = log_freq_images
        self.log_images_fn = log_images_fn


    def log(self, PF_state, log_train_stat, data, step=0):
        dim_data, state_u, state_i, state_conj_u, state_m = PF_state.dim_data, PF_state.state_u, PF_state.state_i, PF_state.state_conj_u, PF_state.state_m
        
        # log training stat
        if step % self.log_freq == 0:                
            wandb.log(log_train_stat)

        # log images
        if step % self.log_freq_images == 0:
            batch = {}
            batch["input"], batch["target"] = data
            self.rng, rng_batch, rng_log = jax.random.split(self.rng, num=3)
            batch["key"] = rng_batch
            self.log_images_fn(state_u, state_conj_u, state_i, state_m, dim_data, batch, rng_log)
        return None

def get_log_images_fn_(apply_grad_conj, conjugate_solver, transport_i, epsilon_target, show=False, F=None, figsize=(4,4), F_stochastic=False, accuracy_f_NN=None):
    @jax.jit
    def compute_for_logging(state_u, state_conj_u, state_i, batch, rng):
        rng, rn_rand= jax.random.split(rng, 2)

        # compute \nabla u^* \circ F(input)
        params_conj_u = state_conj_u.params
        grad_conj_u_point = apply_grad_conj(state_conj_u.apply_fn)
        grad_conj_u = jax.vmap(lambda x: grad_conj_u_point({"params": params_conj_u}, x))
        init_conj_u = grad_conj_u(batch["target"])
        m_input, converged_samples = conjugate_solver(batch["target"], init_conj_u, state_u)
        conv_rate = jnp.sum(converged_samples) / len(converged_samples)
           
        # apply grad_u
        grad_u_point = jax.grad(state_u.apply_fn, argnums=1)
        grad_u = jax.vmap(
            lambda x: grad_u_point({"params": state_u.params}, x)
        )
        grad_u_input = grad_u(batch["input"])

        # compute Sinkhorn divergences
        distance_pred_tar =  sinkhorn_divergence(
                pointcloud.PointCloud, grad_u_input, batch["target"] ,epsilon=epsilon_target
            ).divergence
        
        distance_input_tar =  sinkhorn_divergence(
                pointcloud.PointCloud, batch["input"], batch["target"] ,epsilon=epsilon_target
            ).divergence
        
        # compute 
        i_m_input = transport_i(state_i, m_input, rn_rand)

        return grad_u_input, m_input, i_m_input, conv_rate, distance_pred_tar, distance_input_tar
    
    
    def log_images_fn_(state_u, state_conj_u, state_i, state_m, dim_data, batch, rng):
        """Log all data and images necessary to observe the convergence of u and i"""
        
        grad_u_input, m_input, i_m_input, conv_rate, distance_pred_tar, distance_input_tar = compute_for_logging(state_u, state_conj_u, state_i, batch, rng)
        # log statistics
        wandb.log({"distance_input_tar": distance_input_tar})
        wandb.log({"distance_pred_tar": distance_pred_tar})
        wandb.log({"Conjugate solver convergence": 100 * conv_rate})

        # TSNE predicted vs target
        tsne = TSNE(n_components=2)
        len_pred = len(grad_u_input)
        X = np.array(jnp.concatenate((grad_u_input, batch["target"]), axis=0))
        embedded = tsne.fit_transform(X)
        fig = plt.figure(figsize=(5,5))
        pred = plt.scatter(embedded[:len_pred,0],embedded[:len_pred,1], color = "b", marker="x", s=8)
        tar = plt.scatter(embedded[len_pred:,0],embedded[len_pred:,1], color = "r", s=8)
        plt.legend((tar, pred),
            ('Target', 'Predicted'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=10)
        wandb.log({"Pred vs Target TSNE": [wandb.Image(fig)]})
        plt.close()

        # Compute cosine similarity to assess estimation of M^-1
        if F is not None:
            if F_stochastic:
                rng, rng_stoch = jax.random.split(rng, 2)
                F_i_m_input = F(i_m_input, rng_stoch)
            else:
                F_i_m_input = F(i_m_input)
            cos_sim = jnp.sum(batch["target"] * F_i_m_input, axis=1) / (jnp.sqrt(jnp.sum((batch["target"]**2), axis=1)) * jnp.sqrt(jnp.sum((F_i_m_input**2), axis=1)) )
            fig = plt.figure(figsize=(10,5))
            sns.histplot(cos_sim, bins=40, stat="probability")
            wandb.log({"cosine similarity for i": [wandb.Image(fig)]})
            plt.close()


        if dim_data == 2:
            # visualize transport
            im_input = utils.plot_(batch["input"], batch["input"], "Input ", show=show, figsize=figsize)
            im_m = utils.plot_(
                m_input, batch["input"], "M ", show=show, figsize=figsize
            )
            im_grad_u = utils.plot_(
                grad_u_input, batch["input"], "Grad u ", show=show, figsize=figsize
            )
            im_target = utils.plot_(batch["target"], batch["input"], "F ", show=show, figsize=figsize)

            im_inv_NF = utils.plot_(
                i_m_input, batch["input"], "I inverse of M ", show=show, figsize=figsize
            )
            
            wandb.log({"Input": [wandb.Image(im_input)]})
            wandb.log({"Target F": [wandb.Image(im_target)]})
            wandb.log({"M": [wandb.Image(im_m)]})
            wandb.log({"Grad u": [wandb.Image(im_grad_u)]})
            wandb.log({"Estimated inverse by I": [wandb.Image(im_inv_NF)]})
            
        if state_m is not None and dim_data == 2:
            # apply m
            pred_m = state_m.apply_fn({"params": state_m.params}, batch["input"])
            # apply grad_u
            grad_u_point = jax.grad(state_u.apply_fn, argnums=1)
            grad_u = jax.vmap(lambda x: grad_u_point({"params": state_u.params}, x))
            grad_u_m_input = grad_u(pred_m)

            im_pred_F = utils.plot_(grad_u_m_input, batch["input"], "Predicted F", show=show, figsize=figsize)
            wandb.log({"Predicted F": [wandb.Image(im_pred_F)]})
        
        # estimate accuracy of the input points
        if accuracy_f_NN is not None:
            value_acc_input = accuracy_f_NN(batch["input"])
            fig = plt.figure(figsize=(10,5))
            sns.histplot(value_acc_input, bins=40, stat="probability")
            wandb.log({"classification accuracy of input points": [wandb.Image(fig)]})
            plt.close()

        plt.close("all")
    return log_images_fn_
    
