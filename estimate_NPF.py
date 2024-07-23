import os
import hydra
import wandb
from omegaconf import DictConfig
import jax
from flax.training import checkpoints
import ott.neural.models as models
from ott.geometry import pointcloud
from src.utils.convex_conjugate_fn import get_conjugate_solver
from src.utils.Adam_conjugate_solver import SolverAdam
import src.utils.misc as utils
from src.utils.objectives import get_objective
import src.networks.n_net as n_net
import src.networks.flows as flows
from src.model.PF_learning import PFState, PFTraining
from src.model.training_steps import get_train_step_dual_u, get_step_fn_flow
from src.model.batch_creation import get_create_training_samples_i
from src.utils.logging import PFLogging, get_log_images_fn_
from src.utils.transport_i import get_transport_i


@hydra.main(version_base=None, config_path="src/config", config_name="config_opti")
def experiment(config: DictConfig) -> None:

    # Get config
    config = config["config"]

    # Load objective
    objective_name, dim_data, F, F_sampler, _ = get_objective(config["objective"], config["F_sampler"]["name"])
    

    # Create architecture and optimizer for u
    rng = jax.random.PRNGKey(config["training_hyper"]["seed"])
    config_u = config["architecture"]["neural_u"]
    activation_function = utils.get_act_fn(config_u["act_fn"])
    neural_u = models.ICNN(
        dim_data=dim_data,
        dim_hidden=config_u["layers"],
        pos_weights=True,
        act_fn = activation_function,
        ranks=config_u["rank_pos_def"]
    )
    optimizer_u = utils.get_optimizer(config_u["optimizer"])

    # Create architecture and optimizer for the conjugate NN
    config_conj_u = config["architecture"]["neural_conj_u"]
    activation_function = utils.get_act_fn(config_conj_u["act_fn"])
    neural_conj_u = n_net.MLP(
        config_conj_u["layers"] + [dim_data],
        is_potential=False, 
        act_fn = activation_function,
    )
    optimizer_conj_u = utils.get_optimizer(config_conj_u["optimizer"])

    # Create architecture and optimizer for i
    config_i = config["architecture"]["neural_i"]
    activation_function = utils.get_act_fn(config_i["act_fn"])
    neural_i = flows.VelocityField(
        dim_data, 
        latent_embed_dim = config_i["latent_embed_dim"],
        condition_dim=dim_data,
        num_layers_per_block=config_i["nb_layers"],
        act_fn=activation_function
    )
    optimizer_i = utils.get_optimizer(config_i["optimizer"])
    transport_i = get_transport_i(sigma_diff=config["training_fn"]["sigma_diff"], jit_=False)

    # Create architecture and optimizer for m
    if "neural_m" in config["architecture"]:
        config_m = config["architecture"]["neural_m"]
        activation_function = utils.get_act_fn(config_m["act_fn"])
        neural_m = n_net.MLP(
            config_m["layers"],
            is_potential=False, 
            act_fn = activation_function,
        )
        optimizer_m = utils.get_optimizer(config_m["optimizer"])
    
    else:
        neural_m, optimizer_m = None, None

    # create the associated PFState
    rng, rng_state = jax.random.split(rng, 2) 
    pf_state = PFState(
        rng=rng_state,
        dim_data=dim_data,
        neural_u=neural_u,
        neural_conj_u=neural_conj_u,
        neural_i=neural_i,
        optimizer_u=optimizer_u,
        optimizer_conj_u=optimizer_conj_u,
        optimizer_i=optimizer_i,
        neural_m=neural_m,
        optimizer_m=optimizer_m
    )

    # Create conjugate solver
    solver = SolverAdam(
        max_iter=config["convex_conjugate"]["max_iter"],
        gtol=config["convex_conjugate"]["gtol"],
    )
    conjugate_solver = get_conjugate_solver(
        solver,
        max_iter=config["convex_conjugate"]["max_iter"],
        jit_= False
    )
        
    # Create training function for u
    train_step_u = get_train_step_dual_u(pf_state.apply_grad_conj, solver, max_iter=config["convex_conjugate"]["max_iter"])
    
    # Create training function for i
    create_training_samples_i = get_create_training_samples_i(conjugate_solver, pf_state.apply_grad_conj)
    train_step_i = get_step_fn_flow(sigma_diff=config["training_fn"]["sigma_diff"],
    )

    # Choose the right epsilon to evaluate the NNs
    rng, rng_part = jax.random.split(rng, 2)
    _ , target = F_sampler.generate_samples(rng_part, 4096)
    geom_target = pointcloud.PointCloud(target[:2048], target[2048:])
    epsilon_target = 0.05 * geom_target.mean_cost_matrix

    # Create an instance of PFLogging
    rng, rng_log = jax.random.split(rng, 2)
    log_images_fn = get_log_images_fn_(pf_state.apply_grad_conj, conjugate_solver, transport_i, epsilon_target, F=F, F_stochastic=(objective_name=="LeNet_MNIST"))
    pf_logging = PFLogging(rng_log, config["logging"]["log_freq"], config["logging"]["log_freq_images"], log_images_fn=log_images_fn)


    # Create an instance of PFTraining solver
    pf_training = PFTraining(
        rng,
        pf_state=pf_state,
        create_training_samples_i = create_training_samples_i,
        num_inner_iter_u=config["training_hyper"]["num_iter_u"],
        num_inner_iter_i=config["training_hyper"]["num_iter_i"],
        num_warmup_conj_u = config["training_hyper"]["num_warmup_conj_u"],
        train_step_u=train_step_u,
        train_step_i=train_step_i,
        batch_size_i=config["training_hyper"]["batch_size_i"],
        batch_size_u=config["training_hyper"]["batch_size_u"],
        pf_logging=pf_logging
    )
    
    if config["logging"]["offline"]:
            os.environ["WANDB_MODE"]="offline"

    wandb.init(
            project=config["logging"]["project"],
            name=objective_name,
            dir=config["saving"]["output_dir"],
            config=utils.flatten(dict(config)),
        )
    # Train the NNs
    state_u, state_conj_u, state_i = pf_training(F_sampler)
    
    wandb.run.finish()

    # Save the trained models
    CKPT_DIR_U = os.path.join(config["saving"]["output_dir"], config["saving"]["model_state_dir_u"])
    checkpoints.save_checkpoint(
        ckpt_dir=CKPT_DIR_U, target=state_u, step=config["training_hyper"]["num_optim_iter"]
    )
    if config["training_fn"]["train_step_u"] == "dual_step_ICNN":
        CKPT_DIR_CONJ = os.path.join(config["saving"]["output_dir"], config["saving"]["model_state_dir_conj"])
        checkpoints.save_checkpoint(
            ckpt_dir=CKPT_DIR_CONJ, target=state_conj_u, step=config["training_hyper"]["num_optim_iter"]
        )
    CKPT_DIR_I = os.path.join(config["saving"]["output_dir"], config["saving"]["model_state_dir_i"])
    checkpoints.save_checkpoint(
        ckpt_dir=CKPT_DIR_I, target=state_i, step=config["training_hyper"]["num_optim_iter"]
    )


if __name__ == "__main__":
    experiment()
