"""Helper functions"""
from typing import Callable
import flax.linen as nn
import optax
from collections.abc import MutableMapping
import os
import hydra
import wandb
from pathlib import Path 
from src.utils.objectives import Objective
import yaml
import urllib.request
from tqdm import tqdm
import matplotlib.pyplot as plt

def get_config_name(objective: Objective) -> str:
    config_str = {
        "f": objective.name,
        "input_dim": objective.dim,
    }

    return "-".join([f"{k}:{v}" for k, v in config_str.items()])


def get_config_name_sample(config: dict, objective: Objective) -> str:
    config_str = {
        "f": objective.name,
        "input_dim": objective.dim,
        "dim_noise": config["sampler"]["dim_noise"],
    }

    return "-".join([f"{k}:{v}" for k, v in config_str.items()])


def flatten(dictionary, parent_key='', separator='-'):
    """taken from https://stackoverflow.com/q/6027558"""
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


def softplus(x: float, beta_: float) -> float:
    return 1 / beta_ * nn.activation.softplus(beta_ * x)


def smooth_leaky_relu(x: float, alpha_: float, beta_: float) -> float:
    return alpha_ * x + (1 - alpha_) * softplus(x, beta_)


def get_act_fn(act_fn_name: str, alpha=0.01, beta=0.1, negative_slope=0.2) -> Callable:
    if act_fn_name == "smooth_leaky_relu":
        return lambda x: smooth_leaky_relu(x, alpha, beta)
    elif act_fn_name == "softplus":
        return lambda x: softplus(x, beta)
    elif act_fn_name == "leaky_relu":
        return lambda x: nn.leaky_relu(x, negative_slope)
    else:
        return getattr(nn, act_fn_name)


def get_optimizer(config: dict, b1=0.9, b2=0.999) -> Callable:
    scheduler = getattr(optax, config["scheduler"]["name"])(
        **config["scheduler"]["options"]
    )
    optimizer = getattr(optax, config["name"])(
        learning_rate=scheduler, b1=b1, b2=b2
    )

    return optimizer

def export_log() -> None:
    
    if os.environ.get("SLURM_ARRAY_TASK_ID"):

        log_path = f"{os.environ['SLURM_ARRAY_JOB_ID']}_{os.environ['SLURM_ARRAY_TASK_ID']}"

        hydra_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir).parents[0]
       
        wandb_logs_path = Path(wandb.run.dir).parents[0] / "files" / f"logs.txt"

        with open(wandb_logs_path, "w") as f:
            f.write(str(hydra_path / ".submitit" / log_path))
    else:
        
        print("Not a SLURM array task")


def get_wandb_config_from_url(wandb_path: str, configs_path=Path("/tmp")):
    """Transform a wandb yaml file as a hydra config style"""
    
    # Function to unflatten dict
    # taken from https://stackoverflow.com/a/6037657
    def unflatten(dictionary):
        resultDict = dict()
        for key, value in dictionary.items():
            parts = key.split("-")
            d = resultDict
            for part in parts[:-1]:
                if part not in d:
                    d[part] = dict()
                d = d[part]
            d[parts[-1]] = value
        return resultDict

    # Download file
    config_url = wandb.Api().run(wandb_path).file("config.yaml").directUrl
    
    # Read the file
    with urllib.request.urlopen(config_url) as f:
        config = yaml.load(f, yaml.SafeLoader)

    # Get it to the format
    config = unflatten({key: value["value"]
                        for key, value in config.items() if isinstance(value, dict)})
    config.pop("_wandb")

    return {"config": config}


# Download checkpoints
def get_wandb_checkpoints(wandb_path: str, state: int, checkpoint_path: str):

    run = wandb.Api().run(wandb_path)

    all_checkpoints = [file for file in run.files() if f"checkpoint" in file.name]
    selected_checkpoints = [
        file.download(checkpoint_path, replace=True)
        for file in tqdm(all_checkpoints) if f"{state}" in file.name
    ]

    if len(selected_checkpoints) < 1:
        print(f"WARNING: No checkpoints found for https://wandb.ai/{wandb_path} and state: {state}")
        print(f'WARNING: States available are: {set([int(ckpt.name.split("/")[-2].split("_")[-1]) for ckpt in all_checkpoints])}')
    else:
        print(f"INFO: Checkpoints (state {state}) downloaded to {checkpoint_path}")


def get_wandb_config(path_wandb_file: str):
    """Transform a wandb yaml file as a hydra config style"""

    # Function to unflatten dict
    # taken from https://stackoverflow.com/a/6037657
    def unflatten(dictionary):
        resultDict = dict()
        for key, value in dictionary.items():
            parts = key.split("-")
            d = resultDict
            for part in parts[:-1]:
                if part not in d:
                    d[part] = dict()
                d = d[part]
            d[parts[-1]] = value
        return resultDict

    # Read the file
    with open(path_wandb_file, "r") as f:
        config = yaml.load(f, yaml.SafeLoader)

    # Get it to the format
    config = unflatten({key: value["value"]
                        for key, value in config.items() if isinstance(value, dict)})
    config.pop("_wandb")

    return {"config": config}


def plot_(x, y, title_="", axes=1, cmap="plasma", show=True, figsize=(4, 4), x_lim = None, y_lim = None):
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x[:, 0], x[:, 1], s=10, marker="o", c=y[:, axes], cmap=cmap)
    ax.set_title(title_)
    ax.set_box_aspect(1)
    if show:
        plt.show()
        return None
    else:
        return fig
