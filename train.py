from collections import defaultdict

import magnetic_edge_gnn.configs  # noqa: F401
from magnetic_edge_gnn.experiment import experiment
from magnetic_edge_gnn.train import train
from magnetic_edge_gnn.util import sacred_config_to_dict_config

from magnetic_edge_gnn.models import EdgeLevelTaskModule
import numpy as np


@experiment.command
def print_model_size(
    db_collection: str | None,
    overwrite: str | None,
    save_dir: str,
    checkpoint: bool,
    seed: int,
    data: dict,
    model: dict,
    training: dict,
    optimization: dict,
    logging: dict,
    run_idx: int | None,
    num_splits: int | None = None,
):
    config = sacred_config_to_dict_config(
        {
            "db_collection": db_collection,
            "overwrite": overwrite,
            "save_dir": save_dir,
            "checkpoint": checkpoint,
            "data": data,
            "model": model,
            "training": training,
            "optimization": optimization,
            "logging": logging,
            "run_idx": run_idx,
            "split_idx": -1,
        }
    )
    model = EdgeLevelTaskModule(config)
    # Print number of parameters of the model
    print(model)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")


@experiment.automain
def main(
    db_collection: str | None,
    overwrite: str | None,
    save_dir: str,
    checkpoint: bool,
    seed: int,
    data: dict,
    model: dict,
    training: dict,
    optimization: dict,
    logging: dict,
    run_idx: int | None,
    num_splits: int | None = None,
):
    if num_splits is None:
        num_splits = 1
    results = defaultdict(list)
    
    rng = np.random.default_rng(seed)
    
    for split_idx in range(num_splits):

        dict_config = sacred_config_to_dict_config(
            {
                "db_collection": db_collection,
                "overwrite": overwrite,
                "save_dir": save_dir,
                "checkpoint": checkpoint,
                "seed": int(rng.integers(2**31)),
                "data": data,
                "model": model,
                "training": training,
                "optimization": optimization,
                "logging": logging,
                "run_idx": run_idx,
                "split_idx" : None,
            }
        )
        dict_config.split_idx = split_idx
        for k, v in train(config=dict_config).items():
            results[k].append(v)
    return dict(results)