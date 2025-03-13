from functools import lru_cache

import yaml

from magnetic_edge_gnn.experiment import experiment


@lru_cache
def get_best_hyperparamters(filepath: str, data_name: str, model_name: str):
    with open(filepath, "r") as f:
        hyperparameters = yaml.safe_load(f)
    return hyperparameters[data_name][model_name]


@experiment.named_config
def traffic_flow_denoising_best_hyperparameters(model, data):
    locals().update(
        get_best_hyperparamters(
            "hyperparameters/traffic_flow_denoising.yaml",
            data["name"],
            model["name"],
        )
    )


@experiment.named_config
def traffic_flow_interpolation_best_hyperparameters(model, data):
    locals().update(
        get_best_hyperparamters(
            "hyperparameters/traffic_flow_interpolation.yaml",
            data["name"],
            model["name"],
        )
    )


@experiment.named_config
def traffic_flow_simulation_best_hyperparameters(model, data):
    locals().update(
        get_best_hyperparamters(
            "hyperparameters/traffic_flow_simulation.yaml",
            data["name"],
            model["name"],
        )
    )


@experiment.named_config
def electrical_circuits_denoising_best_hyperparameters(model, data):
    locals().update(
        get_best_hyperparamters(
            "hyperparameters/electrical_circuits.yaml",
            data["name"],
            model["name"],
        )
    )


@experiment.named_config
def electrical_circuits_interpolation_best_hyperparameters(model, data):
    locals().update(
        get_best_hyperparamters(
            "hyperparameters/electrical_circuits.yaml",
            data["name"],
            model["name"],
        )
    )


@experiment.named_config
def electrical_circuits_simulation_best_hyperparameters(model, data):
    locals().update(
        get_best_hyperparamters(
            "hyperparameters/electrical_circuits.yaml",
            data["name"],
            model["name"],
        )
    )


@experiment.named_config
def random_walk_denoising_best_hyperparameters(model, data):
    locals().update(
        get_best_hyperparamters(
            "hyperparameters/random_walk_denoising.yaml",
            data["name"],
            model["name"],
        )
    )


@experiment.named_config
def mixed_longest_cycle_identification_best_hyperparameters(model, data):
    locals().update(
        get_best_hyperparamters(
            "hyperparameters/mixed_longest_cycle_identification.yaml",
            data["name"],
            model["name"],
        )
    )


@experiment.named_config
def typed_triangles_orientation_best_hyperparameters(model, data):
    locals().update(
        get_best_hyperparamters(
            "hyperparameters/typed_triangles_orientation.yaml",
            data["name"],
            model["name"],
        )
    )
