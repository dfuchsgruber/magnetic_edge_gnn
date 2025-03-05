from typing import Mapping

from omegaconf import DictConfig, OmegaConf


def sacred_config_to_dict_config(config) -> DictConfig:
    """Transform a Sacred config to an OmegaConf DictConfig."""

    def to_primitive(value, depth: int = 0):
        if isinstance(value, Mapping):
            return {k: to_primitive(v, depth=depth + 1) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [to_primitive(v, depth=depth + 1) for v in value]
        else:
            return value

    return OmegaConf.create(to_primitive(config))
