import logging
import os
from os import PathLike

from pytorch_lightning.loggers import WandbLogger as WandbLogger_

import wandb


class WandbLogger(WandbLogger_):
    """Logger that logs to wandb using a custom API token path and cache directory."""

    def __init__(
        self,
        *args,
        cache_dir=None,
        log_internal_dir=None,
        api_token_path=None,
        **kwargs,
    ):
        if api_token_path and os.path.exists(api_token_path):
            logging.info(f"Loading W&B API token from {api_token_path}")
            with open(api_token_path, "r") as f:
                os.environ["WANDB_API_KEY"] = f.read().strip()
        if cache_dir:
            self._set_cache_dir(cache_dir)
        if log_internal_dir:
            kwargs["settings"] = kwargs.get("settings", {}).update(
                {"log_internal_dir": log_internal_dir}
            )
        super().__init__(*args, **kwargs)

    def _set_cache_dir(self, cache_dir: PathLike | str):
        """Sets all cache directories W&B uses to stop it from flooding ./cache and .local/share with artifacts..."""
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["WANDB_ARTIFACT_LOCATION"] = str(cache_dir)
        os.environ["WANDB_ARTIFACT_DIR"] = str(cache_dir)
        os.environ["WANDB_CACHE_DIR"] = str(cache_dir)
        os.environ["WANDB_CONFIG_DIR"] = str(cache_dir)
        os.environ["WANDB_DATA_DIR"] = str(cache_dir)

    def _clean(self):
        import subprocess

        subprocess.run(["wandb", "sync", "--clean-force"], check=True)
        subprocess.run(["wandb", "artifact", "cache", "cleanup", "0GB"], check=True)

    def finish(self):
        wandb.finish()
        self._clean()
