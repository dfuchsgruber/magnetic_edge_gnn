from typing import Any
import warnings
from pathlib import Path
from uuid import uuid4

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from magnetic_edge_gnn.datasets import EdgeLevelTaskDataModule
from magnetic_edge_gnn.logging.dict_logger import DictLogger
from magnetic_edge_gnn.logging.wandb import WandbLogger
from magnetic_edge_gnn.models import EdgeLevelTaskModule

warnings.filterwarnings(
    "ignore", ".*Consider increasing the value of the `num_workers` argument*"
)


def train(config: DictConfig) -> dict[str, Any]:
    """Trains and evaluates a model on the edge-level task.
    
    Args:
        config (DictConfig): Configuration file for the model.
    
    Returns:
        dict[str, Any]: Results of the training and evaluation.
    """
    seed_everything(config.seed, workers=True)

    silent = (
        (config.db_collection is not None and config.overwrite is not None)
        if config.logging.silent is None
        else config.logging.silent
    )

    dm = EdgeLevelTaskDataModule(
        config.data,
        batch_size=config.training.batch_size,
        seed=config.seed,
        arbitrary_orientation=config.data.arbitrary_orientation,
    )
    dm.split(seed=config.seed)

    model = EdgeLevelTaskModule(config)
    if not silent:
        print(model)
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # List of callbacks
    callbacks = []

    experiment_dir = str(
        (
            Path(config.save_dir)
            / f"{config.db_collection}_{config.overwrite}"
            / f"{uuid4()}-{config.seed}"
            / f"{config.run_idx}"
            / f"{config.split_idx}"
        ).absolute()
    )
    # Create experiment dir if not existing
    Path(experiment_dir).mkdir(parents=True, exist_ok=True)

    # save DictConfig config as yaml file
    with open(Path(experiment_dir) / "config.yaml", "w") as f:
        OmegaConf.save(config, f)

    if config.checkpoint:
        # Define model checkpoint callback.
        if config.training.loss == "mse_loss":
            checkpoint_callback = ModelCheckpoint(
                dirpath=experiment_dir,
                monitor="val/rmse",
                mode="min",
                filename="{epoch:02d}-{val/rmse:.4f}",
                save_top_k=1,
            )
        elif config.training.loss == "bce_loss":
            checkpoint_callback = ModelCheckpoint(
                dirpath=experiment_dir,
                monitor="val/auc_roc",
                mode="max",
                filename="{epoch:02d}-{val/auc_roc:.4f}",
                save_top_k=1,
            )
        callbacks.append(checkpoint_callback)

    # Define learning rate logger.
    lr_logger = LearningRateMonitor("step")
    callbacks.append(lr_logger)
    dict_logger = DictLogger()
    loggers = [dict_logger]

    if config.logging.wandb is not None:
        wandb_logger = WandbLogger(
            **config.logging.wandb,
            save_dir=Path(experiment_dir) / "wandb",
        )
        loggers.append(wandb_logger)
        wandb_config = OmegaConf.to_container(config, resolve=True) | dict(
            hyperparameter_configuration=f"{config.model.name}-layers_{config.model.num_layers}-dim_{config.model.hidden_dim}-lr_{config.optimization.lr}"
        )
        wandb_logger.log_hyperparams(wandb_config)
    else:
        wandb_logger = None

    trainer = Trainer(
        accelerator="auto",  # Uses GPU if available.
        log_every_n_steps=1,
        callbacks=callbacks,
        max_epochs=config.training.num_epochs,
        gradient_clip_val=config.training.max_grad_norm,
        default_root_dir=experiment_dir,
        logger=loggers,
        enable_model_summary=not silent,
        enable_progress_bar=not silent,
    )

    trainer.fit(model=model, datamodule=dm)
    (result_val,) = trainer.validate(
        datamodule=dm,
        ckpt_path="best",
        verbose=not silent,
    )
    result_test = trainer.test(datamodule=dm, ckpt_path="best", verbose=not silent)[0]

    # Assert that there is no overlap between the keys of the validation and test results.
    assert len(set(result_val.keys()).intersection(set(result_test.keys()))) == 0

    if not silent:
        for metrics in config.logging.plot_to_console:
            dict_logger.print_metrics(*metrics)

    if wandb_logger:
        wandb_logger.finish()

    torch.save(dict_logger.get_metrics(), Path(experiment_dir) / "running_metrics.pt")

    return dict(
        **result_val,
        **result_test,
        results_dir=experiment_dir,
    )
