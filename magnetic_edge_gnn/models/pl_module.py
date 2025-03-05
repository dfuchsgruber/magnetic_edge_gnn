import numpy as np
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from sklearn.feature_selection import r_regression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from torch.optim import SGD, Adam

from .gnn import (
    DirGNN,
    EdgeGNN,
    HodgeGNN,
    LineGraphGNN,
    MagneticChebGNN,
    MagneticEdgeGNN,
    MagneticEdgeGNNHiddenState,
    MagneticEdgeGNNHiddenStateBoth,
    Transformer,
)
from .mlp import MLP
from .mlp_with_fusion import MLPWithFusion


class EdgeLevelTaskModule(LightningModule):
    def __init__(self, config: DictConfig):
        """
        PyTorch Lightning model module class for edge-level tasks.

        Args:
            config (DictConfig): Configuration file for the model.
        """
        super().__init__()

        self.save_hyperparameters()
        self.config = config

        self.model_config, self.training_config, self.optimizer_config = (
            config.model,
            config.training,
            config.optimization,
        )

        self.task_type = (
            "regression"
            if self.training_config.loss == "mse_loss"
            else "classification"
        )

        self.orientation_equivariant_labels = config.data.orientation_equivariant_labels

        if self.model_config.type_ == "MLP":
            self.model = MLP(
                equi_in_channels=self.model_config.equi_input_dim,
                inv_in_channels=self.model_config.inv_input_dim,
                hidden_channels=self.model_config.hidden_dim,
                out_channels=self.model_config.output_dim,
                num_layers=self.model_config.num_layers,
                dropout=self.model_config.dropout,
                equi_act="tanh",
                inv_act="relu",
                classification=(self.task_type == "classification"),
                inputs=self.model_config.inputs,
            )
        elif self.model_config.type_ == "MLPWithFusion":
            self.model = MLPWithFusion(
                equi_in_channels=self.model_config.equi_input_dim,
                inv_in_channels=self.model_config.inv_input_dim,
                equi_hidden_channels=self.model_config.hidden_dim,
                inv_hidden_channels=self.model_config.hidden_dim,
                out_channels=self.model_config.output_dim,
                num_layers=self.model_config.num_layers,
                orientation_equivariant_labels=self.orientation_equivariant_labels,
                dropout=self.model_config.dropout,
                equi_act="tanh",
                inv_act="relu",
                classification=(self.task_type == "classification"),
                equivariant_to_invariant=False,
                invariant_to_equivariant=False,
                use_fusion_layers=self.model_config.use_fusion_layers,
            )
        elif self.model_config.type_ == "LineGraphGNN":
            self.model = LineGraphGNN(
                equi_in_channels=self.model_config.equi_input_dim,
                inv_in_channels=self.model_config.inv_input_dim,
                hidden_channels=self.model_config.hidden_dim,
                out_channels=self.model_config.output_dim,
                num_layers=self.model_config.num_layers,
                dropout=self.model_config.dropout,
                act="relu",
                classification=(self.task_type == "classification"),
                inputs=self.model_config.inputs,
            )
        elif self.model_config.type_ == "HodgeGNN":
            self.model = HodgeGNN(
                equi_in_channels=self.model_config.equi_input_dim,
                inv_in_channels=self.model_config.inv_input_dim,
                hidden_channels=self.model_config.hidden_dim,
                out_channels=self.model_config.output_dim,
                num_layers=self.model_config.num_layers,
                dropout=self.model_config.dropout,
                act="tanh",
                classification=(self.task_type == "classification"),
                bias=False,
                inputs=self.model_config.inputs,
            )
        elif self.model_config.type_ == "DirectedHodgeGNN":
            self.model = HodgeGNN(
                equi_in_channels=self.model_config.equi_input_dim,
                inv_in_channels=self.model_config.inv_input_dim,
                hidden_channels=self.model_config.hidden_dim,
                out_channels=self.model_config.output_dim,
                num_layers=self.model_config.num_layers,
                dropout=self.model_config.dropout,
                act="relu",
                classification=(self.task_type == "classification"),
                inputs=self.model_config.inputs,
            )
        elif self.model_config.type_ == "EdgeGNN":
            self.model = EdgeGNN(
                equi_in_channels=self.model_config.equi_input_dim,
                inv_in_channels=self.model_config.inv_input_dim,
                equi_hidden_channels=self.model_config.hidden_dim,
                inv_hidden_channels=self.model_config.hidden_dim,
                out_channels=self.model_config.output_dim,
                num_layers=self.model_config.num_layers,
                orientation_equivariant_labels=self.orientation_equivariant_labels,
                dropout=self.model_config.dropout,
                equi_act="tanh",
                inv_act="relu",
                classification=(self.task_type == "classification"),
                equivariant_to_invariant=self.model_config.equivariant_to_invariant,
                invariant_to_equivariant=self.model_config.invariant_to_equivariant,
                use_fusion_layers=self.model_config.use_fusion_layers,
            )
        elif self.model_config.type_ == "MagneticEdgeGNN":
            self.model = MagneticEdgeGNN(
                equi_in_channels=self.model_config.equi_input_dim,
                inv_in_channels=self.model_config.inv_input_dim,
                equi_hidden_channels=self.model_config.hidden_dim,
                inv_hidden_channels=self.model_config.hidden_dim,
                out_channels=self.model_config.output_dim,
                num_layers=self.model_config.num_layers,
                orientation_equivariant_labels=self.orientation_equivariant_labels,
                dropout=self.model_config.dropout,
                equi_act="tanh",
                inv_act="relu",
                classification=(self.task_type == "classification"),
                q=self.model_config.q * self.training_config.batch_size,
                equivariant_to_invariant=self.model_config.equivariant_to_invariant,
                invariant_to_equivariant=self.model_config.invariant_to_equivariant,
                use_fusion_layers=self.model_config.use_fusion_layers,
                gcn_normalize=self.model_config.get("gcn_normalize", False),
            )
        elif self.model_config.type_ == "MagneticChebGNN":
            self.model = MagneticChebGNN(
                degree=self.model_config.degree,
                equi_in_channels=self.model_config.equi_input_dim,
                inv_in_channels=self.model_config.inv_input_dim,
                equi_hidden_channels=self.model_config.hidden_dim,
                inv_hidden_channels=self.model_config.hidden_dim,
                out_channels=self.model_config.output_dim,
                num_layers=self.model_config.num_layers,
                orientation_equivariant_labels=self.orientation_equivariant_labels,
                dropout=self.model_config.dropout,
                equi_act="tanh",
                inv_act="relu",
                classification=(self.task_type == "classification"),
                q=self.model_config.q * self.training_config.batch_size,
                equivariant_to_invariant=self.model_config.equivariant_to_invariant,
                invariant_to_equivariant=self.model_config.invariant_to_equivariant,
                use_fusion_layers=self.model_config.use_fusion_layers,
                gcn_normalize=self.model_config.get("gcn_normalize", False),
            )
        elif self.model_config.type_ == "Transformer":
            self.model = Transformer(
                equi_in_channels=self.model_config.equi_input_dim,
                inv_in_channels=self.model_config.inv_input_dim,
                hidden_channels=self.model_config.hidden_dim,
                out_channels=self.model_config.output_dim,
                num_layers=self.model_config.num_layers,
                dropout=self.model_config.dropout,
                act="relu",
                classification=(self.task_type == "classification"),
                num_pos_encodings=self.model_config.num_pos_encodings,
                num_heads=self.model_config.num_heads,
                q=self.model_config.q * self.training_config.batch_size,
            )
        elif self.model_config.type_ == "DirGNN":
            self.model = DirGNN(
                equi_in_channels=self.model_config.equi_input_dim,
                inv_in_channels=self.model_config.inv_input_dim,
                equi_hidden_channels=self.model_config.hidden_dim,
                inv_hidden_channels=self.model_config.hidden_dim,
                out_channels=self.model_config.output_dim,
                num_layers=self.model_config.num_layers,
                orientation_equivariant_labels=self.orientation_equivariant_labels,
                dropout=self.model_config.dropout,
                equi_act="tanh",
                inv_act="relu",
                classification=(self.task_type == "classification"),
                equivariant_to_invariant=self.model_config.equivariant_to_invariant,
                invariant_to_equivariant=self.model_config.invariant_to_equivariant,
                use_fusion_layers=self.model_config.use_fusion_layers,
            )
        elif self.model_config.type_ == "MagneticEdgeGNNHidden":
            self.model = MagneticEdgeGNNHiddenState(
                equi_in_channels=self.model_config.equi_input_dim,
                inv_in_channels=self.model_config.inv_input_dim,
                equi_hidden_channels=self.model_config.hidden_dim,
                inv_hidden_channels=self.model_config.hidden_dim,
                out_channels=self.model_config.output_dim,
                num_layers=self.model_config.num_layers,
                orientation_equivariant_labels=self.orientation_equivariant_labels,
                dropout=self.model_config.dropout,
                equi_act="tanh",
                inv_act="relu",
                classification=(self.task_type == "classification"),
                q=self.model_config.q * self.training_config.batch_size,
                equivariant_to_invariant=self.model_config.equivariant_to_invariant,
                invariant_to_equivariant=self.model_config.invariant_to_equivariant,
                use_fusion_layers=self.model_config.use_fusion_layers,
            )
        elif self.model_config.type_ == "MagneticEdgeGNNHiddenBoth":
            self.model = MagneticEdgeGNNHiddenStateBoth(
                equi_in_channels=self.model_config.equi_input_dim,
                inv_in_channels=self.model_config.inv_input_dim,
                equi_hidden_channels=self.model_config.hidden_dim,
                inv_hidden_channels=self.model_config.hidden_dim,
                out_channels=self.model_config.output_dim,
                num_layers=self.model_config.num_layers,
                orientation_equivariant_labels=self.orientation_equivariant_labels,
                dropout=self.model_config.dropout,
                equi_act="tanh",
                inv_act="relu",
                classification=(self.task_type == "classification"),
                q=self.model_config.q * self.training_config.batch_size,
                equivariant_to_invariant=self.model_config.equivariant_to_invariant,
                invariant_to_equivariant=self.model_config.invariant_to_equivariant,
                use_fusion_layers=self.model_config.use_fusion_layers,
            )
        else:
            raise ValueError(f"The model {self.model_config.type_} is not supported!")

        if self.training_config.loss == "mse_loss":
            self.loss = nn.MSELoss()
        elif self.training_config.loss == "bce_loss":
            self.loss = nn.BCELoss()
        else:
            raise ValueError(f"The loss {self.training_config.loss} is not supported!")

    def _get_step_kwargs(self, batch, batch_idx) -> dict:
        kwargs = {}
        if hasattr(batch, "graph_idx"):
            kwargs["graph_idx"] = batch.graph_idx
        return kwargs

    def training_step(self, batch, batch_idx):
        if self.model.can_return_embeddings:
            predictions, embeddings = self.model(
                edge_index=batch.edge_index,
                equi_edge_attr=batch.equi_edge_attr,
                inv_edge_attr=batch.inv_edge_attr,
                undirected_mask=batch.undirected_mask,
                return_embeddings=True,
                batch=batch,
                **self._get_step_kwargs(batch, batch_idx),
            )
        else:
            predictions = self.model(
                edge_index=batch.edge_index,
                equi_edge_attr=batch.equi_edge_attr,
                inv_edge_attr=batch.inv_edge_attr,
                undirected_mask=batch.undirected_mask,
                batch=batch,
                **self._get_step_kwargs(batch, batch_idx),
            )
            embeddings = None

        if embeddings:
            for i, (emb_equi, emb_inv) in enumerate(embeddings):
                self.log_dict(
                    {f"train/embedding_equi/{i}/mean": emb_equi.mean().item()},
                    batch_size=self.training_config.batch_size,
                )
                self.log_dict(
                    {f"train/embedding_equi/{i}/std": emb_equi.std().item()},
                    batch_size=self.training_config.batch_size,
                )
                self.log_dict(
                    {f"train/embedding_inv/{i}/mean": emb_inv.mean().item()},
                    batch_size=self.training_config.batch_size,
                )
                self.log_dict(
                    {f"train/embedding_inv/{i}/std": emb_inv.std().item()},
                    batch_size=self.training_config.batch_size,
                )

        if len(predictions.size()) == 2:
            predictions = predictions.squeeze(1)
        loss = self.loss(predictions[batch.train_mask], batch.y[batch.train_mask])
        self.log_dict({"train/loss": loss}, batch_size=self.training_config.batch_size)
        return loss

    def any_step(self, batch, batch_idx, which, log: bool = True):
        predictions = self.model(
            edge_index=batch.edge_index,
            equi_edge_attr=batch.equi_edge_attr,
            inv_edge_attr=batch.inv_edge_attr,
            undirected_mask=batch.undirected_mask,
            batch=batch,
            **self._get_step_kwargs(batch, batch_idx),
        )
        match which:
            case "train":
                mask = batch.train_mask
            case "val":
                mask = batch.val_mask
            case "test":
                mask = batch.test_mask
            case _:
                raise ValueError(f"Invalid split {which}")

        if len(predictions.size()) == 2:
            predictions = predictions.squeeze(1)
        loss = self.loss(predictions[mask], batch.y[mask]).item()

        predictions = predictions[mask].detach().cpu().numpy()
        labels = batch.y[mask].detach().cpu().numpy()

        if self.task_type == "regression":
            metrics = self.regression_metrics(
                predictions=predictions, labels=labels, split=which
            )
        elif self.task_type == "classification":
            metrics = self.classification_metrics(
                predictions=predictions, labels=labels, split=which
            )

        metrics = {
            f"{which}/loss": loss,
            **metrics,
        }
        if log:
            self.log_dict(
                metrics,
                batch_size=1,
                on_epoch=True,
            )
        return metrics

    def validation_step(self, batch, batch_idx):
        self.any_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self.any_step(batch, batch_idx, "test")

    @staticmethod
    def classification_metrics(
        predictions: np.ndarray,
        labels: np.ndarray,
        split: str,
    ):
        hard_predictions = predictions > 0.5

        accuracy = accuracy_score(y_pred=hard_predictions, y_true=labels)
        recall = recall_score(y_pred=hard_predictions, y_true=labels)
        precision = precision_score(y_pred=hard_predictions, y_true=labels)
        _f1_score = f1_score(y_pred=hard_predictions, y_true=labels)
        auc_roc = roc_auc_score(y_score=predictions, y_true=labels)

        classification_metrics = {
            f"{split}/accuracy": accuracy,
            f"{split}/recall": recall,
            f"{split}/precision": precision,
            f"{split}/f1_score": _f1_score,
            f"{split}/auc_roc": auc_roc,
        }
        return classification_metrics

    @staticmethod
    def regression_metrics(
        predictions: np.ndarray,
        labels: np.ndarray,
        split: str,
    ):
        mse = mean_squared_error(y_pred=predictions, y_true=labels)
        rmse = root_mean_squared_error(y_pred=predictions, y_true=labels)
        mae = mean_absolute_error(y_pred=predictions, y_true=labels)
        mape = mean_absolute_percentage_error(y_pred=predictions, y_true=labels)
        corr = r_regression(X=predictions.reshape(-1, 1), y=labels)[0]
        regression_metrics = {
            f"{split}/mse": mse,
            f"{split}/rmse": rmse,
            f"{split}/mae": mae,
            f"{split}/mape": mape,
            f"{split}/corr": corr,
        }
        return regression_metrics

    def configure_optimizers(self):
        if self.optimizer_config.optim == "sgd":
            optimizer = SGD(
                self.parameters(),
                lr=self.optimizer_config.lr,
            )
        elif self.optimizer_config.optim == "adam":
            optimizer = Adam(
                self.parameters(),
                lr=self.optimizer_config.lr,
                weight_decay=self.optimizer_config.weight_decay,
            )
        else:
            raise ValueError(
                f"The optimizer {self.optimizer_config.optim} is not supported!"
            )

        return optimizer
