import torch
import torch.nn as nn

from magnetic_edge_gnn.models.gnn.base import BaseModel

from ..model_utils import activation_resolver
from .gnn_layers import (
    LinearLayer,
)


class BasicArchitecture(BaseModel):
    def __init__(
        self,
        equi_in_channels: int,
        inv_in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float = 0.0,
        act: str = "relu",
        classification: bool = False,
        inputs: str = "both",
        **kwargs,
    ):
        """
        An abstract class for implementing basic GNN models and MLPs. They only maintain one representation for edge features.

        Args:
            equi_in_channels (int): Input dimension for orientation-equivariant features.
            inv_in_channels (int): Input dimension for orientation-invariant features.
            hidden_channels (int): Hidden dimension.
            out_channels (int): Output dimension.
            num_layers (int): Number of layers.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
            act (str, optional): Activation function. Defaults to "relu".
            classification (bool, optional): Whether the task is classification.
                In this case, sigmoid activation is applied to the outputs. Defaults to False.
        """

        super().__init__()

        self.inputs = inputs
        match inputs:
            case "both":
                self.in_channels = equi_in_channels + inv_in_channels
            case "equivariant":
                self.in_channels = equi_in_channels
            case "invariant":
                self.in_channels = inv_in_channels
            case _:
                raise ValueError(f"Invalid input type: {inputs}")

        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.classification = classification

        self.dropout = torch.nn.Dropout(p=dropout)
        self.act = activation_resolver(act)

        self.layers = nn.ModuleList()

        assert num_layers >= 1, "The architecture requires at least 1 layer."

        self.layers.append(
            self.init_conv(
                in_channels=self.in_channels,
                out_channels=self.hidden_channels,
                **kwargs,
            )
        )
        for _ in range(num_layers - 1):
            self.layers.append(
                self.init_conv(
                    in_channels=self.hidden_channels,
                    out_channels=self.hidden_channels,
                    **kwargs,
                )
            )

        self.projection_head = LinearLayer(
            in_channels=self.hidden_channels,
            out_channels=self.out_channels,
            **kwargs,
        )

        if self.classification:
            self.sigmoid = nn.Sigmoid()

    def init_conv(self, in_channels: int, out_channels: int, **kwargs) -> nn.Module:
        raise NotImplementedError

    def reset_parameters(self):
        """
        Resets all learnable parameters of the module.
        """
        for layer in self.layers:
            layer.reset_parameters()

        self.projection_head.reset_parameters()

    def forward(
        self,
        edge_index: torch.Tensor,
        equi_edge_attr: torch.Tensor,
        inv_edge_attr: torch.Tensor,
        undirected_mask: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        match self.inputs:
            case "both":
                edge_attr = torch.cat([equi_edge_attr, inv_edge_attr], dim=-1)
            case "equivariant":
                edge_attr = equi_edge_attr
            case "invariant":
                edge_attr = inv_edge_attr
            case _:
                raise ValueError(f"Invalid input type: {self.inputs}")

        for i in range(self.num_layers):
            edge_attr = self.layers[i](
                edge_index=edge_index,
                edge_attr=edge_attr,
                undirected_mask=undirected_mask,
            )
            edge_attr = self.act(edge_attr)
            edge_attr = self.dropout(edge_attr)

        edge_attr = self.projection_head(
            edge_index=edge_index,
            edge_attr=edge_attr,
            undirected_mask=undirected_mask,
        ).squeeze()

        if self.classification:
            edge_attr = self.sigmoid(edge_attr)

        return edge_attr
