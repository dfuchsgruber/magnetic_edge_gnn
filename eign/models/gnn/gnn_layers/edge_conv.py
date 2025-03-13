import torch
import torch.nn as nn

from magnetic_edge_gnn.models.model_utils import (
    degree_normalization,
    edge_laplacian,
)


class EdgeConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        matrix_type: str,
        bias: bool = True,
        skip_connection: bool = True,
        **kwargs,
    ):
        """
        Convolutional layer based on the Edge Laplacian as the shift graph filter.

        Args:
            in_channels (int): Input dimension.
            out_channels (int): Output dimension.
            matrix_type (str): Whether the matrix is orientation-equivariant or orientation-invariant.
            bias (bool, optional): Whether the layer learns an additive bias.
                Defaults to True.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.matrix_type = matrix_type
        self.skip_connection = skip_connection

        self.lin_layer_all = nn.Linear(
            in_features=in_channels,
            out_features=out_channels,
            bias=False,
        )
        if self.skip_connection:
            self.lin_layer_skip = nn.Linear(
                in_features=in_channels,
                out_features=out_channels,
                bias=False,
            )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_layer_all.reset_parameters()
        if self.skip_connection:
            self.lin_layer_skip.reset_parameters()
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        undirected_mask: torch.Tensor,
    ) -> torch.Tensor:
        edge_attr_all = self.lin_layer_all(edge_attr)
        if self.skip_connection:
            edge_attr_skip = self.lin_layer_skip(edge_attr)

        L = edge_laplacian(
            edge_index=edge_index,
            undirected_mask=undirected_mask,
            matrix_type=self.matrix_type,
        )

        # Normalize the matrix based on the degrees.
        L = degree_normalization(L)

        edge_attr = L @ edge_attr_all
        if self.skip_connection:
            edge_attr = edge_attr + edge_attr_skip

        if self.bias is not None:
            edge_attr = edge_attr + self.bias

        return edge_attr
