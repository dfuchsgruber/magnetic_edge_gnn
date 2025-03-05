import torch
import torch.nn as nn

from magnetic_edge_gnn.models.model_utils import (
    degree_normalization,
    edge_laplacian,
)


class DirEdgeConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        matrix_type: str,
        bias: bool = True,
        **kwargs,
    ):
        """
        Directed convolutional layer based on the Edge Laplacian as the shift graph filter.

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

        self.lin_layer_pos = nn.Linear(
            in_features=in_channels,
            out_features=out_channels,
            bias=False,
        )
        self.lin_layer_neg = nn.Linear(
            in_features=in_channels,
            out_features=out_channels,
            bias=False,
        )
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
        self.lin_layer_pos.reset_parameters()
        self.lin_layer_neg.reset_parameters()
        self.lin_layer_skip.reset_parameters()
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        undirected_mask: torch.Tensor,
    ) -> torch.Tensor:
        edge_attr_pos = self.lin_layer_pos(edge_attr)
        edge_attr_neg = self.lin_layer_neg(edge_attr)
        edge_attr_skip = self.lin_layer_skip(edge_attr)

        L = edge_laplacian(
            edge_index=edge_index,
            undirected_mask=undirected_mask,
            matrix_type=self.matrix_type,
        )

        # Undirected edges.
        L_undirected_mask = torch.zeros_like(L, dtype=torch.bool)
        L_undirected_mask[undirected_mask, :] = True
        L_undirected_mask[:, undirected_mask] = True

        # Separately consider the positive and the negative entries of the Edge Laplacian.
        L_pos = L.detach().clone()
        L_pos[L_pos < 0] = 0
        L_pos[L_undirected_mask] = torch.abs(L[L_undirected_mask])

        L_neg = L.detach().clone()
        L_neg[L_neg > 0] = 0
        L_neg[L_undirected_mask] = -torch.abs(L[L_undirected_mask])

        # Normalize the matrices based on the degrees.
        L_pos = degree_normalization(L_pos)
        L_neg = degree_normalization(L_neg)

        edge_attr = L_pos @ edge_attr_pos + L_neg @ edge_attr_neg + edge_attr_skip

        if self.bias is not None:
            edge_attr = edge_attr + self.bias

        return edge_attr
