import torch
import torch.nn as nn

from magnetic_edge_gnn.models.model_utils import (
    degree_normalization,
    magnetic_edge_laplacian,
)

from .magnetic_edge_conv import MagneticEdgeConv


class MagneticChebConv(MagneticEdgeConv):
    """Chebyshev edge convolution."""

    def __init__(self, degree: int, **kwargs):
        super().__init__(**kwargs)
        self.degree = degree
        self.lin_layer_all_inter = nn.ModuleList(
            nn.Linear(
                in_features=self.out_channels,
                out_features=self.out_channels,
                bias=False,
            )
            for _ in range(self.degree + 1)
        )
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, "lin_layer_all_inter"):
            for layer in self.lin_layer_all_inter:
                layer.reset_parameters()

    def forward(self, edge_index, edge_attr, undirected_mask) -> torch.Tensor:
        edge_attr_all = self.lin_layer_all(edge_attr)

        if self.skip_connection:
            edge_attr_skip = self.lin_layer_skip(edge_attr)

        L = magnetic_edge_laplacian(
            edge_index=edge_index,
            undirected_mask=undirected_mask,
            matrix_type=self.matrix_type,
            q=self.q,
        )
        # Get the magnetic edge laplacian for inter-modality
        match self.matrix_type:
            case "orientation-equivariant" | "mixed-orientation-invariant-inputs":
                inter_matrix_type = "orientation-equivariant"
            case "orientation-invariant" | "mixed-orientation-equivariant-inputs":
                inter_matrix_type = "orientation-invariant"
            case _:
                raise ValueError(f"Invalid matrix type: {self.matrix_type}")
        L_inter = magnetic_edge_laplacian(
            edge_index=edge_index,
            undirected_mask=undirected_mask,
            matrix_type=inter_matrix_type,
            q=self.q,
        )

        # Normalize the matrix based on the degrees.
        L = degree_normalization(L)
        L_inter = degree_normalization(L_inter)

        Tx_0 = Tx_1 = edge_attr_all
        if "mixed" in self.matrix_type:
            # Mixing breaks equivariance and invariance without laplacian operators, so no degree zero coefficient
            Tx_0 = Tx_1 = out = 0
        else:
            out = self.lin_layer_all_inter[0](edge_attr_all)

        if self.degree >= 1:
            Tx_1 = torch.view_as_real(
                L
                @ torch.view_as_complex(
                    edge_attr_all.reshape(-1, self.out_channels // 2, 2)
                )
            ).reshape(-1, self.out_channels)
            out += self.lin_layer_all_inter[1](Tx_1)

        for lin_layer in self.lin_layer_all_inter[2:]:
            Tx_2 = (
                2
                * torch.view_as_real(
                    L_inter
                    @ torch.view_as_complex(Tx_1.reshape(-1, self.out_channels // 2, 2))
                ).reshape(-1, self.out_channels)
                - Tx_0
            )
            out += lin_layer(Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2

        edge_attr = out

        if self.skip_connection:
            edge_attr = edge_attr + edge_attr_skip

        if self.bias is not None:
            edge_attr = edge_attr + self.bias

        return edge_attr
