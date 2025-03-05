
from .edge_conv import EdgeConv
import torch
import torch.nn as nn

from magnetic_edge_gnn.models.model_utils import (
    degree_normalization,
    edge_laplacian,
    incidence_matrix,
)


class RossiConv(EdgeConv):
    """Rossi-like edge convolution, which has different weights and paths for undirected edges, and each individual direction."""
    
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.lin_layer_all2 = nn.Linear(
            in_features=self.in_channels,
            out_features=self.out_channels,
            bias=False,
        )
        self.lin_layer_all3 = nn.Linear(
            in_features=self.in_channels,
            out_features=self.out_channels,
            bias=False,
        )
        
        self.reset_parameters()
        
    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, "lin_layer_all2"):
            self.lin_layer_all2.reset_parameters()
        if hasattr(self, "lin_layer_all3"):
            self.lin_layer_all3.reset_parameters()
    
    
    def incidence_matrices(self, edge_index: torch.Tensor,
                            undirected_mask: torch.Tensor,
                         matrix_type: str, 
                         ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = incidence_matrix(edge_index=edge_index, undirected_mask=undirected_mask,
                             matrix_type=matrix_type)
        
        B_undir = B.clone()
        B_undir[:, ~undirected_mask] = 0 # Zero out the directed edges
        B_dir = B.clone()
        B_dir[:, undirected_mask] = 0
        
        # First directed variant, zero out negative entries
        B_dir_pos = B_dir.clone()
        B_dir_pos[B_dir_pos < 0] = 0
        # Second directed variant, zero out positive entries
        B_dir_neg = B_dir.clone()
        B_dir_neg[B_dir_neg > 0] = 0
        return B_undir, B_dir_pos, B_dir_neg
    
    def laplacians(self, edge_index: torch.Tensor, undirected_mask: torch.Tensor, matrix_type: str):
        match matrix_type:
            case "orientation-equivariant" | "orientation-invariant":
                Bs_in = Bs_out = self.incidence_matrices(edge_index=edge_index, undirected_mask=undirected_mask, matrix_type=matrix_type)
            case "mixed-orientation-equivariant-inputs":
                Bs_in = self.incidence_matrices(edge_index=edge_index, undirected_mask=undirected_mask, matrix_type="orientation-equivariant")
                Bs_out = self.incidence_matrices(edge_index=edge_index, undirected_mask=undirected_mask, matrix_type="orientation-invariant")
            case "mixed-orientation-invariant-inputs":
                Bs_in = self.incidence_matrices(edge_index=edge_index, undirected_mask=undirected_mask, matrix_type="orientation-invariant")
                Bs_out = self.incidence_matrices(edge_index=edge_index, undirected_mask=undirected_mask, matrix_type="orientation-equivariant")
            case _:
                raise ValueError(f"Invalid matrix type: {matrix_type}")
        
        return (
            (torch.transpose(B_out, -2, -1) @ B_in, B_in, B_out)
            for B_in, B_out in zip(Bs_in, Bs_out)
        )
        
        
    def forward(self, edge_index, edge_attr, undirected_mask) -> torch.Tensor:
        
        edge_attrs_all = [
            lin(edge_attr)
            for lin in (self.lin_layer_all, self.lin_layer_all2, self.lin_layer_all3)
        ]
        
        if self.skip_connection:
            edge_attr_skip = self.lin_layer_skip(edge_attr)

        edge_attr = 0
        Ls = self.laplacians(edge_index=edge_index, undirected_mask=undirected_mask, matrix_type=self.matrix_type)
        for (L, _, _), edge_attr_all in zip(Ls, edge_attrs_all):
            L = degree_normalization(L)
            edge_attr += L @ edge_attr_all
        
        if self.skip_connection:
            edge_attr = edge_attr + edge_attr_skip

        if self.bias is not None:
            edge_attr = edge_attr + self.bias

        return edge_attr
