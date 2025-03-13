import torch
import torch.nn as nn

from magnetic_edge_gnn.models.model_utils import (
    activation_resolver,
    degree_normalization,
    magnetic_edge_laplacian,
)


class MagneticEdgeConvHiddenState(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        matrix_type: str,
        q: float,
        bias: bool = True,
        skip_connection: bool = True,
        hidden_channels: int | None = None,
        activation: str = "relu",  # can be anything, since the node signals are invariant
        **kwargs,
    ):
        """
        Convolutional layer based on the Magnetic Laplacian where each application of the boundary has its own transformation.
        This leads to a hidden state.

        Args:
            in_channels (int): Input dimension.
            out_channels (int): Output dimension.
            matrix_type (str): Whether the matrix is orientation-equivariant or orientation-invariant.
            q (float): Absolute potential for the Magnetic Edge Laplacian.
            bias (bool, optional): Whether the layer learns an additive bias.
                Defaults to True.
            skip_connection (bool, optional): Whether the layer has a skip connection.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.matrix_type = matrix_type
        self.q = q
        self.skip_connection = skip_connection
        if hidden_channels is None:
            hidden_channels = out_channels
            hidden_channels = 32
        self.hidden_channels = hidden_channels

        assert (
            out_channels % 2 == 0
        ), "The number of output channels has to be divisible by two due to operations in complex numbers!"
        assert (
            hidden_channels % 2 == 0
        ), "The number of hidden channels has to be divisible by two due to operations in complex numbers!"

        if self.skip_connection:
            self.lin_layer_skip = nn.Linear(
                in_features=in_channels,
                out_features=out_channels,
                bias=bias,
            )

        self.lin_in = nn.Linear(
            in_features=in_channels,
            out_features=hidden_channels,
            bias=False,
        )
        self.lin_out = nn.Linear(
            in_features=hidden_channels,
            out_features=out_channels,
            bias=False,
        )
        self.act_hidden = activation_resolver(activation)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_in.reset_parameters()
        self.lin_out.reset_parameters()
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
        if self.skip_connection:
            edge_attr_skip = self.lin_layer_skip(edge_attr)

        L, B_out, B_in = magnetic_edge_laplacian(
            edge_index=edge_index,
            undirected_mask=undirected_mask,
            matrix_type=self.matrix_type,
            q=self.q,
            return_boundary=True,
        )
        L, B_out, B_in = L.to(torch.cfloat), B_out.to(torch.cfloat), B_in.to(torch.cfloat)

        # Normalize the matrix based on the degrees.
        B_inv_sqrt = degree_normalization(L, return_deg_inv_sqrt=True)
        edge_attr = torch.view_as_complex(
            self.lin_in(edge_attr).view(-1, self.hidden_channels // 2, 2)
        )
        edge_attr_hidden = B_in @ (B_inv_sqrt.view(-1, 1) * edge_attr)
        
        
        edge_attr_hidden = torch.view_as_complex(
            self.act_hidden(torch.view_as_real(edge_attr_hidden))
        )
        
        edge_attr_all = torch.view_as_real(
            (B_inv_sqrt.view(-1, 1) * (B_out @ edge_attr_hidden))
        ).view(-1, self.hidden_channels)
        
        
        
        edge_attr = self.lin_out(edge_attr_all)
        
        

        if self.skip_connection:
            edge_attr = edge_attr + edge_attr_skip

        if self.bias is not None:
            edge_attr = edge_attr + self.bias

        return edge_attr
    
    def __repr__(self):
        return super().__repr__() + f"({self.matrix_type}, {self.q})"
