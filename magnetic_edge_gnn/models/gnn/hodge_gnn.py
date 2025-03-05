from torch import nn

from .basic_architecture import BasicArchitecture
from .gnn_layers import EdgeConv


class HodgeGNN(BasicArchitecture):
    """
    GNN based on the Edge Laplacian as the shift graph filter.
    """

    def init_conv(self, in_channels: int, out_channels: int, **kwargs) -> nn.Module:
        return EdgeConv(
            in_channels=in_channels,
            out_channels=out_channels,
            matrix_type="orientation-equivariant",
            **kwargs,
        )
