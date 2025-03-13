import torch.nn as nn

from .dual_architecture import DualArchitecture
from .gnn_layers import RossiConv


class DirGNN(DualArchitecture):
    """
    Dual Rossi-like GNN based on the Orientation-equivariant and Orientation-invariant Edge Laplacians as the shift graph filters.
    """

    def init_equi_conv(
        self, in_channels: int, out_channels: int, **kwargs
    ) -> nn.Module:
        return RossiConv(
            in_channels=in_channels,
            out_channels=out_channels,
            matrix_type="orientation-equivariant",
            bias=False,
            **kwargs,
        )

    def init_inv_conv(self, in_channels: int, out_channels: int, **kwargs) -> nn.Module:
        return RossiConv(
            in_channels=in_channels,
            out_channels=out_channels,
            matrix_type="orientation-invariant",
            **kwargs,
        )

    def init_equi_inv_conv(
        self, in_channels: int, out_channels: int, **kwargs
    ) -> nn.Module:
        return RossiConv(
            in_channels=in_channels,
            out_channels=out_channels,
            matrix_type="mixed-orientation-equivariant-inputs",
            **(
                kwargs | dict(skip_connection=False)
            ),  # The skip connection from equivariant inputs would break invariance
        )

    def init_inv_equi_conv(
        self, in_channels: int, out_channels: int, **kwargs
    ) -> nn.Module:
        return RossiConv(
            in_channels=in_channels,
            out_channels=out_channels,
            matrix_type="mixed-orientation-invariant-inputs",
            bias=False,
            **(
                kwargs | dict(skip_connection=False)
            ),  # The skip connection from invariant inputs would break equivariance
        )
