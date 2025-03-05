import torch.nn as nn

from .dual_architecture import DualArchitecture
from .gnn_layers import MagneticEdgeConv, MagneticEdgeConvHiddenState


class MagneticEdgeGNNHiddenState(DualArchitecture):
    """
    Dual GNN based on the Orientation-equivariant and Orientation-invariant Magnetic Edge Laplacians as the shift graph filters.
    Its convolution blocks allow for hidden states for inter-modality edge signals.
    """

    def init_equi_conv(
        self, in_channels: int, out_channels: int, **kwargs
    ) -> nn.Module:
        return MagneticEdgeConv(
            in_channels=in_channels,
            out_channels=out_channels,
            matrix_type="orientation-equivariant",
            bias=False,
            **kwargs,
        )

    def init_inv_conv(self, in_channels: int, out_channels: int, **kwargs) -> nn.Module:
        return MagneticEdgeConv(
            in_channels=in_channels,
            out_channels=out_channels,
            matrix_type="orientation-invariant",
            **kwargs,
        )

    def init_equi_inv_conv(
        self, in_channels: int, out_channels: int, **kwargs
    ) -> nn.Module:
        return MagneticEdgeConvHiddenState(
            in_channels=in_channels,
            out_channels=out_channels,
            matrix_type="mixed-orientation-equivariant-inputs",
            **(
                kwargs | dict(skip_connection=False)
            ),  # The skip connection from invariant inputs would break equivariance
        )

    def init_inv_equi_conv(
        self, in_channels: int, out_channels: int, **kwargs
    ) -> nn.Module:
        return MagneticEdgeConvHiddenState(
            in_channels=in_channels,
            out_channels=out_channels,
            matrix_type="mixed-orientation-invariant-inputs",
            bias=False,
            **(
                kwargs | dict(skip_connection=False)
            ),  # The skip connection from invariant inputs would break equivariance
        )
        


class MagneticEdgeGNNHiddenStateBoth(DualArchitecture):
    """
    Dual GNN based on the Orientation-equivariant and Orientation-invariant Magnetic Edge Laplacians as the shift graph filters.
    Its convolution blocks allow for hidden states at the node level for intra- and inter-modality edge signals.
    """

    def init_equi_conv(
        self, in_channels: int, out_channels: int, **kwargs
    ) -> nn.Module:
        return MagneticEdgeConvHiddenState(
            in_channels=in_channels,
            out_channels=out_channels,
            matrix_type="orientation-equivariant",
            bias=False,
            **kwargs,
        )

    def init_inv_conv(self, in_channels: int, out_channels: int, **kwargs) -> nn.Module:
        return MagneticEdgeConvHiddenState(
            in_channels=in_channels,
            out_channels=out_channels,
            matrix_type="orientation-invariant",
            **kwargs,
        )

    def init_equi_inv_conv(
        self, in_channels: int, out_channels: int, **kwargs
    ) -> nn.Module:
        return MagneticEdgeConvHiddenState(
            in_channels=in_channels,
            out_channels=out_channels,
            matrix_type="mixed-orientation-equivariant-inputs",
            **(
                kwargs | dict(skip_connection=False)
            ),  # The skip connection from invariant inputs would break equivariance
        )

    def init_inv_equi_conv(
        self, in_channels: int, out_channels: int, **kwargs
    ) -> nn.Module:
        return MagneticEdgeConvHiddenState(
            in_channels=in_channels,
            out_channels=out_channels,
            matrix_type="mixed-orientation-invariant-inputs",
            bias=False,
            **(
                kwargs | dict(skip_connection=False)
            ),  # The skip connection from invariant inputs would break equivariance
        )

