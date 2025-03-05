import torch.nn as nn

from .gnn import DualArchitecture
from .gnn.gnn_layers import LinearLayer


class MLPWithFusion(DualArchitecture):
    """
    Multi-layer perceptron with fusion.
    """

    def init_inv_conv(self, in_channels: int, out_channels: int, **kwargs) -> nn.Module:
        return LinearLayer(in_channels=in_channels, out_channels=out_channels, **kwargs)

    def init_equi_conv(
        self, in_channels: int, out_channels: int, **kwargs
    ) -> nn.Module:
        return LinearLayer(in_channels=in_channels, out_channels=out_channels, **kwargs)
