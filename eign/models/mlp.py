import torch.nn as nn

from .gnn import BasicArchitecture
from .gnn.gnn_layers import LinearLayer


class MLP(BasicArchitecture):
    """
    Multi-layer perceptron.
    """

    def init_conv(self, in_channels: int, out_channels: int, **kwargs) -> nn.Module:
        return LinearLayer(in_channels=in_channels, out_channels=out_channels, **kwargs)
