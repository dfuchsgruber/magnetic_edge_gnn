import torch.nn as nn

from .basic_architecture import BasicArchitecture
from .gnn_layers import LineGraphConv


class LineGraphGNN(BasicArchitecture):
    """
    Node GNN based on the line graph.
    """

    def init_conv(self, in_channels: int, out_channels: int, **kwargs) -> nn.Module:
        return LineGraphConv(
            in_channels=in_channels, out_channels=out_channels, **kwargs
        )
