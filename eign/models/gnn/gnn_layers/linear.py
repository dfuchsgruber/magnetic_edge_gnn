import torch
import torch.nn as nn


class LinearLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        **kwargs,
    ):
        """
        Linear layer.

        Args:
            in_channels (int): Input dimension.
            out_channels (int): Output dimension.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin_layer = nn.Linear(
            in_features=self.in_channels,
            out_features=self.out_channels,
            bias=(kwargs["bias"] if "bias" in kwargs else True),
        )

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        undirected_mask: torch.Tensor,
    ) -> torch.Tensor:
        edge_attr = self.lin_layer(edge_attr)
        return edge_attr
