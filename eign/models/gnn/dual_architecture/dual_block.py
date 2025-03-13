import torch
import torch.nn as nn

from magnetic_edge_gnn.models.gnn.gnn_layers import (
    FusionLayer,
)
from magnetic_edge_gnn.models.model_utils import activation_resolver


class DualBlock(nn.Module):
    """Layer for the dual architecture."""

    def __init__(
        self,
        equi_in_dim: int,
        equi_out_dim: int,
        inv_in_dim: int,
        inv_out_dim: int,
        use_fusion: bool,
        inv_to_equi: bool,
        equi_to_inv: bool,
        init_equi_conv_fn=None,
        init_inv_conv_fn=None,
        init_equi_inv_conv_fn=None,
        init_inv_equi_conv_fn=None,
        equi_act: str = "tanh",
        inv_act: str = "relu",
        gcn_normalize: bool = False,
        **kwargs,
    ):
        super().__init__()
        self._equi_in_dim = equi_in_dim
        self._equi_out_dim = equi_out_dim
        self._inv_in_dim = inv_in_dim
        self._inv_out_dim = inv_out_dim
        self.use_fusion = use_fusion
        self.inv_to_equi = inv_to_equi
        self.equi_to_inv = equi_to_inv
        self.gcn_normalize = gcn_normalize
        self.equi_act = activation_resolver(equi_act)
        self.inv_act = activation_resolver(inv_act)

        # Fusion can only be used once we somehow obtain equi and invariant inputs of the same size
        self.use_fusion = (
            use_fusion
            and (equi_in_dim > 0 or (inv_to_equi and inv_in_dim > 0))
            and (inv_in_dim > 0 or (equi_to_inv and equi_in_dim > 0))
            and equi_out_dim == inv_out_dim
        )

        if self.use_fusion:
            self.equi_fusion_layer = FusionLayer(
                in1_channels=equi_out_dim,
                in2_channels=inv_out_dim,
                out_channels=equi_out_dim,
                bias=False,
                **kwargs,
            )
            self.inv_fusion_layer = FusionLayer(
                in1_channels=equi_out_dim,
                in2_channels=inv_out_dim,
                out_channels=inv_out_dim,
                **kwargs,
            )
        if inv_in_dim > 0:
            self.inv_conv = init_inv_conv_fn(
                in_channels=inv_in_dim,
                out_channels=inv_out_dim,
                gcn_normalize=self.gcn_normalize,
                **kwargs,
            )
            if self.inv_to_equi:
                self.inv_equi_conv = init_inv_equi_conv_fn(
                    inv_in_dim,
                    out_channels=equi_out_dim,
                    gcn_normalize=self.gcn_normalize,
                    **kwargs,
                )
        if equi_in_dim > 0:
            self.equi_conv = init_equi_conv_fn(
                in_channels=equi_in_dim,
                out_channels=equi_out_dim,
                gcn_normalize=self.gcn_normalize,
                **kwargs,
            )
            if self.equi_to_inv:
                self.equi_inv_conv = init_equi_inv_conv_fn(
                    in_channels=equi_in_dim,
                    out_channels=inv_out_dim,
                    gcn_normalize=self.gcn_normalize,
                    **kwargs,
                )

    @property
    def equi_out_dim(self) -> int:
        """The effective output dimension of the block for equivariant features."""
        if self._equi_in_dim > 0 or (self.inv_to_equi and self._inv_in_dim > 0):
            return self._equi_out_dim
        else:
            return 0

    @property
    def inv_out_dim(self) -> int:
        """The effective output dimension of the block for invariant features."""
        if self._inv_in_dim > 0 or (self.equi_to_inv and self._equi_in_dim > 0):
            return self._inv_out_dim
        else:
            return 0

    def fusion(self, equi_edge_attr, inv_edge_attr):
        new_equi_edge_attr = (
            self.equi_fusion_layer(
                equi_edge_attr,
                inv_edge_attr,
            )
            + equi_edge_attr
        )
        # Absolute value to keep representations orientation-invariant.
        new_inv_edge_attr = (
            self.inv_fusion_layer(
                torch.abs(equi_edge_attr),
                inv_edge_attr,
            )
            + inv_edge_attr
        )
        return new_equi_edge_attr, new_inv_edge_attr

    def _mix(self, tensor1, tensor2):
        match tensor1, tensor2:
            case None, None:
                return None
            case None, tensor2:
                return tensor2
            case tensor1, None:
                return tensor1
            case tensor1, tensor2:
                return tensor1 + tensor2

    def forward(
        self,
        edge_index: torch.Tensor,
        equi_edge_attr: torch.Tensor,
        inv_edge_attr: torch.Tensor,
        undirected_mask: torch.Tensor,
    ):
        h_equi_equi, h_inv_equi, h_equi_inv, h_inv_inv = None, None, None, None

        if equi_edge_attr.size(-1) > 0:
            h_equi_equi = self.equi_conv(
                edge_index=edge_index,
                edge_attr=equi_edge_attr,
                undirected_mask=undirected_mask,
            )
            if self.equi_to_inv:
                h_equi_inv = self.equi_inv_conv(
                    edge_index=edge_index,
                    edge_attr=equi_edge_attr,
                    undirected_mask=undirected_mask,
                )

        if inv_edge_attr.size(-1) > 0:
            h_inv_inv = self.inv_conv(
                edge_index=edge_index,
                edge_attr=inv_edge_attr,
                undirected_mask=undirected_mask,
            )
            if self.inv_to_equi:
                h_inv_equi = self.inv_equi_conv(
                    edge_index=edge_index,
                    edge_attr=inv_edge_attr,
                    undirected_mask=undirected_mask,
                )

        h_equi_equi = self._mix(h_equi_equi, h_inv_equi)
        h_inv_inv = self._mix(h_inv_inv, h_equi_inv)

        if h_equi_equi is not None:
            h_equi_equi = self.equi_act(h_equi_equi)
        if h_inv_inv is not None:
            h_inv_inv = self.inv_act(h_inv_inv)

        if self.use_fusion:
            h_equi_equi, h_inv_inv = self.fusion(h_equi_equi, h_inv_inv)

        if h_equi_equi is None:
            assert equi_edge_attr.size(-1) == 0
            h_equi_equi = equi_edge_attr

        if h_inv_inv is None:
            assert inv_edge_attr.size(-1) == 0
            h_inv_inv = inv_edge_attr

        return h_equi_equi, h_inv_inv
