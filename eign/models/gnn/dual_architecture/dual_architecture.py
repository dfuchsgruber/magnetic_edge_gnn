import torch
import torch.nn as nn

from magnetic_edge_gnn.models.gnn.base import BaseModel
from magnetic_edge_gnn.models.gnn.gnn_layers import LinearLayer
from magnetic_edge_gnn.models.model_utils import activation_resolver

from .dual_block import DualBlock


class DualArchitecture(BaseModel):
    def __init__(
        self,
        equi_in_channels: int,
        inv_in_channels: int,
        equi_hidden_channels: int,
        inv_hidden_channels: int,
        out_channels: int,
        num_layers: int,
        orientation_equivariant_labels: bool,
        dropout: float = 0.0,
        equi_act: str = "tanh",
        inv_act: str = "relu",
        classification: bool = False,
        equivariant_to_invariant: bool = False,
        invariant_to_equivariant: bool = False,
        use_fusion_layers: bool = True,
        gcn_normalize: bool = False,
        **kwargs,
    ):
        """
        An abstract class for implementing GNN models with dual architecture that maintains representations for 
        equivariant and invariant edge features.

        Args:
            equi_in_channels (int): Input dimension for orientation-equivariant features.
            inv_in_channels (int): Input dimension for orientation-invariant features.
            equi_hidden_channels (int): Hidden dimension for orientation-equivariant features.
            inv_hidden_channels (int): Hidden dimension for orientation-invariant features.
            out_channels (int): Output dimension.
            num_layers (int): Number of layers.
            orientation_equivariant_labels (bool): Whether the labels are orientation-equivariant or not.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
            equi_act (str, optional): Activation function for orientation-equivariant features. Defaults to "tanh".
            inv_act (str, optional): Activation function for orientation-invariant features. Defaults to "relu".
            classification (bool, optional): Whether the task is classification.
                In this case, sigmoid activation is applied to the outputs. Defaults to False.
            use_fusion_layers (bool, optional): Whether to include fusion layers. Defaults to True.
            equivariant_to_invariant (bool, optional): Whether to include equivariant-to-invariant layers. Defaults to False.
            invariant_to_equivariant (bool, optional): Whether to include invariant-to-equivariant layers. Defaults to False.
        """

        super().__init__()

        self.equi_in_channels = equi_in_channels
        self.inv_in_channels = inv_in_channels
        self.equi_hidden_channels = equi_hidden_channels
        self.inv_hidden_channels = inv_hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.orientation_equivariant_labels = orientation_equivariant_labels
        self.classification = classification
        self.equivariant_to_invariant = equivariant_to_invariant
        self.invariant_to_equivariant = invariant_to_equivariant
        self.use_fusion_layers = use_fusion_layers
        self.gcn_normalize = gcn_normalize

        self.dropout = torch.nn.Dropout(p=dropout)
        self.equi_act = activation_resolver(equi_act)
        self.inv_act = activation_resolver(inv_act)

        assert num_layers >= 1, "The architecture requires at least 1 layer."

        self.blocks = nn.ModuleList()

        equi_in_dim, inv_in_dim = equi_in_channels, inv_in_channels
        for i in range(num_layers):
            block = DualBlock(
                equi_in_dim=equi_in_dim,
                equi_out_dim=self.equi_hidden_channels,
                inv_in_dim=inv_in_dim,
                inv_out_dim=self.inv_hidden_channels,
                use_fusion=use_fusion_layers,
                inv_to_equi=invariant_to_equivariant,
                equi_to_inv=equivariant_to_invariant,
                init_equi_conv_fn=self.init_equi_conv,
                init_inv_conv_fn=self.init_inv_conv,
                init_equi_inv_conv_fn=self.init_equi_inv_conv,
                init_inv_equi_conv_fn=self.init_inv_equi_conv,
                gcn_normalize=gcn_normalize,
                **kwargs,
            )
            self.blocks.append(block)
            equi_in_dim, inv_in_dim = block.equi_out_dim, block.inv_out_dim

        self.projection_head = LinearLayer(
            in_channels=(
                self.equi_hidden_channels
                if self.orientation_equivariant_labels
                else self.inv_hidden_channels
            ),
            out_channels=self.out_channels,
            bias=False if self.orientation_equivariant_labels else True,
            **kwargs,
        )

        if self.classification:
            self.sigmoid = nn.Sigmoid()

    @property
    def can_return_embeddings(self) -> bool:
        return True

    def init_equi_conv(
        self, in_channels: int, out_channels: int, **kwargs
    ) -> nn.Module:
        raise NotImplementedError

    def init_inv_conv(self, in_channels: int, out_channels: int, **kwargs) -> nn.Module:
        raise NotImplementedError

    def init_equi_inv_conv(
        self, in_channels: int, out_channels: int, **kwargs
    ) -> nn.Module:
        """Convolution operator from equivariant inputs to invariant inputs."""
        raise NotImplementedError

    def init_inv_equi_conv(
        self, in_channels: int, out_channels: int, **kwargs
    ) -> nn.Module:
        """Convolution operator from invariant inputs to equivariant inputs."""
        raise NotImplementedError

    def init_conv(self, in_channels: int, out_channels: int, **kwargs) -> nn.Module:
        raise NotImplementedError

    def reset_parameters(self):
        """
        Resets all learnable parameters of the module.
        """
        for layer in self.equi_layers:
            layer.reset_parameters()

        for layer in self.inv_layers:
            layer.reset_parameters()

        if self.equivariant_to_invariant:
            for layer in self.equi_inv_layers:
                layer.reset_parameters()

        if self.invariant_to_equivariant:
            for layer in self.inv_equi_layers:
                layer.reset_parameters()

        for layer in self.equi_fusion_layers:
            layer.reset_parameters()

        for layer in self.inv_fusion_layers:
            layer.reset_parameters()

        self.projection_head.reset_parameters()

    def forward(
        self,
        edge_index: torch.Tensor,
        equi_edge_attr: torch.Tensor,
        inv_edge_attr: torch.Tensor,
        undirected_mask: torch.Tensor,
        *args,
        return_embeddings=False,
        **kwargs,
    ) -> torch.Tensor:
        embeddings = []

        for block in self.blocks:
            equi_edge_attr, inv_edge_attr = block(
                edge_index=edge_index,
                equi_edge_attr=equi_edge_attr,
                inv_edge_attr=inv_edge_attr,
                undirected_mask=undirected_mask,
            )
            embeddings.append([equi_edge_attr, inv_edge_attr])

            equi_edge_attr = self.equi_act(equi_edge_attr)
            inv_edge_attr = self.inv_act(inv_edge_attr)

            if equi_edge_attr is not None and equi_edge_attr.size(-1) > 0:
                equi_edge_attr = self.dropout(equi_edge_attr)
            if inv_edge_attr is not None and inv_edge_attr.size(-1) > 0:
                inv_edge_attr = self.dropout(inv_edge_attr)

        output = self._projection(
            edge_index, equi_edge_attr, inv_edge_attr, undirected_mask
        )
        if return_embeddings:
            return output, embeddings
        else:
            return output

    def _projection(self, edge_index, equi_edge_attr, inv_edge_attr, undirected_mask):
        """The output layer."""
        edge_attr = self.projection_head(
            edge_index=edge_index,
            edge_attr=(
                equi_edge_attr if self.orientation_equivariant_labels else inv_edge_attr
            ),
            undirected_mask=undirected_mask,
        ).squeeze()

        if self.classification:
            edge_attr = self.sigmoid(edge_attr)

        return edge_attr
