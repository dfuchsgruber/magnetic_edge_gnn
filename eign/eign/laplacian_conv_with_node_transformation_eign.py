from .eign import EIGN
from eign.block import EIGNBlockMagneticEdgeLaplacianWithNodeTransformationConv, EIGNBlock

import torch.nn as nn
import torch.nn.functional as F

class EIGNLaplacianWithNodeTransformationConv(EIGN):
    
    def initialize_block(self,
                            in_channels_signed: int,
                            out_channels_signed: int,
                            in_channels_unsigned: int,
                            out_channels_unsigned: int,
                            signed_activation_fn=F.tanh,
                            unsigned_activation_fn=F.relu,
                            *args,
                            **kwargs,
                         ) -> EIGNBlock:
        
        kwargs.setdefault("initialize_node_feature_transformation", self.initialize_node_feature_transformation)
        return EIGNBlockMagneticEdgeLaplacianWithNodeTransformationConv(
            in_channels_signed=in_channels_signed,
            out_channels_signed=out_channels_signed,
            in_channels_unsigned=in_channels_unsigned,
            out_channels_unsigned=out_channels_unsigned,
            signed_activation_fn=signed_activation_fn,
            unsigned_activation_fn=unsigned_activation_fn,
            **kwargs,
        )
        
    @classmethod
    def initialize_node_feature_transformation(cls, in_channels: int, out_channels: int, **kwargs) -> nn.Module:
        """Initialize the node feature transformation layer."""
        return nn.Sequential(
            nn.ReLU(),
        )