from .eign import EIGN
from eign.block import EIGNBlockMagneticEdgeLaplacianConv, EIGNBlock

import torch.nn.functional as F

class EIGNLaplacianConv(EIGN):
    
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
        return EIGNBlockMagneticEdgeLaplacianConv(
            in_channels_signed=in_channels_signed,
            out_channels_signed=out_channels_signed,
            in_channels_unsigned=in_channels_unsigned,
            out_channels_unsigned=out_channels_unsigned,
            signed_activation_fn=signed_activation_fn,
            unsigned_activation_fn=unsigned_activation_fn,
            **kwargs,
        )