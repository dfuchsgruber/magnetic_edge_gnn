"""Block that uses the Mangnetic Edge Laplacian as graph shift operator."""

from .block import EIGNBlock
from eign.conv import MagneticEdgeLaplacianWithNodeTransformationConv
import torch.nn as nn

class EIGNBlockMagneticEdgeLaplacianWithNodeTransformationConv(EIGNBlock):
    r"""Block within the EIGN architecture that models signed (orientation signedvariant) and unsigned (orientation unsignedariant) modalities using the Magnetic Edge Laplacian as graph shift operator."""
    
    def initialize_convolution(
        self,
        in_channels: int,
        out_channels: int,
        signed_in: bool,
        signed_out: bool,
        **kwargs,
    ) -> nn.Module:
        return MagneticEdgeLaplacianWithNodeTransformationConv(
            in_channels=in_channels,
            out_channels=out_channels,
            signed_in=signed_in,
            signed_out=signed_out,
            **kwargs,
        )
    
    
    