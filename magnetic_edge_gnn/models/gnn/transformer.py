import torch
import torch.nn as nn
from torch_geometric.data import Data

from magnetic_edge_gnn.models.gnn.base import BaseModel


class Transformer(BaseModel):
    """Transformer based on the Magnetic Edge Laplacian using its eigenvectors as positional encodings for the edges."""

    def __init__(
        self,
        equi_in_channels: int,
        inv_in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        *args,
        dropout: float = 0.0,
        act: str = "relu",
        classification: bool = False,
        num_pos_encodings: int = 32,
        num_heads: int = 8,
        q: int = 0,
        **kwargs,
    ):
        super().__init__()

        input_dim = equi_in_channels + inv_in_channels
        self.q = q
        self.num_pos_encodings = num_pos_encodings

        self.linear_enc = nn.Linear(4 * num_pos_encodings + input_dim, hidden_channels)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_channels,
                nhead=num_heads,
                activation=act,
                dropout=dropout,
            ),
            num_layers=num_layers,
        )
        self.linear_dec = nn.Linear(hidden_channels, out_channels)

        self.classification = classification

        if self.classification:
            self.sigmoid = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.cached_laplacian_eigenvectors = {}

    def forward(
        self,
        edge_index: torch.Tensor,
        equi_edge_attr: torch.Tensor,
        inv_edge_attr: torch.Tensor,
        undirected_mask: torch.Tensor,
        batch: Data,
        *args,
        return_embeddings=False,
        **kwargs,
    ) -> torch.Tensor:
        x = torch.cat(
            [equi_edge_attr, inv_edge_attr]
            + [
                torch.tensor(
                    enc, dtype=equi_edge_attr.dtype, device=equi_edge_attr.device
                )
                for enc in (
                    batch.eigvec_inv_real[..., : self.num_pos_encodings],
                    batch.eigvec_inv_imag[..., : self.num_pos_encodings],
                    batch.eigvec_equi_real[..., : self.num_pos_encodings],
                    batch.eigvec_equi_imag[..., : self.num_pos_encodings],
                )
            ],
            dim=-1,
        )

        x = self.linear_enc(x)
        x = self.transformer(x)
        x = self.linear_dec(x)

        if self.classification:
            x = self.sigmoid(x)
        return x
