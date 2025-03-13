from abc import abstractmethod
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from eign.block import EIGNBlock


class EIGNOutput(NamedTuple):
    signed: torch.Tensor | None
    unsigned: torch.Tensor | None


class EIGN(nn.Module):
    def __init__(
        self,
        in_channels_signed: int | None,
        out_channels_signed: int | None,
        hidden_channels_signed: int,
        in_channels_unsigned: int,
        hidden_channels_unsigned: int,
        out_channels_unsigned: int,
        num_blocks: int,
        dropout: float = 0.1,
        signed_activation_fn=F.tanh,
        unsigned_activation_fn=F.relu,
        **kwargs_block,
    ):
        super().__init__()
        self.out_channels_signed = out_channels_signed
        self.out_channels_unsigned = out_channels_unsigned
        self.signed_activation_fn = signed_activation_fn
        self.unsigned_activation_fn = unsigned_activation_fn

        self.dropout = nn.Dropout(p=dropout)

        self.blocks = nn.ModuleList()
        _in_channels_signed = in_channels_signed
        _in_channels_unsigned = in_channels_unsigned

        for _ in range(num_blocks):
            block = self.initialize_block(
                in_channels_signed=_in_channels_signed,  # type: ignore
                out_channels_signed=hidden_channels_signed,
                in_channels_unsigned=_in_channels_unsigned,
                out_channels_unsigned=hidden_channels_unsigned,
                signed_activation_fn=signed_activation_fn,
                unsigned_activation_fn=unsigned_activation_fn,
                **kwargs_block,
            )
            self.blocks.append(block)
            _in_channels_signed = hidden_channels_signed
            _in_channels_unsigned = hidden_channels_unsigned

        if self.out_channels_signed:
            self.signed_head = nn.Linear(
                hidden_channels_signed,
                out_channels_signed,  # type: ignore
                bias=False,
            )
        else:
            self.register_buffer('signed_head', None)
        if self.out_channels_unsigned:
            self.unsigned_head = nn.Linear(
                hidden_channels_unsigned, out_channels_unsigned, bias=False
            )
        else:
            self.register_buffer('unsigned_head', None)

    @abstractmethod
    def initialize_block(
        self,
        in_channels_signed: int,
        out_channels_signed: int,
        in_channels_unsigned: int,
        out_channels_unsigned: int,
        signed_activation_fn=F.tanh,
        unsigned_activation_fn=F.relu,
        *args,
        **kwargs,
    ) -> EIGNBlock:
        raise NotImplementedError

    def forward(
        self,
        x_signed: torch.Tensor | None,
        x_unsigned: torch.Tensor | None,
        edge_index: torch.Tensor,
        is_directed: torch.Tensor,
        *args,
        **kwargs,
    ) -> EIGNOutput:
        for block in self.blocks:
            x_signed, x_unsigned = block(
                x_signed=x_signed,
                x_unsigned=x_unsigned,
                edge_index=edge_index,
                is_directed=is_directed,
            )

            if x_signed is not None:
                x_signed = self.signed_activation_fn(x_signed)
                x_signed = self.dropout(x_signed)
            if x_unsigned is not None:
                x_unsigned = self.unsigned_activation_fn(x_unsigned)
                x_unsigned = self.dropout(x_unsigned)

        if self.out_channels_signed:
            x_signed = self.signed_head(x_signed)
        else:
            x_signed = None
        if self.out_channels_unsigned:
            x_unsigned = self.unsigned_head(x_unsigned)
        else:
            x_unsigned = None

        return EIGNOutput(signed=x_signed, unsigned=x_unsigned)
