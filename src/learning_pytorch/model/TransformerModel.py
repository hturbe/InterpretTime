#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
  Transformer model 
  Based on: https://colab.research.google.com/github/PytorchLightning/lightning-tutorials/blob/publication/
            .notebooks/course_UvA-DL/05-transformers-and-MH-attention.ipynb#scrollTo=22b60df6
"""
import math
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder
from torch.nn.parameter import Parameter, UninitializedParameter
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """
        Args:
            input_dim: Dimensionality of the input
            num_heads: Number of heads to use in the attention block
            dim_feedforward: Dimensionality of the hidden layer in the MLP
            dropout: Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.attn = nn.MultiheadAttention(input_dim, num_heads)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, input_dim),
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        inp_x = self.norm1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear_net(self.norm2(x))

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderBlock(**block_args) for _ in range(num_layers)]
        )

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Args
            d_model: Hidden dimensionality of the input.
            max_len: Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class TransformerModel(nn.Module):
    def __init__(self, hparams, **kwargs):
        super().__init__()
        self.length_sequence = kwargs["length_sequence"]
        self.n_features = kwargs["n_features"]
        self.n_classes = kwargs["n_classes"]

        self.hparams = SimpleNamespace(
            num_classes=kwargs["n_classes"],
            dropout=hparams["dropout"],
            input_dropout=hparams.get("input_dropout"),
            num_layers=hparams.get("num_layers"),
            num_heads=hparams.get("num_heads"),
            model_dim=hparams.get("model_dim"),
            mlp_dim=hparams.get("mlp_dim"),
        )
        self.embedding_type = "positional"
        self._create_network()

    def _create_network(self):

        # Positional encoding for sequences
        if self.embedding_type == "positional":
            self.positional_encoding = PositionalEncoding(
                d_model=self.hparams.model_dim, max_len=self.length_sequence
            )
            self.input_feature = self.n_features
        else:
            raise ValueError(
                "Unknown embedding type. Should be one of [positional, time2vec]"
            )
        # Input dim -> Model dim
        self.input_net = nn.Sequential(
            nn.Dropout(self.hparams.input_dropout),
            nn.Linear(self.input_feature, self.hparams.model_dim),
        )
        # Transformer
        self.transformer = TransformerEncoder(
            num_layers=self.hparams.num_layers,
            input_dim=self.hparams.model_dim,
            dim_feedforward=2 * self.hparams.model_dim,
            num_heads=self.hparams.num_heads,
            dropout=self.hparams.dropout,
        )
        # Output classifier per sequence lement
        self.output_net = nn.Sequential(
            nn.Linear(self.length_sequence, self.hparams.mlp_dim),
            nn.LayerNorm(self.hparams.mlp_dim),
            nn.ReLU(),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.mlp_dim, self.hparams.num_classes),
        )
        # self.temp_pool = nn.AdaptiveMaxPool1d((1))
        self.temp_pool = nn.Sequential(nn.AdaptiveAvgPool1d((1)), nn.Flatten())

    def forward(self, x, mask=None, add_positional_encoding=True):
        """
        Args:
            x: Input features of shape [Batch, SeqLen, input_dim]
            mask: Mask to apply on the attention outputs (optional)
            add_positional_encoding: If True, we add the positional encoding to the input.
                                      Might not be desired for some tasks.
        """
        # x = self.input_net(x)
        if (add_positional_encoding) & (self.embedding_type == "positional"):
            # time_embedding = self.positional_encoding(x)
            # x = torch.cat([x,time_embedding],-1)
            x = self.input_net(x)
            x = self.positional_encoding(x)
        elif (add_positional_encoding) & (self.embedding_type == "time2vec"):
            x = self.positional_encoding(x)
            x = self.input_net(x)

        x = self.transformer(x, mask=mask)
        x = self.temp_pool(x)
        x = self.output_net(x)
        return x
