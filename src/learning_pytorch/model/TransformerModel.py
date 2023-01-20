#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
  Transformer model 
  Based on: https://colab.research.google.com/github/PytorchLightning/lightning-tutorials/blob/publication/
            .notebooks/course_UvA-DL/05-transformers-and-MH-attention.ipynb#scrollTo=22b60df6
and https://github.com/bh1995/AF-classification/blob/cabb932d27c63ea493a97770f4b136c28397117f/src/models/model_utils.py#L17
and https://github.com/gzerveas/mvts_transformer
"""
import math
import os
import sys
from types import SimpleNamespace
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import (
    BatchNorm1d,
    Dropout,
    Linear,
    MultiheadAttention,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from torch.nn.parameter import Parameter, UninitializedParameter


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))


# From https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class FixedPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer(
            "pe", pe
        )  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(
            torch.empty(max_len, 1, d_model)
        )  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding

    raise NotImplementedError(
        "pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding)
    )


class TransformerBatchNormEncoderLayer(nn.modules.Module):
    r"""This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"
    ):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = BatchNorm1d(
            d_model, eps=1e-5
        )  # normalizes each feature across batch samples and time steps
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(TransformerBatchNormEncoderLayer, self).__setstate__(state)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        # src = src.reshape([src.shape[0], -1])  # (batch_size, seq_length * d_model)
        src = self.norm1(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        return src


class TransformerModel(nn.Module):
    def __init__(self, hparams, **kwargs):
        super(TransformerModel, self).__init__()
        self.model_type = "Transformer"

        self.length_sequence = kwargs["length_sequence"]
        self.n_features = kwargs["n_features"]
        self.n_classes = kwargs["n_classes"]

        self.hparams = SimpleNamespace(
            dropout=hparams["dropout"],
            d_model=hparams["d_model"],
            nhead=hparams["nhead"],
            dim_feedforward=hparams["dim_feedforward"],
            nlayers=hparams["nlayers"],
            mlp_dim=hparams["mlp_dim"],
        )
        pos_encoding = "fixed"
        freeze = False
        max_len = 5000
        norm = "BatchNorm"
        activation = "gelu"
        self.project_inp = nn.Linear(self.n_features, self.hparams.d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(
            self.hparams.d_model,
            dropout=self.hparams.dropout * (1.0 - freeze),
            max_len=max_len,
        )

        self.conv1 = torch.nn.Conv1d(
            in_channels=self.n_features,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = torch.nn.Conv1d(
            in_channels=128,
            out_channels=self.hparams.d_model,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        if norm == "LayerNorm":
            encoder_layer = TransformerEncoderLayer(
                self.hparams.d_model,
                self.hparams.nhead,
                self.hparams.dim_feedforward,
                self.hparams.dropout * (1.0 - freeze),
                activation=activation,
            )
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(
                self.hparams.d_model,
                self.hparams.nhead,
                self.hparams.dim_feedforward,
                self.hparams.dropout * (1.0 - freeze),
                activation=activation,
            )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, self.hparams.nlayers
        )

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(self.hparams.dropout)
        self.output_layer = self.build_output_module()

    def build_output_module(self):
        output_layer = nn.Sequential(
            nn.Linear(self.hparams.d_model * self.length_sequence, self.hparams.mlp_dim),
            nn.ReLU(),
            self.dropout1,
            nn.Linear(self.hparams.mlp_dim, self.n_classes),
        )
        # no softmax (or log softmax), because CrossEntropyLoss does this internally. If probabilities are needed,
        # add F.log_softmax and use NLLoss
        return output_layer

    def forward(self, X):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, num_classes)
        """
        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp = self.conv1(X)
        inp = self.conv2(inp) * math.sqrt(self.hparams.d_model)
        inp = inp.permute(2, 0, 1)
        # inp = self.project_inp(inp)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.pos_enc(inp)  # add positional encoding

        output = self.transformer_encoder(inp)
        output = self.act(
            output
        )  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)

        # Output
        # output = output * padding_masks.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(
            output.shape[0], -1
        )  # (batch_size, seq_length * d_model)
        output = self.output_layer(output)  # (batch_size, num_classes)

        return output
