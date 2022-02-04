#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
  BiLSTM  implementation in pytorch
  Written by H.Turbé, May 2021.
"""
import os
import re
import sys
from types import SimpleNamespace

import torch.nn as nn

FILEPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FILEPATH))
act_fn_by_name = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "gelu": nn.GELU,
}


class BiLstmModel(nn.Module):
    def __init__(self, hparams, **kwargs):
        super().__init__()

        self.length_sequence = kwargs["length_sequence"]
        self.n_features = kwargs["n_features"]
        self.n_classes = kwargs["n_classes"]

        self.hparams = SimpleNamespace(
            num_classes=kwargs["n_classes"],
            filter_arrays=hparams["cell_array"],
            dropout=hparams["dropout"],
            act_fn_name=hparams["act_fn_name"],
            act_fn=hparams["act_fn"],
        )

        self._create_network()

    def _create_network(self):

        blocks = []
        dropout = self.hparams.dropout
        for idx, filters in enumerate(self.hparams.filter_arrays):
            if idx == 0:  # Case for input layer
                blocks.append(
                    nn.Sequential(
                        LSTM_wraper(
                            input_size=self.n_features,
                            dropout=dropout,
                            hidden_size=filters,
                            bidirectional=True,
                            batch_first=True,
                        )
                    )
                )
            elif idx == len(self.hparams.filter_arrays) - 1:
                # Case for last lstm layer where we only return last hidden state
                blocks.append(
                    nn.Sequential(
                        LSTM_wraper(
                            input_size=self.hparams.filter_arrays[idx - 1] * 2,
                            dropout=dropout,
                            hidden_size=filters,
                            bidirectional=True,
                            batch_first=True,
                            return_last=True,
                        )
                    )
                )
            else:
                # Case for all other layers
                blocks.append(
                    nn.Sequential(
                        LSTM_wraper(
                            input_size=self.hparams.filter_arrays[idx - 1] * 2,
                            dropout=dropout,
                            hidden_size=filters,
                            bidirectional=True,
                            batch_first=True,
                        )
                    )
                )

        self.blocks = nn.Sequential(*blocks)

        self.output_net = nn.Sequential(
            nn.Linear(self.hparams.filter_arrays[-1] * 2, 256),
            nn.ReLU(),
            nn.Linear(256, self.hparams.num_classes),
        )

    def forward(self, inputs):
        x = self.blocks(inputs)
        output = self.output_net(x)
        return output


class LSTM_wraper(nn.Module):
    """
    Wrapper function for the LSTM layer to return only the hidden states or the last one and apply optional dropout
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_first=True,
        dropout=0,
        bidirectional=False,
        return_last=False,
    ):
        """
        Parameters:
            input_size: int
                input size of the LSTM layer
            hidden_size: int
            batch_first: bool
            dropout: float
            bidirectional: bool
            return_last: bool
                whether only the last hidden state should be returned (used for the last layer before the output layer)

        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.dropout = nn.Dropout(p=dropout)
        self.bidirectional = bidirectional
        self.return_last = return_last

        self._create_network()

    def _create_network(self):
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            bidirectional=self.bidirectional,
            batch_first=self.batch_first,
        )

    def forward(self, inputs):
        h, (h_T, c_T) = self.lstm(inputs)

        if self.return_last:
            return self.dropout(h[:, -1, :])
        else:
            return self.dropout(h)
