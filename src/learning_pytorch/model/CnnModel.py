#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
  CNN  implementation
"""
import os
import sys

import torch
import torch.nn as nn
from types import SimpleNamespace
FILEPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FILEPATH))
act_fn_by_name = {"tanh": nn.Tanh, "relu": nn.ReLU, "leakyrelu": nn.LeakyReLU, "gelu": nn.GELU}

class CnnModel(nn.Module):
    def __init__(self, hparams, **kwargs):
        super().__init__()

        self.length_sequence = kwargs["length_sequence"]
        self.n_features = kwargs["n_features"]
        self.n_classes = kwargs["n_classes"]

        self.hparams = SimpleNamespace(
            num_classes=kwargs['n_classes'],
            filter_arrays = hparams["cell_array"],
            dropout = hparams.get("dropout",0.0),
            kernel_size = hparams["kernel_size"],
            act_fn_name=hparams['act_fn_name'],
            act_fn=hparams['act_fn'],
        )

        self._create_network()

    def _create_network(self):
        blocks = []
        kernel_size = self.hparams.kernel_size
        dropout = self.hparams.dropout
        for idx,filters in enumerate(self.hparams.filter_arrays):
            if idx==0:
                blocks.append(nn.Sequential(
                    nn.Conv1d(
                        in_channels = self.n_features,
                        out_channels=filters,
                        kernel_size=kernel_size,
                        stride=1,
                    ), self.hparams.act_fn(),
                    nn.MaxPool1d(kernel_size=2),
                    nn.Dropout(p=dropout)
                ))
            
            else:
                blocks.append(nn.Sequential(
                    nn.Conv1d(
                        in_channels=self.hparams.filter_arrays[idx-1],
                        out_channels=filters,
                        kernel_size=kernel_size,
                        stride=1,
                    ), self.hparams.act_fn(),
                    nn.MaxPool1d(kernel_size=2),
                    nn.Dropout(p=dropout)
                ))
        
        self.blocks = nn.Sequential(*blocks)

        self.output_net = nn.Sequential(
            nn.AdaptiveAvgPool1d((1)), nn.Flatten(), nn.Linear(self.hparams.filter_arrays[-1], 
            self.hparams.num_classes)
        )
        
    def forward(self,x):
        x = self.blocks(x)
        x = self.output_net(x)

        return x

 
