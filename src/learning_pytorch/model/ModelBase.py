#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Model constructor with common routines
    Written by H.Turbé, May 2021.
"""
import os
import sys

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy, MetricCollection, Precision, Recall

FILEPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FILEPATH))


from BiLstmModel import BiLstmModel
from CnnModel import CnnModel
from TransformerModel import TransformerModel

model_dict = {
    "cnn": CnnModel,
    "bilstm": BiLstmModel,
    "transformer": TransformerModel,
}

act_fn_by_name = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "gelu": nn.GELU,
}


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class ModelBase(pl.LightningModule):
    def __init__(
        self,
        signal_name,
        target_name,
        model_hparams,
        optimizer_name,
        optimizer_hparams,
        **kwargs,
    ):
        """
        Inputs:
            model_name - Name of the model/CNN to run. Used for creating the model (see function below)
            model_hparams - Hyperparameters for the model, as dictionary.
            optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """

        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        model_hparams["act_fn"] = act_fn_by_name[model_hparams["act_fn_name"]]
        self.save_hyperparameters()
        # Create model
        self.model = create_model(model_hparams, **kwargs)
        self.signal_name = signal_name
        self.target_name = target_name

        # Create loss module

        if kwargs["loss_type"] == "binary_crossentropy":
            print("Loss type", "binary_crossentropy")
            self.loss_module = nn.BCEWithLogitsLoss()
        elif kwargs["loss_type"] == "categorical_crossentropy":
            self.loss_module = nn.CrossEntropyLoss()
        else:
            raise ValueError("Loss type not supported")

        # Create metrics
        metrics = MetricCollection([Accuracy(), Precision(), Recall()])
        self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def forward(self, imgs):
        # Forward function that is run when visualizing the graph
        return self.model(imgs)

    def configure_optimizers(self):
        # Support Adam or SGD as optimizers.
        if self.hparams.optimizer_name == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here
            # for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.1,
                patience=10,
                min_lr=1e-6,
                verbose=True,
            )
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.1,
                patience=10,
                min_lr=1e-6,
                verbose=True,
            )
        elif self.hparams.optimizer_name == "cosine_warmup":
            p = nn.Parameter(torch.empty(4, 4))
            optimizer = optim.Adam([p], lr=1e-3)
            scheduler = CosineWarmupScheduler(
                optimizer=optimizer, warmup=20, max_iters=500
            )
        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

        return [optimizer], {"scheduler": scheduler, "monitor": "ptl/val_loss"}

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        signal, labels = batch[self.signal_name], batch[self.target_name]
        preds = self.model(signal)
        if isinstance(self.loss_module, torch.nn.modules.loss.BCEWithLogitsLoss):
            loss = self.loss_module(preds, labels.to(dtype=torch.float32))

        else:
            labels = labels.argmax(dim=-1)
            loss = self.loss_module(preds, labels)
        output = self.train_metrics(preds, labels)
        # use log_dict instead of log
        # metrics are logged with keys: train_Accuracy, train_Precision and train_Recall
        self.log_dict(output)
        self.log("train_loss", loss)
        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        # self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        signal, labels = batch[self.signal_name], batch[self.target_name]
        preds = self.model(signal)
        if isinstance(self.loss_module, torch.nn.modules.loss.BCEWithLogitsLoss):
            loss = self.loss_module(preds, labels.to(dtype=torch.float32))

        else:
            labels = labels.argmax(dim=-1)
            loss = self.loss_module(preds, labels)
        # By default logs it per epoch (weighted average over batches)
        output = self.valid_metrics(preds, labels)
        output["val_loss"] = loss

        return output

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_Accuracy"] for x in outputs]).mean()
        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/val_Accuracy", avg_acc)

    def test_step(self, batch, batch_idx):
        signal, labels = batch[self.signal_name], batch[self.target_name]
        preds = self.model(signal)
        preds = self.model(signal)
        if isinstance(self.loss_module, torch.nn.modules.loss.BCEWithLogitsLoss):
            loss = self.loss_module(preds, labels.to(dtype=torch.float32))

        else:
            labels = labels.argmax(dim=-1)
            loss = self.loss_module(preds, labels)
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        output = self.test_metrics(preds, labels)
        self.log_dict(output)


def create_model(model_hparams, **kwargs):
    model_name = model_hparams["model_type"].lower()
    # we deal with situations where recomended config has 0 units for 3rd layer
    cell_config_key = [x for x in model_hparams.keys() if "cell_nb" in x]
    # For HP optimisation we predict each cell nb independently and last one can be 0
    if len(cell_config_key) > 0:
        cell_config_key = sorted(cell_config_key)
        model_hparams["cell_array"] = [
            model_hparams[x] for x in cell_config_key if model_hparams[x] > 0
        ]

    if model_name in model_dict:
        return model_dict[model_name](model_hparams, **kwargs)
    else:
        assert (
            False
        ), f'Unknown model name "{model_name}". Available models are: {str(model_dict.keys())}'
