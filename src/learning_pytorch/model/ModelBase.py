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
sys.path.append(os.path.dirname(FILEPATH))
from model import model_registry
import transform.augment_signal as T
act_fn_by_name = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "gelu": nn.GELU,
}

def unwrap_metrics(output):
    """
    Unwrap metrics from pytorch lightning metrics when they are not averaged in order
    to log them in tensorboard

    Input:
        output: dict of metrics
    Output:
        output: dict of flattened metrics
    """
    to_be_flattened = [x for x in output.items() if x[1].ndim > 0]
    for el in to_be_flattened:
        new_el = {f"{el[0]}_{idx}": el[1][idx] for idx in range(el[1].shape[0])}
        output.update(new_el)
        del output[el[0]]

    return output


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
        if model_hparams.get("augmenter_mag", None) not in [None,0]:
            str_ops = model_hparams.get("augmenter_ops",None) if model_hparams.get("augmenter_ops",None) is not None else "all operations"
            self.augmenter = T.RandAugment(magnitude = model_hparams["augmenter_mag"], augmentation_operations = model_hparams.get("augmenter_ops",None))
        else:
            self.augmenter = None
        # Create loss module

        if kwargs["loss_type"] == "binary_crossentropy":
            self.loss_module = nn.BCEWithLogitsLoss()
        elif kwargs["loss_type"] == "categorical_crossentropy":
            self.loss_module = nn.CrossEntropyLoss(label_smoothing =model_hparams.get("label_smoothing", 0.0))
        else:
            raise ValueError("Loss type not supported")

        # Example input for visualizing the graph in Tensorboard
        # if config['MANIPULATION']['channel_first'] == 'true':
        # self.example_input_array = torch.zeros((16,kwargs['n_features'] , kwargs['length_sequence']), dtype=torch.float32)
        # else:
        # self.example_input_array = torch.zeros((16,kwargs['length_sequence'],kwargs['n_features'] ), dtype=torch.float32)

        # Create metrics. If two classes compute precision, recall for each class else average
        if kwargs["n_classes"] == 2:
            metrics = MetricCollection(
                [
                    Accuracy(task="binary"),
                    Precision(task="binary",average=None, num_classes=kwargs["n_classes"]),
                    Recall(task="binary",average=None, num_classes=kwargs["n_classes"]),
                ]
            )
            self.train_metrics = MetricCollection([Accuracy(task="binary")], prefix="train_")
        else:
            metrics = MetricCollection([Accuracy(task="multiclass"), Precision(task="multiclass"), Recall(task="multiclass")])
            self.train_metrics = MetricCollection([Accuracy(task="multiclass")], prefix="train_")

        # self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def forward(self, imgs):
        # Forward function that is run when visualizing the graph
        return self.model(imgs)

    def configure_optimizers(self):
        print("Configuring optimizers:", self.hparams["optimizer_name"])
        # We will support Adam or SGD as optimizers.
        if self.hparams.optimizer_name == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here
            # for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.1,
                patience=8,
                min_lr=1e-6,
                verbose=True,
            )
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.1,
                patience=8,
                min_lr=1e-6,
                verbose=True,
            )
        elif self.hparams.optimizer_name == "radam":
            optimizer = optim.RAdam(self.parameters(), **self.hparams.optimizer_hparams)

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.1,
                patience=8,
                min_lr=1e-6,
                verbose=True,
            )
        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

        return [optimizer], {"scheduler": scheduler, "monitor": "ptl/val_loss"}

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        signal, labels = batch[self.signal_name], batch[self.target_name]
        if (self.augmenter != None):
            signal = self.augmenter(signal, self.current_epoch)

        if "covariates" in batch.keys():
            covariates = batch["covariates"]
            input = (signal, covariates)
            preds = self.model(input)
        else:
            preds = self.model(signal)

        if isinstance(
            self.loss_module,torch.nn.modules.loss.BCEWithLogitsLoss,
        ):
            loss = self.loss_module(preds, labels)
        else:
            labels = labels.argmax(dim=-1)
            loss = self.loss_module(preds, labels)
        output = self.train_metrics(preds, labels)
        # use log_dict instead of log
        # metrics are logged with keys: train_Accuracy, train_Precision and train_Recall
        output = {f"train/{k}": v for k, v in output.items()}
        self.log_dict(output)
        self.log("train/train_loss", loss)
        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        # self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        signal, labels = batch[self.signal_name], batch[self.target_name]
        if "covariates" in batch.keys():
            covariates = batch["covariates"]
            input = (signal, covariates)
            preds = self.model(input)
        else:
            preds = self.model(signal)

        if isinstance(
            self.loss_module,torch.nn.modules.loss.BCEWithLogitsLoss
        ):
            loss = self.loss_module(preds, labels)
        else:
            labels = labels.argmax(dim=-1)
            loss = self.loss_module(preds, labels)
        # By default logs it per epoch (weighted average over batches)

        # preds = preds.argmax(dim=-1)
        output = self.valid_metrics(preds, labels)
        if preds.shape[1] == 2:
            output = unwrap_metrics(output)
        output["val_loss"] = loss
        output = {f"val/{k}": v for k, v in output.items()}
        self.log_dict(output, on_step=False, on_epoch=True)
        return output

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val/val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val/val_Accuracy"] for x in outputs]).mean()
        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/val_Accuracy", avg_acc)

    def test_step(self, batch, batch_idx):
        signal, labels = batch[self.signal_name], batch[self.target_name]
        if "covariates" in batch.keys():
            covariates = batch["covariates"]
            input = (signal, covariates)
            preds = self.model(input)
        else:
            preds = self.model(signal)
        if isinstance(
            self.loss_module,torch.nn.modules.loss.BCEWithLogitsLoss
        ):
            loss = self.loss_module(preds, labels)
        else:
            labels = labels.argmax(dim=-1)
            loss = self.loss_module(preds, labels)
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        # if preds.shape[1] == 2:
        # preds = preds.argmax(dim=-1)
        output = self.test_metrics(preds, labels)
        if preds.shape[1] == 2:
            output = unwrap_metrics(output)
        output = {f"test/{k}": v for k, v in output.items()}
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

    if model_name in model_registry:
        return model_registry[model_name](model_hparams, **kwargs)
    else:
        assert (
            False
        ), f'Unknown model name "{model_name}". Available models are: {str(model_registry.keys())}'
