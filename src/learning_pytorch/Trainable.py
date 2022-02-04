#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Written by H.Turbé, August 2021.
    Include class to train a network and do hyperparameter search
    Inspired by https://github.com/himanshurawlani/hyper_fcn/blob
    /0252e6809bfa97c6791e44a2b3ca96c8e3bb20e6/generator.py#L11
    
"""
import copy
import math
import os
import sys
from glob import glob

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

FILEPATH = os.path.dirname(os.path.realpath(__file__))
ROOTPATH = os.path.dirname(FILEPATH)
sys.path.append(os.path.join(FILEPATH))
sys.path.append(os.path.join(ROOTPATH))

model_path = os.path.join(FILEPATH, "saved_model")
if not os.path.exists(model_path):
    os.makedirs(model_path)

from ML_Dataset import ML_Dataset

from learning_pytorch.callbacks import create_callbacks
from learning_pytorch.custom_reader import DataLoader
from learning_pytorch.model.ModelBase import ModelBase

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class DataModule(pl.LightningDataModule):
    """
    Pytorch datamodule to interface with Petastorm dataset
    """

    def __init__(
        self,
        train_ML,
        val_ML,
        test_ML,
        desired_fields=None,
        batch_size=64,
        num_workers=40,
        shard_count=8,
    ):
        super().__init__()
        self.train_reader = train_ML
        self.val_reader = val_ML
        self.test_reader = test_ML
        self.batch_size = batch_size
        self.desired_fields = desired_fields

    def train_dataloader(self):
        # train_reader, train_len = self.train_ML.generate_ML_reader("train", num_epochs = 1)
        if self.train_reader.last_row_consumed:
            self.train_reader.reset()
        return DataLoader(
            self.train_reader,
            batch_size=self.batch_size,
            desired_fields=self.desired_fields,
        )

    def val_dataloader(self):
        # val_reader, val_len = self.val_ML.generate_ML_reader("val", num_epochs = 1)
        if self.val_reader.last_row_consumed:
            self.val_reader.reset()
        return DataLoader(
            self.val_reader,
            batch_size=self.batch_size,
            desired_fields=self.desired_fields,
        )

    def test_dataloader(self):
        # test_reader, test_len = self.test_ML.generate_ML_reader("test", num_epochs = 1)
        if self.test_reader.last_row_consumed:
            self.test_reader.reset()
        return DataLoader(
            self.test_reader,
            batch_size=self.batch_size,
            desired_fields=self.desired_fields,
        )


class Trainable:
    """
    Class to train a model with pytorch lightning
    """

    def __init__(self, config, results_path, final_run=False, retrieve_model=False):
        # Initializing state variables for the run
        self.config = config
        self.final_run = final_run
        self.results_path = results_path
        self.retrieve_model = retrieve_model
        self.model_type = self.config["MODEL"]["model_type"]
        self.loss = self.config["CONFIGURATION"]["loss"]

        self.ml_data = ML_Dataset(self.config)
        target_transformed, classes_name = self.ml_data.generate_encoder()
        if config["MANIPULATION"].get("feature_scaling", "none").lower() != "none":
            self.ml_data.generate_scaler()
        self.target_shape = target_transformed.shape[1]
        self.len_seq = self.ml_data.transform_dict["min_size"]
        np.save(os.path.join(self.results_path, "classes_encoder.npy"), classes_name)

    def generate_dataset(self):

        train_ML = copy.deepcopy(self.ml_data)
        train_reader, _ = train_ML.generate_ML_reader("train", num_epochs=1)
        val_ML = copy.deepcopy(self.ml_data)
        val_reader, _ = train_ML.generate_ML_reader("val", num_epochs=1)
        test_ML = copy.deepcopy(self.ml_data)
        test_reader, _ = train_ML.generate_ML_reader("test", num_epochs=1)

        return train_ML, val_ML, test_ML, train_reader, val_reader, test_reader

    def generate_MLdataset(self, batch_size, desired_fields=None):
        (
            train_ML,
            val_ML,
            test_ML,
            train_reader,
            val_reader,
            test_reader,
        ) = self.generate_dataset()

        data_module = DataModule(
            batch_size=batch_size,
            train_ML=train_reader,
            val_ML=val_reader,
            test_ML=test_reader,
            desired_fields=desired_fields,
        )

        return data_module

    def train(self, config_model, checkpoint_dir=None, **kwargs):
        """
        Inputs:
            model_name - Name of the model you want to run. Is used to look up the class in "model_dict"
            save_name (optional) - If specified, this name will be used for creating the checkpoint and logging directory.
        """

        model_name = self.config["MODEL"]["model_type"]
        data_module = self.generate_MLdataset(
            config_model["batch_size"],
            desired_fields=[self.ml_data.featureName, self.ml_data.targetName],
        )

        for key in config_model:
            if key == "batch_size":
                key_tmp = "batch_size_config"
            else:
                key_tmp = key

        save_name = None
        if save_name is None:
            save_name = model_name
        print("PATH", os.path.join(FILEPATH, save_name))
        callbacks = create_callbacks(self.final_run, self.results_path)

        trainer = pl.Trainer(
            default_root_dir=self.results_path,  # Where to save models
            # We run on a single GPU (if possible)
            gpus=1 if str(device) == "cuda:0" else 0,
            reload_dataloaders_every_n_epochs=1,
            # How many epochs to train for if no patience is set
            max_epochs=self.config["CONFIGURATION"]["epochs"],
            log_every_n_steps=1,
            progress_bar_refresh_rate=1,
            callbacks=callbacks,
        )

        # In case your notebook crashes due to the progress bar, consider increasing the refresh rate
        trainer.logger._log_graph = (
            True  # If True, we plot the computation graph in tensorboard
        )
        trainer.logger._default_hp_metric = (
            None  # Optional logging argument that we don't need
        )

        # Check whether pretrained model exists. If yes, load it and skip training
        if self.retrieve_model:
            path_model = os.path.join(self.results_path, "best_model.ckpt")
            print(f"Found pretrained model at {path_model}, loading...")
            # Automatically loads the model with the saved hyperparameters
            model = ModelBase.load_from_checkpoint(path_model)

        else:
            signal_name = self.ml_data.featureName
            target_name = self.ml_data.targetName
            model = ModelBase(
                signal_name=signal_name,
                target_name=target_name,
                model_hparams=config_model,
                optimizer_name=kwargs["optimizer_name"],
                optimizer_hparams=kwargs["optimizer_hparams"],
                length_sequence=self.len_seq,
                n_features=self.ml_data.n_features,
                n_classes=self.target_shape,
                loss_type=self.config["CONFIGURATION"]["loss"],
            )
            trainer.fit(model, datamodule=data_module)
            # Load best checkpoint after training
            model = ModelBase.load_from_checkpoint(
                trainer.checkpoint_callback.best_model_path
            )

        if self.final_run or self.retrieve_model:
            return model
