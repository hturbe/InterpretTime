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



# from focal_loss import BinaryFocalLoss

FILEPATH = os.path.dirname(os.path.realpath(__file__))
ROOTPATH = os.path.dirname(FILEPATH)
sys.path.append(os.path.join(FILEPATH))
sys.path.append(os.path.join(ROOTPATH))


from learning_pytorch.callbacks import create_callbacks
from learning_pytorch.custom_reader import DataLoader, BatchedDataLoader
from learning_pytorch.model.ModelBase import ModelBase
from learning_pytorch.data_module import PetastormDataModule as DataModule

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class Trainable:
    def __init__(self, config, results_path, final_run=False, retrieve_model=False):
        # Initializing state variables for the run
        self.config = config
        self.final_run = final_run
        self.results_path = results_path
        self.retrieve_model = retrieve_model
        self.model_type = self.config["MODEL"]["model_type"]
        self.loss = self.config["CONFIGURATION"]["loss"]

        # ml_data.transform_dict['min_size']
        # save encoder classes into simulation folder


    def train(self, config_model, checkpoint_dir=None, **kwargs):
        """
        Inputs:
            model_name - Name of the model you want to run. Is used to look up the class in "model_dict"
            save_name (optional) - If specified, this name will be used for creating the checkpoint and logging directory.
        """

        data_module_kwargs = {
            "config": self.config,
            "results_path": self.results_path,
            "num_train_epochs": self.config["CONFIGURATION"]["epochs"],
            "train_batch_size": config_model["batch_size"],
            "val_batch_size": config_model["batch_size"],
        }
        dataset = DataModule(**data_module_kwargs)
        dataset.setup()


        self.target_shape = dataset.target_shape
        self.len_seq = dataset.transform_dict["min_size"]

        model_name = self.config["MODEL"]["model_type"]
        desired_fields = [dataset.featureName, dataset.targetName]
        if (dataset.ordinal_covariates is not None) | (
            dataset.ordinal_covariates is not None
        ):
            desired_fields.append("covariates")

        save_name = None
        if save_name is None:
            save_name = model_name
        callbacks = create_callbacks(self.final_run, self.results_path)

        # Create a PyTorch Lightning trainer with the generation callback
        # ncpus, ngpus =  determine_cpus_gpus()
        # plugin = RayPlugin(num_workers=1, use_gpu=False)
        trainer = pl.Trainer(
            default_root_dir=self.results_path,  # Where to save models
            # We run on a single GPU (if possible)
            gpus=1 if str(device) == "cuda:0" else 0,
            # How many epochs to train for if no patience is set
            max_epochs=self.config["CONFIGURATION"]["epochs"],
            log_every_n_steps=1,
            callbacks=callbacks,
        )
        # plugins=[plugin],
        # logger=TensorBoardLogger(
        # save_dir=tune.get_trial_dir(), name="", version="."
        # ),
        # In case your notebook crashes due to the progress bar, consider increasing the refresh rate
        trainer.logger._log_graph = (
            True  # If True, we plot the computation graph in tensorboard
        )
        trainer.logger._default_hp_metric = (
            None  # Optional logging argument that we don't need
        )

        # Check whether pretrained model exists. If yes, load it and skip training

        if self.retrieve_model:
            path_model = sorted(glob(os.path.join(self.results_path, "*.ckpt")), key=os.path.getctime)[-1]
            print(f"Found pretrained model at {path_model}, loading...")
            # Automatically loads the model with the saved hyperparameters
            model = ModelBase.load_from_checkpoint(path_model)
            # target_name = self.ml_data.targetName
            # model.target_name = target_name
            # trainer.fit(model, datamodule=data_module)

        else:
            signal_name = dataset.featureName
            target_name = dataset.targetName
            model = ModelBase(
                signal_name=signal_name,
                target_name=target_name,
                model_hparams=config_model,
                optimizer_name=kwargs["optimizer_name"],
                optimizer_hparams=kwargs["optimizer_hparams"],
                length_sequence=self.len_seq,
                n_features=dataset.nb_features,
                n_classes=self.target_shape,
                loss_type=self.config["CONFIGURATION"]["loss"],
                samples_per_cls=dataset.samples_per_cls,
                dim_covariates = dataset.dim_covariates,
            )
            trainer.fit(model, datamodule=dataset)
            # Load best checkpoint after training
            model = ModelBase.load_from_checkpoint(
                trainer.checkpoint_callback.best_model_path
            )

        if self.final_run or self.retrieve_model:
            return model
