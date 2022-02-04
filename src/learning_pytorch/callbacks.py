#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
  Model Callbacks
  Written by H.Turbé, May 2021.
"""

import logging
import os

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# from ray_lightning.tune import TuneReportCallback
from ray.tune.integration.pytorch_lightning import (
    TuneReportCallback,
    TuneReportCheckpointCallback,
)

logger = logging.getLogger("hyper_search")


def create_callbacks(final_run, path_results=None):
    callbacks = []

    lr_rate_monitor = LearningRateMonitor("epoch")  # Log learning rate every epoch
    earlystopping = EarlyStopping(
        monitor="ptl/val_loss", patience=25, min_delta=1e-3, mode="min"
    )
    callbacks.append(earlystopping)
    callbacks.append(lr_rate_monitor)

    # Creating early stopping callback

    if final_run:
        # Save model if not running hyperparameter search
        logger.info("Creating model checkpoint callback")
        print("Creating model checkpoint callback")

        # Creating model checkpoint callback
        checkpoint = ModelCheckpoint(
            save_weights_only=True,
            mode="min",
            monitor="ptl/val_loss",
            dirpath=path_results,
            filename="best_model",
        )  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
        callbacks.append(checkpoint)

    # Creating ray tune report callback
    else:
        logger.info("Creating tune reporter callback")
        print("Creating tune reporter callback")
        # Creating ray callback which reports metrics of the ongoing run
        # We choose to report metrics after epoch end using freq="epoch"
        # because val_loss is calculated just before the end of epoch
        # tune_reporter = TuneReporter(freq="epoch")
        tune_reporter = TuneReportCallback(
            {"val_loss": "ptl/val_loss", "mean_accuracy": "ptl/val_Accuracy"},
            on="validation_end",
        )

        callbacks.append(tune_reporter)

    return callbacks
