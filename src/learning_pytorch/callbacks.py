import os
import logging
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
# from ray_lightning.tune import TuneReportCallback
logger = logging.getLogger('hyper_search')


def create_callbacks(final_run, path_results = None):
    callbacks = []

    lr_rate_monitor=LearningRateMonitor("epoch") # Log learning rate every epoch
    earlystopping = EarlyStopping(monitor= "ptl/val_loss",  patience=20, min_delta=1e-7,
                             mode='min')
    callbacks.append(earlystopping)
    callbacks.append(lr_rate_monitor)

    callbacks.append( TQDMProgressBar(refresh_rate=1))


    # Creating early stopping callback

    if final_run:
        logger.info("Creating model checkpoint callback")
        print("Creating model checkpoint callback")
        # Make sure the 'snapshots' directory exists
        # os.makedirs(model_path, exist_ok=True)

        # Creating model checkpoint callback
        checkpoint = ModelCheckpoint(
                        save_weights_only=True, mode="min", monitor="ptl/val_loss",
                        dirpath = path_results, filename='best_model'
                ) # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
        callbacks.append(checkpoint)



        # callbacks += [keras.callbacks.LearningRateScheduler(partial(lr_scheduler)]

    return callbacks
