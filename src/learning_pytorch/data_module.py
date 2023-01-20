#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Written by H.Turbé, August 2021.
    Datamodule for interface between petastorm and pytorch lightning.
    Inspired by implementation from Horovod:
    https://github.com/horovod/horovod/blob/master/horovod/spark/lightning/datamodule.py

"""
import math
import os
import sys
import time
from collections.abc import Iterable
from functools import reduce
from operator import attrgetter, itemgetter

# from petastorm.unischema import Unischema, UnischemaField
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from petastorm import TransformSpec, make_batch_reader, make_reader
from petastorm.codecs import NdarrayCodec, ScalarCodec
from petastorm.predicates import in_lambda, in_pseudorandom_split, in_reduce
from petastorm.unischema import UnischemaField
from pyspark.sql.types import IntegerType, StringType
from scipy.sparse import data
from sklearn import preprocessing

# from horovod.spark.common import constants


FILEPATH = os.path.dirname(os.path.realpath(__file__))
ROOTPATH = os.path.dirname(FILEPATH)
sys.path.append(FILEPATH)
sys.path.append(os.path.join(ROOTPATH))


# custom libraries
import shared_utils.utils_data as utils_data
from shared_utils.scaler import TS_Scaler

from horovod_subset.pytorch_data_loaders import (
    PytorchInfiniteAsyncDataLoader,
    PytorchInmemAsyncDataLoader,
)


class PetastormDataModule(pl.LightningDataModule):
    """Default DataModule for Lightning Estimator"""

    def __init__(
        self,
        config: str,
        results_path: str,
        num_train_epochs: int = 1,
        has_val: bool = True,
        has_test: bool = True,
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        shuffle_size: int = 1000,
        num_reader_epochs=None,
        reader_pool_type: str = "process",
        reader_worker_count: int = 2,
        inmemory_cache_all=False,
        cur_shard: int = 0,
        shard_count: int = 1,
        schema_fields=None,
        storage_options=None,
        verbose=False,
        debug_data_loader: bool = False,
        train_async_data_loader_queue_size: int = 0,
        val_async_data_loader_queue_size: int = 0,
        **kwargs,
    ):
        super().__init__()

        self.results_path = results_path
        self.num_train_epochs = num_train_epochs
        self.has_val = has_val
        self.has_test = has_test
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.shuffle_size = shuffle_size
        self.num_reader_epochs = num_reader_epochs
        self.reader_pool_type = reader_pool_type
        self.reader_worker_count = reader_worker_count
        self.inmemory_cache_all = inmemory_cache_all
        self.cur_shard = cur_shard
        self.shard_count = shard_count
        self.schema_fields = schema_fields
        self.storage_options = storage_options
        # self.steps_per_epoch_train = steps_per_epoch_train
        # self.steps_per_epoch_val = steps_per_epoch_val
        self.verbose = verbose
        self.debug_data_loader = debug_data_loader
        self.train_async_data_loader_queue_size = train_async_data_loader_queue_size
        self.val_async_data_loader_queue_size = val_async_data_loader_queue_size

        self.config = config
        self.featureName = self.config["CONFIGURATION"]["feature"]
        self.targetName = self.config["CONFIGURATION"]["target"]

        self.selected_classes = self.config["CONFIGURATION"].get("selected_classes")
        self.dim_covariates = 0
        self.ordinal_covariates = self.config["CONFIGURATION"].get("ordinal_covariates")
        self.categorical_covariates = self.config["CONFIGURATION"].get(
            "categorical_covariates"
        )

        self.data_path = os.path.expanduser(self.config["CONFIGURATION"]["data_path"])
        self.data_path = os.path.abspath(self.data_path)

        self.data_pathPetastorm = f"file://{self.data_path}"

        if debug_data_loader:
            print("Creating data_module")

    def setup(self, stage=None):

        # if self.has_setup_fit:
        # return
        _ = self.evaluate_nbSample()
        _ = self.evaluate_nb_features()
        self.list_transformReader()
        self.generate_encoder()

        # Initialse scaler for TS
        if self.config["MANIPULATION"].get("feature_scaling", "none").lower() != "none":
            self.generate_scaler()

        # Initialise scaler and encoder for covariates
        if (self.config["CONFIGURATION"].get("categorical_covariates") is not None) | (
            self.config["CONFIGURATION"].get("ordinal_covariates") is not None
        ):
            classes_encoder_covariates = self.initialise_covariates()

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            # transform_spec = (
            # TransformSpec(self.transformation) if self.transformation else None
            # )
            # In general, make_batch_reader is faster than make_reader for reading the dataset.
            # However, we found out that make_reader performs data transformations much faster than
            # make_batch_reader with parallel worker processes. Therefore, the default reader
            # we choose is make_batch_reader unless there are data transformations.
            reader_factory_kwargs = dict()
            # if transform_spec:
            reader_factory = make_reader
            reader_factory_kwargs["pyarrow_serialize"] = True
            # else:
            # reader_factory = make_batch_reader
            (
                self.nb_sample_set_train,
                predicate_expr_train,
                transform_train,
            ) = self.generate_ML_reader_predicate_transform(partition="train")
            self.steps_per_epoch_train = int(
                math.floor(float(self.nb_sample_set_train) / self.train_batch_size)
            )

            self.ml_fields = [self.featureName, self.targetName]
            if (self.ordinal_covariates is not None) | (
                self.ordinal_covariates is not None
            ):
                self.ml_fields.append("covariates")

            self.train_reader = reader_factory(
                self.data_pathPetastorm,
                num_epochs=self.num_reader_epochs,
                cur_shard=self.cur_shard,
                shard_count=self.shard_count,
                storage_options=self.storage_options,
                transform_spec=transform_train,
                # Don't shuffle row groups without shuffling.
                shuffle_row_groups=True if self.shuffle_size > 0 else False,
                predicate=predicate_expr_train,
                **reader_factory_kwargs,
            )
            if self.has_val:
                (
                    self.nb_sample_set_val,
                    predicate_expr_val,
                    transform_val,
                ) = self.generate_ML_reader_predicate_transform(partition="val")
                self.steps_per_epoch_val = int(
                    math.floor(float(self.nb_sample_set_val) / self.val_batch_size)
                )
                self.val_reader = reader_factory(
                    self.data_pathPetastorm,
                    num_epochs=self.num_reader_epochs,
                    cur_shard=self.cur_shard,
                    shard_count=self.shard_count,
                    storage_options=self.storage_options,
                    transform_spec=transform_val,
                    shuffle_row_groups=False,
                    predicate=predicate_expr_val,
                    **reader_factory_kwargs,
                )

            if self.has_test:
                (
                    self.nb_sample_set_test,
                    predicate_expr_test,
                    transform_test,
                ) = self.generate_ML_reader_predicate_transform(partition="test")
                self.steps_per_epoch_test = int(
                    math.floor(float(self.nb_sample_set_test) / self.val_batch_size)
                )
                self.test_reader = reader_factory(
                    self.data_pathPetastorm,
                    num_epochs=self.num_reader_epochs,
                    cur_shard=self.cur_shard,
                    shard_count=self.shard_count,
                    storage_options=self.storage_options,
                    transform_spec=transform_test,
                    shuffle_row_groups=False,
                    predicate=predicate_expr_test,
                    **reader_factory_kwargs,
                )

    def teardown(self, stage=None):
        if stage == "fit" or stage is None:
            if self.verbose:
                print("Tear down: closing async dataloaders")
            self.train_dl.close_async_loader()
            if self.has_val:
                self.val_dl.close_async_loader()
            if not self.inmemory_cache_all:
                # Reader was loaded once and stopped for inmemory datalaoder.
                if self.verbose:
                    print("Tear down: closing petastorm readers")
                self.train_reader.stop()
                self.train_reader.join()
                if self.has_val:
                    self.val_reader.stop()
                    self.val_reader.join()
            if self.verbose:
                print("Tear down: async dataloaders closed.")

    def train_dataloader(self):
        if self.verbose:
            print("Setup train dataloader")
        kwargs = dict(
            reader=self.train_reader,
            batch_size=self.train_batch_size,
            name="train dataloader",
            limit_step_per_epoch=self.steps_per_epoch_train,
            verbose=self.verbose,
            desired_fields=self.ml_fields,
        )
        if self.inmemory_cache_all:
            # Use inmem dataloader
            dataloader_class = PytorchInmemAsyncDataLoader
            kwargs["shuffle"] = self.shuffle_size > 0
            kwargs["num_epochs"] = self.num_train_epochs
        else:
            dataloader_class = PytorchInfiniteAsyncDataLoader
            kwargs["shuffling_queue_capacity"] = self.shuffle_size

            if self.debug_data_loader:
                kwargs["debug_data_loader"] = self.debug_data_loader

            if self.train_async_data_loader_queue_size is not None:
                if isinstance(self.train_async_data_loader_queue_size, int):
                    kwargs[
                        "async_loader_queue_size"
                    ] = self.train_async_data_loader_queue_size
                elif isinstance(self.train_async_data_loader_queue_size, float):
                    # use async data loader queue size as ratio of total steps.
                    kwargs["async_loader_queue_size"] = int(
                        kwargs["limit_step_per_epoch"]
                        * self.train_async_data_loader_queue_size
                    )
                else:
                    raise RuntimeError(
                        f"Unsupported type for train_async_data_loader_queue_size={self.train_async_data_loader_queue_size}"
                    )

        self.train_dl = dataloader_class(**kwargs)
        return self.train_dl

    def val_dataloader(self):
        if not self.has_val:
            return None
        if self.verbose:
            print("setup val dataloader")
        kwargs = dict(
            reader=self.val_reader,
            batch_size=self.val_batch_size,
            name="val dataloader",
            limit_step_per_epoch=self.steps_per_epoch_val,
            verbose=self.verbose,
            desired_fields=self.ml_fields,
        )
        if self.inmemory_cache_all:
            # Use inmem dataloader
            dataloader_class = PytorchInmemAsyncDataLoader
            kwargs["shuffle"] = False
            kwargs["num_epochs"] = self.num_train_epochs
        else:
            dataloader_class = PytorchInfiniteAsyncDataLoader
            kwargs["shuffling_queue_capacity"] = 0

            if self.debug_data_loader:
                kwargs["debug_data_loader"] = self.debug_data_loader

            if self.val_async_data_loader_queue_size is not None:
                if isinstance(self.val_async_data_loader_queue_size, int):
                    kwargs[
                        "async_loader_queue_size"
                    ] = self.val_async_data_loader_queue_size
                elif isinstance(self.val_async_data_loader_queue_size, float):
                    # use async data loader queue size as ratio of total steps.
                    kwargs["async_loader_queue_size"] = int(
                        kwargs["limit_step_per_epoch"]
                        * self.val_async_data_loader_queue_size
                    )
                else:
                    raise RuntimeError(
                        f"Unsupported type for val_async_data_loader_queue_size={self.val_async_data_loader_queue_size}"
                    )

        self.val_dl = dataloader_class(**kwargs)
        return self.val_dl

    def test_dataloader(self):
        if not self.has_test:
            return None
        if self.verbose:
            print("setup test dataloader")
        kwargs = dict(
            reader=self.test_reader,
            batch_size=self.val_batch_size,
            name="test dataloader",
            limit_step_per_epoch=self.steps_per_epoch_test,
            verbose=self.verbose,
            desired_fields=self.ml_fields,
        )
        if self.inmemory_cache_all:
            # Use inmem dataloader
            dataloader_class = PytorchInmemAsyncDataLoader
            kwargs["shuffle"] = False
            kwargs["num_epochs"] = self.num_train_epochs
        else:
            dataloader_class = PytorchInfiniteAsyncDataLoader
            kwargs["shuffling_queue_capacity"] = 0

            if self.debug_data_loader:
                kwargs["debug_data_loader"] = self.debug_data_loader

            if self.val_async_data_loader_queue_size is not None:
                if isinstance(self.val_async_data_loader_queue_size, int):
                    kwargs[
                        "async_loader_queue_size"
                    ] = self.val_async_data_loader_queue_size
                elif isinstance(self.val_async_data_loader_queue_size, float):
                    # use async data loader queue size as ratio of total steps.
                    kwargs["async_loader_queue_size"] = int(
                        kwargs["limit_step_per_epoch"]
                        * self.val_async_data_loader_queue_size
                    )
                else:
                    raise RuntimeError(
                        f"Unsupported type for val_async_data_loader_queue_size={self.val_async_data_loader_queue_size}"
                    )

        self.test_dl = dataloader_class(**kwargs)
        return self.test_dl

    def custom_dataloader(self, list_names, epochs=1):
        "Implement a dataloader for a given set of samples names"

        (
            self.nb_sample_set_test,
            predicate_expr_test,
            transform_test,
        ) = self.generate_ML_reader_predicate_transform(list_names=list_names)
        reader_factory_kwargs = dict()

        reader_factory_kwargs["pyarrow_serialize"] = True
        custom_reader = make_reader(
            self.data_pathPetastorm,
            num_epochs=epochs,
            cur_shard=self.cur_shard,
            shard_count=self.shard_count,
            storage_options=self.storage_options,
            transform_spec=transform_test,
            shuffle_row_groups=False,
            predicate=predicate_expr_test,
            **reader_factory_kwargs,
        )

        return custom_reader

    def return_dataloader(self, partition="train"):
        if partition == "train":
            return self.train_dataloader()
        elif partition == "val":
            return self.val_dataloader()
        elif partition == "test":
            return self.test_dataloader()
        else:
            raise RuntimeError(f"Unsupported partition {partition}")

    @property
    def target_shape(self):
        return self._target_shape

    def create_reader(
        self, schema_fields=None, predicate=None, transform_spec=None, num_epochs=1
    ):

        return make_reader(
            self.data_pathPetastorm,
            schema_fields=schema_fields,
            predicate=predicate,
            num_epochs=num_epochs,
            transform_spec=transform_spec,
        )

    def extract_min_size(self, q):
        """Determine min size to retain qth quantile of the data

        Args:
            q ([float]): [quantile of the data which should be retained]

        Returns:
            [int]: [min size to retain qth quantile of the data]
        """
        with self.create_reader(schema_fields=["signal_length"]) as reader:
            l_tot = [l.signal_length for l in reader]
        min_size = np.quantile(l_tot, 1 - q)
        return int(min_size)

    def list_transformReader(self):
        """[Function to list transform which have to be performed when loading dataset]"""
        self.transform_dict = {}
        self.manipulation = self.config.get("MANIPULATION", None)
        resizing = self.manipulation.get("resizing_len", None)
        if isinstance(resizing, list):
            if resizing[0].lower() == "quantile":
                quantile = resizing[1]
                self.transform_dict["min_size"] = self.extract_min_size(quantile)

        target_encoding = self.manipulation.get("target_encoding", None)
        if target_encoding != None:
            self.transform_dict["target_encoding"] = target_encoding

        filter_signal = self.manipulation.get("filter_signal", None)
        if filter_signal != None:
            self.transform_dict["filter_signal"] = filter_signal

        r_segmentation = self.manipulation.get("r_segmentation", None)
        if r_segmentation != None:
            self.transform_dict["r_segmentation"] = r_segmentation

        feature_scaler = self.manipulation.get("feature_scaling", "none").lower()
        if feature_scaler != "none":
            self.transform_dict["feature_scaler"] = feature_scaler


    def generate_ML_reader_predicate_transform(self, partition=None, list_names=None):

        root_path = os.path.abspath(os.path.join(self.data_path, "../"))
        self.partition_split = {}
        # os.path.join(os.path.join(root_path,'train.npy')),
        # os.path.join(os.path.join(root_path,'val.npy')),
        # os.path.join(os.path.join(root_path,'test.npy'))
        # ]
        if partition == "train":
            self.partition_split["train"] = os.path.join(root_path, "train.npy")
            names_sample = np.load(self.partition_split["train"]).astype(str).tolist()
        elif partition == "validation" or partition == "val":
            self.partition_split["val"] = os.path.join(root_path, "val.npy")
            names_sample = np.load(self.partition_split["val"]).astype(str).tolist()
        elif partition == "test":
            self.partition_split["test"] = os.path.join(root_path, "test.npy")
            names_sample = np.load(self.partition_split["test"]).astype(str).tolist()
        elif partition is None and list_names is not None:
            if not isinstance(list_names, list):
                raise ValueError("list_names must be a list")
            names_sample = list_names
        else:
            raise ValueError("Partition should be one of train, validation/val, test")

        nb_sample_set = len(names_sample)
        # print(f'Reader for {partition} set with {self.partition_split[partition_num]} of the data')
        # print(f"Reader for {partition} set with {len(names_sample)}  samples")

        # print(self.transform_dict)
        all_fieds = [self.featureName, self.targetName, "noun_id", "signal_names"]
        if (self.ordinal_covariates is not None) | (
            self.ordinal_covariates is not None
        ):
            all_fieds.append("covariates")
        fields_mod = tuple()
        fields_mod = (
            *fields_mod,
            ("signal", np.float32, (None, None), NdarrayCodec(), False),
        )
        if self.manipulation.get("target_encoding", None) == "label_encoding":
            # if target is encoded scheme need to be changed from string to integet
            fields_mod = (
                *fields_mod,
                (self.targetName, np.int32, (None,), NdarrayCodec(), False),
            )

        elif self.manipulation.get("target_encoding", None) == "binary_encoding":
            # if target is encoded scheme need to be changed from string to integer
            fields_mod = (
                *fields_mod,
                (self.targetName, np.int_, (), ScalarCodec(IntegerType()), False),
            )

        elif self.manipulation.get("target_encoding", None) == "one_hot_encoder":
            # if target is encoded scheme need to be changed from string to integer
            fields_mod = (
                *fields_mod,
                (self.targetName, np.int_, (None,), NdarrayCodec(), False),
            )

        if (self.ordinal_covariates is not None) | (
            self.categorical_covariates is not None
        ):
            fields_mod = (
                *fields_mod,
                UnischemaField("covariates", np.float32, (None), NdarrayCodec(), False),
            )

        if len(self.transform_dict) > 0:
            transform = TransformSpec(
                self._transform_row, edit_fields=fields_mod, selected_fields=all_fieds
            )
        else:
            transform = None

        if self.transform_dict.get("min_size") != None:
            min_size = self.transform_dict.get("min_size")
            # predicate_expr = in_reduce([in_pseudorandom_split(self.partition_split, partition_num, 'noun_id'),
            # in_lambda(['signal_length'], lambda signal_length: signal_length >= min_size)], all)
            predicate_expr = in_reduce(
                [
                    in_lambda(
                        ["noun_id"],
                        lambda noun_id: noun_id.astype("str") in names_sample,
                    ),
                    in_lambda(
                        ["signal_length"],
                        lambda signal_length: signal_length >= min_size,
                    ),
                ],
                all,
            )

        else:
            # predicate_expr = in_pseudorandom_split(self.partition_split,partition_num, 'noun_id')
            predicate_expr = in_lambda(
                ["noun_id"], lambda noun_id: noun_id.astype("str") in names_sample
            )

        return nb_sample_set, predicate_expr, transform

    def transform_features(self, data_row, **kwargs):
        """[Function to add the various steps which should be performed on the features]

        Args:
            data_val ([type]): [description]

        Returns:
            [type]: [description]
        """
        filter_method = self.transform_dict.get("filter_signal")
        data_val = data_row[self.featureName]

        if self.transform_dict.get("min_size") != None:
            min_size = self.transform_dict.get("min_size")
            cropping_val = self.manipulation.get("resizing_method", "end")
            data_val = utils_data.resize_len(data_val, min_size, cropping=cropping_val)


        if self.transform_dict.get("feature_scaler") != None:
            data_val = self.feature_scaler.transform(data_val)

        #we always convert the data to [features, sequence]
        data_val = np.transpose(data_val, (1, 0))

        return data_val.astype(np.float32)

    def transform_targets(self, data_val):
        """Perform transformation on the target

        Args:
            data_val ([type]): [description]

        Returns:
            [type]: [description]
        """

        transform_target = self.transform_dict.get("target_encoding")

        if self.selected_classes != None and "all" not in self.selected_classes:
            data_val = np.setdiff1d(data_val, self.list_remove).astype(int)

        if transform_target == "label_encoding":
            data_val = self.encoder.transform(
                np.expand_dims(data_val, axis=0)
            ).flatten()

        elif transform_target == "binary_encoding":
            if data_val.all() == 0:
                data_val = 0
            else:
                data_val = 1
        elif transform_target == "one_hot_encoder":
            if np.array(data_val).dtype.type is np.string_:
                data_val = np.array(data_val).astype(str)

            if data_val.size > 1:
                # If we one hot encode a list of classes, we join the diagnostics
                # a string and one hot encode it
                data_val = np.array(",".join(data_val.astype(str).tolist()))
            if data_val.size == 0:
                data_val = np.array(["0"])
            data_val = self.encoder.transform(
                data_val.reshape(1, -1).astype(str)
            ).todense()
            data_val = np.array(data_val).squeeze()
        # print(data_val.shape)
        return data_val

    def _transform_row(self, data_row):
        result_row = {
            self.featureName: self.transform_features(data_row),
            self.targetName: self.transform_targets(data_row[self.targetName]),
            "noun_id": data_row["noun_id"],
            "signal_names": data_row["signal_names"],
        }
        if (self.ordinal_covariates is not None) | (
            self.categorical_covariates is not None
        ):
            result_row["covariates"] = self.build_covariates(data_row)
        return result_row

    def initialise_covariates(self):
        """Initialise the covariates

        Returns:
            [type]: [description]
        """
        if "classes_encoder_covariates.npy" in os.listdir(self.results_path):
            print("Loading covariate encoder classes")
            classes_encoder_covariates_saved = np.load(
                os.path.join(self.results_path, "classes_encoder_covariates.npy")
            )
        else:
            classes_encoder_covariates_saved = None

        if self.categorical_covariates is not None:
            nb_dim, categories_encoder = self.generate_covariates_encoder(
                classes_encoder_covariates_saved
            )
            self.dim_covariates += nb_dim
        if self.ordinal_covariates is not None:
            nb_dim = self.generate_covariates_scaler()
            self.dim_covariates += nb_dim

        return categories_encoder

    def build_covariates(self, data_row):
        """
        Build the covariates which will be used by the network
        """
        np_ordinal_covariates = None
        np_categorical_covariates = None
        list_covariates = []
        if self.ordinal_covariates is not None:
            ordinal_covariates = itemgetter(*self.ordinal_covariates)(data_row)
            if not isinstance(ordinal_covariates, Iterable):
                ordinal_covariates = [ordinal_covariates]
            ordinal_covariates = np.array(
                [
                    np.array(x).reshape(
                        -1,
                    )
                    if not isinstance(x, np.ndarray)
                    else x
                    for x in ordinal_covariates
                ]
            )
            ordinal_covariates = np.concatenate(ordinal_covariates)
            if ordinal_covariates.ndim == 1:
                ordinal_covariates = ordinal_covariates.reshape(-1, 1)
            np_ordinal_covariates = self.covariate_scaler.transform(
                ordinal_covariates.reshape(1, -1)
            )
            list_covariates.extend(np_ordinal_covariates.reshape(-1).tolist())

        if self.categorical_covariates is not None:
            categorical_covariates = itemgetter(*self.categorical_covariates)(data_row)
            if categorical_covariates.dtype.type is np.string_:
                categorical_covariates = categorical_covariates.astype(str)
            np_categorical_covariates = self.covariates_encoder.transform(
                categorical_covariates.reshape(1, -1)
            ).todense()
            list_covariates.extend(
                np.asarray(np_categorical_covariates).reshape(-1).tolist()
            )

        np_covariates = np.array(list_covariates).astype(np.float32)
        return np_covariates

    def generate_covariates_encoder(self, categories_encoder):
        with self.create_reader(schema_fields=self.categorical_covariates) as reader:
            categorical_covariates = []
            for item in reader:
                categorical_covariates.append(
                    attrgetter(*self.categorical_covariates)(item)
                )

            # categorical_covariates = [l[0] for l in reader]
        classes = None
        if np.array(categorical_covariates).dtype.type is np.string_:
            categorical_covariates = np.array(categorical_covariates).astype(str)
        elif not isinstance(categorical_covariates, np.ndarray):
            categorical_covariates = np.array(categorical_covariates)

        if categories_encoder is None:
            classes = "auto"
        else:
            classes = categories_encoder
        if isinstance(classes, np.ndarray):
            classes = [classes[idx] for idx in range(classes.shape[0])]

        encoder = preprocessing.OneHotEncoder(categories=classes, drop="if_binary")

        if categorical_covariates.ndim == 1:
            categorical_covariates = categorical_covariates.reshape(-1, 1)

        encoder.fit(categorical_covariates)
        self.covariates_encoder = encoder
        name_classes = encoder.categories_
        if categories_encoder is None:
            np.save(
                os.path.join(self.results_path, "classes_encoder_covariates.npy"),
                name_classes,
            )
        dim_classes = encoder.transform(categorical_covariates).shape[-1]
        return dim_classes, name_classes

    def generate_covariates_scaler(self):
        ordinal_covariates = []
        with self.create_reader(schema_fields=self.ordinal_covariates) as reader:
            for item in reader:
                tmp = attrgetter(*self.ordinal_covariates)(item)

                if not isinstance(tmp, Iterable):
                    tmp = [tmp]
                tmp = np.array(
                    [
                        np.array(x).reshape(
                            -1,
                        )
                        if not isinstance(x, np.ndarray)
                        else x
                        for x in tmp
                    ]
                )
                ordinal_covariates.append(np.concatenate(tmp))

        scaler = preprocessing.StandardScaler()
        if not isinstance(ordinal_covariates, np.ndarray):
            ordinal_covariates = np.array(ordinal_covariates)
        if ordinal_covariates.ndim == 1:
            ordinal_covariates = ordinal_covariates.reshape(-1, 1)

        scaler.fit(ordinal_covariates)
        self.covariate_scaler = scaler
        dim_covariates = ordinal_covariates.shape[-1]

        return dim_covariates

    def generate_encoder(self):
        with self.create_reader(schema_fields=[self.targetName]) as reader:
            target_dataset = [l[0] for l in reader]
        classes = None
        if np.array(target_dataset).dtype.type is np.string_:
            target_dataset = np.array(target_dataset).astype(str)

        if self.selected_classes != None and "all" not in self.selected_classes:
            print(f"Selecting only {self.selected_classes}")
            tmp = [reduce(lambda i, j: np.concatenate((i, j)), target_dataset)]
            unique = set(tmp[0])
            self.list_remove = [x for x in unique if x not in self.selected_classes]
            target_dataset = [np.setdiff1d(x, self.list_remove) for x in target_dataset]
            # target_dataset = [np.zeros(1).astype(int) \
            #  if x.shape[0] ==0 else x.astype(int) for x in new_target ]
            classes = self.selected_classes

        if self.config["MANIPULATION"]["target_encoding"] == "label_encoding":
            encoder = preprocessing.MultiLabelBinarizer(classes=classes)
            encoder.fit(target_dataset)
            tmp = encoder.transform(target_dataset)
            classes_name = encoder.classes_

        elif self.config["MANIPULATION"]["target_encoding"] == "one_hot_encoder":
            classes = "auto"
            encoder = preprocessing.OneHotEncoder(categories=classes)
            if isinstance(target_dataset, list):
                # we want to deal with case where we onehot encode a list of labels
                target_dataset = np.array(
                    [",".join(x.astype(str).tolist()) for x in target_dataset]
                )

            target_dataset = np.array(["0" if x == "" else x for x in target_dataset])
            target_dataset = np.array(target_dataset).reshape(1, -1)
            encoder.fit(target_dataset)
            encoder.fit(target_dataset.reshape(-1, 1))
            tmp = encoder.transform(target_dataset.reshape(-1, 1))
            classes_name = encoder.get_feature_names()
        else:
            raise ValueError("Encoder not recognised")
        self.encoder = encoder
        self.samples_per_cls = tmp.sum(axis=0).A

        self._target_shape = tmp.shape[1]

        np.save(os.path.join(self.results_path, "classes_encoder.npy"), classes_name)
        np.save(
            os.path.join(self.results_path, "samples_per_cls.npy"), self.samples_per_cls
        )
        return True

    def generate_scaler(self):
        if "stats_scaler.csv" in os.listdir(self.results_path):
            df_stats_saved = pd.read_csv(
                os.path.join(self.results_path, "stats_scaler.csv")
            )
        else:
            df_stats_saved = None

        # df_stats = self.ml_data.generate_scaler(df_stats_saved)
        scaler_type = self.config["MANIPULATION"].get("feature_scaling", "none").lower()
        self.feature_scaler = TS_Scaler(scaler_type, df_stats_saved)
        if not isinstance(df_stats_saved, pd.DataFrame):
            with self.create_reader(schema_fields=[self.featureName]) as reader:
                print("Computing train dataset stats for scaler")
                df_stats = self.feature_scaler.fit(reader)

                df_stats.to_csv(os.path.join(self.results_path, "stats_scaler.csv"))
        else:
            print("Loading dataset stats for scaler")

        return True

    def evaluate_nbSample(self):
        with self.create_reader(schema_fields=["noun_id"]) as reader:
            list_noun = [l[0] for l in reader]
            self.nb_sample = len(list_noun)
        return self.nb_sample

    def evaluate_nb_features(self):
        with self.create_reader(schema_fields=[self.featureName]) as reader:
            sample = reader.next()
        self._n_features = sample[0].shape[1]

        return True

    @property
    def nb_features(self):
        return self._n_features
