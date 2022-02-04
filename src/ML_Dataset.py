#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
	Written by H.Turbé, January 2021.
	Dataset class with functions required to prepare the dataset for ML
"""
import os
import sys
import time
from functools import reduce

# from petastorm.unischema import Unischema, UnischemaField
import matplotlib.pyplot as plt
import numpy as np
from petastorm import TransformSpec, make_reader
from petastorm.codecs import NdarrayCodec, ScalarCodec
from petastorm.predicates import in_lambda, in_pseudorandom_split, in_reduce
from pyspark.sql.types import IntegerType, StringType
from scipy.sparse import data
from sklearn import preprocessing

FILEPATH = os.path.dirname(os.path.realpath(__file__))
ROOTPATH = os.path.dirname(FILEPATH)
sys.path.append(os.path.join(ROOTPATH))
results_path = os.path.join(FILEPATH, "results")
# custom libraries
import utils.utils_data as utils_data


class ML_Dataset:
    def __init__(self, config):
        self.config = config
        self.featureName = self.config["CONFIGURATION"]["feature"]
        self.targetName = self.config["CONFIGURATION"]["target"]
        self.selected_classes = self.config["CONFIGURATION"].get("selected_classes")

        if self.config["CONFIGURATION"]["data_type"].lower() == "petastorm":
            self.data_path = os.path.expanduser(
                self.config["CONFIGURATION"]["data_path"]
            )
            self.data_path = os.path.abspath(self.data_path)

            self.data_pathPetastorm = f"file://{self.data_path}"
            _ = self.evaluate_nbSample()

        _ = self.evaluate_nb_features()
        self.list_transformReader()

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

        channel_first = self.manipulation.get("channel_first", "none").lower()
        if channel_first == "true":
            self.transform_dict["channel_first"] = True

    def generate_ML_reader(self, partition="train", num_epochs=None):

        root_path = os.path.abspath(os.path.join(self.data_path, "../"))
        self.partition_split = {}
        if partition == "train":
            self.partition_split["train"] = os.path.join(root_path, "train.npy")
            names_sample = np.load(self.partition_split["train"]).astype(str).tolist()
            nb_sample_set = len(names_sample)
        elif partition == "validation" or partition == "val":
            self.partition_split["val"] = os.path.join(root_path, "val.npy")
            names_sample = np.load(self.partition_split["val"]).astype(str).tolist()
            nb_sample_set = len(names_sample)
        elif partition == "test":
            self.partition_split["test"] = os.path.join(root_path, "test.npy")
            names_sample = np.load(self.partition_split["test"]).astype(str).tolist()
            nb_sample_set = len(names_sample)
        else:
            raise ValueError("Partition should be one of train, validation/val, test")

        self.num_epochs = num_epochs
        all_fieds = [self.featureName, self.targetName, "noun_id", "signal_names"]
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

        if self.manipulation.get("target_encoding", None) == "binary_encoding":
            # if target is encoded scheme need to be changed from string to integer
            fields_mod = (
                *fields_mod,
                (self.targetName, np.int_, (), ScalarCodec(IntegerType()), False),
            )

        if self.manipulation.get("target_encoding", None) == "one_hot_encoder":
            # if target is encoded scheme need to be changed from string to integer
            fields_mod = (
                *fields_mod,
                (self.targetName, np.int_, (None,), NdarrayCodec(), False),
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

        reader = make_reader(
            self.data_pathPetastorm,
            predicate=predicate_expr,
            transform_spec=transform,
            num_epochs=self.num_epochs,
            shuffle_row_drop_partitions=2,
        )

        return reader, nb_sample_set

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

        if self.transform_dict.get("channel_first") == True:
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
        """
        Apply respective transformation on the features and target
        """
        result_row = {
            self.featureName: self.transform_features(data_row),
            self.targetName: self.transform_targets(data_row[self.targetName]),
            "noun_id": data_row["noun_id"],
            "signal_names": data_row["signal_names"],
        }

        return result_row

    def generate_encoder(self):
        """
        Create data encoder. Requires loading full dataset (only target) to create encoder
        """
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
                target_dataset = [
                    np.setdiff1d(x, self.list_remove) for x in target_dataset
                ]
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

                target_dataset = np.array(
                    ["0" if x == "" else x for x in target_dataset]
                )
                target_dataset = np.array(target_dataset).reshape(1, -1)
                encoder.fit(target_dataset)
                encoder.fit(target_dataset.reshape(-1, 1))
                tmp = encoder.transform(target_dataset.reshape(-1, 1))
                classes_name = encoder.get_feature_names()
            else:
                raise ValueError("Encoder not recognised")
            self.encoder = encoder
        return tmp, classes_name

    def generate_scaler(self):
        """
        Create data scaler.
        """
        with self.create_reader(schema_fields=[self.featureName]) as reader:
            feature_dataset = [l[0] for l in reader]
            np_feature = np.array(feature_dataset)

            if np_feature.ndim == 3:
                np_feature = np_feature.reshape(-1, np_feature.shape[-1])

            scaler_type = (
                self.config["MANIPULATION"].get("feature_scaling", "none").lower()
            )
            if scaler_type == "standard":
                self.feature_scaler = preprocessing.StandardScaler()
            elif scaler_type == "minmax":
                self.feature_scaler = preprocessing.MinMaxScaler()
            else:
                raise ValueError(f"Scaler {scaler_type} not supported")

            self.feature_scaler.fit(np_feature)

        return True

    def evaluate_nbSample(self):
        """
        Load the dataset and evaluate the number of samples (only name of the samples are loaded)
        """
        with self.create_reader(schema_fields=["noun_id"]) as reader:
            list_noun = [l[0] for l in reader]
            self.nb_sample = len(list_noun)
        return self.nb_sample

    def evaluate_nb_features(self):
        """
        Evaluate the number of features in each sample
        """
        with self.create_reader(schema_fields=[self.featureName]) as reader:
            sample = reader.next()
        self.n_features = sample[0].shape[1]

        return True

    def retrieve_sample(self, name_sample, fields=None):
        """
        Specific method to retrieve a sample from the dataset
        """
        # self.num_epochs = 1
        if fields == None:
            all_fieds = [self.featureName, self.targetName, "noun_id", "signal_names"]
        else:
            all_fieds = fields
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

        if self.manipulation.get("target_encoding", None) == "binary_encoding":
            # if target is encoded scheme need to be changed from string to integer
            fields_mod = (
                *fields_mod,
                (self.targetName, np.int_, (), ScalarCodec(IntegerType()), False),
            )

        if self.manipulation.get("target_encoding", None) == "one_hot_encoder":
            # if target is encoded scheme need to be changed from string to integer
            fields_mod = (
                *fields_mod,
                (self.targetName, np.int_, (None,), NdarrayCodec(), False),
            )

        if len(self.transform_dict) > 0:
            transform = TransformSpec(
                self._transform_row, edit_fields=fields_mod, selected_fields=all_fieds
            )
        else:
            transform = None

        predicate_expr = in_lambda(
            ["noun_id"], lambda noun_id: noun_id.astype("str") in name_sample
        )

        reader = make_reader(
            self.data_pathPetastorm,
            predicate=predicate_expr,
            transform_spec=transform,
            num_epochs=1,
        )

        return reader
