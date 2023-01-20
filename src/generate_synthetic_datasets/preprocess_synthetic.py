#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Written by H.Turbé, June 2022.

"""
import math
import os
import random
import sys
import warnings

import numpy as np
from petastorm import make_reader
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.unischema import dict_to_spark_row

warnings.simplefilter(action="ignore", category=FutureWarning)
FILEPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(FILEPATH)

from shared_types.schema import ShapeSchema


class PreprocessSynthetic:
    """
    Main class to generate the synthetic dataset
    """
    def __init__(self, save_path=None, data_config=None):
        self.save_path = save_path
        self.data_config = data_config
        print("Generated data save in: ", self.save_path)

        self.nb_simulation = data_config["properties"]["nb_simulation"]
        self.n_points = data_config["properties"]["n_points"]
        self.n_support = data_config["properties"]["n_support"]
        self.n_feature = data_config["properties"]["n_feature"]
        self.f_min_support, self.f_max_support = data_config["properties"]["f_sin"]
        self.f_min_base, self.f_max_base = data_config["properties"]["f_base"]
        # we create a distribution for the sum of sine frequency to
        # have control over class distribution
        f_sine_1 = np.random.randint(
            self.f_min_support, (self.f_max_support + 1), 10000
        )
        f_base_1 = np.random.randint(self.f_min_base, (self.f_max_base + 1), 10000)
        f_sine_2 = np.random.randint(
            self.f_min_support, (self.f_max_support + 1), 10000
        )
        f_base_2 = np.random.randint(self.f_min_base, (self.f_max_base + 1), 10000)
        f_sum = f_sine_1 + f_sine_2
        self.quantile_class_sum = np.quantile(
            f_sum, data_config["properties"]["quantile_class"]
        )
        f_ratio = (f_sine_1 / f_base_1) + (f_sine_2 / f_base_2)
        self.quantile_class_ratio = np.quantile(
            f_ratio, data_config["properties"]["quantile_class"]
        )

    # Abstract methods
    # ----------------------------------------------------------------------------

    def format_to_parquet(self, spark):
        """
        Method to create the dataset in parquet format
        Parameters
        ----------
        spark: SparkSession
        Returns
        -------
        None
        """
        sc = spark.sparkContext

        indices = range(self.nb_simulation)
        ROWGROUP_SIZE_MB = 256

        DEFAULT_PARQUET = os.path.join(self.save_path, "dataParquet")

        output_URL = f"file:///{DEFAULT_PARQUET}"

        with materialize_dataset(spark, output_URL, ShapeSchema, ROWGROUP_SIZE_MB):
            data_list = sc.parallelize(indices).map(
                lambda index: (self._generate_sample(index))
            )

            sql_rows_rdd = data_list.map(lambda r: dict_to_spark_row(ShapeSchema, r))

            # Write out the result
            spark.createDataFrame(
                sql_rows_rdd, ShapeSchema.as_spark_schema()
            ).write.mode("overwrite").option("compression", "none").parquet(output_URL)

    def create_split(self, list_split):
        """
        Method to create the train, val and test split
        Parameters
        ----------
        list_split: list
            list of the percentage of the split
        Returns
        ----------
        None
        """
        split = {"train": list_split[0], "val": list_split[1], "test": list_split[2]}
        counter = 0
        start_set = 0
        namefield = ["noun_id"]

        path_data = f"file://{os.path.join(self.save_path,'dataParquet')}"
        with make_reader(path_data, schema_fields=namefield) as reader_name:
            list_names = [l[0] for l in reader_name]
            list_names = np.array(list_names).astype(str).tolist()

        random.shuffle(list_names)
        nb_samples = len(list_names)
        for key, val in split.items():
            nb_set = math.ceil(val * nb_samples)
            sample_set = list_names[start_set : start_set + nb_set]
            counter += nb_set
            start_set += nb_set

            np_set = np.array(sample_set)
            np.save(os.path.join(self.save_path, key), np_set)
            print(f"sample in {key} = {nb_set}")
        print("Total sample", counter)

    # ----------------------------------------------------------------------------

    # Private methods
    # ----------------------------------------------------------------------------

    def _generate_feature(self, wave: str, f_support: int, f_base: float):
        """
        Generate a given feature of a sample with a support of a given type and frequency
        overlaid over a sine wave of a given frequency
        Parameters
        ----------
        wave: str
            type of the support wave
        f_support: int
            frequency of the support wave
        f_base: float
            frequency of the base wave
        Returns
        -------
        x_feature: np.array
            feature of the sample
        start_idx: int
            idx where the support starts
        """
        x_feature = np.sin(np.linspace(0, 2 * np.pi * f_base, self.n_points)).reshape(
            -1, 1
        )
        x_feature *= 0.5
        start_idx = np.random.randint(0, self.n_points - self.n_support)

        if wave == "sine":
            x_tmp = np.sin(
                np.linspace(0, 2 * np.pi * f_support, self.n_points)
            ).reshape(-1, 1)
            start_tmp = 0
            x_feature[start_idx : start_idx + self.n_support, 0] += x_tmp[
                start_tmp : start_tmp + self.n_support, 0
            ]

        elif wave == "square":
            x_tmp = np.sign(
                np.sin(np.linspace(0, 2 * np.pi * f_support, self.n_points))
            ).reshape(-1, 1)
            start_tmp = 0
            x_feature[start_idx : start_idx + self.n_support, 0] += x_tmp[
                start_tmp : start_tmp + self.n_support, 0
            ]

        elif wave == "line":
            x_feature[start_idx : start_idx + self.n_support, 0] += [0] * self.n_support
        else:
            raise ValueError("wave must be one of sine, square, sawtooth, line")
        return x_feature, start_idx

    def _generate_sample(self, index):
        """
        Generate a sample
        Parameters
        ----------
        index: int
            index of the sample
        Returns
        -------
        dict_all: dict
            dictionary containing the sample with the information saved
            in parquet format
        """
        dict_all = {}
        dict_all["noun_id"] = f"sample_{index}"
        idx_features = np.random.permutation(np.arange(self.n_feature))
        x_sample = np.zeros((self.n_points, self.n_feature))
        f_sine_sum = 0
        f_ratio = 0
        pos_sin_wave = []
        for enum, idx_feature in enumerate(idx_features):
            f_base = np.random.randint(self.f_min_base, (self.f_max_base + 1), 1)[0]
            f_support = np.random.randint(
                self.f_min_support, (self.f_max_support + 1), 1
            )[0]
            if enum < 2:


                f_sine_sum += f_support
                f_ratio += f_support / f_base
                x_tmp, start_idx = self._generate_feature(
                    wave="sine", f_support=f_support, f_base=f_base
                )
                x_sample[:, idx_feature] = x_tmp.squeeze()
                pos_sin_wave.append(start_idx)
            else:

                wave = random.choice(["line", "square"])
                x_tmp, _ = self._generate_feature(
                    wave=wave, f_support=f_support, f_base=f_base
                )
                x_sample[:, idx_feature] = x_tmp.squeeze()

        dict_all["signal"] = x_sample.astype(np.float32)

        dict_all["target_sum"] = (
            np.argwhere(f_sine_sum <= self.quantile_class_sum).min().astype(str)
        )

        bool_class_ratio = f_ratio <= self.quantile_class_ratio
        if np.sum(bool_class_ratio) == 0:
            dict_all["target_ratio"] = str(len(self.quantile_class_ratio) - 1)
        else:
            dict_all["target_ratio"] = np.argwhere(bool_class_ratio).min().astype(str)

        dict_all["signal_length"] = x_sample.shape[0]
        dict_all["signal_names"] = np.array(
            [f"feature_{str(x)}" for x in range(x_sample.shape[1])]
        ).astype(np.string_)
        dict_all["pos_sin_wave"] = np.array(pos_sin_wave).astype(int)
        dict_all["feature_idx_sin_wave"] = idx_features[:2].astype(int)
        dict_all["f_sine_sum"] = f_sine_sum.astype(int)
        dict_all["f_sine_ratio"] = f_ratio.astype(np.float32)

        return dict_all
