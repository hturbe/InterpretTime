#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
	Written by H.Turbé, March 2020.
"""
import argparse
import os
import shutil
import sys
import time

from pyspark import  SparkFiles
from pyspark.sql import  SparkSession

CWD = os.getcwd()
FILEPATH = os.path.dirname(os.path.realpath(__file__))
ROOTPATH = os.path.dirname(FILEPATH)
sys.path.append(os.path.join(ROOTPATH))
from generate_synthetic_datasets.preprocess_synthetic import PreprocessSynthetic as Preprocess
from shared_utils.utils_data import parse_config
from shared_utils.utils_path import config_path, data_path

def main():

    # parse command-line
    parser = argparse.ArgumentParser(description="Process files.")
    parser.add_argument(
        "--config_file",
        default=f"{config_path}/config_generate_synthetic.yaml",
        help="Name of the data catalog to be used",
    )

    # parse arguments
    args = parser.parse_args()
    data_config = parse_config(args.config_file)
    save_path = os.path.join(data_path,data_config["save_name"])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # create spark session
    print("Creating Spark Session ...")
    session_builder = (
        SparkSession.builder.appName("Dataset Creation")
        .config("spark.dynamicAllocation.enabled", "true")
        .config("spark.driver.memory", "10g")
    )
    session_builder.master("local[*]")

    # Set spark environments
    python_interpeter = sys.executable
    os.environ["PYSPARK_PYTHON"] = python_interpeter
    os.environ["PYSPARK_DRIVER_PYTHON"] = python_interpeter
    spark = session_builder.getOrCreate()
    sc = spark.sparkContext
    sc.addFile(
        os.path.join(ROOTPATH, "generate_synthetic_datasets"), recursive = True
    )
    sc.addFile(os.path.join(ROOTPATH, "shared_types"), recursive = True)
    sys.path.insert(0, SparkFiles.getRootDirectory())

    # copy config file into simulation folder
    shutil.copyfile(
        args.config_file, os.path.join(save_path, os.path.split(args.config_file)[-1])
    )

    #Generate data
    dict_pre = Preprocess(
        save_path=save_path,
        data_config=data_config,
    )

    print("Formatting to Parquet")
    s = time.time()
    dict_pre.format_to_parquet(spark)
    print("done Elapsed time ", time.time() - s, "s.")

    list_split = data_config.get("dataset_split")
    if list_split is None:
        print("Split not provided. Default split applied [0.8,0.2]")
        list_split = [0.7,0.15,0.15]

    print("Creating split for model training")
    s = time.time()
    dict_pre.create_split(list_split)
    print("done. Elapsed time ", time.time() - s, "s.")


if __name__ == "__main__":
    main()
