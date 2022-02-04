#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
	Written by G. Mengaldo & H.Turbé, May 2021.
"""
import argparse
import os
import re
import shutil
import sys
import time

import numpy as np
import pandas as pd

CWD = os.getcwd()
FILEPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FILEPATH, "../"))
from learning_pytorch.Trainable import Trainable
from postprocessing_pytorch.manipulation_results import ScoreComputation

# parse command-line parameters
parser = argparse.ArgumentParser(description="Process files.")
parser.add_argument(
    "--results_path",
    type=str,
    help="Path to the folder with the model and its results",
    default="../../trained_models/dynamics/SD1_CNN",
)
parser.add_argument(
    "--method_relevance",
    help="List with the relevance method to use for interpretability results",
    default = ['shapleyvalue','integrated_gradients', 'deeplift', 'gradshap','saliency', 'kernelshap'],
    nargs="+",
)

args = parser.parse_args()
results_path = args.results_path
results_path = os.path.abspath(results_path)
method_relevance = args.method_relevance

# interpretability
plot_signal = 0  # plot every x signal or False & 0 to plot no signal
replacement_methods = ["normal", "permutation"]

res = [
    f
    for f in os.listdir(results_path)
    if re.search(r"config[a-zA-Z0-9_]+(.yaml|.xml)", f)
]
fc = os.path.abspath(os.path.join(results_path, res[0]))

names = None
# names = ['lorenz__245', 'duffing__435', 'duffing__456']
all_quantile = np.arange(0.05, 1.05, 0.10)

# Initiate instance of ScoreComputation class
manipulation_results = ScoreComputation(
    results_path=results_path, names=names, plot_signal=plot_signal
)

# compute relevances using prescribed methods
for method in method_relevance:
    df_rel = manipulation_results.compute_relevance(method)

# Iterate over replacement methods
for replace_method in replacement_methods:

    # Iterate over interpretability methods
    for method in method_relevance:
        _ = manipulation_results.compute_scores_wrapper(
            all_quantile, method, replace_method
        )
        manipulation_results.create_summary(method, replace_method)
    save_results = os.path.join(results_path, "interpretability_results")
    manipulation_results.plot_summary_results(save_results, replace_method)
