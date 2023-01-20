#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
	Written by G. Mengaldo & H.Turbé, May 2021.
"""
import argparse
import os
import re
import sys

import numpy as np

FILEPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FILEPATH, "../"))
from postprocessing_pytorch.manipulation_results import ScoreComputation
from shared_utils.utils_path import trained_model_path
from shared_utils.utils_visualization import plot_DeltaS_results, plot_additional_results

# parse command-line parameters
parser = argparse.ArgumentParser(description="Process files.")
parser.add_argument(
    "--model_path",
    type=str,
    help="Path to the folder with the model and its results",
    default=f"{trained_model_path}/forda_cnn_normal",
)
parser.add_argument(
    "--sample_file",
    help="Name of numpy array with sample. Look into assets/samples_path",
    # default=None,
    default="sample_forda.npy",
)
parser.add_argument(
    "--method_relevance",
    help="List with the relevance method to use for interpretability results",
    default=[
        "integrated_gradients",
        "deeplift",
        "deepliftshap",
        "gradshap",
        "shapleyvalue",
        "kernelshap",
    ],
    nargs="+",
)

args = parser.parse_args()
model_path = args.model_path
model_path = os.path.abspath(model_path)
method_relevance = args.method_relevance
path_asset = os.path.join(FILEPATH, "..", "assets", "samples_post")
if args.sample_file is not None:
    names_sample = np.load(os.path.join(path_asset, args.sample_file)).tolist()
else:
    names_sample = None
# interpretability
plot_signal = 0  # plot every x signal or False & 0 to plot no signal
res = [
    f
    for f in os.listdir(model_path)
    if re.search(r"config[a-zA-Z0-9_]+(.yaml|.xml)", f)
]
fc = os.path.abspath(os.path.join(model_path, res[0]))

all_qfeatures = np.arange(0.05, 1.05, 0.10)
all_qfeatures = np.round(all_qfeatures, 2).tolist()
all_qfeatures.append(1.0)

all_qfeatures = [0.05, 0.15,1.0]

manipulation_results = ScoreComputation(
    model_path=model_path,
    names=names_sample,
    model_output="probabilities",
    # model_output ="logits",
    plot_signal=plot_signal,
)

for method in method_relevance:
    df_rel = manipulation_results.compute_relevance(method)

for method in method_relevance:
    _ = manipulation_results.compute_scores_wrapper(
        all_qfeatures, method
    )
    manipulation_results.create_summary(method)
manipulation_results.summarise_results()

save_results_path = manipulation_results.save_results
plot_DeltaS_results(save_results_path)
plot_additional_results(save_results_path)
