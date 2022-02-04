#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
	Written by H. Turbé, G. Mengaldo November 2021.
    Scripts for manipulation of results
"""

import glob
import os
import re
import sys
from functools import reduce
from os.path import join as pj
from typing import Union
import warnings
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import scipy.integrate as integrate
import torch
from torch.utils.data import DataLoader, TensorDataset
from captum.attr import (
    LRP,
    DeepLift,
    DeepLiftShap,
    GradientShap,
    IntegratedGradients,
    KernelShap,
    Lime,
    NoiseTunnel,
    Saliency,
    ShapleyValueSampling,
)

from matplotlib.gridspec import GridSpec
from pandas.plotting import register_matplotlib_converters
from sklearn import preprocessing
from tqdm import tqdm

# custom packages
FILEPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(pj(FILEPATH, "../utils"))
sys.path.append(pj(FILEPATH, "../data_handling"))
sys.path.append(pj(FILEPATH, ".."))

import utils_data as utils_data
from learning_pytorch.Trainable import Trainable
from ML_Dataset import ML_Dataset

# Dictionaries for methods which require a baseline to compute the relevance
methods_require_baseline = {
    "integrated_gradients": [IntegratedGradients, "mean"],
    "deeplift": [DeepLift, "mean"],
    "deepliftshap": [DeepLiftShap, "sample"],
    "gradshap": [GradientShap, "sample"],
    "shapleyvalue": [ShapleyValueSampling, "mean"],
    "kernelshap": [KernelShap, "mean"],
    "lime": [Lime, "mean"],
}
# Dictionaries for methods which do not require a baseline to compute the relevance
method_wo_baseline = {"saliency": Saliency, "lrp": LRP}

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")
# torch.backends.cudnn.enabled=False

# Define plot properties for matplotlib
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18
mpl.rcParams["lines.linewidth"] = 1.0
register_matplotlib_converters()
plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize


def return_baseline(
    type_baseline: str, signal: Union[np.array, torch.Tensor]
) -> torch.tensor:
    """Function to return baseline
    Parameters
    ----------
    type_baseline : str
        Type of baseline to be used.
    signal : np.array
        Signal to be used to compute baseline.
    Returns
    -------
    Baseline: np.array
    """
    s = signal.copy()
    if not isinstance(s, torch.Tensor):
        s = torch.tensor(s)

    nb_sample = min(signal.shape[0], 250)
    if type_baseline == "zeros":
        # return baseline as zeros
        baseline = torch.zeros(s[:1].shape)
    elif type_baseline == "random":
        # return baseline as random values
        baseline = torch.rand(s[:1].shape)
    elif type_baseline == "mean":
        # return baseline as mean of signal
        baseline = torch.mean(s, dim=0)
    elif type_baseline == "sample":
        # return baseline as sample of given size of signal
        idx_random = np.random.permutation(np.arange(s.shape[0]))
        idx_random = idx_random[:nb_sample]
        baseline = s[idx_random]

    baseline = baseline.type(torch.float32)
    if baseline.ndim == 2:
        baseline = baseline[None, :, :]
    return baseline


def attribute_series_features(algorithm, net, input, labels_idx, **kwargs):
    """Function to compute the attributions of a given algorithm on a given
    input and labels.
    Parameters
    ----------
    algorithm : str
        Name of the algorithm to be used.
    net : torch.nn.Module
        Model to be used.
    input : np.array
        Input to be used.
    labels_idx : np.array
        Labels to be used.
    kwargs : dict
        Additional arguments to be used.
    Returns
    -------
    attributions: np.array
        Attributions computed by the algorithm.
    """

    net.zero_grad()
    if isinstance(input, np.ndarray):
        input = torch.tensor(input.astype(np.float32))
    input = input.to(device)

    labels_idx = torch.tensor(labels_idx.astype(np.int64)).to(device)
    if isinstance(algorithm, ShapleyValueSampling):
        kwargs["perturbations_per_eval"] = 16
    # algorithm = NoiseTunnel(algorithm)
    tensor_attributions = algorithm.attribute(input, target=labels_idx, **kwargs)
    # draw_baseline_from_distrib = True

    return tensor_attributions


class ScoreComputation:
    """
    Main class to compute the relevance as well as the different key interpretability metrics

    """

    def __init__(self, results_path, names, plot_signal=False):
        """
        Parameters
        ----------
        results_path : str
            Path to the folder where the results are stored.
        names : list
            List of names of the datasets to be used.
        plot_signal : bool
            Boolean to indicate if the signal should be plotted.
        """
        self.results_path = results_path
        self.names = names
        self.plot_signal = plot_signal
        if self.plot_signal == False or self.plot_signal == None:
            self.plot_signal = 0
        self.save_results = os.path.join(self.results_path, "interpretability_results")

        if not os.path.exists(self.save_results):
            os.makedirs(self.save_results)

        self.__init_samples()

    def compute_relevance(self, method_relevance):
        """
        Function to compute the relevance for a given method.
        Parameters
        ----------
        method_relevance : str
            Name of the method to be used to compute the relevance.
        Returns
        -------
        attributions_relevance : pd.DataFrame
            Datframe with relevance computed by the method.
        """

        pred = torch.tensor([], device=device, requires_grad=False)
        batch_size = 4
        batched_sample = np.array_split(
            self.np_signal, np.ceil(self.np_signal.shape[0] / batch_size)
        )
        for sample in batched_sample:
            with torch.no_grad():
                pred = torch.cat(
                    (
                        pred,
                        self.model(
                            torch.Tensor(sample.astype(np.float32)).squeeze().to(device)
                        ),
                    ),
                    0,
                )

        # Extract target idx of the labels
        target_class_idx = np.argmax(pred.detach().cpu().numpy(), axis=1)
        print(f"Evaluating relevance using {method_relevance}")
        torch_rel = torch.tensor([], device=device, requires_grad=False)

        # We batch samples together to compute the relevance
        batched_idx = np.array_split(
            target_class_idx, np.ceil(self.np_signal.shape[0] / batch_size)
        )

        if len(batched_idx) != len(batched_sample):
            raise ValueError("Samples and Target should have the same size")

        # Compute relevance for methods with a baseline
        if method_relevance in methods_require_baseline.keys():
            rel_method, baseline_type = methods_require_baseline[method_relevance]
            rel_method = rel_method(self.model)
            baseline = return_baseline(baseline_type, self.baseline).to(device)
            if rel_method.has_convergence_delta():
                for (sample, target_idx) in tqdm(
                    zip(batched_sample, batched_idx),
                    total=len(batched_sample),
                    desc="Computing relevance",
                ):
                    tmp_rel, delta = attribute_series_features(
                        rel_method,
                        self.model,
                        torch.Tensor(sample.astype(np.float32)).squeeze().to(device),
                        target_idx,
                        baselines=baseline,
                        return_convergence_delta=True,
                    )
                    torch_rel = torch.cat((torch_rel, tmp_rel), 0)
                    if torch.sum(delta) > 0.1:
                        print(
                            f"Relevance delta for method {method_relevance} is {delta}"
                        )
            else:
                for (sample, target_idx) in tqdm(
                    zip(batched_sample, batched_idx),
                    total=len(batched_sample),
                    desc="Computing relevance",
                ):
                    tmp_rel = attribute_series_features(
                        rel_method,
                        self.model,
                        torch.Tensor(sample.astype(np.float32)).squeeze().to(device),
                        target_idx,
                        baselines=baseline,
                    )
                    torch_rel = torch.cat((torch_rel, tmp_rel), 0)

        # Compute relevance for methods without a baseline
        elif method_relevance in method_wo_baseline.keys():
            rel_method = method_wo_baseline[method_relevance](self.model)
            if rel_method.has_convergence_delta():
                for (sample, target_idx) in tqdm(
                    zip(batched_sample, batched_idx),
                    total=len(batched_sample),
                    desc="Computing relevance",
                ):
                    tmp_rel, delta = attribute_series_features(
                        rel_method,
                        self.model,
                        torch.Tensor(sample.astype(np.float32)).squeeze(),
                        target_idx,
                        return_convergence_delta=True,
                    )
                    torch_rel = torch.cat((torch_rel, tmp_rel), 0)

                    if torch.sum(delta) > 0.1:
                        print(
                            f"Relevance delta for method {method_relevance} is {delta}"
                        )
            else:
                for (sample, target_idx) in tqdm(
                    zip(batched_sample, batched_idx),
                    total=len(batched_sample),
                    desc="Computing relevance",
                ):
                    tmp_rel = attribute_series_features(
                        rel_method,
                        self.model,
                        torch.Tensor(sample.astype(np.float32)).squeeze(),
                        target_idx,
                    )
                    torch_rel = torch.cat((torch_rel, tmp_rel), 0)

        else:
            raise ValueError(
                f"{method_relevance} \
                is not available. Relevance method should be one of \
               {methods_require_baseline.keys()} or {method_wo_baseline}  "
            )

        np_rel = torch_rel.squeeze().cpu().detach().numpy()

        # If channel first == true we convert to [batch, time-step, feature]
        if self.config["MANIPULATION"]["channel_first"].lower() == "true":
            print("converting to [batch, time-step, feature] format")
            np_rel = np.transpose(np_rel, (0, 2, 1))

        np_rel_df = np_rel.transpose((1, 0, 2)).reshape((np_rel.shape[1], -1))
        df_relevance = pd.DataFrame(np_rel_df)

        names_all = np.repeat(self.list_sample_names, self.nb_features)
        list_signal_names_flat = [
            item for sublist in self.list_signal_names for item in sublist
        ]
        names_columns = [
            f"{names_all[idx]}_{list_signal_names_flat[idx]}"
            for idx in range(np_rel_df.shape[1])
        ]
        df_relevance.columns = names_columns

        df_relevance.to_csv(
            pj(self.save_results, f"intepretability_{method_relevance}.csv")
        )
        np.save(
            pj(self.save_results, f"intepretability_{method_relevance}.npy"),
            np_rel,
        )
        return df_relevance

    def compute_scores_wrapper(
        self, quantiles, method_relevance, replacement_method, neg_relevance=False
    ):
        """
        Wrapper to compute metrics at all prescribed quantiles
        Parameters
        ----------
        quantiles: list
            List of quantiles to compute
        method_relevance: str
            Interpretability method to compute metrics
        replacement_method: str
            Method to oclude samples one of ['normal','permutation']
        neg_relevance: bool
            If true, relevance is computed for negative relevance, else only consider positive relevance
        """

        if replacement_method != "normal" and replacement_method != "permutation":
            raise ValueError(
                "Replacement method should be one of `normal` or `permutation`"
            )

        print(
            f"Computing metrics using {method_relevance} and {replacement_method} alteration"
        )
        for quantile in quantiles:
            _ = self.__compute_scores(
                quantile, method_relevance, replacement_method, neg_relevance
            )

    def create_summary(self, method_relevance, replacement_method):
        """
        Function to create the summary of the results
        Parameters
        ----------
        method_relevance: str
            intepretability method of interest
        replacement_method: str
            replacement method of interest
        Returns
        -------
        df_results: pd.DataFrame
            dataframe with the summary of the results

        """
        # Extract csv with results for all quantile
        path_files = glob.glob(
            os.path.join(
                self.save_results,
                replacement_method,
                f"interpretability_{method_relevance}__0*",
                "results_interp.csv",
            )
        )

        required_columns = [
            "mean_n_pts_removed",
            "mean_ratio_pts_removed",
            "mean_tic",
            "score1",
            "delta_score1",
            "delta_score2",
            "metric_score",
            "expert_score",
            "metric_score_random",
        ]
        quantile = [float(x.split(os.sep)[-2].split("__")[-1]) for x in path_files]
        df_results = pd.DataFrame(index=quantile, columns=required_columns)
        # Compute summary across each metric
        for idx, path in enumerate(path_files):
            df_quantile = pd.read_csv(path, index_col=0)
            df_tmp = df_quantile.loc[:, required_columns]
            df_tmp = pd.DataFrame(
                np.where(np.isinf(df_tmp.values), np.nan, df_tmp.values),
                index=df_tmp.index,
                columns=df_tmp.columns,
            )
            df_tmp = df_tmp.dropna(axis=1, how="all")
            df_results.loc[quantile[idx], df_tmp.columns] = np.nanmean(
                df_tmp.values, axis=0
            )

            df_accuracy = df_quantile.loc[
                :,
                [
                    "initial_classification",
                    "modified_classification",
                    "random_classification",
                ],
            ]
            df_results.loc[quantile[idx], df_accuracy.columns] = (
                np.sum(df_accuracy, axis=0) / df_accuracy.shape[0]
            )

        df_results.to_csv(
            os.path.join(
                self.save_results,
                replacement_method,
                f"summary_relevance_{method_relevance}.csv",
            )
        )

    def __compute_scores(
        self, quantile, method_relevance, replacement_method, neg_relevance
    ):
        """
        Compute metrics at a given quantile
        Parameters
        ----------
        quantile: float
            Quantile to compute metrics at
        method_relevance: str
            Interpretability method to compute metrics
        replacement_method: str
            Method to oclude samples one of ['normal','permutation']
        neg_relevance: bool
            If true, relevance is computed for negative relevance, else only consider positive relevance
        Returns
        -------
        df_results: pd.DataFrame
            Dataframe containing metrics
        """

        path_save = pj(
            self.save_results,
            replacement_method,
            f"interpretability_{method_relevance}__{quantile:.2f}",
        )
        if not os.path.exists(path_save):
            os.makedirs(path_save)

        df_rel = pd.read_csv(
            pj(self.save_results, f"intepretability_{method_relevance}.csv")
        )
        df_results = pd.DataFrame()
        np_all_modified_signal = np.empty(self.np_signal.shape)
        np_all_randomly_modified_signal = np.empty(self.np_signal.shape)

        # Iterate across all samples to compute metrics
        for idx in tqdm(range(self.np_signal.shape[0])):
            target_encoded = self.target_encoded[idx]
            name_sample = self.list_sample_names[idx]
            signal_names = self.list_signal_names[idx]
            signal = self.np_signal[idx, :, :].copy()
            # For this step we always convert to [timesteps,features]
            if signal.shape[-1] != self.nb_features:
                signal = np.transpose(signal, (1, 0))

            bool_columns = [f"{name_sample}_" in x for x in df_rel.columns]
            interp = df_rel.loc[:, bool_columns].values
            if interp.shape[1] > self.nb_features:
                raise ValueError(
                    f"""
                    Interp dataframe shape {interp.shape[1]} 
                    is larger than nb feature {self.nb_features}"""
                )
            if interp.shape[1] == 0 or interp[interp >= 0].shape[0] == 0:
                continue

            class_name = self.encoder.inverse_transform(target_encoded.reshape(1, -1))
            dict_results = self.__interpretability_metrics(
                signal=signal,
                idx_sample=idx,
                name_sample=name_sample,
                signal_names=signal_names,
                target_encoded=target_encoded,
                interp=interp,
                path_save=path_save,
                quantile=quantile,
                replacement_method=replacement_method,
                neg_relevance=neg_relevance,
            )

            dict_summary = {
                "class_name": ",".join(class_name.astype(str)[0].tolist()),
                "min_Rx": np.min(interp),
                "max_Rx": np.max(interp),
                "mean_Rx": np.mean(interp),
                "median_Rx": np.median(interp),
                "mean_n_pts_removed": dict_results["n_pts_removed"].mean(),
                "mean_ratio_pts_removed": dict_results["ratio_pts_removed"].mean(),
                "sum_rel_removed": dict_results["sum_rel_removed"],
                "mean_tic": dict_results["tic"].mean(),
                "sample_expected_values": dict_results["sample_expected_values"],
                "expert_score": dict_results["expert_score"].mean(),
            }

            df_tmp = pd.DataFrame(dict_summary, index=[name_sample])
            np_all_modified_signal[idx, :, :] = dict_results["modified_signal"]
            np_all_randomly_modified_signal[idx, :, :] = dict_results[
                "randomly_modified_signal"
            ]
            # concat all metrics
            df_results = pd.concat([df_results, df_tmp], axis=0)

        # Compute model prediction on initial, modified and randomly modified signals
        df_score = self.__compute_score_pred(
            np_all_modified_signal, np_all_randomly_modified_signal
        )
        if df_results.shape[0] != df_score.shape[0]:
            raise ValueError("Size not matching")
        df_results = pd.concat([df_results, df_score], axis=1)
        # save results to csv

        df_results = df_results.sort_index()

        df_results.to_csv(pj(path_save, "results_interp.csv"))
        return df_results

    def __interpretability_metrics(
        self,
        idx_sample,
        signal,
        name_sample,
        signal_names,
        target_encoded,
        interp,
        path_save,
        quantile,
        replacement_method,
        neg_relevance,
    ):
        """
        Function to compute interpretability metrics based on positive relevance
        ----------
        signal: np.array
            signal to be interpreted
        idx_sample: int
            idx of the current sample used to plot only given sample
        name_sample: str
            name of the sample
        signal_names: list
            list of signal names
        target_encoded: numpy array
            target encoded
        interp: numpy array
            relevance scores
        path_save: str
            path to save results
        quantile: float
            quantile to be used
        ----------

        Returns:
        dict_results: dict
            dict with results for the following keys
            frac_pts_rel: fraction of points removed  againg all points
                        with relevance score > 0 | < 0
            tic: Time Information Content
            n_pts_removed: number of points removed
            ratio_pts_removed: ratio of points removed
            expert_score: ratio of weighted rel with expert weights vs. rel within quantile
            sum_rel_removed: sum of rel removed
            sample_expected_values: expected value of the sample
            modified_signal: numpy array
                signal with removed points
            randomly_modified_signal: numpy array
                signal with points removed randomly
        ----------------
        """

        data_path = os.path.expanduser(self.config["CONFIGURATION"]["data_path"])
        path_weight = os.path.abspath(
            os.path.join(data_path, f"../expert_weights/{name_sample}.csv")
        )

        # load expert weights if they exist
        if os.path.exists(path_weight):
            df_weights = pd.read_csv(
                os.path.join(path_weight), index_col=0
            ).reset_index(drop=True)
            if df_weights.shape[1] == 0:
                df_weights = pd.DataFrame(np.full(signal.shape, np.nan))
        else:
            df_weights = pd.DataFrame(np.full(signal.shape, np.nan))

        epsilon = 1e-12

        frac_pts_rel = np.zeros(interp.shape[1])
        tic = np.zeros(interp.shape[1])
        expert_score = np.zeros(interp.shape[1])

        n_pts_removed = np.zeros(interp.shape[1])
        ratio_pts_removed = np.zeros(interp.shape[1])

        if neg_relevance:
            # Consider case where we are interested in negative relevance
            sn_interp_all = interp[interp < 0]
            if sn_interp_all.shape[0] > 0:
                sn_interp_q = np.quantile(sn_interp_all, 1 - quantile)
            else:
                sn_interp_q = -1e-8
            sn_interp_all = interp[interp < 0]
            thres_p = sn_interp_q

            # In case we are intereseted in evaluating impact negative relevance we cannot start
            # from initial signal, our "baseline" signal is therefore the signal with half of the
            # positive relevance corrupted
            signal = pd.DataFrame(signal)
            median_p = np.median(interp[interp > 0])  # median of the positive relevance
            for coord in range(signal.shape[1]):
                s = signal.iloc[:, coord]
                df_i = pd.Series(interp[:, coord])
                df_i = df_i.loc[df_i > median_p]
                val_above = s[df_i.index]
                shuffled_array = val_above.values.copy()
                np.random.shuffle(shuffled_array)
                s.loc[df_i.index] = shuffled_array
                signal.iloc[:, coord] = s

            modified_signal = signal.copy()
            randomly_modified_signal = signal.copy()

        else:
            # Case interested in positive relevance
            sp_interp_all = interp[interp >= 0]
            if sp_interp_all.shape[0] == 0:
                breakpoint()
            sp_interp_q = np.quantile(sp_interp_all, quantile)
            thres_p = sp_interp_q

            signal = pd.DataFrame(signal)
            modified_signal = signal.copy()
            randomly_modified_signal = signal.copy()

        for coord in range(signal.shape[1]):
            weight_tmp = df_weights.iloc[:, coord]
            # calculate quantile for function bounds
            s = signal.loc[:, coord]
            s_interp = pd.Series(interp[:, coord].copy())
            if neg_relevance:
                sp_interp = s_interp[s_interp < 0]
                sp_tmp = sp_interp[sp_interp <= thres_p]
            else:
                sp_interp = s_interp[s_interp >= 0]
                sp_tmp = sp_interp[sp_interp >= thres_p]

            if sp_tmp.shape[0] == 0:
                frac_pts_rel[coord] = 0
                tic[coord] = 0
                n_pts_removed[coord] = 0
                ratio_pts_removed[coord] = 0
            else:
                # statistical quantities positive
                sp_interp_var = np.var(sp_interp)

                # sp_interp_q = np.quantile(sp_interp, quantile)
                # thres_p = sp_interp_q

                if sp_interp_var < 1e-8:
                    frac_pts_rel[coord] = 0
                else:
                    frac_pts_rel[coord] = sp_tmp.shape[0] / (
                        sp_interp.shape[0] + epsilon
                    )

                if sp_tmp.shape[0] < 1:
                    tic[coord] = 0
                    # expert_score[coord] = 0
                else:
                    tic[coord] = integrate.simpson(sp_tmp) / (
                        integrate.simpson(sp_interp) + epsilon
                    )

                n_pts_removed[coord] = sp_tmp.shape[0]
                ratio_pts_removed[coord] = n_pts_removed[coord] / s.shape[0]

            sp_interp2 = s_interp[s_interp > 0].copy()
            weights_p = weight_tmp.loc[sp_interp2.index]
            count_w = len(weights_p[weights_p > 0])
            count_r = len(sp_interp2)
            count_total = len(s_interp)
            diff_wr = count_w - count_r
            diff_wr = np.max([0, diff_wr])
            ratio = np.sum(sp_interp2 * weights_p) / sum(sp_interp2)
            expert_score[coord] = ratio * (1 - diff_wr / count_total)

            if replacement_method == "normal":
                modified_signal.loc[sp_tmp.index, coord] = (
                    np.random.normal(scale=1 / np.sqrt(3), size=sp_tmp.shape[0]) * 0.5
                )
            elif replacement_method == "permutation":
                shuffled_array = s[sp_tmp.index].values.copy()
                np.random.shuffle(shuffled_array)
                modified_signal.loc[sp_tmp.index, coord] = shuffled_array

        if replacement_method == "normal":
            randomly_modified_signal = randomly_modified_signal.values.reshape(-1)
            idx_random = np.random.permutation(
                np.arange(randomly_modified_signal.shape[0])
            )
            randomly_modified_signal[idx_random[: int(n_pts_removed.sum())]] = (
                np.random.normal(scale=1 / np.sqrt(3), size=int(n_pts_removed.sum()))
                * 0.5
            )

            randomly_modified_signal = pd.DataFrame(
                randomly_modified_signal.reshape(signal.shape)
            )

        elif replacement_method == "permutation":
            col = np.random.randint(
                0, high=signal.shape[1], size=int(n_pts_removed.sum())
            )
            val, count = np.unique(col, return_counts=True)
            for idx, col_nb in enumerate(val):
                nb_item = count[idx]
                random_idx = np.random.permutation(np.arange(s.shape[0]))[:nb_item]
                shuffled_array_random = randomly_modified_signal.loc[
                    random_idx, col_nb
                ].values.copy()
                np.random.shuffle(shuffled_array_random)
                randomly_modified_signal.loc[random_idx, col_nb] = shuffled_array_random

        if self.plot_signal > 0 and idx_sample % self.plot_signal == 0:
            _plot_signal(
                signal,
                modified_signal,
                randomly_modified_signal,
                interp,
                path_save,
                signal_names,
                name_sample,
            )
            # ------------------------------------------------------
        if neg_relevance:
            sum_rel_removed = interp[interp <= thres_p].sum()
        else:
            sum_rel_removed = interp[interp >= thres_p].sum()

        if self.config["MANIPULATION"]["channel_first"].lower() == "true":
            # We return value in correct order for model
            modified_signal = np.transpose(modified_signal.values, (1, 0))
            randomly_modified_signal = np.transpose(
                randomly_modified_signal.values, (1, 0)
            )

        sample_expected_values = self.expected_value * np.asarray(
            target_encoded.reshape(-1)
        )
        sample_expected_values = sample_expected_values[sample_expected_values != 0][0]

        dict_results = {
            "frac_pts_rel": frac_pts_rel,
            "tic": tic,
            "n_pts_removed": n_pts_removed,
            "ratio_pts_removed": ratio_pts_removed,
            "expert_score": expert_score,
            "sum_rel_removed": sum_rel_removed,
            "sample_expected_values": sample_expected_values,
            "modified_signal": modified_signal,
            "randomly_modified_signal": randomly_modified_signal,
        }
        return dict_results

    def __compute_score_pred(
        self, np_all_modified_signal, np_all_randomly_modified_signal
    ):
        """
        Function to compute the score of the prediction on inital and modified signal
        Parameters
        ----------
        np_all_modified_signal: np.array
            np array of the modified signal
        np_all_randomly_modified_signal: np.array
            np array of the randomly modified signal
        Returns
        -------
        df_score: pd.DataFrame
            dataframe with the score of the prediction on the initial and modified signal as well as diff and metric_score metric
        """

        required_columns = [
            "score1",
            "score2",
            "score3",
            "delta_score1",
            "delta_score2",
            "initial_classification",
            "modified_classification",
            "random_classification",
            "metric_score",
            "metric_score_random",
        ]
        df_score = pd.DataFrame(index=self.list_sample_names, columns=required_columns)
        batch_size = 16
        batched_sample = np.array_split(
            self.np_signal, np.ceil(self.np_signal.shape[0] / batch_size)
        )
        score1 = torch.tensor([], device=device, requires_grad=False)
        for sample in batched_sample:
            with torch.no_grad():
                score1 = torch.cat(
                    (
                        score1,
                        self.model(
                            torch.Tensor(sample.astype(np.float32)).to(device).squeeze()
                        ),
                    ),
                    0,
                )

        correct_classification = score1.argmax(
            dim=1
        ).detach().cpu().numpy() == np.argmax(self.target_encoded, axis=1)
        score1 = score1.detach().cpu().numpy() * np.asarray(self.target_encoded)
        df_score.loc[:, "score1"] = np.array([x[x != 0][0] for x in score1])
        df_score.loc[:, "initial_classification"] = correct_classification

        batched_sample = np.array_split(
            np_all_modified_signal, np.ceil(self.np_signal.shape[0] / batch_size)
        )
        score2 = torch.tensor([], device=device, requires_grad=False)
        for sample in batched_sample:
            with torch.no_grad():
                score2 = torch.cat(
                    (
                        score2,
                        self.model(
                            torch.Tensor(sample.astype(np.float32)).to(device).squeeze()
                        ),
                    ),
                    0,
                )
        correct_classification = score2.argmax(
            dim=1
        ).detach().cpu().numpy() == np.argmax(self.target_encoded, axis=1)
        score2 = score2.detach().cpu().numpy() * np.asarray(self.target_encoded)
        df_score.loc[:, "score2"] = np.array([x[x != 0][0] for x in score2])
        df_score.loc[:, "modified_classification"] = correct_classification

        batched_sample = np.array_split(
            np_all_randomly_modified_signal,
            np.ceil(self.np_signal.shape[0] / batch_size),
        )
        score3 = torch.tensor([], device=device, requires_grad=False)
        for sample in batched_sample:
            with torch.no_grad():
                score3 = torch.cat(
                    (
                        score3,
                        self.model(
                            torch.Tensor(sample.astype(np.float32))
                            .to(device)
                            .squeeze()
                            .squeeze()
                        ),
                    ),
                    0,
                )
        correct_classification = score3.argmax(
            dim=1
        ).detach().cpu().numpy() == np.argmax(self.target_encoded, axis=1)
        score3 = score3.detach().cpu().numpy() * np.asarray(self.target_encoded)
        df_score.loc[:, "score3"] = np.array([x[x != 0][0] for x in score3])
        df_score.loc[:, "random_classification"] = correct_classification

        sample_expected_values = self.expected_value * np.asarray(self.target_encoded)
        sample_expected_values = np.array(
            [x[x != 0][0] for x in sample_expected_values]
        )

        # ------------------------------------------------------
        # compute score drop
        df_score.loc[:, "delta_score1"] = (
            df_score.loc[:, "score1"] - df_score.loc[:, "score2"]
        )
        df_score.loc[:, "delta_score2"] = (
            df_score.loc[:, "score1"] - df_score.loc[:, "score3"]
        )
        metric_score = 1 - (df_score.loc[:, "score2"] - sample_expected_values) / (
            df_score.loc[:, "score1"] - sample_expected_values
        )
        metric_score_random = 1 - (
            df_score.loc[:, "score3"] - sample_expected_values
        ) / (df_score.loc[:, "score1"] - sample_expected_values)
        df_score.loc[:, "metric_score"] = np.clip(metric_score, -1, 1)
        df_score.loc[:, "metric_score_random"] = np.clip(metric_score_random, -1, 1)

        del score1
        del score2
        del score3

        return df_score

    def __init_samples(self):
        """
        Function to initialize the samples, models as well as different parameters required for the analysis
        """

        # Find config file used for the simulations
        res = [
            f
            for f in os.listdir(self.results_path)
            if re.search(r"config[a-zA-Z0-9_]+(.yaml|.xml)", f)
        ]
        fc = os.path.abspath(os.path.join(self.results_path, res[0]))

        for file in glob.glob(fc):
            print("file [fc] = ", file)
            config_file = os.path.abspath(file)

        self.config = utils_data.parse_config(config_file)
        trainer = Trainable(self.config, self.results_path, retrieve_model=True)
        self.model = trainer.train(self.config["MODEL"]).to(device).eval()
        df_test = pd.read_csv(
            os.path.join(self.results_path, "results", "results_test.csv")
        )

        if self.names == None:
            names_sample = df_test.noun_id.values.tolist()
            self.names = names_sample
            length_set = len(names_sample)

        else:
            names_sample = set(df_test.noun_id.values.tolist() + self.names)
            length_set = len(names_sample)

        ml_data = ML_Dataset(self.config)
        self.encoder = generate_encoder(ml_data, self.results_path)
        ml_data.encoder = self.encoder
        ml_data.evaluate_nb_features()
        self.nb_features = ml_data.n_features
        if self.config["MANIPULATION"].get("feature_scaling", "none").lower() != "none":
            ml_data.generate_scaler()

        # Fields to be loaded from the dataset
        all_fields = [
            ml_data.featureName,
            ml_data.targetName,
            "noun_id",
            "signal_names",
        ]
        count_large = 0
        target_encoded_all = []
        list_signal_names = []
        list_sample_names = []
        with ml_data.retrieve_sample(
            names_sample, fields=all_fields
        ) as reader_filtered:
            for idx, sample in enumerate(reader_filtered):
                if idx == 0:
                    np_signal = np.zeros(
                        [length_set, sample.signal.shape[0], sample.signal.shape[1]]
                    )
                np_signal[idx, :, :] = sample.signal.astype(np.float32)
                list_sample_names.append(np.array(sample.noun_id).astype(str).tolist())
                target_encoded_all.append(getattr(sample, ml_data.targetName))
                list_signal_names.append(
                    np.array(sample.signal_names).astype(str).tolist()
                )
        # make sure that we remove the lines with zeros created initially in the array
        np_signal = np_signal[: len(list_sample_names)]
        name_not_in_list = [x for x in self.names if x not in list_sample_names]
        if len(name_not_in_list) != 0:
            print(f"{name_not_in_list} in name array are not included in the test set")
            self.names = list(set(self.names) - set(name_not_in_list))
        index_to_explain = np.where(np.isin(list_sample_names, self.names))
        self.np_signal = np_signal[index_to_explain]

        # ------------------------------------------------------
        # define expected value of the networs with different options

        batched_signal = np.array_split(np_signal, np.ceil(np_signal.shape[0] / 8))
        pred = torch.tensor([], device=device)
        for sample in batched_signal:
            with torch.no_grad():
                pred = torch.cat(
                    (
                        pred,
                        self.model(
                            torch.Tensor(sample.astype(np.float32)).to(device).squeeze()
                        ),
                    ),
                    0,
                )

        # option 1: mean of the prediction on the whole dataset
        self.expected_value = pred.mean(axis=0).detach().cpu().numpy()

        # option 2: mean of the prediction on random samples
        # np_random = self.np_signal.copy().reshape(-1)
        # np_random[:] = np.random.normal(scale=1 / np.sqrt(3), size=int(np_random.shape[0])) * 0.5
        # np_random = np_random.reshape(np_signal.shape)
        # self.expected_value = self.model(torch.tensor(np_random[:50].astype(np.float32)).to(device)).mean(axis=0).detach().cpu().numpy()

        # option 3: mean of the prediction on the whole dataset but after multiplication by the target class and excluding 0
        # tmp =pred.detach().cpu().numpy() *np.array(target_encoded_all)
        # tmp[tmp ==0]=np.NaN
        # self.expected_value = np.nanmean(tmp,axis =0)
        #
        # get clean signals
        self.baseline = self.np_signal.copy()
        self.target_encoded = np.array(target_encoded_all)[index_to_explain]
        self.list_sample_names = np.array(list_sample_names)[index_to_explain].tolist()
        self.list_signal_names = np.array(list_signal_names)[index_to_explain].tolist()

    @staticmethod
    def plot_summary_results(save_results, replacement_method):
        """
        Static method to create plot summarising the results of the analysis for the different interpretability methods
        Parameters
        ----------
        save_results: str
            path to the folder where the results are stored
        replacement_method: str
            name of the method used to occlude the features

        Returns
        -------
        None
        """
        import matplotlib.colors as mcolors
        import matplotlib.ticker as plticker
        import seaborn as sns

        color_s = sns.color_palette("tab10", 7)
        name_method_dict = {
            "gradshap": {"name": "GradSHAP", "color": color_s[0], "linestyle": "-o"},
            "integrated_gradients": {
                "name": "Integrated Gradient",
                "color": color_s[1],
                "linestyle": "--o",
            },
            "shapleyvalue": {
                "name": "Shapley Sampling",
                "color": color_s[2],
                "linestyle": "-.o",
            },
            "deeplift": {"name": "DeepLIFT", "color": color_s[3], "linestyle": ":o"},
            "saliency": {"name": "Saliency", "color": color_s[4], "linestyle": "-o"},
            "kernelshap": {
                "name": "KernelSHAP",
                "color": color_s[5],
                "linestyle": "--o",
            },
            "shapley_sampling": {
                "name": "Shapley Sampling",
                "color": color_s[6],
                "linestyle": "-.o",
            },
        }

        path_summary = glob.glob(
            os.path.join(save_results, replacement_method, "summary_relevance_*.csv")
        )
        results_all = {}
        for path in path_summary:
            df_results = pd.read_csv(path, index_col=0)
            name_file = os.path.split(path)[-1]
            method = re.search("summary_relevance_(.+)\.csv", name_file).group(1)
            results_all[method] = df_results
        df_metrics = compute_summary_metrics(results_all)
        df_metrics.to_csv(
            os.path.join(save_results, replacement_method, "metrics_methods.csv")
        )
        fig = plt.figure(figsize=(15, 5))
        width_ratios = [0.33, 0.33, 0.33]
        gridspec = GridSpec(ncols=3, nrows=1, figure=fig, width_ratios=width_ratios)

        # plot 0
        ax0 = plt.subplot(gridspec[0])
        ax1 = plt.subplot(gridspec[1])
        ax2 = plt.subplot(gridspec[2])

        ax0.set_xlabel("$\\tilde{N}_{\\mathrm{r}}$")
        ax0.set_ylabel("$\\tilde{S}_{\\mathbb{E}}$")

        ax1.set_xlabel("TIC")
        ax1.set_ylabel("$\\tilde{S}_{\\mathbb{E}}$")

        ax2.set_xlabel("$\\tilde{N}_{\\mathrm{r}}$")
        ax2.set_ylabel("Accuracy")
        ax0.set_xlim([0, 1])
        ax1.set_xlim([0, 1])
        ax0.set_ylim(bottom=0)
        ax1.set_ylim(bottom=0)
        loc = plticker.MultipleLocator(
            base=0.2
        )  # this locator puts ticks at regular intervals
        ax0.yaxis.set_major_locator(loc)
        ax1.yaxis.set_major_locator(loc)
        ax2.set_xlim([0, 1])

        legend_handles = []
        for idx, key in enumerate(results_all.keys()):
            if key in name_method_dict.keys():
                name_method = name_method_dict[key]["name"]
                color_method = name_method_dict[key]["color"]
                linestyle_method = name_method_dict[key]["linestyle"]
            else:
                name_method = key
            df_tmp = results_all[key]
            df_tmp = df_tmp.sort_index()
            quantile = df_tmp.index
            tmp = ax0.plot(
                # quantile,
                df_tmp.loc[:, "mean_ratio_pts_removed"],
                # (df_tmp.loc[:, "delta_score1"] / df_tmp.loc[:, "score1"]),
                df_tmp.loc[:, "metric_score"],
                "-o",
                color=color_method,
                label=f"{name_method}",
            )
            legend_handles.append(tmp[0])
            ax1.plot(
                df_tmp.loc[:, "mean_tic"],
                df_tmp.loc[:, "metric_score"],
                "-o",
                color=color_method,
                label=f"{name_method}",
            )

            ax2.plot(
                df_tmp.loc[:, "mean_ratio_pts_removed"],
                df_tmp.loc[:, "modified_classification"],
                "-o",
                color=color_method,
                label=f"{name_method}",
            )

            if key == "saliency":
                tmp = ax0.plot(
                    # quantile,
                    df_tmp.loc[:, "mean_ratio_pts_removed"],
                    #    (df_tmp.loc[:, "delta_score2"] / df_tmp.loc[:, "score1"]),
                    df_tmp.loc[:, "metric_score_random"],
                    "-o",
                    color="black",
                    label="Random",
                )
                legend_handles.append(tmp[0])

                tmp = ax1.plot(
                    np.linspace(start=0.0, stop=0.95, num=20),
                    np.linspace(start=0.0, stop=0.95, num=20),
                    color="black",
                    linestyle="--",
                    label="Theoretical estimation",
                )
                legend_handles.append(tmp[0])
                ax2.plot(
                    # quantile,
                    df_tmp.loc[:, "mean_ratio_pts_removed"],
                    df_tmp.loc[:, "random_classification"],
                    "-o",
                    color="black",
                    label="Random",
                )
            ax0.legend(
                handles=legend_handles, bbox_to_anchor=(-0.3, 1.0), loc="upper right"
            )
        plt.tight_layout()
        fig_path = os.path.join(
            save_results, replacement_method, "visualization_results"
        )
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        plt.savefig(os.path.join(fig_path, "score_mean.png"), dpi=200)
        plt.close()


def generate_encoder(ml_data, simulation_path):
    """
    Generate the encoder for the data.
    Parameters
    ----------
    ml_data:
        ml_data class for the simulations
    simulation_path:
        path to the simulation
    Returns
    -------
    encoder:
        encoder for the data
    """

    classes = np.load(
        os.path.join(simulation_path, "classes_encoder.npy"), allow_pickle=True
    )
    with ml_data.create_reader(schema_fields=[ml_data.targetName]) as reader:
        target_dataset = [l[0] for l in reader]
        if np.array(target_dataset).dtype.type is np.string_:
            target_dataset = np.array(target_dataset).astype(str)
        if ml_data.selected_classes != None and "all" not in ml_data.selected_classes:
            print(f"Selecting only {ml_data.selected_classes}")
            tmp = [reduce(lambda i, j: np.concatenate((i, j)), target_dataset)]
            unique = set(tmp[0])
            ml_data.list_remove = [
                x for x in unique if x not in ml_data.selected_classes
            ]
            target_dataset = [
                np.setdiff1d(x, ml_data.list_remove) for x in target_dataset
            ]
            # target_dataset = [np.zeros(1).astype(int) \
            #  if x.shape[0] ==0 else x.astype(int) for x in new_target ]
        if ml_data.config["MANIPULATION"]["target_encoding"] == "one_hot_encoder":
            classes = [np.array([x.replace("x0_", "") for x in classes.astype(str)])]
            encoder = preprocessing.OneHotEncoder(categories=classes)
            if isinstance(target_dataset, list):
                # we want to deal with case where we onehot encode a list of labels
                target_dataset = np.array(
                    [",".join(x.astype(str).tolist()) for x in target_dataset]
                )
            target_dataset = np.array(["0" if x == "" else x for x in target_dataset])
            encoder.fit(target_dataset.reshape(-1, 1))
        elif ml_data.config["MANIPULATION"]["target_encoding"] == "label_encoding":
            classes = np.array(classes).astype(int).tolist()
            encoder = preprocessing.MultiLabelBinarizer(classes=classes)
            encoder.fit(target_dataset)
        else:
            raise ValueError("Encoder not supported in postprocessing")
    return encoder


def _plot_signal(
    signal,
    modified_signal,
    randomly_modified_signal,
    interp,
    path_save,
    signal_names,
    name_sample,
):
    """
    Plot the signal and the modified signal along the relevance
    Parameters
    ----------
    signal: pd.DataFrame
        signal to plot
    modified_signal: pd.DataFrame
        modified signal to plot
    randomly_modified_signal: pd.DataFrame
        randomly modified signal to plot
    interp np.array:
        interpolation method
    path_save: str
        path to save the figures
    signal_names: list
        names of the signals
    name_sample: str
        name of the sample
    """

    # plots
    # ------------------------------------------------------
    color_s = "black"
    color_l = "tab:blue"
    fig = plt.figure(figsize=(12, 1.5 * signal.shape[1]))
    # height_ratios = [0.33, 0.33, 0.33]
    gridspec = GridSpec(
        ncols=1,
        nrows=signal.shape[1],
        figure=fig,
        # height_ratios=height_ratios
    )
    for idx in range(signal.shape[1]):
        if idx == 0:
            ax0 = plt.subplot(gridspec[idx])
            plt.plot(signal.loc[:, idx], color=color_s)
            plt.plot(
                modified_signal.loc[:, idx],
                color="red",
                label="Modified signal",
            )
            plt.plot(
                randomly_modified_signal.loc[:, idx],
                color="green",
                label="Randomly modified signal",
            )
            plt.legend(bbox_to_anchor=(1.1, 1.0), loc="upper left")
            plt.ylabel("Signal [" + signal_names[idx] + "]", color=color_s)
            plt.tick_params(axis="y", labelcolor=color_s)
            ax_t = ax0.twinx()
            ax_t.plot(interp[:, idx], color=color_l)
            ax_t.tick_params(axis="y", labelcolor=color_l)
            ax_t.set_ylabel("Relevance", color=color_l)
            plt.setp(ax0.get_xticklabels(), visible=False)
        else:
            ax = plt.subplot(gridspec[idx], sharex=ax0)
            plt.plot(signal.loc[:, idx], color=color_s)
            plt.plot(
                modified_signal.loc[:, idx],
                color="red",
                label="Modified signal",
            )
            plt.plot(
                randomly_modified_signal.loc[:, idx],
                color="green",
                label="Randomly modified signal",
            )
            plt.ylabel("Signal [" + signal_names[idx] + "]", color=color_s)
            if idx == signal.shape[1] - 1:
                plt.xlabel("Time")
            plt.tick_params(axis="y", labelcolor=color_s)
            ax_t = ax.twinx()
            ax_t.plot(interp[:, idx], color=color_l)
            ax_t.tick_params(axis="y", labelcolor=color_l)
            ax_t.set_ylabel("LRP", color=color_l)
            # make these tick labels invisible
            if idx != signal.shape[1] - 1:
                plt.setp(ax.get_xticklabels(), visible=False)

    fig_path = os.path.join(path_save, "figures")
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, str(name_sample) + ".png"), dpi=200)
    plt.close()


def new_seriesZero(x, y):
    """Function to include zero intercept in serie. This is required to correctly
    separate the positive and negative area which are required to compute the
    various interpretability metrics

    Args:
        x ([type]): [description]
        y ([type]): [description]

    Returns:
        [type]: [description]
    """
    zero_crossings = np.where(np.diff(np.sign(y) >= 0))[0]
    x_new = x.copy()
    y_new = y.copy()

    list_intercept = []
    for val in zero_crossings:
        x0 = x[val]
        x1 = x[val + 1]

        y0 = y[val]
        y1 = y[val + 1]

        a = (y1 - y0) / (x1 - x0)
        b = y0 - a * x0
        intercept = -b / a
        list_intercept.append(intercept)

    y_new = np.insert(y_new, zero_crossings + 1, [0] * len(zero_crossings))
    x_new = np.insert(x_new, zero_crossings + 1, list_intercept)

    return x_new, y_new


def compute_area(x: np.array, y: np.array):
    """Compute positive and negaitve arrea using trapezium rule

    Args:
        x (np.array): [description]
        y (np.array): [description]

    Returns:
        np.array: [description]
    """

    y_right = y[1:]  # right endpoints
    y_left = y[:-1]  # left endpoints
    area = (y_right + y_left) / 2 * np.diff(x)

    area_pos = np.sum(area[area > 0])
    area_neg = np.sum(area[area < 0])

    return area_neg, area_pos


def compute_summary_metrics(results_all: dict):
    """This functions compute the metrics which summarise each interpretability
        method

    Args:
        results_all (dict): [description]

    Returns:
        (pd.DataFrame): [description]
    """

    name_col = ["score_drop_area", "pos_area", "neg_area", "expert_score"]
    df_metrics = pd.DataFrame(index=results_all.keys(), columns=name_col)
    for method in results_all.keys():
        df_tmp = results_all[method].sort_index(ascending=False)

        tsup = df_tmp["mean_tic"]
        metric_score = df_tmp["metric_score"]
        x = pd.concat([pd.Series(0), tsup]).reset_index(drop=True)
        y = pd.concat([pd.Series(0), metric_score]).reset_index(drop=True)
        diff = np.diff(y.to_numpy())
        diff_x = np.diff(x.to_numpy())

        measure_y = (diff / diff_x) - 1
        x_new = x[1:].to_numpy()
        x_zero, y_zero = new_seriesZero(x_new, measure_y)

        neg_area, pos_area = compute_area(x_zero, y_zero)
        df_metrics.loc[method, "pos_area"] = pos_area
        df_metrics.loc[method, "neg_area"] = neg_area
        y_integration = metric_score
        y_integration = np.append(0, y_integration)
        y_integration = np.append(y_integration, y.iloc[-1])
        x_integration = df_tmp.loc[:, "mean_ratio_pts_removed"]
        x_integration = np.append(0, x_integration)
        x_integration = np.append(x_integration, 1)
        df_metrics.loc[method, "score_drop_area"] = integrate.simpson(
            y=y_integration, x=x_integration
        )

        if method == "saliency":
            y_integration = df_tmp.loc[:, "metric_score_random"].copy()
            y_integration = np.append(0, y_integration)
            y_integration = np.append(y_integration, y.iloc[-1])
            x_integration = df_tmp.loc[:, "mean_ratio_pts_removed"]
            x_integration = np.append(0, x_integration)
            x_integration = np.append(x_integration, 1)
            df_metrics.loc["random", "score_drop_area"] = integrate.simpson(
                y=y_integration, x=x_integration
            )

        df_metrics.loc[method, "expert_score"] = df_tmp.loc[:, "expert_score"].iloc[0]
    return df_metrics
