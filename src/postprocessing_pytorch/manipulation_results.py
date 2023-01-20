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
from os.path import join as pj
from typing import  List, Union
import warnings
import more_itertools as mit
import numpy as np
import pandas as pd
import scipy.integrate as integrate
import torch
from torch import nn
from tqdm import tqdm

# custom packages
FILEPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(pj(FILEPATH, ".."))
import shared_utils.utils_data as utils_data
from learning_pytorch.data_module import PetastormDataModule as DataModule
from learning_pytorch.Trainable import Trainable
from learning_pytorch.transform.transform_signal import random_block_enforce_proba
from shared_utils.utils_path import data_path, results_interp_path
from shared_utils.utils_visualization import plot_corrupted_signal
from .method_arguments import dict_method_arguments

global device
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
warnings.simplefilter(action="ignore", category=FutureWarning)

def attribute_series_features(algorithm, net, sample, labels_idx, **kwargs):
    """Function to compute the attributions of a given algorithm on a given
    input and labels.
    Parameters
    ----------
    algorithm : str
        Name of the algorithm to be used.
    net : torch.nn.Module
        Model to be used.
    sample : np.array
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
    if isinstance(sample, np.ndarray):
        sample = torch.tensor(sample.astype(np.float32))
    sample = sample.to(device_interpretability)

    labels_idx = torch.tensor(labels_idx.astype(np.int64)).to(device_interpretability)
    # algorithm = NoiseTunnel(algorithm)
    tensor_attributions = algorithm.attribute(sample, target=labels_idx, **kwargs)
    # draw_baseline_from_distrib = True

    return tensor_attributions


def spaced_choice(low, high, delta, n_samples):
    draw = np.random.choice(
        high - low - (n_samples - 1) * delta, n_samples, replace=False
    )
    idx = np.argsort(draw)
    draw[idx] += np.arange(low, low + delta * n_samples, delta)
    return draw


def resave_score_rel_neg(result_path: str, method_relevance: str):
    """Function to resave the relevance scores of the methods in the result_path
     with the score of samples with only rel neg.
     Parameters
    ----------
    result_path : str
        Path to the result folder.
    method_relevance : str
        Name of the method to be used.

    """
    path_files = os.path.join( result_path, "results_k_feature")
    path_methods = glob.glob(
        os.path.join(path_files, f"interpretability_{method_relevance}__*")
    )
    path_methods = [x for x in path_methods if "__1.00" not in x]
    reference_folder = os.path.join(path_files, f"interpretability_{method_relevance}__1.00")
    if os.path.exists(reference_folder):
        df_reference_top = pd.read_csv(
            os.path.join(reference_folder, "results_interp__top.csv"), index_col=0
        )
        df_reference_top = df_reference_top.loc[:, "score2"]
        df_reference_top.name = "score_onlyrel_neg"

        df_reference_bottom = pd.read_csv(
            os.path.join(reference_folder, "results_interp__bottom.csv"), index_col=0
        )
        df_reference_bottom = df_reference_bottom.loc[:, "score2"]
        df_reference_bottom.name = "score_onlyrel_neg"

        for path in path_methods:
            df_tmp_top = pd.read_csv(
                os.path.join(path, "results_interp__top.csv"), index_col=0
            )
            if "metric_normalised" in df_tmp_top.columns:
                continue
            else:
                df_tmp_top = df_tmp_top.join(df_reference_top)
                df_tmp_top.loc[:, "metric_normalised"] = df_tmp_top.loc[
                    :, "delta_score1"
                ] / (
                    df_tmp_top.loc[:, "score1"] - df_tmp_top.loc[:, "score_onlyrel_neg"]
                )
                df_tmp_top.loc[:, "metric_normalised"] = np.clip(
                    df_tmp_top.loc[:, "metric_normalised"], 0, 1
                )
                df_tmp_top.to_csv(os.path.join(path, "results_interp__top.csv"))

            df_tmp_bottom = pd.read_csv(
                os.path.join(path, "results_interp__bottom.csv"), index_col=0
            )
            if "metric_normalised" in df_tmp_bottom.columns:
                continue
            else:

                df_tmp_bottom = df_tmp_bottom.join(df_reference_bottom)
                df_tmp_bottom.loc[:, "metric_normalised"] = df_tmp_bottom.loc[
                    :, "delta_score1"
                ] / (
                    df_tmp_bottom.loc[:, "score1"]
                    - df_tmp_bottom.loc[:, "score_onlyrel_neg"]
                )
                df_tmp_bottom.loc[:, "metric_normalised"] = np.clip(
                    df_tmp_bottom.loc[:, "metric_normalised"], 0, 1
                )
                df_tmp_bottom.to_csv(os.path.join(path, "results_interp__bottom.csv"))
    else:
        print(
            f"Score for reference with only negative relevance {method_relevance} not found. Please run for k_feature=1"
        )

    return True


class ScoreComputation:
    """
    Main class to compute the relevance as well as the different key interpretability metrics

    """

    def __init__(
        self, model_path, names, model_output="probabilities", plot_signal=False
    ):
        """
        Parameters
        ----------
        model_path : str
            Path to the folder where the results are stored.
        names : list
            List of names of the datasets to be used.
        plot_signal : bool
            Boolean to indicate if the signal should be plotted.
        """
        self.model_path = model_path
        self.names = names
        self.plot_signal = plot_signal
        if self.plot_signal is False or self.plot_signal is None:
            self.plot_signal = 0
        self.model_output = model_output

        name_sim = os.path.split(self.model_path)[-1]
        # self.save_results = os.path.abspath(
        # os.path.join(FILEPATH, "../../results", name_sim)
        # )
        self.save_results = os.path.join(
            results_interp_path, name_sim, "interpretability_results"
        )

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

        lstm_network_bool = any(
            [
                isinstance(module, torch.nn.modules.rnn.LSTM)
                for module in self.model.modules()
            ]
        )
        global device_interpretability
        if (lstm_network_bool) & (
            dict_method_arguments[method_relevance]["noback_cudnn"]
        ):
            print(
                "Attribution for LSTM based network is not supported on GPU. Switching to CPU"
            )
            device_interpretability = torch.device("cpu")
            self.model.to(device_interpretability)
        else:
            device_interpretability = (
                torch.device("cuda:0")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            self.model.to(device_interpretability)

        batch_size = dict_method_arguments[method_relevance].get("batch_size", 16)
        # Extract target idx of the preds
        pred_class_idx = np.argmax(self.score_pred, axis=1)
        print(f"Evaluating relevance using {method_relevance}")
        torch_rel = torch.tensor(
            [], device=device_interpretability, requires_grad=False
        )

        batched_sample = np.array_split(
            self.np_signal, np.ceil(self.np_signal.shape[0] / batch_size)
        )
        # We batch samples together to compute the relevance
        batched_idx = np.array_split(
            pred_class_idx, np.ceil(self.np_signal.shape[0] / batch_size)
        )

        if len(batched_idx) != len(batched_sample):
            raise ValueError("Samples and Target should have the same size")

        # Compute relevance for methods with a baseline
        if dict_method_arguments[method_relevance]["require_baseline"]:
            rel_method = dict_method_arguments[method_relevance]["captum_method"]
            baseline_type = dict_method_arguments[method_relevance]["baseline_type"]
            kwargs_method = dict_method_arguments[method_relevance].get("kwargs_method")

            rel_method = rel_method(
                self.model, **(kwargs_method if kwargs_method is not None else {})
            )

            baseline, expected_value = self._return_baseline(
                baseline_type, self.baseline
            )
            baseline = baseline.to(device_interpretability)
            self.model.to(device_interpretability)
            kwargs_attribution = dict_method_arguments[method_relevance].get(
                "kwargs_attribution"
            )
            if rel_method.has_convergence_delta():
                for (sample, target_idx) in tqdm(
                    zip(batched_sample, batched_idx),
                    total=len(batched_sample),
                    desc="Computing relevance",
                ):
                    tmp_rel, delta = attribute_series_features(
                        rel_method,
                        self.model,
                        torch.Tensor(sample.astype(np.float32)).to(
                            device_interpretability
                        ),
                        target_idx,
                        baselines=baseline,
                        return_convergence_delta=True,
                        **(
                            kwargs_attribution if kwargs_attribution is not None else {}
                        ),
                    )
                    torch_rel = torch.cat((torch_rel, tmp_rel.detach()), 0)
                    if torch.sum(delta) > 0.1:
                        print(
                            f"Relevance delta for method {method_relevance} is {delta}"
                        )

                    del tmp_rel
            else:
                for (sample, target_idx) in tqdm(
                    zip(batched_sample, batched_idx),
                    total=len(batched_sample),
                    desc="Computing relevance",
                ):
                    tmp_rel = attribute_series_features(
                        rel_method,
                        self.model,
                        torch.Tensor(sample.astype(np.float32)).to(
                            device_interpretability
                        ),
                        target_idx,
                        baselines=baseline,
                        **(
                            kwargs_attribution if kwargs_attribution is not None else {}
                        ),
                    )
                    torch_rel = torch.cat((torch_rel, tmp_rel.detach()), 0)

                    del tmp_rel

        # Compute relevance for methods without a baseline
        elif dict_method_arguments[method_relevance]["require_baseline"] == False:
            rel_method = dict_method_arguments[method_relevance]["captum_method"]
            rel_method = rel_method(self.model)
            if rel_method.has_convergence_delta():
                for (sample, target_idx) in tqdm(
                    zip(batched_sample, batched_idx),
                    total=len(batched_sample),
                    desc="Computing relevance",
                ):
                    tmp_rel, delta = attribute_series_features(
                        rel_method,
                        self.model,
                        torch.Tensor(sample.astype(np.float32)),
                        target_idx,
                        return_convergence_delta=True,
                        **(
                            kwargs_attribution if kwargs_attribution is not None else {}
                        ),
                    )
                    torch_rel = torch.cat((torch_rel, tmp_rel.detach()), 0)

                    if torch.sum(delta) > 0.1:
                        print(
                            f"Relevance delta for method {method_relevance} is {delta}"
                        )
                    del tmp_rel
                    del delta
            else:
                for (sample, target_idx) in tqdm(
                    zip(batched_sample, batched_idx),
                    total=len(batched_sample),
                    desc="Computing relevance",
                ):
                    tmp_rel = attribute_series_features(
                        rel_method,
                        self.model,
                        torch.Tensor(sample.astype(np.float32)),
                        target_idx,
                        **(
                            kwargs_attribution if kwargs_attribution is not None else {}
                        ),
                    )
                    torch_rel = torch.cat((torch_rel, tmp_rel.detach()), 0)

                    del tmp_rel

        else:
            raise ValueError(
                f"{method_relevance} \
                is not available. Relevance method should be one of \
               {dict_method_arguments.keys()}"
            )

        np_rel = torch_rel.cpu().detach().numpy()

        # If channel first == true we convert to [batch, time-step, feature]

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
        path_interp_raw = pj(self.save_results, "interpretability_raw")
        if not os.path.exists(path_interp_raw):
            os.makedirs(path_interp_raw)

        df_relevance.to_csv(
            pj(path_interp_raw, f"intepretability_{method_relevance}.csv")
        )
        np.save(
            pj(path_interp_raw, f"intepretability_{method_relevance}.npy"),
            np_rel,
        )

        if (lstm_network_bool) & (
            dict_method_arguments[method_relevance]["noback_cudnn"]
        ):
            device = (
                torch.device("cuda:0")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            self.model.to(device)

        return df_relevance

    def compute_scores_wrapper(
        self,
        q_features: List[float],
        method_relevance: str,
    ):
        """
        Wrapper to compute metrics at all prescribed q_features
        Parameters
        ----------
        q_features: list
            List of k_feature to compute
        method_relevance: str
            Interpretability method to compute metrics
        neg_relevance: bool
            If true, relevance is computed for negative relevance, else only consider positive relevance
        """
        rel_method = dict_method_arguments[method_relevance]["captum_method"]
        baseline_type = dict_method_arguments[method_relevance]["baseline_type"]

        _, expected_value = self._return_baseline(baseline_type, self.baseline)
        self.expected_value = expected_value

        print(f"Computing metrics using {method_relevance}")
        for topk in [True, False]:
            for k_feature in q_features:
                _ = self.__compute_scores(
                    k_feature,
                    topk,
                    method_relevance,
                )
        resave_score_rel_neg(self.save_results, method_relevance)

    def create_summary(self, method_relevance):
        """
        Function to create the summary of the results
        Parameters
        ----------
        method_relevance: str
            intepretability method of interest
        Returns
        -------
        df_results: pd.DataFrame
            dataframe with the summary of the results

        """
        # Extract csv with results for all k_feature
        path_files = glob.glob(
            os.path.join(
                self.save_results,
                "results_k_feature",
                f"interpretability_{method_relevance}__*",
                "results_interp__*.csv",
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
            "metric_score_random",
        ]
        q_features = [
            float(re.search("[\d.]+$", tmp.split(os.sep)[-2]).group())
            for tmp in path_files
        ]
        q_features = list(set(q_features))
        # quantile = [float(x.split(os.sep)[-2].split("__")[-1]) for x in path_files]
        df_results_top = pd.DataFrame(index=q_features, columns=required_columns)
        df_results_bottom = pd.DataFrame(index=q_features, columns=required_columns)
        # Compute summary across each metric
        for idx, path in enumerate(path_files):
            k_feature = float(re.search("[\d.]+$", path.split(os.sep)[-2]).group())

            if k_feature != 1.0:
                required_columns_all = required_columns + ["metric_normalised"]
            else:
                required_columns_all = required_columns

            mask = re.search("(top|bottom)", path.split(os.sep)[-1]).group()
            df_quantile = pd.read_csv(path, index_col=0)
            df_tmp = df_quantile.loc[:, required_columns_all]
            df_tmp = pd.DataFrame(
                np.where(np.isinf(df_tmp.values), np.nan, df_tmp.values),
                index=df_tmp.index,
                columns=df_tmp.columns,
            )
            df_tmp = df_tmp.dropna(axis=1, how="all")

            df_accuracy = df_quantile.loc[
                :,
                [
                    "initial_classification",
                    "modified_classification",
                    "random_classification",
                ],
            ]
            if mask == "top":
                df_results_top.loc[k_feature, df_accuracy.columns] = (
                    np.sum(df_accuracy, axis=0) / df_accuracy.shape[0]
                )

                df_results_top.loc[k_feature, df_tmp.columns] = np.nanmean(
                    df_tmp.values, axis=0
                )
            elif mask == "bottom":
                df_results_bottom.loc[k_feature, df_accuracy.columns] = (
                    np.sum(df_accuracy, axis=0) / df_accuracy.shape[0]
                )

                df_results_bottom.loc[k_feature, df_tmp.columns] = np.nanmean(
                    df_tmp.values, axis=0
                )

        df_results_top.to_csv(
            os.path.join(
                self.save_results,
                f"summary_relevance_{method_relevance}__top.csv",
            )
        )
        df_results_bottom.loc[:, "metric_score"] = df_results_bottom.loc[
            :, "metric_score"
        ].clip(lower=0)
        df_results_bottom.to_csv(
            os.path.join(
                self.save_results,
                f"summary_relevance_{method_relevance}__bottom.csv",
            )
        )

    def _return_baseline(
        self, type_baseline: str, signal: Union[np.array, torch.Tensor]
    ) -> torch.tensor:
        """Function to return baseline
        Parameters
        ----------
        type_baseline : str
            Type of baseline to be used. One of ["zeros", "random", "mean", "sample"]
        signal : np.array
            Signal to be used to compute baseline.
        Returns
        -------
        Baseline: np.array
        """
        s = signal.copy()
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s)
        self.model.to(device)
        nb_sample = min(signal.shape[0], 50)
        if type_baseline == "zeros":
            # return baseline as zeros
            baseline = torch.zeros(s[:1].shape)
            expected_value = (
                self.model(baseline.type(torch.float32).to(device))
                .detach()
                .cpu()
                .numpy()
            )
        elif type_baseline == "random":
            # return baseline as random values
            baseline = torch.normal(mean=0, std=1, size=s[:1].shape)
            expected_value = (
                self.model(baseline.type(torch.float32).to(device))
                .detach()
                .cpu()
                .numpy()
            )
        elif type_baseline == "mean":
            # return baseline as mean of signal
            baseline = torch.mean(s, dim=0)
            expected_value = (
                self.model(baseline[None].type(torch.float32).to(device))
                .detach()
                .cpu()
                .numpy()
            )
        elif type_baseline == "sample":

            if "sample_baseline.npy" in os.listdir(self.save_results):
                print("Loading baseline from file")
                baseline = np.load(pj(self.save_results, "sample_baseline.npy"))
                baseline = torch.tensor(baseline)
            else:
                print("Extracting baseline samples")
                # return baseline as sample of given size of signal
                idx_random = np.random.permutation(np.arange(s.shape[0]))
                idx_random = idx_random[:nb_sample]
                baseline = s[idx_random]
                np.save(pj(self.save_results, "sample_baseline.npy"), baseline)

            batched_signal = np.array_split(baseline, np.ceil(baseline.shape[0] / 16))
            pred = torch.tensor([], device=device)
            for sample in batched_signal:
                with torch.no_grad():
                    pred = torch.cat(
                        (
                            pred,
                            self.model(sample.type(torch.float32).to(device)),
                        ),
                        0,
                    )
                del sample

            # option 1: mean of the prediction on the whole dataset
            expected_value = pred.mean(axis=0).detach().cpu().numpy()

        baseline = baseline.type(torch.float32)
        if baseline.ndim == 2:
            baseline = baseline[None, :, :]

        return baseline, expected_value

    def __compute_scores(
        self,
        k_feature: float,
        topk: bool,
        method_relevance: str,
    ):
        """
        Compute metrics at a given k_feature
        Parameters
        ----------
        k_feature : float
            Proportion (top or bottom) of features to be used.
        topk: bool
            Whether to use top (True) or bottom (False) k features.
        method_relevance: str
            Interpretability method to compute metrics

        Returns
        -------
        df_results: pd.DataFrame
            Dataframe containing metrics
        """

        string_mask = "top" if topk else "bottom"
        path_save = pj(
            self.save_results,
            "results_k_feature",
            f"interpretability_{method_relevance}__{k_feature:.2f}",
        )
        if not os.path.exists(path_save):
            os.makedirs(path_save)

        df_rel = pd.read_csv(
            pj(
                self.save_results,
                "interpretability_raw",
                f"intepretability_{method_relevance}.csv",
            )
        )
        df_results = pd.DataFrame()
        np_all_modified_signal = np.empty(self.np_signal.shape)
        np_all_randomly_modified_signal = np.empty(self.np_signal.shape)

        # Iterate across all samples to compute metrics
        for idx in tqdm(range(self.np_signal.shape[0])):
            pred_idx = self.pred_idx[idx]
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

            class_name = self.encoder.inverse_transform(pred_idx.reshape(1, -1))
            dict_results = self.__interpretability_metrics(
                idx_sample=idx,
                signal=signal,
                name_sample=name_sample,
                signal_names=signal_names,
                pred_idx=pred_idx,
                interp=interp,
                path_save=path_save,
                k_feature=k_feature,
                topk=topk,
            )

            dict_summary = {
                "class_name": ",".join(class_name.astype(str)[0].tolist()),
                "min_Rx": np.min(interp),
                "max_Rx": np.max(interp),
                "mean_Rx": np.mean(interp),
                "median_Rx": np.median(interp),
                "mean_n_pts_removed": dict_results["n_pts_removed"].mean(),
                "mean_ratio_pts_removed": dict_results["ratio_pts_removed"].mean(),
                "mean_tic": dict_results["tic"].mean(),
                "sample_expected_values": dict_results["sample_expected_values"],
                "sum_rel_neg": dict_results["sum_rel_neg"],
                "sum_rel_pos": dict_results["sum_rel_pos"],
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

        df_results.to_csv(pj(path_save, f"results_interp__{string_mask}.csv"))
        return df_results

    def __interpretability_metrics(
        self,
        idx_sample: int,
        signal: np.ndarray,
        name_sample: str,
        signal_names: List[str],
        pred_idx: np.ndarray,
        interp: np.array,
        path_save: str,
        k_feature: float,
        topk: bool,
    ):
        """
        Function to compute interpretability metrics based on positive relevance
        ----------
        idx_sample: int
            idx of the current sample used to plot only given sample
        signal: np.array
            signal to be interpreted
        name_sample: str
            name of the sample
        signal_names: list
            list of signal names
        pred_idx: numpy array
            pred model encoded
        interp: numpy array
            relevance scores
        path_save: str
            path to save results
        k_feature: float
            determine the  k (top or bottom) features to be used
        topk: bool
            determine wheter top (True) or bottom (False) k features are used

        ----------

        Returns:
        dict_results: dict
            dict with results for the following keys
            frac_pts_rel: fraction of points removed  againg all points
                        with relevance score > 0 | < 0
            tic: Time Information Content
            n_pts_removed: number of points removed
            ratio_pts_removed: ratio of points removed
            sum_rel_removed: sum of rel removed
            sample_expected_values: expected value of the sample
            modified_signal: numpy array
                signal with removed points
            randomly_modified_signal: numpy array
                signal with points removed randomly
        ----------------
        """

        epsilon = 1e-12

        frac_pts_rel = np.zeros(interp.shape[1])
        mean_windows_rel = np.zeros(
            interp.shape[1]
        )  # mean len continuous windows with rel>quantile
        tic = np.zeros(interp.shape[1])

        n_pts_removed = np.zeros(interp.shape[1])
        ratio_pts_removed = np.zeros(interp.shape[1])

        sum_rel_neg = interp[interp < 0].sum()
        sum_rel_pos = interp[interp >= 0].sum()
        # Case interested in positive relevance
        sp_interp_all = interp[interp >= 0]
        if sp_interp_all.shape[0] == 0:
            breakpoint()

        if topk:
            # get top k features
            quantile = 1 - k_feature
            thres_p = np.quantile(sp_interp_all, quantile)
        else:
            # get bottom k features
            quantile = k_feature
            thres_p = np.quantile(sp_interp_all, quantile)

        signal = pd.DataFrame(signal)
        modified_signal = signal.copy()
        randomly_modified_signal = signal.copy()

        for coord in range(signal.shape[1]):

            s = signal.loc[:, coord]
            s_interp = pd.Series(interp[:, coord].copy())

            sp_interp = s_interp[s_interp >= 0]
            if topk:
                sp_tmp = sp_interp[sp_interp >= thres_p]
            else:
                sp_tmp = sp_interp[sp_interp <= thres_p]

            grouped_index = mit.consecutive_groups(sp_tmp.index.values)
            window_len = np.array([len(list(x)) for x in grouped_index])
            mean_windows_rel[coord] = (
                np.nanmean(window_len) if len(window_len) > 0 else 0
            )
            mean_windows_rel[coord] = max(1, mean_windows_rel[coord])

            if sp_tmp.shape[0] == 0:
                frac_pts_rel[coord] = 0
                tic[coord] = 0
                n_pts_removed[coord] = 0
                ratio_pts_removed[coord] = 0
            else:
                # statistical quantities positive
                sp_interp_var = np.var(sp_interp)

                if sp_interp_var < 1e-8:
                    frac_pts_rel[coord] = 0
                else:
                    frac_pts_rel[coord] = sp_tmp.shape[0] / (
                        sp_interp.shape[0] + epsilon
                    )

                if sp_tmp.shape[0] < 1:
                    tic[coord] = 0
                else:
                    tic[coord] = integrate.simpson(sp_tmp) / (
                        integrate.simpson(sp_interp) + epsilon
                    )

                n_pts_removed[coord] = sp_tmp.shape[0]
                ratio_pts_removed[coord] = n_pts_removed[coord] / s.shape[0]

            modified_signal.loc[sp_tmp.index, coord] = np.random.normal(
                scale=1, size=sp_tmp.shape[0]
            )

        drop_prob = n_pts_removed.sum() / np.cumprod(signal.shape)[-1]
        block_size = round(np.nanmean(np.array(mean_windows_rel)))

        randomly_modified_signal = random_block_enforce_proba(
            torch.tensor(randomly_modified_signal.values.T[np.newaxis, :]),
            drop_prob=drop_prob,
            block_size=block_size,
        )

        randomly_modified_signal = randomly_modified_signal.detach().numpy()
        randomly_modified_signal = randomly_modified_signal[0].T
        randomly_modified_signal = pd.DataFrame(randomly_modified_signal)

        if self.plot_signal > 0 and idx_sample % self.plot_signal == 0:
            plot_corrupted_signal(
                signal,
                modified_signal,
                randomly_modified_signal,
                interp,
                path_save,
                signal_names,
                name_sample,
                topk,
            )
            # ------------------------------------------------------

        # sum_rel_removed = interp[interp >= thres_p].sum()

        # We return value in correct order for model that is with [feat, sequence]
        modified_signal = np.transpose(modified_signal.values, (1, 0))
        randomly_modified_signal = np.transpose(randomly_modified_signal.values, (1, 0))

        sample_expected_values = self.expected_value * np.asarray(pred_idx.reshape(-1))
        sample_expected_values = sample_expected_values[sample_expected_values != 0][0]

        dict_results = {
            "frac_pts_rel": frac_pts_rel,
            "tic": tic,
            "n_pts_removed": n_pts_removed,
            "ratio_pts_removed": ratio_pts_removed,
            "sample_expected_values": sample_expected_values,
            "modified_signal": modified_signal,
            "randomly_modified_signal": randomly_modified_signal,
            "sum_rel_neg": sum_rel_neg,
            "sum_rel_pos": sum_rel_pos,
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
            dataframe with the score of the prediction on the initial and
            modified signal as well as diff and metric_score metric
        """
        self.model.to(device)
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
        score1 = self.score_pred
        correct_classification = np.argmax(score1, axis=1) == np.argmax(
            self.target_encoded, axis=1
        )
        score1 = score1 * np.asarray(self.pred_idx)
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
                        self.model(torch.Tensor(sample.astype(np.float32)).to(device)),
                    ),
                    0,
                )
        correct_classification = score2.argmax(
            dim=1
        ).detach().cpu().numpy() == np.argmax(self.target_encoded, axis=1)
        score2 = score2.detach().cpu().numpy() * np.asarray(self.pred_idx)
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
                        self.model(torch.Tensor(sample.astype(np.float32)).to(device)),
                    ),
                    0,
                )
        correct_classification = score3.argmax(
            dim=1
        ).detach().cpu().numpy() == np.argmax(self.target_encoded, axis=1)
        score3 = score3.detach().cpu().numpy() * np.asarray(self.pred_idx)
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

        metric_score = df_score.loc[:, "delta_score1"] / df_score.loc[:, "score1"]
        metric_score_random = (
            df_score.loc[:, "delta_score2"] / df_score.loc[:, "score1"]
        )
        df_score.loc[:, "metric_score"] = metric_score
        df_score.loc[:, "metric_score_random"] = metric_score_random

        del score1
        del score2
        del score3

        return df_score

    def summarise_results(self):
        """
        Function to summarise the results from all interpretability methods
        """
        path_summary = glob.glob(
            os.path.join(self.save_results, "summary_relevance_*.csv")
        )
        results_all = {}
        for path in path_summary:
            df_results = pd.read_csv(path, index_col=0)
            name_file = os.path.split(path)[-1]
            method = re.search("summary_relevance_(.+)\.csv", name_file).group(1)
            results_all[method] = df_results
        df_metrics = compute_summary_metrics(results_all)
        df_metrics.to_csv(
            os.path.join(self.save_results, "metrics_methods.csv")
        )

        return True

    def __init_samples(self):
        """
        Function to initialize the samples, models as well as
        different parameters required for the analysis
        """

        # Find config file used for the simulations
        res = [
            f
            for f in os.listdir(self.model_path)
            if re.search(r"config[a-zA-Z0-9_]+(.yaml|.xml)", f)
        ]
        fc = os.path.abspath(os.path.join(self.model_path, res[0]))

        for file in glob.glob(fc):
            print("file [fc] = ", file)
            config_file = os.path.abspath(file)

        self.config = utils_data.parse_config(config_file)
        self.config["CONFIGURATION"]["data_path"] = pj(
            data_path, self.config["CONFIGURATION"]["data_path"]
        )
        trainer = Trainable(self.config, self.model_path, retrieve_model=True)
        model = trainer.train(self.config["MODEL"]).to(device).eval()
        if self.model_output == "probabilities":
            print("Model output is probability")
            self.model = nn.Sequential(model, nn.Softmax(dim=-1)).eval()
        elif self.model_output == "logits":
            self.model = model
            print("Model output is logits")
        else:
            raise ValueError("Model_output must be either probability or logits")

        df_test = pd.read_csv(
            os.path.join(self.model_path, "results", "results_test.csv")
        )

        if self.names == None:
            names_sample = df_test.noun_id.values.tolist()
            self.names = names_sample
            length_set = len(names_sample)

        else:
            names_sample = list(set(df_test.noun_id.values.tolist() + self.names))
            length_set = len(names_sample)

        data_module_kwargs = {
            "config": self.config,
            "results_path": self.model_path,
            "num_train_epochs": self.config["CONFIGURATION"]["epochs"],
            "train_batch_size": self.config["MODEL"]["batch_size"],
            "val_batch_size": self.config["MODEL"]["batch_size"],
            "num_reader_epochs": 1,
        }
        dataset = DataModule(**data_module_kwargs)
        dataset.setup()

        self.encoder = dataset.encoder
        self.nb_features = dataset.nb_features

        # Fields to be loaded from the dataset
        all_fields = [
            dataset.featureName,
            dataset.targetName,
            "noun_id",
            "signal_names",
        ]

        target_encoded_all = []
        list_signal_names = []
        list_sample_names = []
        with dataset.custom_dataloader(list_names=names_sample) as reader_filtered:
            for idx, sample in enumerate(reader_filtered):
                if idx == 0:
                    np_signal = np.zeros(
                        [length_set, sample.signal.shape[0], sample.signal.shape[1]]
                    )
                np_signal[idx, :, :] = sample.signal.astype(np.float32)
                list_sample_names.append(np.array(sample.noun_id).astype(str).tolist())
                target_encoded_all.append(getattr(sample, dataset.targetName))
                list_signal_names.append(
                    np.array(sample.signal_names).astype(str).tolist()
                )
        # make sure that we remove the lines with zeros created initially in the array
        if np_signal.shape[0] != len(list_sample_names):
            raise ValueError("Mismatch in sample shapes")
        np_signal = np_signal[: len(list_sample_names)]
        name_not_in_list = [x for x in self.names if x not in list_sample_names]
        if len(name_not_in_list) != 0:
            print(f"{name_not_in_list} in name array are not included in the test set")
            self.names = list(set(self.names) - set(name_not_in_list))
        index_to_explain = np.where(np.isin(np.array(list_sample_names), self.names))
        self.np_signal = np_signal[index_to_explain]
        if self.np_signal.shape[0] != len(self.names):
            raise ValueError("Missing samples in signal to analyse")
        # Compute pred on self.np_signal
        batch_size = 32
        batched_sample = np.array_split(
            self.np_signal, np.ceil(self.np_signal.shape[0] / batch_size)
        )
        score_pred = torch.tensor([], device=device, requires_grad=False)
        for sample_signal in batched_sample:
            with torch.no_grad():
                score_pred = torch.cat(
                    (
                        score_pred,
                        self.model(
                            torch.Tensor(sample_signal.astype(np.float32)).to(device)
                        ),
                    ),
                    0,
                )
        # get clean signals
        self.baseline = np_signal.copy()
        self.target_encoded = np.array(target_encoded_all)[
            index_to_explain
        ]  # encoded target
        pred_idx = torch.zeros_like(score_pred)
        idx_max = torch.argmax(score_pred, dim=1)
        pred_idx[torch.arange(score_pred.shape[0]), idx_max] = 1  # predicted index
        self.score_pred = score_pred.detach().cpu().numpy()  # predicted score
        self.pred_idx = pred_idx.detach().cpu().numpy()  # predicted index
        self.list_sample_names = np.array(list_sample_names)[index_to_explain].tolist()
        self.list_signal_names = np.array(list_signal_names)[index_to_explain].tolist()


def compute_summary_metrics(results_all: dict):
    """This functions compute the metrics which summarise each interpretability
        method

    Args:
        results_all (dict): [description]

    Returns:
        (pd.DataFrame): [description]
    """

    name_col = ["AUCSE_top", "F_score"]
    keys_dict = list(results_all.keys())
    methods = list(set([x.split("__")[0] for x in keys_dict]))
    df_metrics = pd.DataFrame(index=methods, columns=name_col)
    for method in methods:
        df_tmp_top = results_all[f"{method}__top"].sort_index(ascending=True)
        df_tmp_bottom = results_all[f"{method}__bottom"].sort_index(ascending=True)
        tsup = df_tmp_top["mean_tic"]
        metric_score_top = df_tmp_top["metric_score"]
        metric_score_bottom = df_tmp_bottom["metric_score"]
        # x = pd.concat([pd.Series(0), tsup]).reset_index(drop=True)
        y = pd.concat([pd.Series(0), metric_score_top]).reset_index(drop=True)

        y_integration_aucse = metric_score_top
        y_integration_aucse = np.append(0, y_integration_aucse)
        y_integration_aucse = np.append(y_integration_aucse, y.iloc[-1])
        x_integration_aucse = df_tmp_top.loc[:, "mean_ratio_pts_removed"]
        x_integration_aucse = np.append(0, x_integration_aucse)
        x_integration_aucse = np.append(x_integration_aucse, 1)

        idx_small = np.argwhere(np.diff(x_integration_aucse) < 10**-2)
        if len(idx_small) > 0:
            print(f"deleting points for {method}")
            # drop points as create instability in integration method
            x_integration_aucse = np.delete(x_integration_aucse, idx_small)
            y_integration_aucse = np.delete(y_integration_aucse, idx_small)
        df_metrics.loc[method, "AUCSE_top"] = integrate.simpson(
            y=y_integration_aucse, x=x_integration_aucse
        )

        y_integration_top = metric_score_top
        y_integration_top = np.append(0, y_integration_top)
        x_integration_top = df_tmp_top.loc[:, "mean_ratio_pts_removed"]
        x_integration_top = np.append(0, x_integration_top)
        integral_top = integrate.simpson(y=y_integration_top, x=x_integration_top)

        y_integration_bottom = metric_score_bottom
        y_integration_bottom = np.append(0, y_integration_bottom)
        x_integration_bottom = df_tmp_bottom.loc[:, "mean_ratio_pts_removed"]
        x_integration_bottom = np.append(0, x_integration_bottom)
        integral_bottom = integrate.simpson(
            y=y_integration_bottom, x=x_integration_bottom
        )

        F_score = (integral_top * (1 - integral_bottom)) / (
            integral_top + (1 - integral_bottom)
        )
        df_metrics.loc[method, "F_score"] = F_score

        if method == "integrated_gradients":
            y_integration = df_tmp_top.loc[:, "metric_score_random"].copy()
            y_integration = np.append(0, y_integration)
            y_integration = np.append(y_integration, y.iloc[-1])
            x_integration = df_tmp_top.loc[:, "mean_ratio_pts_removed"]
            x_integration = np.append(0, x_integration)
            x_integration = np.append(x_integration, 1)
            df_metrics.loc["random", "AUCSE_top"] = integrate.simpson(
                y=y_integration, x=x_integration
            )

    return df_metrics
