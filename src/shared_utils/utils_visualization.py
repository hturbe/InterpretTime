import glob
import os
import re
from os.path import join as pj

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

tick_size = 28
label_size = 32

color_s = sns.color_palette("colorblind")
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
    "deepliftshap": {
        "name": "DeepLIFTSHAP",
        "color": color_s[4],
        "linestyle": ":o",
    },
    # "saliency": {"name": "Saliency", "color": color_s[4], "linestyle": "-o"},
    "kernelshap": {
        "name": "KernelSHAP",
        "color": color_s[5],
        "linestyle": "--o",
    },
    "shapley_sampling": {
        "name": "Shapley Sampling",
        "color": color_s[7],
        "linestyle": "-.o",
    },
}


def plot_DeltaS_results(save_results):
    """
    Function to create plot summarising the results of the analysis for the different interpretability methods
    Parameters
    ----------
    save_results: str
        path to the folder where the results are stored

    Returns
    -------
    None
    """
    tick_size = 28
    label_size = 32
    path_summary = glob.glob(os.path.join(save_results, "summary_relevance_*.csv"))

    df_summary_metric = pd.read_csv(
        os.path.join(save_results, "metrics_methods.csv"),
        index_col=0,
    )

    results_all = {}
    for path in path_summary:
        df_results = pd.read_csv(path, index_col=0)
        name_file = os.path.split(path)[-1]

        method = re.search("summary_relevance_(.+)\.csv", name_file).group(1)
        results_all[method] = df_results

    if "lime__top" in results_all.keys():
        del results_all["lime__top"]
    if "lime__bottom" in results_all.keys():
        del results_all["lime__bottom"]
    max_nr = 0
    max_metric_score = 0
    for key, val in results_all.items():
        if max((val["metric_score"])) > max_metric_score:
            max_metric_score = max((val["metric_score"]))
        if max((val["mean_ratio_pts_removed"])) > max_nr:
            max_nr = max((val["mean_ratio_pts_removed"]))
            method_max_nr = key

    methop_top_aucse = df_summary_metric["AUCSE_top"].idxmax()
    methop_top_f = df_summary_metric["F_score"].idxmax()

    fig, axs = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(20, 10),
        sharey=True,
        sharex=True,
        gridspec_kw={"hspace": 0.15, "wspace": 0.15},
    )

    # Defining custom 'xlim' and 'ylim' values.
    custom_xlim = (0, min(round(max_nr * 1.05, 1), 1))
    custom_ylim = (0, min(round(max_metric_score * 1.05, 1), 1))

    loc = plticker.MultipleLocator(
        base=0.2
    )  # this locator puts ticks at regular intervals

    count_ax = 0
    for idx, key in enumerate(sorted(results_all.keys())):
        split_key = key.split("__")
        method = split_key[0]
        mask = split_key[1]
        if mask == "top":
            if method in name_method_dict.keys():
                name_method = name_method_dict[method]["name"]
                color_method = name_method_dict[method]["color"]
                # linestyle_method = name_method_dict[key]["linestyle"]
                linestyle_mask = "-o"
                markersize_s = 7
            else:
                name_method = key

            df_tmp = results_all[key]
            df_tmp = df_tmp.sort_index()
            ax = axs.ravel()[count_ax]
            ax.set_ylim([0, 1])
            tmp = ax.plot(
                df_tmp.loc[:, "mean_ratio_pts_removed"],
                df_tmp.loc[:, "metric_score"],
                "X-",
                # linestyle_mask,
                markersize=markersize_s,
                color=color_method,
                # label=f"{name_method}",
            )
            df_tmp_bottom = results_all[method + "__bottom"]
            df_tmp_bottom = df_tmp_bottom.sort_index()
            tmp = ax.plot(
                df_tmp_bottom.loc[:, "mean_ratio_pts_removed"],
                df_tmp_bottom.loc[:, "metric_score"],
                linestyle_mask,
                markersize=markersize_s,
                color=color_method,
                # label=f"{name_method}",
            )
            tmp = ax.plot(
                results_all[method_max_nr]
                .sort_index()
                .loc[:, "mean_ratio_pts_removed"],
                results_all[method_max_nr].sort_index().loc[:, "metric_score_random"],
                "-^",
                markersize=markersize_s,
                color="black",
                # label="Random",
            )

            ax.fill_between(
                np.insert(df_tmp_bottom.loc[:, "mean_ratio_pts_removed"].values, 0, 0),
                np.insert(df_tmp.loc[:, "metric_score"].values, 0, 0),
                np.insert(df_tmp_bottom.loc[:, "metric_score"].values, 0, 0),
                alpha=0.5,
                color=color_method,
            )
            count_ax += 1
            aucse_tmp = df_summary_metric.loc[method, "AUCSE_top"]
            f1_tmp = df_summary_metric.loc[method, "F_score"]
            if method == methop_top_aucse:
                weight_aucse = "bold"
                color_aucse = "g"
            else:
                weight_aucse = None
                color_aucse = "black"

            if method == methop_top_f:
                weight_f = "bold"
                color_f = "g"
            else:
                weight_f = None
                color_f = "black"

            if (custom_ylim[1] - df_tmp.loc[:, "metric_score"].max()) < 0.2:
                y_loc_1 = 0.45
                y_loc_2 = 0.3
            else:
                y_loc_1 = 0.8
                y_loc_2 = 0.65

            t = ax.text(
                0.97,
                y_loc_1,
                "$AUC\\tilde{S}_{top}$" + f" = {aucse_tmp :.2f}",
                ha="right",
                va="bottom",
                rotation=0,
                size=18,
                weight=weight_aucse,
                color=color_aucse,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color_aucse, lw=2),
                transform=ax.transAxes,
            )
            t = ax.text(
                0.97,
                y_loc_2,
                "$F1\\tilde{S}$" + f" = {f1_tmp :.2f}",
                ha="right",
                va="bottom",
                rotation=0,
                size=18,
                weight=weight_f,
                color=color_f,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color_f, lw=2),
                transform=ax.transAxes,
            )

            ax.grid(True)
            if count_ax in [1, 4]:
                ax.set_ylabel("$\\tilde{S}$", fontsize=label_size)
            if count_ax in [4, 5, 6]:
                ax.set_xlabel("$\\tilde{N}$", fontsize=label_size)

            ax.set_title(f"{name_method}", fontsize=label_size)
            ax.xaxis.set_tick_params(labelsize=tick_size)
            ax.yaxis.set_tick_params(labelsize=tick_size)
            ax.tick_params(axis="x", pad=10)

    fig_path = os.path.join(save_results, "visualization_results")
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    plt.setp(axs, xlim=custom_xlim, ylim=custom_ylim)
    # Setting the values for all axes.
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, "DeltaS_normalised_mean.png"), dpi=200)
    plt.close()


def plot_additional_results(save_results):
    linestyle_mask = "-o"
    markersize_s = 15
    tick_size = 32
    label_size = 37
    path_summary = glob.glob(os.path.join(save_results, "summary_relevance_*.csv"))
    fig_path = os.path.join(save_results, "visualization_results")
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    results_all = {}
    for path in path_summary:
        df_results = pd.read_csv(path, index_col=0)
        name_file = os.path.split(path)[-1]

        method = re.search("summary_relevance_(.+)\.csv", name_file).group(1)
        results_all[method] = df_results

    if "lime__top" in results_all.keys():
        del results_all["lime__top"]
    if "lime__bottom" in results_all.keys():
        del results_all["lime__bottom"]

    results_all = {k: v for k, v in results_all.items() if "top" in k}

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))
    for key in results_all:
        split_key = key.split("__")
        method = split_key[0]
        name_method = name_method_dict[method]["name"]
        color_method = name_method_dict[method]["color"]
        df_tmp = results_all[key]
        df_tmp = df_tmp.sort_index()
        df_accuracy = df_tmp.loc[
            :, ["mean_ratio_pts_removed", "modified_classification"]
        ]
        df_accuracy = df_accuracy.set_index("mean_ratio_pts_removed")
        df_accuracy.loc[0] = df_tmp.loc[1, "initial_classification"]
        df_accuracy = df_accuracy.sort_index()
        ax.plot(
            df_accuracy,
            linestyle_mask,
            color=color_method,
            label=name_method,
            markersize=markersize_s,
            linewidth=5.0,
        )
    ax.xaxis.set_tick_params(labelsize=tick_size)
    ax.yaxis.set_tick_params(labelsize=tick_size)
    ax.set_xlabel("$\\tilde{N}$", fontsize=label_size)
    ax.set_ylabel("Accuracy", fontsize=label_size)
    ax.grid(True)
    # Setting the values for all axes.
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, "accuracy_drop.png"), dpi=200)
    plt.close()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))
    for key in results_all:
        split_key = key.split("__")
        method = split_key[0]
        name_method = name_method_dict[method]["name"]
        color_method = name_method_dict[method]["color"]
        df_tmp = results_all[key]
        df_tmp = df_tmp.sort_index()
        np_tic = df_tmp["mean_tic"].values
        np_tic = np.insert(np_tic, 0, 0)
        np_metric = df_tmp["metric_normalised"].values
        np_metric = np.insert(np_metric, 0, 0)

        ax.plot(
            df_tmp.loc[:, "mean_tic"].iloc[:-1],
            df_tmp.loc[:, "metric_normalised"].iloc[:-1],
            linestyle_mask,
            color=color_method,
            label=name_method,
            markersize=markersize_s,
            linewidth=5.0,
        )
    ax.xaxis.set_tick_params(labelsize=tick_size)
    ax.yaxis.set_tick_params(labelsize=tick_size)

    ax.plot(
        np.linspace(start=0.0, stop=0.95, num=20),
        np.linspace(start=0.0, stop=0.95, num=20),
        color="black",
        linestyle="--",
        label="Theoretical estimation",
        linewidth=4.0,
    )
    ax.set_ylabel("$ \\tilde{S}_A$", fontsize=label_size)
    ax.set_xlabel("$TIC$", fontsize=label_size)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, "rel_attribution.png"), dpi=200)
    plt.close()


def plot_corrupted_signal(
    signal,
    modified_signal,
    randomly_modified_signal,
    interp,
    path_save,
    signal_names,
    name_sample,
    topk,
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

    string_mask = "top" if topk else "bottom"
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

    fig_path = os.path.join(path_save, f"figures__{string_mask}")
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, str(name_sample) + ".png"), dpi=200)
    plt.close()
