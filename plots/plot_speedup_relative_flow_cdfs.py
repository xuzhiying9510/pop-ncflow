#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from lib.plot_utils import (
    save_figure,
    change_poisson_in_df,
    get_ratio_df,
    per_iter_to_nc_df,
    LABEL_NAMES_DICT,
    COLOR_NAMES_DICT,
    LINE_STYLES_DICT,
)

CSV_DIR = "../benchmarks/csvs/total-flow/"

PF_PARAMS = 'num_paths == 4 and edge_disjoint == True and dist_metric == "inv-cap"'


def join_with_fib_entries(df, fib_entries_df, index_cols):
    return (
        df.set_index(index_cols)
        .join(fib_entries_df)
        .reset_index()
        .set_index(["traffic_seed", "problem", "tm_model"])
    )


def plot_cdfs(
    vals_list,
    labels,
    line_style_keys,
    fname,
    *,
    ax=None,
    x_log=False,
    x_label=None,
    figsize=(6, 12),
    bbta=(0, 0, 1, 1),
    ncol=2,
    xlim=None,
    xticklabels=None,
    add_ylabel=True,
    arrow_coords=None,
    show_legend=True,
    save=True
):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    for vals, label, line_style_key in zip(vals_list, labels, line_style_keys):
        vals = sorted([x for x in vals if not np.isnan(x)])
        ax.plot(
            vals,
            np.arange(len(vals)) / len(vals),
            label=LABEL_NAMES_DICT[label] if label in LABEL_NAMES_DICT else label,
            linestyle=LINE_STYLES_DICT[line_style_key],
            color=COLOR_NAMES_DICT[line_style_key],
        )
    if add_ylabel:
        ax.set_ylabel("Fraction of Cases")
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([0.0, 0.25, 0.50, 0.75, 1.0])
    if x_label:
        ax.set_xlabel(x_label)
    if x_log:
        ax.set_xscale("log")
    if xlim:
        ax.set_xlim(xlim)
    if xticklabels:
        if isinstance(xticklabels, tuple):
            xticks, xlabels = xticklabels[0], xticklabels[-1]
        else:
            xticks, xlabels = xticklabels, xticklabels
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
    extra_artists = []
    if show_legend:
        legend = ax.legend(
            ncol=ncol, loc="upper center", bbox_to_anchor=bbta, frameon=False
        )
        extra_artists.append(legend)

    if arrow_coords:
        bbox_props = {
            "boxstyle": "rarrow,pad=0.45",
            "fc": "white",
            "ec": "black",
            "lw": 2,
        }
        t = ax.text(
            arrow_coords[0],
            arrow_coords[1],
            "Better",
            ha="center",
            va="center",
            color="black",
            bbox=bbox_props,
        )
        extra_artists.append(t)
    if save:
        save_figure(fname, extra_artists=extra_artists)
    # plt.show()


def sort_and_set_index(df, drop=False):
    return change_poisson_in_df(
        df.reset_index(drop=drop).sort_values(
            by=["problem", "tm_model", "scale_factor", "traffic_seed"]
        )
    ).set_index(["problem", "tm_model", "traffic_seed", "scale_factor"])


def get_ratio_dataframes(csv_dir, query_str=None):
    # Path Formulation DF
    path_df = (
        pd.read_csv(os.path.join(csv_dir, "path-form-total_flow-slices_0_1_2_3_4.csv"))
        .drop(columns=["num_nodes", "num_edges", "num_commodities"])
        .query(PF_PARAMS)
    )
    path_df = sort_and_set_index(path_df)
    if query_str is not None:
        path_df = path_df.query(query_str)

    # POP DF
    pop_df = pd.read_csv(os.path.join(csv_dir, "pop-total_flow-slice_0-k_16.csv"))
    pop_df = sort_and_set_index(pop_df, drop=True)
    pop_df = pop_df.query(
        '(split_fraction == 0.0 and tm_model != "poisson-high-intra") or (split_fraction == 0.75 and tm_model == "poisson-high-intra")'
    )

    # NC Iterative DF
    nc_iterative_df = per_iter_to_nc_df(
        os.path.join(csv_dir, "ncflow-total_flow-slices_0_1_2_3_4.csv")
    )
    if query_str is not None:
        nc_iterative_df = nc_iterative_df.query(query_str)

    # Fleischer Path DF
    fleischer_path_eps_05_df = (
        pd.read_csv(os.path.join(csv_dir, "fleischer-with-paths.csv"))
        .query("epsilon == 0.5")
        .drop(columns=["epsilon", "tm_attrs"])
    )
    fleischer_path_eps_05_df = sort_and_set_index(fleischer_path_eps_05_df, drop=True)
    if query_str is not None:
        fleischer_path_eps_05_df = fleischer_path_eps_05_df.query(query_str)

    # Fleischer Edge DF
    fleischer_edge_eps_05_df = (
        pd.read_csv(os.path.join(csv_dir, "fleischer-edge.csv"))
        .query("epsilon == 0.5")
        .drop(columns=["epsilon", "tm_attrs"])
    )
    fleischer_edge_eps_05_df = sort_and_set_index(fleischer_edge_eps_05_df, drop=True)
    if query_str is not None:
        fleischer_edge_eps_05_df = fleischer_edge_eps_05_df.query(query_str)

    # Smore DF
    smore_df = pd.read_csv(os.path.join(csv_dir, "smore.csv"))
    smore_df = sort_and_set_index(smore_df, drop=True)
    if query_str is not None:
        smore_df = smore_df.query(query_str)

    # Ratio DFs
    return (
        get_ratio_df(nc_iterative_df, path_df, "obj_val", "_nc"),
        get_ratio_df(smore_df, path_df, "obj_val", "_smore"),
        get_ratio_df(
            fleischer_path_eps_05_df, path_df, "obj_val", "_fleischer_path_eps_05"
        ),
        get_ratio_df(
            fleischer_edge_eps_05_df, path_df, "obj_val", "_fleischer_edge_eps_05"
        ),
        get_ratio_df(pop_df, path_df, "obj_val", "_pop"),
    )


def print_stats(ratio_df, label):
    # Print stats
    print(
        "Flow ratio {} vs PF:\nmin: {},\nmedian: {},\nmean: {},\nmax: {}".format(
            label,
            np.min(ratio_df["flow_ratio"]),
            np.median(ratio_df["flow_ratio"]),
            np.mean(ratio_df["flow_ratio"]),
            np.max(ratio_df["flow_ratio"]),
        )
    )
    print()

    print(
        "Speedup ratio {} vs PF:\nmin: {},\nmedian: {},\nmean: {},\nmax: {}".format(
            label,
            np.min(ratio_df["speedup_ratio"]),
            np.median(ratio_df["speedup_ratio"]),
            np.mean(ratio_df["speedup_ratio"]),
            np.max(ratio_df["speedup_ratio"]),
        )
    )
    print()


def plot_speedup_relative_flow_cdfs(curr_dir):
    (
        nc_iterative_ratio_df,
        smore_ratio_df,
        fleischer_path_eps_05_ratio_df,
        fleischer_edge_eps_05_ratio_df,
        pop_ratio_df,
    ) = get_ratio_dataframes(
        curr_dir, 'problem != "one-wan-bidir.json" and problem != "msft-8075.json"'
    )

    print_stats(nc_iterative_ratio_df, "NCFlow")
    print_stats(pop_ratio_df, "POP")

    # Plot CDFs
    plot_cdfs(
        [
            nc_iterative_ratio_df["speedup_ratio"],
            smore_ratio_df["speedup_ratio"],
            fleischer_path_eps_05_ratio_df["speedup_ratio"],
            fleischer_edge_eps_05_ratio_df["speedup_ratio"],
            pop_ratio_df["speedup_ratio"],
        ],
        ["nc", "smore", "fp", "fe", "pop"],
        ["nc", "smore", "fe", "fp", "pop"],
        "speedup-cdf",
        x_log=True,
        x_label=r"Speedup, relative to standard TE solver (log scale)",
        arrow_coords=(600.0, 0.2),
        bbta=(0, 0, 1, 2.3),
        figsize=(9, 3.5),
        ncol=4,
        show_legend=False,
    )

    plot_cdfs(
        [
            nc_iterative_ratio_df["flow_ratio"],
            smore_ratio_df["flow_ratio"],
            fleischer_path_eps_05_ratio_df["flow_ratio"],
            fleischer_edge_eps_05_ratio_df["flow_ratio"],
            pop_ratio_df["flow_ratio"],
        ],
        ["nc", "smore", "fe", "fp", "pop"],
        ["nc", "smore", "fe", "fp", "pop"],
        "total-flow-cdf",
        x_log=False,
        xlim=(0.5, 1.2),
        x_label=r"Total Flow, relative to standard TE solver",
        arrow_coords=(1.1, 0.2),
        bbta=(0, 0, 1, 1.35),
        figsize=(9, 4.5),
        ncol=4,
    )
    return (
        nc_iterative_ratio_df,
        smore_ratio_df,
        fleischer_edge_eps_05_ratio_df,
        fleischer_path_eps_05_ratio_df,
        pop_ratio_df,
    )


if __name__ == "__main__":
    plot_speedup_relative_flow_cdfs(CSV_DIR)
