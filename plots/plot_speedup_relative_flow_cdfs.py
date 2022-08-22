#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from math import ceil

from lib.plot_utils import (
    save_figure,
    change_poisson_in_df,
    get_ratio_df,
    per_iter_to_nc_df,
    LABEL_NAMES_DICT,
    COLOR_NAMES_DICT,
    LINE_STYLES_DICT,
    CSV_ROOT_DIR,
)

CSV_DIR = os.path.join(CSV_ROOT_DIR, "total-flow")

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
    ax.grid(True)
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


def sort_and_set_index(df, drop=False):
    return change_poisson_in_df(
        df.reset_index(drop=drop).sort_values(
            by=["problem", "tm_model", "scale_factor", "traffic_seed"]
        )
    ).set_index(["problem", "tm_model", "traffic_seed", "scale_factor"])


def get_path_df(csv_dir, query_str=None):
    # Path Formulation DF
    path_df = (
        pd.read_csv(os.path.join(csv_dir, "path-form-total_flow-slice_0_1_2_3_4.csv"))
        .drop(columns=["num_nodes", "num_edges", "num_commodities"])
        .query(PF_PARAMS)
    )
    return sort_and_set_index(path_df)


def get_pop_df(csv_dir, query_str=None):
    # POP DF
    pop_df = pd.read_csv(
        os.path.join(csv_dir, "pop-total_flow-slice_0_1_2_3_4-k_16.csv")
    )
    pop_df = sort_and_set_index(pop_df, drop=True)
    return pop_df.query(
        '(split_fraction == 0.0 and tm_model != "poisson-high-intra") or (split_fraction == 0.75 and tm_model == "poisson-high-intra")'
    )


def get_ncflow_df(csv_dir):
    return per_iter_to_nc_df(
        os.path.join(csv_dir, "ncflow-total_flow-slices_0_1_2_3_4.csv")
    )


def get_fleischer_df(csv_dir, csv_fname, epsilon=0.5):
    fleischer_df = (
        pd.read_csv(os.path.join(csv_dir, csv_fname))
        .query("epsilon == {}".format(epsilon))
        .drop(columns=["epsilon", "tm_attrs"])
    )
    return sort_and_set_index(fleischer_df, drop=True)


def get_smore_df(csv_dir, num_paths=4):
    smore_df = pd.read_csv(os.path.join(csv_dir, "smore.csv")).query(
        "num_paths == {}".format(num_paths)
    )
    return sort_and_set_index(smore_df, drop=True)


def get_cspf_df(csv_dir):
    cspf_df = pd.read_csv(os.path.join(csv_dir, "cspf-total_flow-slice_0_1_2_3_4.csv"))
    return sort_and_set_index(cspf_df, drop=True)


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


def plot_speedup_relative_flow_cdfs(csv_dir):
    query_str = 'problem != "one-wan-bidir.json" and problem != "msft-8075.json"'
    path_df = get_path_df(csv_dir)

    ncflow_df = get_ncflow_df(csv_dir)
    pop_df = get_pop_df(csv_dir)
    cspf_df = get_cspf_df(csv_dir)
    smore_df = get_smore_df(csv_dir)
    fleischer_path_eps_05_df = get_fleischer_df(csv_dir, "fleischer-with-paths.csv")
    fleischer_edge_eps_05_df = get_fleischer_df(csv_dir, "fleischer-edge.csv")

    technique_ratio_dfs = [
        get_ratio_df(technique_df.query(query_str), path_df, "obj_val", suffix)
        for (technique_df, suffix) in [
            (ncflow_df, "_nc"),
            (pop_df, "_pop"),
            (cspf_df, "_cspf"),
            (smore_df, "_smore"),
            (fleischer_path_eps_05_df, "_fleischer_path_eps_05"),
            (fleischer_edge_eps_05_df, "_fleischer_edge_eps_05"),
        ]
    ]

    ratio_df_dict = {
            "cspf" : technique_ratio_dfs[2],
            "smore":  technique_ratio_dfs[3],
            "fp": technique_ratio_dfs[4],
            "fe": technique_ratio_dfs[5],
            "nc": technique_ratio_dfs[0],
            "pop": technique_ratio_dfs[1],
            }

    print_stats(ratio_df_dict["nc"], "NCFlow")
    print_stats(ratio_df_dict["pop"], "POP")

    techniques_to_plot = ["cspf", "smore", "fp", "fe", "nc", "pop"]
    techniques_to_plot_without_cspf = list(techniques_to_plot)
    techniques_to_plot_without_cspf.remove("cspf")

    # fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(15, 3.5))

    # Plot CDFs
    plot_cdfs(
        [
            ratio_df_dict[technique]["speedup_ratio"] for technique in techniques_to_plot_without_cspf
        ],
        techniques_to_plot_without_cspf,
        techniques_to_plot_without_cspf,
        "speedup-cdf",
        # ax=ax2,
        x_log=True,
        x_label=r"Speedup, relative to $\mathrm{PF}_4$ (log scale)",
        arrow_coords=(600.0, 0.2),
        bbta=(0, 0, 1, 2.3),
        figsize=(9, 3.5),
        ncol=len(techniques_to_plot),
        add_ylabel=False,
        show_legend=False,
        save=True,
    )

    plot_cdfs(
        [
            ratio_df_dict[technique]["flow_ratio"] for technique in techniques_to_plot
        ],
        techniques_to_plot,
        techniques_to_plot,
        "total-flow-cdf",
        # ax=ax1,
        x_log=False,
        xlim=(0.2, 1.2),
        x_label=r"Total Flow, relative to $\mathrm{PF}_4$",
        arrow_coords=(1.1, 0.3),
        bbta=(0, 0, 1, 1.3),
        figsize=(9, 4.9),
        ncol=ceil(len(techniques_to_plot) / 2),
        show_legend=True,
        save=True,
    )
    # extra_artists = []
    # legend = ax1.legend(
    #     ncol=len(techniques_to_plot),
    #     loc="upper center",
    #     bbox_to_anchor=(0, 0, 2.2, 1.2),
    #     frameon=False,
    # )
    # extra_artists.append(legend)

    # fig.savefig(
    #     "total-flow-and-speedup-cdfs.pdf",
    #     bbox_inches="tight",
    #     bbox_extra_artists=extra_artists,
    # )


if __name__ == "__main__":
    plot_speedup_relative_flow_cdfs(CSV_DIR)
