
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from .plot_utils import (
    save_figure,
    change_poisson_in_df,
    print_stats,
    sort_and_set_index,
    filter_by_hyperparams,
    LABEL_NAMES_DICT,
)

def get_ratio_df(other_df, baseline_df, target_col, suffix):
    join_df = baseline_df.join(
        other_df, how="inner", lsuffix="_baseline", rsuffix=suffix
    ).reset_index()
    results = []
    for _, row in join_df.iterrows():
        target_col_ratio = row[target_col + suffix] / row["{}_baseline".format(target_col)]
        speedup_ratio = row["runtime_baseline"] / row["runtime{}".format(suffix)]
        results.append(
            [
                row["problem"],
                row["tm_model"],
                row["traffic_seed"],
                row["scale_factor"],
                target_col_ratio,
                speedup_ratio,
            ]
        )

    df = pd.DataFrame(
        columns=[
            "problem",
            "tm_model",
            "traffic_seed",
            "scale_factor",
            "obj_val_ratio",
            "speedup_ratio",
        ],
        data=results,
    )
    return df

def plot_cdfs(
    vals_list,
    labels,
    fname,
    *,
    ax=None,
    title=None,
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

    for vals, label in zip(vals_list, labels):
        vals = sorted([x for x in vals if not np.isnan(x)])
        ax.plot(
            vals,
            np.arange(len(vals)) / len(vals),
            label=LABEL_NAMES_DICT[label] if label in LABEL_NAMES_DICT else label,
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
    if title:
        ax.set_title(title)
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

def per_iter_to_nc_df(per_iter_fname):
    per_iter_df = filter_by_hyperparams(per_iter_fname).drop(
        columns=[
            "num_nodes",
            "num_edges",
            "num_commodities",
            "partition_runtime",
            "size_of_largest_partition",
            "iteration",
            "r1_runtime",
            "r2_runtime",
            "recon_runtime",
            "r3_runtime",
            "kirchoffs_runtime",
        ]
    )
    nc_iterative_df = per_iter_df.groupby(
        [
            "problem",
            "tm_model",
            "traffic_seed",
            "scale_factor",
            "total_demand",
            "algo",
            "clustering_algo",
            "num_partitions",
            "num_paths",
            "edge_disjoint",
            "dist_metric",
        ]
    ).sum()

    return sort_and_set_index(nc_iterative_df)

