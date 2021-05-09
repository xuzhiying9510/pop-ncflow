#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sys

sys.path.append("../..")
from lib.plot_utils import print_stats, sort_and_set_index
from lib.cdf_utils import plot_cdfs, get_ratio_df

PF_PARAMS = 'num_paths == 4 and edge_disjoint == True and dist_metric == "inv-cap"'


def get_ratio_dataframes(curr_dir, query_str=None):
    # Path Formulation DF
    path_form_df = (
        pd.read_csv(curr_dir + "path-form.csv")
        .drop(columns=["num_nodes", "num_edges", "num_commodities"])
        .query(PF_PARAMS)
    )
    path_form_df = sort_and_set_index(path_form_df, drop=True)
    if query_str is not None:
        path_form_df = path_form_df.query(query_str)

    # POP DF
    pop_df = pd.read_csv(curr_dir + "pop.csv")
    pop_df = sort_and_set_index(pop_df, drop=True)
    if query_str is not None:
        pop_df = pop_df.query(query_str)

    def get_pop_dfs(pop_parent_df, suffix):
        pop_random_32_df = pop_parent_df.query(
            'split_method == "random" and num_subproblems == 32'
        )
        pop_random_16_df = pop_parent_df.query(
            'split_method == "random" and num_subproblems == 16'
        )
        pop_random_4_df = pop_parent_df.query(
            'split_method == "random" and num_subproblems == 4'
        )

        pop_means_32_df = pop_parent_df.query(
            'split_method == "means" and num_subproblems == 32'
        )
        pop_means_16_df = pop_parent_df.query(
            'split_method == "means" and num_subproblems == 16'
        )
        pop_means_4_df = pop_parent_df.query(
            'split_method == "means" and num_subproblems == 4'
        )

        return [
            get_ratio_df(df, path_form_df, "obj_val", suffix)
            for df in [
                pop_random_32_df,
                pop_random_16_df,
                pop_random_4_df,
                pop_means_32_df,
                pop_means_16_df,
                pop_means_4_df,
            ]
        ]

    return get_pop_dfs(pop_df, "_pop")


def plot_mcf_cdfs(
    curr_dir,
    title="",
    query_str='problem not in ["Uninett2010.graphml", "Ion.graphml", "Interoute.graphml"]',
):
    ratio_dfs = get_ratio_dataframes(curr_dir, query_str)

    pop_random_32_df = ratio_dfs[0]
    pop_random_16_df = ratio_dfs[1]
    pop_random_4_df = ratio_dfs[2]

    pop_means_32_df = ratio_dfs[4]
    pop_means_16_df = ratio_dfs[4]
    pop_means_4_df = ratio_dfs[5]

    # print_stats(pop_random_32_df, "Random, 32", ["obj_val_ratio", "speedup_ratio"])
    # print_stats(pop_means_32_df, "Power-of-two, 32", ["obj_val_ratio", "speedup_ratio"])

    # print_stats(pop_random_16_df, "Random, 16", ["obj_val_ratio", "speedup_ratio"])
    # print_stats(pop_means_16_df, "Power-of-two, 16", ["obj_val_ratio", "speedup_ratio"])

    # print_stats(pop_random_4_df, "Random, 4", ["obj_val_ratio", "speedup_ratio"])
    # print_stats(pop_means_4_df, "Power-of-two, 4", ["obj_val_ratio", "speedup_ratio"])

    # Plot CDFs
    plot_cdfs(
        [
            pop_random_32_df["speedup_ratio"],
            pop_means_32_df["speedup_ratio"],
            pop_random_16_df["speedup_ratio"],
            pop_means_16_df["speedup_ratio"],
            pop_random_4_df["speedup_ratio"],
            pop_means_4_df["speedup_ratio"],
        ],
        [
            "Random, 32",
            "Power-of-two, 32",
            "Random, 16",
            "Power-of-two, 16",
            "Random, 4",
            "Power-of-two, 4",
        ],
        "speedup-cdf-mcf-{}".format(title),
        x_log=True,
        x_label=r"Speedup, relative to PF4 (log scale)",
        bbta=(0, 0, 1, 1.3),
        figsize=(9, 5.5),
        ncol=3,
        title=title,
    )

    plot_cdfs(
        [
            pop_random_32_df["obj_val_ratio"],
            pop_means_32_df["obj_val_ratio"],
            pop_random_16_df["obj_val_ratio"],
            pop_means_16_df["obj_val_ratio"],
            pop_random_4_df["obj_val_ratio"],
            pop_means_4_df["obj_val_ratio"],
        ],
        [
            "Random, 32",
            "Power-of-two, 32",
            "Random, 16",
            "Power-of-two, 16",
            "Random, 4",
            "Power-of-two, 4",
        ],
        "min-frac-flow-cdf-mcf-{}".format(title),
        x_log=False,
        x_label=r"Min Frac. Flow, relative to PF4",
        bbta=(0, 0, 1, 1.3),
        figsize=(9, 5.5),
        ncol=3,
        title=title,
    )


if __name__ == "__main__":
    plot_mcf_cdfs("./")
