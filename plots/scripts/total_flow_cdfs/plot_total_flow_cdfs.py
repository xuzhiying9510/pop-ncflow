#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sys

sys.path.append("..")

from cdf_utils import get_ratio_df, per_iter_to_nc_df, plot_cdfs

sys.path.append("../..")
from plot_utils import (
    save_figure,
    change_poisson_in_df,
    print_stats,
    sort_and_set_index,
    PF_PARAMS
)


def join_with_fib_entries(df, fib_entries_df, index_cols):
    return (
        df.set_index(index_cols)
        .join(fib_entries_df)
        .reset_index()
        .set_index(["traffic_seed", "problem", "tm_model"])
    )


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

    # NCFlow DF
    nc_iterative_df = per_iter_to_nc_df(curr_dir + "ncflow.csv")
    if query_str is not None:
        nc_iterative_df = nc_iterative_df.query(query_str)

    # POP DF
    pop_df = pd.read_csv(curr_dir + "pop.csv")
    pop_df = sort_and_set_index(pop_df, drop=True)
    if query_str is not None:
        pop_df = pop_df.query(query_str)

    # POP Entity Splitting DF
    pop_entity_splitting_df = pd.read_csv(curr_dir + "pop_entitysplitting.csv")
    pop_entity_splitting_df = sort_and_set_index(pop_entity_splitting_df, drop=True)
    if query_str is not None:
        pop_entity_splitting_df = pop_entity_splitting_df.query(query_str)

    def get_pop_dfs(pop_parent_df, suffix):
        pop_tailored_16_df = pop_parent_df.query(
            'split_method == "tailored" and num_subproblems == 16'
        )
        pop_tailored_4_df = pop_parent_df.query(
            'split_method == "tailored" and num_subproblems == 4'
        )

        pop_random_16_df = pop_parent_df.query(
            'split_method == "random" and num_subproblems == 16'
        )
        pop_random_4_df = pop_parent_df.query(
            'split_method == "random" and num_subproblems == 4'
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
                pop_tailored_16_df,
                pop_tailored_4_df,
                pop_random_16_df,
                pop_random_4_df,
                pop_means_16_df,
                pop_means_4_df,
            ]
        ]

    pop_df_list = get_pop_dfs(pop_df, "_pop")
    pop_entity_splitting_df_list = get_pop_dfs(
        pop_entity_splitting_df, "_pop_entity_splitting"
    )

    # Ratio DFs
    nc_ratio_df = get_ratio_df(nc_iterative_df, path_form_df, "obj_val", "_nc")
    return pop_df_list + pop_entity_splitting_df_list + [nc_ratio_df]


def plot_speedup_relative_flow_cdfs(
    curr_dir,
    title="",
    query_str='problem not in ["Uninett2010.graphml", "Ion.graphml", "Interoute.graphml"]',
):
    ratio_dfs = get_ratio_dataframes(curr_dir, query_str)
    pop_tailored_16_df = ratio_dfs[0]
    # pop_tailored_4_df = ratio_dfs[1]

    pop_random_16_df = ratio_dfs[2]
    # pop_random_4_df = ratio_dfs[3]

    # pop_means_16_df = ratio_dfs[4]
    # pop_means_4_df = ratio_dfs[5]

    pop_entity_splitting_tailored_16_df = ratio_dfs[6]
    # pop_entity_splitting_tailored_4_df = ratio_dfs[7]

    pop_entity_splitting_random_16_df = ratio_dfs[8]
    # pop_entity_splitting_random_4_df = ratio_dfs[9]

    pop_entity_splitting_means_16_df = ratio_dfs[10]
    # pop_entity_splitting_means_4_df = ratio_dfs[11]

    nc_ratio_df = ratio_dfs[-1]

    print_stats(nc_ratio_df, "NCFlow", ["obj_val_ratio", "speedup_ratio"])
    print_stats(pop_random_16_df, "Random 16", ["obj_val_ratio", "speedup_ratio"])
    print_stats(pop_tailored_16_df, "Tailored 16", ["obj_val_ratio", "speedup_ratio"])
    # print_stats(pop_entity_splitting_random_16_df, "Random Entity Splitting 16", ["obj_val_ratio", "speedup_ratio"])
    # print_stats(pop_entity_splitting_tailored_16_df, "Tailored Entity Splitting 16", ["obj_val_ratio", "speedup_ratio"])

    # Plot CDFs
    plot_cdfs(
        [
            nc_ratio_df["speedup_ratio"],
            pop_random_16_df["speedup_ratio"],
            pop_tailored_16_df["speedup_ratio"],
            # pop_entity_splitting_random_16_df["speedup_ratio"],
            # pop_entity_splitting_tailored_16_df["speedup_ratio"],
        ],
        ["NCFlow", "Random, 16", "Tailored, 16"],
        # ["Random, Entity Splitting, 16", "Tailored, Entity Splitting, 16"],
        "speedup-cdf-{}".format(title),
        x_log=True,
        x_label=r"Speedup, relative to PF4 (log scale)",
        bbta=(0, 0, 1, 1.4),
        figsize=(9, 4.5),
        ncol=4,
        title=title,
    )

    plot_cdfs(
        [
            nc_ratio_df["obj_val_ratio"],
            pop_random_16_df["obj_val_ratio"],
            pop_tailored_16_df["obj_val_ratio"],
            # pop_entity_splitting_random_16_df["obj_val_ratio"],
            # pop_entity_splitting_tailored_16_df["obj_val_ratio"],
        ],
        ["NCFlow", "Random, 16", "Tailored, 16"],
        # ["Random, Entity Splitting, 16", "Tailored, Entity Splitting, 16"],
        "total-flow-cdf-{}".format(title),
        x_log=False,
        x_label=r"Total Flow, relative to PF4",
        bbta=(0, 0, 1, 1.4),
        figsize=(9, 4.5),
        ncol=4,
        title=title,
    )


if __name__ == "__main__":
    plot_speedup_relative_flow_cdfs("./")
