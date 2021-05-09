#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sys

sys.path.append("../..")
from lib.plot_utils import sort_and_set_index
from lib.cdf_utils import get_ratio_df, plot_cdfs, per_iter_to_nc_df

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

    # NCFlow DF
    nc_iterative_df = per_iter_to_nc_df(curr_dir + "ncflow.csv")
    if query_str is not None:
        nc_iterative_df = nc_iterative_df.query(query_str)

    # POP DF
    pop_df = pd.read_csv(curr_dir + "pop-total_flow-slice_0-splitsweep.csv")
    pop_df = sort_and_set_index(pop_df, drop=True)
    if query_str is not None:
        pop_df = pop_df.query(query_str)

    def get_pop_dfs(pop_parent_df, suffix):
        pop_e0_poisson_df = pop_parent_df.query('split_fraction == 0 and tm_model == "poisson-high-intra"')
        pop_e25_poisson_df = pop_parent_df.query('split_fraction == 0.25 and tm_model == "poisson-high-intra"')
        pop_e50_poisson_df = pop_parent_df.query('split_fraction == 0.5 and tm_model == "poisson-high-intra"')
        pop_e75_poisson_df = pop_parent_df.query('split_fraction == 0.75 and tm_model == "poisson-high-intra"')
        pop_e100_poisson_df = pop_parent_df.query('split_fraction == 1.0 and tm_model == "poisson-high-intra"')
       
 
        pop_e0_gravity_df = pop_parent_df.query('split_fraction == 0 and tm_model == "gravity"')
        pop_e25_gravity_df = pop_parent_df.query('split_fraction == 0.25 and tm_model == "gravity"')
        pop_e50_gravity_df = pop_parent_df.query('split_fraction == 0.5 and tm_model == "gravity"')
        pop_e75_gravity_df = pop_parent_df.query('split_fraction == 0.75 and tm_model == "gravity"')
        pop_e100_gravity_df = pop_parent_df.query('split_fraction == 1.00 and tm_model == "gravity"')
        

        return [
            get_ratio_df(df, path_form_df, "obj_val", suffix)
            for df in [
                pop_e0_poisson_df,
                pop_e25_poisson_df,
                pop_e50_poisson_df,
                pop_e75_poisson_df,
                pop_e100_poisson_df,
                pop_e0_gravity_df,
                pop_e25_gravity_df,
                pop_e50_gravity_df,
                pop_e75_gravity_df,
                pop_e100_gravity_df,
            ]
        ]

    pop_df_list = get_pop_dfs(pop_df, "_pop")

    # Ratio DFs
    nc_ratio_df = get_ratio_df(nc_iterative_df, path_form_df, "obj_val", "_nc")
    return pop_df_list + [nc_ratio_df]


def plot_client_split_sweep_cdfs(
    curr_dir,
    title="",
    query_str='problem not in ["Uninett2010.graphml", "Ion.graphml", "Interoute.graphml"]',
):
    ratio_dfs = get_ratio_dataframes(curr_dir, query_str)
    
    pop_e0_poisson_df = ratio_dfs[0]
    pop_e25_poisson_df = ratio_dfs[1]
    pop_e50_poisson_df = ratio_dfs[2]
    pop_e75_poisson_df = ratio_dfs[3]
    pop_e100_poisson_df = ratio_dfs[4]

    pop_e0_gravity_df = ratio_dfs[5]
    pop_e25_gravity_df = ratio_dfs[6]
    pop_e50_gravity_df = ratio_dfs[7]
    pop_e75_gravity_df = ratio_dfs[8]
    pop_e100_gravity_df = ratio_dfs[9]


    nc_ratio_df = ratio_dfs[-1]

    def print_stats(df_to_print, name_to_print):
        # Print stats
        print(
            "{} Flow ratio vs PF4:\nmin: {},\nmedian: {},\nmean: {},\nmax: {}".format(
                name_to_print,
                np.min(df_to_print["flow_ratio"]),
                np.median(df_to_print["flow_ratio"]),
                np.mean(df_to_print["flow_ratio"]),
                np.max(df_to_print["flow_ratio"]),
            )
        )
        print()
        print(
            "{} Speedup ratio vs PF4:\nmin: {},\nmedian: {},\nmean: {},\nmax: {}".format(
                name_to_print,
                np.min(df_to_print["speedup_ratio"]),
                np.median(df_to_print["speedup_ratio"]),
                np.mean(df_to_print["speedup_ratio"]),
                np.max(df_to_print["speedup_ratio"]),
            )
        )
        print()

    # Plot CDFs
    plot_cdfs(
        [
            pop_e0_poisson_df["speedup_ratio"],
            pop_e50_poisson_df["speedup_ratio"],
            pop_e100_poisson_df["speedup_ratio"],
            pop_e0_gravity_df["speedup_ratio"],
            pop_e100_gravity_df["speedup_ratio"],
        ],
        ["Poisson, 0%", "Poisson, 50%", "Poisson, 100%", "Gravity, 0%", "Gravity, 100%"],
        "speedup-cdf-client_split_sweep",
        x_log=True,
        x_label=r"Speedup, relative to PF4 (log scale)",
        bbta=(0, 0, 1, 1.4),
        figsize=(9, 4.5),
        ncol=3,
        title=title,
    )

    plot_cdfs(
        [
            pop_e0_poisson_df["flow_ratio"],
            pop_e50_poisson_df["flow_ratio"],
            pop_e100_poisson_df["flow_ratio"],
            pop_e0_gravity_df["flow_ratio"],
            pop_e100_gravity_df["flow_ratio"],
        ],
        ["Poisson, 0%", "Poisson, 50%", "Poisson, 100%", "Gravity, 0%", "Gravity, 100%"],
        "total-flow-cdf-client_split_sweep",
        x_log=False,
        x_label=r"Total Flow, relative to PF4",
        bbta=(0, 0, 1, 1.4),
        figsize=(9, 4.5),
        ncol=3,
        title=title,
    )


if __name__ == "__main__":
    plot_client_split_sweep_cdfs("./")
