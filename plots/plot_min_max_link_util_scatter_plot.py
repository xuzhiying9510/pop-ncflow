#!/usr/bin/env python

import pandas as pd
import os

from lib.plot_utils import CSV_ROOT_DIR, sort_and_set_index
from lib.scatter_utils import pop_scatter_plot

CSV_DIR = os.path.join(CSV_ROOT_DIR, "min-max-link-util")

if __name__ == "__main__":
    traffic_seed = 1876254433  # "Kdl.graphml", "gravity", 8.0
    num_subproblems = [1, 4, 16, 64]
    pop_df = pd.read_csv(os.path.join(CSV_DIR, "pop-min_max_link_util-slice_0_1.csv"))
    pop_df = sort_and_set_index(pop_df, drop=True)
    results_df = pop_df.query(
        "traffic_seed == {} and num_subproblems in {}".format(
            traffic_seed, num_subproblems
        )
    ).sort_values(by="num_subproblems")
    runtimes = results_df["runtime"]
    obj_vals = results_df["obj_val"]
    labels = ["POP-{}".format(n) if n != 1 else "Exact sol." for n in num_subproblems]
    print(runtimes[0] / runtimes[-1])
    print(obj_vals[0] / obj_vals[-1])

    pop_scatter_plot(
        runtimes,
        obj_vals,
        labels,
        ylabel="Max link\nutilization",
        ann_factor_x=0.85,
        ann_factor_y=1.1,
        arrow_coord_x=1,
        arrow_coord_y=0.3,
        arrow_rotation=45,
        annotate_values=True,
        output_filename="runtime_vs_min_max_link_util_pop_scatter.pdf",
    )
