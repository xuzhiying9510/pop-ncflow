#!/usr/bin/env python


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import re
import os

from lib.plot_utils import CSV_ROOT_DIR, save_figure, moving_average
from glob import iglob

CSV_DIR = os.path.join(CSV_ROOT_DIR, "private-wan-csvs")

MA_WINDOW = 5
MARK_EVERY = 150
MARKER_SIZE = 10
LINE_WIDTH = 2.0

TIME_WINDOW = 5

TIME_COLS = ["tm_date", "tm_time"]
DATA_COLS = ["nc_time", "pop_time", "nc_flow", "pop_flow"]
COLS = TIME_COLS + DATA_COLS


def add_timestamp_col(df):
    df["timestamp"] = pd.to_datetime(
        df["tm_date"] + " " + df["tm_time"].map(lambda x: x[:-4]),
        format="%Y%m%d %H%M%S",
    )
    df["timedelta"] = np.arange(0, len(df) * TIME_WINDOW, TIME_WINDOW) / (24 * 60)
    return df.drop(columns=TIME_COLS)


def add_split_threshold_suffix(df, split_threshold):
    return df.rename(
        {col: "{}_{}".format(col, split_threshold) for col in DATA_COLS},
        axis=1,
    )


def plot(ready_to_plot_df):
    fig, [ax0, ax1] = plt.subplots(nrows=2, ncols=1, figsize=(8.38, 6.5), sharex=True)
    ax0.plot(
        ready_to_plot_df["timedelta"][MA_WINDOW - 1 :],
        moving_average(ready_to_plot_df["nc_flow_0.0"], n=MA_WINDOW),
        # marker = 's',
        # markersize = MARKER_SIZE,
        # markevery = MARK_EVERY,
        linestyle="--",
        linewidth=LINE_WIDTH,
        label="NCFlow",
    )
    ax0.plot(
        ready_to_plot_df["timedelta"][MA_WINDOW - 1 :],
        moving_average(ready_to_plot_df["pop_flow_0.0"], n=MA_WINDOW),
        # marker = 'o',
        # markersize = MARKER_SIZE,
        # markevery = MARK_EVERY,
        linestyle=":",
        linewidth=LINE_WIDTH,
        label="POP",
    )
    # ax0.plot(
    #     ready_to_plot_df["timedelta"][MA_WINDOW - 1 :],
    #     moving_average(ready_to_plot_df["pop_flow_0.25"], n=MA_WINDOW),
    #     # marker = '^',
    #     # markersize = MARKER_SIZE,
    #     # markevery = MARK_EVERY,
    #     linestyle="-",
    #     linewidth=LINE_WIDTH,
    #     label="POP, with CS",
    # )
    ax0.set_ylabel("Total flow rel. to\n original problem")
    ax0.set_ylim((0.8297367136563809, 1.0080873810319542))
    ax0.set_yticks([0.85, 0.9, 0.95, 1.0])
    ax0.grid(True)


    ax1.plot(
        ready_to_plot_df["timedelta"][MA_WINDOW - 1 :],
        moving_average(1 / ready_to_plot_df["nc_time_0.0"], n=MA_WINDOW),
        # marker = 's',
        # markersize = MARKER_SIZE,
        # markevery = MARK_EVERY,
        linestyle="--",
        linewidth=LINE_WIDTH,
        label="NCFlow",
    )
    ax1.plot(
        ready_to_plot_df["timedelta"][MA_WINDOW - 1 :],
        moving_average(1 / ready_to_plot_df["pop_time_0.0"], n=MA_WINDOW),
        # marker = 'o',
        # markersize = MARKER_SIZE,
        # markevery = MARK_EVERY,
        linestyle=":",
        linewidth=LINE_WIDTH,
        label="POP, +0x",
    )
    # ax1.plot(
    #     ready_to_plot_df["timedelta"][MA_WINDOW - 1 :],
    #     moving_average(1 / ready_to_plot_df["pop_time_0.25"], n=MA_WINDOW),
    #     # marker = '^',
    #     # markersize = MARKER_SIZE,
    #     # markevery = MARK_EVERY,
    #     linestyle="-",
    #     linewidth=LINE_WIDTH,
    #     label="POP, +0.25x",
    # )
    ax1.set_xlabel("Time (days)")
    ax1.set_ylabel("Speedup rel. to\n original problem")
    ax1.set_ylim((5.804983213663454, 21.029909776264812))
    ax1.set_yticks([10, 15, 20])
    ax1.grid(True)

    legend = ax0.legend(
        frameon=False, ncol=3, loc="center", bbox_to_anchor=(0, 0, 1.0, 2.3)
    )
    save_figure("private-wan", extra_artists=(legend,))


if __name__ == "__main__":
    fnames = list(iglob(CSV_DIR + "/*.csv"))
    dfs = [
        add_split_threshold_suffix(df, split_threshold)
        for split_threshold, df in [
            [
                float(re.search("split_threshold=(.*)-sanitized", fname).group(1)),
                pd.read_csv(
                    fname, dtype={"tm_date": str, "tm_time": str}, usecols=COLS
                ),
            ]
            for fname in fnames
        ]
    ]

    join_df = dfs[0]
    index_cols = ["tm_date", "tm_time"]
    for other_df in dfs[1:]:
        join_df = (
            join_df.set_index(index_cols)
            .join(other_df.set_index(index_cols))
            .reset_index()
        )

    ready_to_plot_df = add_timestamp_col(join_df)

    print("NCFlow, median flow ratio:", ready_to_plot_df["nc_flow_0.0"].median())
    print("POP, no CS, median flow ratio:", ready_to_plot_df["pop_flow_0.0"].median())
    print("POP, 25% CS, median flow ratio:", ready_to_plot_df["pop_flow_0.25"].median())
    print("NCFlow, speedup", 1 / ready_to_plot_df["nc_time_0.0"].median())
    print("POP, no CS, speedup", 1 / ready_to_plot_df["pop_time_0.0"].median())
    print("POP, 25% CS, speedup", 1 / ready_to_plot_df["pop_time_0.25"].median())

    plot(ready_to_plot_df)
