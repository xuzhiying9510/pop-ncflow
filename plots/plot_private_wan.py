#!/usr/bin/env python


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from glob import iglob
import re

from lib.plot_utils import *
from lib.cdf_utils import *


TIME_WINDOW = 5
fnames = list(iglob("private_wan_csvs/*.csv"))
time_cols = ["tm_date", "tm_time"]
data_cols = ["nc_time", "pop_time", "nc_flow", "pop_flow"]
cols = time_cols + data_cols


def add_timestamp_col(df):
    df["timestamp"] = pd.to_datetime(
        df["tm_date"] + " " + df["tm_time"].map(lambda x: x[:-4]),
        format="%Y%m%d %H%M%S",
    )
    df["timedelta"] = np.arange(0, len(df) * TIME_WINDOW, TIME_WINDOW) / (24 * 60)
    return df.drop(columns=time_cols)


def add_split_threshold_suffix(df, split_threshold):
    return df.rename(
        {col: "{}_{}".format(col, split_threshold) for col in data_cols},
        axis=1,
    )


dfs = [
    add_split_threshold_suffix(df, split_threshold)
    for split_threshold, df in [
        [
            float(re.search("split_threshold=(.*)-sanitized", fname).group(1)),
            pd.read_csv(fname, dtype={"tm_date": str, "tm_time": str}, usecols=cols),
        ]
        for fname in fnames
    ]
]

join_df = dfs[0]
index_cols = ["tm_date", "tm_time"]
for other_df in dfs[1:]:
    join_df = (
        join_df.set_index(index_cols).join(other_df.set_index(index_cols)).reset_index()
    )

ready_to_plot_df = add_timestamp_col(join_df)


print(ready_to_plot_df["pop_flow_0.0"].median())
print(ready_to_plot_df["pop_flow_0.25"].median())
print(1 / ready_to_plot_df["pop_time_0.0"].median())
print(1 / ready_to_plot_df["pop_time_0.25"].median())


MA_WINDOW = 5
MARK_EVERY = 150
MARKER_SIZE = 10
LINE_WIDTH = 2.0


def moving_average(a, n=MA_WINDOW):
    a = np.array(a)
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


fig, [ax0, ax1] = plt.subplots(nrows=2, ncols=1, figsize=(8.38, 6.5), sharex=True)
ax0.plot(
    ready_to_plot_df["timedelta"][MA_WINDOW - 1 :],
    moving_average(ready_to_plot_df["nc_flow_0.0"]),
    # marker = 's',
    # markersize = MARKER_SIZE,
    # markevery = MARK_EVERY,
    linestyle="--",
    linewidth=LINE_WIDTH,
    label="NCFlow",
)
ax0.plot(
    ready_to_plot_df["timedelta"][MA_WINDOW - 1 :],
    moving_average(ready_to_plot_df["pop_flow_0.0"]),
    # marker = 'o',
    # markersize = MARKER_SIZE,
    # markevery = MARK_EVERY,
    linestyle=":",
    linewidth=LINE_WIDTH,
    label="POP",
)
ax0.plot(
    ready_to_plot_df["timedelta"][MA_WINDOW - 1 :],
    moving_average(ready_to_plot_df["pop_flow_0.25"]),
    # marker = '^',
    # markersize = MARKER_SIZE,
    # markevery = MARK_EVERY,
    linestyle="-",
    linewidth=LINE_WIDTH,
    label="POP, with CS",
)
ax0.set_ylabel("Total flow rel. to\n original problem")
ax0.grid(True)

ax1.plot(
    ready_to_plot_df["timedelta"][MA_WINDOW - 1 :],
    moving_average(1 / ready_to_plot_df["nc_time_0.0"]),
    # marker = 's',
    # markersize = MARKER_SIZE,
    # markevery = MARK_EVERY,
    linestyle="--",
    linewidth=LINE_WIDTH,
    label="NCFlow",
)
ax1.plot(
    ready_to_plot_df["timedelta"][MA_WINDOW - 1 :],
    moving_average(1 / ready_to_plot_df["pop_time_0.0"]),
    # marker = 'o',
    # markersize = MARKER_SIZE,
    # markevery = MARK_EVERY,
    linestyle=":",
    linewidth=LINE_WIDTH,
    label="POP, +0x",
)
ax1.plot(
    ready_to_plot_df["timedelta"][MA_WINDOW - 1 :],
    moving_average(1 / ready_to_plot_df["pop_time_0.25"]),
    # marker = '^',
    # markersize = MARKER_SIZE,
    # markevery = MARK_EVERY,
    linestyle="-",
    linewidth=LINE_WIDTH,
    label="POP, +0.25x",
)
ax1.set_xlabel("Time (days)")
ax1.set_ylabel("Speedup rel. to\n original problem")
ax1.grid(True)

legend = ax0.legend(
    frameon=False, ncol=3, loc="center", bbox_to_anchor=(0, 0, 1.0, 2.3)
)
save_figure("private-wan", extra_artists=(legend,))
