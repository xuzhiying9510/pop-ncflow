#! /usr/bin/env python

from matplotlib.lines import Line2D
from datetime import datetime, timedelta

import os
import pandas as pd
import matplotlib.pyplot as plt

# import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
import numpy as np
import seaborn as sns


from lib.plot_utils import (
    save_figure,
    LABEL_NAMES_DICT,
    COLOR_NAMES_DICT,
    LINE_STYLES_DICT,
    CSV_ROOT_DIR,
)

plt.rcParams["lines.markersize"] = 14
plt.rcParams["lines.linewidth"] = 3.5
plt.rcParams["legend.handlelength"] = 1.75
plt.rcParams["font.size"] = 19

XLIM_OFFSET = 4
TIME_INTERVAL = 5  # mins
DEMAND_PER_TM = [
    7707397.5,
    15894840.0,
    13724729.0,
    12128114.0,
    8088365.5,
    7173291.5,
    9327278.0,
    12659548.0,
    12846447.0,
    9571880.0,
    9931847.0,
    4002021.75,
    6985683.0,
    4247660.0,
    5741112.5,
    6681249.0,
    8563079.0,
    10961703.0,
    9792799.0,
    12575751.0,
    14174818.0,
    13916833.0,
    13169774.0,
    12087695.0,
    10789603.0,
]


def get_xticks_time(tm_sequence):
    start_time = datetime(
        year=2020, month=4, day=10, hour=12, minute=0, second=0, microsecond=0
    )
    return [start_time + timedelta(minutes=i * 5) for i in tm_sequence]


def get_dataframes(csv_dir):
    pfws_df = pd.read_csv(
        os.path.join(csv_dir, "demand-tracking-path_formulation_warm_start.csv")
    )
    pfws_df["orig_total_demand"] = pd.Series(DEMAND_PER_TM)

    ncflow_df = pd.read_csv(os.path.join(csv_dir, "demand-tracking-ncflow.csv"))
    ncflow_df["orig_total_demand"] = pd.Series(DEMAND_PER_TM)

    oracle_df = None
    oracle_csv_path = os.path.join(
        csv_dir, "demand-tracking-path_formulation-oracle.csv"
    )
    if os.path.exists(oracle_csv_path):
        oracle_df = pd.read_csv(oracle_csv_path)
        oracle_df["orig_total_demand"] = pd.Series(DEMAND_PER_TM)

    pf_df = None
    pf_csv_path = os.path.join(csv_dir, "demand-tracking-path_formulation.csv")
    if os.path.exists(pf_csv_path):
        pf_df = pd.read_csv(pf_csv_path)
        pf_df["orig_total_demand"] = pd.Series(DEMAND_PER_TM)

    return ncflow_df, pf_df, pfws_df, oracle_df


def plot_demand_tracking_flow_volume(csv_dir, figsize=(7, 3.5), ax=None):
    ncflow_df, _, _, _ = get_dataframes(csv_dir)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    # x_ticks_time = get_xticks_time(nci_df['tm_number'])
    # print(x_ticks_time)
    ax.bar(
        ncflow_df["tm_number"] * TIME_INTERVAL,
        # nci_df['orig_total_demand'] / nci_df['orig_total_demand'].min(),
        ncflow_df["orig_total_demand"],
        width=0.8 * TIME_INTERVAL,
        label=LABEL_NAMES_DICT["nc"],
    )
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    # plt.xticks(rotation=15)

    ax.set_xlabel("Time (mins)")
    ax.set_ylabel("Requested\nDemand")
    ax.set_xlim(-8.2, 128.2)
    ax.set_xticks(np.arange(0, 140, 20))
    ax.set_yticks([0.5e7, 1e7, 1.5e7])
    # ax.set_xlabel('TM Number')
    # ax.set_ylabel('Normalized\nRequested Demand')
    # ax.set_xlim(-1, 24.75)
    # ax.set_yticks([0, 1, 2, 3, 4])

    # Change offset text to obfuscate actual flow volume
    # Turn off the offset text that's calculated automatically
    # ax.yaxis.offsetText.set_visible(False)
    # # Add in a text box at the top of the y axis
    # offset_text = r'$10^{m}$'
    # ax.text(0.0, 1.0, offset_text, transform=ax.transAxes,
    #         horizontalalignment='left',
    #         verticalalignment='bottom')

    sns.despine()
    save_figure("demand-tracking-flow-volume")


def plot_demand_tracking(
    csv_dir,
    num_points=25,
    title=None,
    residual_factor=0.0,
    include_demand=False,
    col_to_plot="satisfied_demand",
    ylabel="Satisfied Demand",
    figsize=(9, 5),
):
    def compute_cumulative_demand_never_satisfied(df):
        demand_never_satisfied = list(
            df["demand_not_satisfied"] * (1 - residual_factor)
        )
        # At the end of the sequence, whatever unsatisfied demand is left over
        # becomes never satisfied
        demand_never_satisfied.append(
            df.iloc[-1]["demand_not_satisfied"] * residual_factor
        )
        return pd.Series(demand_never_satisfied).cumsum()

    def add_stats_columns_to_df(df):
        df["demand_not_satisfied"] = df["total_demand"] - df["satisfied_demand"]
        df["cum_demand_not_satisfied"] = df["demand_not_satisfied"].cumsum()
        df["cum_demand_satisfied"] = df["satisfied_demand"].cumsum()
        cum_demand_never_satisfied = compute_cumulative_demand_never_satisfied(df)
        # cum_demand_never_satisfied has 25 rows, so we append the 25th row, then add it
        # as a new column
        df = df.append(
            {
                "tm_number": len(df),
                "total_demand": 0,
                "satisfied_demand": 0,
                "orig_total_demand": 0,
                "demand_not_satisfied": 0,
                "cum_demand_not_satisfied": df.iloc[-1]["cum_demand_not_satisfied"],
                "cum_demand_satisfied": df.iloc[-1]["cum_demand_not_satisfied"],
            },
            ignore_index=True,
        )
        df["cum_demand_never_satisfied"] = cum_demand_never_satisfied
        return df

    _, ax = plt.subplots(figsize=figsize)
    ncflow_df, pf_df, pfws_df, oracle_df = get_dataframes(csv_dir)

    pf_df = add_stats_columns_to_df(pf_df)
    pfws_df = add_stats_columns_to_df(pfws_df)
    ncflow_df = add_stats_columns_to_df(ncflow_df)
    oracle_df = add_stats_columns_to_df(oracle_df)
    print("Satisfied Demand Compared to Oracle:")
    print(
        (oracle_df["satisfied_demand"] - ncflow_df["satisfied_demand"])
        / oracle_df["satisfied_demand"]
    )

    if col_to_plot in ["cum_demand_satisfied", "satisfied_demand"]:
        pf_df = pf_df[:25]
        pfws_df = pfws_df[:25]
        ncflow_df = ncflow_df[:25]
        oracle_df = oracle_df[:25]

    handles, labels = [], []
    if "cum_" in col_to_plot:
        denom = sum(DEMAND_PER_TM) / 100.0
    else:
        denom = np.array(DEMAND_PER_TM) / 100.0
        # denom = 1.0
    if include_demand:
        if "cum_" in col_to_plot:
            y_vals = np.cumsum(np.array(DEMAND_PER_TM))
        else:
            y_vals = np.array(DEMAND_PER_TM)
        handles += ax.plot(
            np.arange(len(DEMAND_PER_TM)),
            y_vals / denom,
            marker="*",
            fillstyle="none",
            color="black",
        )
        labels.append("Requested Demand")

    handles += ax.plot(
        (oracle_df["tm_number"] * TIME_INTERVAL)[:num_points],
        (oracle_df[col_to_plot] / denom)[:num_points],
        marker="o",
        # fillstyle='none',
        # linestyle=LINE_STYLES_DICT['pf-oracle'],
        color=COLOR_NAMES_DICT["pf-oracle"],
    )
    # labels.append(LABEL_NAMES_DICT['pf-oracle'])
    labels.append("Instantaneous TE Solver (No lag)")

    handles += ax.plot(
        ncflow_df["tm_number"] * TIME_INTERVAL,
        ncflow_df[col_to_plot] / denom,
        marker="o",
        fillstyle="none",
        linestyle=LINE_STYLES_DICT["nc"],
        color=COLOR_NAMES_DICT["nc"],
    )
    labels.append(LABEL_NAMES_DICT["nc"])

    def plot_multiple_markers(df, label):
        new_sols = list(np.argwhere(df["new_solution"] == 1.0).flatten())
        if col_to_plot not in ["cum_demand_satisfied", "satisfied_demand"]:
            new_sols.append(len(DEMAND_PER_TM))

        old_sols = list(np.argwhere(df["new_solution"] == 0.0).flatten())
        line = ax.plot(
            df["tm_number"] * TIME_INTERVAL,
            df[col_to_plot] / denom,
            marker="o",
            fillstyle="none",
            markevery=new_sols,
            linestyle=LINE_STYLES_DICT[label],
            color=COLOR_NAMES_DICT[label],
        )[0]
        color = line.get_color()
        handles.append(line)
        labels.append(LABEL_NAMES_DICT[label])
        ax.plot(
            df["tm_number"] * TIME_INTERVAL,
            df[col_to_plot] / denom,
            marker="x",
            fillstyle="none",
            markevery=old_sols,
            linestyle=LINE_STYLES_DICT[label],
            color=color,
        )

    # if 'new_solution' in pf_df.columns:
    #     plot_multiple_markers(pf_df, 'pf')
    # else:
    #     handles += ax.plot(pf_df['tm_number'] * TIME_INTERVAL,
    #                        pf_df[col_to_plot] / denom,
    #                        marker='o',
    #                        fillstyle='none',
    #                        linestyle=LINE_STYLES_DICT['pf'],
    #                        color=COLOR_NAMES_DICT['pf'])
    #     labels.append(LABEL_NAMES_DICT['pf'])

    # if 'new_solution' in pfws_df.columns:
    #     plot_multiple_markers(pfws_df, 'pfws')
    # else:

    handles += ax.plot(
        (pfws_df["tm_number"] * TIME_INTERVAL)[:num_points],
        (pfws_df[col_to_plot] / denom)[:num_points],
        marker="o",
        # fillstyle='none',
        # linestyle=LINE_STYLES_DICT['pfws'],
        color=COLOR_NAMES_DICT["pfws"],
    )
    labels.append("Standard TE Solver (With lag)")

    new_handles = []
    for handle, label in zip(handles, labels):
        new_handle = Line2D(handle.get_xdata(), handle.get_ydata())
        new_handle.update_from(handle)
        if label != "Requested Demand":
            new_handle.set_marker("None")
        new_handles.append(new_handle)
    handles = new_handles

    # handles.append(
    #     Line2D([], [],
    #            marker='o',
    #            color='black',
    #            linestyle='None',
    #            markersize=17))
    # labels.append('New Allocation')
    second_handles = [
        Line2D([], [], marker="x", color="black", linestyle="None", markersize=17)
    ]
    # second_labels = ['Reuse Prev. Allocation']

    first_legend = ax.legend(
        handles,
        labels,
        ncol=4,
        frameon=False,
        columnspacing=1.0,
        loc="upper center",
        bbox_to_anchor=(0, 0, 0.95, 1.15),
    )
    # second_legend = ax.legend(second_handles,
    #                           second_labels,
    #                           ncol=1,
    #                           frameon=True,
    #                           loc='center',
    #                           bbox_to_anchor=(0, 0, 1.0, 0.62))
    # second_legend.get_frame().set_linewidth(0.0)
    ax.add_artist(first_legend)
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    # plt.xticks(rotation=15)
    ax.set_xlim(-XLIM_OFFSET, (len(DEMAND_PER_TM) - 1) * TIME_INTERVAL + XLIM_OFFSET)

    if col_to_plot == "satisfied_demand":
        ax.set_ylim((-4.964740827529834, 104.25955737812652))
        ax.set_yticks(np.arange(0, 100, 20))
    print(ax.get_ylim())

    ax.set_xlabel("Time (mins)")
    if ylabel is not None:
        ax.set_ylabel(ylabel)
        # if 'cum_' in col_to_plot:
        ax.yaxis.set_major_formatter(PercentFormatter())

    if title is not None:
        ax.set_title(title, y=1.05, fontsize=20)

    ax.grid(True)
    sns.despine()
    print("NC:", ncflow_df.iloc[-1]["cum_demand_satisfied"] / sum(DEMAND_PER_TM))
    print("PF:", pf_df.iloc[-1]["cum_demand_satisfied"] / sum(DEMAND_PER_TM))
    print("PFWS:", pfws_df.iloc[-1]["cum_demand_satisfied"] / sum(DEMAND_PER_TM))
    print("Oracle:", oracle_df.iloc[-1]["cum_demand_satisfied"] / sum(DEMAND_PER_TM))
    save_figure(
        "demand-tracking-{}-points".format(num_points), extra_artists=[first_legend]
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        num_points = int(sys.argv[1])
    else:
        num_points = 25
    plot_demand_tracking(
        os.path.join(CSV_ROOT_DIR, "demand-tracking", "no-unmet-demand"), num_points + 1
    )
    if sys.argv[-1] == "--plot-all":
        for n in range(num_points + 1):
           print('plotting {} points'.format(n))
           plot_demand_tracking('no-unmet-demand/', n)
