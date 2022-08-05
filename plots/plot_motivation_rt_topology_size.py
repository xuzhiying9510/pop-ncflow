#! /usr/bin/env python

from matplotlib.lines import Line2D
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

import sys
sys.path.append('..')

from lib.plot_utils import save_figure, PROBLEM_NAMES_DICT, CSV_ROOT_DIR

plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 18

CSV_DIR = os.path.join(CSV_ROOT_DIR, "total-flow")


def plot_motivation_rt_topology_size():
    pf_df = pd.read_csv(os.path.join(CSV_DIR, 'path-form-total_flow-slice_0_1_2_3_4.csv'))
    scale_factor = [8.0, 16.0, 32.0, 64.0, 128.0]
    # model = 'uniform'
    topos_to_include = [
        'AttMpls.graphml',
        'Uninett2010.graphml',
        'Interoute.graphml',
        'TataNld.graphml',
        'Cogentco.graphml',
        'Kdl.graphml',
    ]

    # pf_df['nodes_and_edges'] = (pf_df['num_nodes'] + pf_df['num_edges']) / 2
    # pf_df['unsatisfied_demand'] = (pf_df['total_demand'] - pf_df['total_flow']) / pf_df['total_demand']

    plot_df = pf_df.query(
        'scale_factor in @scale_factor and problem in @topos_to_include')
    plot_df.sort_values(by='num_edges', inplace=True)
    xticklabels = [
    '{}'.format(
            plot_df.query(
                'problem == "{}"'.format(topo)).iloc[-1]['num_nodes'])
        for topo in topos_to_include
    ]

    fig, ax = plt.subplots(figsize=(9, 3.5))
    sns.lineplot(ax=ax,
                 x='num_edges',
                 y='runtime',
                 data=plot_df,
                 marker='o',
                 ci='sd')

    # ax.axhline(300, linestyle='--', color='black')
    # handles = [
    #     Line2D([], [],
    #            color='black',
    #            linestyle='--')
    # ]
    # labels = ['5-minute time window']
    # legend = ax.legend(handles, labels, loc='best')
    # legend.get_frame().set_linewidth(0.0)

    # ax.plot(plot_df['num_edges'], plot_df['runtime'], marker='o')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(plot_df['num_edges'].unique())
    ax.set_xticklabels(xticklabels)
    ax.tick_params(axis='x', which='minor', bottom=False)
    ax.set_xlabel('Number of Nodes, log scale', labelpad=5.0)
    ax.set_ylabel('Runtime (s), log scale', position=(0, 0.5))
    ax.set_yticks([1e-2, 1e-1, 1e0, 1e1, 1e1, 1e2, 1e3])
    ax.grid(True)

    # Plot number of edges on top x-axis
    # top_ax = ax.twiny()
    # top_ax.set_xscale('log')
    # top_ax.set_xlim(ax.get_xlim())
    # top_ax.set_xlabel('Number of Edges, log scale', labelpad=15.0)
    # top_ax.set_xticks(ax.get_xticks())
    # top_ax.xaxis.set_tick_params(which='minor', bottom=False)
    # top_ax.set_xticklabels(plot_df['num_edges'])
    sns.despine()
    # top_ax.tick_params(axis='x', which='minor', top=False)

    # save_figure('motivation-rt-topology-size', extra_artists=(legend,))
    save_figure('motivation-rt-topology-size')


if __name__ == '__main__':
    plot_motivation_rt_topology_size()
