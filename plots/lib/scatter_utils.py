import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from .plot_utils import save_figure, COLOR_NAMES_DICT


def pop_scatter_plot(
    runtimes,
    obj_vals,
    labels,
    ylabel,
    ann_factor_x,
    ann_factor_y,
    arrow_coord_x,
    arrow_coord_y,
    arrow_rotation,
    figsize=(7.5, 2.5),
    annotate_values=False,
    output_filename=None,
):
    plt.figure(figsize=figsize)
    ax = plt.subplot2grid((1, 1), (0, 0), colspan=1)
    for (runtime, obj_val, label) in zip(runtimes, obj_vals, labels):
        ax.scatter(runtime, obj_val, label=label)
        ax.annotate(label, (runtime * ann_factor_x, obj_val * ann_factor_y))
        if annotate_values:
            ax.annotate(
                "{:.3f}".format(obj_val),
                (runtime * ann_factor_x, obj_val * (0.85 / ann_factor_y)),
            )

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Runtime (seconds)")
    ax.set_xscale("log")
    ax.yaxis.major.formatter._useMathText = True
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    plt.xlim(xmin, xmax * 1.2)
    plt.ylim(0, 1.0)
    plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    sns.despine()

    bbox_props = dict(boxstyle="larrow", ec="k", lw=2, fc="white")
    t = ax.text(
        arrow_coord_x,
        arrow_coord_y,
        "Better",
        ha="center",
        va="center",
        rotation=arrow_rotation,
        bbox=bbox_props,
    )
    bb = t.get_bbox_patch()
    bb.set_boxstyle("larrow", pad=0.3)

    if output_filename is not None:
        with PdfPages(output_filename) as pdf:
            pdf.savefig(bbox_inches="tight")


def scatter_plot(
    ratio_dfs,
    labels,
    short_labels,
    x_axis,
    y_axis,
    xlabel=None,
    ylabel=None,
    xlim=0,
    ylim=(10e-3, 5e3),
    title=None,
    xlog=False,
    ylog=False,
    bbta=(0, 0, 1, 1),
    ncol=2,
    figsize=(8, 6),
    ax=None,
    show_legend=True,
    arrow_coords=None,
    show=False,
    save=True,
):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    for ratio_df, label, short_label in zip(ratio_dfs, labels, short_labels):
        ax.scatter(
            ratio_df[x_axis],
            ratio_df[y_axis],
            alpha=0.75,
            label=label,
            marker="o",
            linewidth=1,
            color=COLOR_NAMES_DICT[short_label],
        )
    if xlim:
        ax.set_xlim(0)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(ylim)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

    extra_artists = []
    if arrow_coords:
        bbox_props = {
            "boxstyle": "rarrow,pad=0.5",
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
            rotation=45,
            size=16,
            color="black",
            bbox=bbox_props,
        )
        extra_artists.append(t)

    if show_legend:
        legend = ax.legend(
            loc="center",
            bbox_to_anchor=bbta,
            ncol=ncol,
            frameon=False,
            handletextpad=0.2,
            columnspacing=0.2,
        )
        extra_artists.append(legend)
    sns.despine()
    if show:
        plt.show()
    if save:
        save_figure(
            "scatter-plot-{}-{}-{}".format(x_axis, y_axis, title),
            extra_artists=extra_artists,
        )
