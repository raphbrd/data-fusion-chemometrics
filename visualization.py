import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def plot_match_scores(scores, x_ticks=None, x_tick_labels=None, labels=None, colors=None, ax=None, show=False):
    if ax is None:
        fig, ax = plt.subplots(figsize=(2.75, 2.25), dpi=150)
    else:
        fig = ax.figure

    if x_ticks is None:
        x_ticks = np.arange(len(scores))
    if x_tick_labels is None:
        x_tick_labels = x_ticks
    if colors is None:
        colors = mpl.colormaps['tab20'].colors

    ax.axhline(1, color="grey", linestyle="--", linewidth=0.5)

    scores = np.array(scores)
    if scores.ndim == 1:
        scores = [scores]
    colors = np.array(colors)
    if colors.ndim == 0:
        colors = np.array([colors] * len(scores))
    width = 0.75 / len(scores)
    for idx, score in enumerate(scores):
        offset = width * idx
        ax.bar(x_ticks + offset, score, width, color=colors[idx], label=labels[idx] if labels is not None else None)

    ax.set_xticks(x_ticks + width * max(len(scores) - 2, 0), x_tick_labels, fontsize=8)
    ax.set_yticks([0, 0.5, 1.0], [0, 0.5, 1.0], fontsize=8)
    ax.tick_params(axis="y", labelsize=8, left=True, color="grey")
    # ax.set_xlabel("Component", fontsize=10)
    ax.set_ylabel("Match score", fontsize=10)
    if labels is not None:
        ax.legend(loc="upper right", ncols=len(scores), fontsize=6, frameon=False, bbox_to_anchor=(1, 1.1))
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if show:
        plt.show()

    return fig


def plot_concentrations_per_mixtures(
        gt_concentrations,
        hat_concentrations,
        col_inds,
        col_labels,
        color_gt="black",
        color_hat="tab:blue",
        legend_label="CMTF estimate",
):
    """ Plot the concentrations per mixtures for the ground truth and the estimated concentrations

    Parameters
    ----------
    gt_concentrations : np.ndarray
        The ground truth concentrations
    hat_concentrations : np.ndarray
        The estimated concentrations
    col_inds : list
        The indices of the best permutation match for the match score
    col_labels : list
        The labels for the columns of the concentrations
    color_gt : str
        The color of the ground truth concentrations
    color_hat : str
        The color of the estimated concentrations
    legend_label : str
        The label for the legend of the estimated concentrations
    """
    # because the tensor represents concentrations, we take the absolute value
    # (non-negative components)
    # this is figure 11 of Acar et al. (2014)
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(5, 3.5), sharex=False, sharey=False, dpi=150)
    axes = axes.flatten()

    x_ticks = np.arange(1, gt_concentrations.shape[0] + 1)

    for idx, ax in enumerate(axes[:5]):
        ax.plot(x_ticks, gt_concentrations[:, idx], marker="o", markersize=4, linewidth=1, label="ground truth",
                color=color_gt)
        y = hat_concentrations[:, col_inds[idx]]
        ax.plot(x_ticks, y, marker="o", markersize=4, linewidth=1, label=legend_label, color=color_hat)
        ax.grid(False)
        ax.tick_params(axis="both", labelsize=6, left=True, bottom=True, color="grey")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("mixtures", fontsize=8)
        ax.set_ylabel(col_labels[idx], fontsize=8)
        ax.set_ylim(-0.1, 0.7)
    axes[-1].axis("off")
    handles, labels = ax.get_legend_handles_labels()
    fig.tight_layout(h_pad=0.2)
    fig.legend(handles, labels, loc='lower center', fontsize=8, ncol=1, bbox_to_anchor=(0.8, 0.1),
               bbox_transform=fig.transFigure)
    return fig
