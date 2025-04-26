import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def plot_match_scores(scores, x_ticks=None, x_tick_labels=None, labels=None, colors=None, ax=None, show=False):
    if ax is None:
        fig, ax = plt.subplots(figsize=(2.75, 2.25), dpi=150)

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
        ax.legend(loc="upper right", ncols=len(scores), fontsize=6, frameon=False, bbox_to_anchor=(1, 1.2))
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if show:
        plt.show()

    return fig
