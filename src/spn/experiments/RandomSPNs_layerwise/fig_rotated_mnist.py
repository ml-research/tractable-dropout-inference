#!/usr/bin/env python3
from typing import Tuple
import itertools
from tueplots.bundles import aistats2023
import torchvision
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


colors = sns.color_palette("tab10")


def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])


def adjust_linewidths(ax, width=0.5):
    """Set spine and tick linewidth to 0.5"""
    ax.spines["left"].set_linewidth(width)
    ax.spines["bottom"].set_linewidth(width)
    ax.spines["right"].set_linewidth(width)
    ax.spines["top"].set_linewidth(width)
    ax.tick_params(width=width)


def setup(use_pgf: bool):
    np.random.seed(32)

    # Enable pgf module
    if use_pgf:
        matplotlib.use("pgf")
    matplotlib.rcParams["hatch.linewidth"] = 0.5  # previous pdf hatch linewidth
    matplotlib.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": False,
        }
    )

    # Use SciencePlots  |  pip install SciencePlots
    # plt.style.use(["science", "grid"])
    plt.style.use(["science"])


def get_figsize(scale=1.0, aspect_ratio=(5.0**0.5 - 1.0) / 2.0, num_columns=1) -> Tuple[float, float]:
    """Compute the figure size based on the latex template textwidth in pt. Print via '\the\textwidth' in latex."""
    # Compute optimal figure sizes
    columnwidth_pt = 234.8775  # AISTATS23 columnwidth in pt
    textwidth = columnwidth_pt / 72  # pt to inch
    aspect_ratio *= num_columns
    scale = scale / num_columns
    width = textwidth * scale
    height = width * aspect_ratio
    return width, height


def get_mnist_labels():
    dataset = torchvision.datasets.MNIST(root="~/data/", train=False, download=True)
    return dataset.targets.numpy()


def entropy(data: np.ndarray, axis=-1) -> np.ndarray:
    return -1 * np.sum(data * np.log(data), axis=axis)


def plot_figure(data, rotations):

    plt.style.use(["science", "grid"])

    use_markers = True
    label_fontsize = 8
    markersize = 3

    if use_markers:
        alpha = 0.8
        markers = ["o", "^"]
    else:
        alpha = 1.0
        markers = [None, None]

    # Use tueplots aistats2023 template
    with matplotlib.rc_context(aistats2023()):

        labels = {"dc": "Dropout circuit"}

        mnist_targets = get_mnist_labels()
        mnist_targets = mnist_targets[:, None].repeat(len(rotations), axis=1)

        data_dict = {}
        plt.figure(figsize=get_figsize(scale=0.95, num_columns=1, aspect_ratio=0.5))
        l, k = [], []
        for rot, res in data.items():
            l.append(res[0])
            k.append(res[1])

        data_dict["confs"] = np.exp(np.stack(l, axis=1))
        data_dict["vars"] = np.exp(np.stack(k, axis=2))

        # Create figure with two subplots
        fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, figsize=get_figsize(scale=0.95, num_columns=1), sharex=True)

        ax01 = ax0.twinx()

        #################################
        # Plot entropy on first subplot #
        #################################
        data = data_dict["confs"]
        data_entropy = entropy(data)
        y = np.median(data_entropy, axis=0)

        # Plot pred entropy
        ax0.plot(
            rotations,
            y,
            marker=markers[1],
            label=labels["dc"],
            markersize=markersize,
            markeredgecolor="black",
            markeredgewidth=0.5,
            color=colors[1],
            alpha=alpha,
        )

        # Plot accuracy
        preds = np.argmax(data, axis=2)
        accuracy = (preds == mnist_targets).sum(0) / data.shape[0] * 100
        ax01.plot(
            rotations,
            accuracy,
            "--",
            marker=markers[1],
            label=labels["dc"],
            markersize=markersize,
            markeredgecolor="black",
            markeredgewidth=0.5,
            color=colors[1],
            alpha=alpha,
        )

        ax01.set_ylabel("Accuracy (- -)", fontsize=label_fontsize)
        ax01.grid(False)
        ax01.set_ylim(-5, 105)

        ax0.set_xticks(rotations)
        ax0.set_ylabel("Pred. Entropy", fontsize=label_fontsize)
        # ax0.set_yticks([0.0, 0.1, 0.2], ["0.00", "0.10", "0.20"])
        ax0.set_ylim(np.max(y) - np.max(y) * 1.05, np.max(y) * 1.05)

        ##############################
        # Plot std on second subplot #
        ##############################
        pred_idx_dc = np.argmax(data_dict["confs"], axis=2)
        vars = data_dict["vars"]
        stds = np.sqrt(vars)
        stds = np.take_along_axis(stds, pred_idx_dc[:, None, :], axis=1)[:, 0, :]
        stds = np.median(stds, axis=0)

        ax1.plot(
            rotations,
            stds,
            marker=markers[1],
            label=labels["dc"],
            markersize=markersize,
            markeredgecolor="black",
            markeredgewidth=0.5,
            color=colors[1],
            alpha=alpha,
        )
        ax1.set_ylabel("Pred. Uncertainty", fontsize=label_fontsize)
        ax1.set_ylim(np.max(stds) - np.max(stds) * 1.05, np.max(stds) * 1.05)
        # ax1.set_yticks([0.0, 0.03, 0.06])

        handles, labels = ax0.get_legend_handles_labels()

        # Add legend to the figure
        legend = fig.legend(
            handles,
            labels,
            alignment="center",
            fancybox=False,
            fontsize="x-small",
            edgecolor="white",
            # edgecolor="black",
            loc="lower center",
            ncol=2,
            bbox_to_anchor=(0.5, -0.1),
            columnspacing=1.0,
        )
        legend.get_frame().set_linewidth(0.5)

        fig.align_ylabels([ax0, ax1])
        plt.xlim(0, 90)
        ax1.set_xlabel("Rotation (degrees)", fontsize=label_fontsize)


        plt.tight_layout()
        print("Saving jpg")
        plt.savefig("./rotated_mnist.jpg", dpi=300)
        print("Saving pdf")
        plt.savefig("./rotated_mnist.pdf")