#!/usr/bin/env python3
from matplotlib.lines import Line2D
import tqdm
from typing import Tuple
import tueplots
import itertools
from tueplots.bundles import aistats2023
import scipy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
matplotlib.rcParams['hatch.linewidth'] = 0.5  # previous pdf hatch linewidth


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


def get_figsize(
    scale=1.0, aspect_ratio=(5.0**0.5 - 1.0) / 2.0, num_columns=1
) -> Tuple[float, float]:
    """Compute the figure size based on the latex template textwidth in pt. Print via '\the\textwidth' in latex."""
    # Compute optimal figure sizes
    columnwidth_pt = 234.8775  # AISTATS23 columnwidth in pt
    textwidth = columnwidth_pt / 72  # pt to inch
    aspect_ratio *= num_columns
    scale = scale / num_columns
    width = textwidth * scale
    height = width * aspect_ratio
    return width, height


# Load data
def load_npy(spn, figure, filename, normalize=False):
    data = np.load(os.path.join("data", figure, spn, filename))
    if normalize:
        data = np.exp(
            data - np.repeat(scipy.special.logsumexp(data, axis=1).reshape(-1, 1), 10, axis=1)
        )
    return data


def entropy(data: np.ndarray) -> np.ndarray:
    return -1 * np.sum(data * np.log(data), axis=-1)


def figure_1():

    data_files = {
        "pc": [
            ("class_probs_in_domain_train.npy", False),
            ("class_probs_in_domain_test.npy", False),
            # ("class_probs_ood_test.npy", False),
            ("output_cifar100_test_1.0_0.0_0.0_None.npy", True),
            ("output_cinic_test_1.0_0.0_0.0_None.npy", True),
            ("output_lsun_test_1.0_0.0_0.0_None.npy", True),
        ],
        "dc": [
            ("output_svhn_train_0.1_None.npy", True),
            ("output_svhn_test_0.1_None.npy", True),
            # ("output_cifar_test_0.1_None.npy", True),
            ("output_cifar100_test_1.0_0.1_0.1_None.npy", True),
            ("output_cinic_test_1.0_0.1_0.1_None.npy", True),
            ("output_lsun_test_1.0_0.1_0.1_None.npy", True),
        ],
        "var": [
            ("var_svhn_train_0.1_None.npy", True),
            ("var_svhn_test_0.1_None.npy", True),
            ("var_cifar100_test_1.0_0.1_0.1_None.npy", True),
            ("var_cinic_test_1.0_0.1_0.1_None.npy", True),
            ("var_lsun_test_1.0_0.1_0.1_None.npy", True),

        ]
    }

    labels = [
        "SVHN Train, ID",
        "SVHN Test, ID",
        # "OOD (CIFAR10 Test)",
        "CIFAR100 Test, OOD",
        "CINIC Test, OOD",
        "LSUN Test, OOD",
    ]

    def plot_var():
        fig = plt.figure(figsize=get_figsize(scale=0.95, num_columns=1))

        # Plot each dataset
        for i, ((data_file, normalize), label) in enumerate(zip(data_files["var"], labels)):
            data = load_npy("dc", figure="fig1", filename=data_file, normalize=False)
            data = data.reshape(-1)
            # data_entropy = entropy(data)
            # sns.kdeplot(data, fill=True, common_norm=False, label=label)
            sns.histplot(data, fill=True, bins=20, common_norm=False, label=label)

        # Add legend to the figure
        legend = plt.legend(
            fancybox=False,
            fontsize="xx-small",
            edgecolor="black",
            loc="upper left",
        )
        legend.get_frame().set_linewidth(0.5)



        plt.xlabel("Uncertainty (std)")
        plt.ylabel("Density")
        plt.xlim(-40, 0)

        name = "figure-1-var"
        path = os.path.join("figs", name)
        plt.tight_layout()
        plt.savefig(path + ".pgf")
        plt.savefig(path + ".pdf")
        plt.savefig(path + ".jpg", dpi=300)


    # Plot PC/DC
    def plot(spn_type, ax):
        # Define some hatches
        hatches = ['//////', 'ooo', '-', '+', 'x', '\\', '*', 'o']


        # Plot each dataset
        offset = 0
        for i, ((data_file, normalize), label) in tqdm.tqdm(enumerate(zip(data_files[spn_type], labels))):
            data = load_npy(spn_type, figure="fig1", filename=data_file, normalize=normalize)
            data_entropy = entropy(data)

            if "ID" in label:
                offset += 1
                alpha = 1.0
                color="black"
                fill=False
            else:
                alpha = 0.5
                color=colors[i-offset]
                fill=True

            bars = sns.histplot(data_entropy, bins=20, stat="probability", element="bars", ax=ax, label=label, color=color, binrange=(0, 2.4), alpha=alpha, fill=fill)

            # Add hatches to in-distribution datasets
            if "ID" in label:
                for thisbar in bars.patches[i*20:]:
                    # Set a different hatch for each bar
                    thisbar.set_hatch(hatches[i])


        # Set labels
        if spn_type == "dc":
            ax.set_xlabel("DC Predictive Entropy")
        else:
            ax.set_xlabel("PC Predictive Entropy")

        # Set axis limits
        ax.set_ylabel("Probability")
        ax.set_xlim(0, 2.4)
        ax.set_ylim(0, 0.85)

        # Adjust ax width
        adjust_linewidths(ax)

        # Annotate max entropy line
        if spn_type == "dc":
            start = np.array([np.log(1/10) * -1, 1.0])
            end = np.array([np.log(1/10) * -1, 0.00])
            line = np.array([start, end])
            lw = 0.5
            ax.plot(*line.T, "--", color="red", lw=lw)
            ax.arrow(x=2.0, y=0.765, dx=0.2, dy=0.0, linewidth=0.3, head_width=0.02, fill=True, color="black")
            ax.text(x=1.55, y=0.75, s="max entropy", ha="center", fontsize=6)


    # Use tueplots aistats2023 template
    with matplotlib.rc_context(aistats2023()):
        plot_var()

        fig, (ax0, ax1) = plt.subplots(
            nrows=1, ncols=2, figsize=get_figsize(scale=0.95, num_columns=1, aspect_ratio=0.5), sharey=True
        )
        plot("pc", ax0)
        plot("dc", ax1)
        handles, _ = ax1.get_legend_handles_labels()

        # Add pseudo-handle
        l = Line2D([0],[0],color="w")
        handles.insert(2, l)
        labels.insert(2, "")

        # Add legend to the figure
        legend = fig.legend(
            flip(handles, ncol=3),
            flip(labels, ncol=3),
            alignment="center",
            fancybox=False,
            fontsize="xx-small",
            edgecolor="white",
            # edgecolor="black",
            loc="lower center",
            ncol=3,
            bbox_to_anchor=(0.55, -0.175),
            columnspacing=1.0,
        )
        legend.get_frame().set_linewidth(0.5)

        name = "figure-1"
        path = os.path.join("figs", name)
        plt.tight_layout()
        print("Saving jpg")
        plt.savefig(path + ".jpg", dpi=300)
        # Don't do PGF with hatches -> results in a 30MB+ file
        # print("Saving pgf")
        # plt.savefig(path + ".pgf")
        # PDF without mpl.use("pgf") is also a perfect match regarding fonts
        print("Saving pdf")
        plt.savefig(path + ".pdf")


if __name__ == "__main__":
    setup(use_pgf=False)
    figure_1()
