#!/usr/bin/env python3
from matplotlib.lines import Line2D
import tqdm
from typing import Tuple
import tueplots
import itertools
from tueplots.bundles import aistats2023
import torchvision
import scipy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os


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


# Load data
def load_npy(spn, figure, filename, normalize=False):
    data = np.load(os.path.join("data", figure, spn, filename))
    if normalize:
        data = np.exp(data - np.repeat(scipy.special.logsumexp(data, axis=1).reshape(-1, 1), 10, axis=1))
    return data


def entropy(data: np.ndarray, axis=-1) -> np.ndarray:
    return -1 * np.sum(data * np.log(data), axis=axis)


def figure_3():
    data_files = {
        "dc": [
            ("output_svhn_train_0.1_None.npy", True),
            ("output_svhn_test_0.1_None.npy", True),
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
        ],
    }

    labels = [
        "SVHN Train, ID",
        "SVHN Test, ID",
        # "OOD (CIFAR10 Test)",
        "CIFAR-100, OOD",
        "CINIC, OOD",
        "LSUN, OOD",
    ]

    with matplotlib.rc_context(aistats2023()):
        plt.style.use(["science", "grid"])
        fig = plt.figure(figsize=get_figsize(scale=0.95, num_columns=1))
        hatches = ["//////", "ooo", "-", "+", "x", "\\", "*", "o"]

        # Plot each dataset
        offset = 0
        for i, ((data_file_var, norm_var), (data_file_dc, norm_dc), label) in enumerate(
            zip(data_files["var"], data_files["dc"], labels)
        ):
            data_var = load_npy("dc", figure="fig1", filename=data_file_var, normalize=False)
            data_dc = load_npy("dc", figure="fig1", filename=data_file_dc, normalize=norm_dc)
            stds = np.sqrt(np.exp(data_var))

            pred_idx_dc = np.argmax(data_dc, axis=1)
            stds = np.take_along_axis(stds, pred_idx_dc[:, None], axis=1)[:, 0]

            if "ID" in label:
                offset += 1
                alpha = 1.0
                color = "black"
                fill = False
            else:
                alpha = 0.5
                color = colors[i - offset]
                fill = True

            bars = sns.histplot(
                stds,
                bins=40,
                stat="probability",
                element="bars",
                label=label,
                color=color,
                alpha=alpha,
                fill=fill,
            )

            # Add hatches to in-distribution datasets
            if "ID" in label:
                for thisbar in bars.patches[i * 20 :]:
                    # Set a different hatch for each bar
                    thisbar.set_hatch(hatches[i])

        # Add legend to the figure
        legend = plt.legend(
            fancybox=False,
            fontsize="x-small",
            edgecolor="black",
            loc="upper right",
        )
        legend.get_frame().set_linewidth(0.5)

        plt.xlabel("DC Predictive Uncertainty")
        plt.ylabel("Dataset Proportion")
        plt.xlim(0.0, 0.2)
        plt.xticks([0.0, 0.05, 0.1, 0.15, 0.2])
        plt.grid(False)

        name = "figure-3"
        path = os.path.join("figs", name)
        plt.tight_layout()
        plt.savefig(path + ".pgf")
        plt.savefig(path + ".pdf")
        plt.savefig(path + ".jpg", dpi=300)


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
        ],
    }

    labels = [
        "SVHN Train, ID",
        "SVHN Test, ID",
        # "OOD (CIFAR10 Test)",
        "CIFAR-100, OOD",
        "CINIC, OOD",
        "LSUN, OOD",
    ]

    # Plot PC/DC
    def plot(spn_type, ax):
        # Define some hatches
        hatches = ["//////", "ooo", "-", "+", "x", "\\", "*", "o"]

        # Plot each dataset
        offset = 0
        for i, ((data_file, normalize), label) in tqdm.tqdm(enumerate(zip(data_files[spn_type], labels))):
            data = load_npy(spn_type, figure="fig1", filename=data_file, normalize=normalize)
            data_entropy = entropy(data)

            if "ID" in label:
                offset += 1
                alpha = 1.0
                color = "black"
                fill = False
            else:
                alpha = 0.5
                color = colors[i - offset]
                fill = True

            bars = sns.histplot(
                data_entropy,
                bins=20,
                stat="probability",
                element="bars",
                ax=ax,
                label=label,
                color=color,
                binrange=(0, 2.4),
                alpha=alpha,
                fill=fill,
            )

            # Add hatches to in-distribution datasets
            if "ID" in label:
                for thisbar in bars.patches[i * 20 :]:
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
            start = np.array([np.log(1 / 10) * -1, 1.0])
            end = np.array([np.log(1 / 10) * -1, 0.00])
            line = np.array([start, end])
            lw = 0.5
            ax.plot(*line.T, "--", color="red", lw=lw)
            ax.arrow(x=2.0, y=0.765, dx=0.2, dy=0.0, linewidth=0.3, head_width=0.02, fill=True, color="black")
            ax.text(x=1.55, y=0.75, s="max entropy", ha="center", fontsize=6)

    # Use tueplots aistats2023 template
    with matplotlib.rc_context(aistats2023()):
        fig, (ax0, ax1) = plt.subplots(
            nrows=1, ncols=2, figsize=get_figsize(scale=0.95, num_columns=1, aspect_ratio=0.5), sharey=True
        )
        plot("pc", ax0)
        plot("dc", ax1)
        handles, _ = ax1.get_legend_handles_labels()

        # Add pseudo-handle
        l = Line2D([0], [0], color="w")
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


def load_npy_fig4(figure, filename, normalize=False):
    data = np.load(os.path.join("data", figure, filename))
    if normalize:
        data = np.exp(data - np.repeat(scipy.special.logsumexp(data, axis=1).reshape(-1, 1), 10, axis=1))
    return data


def get_mnist_labels():
    dataset = torchvision.datasets.MNIST(root="~/data/", train=False)
    return dataset.test_labels.numpy()


def get_svhn_labels():
    dataset = torchvision.datasets.SVHN(root="~/data/", split="test")
    return dataset.labels


def figure_4():

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
        rotations = [int(x) for x in np.arange(stop=95, step=5)]

        data_files = {
            "pc": [f"pc_{r}.npy" for r in rotations],
            "dc": [f"dc_{r}.npy" for r in rotations],
            "var": [f"dc_vars_{r}.npy" for r in rotations],
        }

        labels = {"pc": "Probabilistic circuit", "dc": "Dropout circuit"}

        mnist_targets = get_mnist_labels()
        mnist_targets = mnist_targets[:, None].repeat(len(rotations), axis=1)

        data_dict = {}
        plt.figure(figsize=get_figsize(scale=0.95, num_columns=1, aspect_ratio=0.5))
        for i, (key, filelist) in enumerate(data_files.items()):
            l = []
            for data_file in filelist:
                data = load_npy_fig4(figure="fig4", filename=data_file, normalize=key != "var")
                l.append(data)
            if key == "var":
                data_dict[key] = np.exp(np.stack(l, axis=2))
            else:
                data_dict[key] = np.stack(l, axis=1)

        # Create figure with two subplots
        fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, figsize=get_figsize(scale=0.95, num_columns=1), sharex=True)

        ax01 = ax0.twinx()

        #################################
        # Plot entropy on first subplot #
        #################################
        for i, key in enumerate(["pc", "dc"]):
            data = data_dict[key]
            data_entropy = entropy(data)
            y = np.median(data_entropy, axis=0)

            # Plot pred entropy
            ax0.plot(
                rotations,
                y,
                marker=markers[i],
                label=labels[key],
                markersize=markersize,
                markeredgecolor="black",
                markeredgewidth=0.5,
                color=colors[i],
                alpha=alpha,
            )

            # Plot accuracy
            preds = np.argmax(data, axis=2)
            accuracy = (preds == mnist_targets).sum(0) / data.shape[0] * 100
            ax01.plot(
                rotations,
                accuracy,
                "--",
                marker=markers[i],
                label=labels[key],
                markersize=markersize,
                markeredgecolor="black",
                markeredgewidth=0.5,
                color=colors[i],
                alpha=alpha,
            )

        ax01.set_ylabel("Accuracy (- -)", fontsize=label_fontsize)
        ax01.grid(False)
        ax01.set_ylim(-5, 105)

        ax0.set_xticks([0, 15, 30, 45, 60, 75, 90])
        ax0.set_ylabel("Pred. Entropy", fontsize=label_fontsize)
        ax0.set_yticks([0.0, 0.1, 0.2], ["0.00", "0.10", "0.20"])
        ax0.set_ylim(np.max(y) - np.max(y) * 1.05, np.max(y) * 1.05)

        ##############################
        # Plot std on second subplot #
        ##############################
        pred_idx_dc = np.argmax(data_dict["dc"], axis=2)
        vars = data_dict["var"]
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
        ax1.set_yticks([0.0, 0.03, 0.06])

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

        name = "figure-4"
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


def figure_5():
    corruption_labels = [
        "brightness",
        "contrast",
        "defocus_blur",
        "elastic_transform",
        "fog",
        "gaussian_noise",
        "jpeg_compression",
        "motion_blur",
        "pixelate",
        "shot_noise",
        "snow",
        "frost",
    ]

    no_corr_dict = {}
    for typ in ["pc", "dc"]:
        file = f"{typ}_no_corr.npy"
        data = np.load(os.path.join("data", "fig5", file))
        if typ == "dc":
            # DC data is normalized but in logspace
            data = np.exp(data)
        no_corr_dict[typ] = data

    #############
    # Load data #
    #############

    svhn_targets = get_svhn_labels()
    svhn_targets = svhn_targets[:, None].repeat(6, axis=1)

    data_dict = {}
    for corruption in corruption_labels:
        data_dict[corruption] = {}
        for typ in ["pc", "dc"]:
            data_dict[corruption][typ] = {}
            sevs = []

            sevs.append(no_corr_dict[typ])
            for severity in range(1, 6):
                file = f"{typ}_{corruption}_{severity}.npy"
                data = np.load(os.path.join("data", "fig5", file))
                if typ == "pc":
                    # PC data needs to be normalized
                    data = np.exp(data - np.repeat(scipy.special.logsumexp(data, axis=1).reshape(-1, 1), 10, axis=1))
                else:
                    # DC data is normalized but in logspace
                    data = np.exp(data)
                # data_entropy = entropy(data)
                # sevs.append(np.median(data_entropy, axis=0))
                sevs.append(data)

            data_dict[corruption][typ] = np.stack(sevs, axis=-1)

    ############
    # PLOTTING #
    ############
    use_markers = True
    label_fontsize = 8
    markersize = 3

    if use_markers:
        alpha = 0.8
        markers = ["o", "^"]
    else:
        alpha = 1.0
        markers = [None, None]

    plt.style.use(["science", "grid"])

    corruptions = ["brightness", "elastic_transform", "frost", "gaussian_noise"]
    corruption_labels = ["Brightness", "Elastic transformation", "Frost", "Gaussian noise"]
    labels = {"pc": "Probabilistic circuit", "dc": "Dropout circuit"}

    # Use tueplots aistats2023 template
    with matplotlib.rc_context(aistats2023()):

        # Create figure with two subplots
        fig, axs = plt.subplots(
            nrows=1, ncols=4, figsize=get_figsize(scale=0.95, num_columns=0.5, aspect_ratio=0.5), sharey=True
        )
        # axs = axs.reshape(1, 4)
        for row in range(1):
            for col in range(4):
                # ax = axs[row, col]
                ax = axs[col]
                ax1 = ax.twinx()

                corr = corruptions[row * 4 + col]

                #################################
                # Plot entropy on first subplot #
                #################################
                for i, key in enumerate(["pc", "dc"]):
                    x = range(0, 6)
                    y = data_dict[corr][key]
                    y_entropy = np.median(entropy(y, axis=1), axis=0)

                    # Plot pred entropy
                    ax.plot(
                        x,
                        y_entropy,
                        marker=markers[i],
                        label=labels[key],
                        markersize=markersize,
                        markeredgecolor="black",
                        markeredgewidth=0.5,
                        color=colors[i],
                        alpha=alpha,
                    )

                    # Plot accuracy
                    preds = np.argmax(y, axis=1)
                    accuracy = (preds == svhn_targets).sum(0) / data.shape[0] * 100
                    ax1.plot(
                        x,
                        accuracy,
                        "--",
                        marker=markers[i],
                        label=labels[key],
                        markersize=markersize,
                        markeredgecolor="black",
                        markeredgewidth=0.5,
                        color=colors[i],
                        alpha=alpha,
                    )

                ax.set_xticks([0, 1, 2, 3, 4, 5])
                ax.set_xlim(0, 5)
                ax.set_xlabel(corruption_labels[row * 4 + col], fontsize=label_fontsize)
                ax1.grid(False)

                ax1.set_ylim(-5, 85)
                if col < 3:
                    ax1.set_yticks([], [])
                else:
                    ax1.set_ylabel("Accuracy (- -)", fontsize=label_fontsize - 1)

        handles, labels = axs[0].get_legend_handles_labels()


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
            bbox_to_anchor=(0.75, -0.02),
            columnspacing=1.0,
        )
        legend.get_frame().set_linewidth(0.5)

        axs[0].set_ylabel("Predictive Entropy", fontsize=label_fontsize)

        # Set xlabel for the whole figure
        fig.supxlabel("Corruption severity", fontsize=label_fontsize)

        # Save fig
        name = "figure-5"
        path = os.path.join("figs", name)
        plt.tight_layout()
        print("Saving jpg")
        plt.savefig(path + ".jpg", dpi=300)
        print("Saving pdf")
        plt.savefig(path + ".pdf")


if __name__ == "__main__":
    setup(use_pgf=False)
    figure_1()
    figure_3()
    figure_4()
    figure_5()
