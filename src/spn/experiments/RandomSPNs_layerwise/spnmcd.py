from locale import normalize
from os import confstr
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc

import sys

import pandas as pd
import seaborn as sns

from scipy.stats import entropy

import pdb

import torch
from torchvision import datasets, transforms

def plot_mcd_accuracy():
    fig, ax = plt.subplots()

    labels = ["0", "40", "50", "60", "70", "80", "90", "120", "140", "150", "160", "180"]
    res_pc = [95.30, 47.70, 32.39, 22.05, 17.05, 13.90, 12.05, 15.49, 20.88, 20.88, 22.34, 23.47]
    res_pc_mcd = [94.85, 47.20, 32.99, 22.74, 17.94, 14.92, 13.02, 16.86, 22.15, 22.21, 23.09, 23.57]

    plt.plot(labels, res_pc, label='PC', marker='^', linewidth=4, markersize=12)
    plt.plot(labels, res_pc_mcd, label='PC+MCD', marker='o', linewidth=4, markersize=12)

    ax.grid(True)

    plt.ylabel('Test Accuracy')
    plt.xlabel('Rotation (degrees) ')
    ax.set_title('Rotated MNIST')
    ax.legend()

    plt.savefig("./spn_mcd_rotating_mnist.png")



def plot_mcd_accuracy_vs_confidence(accuracies, confidences, filename, fig_title=None, labels=None, xlabel=None):
    fig, ax = plt.subplots()

    #labels = ["-90", "-60", "-30", "0", "30", "60", "90"]
    if labels is None:
        labels = ["-180", "-150", "-120", "-90", "-60", "-30", "0", "30", "60", "90", "120", "150"]

    if xlabel is None:
        ax.set_xlabel('Rotation (degrees) ')
    else:
        ax.set_xlabel(xlabel)


    #ax.plot(np.array(accuracies[0]), np.array(confidences[0])*100, label='PC', marker='o', linewidth=4, markersize=12)
    #plt.plot(accuracies[1], confidences[1], label='PC+MCD', marker='o', linewidth=4, markersize=12)

    #ax.plot(labels, accuracies[0], label='PC', marker='^', linewidth=4, markersize=12)
    #
    # labels = np.delete(labels, -1, None)
    # confidences[0] = confidences[0][:-1]
    # confidences[1] = confidences[1][:-1]
    # accuracies[0] = accuracies[0][:-1]
    # accuracies[1] = accuracies[1][:-1]

    ax.set_ylabel('Classification confidence')
    plt.ylim((-0.1, 1.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    # print(confidences[0])
    # print(confidences[1])
    # ax.plot(labels, confidences[0], label='PC (Confidence)', marker='o', linewidth=4, markersize=12, color='mediumaquamarine')
    # ax.plot(labels, confidences[1], label='DC (Confidence)', marker='s', linewidth=4, markersize=12, color='orange')
    # ax.plot(labels, accuracies[0], label='PC (Accuracy)', marker='X', linewidth=3, markersize=10, color='royalblue', linestyle='--' mfc='mediumaquamarine')
    # ax.plot(labels, accuracies[1], label='DC (Accuracy)', marker='P', linewidth=3, markersize=10, color='royalblue', linestyle='--', mfc='orange')
    ax.plot(labels, confidences[0], label='PC (Confidence)', marker='o', markersize=12, color='mediumaquamarine', linestyle='None')
    ax.plot(labels, confidences[1], label='DC (Confidence)', marker='s', markersize=12, color='orange', linestyle='None')
    ax.plot(labels, accuracies[0], label='PC (Accuracy)', marker='X', markersize=10, color='royalblue', linestyle='None', mfc='mediumaquamarine')
    ax.plot(labels, accuracies[1], label='DC (Accuracy)', marker='P', markersize=10, color='royalblue', linestyle='None', mfc='orange')

    ax2 = ax.twinx()
    ax2.set_ylabel('Classification accuracy', color='royalblue', fontweight='bold')
    ax2.set_yticks([])
    # ax2.set_yticks(np.arange(0, 110, 10))
    #ax2.plot(labels, accuracies[1], label='PC+MCD', marker='x', linewidth=2, markersize=8, color='blue')

    ax.grid(True)


    #ax.set_title('Rotated MNIST')
    if fig_title is None:
        ax.set_title('Rotated MNIST')
    else:
        ax.set_title(fig_title)
    ax.legend()

    leg = ax.legend()

    for te in leg.get_texts():
        te.set_fontweight("roman")
        if te.get_text() == 'PC (Accuracy)' or te.get_text() == 'DC (Accuracy)':
            te.set_color("royalblue")

    plt.savefig("./{}.png".format(filename))
    plt.savefig("./{}.pdf".format(filename))
    plt.close()

def plot_mcd_accuracy_vs_confidence_two_labels(accuracies, confidences, stds, filename, fig_title=None):
    fig, ax = plt.subplots()

    #labels = ["-90", "-60", "-30", "0", "30", "60", "90"]
    labels = ["-180", "-150", "-120", "-90", "-60", "-30", "0", "30", "60", "90", "120", "150"]


    #ax.plot(np.array(accuracies[0]), np.array(confidences[0])*100, label='PC', marker='o', linewidth=4, markersize=12)
    #plt.plot(accuracies[1], confidences[1], label='PC+MCD', marker='o', linewidth=4, markersize=12)

    #ax.plot(labels, accuracies[0], label='PC', marker='^', linewidth=4, markersize=12)

    ax.set_xlabel('Rotation (degrees) ')
    ax.set_ylabel('Classification confidence')
    plt.ylim((-0.1, 1.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    # print(confidences[0])
    # print(confidences[1])
    # ax.plot(labels, accuracies[0], label='PC (Accuracy)', marker='X', linewidth=3, markersize=10, color='royalblue', linestyle='--', mfc='mediumaquamarine')
    ax.plot(labels, accuracies, label='Accuracy', marker='P', linewidth=3, markersize=10, color='royalblue', linestyle='--', mfc='orange')

    # line_cols = ['skyblue', 'plum', 'lightgreen']
    line_cols = ['skyblue', 'plum']

    for idx, (k, v) in enumerate(confidences.items()):
        if idx > 1: continue
        ax.plot(labels, v, label='Confidence label {}'.format(k), marker='s', linewidth=4, markersize=12, color=line_cols[idx])
        v = np.array(v)
        std = np.array(stds[k])
        ax.fill_between(labels, v-std, v+std, alpha=.3, color=line_cols[idx])

    # ax.plot(labels, confidences[0], label='PC (Confidence)', marker='o', linewidth=4, markersize=12, color='mediumaquamarine')


    ax2 = ax.twinx()
    ax2.set_ylabel('Classification accuracy', color='royalblue', fontweight='bold')
    ax2.set_yticks([])
    # ax2.set_yticks(np.arange(0, 110, 10))
    #ax2.plot(labels, accuracies[1], label='PC+MCD', marker='x', linewidth=2, markersize=8, color='blue')

    ax.grid(True)


    #ax.set_title('Rotated MNIST')
    if fig_title is None:
        ax.set_title('Rotated MNIST')
    else:
        ax.set_title(fig_title)
    ax.legend()

    leg = ax.legend()

    for te in leg.get_texts():
        te.set_fontweight("roman")
        if te.get_text() == 'PC (Accuracy)' or te.get_text() == 'DC (Accuracy)':
            te.set_color("royalblue")

    plt.savefig("./{}.png".format(filename))
    plt.savefig("./{}.pdf".format(filename))
    plt.close()

def plot_mcd_accuracy_vs_confidence_many_labels(accuracies, confidences, stds, filename, fig_title=None):
    fig, ax = plt.subplots()

    #labels = ["-90", "-60", "-30", "0", "30", "60", "90"]
    labels = ["-180", "-150", "-120", "-90", "-60", "-30", "0", "30", "60", "90", "120", "150"]


    #ax.plot(np.array(accuracies[0]), np.array(confidences[0])*100, label='PC', marker='o', linewidth=4, markersize=12)
    #plt.plot(accuracies[1], confidences[1], label='PC+MCD', marker='o', linewidth=4, markersize=12)

    #ax.plot(labels, accuracies[0], label='PC', marker='^', linewidth=4, markersize=12)

    ax.set_xlabel('Rotation (degrees) ')
    ax.set_ylabel('Classification confidence')
    plt.ylim((-0.1, 1.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    # print(confidences[0])
    # print(confidences[1])
    # ax.plot(labels, accuracies[0], label='PC (Accuracy)', marker='X', linewidth=3, markersize=10, color='royalblue', linestyle='--', mfc='mediumaquamarine')
    ax.plot(labels, accuracies, label='Accuracy', marker='P', linewidth=3, markersize=10, color='royalblue', linestyle='--', mfc='orange')

    line_cols = ['skyblue', 'plum', 'lightgreen']

    for idx, (k, v) in enumerate(confidences.items()):
        ax.plot(labels, v, label='Confidence label {}'.format(k), marker='s', linewidth=4, markersize=12, color=line_cols[idx])
        v = np.array(v)
        std = np.array(stds[k])
        ax.fill_between(labels, v-std, v+std, alpha=.3, color=line_cols[idx])

    # ax.plot(labels, confidences[0], label='PC (Confidence)', marker='o', linewidth=4, markersize=12, color='mediumaquamarine')


    ax2 = ax.twinx()
    ax2.set_ylabel('Classification accuracy', color='royalblue', fontweight='bold')
    ax2.set_yticks([])
    # ax2.set_yticks(np.arange(0, 110, 10))
    #ax2.plot(labels, accuracies[1], label='PC+MCD', marker='x', linewidth=2, markersize=8, color='blue')

    ax.grid(True)


    #ax.set_title('Rotated MNIST')
    if fig_title is None:
        ax.set_title('Rotated MNIST')
    else:
        ax.set_title(fig_title)
    ax.legend()

    leg = ax.legend()

    for te in leg.get_texts():
        te.set_fontweight("roman")
        if te.get_text() == 'PC (Accuracy)' or te.get_text() == 'DC (Accuracy)':
            te.set_color("royalblue")

    plt.savefig("./{}.png".format(filename))
    plt.savefig("./{}.pdf".format(filename))
    plt.close()



def plot_accuracy_vs_confidence_line(data, data2, filename):
    fig, ax = plt.subplots()

    labels = ["-90", "-60", "-30", "0", "30", "60", "90"]

    ax.set_xlabel('Classification Accuracy')
    ax.set_ylabel('Classification Confidence')
    ax.plot(data[:,0], data[:,1], label='Probabilistic Circuit', marker='o', linewidth=3, markersize=8, color='mediumaquamarine')
    ax.plot(data2[:,0], data2[:,1],  label='Dropout Circuit', marker='s', linewidth=3, markersize=8, color='orange')

    ax.grid(True)

    ax.set_title('Rotated MNIST')
    ax.legend()

    plt.savefig("./{}.png".format(filename))
    plt.savefig("./{}.pdf".format(filename))
    plt.close()


def plot_histograms(lls_dict, filename='histogram', title="", path=None, trained_on_fmnist=False, y_lim=None):

    plt.hist(lls_dict["mnist_train"], label="In-domain (Train)", alpha=0.5, bins=20, color='green')
    plt.hist(lls_dict["mnist_test"], label="In-domain (Test)", alpha=0.5, bins=20, color='blue')
    plt.hist(lls_dict["other_mnist_train"], label="OOD (Train)", alpha=0.5, bins=20, color='orange')
    plt.hist(lls_dict["other_mnist_test"], label="OOD (Test)", alpha=0.5, bins=20, color='red')

    plt.ylabel('# samples')
    plt.xlabel('Classification confidence')

    plt.legend(loc=0)
    plt.title("{}".format(title))

    if y_lim:
	    plt.ylim((None, y_lim))

    plt.savefig(path + filename + '.png')
    plt.savefig(path + filename + '.pdf')
    plt.close()

def plot_boxplot_rotating_digits(data, filename='boxplot_rotating_digits', title="", path=None, ylimits=[-0.1, 1.1], xlabel='Rotation (degrees)', ylabel='Classification confidence'):
	fig, axs = plt.subplots()
	axs.boxplot(data, showmeans=True, meanline=True, labels=['-90', '-60', '-30', '0', '30', '60', '90'], showfliers=False)
	ax = plt.gca()
	ax.set_ylim(ylimits)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	plt.title(title)
	plt.savefig(path + filename + '_.png')
	plt.savefig(path + filename + '_.pdf')
	plt.close()


def plot_multiple_boxplots_rotatiing_digits(d_results):
	drop_class_probs_90 = np.load(d_results + 'dropout_class_probs_90.out.npy')
	drop_class_probs_60 = np.load(d_results + 'dropout_class_probs_60.out.npy')
	drop_class_probs_30 = np.load(d_results + 'dropout_class_probs_30.out.npy')
	drop_class_probs_0 = np.load(d_results + 'dropout_class_probs_0.out.npy')
	drop_class_probs_330 = np.load(d_results + 'dropout_class_probs_330.out.npy')
	drop_class_probs_300 = np.load(d_results + 'dropout_class_probs_300.out.npy')
	drop_class_probs_270 = np.load(d_results + 'dropout_class_probs_270.out.npy')

	class_probs_90 = np.load(d_results + 'class_probs_90.out.npy')
	class_probs_60 = np.load(d_results + 'class_probs_60.out.npy')
	class_probs_30 = np.load(d_results + 'class_probs_30.out.npy')
	class_probs_0 = np.load(d_results + 'class_probs_0.out.npy')
	class_probs_330 = np.load(d_results + 'class_probs_330.out.npy')
	class_probs_300 = np.load(d_results + 'class_probs_300.out.npy')
	class_probs_270 = np.load(d_results + 'class_probs_270.out.npy')

	drop_entropy_class_probs_90 = entropy(drop_class_probs_90.mean(axis=2), axis=1)
	drop_entropy_class_probs_60 = entropy(drop_class_probs_60.mean(axis=2), axis=1)
	drop_entropy_class_probs_30 = entropy(drop_class_probs_30.mean(axis=2), axis=1)
	drop_entropy_class_probs_0 = entropy(drop_class_probs_0.mean(axis=2), axis=1)
	drop_entropy_class_probs_330 = entropy(drop_class_probs_330.mean(axis=2), axis=1)
	drop_entropy_class_probs_300 = entropy(drop_class_probs_300.mean(axis=2), axis=1)
	drop_entropy_class_probs_270 = entropy(drop_class_probs_270.mean(axis=2), axis=1)

	entropy_class_probs_90 = entropy(class_probs_90, axis=1)
	entropy_class_probs_60 = entropy(class_probs_60, axis=1)
	entropy_class_probs_30 = entropy(class_probs_30, axis=1)
	entropy_class_probs_0 = entropy(class_probs_0, axis=1)
	entropy_class_probs_330 = entropy(class_probs_330, axis=1)
	entropy_class_probs_300 = entropy(class_probs_300, axis=1)
	entropy_class_probs_270 = entropy(class_probs_270, axis=1)

	drop_max_class_probs_90 = drop_class_probs_90.mean(axis=2).max(axis=1)
	drop_max_class_probs_60 = drop_class_probs_60.mean(axis=2).max(axis=1)
	drop_max_class_probs_30 = drop_class_probs_30.mean(axis=2).max(axis=1)
	drop_max_class_probs_0 = drop_class_probs_0.mean(axis=2).max(axis=1)
	drop_max_class_probs_330 = drop_class_probs_330.mean(axis=2).max(axis=1)
	drop_max_class_probs_300 = drop_class_probs_300.mean(axis=2).max(axis=1)
	drop_max_class_probs_270 = drop_class_probs_270.mean(axis=2).max(axis=1)

	max_class_probs_90 = class_probs_90.max(axis=1)
	max_class_probs_60 = class_probs_60.max(axis=1)
	max_class_probs_30 = class_probs_30.max(axis=1)
	max_class_probs_0 = class_probs_0.max(axis=1)
	max_class_probs_330 = class_probs_330.max(axis=1)
	max_class_probs_300 = class_probs_300.max(axis=1)
	max_class_probs_270 = class_probs_270.max(axis=1)

	boxplot_data = np.column_stack((max_class_probs_90, max_class_probs_60, max_class_probs_30, max_class_probs_0, max_class_probs_330, max_class_probs_300, max_class_probs_270))
	drop_boxplot_data = np.column_stack((drop_max_class_probs_90, drop_max_class_probs_60, drop_max_class_probs_30, drop_max_class_probs_0, drop_max_class_probs_330, drop_max_class_probs_300, drop_max_class_probs_270))
	entropy_boxplot_data = np.column_stack((entropy_class_probs_90, entropy_class_probs_60, entropy_class_probs_30, entropy_class_probs_0, entropy_class_probs_330, entropy_class_probs_300, entropy_class_probs_270))
	entropy_drop_boxplot_data = np.column_stack((drop_entropy_class_probs_90, drop_entropy_class_probs_60, drop_entropy_class_probs_30, drop_entropy_class_probs_0, drop_entropy_class_probs_330, drop_entropy_class_probs_300, drop_entropy_class_probs_270))

    # gen 4 boxplots in 1 rows
	fig, axes = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=False, figsize=(14,4), tight_layout=True)

	labels_rotation = ['-90°', '-60°', '-30°', '0°', '30°', '60°', '90°']
	medianprops = dict(linestyle='-.', linewidth=2, color='green')
	meanlineprops = dict(linestyle='--', linewidth=1, color='purple')
	meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='indianred')

	bp1 = axes[0].boxplot(boxplot_data, medianprops=medianprops, meanprops=meanpointprops, showmeans=True, meanline=False, labels=labels_rotation, showfliers=False, patch_artist=True)
	bp2 = axes[1].boxplot(drop_boxplot_data, medianprops=medianprops, meanprops=meanpointprops, showmeans=True, meanline=False, labels=labels_rotation, showfliers=False, patch_artist=True)
	bp3 = axes[2].boxplot(entropy_boxplot_data, medianprops=medianprops, meanprops=meanpointprops, showmeans=True, meanline=False, labels=labels_rotation, showfliers=False, patch_artist=True)
	bp4 = axes[3].boxplot(entropy_drop_boxplot_data, medianprops=medianprops, meanprops=meanpointprops, showmeans=True, meanline=False, labels=labels_rotation, showfliers=False, patch_artist=True)
    
	axes[0].set_title('Probabilistic Circuits')
	axes[1].set_title('Dropout Circuits')
	axes[2].set_title('Probabilistic Circuits')
	axes[3].set_title('Dropout Circuits')

	plt.setp(bp1['boxes'], facecolor='linen')
	plt.setp(bp2['boxes'], facecolor='linen')
	plt.setp(bp3['boxes'], facecolor='lightcyan')
	plt.setp(bp4['boxes'], facecolor='lightcyan')


	axes[0].set_ylabel('classification confidence', fontsize='large', fontweight='bold')
	axes[0].set_ylim(0, 1.1)
	axes[1].set_ylim(0, 1.1)
	#axes[1].get_yaxis().set_ticks([])
	axes[2].set_ylabel('classification entropy', fontsize='large', fontweight='bold')
	#axes[3].get_yaxis().set_ticks([])
	axes[2].set_ylim(-0.1, 3)
	axes[3].set_ylim(-0.1, 3)
	
	axes[0].yaxis.grid(True, linestyle='--')
	axes[1].yaxis.grid(True, linestyle='--')
	axes[2].yaxis.grid(True, linestyle='--')
	axes[3].yaxis.grid(True, linestyle='--')

	ax = plt.gca()
	ax.set_xlabel(' ')
	fig.text(0.5,0.03, "digit rotation (degrees)", ha="center", va="center", fontsize='large')
	fig.suptitle('Rotating MNIST', fontsize='large', fontweight='bold')
	plt.savefig('rotating_mnist_all_boxplots.png')
	plt.savefig('rotating_mnist_all_boxplots.pdf')
	plt.close()

def gen_boxplots(d_results):
	drop_class_probs_90 = np.load(d_results + 'dropout_class_probs_90.out.npy')
	drop_class_probs_60 = np.load(d_results + 'dropout_class_probs_60.out.npy')
	drop_class_probs_30 = np.load(d_results + 'dropout_class_probs_30.out.npy')
	drop_class_probs_0 = np.load(d_results + 'dropout_class_probs_0.out.npy')
	drop_class_probs_330 = np.load(d_results + 'dropout_class_probs_330.out.npy')
	drop_class_probs_300 = np.load(d_results + 'dropout_class_probs_300.out.npy')
	drop_class_probs_270 = np.load(d_results + 'dropout_class_probs_270.out.npy')

	class_probs_90 = np.load(d_results + 'class_probs_90.out.npy')
	class_probs_60 = np.load(d_results + 'class_probs_60.out.npy')
	class_probs_30 = np.load(d_results + 'class_probs_30.out.npy')
	class_probs_0 = np.load(d_results + 'class_probs_0.out.npy')
	class_probs_330 = np.load(d_results + 'class_probs_330.out.npy')
	class_probs_300 = np.load(d_results + 'class_probs_300.out.npy')
	class_probs_270 = np.load(d_results + 'class_probs_270.out.npy')

	drop_max_class_probs_90 = drop_class_probs_90.mean(axis=2).max(axis=1)
	drop_max_class_probs_60 = drop_class_probs_60.mean(axis=2).max(axis=1)
	drop_max_class_probs_30 = drop_class_probs_30.mean(axis=2).max(axis=1)
	drop_max_class_probs_0 = drop_class_probs_0.mean(axis=2).max(axis=1)
	drop_max_class_probs_330 = drop_class_probs_330.mean(axis=2).max(axis=1)
	drop_max_class_probs_300 = drop_class_probs_300.mean(axis=2).max(axis=1)
	drop_max_class_probs_270 = drop_class_probs_270.mean(axis=2).max(axis=1)

	max_class_probs_90 = class_probs_90.max(axis=1)
	max_class_probs_60 = class_probs_60.max(axis=1)
	max_class_probs_30 = class_probs_30.max(axis=1)
	max_class_probs_0 = class_probs_0.max(axis=1)
	max_class_probs_330 = class_probs_330.max(axis=1)
	max_class_probs_300 = class_probs_300.max(axis=1)
	max_class_probs_270 = class_probs_270.max(axis=1)

	boxplot_data = np.column_stack((max_class_probs_90, max_class_probs_60, max_class_probs_30, max_class_probs_0, max_class_probs_330, max_class_probs_300, max_class_probs_270))
	drop_boxplot_data = np.column_stack((drop_max_class_probs_90, drop_max_class_probs_60, drop_max_class_probs_30, drop_max_class_probs_0, drop_max_class_probs_330, drop_max_class_probs_300, drop_max_class_probs_270))

	plot_boxplot_rotating_digits(boxplot_data, filename='boxplot_rotating_digits', title='Rotating MNIST - Probabilistic Circuits', path='./')
	plot_boxplot_rotating_digits(drop_boxplot_data, filename='boxplot_rotating_digits_MCD', title='Rotating MNIST - Dropout Circuits', path='./')

def plot_multiple_boxplots_mnist_c(d_results, sort_idxs=None):
	drop_class_probs_c1 = np.load(d_results + 'dropout_class_probs_c1.npy')
	drop_class_probs_c2 = np.load(d_results + 'dropout_class_probs_c2.npy')
	drop_class_probs_c3 = np.load(d_results + 'dropout_class_probs_c3.npy')
	drop_class_probs_c4 = np.load(d_results + 'dropout_class_probs_c4.npy')
	drop_class_probs_c5 = np.load(d_results + 'dropout_class_probs_c5.npy')
	drop_class_probs_c6 = np.load(d_results + 'dropout_class_probs_c6.npy')
	drop_class_probs_c7 = np.load(d_results + 'dropout_class_probs_c7.npy')
	drop_class_probs_c8 = np.load(d_results + 'dropout_class_probs_c8.npy')
	drop_class_probs_c8 = np.load('/Users/fabrizio/Desktop/2022-05-11_16-12-31/results/dropout_class_probs_c_brightness_l2.npy')
	drop_class_probs_c9 = np.load(d_results + 'dropout_class_probs_c9.npy')
	drop_class_probs_c10 = np.load(d_results + 'dropout_class_probs_c10.npy')
	drop_class_probs_c11 = np.load(d_results + 'dropout_class_probs_c11.npy')
	drop_class_probs_c12 = np.load(d_results + 'dropout_class_probs_c12.npy')
	drop_class_probs_c13 = np.load(d_results + 'dropout_class_probs_c13.npy')
	drop_class_probs_c14 = np.load(d_results + 'dropout_class_probs_c14.npy')
	drop_class_probs_c15 = np.load(d_results + 'dropout_class_probs_c15.npy')

	class_probs_c1 = np.load(d_results + 'class_probs_c1.npy')
	class_probs_c2 = np.load(d_results + 'class_probs_c2.npy')
	class_probs_c3 = np.load(d_results + 'class_probs_c3.npy')
	class_probs_c4 = np.load(d_results + 'class_probs_c4.npy')
	class_probs_c5 = np.load(d_results + 'class_probs_c5.npy')
	class_probs_c6 = np.load(d_results + 'class_probs_c6.npy')
	class_probs_c7 = np.load(d_results + 'class_probs_c7.npy')
	class_probs_c8 = np.load(d_results + 'class_probs_c8.npy')
	class_probs_c8 = np.load('/Users/fabrizio/Desktop/2022-05-11_16-12-31/results/class_probs_c_brightness_l2.npy')
	class_probs_c9 = np.load(d_results + 'class_probs_c9.npy')
	class_probs_c10 = np.load(d_results + 'class_probs_c10.npy')
	class_probs_c11 = np.load(d_results + 'class_probs_c11.npy')
	class_probs_c12 = np.load(d_results + 'class_probs_c12.npy')
	class_probs_c13 = np.load(d_results + 'class_probs_c13.npy')
	class_probs_c14 = np.load(d_results + 'class_probs_c14.npy')
	class_probs_c15 = np.load(d_results + 'class_probs_c15.npy')

	drop_entropy_class_probs_c1 = entropy(drop_class_probs_c1.mean(axis=2), axis=1)
	drop_entropy_class_probs_c2 = entropy(drop_class_probs_c2.mean(axis=2), axis=1)
	drop_entropy_class_probs_c3 = entropy(drop_class_probs_c3.mean(axis=2), axis=1)
	drop_entropy_class_probs_c4 = entropy(drop_class_probs_c4.mean(axis=2), axis=1)
	drop_entropy_class_probs_c5 = entropy(drop_class_probs_c5.mean(axis=2), axis=1)
	drop_entropy_class_probs_c6 = entropy(drop_class_probs_c6.mean(axis=2), axis=1)
	drop_entropy_class_probs_c7 = entropy(drop_class_probs_c7.mean(axis=2), axis=1)
	drop_entropy_class_probs_c8 = entropy(drop_class_probs_c8.mean(axis=2), axis=1)
	drop_entropy_class_probs_c9 = entropy(drop_class_probs_c9.mean(axis=2), axis=1)
	drop_entropy_class_probs_c10 = entropy(drop_class_probs_c10.mean(axis=2), axis=1)
	drop_entropy_class_probs_c11 = entropy(drop_class_probs_c11.mean(axis=2), axis=1)
	drop_entropy_class_probs_c12 = entropy(drop_class_probs_c12.mean(axis=2), axis=1)
	drop_entropy_class_probs_c13 = entropy(drop_class_probs_c13.mean(axis=2), axis=1)
	drop_entropy_class_probs_c14 = entropy(drop_class_probs_c14.mean(axis=2), axis=1)
	drop_entropy_class_probs_c15 = entropy(drop_class_probs_c15.mean(axis=2), axis=1)

	entropy_class_probs_c1 = entropy(class_probs_c1, axis=1)
	entropy_class_probs_c2 = entropy(class_probs_c2, axis=1)
	entropy_class_probs_c3 = entropy(class_probs_c3, axis=1)
	entropy_class_probs_c4 = entropy(class_probs_c4, axis=1)
	entropy_class_probs_c5 = entropy(class_probs_c5, axis=1)
	entropy_class_probs_c6 = entropy(class_probs_c6, axis=1)
	entropy_class_probs_c7 = entropy(class_probs_c7, axis=1)
	entropy_class_probs_c8 = entropy(class_probs_c8, axis=1)
	entropy_class_probs_c9 = entropy(class_probs_c9, axis=1)
	entropy_class_probs_c10 = entropy(class_probs_c10, axis=1)
	entropy_class_probs_c11 = entropy(class_probs_c11, axis=1)
	entropy_class_probs_c12 = entropy(class_probs_c12, axis=1)
	entropy_class_probs_c13 = entropy(class_probs_c13, axis=1)
	entropy_class_probs_c14 = entropy(class_probs_c14, axis=1)
	entropy_class_probs_c15 = entropy(class_probs_c15, axis=1)

	drop_max_class_probs_c1 = drop_class_probs_c1.mean(axis=2).max(axis=1)
	drop_max_class_probs_c2 = drop_class_probs_c2.mean(axis=2).max(axis=1)
	drop_max_class_probs_c3 = drop_class_probs_c3.mean(axis=2).max(axis=1)
	drop_max_class_probs_c4 = drop_class_probs_c4.mean(axis=2).max(axis=1)
	drop_max_class_probs_c5 = drop_class_probs_c5.mean(axis=2).max(axis=1)
	drop_max_class_probs_c6 = drop_class_probs_c6.mean(axis=2).max(axis=1)
	drop_max_class_probs_c7 = drop_class_probs_c7.mean(axis=2).max(axis=1)
	drop_max_class_probs_c8 = drop_class_probs_c8.mean(axis=2).max(axis=1)
	drop_max_class_probs_c9 = drop_class_probs_c9.mean(axis=2).max(axis=1)
	drop_max_class_probs_c10 = drop_class_probs_c10.mean(axis=2).max(axis=1)
	drop_max_class_probs_c11 = drop_class_probs_c11.mean(axis=2).max(axis=1)
	drop_max_class_probs_c12 = drop_class_probs_c12.mean(axis=2).max(axis=1)
	drop_max_class_probs_c13 = drop_class_probs_c13.mean(axis=2).max(axis=1)
	drop_max_class_probs_c14 = drop_class_probs_c14.mean(axis=2).max(axis=1)
	drop_max_class_probs_c15 = drop_class_probs_c15.mean(axis=2).max(axis=1)

	max_class_probs_c1 = class_probs_c1.max(axis=1)
	max_class_probs_c2 = class_probs_c2.max(axis=1)
	max_class_probs_c3 = class_probs_c3.max(axis=1)
	max_class_probs_c4 = class_probs_c4.max(axis=1)
	max_class_probs_c5 = class_probs_c5.max(axis=1)
	max_class_probs_c6 = class_probs_c6.max(axis=1)
	max_class_probs_c7 = class_probs_c7.max(axis=1)
	max_class_probs_c8 = class_probs_c8.max(axis=1)
	max_class_probs_c9 = class_probs_c9.max(axis=1)
	max_class_probs_c10 = class_probs_c10.max(axis=1)
	max_class_probs_c11 = class_probs_c11.max(axis=1)
	max_class_probs_c12 = class_probs_c12.max(axis=1)
	max_class_probs_c13 = class_probs_c13.max(axis=1)
	max_class_probs_c14 = class_probs_c14.max(axis=1)
	max_class_probs_c15 = class_probs_c15.max(axis=1)

	boxplot_data = np.column_stack((max_class_probs_c1, max_class_probs_c2, max_class_probs_c3, max_class_probs_c4, max_class_probs_c5, max_class_probs_c6, max_class_probs_c7, max_class_probs_c8, max_class_probs_c9, max_class_probs_c10, max_class_probs_c11, max_class_probs_c12, max_class_probs_c13, max_class_probs_c14, max_class_probs_c15))
	drop_boxplot_data = np.column_stack((drop_max_class_probs_c1, drop_max_class_probs_c2, drop_max_class_probs_c3, drop_max_class_probs_c4, drop_max_class_probs_c5, drop_max_class_probs_c6, drop_max_class_probs_c7, drop_max_class_probs_c8, drop_max_class_probs_c9, drop_max_class_probs_c10, drop_max_class_probs_c11, drop_max_class_probs_c12, drop_max_class_probs_c13, drop_max_class_probs_c14, drop_max_class_probs_c15))
	entropy_boxplot_data = np.column_stack((entropy_class_probs_c1, entropy_class_probs_c2, entropy_class_probs_c3, entropy_class_probs_c4, entropy_class_probs_c5, entropy_class_probs_c6, entropy_class_probs_c7, entropy_class_probs_c8, entropy_class_probs_c9, entropy_class_probs_c10, entropy_class_probs_c11, entropy_class_probs_c12, entropy_class_probs_c13, entropy_class_probs_c14, entropy_class_probs_c15))
	entropy_drop_boxplot_data = np.column_stack((drop_entropy_class_probs_c1, drop_entropy_class_probs_c2, drop_entropy_class_probs_c3, drop_entropy_class_probs_c4, drop_entropy_class_probs_c5, drop_entropy_class_probs_c6, drop_entropy_class_probs_c7, drop_entropy_class_probs_c8, drop_entropy_class_probs_c9, drop_entropy_class_probs_c10, drop_entropy_class_probs_c11, drop_entropy_class_probs_c12, drop_entropy_class_probs_c13, drop_entropy_class_probs_c14, drop_entropy_class_probs_c15))

    # gen 4 boxplots in 1 rows
	# fig, axes = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=False, figsize=(16,4), tight_layout=True)
	fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False, figsize=(16,8), tight_layout=True)
	labels_rotation = ['SN', 'IN', 'GB', 'MB', 'SH', 'SC', 'RO', 'BR', 'TR', 'ST', 'FO', 'SP', 'DO', 'ZI', 'CA']
	# labels_rotation = ['SN', 'IN', 'GB', 'MB', 'SH', 'SC', 'RO', 'TR', 'ST', 'FO', 'SP', 'DO', 'ZI', 'CA']
	# boxplot_data = np.delete(boxplot_data, 7, 1)
	# drop_boxplot_data = np.delete(drop_boxplot_data, 7, 1)
	# entropy_boxplot_data = np.delete(entropy_boxplot_data, 7, 1)
	# entropy_drop_boxplot_data = np.delete(entropy_drop_boxplot_data, 7, 1)

	if sort_idxs is not None:
		labels_rotation = np.array(labels_rotation)
		labels_rotation = labels_rotation[sort_idxs[::-1]]
		labels_rotation = labels_rotation.tolist()
		boxplot_data = boxplot_data[:, sort_idxs[::-1]]
		drop_boxplot_data = drop_boxplot_data[:, sort_idxs[::-1]]
		entropy_boxplot_data = entropy_boxplot_data[:, sort_idxs[::-1]]
		entropy_drop_boxplot_data = entropy_drop_boxplot_data[:, sort_idxs[::-1]]

	medianprops = dict(linestyle='-.', linewidth=2, color='green')
	meanlineprops = dict(linestyle='--', linewidth=1, color='purple')
	meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='indianred')

	bp1 = axes[0][0].boxplot(boxplot_data, medianprops=medianprops, meanprops=meanpointprops, showmeans=True, meanline=False, labels=labels_rotation, showfliers=False, patch_artist=True)
	bp2 = axes[0][1].boxplot(drop_boxplot_data, medianprops=medianprops, meanprops=meanpointprops, showmeans=True, meanline=False, labels=labels_rotation, showfliers=False, patch_artist=True)
	axes[0][0].get_xaxis().set_ticks([])
	axes[0][1].get_xaxis().set_ticks([])
	bp3 = axes[1][0].boxplot(entropy_boxplot_data, medianprops=medianprops, meanprops=meanpointprops, showmeans=True, meanline=False, labels=labels_rotation, showfliers=False, patch_artist=True)
	bp4 = axes[1][1].boxplot(entropy_drop_boxplot_data, medianprops=medianprops, meanprops=meanpointprops, showmeans=True, meanline=False, labels=labels_rotation, showfliers=False, patch_artist=True)

	axes[0][0].set_title('Probabilistic Circuits')
	axes[0][1].set_title('Dropout Circuits')
	# axes[1][0].set_title('Probabilistic Circuits')
	# axes[1][1].set_title('Dropout Circuits')

	plt.setp(bp1['boxes'], facecolor='linen')
	plt.setp(bp2['boxes'], facecolor='linen')
	plt.setp(bp3['boxes'], facecolor='lightcyan')
	plt.setp(bp4['boxes'], facecolor='lightcyan')


	axes[0][0].set_ylabel('classification confidence', fontsize='large', fontweight='bold')
	axes[0][0].set_ylim(0, 1.1)
	axes[0][1].set_ylim(0, 1.1)
	axes[1][0].set_ylabel('classification entropy', fontsize='large', fontweight='bold')
	#axes[3].get_yaxis().set_ticks([])
	axes[1][0].set_ylim(-0.1, 3)
	axes[1][1].set_ylim(-0.1, 3)

	axes[0][0].yaxis.grid(True, linestyle='--')
	axes[0][1].yaxis.grid(True, linestyle='--')
	axes[1][0].yaxis.grid(True, linestyle='--')
	axes[1][1].yaxis.grid(True, linestyle='--')

	ax = plt.gca()
	ax.set_xlabel(' ')
	fig.text(0.5,0.03, "corruption ID", ha="center", va="center", fontsize='large')
	fig.suptitle('Corrupted MNIST', fontsize='large', fontweight='bold')
	plt.savefig('corrupting_mnist_all_boxplots_tmp.png')
	plt.savefig('corrupting_mnist_all_boxplots_tmp.pdf')
	plt.close()

def plot_multiple_boxplots_svhn_c(d_results, sort_idxs=None):
	from imagecorruptions import corrupt, get_corruption_names

	dc_results, pc_results = {}, {}
	for corruption in get_corruption_names('common'):
		for cl in range(1):
			cl += 1
			print("Load corruption {} Level {}".format(corruption, cl))
			dc_results['c_{}_l{}'.format(corruption, cl)] = np.load(
				d_results + 'dropout_class_probs_c_{}_l{}.npy'.format(corruption, cl))
			pc_results['c_{}_l{}'.format(corruption, cl)] = np.load(
				d_results + 'class_probs_c_{}_l{}.npy'.format(corruption, cl))

	entropy_class_probs_pc = [entropy(class_probs, axis=1) for class_probs in pc_results.values()]
	entropy_class_probs_dc = [entropy(class_probs.mean(axis=2), axis=1) for class_probs in dc_results.values()]
	max_class_probs_pc = [class_probs.max(axis=1) for class_probs in pc_results.values()]
	max_class_probs_dc = [class_probs.mean(axis=2).max(axis=1) for class_probs in dc_results.values()]

	boxplot_data = np.column_stack((max_class_probs_pc))
	print(boxplot_data.shape)
	drop_boxplot_data = np.column_stack((max_class_probs_dc))
	entropy_boxplot_data = np.column_stack((entropy_class_probs_pc))
	entropy_drop_boxplot_data = np.column_stack((entropy_class_probs_dc))

    # gen 4 boxplots in 1 rows
	# fig, axes = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=False, figsize=(16,4), tight_layout=True)
	fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False, figsize=(16,8), tight_layout=True)
	labels_rotation = ['GN', 'SN', 'IN', 'DB', 'GB', 'MB', 'ZB', 'SN', 'FR', 'FO', 'BR', 'CO', 'ET', 'PX', 'JP']
	# labels_rotation = ['SN', 'IN', 'GB', 'MB', 'SH', 'SC', 'RO', 'TR', 'ST', 'FO', 'SP', 'DO', 'ZI', 'CA']
	# boxplot_data = np.delete(boxplot_data, 7, 1)
	# drop_boxplot_data = np.delete(drop_boxplot_data, 7, 1)
	# entropy_boxplot_data = np.delete(entropy_boxplot_data, 7, 1)
	# entropy_drop_boxplot_data = np.delete(entropy_drop_boxplot_data, 7, 1)

	if sort_idxs is not None:
		labels_rotation = np.array(labels_rotation)
		labels_rotation = labels_rotation[sort_idxs[::-1]]
		labels_rotation = labels_rotation.tolist()
		boxplot_data = boxplot_data[:, sort_idxs[::-1]]
		drop_boxplot_data = drop_boxplot_data[:, sort_idxs[::-1]]
		entropy_boxplot_data = entropy_boxplot_data[:, sort_idxs[::-1]]
		entropy_drop_boxplot_data = entropy_drop_boxplot_data[:, sort_idxs[::-1]]

	medianprops = dict(linestyle='-.', linewidth=2, color='green')
	meanlineprops = dict(linestyle='--', linewidth=1, color='purple')
	meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='indianred')

	bp1 = axes[0][0].boxplot(boxplot_data, medianprops=medianprops, meanprops=meanpointprops, showmeans=True, meanline=False, labels=labels_rotation, showfliers=False, patch_artist=True)
	bp2 = axes[0][1].boxplot(drop_boxplot_data, medianprops=medianprops, meanprops=meanpointprops, showmeans=True, meanline=False, labels=labels_rotation, showfliers=False, patch_artist=True)
	axes[0][0].get_xaxis().set_ticks([])
	axes[0][1].get_xaxis().set_ticks([])
	bp3 = axes[1][0].boxplot(entropy_boxplot_data, medianprops=medianprops, meanprops=meanpointprops, showmeans=True, meanline=False, labels=labels_rotation, showfliers=False, patch_artist=True)
	bp4 = axes[1][1].boxplot(entropy_drop_boxplot_data, medianprops=medianprops, meanprops=meanpointprops, showmeans=True, meanline=False, labels=labels_rotation, showfliers=False, patch_artist=True)

	axes[0][0].set_title('Probabilistic Circuits')
	axes[0][1].set_title('Dropout Circuits')
	# axes[1][0].set_title('Probabilistic Circuits')
	# axes[1][1].set_title('Dropout Circuits')

	plt.setp(bp1['boxes'], facecolor='linen')
	plt.setp(bp2['boxes'], facecolor='linen')
	plt.setp(bp3['boxes'], facecolor='lightcyan')
	plt.setp(bp4['boxes'], facecolor='lightcyan')


	axes[0][0].set_ylabel('classification confidence', fontsize='large', fontweight='bold')
	axes[0][0].set_ylim(0, 1.1)
	axes[0][1].set_ylim(0, 1.1)
	axes[1][0].set_ylabel('classification entropy', fontsize='large', fontweight='bold')
	#axes[3].get_yaxis().set_ticks([])
	axes[1][0].set_ylim(-0.1, 3)
	axes[1][1].set_ylim(-0.1, 3)

	axes[0][0].yaxis.grid(True, linestyle='--')
	axes[0][1].yaxis.grid(True, linestyle='--')
	axes[1][0].yaxis.grid(True, linestyle='--')
	axes[1][1].yaxis.grid(True, linestyle='--')

	ax = plt.gca()
	ax.set_xlabel(' ')
	fig.text(0.5,0.03, "corruption ID", ha="center", va="center", fontsize='large')
	fig.suptitle('Corrupted SVHN', fontsize='large', fontweight='bold')
	plt.savefig('corrupting_svhn_all_boxplots_tmp.png')
	plt.savefig('corrupting_svhn_all_boxplots_tmp.pdf')
	plt.close()

def gen_histograms():
    class_probs_in_domain_train = np.load('/Users/fabrizio/Documents/2022-03-25_11-38-35/results/class_probs_in_domain_train.npy')
    class_probs_in_domain_test = np.load('/Users/fabrizio/Documents/2022-03-25_11-38-35/results/class_probs_in_domain_test.npy')
    class_probs_ood_train = np.load('/Users/fabrizio/Documents/2022-03-25_11-38-35/results/class_probs_ood_train.npy')
    class_probs_ood_test = np.load('/Users/fabrizio/Documents/2022-03-25_11-38-35/results/class_probs_ood_test.npy')

    # class_probs_in_domain_train = np.load('/Users/fabrizio/Desktop/2022-04-28_18-57-33_kmnist/results/class_probs_in_domain_train.npy')
    # class_probs_in_domain_test = np.load('/Users/fabrizio/Desktop/2022-04-28_18-57-33_kmnist/results/class_probs_in_domain_test.npy')
    # class_probs_ood_train = np.load('/Users/fabrizio/Desktop/2022-04-28_18-57-33_kmnist/results/class_probs_ood_train.npy')
    # class_probs_ood_test = np.load('/Users/fabrizio/Desktop/2022-04-28_18-57-33_kmnist/results/class_probs_ood_test.npy')

    class_probs_in_domain_test = np.repeat(class_probs_in_domain_test, 6, axis=0)
    class_probs_ood_test = np.repeat(class_probs_ood_test, 6, axis=0)
    class_probs_dict = {'mnist_train':class_probs_in_domain_train.max(axis=1), 'mnist_test':class_probs_in_domain_test.max(axis=1) , 'other_mnist_train':class_probs_ood_train.max(axis=1), 'other_mnist_test':class_probs_ood_test.max(axis=1)}


    class_probs_in_domain_train_drop = np.load('/Users/fabrizio/Documents/2022-03-25_11-38-35/results/class_probs_in_domain_train_dropout.npy')
    class_probs_in_domain_test_drop = np.load('/Users/fabrizio/Documents/2022-03-25_11-38-35/results/class_probs_in_domain_test_dropout.npy')
    class_probs_ood_train_drop = np.load('/Users/fabrizio/Documents/2022-03-25_11-38-35/results/class_probs_ood_train_dropout.npy')
    class_probs_ood_test_drop = np.load('/Users/fabrizio/Documents/2022-03-25_11-38-35/results/class_probs_ood_test_dropout.npy')

    # class_probs_in_domain_train_drop = np.load('/Users/fabrizio/Desktop/2022-04-28_18-57-33_kmnist/results/class_probs_in_domain_train_dropout.npy')
    # class_probs_in_domain_test_drop = np.load('/Users/fabrizio/Desktop/2022-04-28_18-57-33_kmnist/results/class_probs_in_domain_test_dropout.npy')
    # class_probs_ood_train_drop = np.load('/Users/fabrizio/Desktop/2022-04-28_18-57-33_kmnist/results/class_probs_ood_train_dropout.npy')
    # class_probs_ood_test_drop = np.load('/Users/fabrizio/Desktop/2022-04-28_18-57-33_kmnist/results/class_probs_ood_test_dropout.npy')


    class_probs_in_domain_test_drop = np.repeat(class_probs_in_domain_test_drop, 6, axis=0)
    class_probs_ood_test_drop = np.repeat(class_probs_ood_test_drop, 6, axis=0)
    dropout_class_probs_dict = {'mnist_train':class_probs_in_domain_train_drop.max(axis=1), 'mnist_test':class_probs_in_domain_test_drop.max(axis=1), 'other_mnist_train':class_probs_ood_train_drop.max(axis=1), 'other_mnist_test':class_probs_ood_test_drop.max(axis=1)}



    # gen 4 histograms in 2 rows
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(14,5), tight_layout=True)
    bins = 20
    density = True

    # weights = np.ones_like(class_probs_dict["mnist_train"], dtype=np.float64) / (float(len(class_probs_dict["mnist_train"])))
    # print(weights)
    axes[0][0].hist(class_probs_dict["mnist_train"], label="In-domain (Train)", alpha=0.5, bins=bins, color='green', density=density)
    weights = np.ones_like(class_probs_dict["mnist_test"]) / (len(class_probs_dict["mnist_test"]))
    axes[0][0].hist(class_probs_dict["mnist_test"], label="In-domain (Test)", alpha=0.5, bins=bins, color='blue', density=density)
    # axes[0][0].hist(class_probs_dict["other_mnist_train"], label="OOD (Train)", alpha=0.5, bins=bins, color='orange', density=density)
    # weights = np.ones_like(class_probs_dict["other_mnist_test"]) / (len(class_probs_dict["other_mnist_test"]))
    axes[0][0].hist(class_probs_dict["other_mnist_test"], label="OOD (Test)", alpha=0.5, bins=bins, color='red', density=density)
    axes[0][0].set_title('Probabilistic Circuits')
    #axes[0][0].set_ylabel('# samples')
    axes[0][0].grid(True)

    # weights = np.ones_like(dropout_class_probs_dict["mnist_train"]) / (len(dropout_class_probs_dict["mnist_train"]))
    axes[0][1].hist(dropout_class_probs_dict["mnist_train"], label="In-domain (Train)", alpha=0.5, bins=bins, color='green', density=density)
    # weights = np.ones_like(dropout_class_probs_dict["mnist_test"]) / (len(dropout_class_probs_dict["mnist_test"]))
    axes[0][1].hist(dropout_class_probs_dict["mnist_test"], label="In-domain (Test)", alpha=0.5, bins=bins, color='blue', density=density)
    # axes[0][1].hist(dropout_class_probs_dict["other_mnist_train"], label="OOD (Train)", alpha=0.5, bins=bins, color='orange', density=True)
    # weights = np.ones_like(dropout_class_probs_dict["other_mnist_test"]) / (len(dropout_class_probs_dict["other_mnist_test"]))
    axes[0][1].hist(dropout_class_probs_dict["other_mnist_test"], label="OOD (Test)", alpha=0.5, bins=bins, color='red', density=density)
    axes[0][1].set_title('Dropout Circuits')
    axes[0][1].grid(True)
    axes[0][1].legend(loc=0)
    #ax2 = axes[0][1].twinx()
    #ax2.get_yaxis().set_visible(False)
    #ax2.get_yaxis().set_ticks([])
    #ax2.set_ylabel('Trained on MNIST', color='slategrey', fontweight='bold')

    ax2 = axes[0][1].secondary_yaxis('right')
    ax2.set_ylabel('Trained on MNIST', color='slategrey', fontweight='bold', fontsize='large')
    # ax2.set_ylabel('Trained on K-MNIST', color='slategrey', fontweight='bold', fontsize='large')
    ax2.get_yaxis().set_ticks([])

    #fig.set_xlabel('Classification confidence')

    ## Get extents of subplot
    #x0 = min([ax.get_position().x0 for ax in axes])
    #y0 = min([ax.get_position().y0 for ax in axes])
    #x1 = max([ax.get_position().x1 for ax in axes])
    #y1 = max([ax.get_position().y1 for ax in axes])

    ## Hidden axes for common x and y labels
    #plt.axes([x0, y0, x1 - x0, y1 - y0], frameon=False)
    #plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    ## Labelize
    #plt.xlabel(" ", labelpad=None)
    #plt.ylabel("common Y")
    #plt.title('Common Title')

    #plt.legend(loc=0)
    #plt.grid(True)

    #fig.suptitle("Training on MNIST")
    #fig.text(0.5,0.05, "classification confidence", ha="center", va="center", fontsize='large')


    #if y_lim:
    # plt.ylim((None, 60000))

    #plt.savefig('class_confidence_hist_mnist.png')
    #plt.savefig('class_confidence_hist_mnist.pdf')
    #plt.close()

    #train_on_fmnist = False
    #plot_histograms(class_probs_dict, filename='class_probs_histograms_mnist', title="Probabilistic Circuits", path='./', trained_on_fmnist=train_on_fmnist, y_lim=60000)
    #plot_histograms(dropout_class_probs_dict, filename='dropout_class_probs_histograms_mnist', title="Dropout Circuits", path='./', trained_on_fmnist=train_on_fmnist, y_lim=60000)

    class_probs_in_domain_train = np.load('/Users/fabrizio/Documents/2022-03-25_19-00-15/results/class_probs_in_domain_train.npy')
    class_probs_in_domain_test = np.load('/Users/fabrizio/Documents/2022-03-25_19-00-15/results/class_probs_in_domain_test.npy')
    class_probs_ood_train = np.load('/Users/fabrizio/Documents/2022-03-25_19-00-15/results/class_probs_ood_train.npy')
    class_probs_ood_test = np.load('/Users/fabrizio/Documents/2022-03-25_19-00-15/results/class_probs_ood_test.npy')

    # class_probs_in_domain_train = np.load('/Users/fabrizio/Desktop/2022-04-29_08-10-04_emnist/results/class_probs_in_domain_train.npy')
    # class_probs_in_domain_test = np.load('/Users/fabrizio/Desktop/2022-04-29_08-10-04_emnist/results/class_probs_in_domain_test.npy')
    # class_probs_ood_train = np.load('/Users/fabrizio/Desktop/2022-04-29_08-10-04_emnist/results/class_probs_ood_train.npy')
    # class_probs_ood_test = np.load('/Users/fabrizio/Desktop/2022-04-29_08-10-04_emnist/results/class_probs_ood_test.npy')

    class_probs_in_domain_test = np.repeat(class_probs_in_domain_test, 6, axis=0)
    class_probs_ood_test = np.repeat(class_probs_ood_test, 6, axis=0)
    class_probs_dict = {'mnist_train':class_probs_in_domain_train.max(axis=1), 'mnist_test':class_probs_in_domain_test.max(axis=1) , 'other_mnist_train':class_probs_ood_train.max(axis=1), 'other_mnist_test':class_probs_ood_test.max(axis=1)}


    class_probs_in_domain_train_drop = np.load('/Users/fabrizio/Documents/2022-03-25_19-00-15/results/class_probs_in_domain_train_dropout.npy')
    class_probs_in_domain_test_drop = np.load('/Users/fabrizio/Documents/2022-03-25_19-00-15/results/class_probs_in_domain_test_dropout.npy')
    class_probs_ood_train_drop = np.load('/Users/fabrizio/Documents/2022-03-25_19-00-15/results/class_probs_ood_train_dropout.npy')
    class_probs_ood_test_drop = np.load('/Users/fabrizio/Documents/2022-03-25_19-00-15/results/class_probs_ood_test_dropout.npy')

    # class_probs_in_domain_train_drop = np.load('/Users/fabrizio/Desktop/2022-04-29_08-10-04_emnist/results/class_probs_in_domain_train_dropout.npy')
    # class_probs_in_domain_test_drop = np.load('/Users/fabrizio/Desktop/2022-04-29_08-10-04_emnist/results/class_probs_in_domain_test_dropout.npy')
    # class_probs_ood_train_drop = np.load('/Users/fabrizio/Desktop/2022-04-29_08-10-04_emnist/results/class_probs_ood_train_dropout.npy')
    # class_probs_ood_test_drop = np.load('/Users/fabrizio/Desktop/2022-04-29_08-10-04_emnist/results/class_probs_ood_test_dropout.npy')

    class_probs_in_domain_test_drop = np.repeat(class_probs_in_domain_test_drop, 6, axis=0)
    class_probs_ood_test_drop = np.repeat(class_probs_ood_test_drop, 6, axis=0)
    dropout_class_probs_dict = {'mnist_train':class_probs_in_domain_train_drop.max(axis=1), 'mnist_test':class_probs_in_domain_test_drop.max(axis=1), 'other_mnist_train':class_probs_ood_train_drop.max(axis=1), 'other_mnist_test':class_probs_ood_test_drop.max(axis=1)}



    #train_on_fmnist = True
    #plot_histograms(class_probs_dict, filename='class_probs_histograms_fmnist', title="Probabilistic Circuits", path='./', trained_on_fmnist=train_on_fmnist, y_lim=60000)
    #plot_histograms(dropout_class_probs_dict, filename='dropout_class_probs_histograms_fmnist', title="Dropout Circuits", path='./', trained_on_fmnist=train_on_fmnist, y_lim=60000)


    ## gen 2 histograms in one column
    #fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(10,4), tight_layout=True)
    #bins = 20

    axes[1][0].hist(class_probs_dict["mnist_train"], label="In-domain (Train)", alpha=0.5, bins=bins, color='green', density=density)
    axes[1][0].hist(class_probs_dict["mnist_test"], label="In-domain (Test)", alpha=0.5, bins=bins, color='blue', density=density)
    # axes[1][0].hist(class_probs_dict["other_mnist_train"], label="OOD (Train)", alpha=0.5, bins=bins, color='orange', density=density)
    axes[1][0].hist(class_probs_dict["other_mnist_test"], label="OOD (Test)", alpha=0.5, bins=bins, color='red', density=density)
    #axes[1][0].set_title('Probabilistic Circuits')
    #axes[1][0].set_ylabel('# samples')
    #axes[1][0].set_xlabel(' ')
    axes[1][0].set_ylabel(' ')
    axes[1][0].grid(True)

    axes[1][1].hist(dropout_class_probs_dict["mnist_train"], label="In-domain (Train)", alpha=0.5, bins=bins, color='green', density=density)
    axes[1][1].hist(dropout_class_probs_dict["mnist_test"], label="In-domain (Test)", alpha=0.5, bins=bins, color='blue', density=density)
    # axes[1][1].hist(dropout_class_probs_dict["other_mnist_train"], label="OOD (Train)", alpha=0.5, bins=bins, color='orange', density=density)
    axes[1][1].hist(dropout_class_probs_dict["other_mnist_test"], label="OOD (Test)", alpha=0.5, bins=bins, color='red', density=density)
    #axes[1][1].set_title('Dropout Circuits')
    axes[1][1].grid(True)
    ax3 = axes[1][1].secondary_yaxis('right')
    #print(ax3)
    ax3.set_ylabel('Trained on F-MNIST', color='slategrey', fontweight='bold', fontsize='large')
    # ax3.set_ylabel('Trained on EMNIST', color='slategrey', fontweight='bold', fontsize='large')
    ax3.get_yaxis().set_ticks([])
    #ax3.get_yaxis().set_label('Trained on F-MNIST', color='slategrey', fontweight='bold')


    #fig.set_xlabel('Classification confidence')

    ## Get extents of subplot
    #x0 = min([ax.get_position().x0 for ax in axes])
    #y0 = min([ax.get_position().y0 for ax in axes])
    #x1 = max([ax.get_position().x1 for ax in axes])
    #y1 = max([ax.get_position().y1 for ax in axes])

    ## Hidden axes for common x and y labels
    #plt.axes([x0, y0, x1 - x0, y1 - y0], frameon=False)
    #plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    ## Labelize
    plt.xlabel(" ", labelpad=None)
    plt.ylabel(" ", labelpad=None)
    #plt.ylabel("common Y")
    #plt.title('Common Title')

    #plt.grid(True)
    #plt.legend(loc=1)

    #fig.suptitle("Training on F-MNIST")
    fig.text(0.5,0.02, "classification confidence", ha="center", va="center", fontsize='large')
    fig.text(0.01,0.5, "# samples", ha="center", va="center", rotation=90, fontsize='large')


    #if y_lim:
    #plt.ylim((None, 60000))


    plt.savefig('class_confidence_histograms.png')
    plt.savefig('class_confidence_histograms.pdf')
    plt.close()

    class_probs_in_domain_train = np.load('/Users/fabrizio/Documents/2022-03-25_19-00-15/results/class_probs_in_domain_train.npy').max(axis=1)
    class_probs_in_domain_test = np.load('/Users/fabrizio/Documents/2022-03-25_19-00-15/results/class_probs_in_domain_test.npy').max(axis=1)
    class_probs_ood_train = np.load('/Users/fabrizio/Documents/2022-03-25_19-00-15/results/class_probs_ood_train.npy').max(axis=1)
    class_probs_ood_test = np.load('/Users/fabrizio/Documents/2022-03-25_19-00-15/results/class_probs_ood_test.npy').max(axis=1)

    # class_probs_in_domain_test = np.repeat(class_probs_in_domain_test, 6)
    # class_probs_ood_test = np.repeat(class_probs_ood_test, 6)


    df_in_train = pd.DataFrame({'class_probs_in_domain_train':class_probs_in_domain_train})
    df_in_test = pd.DataFrame({'class_probs_in_domain_test':class_probs_in_domain_test})
    # df_ood_train = pd.DataFrame({'class_probs_ood_train':class_probs_ood_train})
    df_ood_test = pd.DataFrame({'class_probs_ood_test':class_probs_ood_test})

    # data_ary = [df_in_train, df_in_test, df_ood_train, df_ood_test]
    #data_ary = [df_in_train, df_ood_train]

    data = pd.concat([df_in_train, df_in_test, df_ood_test], ignore_index=True, axis=1)
    # data = pd.concat([df_in_train, df_in_test, df_ood_train, df_ood_test], ignore_index=True, axis=1)
    # data = data.rename({0:'F-MNIST Train (In-domain)', 1:'F-MNIST Test (In-domain)', 2:'MNIST Train (OOD)', 3:'MNIST Test (OOD)'}, axis=1)
    data = data.rename({0:'F-MNIST Train (In-domain)', 1:'F-MNIST Test (In-domain)', 2:'MNIST Test (OOD)'}, axis=1)

    #data = pd.concat([df_in_train, df_ood_train], ignore_index=True, axis=1)
    #data = data.rename({0:'F-MNIST Train (In-domain)', 1:'MNIST Train (OOD)'}, axis=1)
    #print(data)

    #df_melted = data.melt(var_name='column')

    #p2 = sns.histplot(data=data, bins=10, multiple='layer')
    #fig2 = p1.get_figure()
    #fig2.savefig("confidence_histograms_fmnist.pdf")

    #sns.set(color_codes=True)
    #sns.set(style="white", palette="muted")

    #sns.distplot(data)

    #plt.figure()
    #for col in data:
    #    sns.distplot(col, hist=True)
    #plt.savefig('test_distplot.pdf')

    print(data)

    #sns.color_palette("Paired")
    # palette ={"F-MNIST Train (In-domain)": "green", "F-MNIST Test (In-domain)": "C0", "MNIST Train (OOD)": "yellow", "MNIST Test (OOD)": "red"}
    palette ={"F-MNIST Train (In-domain)": "yellow", "F-MNIST Test (In-domain)": "green", "MNIST Test (OOD)": "blue"}
    # palette ={"F-MNIST Train (In-domain)": (0.011, 0.988, 0.662, 1.), "F-MNIST Test (In-domain)": (0.011, 0.043, 0.988, 0.5), "MNIST Test (OOD)": (0.99, 0., 0., 0.6)}
    pastel = sns.color_palette("pastel", 3)
    # print(my_palette)
    husl = sns.color_palette("husl", 3)
    # print(my_palette)
    #my_palette = [pastel[0], husl[0], husl[1]]
    my_palette = palette
    #p3 = sns.histplot(data=data, x='value', hue='column',  bins=50, multiple='layer', kde=True)
    # p3 = sns.histplot(data=data, multiple='layer', stat="density", palette=palette, cbar_kws={'alpha':0.3})
    p3 = sns.histplot(data=data, stat="probability", element="bars", bins=20, common_norm=False, palette=my_palette)
    # p3 = sns.distplot(a=data, bins=20,  hist_kws={"alpha":0.2, "stat":"probability", "element":"step", "common_norm":False})
    #p3 = sns.distplot(df_melted,  bins=50)
    p3.set(xlabel='Classification confidence', ylabel='rel. perc. of samples')
    # p3.map(plt.hist, alpha=0.5)
    #p3 = sns.histplot(data=data, bins=20, multiple='layer')
    fig3 = p3.get_figure()
    fig3.savefig("confidence_histograms_fmnist_3.pdf")

def gen_lls_histograms(d_results, filename="lls_histograms.pdf"):

    lls_in_domain_train = np.load(d_results + 'train_lls.npy')
    lls_in_domain_test = np.load(d_results + 'test_lls.npy')
    # other_train_lls = np.load(d_results + 'other_mnist_train_lls.npy')
    # other_test_lls = np.load(d_results + 'other_mnist_test_lls.npy')
    other_train_lls = np.load(d_results + 'other_train_lls.npy')
    other_test_lls = np.load(d_results + 'other_test_lls.npy')

    df_in_train = pd.DataFrame({'lls_train':lls_in_domain_train})
    df_in_test = pd.DataFrame({'lls_test':lls_in_domain_test})
    # df_ood_train = pd.DataFrame({'class_probs_ood_train':class_probs_ood_train})
    df_ood_test = pd.DataFrame({'other_test_lls':other_test_lls})

    data = pd.concat([df_in_train, df_in_test, df_ood_test], ignore_index=True, axis=1)
    data = data.rename({0:'F-MNIST Train (In-domain)', 1:'F-MNIST Test (In-domain)', 2:'MNIST Test (OOD)'}, axis=1)
    print(data)

    palette ={"F-MNIST Train (In-domain)": "yellow", "F-MNIST Test (In-domain)": "green", "MNIST Test (OOD)": "blue"}
    my_palette = palette
    #p3 = sns.histplot(data=data, x='value', hue='column',  bins=50, multiple='layer', kde=True)
    # p3 = sns.histplot(data=data, multiple='layer', stat="density", palette=palette, cbar_kws={'alpha':0.3})
    p3 = sns.histplot(data=data, stat="probability", element="bars", common_norm=False, palette=my_palette)
    # p3 = sns.distplot(a=data, bins=20,  hist_kws={"alpha":0.2, "stat":"probability", "element":"step", "common_norm":False})
    #p3 = sns.distplot(df_melted,  bins=50)
    p3.set(xlabel='Data LL', ylabel='rel. perc. of samples')
    # p3.map(plt.hist, alpha=0.5)
    #p3 = sns.histplot(data=data, bins=20, multiple='layer')
    fig3 = p3.get_figure()
    fig3.savefig(filename)
    plt.close()

    lls_in_domain_train = np.load(d_results + 'drop_train_lls.npy')
    lls_in_domain_test = np.load(d_results + 'drop_test_lls.npy')
    # other_train_lls = np.load(d_results + 'drop_other_mnist_train_lls.npy')
    # other_test_lls = np.load(d_results + 'drop_other_mnist_test_lls.npy')
    other_train_lls = np.load(d_results + 'drop_train_lls.npy')
    other_test_lls = np.load(d_results + 'drop_test_lls.npy')

    df_in_train = pd.DataFrame({'lls_train':lls_in_domain_train})
    df_in_test = pd.DataFrame({'lls_test':lls_in_domain_test})
    # df_ood_train = pd.DataFrame({'class_probs_ood_train':class_probs_ood_train})
    df_ood_test = pd.DataFrame({'other_test_lls':other_test_lls})

    data = pd.concat([df_in_train, df_in_test, df_ood_test], ignore_index=True, axis=1)
    data = data.rename({0:'F-MNIST Train (In-domain)', 1:'F-MNIST Test (In-domain)', 2:'MNIST Test (OOD)'}, axis=1)
    p3 = sns.histplot(data=data, stat="probability", element="bars", common_norm=False, palette=my_palette)
    p3.set(xlabel='Data LL', ylabel='rel. perc. of samples')
    fig3 = p3.get_figure()
    fig3.savefig("drop" + filename)
    plt.close()

def gen_plot_conf_vs_acc_auto(d_results, filename="acc_vs_conf_mcd_p_02"):
    mean = 0.1307
    std = 0.3081

    transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean), (std))])
    test_set = datasets.MNIST(root='./data', download=True, train=False, transform=transformer)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False)

    drop_class_probs_180 = np.load(d_results + 'dropout_class_probs_180.npy')
    drop_class_probs_150 = np.load(d_results + 'dropout_class_probs_150.npy')
    drop_class_probs_120 = np.load(d_results + 'dropout_class_probs_120.npy')
    drop_class_probs_90 = np.load(d_results + 'dropout_class_probs_90.npy')
    drop_class_probs_60 = np.load(d_results + 'dropout_class_probs_60.npy')
    drop_class_probs_30 = np.load(d_results + 'dropout_class_probs_30.npy')
    drop_class_probs_0 = np.load(d_results + 'dropout_class_probs_0.npy')
    drop_class_probs_330 = np.load(d_results + 'dropout_class_probs_330.npy')
    drop_class_probs_300 = np.load(d_results + 'dropout_class_probs_300.npy')
    drop_class_probs_270 = np.load(d_results + 'dropout_class_probs_270.npy')
    drop_class_probs_240 = np.load(d_results + 'dropout_class_probs_240.npy')
    drop_class_probs_210 = np.load(d_results + 'dropout_class_probs_210.npy')

    class_probs_180 = np.load(d_results + 'class_probs_180.npy')
    class_probs_150 = np.load(d_results + 'class_probs_150.npy')
    class_probs_120 = np.load(d_results + 'class_probs_120.npy')
    class_probs_90 = np.load(d_results + 'class_probs_90.npy')
    class_probs_60 = np.load(d_results + 'class_probs_60.npy')
    class_probs_30 = np.load(d_results + 'class_probs_30.npy')
    class_probs_0 = np.load(d_results + 'class_probs_0.npy')
    class_probs_330 = np.load(d_results + 'class_probs_330.npy')
    class_probs_300 = np.load(d_results + 'class_probs_300.npy')
    class_probs_270 = np.load(d_results + 'class_probs_270.npy')
    class_probs_240 = np.load(d_results + 'class_probs_240.npy')
    class_probs_210 = np.load(d_results + 'class_probs_210.npy')

    drop_class_probs_dict = {180:drop_class_probs_180, 150:drop_class_probs_150, 120:drop_class_probs_120, 90:drop_class_probs_90, 60:drop_class_probs_60, 30:drop_class_probs_30, 0:drop_class_probs_0, 330:drop_class_probs_330, 300:drop_class_probs_300, 270:drop_class_probs_270, 240:drop_class_probs_240, 210:drop_class_probs_210}
    class_probs_dict = {180:class_probs_180, 150:class_probs_150, 120:class_probs_120, 90:class_probs_90, 60:class_probs_60, 30:class_probs_30, 0:class_probs_0, 330:class_probs_330, 300:class_probs_300, 270:class_probs_270, 240:class_probs_240, 210:class_probs_210}


    drop_max_class_probs_180 = drop_class_probs_180.mean(axis=2).max(axis=1)
    drop_max_class_probs_150 = drop_class_probs_150.mean(axis=2).max(axis=1)
    drop_max_class_probs_120 = drop_class_probs_120.mean(axis=2).max(axis=1)
    drop_max_class_probs_90 = drop_class_probs_90.mean(axis=2).max(axis=1)
    drop_max_class_probs_60 = drop_class_probs_60.mean(axis=2).max(axis=1)
    drop_max_class_probs_30 = drop_class_probs_30.mean(axis=2).max(axis=1)
    drop_max_class_probs_0 = drop_class_probs_0.mean(axis=2).max(axis=1)
    drop_max_class_probs_330 = drop_class_probs_330.mean(axis=2).max(axis=1)
    drop_max_class_probs_300 = drop_class_probs_300.mean(axis=2).max(axis=1)
    drop_max_class_probs_270 = drop_class_probs_270.mean(axis=2).max(axis=1)
    drop_max_class_probs_240 = drop_class_probs_240.mean(axis=2).max(axis=1)
    drop_max_class_probs_210 = drop_class_probs_210.mean(axis=2).max(axis=1)

    max_class_probs_180 = class_probs_180.max(axis=1)
    max_class_probs_150 = class_probs_150.max(axis=1)
    max_class_probs_120 = class_probs_120.max(axis=1)
    max_class_probs_90 = class_probs_90.max(axis=1)
    max_class_probs_60 = class_probs_60.max(axis=1)
    max_class_probs_30 = class_probs_30.max(axis=1)
    max_class_probs_0 = class_probs_0.max(axis=1)
    max_class_probs_330 = class_probs_330.max(axis=1)
    max_class_probs_300 = class_probs_300.max(axis=1)
    max_class_probs_270 = class_probs_270.max(axis=1)
    max_class_probs_240 = class_probs_240.max(axis=1)
    max_class_probs_210 = class_probs_210.max(axis=1)


    pc_test_accuracy = [(test_loader.dataset.targets.numpy()  == res_array.argmax(axis=1)).sum()/test_loader.dataset.targets.numpy().shape[0] for res_array in class_probs_dict.values()]
    mcd_pc_test_accuracy = [(test_loader.dataset.targets.numpy()  == res_array.mean(axis=2).argmax(axis=1)).sum()/test_loader.dataset.targets.numpy().shape[0] for res_array in drop_class_probs_dict.values()]

    mcd_avg_class_confidence = [drop_max_class_probs_180.mean(), drop_max_class_probs_150.mean(), drop_max_class_probs_120.mean(), drop_max_class_probs_90.mean(), drop_max_class_probs_60.mean(), drop_max_class_probs_30.mean(), drop_max_class_probs_0.mean(), drop_max_class_probs_330.mean(), drop_max_class_probs_300.mean(), drop_max_class_probs_270.mean(), drop_max_class_probs_240.mean(), drop_max_class_probs_210.mean()]
    pc_avg_class_confidence = [max_class_probs_180.mean(), max_class_probs_150.mean(), max_class_probs_120.mean(), max_class_probs_90.mean(), max_class_probs_60.mean(), max_class_probs_30.mean(), max_class_probs_0.mean(), max_class_probs_330.mean(), max_class_probs_300.mean(), max_class_probs_270.mean(), max_class_probs_240.mean(), max_class_probs_210.mean()]

    assert len(pc_test_accuracy) == len(mcd_pc_test_accuracy)
    assert len(mcd_avg_class_confidence) == len(pc_avg_class_confidence)
    assert len(pc_test_accuracy) == len(pc_avg_class_confidence)

    plot_mcd_accuracy_vs_confidence([pc_test_accuracy, mcd_pc_test_accuracy], [pc_avg_class_confidence, mcd_avg_class_confidence], filename, fig_title='Rotated MNIST - DC')


    digits = np.arange(10)
    degrees = np.array([180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210])
    #labels = ["-180", "-150", "-120", "-90", "-60", "-30", "0", "30", "60", "90", "120", "150"]


    drop_class_probs_dict = {}
    class_probs_dict = {}

    for n in digits:
        for deg in degrees:
            drop_class_probs_dict[deg] = np.load('{}dropout_class_probs_{}.out_{}.npy'.format(d_results, deg, n))
            class_probs_dict[deg] = np.load('{}class_probs_{}_{}.npy'.format(d_results, deg, n))


        pc_test_accuracy = [(n  == res_array.argmax(axis=1)).sum()/res_array.shape[0] for res_array in class_probs_dict.values()]
        mcd_pc_test_accuracy = [(n  == res_array.mean(axis=2).argmax(axis=1)).sum()/res_array.shape[0] for res_array in drop_class_probs_dict.values()]

        # take always the max label (not the label of the rotated digit)
        mcd_avg_class_confidence = [res_array.mean(axis=2).max(axis=1).mean() for res_array in drop_class_probs_dict.values()]
        pc_avg_class_confidence = [res_array.max(axis=1).mean() for res_array in class_probs_dict.values()]
        #
        # Here consider the confidence over the original true label of the digit
        # mcd_avg_class_confidence = [res_array.mean(axis=2)[:, n].mean() for res_array in drop_class_probs_dict.values()]
        # pc_avg_class_confidence = [res_array[:, n].mean() for res_array in class_probs_dict.values()]

        assert len(pc_test_accuracy) == len(mcd_pc_test_accuracy)
        assert len(mcd_avg_class_confidence) == len(pc_avg_class_confidence)
        assert len(pc_test_accuracy) == len(pc_avg_class_confidence)

        plot_mcd_accuracy_vs_confidence([pc_test_accuracy, mcd_pc_test_accuracy], [pc_avg_class_confidence, mcd_avg_class_confidence], filename+"_{}".format(n), fig_title='Rotated MNIST - DC (Digit {})'.format(n))

        print("\n")
        for k, v in drop_class_probs_dict.items():
            print("digit {} degrees {}".format(n, k))
            print(v.mean(axis=2).mean(axis=0))

        if n == 6:
            confidences = {6:[], 9:[], 4:[]}
            stds = {6:[], 9:[], 4:[]}

            confidences[6] = [res_array.mean(axis=2)[:, n].mean() for res_array in drop_class_probs_dict.values()]
            confidences[9] = [res_array.mean(axis=2)[:, 9].mean() for res_array in drop_class_probs_dict.values()]
            confidences[4] = [res_array.mean(axis=2)[:, 4].mean() for res_array in drop_class_probs_dict.values()]
            stds[6] = [res_array.mean(axis=2)[:, n].std() for res_array in drop_class_probs_dict.values()]
            stds[9] = [res_array.mean(axis=2)[:, 9].std() for res_array in drop_class_probs_dict.values()]
            stds[4] = [res_array.mean(axis=2)[:, 4].std() for res_array in drop_class_probs_dict.values()]
            plot_mcd_accuracy_vs_confidence_many_labels(mcd_pc_test_accuracy, confidences, stds=stds, filename=filename+"_{}".format(n) + "_dropout_circuits_labels", fig_title='Rotated MNIST - DC (Digit {})'.format(n))
            plot_mcd_accuracy_vs_confidence_two_labels(mcd_pc_test_accuracy, confidences, stds=stds, filename=filename+"_two_{}".format(n) + "_dropout_circuits_labels", fig_title='Rotated MNIST - DC (Digit {})'.format(n))

            dc_class_confidences = [drop_max_class_probs_180, drop_max_class_probs_150,
										drop_max_class_probs_120, drop_max_class_probs_90,
										drop_max_class_probs_60, drop_max_class_probs_30,
										drop_max_class_probs_0, drop_max_class_probs_330,
										drop_max_class_probs_300, drop_max_class_probs_270,
										drop_max_class_probs_240, drop_max_class_probs_210]

            dc_class_stds = [drop_class_probs_180, drop_class_probs_150,
									drop_class_probs_120, drop_class_probs_90,
									drop_class_probs_60, drop_class_probs_30,
									drop_class_probs_0, drop_class_probs_330,
									drop_class_probs_300, drop_class_probs_270,
									drop_class_probs_240, drop_class_probs_210]


		    # 3d plot just for label 6
            from matplotlib import cm
            from matplotlib.ticker import LinearLocator

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

            labels = [-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150]
            colors = ['brown', 'red', 'tomato', 'orange', 'yellow', 'lime', 'black', 'lime', 'yellow', 'orange', 'tomato', 'red']
            # breakpoint()

            for idx in range(len(dc_class_confidences)):
                ax.scatter(labels[idx], dc_class_confidences[idx][:1], np.take(dc_class_stds[idx].std(axis=2),
																	  dc_class_stds[idx].mean(axis=2).argmax(axis=1))[:1],
						   c=colors[idx])
                print(labels[idx])
                print(dc_class_confidences[idx][:1])
                print(np.take(dc_class_stds[idx].std(axis=2), dc_class_stds[idx].mean(axis=2).argmax(axis=1))[:1])

            ax.set_xlabel('rotation (degrees)')
            ax.set_ylabel('classification confidence')
            ax.set_zlabel('std')

            fig.savefig("3D_scatter_drop" + filename)
            plt.close()

            fig = plt.figure()
            ax = fig.add_subplot()

            labels = [-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150]
            colors = ['brown', 'red', 'tomato', 'orange', 'yellow', 'lime', 'black', 'lime', 'yellow', 'orange',
					  'tomato', 'red']
			# breakpoint()

            for idx in range(len(dc_class_confidences)):
                ax.scatter([labels[idx]]*dc_class_stds[idx].shape[0], dc_class_stds[idx].mean(axis=2).max(axis=1),
						   c=np.take(dc_class_stds[idx].std(axis=2), dc_class_stds[idx].mean(axis=2).argmax(axis=1)),
						   ec='none',
						   cmap='Blues',
						   alpha=0.5)

				

            ax.set_xlabel('rotation (degrees)')
			# ax.set_ylabel('classification confidence')
            ax.set_ylabel('confidence')

            ax.set_ylim(0, 1.1)
            ax.set_xticks(labels)

            fig.savefig("2D_scatter_drop_rotated_mnist.pdf")
            fig.savefig("2D_scatter_drop_rotated_mnist.pdf")
            plt.close()

            # plot only 0, 30, 60 and 90 degrees and maybe also for another digit (e.g. 1)
            # TODO add colorbar (stds as values), make marker bigger
            fig = plt.figure()
            ax = fig.add_subplot()

			# breakpoint()

            min_std_plot = +np.INF
			max_std_plot = -np.INF

            for idx in range(6, 10, 1):
				c_values = np.take(dc_class_stds[idx].std(axis=2), dc_class_stds[idx].mean(axis=2).argmax(axis=1))
				min_std_plot = min(min_std_plot, c_values.min())
				max_std_plot = max(max_std_plot, c_values.max())
                sc = ax.scatter([labels[idx]]*dc_class_stds[idx].shape[0], dc_class_stds[idx].mean(axis=2).max(axis=1),
						        c=c_values,
						        ec='none',
						        cmap='Blues',
						        alpha=0.7,
						        s=600)



            ax.set_xlabel('rotation (degrees)')
			# ax.set_ylabel('classification confidence')
            ax.set_ylabel('confidence')

            ax.set_ylim(0, 1.1)
            ax.set_xticks(labels[6:10])

            # legend1 = ax.legend(*sc.legend_elements(),
			# 					loc="lower left", title="Standard deviation")
            # ax.add_artist(legend1)

            plt.colorbar(sc)

            fig.savefig("2D_scatter_drop_rotated_mnist_2.pdf")
            fig.savefig("2D_scatter_drop_rotated_mnist_2.pdf")
            plt.close()


        elif n == 7:
            confidences = {7:[], 5:[], 6:[]}
            stds = {7:[], 5:[], 6:[]}

            confidences[7] = [res_array.mean(axis=2)[:,n].mean() for res_array in drop_class_probs_dict.values()]
            confidences[5] = [res_array.mean(axis=2)[:,5].mean() for res_array in drop_class_probs_dict.values()]
            confidences[6] = [res_array.mean(axis=2)[:,6].mean() for res_array in drop_class_probs_dict.values()]
            stds[7] = [res_array.mean(axis=2)[:,n].std() for res_array in drop_class_probs_dict.values()]
            stds[5] = [res_array.mean(axis=2)[:,5].std() for res_array in drop_class_probs_dict.values()]
            stds[6] = [res_array.mean(axis=2)[:,6].std() for res_array in drop_class_probs_dict.values()]
            plot_mcd_accuracy_vs_confidence_many_labels(mcd_pc_test_accuracy, confidences, stds=stds, filename=filename+"_{}".format(n) + "_dropout_circuits_labels", fig_title='Rotated MNIST - DC (Digit {})'.format(n))
            plot_mcd_accuracy_vs_confidence_two_labels(mcd_pc_test_accuracy, confidences, stds=stds, filename=filename+"_two_{}".format(n) + "_dropout_circuits_labels", fig_title='Rotated MNIST - DC (Digit {})'.format(n))
        elif n == 9:
            confidences = {9:[], 6:[], 4:[]}
            stds = {9:[], 6:[], 4:[]}
            confidences[9] = [res_array.mean(axis=2)[:,n].mean() for res_array in drop_class_probs_dict.values()]
            confidences[6] = [res_array.mean(axis=2)[:,6].mean() for res_array in drop_class_probs_dict.values()]
            confidences[4] = [res_array.mean(axis=2)[:,4].mean() for res_array in drop_class_probs_dict.values()]
            stds[9] = [res_array.mean(axis=2)[:,n].std() for res_array in drop_class_probs_dict.values()]
            stds[6] = [res_array.mean(axis=2)[:,6].std() for res_array in drop_class_probs_dict.values()]
            stds[4] = [res_array.mean(axis=2)[:,4].std() for res_array in drop_class_probs_dict.values()]
            plot_mcd_accuracy_vs_confidence_many_labels(mcd_pc_test_accuracy, confidences, stds=stds, filename=filename+"_{}".format(n) + "_dropout_circuits_labels", fig_title='Rotated MNIST - DC (Digit {})'.format(n))
            plot_mcd_accuracy_vs_confidence_two_labels(mcd_pc_test_accuracy, confidences, stds=stds, filename=filename+"_two_{}".format(n) + "_dropout_circuits_labels", fig_title='Rotated MNIST - DC (Digit {})'.format(n))
        elif n == 1:
            confidences = {1:[], 4:[], 7:[]}
            stds = {1:[], 4:[], 7:[]}
            confidences[1] = [res_array.mean(axis=2)[:,n].mean() for res_array in drop_class_probs_dict.values()]
            confidences[4] = [res_array.mean(axis=2)[:,4].mean() for res_array in drop_class_probs_dict.values()]
            confidences[7] = [res_array.mean(axis=2)[:,7].mean() for res_array in drop_class_probs_dict.values()]
            stds[1] = [res_array.mean(axis=2)[:,n].std() for res_array in drop_class_probs_dict.values()]
            stds[4] = [res_array.mean(axis=2)[:,4].std() for res_array in drop_class_probs_dict.values()]
            stds[7] = [res_array.mean(axis=2)[:,7].std() for res_array in drop_class_probs_dict.values()]
            plot_mcd_accuracy_vs_confidence_many_labels(mcd_pc_test_accuracy, confidences, stds=stds, filename=filename+"_{}".format(n) + "_dropout_circuits_labels", fig_title='Rotated MNIST - DC (Digit {})'.format(n))
            plot_mcd_accuracy_vs_confidence_two_labels(mcd_pc_test_accuracy, confidences, stds=stds, filename=filename+"_two_{}".format(n) + "_dropout_circuits_labels", fig_title='Rotated MNIST - DC (Digit {})'.format(n))
        elif n == 3:
            confidences = {3:[], 6:[], 8:[]}
            stds = {3:[], 6:[], 8:[]}
            confidences[3] = [res_array.mean(axis=2)[:,n].mean() for res_array in drop_class_probs_dict.values()]
            confidences[6] = [res_array.mean(axis=2)[:,6].mean() for res_array in drop_class_probs_dict.values()]
            confidences[8] = [res_array.mean(axis=2)[:,8].mean() for res_array in drop_class_probs_dict.values()]
            stds[3] = [res_array.mean(axis=2)[:,n].std() for res_array in drop_class_probs_dict.values()]
            stds[6] = [res_array.mean(axis=2)[:,6].std() for res_array in drop_class_probs_dict.values()]
            stds[8] = [res_array.mean(axis=2)[:,8].std() for res_array in drop_class_probs_dict.values()]
            plot_mcd_accuracy_vs_confidence_many_labels(mcd_pc_test_accuracy, confidences, stds=stds, filename=filename+"_{}".format(n) + "_dropout_circuits_labels", fig_title='Rotated MNIST - DC (Digit {})'.format(n))
            plot_mcd_accuracy_vs_confidence_two_labels(mcd_pc_test_accuracy, confidences, stds=stds, filename=filename+"_two_{}".format(n) + "_dropout_circuits_labels", fig_title='Rotated MNIST - DC (Digit {})'.format(n))

def gen_plot_conf_vs_acc_corrupted(d_results, filename="acc_vs_conf_mcd_p_02_corrupted_sort_new"):
	mean = 0.1307
	std = 0.3081

	transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean), (std))])
	test_set = datasets.MNIST(root='./data', download=True, train=False, transform=transformer)
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False)

	drop_class_probs_c1 = np.load(d_results + 'dropout_class_probs_c1.npy')
	drop_class_probs_c2 = np.load(d_results + 'dropout_class_probs_c2.npy')
	drop_class_probs_c3 = np.load(d_results + 'dropout_class_probs_c3.npy')
	drop_class_probs_c4 = np.load(d_results + 'dropout_class_probs_c4.npy')
	drop_class_probs_c5 = np.load(d_results + 'dropout_class_probs_c5.npy')
	drop_class_probs_c6 = np.load(d_results + 'dropout_class_probs_c6.npy')
	drop_class_probs_c7 = np.load(d_results + 'dropout_class_probs_c7.npy')
	drop_class_probs_c8 = np.load(d_results + 'dropout_class_probs_c8.npy')
	drop_class_probs_c8 = np.load('/Users/fabrizio/Desktop/2022-05-11_16-12-31/results/dropout_class_probs_c_brightness_l2.npy')
	drop_class_probs_c9 = np.load(d_results + 'dropout_class_probs_c9.npy')
	drop_class_probs_c10 = np.load(d_results + 'dropout_class_probs_c10.npy')
	drop_class_probs_c11 = np.load(d_results + 'dropout_class_probs_c11.npy')
	drop_class_probs_c12 = np.load(d_results + 'dropout_class_probs_c12.npy')
	drop_class_probs_c13 = np.load(d_results + 'dropout_class_probs_c13.npy')
	drop_class_probs_c14 = np.load(d_results + 'dropout_class_probs_c14.npy')
	drop_class_probs_c15 = np.load(d_results + 'dropout_class_probs_c15.npy')

	class_probs_c1 = np.load(d_results + 'class_probs_c1.npy')
	class_probs_c2 = np.load(d_results + 'class_probs_c2.npy')
	class_probs_c3 = np.load(d_results + 'class_probs_c3.npy')
	class_probs_c4 = np.load(d_results + 'class_probs_c4.npy')
	class_probs_c5 = np.load(d_results + 'class_probs_c5.npy')
	class_probs_c6 = np.load(d_results + 'class_probs_c6.npy')
	class_probs_c7 = np.load(d_results + 'class_probs_c7.npy')
	class_probs_c8 = np.load(d_results + 'class_probs_c8.npy')
	class_probs_c8 = np.load('/Users/fabrizio/Desktop/2022-05-11_16-12-31/results/class_probs_c_brightness_l2.npy')
	class_probs_c9 = np.load(d_results + 'class_probs_c9.npy')
	class_probs_c10 = np.load(d_results + 'class_probs_c10.npy')
	class_probs_c11 = np.load(d_results + 'class_probs_c11.npy')
	class_probs_c12 = np.load(d_results + 'class_probs_c12.npy')
	class_probs_c13 = np.load(d_results + 'class_probs_c13.npy')
	class_probs_c14 = np.load(d_results + 'class_probs_c14.npy')
	class_probs_c15 = np.load(d_results + 'class_probs_c15.npy')

	drop_class_probs_dict = {1:drop_class_probs_c1, 2:drop_class_probs_c2, 3:drop_class_probs_c3, 4:drop_class_probs_c4, 5:drop_class_probs_c5, 6:drop_class_probs_c6, 7:drop_class_probs_c7, 8:drop_class_probs_c8, 9:drop_class_probs_c9, 10:drop_class_probs_c10, 11:drop_class_probs_c11, 12:drop_class_probs_c12, 13:drop_class_probs_c13, 14:drop_class_probs_c14, 15:drop_class_probs_c15}
	class_probs_dict = {1:class_probs_c1, 2:class_probs_c2, 3:class_probs_c3, 4:class_probs_c4, 5:class_probs_c5, 6:class_probs_c6, 7:class_probs_c7, 8:class_probs_c8, 9:class_probs_c9, 10:class_probs_c10, 11:class_probs_c11, 12:class_probs_c12, 13:class_probs_c13, 14:class_probs_c14, 15:class_probs_c15}

	drop_max_class_probs_c1 = drop_class_probs_c1.mean(axis=2).max(axis=1)
	drop_max_class_probs_c2 = drop_class_probs_c2.mean(axis=2).max(axis=1)
	drop_max_class_probs_c3 = drop_class_probs_c3.mean(axis=2).max(axis=1)
	drop_max_class_probs_c4 = drop_class_probs_c4.mean(axis=2).max(axis=1)
	drop_max_class_probs_c5 = drop_class_probs_c5.mean(axis=2).max(axis=1)
	drop_max_class_probs_c6 = drop_class_probs_c6.mean(axis=2).max(axis=1)
	drop_max_class_probs_c7 = drop_class_probs_c7.mean(axis=2).max(axis=1)
	drop_max_class_probs_c8 = drop_class_probs_c8.mean(axis=2).max(axis=1)
	drop_max_class_probs_c9 = drop_class_probs_c9.mean(axis=2).max(axis=1)
	drop_max_class_probs_c10 = drop_class_probs_c10.mean(axis=2).max(axis=1)
	drop_max_class_probs_c11 = drop_class_probs_c11.mean(axis=2).max(axis=1)
	drop_max_class_probs_c12 = drop_class_probs_c12.mean(axis=2).max(axis=1)
	drop_max_class_probs_c13 = drop_class_probs_c13.mean(axis=2).max(axis=1)
	drop_max_class_probs_c14 = drop_class_probs_c14.mean(axis=2).max(axis=1)
	drop_max_class_probs_c15 = drop_class_probs_c15.mean(axis=2).max(axis=1)

	max_class_probs_c1 = class_probs_c1.max(axis=1)
	max_class_probs_c2 = class_probs_c2.max(axis=1)
	max_class_probs_c3 = class_probs_c3.max(axis=1)
	max_class_probs_c4 = class_probs_c4.max(axis=1)
	max_class_probs_c5 = class_probs_c5.max(axis=1)
	max_class_probs_c6 = class_probs_c6.max(axis=1)
	max_class_probs_c7 = class_probs_c7.max(axis=1)
	max_class_probs_c8 = class_probs_c8.max(axis=1)
	max_class_probs_c9 = class_probs_c9.max(axis=1)
	max_class_probs_c10 = class_probs_c10.max(axis=1)
	max_class_probs_c11 = class_probs_c11.max(axis=1)
	max_class_probs_c12 = class_probs_c12.max(axis=1)
	max_class_probs_c13 = class_probs_c13.max(axis=1)
	max_class_probs_c14 = class_probs_c14.max(axis=1)
	max_class_probs_c15 = class_probs_c15.max(axis=1)


	pc_test_accuracy = [(test_loader.dataset.targets.numpy()  == res_array.argmax(axis=1)).sum()/test_loader.dataset.targets.numpy().shape[0] for res_array in class_probs_dict.values()]
	mcd_pc_test_accuracy = [(test_loader.dataset.targets.numpy()  == res_array.mean(axis=2).argmax(axis=1)).sum()/test_loader.dataset.targets.numpy().shape[0] for res_array in drop_class_probs_dict.values()]

	mcd_avg_class_confidence = [drop_max_class_probs_c1.mean(), drop_max_class_probs_c2.mean(), drop_max_class_probs_c3.mean(), drop_max_class_probs_c4.mean(), drop_max_class_probs_c5.mean(), drop_max_class_probs_c6.mean(), drop_max_class_probs_c7.mean(), drop_max_class_probs_c8.mean(), drop_max_class_probs_c9.mean(), drop_max_class_probs_c10.mean(), drop_max_class_probs_c11.mean(), drop_max_class_probs_c12.mean(), drop_max_class_probs_c13.mean(), drop_max_class_probs_c14.mean(), drop_max_class_probs_c15.mean()]
	pc_avg_class_confidence = [max_class_probs_c1.mean(), max_class_probs_c2.mean(), max_class_probs_c3.mean(), max_class_probs_c4.mean(), max_class_probs_c5.mean(), max_class_probs_c6.mean(), max_class_probs_c7.mean(), max_class_probs_c8.mean(), max_class_probs_c9.mean(), max_class_probs_c10.mean(), max_class_probs_c11.mean(), max_class_probs_c12.mean(), max_class_probs_c13.mean(), max_class_probs_c14.mean(), max_class_probs_c15.mean()]

	assert len(pc_test_accuracy) == len(mcd_pc_test_accuracy)
	assert len(mcd_avg_class_confidence) == len(pc_avg_class_confidence)
	assert len(pc_test_accuracy) == len(pc_avg_class_confidence)

	labels_corruption = ['SN', 'IN', 'GB', 'MB', 'SH', 'SC', 'RO', 'BR', 'TR', 'ST', 'FO', 'SP', 'DO', 'ZI', 'CA']

	mcd_pc_test_accuracy = np.array(mcd_pc_test_accuracy)
	pc_test_accuracy = np.array(pc_test_accuracy)
	pc_avg_class_confidence = np.array(pc_avg_class_confidence)
	mcd_avg_class_confidence = np.array(mcd_avg_class_confidence)

	# labels_corruption = ['SN', 'IN', 'GB', 'MB', 'SH', 'SC', 'RO', 'TR', 'ST', 'FO', 'SP', 'DO', 'ZI', 'CA']
	# pc_test_accuracy = np.delete(pc_test_accuracy, 7, 0)
	# mcd_pc_test_accuracy = np.delete(mcd_pc_test_accuracy, 7, 0)
	# pc_avg_class_confidence = np.delete(pc_avg_class_confidence, 7, 0)
	# mcd_avg_class_confidence = np.delete(mcd_avg_class_confidence, 7, 0)

	sort_idxs = mcd_pc_test_accuracy.argsort()
	pc_test_accuracy = pc_test_accuracy[sort_idxs[::-1]].tolist()

	labels_corruption = np.array(labels_corruption)
	labels_corruption = labels_corruption[sort_idxs[::-1]]
	labels_corruption = labels_corruption.tolist()

	pc_avg_class_confidence = pc_avg_class_confidence[sort_idxs[::-1]].tolist()
	mcd_avg_class_confidence = mcd_avg_class_confidence[sort_idxs[::-1]].tolist()
	mcd_pc_test_accuracy = mcd_pc_test_accuracy[sort_idxs[::-1]].tolist()

	plot_mcd_accuracy_vs_confidence([pc_test_accuracy, mcd_pc_test_accuracy], [pc_avg_class_confidence, mcd_avg_class_confidence], filename, fig_title='Corrupted MNIST', labels=labels_corruption, xlabel="Corruption ID")

	return sort_idxs

def gen_plot_conf_vs_acc_corrupted_single(d_results, filename="acc_vs_conf_mcd_p_02_corrupted_single"):
	mean = 0.1307
	std = 0.3081

	transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean), (std))])
	test_set = datasets.MNIST(root='./data', download=True, train=False, transform=transformer)
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False)


	import importlib.util
	spec = importlib.util.spec_from_file_location("corruptions", "/Users/fabrizio/research/mnist-c/mnist-c/corruptions.py")
	corruptions = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(corruptions)
	corruption_method = [corruptions.brightness, corruptions.shot_noise, corruptions.impulse_noise, corruptions.glass_blur,
						 corruptions.motion_blur, corruptions.shear, corruptions.scale,
						 corruptions.rotate, corruptions.translate, corruptions.fog, corruptions.spatter]
	severity = [1, 2, 3, 4, 5]

	for cm in corruption_method:
		res_pc = {}
		res_dc = {}
		for sl in severity:
			res_pc[sl] = np.load(d_results + 'class_probs_c_{}_l{}.npy'.format(cm.__name__, sl))
			res_dc[sl] = np.load(d_results + 'dropout_class_probs_c_{}_l{}.npy'.format(cm.__name__, sl))

		pc_test_accuracy = [(test_loader.dataset.targets.numpy()  == res_array.argmax(axis=1)).sum()/test_loader.dataset.targets.numpy().shape[0] for res_array in res_pc.values()]
		mcd_pc_test_accuracy = [(test_loader.dataset.targets.numpy()  == res_array.mean(axis=2).argmax(axis=1)).sum()/test_loader.dataset.targets.numpy().shape[0] for res_array in res_dc.values()]

		mcd_avg_class_confidence = [res_array.mean(axis=2).max(axis=1).mean() for res_array in res_dc.values()]
		pc_avg_class_confidence = [res_array.max(axis=1).mean() for res_array in res_pc.values()]

		assert len(pc_test_accuracy) == len(mcd_pc_test_accuracy)
		assert len(mcd_avg_class_confidence) == len(pc_avg_class_confidence)
		assert len(pc_test_accuracy) == len(pc_avg_class_confidence)

		# labels_corruption = ['BR', 'SN', 'IN', 'GB', 'MB', 'SH', 'SC', 'RO', 'TR', 'FO', 'SP']
		labels_corruption = ['1', '2', '3', '4', '5']

		final_filename = filename + '_' + cm.__name__

		plot_mcd_accuracy_vs_confidence([pc_test_accuracy, mcd_pc_test_accuracy], [pc_avg_class_confidence, mcd_avg_class_confidence], final_filename, fig_title='Corrupted MNIST ({})'.format(cm.__name__).replace('_', ' '), labels=labels_corruption, xlabel="Severity")

		boxplot_data = np.column_stack((res_array.max(axis=1) for res_array in res_pc.values()))
		drop_boxplot_data = np.column_stack((res_array.mean(axis=2).max(axis=1) for res_array in res_dc.values()))
		entropy_boxplot_data = np.column_stack((entropy(res_array, axis=1) for res_array in res_pc.values()))
		entropy_drop_boxplot_data = np.column_stack((entropy(res_array.mean(axis=2), axis=1) for res_array in res_dc.values()))

		# gen 4 boxplots in 1 rows
		fig, axes = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=False, figsize=(14,4), tight_layout=True)

		medianprops = dict(linestyle='-.', linewidth=2, color='green')
		meanlineprops = dict(linestyle='--', linewidth=1, color='purple')
		meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='indianred')

		bp1 = axes[0].boxplot(boxplot_data, medianprops=medianprops, meanprops=meanpointprops, showmeans=True, meanline=False, labels=labels_corruption, showfliers=False, patch_artist=True)
		bp2 = axes[1].boxplot(drop_boxplot_data, medianprops=medianprops, meanprops=meanpointprops, showmeans=True, meanline=False, labels=labels_corruption, showfliers=False, patch_artist=True)
		bp3 = axes[2].boxplot(entropy_boxplot_data, medianprops=medianprops, meanprops=meanpointprops, showmeans=True, meanline=False, labels=labels_corruption, showfliers=False, patch_artist=True)
		bp4 = axes[3].boxplot(entropy_drop_boxplot_data, medianprops=medianprops, meanprops=meanpointprops, showmeans=True, meanline=False, labels=labels_corruption, showfliers=False, patch_artist=True)

		axes[0].set_title('Probabilistic Circuits')
		axes[1].set_title('Dropout Circuits')
		axes[2].set_title('Probabilistic Circuits')
		axes[3].set_title('Dropout Circuits')

		plt.setp(bp1['boxes'], facecolor='linen')
		plt.setp(bp2['boxes'], facecolor='linen')
		plt.setp(bp3['boxes'], facecolor='lightcyan')
		plt.setp(bp4['boxes'], facecolor='lightcyan')


		axes[0].set_ylabel('classification confidence', fontsize='large', fontweight='bold')
		axes[0].set_ylim(0, 1.1)
		axes[1].set_ylim(0, 1.1)
		#axes[1].get_yaxis().set_ticks([])
		axes[2].set_ylabel('classification entropy', fontsize='large', fontweight='bold')
		#axes[3].get_yaxis().set_ticks([])
		axes[2].set_ylim(-0.1, 3)
		axes[3].set_ylim(-0.1, 3)

		axes[0].yaxis.grid(True, linestyle='--')
		axes[1].yaxis.grid(True, linestyle='--')
		axes[2].yaxis.grid(True, linestyle='--')
		axes[3].yaxis.grid(True, linestyle='--')

		ax = plt.gca()
		ax.set_xlabel(' ')
		fig.text(0.5,0.03, "severity level", ha="center", va="center", fontsize='large')
		fig.suptitle('Corrupted MNIST ({})'.format(cm.__name__.replace('_', ' ')), fontsize='large', fontweight='bold')
		plt.savefig('corrupted_mnist_all_boxplots_{}.png'.format(cm.__name__))
		plt.savefig('rotating_mnist_all_boxplots_{}.pdf'.format(cm.__name__))
		plt.close()

def gen_plot_conf_vs_acc_corrupted_svhn(d_results, filename="acc_vs_conf_mcd_p_02_corrupted_sort_new"):
	svhn_transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4377, 0.4438, 0.4728), (0.198, 0.201, 0.197))])

	test_set = datasets.SVHN(root='../data', download=True, split='test', transform=svhn_transformer)
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False)

	from imagecorruptions import corrupt, get_corruption_names

	dc_results, pc_results = {}, {}
	for corruption in get_corruption_names('common'):
		for cl in range(1):
			cl += 1
			print("Load corruption {} Level {}".format(corruption, cl))
			dc_results['c_{}_l{}'.format(corruption, cl)] = np.load(
				d_results + 'dropout_class_probs_c_{}_l{}.npy'.format(corruption, cl))
			pc_results['c_{}_l{}'.format(corruption, cl)] = np.load(
				d_results + 'class_probs_c_{}_l{}.npy'.format(corruption, cl))

	pc_test_accuracy = [(test_loader.dataset.labels == res_array.argmax(axis=1)).sum()/test_loader.dataset.labels.shape[0] for res_array in pc_results.values()]
	dc_test_accuracy = [(test_loader.dataset.labels == res_array.mean(axis=2).argmax(axis=1)).sum()/test_loader.dataset.labels.shape[0] for res_array in dc_results.values()]

	dc_avg_class_confidence = [res_array.mean(axis=2).max(axis=1).mean() for res_array in dc_results.values()]
	pc_avg_class_confidence = [res_array.max(axis=1).mean() for res_array in pc_results.values()]

	assert len(pc_test_accuracy) == len(dc_test_accuracy)
	assert len(dc_avg_class_confidence) == len(pc_avg_class_confidence)
	assert len(pc_test_accuracy) == len(pc_avg_class_confidence)

	labels_corruption = ['GN', 'SN', 'IN', 'DB', 'GB', 'MB', 'ZB', 'SN', 'FR', 'FO', 'BR', 'CO', 'ET', 'PX', 'JP']

	dc_test_accuracy = np.array(dc_test_accuracy)
	pc_test_accuracy = np.array(pc_test_accuracy)
	pc_avg_class_confidence = np.array(pc_avg_class_confidence)
	dc_avg_class_confidence = np.array(dc_avg_class_confidence)

	sort_idxs = dc_test_accuracy.argsort()
	pc_test_accuracy = pc_test_accuracy[sort_idxs[::-1]].tolist()

	labels_corruption = np.array(labels_corruption)
	labels_corruption = labels_corruption[sort_idxs[::-1]]
	labels_corruption = labels_corruption.tolist()

	pc_avg_class_confidence = pc_avg_class_confidence[sort_idxs[::-1]].tolist()
	dc_avg_class_confidence = dc_avg_class_confidence[sort_idxs[::-1]].tolist()
	dc_test_accuracy = dc_test_accuracy[sort_idxs[::-1]].tolist()

	# print(dc_test_accuracy)
	# print(dc_avg_class_confidence)
	# print(pc_test_accuracy)
	# print(pc_avg_class_confidence)
	# breakpoint()
	plot_mcd_accuracy_vs_confidence([pc_test_accuracy, dc_test_accuracy], [pc_avg_class_confidence, dc_avg_class_confidence], filename, fig_title='Corrupted SVHN', labels=labels_corruption, xlabel="Corruption ID")

	return sort_idxs

def gen_plot_conf_vs_acc_svhn_c_single(d_results, filename="acc_vs_conf_mcd_p_02_corrupted_single"):

	severity = [1, 2, 3, 4, 5]

	svhn_transformer = transforms.Compose(
		[transforms.ToTensor(), transforms.Normalize((0.4377, 0.4438, 0.4728), (0.198, 0.201, 0.197))])

	test_set = datasets.SVHN(root='../data', download=True, split='test', transform=svhn_transformer)
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False)

	from imagecorruptions import corrupt, get_corruption_names

	dc_results, pc_results = {}, {}
	for corruption in get_corruption_names('common'):
		res_pc = {}
		res_dc = {}
		for cl in severity:
			print("Load corruption {} Level {}".format(corruption, cl))
			res_dc[cl] = np.load(
				d_results + 'dropout_class_probs_c_{}_l{}.npy'.format(corruption, cl))
			res_pc[cl] = np.load(
				d_results + 'class_probs_c_{}_l{}.npy'.format(corruption, cl))




		pc_test_accuracy = [(test_loader.dataset.labels  == res_array.argmax(axis=1)).sum()/test_loader.dataset.labels.shape[0] for res_array in res_pc.values()]
		mcd_pc_test_accuracy = [(test_loader.dataset.labels == res_array.mean(axis=2).argmax(axis=1)).sum()/test_loader.dataset.labels.shape[0] for res_array in res_dc.values()]

		mcd_avg_class_confidence = [res_array.mean(axis=2).max(axis=1).mean() for res_array in res_dc.values()]
		pc_avg_class_confidence = [res_array.max(axis=1).mean() for res_array in res_pc.values()]

		assert len(pc_test_accuracy) == len(mcd_pc_test_accuracy)
		assert len(mcd_avg_class_confidence) == len(pc_avg_class_confidence)
		assert len(pc_test_accuracy) == len(pc_avg_class_confidence)

		# labels_corruption = ['BR', 'SN', 'IN', 'GB', 'MB', 'SH', 'SC', 'RO', 'TR', 'FO', 'SP']
		labels_corruption = ['1', '2', '3', '4', '5']

		final_filename = filename + '_' + corruption

		plot_mcd_accuracy_vs_confidence([pc_test_accuracy, mcd_pc_test_accuracy], [pc_avg_class_confidence, mcd_avg_class_confidence], final_filename, fig_title='Corrupted SVHN ({})'.format(corruption).replace('_', ' '), labels=labels_corruption, xlabel="Severity")

		boxplot_data = np.column_stack((res_array.max(axis=1) for res_array in res_pc.values()))
		drop_boxplot_data = np.column_stack((res_array.mean(axis=2).max(axis=1) for res_array in res_dc.values()))
		entropy_boxplot_data = np.column_stack((entropy(res_array, axis=1) for res_array in res_pc.values()))
		entropy_drop_boxplot_data = np.column_stack((entropy(res_array.mean(axis=2), axis=1) for res_array in res_dc.values()))

		# gen 4 boxplots in 1 rows
		fig, axes = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=False, figsize=(14,4), tight_layout=True)

		medianprops = dict(linestyle='-.', linewidth=2, color='green')
		meanlineprops = dict(linestyle='--', linewidth=1, color='purple')
		meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='indianred')

		bp1 = axes[0].boxplot(boxplot_data, medianprops=medianprops, meanprops=meanpointprops, showmeans=True, meanline=False, labels=labels_corruption, showfliers=False, patch_artist=True)
		bp2 = axes[1].boxplot(drop_boxplot_data, medianprops=medianprops, meanprops=meanpointprops, showmeans=True, meanline=False, labels=labels_corruption, showfliers=False, patch_artist=True)
		bp3 = axes[2].boxplot(entropy_boxplot_data, medianprops=medianprops, meanprops=meanpointprops, showmeans=True, meanline=False, labels=labels_corruption, showfliers=False, patch_artist=True)
		bp4 = axes[3].boxplot(entropy_drop_boxplot_data, medianprops=medianprops, meanprops=meanpointprops, showmeans=True, meanline=False, labels=labels_corruption, showfliers=False, patch_artist=True)

		axes[0].set_title('Probabilistic Circuits')
		axes[1].set_title('Dropout Circuits')
		axes[2].set_title('Probabilistic Circuits')
		axes[3].set_title('Dropout Circuits')

		plt.setp(bp1['boxes'], facecolor='linen')
		plt.setp(bp2['boxes'], facecolor='linen')
		plt.setp(bp3['boxes'], facecolor='lightcyan')
		plt.setp(bp4['boxes'], facecolor='lightcyan')


		axes[0].set_ylabel('classification confidence', fontsize='large', fontweight='bold')
		axes[0].set_ylim(0, 1.1)
		axes[1].set_ylim(0, 1.1)
		#axes[1].get_yaxis().set_ticks([])
		axes[2].set_ylabel('classification entropy', fontsize='large', fontweight='bold')
		#axes[3].get_yaxis().set_ticks([])
		axes[2].set_ylim(-0.1, 3)
		axes[3].set_ylim(-0.1, 3)

		axes[0].yaxis.grid(True, linestyle='--')
		axes[1].yaxis.grid(True, linestyle='--')
		axes[2].yaxis.grid(True, linestyle='--')
		axes[3].yaxis.grid(True, linestyle='--')

		ax = plt.gca()
		ax.set_xlabel(' ')
		fig.text(0.5,0.03, "severity level", ha="center", va="center", fontsize='large')
		fig.suptitle('Corrupted SVHN ({})'.format(corruption.replace('_', ' ')), fontsize='large', fontweight='bold')
		plt.savefig('corrupted_svhn_all_boxplots_{}.png'.format(corruption))
		plt.savefig('corrupted_svhn_all_boxplots_{}.pdf'.format(corruption))
		plt.close()

def gen_plot_conf_vs_acc():

    # MCD d = 0.1
    pc_test_accuracy = [13.30, 21.72, 65.61, 95.39, 61.00, 21.63, 15.15]
    mcd_pc_test_accuracy = [13.40, 21.78, 65.92, 95.43, 61.85, 21.94, 15.20]

    pc_avg_class_confidence = [0.8938149, 0.8944525, 0.9368199, 0.986387, 0.92594326, 0.9100882, 0.9091338]
    mcd_avg_class_confidence = [0.6953, 0.7010, 0.7912, 0.9080, 0.7771, 0.7161, 0.7387]

    plot_mcd_accuracy_vs_confidence([pc_test_accuracy, mcd_pc_test_accuracy], [pc_avg_class_confidence, mcd_avg_class_confidence], "acc_vs_conf_mcd_p_01")

    pc = np.hstack(( np.array(pc_test_accuracy).reshape(-1,1), np.array(pc_avg_class_confidence).reshape(-1,1) ))
    pc_mcd = np.hstack(( np.array(mcd_pc_test_accuracy).reshape(-1,1), np.array(mcd_avg_class_confidence).reshape(-1,1) ))

    pc = pc[pc[:,0].argsort()]
    pc_mcd = pc_mcd[pc_mcd[:,0].argsort()]

    plot_accuracy_vs_confidence_line(pc, pc_mcd, "acc_vs_conf_mcd_p_01_line")

    # MCD d = 0.2
    pc_test_accuracy = [13.54, 21.88, 67.83, 95.07, 61.69, 20.54, 14.42]
    mcd_pc_test_accuracy = [13.75, 22.21, 68.53, 95.32, 61.13, 20.26, 14.03]

    pc_avg_class_confidence = [0.8726, 0.8797, 0.9408, 0.9867, 0.9343, 0.8823, 0.8807]
    mcd_avg_class_confidence = [0.5148, 0.5226, 0.6217, 0.7643, 0.6054, 0.5298, 0.5257]

    plot_mcd_accuracy_vs_confidence([pc_test_accuracy, mcd_pc_test_accuracy], [pc_avg_class_confidence, mcd_avg_class_confidence], "acc_vs_conf_mcd_p_02")

    pc = np.hstack(( np.array(pc_test_accuracy).reshape(-1,1), np.array(pc_avg_class_confidence).reshape(-1,1) ))
    pc_mcd = np.hstack(( np.array(mcd_pc_test_accuracy).reshape(-1,1), np.array(mcd_avg_class_confidence).reshape(-1,1) ))

    pc = pc[pc[:,0].argsort()]
    pc_mcd = pc_mcd[pc_mcd[:,0].argsort()]

    plot_accuracy_vs_confidence_line(pc, pc_mcd, "acc_vs_conf_mcd_p_02_line")

    ### MCD p = 0.1 ###
    # Digit 0
    pc_test_accuracy = [54.18, 62.86, 89.59, 98.16, 93.67, 79.29, 60.71]
    mcd_pc_test_accuracy = [56.43, 64.90, 90.71, 98.27, 94.10, 80.61, 62.04]

    pc_avg_class_confidence = [0.91, 0.93, 0.97, 0.995, 0.99, 0.96, 0.93]
    mcd_avg_class_confidence = [0.76, 0.80, 0.91, 0.97, 0.94, 0.88, 0.81]

    plot_mcd_accuracy_vs_confidence([pc_test_accuracy, mcd_pc_test_accuracy], [pc_avg_class_confidence, mcd_avg_class_confidence], "acc_vs_conf_mcd_p_01_digit_0", "Rotating MNIST - Digit 0")


    # Digit 1
    pc_test_accuracy = [0, 14.19, 79.82, 99.03, 72.33, 6.96, 0.18]
    mcd_pc_test_accuracy = [0, 14.45, 80.88, 98.94, 72.95, 6.52, 0.18]

    pc_avg_class_confidence = [0.92, 0.91, 0.97, 0.996, 0.93, 0.92, 0.97]
    mcd_avg_class_confidence = [0.74950427, 0.72899824, 0.90689003, 0.9920027, 0.858448, 0.7936631, 0.885084]

    plot_mcd_accuracy_vs_confidence([pc_test_accuracy, mcd_pc_test_accuracy], [pc_avg_class_confidence, mcd_avg_class_confidence], "acc_vs_conf_mcd_p_01_digit_1", "Rotating MNIST - Digit 1")

    # Digit 2
    pc_test_accuracy = [4.84, 5.72, 52.81, 94.96, 54.84, 16.67, 16.76]
    mcd_pc_test_accuracy = [3.39, 4.26, 50.68, 94.57, 53.49, 14.44, 13.18]

    pc_avg_class_confidence = [0.90052265, 0.8934769, 0.9244727, 0.988453, 0.9337067, 0.91467637, 0.90259516]
    mcd_avg_class_confidence = [0.69871, 0.69955754, 0.69449437, 0.79105586, 0.7279546, 0.7349449, 0.7295492]

    plot_mcd_accuracy_vs_confidence([pc_test_accuracy, mcd_pc_test_accuracy], [pc_avg_class_confidence, mcd_avg_class_confidence], "acc_vs_conf_mcd_p_01_digit_2", "Rotating MNIST - Digit 2")
    
    # Digit 3
    pc_test_accuracy = [3.47, 20.59, 73.96, 94.75, 69.80, 29.50, 7.33]
    mcd_pc_test_accuracy = [3.76, 21.68, 75.94, 95.45, 72.38, 29.90, 7.72 ]

    pc_avg_class_confidence = [0.88564223, 0.8779888, 0.9404398, 0.97786987, 0.9248703, 0.8818537, 0.88763726]
    mcd_avg_class_confidence = [0.6874759, 0.67964464, 0.82583153, 0.93764263, 0.78587365, 0.67695415, 0.69217515]

    plot_mcd_accuracy_vs_confidence([pc_test_accuracy, mcd_pc_test_accuracy], [pc_avg_class_confidence, mcd_avg_class_confidence], "acc_vs_conf_mcd_p_01_digit_3", "Rotating MNIST - Digit 3")
    
    # Digit 4
    pc_test_accuracy = [11.81, 13.34, 43.28, 94.09, 46.64, 21.38, 30.24]
    mcd_pc_test_accuracy = [12.73, 16.09, 46.33, 94.70, 50.20, 23.32, 32.18]

    pc_avg_class_confidence = [0.88564223, 0.8779888, 0.9404398, 0.97786987, 0.9248703, 0.8818537, 0.88763726]
    mcd_avg_class_confidence = [0.6570231, 0.6476715, 0.69988304, 0.91781825, 0.73084897, 0.69716495, 0.6530819]

    plot_mcd_accuracy_vs_confidence([pc_test_accuracy, mcd_pc_test_accuracy], [pc_avg_class_confidence, mcd_avg_class_confidence], "acc_vs_conf_mcd_p_01_digit_4", "Rotating MNIST - Digit 4")
    
    # Digit 5
    pc_test_accuracy = [4.04, 22.42, 60.65, 93.72, 55.04, 10.54, 3.36]
    mcd_pc_test_accuracy = [3.48, 20.74, 58.74, 92.60, 52.02, 8.86, 3.25]

    pc_avg_class_confidence = [0.89075506, 0.90212584, 0.9342961, 0.9801143, 0.89843947, 0.86931735, 0.8770746]
    mcd_avg_class_confidence = [0.7154905, 0.725274, 0.7220949, 0.8220068, 0.6821552, 0.64897645, 0.6606313]

    plot_mcd_accuracy_vs_confidence([pc_test_accuracy, mcd_pc_test_accuracy], [pc_avg_class_confidence, mcd_avg_class_confidence], "acc_vs_conf_mcd_p_01_digit_5", "Rotating MNIST - Digit 5")
    
    # Digit 6
    pc_test_accuracy = [2.82, 16.81, 75.16, 96.24, 56.99, 24.01, 20.98]
    mcd_pc_test_accuracy = [3.03, 17.22, 76.10, 96.66, 58.35, 25.26, 23.90]

    pc_avg_class_confidence = [0.9012232, 0.8862662, 0.9562003, 0.9915845, 0.9250488, 0.89030874, 0.8724931]
    mcd_avg_class_confidence = [0.69099736, 0.7134204, 0.8650135, 0.95733297, 0.78785634, 0.6858617, 0.67832017]

    plot_mcd_accuracy_vs_confidence([pc_test_accuracy, mcd_pc_test_accuracy], [pc_avg_class_confidence, mcd_avg_class_confidence], "acc_vs_conf_mcd_p_01_digit_6", "Rotating MNIST - Digit 6")
    
    # Digit 7
    pc_test_accuracy = [27.53, 19.94, 40.37, 95.33, 48.15, 6.23, 1.75]
    mcd_pc_test_accuracy = [25.97, 18.19, 37.84, 94.75, 47.86, 6.23, 1.36]

    pc_avg_class_confidence = [0.9112222, 0.9012318, 0.8998183, 0.98414975, 0.9542258, 0.94954824, 0.94692415]
    mcd_avg_class_confidence = [0.71157044, 0.70606977, 0.7077255, 0.8654702, 0.76602495, 0.7251801, 0.8185596]

    plot_mcd_accuracy_vs_confidence([pc_test_accuracy, mcd_pc_test_accuracy], [pc_avg_class_confidence, mcd_avg_class_confidence], "acc_vs_conf_mcd_p_01_digit_7", "Rotating MNIST - Digit 7")
    
    # Digit 8
    pc_test_accuracy = [8.93, 16.12, 64.37, 93.43, 62.73, 15.61, 6.98]
    mcd_pc_test_accuracy = [9.65, 16.74, 65.50, 94.05, 65.61, 17.35, 7.19]

    pc_avg_class_confidence = [0.86429065, 0.8767025, 0.92782617, 0.97846264, 0.92970204, 0.8820493, 0.90632606]
    mcd_avg_class_confidence = [0.6601284, 0.67979926, 0.7754102, 0.9136181, 0.7818706, 0.6545178, 0.73271656]

    plot_mcd_accuracy_vs_confidence([pc_test_accuracy, mcd_pc_test_accuracy], [pc_avg_class_confidence, mcd_avg_class_confidence], "acc_vs_conf_mcd_p_01_digit_8", "Rotating MNIST - Digit 8")
    
    # Digit 9
    pc_test_accuracy = [16.35, 27.16, 75.02, 93.56, 48.46, 8.62, 5.65]
    mcd_pc_test_accuracy = [15.86, 26.66, 76.31, 93.26, 49.06, 8.62, 5.25]

    pc_avg_class_confidence = [0.8574678, 0.8680167, 0.9335387, 0.98327655, 0.88295984, 0.9081661, 0.8962768]
    mcd_avg_class_confidence = [0.6247081, 0.6374549, 0.7887788, 0.8902882, 0.6971628, 0.66210395, 0.69889003]

    plot_mcd_accuracy_vs_confidence([pc_test_accuracy, mcd_pc_test_accuracy], [pc_avg_class_confidence, mcd_avg_class_confidence], "acc_vs_conf_mcd_p_01_digit_9", "Rotating MNIST - Digit 9")

def nn_accuracy_vs_confidence():
    # MCD p = 0.2

    # pc_test_accuracy = [(test_loader.dataset.targets.numpy()  == res_array.argmax(axis=1)).sum()/test_loader.dataset.targets.numpy().shape[0] for res_array in class_probs_dict.values()]
    # mcd_pc_test_accuracy = [(test_loader.dataset.targets.numpy()  == res_array.mean(axis=2).argmax(axis=1)).sum()/test_loader.dataset.targets.numpy().shape[0] for res_array in drop_class_probs_dict.values()]

    # labels = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 0]
    labels = ["-180", "-150", "-120", "-90", "-60", "-30", "0", "30", "60", "90", "120", "150"]

    # pc_test_accuracy = [0.3092, 0.2596, 0.1512, 0.1321, 0.2497, 0.7525, 0.9713, 0.6877, 0.212, 0.121, 0.1497, 0.2687]
    # mcd_pc_test_accuracy = [0.3092, 0.2598, 0.1511, 0.1315, 0.2479, 0.7515, 0.9713, 0.6863, 0.2107, 0.1206, 0.1497, 0.2684]

    # pc_test_conf = [0.9994893725216388, 0.9994722199678421, 0.9991746889710427, 0.9992639874815941, 0.9991397645175457, 0.9996216004669666, 0.9999360091328621, 0.999648166179657, 0.9996217341125011, 0.9994376227736473, 0.9993582387685775, 0.9992988487660884]
    # mcd_test_conf = [0.8941314125322828, 0.8813078832309582, 0.8709883483422384, 0.8468019912679036, 0.838076009036793, 0.9140075893612275, 0.9867131624806834, 0.9221946048704818, 0.8901952232674841, 0.8736242637878915, 0.837331470342473, 0.8582712244592489]

    # MCD p = 0.5
    pc_test_accuracy = np.array([0.6898, 0.2408, 0.1265, 0.1681, 0.248, 0.2713, 0.2277, 0.168, 0.1784, 0.1916, 0.632, 0.9651])
    mcd_pc_test_accuracy = np.array([0.6861, 0.2391, 0.1308, 0.1674, 0.2484, 0.2722, 0.2284, 0.1675, 0.1795, 0.193, 0.6276, 0.964])

    pc_test_conf = np.array([0.9995908941090107, 0.999209826350212, 0.9995388736128807, 0.999325581842661, 0.9996194237828254, 0.9996447681069374, 0.9994001618385315, 0.9992742420852184, 0.9997053436815738, 0.9996422296643257, 0.9998249816954136, 0.9999406792521477])
    mcd_test_conf = np.array([0.8414500771113622, 0.7203332337849803, 0.7548855906350697, 0.7799100487649651, 0.8031128558081028, 0.825133473971939, 0.7851172262658751, 0.7476283550736273, 0.8016413770959246, 0.7707684216636047, 0.8480479367231663, 0.9673197068951317])

    assert pc_test_conf.shape == mcd_test_conf.shape, (pc_test_conf.shape, mcd_test_conf.shape)

    pc_test_accuracy = np.hstack((np.flip(pc_test_accuracy[:6]), pc_test_accuracy[-1], np.flip(pc_test_accuracy[6:11] )))
    mcd_pc_test_accuracy = np.hstack((np.flip(mcd_pc_test_accuracy[:6]), mcd_pc_test_accuracy[-1], np.flip(mcd_pc_test_accuracy[6:11] )))
    pc_test_conf = np.hstack((np.flip(pc_test_conf[:6]), pc_test_conf[-1], np.flip(pc_test_conf[6:11] )))
    mcd_test_conf = np.hstack((np.flip(mcd_test_conf[:6]), mcd_test_conf[-1], np.flip(mcd_test_conf[6:11] )))

    # pc_test_accuracy = [0.7525, 0.2497, 0.1321, 0.1512, 0.2596, 0.3092, 0.2687, 0.1497, 0.121, 0.212, 0.6877, 0.9713]
    # mcd_pc_test_accuracy = [0.7515, 0.2479, 0.1315, 0.1511, 0.2598, 0.3092, 0.2684, 0.1497, 0.1206, 0.2107, 0.6863, 0.9713]

    # plt.rcParams['text.usetex'] = True
    fig, ax = plt.subplots()

    fig_title = None
    filename = 'nn_acc_vs_conf_p05'


    #ax.plot(np.array(accuracies[0]), np.array(confidences[0])*100, label='PC', marker='o', linewidth=4, markersize=12)
    #plt.plot(accuracies[1], confidences[1], label='PC+MCD', marker='o', linewidth=4, markersize=12)

    #ax.plot(labels, accuracies[0], label='PC', marker='^', linewidth=4, markersize=12)

    ax.set_xlabel('Rotation (degrees) ')
    ax.set_ylabel('Classification confidence')
    ax.set_ylim([-0.1, 1.1])
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.plot(labels, pc_test_conf, label='DNN (Confidence)', marker='o', linewidth=4, markersize=12, color='mediumaquamarine')
    ax.plot(labels, mcd_test_conf, label='MCD (Confidence)', marker='s', linewidth=4, markersize=12, color='orange')
    ax.plot(labels, pc_test_accuracy, label='DNN (Accuracy)', marker='X', linewidth=3, markersize=10, color='royalblue', linestyle='--', mfc='mediumaquamarine')
    ax.plot(labels, mcd_pc_test_accuracy, label='MCD (Accuracy)', marker='P', linewidth=3, markersize=10, color='royalblue', linestyle='--', mfc='orange')

    ax2 = ax.twinx()
    ax2.set_ylabel('Classification accuracy', color='royalblue', fontweight='bold')
    # # ax2.set_yticks(np.arange(0, 1.1, 0.1))
    ax2.set_yticks([])

    ax.grid(True)


    #ax.set_title('Rotated MNIST')
    if fig_title is None:
        ax.set_title('Rotated MNIST - DNN')
    else:
        ax.set_title(fig_title)
    leg = ax.legend()

    for te in leg.get_texts():
        te.set_fontweight("roman")
        if te.get_text() == 'DNN (Accuracy)' or te.get_text() == 'MCD (Accuracy)':
            te.set_color("royalblue")

    plt.savefig("./{}.png".format(filename))
    plt.savefig("./{}.pdf".format(filename))
    plt.close()



def plot_some_mnist_digits(element_idx=0, d_results=None):

    drop_class_probs_90 = np.load(d_results + 'dropout_class_probs_90.out.npy')
    drop_class_probs_60 = np.load(d_results + 'dropout_class_probs_60.out.npy')
    drop_class_probs_30 = np.load(d_results + 'dropout_class_probs_30.out.npy')
    drop_class_probs_0 = np.load(d_results + 'dropout_class_probs_0.out.npy')
    drop_class_probs_330 = np.load(d_results + 'dropout_class_probs_330.out.npy')
    drop_class_probs_300 = np.load(d_results + 'dropout_class_probs_300.out.npy')
    drop_class_probs_270 = np.load(d_results + 'dropout_class_probs_270.out.npy')

    class_probs_90 = np.load(d_results + 'class_probs_90.out.npy')
    class_probs_60 = np.load(d_results + 'class_probs_60.out.npy')
    class_probs_30 = np.load(d_results + 'class_probs_30.out.npy')
    class_probs_0 = np.load(d_results + 'class_probs_0.out.npy')
    class_probs_330 = np.load(d_results + 'class_probs_330.out.npy')
    class_probs_300 = np.load(d_results + 'class_probs_300.out.npy')
    class_probs_270 = np.load(d_results + 'class_probs_270.out.npy')

    mcd_rotating_results = {90:drop_class_probs_90, 60:drop_class_probs_60, 30:drop_class_probs_30, 0:drop_class_probs_0, 330:drop_class_probs_330, 300:drop_class_probs_300, 270:drop_class_probs_270}
    rotating_results = {90:class_probs_90, 60:class_probs_60, 30:class_probs_30, 0:class_probs_0, 330:class_probs_330, 300:class_probs_300, 270:class_probs_270}

    mean = 0.1307
    std = 0.3081
    
    transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean), (std))])
    test_set = datasets.MNIST(root='./data', download=True, train=False, transform=transformer)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False)
    
    if type(element_idx) == int:
        img_filename = 'mnist_test_sample_' + str(element_idx) + '_label_' + str(test_set.targets[element_idx].item()) + '.png'
        plt.imsave(img_filename, test_set.data[element_idx], cmap='gray')
        print('\n0 ', class_probs_0[element_idx], class_probs_0[element_idx].argmax())
        print('30 ', class_probs_30[element_idx], class_probs_30[element_idx].argmax())
        print('330 ', class_probs_330[element_idx], class_probs_330[element_idx].argmax())
        print('60 ', class_probs_60[element_idx], class_probs_60[element_idx].argmax())
        print('300 ', class_probs_300[element_idx], class_probs_300[element_idx].argmax())
        print('90 ', class_probs_90[element_idx], class_probs_90[element_idx].argmax())
        print('270 ', class_probs_270[element_idx], class_probs_270[element_idx].argmax())
    elif type(element_idx) == list:
        for el in element_idx:
            plt.imsave('mnist_test_sample_' + str(el) + '_label_' + str(test_set.targets[el].item()) + '.png', test_set.data[el], cmap='gray')
            print("\n********* sample id {} gt label {} *********".format(el, str(test_set.targets[el].item())))
            print('0 ', class_probs_0[el], class_probs_0[el].argmax())
            print('30 ', class_probs_30[el], class_probs_30[el].argmax())
            print('330 ', class_probs_330[el], class_probs_330[el].argmax())
            print('60 ', class_probs_60[el], class_probs_60[el].argmax())
            print('300 ', class_probs_300[el], class_probs_300[el].argmax())
            print('90 ', class_probs_90[el], class_probs_90[el].argmax())
            print('270 ', class_probs_270[el], class_probs_270[el].argmax())

            #data = np.column_stack((drop_class_probs_90[el][1], drop_class_probs_90[el][5], drop_class_probs_90[el][7]))
            labels_ary = np.arange(10).tolist()
            
            for key, drop_class_probs in mcd_rotating_results.items():

                fig, axs = plt.subplots()
                axs.boxplot(drop_class_probs[el].transpose(), showmeans=True, meanline=True, labels=labels_ary, showfliers=False)
                ax = plt.gca()
                ax.set_ylim([-0.1, 1.1])
                ax.set_xlabel('Label')
                ax.set_ylabel('classification confidence')
                plt.title('Rotating MNIST test digit sample {} rotated by {} degrees'.format(el, key))
                plot_filename = 'boxplot_mnist_test_sample_{}_gt_label_{}_degrees_{}'.format(el, str(test_set.targets[el].item()), key)

                plt.savefig(plot_filename + '_.png')
                plt.savefig(plot_filename + '_.pdf')
                plt.close()

def cond_means_pandas():
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_theme(style="whitegrid")
    iris = sns.load_dataset("iris")

    # "Melt" the dataset to "long-form" or "tidy" representation
    iris = pd.melt(iris, "species", var_name="measurement")

    # Initialize the figure
    f, ax = plt.subplots()
    sns.despine(bottom=True, left=True)

    # Show each observation with a scatterplot
    sns.stripplot(x="value", y="measurement", hue="species",
                data=iris, dodge=True, alpha=.25, zorder=1)

    # Show the conditional means, aligning each pointplot in the
    # center of the strips by adjusting the width allotted to each
    # category (.8 by default) by the number of hue levels
    p1 = sns.pointplot(x="value", y="measurement", hue="species",
                data=iris, dodge=.8 - .8 / 3,
                join=False, palette="dark",
                markers="d", scale=.75, ci=None)

    # Improve the legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[3:], labels[3:], title="species",
            handletextpad=0, columnspacing=1,
            loc="lower right", ncol=3, frameon=True)
    fig2 = p1.get_figure()
    fig2.savefig("cond_means.pdf")

if __name__ == "__main__":
    d_results_p01 = '/home/fabrizio/research/mc_dropout/SPFlow/src/spn/experiments/RandomSPNs_layerwise/results/2022-04-08_11-29-28/results/'
    d_results_p02 = '/home/fabrizio/research/mc_dropout/SPFlow/src/spn/experiments/RandomSPNs_layerwise/results/2022-04-08_11-29-28/results/'
    gen_plot_conf_vs_acc_auto(d_results_p01, "acc_vs_conf_mcd_p_01")
    gen_plot_conf_vs_acc_auto(d_results_p02, "acc_vs_conf_mcd_p_02")
    sys.exit()

    sort_idxs = gen_plot_conf_vs_acc_corrupted_svhn('/home/fabrizio/research/mc_dropout/SPFlow/src/spn/experiments/RandomSPNs_layerwise/results/2022-05-15_00-45-40/model/post_hoc_results/svhn_c/',
    												filename='acc_vs_conf_mcd_p_02_SVHN_C_sort_new')
    plot_multiple_boxplots_svhn_c('/home/fabrizio/research/mc_dropout/SPFlow/src/spn/experiments/RandomSPNs_layerwise/results/2022-05-15_00-45-40/model/post_hoc_results/svhn_c/',
								  sort_idxs=sort_idxs)
    gen_plot_conf_vs_acc_svhn_c_single('/home/fabrizio/research/mc_dropout/SPFlow/src/spn/experiments/RandomSPNs_layerwise/results/2022-05-15_00-45-40/model/post_hoc_results/svhn_c/',
									   filename='SVHN_conf_v_acc_corr_single_')
    sys.exit()

	# gen_lls_histograms('/Users/fabrizio/Desktop/results/2022-05-16_16-10-50/model/likelihoods/', filename="data_lls_fmnist_lambda_0.pdf")
    # gen_lls_histograms('/Users/fabrizio/Desktop/results 2/2022-05-16_04-34-36/model/likelihoods/', filename="data_lls_fmnist_lambda_0_2.pdf")
    gen_lls_histograms('/Users/fabrizio/research/exps_cache/mcd/2022-06-02_23-13-30/results/likelihoods/', filename="data_lls_fmnist_lambda_0_2_j3_1.pdf")
    gen_lls_histograms('/Users/fabrizio/research/exps_cache/mcd/2022-06-03_04-57-49/results/likelihoods/', filename="data_lls_fmnist_lambda_0_2_j3_2.pdf")
    sys.exit()
    gen_histograms()
    # cond_means_pandas()
    # sort_idxs = gen_plot_conf_vs_acc_corrupted('/Users/fabrizio/Documents/2022-04-29_21-27-25_mnist-c/results/')
    # plot_multiple_boxplots_mnist_c('/Users/fabrizio/Documents/2022-04-29_21-27-25_mnist-c/results/', sort_idxs=sort_idxs)
    # gen_plot_conf_vs_acc_corrupted_single('/Users/fabrizio/Desktop/2022-05-11_16-12-31/results/', filename='conf_v_acc_corr_single_')
    # sys.exit()

    nn_accuracy_vs_confidence()
    #d_results = '/Users/fabrizio/Documents/2022-03-25_11-38-35/results/'
    # d_results = '/Users/fabrizio/Desktop/2022-04-04_20-09-44/results/'
    # d_results_p01 = ' /home/fabrizio/research/mc_dropout/SPFlow/src/spn/experiments/RandomSPNs_layerwise/results/2022-04-08_11-29-28/'
    # d_results_p02 = '/home/fabrizio/research/mc_dropout/SPFlow/src/spn/experiments/RandomSPNs_layerwise/results/2022-04-08_11-29-28/results/'
    # gen_plot_conf_vs_acc_auto(d_results_p01, "acc_vs_conf_mcd_p_01")
    # gen_plot_conf_vs_acc_auto(d_results_p02, "acc_vs_conf_mcd_p_02")
    # plot_multiple_boxplots_rotatiing_digits(d_results)

    #idx_ary = [0, 110, 340, 10, 345, 3, 234, 34, 56, 90, 765, 562, 10, 20, 40, 50, 60, 90]
    #plot_some_mnist_digits(element_idx=idx_ary, d_results=d_results)
    sys.exit()

    # gen_boxplots('/Users/fabrizio/Desktop/2022-03-25_11-38-35/results/')
    #plot_multiple_boxplots_rotatiing_digits('/Users/fabrizio/Documents/2022-03-25_11-38-35/results/')
    #gen_histograms()
    #gen_plot_conf_vs_acc()
    #sys.exit()

    class_probs_in_domain_train = np.load('/Users/fabrizio/Documents/2022-03-25_11-38-35/results/class_probs_in_domain_train.npy').max(axis=1)
    class_probs_in_domain_test = np.load('/Users/fabrizio/Documents/2022-03-25_11-38-35/results/class_probs_in_domain_test.npy').max(axis=1)
    class_probs_ood_train = np.load('/Users/fabrizio/Documents/2022-03-25_11-38-35/results/class_probs_ood_train.npy').max(axis=1)
    class_probs_ood_test = np.load('/Users/fabrizio/Documents/2022-03-25_11-38-35/results/class_probs_ood_test.npy').max(axis=1)

    df_in_train = pd.DataFrame({'class_probs_in_domain_train':class_probs_in_domain_train})
    df_in_test = pd.DataFrame({'class_probs_in_domain_test':class_probs_in_domain_test})
    df_ood_train = pd.DataFrame({'class_probs_ood_train':class_probs_ood_train})
    df_ood_test = pd.DataFrame({'class_probs_ood_test':class_probs_ood_test})

    data = pd.concat([df_in_train, df_in_test, df_ood_train, df_ood_test], ignore_index=True, axis=1)
    data = data.rename({0:'MNIST Train (In-domain)', 1:'MNIST Test (In-domain)', 2:'F-MNIST Train (OOD)', 3:'F-MNIST Test (OOD)'}, axis=1)

    #print(data)


    #data = np.column_stack((class_probs_in_domain_train, class_probs_in_domain_test, class_probs_ood_train, class_probs_ood_test))
    #data1 = pd.DataFrame({'in_train':class_probs_in_domain_train, 'in_test':class_probs_in_domain_test, 'ood_train':class_probs_ood_train, 'ood_test':class_probs_ood_test}, index=np.arange(60000))


    #p1 = sns.histplot(data=data, bins=10, multiple='layer')
    #fig1 = p1.get_figure()
    #fig1.savefig("confidence_histograms_mnist.pdf")


    class_probs_in_domain_train = np.load('/Users/fabrizio/Documents/2022-03-25_19-00-15/results/class_probs_in_domain_train.npy').max(axis=1)
    class_probs_in_domain_test = np.load('/Users/fabrizio/Documents/2022-03-25_19-00-15/results/class_probs_in_domain_test.npy').max(axis=1)
    class_probs_ood_train = np.load('/Users/fabrizio/Documents/2022-03-25_19-00-15/results/class_probs_ood_train.npy').max(axis=1)
    class_probs_ood_test = np.load('/Users/fabrizio/Documents/2022-03-25_19-00-15/results/class_probs_ood_test.npy').max(axis=1)


    df_in_train = pd.DataFrame({'class_probs_in_domain_train':class_probs_in_domain_train})
    df_in_test = pd.DataFrame({'class_probs_in_domain_test':class_probs_in_domain_test})
    df_ood_train = pd.DataFrame({'class_probs_ood_train':class_probs_ood_train})
    df_ood_test = pd.DataFrame({'class_probs_ood_test':class_probs_ood_test})

    data_ary = [df_in_train, df_in_test, df_ood_train, df_ood_test]
    #data_ary = [df_in_train, df_ood_train]

    data = pd.concat([df_in_train, df_in_test, df_ood_train, df_ood_test], ignore_index=True, axis=1)
    data = data.rename({0:'F-MNIST Train (In-domain)', 1:'F-MNIST Test (In-domain)', 2:'MNIST Train (OOD)', 3:'MNIST Test (OOD)'}, axis=1)

    #data = pd.concat([df_in_train, df_ood_train], ignore_index=True, axis=1)
    #data = data.rename({0:'F-MNIST Train (In-domain)', 1:'MNIST Train (OOD)'}, axis=1)
    #print(data)

    #df_melted = data.melt(var_name='column')

    #p2 = sns.histplot(data=data, bins=10, multiple='layer')
    #fig2 = p1.get_figure()
    #fig2.savefig("confidence_histograms_fmnist.pdf")

    #sns.set(color_codes=True)
    #sns.set(style="white", palette="muted")

    #sns.distplot(data)

    #plt.figure()
    #for col in data:
    #    sns.distplot(col, hist=True)
    #plt.savefig('test_distplot.pdf')

    print(data)

    #sns.color_palette("Paired")
    palette ={"F-MNIST Train (In-domain)": "green", "F-MNIST Test (In-domain)": "C0", "MNIST Train (OOD)": "yellow", "MNIST Test (OOD)": "red"}
    #p3 = sns.histplot(data=data, x='value', hue='column',  bins=50, multiple='layer', kde=True)
    p3 = sns.histplot(data=data, bins=50, multiple='layer', palette=palette, cbar_kws={'alpha':0.3})
    #p3 = sns.distplot(df_melted,  bins=50)
    p3.set(xlabel='Classification confidence', ylabel='# samples')
    #p3 = sns.histplot(data=data, bins=20, multiple='layer')
    fig3 = p3.get_figure()
    fig3.savefig("confidence_histograms_fmnist_3.pdf")


    #plt.hist(class_probs_in_domain_train, label="In-domain (Train)", alpha=0.5, bins=30, color='green')
    #plt.hist(class_probs_ood_train, label="OOD (Train)", alpha=0.5, bins=30, color='orange')
    #plt.hist(class_probs_in_domain_test, label="In-domain (Test)", alpha=0.5, bins=30, color='blue')
    #plt.hist(class_probs_ood_test, label="OOD (Test)", alpha=0.5, bins=30, color='red')

    #plt.ylabel('# samples')
    #plt.xlabel('Classification confidence')

    #plt.legend(loc=0)
    #plt.title("histogram")

    #plt.savefig('histo_tmp.pdf')
    #plt.close()

