import os
import random
import sys
import time

import imageio
import numpy as np
import skimage
import torch
import torchvision
from torch import nn
from torchvision import datasets, transforms

from spn.experiments.RandomSPNs_layerwise.distributions import RatNormal
from spn.experiments.RandomSPNs_layerwise.rat_spn import RatSpn, RatSpnConfig
from spn.algorithms.layerwise.layers import CrossProduct, Sum
from spn.algorithms.layerwise.distributions import Bernoulli

import matplotlib.pyplot as plt
import matplotlib

from torch.utils.data import Dataset, ConcatDataset
from PIL import Image
import datetime

import pdb
from scipy.stats import entropy

import spn.experiments.RandomSPNs_layerwise.mnist_c.corruptions as corruptions
from sklearn import datasets as sk_datasets

import csv
import subprocess

from icecream import ic

def maybe_download_debd():
    if os.path.isdir('../data/debd'):
        return
    subprocess.run(['git', 'clone', 'https://github.com/arranger1044/DEBD', '../data/debd'])
    wd = os.getcwd()
    os.chdir('../data/debd')
    subprocess.run(['git', 'checkout', '80a4906dcf3b3463370f904efa42c21e8295e85c'])
    subprocess.run(['rm', '-rf', '.git'])
    os.chdir(wd)


def load_debd(name, dtype='int32'):
    """Load one of the twenty binary density esimtation benchmark datasets."""

    maybe_download_debd()

    data_dir = '../data/debd'

    train_path = os.path.join(data_dir, 'datasets', name, name + '.train.data')
    test_path = os.path.join(data_dir, 'datasets', name, name + '.test.data')
    valid_path = os.path.join(data_dir, 'datasets', name, name + '.valid.data')

    reader = csv.reader(open(train_path, 'r'), delimiter=',')
    train_x = np.array(list(reader)).astype(dtype)

    reader = csv.reader(open(test_path, 'r'), delimiter=',')
    test_x = np.array(list(reader)).astype(dtype)

    reader = csv.reader(open(valid_path, 'r'), delimiter=',')
    valid_x = np.array(list(reader)).astype(dtype)

    return train_x, test_x, valid_x


DEBD = ['accidents', 'ad', 'baudio', 'bbc', 'bnetflix', 'book', 'c20ng', 'cr52', 'cwebkb', 'dna', 'jester', 'kdd',
        'kosarek', 'moviereview', 'msnbc', 'msweb', 'nltcs', 'plants', 'pumsb_star', 'tmovie', 'tretail', 'voting']

DEBD_shapes = {
    'accidents': dict(train=(12758, 111), valid=(2551, 111), test=(1700, 111)),
    'ad': dict(train=(2461, 1556), valid=(491, 1556), test=(327, 1556)),
    'baudio': dict(train=(15000, 100), valid=(3000, 100), test=(2000, 100)),
    'bbc': dict(train=(1670, 1058), valid=(330, 1058), test=(225, 1058)),
    'bnetflix': dict(train=(15000, 100), valid=(3000, 100), test=(2000, 100)),
    'book': dict(train=(8700, 500), valid=(1739, 500), test=(1159, 500)),
    'c20ng': dict(train=(11293, 910), valid=(3764, 910), test=(3764, 910)),
    'cr52': dict(train=(6532, 889), valid=(1540, 889), test=(1028, 889)),
    'cwebkb': dict(train=(2803, 839), valid=(838, 839), test=(558, 839)),
    'dna': dict(train=(1600, 180), valid=(1186, 180), test=(400, 180)),
    'jester': dict(train=(9000, 100), valid=(4116, 100), test=(1000, 100)),
    'kdd': dict(train=(180092, 64), valid=(34955, 64), test=(19907, 64)),
    'kosarek': dict(train=(33375, 190), valid=(6675, 190), test=(4450, 190)),
    'moviereview': dict(train=(1600, 1001), valid=(250, 1001), test=(150, 1001)),
    'msnbc': dict(train=(291326, 17), valid=(58265, 17), test=(38843, 17)),
    'msweb': dict(train=(29441, 294), valid=(5000, 294), test=(3270, 294)),
    'nltcs': dict(train=(16181, 16), valid=(3236, 16), test=(2157, 16)),
    'plants': dict(train=(17412, 69), valid=(3482, 69), test=(2321, 69)),
    'pumsb_star': dict(train=(12262, 163), valid=(2452, 163), test=(1635, 163)),
    'tmovie': dict(train=(4524, 500), valid=(591, 500), test=(1002, 500)),
    'tretail': dict(train=(22041, 135), valid=(4408, 135), test=(2938, 135)),
    'voting': dict(train=(1214, 1359), valid=(350, 1359), test=(200, 1359)),
}

DEBD_display_name = {
    'accidents': 'accidents',
    'ad': 'ad',
    'baudio': 'audio',
    'bbc': 'bbc',
    'bnetflix': 'netflix',
    'book': 'book',
    'c20ng': '20ng',
    'cr52': 'reuters-52',
    'cwebkb': 'web-kb',
    'dna': 'dna',
    'jester': 'jester',
    'kdd': 'kdd-2k',
    'kosarek': 'kosarek',
    'moviereview': 'moviereview',
    'msnbc': 'msnbc',
    'msweb': 'msweb',
    'nltcs': 'nltcs',
    'plants': 'plants',
    'pumsb_star': 'pumsb-star',
    'tmovie': 'each-movie',
    'tretail': 'retail',
    'voting': 'voting'}


class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].shape[0] == tensor.shape[0] for tensor in tensors)
        self.tensors = tensors
        self.transform = transform
        self.data = self.tensors[0]
        self.targets = self.tensors[1]

    def __getitem__(self, index):
        x = self.tensors[0][index]
        x = Image.fromarray(x.numpy(), mode=None)

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].shape[0]

class Simple2dTensorDataset(Dataset):
    """TensorDataset for 2d datasets.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].shape[0] == tensor.shape[0] for tensor in tensors)
        self.tensors = tensors
        self.transform = transform
        self.data = self.tensors[0]
        self.targets = self.tensors[1]

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].shape[0]

def round_and_astype(x):
    return np.round(x).astype(np.uint8)

def one_hot(vector):
    result = np.zeros((vector.size, vector.max() + 1))
    result[np.arange(vector.size), vector] = 1
    return result


def time_delta_now(t_start: float) -> str:
    """
    Convert a timestamp into a human readable timestring.
    Args:
        t_start (float): Timestamp.

    Returns:
        Human readable timestring.
    """
    a = t_start
    b = time.time()  # current epoch time
    c = b - a  # seconds
    days = round(c // 86400)
    hours = round(c // 3600 % 24)
    minutes = round(c // 60 % 60)
    seconds = round(c % 60)
    millisecs = round(c % 1 * 1000)
    return f"{days} days, {hours} hours, {minutes} minutes, {seconds} seconds, {millisecs} milliseconds"


def count_params(model: torch.nn.Module) -> int:
    """
    Count the number of parameters in a modinference

    Args:
        model (torch.nn.Module): PyTorch model.

    Returns:
        int: Number of learnable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_2moons_loaders(use_cuda, device, batch_size, fourD=True, n_samples=1000):
    """
    Get the 2 moons pytorch data loader.

    Args:
        use_cuda: Use cuda flag.
    """
    kwargs = {"num_workers": 8, "pin_memory": True} if use_cuda else {}

    test_batch_size = batch_size

    noisy_moons = sk_datasets.make_moons(n_samples=n_samples, noise=.05)[0]
    noisy_moons_neg = noisy_moons + 4

    # standardization
    noisy_moons -= np.mean(noisy_moons, axis=0)
    noisy_moons /= np.std(noisy_moons, axis=0)

    noisy_moons_neg -= np.mean(noisy_moons_neg, axis=0)
    noisy_moons_neg /= np.std(noisy_moons_neg, axis=0)

    print("Noisy moon mean {}, std {}".format(noisy_moons.mean(axis=0), noisy_moons.std(axis=0)))
    print("Noisy moon neg mean {}, std {}".format(noisy_moons_neg.mean(axis=0), noisy_moons_neg.std(axis=0)))

    training_data = np.row_stack((noisy_moons, noisy_moons_neg))
    if fourD:
        training_data = np.column_stack((training_data, training_data))
    training_targets = np.concatenate((np.ones(noisy_moons.shape[0]), np.zeros(noisy_moons_neg.shape[0])))
    # print(training_targets)
    # training_targets = one_hot(training_targets.astype(int))
    # print(training_targets)

    test_data = training_data
    test_targets = training_targets

    training_set = Simple2dTensorDataset(tensors=[torch.tensor(training_data, dtype=torch.float32), torch.tensor(training_targets, dtype=torch.int64)])
    test_set = Simple2dTensorDataset(tensors=[torch.tensor(test_data, dtype=torch.float32), torch.tensor(test_targets, dtype=torch.int64)])

    assert batch_size <= training_data.shape[0]

    # Train data loader
    train_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )

    assert batch_size <= test_data.shape[0]

    # Test data loader
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=test_batch_size,
        shuffle=False,
        **kwargs,
    )
    return train_loader, test_loader

def get_4gaussians_loaders(use_cuda, device, batch_size, n_samples=1000):
    """
    Get the 2 moons pytorch data loader.

    Args:
        use_cuda: Use cuda flag.
    """
    kwargs = {"num_workers": 8, "pin_memory": True} if use_cuda else {}

    test_batch_size = batch_size

    train_4g = np.c_[np.r_[np.random.normal(5, 0.2, (n_samples, 1))], np.r_[np.random.normal(10, 0.2, (n_samples, 1))],
                            np.r_[np.random.normal(15, 0.2, (n_samples, 1))], np.r_[np.random.normal(20, 0.2, (n_samples, 1))]]
    train_4g_neg = np.c_[np.r_[np.random.normal(-5, 0.2, (n_samples, 1))], np.r_[np.random.normal(-10, 0.2, (n_samples, 1))],
                            np.r_[np.random.normal(-15, 0.2, (n_samples, 1))], np.r_[np.random.normal(-30, 0.2, (n_samples, 1))]]

    # print(train_4g)
    # print(train_4g_neg)
    print(train_4g.shape)
    print(train_4g_neg.shape)

    # standardization
    train_4g -= np.mean(train_4g, axis=0)
    train_4g /= np.std(train_4g, axis=0)

    train_4g_neg -= np.mean(train_4g_neg, axis=0)
    train_4g_neg /= np.std(train_4g_neg, axis=0)

    print("4 gaussians mean {}, std {}".format(train_4g.mean(axis=0), train_4g.std(axis=0)))
    print("4 gaussians neg mean {}, std {}".format(train_4g_neg.mean(axis=0), train_4g_neg.std(axis=0)))

    training_data = np.row_stack((train_4g, train_4g_neg))
    training_targets = np.concatenate((np.ones(train_4g.shape[0]), np.zeros(train_4g_neg.shape[0])))

    test_data = training_data
    test_targets = training_targets

    training_set = Simple2dTensorDataset(tensors=[torch.tensor(training_data, dtype=torch.float32), torch.tensor(training_targets, dtype=torch.int64)])
    test_set = Simple2dTensorDataset(tensors=[torch.tensor(test_data, dtype=torch.float32), torch.tensor(test_targets, dtype=torch.int64)])

    assert batch_size <= training_data.shape[0]

    # Train data loader
    train_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )

    assert batch_size <= test_data.shape[0]

    # Test data loader
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=test_batch_size,
        shuffle=False,
        **kwargs,
    )
    return train_loader, test_loader

def get_2gaussians_loaders(use_cuda, device, batch_size, fourD=True, n_samples=1000):
    """
    Get the 2 moons pytorch data loader.

    Args:
        use_cuda: Use cuda flag.
    """
    kwargs = {"num_workers": 8, "pin_memory": True} if use_cuda else {}

    test_batch_size = batch_size

    train_2g = np.c_[np.r_[np.random.normal(5, 1, (n_samples, 2))]]
    train_2g_neg = np.c_[np.r_[np.random.normal(10, 1, (n_samples, 2))]]

    # standardization
    train_2g -= np.mean(train_2g, axis=0)
    train_2g /= np.std(train_2g, axis=0)

    train_2g_neg -= np.mean(train_2g_neg, axis=0)
    train_2g_neg /= np.std(train_2g_neg, axis=0)

    print("2gaussians mean {}, std {}".format(train_2g.mean(axis=0), train_2g.std(axis=0)))
    print("2gaussians neg mean {}, std {}".format(train_2g_neg.mean(axis=0), train_2g_neg.std(axis=0)))

    training_data = np.row_stack((train_2g, train_2g_neg))
    if fourD:
        #training_data = np.column_stack((training_data, training_data))
        training_data = np.column_stack((np.c_[np.r_[np.random.normal(0, 1, (n_samples * 2, 2))]],
                                         np.c_[np.r_[np.random.normal(0, 1, (n_samples * 2, 2))]]))
    training_targets = np.concatenate((np.ones(train_2g.shape[0]), np.zeros(train_2g_neg.shape[0])))
    # print(training_targets)
    # training_targets = one_hot(training_targets.astype(int))
    # print(training_targets)

    test_data = training_data
    test_targets = training_targets

    training_set = Simple2dTensorDataset(tensors=[torch.tensor(training_data, dtype=torch.float32), torch.tensor(training_targets, dtype=torch.int64)])
    test_set = Simple2dTensorDataset(tensors=[torch.tensor(test_data, dtype=torch.float32), torch.tensor(test_targets, dtype=torch.int64)])

    assert batch_size <= training_data.shape[0]

    # Train data loader
    train_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )

    assert batch_size <= test_data.shape[0]

    # Test data loader
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=test_batch_size,
        shuffle=False,
        **kwargs,
    )
    return train_loader, test_loader

def get_d_mnist_loaders(use_cuda, device, batch_size):
    """
    Get the MNIST pytorch data loader.

    Args:
        use_cuda: Use cuda flag.
    """
    kwargs = {"num_workers": 8, "pin_memory": True} if use_cuda else {}

    test_batch_size = batch_size

    transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # Train data loader
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST("../data", train=True, download=True, transform=transformer),
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )

    # Test data loader
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("../data", train=False, transform=transformer),
        batch_size=test_batch_size,
        shuffle=False,
        **kwargs,
    )
    return train_loader, test_loader


def get_f_mnist_loaders(use_cuda, device, batch_size):
    """
    Get the Fashion-MNIST pytorch data loader.

    Args:
        use_cuda: Use cuda flag.

    """
    kwargs = {"num_workers": 8, "pin_memory": True} if use_cuda else {}

    test_batch_size = batch_size

    transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860), (0.3530))])
    # Train data loader
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST("../data", train=True, download=True, transform=transformer),
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )

    # Test data loader
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST("../data", train=False, transform=transformer),
        batch_size=test_batch_size,
        shuffle=False,
        **kwargs,
    )
    return train_loader, test_loader

def get_emnist_loaders(use_cuda, device, batch_size):
    """
    Get the EMNIST pytorch data loader.

    Args:
        use_cuda: Use cuda flag.

    """
    kwargs = {"num_workers": 8, "pin_memory": True} if use_cuda else {}

    test_batch_size = batch_size

    transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1733), (0.3317))])
    # Train data loader
    train_loader = torch.utils.data.DataLoader(
        datasets.EMNIST("../data", train=True, download=True, split='mnist', transform=transformer),
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )

    # Test data loader
    test_loader = torch.utils.data.DataLoader(
        datasets.EMNIST("../data", train=False, split='mnist', transform=transformer),
        batch_size=test_batch_size,
        shuffle=False,
        **kwargs,
    )
    return train_loader, test_loader

def get_kmnist_loaders(use_cuda, device, batch_size):
    """
    Get the KMNIST pytorch data loader.

    Args:
        use_cuda: Use cuda flag.

    """
    kwargs = {"num_workers": 8, "pin_memory": True} if use_cuda else {}

    test_batch_size = batch_size

    transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1918), (0.3483))])
    # Train data loader
    train_loader = torch.utils.data.DataLoader(
        datasets.KMNIST("../data", train=True, download=True, transform=transformer),
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )

    # Test data loader
    test_loader = torch.utils.data.DataLoader(
        datasets.KMNIST("../data", train=False, transform=transformer),
        batch_size=test_batch_size,
        shuffle=False,
        **kwargs,
    )
    return train_loader, test_loader

def get_cifar_loaders(use_cuda, device, batch_size):
    """
    Get the CIFAR10 pytorch data loader.

    Args:
        use_cuda: Use cuda flag.

    """
    kwargs = {"num_workers": 8, "pin_memory": True} if use_cuda else {}
    cifar10_transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

    cifar10_train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='../data', train=True, download=True, transform=cifar10_transformer),
        batch_size=batch_size,
        shuffle=True,
        **kwargs
    )

    cifar10_test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='../data', train=False, download=True, transform=cifar10_transformer),
        batch_size=batch_size,
        shuffle=False,
        **kwargs
    )
    return cifar10_train_loader, cifar10_test_loader

def get_nltcs_loaders(use_cuda, device, batch_size):
    """
    Get the NLTCS pytorch data loader.

    Args:
        use_cuda: Use cuda flag.

    """
    kwargs = {"num_workers": 8, "pin_memory": True} if use_cuda else {}

    test_batch_size = batch_size

    train, test, valid = load_debd('nltcs')
    print(train.shape)
    print(train[:, :15].shape)

    training_set = Simple2dTensorDataset(
        tensors=[torch.tensor(train[:, :15], dtype=torch.float32), torch.tensor(train[:, 15], dtype=torch.int64)])
    test_set = Simple2dTensorDataset(
        tensors=[torch.tensor(test[:, :15], dtype=torch.float32), torch.tensor(test[:, 15], dtype=torch.int64)])

    assert batch_size <= train.shape[0]

    # Train data loader
    train_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )

    assert batch_size <= test.shape[0]

    # Test data loader
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=test_batch_size,
        shuffle=False,
        **kwargs,
    )
    return train_loader, test_loader

def get_msnbc_loaders(use_cuda, device, batch_size):
    """
    Get the MSNBC pytorch data loader.

    Args:
        use_cuda: Use cuda flag.

    """
    kwargs = {"num_workers": 8, "pin_memory": True} if use_cuda else {}

    test_batch_size = batch_size

    train, test, valid = load_debd('msnbc')

    training_set = Simple2dTensorDataset(
        tensors=[torch.tensor(train[:, :15], dtype=torch.float32), torch.tensor(train[:, 15], dtype=torch.int64)])
    test_set = Simple2dTensorDataset(
        tensors=[torch.tensor(test[:, :15], dtype=torch.float32), torch.tensor(test[:, 15], dtype=torch.int64)])

    assert batch_size <= train.shape[0]

    # Train data loader
    train_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )

    assert batch_size <= test.shape[0]

    # Test data loader
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=test_batch_size,
        shuffle=False,
        **kwargs,
    )
    return train_loader, test_loader

def get_svhn_loaders(use_cuda, device, batch_size, add_extra=True):
    """
    Get the SVHN pytorch data loader.

    Args:
        use_cuda: Use cuda flag.

    """
    kwargs = {"num_workers": 8, "pin_memory": True} if use_cuda else {}
    svhn_transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4377, 0.4438, 0.4728), (0.198, 0.201, 0.197))])

    if add_extra:
        train_dataset = ConcatDataset([datasets.SVHN(root='../data', split='train', download=True, transform=svhn_transformer),
                                       datasets.SVHN(root='../data', split='extra', download=True, transform=svhn_transformer)])
    else:
        train_dataset = datasets.SVHN(root='../data', split='train', download=True, transform=svhn_transformer)

    svhn_train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **kwargs
    )

    svhn_test_loader = torch.utils.data.DataLoader(
        datasets.SVHN(root='../data', split='test', download=True, transform=svhn_transformer),
        batch_size=batch_size,
        shuffle=False,
        **kwargs
    )

    return svhn_train_loader, svhn_test_loader


def get_data_loaders(use_cuda, device, batch_size, dataset='mnist'):
    if dataset == 'mnist':
        return get_d_mnist_loaders(use_cuda, device, batch_size)
    elif dataset == 'fmnist':
        return get_f_mnist_loaders(use_cuda, device, batch_size)
    elif dataset == 'kmnist':
        return get_kmnist_loaders(use_cuda, device, batch_size)
    elif dataset == 'emnist':
        return get_emnist_loaders(use_cuda, device, batch_size)
    elif dataset == 'cifar':
        return get_cifar_loaders(use_cuda, device, batch_size)
    elif dataset == 'svhn':
        return get_svhn_loaders(use_cuda, device, batch_size)
    elif dataset == '2moons':
        return get_2moons_loaders(use_cuda, device, batch_size, n_samples=1000)
    elif dataset == '2gaussians':
        return get_2gaussians_loaders(use_cuda, device, batch_size, n_samples=1000)
    elif dataset == '4gaussians':
        return get_4gaussians_loaders(use_cuda, device, batch_size, n_samples=100)
    elif dataset == 'nltcs':
        return get_nltcs_loaders(use_cuda, device, batch_size)
    elif dataset == 'msnbc':
        return get_msnbc_loaders(use_cuda, device, batch_size)


def get_data_flatten_shape(data_loader):
    if isinstance(data_loader.dataset, ConcatDataset):
        return (data_loader.dataset.cumulative_sizes[-1],
                torch.prod(torch.tensor(data_loader.dataset.datasets[0].data.shape[1:])).int().item())
    return (data_loader.dataset.data.shape[0],
            torch.prod(torch.tensor(data_loader.dataset.data.shape[1:])).int().item())


def make_spn(S, I, R, D, dropout, device, F=28 ** 2, C=10, leaf_distribution=RatNormal) -> RatSpn:
    """Construct the RatSpn"""
    print(leaf_distribution)

    # Setup RatSpnConfig
    config = RatSpnConfig()
    config.F = F
    config.R = R
    config.D = D
    config.I = I
    config.S = S
    config.C = C
    config.dropout = dropout
    config.leaf_base_class = leaf_distribution
    config.leaf_base_kwargs = {}
    #config.leaf_base_kwargs = {"min_sigma": 0.1, "max_sigma": 2.0}
    #config.leaf_base_kwargs = {"min_sigma": 0.1, "max_sigma": 2.0, "min_mean": -1., "max_mean": 7.}

    # Construct RatSpn from config
    model = RatSpn(config)

    model = model.to(device)
    model.train()

    print("Using device:", device)
    print("Dropout SPN: ", dropout)
    return model

def get_dataset_display_name(dataset_name):
    display_names = {}
    display_names['fmnist'] = 'F-MNIST'
    display_names['mnist'] = 'MNIST'
    display_names['kmnist'] = 'K-MNIST'
    display_names['emnist'] = 'E-MNIST'
    display_names['cifar'] = 'CIFAR'
    display_names['svhn'] = 'SVHN'
    display_names['2moons'] = '2 moons'
    display_names['2gaussians'] = '2 gaussians'
    display_names['4guassians'] = '4 gaussians'
    display_names['nltcs'] = 'NLTCS'
    display_names['msnbc'] = 'MSNBC'
    return display_names[dataset_name]

def get_other_dataset_name(training_dataset):
    if training_dataset == 'mnist':
        return 'fmnist'
    elif training_dataset == 'fmnist':
        return 'mnist'
    elif training_dataset == 'kmnist':
        return 'emnist'
    elif training_dataset == 'emnist':
        return 'kmnist'
    elif training_dataset == 'cifar':
        return 'svhn'
    elif training_dataset == 'svhn':
        return 'cifar'
    elif training_dataset == '2moons':
        return '2gaussians'
    elif training_dataset == '2gaussians':
        return '2moons'
    elif training_dataset == '4gaussians':
        return '2moons'
    elif training_dataset == 'nltcs':
        return 'msnbc'

def plot_sum_weights(model_dir=None, training_dataset=None, dropout_inference=None, n_mcd_passes=100, batch_size=512,
                     rat_S=20, rat_I=20, rat_D=5, rat_R=5):
    import pandas as pd
    import seaborn as sns

    dev = sys.argv[1]
    device = torch.device("cuda:0")
    use_cuda = True
    torch.cuda.benchmark = True

    d = model_dir + "post_hoc_results/plots/"
    ensure_dir(d)
    train_loader, test_loader = get_data_loaders(use_cuda, batch_size=batch_size, device=device, dataset=training_dataset)
    n_features = get_data_flatten_shape(train_loader)[1]
    if training_dataset in DEBD:
        leaves = Bernoulli
        rat_C = 2
    else:
        leaves = RatNormal
        rat_C = 10
    print(leaves)
    model = make_spn(S=rat_S, I=rat_I, D=rat_D, R=rat_R, device=dev, dropout=dropout_inference, F=n_features,
                     C=rat_C, leaf_distribution=leaves)
    print(model)

    checkpoint = torch.load(model_dir + 'checkpoint.tar')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    sum_layer_idx = 0


    for hn in model._inner_layers:
        if isinstance(hn, Sum):
            sum_layer_idx += 1
            filename = "sum_node_weights_l_{}_histogram.pdf".format(sum_layer_idx)
            norm_weights = torch.nn.functional.softmax(hn.weights, dim=1)
            print(norm_weights.shape)

            # num of sum node weights greather than 0.4 for each sum node
            weight_threshold = 0.4

            big_weights_nodes = torch.any(norm_weights > weight_threshold, dim=1)
            n_big_weights_nodes = big_weights_nodes.sum().item()
            big_weights_nodes_ratio = n_big_weights_nodes / torch.prod(torch.tensor(big_weights_nodes.shape)).item()

            big_weights = (norm_weights > weight_threshold)
            n_big_weights = big_weights.sum().item()
            big_weights_ratio = n_big_weights / torch.prod(torch.tensor(big_weights.shape)).item()

            norm_weights_flatten = norm_weights.flatten().cpu().detach().numpy()
            # nw = pd.DataFrame({'Sum node wights': norm_weights}, index=[0])
            # nw = nw.rename({0: 'Sum node weights'})
            p3 = sns.histplot(data=norm_weights_flatten, stat="density", element="bars", common_norm=False,
                              color='navy')
            additional_text = "{:.2f}% of sum nodes has one (or more) weight > {}".format(big_weights_nodes_ratio * 100,
                                                                                              weight_threshold)
            additional_text_2 = "{:.2f}% of weights is > {}".format(
                big_weights_ratio * 100,
                weight_threshold)

            p3.text(0.1, 200, additional_text, fontsize=9, color='blue')
            p3.text(0.1, 600, additional_text_2, fontsize=9, color='blue')

            p3.set(xlabel='weight value', ylabel='# weights')
            p3.set_title("{}\u2070 sum node layer".format(sum_layer_idx))
            # p3.set_xlim(x_lim_left, x_lim_right)
            fig3 = p3.get_figure()
            fig3.savefig(d + filename)
            plt.close()


def evaluate_corrupted_svhn(model_dir=None, dropout_inference=None, n_mcd_passes=100, batch_size=512,
                  rat_S=20, rat_I=20, rat_D=5, rat_R=5, corrupted_svhn_dir=''):

    from imagecorruptions import corrupt, get_corruption_names

    import pandas as pd
    import seaborn as sns

    dev = sys.argv[1]
    device = torch.device("cuda:0")
    use_cuda = True
    torch.cuda.benchmark = True

    d = model_dir + "post_hoc_results/svhn_c/mcd_p_{}/".format(str(dropout_inference).replace('.', ''))
    ensure_dir(d)
    train_loader, test_loader = get_data_loaders(use_cuda, batch_size=batch_size, device=device,
                                                 dataset='svhn')
    n_features = get_data_flatten_shape(train_loader)[1]
    leaves = RatNormal
    rat_C = 10
    model = make_spn(S=rat_S, I=rat_I, D=rat_D, R=rat_R, device=dev, dropout=dropout_inference, F=n_features,
                     C=rat_C, leaf_distribution=leaves)

    # new models
    # checkpoint = torch.load(model_dir + 'checkpoint.tar')
    # model.load_state_dict(checkpoint['model_state_dict'])
    # old models
    model.load_state_dict(torch.load(model_dir + 'model.pt'))
    model.eval()

    results_dict = {}
    for corruption in get_corruption_names('common'):
        for cl in range(5):
            cl += 1
            print("Corruption {} Level {}".format(corruption, cl))
            results_dict['c_{}_l{}'.format(corruption, cl)] = evaluate_model_corrupted_svhn(model, device, test_loader,
                                                                                             "Test DROP corrupted SVHN C {} L {}".format(
                                                                                                 corruption, cl),
                                                                                             dropout_inference=dropout_inference,
                                                                                             n_dropout_iters=n_mcd_passes,
                                                                                             output_dir=d,
                                                                                             corruption=corruption,
                                                                                             corruption_level=cl,
                                                                                             corrupted_svhn_dir=corrupted_svhn_dir)
            np.save(d + 'dropout_class_probs_c_{}_l{}'.format(corruption, cl),
                    results_dict['c_{}_l{}'.format(corruption, cl)][0].cpu().detach().numpy())
            np.save(d + 'class_probs_c_{}_l{}'.format(corruption, cl),
                    results_dict['c_{}_l{}'.format(corruption, cl)][1].cpu().detach().numpy())


def test_closed_form(model_dir=None, training_dataset=None, dropout_inference=None, batch_size=20,
                  rat_S=20, rat_I=20, rat_D=5, rat_R=5, rotation=None, model=None):
    ic(training_dataset)
    ic(rotation)
    # dev = sys.argv[1]
    # device = torch.device("cuda:0")
    device = sys.argv[1]
    use_cuda = True
    torch.cuda.benchmark = True

    d = model_dir + "post_hoc_results/closed_form/"
    ensure_dir(d)
    train_loader, test_loader = get_data_loaders(use_cuda, batch_size=batch_size, device=device,
                                                 dataset=training_dataset)
    n_features = get_data_flatten_shape(train_loader)[1]
    if training_dataset in DEBD:
        leaves = Bernoulli
        rat_C = 2
    else:
        leaves = RatNormal
        rat_C = 10

    if not model:
        model = make_spn(S=rat_S, I=rat_I, D=rat_D, R=rat_R, device=device, dropout=dropout_inference, F=n_features,
                         C=rat_C, leaf_distribution=leaves)

        checkpoint = torch.load(model_dir + 'checkpoint.tar')
        model.load_state_dict(checkpoint['model_state_dict'])
        # old models
        # model.load_state_dict(torch.load(model_dir + 'model.pt'))
        model.eval()
        device = torch.device("cuda")
        model.to(device)
        print(model)
        # breakpoint()

    tag = "Testing Closed Form dropout: "

    loss_ce = 0
    loss_nll = 0
    data_ll = []
    data_ll_super = []  # pick it from the label-th head
    data_ll_unsup = []  # pick the max one
    class_probs = torch.zeros((get_data_flatten_shape(test_loader)[0], model.config.C)).to(device)
    correct = 0

    mean = 0.1307
    std = 0.3081

    if model.config.C == 2:
        criterion = nn.BCELoss(reduction="sum")
    else:
        criterion = nn.CrossEntropyLoss(reduction="sum")
    # criterion = nn.NLLLoss(reduction="sum")

    # max_std_eq_max_conds = 0

    with torch.no_grad():
        t_start = time.time()
        for batch_index, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            if rotation:
                data = torchvision.transforms.functional.rotate(data.reshape(-1, 1, 28, 28), rotation,
                                                                fill=-mean / std).reshape(-1, 28, 28)
            data = data.view(data.shape[0], -1)

            output, stds, ll_x, var_x = model(data, test_dropout=True, dropout_inference=dropout_inference, dropout_cf=True)
            if batch_index == 0:
                output_res = output.detach().cpu().numpy()
                var_res = stds.detach().cpu().numpy()
                ll_x_res = ll_x.detach().cpu().numpy().flatten()
                var_x_res = var_x.detach().cpu().numpy().flatten()
            else:
                output_res = np.vstack((output_res, output.detach().cpu().numpy()))
                var_res = np.vstack((var_res, stds.detach().cpu().numpy()))
                ll_x_res = np.vstack((ll_x_res, ll_x.detach().cpu().numpy().flatten()))
                var_x_res = np.vstack((var_x_res, var_x.detach().cpu().numpy().flatten()))
            # output = model(data, test_dropout=False, dropout_inference=0.0, dropout_cf=False)
            #ic(output.exp())
            # ic(output)
            #ic(stds)
            # ic(stds.gather(1, output.argmax(dim=1).view(-1, 1)).mean())
            # ic(output.max(dim=1)[0].exp().mean())
            # ic((output - torch.logsumexp(output, dim=1, keepdims=True)).exp())
            # ic(output.max(dim=1))
            # ic(output.max(dim=1)[0].exp())
            #ic(output.argmax(dim=1))
            # ic(stds.argmin(dim=1))
            # ic(stds.argmax(dim=1))
            #ic(target)
            # ic(stds)
            # # ic(stds.exp())
            # # ic(stds.gather(1, output.argmax(dim=1).view(-1,1)))
            # # ic(stds.gather(1, output.argmax(dim=1).view(-1, 1)).exp())
            # ic(stds.gather(1, output.argmax(dim=1).view(-1, 1)).mean())
            # ic(stds.mean())
            # breakpoint()
            # max_std_eq_max_conds += (output.argmax(dim=1) == stds.argmax(dim=1)).sum().item()
            # ic(torch.where(output.max(dim=1)[0].exp().isnan(), torch.tensor([0.0]).to(output.device), output.max(dim=1)[0].exp()).mean())
            # ic((torch.where(stds.isnan(), torch.tensor([0.0]).to(stds.device), stds).sum(dim=1)/9.0).mean())
            # assert stds.isfinite().sum() == torch.prod(torch.tensor(stds.shape)), "there is at least a non-finite value"
            # assert output.isfinite().sum() == torch.prod(torch.tensor(output.shape)), breakpoint()

            # sum up batch loss
            if model.config.C == 2:
                c_probs = (output - torch.logsumexp(output, dim=1, keepdims=True)).exp()
                c_probs = c_probs.gather(1, target.reshape(-1, 1)).flatten()
                loss_ce += criterion(c_probs, target.float()).item()
            else:
                loss_ce += criterion(output, target).item()

            data_ll.extend(torch.logsumexp(output, dim=1).detach().cpu().numpy())
            data_ll_unsup.extend(output.max(dim=1)[0].detach().cpu().numpy())
            data_ll_super.extend(output.gather(1, target.reshape(-1, 1)).squeeze().detach().cpu().numpy())
            class_probs[batch_index * test_loader.batch_size: (batch_index + 1) * test_loader.batch_size, :] = (
                        output - torch.logsumexp(output, dim=1, keepdims=True)).exp()

            loss_nll += -output.sum()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()

        # ic(max_std_eq_max_conds)

        t_delta = time_delta_now(t_start)
        print("Eval took {}".format(t_delta))

    np.save(d + 'output_{}_{}_{}'.format(training_dataset, dropout_inference, rotation), output_res)
    np.save(d + 'var_{}_{}_{}'.format(training_dataset, dropout_inference, rotation), var_res)
    np.save(d + 'll_x_{}_{}_{}'.format(training_dataset, dropout_inference, rotation), ll_x_res)
    np.save(d + 'var_x_{}_{}_{}'.format(training_dataset, dropout_inference, rotation), var_x_res)

    # ic((output_res - np.logsumexp(output_res, axis=1)).exp().max(axis=1).mean())
    ic(class_probs.max(dim=1)[0].mean())
    ic(np.exp(output_res).max(axis=1).mean())
    cf_std_0 = np.take_along_axis(np.exp(var_res), np.expand_dims(np.argmax(output_res, axis=1), axis=1),
                                  axis=1).flatten()
    cf_std_0 = cf_std_0.sum() / 10000  # dataset size
    cf_std_0 = np.sqrt(cf_std_0)
    ic(cf_std_0)

    loss_ce /= len(test_loader.dataset)
    loss_nll /= len(test_loader.dataset) + get_data_flatten_shape(test_loader)[1]
    accuracy = 100.0 * correct / len(test_loader.dataset)

    output_string = "{} set: Average loss_ce: {:.4f} Average loss_nll: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
        tag, loss_ce, loss_nll, correct, len(test_loader.dataset), accuracy
    )
    print(output_string)
    with open(d + 'training.out', 'a') as writer:
        writer.write(output_string + "\n")
    assert len(data_ll) == get_data_flatten_shape(test_loader)[0]
    assert len(data_ll_super) == get_data_flatten_shape(test_loader)[0]
    assert len(data_ll_unsup) == get_data_flatten_shape(test_loader)[0]

    # return data_ll, class_probs.detach().cpu().numpy(), data_ll_super, data_ll_unsup, loss_ce, loss_nll

    return model


def post_hoc_exps(model_dir=None, training_dataset=None, dropout_inference=None, n_mcd_passes=100, batch_size=512,
                  rat_S=20, rat_I=20, rat_D=5, rat_R=5):
    import pandas as pd
    import seaborn as sns

    dev = sys.argv[1]
    device = torch.device("cuda:0")
    use_cuda = True
    torch.cuda.benchmark = True

    d = model_dir + "post_hoc_results/"
    ensure_dir(d)
    train_loader, test_loader = get_data_loaders(use_cuda, batch_size=batch_size, device=device, dataset=training_dataset)
    n_features = get_data_flatten_shape(train_loader)[1]
    if training_dataset in DEBD:
        leaves = Bernoulli
        rat_C = 2
    else:
        leaves = RatNormal
        rat_C = 10
    model = make_spn(S=rat_S, I=rat_I, D=rat_D, R=rat_R, device=dev, dropout=dropout_inference[0], F=n_features,
                     C=rat_C, leaf_distribution=leaves)

    checkpoint = torch.load(model_dir + 'checkpoint.tar')
    model.load_state_dict(checkpoint['model_state_dict'])
    # old models
    # model.load_state_dict(torch.load(model_dir + 'model.pt'))
    model.eval()

    if isinstance(dropout_inference, float):
        d_inf = [dropout_inference]
    else:
        d_inf = dropout_inference

    if isinstance(n_mcd_passes, float):
        fwd_passes = [n_mcd_passes]
    else:
        fwd_passes = n_mcd_passes

    for dropout_inference in d_inf:
        for n_mcd_passes in fwd_passes:
            print("MCD p {} FWD passes {}".format(dropout_inference, n_mcd_passes))

            train_lls, class_probs_train, train_lls_sup, train_lls_unsup, train_ce, train_nll = evaluate_model(model, device, train_loader, "Train", output_dir=d)
            np.save(d + 'train_lls', train_lls)
            np.save(d + 'train_nll', train_nll.cpu().numpy())
            print(np.average(train_lls))

            test_lls, class_probs_test, test_lls_sup, test_lls_unsup, test_ce, test_nll = evaluate_model(model, device, test_loader, "Test", output_dir=d)
            np.save(d + 'test_lls', test_lls)
            np.save(d + 'test_nll', test_nll.cpu().numpy())
            print(np.average(test_lls))

            other_mnist_train_lls, other_mnist_test_lls, other_class_probs_train, other_class_probs_test, other_mnist_train_lls_sup, other_mnist_train_lls_unsup, other_mnist_test_lls_sup, other_mnist_test_lls_unsup = get_other_lls(model, device, d, use_cuda, batch_size, training_dataset=training_dataset)
            np.save(d + 'other_train_lls', other_mnist_train_lls)
            np.save(d + 'other_test_lls', other_mnist_test_lls)
            print(np.average(other_mnist_train_lls))
            print(np.average(other_mnist_test_lls))



            lls_dict = {"train_lls":train_lls, "test_lls":test_lls, "other_mnist_train_lls":other_mnist_train_lls, "other_mnist_test_lls":other_mnist_test_lls}
            head_lls_dict_sup = {"train_lls_sup":train_lls_sup, "test_lls_sup":test_lls_sup, "other_mnist_train_lls_sup":other_mnist_train_lls_sup, "other_mnist_test_lls_sup":other_mnist_test_lls_sup}
            head_lls_dict_unsup = {"train_lls_unsup":train_lls_unsup, "test_lls_unsup":test_lls_unsup, "other_mnist_train_unsup":other_mnist_train_lls_unsup, "other_mnist_test_unsup":other_mnist_test_lls_unsup}


            train_lls_dropout, class_probs_train_dropout, train_lls_sup_drop, train_lls_unsup_drop, train_lls_dropout_heads = evaluate_model_dropout(model, device, train_loader, "Train DROP", dropout_inference=dropout_inference, n_dropout_iters=n_mcd_passes, output_dir=d)
            np.save(d + 'drop_train_lls', train_lls_dropout)
            np.save(d + 'drop_train_nll', train_lls_dropout_heads)
            print(np.average(train_lls_dropout))
            test_lls_dropout, class_probs_test_dropout, test_lls_sup_drop, test_lls_unsup_drop, test_lls_dropout_heads = evaluate_model_dropout(model, device, test_loader, "Test DROP", dropout_inference=dropout_inference, n_dropout_iters=n_mcd_passes, output_dir=d)
            np.save(d + 'drop_test_lls', test_lls_dropout)
            np.save(d + 'drop_test_nll', test_lls_dropout_heads)
            print(np.average(test_lls_dropout))

            other_train_loader, other_test_loader = get_data_loaders(use_cuda=use_cuda, device=device, batch_size=batch_size, dataset=get_other_dataset_name(training_dataset))
            other_train_lls_dropout, other_class_probs_train_dropout, other_train_lls_sup_drop, other_train_lls_unsup_drop, other_train_lls_dropout_heads = evaluate_model_dropout(model, device, other_train_loader, "Other Train DROP", dropout_inference=dropout_inference, n_dropout_iters=n_mcd_passes, output_dir=d)
            np.save(d + 'drop_other_train_lls', other_train_lls_dropout)
            np.save(d + 'drop_other_train_nll', other_train_lls_dropout_heads)
            print(np.average(other_train_lls_dropout))
            other_test_lls_dropout, other_class_probs_test_dropout, other_test_lls_sup_drop, other_test_lls_unsup_drop, other_test_lls_dropout_heads = evaluate_model_dropout(model, device, other_test_loader, "Other Test DROP", dropout_inference=dropout_inference, n_dropout_iters=n_mcd_passes, output_dir=d)
            np.save(d + 'drop_other_test_lls', other_test_lls_dropout)
            np.save(d + 'drop_other_test_nll', other_test_lls_dropout_heads)
            print(np.average(other_test_lls_dropout))


            dropout_lls_dict = {"drop_train_lls":train_lls_dropout, "drop_test_lls":test_lls_dropout, "drop_other_mnist_train_lls":other_train_lls_dropout, "drop_other_mnist_test_lls":other_test_lls_dropout}
            dropout_head_lls_dict_sup = {"drop_train_lls_sup":train_lls_sup_drop, "drop_test_lls_sup":test_lls_sup_drop, "drop_other_mnist_train_lls_sup":other_train_lls_sup_drop, "drop_other_mnist_test_lls_sup":other_test_lls_sup_drop}
            dropout_head_lls_dict_unsup = {"drop_train_lls_unsup":train_lls_unsup_drop, "drop_test_lls_unsup":test_lls_unsup_drop, "drop_other_mnist_train_lls_unsup":other_train_lls_unsup_drop, "drop_other_mnist_test_lls_unsup":other_test_lls_unsup_drop}



            # plots
            filename = "lls_histograms.pdf"
            x_lim_left = min(min(train_lls), min(test_lls), min(other_mnist_test_lls), min(train_lls_dropout),
                             min(test_lls_dropout), min(other_test_lls_dropout))
            x_lim_right = max(max(train_lls), max(test_lls), max(other_mnist_test_lls), max(train_lls_dropout),
                             max(test_lls_dropout), max(other_test_lls_dropout))
            # print(x_lim_left, x_lim_right)
            y_lim_left = 0.0
            y_lim_right = 0.3 # 0.06
            # lls_in_domain_train = np.load(d_results + 'train_lls.npy')
            # lls_in_domain_test = np.load(d_results + 'test_lls.npy')
            # # other_train_lls = np.load(d_results + 'other_mnist_train_lls.npy')
            # # other_test_lls = np.load(d_results + 'other_mnist_test_lls.npy')
            # other_train_lls = np.load(d_results + 'other_train_lls.npy')
            # other_test_lls = np.load(d_results + 'other_test_lls.npy')

            df_in_train = pd.DataFrame({'lls_train': train_lls})
            df_in_test = pd.DataFrame({'lls_test': test_lls})
            # df_ood_train = pd.DataFrame({'other_train_lls':other_mnist_train_lls})
            df_ood_test = pd.DataFrame({'other_test_lls': other_mnist_test_lls})

            data = pd.concat([df_in_train, df_in_test, df_ood_test], ignore_index=True, axis=1)
            data = data.rename({0: '{} Train (In-domain)'.format(get_dataset_display_name(training_dataset)),
                                1: '{} Test (In-domain)'.format(get_dataset_display_name(training_dataset)),
                                2: '{} Test (OOD)'.format(get_dataset_display_name(get_other_dataset_name(training_dataset)))},
                               axis=1)

            palette = {"{} Train (In-domain)".format(get_dataset_display_name(training_dataset)): "yellow",
                       "{} Test (In-domain)".format(get_dataset_display_name(training_dataset)): "green",
                       "{} Test (OOD)".format(get_dataset_display_name(get_other_dataset_name(training_dataset=training_dataset))): "blue"}
            my_palette = palette
            # p3 = sns.histplot(data=data, x='value', hue='column',  bins=50, multiple='layer', kde=True)
            # p3 = sns.histplot(data=data, multiple='layer', stat="density", palette=palette, cbar_kws={'alpha':0.3})
            p3 = sns.histplot(data=data, stat="density", element="bars", common_norm=False, palette=my_palette)
            # p3 = sns.distplot(a=data, bins=20,  hist_kws={"alpha":0.2, "stat":"probability", "element":"step", "common_norm":False})
            # p3 = sns.distplot(df_melted,  bins=50)
            p3.set(xlabel='Data LL', ylabel='proportion of samples')
            p3.set_xlim(x_lim_left, x_lim_right)
            # p3.set_ylim(y_lim_left, y_lim_right)
            p3.set_title("Probabilistic Circuit")
            # p3.map(plt.hist, alpha=0.5)
            # p3 = sns.histplot(data=data, bins=20, multiple='layer')
            fig3 = p3.get_figure()
            fig3.savefig(d + "DE_" + filename)
            plt.close()

            p3 = sns.histplot(data=data, stat="probability", element="bars", common_norm=False, palette=my_palette)
            p3.set(xlabel='Data LL', ylabel='proportion of samples')
            p3.set_xlim(x_lim_left, x_lim_right)
            p3.set_ylim(y_lim_left, y_lim_right)
            p3.set_title("Probabilistic Circuit")
            # p3.map(plt.hist, alpha=0.5)
            # p3 = sns.histplot(data=data, bins=20, multiple='layer')
            fig3 = p3.get_figure()
            fig3.savefig(d + "pc_{}_passes_{}_".format(dropout_inference, n_mcd_passes) + filename)
            plt.close()

            # lls_in_domain_train = np.load(d_results + 'drop_train_lls.npy')
            # lls_in_domain_test = np.load(d_results + 'drop_test_lls.npy')
            # # other_train_lls = np.load(d_results + 'drop_other_mnist_train_lls.npy')
            # # other_test_lls = np.load(d_results + 'drop_other_mnist_test_lls.npy')
            # other_train_lls = np.load(d_results + 'drop_train_lls.npy')
            # other_test_lls = np.load(d_results + 'drop_test_lls.npy')

            df_in_train = pd.DataFrame({'lls_train': train_lls_dropout})
            df_in_test = pd.DataFrame({'lls_test': test_lls_dropout})
            # df_ood_train = pd.DataFrame({'other_train_lls':other_train_lls_dropout})
            df_ood_test = pd.DataFrame({'other_test_lls': other_test_lls_dropout})

            data = pd.concat([df_in_train, df_in_test, df_ood_test], ignore_index=True, axis=1)
            data = data.rename({0: '{} Train (In-domain)'.format(get_dataset_display_name(training_dataset)),
                                1: '{} Test (In-domain)'.format(get_dataset_display_name(training_dataset)),
                                2: '{} Test (OOD)'.format(
                                    get_dataset_display_name(get_other_dataset_name(training_dataset)))},
                               axis=1)
            p3 = sns.histplot(data=data, stat="density", element="bars", common_norm=False, palette=my_palette)
            p3.set(xlabel='Data LL', ylabel='proportion of samples')
            p3.set_xlim(x_lim_left, x_lim_right)
            # p3.set_ylim(y_lim_left, y_lim_right)
            p3.set_title("Dropout Circuit [p {} passes {}]".format(dropout_inference, n_mcd_passes))

            fig3 = p3.get_figure()
            fig3.savefig(d + "DE_drop_{}_passes_{}_".format(dropout_inference, n_mcd_passes) + filename)
            plt.close()

            p3 = sns.histplot(data=data, stat="probability", element="bars", common_norm=False, palette=my_palette)
            p3.set(xlabel='Data LL', ylabel='proportion of samples')
            p3.set_xlim(x_lim_left, x_lim_right)
            p3.set_ylim(y_lim_left, y_lim_right)
            p3.set_title("Dropout Circuit [p {} passes {}]".format(dropout_inference, n_mcd_passes))

            fig3 = p3.get_figure()
            fig3.savefig(d + "drop_{}_passes_{}_".format(dropout_inference, n_mcd_passes) + filename)
            plt.close()

def load_torch(model_dir=None, training_dataset=None, dropout_inference=None, n_mcd_passes=100, batch_size=512, rat_S=20, rat_I=20, rat_D=5, rat_R=5):
    dev = sys.argv[1]
    device = torch.device("cuda:0")
    use_cuda = True
    torch.cuda.benchmark = True

    d = model_dir + "likelihoods/"
    ensure_dir(d)
    train_loader, test_loader = get_data_loaders(use_cuda, batch_size=batch_size, device=device, dataset=training_dataset)
    n_features = get_data_flatten_shape(train_loader)[1]
    model = make_spn(S=rat_S, I=rat_I, D=rat_D, R=rat_R, device=dev, dropout=dropout_inference, F=n_features)
    model.load_state_dict(torch.load(model_dir + 'model.pt'))
    model.eval()

    train_lls, class_probs_train, train_lls_sup, train_lls_unsup, train_ce, train_nll = evaluate_model(model, device, train_loader, "Train", output_dir=d)
    test_lls, class_probs_test, test_lls_sup, test_lls_unsup, test_ce, test_nll = evaluate_model(model, device, test_loader, "Test", output_dir=d)
    other_mnist_train_lls, other_mnist_test_lls, other_class_probs_train, other_class_probs_test, other_mnist_train_lls_sup, other_mnist_train_lls_unsup, other_mnist_test_lls_sup, other_mnist_test_lls_unsup = get_other_lls(model, device, d, use_cuda, batch_size, training_dataset=training_dataset)

    lls_dict = {"train_lls":train_lls, "test_lls":test_lls, "other_mnist_train_lls":other_mnist_train_lls, "other_mnist_test_lls":other_mnist_test_lls}
    head_lls_dict_sup = {"train_lls_sup":train_lls_sup, "test_lls_sup":test_lls_sup, "other_mnist_train_lls_sup":other_mnist_train_lls_sup, "other_mnist_test_lls_sup":other_mnist_test_lls_sup}
    head_lls_dict_unsup = {"train_lls_unsup":train_lls_unsup, "test_lls_unsup":test_lls_unsup, "other_mnist_train_unsup":other_mnist_train_lls_unsup, "other_mnist_test_unsup":other_mnist_test_lls_unsup}

    for k, v in lls_dict.items():
        np.save(d + k, v)
    for k, v in head_lls_dict_sup.items():
        np.save(d + k, v)
    for k, v in head_lls_dict_unsup.items():
        np.save(d + k, v)

    train_lls_dropout, class_probs_train_dropout, train_lls_sup_drop, train_lls_unsup_drop, train_lls_dropout_heads = evaluate_model_dropout(model, device, train_loader, "Train DROP", dropout_inference=dropout_inference, n_dropout_iters=n_mcd_passes, output_dir=d)
    test_lls_dropout, class_probs_test_dropout, test_lls_sup_drop, test_lls_unsup_drop, test_lls_dropout_heads = evaluate_model_dropout(model, device, test_loader, "Test DROP", dropout_inference=dropout_inference, n_dropout_iters=n_mcd_passes, output_dir=d)

    other_train_loader, other_test_loader = get_data_loaders(use_cuda=use_cuda, device=device, batch_size=batch_size, dataset=get_other_dataset_name(training_dataset))
    other_train_lls_dropout, other_class_probs_train_dropout, other_train_lls_sup_drop, other_train_lls_unsup_drop, other_train_lls_dropout_heads = evaluate_model_dropout(model, device, other_train_loader, "Other Train DROP", dropout_inference=dropout_inference, n_dropout_iters=n_mcd_passes, output_dir=d)
    other_test_lls_dropout, other_class_probs_test_dropout, other_test_lls_sup_drop, other_test_lls_unsup_drop, other_test_lls_dropout_heads = evaluate_model_dropout(model, device, other_test_loader, "Other Test DROP", dropout_inference=dropout_inference, n_dropout_iters=n_mcd_passes, output_dir=d)

    np.save(d + 'train_lls_dropout_heads', train_lls_dropout_heads)
    np.save(d + 'test_lls_dropout_heads', test_lls_dropout_heads)
    np.save(d + 'other_train_lls_dropout_heads', other_train_lls_dropout_heads)
    np.save(d + 'other_test_lls_dropout_heads', other_test_lls_dropout_heads)

    dropout_lls_dict = {"drop_train_lls":train_lls_dropout, "drop_test_lls":test_lls_dropout, "drop_other_mnist_train_lls":other_train_lls_dropout, "drop_other_mnist_test_lls":other_test_lls_dropout}
    dropout_head_lls_dict_sup = {"drop_train_lls_sup":train_lls_sup_drop, "drop_test_lls_sup":test_lls_sup_drop, "drop_other_mnist_train_lls_sup":other_train_lls_sup_drop, "drop_other_mnist_test_lls_sup":other_test_lls_sup_drop}
    dropout_head_lls_dict_unsup = {"drop_train_lls_unsup":train_lls_unsup_drop, "drop_test_lls_unsup":test_lls_unsup_drop, "drop_other_mnist_train_lls_unsup":other_train_lls_unsup_drop, "drop_other_mnist_test_lls_unsup":other_test_lls_unsup_drop}

    for k, v in dropout_lls_dict.items():
        np.save(d + k, v)
    for k, v in dropout_head_lls_dict_sup.items():
        np.save(d + k, v)
    for k, v in dropout_head_lls_dict_unsup.items():
        np.save(d + k, v)


def run_torch(n_epochs=100, batch_size=256, dropout_inference=0.1, dropout_spn=0.0, training_dataset='mnist',
              lmbda=0.0, eval_single_digit=False, toy_setting=False, eval_rotation=False, mnist_corruptions=False,
              n_mcd_passes=100, corrupted_cifar_dir='', eval_every_n_epochs=5, lr=1e-3, dropout_cf=False):
    """Run the torch code.

    Args:
        n_epochs (int, optional): Number of epochs.
        batch_size (int, optional): Batch size.
    """
    from torch import optim
    from torch import nn

    assert len(sys.argv) == 2, "Usage: train.mnist cuda/cpu"
    dev = sys.argv[1]

    if dev == "cpu":
        device = torch.device("cpu")
        use_cuda = False
    else:
        device = torch.device("cuda:0")
        use_cuda = True
        torch.cuda.benchmark = True

    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    d = "results/{}/".format(datetime_str)
    ensure_dir(d)


    train_loader, test_loader = get_data_loaders(use_cuda, batch_size=batch_size, device=device, dataset=training_dataset)
    n_features = get_data_flatten_shape(train_loader)[1]
    print(n_features)


    if toy_setting:
        rat_S, rat_I, rat_D, rat_R, rat_C = 2, 4, 2, 1, 10
        if training_dataset in DEBD:
            leaves = Bernoulli
            rat_C = 2
        else:
            leaves = RatNormal
    elif training_dataset == 'nltcs' or training_dataset == 'msnbc':
        rat_S, rat_I, rat_D, rat_R, rat_C, leaves = 2, 4, 2, 1, 2, Bernoulli
    elif training_dataset == '2moons' or training_dataset == '2gaussians' or training_dataset == '4gaussians':
        rat_S, rat_I, rat_D, rat_R, rat_C, leaves = 4, 32, 2, 2, 2, RatNormal
    elif training_dataset == 'cifar': # or training_dataset == 'svhn':
        rat_S, rat_I, rat_D, rat_R, rat_C, leaves = 30, 30, 5, 10, 10, RatNormal
    else:
        rat_S, rat_I, rat_D, rat_R, rat_C, leaves = 20, 20, 5, 5, 10, RatNormal

    model = make_spn(S=rat_S, I=rat_I, D=rat_D, R=rat_R, device=dev, dropout=dropout_spn, F=n_features, C=rat_C,
                     leaf_distribution=leaves)
    model.train()
    print(model)
    print(model._leaf.base_leaf)
    n_rat_params = count_params(model)
    print("Number of pytorch parameters: ", n_rat_params)

    # Define optimizer
    if model.config.C == 2:
        loss_fn = nn.BCELoss()
    else:
        loss_fn = nn.CrossEntropyLoss()
    #loss_fn = nn.NLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    print("Learning rate: {}".format(lr))

    log_interval = 100

    training_string = ""

    d_samples = d + "samples/"
    d_results = d + "results/"
    d_model = d + "model/"
    ensure_dir(d_samples)
    ensure_dir(d_results)
    ensure_dir(d_model)

    with open(d + 'training_details.out', 'a') as writer:
        writer.write("Dataset: {}, N features {}".format(training_dataset, n_features))
        writer.write("\nn epochs: {}, batch size: {}".format(n_epochs, batch_size))
        writer.write("\nLearning rate {}".format(lr))
        writer.write("\nMC dropout p {}, dropout (learning) {}".format(dropout_inference, dropout_spn))
        writer.write("\nRAT lambda {}".format(lmbda))
        writer.write("\nRAT hyperparameters S {} I {} D {} R {} C {}".format(rat_S, rat_I, rat_D, rat_R, rat_C))
        writer.write("\nRAT n of model params: {}".format(n_rat_params))
        writer.write("\nN MCD passes: {}".format(n_mcd_passes))

    loss_epoch = []
    loss_ce_epoch = []
    loss_nll_epoch = []

    test_loss = []
    test_loss_ce = []
    test_loss_nll = []


    for epoch in range(n_epochs):
        model.train()
        t_start = time.time()

        running_loss = 0.0
        running_loss_ce = 0.0
        running_loss_nll = 0.0

        epoch_loss = 0.0
        epoch_loss_ce = 0.0
        epoch_loss_nll = 0.0

        for batch_index, (data, target) in enumerate(train_loader):
            if model.config.C == 2:
                target = torch.tensor(one_hot(target.numpy()), dtype=torch.float)

            # Send data to correct device
            data, target = data.to(device), target.to(device)
            data = data.view(data.shape[0], -1)
            # print(data)
            # print(target)

            # Reset gradients
            optimizer.zero_grad()

            # Inference
            output = model(data)

            # Compute loss
            if model.config.C == 2:
                class_probs = (output - torch.logsumexp(output, dim=1, keepdims=True)).exp()
                # class_probs = class_probs.gather(1, target.reshape(-1, 1)).flatten()
                loss_ce = loss_fn(class_probs, target)
            else:
                loss_ce = loss_fn(output, target)

            loss_nll = -output.sum() / (data.shape[0] * n_features)

            loss = (1 - lmbda) * loss_nll + lmbda * loss_ce

            # Backprop
            loss.backward()
            optimizer.step()
            # print([p for p in model.parameters()])

            epoch_loss += loss.item()
            epoch_loss_ce += loss_ce.item()
            epoch_loss_nll += loss_nll.item()

            # Log stuff
            running_loss += loss.item()
            running_loss_ce += loss_ce.item()
            running_loss_nll += loss_nll.item()

            if batch_index % log_interval == (log_interval - 1):
                if model.config.C == 2:
                    target = target.argmax(dim=1)
                pred = output.argmax(1).eq(target).sum().cpu().numpy() / data.shape[0] * 100
                print(
                    "Train Epoch: {} [{: >5}/{: <5} ({:.0f}%)]\tLoss_ce: {:.6f}\tLoss_nll: {:.6f}\tAccuracy: {:.0f}%".format(
                        epoch,
                        batch_index * len(data),
                        60000,
                        100.0 * batch_index / len(train_loader),
                        running_loss_ce / log_interval,
                        running_loss_nll / log_interval,
                        pred,
                    ),
                    end="\r",
                )
                running_loss = 0.0
                running_loss_ce = 0.0
                running_loss_nll = 0.0

        with torch.no_grad():
            if (training_dataset == 'mnist' or training_dataset == 'fmnist'):
                set_seed(0)
                samples = model.sample(class_index=list(range(10)) * 5)
                save_samples(samples, iteration=epoch)

        t_delta = time_delta_now(t_start)
        print("Train Epoch {} took {}".format(epoch, t_delta))
        loss_epoch.append(epoch_loss / len(train_loader))
        loss_ce_epoch.append(epoch_loss_ce / len(train_loader))
        loss_nll_epoch.append(epoch_loss_nll / len(train_loader))


        if epoch % eval_every_n_epochs == (eval_every_n_epochs-1):
            print("Evaluating model...")
            lls_train, class_probs_train, _, _, train_loss_ce_ep, train_loss_nll_ep = evaluate_model(model, device, train_loader, "{}^ epoch - Train".format(epoch+1), output_dir=d)
            test_lls, class_probs_test, _, _, test_loss_ce_ep, test_loss_nll_ep = evaluate_model(model, device, test_loader, "{}^ epoch - Test".format(epoch+1), output_dir=d)

            test_loss_ce.append([epoch, test_loss_ce_ep])
            test_loss_nll.append([epoch, test_loss_nll_ep.cpu().item()])
            test_loss.append([epoch, (1 - lmbda) * test_loss_nll_ep.cpu().item() + lmbda * test_loss_ce_ep])

            print("Train class entropy: {} Test class entropy: {}".format(entropy(class_probs_train, axis=1).sum(), entropy(class_probs_test, axis=1).sum()))
            print('Saving model... epoch {}'.format(epoch))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'loss_ce': loss_ce,
                'loss_nll': loss_nll,
                'lmbda': lmbda,
                'lr': lr,
            }, d + 'model/checkpoint.tar')


    print('Saving model...')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'loss_ce': loss_ce,
        'loss_nll': loss_nll,
        'lmbda': lmbda,
        'lr': lr,
    }, d + 'model/checkpoint.tar')

    train_lls, class_probs_train, train_lls_sup, train_lls_unsup, train_loss_ce_final, train_loss_nll_final = evaluate_model(model, device, train_loader, "Train", output_dir=d)
    test_lls, class_probs_test, test_lls_sup, test_lls_unsup, test_loss_ce_final, test_loss_nll_final = evaluate_model(model, device, test_loader, "Test", output_dir=d)

    test_loss_ce.append([n_epochs-1, test_loss_ce_final])
    test_loss_nll.append([n_epochs-1, test_loss_nll_final.cpu().item()])
    test_loss.append([n_epochs-1, (1 - lmbda) * test_loss_nll_final.cpu().item() + lmbda * test_loss_ce_final])

    print("Train class entropy: {} Test class entropy: {}".format(entropy(class_probs_train, axis=1).sum(), entropy(class_probs_test, axis=1).sum()))
    training_curves_data = {"training_loss":loss_epoch, "training_loss_ce":loss_ce_epoch, "training_loss_nll":loss_nll_epoch,
                            "test_loss":test_loss, "test_loss_ce":test_loss_ce, "test_loss_nll":test_loss_nll}
    plot_training_curves(training_curves_data, filename='loss_curves', title='Loss(es)', path=d_samples)


    if training_dataset == 'cifar':
        corruptions = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 'frost', 'gaussian_blur', 'gaussian_noise',
                          'glass_blur', 'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate', 'saturate', 'shot_noise', 'snow',
                          'spatter', 'speckle_noise', 'zoom_blur']

        results_dict = {}
        for corruption in corruptions:
            for cl in range(5):
                cl += 1
                print("Corruption {} Level {}".format(corruption, cl))
                results_dict['c_{}_l{}'.format(corruption, cl)] = evaluate_model_corrupted_cifar(model, device, test_loader, "Test DROP corrupted CIFAR C {} L {}".format(corruption, cl),
                                                                                                 dropout_inference=dropout_inference, n_dropout_iters=n_mcd_passes, output_dir=d,
                                                                                                 corruption=corruption, corruption_level=cl, corrupted_cifar_dir=corrupted_cifar_dir)
                np.save(d_results + 'dropout_class_probs_c_{}_l{}'.format(corruption, cl), results_dict['c_{}_l{}'.format(corruption, cl)][0].cpu().detach().numpy())
                np.save(d_results + 'class_probs_c_{}_l{}'.format(corruption, cl), results_dict['c_{}_l{}'.format(corruption, cl)][1].cpu().detach().numpy())


    if training_dataset == 'mnist' and mnist_corruptions:
        import spn.experiments.RandomSPNs_layerwise.mnist_c.corruptions as corruptions
        severity = [1, 2, 3, 4, 5]
        corruption_method = [corruptions.brightness, corruptions.shot_noise, corruptions.impulse_noise, corruptions.glass_blur, corruptions.motion_blur, corruptions.shear, corruptions.scale,
                             corruptions.rotate, corruptions.translate, corruptions.fog, corruptions.spatter]
        corruption_methods_no_severity = [corruptions.stripe, corruptions.dotted_line, corruptions.zigzag, corruptions.canny_edges]

        results_dict = {}
        corruption_methods_full = []
        corruption_methods_full.extend(corruption_method)
        corruption_methods_full.extend(corruption_methods_no_severity)

        for cm in corruption_methods_full:
            if cm in corruption_methods_no_severity: severity = [None]
            for sl in severity:
                print("Corruption {}, Severity {}".format(cm.__name__, sl))
                results_dict['c_{}_l{}'.format(cm.__name__, sl)] = evaluate_model_corrupted_digits(model, device, test_loader, "Test DROP corrupted", dropout_inference=dropout_inference,
                                                                                                   n_dropout_iters=n_mcd_passes, output_dir=d, corruption=cm, severity=sl)
                np.save(d_results + 'dropout_class_probs_c_{}_l{}'.format(cm.__name__, sl), results_dict['c_{}_l{}'.format(cm.__name__, sl)][0].cpu().detach().numpy())
                np.save(d_results + 'class_probs_c_{}_l{}'.format(cm.__name__, sl), results_dict['c_{}_l{}'.format(cm.__name__, sl)][1].cpu().detach().numpy())

        # compute max class probs
        drop_max_class_probs = [torch.max(torch.mean(results_dict['c_{}_l1'.format(cm.__name__)][0], dim=2), dim=1)[0].cpu().detach().numpy() for cm in corruption_method]
        drop_max_class_probs.extend([torch.max(torch.mean(results_dict['c_{}_lNone'.format(cm.__name__)][0], dim=2), dim=1)[0].cpu().detach().numpy() for cm in corruption_methods_no_severity])

        max_class_probs = [torch.max(results_dict['c_{}_l1'.format(cm.__name__)][1], dim=1)[0].cpu().detach().numpy() for cm in corruption_method]
        max_class_probs.extend([torch.max(results_dict['c_{}_lNone'.format(cm.__name__)][1], dim=1)[0].cpu().detach().numpy() for cm in corruption_methods_no_severity])

        # prepare boxplots data
        boxplot_data = np.column_stack(max_class_probs)
        drop_boxplot_data = np.column_stack(drop_max_class_probs)

        # compute entropy
        drop_entropy_class_probs = [entropy(torch.mean(results_dict['c_{}_l1'.format(cm.__name__)][0], dim=2).detach().cpu().numpy(), axis=1) for cm in corruption_method]
        drop_entropy_class_probs.extend([entropy(torch.mean(results_dict['c_{}_lNone'.format(cm.__name__)][0], dim=2).detach().cpu().numpy(), axis=1) for cm in corruption_methods_no_severity])
        entropy_class_probs = [entropy(results_dict['c_{}_l1'.format(cm.__name__)][1].detach().cpu().numpy(), axis=1) for cm in corruption_method]
        entropy_class_probs.extend([entropy(results_dict['c_{}_lNone'.format(cm.__name__)][1].detach().cpu().numpy(), axis=1) for cm in corruption_methods_no_severity])

        # prepare boxplots data
        entropy_boxplot_data = np.column_stack(entropy_class_probs)
        drop_entropy_boxplot_data = np.column_stack(drop_entropy_class_probs)

        # plot bloxplot
        plot_boxplot_corrupted_digits(boxplot_data, filename='boxplot_corrupted_digits', title='Corrupted MNIST', path=d_samples)
        plot_boxplot_corrupted_digits(drop_boxplot_data, filename='boxplot_corrupted_digits_MCD', title='Corrupted MNIST - MCD', path=d_samples)
        plot_boxplot_corrupted_digits(entropy_boxplot_data, filename='entropy_boxplot_corrupted_digits', title='Corrupted MNIST', path=d_samples, ylimits=[-0.1, 3.0], ylabel='Classification Entropy')
        plot_boxplot_corrupted_digits(drop_entropy_boxplot_data, filename='entropy_boxplot_corrupted_digits_MCD', title='Corrupted MNIST - MCD', path=d_samples, ylimits=[-0.1, 3.0], ylabel='Classification Entropy')


    if training_dataset == 'mnist' and eval_rotation:
        results_dict = {}
        degrees = [180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210]
        for deg in degrees:
            print("Rotation degrees {}".format(deg))
            results_dict['rotation_{}'.format(deg)] = evaluate_model_rotated_digits(model, device, test_loader, "Test DROP rotated digit {} degrees".format(deg), dropout_inference=dropout_inference,
                                                                                    n_dropout_iters=n_mcd_passes, output_dir=d, degrees=deg)
            np.save(d_results + 'dropout_class_probs_{}'.format(deg), results_dict['rotation_{}'.format(deg)][0].cpu().detach().numpy())
            np.save(d_results + 'class_probs_{}'.format(deg), results_dict['rotation_{}'.format(deg)][1].cpu().detach().numpy())


        drop_max_class_probs = [torch.max(torch.mean(results_dict['rotation_{}'.format(deg)][0], dim=2), dim=1)[0].cpu().detach().numpy() for deg in degrees]
        max_class_probs = [torch.max(results_dict['rotation_{}'.format(deg)][1], dim=1)[0].cpu().detach().numpy() for deg in degrees]

        boxplot_data = np.column_stack(max_class_probs)
        drop_boxplot_data = np.column_stack(drop_max_class_probs)

        plot_boxplot_rotating_digits(boxplot_data, filename='boxplot_rotating_digits', title='Rotating MNIST', path=d_samples)
        plot_boxplot_rotating_digits(drop_boxplot_data, filename='boxplot_rotating_digits_MCD', title='Rotating MNIST - MCD', path=d_samples)

        # plot classification entropy
        drop_entropy_class_probs = [entropy(torch.mean(results_dict['rotation_{}'.format(deg)][0], dim=2).detach().cpu().numpy(), axis=1) for deg in degrees]
        entropy_class_probs = [entropy(results_dict['rotation_{}'.format(deg)][1].detach().cpu().numpy(), axis=1) for deg in degrees]

        entropy_boxplot_data = np.column_stack(entropy_class_probs)
        drop_entropy_boxplot_data = np.column_stack(drop_entropy_class_probs)

        plot_boxplot_rotating_digits(entropy_boxplot_data, filename='entropy_boxplot_rotating_digits', title='Rotating MNIST', path=d_samples, ylimits=[-0.1, 3.0], ylabel='Classification Entropy')
        plot_boxplot_rotating_digits(drop_entropy_boxplot_data, filename='entropy_boxplot_rotating_digits_MCD', title='Rotating MNIST - MCD', path=d_samples, ylimits=[-0.1, 3.0], ylabel='Classification Entropy')



    if training_dataset == 'mnist' and eval_rotation and eval_single_digit:
        degrees = [180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210]

        for class_idx in range(model.config.C):
            results_dict = {}
            print("Evaluate by rotating digits, for each class separately: class {}".format(class_idx))
            for deg in degrees:
                print("Rotation degrees {}".format(deg))
                results_dict['digit_{}_rotation_{}'.format(class_idx, deg)] = evaluate_model_rotated_digits(model, device, test_loader, "Test DROP rotated digit {} degrees".format(deg), dropout_inference=dropout_inference,
                                                                                        n_dropout_iters=n_mcd_passes, output_dir=d, degrees=deg, class_label=class_idx)
                np.save(d_results + 'dropout_class_probs_{}_{}'.format(deg, class_idx), results_dict['digit_{}_rotation_{}'.format(class_idx, deg)][0].cpu().detach().numpy())
                np.save(d_results + 'class_probs_{}_{}'.format(deg, class_idx), results_dict['digit_{}_rotation_{}'.format(class_idx, deg)][1].cpu().detach().numpy())


            drop_max_class_probs = [torch.max(torch.mean(results_dict['digit_{}_rotation_{}'.format(class_idx, deg)][0], dim=2), dim=1)[0].cpu().detach().numpy() for deg in degrees]
            max_class_probs = [torch.max(results_dict['digit_{}_rotation_{}'.format(class_idx, deg)][1], dim=1)[0].cpu().detach().numpy() for deg in degrees]

            boxplot_data = np.column_stack(max_class_probs)
            drop_boxplot_data = np.column_stack(drop_max_class_probs)

            plot_boxplot_rotating_digits(boxplot_data, filename='boxplot_rotating_digits_' + str(class_idx), title='Rotating MNIST', path=d_samples)
            plot_boxplot_rotating_digits(drop_boxplot_data, filename='boxplot_rotating_digits_MCD_' + str(class_idx), title='Rotating MNIST - MCD', path=d_samples)

            # plot classification entropy
            drop_entropy_class_probs = [entropy(torch.mean(results_dict['digit_{}_rotation_{}'.format(deg)][0], dim=2).detach().cpu().numpy(), axis=1) for deg in degrees]
            entropy_class_probs = [entropy(results_dict['digit_{}_rotation_{}'.format(deg)][1].detach().cpu().numpy(), axis=1) for deg in degrees]

            entropy_boxplot_data = np.column_stack(entropy_class_probs)
            drop_entropy_boxplot_data = np.column_stack(drop_entropy_class_probs)

            plot_boxplot_rotating_digits(entropy_boxplot_data, filename='entropy_boxplot_rotating_digits_' + str(class_idx), title='Rotating MNIST', path=d_samples, ylimits=[-0.1, 3.0], ylabel='Classification Entropy')
            plot_boxplot_rotating_digits(drop_entropy_boxplot_data, filename='entropy_boxplot_rotating_digits_MCD_' + str(class_idx), title='Rotating MNIST - MCD', path=d_samples, ylimits=[-0.1, 3.0], ylabel='Classification Entropy')


    other_mnist_train_lls, other_mnist_test_lls, other_class_probs_train, other_class_probs_test, other_mnist_train_lls_sup, other_mnist_train_lls_unsup, other_mnist_test_lls_sup, other_mnist_test_lls_unsup = get_other_lls(model, device, d, use_cuda, batch_size, training_dataset=training_dataset)
    print("OTHER Train class entropy: {} OTHER Test class entropy: {}".format(entropy(other_class_probs_train, axis=1).sum(), entropy(other_class_probs_test, axis=1).sum()))

    lls_dict = {"train_lls": train_lls, "test_lls": test_lls, "other_train_lls": other_mnist_train_lls, "other_test_lls":other_mnist_test_lls}
    head_lls_dict_sup = {"train_lls_sup":train_lls_sup, "test_lls_sup":test_lls_sup, "other_train_lls_sup":other_mnist_train_lls_sup, "other_test_lls_sup":other_mnist_test_lls_sup}
    head_lls_dict_unsup = {"train_lls_unsup":train_lls_unsup, "test_lls_unsup":test_lls_unsup, "other_train_unsup":other_mnist_train_lls_unsup, "other_test_unsup":other_mnist_test_lls_unsup}
    class_probs_dict = {"class_probs_train":class_probs_train.max(axis=1), "class_probs_test":class_probs_test.max(axis=1), "other_class_probs_train":other_class_probs_train.max(axis=1), "other_class_probs_test":other_class_probs_test.max(axis=1)}

    np.save(d_results + 'class_probs_in_domain_train', class_probs_train)
    np.save(d_results + 'class_probs_in_domain_test', class_probs_test)
    np.save(d_results + 'class_probs_ood_train', other_class_probs_train)
    np.save(d_results + 'class_probs_ood_test', other_class_probs_test)

    d_likelihoods = d_results + "likelihoods/"
    ensure_dir(d_likelihoods)

    for k, v in lls_dict.items():
        np.save(d_likelihoods + k, v)
        print(np.array(v).mean())
    for k, v in head_lls_dict_sup.items():
        np.save(d_likelihoods + k, v)
    for k, v in head_lls_dict_unsup.items():
        np.save(d_likelihoods + k, v)

    plot_histograms(lls_dict, filename='histograms_lls', title="Data LLs", path=d_samples, trained_on_fmnist=training_dataset)
    plot_histograms(head_lls_dict_sup, filename='histograms_lls_sup', title="Data SUP LLs", path=d_samples, trained_on_fmnist=training_dataset)
    plot_histograms(head_lls_dict_unsup, filename='histograms_lls_unsup', title="Data UNSUP LLs", path=d_samples, trained_on_fmnist=training_dataset)
    plot_histograms(class_probs_dict, filename='class_probs_histograms', title="Class Probs", path=d_samples, trained_on_fmnist=training_dataset, y_lim=30000)

    train_lls_dropout, class_probs_train_dropout, train_lls_sup_drop, train_lls_unsup_drop, train_lls_dropout_heads = evaluate_model_dropout(model, device, train_loader, "Train DROP",
                                                                                                                    dropout_inference=dropout_inference, n_dropout_iters=n_mcd_passes, output_dir=d,
                                                                                                                                             dropout_cf=dropout_cf)
    test_lls_dropout, class_probs_test_dropout, test_lls_sup_drop, test_lls_unsup_drop, test_lls_dropout_heads = evaluate_model_dropout(model, device, test_loader, "Test DROP",
                                                                                                                dropout_inference=dropout_inference, n_dropout_iters=n_mcd_passes, output_dir=d,
                                                                                                                                        dropout_cf=dropout_cf)
    print("DROP Train class entropy: {} DROP Test class entropy: {}".format(entropy(class_probs_train_dropout.mean(axis=2), axis=1).sum(), entropy(class_probs_test_dropout.mean(axis=2), axis=1).sum()))

    other_train_loader, other_test_loader = get_data_loaders(use_cuda=use_cuda, device=device, batch_size=batch_size, dataset=get_other_dataset_name(training_dataset))
    other_train_lls_dropout, other_class_probs_train_dropout, other_train_lls_sup_drop, other_train_lls_unsup_drop, other_train_lls_dropout_heads = evaluate_model_dropout(model, device, other_train_loader, "Other Train DROP",
                                                                                                                                            dropout_inference=dropout_inference, n_dropout_iters=n_mcd_passes, output_dir=d,
                                                                                                                                                                           dropout_cf=dropout_cf)
    other_test_lls_dropout, other_class_probs_test_dropout, other_test_lls_sup_drop, other_test_lls_unsup_drop, other_test_lls_dropout_heads = evaluate_model_dropout(model, device, other_test_loader, "Other Test DROP",
                                                                                                                                        dropout_inference=dropout_inference, n_dropout_iters=n_mcd_passes, output_dir=d,
                                                                                                                                                                      dropout_cf=dropout_cf)
    print("DROP OTHER Train class entropy: {} DROP OTHER Test class entropy: {}".format(entropy(other_class_probs_train_dropout.mean(axis=2), axis=1).sum(), entropy(other_class_probs_test_dropout.mean(axis=2), axis=1).sum()))

    np.save(d_results + 'class_probs_in_domain_train_dropout', class_probs_train_dropout)
    np.save(d_results + 'class_probs_in_domain_test_dropout', class_probs_test_dropout)
    np.save(d_results + 'class_probs_ood_train_dropout', other_class_probs_train_dropout)
    np.save(d_results + 'class_probs_ood_test_dropout', other_class_probs_test_dropout)

    np.save(d_results + 'train_lls_dropout_in_domain_heads', train_lls_dropout_heads)
    np.save(d_results + 'test_lls_dropout_in_domain_heads', test_lls_dropout_heads)
    np.save(d_results + 'train_lls_dropout_ood_heads', other_train_lls_dropout_heads)
    np.save(d_results + 'test_lls_dropout_ood_heads', other_test_lls_dropout_heads)

    dropout_lls_dict = {"drop_train_lls":train_lls_dropout, "drop_test_lls":test_lls_dropout, "drop_other_train_lls":other_train_lls_dropout, "drop_other_test_lls":other_test_lls_dropout}
    dropout_head_lls_dict_sup = {"drop_train_lls_sup":train_lls_sup_drop, "drop_test_lls_sup":test_lls_sup_drop, "drop_other_train_lls_sup":other_train_lls_sup_drop, "drop_other_test_lls_sup":other_test_lls_sup_drop}
    dropout_head_lls_dict_unsup = {"drop_train_lls_unsup":train_lls_unsup_drop, "drop_test_lls_unsup":test_lls_unsup_drop, "drop_other_train_lls_unsup":other_train_lls_unsup_drop, "drop_other_test_lls_unsup":other_test_lls_unsup_drop}
    dropout_class_probs_dict = {"drop_class_probs_train":class_probs_train_dropout.mean(axis=2).max(axis=1), "drop_class_probs_test":class_probs_test_dropout.mean(axis=2).max(axis=1),
                                "drop_other_class_probs_train":other_class_probs_train_dropout.mean(axis=2).max(axis=1), "drop_other_class_probs_test":other_class_probs_test_dropout.mean(axis=2).max(axis=1)}

    for k, v in dropout_lls_dict.items():
        np.save(d_likelihoods + k, v)
        print(np.array(v).mean())
    for k, v in dropout_head_lls_dict_sup.items():
        np.save(d_likelihoods + k, v)
    for k, v in dropout_head_lls_dict_unsup.items():
        np.save(d_likelihoods + k, v)

    plot_histograms(dropout_lls_dict, filename='dropout_histograms_lls', title="DROPOUT Data LLs", path=d_samples, trained_on_fmnist=training_dataset)
    plot_histograms(dropout_head_lls_dict_sup, filename='dropout_histograms_lls_sup', title="DROPOUT Data SUP LLs", path=d_samples, trained_on_fmnist=training_dataset)
    plot_histograms(dropout_head_lls_dict_unsup, filename='dropout_histograms_lls_unsup', title="DROPOUT Data UNSUP LLs", path=d_samples, trained_on_fmnist=training_dataset)
    plot_histograms(dropout_class_probs_dict, filename='dropout_class_probs_histograms', title="DROPOUT Class Probs", path=d_samples, trained_on_fmnist=training_dataset, y_lim=30000)


def get_other_lls(model, device, output_dir='./', use_cuda=False, batch_size=100, training_dataset='mnist'):
    train_loader, test_loader = get_data_loaders(use_cuda=use_cuda, device=device, batch_size=batch_size, dataset=get_other_dataset_name(training_dataset))
    log_string = get_other_dataset_name(training_dataset)
    train_lls, class_probs_train, train_lls_sup, train_lls_unsup, _, _ = evaluate_model(model, device, train_loader, "Train " + log_string, output_dir=output_dir)
    test_lls, class_probs_test, test_lls_sup, test_lls_unsup, _, _ = evaluate_model(model, device, test_loader, "Test " + log_string, output_dir=output_dir)
    return train_lls, test_lls, class_probs_train, class_probs_test, train_lls_sup, train_lls_unsup, test_lls_sup, test_lls_unsup

def evaluate_model(model: torch.nn.Module, device, loader, tag, output_dir="") -> float:
    """
    Description for method evaluate_model.

    Args:
        model (nn.Module): PyTorch module.
        device: Execution device.
        loader: Data loader.
        tag (str): Tag for information.

    Returns:
        float: Tuple of loss and accuracy.
    """
    model.eval()
    loss_ce = 0
    loss_nll = 0
    data_ll = []
    data_ll_super = [] # pick it from the label-th head
    data_ll_unsup = [] # pick the max one
    class_probs = torch.zeros((get_data_flatten_shape(loader)[0], model.config.C)).to(device)
    correct = 0

    if model.config.C == 2:
        criterion = nn.BCELoss(reduction="sum")
    else:
        criterion = nn.CrossEntropyLoss(reduction="sum")
    # criterion = nn.NLLLoss(reduction="sum")

    with torch.no_grad():
        for batch_index, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            #print(target)
            data = data.view(data.shape[0], -1)
            output = model(data)

            # sum up batch loss
            if model.config.C == 2:
                c_probs = (output - torch.logsumexp(output, dim=1, keepdims=True)).exp()
                c_probs = c_probs.gather(1, target.reshape(-1, 1)).flatten()
                loss_ce += criterion(c_probs, target.float()).item()
            else:
                loss_ce += criterion(output, target).item()



            data_ll.extend(torch.logsumexp(output, dim=1).detach().cpu().numpy())
            data_ll_unsup.extend(output.max(dim=1)[0].detach().cpu().numpy())
            data_ll_super.extend(output.gather(1, target.reshape(-1, 1)).squeeze().detach().cpu().numpy())
            class_probs[batch_index * loader.batch_size: (batch_index+1)*loader.batch_size, :] = (output - torch.logsumexp(output, dim=1, keepdims=True)).exp()

            loss_nll += -output.sum()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()

    loss_ce /= len(loader.dataset)
    loss_nll /= len(loader.dataset) + get_data_flatten_shape(loader)[1]
    accuracy = 100.0 * correct / len(loader.dataset)

    output_string = "{} set: Average loss_ce: {:.4f} Average loss_nll: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            tag, loss_ce, loss_nll, correct, len(loader.dataset), accuracy
    )
    print(output_string)
    with open(output_dir + 'training.out', 'a') as writer:
        writer.write(output_string + "\n")
    assert len(data_ll) == get_data_flatten_shape(loader)[0]
    assert len(data_ll_super) == get_data_flatten_shape(loader)[0]
    assert len(data_ll_unsup) == get_data_flatten_shape(loader)[0]
    return data_ll, class_probs.detach().cpu().numpy(), data_ll_super, data_ll_unsup, loss_ce, loss_nll




def evaluate_model_dropout(model: torch.nn.Module, device, loader, tag, dropout_inference=0.01, n_dropout_iters=100,
                           output_dir="", dropout_cf=False) -> float:
    """
    Description for method evaluate_model.

    Args:
        model (nn.Module): PyTorch module.
        device: Execution device.
        loader: Data loader.
        tag (str): Tag for information.

    Returns:
        float: Tuple of loss and accuracy.
    """
    model.eval()
    data_ll_super = [] # pick it from the label-th head
    data_ll_unsup = [] # pick the max one
    data_ll = []
    dropout_data_lls = torch.zeros(get_data_flatten_shape(loader)[0], model.config.C, n_dropout_iters).to(device)
    n_dropout_iters = n_dropout_iters
    class_probs = torch.zeros((get_data_flatten_shape(loader)[0], model.config.C, n_dropout_iters)).to(device)
    loss_nll = [0] * n_dropout_iters
    drop_corrects = 0
    loss_ce = [0] * n_dropout_iters
    # criterion = nn.CrossEntropyLoss(reduction="sum") #TODO NOTE same as cross_entropy_loss_with_logits in this case?
    criterion = nn.NLLLoss(reduction="sum")
    with torch.no_grad():
        for batch_index, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            data = data.view(data.shape[0], -1)
            data_ll_it = torch.zeros(data.shape[0]).to(device)
            data_ll_it_sq = torch.zeros(data.shape[0], n_dropout_iters).to(device)
            data_ll_it_heads = torch.zeros(data.shape[0], model.config.C, n_dropout_iters).to(device)
            for i in range(n_dropout_iters):
                #print(i)
                output = model(data, test_dropout=True, dropout_inference=dropout_inference, dropout_cf=dropout_cf)
                cprobs = output - torch.logsumexp(output, dim=1, keepdims=True)
                #print(output)
                data_ll_it += torch.logsumexp(output, dim=1)
                data_ll_it_sq[:, i] = torch.logsumexp(output, dim=1)
                data_ll_it_heads[:, :, i] = output
                class_probs[batch_index * loader.batch_size: (batch_index+1)*loader.batch_size, :, i] = (output - torch.logsumexp(output, dim=1, keepdims=True)).exp()
                loss_ce[i] += criterion(cprobs, target).item()  # sum up batch loss
                loss_nll[i] += -output.sum().cpu()

                if batch_index == len(loader) - 1:
                    dropout_data_lls[batch_index * loader.batch_size: batch_index * loader.batch_size + output.shape[0], :, i] = output
                else:
                    dropout_data_lls[batch_index * loader.batch_size: (batch_index+1)*loader.batch_size, :, i] = output

            drop_preds = class_probs[batch_index * loader.batch_size: (batch_index+1)*loader.batch_size, :, :].mean(dim=2).argmax(dim=1)
            drop_corrects += (drop_preds == target).sum()


            data_ll_it /= n_dropout_iters
            data_ll.extend(data_ll_it_sq.mean(dim=1).detach().cpu().numpy())
            data_ll_super.extend(data_ll_it_heads.mean(dim=2).gather(1, target.reshape(-1,1)).squeeze().detach().cpu().numpy())
            data_ll_unsup.extend(data_ll_it_heads.mean(dim=2).max(dim=1)[0].detach().cpu().numpy())
            #breakpoint()

    loss_ce = np.array(loss_ce)
    loss_nll = np.array(loss_nll)

    loss_ce /= len(loader.dataset)
    loss_nll /= len(loader.dataset) + get_data_flatten_shape(loader)[1]

    print("drop corrects: {}".format(drop_corrects/get_data_flatten_shape(loader)[0]))

    output_string = "{} set: Average loss_ce: {:.4f} \u00B1{:.4f} Average loss_nll: {:.4f} \u00B1{:.4f}, Accuracy: {:.4f}".format(
            tag, np.mean(loss_ce), np.std(loss_ce), np.mean(loss_nll), np.std(loss_nll), drop_corrects/get_data_flatten_shape(loader)[0])
    print(output_string)

    with open(output_dir + 'training.out', 'a') as writer:
        writer.write(output_string + "\n")
    assert len(data_ll) == get_data_flatten_shape(loader)[0]
    return data_ll, class_probs.detach().cpu().numpy(), data_ll_super, data_ll_unsup, dropout_data_lls.detach().cpu().numpy()


def evaluate_model_corrupted_digits(model: torch.nn.Module, device, loader, tag, dropout_inference=0.01, n_dropout_iters=100, output_dir="", corruption=corruptions.glass_blur, class_label=None, severity=None) -> float:

    d = output_dir
    d_samples = d + "samples/"
    d_model = d + "model/"
    ensure_dir(d)
    ensure_dir(d_samples)
    ensure_dir(d_model)

    model.eval()
    mean = 0.1307
    std = 0.3081

    transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean), (std))])
    dataset = datasets.MNIST(root='../data', train=False, download=True, transform=None)

    kwargs = {}
    if severity is not None: kwargs = {'severity': severity}
    # apply corruption
    corrupted_images = np.empty((len(dataset), 28, 28), dtype=np.uint8)
    for i in range(len(dataset)):
        corrupted_images[i] = round_and_astype(np.array(corruption(dataset[i][0], **kwargs)))

    corrupted_dataset = CustomTensorDataset(tensors=[torch.tensor(corrupted_images), torch.tensor(dataset.targets)], transform=transformer)

    if class_label is not None:
        targets = torch.tensor(corrupted_dataset.targets.clone().detach())
        target_idx = (targets == class_label).nonzero()
        sampler = torch.utils.data.sampler.SubsetRandomSampler(target_idx.reshape((-1,)))
        data_loader = torch.utils.data.DataLoader(corrupted_dataset, batch_size=100, shuffle=False, sampler=sampler)
    else:
        data_loader = torch.utils.data.DataLoader(corrupted_dataset, batch_size=100, shuffle=False)

    n_samples = 0

    with torch.no_grad():

        if dropout_inference == 0.0:
            n_dropout_iters = 1

        class_probs = torch.zeros(data_loader.dataset.data.shape[0], model.config.C).to(device)

        dropout_output = torch.zeros(data_loader.dataset.data.shape[0], model.config.C, n_dropout_iters).to(device)
        dropout_class_probs = torch.zeros(data_loader.dataset.data.shape[0], model.config.C, n_dropout_iters).to(device)
        dropout_softmax_output = torch.zeros(data_loader.dataset.data.shape[0], model.config.C, n_dropout_iters).to(device)

        n_correct = 0
        n_samples = 0


        for batch_index, (data, target) in enumerate(data_loader):
            data = data.to(device)
            data = data.view(data.shape[0], -1)

            n_samples += data.shape[0]

            output = model(data, test_dropout=False, dropout_inference=0.0)
            if batch_index == len(data_loader) - 1:
                class_probs[batch_index * data_loader.batch_size: batch_index * data_loader.batch_size + output.shape[0] , :] = (output - torch.logsumexp(output, dim=1, keepdims=True)).exp()
                assert torch.all(torch.isclose(class_probs[batch_index * data_loader.batch_size: batch_index * data_loader.batch_size + output.shape[0], :].sum(-1), torch.tensor((1.)), atol=1e-03)), class_probs[batch_index * data_loader.batch_size:batch_index * data_loader.batch_size + output.shape[0], :].sum(-1)
            else:
                class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :] = (output - torch.logsumexp(output, dim=1, keepdims=True)).exp()
                assert torch.all(torch.isclose(class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :].sum(-1), torch.tensor((1.)), atol=1e-03)), class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :].sum(-1)

            pred = output.argmax(dim=1)
            n_correct += (pred == target.to(device)).sum().item()

            for i in range(n_dropout_iters):
                if batch_index == len(data_loader) - 1:
                    dropout_output[batch_index * data_loader.batch_size: batch_index * data_loader.batch_size + output.shape[0], :, i] = model(data, test_dropout=True, dropout_inference=dropout_inference)

                    dropout_class_probs[batch_index * data_loader.batch_size:batch_index * data_loader.batch_size + output.shape[0], :, i] = \
                        (dropout_output[batch_index * data_loader.batch_size:batch_index * data_loader.batch_size + output.shape[0], :, i] - \
                        torch.logsumexp(dropout_output[batch_index * data_loader.batch_size:batch_index * data_loader.batch_size + output.shape[0], :, i], dim=1, keepdims=True)).exp()
                    assert torch.all(torch.isclose(dropout_class_probs[batch_index * data_loader.batch_size: batch_index * data_loader.batch_size + output.shape[0], :, i].sum(-1), torch.tensor((1.)), atol=1e-03)), dropout_class_probs[batch_index * data_loader.batch_size: batch_index * data_loader.batch_size + output.shape[0], :, i].sum(-1)
                else:
                    dropout_output[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i] = model(data, test_dropout=True, dropout_inference=dropout_inference)


                    dropout_class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i] = \
                        (dropout_output[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i] - \
                        torch.logsumexp(dropout_output[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i], dim=1, keepdims=True)).exp()
                    assert torch.all(torch.isclose(dropout_class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i].sum(-1), torch.tensor((1.)), atol=1e-03)), dropout_class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i].sum(-1)


        print("N of correct test predictions (w/o MCD): {}/{} ({}%)".format(n_correct, n_samples, (n_correct / n_samples)*100))

        dropout_class_probs = dropout_class_probs[:n_samples, :, :]
        class_probs = class_probs[:n_samples, :]

        if class_label is not None:
            drop_n_correct = (torch.argmax(torch.mean(dropout_class_probs, 2), dim=1) == class_label).sum()
        else:
            drop_n_correct = (torch.argmax(torch.mean(dropout_class_probs, 2), dim=1) == data_loader.dataset.targets.to(device)).sum()
        print("N of correct test precictions with MCD: DROP N of correct predictions of test samples: {}/{} ({}%)".format(drop_n_correct, n_samples, (drop_n_correct / n_samples)*100))

        return dropout_class_probs, class_probs

def evaluate_model_corrupted_cifar(model: torch.nn.Module, device, loader, tag, dropout_inference=0.01, n_dropout_iters=100, output_dir="", corruption='fog', corruption_level=1, class_label=None, corrupted_cifar_dir='') -> float:

    d = output_dir
    d_samples = d + "samples/"
    d_model = d + "model/"
    ensure_dir(d)
    ensure_dir(d_samples)
    ensure_dir(d_model)

    assert corruption in ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 'frost', 'gaussian_blur', 'gaussian_noise',
                          'glass_blur', 'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate', 'saturate', 'shot_noise', 'snow',
                          'spatter', 'speckle_noise', 'zoom_blur']
    corrupted_dataset = np.load(corrupted_cifar_dir + '{}.npy'.format(corruption))
    corrupted_dataset = corrupted_dataset[(corruption_level - 1)*10000 : (10000) * corruption_level ]

    labels = np.load(corrupted_cifar_dir + 'labels.npy')
    labels = labels[(corruption_level - 1)*10000 : (10000) * corruption_level ]
    assert corrupted_dataset.shape[0] == 10000
    assert labels.shape[0] == 10000

    model.eval()

    cifar10_transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

    corrupted_dataset = CustomTensorDataset(tensors=[torch.tensor(corrupted_dataset), torch.tensor(labels)], transform=cifar10_transformer)

    if class_label is not None:
        targets = torch.tensor(corrupted_dataset.targets.clone().detach())
        target_idx = (targets == class_label).nonzero()
        sampler = torch.utils.data.sampler.SubsetRandomSampler(target_idx.reshape((-1,)))
        data_loader = torch.utils.data.DataLoader(corrupted_dataset, batch_size=100, shuffle=False, sampler=sampler)
    else:
        data_loader = torch.utils.data.DataLoader(corrupted_dataset, batch_size=100, shuffle=False)


    n_samples = 0

    with torch.no_grad():

        if dropout_inference == 0.0:
            n_dropout_iters = 1

        class_probs = torch.zeros(data_loader.dataset.data.shape[0], model.config.C).to(device)

        dropout_output = torch.zeros(data_loader.dataset.data.shape[0], model.config.C, n_dropout_iters).to(device)
        dropout_class_probs = torch.zeros(data_loader.dataset.data.shape[0], model.config.C, n_dropout_iters).to(device)
        dropout_softmax_output = torch.zeros(data_loader.dataset.data.shape[0], model.config.C, n_dropout_iters).to(device)

        n_correct = 0
        n_samples = 0


        for batch_index, (data, target) in enumerate(data_loader):
            data = data.to(device)
            data = data.view(data.shape[0], -1)

            n_samples += data.shape[0]

            output = model(data, test_dropout=False, dropout_inference=0.0)
            if batch_index == len(data_loader) - 1:
                class_probs[batch_index * data_loader.batch_size: batch_index * data_loader.batch_size + output.shape[0] , :] = (output - torch.logsumexp(output, dim=1, keepdims=True)).exp()
                assert torch.all(torch.isclose(class_probs[batch_index * data_loader.batch_size: batch_index * data_loader.batch_size + output.shape[0], :].sum(-1), torch.tensor((1.)), atol=1e-03)), class_probs[batch_index * data_loader.batch_size:batch_index * data_loader.batch_size + output.shape[0], :].sum(-1)
            else:
                class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :] = (output - torch.logsumexp(output, dim=1, keepdims=True)).exp()
                assert torch.all(torch.isclose(class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :].sum(-1), torch.tensor((1.)), atol=1e-03)), class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :].sum(-1)

            pred = output.argmax(dim=1)
            n_correct += (pred == target.to(device)).sum().item()

            for i in range(n_dropout_iters):
                if batch_index == len(data_loader) - 1:
                    dropout_output[batch_index * data_loader.batch_size: batch_index * data_loader.batch_size + output.shape[0], :, i] = model(data, test_dropout=True, dropout_inference=dropout_inference)

                    dropout_class_probs[batch_index * data_loader.batch_size:batch_index * data_loader.batch_size + output.shape[0], :, i] = \
                        (dropout_output[batch_index * data_loader.batch_size:batch_index * data_loader.batch_size + output.shape[0], :, i] - \
                        torch.logsumexp(dropout_output[batch_index * data_loader.batch_size:batch_index * data_loader.batch_size + output.shape[0], :, i], dim=1, keepdims=True)).exp()
                    assert torch.all(torch.isclose(dropout_class_probs[batch_index * data_loader.batch_size: batch_index * data_loader.batch_size + output.shape[0], :, i].sum(-1), torch.tensor((1.)), atol=1e-03)), dropout_class_probs[batch_index * data_loader.batch_size: batch_index * data_loader.batch_size + output.shape[0], :, i].sum(-1)
                else:
                    dropout_output[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i] = model(data, test_dropout=True, dropout_inference=dropout_inference)

                    dropout_class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i] = \
                        (dropout_output[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i] - \
                        torch.logsumexp(dropout_output[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i], dim=1, keepdims=True)).exp()
                    assert torch.all(torch.isclose(dropout_class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i].sum(-1), torch.tensor((1.)), atol=1e-03)), dropout_class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i].sum(-1)

        print("N of correct test predictions (w/o MCD): {}/{} ({}%)".format(n_correct, n_samples, (n_correct / n_samples)*100))

        dropout_class_probs = dropout_class_probs[:n_samples, :, :]
        class_probs = class_probs[:n_samples, :]

        if class_label is not None:
            drop_n_correct = (torch.argmax(torch.mean(dropout_class_probs, 2), dim=1) == class_label).sum()
        else:
            drop_n_correct = (torch.argmax(torch.mean(dropout_class_probs, 2), dim=1) == data_loader.dataset.targets.to(device)).sum()
        print("N of correct test precictions with MCD: DROP N of correct predictions of test samples: {}/{} ({}%)".format(drop_n_correct, n_samples, (drop_n_correct / n_samples)*100))

        return dropout_class_probs, class_probs

def evaluate_model_corrupted_svhn(model: torch.nn.Module, device, loader, tag, dropout_inference=0.01,
                                  n_dropout_iters=100, output_dir="", corruption='fog', corruption_level=1,
                                  class_label=None, corrupted_svhn_dir='') -> float:

    d = output_dir
    d_samples = d + "samples/"
    d_model = d + "model/"
    ensure_dir(d)
    ensure_dir(d_samples)
    ensure_dir(d_model)


    corrupted_dataset = np.load(corrupted_svhn_dir + 'svhn_test_{}_l{}.npy'.format(corruption, corruption_level))
    labels = np.load(corrupted_svhn_dir + 'svhn_test_{}_l{}_labels.npy'.format(corruption, corruption_level))
    assert corrupted_dataset.shape[0] == labels.shape[0]
    assert labels.shape[0] == 26032

    model.eval()

    svhn_transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4377, 0.4438, 0.4728), (0.198, 0.201, 0.197))])

    corrupted_dataset = CustomTensorDataset(tensors=[torch.tensor(corrupted_dataset), torch.tensor(labels)], transform=svhn_transformer)

    if class_label is not None:
        targets = torch.tensor(corrupted_dataset.targets.clone().detach())
        target_idx = (targets == class_label).nonzero()
        sampler = torch.utils.data.sampler.SubsetRandomSampler(target_idx.reshape((-1,)))
        data_loader = torch.utils.data.DataLoader(corrupted_dataset, batch_size=100, shuffle=False, sampler=sampler)
    else:
        data_loader = torch.utils.data.DataLoader(corrupted_dataset, batch_size=100, shuffle=False)


    n_samples = 0

    with torch.no_grad():

        if dropout_inference == 0.0:
            n_dropout_iters = 1

        class_probs = torch.zeros(data_loader.dataset.data.shape[0], model.config.C).to(device)

        dropout_output = torch.zeros(data_loader.dataset.data.shape[0], model.config.C, n_dropout_iters).to(device)
        dropout_class_probs = torch.zeros(data_loader.dataset.data.shape[0], model.config.C, n_dropout_iters).to(device)
        dropout_softmax_output = torch.zeros(data_loader.dataset.data.shape[0], model.config.C, n_dropout_iters).to(device)

        n_correct = 0
        n_samples = 0


        for batch_index, (data, target) in enumerate(data_loader):
            data = data.to(device)
            data = data.view(data.shape[0], -1)

            n_samples += data.shape[0]

            output = model(data, test_dropout=False, dropout_inference=0.0)
            if batch_index == len(data_loader) - 1:
                class_probs[batch_index * data_loader.batch_size: batch_index * data_loader.batch_size + output.shape[0] , :] = (output - torch.logsumexp(output, dim=1, keepdims=True)).exp()
                assert torch.all(torch.isclose(class_probs[batch_index * data_loader.batch_size: batch_index * data_loader.batch_size + output.shape[0], :].sum(-1), torch.tensor((1.)), atol=1e-03)), class_probs[batch_index * data_loader.batch_size:batch_index * data_loader.batch_size + output.shape[0], :].sum(-1)
            else:
                class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :] = (output - torch.logsumexp(output, dim=1, keepdims=True)).exp()
                assert torch.all(torch.isclose(class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :].sum(-1), torch.tensor((1.)), atol=1e-03)), class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :].sum(-1)

            pred = output.argmax(dim=1)
            n_correct += (pred == target.to(device)).sum().item()

            for i in range(n_dropout_iters):
                if batch_index == len(data_loader) - 1:
                    dropout_output[batch_index * data_loader.batch_size: batch_index * data_loader.batch_size + output.shape[0], :, i] = model(data, test_dropout=True, dropout_inference=dropout_inference)

                    dropout_class_probs[batch_index * data_loader.batch_size:batch_index * data_loader.batch_size + output.shape[0], :, i] = \
                        (dropout_output[batch_index * data_loader.batch_size:batch_index * data_loader.batch_size + output.shape[0], :, i] - \
                        torch.logsumexp(dropout_output[batch_index * data_loader.batch_size:batch_index * data_loader.batch_size + output.shape[0], :, i], dim=1, keepdims=True)).exp()
                    assert torch.all(torch.isclose(dropout_class_probs[batch_index * data_loader.batch_size: batch_index * data_loader.batch_size + output.shape[0], :, i].sum(-1), torch.tensor((1.)), atol=1e-03)), dropout_class_probs[batch_index * data_loader.batch_size: batch_index * data_loader.batch_size + output.shape[0], :, i].sum(-1)
                else:
                    dropout_output[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i] = model(data, test_dropout=True, dropout_inference=dropout_inference)

                    dropout_class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i] = \
                        (dropout_output[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i] - \
                        torch.logsumexp(dropout_output[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i], dim=1, keepdims=True)).exp()
                    assert torch.all(torch.isclose(dropout_class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i].sum(-1), torch.tensor((1.)), atol=1e-03)), dropout_class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i].sum(-1)

        print("N of correct test predictions (w/o MCD): {}/{} ({}%)".format(n_correct, n_samples, (n_correct / n_samples)*100))

        dropout_class_probs = dropout_class_probs[:n_samples, :, :]
        class_probs = class_probs[:n_samples, :]

        if class_label is not None:
            drop_n_correct = (torch.argmax(torch.mean(dropout_class_probs, 2), dim=1) == class_label).sum()
        else:
            drop_n_correct = (torch.argmax(torch.mean(dropout_class_probs, 2), dim=1) == data_loader.dataset.targets.to(device)).sum()
        print("N of correct test precictions with MCD: DROP N of correct predictions of test samples: {}/{} ({}%)".format(drop_n_correct, n_samples, (drop_n_correct / n_samples)*100))

        return dropout_class_probs, class_probs

def evaluate_model_rotated_digits(model: torch.nn.Module, device, loader, tag, dropout_inference=0.01,
                                  n_dropout_iters=100, output_dir="", degrees=30, class_label=None) -> float:

    d = output_dir
    d_samples = d + "samples/"
    d_model = d + "model/"
    ensure_dir(d)
    ensure_dir(d_samples)
    ensure_dir(d_model)

    model.eval()
    mean = 0.1307
    std = 0.3081

    transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean), (std))])
    dataset = datasets.MNIST(root='../data', train=False, download=True, transform=transformer)

    if class_label is not None:
        targets = torch.tensor(dataset.targets.clone().detach())
        target_idx = (targets == class_label).nonzero()
        sampler = torch.utils.data.sampler.SubsetRandomSampler(target_idx.reshape((-1,)))
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, sampler=sampler)
    else:
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False)


    n_samples = 0

    with torch.no_grad():

        if dropout_inference == 0.0:
            n_dropout_iters = 1

        class_probs = torch.zeros(data_loader.dataset.data.shape[0], model.config.C).to(device)

        dropout_output = torch.zeros(data_loader.dataset.data.shape[0], model.config.C, n_dropout_iters).to(device)
        dropout_class_probs = torch.zeros(data_loader.dataset.data.shape[0], model.config.C, n_dropout_iters).to(device)
        dropout_softmax_output = torch.zeros(data_loader.dataset.data.shape[0], model.config.C, n_dropout_iters).to(device)

        n_correct = 0
        n_samples = 0


        for batch_index, (data, target) in enumerate(data_loader):
            data = data.to(device)
            data = torchvision.transforms.functional.rotate(data.reshape(-1,1,28,28), degrees, fill=-mean/std).reshape(-1,28,28)
            data = data.view(data.shape[0], -1)

            n_samples += data.shape[0]

            output = model(data, test_dropout=False, dropout_inference=0.0)
            if batch_index == len(data_loader) - 1:
                class_probs[batch_index * data_loader.batch_size: batch_index * data_loader.batch_size + output.shape[0] , :] = (output - torch.logsumexp(output, dim=1, keepdims=True)).exp()
                assert torch.all(torch.isclose(class_probs[batch_index * data_loader.batch_size: batch_index * data_loader.batch_size + output.shape[0], :].sum(-1), torch.tensor((1.)), atol=1e-03)), class_probs[batch_index * data_loader.batch_size:batch_index * data_loader.batch_size + output.shape[0], :].sum(-1)
            else:
                class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :] = (output - torch.logsumexp(output, dim=1, keepdims=True)).exp()
                assert torch.all(torch.isclose(class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :].sum(-1), torch.tensor((1.)), atol=1e-03)), class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :].sum(-1)

            pred = output.argmax(dim=1)
            n_correct += (pred == target.to(device)).sum().item()

            for i in range(n_dropout_iters):
                if batch_index == len(data_loader) - 1:
                    dropout_output[batch_index * data_loader.batch_size: batch_index * data_loader.batch_size + output.shape[0], :, i] = model(data, test_dropout=True, dropout_inference=dropout_inference)

                    dropout_class_probs[batch_index * data_loader.batch_size:batch_index * data_loader.batch_size + output.shape[0], :, i] = \
                        (dropout_output[batch_index * data_loader.batch_size:batch_index * data_loader.batch_size + output.shape[0], :, i] - \
                        torch.logsumexp(dropout_output[batch_index * data_loader.batch_size:batch_index * data_loader.batch_size + output.shape[0], :, i], dim=1, keepdims=True)).exp()
                    assert torch.all(torch.isclose(dropout_class_probs[batch_index * data_loader.batch_size: batch_index * data_loader.batch_size + output.shape[0], :, i].sum(-1), torch.tensor((1.)), atol=1e-03)), dropout_class_probs[batch_index * data_loader.batch_size: batch_index * data_loader.batch_size + output.shape[0], :, i].sum(-1)
                else:
                    dropout_output[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i] = model(data, test_dropout=True, dropout_inference=dropout_inference)


                    dropout_class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i] = \
                        (dropout_output[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i] - \
                        torch.logsumexp(dropout_output[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i], dim=1, keepdims=True)).exp()
                    assert torch.all(torch.isclose(dropout_class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i].sum(-1), torch.tensor((1.)), atol=1e-03)), dropout_class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i].sum(-1)

        print("N of correct test predictions (w/o MCD): {}/{} ({}%)".format(n_correct, n_samples, (n_correct / n_samples)*100))

        dropout_class_probs = dropout_class_probs[:n_samples, :, :]
        class_probs = class_probs[:n_samples, :]

        if class_label is not None:
            drop_n_correct = (torch.argmax(torch.mean(dropout_class_probs, 2), dim=1) == class_label).sum()
        else:
            drop_n_correct = (torch.argmax(torch.mean(dropout_class_probs, 2), dim=1) == data_loader.dataset.targets.to(device)).sum()
        print("N of correct test precictions with MCD: DROP N of correct predictions of test samples: {}/{} ({}%)".format(drop_n_correct, n_samples, (drop_n_correct / n_samples)*100))

        return dropout_class_probs, class_probs


def ensure_dir(path: str):
    """
    Ensure that a directory exists.

    For 'foo/bar/baz.csv' the directories 'foo' and 'bar' will be created if not already present.

    Args:
        path (str): Directory path.
    """
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)


def plot_samples(x: torch.Tensor, path):
    """
    Plot a single sample witht the target and prediction in the title.

    Args:
        x (torch.Tensor): Batch of input images. Has to be shape: [N, C, H, W].
    """
    # Normalize in valid range
    for i in range(x.shape[0]):
        x[i, :] = (x[i, :] - x[i, :].min()) / (x[i, :].max() - x[i, :].min())

    tensors = torchvision.utils.make_grid(x, nrow=10, padding=1).cpu()
    arr = tensors.permute(1, 2, 0).numpy()
    arr = skimage.img_as_ubyte(arr)
    imageio.imwrite(path, arr)

def plot_training_curves(data, filename='loss_curves', title="", path=None):
    training_loss = np.hstack((np.arange(len(data['training_loss'])).reshape(-1,1), np.array(data['training_loss']).reshape(-1,1)))
    training_loss_ce = np.hstack((np.arange(len(data['training_loss_ce'])).reshape(-1,1), np.array(data['training_loss_ce']).reshape(-1,1)))
    training_loss_nll = np.hstack((np.arange(len(data['training_loss_nll'])).reshape(-1,1), np.array(data['training_loss_nll']).reshape(-1,1)))

    test_loss = np.array(data['test_loss'])
    test_loss_ce = np.array(data['test_loss_ce'])
    test_loss_nll = np.array(data['test_loss_nll'])

    fig, axs = plt.subplots()
    axs.plot(training_loss[:,0], training_loss[:,1], label='Training loss')
    axs.plot(test_loss[:,0], test_loss[:,1], label='Test loss')

    plt.legend(loc=0)
    plt.title('Loss curves')

    plt.savefig(path + 'loss_curves.png')
    plt.savefig(path + 'loss_curves.pdf')
    plt.close()

    fig, axs = plt.subplots()
    axs.plot(training_loss_ce[:,0], training_loss_ce[:,1], label='Training CE loss')
    axs.plot(test_loss_ce[:,0], test_loss_ce[:,1], label='Test CE loss')

    plt.legend(loc=0)
    plt.title('CE loss curves')

    plt.savefig(path + 'ce_loss_curves.png')
    plt.savefig(path + 'ce_loss_curves.pdf')
    plt.close()


    fig, axs = plt.subplots()
    axs.plot(training_loss_nll[:,0], training_loss_nll[:,1], label='Training NLL loss')
    axs.plot(test_loss_nll[:,0], test_loss_nll[:,1], label='Test NLL loss')

    plt.legend(loc=0)
    plt.title('NLL loss curves')

    plt.savefig(path + 'nll_loss_curves.png')
    plt.savefig(path + 'nll_loss_curves.pdf')
    plt.close()

    # ax = plt.gca()
    # ax.set_ylim(ylimits)
    # ax.set_xlabel(xlabel)
    # ax.set_ylabel(ylabel)



def plot_boxplot(data, filename='boxplot', title="", path=None):

    for i in range(data.shape[1]):
        fig, axs = plt.subplots()
        axs.boxplot(np.transpose(data[:, i, :]), showmeans=True, meanline=True)
        ax = plt.gca()
        ax.set_ylim([-0.1, 1.1])
        plt.title("{}Digit {}".format(title, i))
        plt.savefig(path + filename + '_{}.png'.format(i))
        plt.savefig(path + filename + '_{}.pdf'.format(i))
        plt.close()

def plot_boxplot_rotating_digits(data, filename='boxplot_rotating_digits', title="", path=None, ylimits=[-0.1, 1.1], xlabel='Rotation (degrees)', ylabel='Classification confidence'):

    fig, axs = plt.subplots()
    axs.boxplot(data, showmeans=True, meanline=True, labels=['-180', '-150', '-120', '-90', '-60', '-30', '0', '30', '60', '90', '120', '150'], showfliers=False)
    ax = plt.gca()
    ax.set_ylim(ylimits)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.title(title)
    plt.savefig(path + filename + '.png')
    plt.savefig(path + filename + '.pdf')
    plt.close()

def plot_boxplot_corrupted_digits(data, filename='boxplot_corrupted_digits', title="", path=None, ylimits=[-0.1, 1.1], xlabel='Corruption ID', ylabel='Classification confidence'):

    fig, axs = plt.subplots()
    axs.boxplot(data, showmeans=True, meanline=True, labels=['SN', 'IN', 'GB', 'MB', 'SH', 'SC', 'RO', 'BR', 'TR', 'ST', 'FOG', 'SPA', 'DOT', 'ZIG', 'CAE'], showfliers=False)
    ax = plt.gca()
    ax.set_ylim(ylimits)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.title(title)
    plt.savefig(path + filename + '.png')
    plt.savefig(path + filename + '.pdf')
    plt.close()

def plot_histograms(lls_dict, filename='histogram', title="", path=None, trained_on_fmnist=False, y_lim=None):

    dataset_names = [trained_on_fmnist.upper(), get_other_dataset_name(trained_on_fmnist).upper()]
    if not trained_on_fmnist: dataset_names.reverse()

    if not "train_lls" in lls_dict.keys():
        lls_dict_renamed = {}
        for k, v in lls_dict.items():
            if "other_train" in k or "other_class_probs_train" in k:
                lls_dict_renamed["other_train_lls"] = v
            if "other_test" in k or "other_class_probs_test" in k:
                lls_dict_renamed["other_test_lls"] = v
            if "train_lls" in k or "class_probs_train" in k:
                lls_dict_renamed["train_lls"] = v
            if "test_lls" in k or "class_probs_test" in k:
                lls_dict_renamed["test_lls"] = v
        lls_dict = lls_dict_renamed

    plt.hist(np.nan_to_num(lls_dict["train_lls"]), label=dataset_names[0] + " Train", alpha=0.5, bins=20, color='red')
    plt.hist(np.nan_to_num(lls_dict["test_lls"]), label=dataset_names[0] + " Test", alpha=0.5, bins=20, color='blue')
    plt.hist(np.nan_to_num(lls_dict["other_train_lls"]), label=dataset_names[1] + " Train (OOD)", alpha=0.5, bins=20, color='orange')
    plt.hist(np.nan_to_num(lls_dict["other_test_lls"]), label=dataset_names[1] + " Test (OOD)", alpha=0.5, bins=20, color='yellow')
    plt.legend(loc=0)
    plt.title("Histogram {}".format(title))
    if y_lim:
        plt.ylim((None, y_lim))
    plt.savefig(path + filename + '.png')
    plt.savefig(path + filename + '.pdf')
    plt.close()

def save_samples(samples, iteration: int):
    d = "results/samples/"
    ensure_dir(d)
    plot_samples(samples.view(-1, 1, 28, 28), path=os.path.join(d, f"mnist-{iteration:03}.png"))


def set_seed(seed: int):
    """
    Set the seed globally for python, numpy and torch.

    Args:
        seed (int): Seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    torch.cuda.benchmark = True
    set_seed(0)

    corrupted_cifar_dir = '/home/fabrizio/research/CIFAR-10-C/'
    corrupted_svhn_dir = '/home/fabrizio/research/svhn_c/'

    # torch.autograd.set_detect_anomaly(True)

    # run_torch(200, 200, dropout_inference=0.2, dropout_spn=0.2,
    #           training_dataset='nltcs', lmbda=1.0, eval_single_digit=False, toy_setting=False,
    #           eval_rotation=False, mnist_corruptions=False, n_mcd_passes=10, corrupted_cifar_dir=corrupted_cifar_dir,
    #           eval_every_n_epochs=5, lr=0.001)
    # evaluate_corrupted_svhn(model_dir='results/2022-05-15_00-45-40/model/', dropout_inference=0.2,
    #                         n_mcd_passes=100, corrupted_svhn_dir=corrupted_svhn_dir)
    # evaluate_corrupted_svhn(model_dir='results/2022-05-15_00-45-40/model/', dropout_inference=0.1,
    #                         n_mcd_passes=100, corrupted_svhn_dir=corrupted_svhn_dir)

    # plot_sum_weights(model_dir='results/2022-07-04_15-09-25/model/', training_dataset='nltcs',
    #                dropout_inference=0.2, n_mcd_passes=100, rat_S=2, rat_I=4, rat_D=2, rat_R=1)
    # post_hoc_exps(model_dir='results/2022-07-04_15-09-25/model/', training_dataset='nltcs',
    #                dropout_inference=[0.8], n_mcd_passes=[10], rat_S=2, rat_I=4, rat_D=2, rat_R=1)
    # plot_sum_weights(model_dir='results/2022-07-01_19-36-52/model/', training_dataset='nltcs',
    #                dropout_inference=0.2, n_mcd_passes=100, rat_S=2, rat_I=4, rat_D=2, rat_R=1)
    # post_hoc_exps(model_dir='results/2022-07-01_19-36-52/model/', training_dataset='nltcs',
    #                dropout_inference=[0.8], n_mcd_passes=[10], rat_S=2, rat_I=4, rat_D=2, rat_R=1)

    # test_closed_form(model_dir='results/2022-08-29_13-50-24/model/', training_dataset='mnist', dropout_inference=0.01)
    # test_closed_form(model_dir='results/2022-08-29_13-50-24/model/', training_dataset='mnist', dropout_inference=0.05)
    # test_closed_form(model_dir='results/2022-08-29_13-50-24/model/', training_dataset='mnist', dropout_inference=0.1)
    # m1 = test_closed_form(model_dir='results/2022-08-29_13-50-24/model/', training_dataset='mnist', dropout_inference=0.2,
    #                  batch_size=500, rotation=None)
    # test_closed_form(model_dir='results/2022-08-29_13-50-24/model/', training_dataset='mnist', dropout_inference=0.2,
    #                  batch_size=500, rotation=None, model=m1)
    # test_closed_form(model_dir='results/2022-08-29_13-50-24/model/', training_dataset=get_other_dataset_name('mnist'),
    #                  dropout_inference=0.2, batch_size=500, rotation=None, model=m1)

    m1 = test_closed_form(model_dir='results/2022-09-14_14-28-01/model/', training_dataset='fmnist',
                          dropout_inference=0.2,
                          batch_size=500, rotation=None)
    # test_closed_form(model_dir='results/2022-09-14_14-28-01/model/', training_dataset='fmnist', dropout_inference=0.2,
    #                  batch_size=500, rotation=None, model=m1)
    test_closed_form(model_dir='results/2022-09-14_14-28-01/model/', training_dataset=get_other_dataset_name('fmnist'),
                     dropout_inference=0.2, batch_size=500, rotation=None, model=m1)

    # test_closed_form(model_dir='results/2022-08-29_13-50-24/model/', training_dataset='mnist', dropout_inference=0.2,
    #                  batch_size=500, rotation=None)
    # test_closed_form(model_dir='results/2022-08-29_13-50-24/model/', training_dataset='mnist', dropout_inference=0.2,
    #                  batch_size=500, rotation=None, model=m1)
    # test_closed_form(model_dir='results/2022-08-29_13-50-24/model/', training_dataset='mnist', dropout_inference=0.2,
    #                  batch_size=500, rotation=None)
    # test_closed_form(model_dir='results/2022-09-09_14-48-39/model/', training_dataset='mnist', dropout_inference=0.2,
    #                  batch_size=200, rotation=90)
    # test_closed_form(model_dir='results/2022-08-29_13-50-24/model/', training_dataset='mnist', dropout_inference=0.2)
    # test_closed_form(model_dir='results/2022-08-29_13-50-24/model/', training_dataset='mnist', dropout_inference=0.2)
    # test_closed_form(model_dir='results/2022-08-29_13-50-24/model/', training_dataset='mnist', dropout_inference=0.2)
    # test_closed_form(model_dir='results/2022-08-29_13-50-24/model/', training_dataset='mnist', dropout_inference=0.5)

    # run_torch(200, 200, dropout_inference=0.2, dropout_spn=0.2,
    #           training_dataset='mnist', lmbda=1.0, eval_single_digit=False, toy_setting=False,
    #           eval_rotation=False, mnist_corruptions=False, n_mcd_passes=10, corrupted_cifar_dir=corrupted_cifar_dir,
    #           eval_every_n_epochs=5, lr=0.001, dropout_cf=True)

    # plot_sum_weights(model_dir='results/2022-06-02_23-13-30/model/', training_dataset='fmnist',
    #               dropout_inference=0.2, n_mcd_passes=100)
    # post_hoc_exps(model_dir='results/2022-06-02_23-13-30/model/', training_dataset='fmnist',
    #               dropout_inference=[0.2], n_mcd_passes=[100])
    # post_hoc_exps(model_dir='results/2022-05-15_00-45-40/model/', training_dataset='svhn',
    #                dropout_inference=[0.2], n_mcd_passes=[100])

    # # experiment useful to check the variance on OODs with the closed form
    # # dropout p = 0.2 (also at learning time)
    # run_torch(200, 200, dropout_inference=0.2, dropout_spn=0.2,
    #           training_dataset='fmnist', lmbda=1.0, eval_single_digit=False, toy_setting=False,
    #           eval_rotation=False, mnist_corruptions=False, n_mcd_passes=100, corrupted_cifar_dir=corrupted_cifar_dir,
    #           eval_every_n_epochs=5, lr=0.001)
    # # dropout p = 0.1 (also at learning time)
    # run_torch(200, 200, dropout_inference=0.1, dropout_spn=0.1,
    #           training_dataset='fmnist', lmbda=1.0, eval_single_digit=False, toy_setting=False,
    #           eval_rotation=False, mnist_corruptions=False, n_mcd_passes=100, corrupted_cifar_dir=corrupted_cifar_dir,
    #           eval_every_n_epochs=5, lr=0.001)
    # dropout p = 0.2 (also at learning time)
    # run_torch(200, 200, dropout_inference=0.2, dropout_spn=0.2,
    #           training_dataset='mnist', lmbda=1.0, eval_single_digit=False, toy_setting=False,
    #           eval_rotation=False, mnist_corruptions=False, n_mcd_passes=100, corrupted_cifar_dir=corrupted_cifar_dir,
    #           eval_every_n_epochs=5, lr=0.001)
    # # dropout p = 0.1 (also at learning time)
    # run_torch(200, 200, dropout_inference=0.1, dropout_spn=0.1,
    #           training_dataset='mnist', lmbda=1.0, eval_single_digit=False, toy_setting=False,
    #           eval_rotation=False, mnist_corruptions=False, n_mcd_passes=100, corrupted_cifar_dir=corrupted_cifar_dir,
    #           eval_every_n_epochs=5, lr=0.001)
    #
    # # run experiments on MNIST with corrupted MNIST and rotating digits
    # # dropout p = 0.2 (also at learning time)
    # run_torch(200, 200, dropout_inference=0.2, dropout_spn=0.2,
    #           training_dataset='mnist', lmbda=1.0, eval_single_digit=False, toy_setting=False,
    #           eval_rotation=True, mnist_corruptions=True, n_mcd_passes=100, corrupted_cifar_dir=corrupted_cifar_dir,
    #           eval_every_n_epochs=5, lr=0.001)
    #
    # # dropout p = 0.1 (also at learning time)
    # run_torch(200, 200, dropout_inference=0.1, dropout_spn=0.1,
    #           training_dataset='mnist', lmbda=1.0, eval_single_digit=False, toy_setting=False,
    #           eval_rotation=True, mnist_corruptions=True, n_mcd_passes=100, corrupted_cifar_dir=corrupted_cifar_dir,
    #           eval_every_n_epochs=5, lr=0.001)
    #
    # # Set of experiments to learn the impact of lambda on LL separation vs classification accuracy
    # run_torch(200, 200, dropout_inference=0.0, dropout_spn=0.0,
    #           training_dataset='mnist', lmbda=0.8, eval_single_digit=False, toy_setting=False,
    #           eval_rotation=False, mnist_corruptions=False, n_mcd_passes=100, corrupted_cifar_dir=corrupted_cifar_dir,
    #           eval_every_n_epochs=5, lr=0.001)
    #
    # run_torch(200, 200, dropout_inference=0.0, dropout_spn=0.0,
    #           training_dataset='mnist', lmbda=0.6, eval_single_digit=False, toy_setting=False,
    #           eval_rotation=False, mnist_corruptions=False, n_mcd_passes=100, corrupted_cifar_dir=corrupted_cifar_dir,
    #           eval_every_n_epochs=5, lr=0.001)
    #
    # run_torch(200, 200, dropout_inference=0.0, dropout_spn=0.0,
    #           training_dataset='mnist', lmbda=0.4, eval_single_digit=False, toy_setting=False,
    #           eval_rotation=False, mnist_corruptions=False, n_mcd_passes=100, corrupted_cifar_dir=corrupted_cifar_dir,
    #           eval_every_n_epochs=5, lr=0.001)
    #
    # run_torch(200, 200, dropout_inference=0.0, dropout_spn=0.0,
    #           training_dataset='mnist', lmbda=0.2, eval_single_digit=False, toy_setting=False,
    #           eval_rotation=False, mnist_corruptions=False, n_mcd_passes=100, corrupted_cifar_dir=corrupted_cifar_dir,
    #           eval_every_n_epochs=5, lr=0.001)
    #
    # run_torch(200, 200, dropout_inference=0.0, dropout_spn=0.0,
    #           training_dataset='mnist', lmbda=0.0, eval_single_digit=False, toy_setting=False,
    #           eval_rotation=False, mnist_corruptions=False, n_mcd_passes=100, corrupted_cifar_dir=corrupted_cifar_dir,
    #           eval_every_n_epochs=5, lr=0.001)
    #
    # run_torch(200, 200, dropout_inference=0.0, dropout_spn=0.0,
    #           training_dataset='fmnist', lmbda=1.0, eval_single_digit=False, toy_setting=False,
    #           eval_rotation=False, mnist_corruptions=False, n_mcd_passes=100, corrupted_cifar_dir=corrupted_cifar_dir,
    #           eval_every_n_epochs=5, lr=0.001)
    #
    # run_torch(200, 200, dropout_inference=0.0, dropout_spn=0.0,
    #           training_dataset='fmnist', lmbda=0.8, eval_single_digit=False, toy_setting=False,
    #           eval_rotation=False, mnist_corruptions=False, n_mcd_passes=100, corrupted_cifar_dir=corrupted_cifar_dir,
    #           eval_every_n_epochs=5, lr=0.001)
    #
    # run_torch(200, 200, dropout_inference=0.0, dropout_spn=0.0,
    #           training_dataset='fmnist', lmbda=0.6, eval_single_digit=False, toy_setting=False,
    #           eval_rotation=False, mnist_corruptions=False, n_mcd_passes=100, corrupted_cifar_dir=corrupted_cifar_dir,
    #           eval_every_n_epochs=5, lr=0.001)
    #
    # run_torch(200, 200, dropout_inference=0.0, dropout_spn=0.0,
    #           training_dataset='fmnist', lmbda=0.4, eval_single_digit=False, toy_setting=False,
    #           eval_rotation=False, mnist_corruptions=False, n_mcd_passes=100, corrupted_cifar_dir=corrupted_cifar_dir,
    #           eval_every_n_epochs=5, lr=0.001)
    #
    # run_torch(200, 200, dropout_inference=0.0, dropout_spn=0.0,
    #           training_dataset='fmnist', lmbda=0.2, eval_single_digit=False, toy_setting=False,
    #           eval_rotation=False, mnist_corruptions=False, n_mcd_passes=100, corrupted_cifar_dir=corrupted_cifar_dir,
    #           eval_every_n_epochs=5, lr=0.001)
    #
    # run_torch(200, 200, dropout_inference=0.0, dropout_spn=0.0,
    #           training_dataset='fmnist', lmbda=0.0, eval_single_digit=False, toy_setting=False,
    #           eval_rotation=False, mnist_corruptions=False, n_mcd_passes=100, corrupted_cifar_dir=corrupted_cifar_dir,
    #           eval_every_n_epochs=5, lr=0.001)
    #
    # # MCD experiments on SVHN
    # # dropout p = 0.1 (also at learning time)
    # run_torch(200, 200, dropout_inference=0.1, dropout_spn=0.1,
    #           training_dataset='svhn', lmbda=1.0, eval_single_digit=False, toy_setting=False,
    #           eval_rotation=False, mnist_corruptions=False, n_mcd_passes=100, corrupted_cifar_dir=corrupted_cifar_dir,
    #           eval_every_n_epochs=5, lr=0.001)
    #
    # # dropout p = 0.2 (also at learning time)
    # run_torch(200, 200, dropout_inference=0.2, dropout_spn=0.2,
    #           training_dataset='svhn', lmbda=1.0, eval_single_digit=False, toy_setting=False,
    #           eval_rotation=False, mnist_corruptions=False, n_mcd_passes=100, corrupted_cifar_dir=corrupted_cifar_dir,
    #           eval_every_n_epochs=5, lr=0.001)



    # run_torch(200, 200, dropout_inference=0.2, dropout_spn=0.2,
    #           training_dataset='fmnist', lmbda=1.0, eval_single_digit=False, toy_setting=False,
    #           eval_rotation=False, mnist_corruptions=False, n_mcd_passes=100, corrupted_cifar_dir=corrupted_cifar_dir,
    #           eval_every_n_epochs=5, lr=0.001)
    # run_torch(200, 200, dropout_inference=0.2, dropout_spn=0.2,
    #           training_dataset='svhn', lmbda=1.0, eval_single_digit=False, toy_setting=False,
    #           eval_rotation=False, mnist_corruptions=False, n_mcd_passes=100, corrupted_cifar_dir=corrupted_cifar_dir,
    #           eval_every_n_epochs=5, lr=0.001)

    # run_torch(5, 200, dropout_inference=0.2, dropout_spn=0.0,
    #           training_dataset='mnist', lmbda=1.0, eval_single_digit=False, toy_setting=True,
    #           eval_rotation=False, mnist_corruptions=False, n_mcd_passes=10, corrupted_cifar_dir=corrupted_cifar_dir,
    #           eval_every_n_epochs=5, lr=0.001)

    # run_torch(5, 200, dropout_inference=0.9, dropout_spn=0.2,
    #           training_dataset='mnist', lmbda=1.0, eval_single_digit=False, toy_setting=False,
    #           eval_rotation=False, mnist_corruptions=False, n_mcd_passes=10, corrupted_cifar_dir=corrupted_cifar_dir,
    #           eval_every_n_epochs=5)

    # run_torch(100, 200, dropout_inference=0.2, dropout_spn=0.2,
    #           training_dataset='mnist', lmbda=1.0, eval_single_digit=False, toy_setting=True,
    #           eval_rotation=False, mnist_corruptions=False, n_mcd_passes=10, corrupted_cifar_dir=corrupted_cifar_dir,
    #           eval_every_n_epochs=5)

    # run_torch(200, 200, dropout_inference=0.8, dropout_spn=0.2,
    #           training_dataset='mnist', lmbda=1.0, eval_single_digit=False, toy_setting=True,
    #           eval_rotation=False, mnist_corruptions=False, n_mcd_passes=10, corrupted_cifar_dir=corrupted_cifar_dir,
    #           eval_every_n_epochs=5)
    #
    # run_torch(200, 200, dropout_inference=0.8, dropout_spn=0.2,
    #           training_dataset='mnist', lmbda=1.0, eval_single_digit=False, toy_setting=True,
    #           eval_rotation=False, mnist_corruptions=False, n_mcd_passes=100, corrupted_cifar_dir=corrupted_cifar_dir,
    #           eval_every_n_epochs=5)
    #
    # run_torch(200, 200, dropout_inference=0.2, dropout_spn=0.2,
    #           training_dataset='mnist', lmbda=1.0, eval_single_digit=False, toy_setting=True,
    #           eval_rotation=False, mnist_corruptions=False, n_mcd_passes=10, corrupted_cifar_dir=corrupted_cifar_dir,
    #           eval_every_n_epochs=5)
    #
    # run_torch(200, 200, dropout_inference=0.2, dropout_spn=0.2,
    #           training_dataset='mnist', lmbda=1.0, eval_single_digit=False, toy_setting=True,
    #           eval_rotation=False, mnist_corruptions=False, n_mcd_passes=100, corrupted_cifar_dir=corrupted_cifar_dir,
    #           eval_every_n_epochs=5)

    # run_torch(200, 200, dropout_inference=0.0, dropout_spn=0.1,
    #           training_dataset='mnist', lmbda=1.0, eval_single_digit=False, toy_setting=False,
    #           eval_rotation=False, mnist_corruptions=False, n_mcd_passes=1, corrupted_cifar_dir=corrupted_cifar_dir,
    #           eval_every_n_epochs=5)
    #
    # run_torch(200, 200, dropout_inference=0.0, dropout_spn=0.1,
    #           training_dataset='mnist', lmbda=0.8, eval_single_digit=False, toy_setting=False,
    #           eval_rotation=False, mnist_corruptions=False, n_mcd_passes=1, corrupted_cifar_dir=corrupted_cifar_dir,
    #           eval_every_n_epochs=5)
    #
    # run_torch(200, 200, dropout_inference=0.0, dropout_spn=0.1,
    #           training_dataset='mnist', lmbda=0.6, eval_single_digit=False, toy_setting=False,
    #           eval_rotation=False, mnist_corruptions=False, n_mcd_passes=1, corrupted_cifar_dir=corrupted_cifar_dir,
    #           eval_every_n_epochs=5)
    #
    # run_torch(200, 200, dropout_inference=0.0, dropout_spn=0.1,
    #           training_dataset='mnist', lmbda=0.4, eval_single_digit=False, toy_setting=False,
    #           eval_rotation=False, mnist_corruptions=False, n_mcd_passes=1, corrupted_cifar_dir=corrupted_cifar_dir,
    #           eval_every_n_epochs=5)
    #
    # run_torch(200, 200, dropout_inference=0.0, dropout_spn=0.1,
    #           training_dataset='mnist', lmbda=0.2, eval_single_digit=False, toy_setting=False,
    #           eval_rotation=False, mnist_corruptions=False, n_mcd_passes=1, corrupted_cifar_dir=corrupted_cifar_dir,
    #           eval_every_n_epochs=5)
    #
    # run_torch(200, 200, dropout_inference=0.0, dropout_spn=0.1,
    #           training_dataset='mnist', lmbda=0.1, eval_single_digit=False, toy_setting=False,
    #           eval_rotation=False, mnist_corruptions=False, n_mcd_passes=1, corrupted_cifar_dir=corrupted_cifar_dir,
    #           eval_every_n_epochs=5)
    #
    # run_torch(200, 200, dropout_inference=0.0, dropout_spn=0.1,
    #           training_dataset='mnist', lmbda=0.0, eval_single_digit=False, toy_setting=False,
    #           eval_rotation=False, mnist_corruptions=False, n_mcd_passes=1, corrupted_cifar_dir=corrupted_cifar_dir,
    #           eval_every_n_epochs=5)
    #
    # run_torch(200, 200, dropout_inference=0.0, dropout_spn=0.1,
    #           training_dataset='fmnist', lmbda=1.0, eval_single_digit=False, toy_setting=False,
    #           eval_rotation=False, mnist_corruptions=False, n_mcd_passes=1, corrupted_cifar_dir=corrupted_cifar_dir,
    #           eval_every_n_epochs=5)
    #
    # run_torch(200, 200, dropout_inference=0.0, dropout_spn=0.1,
    #           training_dataset='fmnist', lmbda=0.8, eval_single_digit=False, toy_setting=False,
    #           eval_rotation=False, mnist_corruptions=False, n_mcd_passes=1, corrupted_cifar_dir=corrupted_cifar_dir,
    #           eval_every_n_epochs=5)
    #
    # run_torch(200, 200, dropout_inference=0.0, dropout_spn=0.1,
    #           training_dataset='fmnist', lmbda=0.6, eval_single_digit=False, toy_setting=False,
    #           eval_rotation=False, mnist_corruptions=False, n_mcd_passes=1, corrupted_cifar_dir=corrupted_cifar_dir,
    #           eval_every_n_epochs=5)
    #
    # run_torch(200, 200, dropout_inference=0.0, dropout_spn=0.1,
    #           training_dataset='fmnist', lmbda=0.4, eval_single_digit=False, toy_setting=False,
    #           eval_rotation=False, mnist_corruptions=False, n_mcd_passes=1, corrupted_cifar_dir=corrupted_cifar_dir,
    #           eval_every_n_epochs=5)
    #
    # run_torch(200, 200, dropout_inference=0.0, dropout_spn=0.1,
    #           training_dataset='fmnist', lmbda=0.2, eval_single_digit=False, toy_setting=False,
    #           eval_rotation=False, mnist_corruptions=False, n_mcd_passes=1, corrupted_cifar_dir=corrupted_cifar_dir,
    #           eval_every_n_epochs=5)
    #
    # run_torch(200, 200, dropout_inference=0.0, dropout_spn=0.1,
    #           training_dataset='fmnist', lmbda=0.1, eval_single_digit=False, toy_setting=False,
    #           eval_rotation=False, mnist_corruptions=False, n_mcd_passes=1, corrupted_cifar_dir=corrupted_cifar_dir,
    #           eval_every_n_epochs=5)
    #
    # run_torch(200, 200, dropout_inference=0.0, dropout_spn=0.1,
    #           training_dataset='fmnist', lmbda=0.0, eval_single_digit=False, toy_setting=False,
    #           eval_rotation=False, mnist_corruptions=False, n_mcd_passes=1, corrupted_cifar_dir=corrupted_cifar_dir,
    #           eval_every_n_epochs=5)

    # run_torch(200, 200, dropout_inference=0.2, dropout_spn=0.2,
    #           training_dataset='fmnist', lmbda=1.0, eval_single_digit=False, toy_setting=False,
    #           eval_rotation=False, mnist_corruptions=False, n_mcd_passes=100, corrupted_cifar_dir=corrupted_cifar_dir,
    #           eval_every_n_epochs=5)
    # run_torch(200, 200, dropout_inference=0.2, dropout_spn=0.2,
    #           training_dataset='fmnist', lmbda=1.0, eval_single_digit=False, toy_setting=False,
    #           eval_rotation=False, mnist_corruptions=False, n_mcd_passes=10, corrupted_cifar_dir=corrupted_cifar_dir,
    #           eval_every_n_epochs=5)
    # run_torch(3, 200, dropout_inference=0.2, dropout_spn=0.2,
    #           training_dataset='svhn', lmbda=1.0, eval_single_digit=False, toy_setting=True,
    #           eval_rotation=False, mnist_corruptions=False, n_mcd_passes=20, corrupted_cifar_dir=corrupted_cifar_dir)
    # run_torch(3, 200, dropout_inference=0.2, dropout_spn=0.2,
    #           training_dataset='cifar', lmbda=1.0, eval_single_digit=False, toy_setting=True,
    #           eval_rotation=False, mnist_corruptions=False, n_mcd_passes=10, corrupted_cifar_dir=corrupted_cifar_dir)

    # load_torch(model_dir='results/2022-05-15_05-20-35/model/', train_on_fmnist=False, dropout_inference=0.2) # lambda 0. mnist
    # run_torch(200, 200, dropout_inference=0.2, dropout_spn=0.2, class_label=1, train_on_fmnist=False, lmbda=1.0, eval_single_digit=False, toy_setting=False, eval_rotation=False)



# TODO
# - add logging instead of prints
# - double check batch index stuff to do not skip last samples < batch size
# - another interesting experiment with rotating mnist would be showing a confusion matrix over the rotations
# - MCD over SPN with one forward pass (pass a vector of values for each sample, or add dim on sum node weight matrices)
# - formal description on what MCD is doing on a SPN (and is Bernoulli the only good way for dropouts? are there alternative distributions?)
