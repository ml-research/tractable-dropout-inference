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

import matplotlib.pyplot as plt
import matplotlib

from torch.utils.data import Dataset, ConcatDataset
from PIL import Image
import datetime

import pdb
from scipy.stats import entropy

import spn.experiments.RandomSPNs_layerwise.mnist_c.corruptions as corruptions

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
        # x = Image.fromarray(x.numpy(), mode='L')
        x = Image.fromarray(x.numpy(), mode=None)

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
        shuffle=True,
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


def get_data_flatten_shape(data_loader):
    if isinstance(data_loader.dataset, ConcatDataset):
        return (data_loader.dataset.cummulative_sizes[-1],
                torch.prod(torch.tensor(data_loader.dataset.datasets[0].data.shape[1:])).int().item())
    return (data_loader.dataset.data.shape[0],
            torch.prod(torch.tensor(data_loader.dataset.data.shape[1:])).int().item())


def make_spn(S, I, R, D, dropout, device, F=28 ** 2) -> RatSpn:
    """Construct the RatSpn"""

    # Setup RatSpnConfig
    config = RatSpnConfig()
    config.F = F
    config.R = R
    config.D = D
    config.I = I
    config.S = S
    config.C = 10
    config.dropout = dropout
    config.leaf_base_class = RatNormal
    config.leaf_base_kwargs = {}

    # Construct RatSpn from config
    model = RatSpn(config)

    model = model.to(device)
    model.train()

    print("Using device:", device)
    print("Dropout SPN: ", dropout)
    return model


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


def load_torch(model_dir=None, training_dataset=None, dropout_inference=None, batch_size=512):
    from torch import optim
    from torch import nn

    dev = sys.argv[1]
    device = torch.device("cuda:0")
    use_cuda = True
    torch.cuda.benchmark = True

    d = model_dir + "likelihoods/"
    ensure_dir(d)
    train_loader, test_loader = get_data_loaders(use_cuda, batch_size=batch_size, device=device, dataset=training_dataset)
    n_features = get_data_flatten_shape(train_loader)[1]
    model = make_spn(S=20, I=20, D=5, R=5, device=dev, dropout=dropout_inference, F=n_features)
    model.load_state_dict(torch.load(model_dir + 'model.pt'))
    model.eval()

    train_lls, class_probs_train, train_lls_sup, train_lls_unsup = evaluate_model(model, device, train_loader, "Train", output_dir=d)
    test_lls, class_probs_test, test_lls_sup, test_lls_unsup = evaluate_model(model, device, test_loader, "Test", output_dir=d)
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

    train_lls_dropout, class_probs_train_dropout, train_lls_sup_drop, train_lls_unsup_drop = evaluate_model_dropout(model, device, train_loader, "Train DROP", dropout_inference=dropout_inference, output_dir=d)
    test_lls_dropout, class_probs_test_dropout, test_lls_sup_drop, test_lls_unsup_drop = evaluate_model_dropout(model, device, test_loader, "Test DROP", dropout_inference=dropout_inference, output_dir=d)

    other_train_loader, other_test_loader = get_data_loaders(use_cuda=use_cuda, device=device, batch_size=batch_size, dataset=get_other_dataset_name(training_dataset))
    other_train_lls_dropout, other_class_probs_train_dropout, other_train_lls_sup_drop, other_train_lls_unsup_drop = evaluate_model_dropout(model, device, other_train_loader, "Other Train DROP", dropout_inference=dropout_inference, output_dir=d)
    other_test_lls_dropout, other_class_probs_test_dropout, other_test_lls_sup_drop, other_test_lls_unsup_drop = evaluate_model_dropout(model, device, other_test_loader, "Other Test DROP", dropout_inference=dropout_inference, output_dir=d)

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
              n_mcd_passes=100):
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

    if toy_setting:
        rat_S, rat_I, rat_D, rat_R = 2, 4, 2, 1
    elif training_dataset == 'cifar':
        rat_S, rat_I, rat_D, rat_R = 20, 20, 5, 10
    else:
        rat_S, rat_I, rat_D, rat_R = 20, 20, 5, 5

    model = make_spn(S=rat_S, I=rat_I, D=rat_D, R=rat_R, device=dev, dropout=dropout_spn, F=n_features)
    model.train()
    print(model)
    n_rat_params = count_params(model)
    print("Number of pytorch parameters: ", n_rat_params)

    # Define optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    log_interval = 100

    training_string = ""

    d_samples = d + "samples/"
    d_results = d + "results/"
    d_model = d + "model/"
    ensure_dir(d_samples)
    ensure_dir(d_results)
    ensure_dir(d_model)

    with open(d + 'trainig_details.out', 'a') as writer:
        writer.write("Dataset: {}, N features {}".format(training_dataset, n_features))
        writer.write("n epochs: {}, batch size: {}".format(n_epochs, batch_size))
        writer.write("MC dropout p {}, dropout (learning) {}".format(dropout_inference, dropout_spn))
        writer.write("RAT lambda {}".format(lmbda))
        writer.write("RAT hyperparameters S {} I {} D {} R {}".format(S, I, D, R))
        writer.write("RAT n of model params: {}".format(n_rat_params))
        writer.write("N MCD passes: {}".format(n_mcd_passes))

    for epoch in range(n_epochs):
        t_start = time.time()
        running_loss = 0.0
        running_loss_ce = 0.0
        running_loss_nll = 0.0
        for batch_index, (data, target) in enumerate(train_loader):
            # Send data to correct device
            data, target = data.to(device), target.to(device)
            data = data.view(data.shape[0], -1)

            # Reset gradients
            optimizer.zero_grad()

            # Inference
            output = model(data)

            # Compute loss
            loss_ce = loss_fn(output, target)
            loss_nll = -output.sum() / (data.shape[0] * n_features)

            loss = (1 - lmbda) * loss_nll + lmbda * loss_ce

            # Backprop
            loss.backward()
            optimizer.step()

            # Log stuff
            running_loss += loss.item()
            running_loss_ce += loss_ce.item()
            running_loss_nll += loss_nll.item()
            if batch_index % log_interval == (log_interval - 1):
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
            if not (training_dataset == 'cifar' or training_dataset == 'svhn'):
                set_seed(0)
                samples = model.sample(class_index=list(range(10)) * 5)
                save_samples(samples, iteration=epoch)

        t_delta = time_delta_now(t_start)
        print("Train Epoch {} took {}".format(epoch, t_delta))
        if epoch % 5 == 4:
            print("Evaluating model...")
            lls_train, class_probs_train, _, _ = evaluate_model(model, device, train_loader, "Train", output_dir=d)
            test_lls, class_probs_test, _, _ = evaluate_model(model, device, test_loader, "Test", output_dir=d)
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
    }, d + 'model/checkpoint.tar')

    train_lls, class_probs_train, train_lls_sup, train_lls_unsup = evaluate_model(model, device, train_loader, "Train", output_dir=d)
    test_lls, class_probs_test, test_lls_sup, test_lls_unsup = evaluate_model(model, device, test_loader, "Test", output_dir=d)

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
                                                                                                 dropout_inference=dropout_inference, n_dropout_iters=n_mcd_passes, output_dir=d, corruption=corruption, corruption_level=cl)
                np.save(d_results + 'dropout_class_probs_c_{}_l{}'.format(corruption, cl), results_dict['c_{}_l{}'.format(corruption, cl)][0].cpu().detach().numpy())
                np.save(d_results + 'class_probs_c_{}_l{}'.format(corruption, cl), results_dict['c_{}_l{}'.format(corruption, cl)][1].cpu().detach().numpy())


        sys.exit("done.")

    if training_dataset == 'mnist' and mnist_corruptions:
        import spn.experiments.RandomSPNs_layerwise.mnist_c.corruptions as corruptions
        severity = [1, 2, 3, 4, 5]
        corruption_method = [corruptions.brightness, corruptions.shot_noise, corruptions.impulse_noise, corruptions.glass_blur, corruptions.motion_blur, corruptions.shear, corruptions.scale,
                             corruptions.rotate, corruptions.translate, corruptions.fog, corruptions.spatter]
        corruption_methods_no_severity = [corruptions.stripe, corruptions.dotted_line, corruptions.zigzag, corruptions.canny_edges]

        results_dict = {}

        for cm in corruption_method.extend(corruption_methods_no_severity):
            if cm in corruption_methods_no_severity: severity = [None]
            for sl in severity:
                print("Corruption {}, Severity {}".format(cm.__name__, sl))
                results_dict['c_{}_l{}'.format(cm.__name__, sl)] = evaluate_model_corrupted_digits(model, device, test_loader, "Test DROP corrupted", dropout_inference=dropout_inference, output_dir=d, corruption=cm, severity=sl)
                np.save(d_results + 'dropout_class_probs_c_{}_l{}'.format(cm.__name__, sl), results_dict['c_{}_l{}'.format(cm.__name__, sl)][0].cpu().detach().numpy())
                np.save(d_results + 'class_probs_c_{}_l{}'.format(cm.__name__, sl), results_dict['c_{}_l{}'.format(cm.__name__, sl)][1].cpu().detach().numpy())
        sys.exit()



        # compute max
        drop_max_class_probs_c1 = torch.max(torch.mean(drop_class_probs_c1, dim=2), dim=1)[0].detach().cpu().numpy()
        drop_max_class_probs_c2 = torch.max(torch.mean(drop_class_probs_c2, dim=2), dim=1)[0].detach().cpu().numpy()
        drop_max_class_probs_c3 = torch.max(torch.mean(drop_class_probs_c3, dim=2), dim=1)[0].detach().cpu().numpy()
        drop_max_class_probs_c4 = torch.max(torch.mean(drop_class_probs_c4, dim=2), dim=1)[0].detach().cpu().numpy()
        drop_max_class_probs_c5 = torch.max(torch.mean(drop_class_probs_c5, dim=2), dim=1)[0].detach().cpu().numpy()
        drop_max_class_probs_c6 = torch.max(torch.mean(drop_class_probs_c6, dim=2), dim=1)[0].detach().cpu().numpy()
        drop_max_class_probs_c7 = torch.max(torch.mean(drop_class_probs_c7, dim=2), dim=1)[0].detach().cpu().numpy()
        drop_max_class_probs_c8 = torch.max(torch.mean(drop_class_probs_c8, dim=2), dim=1)[0].detach().cpu().numpy()
        drop_max_class_probs_c9 = torch.max(torch.mean(drop_class_probs_c9, dim=2), dim=1)[0].detach().cpu().numpy()
        drop_max_class_probs_c10 = torch.max(torch.mean(drop_class_probs_c10, dim=2), dim=1)[0].detach().cpu().numpy()
        drop_max_class_probs_c11 = torch.max(torch.mean(drop_class_probs_c11, dim=2), dim=1)[0].detach().cpu().numpy()
        drop_max_class_probs_c12 = torch.max(torch.mean(drop_class_probs_c12, dim=2), dim=1)[0].detach().cpu().numpy()
        drop_max_class_probs_c13 = torch.max(torch.mean(drop_class_probs_c13, dim=2), dim=1)[0].detach().cpu().numpy()
        drop_max_class_probs_c14 = torch.max(torch.mean(drop_class_probs_c14, dim=2), dim=1)[0].detach().cpu().numpy()
        drop_max_class_probs_c15 = torch.max(torch.mean(drop_class_probs_c15, dim=2), dim=1)[0].detach().cpu().numpy()

        max_class_probs_c1 = torch.max(class_probs_c1, dim=1)[0].detach().cpu().numpy()
        max_class_probs_c2 = torch.max(class_probs_c2, dim=1)[0].detach().cpu().numpy()
        max_class_probs_c3 = torch.max(class_probs_c3, dim=1)[0].detach().cpu().numpy()
        max_class_probs_c4 = torch.max(class_probs_c4, dim=1)[0].detach().cpu().numpy()
        max_class_probs_c5 = torch.max(class_probs_c5, dim=1)[0].detach().cpu().numpy()
        max_class_probs_c6 = torch.max(class_probs_c6, dim=1)[0].detach().cpu().numpy()
        max_class_probs_c7 = torch.max(class_probs_c7, dim=1)[0].detach().cpu().numpy()
        max_class_probs_c8 = torch.max(class_probs_c8, dim=1)[0].detach().cpu().numpy()
        max_class_probs_c9 = torch.max(class_probs_c9, dim=1)[0].detach().cpu().numpy()
        max_class_probs_c10 = torch.max(class_probs_c10, dim=1)[0].detach().cpu().numpy()
        max_class_probs_c11 = torch.max(class_probs_c11, dim=1)[0].detach().cpu().numpy()
        max_class_probs_c12 = torch.max(class_probs_c12, dim=1)[0].detach().cpu().numpy()
        max_class_probs_c13 = torch.max(class_probs_c13, dim=1)[0].detach().cpu().numpy()
        max_class_probs_c14 = torch.max(class_probs_c14, dim=1)[0].detach().cpu().numpy()
        max_class_probs_c15 = torch.max(class_probs_c15, dim=1)[0].detach().cpu().numpy()

        # prepare boxplots data
        boxplot_data = np.column_stack((max_class_probs_c1, max_class_probs_c2, max_class_probs_c3, max_class_probs_c4, max_class_probs_c5, max_class_probs_c6, max_class_probs_c7, max_class_probs_c8, max_class_probs_c9, max_class_probs_c10, max_class_probs_c11, max_class_probs_c12, max_class_probs_c13, max_class_probs_c14, max_class_probs_c15))
        drop_boxplot_data = np.column_stack((drop_max_class_probs_c1, drop_max_class_probs_c2, drop_max_class_probs_c3, drop_max_class_probs_c4, drop_max_class_probs_c5, drop_max_class_probs_c6, drop_max_class_probs_c7, drop_max_class_probs_c8, drop_max_class_probs_c9, drop_max_class_probs_c10, drop_max_class_probs_c11, drop_max_class_probs_c12, drop_max_class_probs_c13, drop_max_class_probs_c14, drop_max_class_probs_c15))

        # compute entropy
        drop_entropy_class_probs_c1 = entropy(torch.mean(drop_class_probs_c1, dim=2).detach().cpu().numpy(), axis=1)
        drop_entropy_class_probs_c2 = entropy(torch.mean(drop_class_probs_c2, dim=2).detach().cpu().numpy(), axis=1)
        drop_entropy_class_probs_c3 = entropy(torch.mean(drop_class_probs_c3, dim=2).detach().cpu().numpy(), axis=1)
        drop_entropy_class_probs_c4 = entropy(torch.mean(drop_class_probs_c4, dim=2).detach().cpu().numpy(), axis=1)
        drop_entropy_class_probs_c5 = entropy(torch.mean(drop_class_probs_c5, dim=2).detach().cpu().numpy(), axis=1)
        drop_entropy_class_probs_c6 = entropy(torch.mean(drop_class_probs_c6, dim=2).detach().cpu().numpy(), axis=1)
        drop_entropy_class_probs_c7 = entropy(torch.mean(drop_class_probs_c7, dim=2).detach().cpu().numpy(), axis=1)
        drop_entropy_class_probs_c8 = entropy(torch.mean(drop_class_probs_c8, dim=2).detach().cpu().numpy(), axis=1)
        drop_entropy_class_probs_c9 = entropy(torch.mean(drop_class_probs_c9, dim=2).detach().cpu().numpy(), axis=1)
        drop_entropy_class_probs_c10 = entropy(torch.mean(drop_class_probs_c10, dim=2).detach().cpu().numpy(), axis=1)
        drop_entropy_class_probs_c11 = entropy(torch.mean(drop_class_probs_c11, dim=2).detach().cpu().numpy(), axis=1)
        drop_entropy_class_probs_c12 = entropy(torch.mean(drop_class_probs_c12, dim=2).detach().cpu().numpy(), axis=1)
        drop_entropy_class_probs_c13 = entropy(torch.mean(drop_class_probs_c13, dim=2).detach().cpu().numpy(), axis=1)
        drop_entropy_class_probs_c14 = entropy(torch.mean(drop_class_probs_c14, dim=2).detach().cpu().numpy(), axis=1)
        drop_entropy_class_probs_c15 = entropy(torch.mean(drop_class_probs_c15, dim=2).detach().cpu().numpy(), axis=1)

        entropy_class_probs_c1 = entropy(class_probs_c1.detach().cpu().numpy(), axis=1)
        entropy_class_probs_c2 = entropy(class_probs_c2.detach().cpu().numpy(), axis=1)
        entropy_class_probs_c3 = entropy(class_probs_c3.detach().cpu().numpy(), axis=1)
        entropy_class_probs_c4 = entropy(class_probs_c4.detach().cpu().numpy(), axis=1)
        entropy_class_probs_c5 = entropy(class_probs_c5.detach().cpu().numpy(), axis=1)
        entropy_class_probs_c6 = entropy(class_probs_c6.detach().cpu().numpy(), axis=1)
        entropy_class_probs_c7 = entropy(class_probs_c7.detach().cpu().numpy(), axis=1)
        entropy_class_probs_c8 = entropy(class_probs_c8.detach().cpu().numpy(), axis=1)
        entropy_class_probs_c9 = entropy(class_probs_c9.detach().cpu().numpy(), axis=1)
        entropy_class_probs_c10 = entropy(class_probs_c10.detach().cpu().numpy(), axis=1)
        entropy_class_probs_c11 = entropy(class_probs_c11.detach().cpu().numpy(), axis=1)
        entropy_class_probs_c12 = entropy(class_probs_c12.detach().cpu().numpy(), axis=1)
        entropy_class_probs_c13 = entropy(class_probs_c13.detach().cpu().numpy(), axis=1)
        entropy_class_probs_c14 = entropy(class_probs_c14.detach().cpu().numpy(), axis=1)
        entropy_class_probs_c15 = entropy(class_probs_c15.detach().cpu().numpy(), axis=1)

        # prepare boxplots data
        entropy_boxplot_data = np.column_stack((entropy_class_probs_c1, entropy_class_probs_c2, entropy_class_probs_c3, entropy_class_probs_c4, entropy_class_probs_c5, entropy_class_probs_c6, entropy_class_probs_c7, entropy_class_probs_c8, entropy_class_probs_c9, entropy_class_probs_c10, entropy_class_probs_c11, entropy_class_probs_c12, entropy_class_probs_c13, entropy_class_probs_c14, entropy_class_probs_c15))
        drop_entropy_boxplot_data = np.column_stack((drop_entropy_class_probs_c1, drop_entropy_class_probs_c2, drop_entropy_class_probs_c3, drop_entropy_class_probs_c4, drop_entropy_class_probs_c5, drop_entropy_class_probs_c6, drop_entropy_class_probs_c7, drop_entropy_class_probs_c8, drop_entropy_class_probs_c9, drop_entropy_class_probs_c10, drop_entropy_class_probs_c11, drop_entropy_class_probs_c12, drop_entropy_class_probs_c13, drop_entropy_class_probs_c14, drop_entropy_class_probs_c15))

        # plot bloxplot
        plot_boxplot_corrupted_digits(boxplot_data, filename='boxplot_corrupted_digits', title='Corrupted MNIST', path=d_samples)
        plot_boxplot_corrupted_digits(drop_boxplot_data, filename='boxplot_corrupted_digits_MCD', title='Corrupted MNIST - MCD', path=d_samples)
        plot_boxplot_corrupted_digits(entropy_boxplot_data, filename='entropy_boxplot_corrupted_digits', title='Corrupted MNIST', path=d_samples, ylimits=[-0.1, 3.0], ylabel='Classification Entropy')
        plot_boxplot_corrupted_digits(drop_entropy_boxplot_data, filename='entropy_boxplot_corrupted_digits_MCD', title='Corrupted MNIST - MCD', path=d_samples, ylimits=[-0.1, 3.0], ylabel='Classification Entropy')

        sys.exit("Done!")

    if training_dataset == 'mnist' and eval_rotation:
        drop_class_probs_180, class_probs_180 = evaluate_model_rotated_digits(model, device, test_loader, "Test DROP rotated one 180 degrees", dropout_inference=dropout_inference, output_dir=d, degrees=180) #TODO add batch_size and drop iters
        drop_class_probs_150, class_probs_150 = evaluate_model_rotated_digits(model, device, test_loader, "Test DROP rotated one 150 degrees", dropout_inference=dropout_inference, output_dir=d, degrees=150) #TODO add batch_size and drop iters
        drop_class_probs_120, class_probs_120 = evaluate_model_rotated_digits(model, device, test_loader, "Test DROP rotated one 120 degrees", dropout_inference=dropout_inference, output_dir=d, degrees=120) #TODO add batch_size and drop iters
        drop_class_probs_90, class_probs_90 = evaluate_model_rotated_digits(model, device, test_loader, "Test DROP rotated one 90 degrees", dropout_inference=dropout_inference, output_dir=d, degrees=90) #TODO add batch_size and drop iters
        drop_class_probs_60, class_probs_60 = evaluate_model_rotated_digits(model, device, test_loader, "Test DROP rotated one 60 degrees", dropout_inference=dropout_inference, output_dir=d, degrees=60) #TODO add batch_size and drop iters
        drop_class_probs_30, class_probs_30 = evaluate_model_rotated_digits(model, device, test_loader, "Test DROP rotated one 30 degrees", dropout_inference=dropout_inference, output_dir=d, degrees=30) #TODO add batch_size and drop iters
        drop_class_probs_0, class_probs_0 = evaluate_model_rotated_digits(model, device, test_loader, "Test DROP rotated one 0 degrees", dropout_inference=dropout_inference, output_dir=d, degrees=0) #TODO add batch_size and drop iters
        drop_class_probs_330, class_probs_330 = evaluate_model_rotated_digits(model, device, test_loader, "Test DROP rotated one 330 degrees", dropout_inference=dropout_inference, output_dir=d, degrees=330) #TODO add batch_size and drop iters
        drop_class_probs_300, class_probs_300 = evaluate_model_rotated_digits(model, device, test_loader, "Test DROP rotated one 300 degrees", dropout_inference=dropout_inference, output_dir=d, degrees=300) #TODO add batch_size and drop iters
        drop_class_probs_270, class_probs_270 = evaluate_model_rotated_digits(model, device, test_loader, "Test DROP rotated one 270 degrees", dropout_inference=dropout_inference, output_dir=d, degrees=270) #TODO add batch_size and drop iters
        drop_class_probs_240, class_probs_240 = evaluate_model_rotated_digits(model, device, test_loader, "Test DROP rotated one 240 degrees", dropout_inference=dropout_inference, output_dir=d, degrees=240) #TODO add batch_size and drop iters
        drop_class_probs_210, class_probs_210 = evaluate_model_rotated_digits(model, device, test_loader, "Test DROP rotated one 210 degrees", dropout_inference=dropout_inference, output_dir=d, degrees=210) #TODO add batch_size and drop iters
        #

        np.save(d_results + 'dropout_class_probs_180', drop_class_probs_180.cpu().detach().numpy())
        np.save(d_results + 'dropout_class_probs_150', drop_class_probs_150.cpu().detach().numpy())
        np.save(d_results + 'dropout_class_probs_120', drop_class_probs_120.cpu().detach().numpy())
        np.save(d_results + 'dropout_class_probs_90', drop_class_probs_90.cpu().detach().numpy())
        np.save(d_results + 'dropout_class_probs_60', drop_class_probs_60.cpu().detach().numpy())
        np.save(d_results + 'dropout_class_probs_30', drop_class_probs_30.cpu().detach().numpy())
        np.save(d_results + 'dropout_class_probs_0', drop_class_probs_0.cpu().detach().numpy())
        np.save(d_results + 'dropout_class_probs_330', drop_class_probs_330.cpu().detach().numpy())
        np.save(d_results + 'dropout_class_probs_300', drop_class_probs_300.cpu().detach().numpy())
        np.save(d_results + 'dropout_class_probs_270', drop_class_probs_270.cpu().detach().numpy())
        np.save(d_results + 'dropout_class_probs_240', drop_class_probs_240.cpu().detach().numpy())
        np.save(d_results + 'dropout_class_probs_210', drop_class_probs_210.cpu().detach().numpy())

        np.save(d_results + 'class_probs_180', class_probs_180.cpu().detach().numpy())
        np.save(d_results + 'class_probs_150', class_probs_150.cpu().detach().numpy())
        np.save(d_results + 'class_probs_120', class_probs_120.cpu().detach().numpy())
        np.save(d_results + 'class_probs_90', class_probs_90.cpu().detach().numpy())
        np.save(d_results + 'class_probs_60', class_probs_60.cpu().detach().numpy())
        np.save(d_results + 'class_probs_30', class_probs_30.cpu().detach().numpy())
        np.save(d_results + 'class_probs_0', class_probs_0.cpu().detach().numpy())
        np.save(d_results + 'class_probs_330', class_probs_330.cpu().detach().numpy())
        np.save(d_results + 'class_probs_300', class_probs_300.cpu().detach().numpy())
        np.save(d_results + 'class_probs_270', class_probs_270.cpu().detach().numpy())
        np.save(d_results + 'class_probs_240', class_probs_240.cpu().detach().numpy())
        np.save(d_results + 'class_probs_210', class_probs_210.cpu().detach().numpy())

        drop_max_class_probs_180 = torch.max(torch.mean(drop_class_probs_180, dim=2), dim=1)[0].detach().cpu().numpy()
        drop_max_class_probs_150 = torch.max(torch.mean(drop_class_probs_150, dim=2), dim=1)[0].detach().cpu().numpy()
        drop_max_class_probs_120 = torch.max(torch.mean(drop_class_probs_120, dim=2), dim=1)[0].detach().cpu().numpy()
        drop_max_class_probs_90 = torch.max(torch.mean(drop_class_probs_90, dim=2), dim=1)[0].detach().cpu().numpy()
        drop_max_class_probs_60 = torch.max(torch.mean(drop_class_probs_60, dim=2), dim=1)[0].detach().cpu().numpy()
        drop_max_class_probs_30 = torch.max(torch.mean(drop_class_probs_30, dim=2), dim=1)[0].detach().cpu().numpy()
        drop_max_class_probs_0 = torch.max(torch.mean(drop_class_probs_0, dim=2), dim=1)[0].detach().cpu().numpy()
        drop_max_class_probs_330 = torch.max(torch.mean(drop_class_probs_330, dim=2), dim=1)[0].detach().cpu().numpy()
        drop_max_class_probs_300 = torch.max(torch.mean(drop_class_probs_300, dim=2), dim=1)[0].detach().cpu().numpy()
        drop_max_class_probs_270 = torch.max(torch.mean(drop_class_probs_270, dim=2), dim=1)[0].detach().cpu().numpy()
        drop_max_class_probs_240 = torch.max(torch.mean(drop_class_probs_240, dim=2), dim=1)[0].detach().cpu().numpy()
        drop_max_class_probs_210 = torch.max(torch.mean(drop_class_probs_210, dim=2), dim=1)[0].detach().cpu().numpy()

        max_class_probs_180 = torch.max(class_probs_180, dim=1)[0].detach().cpu().numpy()
        max_class_probs_150 = torch.max(class_probs_150, dim=1)[0].detach().cpu().numpy()
        max_class_probs_120 = torch.max(class_probs_120, dim=1)[0].detach().cpu().numpy()
        max_class_probs_90 = torch.max(class_probs_90, dim=1)[0].detach().cpu().numpy()
        max_class_probs_60 = torch.max(class_probs_60, dim=1)[0].detach().cpu().numpy()
        max_class_probs_30 = torch.max(class_probs_30, dim=1)[0].detach().cpu().numpy()
        max_class_probs_0 = torch.max(class_probs_0, dim=1)[0].detach().cpu().numpy()
        max_class_probs_330 = torch.max(class_probs_330, dim=1)[0].detach().cpu().numpy()
        max_class_probs_300 = torch.max(class_probs_300, dim=1)[0].detach().cpu().numpy()
        max_class_probs_270 = torch.max(class_probs_270, dim=1)[0].detach().cpu().numpy()
        max_class_probs_240 = torch.max(class_probs_240, dim=1)[0].detach().cpu().numpy()
        max_class_probs_210 = torch.max(class_probs_210, dim=1)[0].detach().cpu().numpy()

        boxplot_data = np.column_stack((max_class_probs_180, max_class_probs_150, max_class_probs_120, max_class_probs_90, max_class_probs_60, max_class_probs_30, max_class_probs_0, max_class_probs_330, max_class_probs_300, max_class_probs_270, max_class_probs_240, max_class_probs_210))
        drop_boxplot_data = np.column_stack((drop_max_class_probs_180, drop_max_class_probs_150, drop_max_class_probs_120, drop_max_class_probs_90, drop_max_class_probs_60, drop_max_class_probs_30, drop_max_class_probs_0, drop_max_class_probs_330, drop_max_class_probs_300, drop_max_class_probs_270, drop_max_class_probs_240, drop_max_class_probs_210))

        # plot classification entropy
        drop_entropy_class_probs_180 = entropy(torch.mean(drop_class_probs_180, dim=2).detach().cpu().numpy(), axis=1)
        drop_entropy_class_probs_150 = entropy(torch.mean(drop_class_probs_150, dim=2).detach().cpu().numpy(), axis=1)
        drop_entropy_class_probs_120 = entropy(torch.mean(drop_class_probs_120, dim=2).detach().cpu().numpy(), axis=1)
        drop_entropy_class_probs_90 = entropy(torch.mean(drop_class_probs_90, dim=2).detach().cpu().numpy(), axis=1)
        drop_entropy_class_probs_60 = entropy(torch.mean(drop_class_probs_60, dim=2).detach().cpu().numpy(), axis=1)
        drop_entropy_class_probs_30 = entropy(torch.mean(drop_class_probs_30, dim=2).detach().cpu().numpy(), axis=1)
        drop_entropy_class_probs_0 = entropy(torch.mean(drop_class_probs_0, dim=2).detach().cpu().numpy(), axis=1)
        drop_entropy_class_probs_330 = entropy(torch.mean(drop_class_probs_330, dim=2).detach().cpu().numpy(), axis=1)
        drop_entropy_class_probs_300 = entropy(torch.mean(drop_class_probs_300, dim=2).detach().cpu().numpy(), axis=1)
        drop_entropy_class_probs_270 = entropy(torch.mean(drop_class_probs_270, dim=2).detach().cpu().numpy(), axis=1)
        drop_entropy_class_probs_240 = entropy(torch.mean(drop_class_probs_240, dim=2).detach().cpu().numpy(), axis=1)
        drop_entropy_class_probs_210 = entropy(torch.mean(drop_class_probs_210, dim=2).detach().cpu().numpy(), axis=1)

        entropy_class_probs_180 = entropy(class_probs_180.detach().cpu().numpy(), axis=1)
        entropy_class_probs_150 = entropy(class_probs_150.detach().cpu().numpy(), axis=1)
        entropy_class_probs_120 = entropy(class_probs_120.detach().cpu().numpy(), axis=1)
        entropy_class_probs_90 = entropy(class_probs_90.detach().cpu().numpy(), axis=1)
        entropy_class_probs_60 = entropy(class_probs_60.detach().cpu().numpy(), axis=1)
        entropy_class_probs_30 = entropy(class_probs_30.detach().cpu().numpy(), axis=1)
        entropy_class_probs_0 = entropy(class_probs_0.detach().cpu().numpy(), axis=1)
        entropy_class_probs_330 = entropy(class_probs_330.detach().cpu().numpy(), axis=1)
        entropy_class_probs_300 = entropy(class_probs_300.detach().cpu().numpy(), axis=1)
        entropy_class_probs_270 = entropy(class_probs_270.detach().cpu().numpy(), axis=1)
        entropy_class_probs_240 = entropy(class_probs_240.detach().cpu().numpy(), axis=1)
        entropy_class_probs_210 = entropy(class_probs_210.detach().cpu().numpy(), axis=1)

        entropy_boxplot_data = np.column_stack((entropy_class_probs_180, entropy_class_probs_150, entropy_class_probs_120, entropy_class_probs_90, entropy_class_probs_60, entropy_class_probs_30, entropy_class_probs_0, entropy_class_probs_330, entropy_class_probs_300, entropy_class_probs_270, entropy_class_probs_240, entropy_class_probs_210))
        drop_entropy_boxplot_data = np.column_stack((drop_entropy_class_probs_180, drop_entropy_class_probs_150, drop_entropy_class_probs_120, drop_entropy_class_probs_90, drop_entropy_class_probs_60, drop_entropy_class_probs_30, drop_entropy_class_probs_0, drop_entropy_class_probs_330, drop_entropy_class_probs_300, drop_entropy_class_probs_270, drop_entropy_class_probs_240, drop_entropy_class_probs_210))

        plot_boxplot_rotating_digits(entropy_boxplot_data, filename='entropy_boxplot_rotating_digits', title='Rotating MNIST', path=d_samples, ylimits=[-0.1, 3.0], ylabel='Classification Entropy')
        plot_boxplot_rotating_digits(drop_entropy_boxplot_data, filename='entropy_boxplot_rotating_digits_MCD', title='Rotating MNIST - MCD', path=d_samples, ylimits=[-0.1, 3.0], ylabel='Classification Entropy')

    # evaluate_model_rotated_digits(model, device, test_loader, "Test DROP rotated one", dropout_inference=dropout_inference, output_dir=d, degrees=50) #TODO add batch_size and drop iters
    # evaluate_model_rotated_digits(model, device, test_loader, "Test DROP rotated one", dropout_inference=dropout_inference, output_dir=d, degrees=60) #TODO add batch_size and drop iters
    # evaluate_model_rotated_digits(model, device, test_loader, "Test DROP rotated one", dropout_inference=dropout_inference, output_dir=d, degrees=70) #TODO add batch_size and drop iters
    # evaluate_model_rotated_digits(model, device, test_loader, "Test DROP rotated one", dropout_inference=dropout_inference, output_dir=d, degrees=80) #TODO add batch_size and drop iters
    # evaluate_model_rotated_digits(model, device, test_loader, "Test DROP rotated one", dropout_inference=dropout_inference, output_dir=d, degrees=90) #TODO add batch_size and drop iters
    # evaluate_model_rotated_digits(model, device, test_loader, "Test DROP rotated one", dropout_inference=dropout_inference, output_dir=d, degrees=120) #TODO add batch_size and drop iters
    # evaluate_model_rotated_digits(model, device, test_loader, "Test DROP rotated one", dropout_inference=dropout_inference, output_dir=d, degrees=140) #TODO add batch_size and drop iters
    # evaluate_model_rotated_digits(model, device, test_loader, "Test DROP rotated one", dropout_inference=dropout_inference, output_dir=d, degrees=150) #TODO add batch_size and drop iters
    # evaluate_model_rotated_digits(model, device, test_loader, "Test DROP rotated one", dropout_inference=dropout_inference, output_dir=d, degrees=160) #TODO add batch_size and drop iters
    # evaluate_model_rotated_digits(model, device, test_loader, "Test DROP rotated one", dropout_inference=dropout_inference, output_dir=d, degrees=180) #TODO add batch_size and drop iters


        print("Train class entropy: {} Test class entropy: {}".format(entropy(class_probs_train, axis=1).sum(), entropy(class_probs_test, axis=1).sum()))

        plot_boxplot_rotating_digits(boxplot_data, filename='boxplot_rotating_digits', title='Rotating MNIST', path=d_samples)
        plot_boxplot_rotating_digits(drop_boxplot_data, filename='boxplot_rotating_digits_MCD', title='Rotating MNIST - MCD', path=d_samples)


    if training_dataset == 'mnist' and eval_rotation:
        if eval_single_digit:

            for class_idx in range(10):
                print("Evaluate by rotating digits, for each class separately: class {}".format(class_idx))

                drop_class_probs_180, class_probs_180 = evaluate_model_rotated_digits(model, device, test_loader, "Test DROP rotated one 180 degrees", dropout_inference=dropout_inference, output_dir=d, degrees=180, class_label=class_idx) #TODO add batch_size and drop iters
                drop_class_probs_150, class_probs_150 = evaluate_model_rotated_digits(model, device, test_loader, "Test DROP rotated one 150 degrees", dropout_inference=dropout_inference, output_dir=d, degrees=150, class_label=class_idx) #TODO add batch_size and drop iters
                drop_class_probs_120, class_probs_120 = evaluate_model_rotated_digits(model, device, test_loader, "Test DROP rotated one 120 degrees", dropout_inference=dropout_inference, output_dir=d, degrees=120, class_label=class_idx) #TODO add batch_size and drop iters
                drop_class_probs_90, class_probs_90 = evaluate_model_rotated_digits(model, device, test_loader, "Test DROP rotated one 90 degrees", dropout_inference=dropout_inference, output_dir=d, degrees=90, class_label=class_idx) #TODO add batch_size and drop iters
                drop_class_probs_60, class_probs_60 = evaluate_model_rotated_digits(model, device, test_loader, "Test DROP rotated one 60 degrees", dropout_inference=dropout_inference, output_dir=d, degrees=60, class_label=class_idx) #TODO add batch_size and drop iters
                drop_class_probs_30, class_probs_30 = evaluate_model_rotated_digits(model, device, test_loader, "Test DROP rotated one 30 degrees", dropout_inference=dropout_inference, output_dir=d, degrees=30, class_label=class_idx) #TODO add batch_size and drop iters
                drop_class_probs_0, class_probs_0 = evaluate_model_rotated_digits(model, device, test_loader, "Test DROP rotated one 0 degrees", dropout_inference=dropout_inference, output_dir=d, degrees=0, class_label=class_idx) #TODO add batch_size and drop iters
                drop_class_probs_330, class_probs_330 = evaluate_model_rotated_digits(model, device, test_loader, "Test DROP rotated one 330 degrees", dropout_inference=dropout_inference, output_dir=d, degrees=330, class_label=class_idx) #TODO add batch_size and drop iters
                drop_class_probs_300, class_probs_300 = evaluate_model_rotated_digits(model, device, test_loader, "Test DROP rotated one 300 degrees", dropout_inference=dropout_inference, output_dir=d, degrees=300, class_label=class_idx) #TODO add batch_size and drop iters
                drop_class_probs_270, class_probs_270 = evaluate_model_rotated_digits(model, device, test_loader, "Test DROP rotated one 270 degrees", dropout_inference=dropout_inference, output_dir=d, degrees=270, class_label=class_idx) #TODO add batch_size and drop iters
                drop_class_probs_240, class_probs_240 = evaluate_model_rotated_digits(model, device, test_loader, "Test DROP rotated one 240 degrees", dropout_inference=dropout_inference, output_dir=d, degrees=240, class_label=class_idx) #TODO add batch_size and drop iters
                drop_class_probs_210, class_probs_210 = evaluate_model_rotated_digits(model, device, test_loader, "Test DROP rotated one 210 degrees", dropout_inference=dropout_inference, output_dir=d, degrees=210, class_label=class_idx) #TODO add batch_size and drop iters

                np.save(d_results + 'dropout_class_probs_180_' + str(class_idx), drop_class_probs_180.cpu().detach().numpy())
                np.save(d_results + 'dropout_class_probs_150_' + str(class_idx), drop_class_probs_150.cpu().detach().numpy())
                np.save(d_results + 'dropout_class_probs_120_' + str(class_idx), drop_class_probs_120.cpu().detach().numpy())
                np.save(d_results + 'dropout_class_probs_90_' + str(class_idx), drop_class_probs_90.cpu().detach().numpy())
                np.save(d_results + 'dropout_class_probs_60.out_' + str(class_idx), drop_class_probs_60.cpu().detach().numpy())
                np.save(d_results + 'dropout_class_probs_30.out_' + str(class_idx), drop_class_probs_30.cpu().detach().numpy())
                np.save(d_results + 'dropout_class_probs_0.out_' + str(class_idx), drop_class_probs_0.cpu().detach().numpy())
                np.save(d_results + 'dropout_class_probs_330.out_' + str(class_idx), drop_class_probs_330.cpu().detach().numpy())
                np.save(d_results + 'dropout_class_probs_300.out_' + str(class_idx), drop_class_probs_300.cpu().detach().numpy())
                np.save(d_results + 'dropout_class_probs_270.out_' + str(class_idx), drop_class_probs_270.cpu().detach().numpy())
                np.save(d_results + 'dropout_class_probs_240.out_' + str(class_idx), drop_class_probs_240.cpu().detach().numpy())
                np.save(d_results + 'dropout_class_probs_210.out_' + str(class_idx), drop_class_probs_210.cpu().detach().numpy())

                np.save(d_results + 'class_probs_180_' + str(class_idx), class_probs_180.cpu().detach().numpy())
                np.save(d_results + 'class_probs_150_' + str(class_idx), class_probs_150.cpu().detach().numpy())
                np.save(d_results + 'class_probs_120_' + str(class_idx), class_probs_120.cpu().detach().numpy())
                np.save(d_results + 'class_probs_90_' + str(class_idx), class_probs_90.cpu().detach().numpy())
                np.save(d_results + 'class_probs_60_' + str(class_idx), class_probs_60.cpu().detach().numpy())
                np.save(d_results + 'class_probs_30_' + str(class_idx), class_probs_30.cpu().detach().numpy())
                np.save(d_results + 'class_probs_0_' + str(class_idx), class_probs_0.cpu().detach().numpy())
                np.save(d_results + 'class_probs_330_' + str(class_idx), class_probs_330.cpu().detach().numpy())
                np.save(d_results + 'class_probs_300_' + str(class_idx), class_probs_300.cpu().detach().numpy())
                np.save(d_results + 'class_probs_270_' + str(class_idx), class_probs_270.cpu().detach().numpy())
                np.save(d_results + 'class_probs_240_' + str(class_idx), class_probs_240.cpu().detach().numpy())
                np.save(d_results + 'class_probs_210_' + str(class_idx), class_probs_210.cpu().detach().numpy())

                drop_max_class_probs_180 = torch.max(torch.mean(drop_class_probs_180, dim=2), dim=1)[0].detach().cpu().numpy()
                drop_max_class_probs_150 = torch.max(torch.mean(drop_class_probs_150, dim=2), dim=1)[0].detach().cpu().numpy()
                drop_max_class_probs_120 = torch.max(torch.mean(drop_class_probs_120, dim=2), dim=1)[0].detach().cpu().numpy()
                drop_max_class_probs_90 = torch.max(torch.mean(drop_class_probs_90, dim=2), dim=1)[0].detach().cpu().numpy()
                drop_max_class_probs_60 = torch.max(torch.mean(drop_class_probs_60, dim=2), dim=1)[0].detach().cpu().numpy()
                drop_max_class_probs_30 = torch.max(torch.mean(drop_class_probs_30, dim=2), dim=1)[0].detach().cpu().numpy()
                drop_max_class_probs_0 = torch.max(torch.mean(drop_class_probs_0, dim=2), dim=1)[0].detach().cpu().numpy()
                drop_max_class_probs_330 = torch.max(torch.mean(drop_class_probs_330, dim=2), dim=1)[0].detach().cpu().numpy()
                drop_max_class_probs_300 = torch.max(torch.mean(drop_class_probs_300, dim=2), dim=1)[0].detach().cpu().numpy()
                drop_max_class_probs_270 = torch.max(torch.mean(drop_class_probs_270, dim=2), dim=1)[0].detach().cpu().numpy()
                drop_max_class_probs_240 = torch.max(torch.mean(drop_class_probs_240, dim=2), dim=1)[0].detach().cpu().numpy()
                drop_max_class_probs_210 = torch.max(torch.mean(drop_class_probs_210, dim=2), dim=1)[0].detach().cpu().numpy()

                max_class_probs_180 = torch.max(class_probs_180, dim=1)[0].detach().cpu().numpy()
                max_class_probs_150 = torch.max(class_probs_150, dim=1)[0].detach().cpu().numpy()
                max_class_probs_120 = torch.max(class_probs_120, dim=1)[0].detach().cpu().numpy()
                max_class_probs_90 = torch.max(class_probs_90, dim=1)[0].detach().cpu().numpy()
                max_class_probs_60 = torch.max(class_probs_60, dim=1)[0].detach().cpu().numpy()
                max_class_probs_30 = torch.max(class_probs_30, dim=1)[0].detach().cpu().numpy()
                max_class_probs_0 = torch.max(class_probs_0, dim=1)[0].detach().cpu().numpy()
                max_class_probs_330 = torch.max(class_probs_330, dim=1)[0].detach().cpu().numpy()
                max_class_probs_300 = torch.max(class_probs_300, dim=1)[0].detach().cpu().numpy()
                max_class_probs_270 = torch.max(class_probs_270, dim=1)[0].detach().cpu().numpy()
                max_class_probs_240 = torch.max(class_probs_240, dim=1)[0].detach().cpu().numpy()
                max_class_probs_210 = torch.max(class_probs_210, dim=1)[0].detach().cpu().numpy()

                boxplot_data = np.column_stack((max_class_probs_180, max_class_probs_150, max_class_probs_120, max_class_probs_90, max_class_probs_60, max_class_probs_30, max_class_probs_0, max_class_probs_330, max_class_probs_300, max_class_probs_270, max_class_probs_240, max_class_probs_210))
                drop_boxplot_data = np.column_stack((drop_max_class_probs_180, drop_max_class_probs_150, drop_max_class_probs_120, drop_max_class_probs_90, drop_max_class_probs_60, drop_max_class_probs_30, drop_max_class_probs_0, drop_max_class_probs_330, drop_max_class_probs_300, drop_max_class_probs_270, drop_max_class_probs_240, drop_max_class_probs_210))

                # plot classification entropy
                drop_entropy_class_probs_180 = entropy(torch.mean(drop_class_probs_180, dim=2).detach().cpu().numpy(), axis=1)
                drop_entropy_class_probs_150 = entropy(torch.mean(drop_class_probs_150, dim=2).detach().cpu().numpy(), axis=1)
                drop_entropy_class_probs_120 = entropy(torch.mean(drop_class_probs_120, dim=2).detach().cpu().numpy(), axis=1)
                drop_entropy_class_probs_90 = entropy(torch.mean(drop_class_probs_90, dim=2).detach().cpu().numpy(), axis=1)
                drop_entropy_class_probs_60 = entropy(torch.mean(drop_class_probs_60, dim=2).detach().cpu().numpy(), axis=1)
                drop_entropy_class_probs_30 = entropy(torch.mean(drop_class_probs_30, dim=2).detach().cpu().numpy(), axis=1)
                drop_entropy_class_probs_0 = entropy(torch.mean(drop_class_probs_0, dim=2).detach().cpu().numpy(), axis=1)
                drop_entropy_class_probs_330 = entropy(torch.mean(drop_class_probs_330, dim=2).detach().cpu().numpy(), axis=1)
                drop_entropy_class_probs_300 = entropy(torch.mean(drop_class_probs_300, dim=2).detach().cpu().numpy(), axis=1)
                drop_entropy_class_probs_270 = entropy(torch.mean(drop_class_probs_270, dim=2).detach().cpu().numpy(), axis=1)
                drop_entropy_class_probs_240 = entropy(torch.mean(drop_class_probs_240, dim=2).detach().cpu().numpy(), axis=1)
                drop_entropy_class_probs_210 = entropy(torch.mean(drop_class_probs_210, dim=2).detach().cpu().numpy(), axis=1)

                entropy_class_probs_180 = entropy(class_probs_180.detach().cpu().numpy(), axis=1)
                entropy_class_probs_150 = entropy(class_probs_150.detach().cpu().numpy(), axis=1)
                entropy_class_probs_120 = entropy(class_probs_120.detach().cpu().numpy(), axis=1)
                entropy_class_probs_90 = entropy(class_probs_90.detach().cpu().numpy(), axis=1)
                entropy_class_probs_60 = entropy(class_probs_60.detach().cpu().numpy(), axis=1)
                entropy_class_probs_30 = entropy(class_probs_30.detach().cpu().numpy(), axis=1)
                entropy_class_probs_0 = entropy(class_probs_0.detach().cpu().numpy(), axis=1)
                entropy_class_probs_330 = entropy(class_probs_330.detach().cpu().numpy(), axis=1)
                entropy_class_probs_300 = entropy(class_probs_300.detach().cpu().numpy(), axis=1)
                entropy_class_probs_270 = entropy(class_probs_270.detach().cpu().numpy(), axis=1)
                entropy_class_probs_240 = entropy(class_probs_240.detach().cpu().numpy(), axis=1)
                entropy_class_probs_210 = entropy(class_probs_210.detach().cpu().numpy(), axis=1)

                entropy_boxplot_data = np.column_stack((entropy_class_probs_180, entropy_class_probs_150, entropy_class_probs_120, entropy_class_probs_90, entropy_class_probs_60, entropy_class_probs_30, entropy_class_probs_0, entropy_class_probs_330, entropy_class_probs_300, entropy_class_probs_270, entropy_class_probs_240, entropy_class_probs_210))
                drop_entropy_boxplot_data = np.column_stack((drop_entropy_class_probs_180, drop_entropy_class_probs_150, drop_entropy_class_probs_120, drop_entropy_class_probs_90, drop_entropy_class_probs_60, drop_entropy_class_probs_30, drop_entropy_class_probs_0, drop_entropy_class_probs_330, drop_entropy_class_probs_300, drop_entropy_class_probs_270, drop_entropy_class_probs_240, drop_entropy_class_probs_210))

                plot_boxplot_rotating_digits(entropy_boxplot_data, filename='entropy_boxplot_rotating_digits_' + str(class_idx), title='Rotating MNIST', path=d_samples, ylimits=[-0.1, 3.0], ylabel='Classification Entropy')
                plot_boxplot_rotating_digits(drop_entropy_boxplot_data, filename='entropy_boxplot_rotating_digits_MCD_' + str(class_idx), title='Rotating MNIST - MCD', path=d_samples, ylimits=[-0.1, 3.0], ylabel='Classification Entropy')

                plot_boxplot_rotating_digits(boxplot_data, filename='boxplot_rotating_digits_' + str(class_idx), title='Rotating MNIST', path=d_samples)
                plot_boxplot_rotating_digits(drop_boxplot_data, filename='boxplot_rotating_digits_MCD_' + str(class_idx), title='Rotating MNIST - MCD', path=d_samples)



    other_mnist_train_lls, other_mnist_test_lls, other_class_probs_train, other_class_probs_test, other_mnist_train_lls_sup, other_mnist_train_lls_unsup, other_mnist_test_lls_sup, other_mnist_test_lls_unsup = get_other_lls(model, device, d, use_cuda, batch_size, training_dataset=training_dataset)
    print("OTHER Train class entropy: {} OTHER Test class entropy: {}".format(entropy(other_class_probs_train, axis=1).sum(), entropy(other_class_probs_test, axis=1).sum()))

    lls_dict = {"mnist_train":train_lls, "mnist_test":test_lls, "other_mnist_train":other_mnist_train_lls, "other_mnist_test":other_mnist_test_lls}
    head_lls_dict_sup = {"mnist_train":train_lls_sup, "mnist_test":test_lls_sup, "other_mnist_train":other_mnist_train_lls_sup, "other_mnist_test":other_mnist_test_lls_sup}
    head_lls_dict_unsup = {"mnist_train":train_lls_unsup, "mnist_test":test_lls_unsup, "other_mnist_train":other_mnist_train_lls_unsup, "other_mnist_test":other_mnist_test_lls_unsup}
    class_probs_dict = {"mnist_train":class_probs_train.max(axis=1), "mnist_test":class_probs_test.max(axis=1), "other_mnist_train":other_class_probs_train.max(axis=1), "other_mnist_test":other_class_probs_test.max(axis=1)}


    np.save(d_results + 'class_probs_in_domain_train', class_probs_train)
    np.save(d_results + 'class_probs_in_domain_test', class_probs_test)
    np.save(d_results + 'class_probs_ood_train', other_class_probs_train)
    np.save(d_results + 'class_probs_ood_test', other_class_probs_test)

    plot_histograms(lls_dict, filename='histograms_lls', title="Data LLs", path=d_samples, trained_on_fmnist=training_dataset)
    plot_histograms(head_lls_dict_sup, filename='histograms_lls_sup', title="Data SUP LLs", path=d_samples, trained_on_fmnist=training_dataset)
    plot_histograms(head_lls_dict_unsup, filename='histograms_lls_unsup', title="Data UNSUP LLs", path=d_samples, trained_on_fmnist=training_dataset)
    plot_histograms(class_probs_dict, filename='class_probs_histograms', title="Class Probs", path=d_samples, trained_on_fmnist=training_dataset, y_lim=30000)

    train_lls_dropout, class_probs_train_dropout, train_lls_sup_drop, train_lls_unsup_drop = evaluate_model_dropout(model, device, train_loader, "Train DROP", dropout_inference=dropout_inference, output_dir=d)
    test_lls_dropout, class_probs_test_dropout, test_lls_sup_drop, test_lls_unsup_drop = evaluate_model_dropout(model, device, test_loader, "Test DROP", dropout_inference=dropout_inference, output_dir=d)
    print("DROP Train class entropy: {} DROP Test class entropy: {}".format(entropy(class_probs_train_dropout.mean(axis=2), axis=1).sum(), entropy(class_probs_test_dropout.mean(axis=2), axis=1).sum()))

    other_train_loader, other_test_loader = get_data_loaders(use_cuda=use_cuda, device=device, batch_size=batch_size, dataset=get_other_dataset_name(training_dataset))
    other_train_lls_dropout, other_class_probs_train_dropout, other_train_lls_sup_drop, other_train_lls_unsup_drop = evaluate_model_dropout(model, device, other_train_loader, "Other Train DROP", dropout_inference=dropout_inference, output_dir=d)
    other_test_lls_dropout, other_class_probs_test_dropout, other_test_lls_sup_drop, other_test_lls_unsup_drop = evaluate_model_dropout(model, device, other_test_loader, "Other Test DROP", dropout_inference=dropout_inference, output_dir=d)
    print("DROP OTHER Train class entropy: {} DROP OTHER Test class entropy: {}".format(entropy(other_class_probs_train_dropout.mean(axis=2), axis=1).sum(), entropy(other_class_probs_test_dropout.mean(axis=2), axis=1).sum()))

    dropout_lls_dict = {"mnist_train":train_lls_dropout, "mnist_test":test_lls_dropout, "other_mnist_train":other_train_lls_dropout, "other_mnist_test":other_test_lls_dropout}
    dropout_head_lls_dict_sup = {"mnist_train":train_lls_sup_drop, "mnist_test":test_lls_sup_drop, "other_mnist_train":other_train_lls_sup_drop, "other_mnist_test":other_test_lls_sup_drop}
    dropout_head_lls_dict_unsup = {"mnist_train":train_lls_unsup_drop, "mnist_test":test_lls_unsup_drop, "other_mnist_train":other_train_lls_unsup_drop, "other_mnist_test":other_test_lls_unsup_drop}
    dropout_class_probs_dict = {"mnist_train":class_probs_train_dropout.mean(axis=2).max(axis=1), "mnist_test":class_probs_test_dropout.mean(axis=2).max(axis=1), "other_mnist_train":other_class_probs_train_dropout.mean(axis=2).max(axis=1), "other_mnist_test":other_class_probs_test_dropout.mean(axis=2).max(axis=1)}

    np.save(d_results + 'class_probs_in_domain_train_dropout', class_probs_train_dropout)
    np.save(d_results + 'class_probs_in_domain_test_dropout', class_probs_test_dropout)
    np.save(d_results + 'class_probs_ood_train_dropout', other_class_probs_train_dropout)
    np.save(d_results + 'class_probs_ood_test_dropout', other_class_probs_test_dropout)

    plot_histograms(dropout_lls_dict, filename='dropout_histograms_lls', title="DROPOUT Data LLs", path=d_samples, trained_on_fmnist=training_dataset)
    plot_histograms(dropout_head_lls_dict_sup, filename='dropout_histograms_lls_sup', title="DROPOUT Data SUP LLs", path=d_samples, trained_on_fmnist=training_dataset)
    plot_histograms(dropout_head_lls_dict_unsup, filename='dropout_histograms_lls_unsup', title="DROPOUT Data UNSUP LLs", path=d_samples, trained_on_fmnist=training_dataset)
    plot_histograms(dropout_class_probs_dict, filename='dropout_class_probs_histograms', title="DROPOUT Class Probs", path=d_samples, trained_on_fmnist=training_dataset, y_lim=30000)

def get_other_lls(model, device, output_dir='./', use_cuda=False, batch_size=100, training_dataset='mnist'):
    train_loader, test_loader = get_data_loaders(use_cuda=use_cuda, device=device, batch_size=batch_size, dataset=get_other_dataset_name(training_dataset))
    log_string = get_other_dataset_name(training_dataset)
    train_lls, class_probs_train, train_lls_sup, train_lls_unsup = evaluate_model(model, device, train_loader, "Train " + log_string, output_dir=output_dir)
    test_lls, class_probs_test, test_lls_sup, test_lls_unsup = evaluate_model(model, device, test_loader, "Test " + log_string, output_dir=output_dir)
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
    # class_probs = torch.zeros((loader.dataset.data.shape[0],10)).to(device)
    class_probs = torch.zeros((get_data_flatten_shape(loader)[0],10)).to(device)
    correct = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for batch_index, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            data = data.view(data.shape[0], -1)
            output = model(data)
            data_ll.extend(torch.logsumexp(output, dim=1).detach().cpu().numpy())
            data_ll_unsup.extend(output.max(dim=1)[0].detach().cpu().numpy())
            # data_ll_super.extend(output[:, target].detach().cpu().numpy())
            data_ll_super.extend(output.gather(1, target.reshape(-1,1)).squeeze().detach().cpu().numpy())
            # breakpoint()
            # print((output - torch.logsumexp(output, dim=1, keepdims=True)).exp().sum(-1))
            class_probs[batch_index * loader.batch_size: (batch_index+1)*loader.batch_size, :] = (output - torch.logsumexp(output, dim=1, keepdims=True)).exp()
            loss_ce += criterion(output, target).item()  # sum up batch loss
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
    with open(output_dir + 'trainig.out', 'a') as writer:
        writer.write(output_string + "\n")
    assert len(data_ll) == get_data_flatten_shape(loader)[0]
    assert len(data_ll_super) == get_data_flatten_shape(loader)[0]
    assert len(data_ll_unsup) == get_data_flatten_shape(loader)[0]
    return data_ll, class_probs.detach().cpu().numpy(), data_ll_super, data_ll_unsup




def evaluate_model_dropout(model: torch.nn.Module, device, loader, tag, dropout_inference=0.01, n_dropout_iters=100, output_dir="") -> float:
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
    data_ll_super = [] # pix it from the label-th head
    data_ll_unsup = [] # pick the max one
    data_ll = []
    n_dropout_iters = n_dropout_iters
    class_probs = torch.zeros((get_data_flatten_shape(loader)[0], 10, n_dropout_iters)).to(device)
    loss_nll = [0] * n_dropout_iters
    correct = [0] * n_dropout_iters
    drop_corrects = 0
    loss_ce = [0] * n_dropout_iters
    criterion = nn.CrossEntropyLoss(reduction="sum") #TODO NOTE same as cross_entropy_loss_with_logits in this case?
    with torch.no_grad():
        # for data, target in loader:
        for batch_index, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            data = data.view(data.shape[0], -1)
            data_ll_it = torch.zeros(data.shape[0]).to(device)
            data_ll_it_sq = torch.zeros(data.shape[0], n_dropout_iters).to(device)
            data_ll_it_heads = torch.zeros(data.shape[0], 10, n_dropout_iters).to(device)
            for i in range(n_dropout_iters):
                output = model(data, test_dropout=True, dropout_inference=dropout_inference)
                data_ll_it += torch.logsumexp(output, dim=1)
                data_ll_it_sq[:, i] = torch.logsumexp(output, dim=1)
                data_ll_it_heads[:, :, i] = output
                class_probs[batch_index * loader.batch_size: (batch_index+1)*loader.batch_size, :, i] = (output - torch.logsumexp(output, dim=1, keepdims=True)).exp()
                # print("test sum up to 1 -------------------")
                # print((output - torch.logsumexp(output, dim=1, keepdims=True)).exp().sum(-1))
                loss_ce[i] += criterion(output, target).item()  # sum up batch loss
                loss_nll[i] += -output.sum().cpu()
                pred = output.argmax(dim=1)
                correct[i] += (pred == target).sum().item()

            drop_preds = class_probs[batch_index * loader.batch_size: (batch_index+1)*loader.batch_size, :, :].mean(dim=2).argmax(dim=1)
            drop_corrects += (drop_preds == target).sum()


            data_ll_it /= n_dropout_iters # NOTE maybe we should analyze the std as well
            # print("*"*80)
            # print(data_ll_it_sq.mean(dim=1))
            # print(data_ll_it_sq.std(dim=1))
            # data_ll.extend(data_ll_it.detach().cpu().numpy())
            data_ll.extend(data_ll_it_sq.mean(dim=1).detach().cpu().numpy())
            # data_ll_super.extend(data_ll_it_heads.mean(dim=2)[:, target].detach().cpu().numpy())
            data_ll_super.extend(data_ll_it_heads.mean(dim=2).gather(1, target.reshape(-1,1)).squeeze().detach().cpu().numpy())
            data_ll_unsup.extend(data_ll_it_heads.mean(dim=2).max(dim=1)[0].detach().cpu().numpy())
            # breakpoint()

    loss_ce = np.array(loss_ce)
    loss_nll = np.array(loss_nll)
    correct = np.array(correct)

    loss_ce /= len(loader.dataset)
    loss_nll /= len(loader.dataset) + get_data_flatten_shape(loader)[1]

    accuracy = 100.0 * correct / len(loader.dataset)

    # class_probs /= n_dropout_iters
    # class_probs = class_probs.mean(dim=2)
    # drop_preds = class_probs.argmax(dim=1)
    # drop_corrects = (drop_preds == loader.dataset.targets.to(device)).sum().item() #TODO double check here
    # print("drop corrects: {}".format(drop_corrects/loader.dataset.data.shape[0]))
    print("drop corrects: {}".format(drop_corrects/get_data_flatten_shape(loader)[0]))
    # breakpoint()

    output_string = "{} set: Average loss_ce: {:.4f} \u00B1{:.4f} Average loss_nll: {:.4f} \u00B1{:.4f}, Accuracy: {:.4f} \u00B1{:.4f}/{} ({:.0f}% \u00B1{:.4f})".format(
            tag, np.mean(loss_ce), np.std(loss_ce), np.mean(loss_nll), np.std(loss_nll), np.mean(correct), np.std(correct), len(loader.dataset), np.mean(accuracy), np.std(accuracy))
    print(output_string)

    with open(output_dir + 'trainig.out', 'a') as writer:
        writer.write(output_string + "\n")
    assert len(data_ll) == get_data_flatten_shape(loader)[0]
    return data_ll, class_probs.detach().cpu().numpy(), data_ll_super, data_ll_unsup


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
    if severity is not None: kwargs = {'severity':severity}
    # apply corruptiona
    corrupted_images = np.empty((len(dataset), 28, 28), dtype=np.uint8)
    for i in range(len(dataset)):
        corrupted_images[i] = round_and_astype(np.array(corruption(dataset[i][0], **kwargs)))

    corrupted_dataset = CustomTensorDataset(tensors=[torch.tensor(corrupted_images), torch.tensor(dataset.targets)], transform=transformer)

    if class_label is not None:
        targets = torch.tensor(corrupted_dataset.targets.clone().detach())
        target_idx = (targets == class_label).nonzero()
        sampler = torch.utils.data.sampler.SubsetRandomSampler(target_idx.reshape((-1,)))
        data_loader = torch.utils.data.DataLoader(corrputed_dataset, batch_size=100, shuffle=False, sampler=sampler)
    else:
        data_loader = torch.utils.data.DataLoader(corrupted_dataset, batch_size=100, shuffle=False)


    n_samples = 0

    with torch.no_grad():

        if dropout_inference == 0.0:
            n_dropout_iters = 1

        class_probs = torch.zeros(data_loader.dataset.data.shape[0], 10).to(device)

        dropout_output = torch.zeros(data_loader.dataset.data.shape[0], 10, n_dropout_iters).to(device)
        dropout_class_probs = torch.zeros(data_loader.dataset.data.shape[0], 10, n_dropout_iters).to(device)
        dropout_softmax_output = torch.zeros(data_loader.dataset.data.shape[0], 10, n_dropout_iters).to(device)

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
            # assert torch.all(torch.isclose(class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :].sum(-1), torch.tensor((1.)), atol=1e-03)), class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :].sum(-1)

            pred = output.argmax(dim=1)
            n_correct += (pred == target.to(device)).sum().item()

            for i in range(n_dropout_iters):
                if batch_index == len(data_loader) - 1:
                    dropout_output[batch_index * data_loader.batch_size: batch_index * data_loader.batch_size + output.shape[0], :, i] = model(data, test_dropout=True, dropout_inference=dropout_inference)

                    dropout_class_probs[batch_index * data_loader.batch_size:batch_index * data_loader.batch_size + output.shape[0], :, i] = \
                        (dropout_output[batch_index * data_loader.batch_size:batch_index * data_loader.batch_size + output.shape[0], :, i] - \
                        torch.logsumexp(dropout_output[batch_index * data_loader.batch_size:batch_index * data_loader.batch_size + output.shape[0], :, i], dim=1, keepdims=True)).exp()
                    # dropout_class_probs[batch_index * data_loader.batch_size:batch_index * data_loader.batch_size + output.shape[0], :, i] = torch.nan_to_num(dropout_class_probs[batch_index * data_loader.batch_size:batch_index * data_loader.batch_size + output.shape[0], :, i], nan=0.1)
                    assert torch.all(torch.isclose(dropout_class_probs[batch_index * data_loader.batch_size: batch_index * data_loader.batch_size + output.shape[0], :, i].sum(-1), torch.tensor((1.)), atol=1e-03)), dropout_class_probs[batch_index * data_loader.batch_size: batch_index * data_loader.batch_size + output.shape[0], :, i].sum(-1)
                else:
                    dropout_output[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i] = model(data, test_dropout=True, dropout_inference=dropout_inference)


                    dropout_class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i] = \
                        (dropout_output[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i] - \
                        torch.logsumexp(dropout_output[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i], dim=1, keepdims=True)).exp()
                    # dropout_class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i] = torch.nan_to_num(dropout_class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i], nan=0.1)
                    # print(dropout_output[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i])
                    # print(torch.logsumexp(dropout_output[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i], dim=1, keepdims=True))
                    assert torch.all(torch.isclose(dropout_class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i].sum(-1), torch.tensor((1.)), atol=1e-03)), dropout_class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i].sum(-1)

                # assert torch.all(torch.isclose(dropout_class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i].sum(-1), torch.tensor((1.)), atol=1e-03)), dropout_class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i].sum(-1)
        # assert torch.all(torch.isclose(dropout_class_probs[:,:,i].sum(-1), torch.tensor((1.)), atol=1e-04)), dropout_softmax_output[:,:,i].sum(-1)

        print("N of correct test predictions (w/o MCD): {}/{} ({}%)".format(n_correct, n_samples, (n_correct / n_samples)*100))

        dropout_class_probs = dropout_class_probs[:n_samples, :, :]
        class_probs = class_probs[:n_samples, :]

        if class_label is not None:
            drop_n_correct = (torch.argmax(torch.mean(dropout_class_probs, 2), dim=1) == class_label).sum()
        else:
            drop_n_correct = (torch.argmax(torch.mean(dropout_class_probs, 2), dim=1) == data_loader.dataset.targets.to(device)).sum()
        # print("N of correct test predictions with MCD: {}".format(drop_n_correct))
        print("N of correct test precictions with MCD: DROP N of correct predictions of test samples: {}/{} ({}%)".format(drop_n_correct, n_samples, (drop_n_correct / n_samples)*100))

        # return (dropout_class_probs.mean(dim=2).max(dim=1)).detach().cpu().numpy()
        # breakpoint()
        # return torch.max(torch.mean(dropout_class_probs, dim=2), dim=1)[0].detach().cpu().numpy()
        return dropout_class_probs, class_probs

def evaluate_model_corrupted_cifar(model: torch.nn.Module, device, loader, tag, dropout_inference=0.01, n_dropout_iters=100, output_dir="", corruption='fog', corruption_level=1, class_label=None) -> float:

    d = output_dir
    d_samples = d + "samples/"
    d_model = d + "model/"
    ensure_dir(d)
    ensure_dir(d_samples)
    ensure_dir(d_model)

    assert corruption in ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 'frost', 'gaussian_blur', 'gaussian_noise',
                          'glass_blur', 'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate', 'saturate', 'shot_noise', 'snow',
                          'spatter', 'speckle_noise', 'zoom_blur']
    corrupted_dataset = np.load('/home/fabrizio/research/CIFAR-10-C/{}.npy'.format(corruption))
    corrupted_dataset = corrupted_dataset[(corruption_level - 1)*10000 : (10000) * corruption_level ]

    labels = np.load('/home/fabrizio/research/CIFAR-10-C/labels.npy')
    labels = labels[(corruption_level - 1)*10000 : (10000) * corruption_level ]
    # breakpoint()
    assert corrupted_dataset.shape[0] == 10000
    assert labels.shape[0] == 10000

    model.eval()
    # kwargs = {"num_workers": 8, "pin_memory": True} if use_cuda else {}

    cifar10_transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

    # cifar10_test_loader = torch.utils.data.DataLoader(
    #     datasets.CIFAR10(root='../data', train=False, download=True, transform=cifar10_transformer),
    #     batch_size=batch_size,
    #     shuffle=False,
    #     **kwargs
    # )

    corrupted_dataset = CustomTensorDataset(tensors=[torch.tensor(corrupted_dataset), torch.tensor(labels)], transform=cifar10_transformer)

    if class_label is not None:
        targets = torch.tensor(corrupted_dataset.targets.clone().detach())
        target_idx = (targets == class_label).nonzero()
        sampler = torch.utils.data.sampler.SubsetRandomSampler(target_idx.reshape((-1,)))
        data_loader = torch.utils.data.DataLoader(corrputed_dataset, batch_size=100, shuffle=False, sampler=sampler)
    else:
        data_loader = torch.utils.data.DataLoader(corrupted_dataset, batch_size=100, shuffle=False)


    n_samples = 0

    with torch.no_grad():

        if dropout_inference == 0.0:
            n_dropout_iters = 1

        class_probs = torch.zeros(data_loader.dataset.data.shape[0], 10).to(device)

        dropout_output = torch.zeros(data_loader.dataset.data.shape[0], 10, n_dropout_iters).to(device)
        dropout_class_probs = torch.zeros(data_loader.dataset.data.shape[0], 10, n_dropout_iters).to(device)
        dropout_softmax_output = torch.zeros(data_loader.dataset.data.shape[0], 10, n_dropout_iters).to(device)

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
            # assert torch.all(torch.isclose(class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :].sum(-1), torch.tensor((1.)), atol=1e-03)), class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :].sum(-1)

            pred = output.argmax(dim=1)
            n_correct += (pred == target.to(device)).sum().item()

            for i in range(n_dropout_iters):
                if batch_index == len(data_loader) - 1:
                    dropout_output[batch_index * data_loader.batch_size: batch_index * data_loader.batch_size + output.shape[0], :, i] = model(data, test_dropout=True, dropout_inference=dropout_inference)

                    dropout_class_probs[batch_index * data_loader.batch_size:batch_index * data_loader.batch_size + output.shape[0], :, i] = \
                        (dropout_output[batch_index * data_loader.batch_size:batch_index * data_loader.batch_size + output.shape[0], :, i] - \
                        torch.logsumexp(dropout_output[batch_index * data_loader.batch_size:batch_index * data_loader.batch_size + output.shape[0], :, i], dim=1, keepdims=True)).exp()
                    # dropout_class_probs[batch_index * data_loader.batch_size:batch_index * data_loader.batch_size + output.shape[0], :, i] = torch.nan_to_num(dropout_class_probs[batch_index * data_loader.batch_size:batch_index * data_loader.batch_size + output.shape[0], :, i], nan=0.1)
                    assert torch.all(torch.isclose(dropout_class_probs[batch_index * data_loader.batch_size: batch_index * data_loader.batch_size + output.shape[0], :, i].sum(-1), torch.tensor((1.)), atol=1e-03)), dropout_class_probs[batch_index * data_loader.batch_size: batch_index * data_loader.batch_size + output.shape[0], :, i].sum(-1)
                else:
                    dropout_output[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i] = model(data, test_dropout=True, dropout_inference=dropout_inference)


                    dropout_class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i] = \
                        (dropout_output[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i] - \
                        torch.logsumexp(dropout_output[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i], dim=1, keepdims=True)).exp()
                    # dropout_class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i] = torch.nan_to_num(dropout_class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i], nan=0.1)
                    # print(dropout_output[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i])
                    # print(torch.logsumexp(dropout_output[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i], dim=1, keepdims=True))
                    assert torch.all(torch.isclose(dropout_class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i].sum(-1), torch.tensor((1.)), atol=1e-03)), dropout_class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i].sum(-1)

                # assert torch.all(torch.isclose(dropout_class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i].sum(-1), torch.tensor((1.)), atol=1e-03)), dropout_class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i].sum(-1)
        # assert torch.all(torch.isclose(dropout_class_probs[:,:,i].sum(-1), torch.tensor((1.)), atol=1e-04)), dropout_softmax_output[:,:,i].sum(-1)

        print("N of correct test predictions (w/o MCD): {}/{} ({}%)".format(n_correct, n_samples, (n_correct / n_samples)*100))

        dropout_class_probs = dropout_class_probs[:n_samples, :, :]
        class_probs = class_probs[:n_samples, :]

        if class_label is not None:
            drop_n_correct = (torch.argmax(torch.mean(dropout_class_probs, 2), dim=1) == class_label).sum()
        else:
            drop_n_correct = (torch.argmax(torch.mean(dropout_class_probs, 2), dim=1) == data_loader.dataset.targets.to(device)).sum()
        # print("N of correct test predictions with MCD: {}".format(drop_n_correct))
        print("N of correct test precictions with MCD: DROP N of correct predictions of test samples: {}/{} ({}%)".format(drop_n_correct, n_samples, (drop_n_correct / n_samples)*100))

        # return (dropout_class_probs.mean(dim=2).max(dim=1)).detach().cpu().numpy()
        # breakpoint()
        # return torch.max(torch.mean(dropout_class_probs, dim=2), dim=1)[0].detach().cpu().numpy()
        return dropout_class_probs, class_probs

def evaluate_model_rotated_digits(model: torch.nn.Module, device, loader, tag, dropout_inference=0.01, n_dropout_iters=100, output_dir="", degrees=30, class_label=None) -> float:

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
        # sanity check, MCD inference for non-rotated data
        #
        # dropout_output = torch.zeros(data_loader.dataset.data.shape[0], 10, n_dropout_iters).to(device)
        # dropout_class_probs = torch.zeros(data_loader.dataset.data.shape[0], 10, n_dropout_iters).to(device)
        # dropout_softmax_output = torch.zeros(data_loader.dataset.data.shape[0], 10, n_dropout_iters).to(device)

        # for batch_index, (data, target) in enumerate(data_loader):
        #     data = data.to(device)
        #     data = data.view(data.shape[0], -1)

        #     n_samples += data.shape[0]

        #     for i in range(n_dropout_iters):
        #         dropout_output[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i] = model(data, test_dropout=True, dropout_inference=dropout_inference)

        #         dropout_class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i] = \
        #             (dropout_output[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i] - \
        #              torch.logsumexp(dropout_output[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i], dim=1, keepdims=True)).exp()

        #         assert torch.all(torch.isclose(dropout_class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i].sum(-1), torch.tensor((1.)), atol=1e-03)), dropout_class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i].sum(-1)

        # drop_n_correct = (torch.argmax(torch.mean(dropout_class_probs, 2), dim=1) == data_loader.dataset.targets.to(device)).sum()
        # print("DROP N CORRECT: {}".format(drop_n_correct))
        # print("N of correct predictions of samples: {}/{} ({}%)".format(drop_n_correct, n_samples, (drop_n_correct / n_samples)))

        if dropout_inference == 0.0:
            n_dropout_iters = 1

        class_probs = torch.zeros(data_loader.dataset.data.shape[0], 10).to(device)
        # print(class_probs.shape)

        dropout_output = torch.zeros(data_loader.dataset.data.shape[0], 10, n_dropout_iters).to(device)
        dropout_class_probs = torch.zeros(data_loader.dataset.data.shape[0], 10, n_dropout_iters).to(device)
        dropout_softmax_output = torch.zeros(data_loader.dataset.data.shape[0], 10, n_dropout_iters).to(device)

        n_correct = 0
        n_samples = 0


        for batch_index, (data, target) in enumerate(data_loader):
            data = data.to(device)
            data = torchvision.transforms.functional.rotate(data.reshape(-1,1,28,28), degrees, fill=-mean/std).reshape(-1,28,28)
            data = data.view(data.shape[0], -1)

            n_samples += data.shape[0]

            output = model(data, test_dropout=False, dropout_inference=0.0)
            if batch_index == len(data_loader) - 1:
                # print(batch_index)
                # print(len(data_loader)-1)
                class_probs[batch_index * data_loader.batch_size: batch_index * data_loader.batch_size + output.shape[0] , :] = (output - torch.logsumexp(output, dim=1, keepdims=True)).exp()
                assert torch.all(torch.isclose(class_probs[batch_index * data_loader.batch_size: batch_index * data_loader.batch_size + output.shape[0], :].sum(-1), torch.tensor((1.)), atol=1e-03)), class_probs[batch_index * data_loader.batch_size:batch_index * data_loader.batch_size + output.shape[0], :].sum(-1)
            else:
                class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :] = (output - torch.logsumexp(output, dim=1, keepdims=True)).exp()
                assert torch.all(torch.isclose(class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :].sum(-1), torch.tensor((1.)), atol=1e-03)), class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :].sum(-1)
            # assert torch.all(torch.isclose(class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :].sum(-1), torch.tensor((1.)), atol=1e-03)), class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :].sum(-1)

            pred = output.argmax(dim=1)
            n_correct += (pred == target.to(device)).sum().item()

            for i in range(n_dropout_iters):
                if batch_index == len(data_loader) - 1:
                    dropout_output[batch_index * data_loader.batch_size: batch_index * data_loader.batch_size + output.shape[0], :, i] = model(data, test_dropout=True, dropout_inference=dropout_inference)

                    dropout_class_probs[batch_index * data_loader.batch_size:batch_index * data_loader.batch_size + output.shape[0], :, i] = \
                        (dropout_output[batch_index * data_loader.batch_size:batch_index * data_loader.batch_size + output.shape[0], :, i] - \
                        torch.logsumexp(dropout_output[batch_index * data_loader.batch_size:batch_index * data_loader.batch_size + output.shape[0], :, i], dim=1, keepdims=True)).exp()
                    # dropout_class_probs[batch_index * data_loader.batch_size:batch_index * data_loader.batch_size + output.shape[0], :, i] = torch.nan_to_num(dropout_class_probs[batch_index * data_loader.batch_size:batch_index * data_loader.batch_size + output.shape[0], :, i], nan=0.1)
                    assert torch.all(torch.isclose(dropout_class_probs[batch_index * data_loader.batch_size: batch_index * data_loader.batch_size + output.shape[0], :, i].sum(-1), torch.tensor((1.)), atol=1e-03)), dropout_class_probs[batch_index * data_loader.batch_size: batch_index * data_loader.batch_size + output.shape[0], :, i].sum(-1)
                else:
                    dropout_output[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i] = model(data, test_dropout=True, dropout_inference=dropout_inference)


                    dropout_class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i] = \
                        (dropout_output[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i] - \
                        torch.logsumexp(dropout_output[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i], dim=1, keepdims=True)).exp()
                    # dropout_class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i] = torch.nan_to_num(dropout_class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i], nan=0.1)
                    # print(dropout_output[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i])
                    # print(torch.logsumexp(dropout_output[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i], dim=1, keepdims=True))
                    assert torch.all(torch.isclose(dropout_class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i].sum(-1), torch.tensor((1.)), atol=1e-03)), dropout_class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i].sum(-1)

                # assert torch.all(torch.isclose(dropout_class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i].sum(-1), torch.tensor((1.)), atol=1e-03)), dropout_class_probs[batch_index * data_loader.batch_size: (batch_index+1)*data_loader.batch_size, :, i].sum(-1)
        # assert torch.all(torch.isclose(dropout_class_probs[:,:,i].sum(-1), torch.tensor((1.)), atol=1e-04)), dropout_softmax_output[:,:,i].sum(-1)

        print("N of correct test predictions (w/o MCD): {}/{} ({}%)".format(n_correct, n_samples, (n_correct / n_samples)*100))

        dropout_class_probs = dropout_class_probs[:n_samples, :, :]
        class_probs = class_probs[:n_samples, :]

        if class_label is not None:
            drop_n_correct = (torch.argmax(torch.mean(dropout_class_probs, 2), dim=1) == class_label).sum()
        else:
            drop_n_correct = (torch.argmax(torch.mean(dropout_class_probs, 2), dim=1) == data_loader.dataset.targets.to(device)).sum()
        # print("N of correct test predictions with MCD: {}".format(drop_n_correct))
        print("N of correct test precictions with MCD: DROP N of correct predictions of test samples: {}/{} ({}%)".format(drop_n_correct, n_samples, (drop_n_correct / n_samples)*100))

        # return (dropout_class_probs.mean(dim=2).max(dim=1)).detach().cpu().numpy()
        # breakpoint()
        # return torch.max(torch.mean(dropout_class_probs, dim=2), dim=1)[0].detach().cpu().numpy()
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
    plt.savefig(path + filename + '_.png')
    plt.savefig(path + filename + '_.pdf')
    plt.close()

def plot_boxplot_corrupted_digits(data, filename='boxplot_corrupted_digits', title="", path=None, ylimits=[-0.1, 1.1], xlabel='Corruption ID', ylabel='Classification confidence'):

    fig, axs = plt.subplots()
    axs.boxplot(data, showmeans=True, meanline=True, labels=['SN', 'IN', 'GB', 'MB', 'SH', 'SC', 'RO', 'BR', 'TR', 'ST', 'FOG', 'SPA', 'DOT', 'ZIG', 'CAE'], showfliers=False)
    ax = plt.gca()
    ax.set_ylim(ylimits)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.title(title)
    plt.savefig(path + filename + '_.png')
    plt.savefig(path + filename + '_.pdf')
    plt.close()

def plot_histograms(lls_dict, filename='histogram', title="", path=None, trained_on_fmnist=False, y_lim=None):

    dataset_names = ["F-MNIST", "MNIST"]
    if not trained_on_fmnist: dataset_names.reverse()

    plt.hist(np.nan_to_num(lls_dict["mnist_train"]), label=dataset_names[0] + " Train", alpha=0.5, bins=20, color='red')
    plt.hist(np.nan_to_num(lls_dict["mnist_test"]), label=dataset_names[0] + " Test", alpha=0.5, bins=20, color='blue')
    plt.hist(np.nan_to_num(lls_dict["other_mnist_train"]), label=dataset_names[1] + " Train (OOD)", alpha=0.5, bins=20, color='orange')
    plt.hist(np.nan_to_num(lls_dict["other_mnist_test"]), label=dataset_names[1] + " Test (OOD)", alpha=0.5, bins=20, color='yellow')
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
    # TODO double check batch index stuff to do not skip last samples < batch size
    #
    # run_torch(10, 100, dropout_inference=0.1, dropout_spn=0.1, class_label=1)
    # run_torch(100, 100, dropout_inference=0.1, dropout_spn=0.1, class_label=1)
    # run_torch(100, 100, dropout_inference=0.1, dropout_spn=0.1, class_label=5)
    # run_torch(100, 100, dropout_inference=0.1, dropout_spn=0.1, class_label=9)
    # run_torch(100, 100, dropout_inference=0.2, dropout_spn=0.2, class_label=1)
    # run_torch(10, 200, dropout_inference=0.2, dropout_spn=0.2, class_label=1, train_on_fmnist=True, lmbda=1.0)

    # load_torch(model_dir='results/2022-05-16_16-10-50/model/', train_on_fmnist=True, dropout_inference=0.2) # lambda 0
    # load_torch(model_dir='results/2022-05-16_04-34-36/model/', train_on_fmnist=True, dropout_inference=0.2) # lambda 0.2
    # load_torch(model_dir='results/2022-05-15_11-08-35/model/', train_on_fmnist=True, dropout_inference=0.2) # lambda 0.8
    # load_torch(model_dir='results/2022-05-14_00-30-37/model/', train_on_fmnist=False, dropout_inference=0.2) # lambda 0.8 mnist
    # load_torch(model_dir='results/2022-05-13_18-43-00/model/', train_on_fmnist=True, dropout_inference=0.2) # lambda 1
    # load_torch(model_dir='results/2022-05-13_12-55-49/model/', train_on_fmnist=False, dropout_inference=0.2) # lambda 1. mnist
    # load_torch(model_dir='results/2022-05-14_17-46-17/model/', train_on_fmnist=False, dropout_inference=0.2) # lambda 0.2 mnist
    load_torch(model_dir='results/2022-05-15_05-20-35/model/', train_on_fmnist=False, dropout_inference=0.2) # lambda 0. mnist
    sys.exit()
    run_torch(200, 200, dropout_inference=0.2, dropout_spn=0.2, class_label=1, train_on_fmnist='svhn', lmbda=1.0, eval_single_digit=False, toy_setting=False, eval_rotation=False)

    run_torch(200, 200, dropout_inference=0.2, dropout_spn=0.2, class_label=1, train_on_fmnist=False, lmbda=1.0, eval_single_digit=False, toy_setting=False, eval_rotation=False)
    run_torch(200, 200, dropout_inference=0.2, dropout_spn=0.2, class_label=1, train_on_fmnist=True, lmbda=1.0, eval_single_digit=False, toy_setting=False, eval_rotation=False)

    run_torch(200, 200, dropout_inference=0.2, dropout_spn=0.2, class_label=1, train_on_fmnist=False, lmbda=0.8, eval_single_digit=False, toy_setting=False, eval_rotation=False)
    run_torch(200, 200, dropout_inference=0.2, dropout_spn=0.2, class_label=1, train_on_fmnist=False, lmbda=0.6, eval_single_digit=False, toy_setting=False, eval_rotation=False)
    run_torch(200, 200, dropout_inference=0.2, dropout_spn=0.2, class_label=1, train_on_fmnist=False, lmbda=0.4, eval_single_digit=False, toy_setting=False, eval_rotation=False)
    run_torch(200, 200, dropout_inference=0.2, dropout_spn=0.2, class_label=1, train_on_fmnist=False, lmbda=0.2, eval_single_digit=False, toy_setting=False, eval_rotation=False)
    run_torch(200, 200, dropout_inference=0.2, dropout_spn=0.2, class_label=1, train_on_fmnist=False, lmbda=0.1, eval_single_digit=False, toy_setting=False, eval_rotation=False)
    run_torch(200, 200, dropout_inference=0.2, dropout_spn=0.2, class_label=1, train_on_fmnist=False, lmbda=0.0, eval_single_digit=False, toy_setting=False, eval_rotation=False)

    run_torch(200, 200, dropout_inference=0.2, dropout_spn=0.2, class_label=1, train_on_fmnist=True, lmbda=0.8, eval_single_digit=False, toy_setting=False, eval_rotation=False)
    run_torch(200, 200, dropout_inference=0.2, dropout_spn=0.2, class_label=1, train_on_fmnist=True, lmbda=0.6, eval_single_digit=False, toy_setting=False, eval_rotation=False)
    run_torch(200, 200, dropout_inference=0.2, dropout_spn=0.2, class_label=1, train_on_fmnist=True, lmbda=0.4, eval_single_digit=False, toy_setting=False, eval_rotation=False)
    run_torch(200, 200, dropout_inference=0.2, dropout_spn=0.2, class_label=1, train_on_fmnist=True, lmbda=0.2, eval_single_digit=False, toy_setting=False, eval_rotation=False)
    run_torch(200, 200, dropout_inference=0.2, dropout_spn=0.2, class_label=1, train_on_fmnist=True, lmbda=0.1, eval_single_digit=False, toy_setting=False, eval_rotation=False)
    run_torch(200, 200, dropout_inference=0.2, dropout_spn=0.2, class_label=1, train_on_fmnist=True, lmbda=0.0, eval_single_digit=False, toy_setting=False, eval_rotation=False)
    ####
    ####
    run_torch(200, 200, dropout_inference=0.2, dropout_spn=0.2, class_label=1, train_on_fmnist=True, lmbda=0.8, toy_setting=False)
    run_torch(200, 200, dropout_inference=0.2, dropout_spn=0.2, class_label=1, train_on_fmnist=True, lmbda=0.6, toy_setting=False)
    run_torch(200, 200, dropout_inference=0.2, dropout_spn=0.2, class_label=1, train_on_fmnist=True, lmbda=0.4, toy_setting=False)
    run_torch(200, 200, dropout_inference=0.2, dropout_spn=0.2, class_label=1, train_on_fmnist=True, lmbda=0.2, toy_setting=False)
    run_torch(200, 200, dropout_inference=0.2, dropout_spn=0.2, class_label=1, train_on_fmnist=True, lmbda=0.1, toy_setting=False)
    run_torch(200, 200, dropout_inference=0.2, dropout_spn=0.2, class_label=1, train_on_fmnist=True, lmbda=0.0, toy_setting=False)
    # run_torch(200, 200, dropout_inference=0.2, dropout_spn=0.2, class_label=1, train_on_fmnist='cifar', lmbda=1.0, toy_setting=False)
    # run_torch(3, 200, dropout_inference=0.2, dropout_spn=0.2, class_label=1, train_on_fmnist=False, lmbda=1.0, toy_setting=False)
    # run_torch(200, 200, dropout_inference=0.2, dropout_spn=0.2, class_label=1, train_on_fmnist=False, lmbda=1.0, toy_setting=False)
    sys.exit()
    run_torch(200, 200, dropout_inference=0.2, dropout_spn=0.2, class_label=1, train_on_fmnist=False, lmbda=1.0, toy_setting=False)
    run_torch(200, 200, dropout_inference=0.2, dropout_spn=0.2, class_label=1, train_on_fmnist='kmnist', lmbda=1.0, toy_setting=False)
    run_torch(200, 200, dropout_inference=0.2, dropout_spn=0.2, class_label=1, train_on_fmnist='emnist', lmbda=1.0, toy_setting=False)

    run_torch(200, 200, dropout_inference=0.3, dropout_spn=0.3, class_label=1, train_on_fmnist=False, lmbda=1.0)
    run_torch(200, 200, dropout_inference=0.5, dropout_spn=0.5, class_label=1, train_on_fmnist=False, lmbda=1.0)

    run_torch(200, 200, dropout_inference=0.1, dropout_spn=0.1, class_label=1, train_on_fmnist=True, lmbda=1.0)
    run_torch(200, 200, dropout_inference=0.2, dropout_spn=0.1, class_label=1, train_on_fmnist=False, lmbda=1.0)
    run_torch(200, 200, dropout_inference=0.2, dropout_spn=0.2, class_label=1, train_on_fmnist=False, lmbda=1.0)
    run_torch(200, 200, dropout_inference=0.2, dropout_spn=0.2, class_label=1, train_on_fmnist=True, lmbda=1.0)
    run_torch(200, 200, dropout_inference=0.2, dropout_spn=0.1, class_label=1, train_on_fmnist=True, lmbda=1.0)


    run_torch(200, 200, dropout_inference=0.2, dropout_spn=0.2, class_label=1, train_on_fmnist=True, lmbda=1.0)
    run_torch(200, 200, dropout_inference=0.5, dropout_spn=0.5, class_label=1, train_on_fmnist=True, lmbda=1.0)
    run_torch(200, 200, dropout_inference=0.8, dropout_spn=0.8, class_label=1, train_on_fmnist=True, lmbda=1.0)


    run_torch(200, 200, dropout_inference=0.4, dropout_spn=0.4, class_label=1, train_on_fmnist=False, lmbda=1.0)
    run_torch(200, 200, dropout_inference=0.5, dropout_spn=0.5, class_label=1, train_on_fmnist=False, lmbda=1.0)
    run_torch(200, 200, dropout_inference=0.7, dropout_spn=0.7, class_label=1, train_on_fmnist=False, lmbda=1.0)
    run_torch(200, 200, dropout_inference=0.8, dropout_spn=0.8, class_label=1, train_on_fmnist=False, lmbda=1.0)

    run_torch(100, 200, dropout_inference=0.2, dropout_spn=0.2, class_label=1, train_on_fmnist=False, lmbda=0.5)
    run_torch(100, 200, dropout_inference=0.2, dropout_spn=0.2, class_label=1, train_on_fmnist=False, lmbda=0.0)
    run_torch(100, 200, dropout_inference=0.2, dropout_spn=0.2, class_label=1, train_on_fmnist=False, lmbda=1.0)


    run_torch(100, 200, dropout_inference=0.1, dropout_spn=0.1, class_label=1, train_on_fmnist=True)
    run_torch(100, 200, dropout_inference=0.2, dropout_spn=0.2, class_label=1, train_on_fmnist=True)
    run_torch(100, 200, dropout_inference=0.4, dropout_spn=0.4, class_label=1, train_on_fmnist=True)
    run_torch(100, 200, dropout_inference=0.7, dropout_spn=0.7, class_label=1, train_on_fmnist=True)
    run_torch(100, 200, dropout_inference=0.8, dropout_spn=0.8, class_label=1, train_on_fmnist=True)


    run_torch(100, 200, dropout_inference=0.5, dropout_spn=0.5, class_label=1, train_on_fmnist=True)
    run_torch(100, 200, dropout_inference=0.5, dropout_spn=0.5, class_label=1, train_on_fmnist=False)

    run_torch(100, 200, dropout_inference=0.5, dropout_spn=0.5, class_label=1, train_on_fmnist=False)
    # run_torch(100, 100, dropout_inference=0.2, dropout_spn=0.0, class_label=1, train_on_fmnist=False)
    run_torch(100, 200, dropout_inference=0.1, dropout_spn=0.1, class_label=1, train_on_fmnist=True)
    run_torch(100, 200, dropout_inference=0.1, dropout_spn=0.1, class_label=1, train_on_fmnist=False)

    run_torch(100, 200, dropout_inference=0.2, dropout_spn=0.2, class_label=1, train_on_fmnist=True)
    run_torch(100, 200, dropout_inference=0.2, dropout_spn=0.2, class_label=1, train_on_fmnist=False)

    run_torch(100, 200, dropout_inference=0.7, dropout_spn=0.7, class_label=1, train_on_fmnist=True)
    run_torch(100, 200, dropout_inference=0.7, dropout_spn=0.7, class_label=1, train_on_fmnist=False)

    run_torch(100, 200, dropout_inference=0.9, dropout_spn=0.9, class_label=1, train_on_fmnist=True)
    run_torch(100, 200, dropout_inference=0.9, dropout_spn=0.9, class_label=1, train_on_fmnist=False)


    run_torch(100, 100, dropout_inference=0.0, dropout_spn=0.0, class_label=1)
    run_torch(100, 100, dropout_inference=0.1, dropout_spn=0.0, class_label=1)
    run_torch(100, 100, dropout_inference=0.2, dropout_spn=0.0, class_label=1)
    run_torch(100, 100, dropout_inference=0.0, dropout_spn=0.1, class_label=1)
    run_torch(100, 100, dropout_inference=0.1, dropout_spn=0.1, class_label=1)
    run_torch(100, 100, dropout_inference=0.2, dropout_spn=0.1, class_label=1)
    run_torch(100, 100, dropout_inference=0.0, dropout_spn=0.2, class_label=1)
    run_torch(100, 100, dropout_inference=0.1, dropout_spn=0.2, class_label=1)
    run_torch(100, 100, dropout_inference=0.2, dropout_spn=0.2, class_label=1)

    run_torch(100, 100, dropout_inference=0.0, dropout_spn=0.0, class_label=5)
    run_torch(100, 100, dropout_inference=0.1, dropout_spn=0.0, class_label=5)
    run_torch(100, 100, dropout_inference=0.2, dropout_spn=0.0, class_label=5)
    run_torch(100, 100, dropout_inference=0.0, dropout_spn=0.1, class_label=5)
    run_torch(100, 100, dropout_inference=0.1, dropout_spn=0.1, class_label=5)
    run_torch(100, 100, dropout_inference=0.2, dropout_spn=0.1, class_label=5)
    run_torch(100, 100, dropout_inference=0.0, dropout_spn=0.2, class_label=5)
    run_torch(100, 100, dropout_inference=0.1, dropout_spn=0.2, class_label=5)
    run_torch(100, 100, dropout_inference=0.2, dropout_spn=0.2, class_label=5)
    # run_torch(100, 100, dropout_inference=0.1)
    # run_torch(200, 256, dropout_inference=0.1)
    run_torch(100, 100, dropout_inference=0.0, dropout_spn=0.0, class_label=9)
    run_torch(100, 100, dropout_inference=0.1, dropout_spn=0.0, class_label=9)
    run_torch(100, 100, dropout_inference=0.2, dropout_spn=0.0, class_label=9)
    run_torch(100, 100, dropout_inference=0.0, dropout_spn=0.1, class_label=9)
    run_torch(100, 100, dropout_inference=0.1, dropout_spn=0.1, class_label=9)
    run_torch(100, 100, dropout_inference=0.2, dropout_spn=0.1, class_label=9)
    run_torch(100, 100, dropout_inference=0.0, dropout_spn=0.2, class_label=9)
    run_torch(100, 100, dropout_inference=0.1, dropout_spn=0.2, class_label=9)
    run_torch(100, 100, dropout_inference=0.2, dropout_spn=0.2, class_label=9)

# print(torch.mean(torch.nn.functional.softmax(output, dim=1),0))
# print(torch.std(torch.nn.functional.softmax(output, dim=1),0))
# print( (output.argmax(dim=1) == 1).sum() )
# print( (output.argmax(dim=1) == 1).sum() / output.shape[0])

# output = model(one_train_samples, test_dropout=False)
# prior = np.log(0.1); print((output + prior - torch.logsumexp(output + prior, dim=1, keepdims=True)).exp().sum(-1))

# test_output_probs = model(rotated_one.reshape(rotated_one.shape[0], -1)[:3,:])
# #     (test_output_probs + torch.log(torch.Tensor([0.1]).to(device))) -
# #         # (torch.logsumexp(test_output_probs + torch.log(torch.Tensor([0.1]).to(device)), dim=1))
# #         (torch.logsumexp(test_output_probs, dim=1) + torch.log(torch.Tensor([0.1]).to(device)))
# #     )
# # )
# prior = np.log(0.1); print((test_output_probs + prior - torch.logsumexp(test_output_probs + prior, dim=1, keepdims=True)).exp().sum(-1))
# print("*"*80)
# print(test_output_probs)
# print(torch.logsumexp(test_output_probs, dim=1, keepdims=True))
# print(torch.logsumexp(test_output_probs, dim=1, keepdims=True).shape)
# norm_probs = (test_output_probs - torch.logsumexp(test_output_probs, dim=1, keepdims=True)).exp().sum(-1)
# # assert norm_probs[0] == 1.0, norm_probs
# print((test_output_probs - torch.logsumexp(test_output_probs, dim=1, keepdims=True)).exp().sum(-1))

# rotated_one = rotated_one.to(device)
# rotated_one = rotated_one.sub_(mean).div_(std)
# one_train_samples = one_train_samples.sub_(mean).div_(std)
#

# fig, axs = plt.subplots()
# axs.boxplot(np.transpose(data[:, 1, :]))
# ax = plt.gca()
# ax.set_ylim([-0.1, 1.1])
# plt.title("Digit 5")
# plt.savefig('results/samples/boxplot_5.png')

# fig, axs = plt.subplots()
# axs.boxplot(np.transpose(data[:, 2, :]))
# ax = plt.gca()
# ax.set_ylim([-0.1, 1.1])
# plt.title("Digit 7")
# plt.savefig('results/samples/boxplot_7.png')

# def plot_violinplot(data, path=None):
#     # colors = ['C{}'.format(i) for i in range(3)]
#     fig, axs = plt.subplots()
#     axs.violinplot(np.transpose(data[:, 0, :]))
#     axs.violinplot(np.transpose(data[:, 1, :]))
#     axs.violinplot(np.transpose(data[:, 2, :]))
#     #axs.legend(['one', 'five', 'seven'])
#     plt.savefig('results/samples/violinplot.png')

# def scatterplot(data, path=None):
#     fig, ax = plt.subplots()
#     labels = ["one", "five", "seven"]
#     colors = ['tab:blue', 'tab:orange', 'tab:green']
#     for idx, color in enumerate(colors):
#         for j in np.arange(10):
#             ax.scatter(x=np.array([j] * data.shape[2]), y=data[j, idx, :], c=color, alpha=0.3, edgecolors='none')
#     #ax.legend(['one', 'five', 'seven'])
#     ax.legend(['one', 'five', 'seven'], loc='center left', bbox_to_anchor=(1, 0.5))
#     leg = ax.get_legend()
#     for i in range(3):
#         leg.legendHandles[i].set_color(colors[i])
#     plt.savefig('results/samples/scatterplot.png')

# def plot_eventplot(data, path=None):
#     colors = ['C{}'.format(i) for i in range(3)]
#     linelengths = [1] * 10
#     lineoffsets = np.arange(10)
#     print(linelengths)
#     print(lineoffsets)
#     print(data[:, 0, :].shape)
#     fig, axs = plt.subplots()
#     axs.eventplot(data[:, 0, :], colors=colors[0], lineoffsets=lineoffsets, linelengths=linelengths, orientation='vertical')
#     axs.eventplot(data[:, 1, :], colors=colors[1], lineoffsets=lineoffsets, linelengths=linelengths, orientation='vertical')
#     axs.eventplot(data[:, 2, :], colors=colors[2], lineoffsets=lineoffsets, linelengths=linelengths, orientation='vertical')
#     axs.legend(['one', 'five', 'seven'])
#     plt.savefig('results/samples/eventplot.png')

# TODO

# rotating mnist experiment + plot where we average the class probs, maybe starts with -90 to +90 degrees when rotating
# regarding averaging should we consider only the predicted class probs or all?
#
# another interesting experiment with rotating mnist would be showing a confusion matrix over the rotations

# MCD over SPN with one forward pass (pass a vector of values for each sample)
# formal description on what MCD is doing on a SPN (and is Bernoulli the only good way for dropouts? are there alternative distributions?)
#
