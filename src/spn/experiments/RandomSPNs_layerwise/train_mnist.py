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

from torch.utils.data import Dataset
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
    Count the number of parameters in a model.

    Args:
        model (torch.nn.Module): PyTorch model.

    Returns:
        int: Number of learnable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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



def get_mnist_loaders(use_cuda, device, batch_size, f_mnist=False):
    if isinstance(f_mnist, bool):
        return get_f_mnist_loaders(use_cuda, device, batch_size) if f_mnist else get_d_mnist_loaders(use_cuda, device, batch_size)
    elif f_mnist == 'kmnist':
        return get_kmnist_loaders(use_cuda, device, batch_size)
    elif f_mnist == 'emnist':
        return get_emnist_loaders(use_cuda, device, batch_size)
    elif f_mnist == 'cifar':
        return get_cifar_loaders(use_cuda, device, batch_size)

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

def get_other_mnist_dataset_name(train_on_fmnist):
    if isinstance(train_on_fmnist, bool):
        other_mnist_dataset = (not train_on_fmnist)
    elif train_on_fmnist == 'kmnist':
        other_mnist_dataset = 'emnist'
    elif train_on_fmnist == 'emnist':
        other_mnist_dataset = 'kmnist'
    return other_mnist_dataset

def run_torch(n_epochs=100, batch_size=256, dropout_inference=0.1, dropout_spn=0.0, class_label=1, train_on_fmnist=False, lmbda=0.0, eval_single_digit=True, toy_setting=False):
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

    with open(d + 'trainig_details.out', 'a') as writer:
        writer.write("n epochs: {}, batch size: {}".format(n_epochs, batch_size))

    train_loader, test_loader = get_mnist_loaders(use_cuda, batch_size=batch_size, device=device, f_mnist=train_on_fmnist)
    n_features = torch.prod(torch.Tensor(train_loader.dataset.data.shape[1:])).int().item()

    #model = make_spn(S=10, I=10, D=3, R=5, device=dev, dropout=0.0)
    # model = make_spn(S=10, I=10, D=3, R=5, device=dev, dropout=dropout_spn)
    # model = make_spn(S=2, I=4, D=2, R=1, device=dev, dropout=dropout_spn)
    # model = make_spn(S=2, I=20, D=2, R=1, device=dev, dropout=dropout_spn)
    if toy_setting:
        model = make_spn(S=2, I=4, D=2, R=1, device=dev, dropout=dropout_spn, F=n_features)
        # model = make_spn(S=20, I=20, D=5, R=5, device=dev, dropout=dropout_spn)
    elif train_on_fmnist == 'cifar':
        model = make_spn(S=20, I=20, D=5, R=10, device=dev, dropout=dropout_spn, F=n_features)
    else:
        model = make_spn(S=20, I=20, D=5, R=5, device=dev, dropout=dropout_spn, F=n_features)

    model.train()
    print(model)
    print("Number of pytorch parameters: ", count_params(model))

    # Define optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    log_interval = 100

    lmbda = lmbda

    training_string = ""



    for epoch in range(n_epochs):
    # TODO change
        #if epoch > 20:
        #    # lmbda = lmbda_0 + lmbda_rel * (0.95 ** (epoch - 20))
        #    lmbda = 0.5
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
            # scheduler.step()

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
            if not train_on_fmnist == 'cifar':
                set_seed(0)
                # samples = model.sample(n=25)
                samples = model.sample(class_index=list(range(10)) * 5)
                save_samples(samples, iteration=epoch)

        t_delta = time_delta_now(t_start)
        print("Train Epoch {} took {}".format(epoch, t_delta))
        if epoch % 5 == 4:
            print("Evaluating model...")
            lls_train, class_probs_train, _, _ = evaluate_model(model, device, train_loader, "Train", output_dir=d)
            test_lls, class_probs_test, _, _ = evaluate_model(model, device, test_loader, "Test", output_dir=d)
            print("Train class entropy: {} Test class entropy: {}".format(entropy(class_probs_train, axis=1).sum(), entropy(class_probs_test, axis=1).sum()))

    d_samples = d + "samples/"
    d_results = d + "results/"
    d_model = d + "model/"
    ensure_dir(d_samples)
    ensure_dir(d_results)
    ensure_dir(d_model)

    print('Saving model...')
    torch.save(model.state_dict(), d + 'model/model.pt')

    train_lls, class_probs_train, train_lls_sup, train_lls_unsup = evaluate_model(model, device, train_loader, "Train", output_dir=d)
    test_lls, class_probs_test, test_lls_sup, test_lls_unsup = evaluate_model(model, device, test_loader, "Test", output_dir=d)
    # evaluate_model_rotated_one(model, device, test_loader, "Test DROP rotated one", dropout_inference=dropout_inference, output_dir=d, class_label=class_label)
    #
    #

    if isinstance(train_on_fmnist, str) and train_on_fmnist == 'cifar':
        corruptions = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 'frost', 'gaussian_blur', 'gaussian_noise',
                          'glass_blur', 'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate', 'saturate', 'shot_noise', 'snow',
                          'spatter', 'speckle_noise', 'zoom_blur']
        # drop_class_probs_c0, class_probs_c0 = evaluate_model_corrupted_cifar(model, device, test_loader, "Test DROP corrupted CIFAR 0", dropout_inference=dropout_inference, output_dir=d, corruption=0) #TODO add batch_size and drop iters
        # drop_class_probs_c1_l1, class_probs_c1_l1 = evaluate_model_corrupted_cifar(model, device, test_loader, "Test DROP corrupted CIFAR 1 L1", dropout_inference=dropout_inference, output_dir=d, corruption='gaussian_noise', corruption_level=1) #TODO add batch_size and drop iters
        # drop_class_probs_c1_l2, class_probs_c1_l2 = evaluate_model_corrupted_cifar(model, device, test_loader, "Test DROP corrupted CIFAR 1 L2", dropout_inference=dropout_inference, output_dir=d, corruption='gaussian_noise', corruption_level=2) #TODO add batch_size and drop iters
        # drop_class_probs_c1_l3, class_probs_c1_l3 = evaluate_model_corrupted_cifar(model, device, test_loader, "Test DROP corrupted CIFAR 1 L3", dropout_inference=dropout_inference, output_dir=d, corruption='gaussian_noise', corruption_level=3) #TODO add batch_size and drop iters
        # drop_class_probs_c1_l4, class_probs_c1_l4 = evaluate_model_corrupted_cifar(model, device, test_loader, "Test DROP corrupted CIFAR 1 L4", dropout_inference=dropout_inference, output_dir=d, corruption='gaussian_noise', corruption_level=4) #TODO add batch_size and drop iters
        # drop_class_probs_c1_l5, class_probs_c1_l5 = evaluate_model_corrupted_cifar(model, device, test_loader, "Test DROP corrupted CIFAR 1 L5", dropout_inference=dropout_inference, output_dir=d, corruption='gaussian_noise', corruption_level=5) #TODO add batch_size and drop iters
        # drop_class_probs_c2_l1, class_probs_c2_l1 = evaluate_model_corrupted_cifar(model, device, test_loader, "Test DROP corrupted CIFAR 2 L1", dropout_inference=dropout_inference, output_dir=d, corruption='shot_noise', corruption_level=1) #TODO add batch_size and drop iters
        # drop_class_probs_c2_l2, class_probs_c2_l2 = evaluate_model_corrupted_cifar(model, device, test_loader, "Test DROP corrupted CIFAR 2 L2", dropout_inference=dropout_inference, output_dir=d, corruption='shot_noise', corruption_level=2) #TODO add batch_size and drop iters

        results_dict = {}
        for corruption in corruptions:
            for cl in range(5):
                cl += 1
                print("Corruption {} Level {}".format(corruption, cl))
                results_dict['c_{}_l{}'.format(corruption, cl)] = evaluate_model_corrupted_cifar(model, device, test_loader, "Test DROP corrupted CIFAR C {} L {}".format(corruption, cl),
                                                                                                dropout_inference=dropout_inference, output_dir=d, corruption=corruption, corruption_level=cl) #TODO add batch_size and drop iters
                np.save(d_results + 'dropout_class_probs_c_{}_l{}'.format(corruption, cl), results_dict['c_{}_l{}'.format(corruption, cl)][0].cpu().detach().numpy())
                np.save(d_results + 'class_probs_c_{}_l{}'.format(corruption, cl), results_dict['c_{}_l{}'.format(corruption, cl)][1].cpu().detach().numpy())


        # eval on corrupted cifar
        sys.exit("ciao.")

    if isinstance(train_on_fmnist, bool) and train_on_fmnist == False:
        drop_class_probs_c1, class_probs_c1 = evaluate_model_corrupted_digits(model, device, test_loader, "Test DROP corrupted 1", dropout_inference=dropout_inference, output_dir=d, corruption=corruptions.shot_noise) #TODO add batch_size and drop iters
        drop_class_probs_c2, class_probs_c2 = evaluate_model_corrupted_digits(model, device, test_loader, "Test DROP corrupted 2", dropout_inference=dropout_inference, output_dir=d, corruption=corruptions.impulse_noise) #TODO add batch_size and drop iters
        drop_class_probs_c3, class_probs_c3 = evaluate_model_corrupted_digits(model, device, test_loader, "Test DROP corrupted 3", dropout_inference=dropout_inference, output_dir=d, corruption=corruptions.glass_blur) #TODO add batch_size and drop iters
        drop_class_probs_c4, class_probs_c4 = evaluate_model_corrupted_digits(model, device, test_loader, "Test DROP corrupted 4", dropout_inference=dropout_inference, output_dir=d, corruption=corruptions.motion_blur) #TODO add batch_size and drop iters
        drop_class_probs_c5, class_probs_c5 = evaluate_model_corrupted_digits(model, device, test_loader, "Test DROP corrupted 5", dropout_inference=dropout_inference, output_dir=d, corruption=corruptions.shear) #TODO add batch_size and drop iters
        drop_class_probs_c6, class_probs_c6 = evaluate_model_corrupted_digits(model, device, test_loader, "Test DROP corrupted 6", dropout_inference=dropout_inference, output_dir=d, corruption=corruptions.scale) #TODO add batch_size and drop iters
        drop_class_probs_c7, class_probs_c7 = evaluate_model_corrupted_digits(model, device, test_loader, "Test DROP corrupted 7", dropout_inference=dropout_inference, output_dir=d, corruption=corruptions.rotate) #TODO add batch_size and drop iters
        drop_class_probs_c8, class_probs_c8 = evaluate_model_corrupted_digits(model, device, test_loader, "Test DROP corrupted 8", dropout_inference=dropout_inference, output_dir=d, corruption=corruptions.brightness) #TODO add batch_size and drop iters
        drop_class_probs_c9, class_probs_c9 = evaluate_model_corrupted_digits(model, device, test_loader, "Test DROP corrupted 9", dropout_inference=dropout_inference, output_dir=d, corruption=corruptions.translate) #TODO add batch_size and drop iters
        drop_class_probs_c10, class_probs_c10 = evaluate_model_corrupted_digits(model, device, test_loader, "Test DROP corrupted 10", dropout_inference=dropout_inference, output_dir=d, corruption=corruptions.stripe) #TODO add batch_size and drop iters
        drop_class_probs_c11, class_probs_c11 = evaluate_model_corrupted_digits(model, device, test_loader, "Test DROP corrupted 11", dropout_inference=dropout_inference, output_dir=d, corruption=corruptions.fog) #TODO add batch_size and drop iters
        drop_class_probs_c12, class_probs_c12 = evaluate_model_corrupted_digits(model, device, test_loader, "Test DROP corrupted 12", dropout_inference=dropout_inference, output_dir=d, corruption=corruptions.spatter) #TODO add batch_size and drop iters
        drop_class_probs_c13, class_probs_c13 = evaluate_model_corrupted_digits(model, device, test_loader, "Test DROP corrupted 13", dropout_inference=dropout_inference, output_dir=d, corruption=corruptions.dotted_line) #TODO add batch_size and drop iters
        drop_class_probs_c14, class_probs_c14 = evaluate_model_corrupted_digits(model, device, test_loader, "Test DROP corrupted 14", dropout_inference=dropout_inference, output_dir=d, corruption=corruptions.zigzag) #TODO add batch_size and drop iters
        drop_class_probs_c15, class_probs_c15 = evaluate_model_corrupted_digits(model, device, test_loader, "Test DROP corrupted 15", dropout_inference=dropout_inference, output_dir=d, corruption=corruptions.canny_edges) #TODO add batch_size and drop iters

        # save class probs
        np.save(d_results + 'dropout_class_probs_c1', drop_class_probs_c1.cpu().detach().numpy())
        np.save(d_results + 'dropout_class_probs_c2', drop_class_probs_c2.cpu().detach().numpy())
        np.save(d_results + 'dropout_class_probs_c3', drop_class_probs_c3.cpu().detach().numpy())
        np.save(d_results + 'dropout_class_probs_c4', drop_class_probs_c4.cpu().detach().numpy())
        np.save(d_results + 'dropout_class_probs_c5', drop_class_probs_c5.cpu().detach().numpy())
        np.save(d_results + 'dropout_class_probs_c6', drop_class_probs_c6.cpu().detach().numpy())
        np.save(d_results + 'dropout_class_probs_c7', drop_class_probs_c7.cpu().detach().numpy())
        np.save(d_results + 'dropout_class_probs_c8', drop_class_probs_c8.cpu().detach().numpy())
        np.save(d_results + 'dropout_class_probs_c9', drop_class_probs_c9.cpu().detach().numpy())
        np.save(d_results + 'dropout_class_probs_c10', drop_class_probs_c10.cpu().detach().numpy())
        np.save(d_results + 'dropout_class_probs_c11', drop_class_probs_c11.cpu().detach().numpy())
        np.save(d_results + 'dropout_class_probs_c12', drop_class_probs_c12.cpu().detach().numpy())
        np.save(d_results + 'dropout_class_probs_c13', drop_class_probs_c13.cpu().detach().numpy())
        np.save(d_results + 'dropout_class_probs_c14', drop_class_probs_c14.cpu().detach().numpy())
        np.save(d_results + 'dropout_class_probs_c15', drop_class_probs_c15.cpu().detach().numpy())

        np.save(d_results + 'class_probs_c1', class_probs_c1.cpu().detach().numpy())
        np.save(d_results + 'class_probs_c2', class_probs_c2.cpu().detach().numpy())
        np.save(d_results + 'class_probs_c3', class_probs_c3.cpu().detach().numpy())
        np.save(d_results + 'class_probs_c4', class_probs_c4.cpu().detach().numpy())
        np.save(d_results + 'class_probs_c5', class_probs_c5.cpu().detach().numpy())
        np.save(d_results + 'class_probs_c6', class_probs_c6.cpu().detach().numpy())
        np.save(d_results + 'class_probs_c7', class_probs_c7.cpu().detach().numpy())
        np.save(d_results + 'class_probs_c8', class_probs_c8.cpu().detach().numpy())
        np.save(d_results + 'class_probs_c9', class_probs_c9.cpu().detach().numpy())
        np.save(d_results + 'class_probs_c10', class_probs_c10.cpu().detach().numpy())
        np.save(d_results + 'class_probs_c11', class_probs_c11.cpu().detach().numpy())
        np.save(d_results + 'class_probs_c12', class_probs_c12.cpu().detach().numpy())
        np.save(d_results + 'class_probs_c13', class_probs_c13.cpu().detach().numpy())
        np.save(d_results + 'class_probs_c14', class_probs_c14.cpu().detach().numpy())
        np.save(d_results + 'class_probs_c15', class_probs_c15.cpu().detach().numpy())

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

        sys.exit()

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



    other_mnist_train_lls, other_mnist_test_lls, other_class_probs_train, other_class_probs_test, other_mnist_train_lls_sup, other_mnist_train_lls_unsup, other_mnist_test_lls_sup, other_mnist_test_lls_unsup = get_other_mnist_lls(model, device, d, use_cuda, batch_size, trained_on_fmnist=train_on_fmnist)
    print("OTHER Train class entropy: {} OTHER Test class entropy: {}".format(entropy(other_class_probs_train, axis=1).sum(), entropy(other_class_probs_test, axis=1).sum()))

    lls_dict = {"mnist_train":train_lls, "mnist_test":test_lls, "other_mnist_train":other_mnist_train_lls, "other_mnist_test":other_mnist_test_lls}
    head_lls_dict_sup = {"mnist_train":train_lls_sup, "mnist_test":test_lls_sup, "other_mnist_train":other_mnist_train_lls_sup, "other_mnist_test":other_mnist_test_lls_sup}
    head_lls_dict_unsup = {"mnist_train":train_lls_unsup, "mnist_test":test_lls_unsup, "other_mnist_train":other_mnist_train_lls_unsup, "other_mnist_test":other_mnist_test_lls_unsup}
    class_probs_dict = {"mnist_train":class_probs_train.max(axis=1), "mnist_test":class_probs_test.max(axis=1), "other_mnist_train":other_class_probs_train.max(axis=1), "other_mnist_test":other_class_probs_test.max(axis=1)}


    np.save(d_results + 'class_probs_in_domain_train', class_probs_train)
    np.save(d_results + 'class_probs_in_domain_test', class_probs_test)
    np.save(d_results + 'class_probs_ood_train', other_class_probs_train)
    np.save(d_results + 'class_probs_ood_test', other_class_probs_test)

    plot_histograms(lls_dict, filename='histograms_lls', title="Data LLs", path=d_samples, trained_on_fmnist=train_on_fmnist)
    plot_histograms(head_lls_dict_sup, filename='histograms_lls_sup', title="Data SUP LLs", path=d_samples, trained_on_fmnist=train_on_fmnist)
    plot_histograms(head_lls_dict_unsup, filename='histograms_lls_unsup', title="Data UNSUP LLs", path=d_samples, trained_on_fmnist=train_on_fmnist)
    plot_histograms(class_probs_dict, filename='class_probs_histograms', title="Class Probs", path=d_samples, trained_on_fmnist=train_on_fmnist, y_lim=30000)

    train_lls_dropout, class_probs_train_dropout, train_lls_sup_drop, train_lls_unsup_drop = evaluate_model_dropout(model, device, train_loader, "Train DROP", dropout_inference=dropout_inference, output_dir=d)
    test_lls_dropout, class_probs_test_dropout, test_lls_sup_drop, test_lls_unsup_drop = evaluate_model_dropout(model, device, test_loader, "Test DROP", dropout_inference=dropout_inference, output_dir=d)
    print("DROP Train class entropy: {} DROP Test class entropy: {}".format(entropy(class_probs_train_dropout, axis=1).sum(), entropy(class_probs_test_dropout, axis=1).sum()))

    other_train_loader, other_test_loader = get_mnist_loaders(use_cuda=use_cuda, device=device, batch_size=batch_size, f_mnist=get_other_mnist_dataset_name(train_on_fmnist))
    other_train_lls_dropout, other_class_probs_train_dropout, other_train_lls_sup_drop, other_train_lls_unsup_drop = evaluate_model_dropout(model, device, other_train_loader, "Other Train DROP", dropout_inference=dropout_inference, output_dir=d)
    other_test_lls_dropout, other_class_probs_test_dropout, other_test_lls_sup_drop, other_test_lls_unsup_drop = evaluate_model_dropout(model, device, other_test_loader, "Other Test DROP", dropout_inference=dropout_inference, output_dir=d)
    print("DROP OTHER Train class entropy: {} DROP OTHER Test class entropy: {}".format(entropy(other_class_probs_train_dropout, axis=1).sum(), entropy(other_class_probs_test_dropout, axis=1).sum()))

    dropout_lls_dict = {"mnist_train":train_lls_dropout, "mnist_test":test_lls_dropout, "other_mnist_train":other_train_lls_dropout, "other_mnist_test":other_test_lls_dropout}
    dropout_head_lls_dict_sup = {"mnist_train":train_lls_sup_drop, "mnist_test":test_lls_sup_drop, "other_mnist_train":other_train_lls_sup_drop, "other_mnist_test":other_test_lls_sup_drop}
    dropout_head_lls_dict_unsup = {"mnist_train":train_lls_unsup_drop, "mnist_test":test_lls_unsup_drop, "other_mnist_train":other_train_lls_unsup_drop, "other_mnist_test":other_test_lls_unsup_drop}
    dropout_class_probs_dict = {"mnist_train":class_probs_train_dropout.max(axis=1), "mnist_test":class_probs_test_dropout.max(axis=1), "other_mnist_train":other_class_probs_train_dropout.max(axis=1), "other_mnist_test":other_class_probs_test_dropout.max(axis=1)}

    np.save(d_results + 'class_probs_in_domain_train_dropout', class_probs_train_dropout)
    np.save(d_results + 'class_probs_in_domain_test_dropout', class_probs_test_dropout)
    np.save(d_results + 'class_probs_ood_train_dropout', other_class_probs_train_dropout)
    np.save(d_results + 'class_probs_ood_test_dropout', other_class_probs_test_dropout)

    plot_histograms(dropout_lls_dict, filename='dropout_histograms_lls', title="DROPOUT Data LLs", path=d_samples, trained_on_fmnist=train_on_fmnist)
    plot_histograms(dropout_head_lls_dict_sup, filename='dropout_histograms_lls_sup', title="DROPOUT Data SUP LLs", path=d_samples, trained_on_fmnist=train_on_fmnist)
    plot_histograms(dropout_head_lls_dict_unsup, filename='dropout_histograms_lls_unsup', title="DROPOUT Data UNSUP LLs", path=d_samples, trained_on_fmnist=train_on_fmnist)
    plot_histograms(dropout_class_probs_dict, filename='dropout_class_probs_histograms', title="DROPOUT Class Probs", path=d_samples, trained_on_fmnist=train_on_fmnist, y_lim=30000)
    # evaluate_model_rotated_one(model, device, test_loader, "Test DROP rotated one", dropout_inference=dropout_inference, output_dir=d, class_label=class_label)
    # TODO
    # save model
    # eval the best one? on a valid set?

def get_other_mnist_lls(model, device, output_dir='./', use_cuda=False, batch_size=100, trained_on_fmnist=False):
    train_loader, test_loader = get_mnist_loaders(use_cuda=use_cuda, device=device, batch_size=batch_size, f_mnist=get_other_mnist_dataset_name(trained_on_fmnist))
    log_string = "MNIST" if trained_on_fmnist else "F-MNIST"
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
    class_probs = torch.zeros((loader.dataset.data.shape[0],10)).to(device)
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
    loss_nll /= len(loader.dataset) + torch.prod(torch.Tensor(loader.dataset.data.shape[1:]))
    accuracy = 100.0 * correct / len(loader.dataset)

    output_string = "{} set: Average loss_ce: {:.4f} Average loss_nll: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            tag, loss_ce, loss_nll, correct, len(loader.dataset), accuracy
    )
    print(output_string)
    with open(output_dir + 'trainig.out', 'a') as writer:
        writer.write(output_string + "\n")
    assert len(data_ll) == loader.dataset.data.shape[0]
    assert len(data_ll_super) == loader.dataset.data.shape[0]
    assert len(data_ll_unsup) == loader.dataset.data.shape[0]
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
    class_probs = torch.zeros((loader.dataset.data.shape[0], 10, n_dropout_iters)).to(device)
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
    loss_nll /= len(loader.dataset) + torch.prod(torch.Tensor(loader.dataset.data.shape[1:]))

    accuracy = 100.0 * correct / len(loader.dataset)

    # class_probs /= n_dropout_iters
    class_probs = class_probs.mean(dim=2)
    # drop_preds = class_probs.argmax(dim=1)
    # drop_corrects = (drop_preds == loader.dataset.targets.to(device)).sum().item() #TODO double check here
    print("drop corrects: {}".format(drop_corrects/loader.dataset.data.shape[0]))
    # breakpoint()

    output_string = "{} set: Average loss_ce: {:.4f} \u00B1{:.4f} Average loss_nll: {:.4f} \u00B1{:.4f}, Accuracy: {:.4f} \u00B1{:.4f}/{} ({:.0f}% \u00B1{:.4f})".format(
            tag, np.mean(loss_ce), np.std(loss_ce), np.mean(loss_nll), np.std(loss_nll), np.mean(correct), np.std(correct), len(loader.dataset), np.mean(accuracy), np.std(accuracy))
    print(output_string)

    with open(output_dir + 'trainig.out', 'a') as writer:
        writer.write(output_string + "\n")
    assert len(data_ll) == loader.dataset.data.shape[0]
    return data_ll, class_probs.detach().cpu().numpy(), data_ll_super, data_ll_unsup

def evaluate_model_rotated_one(model: torch.nn.Module, device, loader, tag, dropout_inference=0.01, n_dropout_iters=100, output_dir="", class_label=1) -> float:

    d = output_dir
    d_samples = d + "samples/"
    d_model = d + "model/"
    ensure_dir(d)
    ensure_dir(d_samples)
    ensure_dir(d_model)

    class_label = class_label

    model.eval()
    mean = 0.1307
    std = 0.3081
    transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean), (std))])

    dataset = datasets.MNIST(root='../data', train=False, download=True, transform=transformer)

    targets = torch.tensor(dataset.targets)
    target_idx = (targets == class_label).nonzero()
    sampler = torch.utils.data.sampler.SubsetRandomSampler(target_idx.reshape((-1,)))

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, sampler=sampler)
    n_correct = 0
    n_samples = 0

    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            data = data.view(data.shape[0], -1)
            n_samples += data.shape[0]
            output = model(data, test_dropout=True, dropout_inference=dropout_inference)
            n_correct += (output.argmax(dim=1) == class_label).sum()
            # breakpoint()
        # print("N of correct predictions of digit 1 test samples: {}/{} ({}%)".format(n_correct, n_samples, (n_correct / n_samples)))

        with open(d_model + 'log.out', 'a') as writer:
            writer.write("dropout inference: {}\n".format(dropout_inference))
            writer.write("dropout iters: {}\n".format(n_dropout_iters))
            writer.write("N of correct predictions of digit {} test samples: {}/{} ({}%)\n".format(class_label, n_correct, n_samples, (n_correct / n_samples)))


        one_digit_samples = dataset.data[target_idx.reshape((-1,))].view(-1,1,28,28)
        special_one = one_digit_samples[5] #TODO it was 1

        rotated_one = one_digit_samples[:10].to(device)
        # rotated_one = torch.zeros(11, 1, 28, 28).to(device)
        # # breakpoint()

        # rotated_one[0] = special_one
        # rotated_one[1] = torchvision.transforms.functional.rotate(special_one, 10)
        # rotated_one[2] = torchvision.transforms.functional.rotate(special_one, 20)
        # rotated_one[3] = torchvision.transforms.functional.rotate(special_one, 30)
        # rotated_one[4] = torchvision.transforms.functional.rotate(special_one, 40)
        # rotated_one[5] = torchvision.transforms.functional.rotate(special_one, 50)
        # rotated_one[6] = torchvision.transforms.functional.rotate(special_one, 60)
        # rotated_one[7] = torchvision.transforms.functional.rotate(special_one, 70)
        # rotated_one[8] = torchvision.transforms.functional.rotate(special_one, 80)
        # rotated_one[9] = torchvision.transforms.functional.rotate(special_one, 90)
        # rotated_one[10] = torchvision.transforms.functional.rotate(special_one, 90)
        # # breakpoint()

        transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean), (std))])
        custom_dataset = CustomTensorDataset(tensors=[rotated_one.reshape(10, 28, 28).clone().cpu(), torch.tensor([class_label]*rotated_one.shape[0])], transform=transformer)
        custom_data_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=10, shuffle=False)

        dropout_output = torch.zeros(10,10,n_dropout_iters).to(device)
        dropout_class_probs = torch.zeros(10,10,n_dropout_iters).to(device)
        dropout_softmax_output = torch.zeros(10,10,n_dropout_iters).to(device)

        for data, target in custom_data_loader:
            data = data.to(device)

            data[0] = data[5].reshape(28,28)
            data[1] = torchvision.transforms.functional.rotate(data[0].reshape(-1,1,28,28), 10, fill=-mean/std).reshape(28,28)
            data[2] = torchvision.transforms.functional.rotate(data[0].reshape(-1,1,28,28), 20, fill=-mean/std).reshape(28,28)
            data[3] = torchvision.transforms.functional.rotate(data[0].reshape(-1,1,28,28), 30, fill=-mean/std).reshape(28,28)
            data[4] = torchvision.transforms.functional.rotate(data[0].reshape(-1,1,28,28), 40, fill=-mean/std).reshape(28,28)
            data[5] = torchvision.transforms.functional.rotate(data[0].reshape(-1,1,28,28), 50, fill=-mean/std).reshape(28,28)
            data[6] = torchvision.transforms.functional.rotate(data[0].reshape(-1,1,28,28), 60, fill=-mean/std).reshape(28,28)
            data[7] = torchvision.transforms.functional.rotate(data[0].reshape(-1,1,28,28), 70, fill=-mean/std).reshape(28,28)
            data[8] = torchvision.transforms.functional.rotate(data[0].reshape(-1,1,28,28), 80, fill=-mean/std).reshape(28,28)
            data[9] = torchvision.transforms.functional.rotate(data[0].reshape(-1,1,28,28), 90, fill=-mean/std).reshape(28,28)
            # data[10] = torchvision.transforms.functional.rotate(data[0].reshape(-1,1,28,28), 100).reshape(28,28)

            data = data.view(data.shape[0], -1)


            # breakpoint()
            output = model(data, test_dropout=False, dropout_inference=0.0)
            # breakpoint()
            # plot_samples(rotated_one, path=os.path.join(d_samples, f"rotated-one_TEST.png"))
            # print("From root heads: ", output.argmax(dim=1))
            # print("From softmax: ", torch.nn.functional.softmax(output, dim=1).argmax(dim=1)) # 0.01 is prolly more reasonable
            # print((output - torch.logsumexp(output, dim=1, keepdims=True)).exp())
            # print("test sum up to 1 -------------------")
            class_probs = (output - torch.logsumexp(output, dim=1, keepdims=True)).exp()
            # print("From class probs: ", class_probs.argmax(dim=1))
            assert torch.all(torch.isclose(class_probs.sum(-1), torch.tensor((1.)), atol=1e-04)), class_probs.sum(-1)

            for i in range(n_dropout_iters):
                dropout_output[:,:,i] = model(data, test_dropout=True, dropout_inference=dropout_inference)
                dropout_class_probs[:,:,i] = (dropout_output[:,:,i] - torch.logsumexp(dropout_output[:,:,i], dim=1, keepdims=True)).exp()
                dropout_softmax_output[:,:,i] = torch.nn.functional.softmax(model(data, test_dropout=True, dropout_inference=dropout_inference), dim=1) # 0.01 is prolly more reasonable
                assert torch.all(torch.isclose(dropout_class_probs[:,:,i].sum(-1), torch.tensor((1.)), atol=1e-04)), dropout_softmax_output[:,:,i].sum(-1)
                # breakpoint()
                # print(dropout_class_probs[:,:,i].sum(-1))


        torch.save(model.state_dict(), d_model + 'model.pt')

        np.savetxt(d_model + 'mean_dropout_probs.out', torch.mean(dropout_class_probs, 2).cpu().detach().numpy(), delimiter=',')
        np.savetxt(d_model + 'std_dropout_probs.out', torch.std(dropout_class_probs, 2).cpu().detach().numpy(), delimiter=',')

        with open(d_model + 'log.out', 'a') as writer:
            writer.write(np.array2string((torch.argmax(torch.mean(dropout_class_probs, 2), dim=1) == 1).sum().cpu().detach().numpy()) + "\n")
            writer.write(str(model.config) + "\n")
            writer.write(str(model) + "\n")
            writer.write("Number of pytorch parameters: {}\n".format(count_params(model)))
            writer.write("From root heads: {}\n".format(output.argmax(dim=1)))
            writer.write("From softmax: {}\n".format(torch.nn.functional.softmax(output, dim=1).argmax(dim=1))) # 0.01 is prolly more reasonable
            writer.write("From class probs: {}\n".format(class_probs.argmax(dim=1)))

        plot_boxplot(data=dropout_class_probs.cpu().numpy(), filename='boxplot_class_probs', title="Class probs - dropout {} ".format(dropout_inference), path=d_samples)
        plot_boxplot(data=dropout_softmax_output.cpu().numpy(), filename='boxplot_softmax', title="Softmax - dropout {} ".format(dropout_inference), path=d_samples)

        inv_normalize = transforms.Normalize((-0.1307/0.3081,), (1/0.3081,))
        result_samples = inv_normalize(one_digit_samples[:300].float())
        # rotated_one = inv_normalize(rotated_one.float())
        rotated_one = inv_normalize(data.reshape(-1,1,28,28))
        plot_samples(result_samples, path=os.path.join(d_samples, f"some-test-samples.png"))
        plot_samples(rotated_one, path=os.path.join(d_samples, f"rotated-one.png"))


def evaluate_model_corrupted_digits(model: torch.nn.Module, device, loader, tag, dropout_inference=0.01, n_dropout_iters=100, output_dir="", corruption=corruptions.glass_blur, class_label=None) -> float:

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

    # apply corruptiona
    corrupted_images = np.empty((len(dataset), 28, 28), dtype=np.uint8)
    for i in range(len(dataset)):
        corrupted_images[i] = round_and_astype(np.array(corruption(dataset[i][0])))

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

        # for data, target in data_loader:
        #     data[0] = data[5].reshape(28,28)
        #     data[1] = torchvision.transforms.functional.rotate(data[0].reshape(-1,1,28,28), 10, fill=-mean/std).reshape(28,28)
        #     data[2] = torchvision.transforms.functional.rotate(data[0].reshape(-1,1,28,28), 20, fill=-mean/std).reshape(28,28)
        #     data[3] = torchvision.transforms.functional.rotate(data[0].reshape(-1,1,28,28), 30, fill=-mean/std).reshape(28,28)
        #     data[4] = torchvision.transforms.functional.rotate(data[0].reshape(-1,1,28,28), 40, fill=-mean/std).reshape(28,28)
        #     data[5] = torchvision.transforms.functional.rotate(data[0].reshape(-1,1,28,28), 50, fill=-mean/std).reshape(28,28)
        #     data[6] = torchvision.transforms.functional.rotate(data[0].reshape(-1,1,28,28), 60, fill=-mean/std).reshape(28,28)
        #     data[7] = torchvision.transforms.functional.rotate(data[0].reshape(-1,1,28,28), 70, fill=-mean/std).reshape(28,28)
        #     data[8] = torchvision.transforms.functional.rotate(data[0].reshape(-1,1,28,28), 80, fill=-mean/std).reshape(28,28)
        #     data[9] = torchvision.transforms.functional.rotate(data[0].reshape(-1,1,28,28), 90, fill=-mean/std).reshape(28,28)

        #     data = data.view(data.shape[0], -1)

        #     output = model(data, test_dropout=False, dropout_inference=0.0)
        #     class_probs = (output - torch.logsumexp(output, dim=1, keepdims=True)).exp()
        #     assert torch.all(torch.isclose(class_probs.sum(-1), torch.tensor((1.)), atol=1e-04)), class_probs.sum(-1)

        #     for i in range(n_dropout_iters):
        #         dropout_output[:,:,i] = model(data, test_dropout=True, dropout_inference=dropout_inference)
        #         dropout_class_probs[:,:,i] = (dropout_output[:,:,i] - torch.logsumexp(dropout_output[:,:,i], dim=1, keepdims=True)).exp()
        #         dropout_softmax_output[:,:,i] = torch.nn.functional.softmax(model(data, test_dropout=True, dropout_inference=dropout_inference), dim=1) # 0.01 is prolly more reasonable
        #         assert torch.all(torch.isclose(dropout_class_probs[:,:,i].sum(-1), torch.tensor((1.)), atol=1e-04)), dropout_softmax_output[:,:,i].sum(-1)

        # torch.save(model.state_dict(), d_model + 'model.pt')

        # np.savetxt(d_model + 'mean_dropout_probs.out', torch.mean(dropout_class_probs, 2).cpu().detach().numpy(), delimiter=',')
        # np.savetxt(d_model + 'std_dropout_probs.out', torch.std(dropout_class_probs, 2).cpu().detach().numpy(), delimiter=',')

        # with open(d_model + 'log.out', 'a') as writer:
        #     writer.write(np.array2string((torch.argmax(torch.mean(dropout_class_probs, 2), dim=1) == 1).sum().cpu().detach().numpy()) + "\n")
        #     writer.write(str(model.config) + "\n")
        #     writer.write(str(model) + "\n")
        #     writer.write("Number of pytorch parameters: {}\n".format(count_params(model)))
        #     writer.write("From root heads: {}\n".format(output.argmax(dim=1)))
        #     writer.write("From softmax: {}\n".format(torch.nn.functional.softmax(output, dim=1).argmax(dim=1))) # 0.01 is prolly more reasonable
        #     writer.write("From class probs: {}\n".format(class_probs.argmax(dim=1)))

        # plot_boxplot(data=dropout_class_probs.cpu().numpy(), filename='boxplot_class_probs', title="Class probs - dropout {} ".format(dropout_inference), path=d_samples)
        # plot_boxplot(data=dropout_softmax_output.cpu().numpy(), filename='boxplot_softmax', title="Softmax - dropout {} ".format(dropout_inference), path=d_samples)

        # inv_normalize = transforms.Normalize((-0.1307/0.3081,), (1/0.3081,))
        # result_samples = inv_normalize(one_digit_samples[:300].float())

        # rotated_one = inv_normalize(data.reshape(-1,1,28,28))
        # plot_samples(result_samples, path=os.path.join(d_samples, f"some-test-samples.png"))
        # plot_samples(rotated_one, path=os.path.join(d_samples, f"rotated-one.png"))


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

    run_torch(200, 200, dropout_inference=0.2, dropout_spn=0.2, class_label=1, train_on_fmnist='cifar', lmbda=1.0, toy_setting=False)
    # run_torch(3, 200, dropout_inference=0.2, dropout_spn=0.2, class_label=1, train_on_fmnist=False, lmbda=1.0, toy_setting=False)
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
