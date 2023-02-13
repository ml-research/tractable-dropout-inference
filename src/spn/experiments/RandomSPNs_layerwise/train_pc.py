import os
import random
import sys
import time

import numpy as np
import torch
import torchvision
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from imagecorruptions import get_corruption_names

from spn.experiments.RandomSPNs_layerwise.distributions import RatNormal
from spn.experiments.RandomSPNs_layerwise.rat_spn import RatSpn, RatSpnConfig

from torch.utils.data import Dataset, ConcatDataset
from PIL import Image
import datetime
import scipy

from fig_rotated_mnist import plot_figure


class CustomTensorDataset(Dataset):
    """
    TensorDataset with support of transforms.
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


def time_delta_now(t_start: float) -> str:
    """
    Convert a timestamp into a human-readable timestring.
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
    Count the number of parameters in a model

    Args:
        model (torch.nn.Module): PyTorch model.

    Returns:
        int: Number of learnable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_mnist_loaders(use_cuda, batch_size):
    """
    Get the MNIST pytorch data loader.

    Args:
        use_cuda: Use cuda flag.
        batch_size: The size of the batch.
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


def get_lsun_loaders(use_cuda, batch_size):
    """
    Get the LSUN pytorch data loader.

    Args:
        use_cuda: Use cuda flag.
        batch_size: The size of the batch.

    """
    kwargs = {"num_workers": 8, "pin_memory": True} if use_cuda else {}

    lsun_classes = ['church_outdoor']

    # mean and std for church_outdoor (train set)
    lsun_mean = (0.4846, 0.5057, 0.5166)
    lsun_std = (0.0356, 0.0352, 0.0414)

    lsun_transformer = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(),
                                           transforms.Normalize(mean=lsun_mean, std=lsun_std),
                                           ])

    lsun_train_loader = torch.utils.data.DataLoader(
        datasets.LSUN(root=LSUN_DIR, classes=[lc + '_train' for lc in lsun_classes],
                      transform=lsun_transformer),
        batch_size=batch_size,
        shuffle=True,
        **kwargs
    )

    # mean and standard deviation computed on the valid set
    lsun_valid_mean = (0.5071, 0.4699, 0.4325)
    lsun_valid_std = (0.0355, 0.0356, 0.0382)

    lsun_transformer_valid = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(),
                                                 transforms.Normalize(mean=lsun_valid_mean, std=lsun_valid_std),
                                                 ])

    # use valid set instead of the test set, since it has a more reasonable size i.e. 3k samples
    lsun_test_loader = torch.utils.data.DataLoader(
        datasets.LSUN(root=LSUN_DIR, classes='val', transform=lsun_transformer_valid),
        batch_size=batch_size,
        shuffle=False,
        **kwargs
    )
    return lsun_train_loader, lsun_test_loader


def get_cifar_loaders(use_cuda, batch_size):
    """
    Get the CIFAR-10 pytorch data loader.

    Args:
        use_cuda: Use cuda flag.
        batch_size: The size of the batch.
    """
    kwargs = {"num_workers": 8, "pin_memory": True} if use_cuda else {}
    cifar10_transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

    cifar10_train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=CIFAR10_DIR, train=True, download=True, transform=cifar10_transformer),
        batch_size=batch_size,
        shuffle=True,
        **kwargs
    )

    cifar10_test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=CIFAR10_DIR, train=False, download=True, transform=cifar10_transformer),
        batch_size=batch_size,
        shuffle=False,
        **kwargs
    )
    return cifar10_train_loader, cifar10_test_loader


def get_cifar100_loaders(use_cuda, batch_size):
    """
    Get the CIFAR-100 pytorch data loader.

    Args:
        use_cuda: Use cuda flag.
        batch_size: The size of the batch.

    """
    kwargs = {"num_workers": 8, "pin_memory": True} if use_cuda else {}
    cifar100_transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])

    cifar100_train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root=CIFAR100_DIR, train=True, download=True, transform=cifar100_transformer),
        batch_size=batch_size,
        shuffle=True,
        **kwargs
    )

    cifar100_test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root=CIFAR100_DIR, train=False, download=True, transform=cifar100_transformer),
        batch_size=batch_size,
        shuffle=False,
        **kwargs
    )
    return cifar100_train_loader, cifar100_test_loader


def get_cinic_loaders(use_cuda, batch_size):
    """
    Get the CINIC pytorch data loader.

    Args:
        use_cuda: Use cuda flag.
        batch_size: The size of the batch.

    """
    kwargs = {"num_workers": 8, "pin_memory": True} if use_cuda else {}

    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]

    cinic_transformer = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=cinic_mean,
                                                                 std=cinic_std)])

    cinic_train = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(CINIC_DIR + '/train',
                                                                               transform=cinic_transformer),
                                              batch_size=batch_size, shuffle=True, **kwargs)

    cinic_test = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(CINIC_DIR + '/test',
                                                                              transform=cinic_transformer),
                                             batch_size=batch_size, shuffle=True, **kwargs)

    return cinic_train, cinic_test


def get_svhn_loaders(use_cuda, batch_size, add_extra=True):
    """
    Get the SVHN pytorch data loader.

    Args:
        use_cuda: Use cuda flag.
        batch_size: The size of the batch.
        add_extra: Add extra data samples.

    """
    kwargs = {"num_workers": 8, "pin_memory": True} if use_cuda else {}
    svhn_transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        (0.4377, 0.4438, 0.4728), (0.198, 0.201, 0.197))])

    if add_extra:
        train_dataset = ConcatDataset([datasets.SVHN(
            root='../data', split='train', download=True, transform=svhn_transformer),
            datasets.SVHN(
                root=SVHN_DIR, split='extra', download=True, transform=svhn_transformer)])
    else:
        train_dataset = datasets.SVHN(root=SVHN_DIR, split='train', download=True, transform=svhn_transformer)

    svhn_train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **kwargs
    )

    svhn_test_loader = torch.utils.data.DataLoader(
        datasets.SVHN(root=SVHN_DIR, split='test', download=True, transform=svhn_transformer),
        batch_size=batch_size,
        shuffle=False,
        **kwargs
    )

    return svhn_train_loader, svhn_test_loader


def get_data_loaders(use_cuda, batch_size, dataset='mnist'):
    """
    Get the data loaders for a specific dataset.

    Args:
        use_cuda: Use cuda flag.
        batch_size: The size of the batch.
        dataset: The dataset of the corresponding data loaders.
    """
    if dataset == 'mnist':
        return get_mnist_loaders(use_cuda, batch_size)
    elif dataset == 'cifar':
        return get_cifar_loaders(use_cuda, batch_size)
    elif dataset == 'svhn':
        return get_svhn_loaders(use_cuda, batch_size)
    elif dataset == 'cifar100':
        return get_cifar100_loaders(use_cuda, batch_size)
    elif dataset == 'cinic':
        return get_cinic_loaders(use_cuda, batch_size)
    elif dataset == 'lsun':
        return get_lsun_loaders(use_cuda, batch_size)


def get_data_flatten_shape(data_loader):
    """
    Get the flattened data shape of data loader's dataset.

    Args:
        data_loader: The data loader.
    """
    if isinstance(data_loader.dataset, ConcatDataset):
        return (data_loader.dataset.cumulative_sizes[-1],
                torch.prod(torch.tensor(data_loader.dataset.datasets[0].data.shape[1:])).int().item())
    if 'lsun' in data_loader.dataset.root:
        return data_loader.dataset.length, 3 * 32 * 32
    if isinstance(data_loader.dataset, torchvision.datasets.ImageFolder):
        return len(data_loader.dataset), 3 * 32 * 32
    return (data_loader.dataset.data.shape[0],
            torch.prod(torch.tensor(data_loader.dataset.data.shape[1:])).int().item())


def make_spn(S, I, R, D, dropout, device, F=28 ** 2, C=10, leaf_distribution=RatNormal) -> RatSpn:
    """
    Construct the RatSpn. See Peharz et al., UAI 2019 (http://proceedings.mlr.press/v115/peharz20a/peharz20a.pdf).

    Args:
        S: The number of sum nodes in a sum region.
        I: The number of input distributions at leaves.
        R: The number of replicas.
        D: The depth in number of layers.
        dropout: The dropout parameter to apply during training.
        device: The device where to put the model.
        F: The number of data features.
        C: The number of classes i.e. the root heads.
        leaf_distribution: The type of leaf distributions.
    """

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

    # Construct RatSpn from config
    model = RatSpn(config)

    model = model.to(device)
    model.train()

    return model


def evaluate_corrupted_svhn_tdi(model_dir=None, dropout_inference=None, rat_S=20, rat_I=20, rat_D=5, rat_R=5,
                                corrupted_svhn_dir='', model=None, class_label=None):
    """
    Evaluate the corrupted SVHN images with Tractable Dropout Inference on a given probabilistic circuit (SPN).
    Args:
        model_dir: The directory where the model is stored.
        dropout_inference: The dropout p parameter for the Tractable Dropout Inference.
        rat_S: The number of sum nodes in a sum region of the SPN.
        rat_I: The number of input leaf distributions.
        rat_D: The depth of the SPN.
        rat_R: The number of replicas.
        corrupted_svhn_dir: The directory where the corrupted SVHN images are stored.
        model: An already loaded model to evaluate (optional).
        class_label: The class label of the data on which we want to perform the evaluation, if None the evaluation
        is performed over the whole dataset.
    """

    device = sys.argv[1]
    torch.cuda.benchmark = True

    d = model_dir + "tdi/svhn_c/tdi_p_{}/".format(str(dropout_inference).replace('.', ''))
    ensure_dir(d)
    n_features = 3 * 32 * 32
    leaves = RatNormal
    rat_C = 10

    if not model:
        model = make_spn(S=rat_S, I=rat_I, D=rat_D, R=rat_R, device=device, dropout=dropout_inference, F=n_features,
                         C=rat_C, leaf_distribution=leaves)

        checkpoint = torch.load(model_dir + 'checkpoint.tar')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        device = torch.device("cuda")
        model.to(device)

    for corruption in get_corruption_names('common'):
        for cl in range(5):
            cl += 1
            print("Corruption {} Level {}".format(corruption, cl))

            corrupted_dataset = np.load(
                corrupted_svhn_dir + 'svhn_test_{}_l{}.npy'.format(corruption, cl))
            labels = np.load(corrupted_svhn_dir + 'svhn_test_{}_l{}_labels.npy'.format(corruption, cl))
            assert corrupted_dataset.shape[0] == labels.shape[0]
            assert labels.shape[0] == 26032

            model.eval()

            svhn_transformer = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.4377, 0.4438, 0.4728), (0.198, 0.201, 0.197))])

            corrupted_dataset = CustomTensorDataset(tensors=[torch.tensor(corrupted_dataset), torch.tensor(labels)],
                                                    transform=svhn_transformer)

            if class_label is not None:
                targets = torch.tensor(corrupted_dataset.targets.clone().detach())
                target_idx = (targets == class_label).nonzero()
                sampler = torch.utils.data.sampler.SubsetRandomSampler(target_idx.reshape((-1,)))
                data_loader = torch.utils.data.DataLoader(corrupted_dataset, batch_size=100, shuffle=False,
                                                          sampler=sampler)
            else:
                data_loader = torch.utils.data.DataLoader(corrupted_dataset, batch_size=100, shuffle=False)

            with torch.no_grad():
                n_correct = 0
                n_samples = 0

                for batch_index, (data, target) in enumerate(data_loader):
                    data = data.to(device)
                    data = data.view(data.shape[0], -1)
                    n_samples += data.shape[0]

                    if dropout_inference > 0.0:
                        output, vars, ll_x, var_x, root_heads, heads_vars = \
                            model(data, dropout_inference=dropout_inference, dropout_cf=True)
                    else:
                        output = model(data, dropout_inference=dropout_inference, dropout_cf=False)

                    if batch_index == 0:
                        output_res = output.detach().cpu().numpy()
                        if dropout_inference > 0.0:
                            var_res = vars.detach().cpu().numpy()
                            ll_x_res = ll_x.detach().cpu().numpy().flatten()
                            var_x_res = var_x.detach().cpu().numpy().flatten()
                            root_heads_res = root_heads.detach().cpu().numpy()
                            heads_vars_res = heads_vars.detach().cpu().numpy()
                        else:
                            # normalize output in log space when dropout is not applied at inference time
                            output_res = output_res - np.repeat(
                                scipy.special.logsumexp(output_res, axis=1).reshape(-1, 1), 10, axis=1)
                    else:
                        if dropout_inference > 0.0:
                            output_res = np.concatenate((output_res, output.detach().cpu().numpy()))
                            var_res = np.concatenate((var_res, vars.detach().cpu().numpy()))
                            ll_x_res = np.concatenate((ll_x_res, ll_x.detach().cpu().numpy().flatten()))
                            var_x_res = np.concatenate((var_x_res, var_x.detach().cpu().numpy().flatten()))
                            root_heads_res = np.concatenate((root_heads_res, root_heads.detach().cpu().numpy()))
                            heads_vars_res = np.concatenate((heads_vars_res, heads_vars.detach().cpu().numpy()))
                        else:
                            output_tmp = output.detach().cpu().numpy()
                            # normalize output in log space when dropout is not applied at inference time
                            output_tmp = output_tmp - np.repeat(
                                scipy.special.logsumexp(output_tmp, axis=1).reshape(-1, 1), 10, axis=1)
                            output_res = np.concatenate((output_res, output_tmp))

                    if class_label:
                        n_correct += (output.argmax(dim=1) == class_label).sum()
                    else:
                        n_correct += (output.argmax(dim=1) == target.to(device)).sum().item()

                print("N of correct test predictions: {}/{} ({}%)".format(n_correct, n_samples,
                                                                          (n_correct / n_samples) * 100))

            fold = 'test'
            training_dataset = 'svhn'
            np.save(d + 'output_{}_{}_{}_{}_{}'.format(
                training_dataset, fold, dropout_inference, corruption, cl), output_res)
            if dropout_inference > 0.0:
                np.save(d + 'var_{}_{}_{}_{}_{}'.format(
                    training_dataset, fold, dropout_inference, corruption, cl), var_res)
                np.save(d + 'll_x_{}_{}_{}_{}_{}'.format(
                    training_dataset, fold, dropout_inference, corruption, cl), ll_x_res)
                np.save(d + 'var_x_{}_{}_{}_{}_{}'.format(
                    training_dataset, fold, dropout_inference, corruption, cl), var_x_res)
                np.save(d + 'heads_x_{}_{}_{}_{}_{}'.format(
                    training_dataset, fold, dropout_inference, corruption, cl),
                        root_heads_res)
                np.save(d + 'heads_vars_{}_{}_{}_{}_{}'.format(
                    training_dataset, fold, dropout_inference, corruption, cl),
                        heads_vars_res)
    return model


def test_tdi(model_dir=None, training_dataset=None, dropout_inference=None, batch_size=20,
             rat_S=20, rat_I=20, rat_D=5, rat_R=5, rotation=None, model=None, eval_train_set=False,
             ll_correction=False):
    """
    Perform inference with Tractable Dropout Inference.
    Args:
        model_dir: The directory where the model is stored.
        training_dataset: The dataset used to train the model.
        dropout_inference: The dropout p parameter for the Tractable Dropout Inference, if 0 it performs.
        the conventional probabilistic inference.
        batch_size: The batch size.
        rat_S: The number of sum nodes in a sum region of the SPN.
        rat_I: The number of input leaf distributions.
        rat_D: The depth of the SPN.
        rat_R: The number of replicas.
        rotation: The degrees of the rotation to apply to the image.
        model: An already loaded model to evaluate (optional).
        eval_train_set: If True, it evaluates the model with TDI on the training dataset.
        ll_correction: If True it applies the likelihood correction.
    """
    device = sys.argv[1]
    use_cuda = True
    torch.cuda.benchmark = True

    d = model_dir + "tdi/"
    ensure_dir(d)
    train_loader, test_loader = get_data_loaders(use_cuda, batch_size=batch_size, dataset=training_dataset)
    if eval_train_set:
        test_loader = train_loader

    n_features = get_data_flatten_shape(train_loader)[1]

    leaves = RatNormal
    rat_C = 10

    if not model:
        model = make_spn(S=rat_S, I=rat_I, D=rat_D, R=rat_R, device=device, dropout=dropout_inference, F=n_features,
                         C=rat_C, leaf_distribution=leaves)

        checkpoint = torch.load(model_dir + 'checkpoint.tar')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        device = torch.device("cuda")
        model.to(device)

    tag = "Testing Tractable Dropout Inference."

    correct = 0

    if rotation is not None:
        if training_dataset == 'mnist':
            mean = 0.1307
            std = 0.3081
            print("Performing inference with rotation of {} degrees...".format(rotation))
        else:
            raise NotImplementedError()

    with torch.no_grad():
        t_start = time.time()
        for batch_index, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            if rotation and training_dataset == 'mnist':
                data = torchvision.transforms.functional.rotate(data.reshape(-1, 1, 28, 28), rotation,
                                                                fill=-mean / std).reshape(-1, 28, 28)
            data = data.view(data.shape[0], -1)

            if dropout_inference > 0.0:
                output, stds, ll_x, var_x, root_heads, heads_vars = \
                    model(data, dropout_inference=dropout_inference, dropout_cf=True,
                          ll_correction=ll_correction)
            else:
                output = model(data, dropout_inference=dropout_inference, dropout_cf=False)

            if batch_index == 0:
                output_res = output.detach().cpu().numpy()
                if dropout_inference > 0.0:
                    var_res = stds.detach().cpu().numpy()
                    ll_x_res = ll_x.detach().cpu().numpy().flatten()
                    var_x_res = var_x.detach().cpu().numpy().flatten()
                    root_heads_res = root_heads.detach().cpu().numpy()
                    heads_vars_res = heads_vars.detach().cpu().numpy()
                else:
                    # normalize output in log space when dropout is not applied at inference time
                    output_res = output_res - np.repeat(
                        scipy.special.logsumexp(output_res, axis=1).reshape(-1, 1), 10, axis=1)

            else:
                if dropout_inference > 0.0:
                    output_res = np.concatenate((output_res, output.detach().cpu().numpy()))
                    var_res = np.concatenate((var_res, stds.detach().cpu().numpy()))
                    ll_x_res = np.concatenate((ll_x_res, ll_x.detach().cpu().numpy().flatten()))
                    var_x_res = np.concatenate((var_x_res, var_x.detach().cpu().numpy().flatten()))
                    root_heads_res = np.concatenate((root_heads_res, root_heads.detach().cpu().numpy()))
                    heads_vars_res = np.concatenate((heads_vars_res, heads_vars.detach().cpu().numpy()))
                else:
                    output_tmp = output.detach().cpu().numpy()
                    # normalize output in log space when dropout is not applied at inference time
                    output_tmp = output_tmp - np.repeat(
                        scipy.special.logsumexp(output_tmp, axis=1).reshape(-1, 1), 10, axis=1)
                    output_res = np.concatenate((output_res, output_tmp))

            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()

        t_delta = time_delta_now(t_start)
        print("Eval took {}".format(t_delta))

    if eval_train_set:
        fold = 'train'
    else:
        fold = 'test'
    np.save(d + 'output_{}_{}_{}_{}_{}'.format(
        training_dataset, fold, dropout_inference, rotation, ll_correction), output_res)
    if dropout_inference > 0.0:
        np.save(d + 'var_{}_{}_{}_{}_{}'.format(
            training_dataset, fold, dropout_inference, rotation, ll_correction),
                var_res)
        np.save(d + 'll_x_{}_{}_{}_{}_{}'.format(
            training_dataset, fold, dropout_inference, rotation, ll_correction),
                ll_x_res)
        np.save(d + 'var_x_{}_{}_{}_{}_{}'.format(
            training_dataset, fold, dropout_inference, rotation, ll_correction),
                var_x_res)
        np.save(d + 'heads_x_{}_{}_{}_{}_{}'.format(
            training_dataset, fold, dropout_inference, rotation, ll_correction),
                root_heads_res)
        np.save(d + 'heads_vars_{}_{}_{}_{}_{}'.format(
            training_dataset, fold, dropout_inference, rotation, ll_correction),
                heads_vars_res)

    accuracy = 100.0 * correct / len(test_loader.dataset)

    output_string = "{} -- Accuracy: {}/{} ({:.0f}%)".format(
        tag, correct, len(test_loader.dataset), accuracy
    )
    print(output_string)
    with open(d + 'training.out', 'a') as writer:
        writer.write(output_string + "\n")

    if dropout_inference == 0.0:
        var_res = []
    return model, output_res, var_res


def run_torch(n_epochs=100, batch_size=256, dropout=0.0, training_dataset='mnist', eval_every_n_epochs=5, lr=1e-3):
    """
    Run the probabilistic circuit training with PyTorch.

    Args:
        n_epochs (int, optional): Number of epochs for model training.
        batch_size (int, optional): The batch size.
        dropout (float, optional): The dropout p parameter during training.
        training_dataset (str, optional): The name of the dataset to be used for training.
        eval_every_n_epochs (int, optional): The interval of epochs before evaluating the model during training.
        lr (float, optional): The learning rate.
    """

    assert len(sys.argv) == 2, "Usage: train_pc.py cuda/cpu"
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

    train_loader, test_loader = get_data_loaders(use_cuda, batch_size=batch_size, dataset=training_dataset)
    n_features = get_data_flatten_shape(train_loader)[1]

    rat_S, rat_I, rat_D, rat_R, rat_C, leaves = 20, 20, 5, 5, 10, RatNormal

    model = make_spn(S=rat_S, I=rat_I, D=rat_D, R=rat_R, device=dev, dropout=dropout, F=n_features, C=rat_C,
                     leaf_distribution=leaves)
    model.train()
    n_rat_params = count_params(model)
    print("Number of parameters: ", n_rat_params)

    # Define optimizer
    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    print("Learning rate: {}".format(lr))

    log_interval = 100
    d_results = d + "results/"
    d_model = d + "model/"
    ensure_dir(d_results)
    ensure_dir(d_model)

    for epoch in range(n_epochs):
        model.train()
        t_start = time.time()
        running_loss = 0.0

        for batch_index, (data, target) in enumerate(train_loader):

            # Send data to the correct device
            data, target = data.to(device), target.to(device)
            data = data.view(data.shape[0], -1)

            # Reset gradients
            optimizer.zero_grad()

            # Inference
            output = model(data)

            # Compute loss
            loss = loss_fn(output, target)

            # Backprop
            loss.backward()
            optimizer.step()

            # Log stuff
            running_loss += loss.item()

            if batch_index % log_interval == (log_interval - 1):
                pred = output.argmax(1).eq(target).sum().cpu().numpy() / data.shape[0] * 100
                print(
                    "Train Epoch: {} [{: >5}/{: <5} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.0f}%".format(
                        epoch,
                        batch_index * len(data),
                        get_data_flatten_shape(train_loader)[0],
                        100.0 * batch_index / len(train_loader),
                        running_loss / log_interval,
                        pred,
                    ),
                    end="\r",
                )
                running_loss = 0.0

        t_delta = time_delta_now(t_start)
        print("Train Epoch {} took {}".format(epoch, t_delta))

        if epoch % eval_every_n_epochs == (eval_every_n_epochs - 1):
            print("Evaluating model...")
            evaluate_model(model, device, train_loader, "{}^ epoch - Train".format(epoch + 1), output_dir=d)
            evaluate_model(model, device, test_loader, "{}^ epoch - Test".format(epoch + 1), output_dir=d)

            print('Saving model... epoch {}'.format(epoch))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'lr': lr,
            }, d + 'model/checkpoint.tar')

    print('Saving model...')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'lr': lr,
    }, d + 'model/checkpoint.tar')

    evaluate_model(model, device, train_loader, "Train", output_dir=d)
    evaluate_model(model, device, test_loader, "Test", output_dir=d)

    return model


def evaluate_model(model: torch.nn.Module, device, loader, tag, output_dir="") -> float:
    """
    Evaluate the model.

    Args:
        model (nn.Module): The model to evalaute.
        device: Execution device.
        loader: Data loader.
        tag (str): Tag for information.
        output_dir (str): Path of the directory where to store training information.

    Returns:
        float: Tuple of loss and accuracy.
    """
    model.eval()
    loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")

    with torch.no_grad():
        for batch_index, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            data = data.view(data.shape[0], -1)
            output = model(data)

            # sum up batch loss
            loss += criterion(output, target).item()

            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()

    loss /= len(loader.dataset)
    accuracy = 100.0 * correct / len(loader.dataset)

    output_string = "{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
        tag, loss, correct, len(loader.dataset), accuracy
    )
    print(output_string)
    with open(output_dir + 'training.out', 'a') as writer:
        writer.write(output_string + "\n")

    return loss


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

    # data set directories
    LSUN_DIR = ''
    CIFAR10_DIR = ''
    CIFAR100_DIR = ''
    CINIC_DIR = ''
    SVHN_DIR = ''

    # To run experiments on the corrupted SVHN data, follow the documentation of the
    # Python package "imagecorruptions" and Hendrycks et al., ICLR 2019 (https://openreview.net/forum?id=HJz6tiCqYm)
    # To generate the corrupted images save them as NumPy tensors with the following file name
    # 'svhn_test_<CORRUPTION_NAME>_l<SEVERITY>_labels.npy' in the CORRUPTED_SVHN_DIR
    # directory. Once having a model trained on SVHN, one can use the method evaluate_corrupted_svhn_tdi()
    # to perform Tractable Dropout Inference with SVHN corrupted data.
    CORRUPTED_SVHN_DIR = ''

    # learn and save a model
    pc = run_torch(n_epochs=20, batch_size=200, dropout=0.2, training_dataset='mnist', eval_every_n_epochs=5, lr=0.01)

    # run inference
    rotations = [rot for rot in range(0, 95, 5)]

    results = {}

    for rotation in rotations:
        pc, class_probs, vars = test_tdi(model_dir='', training_dataset='mnist', dropout_inference=0.2,
                                         batch_size=100, rotation=rotation, model=pc, eval_train_set=False)
        results[rotation] = (class_probs, vars)

    plot_figure(results, rotations)
