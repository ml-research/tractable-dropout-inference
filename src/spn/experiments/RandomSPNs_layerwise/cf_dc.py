import argparse
import numpy as np
from typing import List, Tuple
from abc import ABC, abstractmethod
import torch
from torch import dropout, nn
from torch.distributions import Normal
from rich.traceback import install
from icecream import ic
import tqdm

install()


def logsumexp(left, right, mask=None):
    """
    Source: https://github.com/pytorch/pytorch/issues/32097

    Logsumexp with custom scalar mask to allow for negative values in the sum.

    Args:
      tensor:
      other:
      mask:  (Default value = None)

    Returns:

    """
    if mask is None:
        mask = torch.tensor([1, 1])
    else:
        assert mask.shape == (2,), "Invalid mask shape"

    maxes = torch.max(left, right)
    return maxes + ((left - maxes).exp() * mask[0] + (right - maxes).exp() * mask[1]).log()


class Node(ABC, nn.Module):
    """Abstract node."""

    def __init__(self, chs: List["Node"], dropout: float) -> None:
        super().__init__()
        self.chs = chs
        self.dropout = dropout

    @abstractmethod
    def forward_dropout_mc(self, x: torch.Tensor) -> torch.Tensor:
        """MC n_childrenropout forward pass."""
        pass

    @abstractmethod
    def forward_dropout_cf(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Closed-form dropout forward pass."""

    @property
    def num_children(self):
        return len(self.chs)


class SumNode(Node):
    """Sum node: convex combination of children."""

    def __init__(self, chs: List["Node"], dropout: float) -> None:
        super().__init__(chs, dropout)

        # Random weights
        self.ws = torch.rand(len(chs))

    def forward_dropout_mc(self, x: torch.Tensor) -> torch.Tensor:
        while True:
            dropout_mask = torch.rand(x.shape[0], self.num_children) > self.dropout
            if torch.all(torch.sum(dropout_mask, dim=1) > 0):
                break

        ll_children = torch.stack([ch.forward_dropout_mc(x) for ch in self.chs], dim=1)

        ws = self.ws

        # Normalize
        ws = ws / ws.sum()

        # Apply dropout mask
        ws = ws * dropout_mask

        log_ws = ws.log()
        ll = torch.logsumexp(ll_children + log_ws, dim=1)


        # Constant to account for reduction in expected LL
        ll = ll - np.log(1 - self.dropout)

        return ll

    def forward_dropout_cf(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_exps = []
        log_vars = []
        for ch in self.chs:
            log_exp_ch, log_var_ch = ch.forward_dropout_cf(x)
            log_exps.append(log_exp_ch)
            log_vars.append(log_var_ch)

        log_exps = torch.stack(log_exps, dim=-1)
        log_vars = torch.stack(log_vars, dim=-1)

        # Normalize
        ws = self.ws / self.ws.sum()
        log_ws = ws.log()

        # E[N_S] =  \sum_i w_i * E[N_i]
        # logE[N_S] = logsumexp( log_ws + log_E[N_i] )
        log_exp_sum = torch.logsumexp(log_ws + log_exps, dim=1)

        # Var[N_S] = 1/(1-p) \sum_i w_i^2 * Var[N_i] + p/(1-p) \sum_i w_i^2 * E[N_i]^2
        # logVar[N_S] = logsumexp( [log(1/(1-p)) + logsumexp(2*log_ws + log_Var[Ns]), log(p/(1-p)) + logsumexp(2*log_ws + 2*log_E[Ns])] )
        # logVar[N_S] = logsumexp(  log(1/(1-p)) +   term_left                      , log(p/(1-p)) + term_right                     ] )

        term_left = torch.logsumexp(2 * log_ws + log_vars, dim=1)
        term_right = torch.logsumexp(2 * log_ws + 2 * log_exps, dim=1)

        log_var_sum = torch.logsumexp(
            torch.stack(
                [
                    np.log(1 / (1 - self.dropout)) + term_left,
                    np.log(self.dropout / (1 - self.dropout)) + term_right,
                ],
                dim=-1,
            ),
            dim=-1,
        )

        return log_exp_sum, log_var_sum


class ProductNode(Node):
    def __init__(self, chs: List["Node"], dropout: float) -> None:
        super().__init__(chs, dropout)

    def forward_dropout_mc(self, x: torch.Tensor) -> torch.Tensor:
        ll_children = torch.stack([ch.forward_dropout_mc(x) for ch in self.chs], dim=1)
        ll = torch.sum(ll_children, dim=1)

        return ll

    def forward_dropout_cf(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        log_exps = []
        log_vars = []
        for ch in self.chs:
            log_exp_ch, log_var_ch = ch.forward_dropout_cf(x)
            log_exps.append(log_exp_ch)
            log_vars.append(log_var_ch)

        log_exps = torch.stack(log_exps, dim=-1)
        log_vars = torch.stack(log_vars, dim=-1)

        # E[N_P] = \prod_i E[N_i]
        # logE[N_P] = \sum_i logE[N_i]
        log_exp_prod = torch.sum(log_exps, dim=-1)

        # Var[N_P] = \prod_i ( Var[N_i] + E[N_i]^2 ) - \prod_i E[N_i]^2
        # logVar[N_P] = logsumexp(\sum_i ( logsumexp(logVar[N_i]),2*logE[N_i]), \sum_i 2*logE[N_i], mask=[1, -1])
        # logVar[N_P] =               term_left                        -          term_right
        term_left = torch.sum(
            torch.logsumexp(torch.stack((log_vars, 2 * log_exps), dim=-1), dim=-1), dim=-1
        )
        term_right = torch.sum(2 * log_exps, dim=1)
        mask = torch.tensor([1, -1])
        log_var_prod = logsumexp(term_left, term_right, mask=mask)

        return log_exp_prod, log_var_prod


class GaussianNode(Node):
    def __init__(self, scope: int) -> None:
        super().__init__(chs=[], dropout=0.0)

        self.scope = scope
        self.mean = torch.randn(1)
        self.std = torch.rand(1)

    def forward_dropout_mc(self, x: torch.Tensor) -> torch.Tensor:
        x_i = x[:, self.scope]
        normal = Normal(loc=self.mean, scale=self.std)
        ll = normal.log_prob(x_i)
        return ll

    def forward_dropout_cf(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ll = self.forward_dropout_mc(x)
        ll_exp = ll
        ll_var = torch.zeros(x.shape[0]).log()
        return ll_exp, ll_var


def compare(node: Node, n_children: int, n_mc: int) -> None:
    """

    Args:
      node: Node to be evaluated.
      n_children: Number of random variables.
      n_mc: Number of MC trials.
    """
    # Init some random data
    batch_size = 5
    x = torch.randn(batch_size, n_children)

    # Compute MC log-likelihoods
    lls_mc = []
    for l in tqdm.trange(n_mc, desc="Monte Carlo Simulation"):
        lls_mc.append(node.forward_dropout_mc(x))
    lls_mc = torch.stack(lls_mc, dim=1)

    # Compute MC expectation and variance
    probs_mc = lls_mc.exp()
    log_mc_exp = probs_mc.mean(dim=1).log()
    log_mc_var = probs_mc.var(dim=1).log()

    # Compute closed-form expectation and variance
    log_cf_exp, log_cf_var = node.forward_dropout_cf(x)

    # Compute differences
    log_exp_rel_diffs = abs((log_cf_exp - log_mc_exp) / log_mc_exp)
    log_var_rel_diffs = abs((log_cf_var - log_mc_var) / log_mc_var)

    # Print
    print("-- Expectations:")
    ic(log_mc_exp, log_cf_exp, log_exp_rel_diffs)
    print("-- Variances:")
    ic(log_mc_var, log_cf_var, log_var_rel_diffs)
    print("-" * 80)


def compare_sum(dropout: float, n_children, n_mc):
    print(f"Comparing a single sum node with {n_children} leaf node children ...")

    leaves = []
    # Construct sum(gauss, gauss, ...) node
    for d in range(n_children):
        leaves.append(GaussianNode(scope=0))
    sum_node = SumNode(chs=leaves, dropout=dropout)

    compare(sum_node, 1, n_mc)


def compare_prod(dropout: float, n_children, n_mc):
    print(f"Comparing a single product node with {n_children} leaf node children ...")
    print("NOTE: Variance should be '-inf' since no dropout is involved!")

    # Construct product(gauss, gauss, ...) node
    leaves = []
    for d in range(n_children):
        leaves.append(GaussianNode(scope=d))
    prod_node = ProductNode(chs=leaves, dropout=dropout)

    compare(prod_node, n_children, n_mc)


def compare_sum_prod(dropout: float, n_children, n_mc):
    print(
        f"Comparing a sum node with {n_children} product node childen with {n_children} leaf node children ..."
    )

    # Construct sum(prod(gauss, gauss, ...), prod(gauss, gauss, ...), ...) node
    prods = []
    for d_sum in range(n_children):
        leaves = []
        for d_prod in range(n_children):
            leaves.append(GaussianNode(scope=d_prod))
        prods.append(ProductNode(chs=leaves, dropout=dropout))

    sum_root = SumNode(chs=prods, dropout=dropout)

    compare(sum_root, n_children, n_mc)


def compare_prod_sum(dropout: float, n_children, n_mc):
    print(
        f"Comparing a product node with {n_children} sum node childen with {n_children} leaf node children ..."
    )

    # Construct prod(sum(gauss, gauss, ...), sum(gauss, gauss, ...), ...) node
    sums = []
    for d_prod in range(n_children):
        leaves = []
        for d_sum in range(n_children):
            leaves.append(GaussianNode(scope=d_prod))
        sums.append(SumNode(chs=leaves, dropout=dropout))

    prod_root = ProductNode(chs=sums, dropout=dropout)

    compare(prod_root, n_children, n_mc)


if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--n-mc", default=1000, type=int, help="Number of Monte Carlo simulation trials."
    )
    parser.add_argument("--dropout", default=0.1, type=float, help="n_childrenropout probability.")
    parser.add_argument(
        "--n-children", default=5, type=int, help="Number of children for all nodes."
    )
    parser.add_argument("--seed", default=0, type=int, help="Random seed for reproducibility.")
    args = parser.parse_args()
    dropout = args.dropout
    n_children = args.n_children
    n_mc = args.n_mc

    # Set seed to ensure reproducibility
    torch.manual_seed(args.seed)

    # Run comparisons
    compare_sum(dropout=dropout, n_children=n_children, n_mc=n_mc)
    compare_prod(dropout=dropout, n_children=n_children, n_mc=n_mc)
    compare_sum_prod(dropout=dropout, n_children=n_children, n_mc=n_mc)
    compare_prod_sum(dropout=dropout, n_children=n_children, n_mc=n_mc)