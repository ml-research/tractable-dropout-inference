import logging
from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from spn.algorithms.layerwise.type_checks import check_valid
from spn.algorithms.layerwise.utils import SamplingContext

from icecream import ic
logger = logging.getLogger(__name__)


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

class AbstractLayer(nn.Module, ABC):
    def __init__(self, in_features: int, num_repetitions: int = 1):
        super().__init__()
        self.in_features = check_valid(in_features, int, 1)
        self.num_repetitions = check_valid(num_repetitions, int, 1)

    @abstractmethod
    def sample(self, n: int = None, context: SamplingContext = None) -> Union[SamplingContext, torch.Tensor]:
        """
        Sample from this layer.
        Args:
            context: Sampling context.
            n: Number of samples.

        Returns:
            torch.Tensor: Generated samples.
        """
        pass


class Sum(AbstractLayer):
    def __init__(
        self, in_channels: int, in_features: int, out_channels: int, num_repetitions: int = 1, dropout: float = 0.0
    ):
        """
        Create a Sum layer.

        Input is expected to be of shape [n, d, ic, r].
        Output will be of shape [n, d, oc, r].

        Args:
            in_channels (int): Number of output channels from the previous layer.
            in_features (int): Number of input features.
            out_channels (int): Multiplicity of a sum node for a given scope set.
            num_repetitions(int): Number of layer repetitions in parallel.
            dropout (float, optional): Dropout percentage.
        """
        super().__init__(in_features, num_repetitions)

        self.in_channels = check_valid(in_channels, int, 1)
        self.out_channels = check_valid(out_channels, int, 1)
        self.dropout = nn.Parameter(torch.tensor(check_valid(dropout, float, 0.0, 1.0)), requires_grad=False)

        # Weights, such that each sumnode has its own weights
        ws = torch.randn(self.in_features, self.in_channels, self.out_channels, self.num_repetitions)
        self.weights = nn.Parameter(ws)
        self._bernoulli_dist = torch.distributions.Bernoulli(probs=self.dropout)

        self.out_shape = f"(N, {self.in_features}, {self.out_channels}, {self.num_repetitions})"

        # Necessary for sampling with evidence: Save input during forward pass.
        self._is_input_cache_enabled = False
        self._input_cache = None

    def _enable_input_cache(self):
        """Enables the input cache. This will store the input in forward passes into `self.__input_cache`."""
        self._is_input_cache_enabled = True

    def _disable_input_cache(self):
        """Disables and clears the input cache."""
        self._is_input_cache_enabled = False
        self._input_cache = None

    @property
    def __device(self):
        """Hack to obtain the current device, this layer lives on."""
        return self.weights.device

    def forward(self, x: torch.Tensor, test_dropout=False, dropout_inference=0.0, dropout_cf=False, vars=None):
        """
        Sum layer foward pass.

        Args:
            x: Input of shape [batch, in_features, in_channels].

        Returns:
            torch.Tensor: Output of shape [batch, in_features, out_channels]
        """
        # Save input if input cache is enabled
        if self._is_input_cache_enabled:
            self._input_cache = x.clone()

        # Apply dropout: Set random sum node children to 0 (-inf in log domain)
        # TODO change
        if self.training and self.dropout > 0.0:
            dropout_indices = self._bernoulli_dist.sample(x.shape).bool()
            while torch.any(torch.all(dropout_indices, dim=2)):
                dropout_indices = self._bernoulli_dist.sample(x.shape).bool()
            x[dropout_indices] = np.NINF
        if test_dropout and dropout_inference > 0.0 and not dropout_cf:
            dropout_indices = torch.distributions.Bernoulli(probs=dropout_inference).sample(x.shape).bool() #NOTE not the most efficient way
            while torch.any(torch.all(dropout_indices, dim=2)):
                dropout_indices = self._bernoulli_dist.sample(x.shape).bool()
            x[dropout_indices] = np.NINF

        # Dimensions
        n, d, ic, r = x.size()
        oc = self.weights.size(2)

        x = x.unsqueeze(3)  # Shape: [n, d, ic, 1, r]

        # Normalize weights in log-space along in_channel dimension
        # Weights is of shape [d, ic, oc, r]
        logweights = F.log_softmax(self.weights, dim=1)

        # Multiply (add in log-space) input features and weights
        log_probs = x + logweights  # Shape: [n, d, ic, oc, r]
        # print(x.shape)
        # print(logweights.shape)

        # Compute sum via logsumexp along in_channels dimension
        log_probs = torch.logsumexp(log_probs, dim=2)  # Shape: [n, d, oc, r]

        # Assert correct dimensions
        assert log_probs.size() == (n, d, oc, r)

        if vars is not None:
            squared_weights = logweights * 2
            squared_exps = x * 2
            input_vars = vars.unsqueeze(3)

            right_term = torch.logsumexp((squared_weights + squared_exps), dim=2)

            # if the sum node corresponds to a root node
            # if not dropout_cf:
            #     print("double check here, what's happening with the reshaping")
            #     breakpoint()
            #     input_vars = input_vars.reshape((input_vars.shape[0], 1, input_vars.shape[2] * input_vars.shape[4], 1, 1))

            left_term = torch.logsumexp((squared_weights + input_vars), dim=2)

            if dropout_cf:
                right_term += torch.log(torch.tensor(dropout_inference / (1 - dropout_inference)).to(x.device))
                left_term += torch.log(torch.tensor(1 / (1 - dropout_inference)).to(x.device))

            log_vars = logsumexp(left_term, right_term)

            assert log_vars.isnan().sum() == 0, "nan values when propagating from a sum node"
            assert log_vars.isfinite().sum() > 0, "no finite values while propagating from a sum node"

            return log_probs, log_vars

        return log_probs

    def sample(self, n: int = None, context: SamplingContext = None) -> SamplingContext:
        """Method to sample from this layer, based on the parents output.

        Output is always a vector of indices into the channels.

        Args:
            repetition_indices (List[int]): An index into the repetition axis for each sample.
                Can be None if `num_repetitions==1`.
            indices (torch.Tensor): Parent sampling output.
            n (int): Number of samples.
        Returns:
            torch.Tensor: Index into tensor which paths should be followed.
        """

        # Sum weights are of shape: [D, IC, OC, R]
        # We now want to use `indices` to access one in_channel for each in_feature x out_channels block
        # index is of size in_feature
        weights = self.weights.data
        d, ic, oc, r = weights.shape
        n = context.n

        # Create sampling context if this is a root layer
        if context.is_root:
            assert oc == 1 and r == 1, "Cannot start sampling from non-root layer."

            # Initialize rep indices
            context.repetition_indices = torch.zeros(n, dtype=int, device=self.__device)

            # Select weights, repeat n times along the last dimension
            weights = weights[:, :, [0] * n, 0]  # Shape: [D, IC, N]

            # Move sample dimension to the first axis: [feat, channels, batch] -> [batch, feat, channels]
            weights = weights.permute(2, 0, 1)  # Shape: [N, D, IC]
        else:
            # If this is not the root node, use the paths (out channels), specified by the parent layer
            self._check_repetition_indices(context)

            tmp = torch.zeros(n, d, ic, device=self.__device)
            for i in range(n):
                tmp[i, :, :] = weights[
                    range(self.in_features), :, context.parent_indices[i], context.repetition_indices[i]
                ]
            weights = tmp

        # Check dimensions
        assert weights.shape == (n, d, ic)

        # Apply softmax to ensure they are proper probabilities
        log_weights = F.log_softmax(weights, dim=2)

        # If evidence is given, adjust the weights with the likelihoods of the observed paths
        if self._is_input_cache_enabled and self._input_cache is not None:
            for i in range(n):
                # Reweight the i-th samples weights by its likelihood values at the correct repetition
                log_weights[i, :, :] += self._input_cache[i, :, :, context.repetition_indices[i]]

        # If sampling context is MPE, set max weight to 1 and rest to zero, such that the maximum index will be sampled
        if context.is_mpe:
            # Get index of largest weight along in-channel dimension
            indices = log_weights.argmax(dim=2)
        else:
            # Create categorical distribution and use weights as logits.
            #
            # Use the Gumble-Softmax trick to obtain one-hot indices of the categorical distribution
            # represented by the given logits. (Use Gumble-Softmax instead of Categorical
            # to allow for gradients).
            #
            # The code below is an approximation of:
            #
            # >> dist = torch.distributions.Categorical(logits=log_weights)
            # >> indices = dist.sample()

            cats = torch.arange(ic, device=log_weights.device)
            one_hot = F.gumbel_softmax(logits=log_weights, hard=True, dim=-1)
            indices = (one_hot * cats).sum(-1).long()

        # Update parent indices
        context.parent_indices = indices

        return context

    def _check_repetition_indices(self, context: SamplingContext):
        assert context.repetition_indices.shape[0] == context.parent_indices.shape[0]
        if self.num_repetitions > 1 and context.repetition_indices is None:
            raise Exception(
                f"Sum layer has multiple repetitions (num_repetitions=={self.num_repetitions}) but repetition_indices argument was None, expected a Long tensor size #samples."
            )
        if self.num_repetitions == 1 and context.repetition_indices is None:
            context.repetition_indices = torch.zeros(context.n, dtype=int, device=self.__device)

    def __repr__(self):
        return "Sum(in_channels={}, in_features={}, out_channels={}, dropout={}, out_shape={})".format(
            self.in_channels, self.in_features, self.out_channels, self.dropout, self.out_shape
        )


class Product(AbstractLayer):
    """
    Product Node Layer that chooses k scopes as children for a product node.
    """

    def __init__(self, in_features: int, cardinality: int, num_repetitions: int = 1):
        """
        Create a product node layer.

        Args:
            in_features (int): Number of input features.
            cardinality (int): Number of random children for each product node.
        """

        super().__init__(in_features, num_repetitions)

        self.cardinality = check_valid(cardinality, int, 1, in_features + 1)

        # Implement product as convolution
        self._conv_weights = nn.Parameter(torch.ones(1, 1, cardinality, 1, 1), requires_grad=False)
        self._pad = (self.cardinality - self.in_features % self.cardinality) % self.cardinality

        self._out_features = np.ceil(self.in_features / self.cardinality).astype(int)
        self.out_shape = f"(N, {self._out_features}, in_channels, {self.num_repetitions})"

    @property
    def __device(self):
        """Hack to obtain the current device, this layer lives on."""
        return self._conv_weights.device

    def _forward_cf(self, x: torch.Tensor, dropout_inference=0.0):
        exps, vars = x

        # Only one product node
        if self.cardinality == exps.shape[1]:
            exps = exps.sum(1, keepdim=True)
            # TODO
            vars = vars #TODO
            ic()
            breakpoint()

        # Special case: if cardinality is 1 (one child per product node), this is a no-op
        if self.cardinality == 1:
            # ic()
            assert exps.isnan().sum() == 0
            assert vars.isnan().sum() == 0
            return exps, vars

        # Pad if in_features % cardinality != 0
        if self._pad > 0:
            exps = F.pad(exps, pad=(0, 0, 0, 0, 0, self._pad), value=0)

        # Dimensions
        n, d, c, r = exps.size()
        d_out = d // self.cardinality

        # Use convolution with 3D weight tensor filled with ones to simulate the product node
        exps = exps.unsqueeze(1)  # Shape: [n, 1, d, c, r]
        result = F.conv3d(exps, weight=self._conv_weights, stride=(self.cardinality, 1, 1))

        # Remove simulated channel
        result = result.squeeze(1)

        assert result.size() == (n, d_out, c, r)
        return result, torch.zeros_like(result).log() # TODO double chek here, this holds only right after leaves

        # log_exps = []
        # log_vars = []
        # for ch in self.chs:
        #     log_exp_ch, log_var_ch = ch.forward_dropout_cf(x)
        #     log_exps.append(log_exp_ch)
        #     log_vars.append(log_var_ch)
        #
        # log_exps = torch.stack(log_exps, dim=-1)
        # log_vars = torch.stack(log_vars, dim=-1)
        #
        # # E[N_P] = \prod_i E[N_i]
        # # logE[N_P] = \sum_i logE[N_i]
        # log_exp_prod = torch.sum(log_exps, dim=-1)
        #
        # # Var[N_P] = \prod_i ( Var[N_i] + E[N_i]^2 ) - \prod_i E[N_i]^2
        # # logVar[N_P] = logsumexp(\sum_i ( logsumexp(logVar[N_i]),2*logE[N_i]), \sum_i 2*logE[N_i], mask=[1, -1])
        # # logVar[N_P] =               term_left                        -          term_right
        # term_left = torch.sum(
        #     torch.logsumexp(torch.stack((log_vars, 2 * log_exps), dim=-1), dim=-1), dim=-1
        # )
        # term_right = torch.sum(2 * log_exps, dim=1)
        # mask = torch.tensor([1, -1])
        # log_var_prod = logsumexp(term_left, term_right, mask=mask)

        # ic()
        # ic(log_exp_prod)
        # ic(log_var_prod)

        # return log_exp_prod, log_var_prod


    def forward(self, x: torch.Tensor, test_dropout=False, dropout_inference=0.0, dropout_cf=False, vars=None):
        """
        Product layer forward pass.

        Args:
            x: Input of shape [batch, in_features, channel].

        Returns:
            torch.Tensor: Output of shape [batch, ceil(in_features/cardinality), channel].
        """
        if isinstance(x, tuple) and dropout_cf:
            return self._forward_cf(x, dropout_inference=dropout_inference)

        # Only one product node
        if self.cardinality == x.shape[1]:
            return x.sum(1, keepdim=True)

        # Special case: if cardinality is 1 (one child per product node), this is a no-op
        if self.cardinality == 1:
            return x

        # Pad if in_features % cardinality != 0
        if self._pad > 0:
            x = F.pad(x, pad=(0, 0, 0, 0, 0, self._pad), value=0)

        # Dimensions
        n, d, c, r = x.size()
        d_out = d // self.cardinality

        # Use convolution with 3D weight tensor filled with ones to simulate the product node
        x = x.unsqueeze(1)  # Shape: [n, 1, d, c, r]
        result = F.conv3d(x, weight=self._conv_weights, stride=(self.cardinality, 1, 1))

        # Remove simulated channel
        result = result.squeeze(1)

        assert result.size() == (n, d_out, c, r)
        return result

    def sample(self, n: int = None, context: SamplingContext = None) -> SamplingContext:
        """Method to sample from this layer, based on the parents output.

        Args:
            n (int): Number of instances to sample.
            indices (torch.Tensor): Parent sampling output.
        Returns:
            torch.Tensor: Index into tensor which paths should be followed.
                          Output should be of size: in_features, out_channels.
        """

        # If this is a root node
        if context.is_root:

            if self.num_repetitions == 1:
                # If there is only a single repetition, create new sampling context
                context.parent_indices = torch.zeros(context.n, 1, dtype=int, device=self.__device)
                context.repetition_indices = torch.zeros(context.n, dtype=int, device=self.__device)
                return context
            else:
                raise Exception(
                    "Cannot start sampling from Product layer with num_repetitions > 1 and no context given."
                )
        else:
            # Repeat the parent indices, e.g. [0, 2, 3] -> [0, 0, 2, 2, 3, 3] depending on the cardinality
            indices = torch.repeat_interleave(context.parent_indices, repeats=self.cardinality, dim=1)

            # Remove padding
            if self._pad:
                indices = indices[:, : -self._pad]

            context.parent_indices = indices
            return context

    def __repr__(self):
        return "Product(in_features={}, cardinality={}, out_shape={})".format(
            self.in_features, self.cardinality, self.out_shape
        )


class CrossProduct(AbstractLayer):
    """
    Layerwise implementation of a RAT Product node.

    Builds the the combination of all children in two regions:
    res = []
    for n1 in R1, n2 in R2:
        res += [n1 * n2]

    TODO: Generalize to k regions (cardinality = k).
    """

    def __init__(self, in_features: int, in_channels: int, num_repetitions: int = 1):
        """
        Create a rat product node layer.

        Args:
            in_features (int): Number of input features.
            in_channels (int): Number of input channels. This is only needed for the sampling pass.
        """

        super().__init__(in_features, num_repetitions)
        self.in_channels = check_valid(in_channels, int, 1)
        cardinality = 2  # Fixed to binary graphs for now
        self.cardinality = check_valid(cardinality, int, 2, in_features + 1)
        self._out_features = np.ceil(self.in_features / self.cardinality).astype(int)
        self._pad = 0

        # Collect scopes for each product child
        self._scopes = [[] for _ in range(self.cardinality)]

        # Create sequence of scopes
        scopes = np.arange(self.in_features)

        # For two consecutive scopes
        for i in range(0, self.in_features, self.cardinality):
            for j in range(cardinality):
                if i + j < in_features:
                    self._scopes[j].append(scopes[i + j])
                else:
                    # Case: d mod cardinality != 0 => Create marginalized nodes with prob 1.0
                    # Pad x in forward pass on the right: [n, d, c] -> [n, d+1, c] where index
                    # d+1 is the marginalized node (index "in_features")
                    self._scopes[j].append(self.in_features)

        # Transform into numpy array for easier indexing
        self._scopes = np.array(self._scopes)

        # Create index map from flattened to coordinates (only needed in sampling)
        self.unraveled_channel_indices = nn.Parameter(
            torch.tensor([(i, j) for i in range(self.in_channels) for j in range(self.in_channels)]),
            requires_grad=False,
        )

        self.out_shape = f"(N, {self._out_features}, {self.in_channels ** 2}, {self.num_repetitions})"

    @property
    def __device(self):
        """Hack to obtain the current device, this layer lives on."""
        return self.unraveled_channel_indices.device

    def forward(self, x: torch.Tensor, test_dropout=False, dropout_inference=0.0, dropout_cf=False, vars=None):
        """
        Product layer forward pass.

        Args:
            x: Input of shape [batch, in_features, channel].

        Returns:
            torch.Tensor: Output of shape [batch, ceil(in_features/2), channel * channel].
        """
        # Check if padding to next power of 2 is necessary
        if self.in_features != x.shape[1]:
            # Compute necessary padding to the next power of 2
            self._pad = 2 ** np.ceil(np.log2(x.shape[1])).astype(np.int) - x.shape[1]

            # Pad marginalized node
            x = F.pad(x, pad=[0, 0, 0, 0, 0, self._pad], mode="constant", value=0.0)

            if vars is not None and dropout_cf:
                vars = F.pad(vars, pad=[0, 0, 0, 0, 0, self._pad], mode="constant", value=0.0)

        # Dimensions
        n, d, c, r = x.size()
        d_out = d // self.cardinality

        # Build outer sum, using broadcasting, this can be done with
        # modifying the tensor dimensions:
        # left: [n, d/2, c, r] -> [n, d/2, c, 1, r]
        # right: [n, d/2, c, r] -> [n, d/2, 1, c, r]
        left = x[:, self._scopes[0, :], :, :].unsqueeze(3)
        right = x[:, self._scopes[1, :], :, :].unsqueeze(2)

        # left + right with broadcasting: [n, d/2, c, 1, r] + [n, d/2, 1, c, r] -> [n, d/2, c, c, r]
        result = left + right

        # Put the two channel dimensions from the outer sum into one single dimension:
        # [n, d/2, c, c, r] -> [n, d/2, c * c, r]
        result = result.view(n, d_out, c * c, r)

        assert result.size() == (n, d_out, c * c, r)

        if vars is not None and dropout_cf:
            left_vars = vars[:, self._scopes[0, :], :, :].unsqueeze(3).to(x.device)
            right_vars = vars[:, self._scopes[1, :], :, :].unsqueeze(2).to(x.device)

            left_squared = left * 2
            right_squared = right * 2

            # perform variance computation at product nodes
            var_right_term = left_squared + right_squared

            # var_left_term_left = torch.log(torch.exp(left_vars) + torch.exp(left_squared))
            var_left_term_left = logsumexp(left_vars, left_squared)
            # var_left_term_right = torch.log(torch.exp(right_vars) + torch.exp(right_squared))
            var_left_term_right = logsumexp(right_vars, right_squared)

            var_left_term = var_left_term_left + var_left_term_right

            mask = torch.tensor([1, -1])
            log_var_prod = logsumexp(var_left_term, var_right_term, mask=mask)

            # Put the two channel dimensions from the outer sum into one single dimension:
            log_var_prod = log_var_prod.view(n, d_out, c * c, r)
            assert log_var_prod.size() == (n, d_out, c * c, r)

            assert result.isnan().sum() == 0, "nan values"
            assert log_var_prod.isnan().sum() == 0, breakpoint()

            return result, log_var_prod
        else:
            return result

    def sample(self, n: int = None, context: SamplingContext = None) -> SamplingContext:
        """Method to sample from this layer, based on the parents output.

        Args:
            n: Number of samples.
            indices (torch.Tensor): Parent sampling output
        Returns:
            torch.Tensor: Index into tensor which paths should be followed.
                          Output should be of size: in_features, out_channels.
        """

        # If this is a root node
        if context.is_root:
            if self.num_repetitions == 1:
                # If there is only a single repetition, create new sampling context
                context.parent_indices = torch.zeros(context.n, 1, dtype=int, device=self.__device)
                context.repetition_indices = torch.zeros(context.n, dtype=int, device=self.__device)
                return context
            else:
                raise Exception(
                    "Cannot start sampling from CrossProduct layer with num_repetitions > 1 and no context given."
                )

        # Map flattened indexes back to coordinates to obtain the chosen input_channel for each feature
        indices = self.unraveled_channel_indices[context.parent_indices]
        indices = indices.view(indices.shape[0], -1)

        # Remove padding
        if self._pad:
            indices = indices[:, : -self._pad]

        context.parent_indices = indices
        return context

    def __repr__(self):
        return "CrossProduct(in_features={}, out_shape={})".format(self.in_features, self.out_shape)
