#!/usr/bin/env python3
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

import torch
from torch import nn

def logsumexp(left, right, mask=None):
    """
    Source: https://github.com/pytorch/pytorch/issues/32097

    Logsumexp with custom scalar mask to allow for negative values in the sum.

    Args:
      left: First operand of the addition
      right: Second operand of the addition
      mask:  (Default value = None)

    Returns: Tensor (in log space) result of the addition via logsumexp

    """
    if mask is None:
        mask = torch.tensor([1, 1])
    else:
        assert mask.shape == (2,), "Invalid mask shape"

    maxes = torch.max(left, right)
    return maxes + ((left - maxes).exp() * mask[0] + (right - maxes).exp() * mask[1]).log()

@contextmanager
def provide_evidence(spn: nn.Module, evidence: torch.Tensor, requires_grad=False):
    """
    Context manager for sampling with evidence. In this context, the SPN graph is re-weighted with the likelihoods
    computed using the given evidence.

    Args:
        spn: SPN that is being used to perform the sampling.
        evidence: Provided evidence. The SPN will perform a forward pass prior to entering this contex.
        requires_grad: If False, runs in torch.no_grad() context. (default: False)
    """
    # If no gradients are required, run in no_grad context
    if not requires_grad:
        context = torch.no_grad
    else:
        # Else provide null context
        context = nullcontext

    # Run forward pass in given context
    with context():
        # Enter
        for module in spn.modules():
            if hasattr(module, "_enable_input_cache"):
                module._enable_input_cache()

        if evidence is not None:
            _ = spn(evidence)

        # Run in context (nothing needs to be yielded)
        yield

        # Exit
        for module in spn.modules():
            if hasattr(module, "_enable_input_cache"):
                module._disable_input_cache()


@dataclass
class SamplingContext:
    # Number of samples
    n: int = None

    # Indices into the out_channels dimension
    parent_indices: torch.Tensor = None

    # Indices into the repetition dimension
    repetition_indices: torch.Tensor = None

    # MPE flag, if true, will perform most probable explanation sampling
    is_mpe: bool = False

    def __setattr__(self, key, value):
        if hasattr(self, key):
            super().__setattr__(key, value)
        else:
            raise AttributeError(f"SamplingContext object has no attribute {key}")

    @property
    def is_root(self):
        return self.parent_indices == None and self.repetition_indices == None
