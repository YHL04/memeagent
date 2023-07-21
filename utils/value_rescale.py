

import torch


def rescale(x, eps=1e-3):
    """
    Rescaling the expected q values with the following function:
    h(z) = sign(z) * sqrt(abs(z) + 1) - 1) + eps * z

    Proposed in R2D2: https://openreview.net/pdf?id=r1lyTjAqYX (pg 3)

    Args:
        x (torch.tensor): value to be rescaled
        eps (int=1e-3): small constant

    Returns:
        x (torch.tensor): rescaled value
    """

    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + eps * x


def inv_rescale(x, eps=1e-3):
    """
    Inverse of h(x) function

    Derived from:
    https://github.com/deepmind/rlax/blob/master/rlax/_src/transforms.py

    *NGU paper derivation (pg 20) is missing a torch.square() causing confusion

    Args:
        z (torch.tensor): value to be inverse rescaled
        eps (1e-3): small constant

    Returns:
        z (torch.tensor): inverse rescaled value
    """
    z = torch.sqrt(1 + 4 * eps * (eps + 1 + torch.abs(x))) / 2 / eps - 1 / 2 / eps
    return torch.sign(x) * (torch.square(z) - 1)

    # NGU paper is wrong
    # numerator = torch.sqrt(1 + 4 * eps * (torch.abs(z) + 1 + eps)) - 1
    # denominator = 2 * eps

    # z = torch.sign(z) * ((numerator / denominator) - 1)
    # return z
