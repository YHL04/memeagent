

import torch
import numpy as np


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def get_beta(i, N, beta):
    """
    Args:
        i (int): ID of actor associated with a combination of beta and discount
        N (int): N constant representing total different combinations of betas and discounts
        beta (float): Maximum beta value

    Returns:
        beta (float): Betas associated with each env
    """
    if i == 0:
        return 0
    elif i == N - 1:
        return beta
    else:
        x = 10 * (2 * i - (N - 2)) / (N - 2)
        return beta * sigmoid(x)


def get_discount(i, N, discount_max, discount_min):
    """
    Args:
        i (int): ID of actor associated with a combination of beta and discount
        N (int): N constant representing total different combinations of betas and discount
        discount_max (float): Maximum discount value
        discount_min (float): Minimum discount value

    Returns:
        discount (float): discount associated with each env
    """
    if N == 1:
        return discount_max

    numerator = (N - 1 - i) * np.log(1 - discount_max) + i * np.log(1 - discount_min)
    denominator = N - 1

    return 1 - np.exp(numerator / denominator)


def get_betas(N, beta):
    """
    Returns a list of betas according to N and beta
    """
    return torch.tensor([get_beta(i, N, beta) for i in range(N)])


def get_discounts(N, discount_max, discount_min):
    """
    Returns a list of discounts according to N, discount_min, and discount_max
    """
    return torch.tensor([get_discount(i, N, discount_min, discount_max) for i in range(N)])

def get_actor_exploration_epsilon(n: int):
    """Returns exploration epsilon for actor. This follows the Ape-x, R2D2 papers.

    Example for 4 actors: [0.4, 0.04715560318259695, 0.005559127278786369, 0.0006553600000000003]

    """
    assert 1 <= n
    return np.power(0.4, np.linspace(1.0, 8.0, num=n)).flatten().tolist()

