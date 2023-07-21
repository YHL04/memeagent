

import torch
import numpy as np


def tonumpy(tensor):
    return tensor.cpu().numpy()


def tosqueeze(arr):
    return arr.squeeze()


def tounsqueeze(arr):
    return np.expand_dims(arr)


def totensor(arr, dtype=torch.float32, device="cuda"):
    return torch.tensor(arr, dtype=dtype, device=device)


def toconcat(lst, axis=0):
    return np.concatenate(lst, axis=axis)

