

import torch
import torch.nn as nn

from .convnet import ConvNet
from .nfnet import NFNet
from .gtrxl import GTrXL


class SingleModel(nn.Module):
    """
    Convolutional Neural Network for Atari with LSTM added on top
    for R2D2 Agent memory. Head uses dueling architecture.
    """

    def __init__(self, action_size, cls):
        super(SingleModel, self).__init__()
        self.torso = cls(512)
        self.lstm = nn.LSTMCell(512, 512)

        self.value = nn.Linear(512, 1)
        self.adv = nn.Linear(512, action_size)

    def forward(self, x, state):
        x = self.torso(x)
        x, state = self.lstm(x, state)
        state = (x, state)

        # Dueling network architecture
        value = self.value(x)
        adv = self.adv(x)
        x = value + (adv - torch.mean(adv, axis=-1, keepdim=True))

        return x, state


# class Model(nn.Module):
#     """
#     Agent57 Model with separate models for intrinsic and extrinsic reward.
#     """
#
#     def __init__(self, action_size, cls=NFNet):
#         super(Model, self).__init__()
#         self.extr = SingleModel(action_size, cls)
#         self.intr = SingleModel(action_size, cls)
#
#     def forward(self, x, state1, state2):
#         qe, state1 = self.extr(x, state1)
#         qi, state2 = self.intr(x, state2)
#
#         return qe, qi, state1, state2


class Torso(nn.Module):

    def __init__(self):
        pass

    def forward(self):
        pass


class Model(nn.Module):

    def __init__(self):
        pass

    def forward(self):
        pass

