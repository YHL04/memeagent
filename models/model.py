

import torch
import torch.nn as nn
import torch.nn.functional as F

from .convnet import ConvNet
from .nfnet import NFNet
from .gtrxl import GTrXL


# class SingleModel(nn.Module):
#     """
#     Convolutional Neural Network for Atari with LSTM added on top
#     for R2D2 Agent memory. Head uses dueling architecture.
#     """
#
#     def __init__(self, action_size, cls):
#         super(SingleModel, self).__init__()
#         self.torso = cls(512)
#         self.lstm = nn.LSTMCell(512, 512)
#
#         self.value = nn.Linear(512, 1)
#         self.adv = nn.Linear(512, action_size)
#
#     def forward(self, x, state):
#         x = self.torso(x)
#         x, state = self.lstm(x, state)
#         state = (x, state)
#
#         # Dueling network architecture
#         value = self.value(x)
#         adv = self.adv(x)
#         x = value + (adv - torch.mean(adv, axis=-1, keepdim=True))
#
#         return x, state


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

    def __init__(self, cls):
        super(Torso, self).__init__()

        self.body = cls(512)
        self.lstm = nn.LSTMCell(512, 512)

    def forward(self, x, state):
        x = self.body(x)
        x, state = self.lstm(x, state)
        state = (x, state)

        return x, state


class Head(nn.Module):

    def __init__(self, action_size):
        super(Head, self).__init__()

        self.linear1 = nn.Linear(512, 512)
        self.linear2 = nn.Linear(512, 512)

        self.value = nn.Linear(512, 1)
        self.adv = nn.Linear(512, action_size)

    def forward(self, x):
        x = F.gelu(self.linear1(x))
        x = F.gelu(self.linear2(x))

        # Dueling architecture
        value = self.value(x)
        adv = self.adv(x)
        x = value + (adv - torch.mean(adv, axis=-1, keepdim=True))

        return x


class Model(nn.Module):
    """
    From MEME paper, shared torso with separate heads with combined loss
    """

    def __init__(self, N, action_size, cls=ConvNet):
        super(Model, self).__init__()

        self.N = N
        self.action_size = action_size

        self.torso = Torso(cls)
        self.heads = nn.ModuleList([Head(action_size) for _ in range(N)])

    def forward(self, x, state):
        """
        TODO:
            parallelize heads using nn.conv2d

        Args:
            x (B, C, H, W): batched observations
            state (Tuple(B, dim)): recurrent state

        Returns:
            q (B, N, action_size): q values associated with each policy
            state (Tuple(B, dim)): next recurrent state
        """
        x, state = self.torso(x, state)

        q = []
        for head in self.heads:
            _q = head(x)
            q.append(_q)

        q = torch.stack(q).transpose(0, 1)

        return q, state
