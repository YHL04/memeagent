

import torch
import torch.nn as nn
import torch.nn.functional as F

from .convnet import ConvNet
from .nfnet import NFNet
from .gtrxl import GTrXL


class Model(nn.Module):
    """
    From MEME paper, shared torso with separate heads with combined loss
    """

    def __init__(self, N, action_size, cls=ConvNet):
        super(Model, self).__init__()

        self.N = N
        self.action_size = action_size

        self.torso = Torso(cls)
        self.v_heads = nn.ModuleList([ValueHead(action_size) for _ in range(N)])
        self.p_heads = nn.ModuleList([PolicyHead(action_size) for _ in range(N)])

    def forward(self, x, state):
        """
        Args:
            x (B, C, H, W): batched observations
            state (Tuple(B, dim)): recurrent state

        Returns:
            q (B, N, action_size): q values associated with each policy
            state (Tuple(B, dim)): next recurrent state
        """
        x, state = self.torso(x, state)

        q, pi = [], []
        for v_head, p_head in zip(self.v_heads, self.p_heads):
            # detach x into policy head to stop policy gradient from flowing into torso
            _q, _p = v_head(x), p_head(x.detach())
            q.append(_q)
            pi.append(_p)

        q = torch.stack(q).transpose(0, 1)
        pi = torch.stack(pi).transpose(0, 1)

        return q, pi, state


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


class ValueHead(nn.Module):
    """
    Value Head in MEME paper, for each policy
    """

    def __init__(self, action_size):
        super(ValueHead, self).__init__()

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


class PolicyHead(nn.Module):
    """
    Policy Head in MEME paper, for each policy
    """

    def __init__(self, action_size):
        super(PolicyHead, self).__init__()

        self.linear1 = nn.Linear(512, 512)
        self.linear2 = nn.Linear(512, 512)
        self.out = nn.Linear(512, action_size)

    def forward(self, x):
        x = F.gelu(self.linear1(x))
        x = F.gelu(self.linear2(x))
        x = F.softmax(self.out(x), dim=-1)

        return x


class ParallelHead(nn.Module):
    """
    Parallelized Head in MEME Agent using Conv1d (3x Slower)
    Probably has to do with how Conv1d is computed
    """

    def __init__(self, N, action_size, dim=512):
        super(ParallelHead, self).__init__()
        self.N = N
        self.action_size = action_size
        self.dim = dim

        self.linear1 = nn.Linear(dim, N * dim)
        self.linear2 = nn.Conv1d(N, N * dim, kernel_size=dim, stride=1, groups=N)

        self.value = nn.Conv1d(N, N, kernel_size=dim, stride=1, groups=N)
        self.adv = nn.Conv1d(N, N * action_size, kernel_size=dim, stride=1, groups=N)

    def forward(self, x):
        B = x.size(0)

        # (B, dim)
        x = self.linear1(x).view(B, self.N, self.dim)
        # (B, N, dim)
        x = self.linear2(x)
        # (B, dim * N, 1)
        x = x.view(B, self.N, self.dim)
        # (B, N, dim)

        # Dueling architecture
        value = self.value(x)
        # (B, N, 1)
        adv = self.adv(x).view(B, self.N, self.action_size)
        # (B, N, action_size)
        x = value + (adv - torch.mean(adv, axis=-1, keepdim=True))

        # (B, N, action_size)
        return x

