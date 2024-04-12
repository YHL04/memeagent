

import torch
import torch.nn as nn
import torch.nn.functional as F

from nfnet import NFNet


class Model(nn.Module):
    """
    Agent57 Model with separate models for intrinsic and extrinsic reward.
    """

    def __init__(self, N, action_size):
        super(Model, self).__init__()
        self.N = N
        self.action_size = action_size

        self.torso = Torso()
        self.v_heads = nn.ModuleList([ValueHead(action_size) for _ in range(N)])
        self.p_heads = nn.ModuleList([PolicyHead(action_size) for _ in range(N)])

    def forward(self, x, state):
        x, state = self.torso(x, state)
        p_x = x.detach().clone()

        assert not torch.isnan(p_x).any(), p_x

        q, pi = [], []
        for v_head, p_head in zip(self.v_heads, self.p_heads):
            _q, _p = v_head(x), p_head(p_x)
            q.append(_q)
            pi.append(_p)

        q = torch.stack(q).transpose(0, 1)
        pi = torch.stack(pi).transpose(0, 1)

        return q, pi, state


class Torso(nn.Module):
    """
    Convolutional Neural Network for Atari with LSTM added on top
    for R2D2 Agent memory. Head uses dueling architecture.
    """

    def __init__(self):
        super(Torso, self).__init__()
        self.torso = ConvNet()
        # self.torso = NFNet(512)
        self.lstm = nn.LSTMCell(512, 512)

    def forward(self, x, state):
        x = self.torso(x)
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
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

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
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.out(x)

        return x


class EmbeddingNet(nn.Module):
    """
    Embedding Network first proposed in Never Give Up for exploration

    """

    def __init__(self, action_size):
        super(EmbeddingNet, self).__init__()

        self.convnet = ConvNet()

        self.inverse_head = nn.Sequential(
            nn.Linear(2 * 512, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )

    def forward(self, obs):
        emb = self.convnet(obs)
        return emb

    def inverse(self, emb):
        logits = self.inverse_head(emb)
        return logits


class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(3456, 512)

        # self.conv1 = nn.Linear(200, 512)
        # self.conv2 = nn.Linear(512, 512)
        # self.conv3 = nn.Linear(512, 512)
        # self.fc = nn.Linear(512, 512)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc(x))

        return x

