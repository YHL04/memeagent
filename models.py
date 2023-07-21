

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(3456, 512)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc(x))

        return x


class SingleModel(nn.Module):
    """
    Convolutional Neural Network for Atari with LSTM added on top
    for R2D2 Agent memory. Head uses dueling architecture.
    """

    def __init__(self, action_size):
        super(SingleModel, self).__init__()
        self.convnet = ConvNet()
        self.lstm = nn.LSTMCell(512, 512)

        self.value = nn.Linear(512, 1)
        self.adv = nn.Linear(512, action_size)

    def forward(self, x, state):
        x = self.convnet(x)
        x, state = self.lstm(x, state)
        state = (x, state)

        # Dueling network architecture
        value = self.value(x)
        adv = self.adv(x)
        x = value + (adv - torch.mean(adv, axis=-1, keepdim=True))

        return x, state


class Model(nn.Module):
    """
    Agent57 Model with separate models for intrinsic and extrinsic reward.
    """

    def __init__(self, action_size):
        super(Model, self).__init__()
        self.extr = SingleModel(action_size)
        self.intr = SingleModel(action_size)

    def forward(self, x, state1, state2):
        qe, state1 = self.extr(x, state1)
        qi, state2 = self.intr(x, state2)

        return qe, qi, state1, state2


class EmbeddingNet(nn.Module):
    """
    Embedding Network first proposed in Never Give Up for
    exploration

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

