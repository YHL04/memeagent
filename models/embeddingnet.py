

import torch.nn as nn

from .convnet import ConvNet
from .nfnet import NFNet
from .gtrxl import GTrXL


class EmbeddingNet(nn.Module):
    """
    Embedding Network first proposed in Never Give Up for exploration

    """

    def __init__(self, action_size, cls=ConvNet):
        super(EmbeddingNet, self).__init__()
        self.torso = cls(512)

        self.inverse_head = nn.Sequential(
            nn.Linear(2 * 512, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )

    def forward(self, obs):
        emb = self.torso(obs)
        return emb

    def inverse(self, emb):
        logits = self.inverse_head(emb)
        return logits

