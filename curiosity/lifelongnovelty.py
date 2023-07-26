

import torch
import torch.nn.functional as F
import torch.optim as optim

from models import ConvNet
from utils import RunningMeanStd


class LifelongNovelty:

    def __init__(self, lr=5e-4, L=5, device="cuda"):

        self.predictor = ConvNet(512).to(device)
        self.target = ConvNet(512).to(device)

        self.eval_predictor = ConvNet(512).to(device)
        self.eval_target = ConvNet(512).to(device)

        self.opt = optim.Adam(self.predictor.parameters(), lr=lr)

        self.normalizer = RunningMeanStd()

        self.L = L
        self.device = device

    def normalize_reward(self, reward):
        """Compute returns then normalize the intrinsic reward based on these returns"""

        self.normalizer.update(reward.cpu().numpy())

        norm_reward = reward / torch.sqrt(torch.tensor(self.normalizer.var, device=reward.device) + 1e-8)
        return norm_reward

    @torch.no_grad()
    def get_reward(self, obs, device="cpu"):

        # ngu paper normalizes obs but we dont since its already 0-1
        pred = self.predictor(obs)
        target = self.target(obs)

        reward = torch.square(pred - target).mean(-1)
        reward = self.normalize_reward(reward)
        reward = torch.minimum(torch.maximum(reward, torch.tensor(1., device=reward.device)),
                               torch.tensor(self.L, device=reward.device))

        return reward.to(device)

    def update(self, obs):
        target = self.predictor(obs)
        expected = self.target(obs)

        self.opt.zero_grad()
        loss = F.mse_loss(expected, target)
        loss = loss.mean()
        self.opt.step()

        return loss.item()

    def update_eval(self):
        self.eval_predictor.load_state_dict(self.predictor.state_dict())
        self.eval_target.load_state_dict(self.target.state_dict())

