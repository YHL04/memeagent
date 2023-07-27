

import torch
import torch.nn as nn

import numpy as np

from environment import Env
from models import Model


def get_action(model, obs, state, device="cuda"):
    obs = torch.tensor(np.stack(obs), dtype=torch.float32, device=device).unsqueeze(0) / 255.

    q, pi, state = model(obs, state)
    return torch.argmax(pi[:, 0, :]).squeeze().item(), state


def main(env_name, path, device):
    env = Env(env_name, render_mode="human")
    model = nn.DataParallel(Model(N=2, action_size=env.action_size)).to(device)
    model.load_state_dict(torch.load(path))

    while True:
        obs = env.reset()
        state = (torch.zeros((1, 512)).to(device), torch.zeros((1, 512)).to(device))

        done = False
        total_reward = 0.

        while not done:
            action, next_state = get_action(model, obs, state)

            next_obs, reward, done = env.step(action)

            obs = next_obs
            state = next_state

            total_reward += reward

        print("Total Reward: ", total_reward)


if __name__ == "__main__":
    main(env_name="BreakoutDeterministic-v4",
         path="saved/final",
         device="cuda")

