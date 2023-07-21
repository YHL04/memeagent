

import gym

from collections import deque
import numpy as np
import random


def preprocess_frame(frame):
    frame = np.mean(frame, axis=2).astype(np.uint8)
    frame = frame[::2, ::2]

    # temporary
    # frame = frame.astype(np.float32)
    # frame /= 255.0

    return frame


class Env:

    def __init__(self, env_name, render_mode=None, no_op_max=10):
        self.env = gym.make(env_name, render_mode=render_mode)
        self.last_lives = 0

        self.fire = False
        self.no_op_max = no_op_max

        self.stack = deque(maxlen=4)

    @property
    def state_size(self):
        return self.env.observation_space.shape

    @property
    def action_size(self):
        return self.env.action_space.n

    def reset(self):
        for i in range(4):
            self.stack.append(np.zeros((105, 80)))

        self.fire = True
        frame, _ = self.env.reset()

        for i in range(random.randint(1, self.no_op_max)):
            frame, _, _, _, _ = self.env.step(1)

        self.stack.append(preprocess_frame(frame))
        return np.array(self.stack, dtype=np.float32)

    def step(self, action):
        frame, reward, _, terminal, info = self.env.step(action)

        if info["lives"] < self.last_lives:
            life_lost = True

        else:
            life_lost = terminal

        self.last_lives = info["lives"]

        self.stack.append(preprocess_frame(frame))
        frame = np.array(self.stack, dtype=np.float32)
        return frame, reward, life_lost

    def render(self):
        """Called at each timestep to render"""
        self.env.render()

