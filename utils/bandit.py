

import numpy as np
import random


class UCB:
    """
    Upper Confidence Bound (UCB) Bandit Algorithm

    """

    def __init__(self,
                 num_arms,
                 window_size,
                 beta,
                 epsilon):

        self.num_arms = num_arms
        self.window_size = window_size
        self.beta = beta
        self.epsilon = epsilon

        self.rewards = np.zeros((window_size, num_arms), dtype=np.float32)
        self.counts = np.zeros((window_size, num_arms), dtype=np.int32)

        self.t = 0

    def update(self, arm, reward):
        idx = self.t % self.window_size
        self.t += 1

        self.rewards[idx, arm] = reward
        self.counts[idx, arm] = 1

    def sample(self):
        if self.t < self.num_arms:
            return self.t
        elif random.random() <= self.epsilon:
            return random.randrange(0, self.num_arms)
        else:
            i = min(self.t, self.window_size)
            rewards_sum = np.sum(self.rewards[:i], axis=0)
            count = np.sum(self.counts[:i], axis=0)
            mean = rewards_sum / (count + 1e-8)

            # Calculate values from UCB equation
            values = mean + self.beta * np.sqrt(1 / (count + 1e-8))

            return np.argmax(values)

