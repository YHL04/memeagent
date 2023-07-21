

import numpy as np


class RunningMeanStd:
    """
    Computes running mean and std.
    """

    def __init__(self):
        self.mean = np.zeros((1,), dtype=np.float32)
        self.var = np.ones((1,), dtype=np.float32)
        self.count = 0

        self.deltas = []
        self.min_size = 10

    def update(self, x):
        assert len(x.shape) == 1
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)

        # update count
        n = x.shape[0]
        self.count += n

        # update mean
        delta = batch_mean - self.mean
        self.mean += delta * n / self.count

        # update var
        m_a = self.var * (self.count - n)
        m_b = batch_var * n
        M2 = m_a + m_b + np.square(delta) * n
        self.var = M2 / self.count

    def std(self):
        std = np.sqrt(self.var).item()
        assert type(std) == float
        return std

    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


