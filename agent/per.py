

import numpy as np
import math


class SumTree:
    """
    Sum Tree to store all the priorities for Prioritized Experience Replay
    Use it for each individual episode so that ReplayBuffer can choose
    experience according to priorities at the buffer level and priorities
    within each episode.

    Sum Tree is used twice, once for individual episodes in buffer, and another
    for global Sum Tree to keep track of total priorities and retrieval over
    the entire buffer.

    Need to review for bugs...
    """
    e = 0.01

    def __init__(self, size, max_error, fill_zero=False):
        """make sure size is powers of two and pad extra spots, not an optimal solution"""
        orig_size = size
        size = 2 ** math.ceil(math.log(size, 2))

        self.size = size
        self.orig_size = orig_size
        self.tree = np.zeros((2 * size - 1,), dtype=np.float32)

        if not fill_zero:
            for t in range(orig_size):
                self.add(t, max_error, got_p=False)

        # self._check_graph()

    def _check_graph(self):
        """testing function to make sure that the tree is constructed correctly"""
        for idx in range(1, len(self.tree)):
            parent = (idx - 1) // 2
            left = 2 * parent + 1
            right = left + 1

            # make sure that parent is the sum of left and right
            # and answer is within 1e-6 to account for np.float32 precision when comparing
            error = str(self.tree[parent]) + " " + str(self.tree[left]) + " " + str(self.tree[right])
            assert 1e-3 > (self.tree[parent] - (self.tree[left] + self.tree[right])), error

    def total(self):
        return self.tree[0]

    def get_priority(self, e):
        # proportional prioritization
        return np.abs(e) + self.e

    def add(self, idx, p, got_p):
        self.update(idx, p, got_p=got_p)

        # self._check_graph()

    def update(self, idx, e, got_p):
        """changes tree[idx] to p and then propagate change through the tree"""
        # convert data idx to tree idx
        idx = idx + self.size - 1

        # get priority from equation
        if not got_p:
            p = self.get_priority(e)
        else:
            p = e

        change = p - self.tree[idx]

        self.tree[idx] = p
        self.propagate(idx, change)

    def propagate(self, idx, change):
        """recursively propagate up the tree"""
        parent = (idx - 1) // 2
        self.tree[parent] += change

        # if not root node, do recursion
        if parent != 0:
            self.propagate(parent, change)

    def get(self, s):
        """get index of data and its priority"""
        # log = str(s) + " " + str(self.total())
        # assert s <= self.total(), log

        idx, s = self.retrieve(0, s)

        data_idx = idx - self.size + 1

        # assert data_idx < self.orig_size
        # assert s <= self.tree[idx]

        return data_idx, self.tree[idx], s

    def retrieve(self, idx, s):
        """when retrieve -1 is padding for sum tree if episode timesteps are not powers of two"""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx, s

        # temporary fix for a bug that happens infrequently ( or self.tree[right] == 0: )
        # BUG: buffer sum tree sometimes retrieve a leaf node with priority 0
        # best explanation is due to numerical precision of np.float32, occasionally
        # messes up comparison of s and node (Example: 6.0600004 = 4.04 + 2.02)
        if s <= self.tree[left] or self.tree[right] == 0:
            return self.retrieve(left, s)
        else:
            return self.retrieve(right, s - self.tree[left])
