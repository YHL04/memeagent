

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

    NOTE:
        AssertionError: 133.14080000000007 66.5704 66.5704
        during _check_graph() caused by numerical precision
        this causes issues occasionally when errors accumulate
        the workaround is to use np.int64 array and multiply
        float by a million and round it off so there is no numerical
        precision errors that can cause a crash

    """

    def __init__(self, size, fill_value=0):
        """make sure size is powers of two and pad extra spots, not an optimal solution"""
        fill_value = round(fill_value * 1000000)

        orig_size = size
        size = 2 ** math.ceil(math.log(size, 2))

        self.size = size
        self.orig_size = orig_size
        self.tree = np.zeros((2 * size - 1,), dtype=np.int64)

        if fill_value != 0:
            for t in range(orig_size):
                self.update(t, fill_value)

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
            assert 0 == (self.tree[parent] - (self.tree[left] + self.tree[right])), error

    def total(self):
        return self.tree[0] / 1000000

    def update(self, data_idx, p):
        """changes tree[idx] to p and then propagate change through the tree"""
        p = round(p * 1000000)

        # convert data idx to tree idx
        idx = data_idx + self.size - 1

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
        s = round(s * 1000000)

        assert s != 0
        assert s < self.tree[0]
        idx, s = self.retrieve(0, s)
        data_idx = idx - self.size + 1

        assert self.tree[idx] != 0
        p = self.tree[idx] / 1000000
        s = s / 1000000
        return data_idx, p, s

    def retrieve(self, idx, s):
        """when retrieve -1 is padding for sum tree if episode timesteps are not powers of two"""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx, s

        if s <= self.tree[left] or self.tree[right] == 0:
            return self.retrieve(left, s)
        else:
            return self.retrieve(right, s - self.tree[left])


if __name__ == "__main__":
    b = SumTree(size=16, fill_value=8.3223)
    print(b._check_graph())
    print(b.tree)

    b = SumTree(size=16, fill_value=8.3212)
    print(b._check_graph())
    print(b.tree)

    b = SumTree(size=16, fill_value=8.7632)
    print(b._check_graph())
    print(b.tree)

    b = SumTree(size=16, fill_value=8.1122)
    print(b._check_graph())
    print(b.tree)

    b = SumTree(size=16, fill_value=8.5123)
    print(b._check_graph())
    print(b.tree)

    b = SumTree(size=16, fill_value=8.1842)
    print(b._check_graph())
    print(b.tree)

