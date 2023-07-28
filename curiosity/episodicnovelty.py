

import torch
import torch.optim as optim
import torch.nn.functional as F

import faiss
import numpy as np
from copy import deepcopy

from models import EmbeddingNet
from utils import RunningMeanStd


class EpisodicNovelty:
    """
    Different from original implementation, index resets after n timesteps
    """

    def __init__(self,
                 num_envs,
                 action_size,
                 N=10,
                 lr=5e-4,
                 kernel_epsilon=0.0001,
                 cluster_distance=0.008,
                 max_similarity=8.0,
                 c_constant=0.001,
                 device="cuda"
                 ):

        self.num_envs = num_envs

        # dimension is always 512
        model = EmbeddingNet(action_size=action_size).to(device)

        self.model = deepcopy(model)
        self.eval_model = deepcopy(model)

        self.opt = optim.Adam(self.model.parameters(), lr=lr)

        self.index = [faiss.IndexFlatL2(512) for _ in range(num_envs)]
        self.normalizer = RunningMeanStd()

        self.N = N
        self.kernel_epsilon = kernel_epsilon
        self.cluster_distance = cluster_distance
        self.max_similarity = max_similarity
        self.c_constant = c_constant

        self.counts = torch.zeros((num_envs,))

    def reset(self, id):
        self.index[id].reset()
        self.counts[id] = 0

    def add(self, ids, emb):
        for i, id in enumerate(ids):
            self.index[id].add(np.expand_dims(emb[i], 0))
            self.counts[id] += 1

    def knn_query(self, ids, emb):
        distances = []
        for i, id in enumerate(ids):
            distance, _ = self.index[id].search(np.expand_dims(emb[i], 0), self.N)
            distances.append(distance)

        distances = np.stack(distances)
        # mask out very big value placeholder by faiss to avoid overflow
        distances[distances > 1e30] = 0
        return distances

    @torch.no_grad()
    def get_reward(self, ids, obs, device="cpu"):
        ids = ids.cpu().numpy()
        emb = self.eval_model(obs).cpu().numpy()

        dist = self.knn_query(ids, emb)
        for id in ids:
            if self.counts[id] >= self.N:
                self.normalizer.update(dist[id].flatten())
        self.add(ids, emb)

        # Calculate kernel output
        distance_rate = dist / (self.normalizer.mean + 1e-8)

        distance_rate = np.maximum((distance_rate - self.cluster_distance), np.array(0.))
        kernel_output = self.kernel_epsilon / (distance_rate + self.kernel_epsilon)

        # Calculate denominator
        similarity = np.sqrt(np.sum(kernel_output)) + self.c_constant

        similarity = torch.tensor(similarity)
        mask = (self.counts < self.N) | torch.isnan(similarity) | (similarity > self.max_similarity)
        intr = torch.where(mask, 0., 1 / similarity)

        return intr.to(device)

    def update(self, obs1, obs2, actions):
        emb1 = self.model.forward(obs1)
        emb2 = self.model.forward(obs2)
        emb = torch.concat([emb1, emb2], dim=-1)
        logits = self.model.inverse(emb)

        self.opt.zero_grad()
        loss = F.cross_entropy(logits, actions.to(torch.int64).squeeze())
        loss = loss.mean()
        self.opt.step()

        return loss.item()

    def update_eval(self):
        self.eval_model.load_state_dict(self.model.state_dict())

