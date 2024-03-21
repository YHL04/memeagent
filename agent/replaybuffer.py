

import torch
import threading
import time
import random
import numpy as np
from dataclasses import dataclass
from typing import List

from utils import tosqueeze

from .logger import Logger
from .per import SumTree


@dataclass
class Episode:
    """
    Episode dataclass used to store completed episodes from actor
    """
    obs: np.array
    actions: np.array
    probs: np.array
    extr: np.array
    intr: np.array
    states: np.array
    dones: np.array
    length: int
    arm: int
    total_extr: float
    total_intr: float
    total_time: float
    signature: str


@dataclass
class Block:
    """
    Block dataclass used to store preprocessed batches for training
    """
    obs: torch.tensor
    actions: torch.tensor
    probs: torch.tensor
    extr: torch.tensor
    intr: torch.tensor
    states: torch.tensor
    dones: torch.tensor
    arms: torch.tensor
    is_weights: torch.tensor
    idxs: List[List[int]]


class ReplayBuffer:
    """
    Replay Buffer will be used inside Learner where start_threads is called
    before the main training the loop. The Learner will asynchronously queue
    Episodes into the buffer, logs the data, and prepare Block for training.

    Args:
        size (int): Size of self.buffer
        B (int): Training batch size
        T (int): Time step length of blocks
        N (int): N value in Never Give Up agent
        sample_queue (mp.Queue): FIFO queue to store Episode into ReplayBuffer
        batch_queue (mp.Queue): FIFO queue to sample batches for training from ReplayBuffer
        priority_queue (mp.Queue): FIFO queue to update new recurrent states from training to ReplayBuffer

    """
    e = 0.01
    p_a = 0.6
    p_beta = 0.4
    p_beta_increment_per_sampling = 0.001

    def __init__(self, max_frames, B, T, N,
                 sample_queue, batch_queue, priority_queue
                 ):

        self.max_frames = int(max_frames)
        self.size = int(max_frames)
        self.B = B
        self.T = T
        self.N = N

        self.lock = threading.Lock()
        self.sample_queue = sample_queue
        self.batch_queue = batch_queue
        self.priority_queue = priority_queue

        # Buffer
        self.buffer = np.empty((self.size,), dtype=object)
        self.ptr, self.count, self.n_entries, self.updates = 0, 0, 0, 0

        # Global Sum Tree
        self.sumtree = SumTree(self.size)

        # Logger
        self.logger = Logger()

        self.frames = 0
        self.total_frames = 0
        self.max_error = 1
        self.total_p = 0

    def get_priority(self, e):
        return np.abs(e) + self.e

    def __len__(self):
        return len(self.buffer)

    def start_threads(self):
        """Wrapper function to start all the threads in ReplayBuffer"""
        thread = threading.Thread(target=self.add_data, daemon=True)
        thread.start()

        thread = threading.Thread(target=self.prepare_data, daemon=True)
        thread.start()

        thread = threading.Thread(target=self.update_data, daemon=True)
        thread.start()

        thread = threading.Thread(target=self.log_data, daemon=True)
        thread.start()

    def add_data(self):
        """asynchronously add episodes to buffer by calling add()"""
        while True:
            time.sleep(0.001)

            if not self.sample_queue.empty():
                data = self.sample_queue.get_nowait()
                self.add(data)

    def prepare_data(self):
        """asynchronously add batches to batch_queue by calling sample_batch()"""
        while True:
            time.sleep(0.001)

            if not self.batch_queue.full() and self.count != 0:
                self.updates += 1

                data = self.sample_batch()
                self.batch_queue.put(data)

    def update_data(self):
        """asynchronously update states inside buffer by calling update_priorities()"""
        while True:
            time.sleep(0.001)

            if not self.priority_queue.empty():
                data = self.priority_queue.get_nowait()
                self.update_priorities(*data)

    def log_data(self):
        """asynchronously prints out logs and write into file by calling logs()"""
        while True:
            time.sleep(10)

            self.log()

    def add(self, episode):
        """Add Episode to self.buffer and update size, ptr, and logs"""

        with self.lock:

            # add to buffer

            # total frames received
            self.frames += episode.length

            # total frames in buffer
            self.total_frames += episode.length

            # remove episodes until total_frames can fit into max_frames
            while self.total_frames > self.max_frames:
                if self.buffer[-1] is not None:
                    self.total_frames -= self.buffer[-1].length
                    self.n_entries -= (self.buffer[-1].length - self.T)
                    self.count -= 1

                self.size -= 1
                self.sumtree.update(self.size, 0)
                self.buffer.resize(self.size, refcheck=False)
                assert len(self.buffer) == self.size

            if self.ptr >= self.size:
                self.ptr = 0

            # assert sum(self.sumtree.tree[len(self.buffer)+self.sumtree.size-1:]) == 0

            # create sum tree for priority queue
            episode.sumtree = SumTree(episode.length-self.T, fill_value=self.get_priority(self.max_error))
            # subtract entries that will be removed along with episode
            if self.buffer[self.ptr] is not None:
                self.n_entries -= episode.length-self.T
                self.total_frames -= episode.length
            # append episode to buffer
            self.buffer[self.ptr] = episode
            # update global sumtree
            assert episode.sumtree.total() != 0
            self.sumtree.update(self.ptr, episode.sumtree.total())

            # increment pointers and counts
            self.n_entries += episode.length-self.T

            self.count = min(self.count + 1, self.size)
            self.ptr += 1
            if self.ptr >= self.size:
                self.ptr = 0

            # logs
            self.logger.total_frames += episode.length
            self.logger.arm = episode.arm
            self.logger.replay_ratio = (self.updates * self.B * self.T) / self.frames

            # obtain extrinsic reward from purely exploitative policy
            if episode.arm == 0:
                self.logger.reward = episode.total_extr

            # obtain intrinsic reward from purely exploration policy
            if episode.arm == self.N - 1:
                self.logger.intrinsic = episode.total_intr

    def sample_batch(self):
        """
        Sample batch from buffer by sampling allocs, ids, actions, rewards, states, idxs.
        Then create bert targets from ids and precompute rewards with n step and gamma.
        Finally return finished Block for training.

        Returns:
            block (Block): completed block

        """

        with self.lock:

            obs = []
            actions = []
            probs = []
            extr = []
            intr = []
            states = []
            dones = []
            arms = []
            priorities = []
            idxs = []

            # increment per constants
            self.p_beta = min(1., self.p_beta + self.p_beta_increment_per_sampling)

            total = self.sumtree.total()
            segment = total / self.B

            for i in range(self.B):

                # sample from prioritized experience replay
                a = segment * i
                b = segment * (i + 1)
                s = random.uniform(a, b)

                b_idx, p, s = self.sumtree.get(s)
                assert b_idx < self.size
                t_idx, p, _ = self.buffer[b_idx].sumtree.get(s)
                assert t_idx < self.buffer[b_idx].length - self.T

                assert p >= 0
                priorities.append(p)
                idxs.append([b_idx, t_idx, self.buffer[b_idx].signature])

                extr.append(self.buffer[b_idx].extr[t_idx:t_idx+self.T])
                intr.append(self.buffer[b_idx].intr[t_idx:t_idx+self.T])
                obs.append(self.buffer[b_idx].obs[t_idx:t_idx+self.T+1])
                actions.append(self.buffer[b_idx].actions[t_idx:t_idx+self.T+1])
                probs.append(self.buffer[b_idx].probs[t_idx:t_idx+self.T+1])
                states.append(self.buffer[b_idx].states[t_idx])
                dones.append(self.buffer[b_idx].dones[t_idx:t_idx+self.T])
                arms.append(self.buffer[b_idx].arm)

            obs = torch.tensor(np.stack(obs), dtype=torch.float32) / 255.
            actions = torch.tensor(np.stack(actions), dtype=torch.int32)
            probs = torch.tensor(np.stack(probs), dtype=torch.float32)

            extr = torch.tensor(np.stack(extr), dtype=torch.float32)
            intr = torch.tensor(np.stack(intr), dtype=torch.float32)

            states = torch.tensor(np.stack(states), dtype=torch.float32)
            states = (states[:, 0, :], states[:, 1, :])

            dones = torch.tensor(np.stack(dones), dtype=torch.bool)
            arms = torch.tensor(arms, dtype=torch.int32)

            obs = obs.transpose(0, 1)
            actions = actions.transpose(0, 1)
            probs = probs.transpose(0, 1)
            extr = extr.transpose(0, 1)
            intr = intr.transpose(0, 1)
            dones = dones.transpose(0, 1)

            # prioritized experience replay
            assert torch.isnan(torch.tensor(priorities)).any() == False, priorities

            priorities = torch.tensor(priorities) ** self.p_a
            assert torch.isnan(priorities).any() == False, priorities
            sampling_probabilities = priorities / priorities.sum()
            assert priorities.sum() != 0
            assert torch.isnan(sampling_probabilities).any() == False, sampling_probabilities
            is_weights = torch.pow(self.n_entries * sampling_probabilities, -self.p_beta)
            assert is_weights.max() != 0
            assert torch.isnan(is_weights).any() == False, is_weights

            is_weights /= is_weights.max()
            assert torch.isnan(is_weights).any() == False

            assert obs.shape == (self.T+1, self.B, 4, 105, 80)
            assert actions.shape == (self.T+1, self.B)
            assert probs.shape == (self.T+1, self.B)
            assert extr.shape == (self.T, self.B)
            assert intr.shape == (self.T, self.B)
            assert states[0].shape == (self.B, 512) and states[1].shape == (self.B, 512)
            assert dones.shape == (self.T, self.B)
            assert arms.shape == (self.B,)
            assert is_weights.shape == (self.B,)

            block = Block(obs=obs,
                          actions=actions,
                          probs=probs,
                          extr=extr,
                          intr=intr,
                          states=states,
                          dones=dones,
                          arms=arms,
                          idxs=idxs,
                          is_weights=is_weights
                          )

        return block

    def update_priorities(self, idxs, states, errors, loss, intr_loss, epsilon):
        """
        Update recurrent states from new recurrent states obtained during training
        with most up-to-date model weights. Data are the training batch

        Args:
            idxs (List[List[b_idx, t_idx]]): indices of states
        """
        assert states.shape == (self.B, self.T+1, 2, 512)
        assert errors.shape == (self.B,)

        with self.lock:

            for (idx, state, error) in zip(idxs, states, errors):
                # b_idx is the index of episode and t_idx is the starting index of burnin
                b_idx, t_idx, signature = idx

                if b_idx >= self.size:
                    continue

                if signature != self.buffer[b_idx].signature:
                    continue

                assert np.abs(error) != np.nan, loss

                self.buffer[b_idx].sumtree.update(t_idx, self.get_priority(error))
                self.sumtree.update(b_idx, self.buffer[b_idx].sumtree.total())

                if error > self.max_error and np.abs(error) != np.nan:
                    self.max_error = error

            # update new state for each sample in batch
            # for idx, state1, state2 in zip(idxs, states1, states2):
            #     b_idx, t_idx = idx
            #
            #     try:
            #         self.buffer[b_idx].states1[t_idx:t_idx+self.T+1] = state1
            #         self.buffer[b_idx].states2[t_idx:t_idx+self.T+1] = state2
            #
            #     except IndexError:
            #         pass
            #
            #     except ValueError:
            #         pass

            # log
            self.logger.total_updates += 1
            self.logger.loss = loss
            self.logger.intr_loss = intr_loss
            self.logger.epsilon = epsilon

    def log(self):
        """
        Calls logger.print() to print out all the tracked values during training,
        lock to make sure its thread safe
        """

        with self.lock:
            self.logger.print()


class LocalBuffer:
    """
    Used by Actor to store data. Once the episode is finished
    finish() is called to return Episode to Learner to store in ReplayBuffer
    """

    def __init__(self, T):
        self.T = T
        self._reset()

    def _reset(self):
        self.obs_buffer = []
        self.action_buffer = []
        self.prob_buffer = []
        self.extr_buffer = []
        self.intr_buffer = []
        self.state_buffer = []

        for i in range(self.T-1):
            self.add_zeros()

    def add(self, obs, action, prob, extr, intr, state):
        """
        This function is called after every time step to store data into list

        Args:
            obs (Array): observed frame
            action (float): recorded action
            extr (float): recorded extrinsic reward
            intr (float): recorded intrinsic reward
            state (Array): recurrent state before model newly generated recurrent state
        """
        self.obs_buffer.append(obs)
        self.action_buffer.append(action)
        self.prob_buffer.append(prob)
        self.extr_buffer.append(extr)
        self.intr_buffer.append(intr)
        self.state_buffer.append(state)

    def add_zeros(self):
        """
        This function is called to pad for big burnin and rollouts
        """
        self.obs_buffer.append(np.zeros((4, 105, 80)))
        self.action_buffer.append(0)
        self.prob_buffer.append(0)
        self.extr_buffer.append(0)
        self.intr_buffer.append(0)
        self.state_buffer.append(tuple(map(tosqueeze, (np.zeros((1, 512)), np.zeros((1, 512))))))

    def finish(self, arm, total_time, signature):
        """
        This function is called after episode ends. lists are
        converted into numpy arrays and lists are cleared for
        next episode

        Args:
            arm (int): arm of actor
            total_time (float): total time for actor to complete episode in seconds

        """

        # pad obs, action, prob for retrace so that done can be included
        # since last element is not needed for retrace loss
        self.obs_buffer.append(np.zeros_like(self.obs_buffer[-1]))
        self.action_buffer.append(np.zeros_like(self.action_buffer[-1]))
        self.prob_buffer.append(np.zeros_like(self.prob_buffer[-1]))

        obs = np.stack(self.obs_buffer).astype(np.uint8)
        actions = np.stack(self.action_buffer).astype(np.int32)
        probs = np.stack(self.prob_buffer).astype(np.float32)
        extr = np.stack(self.extr_buffer).astype(np.float32)
        intr = np.stack(self.intr_buffer).astype(np.float32)
        states = np.stack(self.state_buffer).astype(np.float32)

        dones = np.zeros_like(extr)
        dones[-1] = 1
        dones = dones.astype(np.bool)

        length = len(extr)

        total_extr = np.sum(extr).item()
        total_intr = np.sum(intr).item()

        self.obs_buffer.clear()
        self.action_buffer.clear()
        self.prob_buffer.clear()
        self.extr_buffer.clear()
        self.intr_buffer.clear()
        self.state_buffer.clear()

        return Episode(obs=obs,
                       actions=actions,
                       probs=probs,
                       extr=extr,
                       intr=intr,
                       states=states,
                       dones=dones,
                       length=length,
                       arm=arm,
                       total_extr=total_extr,
                       total_intr=total_intr,
                       total_time=total_time,
                       signature=signature
                       )
