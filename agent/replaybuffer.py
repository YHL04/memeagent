

import torch
import threading
import time
import random
import numpy as np
from dataclasses import dataclass
from typing import List
from collections import deque

from .logger import Logger


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

    def __init__(self, size, B, T, N,
                 sample_queue, batch_queue, priority_queue
                 ):

        self.size = size
        self.B = B
        self.T = T
        self.N = N

        self.lock = threading.Lock()
        self.sample_queue = sample_queue
        self.batch_queue = batch_queue
        self.priority_queue = priority_queue

        self.buffer = deque()
        self.logger = Logger()

        self.frames = 0

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

            if not self.batch_queue.full() and len(self.buffer) != 0:
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
            self.frames += episode.length
            self.buffer.append(episode)

            while self.frames > self.size:
                self.frames -= self.buffer[0].length
                self.buffer.popleft()

            # logs
            self.logger.total_frames += episode.length
            self.logger.arm = episode.arm

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
            idxs = []

            for _ in range(self.B):
                buffer_idx = random.randrange(0, len(self.buffer))
                time_idx = random.randrange(0, self.buffer[buffer_idx].length-self.T+1)
                idxs.append([buffer_idx, time_idx])

                extr.append(self.buffer[buffer_idx].extr[time_idx:time_idx+self.T])
                intr.append(self.buffer[buffer_idx].intr[time_idx:time_idx+self.T])
                obs.append(self.buffer[buffer_idx].obs[time_idx:time_idx+self.T+1])
                actions.append(self.buffer[buffer_idx].actions[time_idx:time_idx+self.T+1])
                probs.append(self.buffer[buffer_idx].probs[time_idx:time_idx+self.T+1])
                states.append(self.buffer[buffer_idx].states[time_idx])
                dones.append(self.buffer[buffer_idx].dones[time_idx:time_idx+self.T])

            obs = torch.tensor(np.stack(obs), dtype=torch.float32) / 255.
            actions = torch.tensor(np.stack(actions), dtype=torch.int32)
            probs = torch.tensor(np.stack(probs), dtype=torch.float32)

            extr = torch.tensor(np.stack(extr), dtype=torch.float32)
            intr = torch.tensor(np.stack(intr), dtype=torch.float32)

            states = torch.tensor(np.stack(states), dtype=torch.float32)
            states = (states[:, 0, :], states[:, 1, :])

            dones = torch.tensor(np.stack(dones), dtype=torch.bool)

            obs = obs.transpose(0, 1)
            actions = actions.transpose(0, 1)
            probs = probs.transpose(0, 1)
            extr = extr.transpose(0, 1)
            intr = intr.transpose(0, 1)
            dones = dones.transpose(0, 1)

            assert obs.shape == (self.T+1, self.B, 4, 105, 80)
            assert actions.shape == (self.T+1, self.B)
            assert probs.shape == (self.T+1, self.B)
            assert extr.shape == (self.T, self.B)
            assert intr.shape == (self.T, self.B)
            assert states[0].shape == (self.B, 512) and states[1].shape == (self.B, 512)
            assert dones.shape == (self.T, self.B)

            block = Block(obs=obs,
                          actions=actions,
                          probs=probs,
                          extr=extr,
                          intr=intr,
                          states=states,
                          dones=dones,
                          idxs=idxs
                          )

        return block

    def update_priorities(self, idxs, states, loss, intr_loss, epsilon):
        """
        Update recurrent states from new recurrent states obtained during training
        with most up-to-date model weights

        Args:
            idxs (List[List[buffer_idx, time_idx]]): indices of states
        """
        assert states.shape == (self.B, self.T+1, 2, 512)

        with self.lock:

            # update new state for each sample in batch
            # for idx, state1, state2 in zip(idxs, states1, states2):
            #     buffer_idx, time_idx = idx
            #
            #     try:
            #         self.buffer[buffer_idx].states1[time_idx:time_idx+self.T+1] = state1
            #         self.buffer[buffer_idx].states2[time_idx:time_idx+self.T+1] = state2
            #
            #     except IndexError:
            #         pass
            #
            #     except ValueError:
            #         pass

            # logs
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

    def __init__(self):
        self.obs_buffer = []
        self.action_buffer = []
        self.prob_buffer = []
        self.extr_buffer = []
        self.intr_buffer = []
        self.state_buffer = []

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

    def finish(self, arm, total_time):
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
                       total_time=total_time
                       )
