

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef
from torch.distributed.rpc.functions import async_execution
from torch.futures import Future

import gym
import numpy as np
import threading
import time
import random
from copy import deepcopy

from .actor import Actor
from .replaybuffer import ReplayBuffer

from models import Model
from curiosity import EpisodicNovelty, LifelongNovelty
from utils import UCB, RunningMeanStd, \
    compute_loss, \
    compute_policy_loss, \
    get_betas, get_discounts, \
    totensor, toconcat


class Learner:
    """
    Main class used to train the agent. Called by rpc remote.
    Call run() to start the main training loop.

    Args:
        env_name (string): Environment name in gym[atari]
        size (int): The size of the buffer in ReplayBuffer
        B (int): Batch size for training
        burnin (int): Length of burnin, concept from R2D2 paper
        rollout (int): Length of rollout, concept from R2D2 paper

    """
    epsilon = 1
    epsilon_min = 0.1
    epsilon_decay = 0.0001

    lr = 1e-4
    weight_decay = 0.05
    adam_betas = (0.9, 0.999)
    adam_eps = 1e-8

    beta = 0.3
    discount_max = 0.997
    discount_min = 0.99
    tau = 0.25

    update_every = 400
    save_every = 400
    device = "cuda"

    bandit_window_size = 90
    bandit_beta = 1.0
    bandit_epsilon = 0.5

    def __init__(self, env_name, N, B, size, burnin, rollout):
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        self.N, self.B = N, B
        self.size = size

        # models
        self.action_size = gym.make(env_name).action_space.n
        model = Model(self.N, self.action_size)

        # episodic novelty module / lifelong novelty module
        self.episodic_novelty = EpisodicNovelty(N, self.action_size)
        self.lifelong_novelty = LifelongNovelty(N)

        self.model = nn.DataParallel(deepcopy(model)).cuda()
        self.target_model = nn.DataParallel(deepcopy(model)).cuda()
        self.eval_model = nn.DataParallel(deepcopy(model)).cuda()

        # set model modes
        self.model.train()
        self.target_model.eval()
        self.eval_model.eval()

        # locks
        self.lock = mp.Lock()
        self.lock_model = mp.Lock()

        # hyper-parameters
        self.burnin = burnin
        self.rollout = rollout
        self.T = burnin + rollout

        # optimizer and loss functions
        # self.opt = optim.Adam(self.model.parameters(),
        #                       lr=self.lr,
        #                       betas=self.adam_betas,
        #                       eps=self.adam_eps,
        #                       weight_decay=self.weight_decay
        #                       )
        self.opt = optim.Adam(self.model.parameters(), lr=self.lr)

        # queues
        self.sample_queue = mp.Queue()
        self.batch_queue = mp.Queue()
        self.priority_queue = mp.Queue()

        self.batch_queue = mp.Queue(8)
        self.priority_queue = mp.Queue(8)

        # params, batched_data (feeds batch), request_rpcs (answer calls)
        self.batch_data = []

        # start replay buffer
        self.replay_buffer = ReplayBuffer(max_frames=size,
                                          B=B,
                                          T=burnin+rollout,
                                          N=N,
                                          sample_queue=self.sample_queue,
                                          batch_queue=self.batch_queue,
                                          priority_queue=self.priority_queue
                                          )

        # start actors
        self.request_futures = [Future() for _ in range(N)]
        self.return_futures = [Future() for _ in range(N)]

        self.request_rpcs = [None for _ in range(N)]
        self.return_rpcs = [None for _ in range(N)]
        self.request_rpcs_count = 0

        self.betas = get_betas(N, self.beta)
        self.discounts = get_discounts(N, self.discount_max, self.discount_min)

        self.controller = UCB(num_arms=self.N,
                              window_size=self.bandit_window_size,
                              beta=self.bandit_beta,
                              epsilon=self.bandit_epsilon
                              )

        self.actor_rref = self.spawn_actors(learner_rref=RRef(self),
                                            env_name=env_name,
                                            N=N,
                                            T=self.T
                                            )

        self.running_errors = [RunningMeanStd() for _ in range(N)]

        self.updates = 0

    @staticmethod
    def spawn_actors(learner_rref, env_name, N, T):
        """
        Start actor by calling actor.remote().run()
        Actors communicate with learner through rpc and RRef

        Args:
            learner_rref (RRef): learner RRef for actor to reference the learner
            env_name (string): Name of environment


        Returns:
            actor_rref (RRef): to reference the actor from the learner
        """
        actor_rrefs = []

        for i in range(N):
            actor_rref = rpc.remote(f"actor{i}",
                                    Actor,
                                    args=(learner_rref, i, env_name, T),
                                    timeout=0
                                    )
            actor_rref.remote().run()
            actor_rrefs.append(actor_rref)

        return actor_rrefs

    @async_execution
    def queue_request(self, id, *args):
        """
        Called by actor asynchronously to queue requests

        Returns:
            future (Future.wait): Halts until value is ready
        """
        future = self.request_futures[id].then(lambda f: f.wait())
        with self.lock:
            self.request_rpcs[id] = (id, *args)
            self.request_rpcs_count += 1

        return future

    @async_execution
    def return_episode(self, id, episode):
        """
        Called by actor to asynchronously to return completed Episode
        to Learner

        Returns:
            future (Future.wait): Halts until value is ready
        """
        future = self.return_futures[id].then(lambda f: f.wait())
        self.sample_queue.put(episode)
        self.return_rpcs[id] = episode

        # Never Give Up
        with self.lock_model:
            self.episodic_novelty.reset(id)

        return future

    @torch.inference_mode()
    def get_policy(self, id, obs, state, arm):
        """
        Args:
            id (B,): actor IDs
            obs (B, c, h, w): batched observation
            state (Tuple(B, dim)): batched recurrent state
            arm (B,): each actor's bandit arm

        Returns:
            action (B,): action indices
            prob (B,): action probabilities
            state (Tuple(B, dim)): batched recurrent state
            intr (B,): each actor intrinsic reward
        """
        B = id.size(0)

        self.epsilon -= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

        with self.lock_model:
            _, pi, state = self.eval_model(obs, state)
            pi = pi[torch.arange(B), arm]

            intr_e = self.episodic_novelty.get_reward(id, obs)
            intr_l = self.lifelong_novelty.get_reward(obs)
            intr = intr_e * intr_l

        if random.random() <= self.epsilon:
            action = torch.randint(0, self.action_size, size=(B,))
            prob = torch.full_like(action, self.epsilon / self.action_size)

            return action, prob, state, intr

        # get action and probability of that action according to Agent57 (pg 19)
        action = torch.argmax(pi, dim=-1)
        prob = torch.full_like(action, 1 - (self.epsilon * ((self.action_size - 1) / self.action_size)))

        return action, prob, state, intr

    def get_action(self, id, obs, state, arm):
        """
        Convert everything into tensor and into the right shape to pass
        into get_policy() and then convert everything back to numpy.

        Args:
            id (List(int)): ID of actor
            obs (List(np.array)): A list of observations in numpy.uint8
            state (List(Tuple(np.array)): A list of tuples of recurrent states in numpy
            arm (List(float)): The bandit arm of the actor

        Returns:
            action (np.array)
            prob (np.array)
            state (List(Tuple(np.array))
            arm (int)
        """
        # state1 = List((h, c)) to batched (h, c)

        id = torch.tensor(id, device=self.device)
        arm = torch.tensor(arm, device=self.device)

        obs = torch.tensor(np.stack(obs), dtype=torch.float32, device=self.device) / 255.

        # list of tuples to size 2 tuple of lists
        state = tuple(map(list, zip(*state)))
        # concatenate lists inside tuple
        state = tuple(map(totensor, tuple(map(toconcat, state))))
        # tuple of two batched tensors

        action, prob, state, intr = self.get_policy(id, obs, state, arm)

        # state = batched (h, c) to List((h, c))

        action = action.cpu().numpy()
        prob = prob.cpu().numpy()

        # convert states to numpy and separate each array into a list
        state = tuple(map(lambda x: list(np.moveaxis(np.expand_dims(x.cpu().numpy(), 1), 0, 0)), state))
        # turn tuple of two lists into lists of size 2 tuples
        state = list(map(list, zip(*state)))

        return action, prob, state, intr

    def sample_controller(self, episode):
        # update controller's arm with obtained extrinsic reward
        self.controller.update(episode.arm, episode.total_extr)

        # sample new policy with new beta and discount
        return self.controller.sample()

    def answer_requests(self):
        """
        Thread to answer actor requests from queue_request and return_episode.
        Loops through with a time gap of 0.0001 sec
        """

        while True:
            time.sleep(0.0001)

            with self.lock:

                # clear self.return_futures to store episodes
                for i in range(len(self.return_rpcs)):
                    if self.return_rpcs[i] is not None:
                        future = self.return_futures[i]
                        self.return_futures[i] = Future()
                        future.set_result(self.sample_controller(self.return_rpcs[i]))

                        self.return_rpcs[i] = None

                if self.request_rpcs_count == self.N:
                    results = self.get_action(*list(map(list, (zip(*self.request_rpcs)))))
                    self.request_rpcs = [None for _ in range(self.N)]
                    self.request_rpcs_count = 0

                    for i, result in enumerate(zip(*results)):
                        future = self.request_futures[i]
                        self.request_futures[i] = Future()
                        future.set_result(result)

    def prepare_data(self):
        """
        Thread to prepare batch for update, batch_queue is filled by ReplayBuffer
        Loops through with a time gap of 0.01 sec
        """

        while True:
            time.sleep(0.001)

            if not self.batch_queue.empty() and len(self.batch_data) < 4:
                data = self.batch_queue.get_nowait()
                self.batch_data.append(data)

    def run(self):
        """
        Main training loop. Start ReplayBuffer threads, answer_requests thread,
        and prepare_data thread. Then starts training
        """
        self.replay_buffer.start_threads()

        inference_thread = threading.Thread(target=self.answer_requests, daemon=True)
        inference_thread.start()

        background_thread = threading.Thread(target=self.prepare_data, daemon=True)
        background_thread.start()

        while True:

            while not self.batch_data:
                time.sleep(0.001)
            block = self.batch_data.pop(0)

            self.update(block)

    def update(self, block):
        """
        An update step. Performs a training step, update new recurrent states,
        hard update target model occasionally and transfer weights to eval model
        """
        loss, new_states, error = self.train_step(
            obs=block.obs.cuda(),
            actions=block.actions.cuda(),
            probs=block.probs.cuda(),
            extr=block.extr.cuda(),
            intr=block.intr.cuda(),
            states=(block.states[0].cuda(), block.states[1].cuda()),
            dones=block.dones.cuda(),
            arms=block.arms.cuda(),
            is_weights=block.is_weights.cuda()
        )
        intr_loss = self.train_novelty_step(
            obs=block.obs.cuda(),
            actions=block.actions.cuda()
        )

        # reformat List[Tuple(Tensor, Tensor)] to array of shape (bsz, block_len+n_step, 2, dim)
        states1, states2 = zip(*new_states)
        states1 = torch.stack(states1).transpose(0, 1).cpu().numpy()
        states2 = torch.stack(states2).transpose(0, 1).cpu().numpy()
        new_states = np.stack([states1, states2], 2)

        error = error.sum(0).cpu().numpy()

        # update new states to buffer
        self.priority_queue.put((block.idxs, new_states, error, loss, intr_loss, self.epsilon))

        # hard update target model
        if self.updates % self.update_every == 0:
            self.hard_update(self.target_model, self.model)

        # save model
        if self.updates % self.save_every == 0:
            self.save(self.model)

        self.updates += 1

        # transfer weights to eval model
        with self.lock_model:
            self.hard_update(self.eval_model, self.model)
            self.episodic_novelty.update_eval()
            self.lifelong_novelty.update_eval()

        return loss, intr_loss

    def train_step(self, obs, actions, probs, extr, intr, states, dones, arms, is_weights):
        """
        Accumulate gradients to increase batch size
        Gradients are cached for n_accumulate steps before optimizer.step()

        Args:
            obs (T+1, B, channels, h, w]): tokens
            actions (T+1, B): actions
            probs (T+1, B): probs
            extr (T, B): extrinsic rewards
            intr (T, B): extrinsic rewards
            states (B, dim): recurrent states
            dones (T+1, B): boolean indicating episode termination
            arms (B,): arm index of each batch sample

        Returns:
            loss (float): Loss of critic model
            bert_loss (float): Loss of bert masked language modeling
            new_states (B, dim): for lstm
        """

        with torch.no_grad():
            state = (states[0].detach().clone(), states[1].detach().clone())

            new_states = []
            for t in range(self.burnin):
                new_states.append((state[0].detach(), state[1].detach()))

                _, _, state = self.target_model(obs[t], state)

            target_q, target_pi = [], []
            for t in range(self.burnin, self.T+1):
                new_states.append((state[0].detach(), state[1].detach()))

                target_q_, target_pi_, state = self.target_model(obs[t], state)
                target_q.append(target_q_)
                target_pi.append(target_pi_)

            target_q = torch.stack(target_q)
            target_pi = torch.stack(target_pi)

        self.model.zero_grad()

        state = (states[0].detach().clone(), states[1].detach().clone())

        for t in range(self.burnin):
            _, _, state = self.model(obs[t], state)

        q, pi = [], []
        for t in range(self.burnin, self.T+1):
            q_, pi_, state = self.model(obs[t], state)
            q.append(q_)
            pi.append(pi_)

        q = torch.stack(q)
        pi = torch.stack(pi)

        pi = F.softmax(torch.log(pi) / self.tau, dim=-1)
        target_pi = F.softmax(torch.log(target_pi) / self.tau, dim=-1)

        probs = probs.unsqueeze(-1).repeat(1, 1, self.N)
        rewards = extr.unsqueeze(-1) + self.betas.view(1, 1, self.N).to(extr.device) * intr.unsqueeze(-1)
        discount_t = (~dones).float().unsqueeze(-1) * self.discounts.view(1, 1, self.N).to(dones.device)
        actions = actions.unsqueeze(-1).repeat(1, 1, self.N)

        q_loss, error = compute_loss(
            q_t=q,
            qT_t=target_q[:-1],
            a_t=actions[self.burnin:-1],
            a_t1=actions[self.burnin+1:],
            r_t=rewards[self.burnin:],
            pi_t1=target_pi[1:],
            mu_t1=probs[self.burnin+1:],
            discount_t=discount_t[self.burnin:],
            arms=arms,
            running_errors=self.running_errors,
            is_weights=is_weights,
        )

        p_loss = compute_policy_loss(
            q_t=q,
            pi_t=pi,
            piT_t=target_pi
        )

        loss = q_loss + p_loss

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
        self.opt.step()

        loss = loss.item()
        return loss, new_states, error

    def train_novelty_step(self, obs, actions):
        emb_loss = self.train_emb_step(obs, actions)
        lifelong_loss = self.train_lifelong_step(obs)

        return emb_loss + lifelong_loss

    def train_emb_step(self, obs, actions):
        """
        Args:
            obs (torch.Tensor): shape (block+1, bsz, 4, 105, 80)
            actions (torch.Tensor): shape (block+1, bsz, 1)
        """

        # Get obs, next_obs and action, and flatten time and batch dimension
        obs1 = torch.flatten(obs[:-1, ...], 0, 1)
        obs2 = torch.flatten(obs[1:, ...], 0, 1)
        actions = torch.flatten(actions[:-1, ...], 0, 1)

        loss = self.episodic_novelty.update(obs1, obs2, actions)
        return loss

    def train_lifelong_step(self, obs):
        obs = torch.flatten(obs, 0, 1)

        loss = self.lifelong_novelty.update(obs)
        return loss

    @staticmethod
    def soft_update(target, source, tau):
        """
        Soft weight updates: target slowly track the weights of source with constant tau
        See DDPG paper page 4: https://arxiv.org/pdf/1509.02971.pdf

        Args:
            target (nn.Module): target model
            source (nn.Module): source model
            tau (float): soft update constant
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    @staticmethod
    def hard_update(target, source):
        """
        Copy weights from source to target

        Args:
            target (nn.Module): target model
            source (nn.Module): source model
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    @staticmethod
    def save(source, path="saved/final"):
        torch.save(source.state_dict(), path)

