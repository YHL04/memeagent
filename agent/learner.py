

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
from utils import UCB, compute_retrace_loss, \
    rescale, inv_rescale, \
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
    beta = 0.3
    discount_max = 0.997
    discount_min = 0.99

    update_every = 400
    save_every = 400
    device = "cuda"

    bandit_window_size = 90
    bandit_beta = 1.0
    bandit_epsilon = 0.5

    def __init__(self,
                 env_name,
                 N,
                 size,
                 B,
                 burnin,
                 rollout
                 ):
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        self.size = size
        self.B = B
        self.N = N

        # models
        self.action_size = gym.make(env_name).action_space.n
        model = Model(action_size=self.action_size)

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
        self.replay_buffer = ReplayBuffer(size=size,
                                          B=B,
                                          T=burnin+rollout,
                                          beta=self.beta,
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

        # self.pending_rpc = None
        # self.await_rpc = False

        self.betas = get_betas(N, self.beta)
        self.discounts = get_discounts(N, self.discount_max, self.discount_min)

        self.controller = UCB(num_arms=self.N,
                              window_size=self.bandit_window_size,
                              beta=self.bandit_beta,
                              epsilon=self.bandit_epsilon
                              )

        self.actor_rref = self.spawn_actors(learner_rref=RRef(self),
                                            env_name=env_name,
                                            N=self.N,
                                            betas=self.betas,
                                            discounts=self.discounts
                                            )

        self.updates = 0

    @staticmethod
    def spawn_actors(learner_rref, env_name, N, betas, discounts):
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
                                    args=(learner_rref,
                                          i,
                                          env_name,
                                          betas[i],
                                          discounts[i]
                                          ),
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
    def get_policy(self, id, obs, state1, state2, beta):
        """
        Args:
            id (B,): actor IDs
            obs (B, c, h, w): batched observation
            state1 (Tuple(B, dim)): batched recurrent state 1
            state2 (Tuple(B, dim)): batched recurrent state 2
            beta (B,): each actor beta values

        Returns:
            action (B,): action indices
            prob (B,): action probabilities
            state1 (Tuple(B, dim)): batched recurrent state 1
            state2 (Tuple(B, dim)): batched recurrent state 2
            intr (B,): each actor intrinsic reward
        """
        B = id.size(0)

        self.epsilon -= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

        with self.lock_model:
            qe, qi, state1, state2 = self.eval_model(obs, state1, state2)
            q_values = rescale(inv_rescale(qe) + beta.unsqueeze(-1) * inv_rescale(qi))

            intr_e = self.episodic_novelty.get_reward(id, obs)
            intr_l = self.lifelong_novelty.get_reward(obs)
            intr = intr_e * intr_l

        if random.random() <= self.epsilon:
            action = torch.randint(0, self.action_size, size=(B,))
            prob = torch.full_like(action, self.epsilon / self.action_size)

            return action, prob, state1, state2, intr

        # get action and probability of that action according to Agent57 (pg 19)
        action = torch.argmax(q_values, dim=-1).squeeze()
        prob = torch.full_like(action, 1 - (self.epsilon * ((self.action_size - 1) / self.action_size)))

        return action, prob, state1, state2, intr

    def get_action(self, id, obs, state1, state2, beta):
        """
        Convert everything into tensor and into the right shape to pass
        into get_policy() and then convert everything back to numpy.

        Args:
            id (List(int)): ID of actor
            obs (List(np.array)): A list of observations in numpy.uint8
            state1 (List(Tuple(np.array)): A list of tuples of recurrent states in numpy
            state2 (List(Tuple(np.array)): A list of tuples of recurrent states in numpy
            beta (List(float)): The beta value of the actor

        Returns:
            action (np.array)
            prob (np.array)
            state1 (List(Tuple(np.array))
            state2 (List(Tuple(np.array))
            beta (int)
        """
        # state1 = List((h, c)) to batched (h, c)

        id = torch.tensor(id, device=self.device)
        obs = torch.tensor(np.stack(obs), dtype=torch.float32, device=self.device) / 255.

        # list of tuples to size 2 tuple of lists
        state1 = tuple(map(list, zip(*state1)))
        state2 = tuple(map(list, zip(*state2)))
        # concatenate lists inside tuple
        state1 = tuple(map(totensor, tuple(map(toconcat, state1))))
        state2 = tuple(map(totensor, tuple(map(toconcat, state2))))
        # tuple of two batched tensors
        beta = torch.tensor(beta, device=self.device)

        action, prob, state1, state2, beta = self.get_policy(id, obs, state1, state2, beta)

        # state1 = batched (h, c) to List((h, c))

        action = action.cpu().numpy()
        prob = prob.cpu().numpy()

        # convert states to numpy and separate each array into a list
        state1 = tuple(map(lambda x: list(np.moveaxis(np.expand_dims(x.cpu().numpy(), 1), 0, 0)), state1))
        state2 = tuple(map(lambda x: list(np.moveaxis(np.expand_dims(x.cpu().numpy(), 1), 0, 0)), state2))
        # turn tuple of two lists into lists of size 2 tuples
        state1 = list(map(list, zip(*state1)))
        state2 = list(map(list, zip(*state2)))

        beta = beta.cpu().numpy()

        return action, prob, state1, state2, beta

    def sample_controller(self, episode):
        # update controller's policy index with obtained extrinsic reward
        idx = self.betas.index(episode.beta)
        self.controller.update(idx, episode.total_extr)

        # sample new policy with new beta and discount
        new_idx = self.controller.sample()
        new_beta = self.betas[new_idx]
        new_discount = self.discounts[new_idx]

        return new_beta, new_discount

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

                # if self.await_rpc:
                #     self.await_rpc = False
                #
                #     future = self.future2
                #     self.future2 = Future()
                #     future.set_result(None)

                if self.request_rpcs_count == self.N:
                    results = self.get_action(*list(map(list, (zip(*self.request_rpcs)))))
                    self.request_rpcs = [None for _ in range(self.N)]
                    self.request_rpcs_count = 0

                    for i, result in enumerate(zip(*results)):
                        future = self.request_futures[i]
                        self.request_futures[i] = Future()
                        future.set_result(result)

                # clear self.request_futures to answer requests
                # for i in range(len(self.request_rpcs)):
                #     if self.request_rpcs[i] is not None:
                #         results = self.get_action(*self.request_rpcs[i])
                #         self.request_rpcs[i] = None
                #
                #         future = self.request_futures[i]
                #         self.request_futures[i] = Future()
                #         future.set_result(results)

                # if self.pending_rpc is not None:
                #     results = self.get_action(*self.pending_rpc)
                #     self.pending_rpc = None
                #
                #     future = self.future1
                #     self.future1 = Future()
                #     future.set_result(results)

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

            self.update(obs=block.obs,
                        actions=block.actions,
                        probs=block.probs,
                        extr=block.extr,
                        intr=block.intr,
                        states1=block.states1,
                        states2=block.states2,
                        dones=block.dones,
                        discounts=block.discounts,
                        idxs=block.idxs
                        )

    def update(self, obs, actions, probs, extr, intr, states1, states2, dones, discounts, idxs):
        """
        An update step. Performs a training step, update new recurrent states,
        hard update target model occasionally and transfer weights to eval model
        """
        loss, new_states1, new_states2 = self.train_step(
            obs=obs.cuda(),
            actions=actions.cuda(),
            probs=probs.cuda(),
            extr=extr.cuda(),
            intr=intr.cuda(),
            states1=(states1[0].cuda(), states1[1].cuda()),
            states2=(states2[0].cuda(), states2[1].cuda()),
            dones=dones.cuda(),
            discounts=discounts.cuda()
        )
        intr_loss = self.train_novelty_step(
            obs=obs.cuda(),
            actions=actions.cuda()
        )

        # reformat List[Tuple(Tensor, Tensor)] to array of shape (bsz, block_len+n_step, 2, dim)
        states11, states12 = zip(*new_states1)
        states11 = torch.stack(states11).transpose(0, 1).cpu().numpy()
        states12 = torch.stack(states12).transpose(0, 1).cpu().numpy()
        new_states1 = np.stack([states11, states12], 2)

        states21, states22 = zip(*new_states2)
        states21 = torch.stack(states21).transpose(0, 1).cpu().numpy()
        states22 = torch.stack(states22).transpose(0, 1).cpu().numpy()
        new_states2 = np.stack([states21, states22], 2)

        # update new states to buffer
        self.priority_queue.put((idxs, new_states1, new_states2, loss, intr_loss, self.epsilon))

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

    def train_step(self, obs, actions, probs, extr, intr, states1, states2, dones, discounts):
        """
        Accumulate gradients to increase batch size
        Gradients are cached for n_accumulate steps before optimizer.step()

        Args:
            obs (block+1, B, channels, h, w]): tokens
            actions (block+1, B): actions
            probs (block+1, B): probs
            extr (block, B): extrinsic rewards
            intr (block, B): extrinsic rewards
            states1 (B, dim): recurrent states
            states2 (B, dim): recurrent states
            dones (block+1, B): boolean indicating episode termination

        Returns:
            loss (float): Loss of critic model
            bert_loss (float): Loss of bert masked language modeling
            new_states1 (B, dim): for lstm
            new_states2 (B, dim): for lstm
        """

        with torch.no_grad():
            state1 = (states1[0].detach().clone(), states1[1].detach().clone())
            state2 = (states2[0].detach().clone(), states2[1].detach().clone())

            new_states1, new_states2 = [], []
            for t in range(self.burnin):
                new_states1.append((state1[0].detach(), state1[1].detach()))
                new_states2.append((state2[0].detach(), state2[1].detach()))

                _, _, state1, state2 = self.target_model(obs[t], state1, state2)

            target_q1, target_q2 = [], []
            for t in range(self.burnin, self.T+1):
                new_states1.append((state1[0].detach(), state1[1].detach()))
                new_states2.append((state2[0].detach(), state2[1].detach()))

                target_q1_, target_q2_, state1, state2 = self.target_model(obs[t], state1, state2)
                target_q1.append(target_q1_)
                target_q2.append(target_q2_)

            target_q1 = torch.stack(target_q1)
            target_q2 = torch.stack(target_q2)

        self.model.zero_grad()

        state1 = (states1[0].detach().clone(), states1[1].detach().clone())
        state2 = (states2[0].detach().clone(), states2[1].detach().clone())

        for t in range(self.burnin):
            _, _, state1, state2 = self.model(obs[t], state1, state2)

        q1, q2 = [], []
        for t in range(self.burnin, self.T+1):
            q1_, q2_, state1, state2 = self.model(obs[t], state1, state2)
            q1.append(q1_)
            q2.append(q2_)

        q1 = torch.stack(q1)
        q2 = torch.stack(q2)

        pi_t1 = F.softmax(q1, dim=-1)
        pi_t2 = F.softmax(q2, dim=-1)

        # not sure how variable discounts are trained
        # just pass it in for now
        # discount_t = (~dones).float() * discounts
        discount_t = (~dones).float() * 0.99

        extr_loss = compute_retrace_loss(
            q_t=q1,
            q_t1=target_q1[1:],
            a_t=actions[:-1],
            a_t1=actions[1:],
            r_t=extr,
            pi_t1=pi_t1[1:],
            mu_t1=probs[1:],
            discount_t=discount_t
        )
        intr_loss = compute_retrace_loss(
            q_t=q2,
            q_t1=target_q2[1:],
            a_t=actions[:-1],
            a_t1=actions[1:],
            r_t=intr,
            pi_t1=pi_t2[1:],
            mu_t1=probs[1:],
            discount_t=discount_t
        )

        loss = extr_loss + intr_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        loss = loss.item()
        return loss, new_states1, new_states2

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

