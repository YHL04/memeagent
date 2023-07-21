

import time
import numpy as np

from .replaybuffer import LocalBuffer
from environment import Env
from utils import tosqueeze


class Actor:
    """
    Class to be asynchronously run by Learner, use self.run() for main training
    loop. This class creates a local buffer to store data before sending completed
    Episode to Learner through rpc. All communication is through numpy array.


    Args:
        learner_rref (RRef): Learner RRef to reference the learner
        id (int): ID of the actor
        env_name (string): Environment name
        beta (float): initial beta value of actor
        discount (float): initial discount value of actor
    """

    def __init__(self, learner_rref, id, env_name, beta, discount):
        self.learner_rref = learner_rref
        self.id = id

        self.beta = beta
        self.discount = discount

        self.env = Env(env_name)
        self.local_buffer = LocalBuffer()

    def get_action(self, obs, state1, state2, beta):
        """
        Uses learner RRef and rpc async to call queue_request to get action
        from learner.

        Args:
            obs (List[np.array]): frames with shape (batch_size, n_channels, h, w)
            state1 (List[np.array]): recurrent states with shape (batch_size, state_len, d_model)
            state2 (List[np.array]): recurrent states with shape (batch_size, state_len, d_model)
            beta (List[np.array]): array of betas associated with each obs

        Returns:
            Future() object that when used with .wait(), halts until value is ready from
            the learner. Future() returns (action, prob, next_state1, next_state2, intr)

        """
        return self.learner_rref.rpc_async().queue_request(self.id, obs, state1, state2, beta)

    def return_episode(self, episode):
        """
        Once episode is completed return_episode uses learner_rref and rpc_async
        to call return_episode to return Episode object to learner for training.

        Args:
            id (int): Actor ID
            episode (Episode): Finished episode

        Returns:
            Future() object that when used with .wait(), halts until value is ready from
            the learner. Future() returns (new_beta, new_discount)

        """
        return self.learner_rref.rpc_async().return_episode(self.id, episode)

    def run(self):
        """
        Main actor training loop, calls queue_request to get action and
        return_episode to return finished episode

        TODO:
            finish batched actor
            adapt learner to batched actor
        """

        while True:
            obs = self.env.reset()
            state1 = (np.zeros((1, 512)), np.zeros((1, 512)))
            state2 = (np.zeros((1, 512)), np.zeros((1, 512)))

            start = time.time()
            done = False

            while not done:
                action, prob, next_state1, next_state2, intr = self.get_action(obs, state1, state2, self.beta).wait()

                next_obs, reward, done = self.env.step(action)

                self.local_buffer.add(obs, action, prob, reward, intr,
                                      tuple(map(tosqueeze, state1)),
                                      tuple(map(tosqueeze, state2))
                                      )

                obs = next_obs
                state1 = next_state1
                state2 = next_state2

            episode = self.local_buffer.finish(time.time()-start, self.beta, self.discount)
            self.beta, self.discount = self.return_episode(episode).wait()

