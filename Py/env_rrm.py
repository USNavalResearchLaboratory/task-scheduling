import math
from itertools import permutations
import numpy as np

import gym
from gym.spaces import Discrete, Box, Space
from gym.utils import seeding

from baselines import logger

from tree_search import TreeNode
from tasks import ReluDropGenerator


class Sequence(Space):
    """Gym Space for index sequences."""

    def __init__(self, n):
        self.n = n
        super().__init__((n,), np.int)

    def sample(self):
        return self.np_random.permutation(self.n)

    def contains(self, x):
        if (np.sort(np.asarray(x, dtype=int)) == np.arange(self.n)).all():
            return True
        else:
            return False

    def __repr__(self):
        return f"Sequence({self.n})"

    def __eq__(self, other):
        return isinstance(other, Sequence) and self.n == other.n


class RadarTaskEnv(gym.Env):

    def __init__(self):     # TODO: specify task parameters during initialization
        self.episode = 0 
        self.counter = 0 
        self.agg = 0

        self.ch_avail = 2 * [0]

        self.n_tasks = 8
        self.task_generator = ReluDropGenerator()

        self.tasks, params = self.task_generator.rand_tasks(self.n_tasks, return_params=True)
        self.obs = params.view(np.float).reshape(*params.shape, -1)

        TreeNode._tasks = self.tasks
        TreeNode._ch_avail_init = self.ch_avail

        self.observation_space = Box(low=0, high=10, shape=(self.n_tasks, 5), dtype=np.float32)     # TODO: proper bounds
        self.action_space = Sequence(self.n_tasks)

        self.reward_range = (-float('inf'), 0)

    def step(self, action: list):
        # self.counter += 1
        # self.episode += 1
        # logger.record_tabular("action", action)

        reward = -1 * TreeNode(action).l_ex

        # self.agg += reward
        # if self.counter % 100 == 0:
        #     print('Average last  100 rewards: ', self.agg/100)
        #     self.counter = 0
        #     self.agg = 0
        #     print('episode', self.episode)

        return self.obs, reward, True, {}

    def reset(self):
        self.tasks, params = ReluDropGenerator().rand_tasks(self.n_tasks, return_params=True)
        self.obs = params.view(np.float).reshape(*params.shape, -1)
        return self.obs

    def render(self, mode='human'):
        pass    # some kind of plot function


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()
