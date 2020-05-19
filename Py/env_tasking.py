import math
from itertools import permutations
import numpy as np
import matplotlib.pyplot as plt

import gym
from gym.spaces import Discrete, Box, Space
from gym.utils import seeding

from baselines import logger

from tree_search import TreeNode
from tasks import ReluDropGenerator
from util.plot import plot_task_losses


def obs_relu_drop(tasks):
    """Convert tasks list into Gym observation."""

    # _params = [(task.duration, task.t_release, task.slope, task.t_drop, task.l_drop) for task in tasks]
    # params = np.array(_params, dtype=[('duration', np.float), ('t_release', np.float),
    #                                   ('slope', np.float), ('t_drop', np.float), ('l_drop', np.float)])
    # params.view(np.float).reshape(*params.shape, -1)
    return np.asarray([[task.duration, task.t_release, task.slope, task.t_drop, task.l_drop] for task in tasks])


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


class TaskingEnv(gym.Env):

    def __init__(self, task_gen, ch_avail_gen):     # TODO: specify task parameters during initialization
        self.episode = 0 
        self.counter = 0 
        self.agg = 0

        self.task_gen = task_gen
        self.tasks = self.task_gen()
        self.n_tasks = len(self.tasks)

        self.ch_avail_gen = ch_avail_gen
        self.ch_avail = self.ch_avail_gen()

        self.obs = obs_relu_drop(self.tasks)

        TreeNode._tasks = self.tasks
        TreeNode._ch_avail_init = self.ch_avail

        self.observation_space = Box(low=0, high=10, shape=self.obs.shape, dtype=np.float32)     # TODO: proper bounds
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
        # self.tasks, params = ReluDropGenerator().rand_tasks(self.n_tasks)
        # self.obs = params.view(np.float).reshape(*params.shape, -1)
        return self.obs

    def render(self, mode='human'):
        fig_env, ax_env = plt.subplots(num='Task Scheduling Env', clear=True)
        plot_task_losses(self.tasks, ax=ax_env)

    # def close(self):
    #     plt.close('all')


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()
