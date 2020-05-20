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

    def __init__(self, n_tasks, task_gen, n_ch, ch_avail_gen):
        self.n_tasks = n_tasks
        self.task_gen = task_gen

        self.n_ch = n_ch
        self.ch_avail_gen = ch_avail_gen

        self.ch_avail = None
        self.tasks = None
        self.node = None

        _low, _high = list(zip(task_gen.duration_lim, task_gen.t_release_lim, task_gen.slope_lim,
                               task_gen.t_drop_lim, task_gen.l_drop_lim,))
        obs_low = np.broadcast_to(np.asarray(_low), (n_tasks, 5))
        obs_high = np.broadcast_to(np.asarray(_high), (n_tasks, 5))

        self.observation_space = Box(obs_low, obs_high, dtype=np.float64)
        self.action_space = Sequence(n_tasks)

        self.reward_range = (-float('inf'), 0)

    def reset(self, tasks=None, ch_avail=None):     # TODO: added arguments to control Env state. OK?
        if tasks is None:
            self.tasks = self.task_gen.rand_tasks(self.n_tasks)
        else:
            self.tasks = tasks

        if ch_avail is None:
            self.ch_avail = self.ch_avail_gen(self.n_ch)
        else:
            self.ch_avail = ch_avail

        TreeNode._tasks = self.tasks
        TreeNode._ch_avail_init = self.ch_avail
        self.node = TreeNode([])

        return obs_relu_drop(self.tasks)

    def step(self, action: list):
        obs = obs_relu_drop(self.tasks)

        self.node.seq = action
        reward = -1 * self.node.l_ex

        return obs, reward, True, {}

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


def train_random_agent(n_tasks, task_gen, n_ch, ch_avail_gen):
    env = TaskingEnv(n_tasks, task_gen, n_ch, ch_avail_gen)
    agent = RandomAgent(env.action_space)

    def random_agent(tasks, ch_avail):
        observation, reward, done = env.reset(tasks, ch_avail), 0, False
        while not done:
            action = agent.act(observation, reward, done)
            observation, reward, done, info = env.step(action)

        return env.node.t_ex, env.node.ch_ex

    return random_agent
