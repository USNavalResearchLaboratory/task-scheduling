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

    def __init__(self, ch_avail, tasks):
        self.n_ch = len(ch_avail)

        self.n_tasks = len(tasks)

        self.ch_avail = ch_avail
        self.tasks = tasks

        TreeNode._ch_avail_init = ch_avail
        TreeNode._tasks = tasks
        self.node = TreeNode([])

        self.observation_space = Discrete(0)
        self.action_space = Sequence(self.n_tasks)

        self.reward_range = (-float('inf'), 0)

    # def __init__(self, n_ch, ch_avail_gen, n_tasks, task_gen):
    #     self.n_ch = n_ch
    #     self.ch_avail_gen = ch_avail_gen
    #
    #     self.n_tasks = n_tasks
    #     self.task_gen = task_gen
    #
    #     self.ch_avail = None
    #     self.tasks = None
    #     self.node = None
    #
    #     _low, _high = list(zip(task_gen.duration_lim, task_gen.t_release_lim, task_gen.slope_lim,
    #                            task_gen.t_drop_lim, task_gen.l_drop_lim,))
    #     obs_low = np.broadcast_to(np.asarray(_low), (n_tasks, 5))
    #     obs_high = np.broadcast_to(np.asarray(_high), (n_tasks, 5))
    #
    #     self.observation_space = Box(obs_low, obs_high, dtype=np.float32)
    #     self.action_space = Sequence(n_tasks)
    #
    #     self.reward_range = (-float('inf'), 0)

    def reset(self):
        return 0

    # def reset(self):
    #     self.ch_avail = self.ch_avail_gen(self.n_ch)
    #     self.tasks = self.task_gen.rand_tasks(self.n_tasks)
    #
    #     TreeNode._ch_avail_init = self.ch_avail
    #     TreeNode._tasks = self.tasks
    #     self.node = TreeNode([])
    #
    #     return obs_relu_drop(self.tasks)

    def step(self, action: list):
        # obs = obs_relu_drop(self.tasks)
        obs = 0

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

    # def act(self, observation, reward, done):
    def act(self):
        return self.action_space.sample()


def random_agent(tasks, ch_avail):
    # env = TaskingEnv(n_ch, ch_avail_gen, n_tasks, task_gen)   # TODO: random environment initialization?
    env = TaskingEnv(ch_avail, tasks)
    agent = RandomAgent(env.action_space)

    observation = env.reset()
    action = agent.act()
    observation, reward, done, info = env.step(action)
    if done:
        return env.node.t_ex, env.node.ch_ex
