from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt

import gym
from gym.spaces import Discrete, Box, Space
from gym.utils import seeding

from baselines import logger
# from baselines import deepq

from util.generic import check_rng
from util.plot import plot_task_losses

from tree_search import TreeNode, TreeNodeShift
from tasks import ReluDropGenerator

np.set_printoptions(precision=2)


# Gym Spaces
class Sequence(Space):
    """Gym Space for index sequences."""

    def __init__(self, n):
        self.n = n
        super().__init__((n,), np.int)

    def sample(self):
        return self.np_random.permutation(self.n)

    def contains(self, x):
        return True if (np.sort(np.asarray(x, dtype=int)) == np.arange(self.n)).all() else False

    def __repr__(self):
        return f"Sequence({self.n})"

    def __eq__(self, other):
        return isinstance(other, Sequence) and self.n == other.n


class DiscreteSet(Space):
    """Gym Space for discrete, non-integral elements."""

    def __init__(self, elements):
        self.elements = np.sort(np.asarray(list(elements)).flatten())
        self.n = self.elements.size
        super().__init__((self.n,), self.elements.dtype)

    def sample(self):
        return self.np_random.choice(self.elements)

    def contains(self, x):
        return True if x in self.elements else False

    def __repr__(self):
        return f"DiscreteSet({self.elements})"

    def __eq__(self, other):
        return isinstance(other, DiscreteSet) and self.elements == other.elements


# Gym Environments
class BaseTaskingEnv(gym.Env):
    """Base environment for task scheduling."""

    def __init__(self, n_tasks, task_gen, n_ch, ch_avail_gen, cls_node=TreeNode, feature_funcs=None):
        self.n_tasks = n_tasks
        self.task_gen = task_gen

        self.n_ch = n_ch
        self.ch_avail_gen = ch_avail_gen

        self.feature_funcs = feature_funcs if feature_funcs is not None else {}

        self.cls_node = cls_node
        self.node = None
        self.reset()

        self.reward_range = (-float('inf'), 0)

        _low, _high = self.task_gen.param_rep_lim
        if self.cls_node == TreeNodeShift:      # FIXME: limits change for shift node class!
            _low = (_low[0], 0., _low[2], 0., 0.)   # FIXME: RELU specific!!
        self._param_rep_low = np.broadcast_to(np.array(_low), (self.n_tasks, len(_low)))
        self._param_rep_high = np.broadcast_to(np.array(_high), (self.n_tasks, len(_high)))

    @property
    def feature_names(self):
        return tuple(self.feature_funcs.keys())

    @property
    def _state_tasks(self):
        return np.array([task.gen_features(*self.feature_funcs.values()) for task in self.node.tasks])

    def reset(self, tasks=None, ch_avail=None, persist=False):
        if not persist:     # use new random (or user-specified) tasks/channels
            self.cls_node._tasks_init = self.task_gen(self.n_tasks) if tasks is None else tasks
            self.cls_node._ch_avail_init = self.ch_avail_gen(self.n_ch) if ch_avail is None else ch_avail

        self.node = self.cls_node()

    def step(self, action: list):
        raise NotImplementedError

    def render(self, mode='human'):
        if mode == 'human':
            fig_env, ax_env = plt.subplots(num='Task Scheduling Env', clear=True)
            plot_task_losses(self.node.tasks, ax=ax_env)

    def close(self):
        plt.close('all')


class SeqTaskingEnv(BaseTaskingEnv):
    """Tasking environment, entire sequence selected at once."""

    def __init__(self, n_tasks, task_gen, n_ch, ch_avail_gen, cls_node=TreeNode, feature_funcs=None):
        super().__init__(n_tasks, task_gen, n_ch, ch_avail_gen, cls_node, feature_funcs)
        self.observation_space = Box(self._param_rep_low, self._param_rep_high, dtype=np.float64)
        self.action_space = Sequence(self.n_tasks)

    def reset(self, tasks=None, ch_avail=None, persist=False):
        super().reset(tasks, ch_avail, persist)
        return self._state_tasks

    def step(self, action: list):
        self.node.seq = action
        reward = -1 * self.node.l_ex

        return None, reward, True, {}       # Episode is done, no observation


class StepTaskingEnv(BaseTaskingEnv):
    """Tasking environment, tasks scheduled sequentially."""

    # TODO: add option for task reordering?

    def __init__(self, n_tasks, task_gen, n_ch, ch_avail_gen, cls_node=TreeNode, feature_funcs=None,
                 seq_encoding='binary', masking=False):

        self.state_params = {'seq_encoding': seq_encoding,
                             'masking': masking}

        if self.state_params['seq_encoding'] == 'binary':
            self._state_seq_init = np.ones((n_tasks, 1))
        elif self.state_params['seq_encoding'] == 'one-hot':
            self._state_seq_init = np.zeros(2 * (n_tasks,))
        else:
            raise ValueError("Unrecognized state type.")
        self._state_seq = None

        super().__init__(n_tasks, task_gen, n_ch, ch_avail_gen, cls_node, feature_funcs)

        _state_low = np.concatenate((np.zeros(self._state_seq_init.shape), self._param_rep_low), axis=1)
        _state_high = np.concatenate((np.ones(self._state_seq_init.shape), self._param_rep_high), axis=1)
        self.observation_space = Box(_state_low, _state_high, dtype=np.float64)

        self.loss_agg = None

    @property
    def action_space(self):
        return DiscreteSet(self.node.seq_rem)

    @property
    def state(self):
        state_tasks = self._state_tasks
        if self.state_params['masking']:
            state_tasks[self.node.seq] = 0.
        return np.concatenate((self._state_seq, state_tasks), axis=1)

    def reset(self, tasks=None, ch_avail=None, persist=False):
        super().reset(tasks, ch_avail, persist)

        self.loss_agg = self.node.l_ex

        self._state_seq = self._state_seq_init.copy()
        return self.state

    def step(self, action: int):
        if self.state_params['seq_encoding'] == 'binary':
            self._state_seq[action][0] = 0
        elif self.state_params['seq_encoding'] == 'one-hot':
            self._state_seq[action][len(self.node.seq)] = 1
        else:
            raise ValueError("Unrecognized state type.")

        self.node.seq_extend([action])
        reward, self.loss_agg = self.loss_agg - self.node.l_ex, self.node.l_ex
        # reward = self.node.tasks[action](self.node.t_ex[action])      # TODO: reward OK for shift node?

        done = len(self.node.seq_rem) == 0

        return self.state, reward, done, {}


def wrap_agent(env, agent):
    """Generate scheduling function by running an agent on a single environment episode."""

    def scheduling_agent(tasks, ch_avail):
        observation, reward, done = env.reset(tasks, ch_avail), 0, False
        while not done:
            agent.action_space = env.action_space       # FIXME: hacked to allow proper StepTasking behavior
            action = agent.act(observation, reward, done)
            observation, reward, done, info = env.step(action)

        return env.node.t_ex, env.node.ch_ex

    return scheduling_agent


def wrap_agent_run_lim(env, agent):
    """Generate scheduling function by running an agent on a single environment episode, enforcing max runtime."""

    def scheduling_agent(tasks, ch_avail, max_runtime):

        t_run = perf_counter()

        observation, reward, done = env.reset(tasks, ch_avail), 0, False
        while not done:
            agent.action_space = env.action_space
            action = agent.act(observation, reward, done)
            observation, reward, done, info = env.step(action)

        runtime = perf_counter() - t_run
        if runtime >= max_runtime:
            raise RuntimeError(f"Algorithm timeout: {runtime} > {max_runtime}.")

        return env.node.t_ex, env.node.ch_ex

    return scheduling_agent


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


def main():
    def ch_avail_generator(n_ch, rng=check_rng(None)):  # channel availability time generator
        return rng.uniform(0, 2, n_ch)

    feature_dict = {'duration': lambda self: self.duration,
                    'release time': lambda self: self.t_release,
                    'slope': lambda self: self.slope,
                    'drop time': lambda self: self.t_drop,
                    'drop loss': lambda self: self.l_drop,
                    'is available': lambda self: self.t_release == 0.,
                    'is dropped': lambda self: self.t_release == 0. and self.t_drop == 0.
                    }

    params = {'n_tasks': 5,
              'task_gen': ReluDropGenerator(duration_lim=(3, 6), t_release_lim=(0, 4), slope_lim=(0.5, 2),
                                            t_drop_lim=(6, 12), l_drop_lim=(35, 50), rng=None),
              'n_ch': 2,
              'ch_avail_gen': ch_avail_generator,
              'cls_node': TreeNodeShift,
              'feature_funcs': feature_dict,
              'seq_encoding': 'binary',
              'masking': False}

    # env = SeqTaskingEnv(**params)
    env = StepTaskingEnv(**params)

    agent = RandomAgent(env.action_space)

    observation, reward, done = env.reset(), 0, False
    while not done:
        agent.action_space = env.action_space
        act = agent.act(observation, reward, done)
        print(observation)
        observation, reward, done, info = env.step(act)
        print(reward)

    # act = deepq.learn(task_env,
    #                   network='mlp',
    #                   lr=1e-3,
    #                   total_timesteps=100000,
    #                   buffer_size=50000,
    #                   exploration_fraction=0.1,
    #                   exploration_final_eps=0.02,
    #                   print_freq=10,
    #                   )


if __name__ == '__main__':
    main()
