import time
from functools import partial
from types import MethodType
import dill

import numpy as np
import matplotlib.pyplot as plt

import gym
from gym.spaces import Discrete, Box, Space
from gym.utils import seeding

from baselines import logger
# from baselines import deepq

from util.generic import check_rng
from util.plot import plot_task_losses

from tree_search import TreeNode, TreeNodeShift, branch_bound
from tasks import ReluDropGenerator

# np.set_printoptions(precision=2)


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
        # self.n = len(self.elements)
        # super().__init__((self.n,), type(self.elements[0]))

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

    def __init__(self, n_tasks, task_gen, n_ch, ch_avail_gen, cls_node=TreeNode, features=None, sort_key=None):
        self.n_tasks = n_tasks
        self.task_gen = task_gen

        self.n_ch = n_ch
        self.ch_avail_gen = ch_avail_gen

        if features is not None:
            self.features = features
            _low, _high = self.features['lims'].transpose()
        else:
            self.features = np.array([], dtype=[('name', '<U16'), ('func', object), ('lims', np.float, 2)])
            _low, _high = self.task_gen.param_rep_lim

        if type(sort_key) == str:       # TODO: expand sorting functionality for simple string inputs
            attr_str = sort_key

            def sort_key(env, n):
                return getattr(env.node.tasks[n], attr_str)

        if callable(sort_key):
            self.sort_key = MethodType(sort_key, self)
        else:
            self.sort_key = None

        self.masking = False

        self._state_tasks_low = np.broadcast_to(_low, (self.n_tasks, len(_low)))
        self._state_tasks_high = np.broadcast_to(_high, (self.n_tasks, len(_high)))

        self.reward_range = (-float('inf'), 0)

        self.cls_node = cls_node
        self.node = None
        self.reset()

    @property
    def sorted_index(self):
        # return np.argsort([self.sort_key(n) for n in range(self.n_tasks)])
        if callable(self.sort_key):  # sort individual task states
            return np.argsort([self.sort_key(n) for n in range(self.n_tasks)])
        else:
            # return np.arange(self.n_tasks)
            return None

    @property
    def _state_tasks(self):
        state_tasks = np.array([task.gen_features(*self.features['func']) for task in self.node.tasks])
        if self.masking:
            state_tasks[self.node.seq] = 0.

        if callable(self.sort_key):  # sort individual task states
            return state_tasks[self.sorted_index]
        else:
            return state_tasks

    def reset(self, tasks=None, ch_avail=None, persist=False):
        if not persist:     # use new random (or user-specified) tasks/channels
            self.cls_node._tasks_init = self.task_gen(self.n_tasks) if tasks is None else tasks
            self.cls_node._ch_avail_init = self.ch_avail_gen(self.n_ch) if ch_avail is None else ch_avail

        self.node = self.cls_node()

    def step(self, action):
        if callable(self.sort_key):  # decode task index to original order
            action = self.sorted_index[action]

        self.node.seq_extend(action)  # updates sequence, loss, task parameters, etc.

    def render(self, mode='human'):
        if mode == 'human':
            fig_env, ax_env = plt.subplots(num='Task Scheduling Env', clear=True)
            plot_task_losses(self.node.tasks, ax=ax_env)

    def close(self):
        plt.close('all')


class SeqTaskingEnv(BaseTaskingEnv):        # TODO: rename subclasses?
    """Tasking environment, entire sequence selected at once."""

    def __init__(self, n_tasks, task_gen, n_ch, ch_avail_gen, cls_node=TreeNode, features=None, sort_key=None):
        super().__init__(n_tasks, task_gen, n_ch, ch_avail_gen, cls_node, features, sort_key)
        self.observation_space = Box(self._state_tasks_low, self._state_tasks_high, dtype=np.float64)
        self.action_space = Sequence(self.n_tasks)

    def reset(self, tasks=None, ch_avail=None, persist=False):
        super().reset(tasks, ch_avail, persist)
        return self._state_tasks

    def step(self, action: list):
        super().step(action)
        reward = -1 * self.node.l_ex

        return None, reward, True, {}       # Episode is done, no observation


class StepTaskingEnv(BaseTaskingEnv):
    """Tasking environment, tasks scheduled sequentially."""

    def __init__(self, n_tasks, task_gen, n_ch, ch_avail_gen,
                 cls_node=TreeNode, features=None, sort_key=None, seq_encoding='indicator', masking=False):

        self.seq_encoding = seq_encoding

        super().__init__(n_tasks, task_gen, n_ch, ch_avail_gen, cls_node, features, sort_key)
        self.masking = masking

        _state_low = np.concatenate((np.zeros(self._state_seq.shape), self._state_tasks_low), axis=1)
        _state_high = np.concatenate((np.ones(self._state_seq.shape), self._state_tasks_high), axis=1)
        self.observation_space = Box(_state_low, _state_high, dtype=np.float64)

        self.loss_agg = None

    @property
    def action_space(self):
        if callable(self.sort_key):
            seq_rem_sort = np.flatnonzero(np.isin(self.sorted_index, list(self.node.seq_rem)))
            return DiscreteSet(seq_rem_sort)
        else:
            return DiscreteSet(self.node.seq_rem)

    @property
    def _state_seq(self):
        if self.seq_encoding == 'indicator':
            state_seq = np.array([[0] if n in self.node.seq else [1] for n in range(self.n_tasks)])
        elif self.seq_encoding == 'one-hot':
            _eye = np.eye(self.n_tasks)
            state_seq = np.array([_eye[self.node.seq.index(n)] if n in self.node.seq else np.zeros(self.n_tasks)
                                  for n in range(self.n_tasks)])
        else:
            raise ValueError("Unrecognized state type.")

        if callable(self.sort_key):     # sort individual sequence states
            return state_seq[self.sorted_index]
        else:
            return state_seq

    @property
    def state(self):
        return np.concatenate((self._state_seq, self._state_tasks), axis=1)

    def reset(self, tasks=None, ch_avail=None, persist=False):
        super().reset(tasks, ch_avail, persist)
        self.loss_agg = self.node.l_ex

        return self.state

    def step(self, action: int):
        super().step(action)

        # TODO: different aggregate loss increments for shift node! Test both...
        reward, self.loss_agg = self.loss_agg - self.node.l_ex, self.node.l_ex
        # reward = self.node.tasks[action](self.node.t_ex[action])

        done = len(self.node.seq_rem) == 0

        return self.state, reward, done, {}


# Agents
class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


# Learning
def data_gen(env, n_gen=1):
    """Generate state-action data for learner training and evaluation."""

    if not isinstance(env, StepTaskingEnv):
        raise NotImplementedError("Tasking environment must be step Env.")

    # TODO: generate sample weights to prioritize earliest task selections??
    # TODO: train using complete tree info, not just B&B solution?

    x_gen = []
    y_gen = []
    for i_gen in range(n_gen):
        print(f'Task Set: {i_gen + 1}/{n_gen}', end='\n')

        env.reset()     # initializes environment state

        t_ex, ch_ex = branch_bound(env.node.tasks, env.node.ch_avail, verbose=True)
        seq = np.argsort(t_ex)     # optimal sequence

        # Generate samples for each scheduling step of the optimal sequence
        for n in seq:
            n_sort = env.sorted_index.tolist().index(n)

            x_gen.append(env.state.copy())
            y_gen.append(n_sort)

            env.step(n_sort)     # updates environment state

    return np.array(x_gen), np.array(y_gen)


def train_agent(n_tasks, task_gen, n_channels, ch_avail_gen,
                n_gen_train=0, n_gen_val=0, env_cls=StepTaskingEnv, env_params=None,
                save=False, save_dir=None):

    if env_params is None:
        env_params = {}

    # Create environment
    env = env_cls(n_tasks, task_gen, n_channels, ch_avail_gen, **env_params)

    # Generate state-action data pairs
    d_train = data_gen(env, n_gen_train)
    d_val = data_gen(env, n_gen_val)

    # Train agent
    agent = RandomAgent(env.action_space)

    if save:
        if save_dir is None:
            save_dir = 'temp/{}'.format(time.strftime('%Y-%m-%d_%H-%M-%S'))

        with open('./agents/' + save_dir, 'wb') as file:
            dill.dump({'env': env, 'agent': agent}, file)    # save environment

    return wrap_agent(env, agent)


def load_agent(load_dir):
    with open('./agents/' + load_dir, 'rb') as file:
        pkl_dict = dill.load(file)
    return wrap_agent(**pkl_dict)


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

        t_run = time.perf_counter()

        observation, reward, done = env.reset(tasks, ch_avail), 0, False
        while not done:
            agent.action_space = env.action_space
            action = agent.act(observation, reward, done)
            observation, reward, done, info = env.step(action)

        runtime = time.perf_counter() - t_run
        if runtime >= max_runtime:
            raise RuntimeError(f"Algorithm timeout: {runtime} > {max_runtime}.")

        return env.node.t_ex, env.node.ch_ex

    return scheduling_agent


def main():

    task_gen = ReluDropGenerator(duration_lim=(3, 6), t_release_lim=(0, 4), slope_lim=(0.5, 2),
                                 t_drop_lim=(6, 12), l_drop_lim=(35, 50), rng=None)

    def ch_avail_generator(n_ch, rng=check_rng(None)):  # channel availability time generator
        return rng.uniform(0, 2, n_ch)

    features = np.array([('duration', lambda self: self.duration, task_gen.duration_lim),
                         ('release time', lambda self: self.t_release, (0., task_gen.t_release_lim[1])),
                         ('slope', lambda self: self.slope, task_gen.slope_lim),
                         ('drop time', lambda self: self.t_drop, (0., task_gen.t_drop_lim[1])),
                         ('drop loss', lambda self: self.l_drop, (0., task_gen.l_drop_lim[1])),
                         ('is available', lambda self: 1 if self.t_release == 0. else 0, (0, 1)),
                         ('is dropped', lambda self: 1 if self.l_drop == 0. else 0, (0, 1)),
                         ],
                        dtype=[('name', '<U16'), ('func', object), ('lims', np.float, 2)])

    def sort_key(self, n):
        if n in self.node.seq:
            return float('inf')
        else:
            return self.node.tasks[n].t_release
            # return 1 if self.node.tasks[n].l_drop == 0. else 0
            # return self.node.tasks[n].l_drop / self.node.tasks[n].t_drop

    # sort_key = 't_release'

    params = {'n_tasks': 4,
              'task_gen': task_gen,
              'n_ch': 2,
              'ch_avail_gen': ch_avail_generator,
              'cls_node': TreeNodeShift,
              'features': features,
              'seq_encoding': 'one-hot',
              'masking': False,
              'sort_key': sort_key
              }

    # env = SeqTaskingEnv(**params)
    env = StepTaskingEnv(**params)

    agent = RandomAgent(env.action_space)

    observation, reward, done = env.reset(), 0, False
    while not done:
        print(observation)
        print(env.sorted_index)
        print(env.node.seq)
        print(env.node.tasks)
        agent.action_space = env.action_space
        act = agent.act(observation, reward, done)
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
