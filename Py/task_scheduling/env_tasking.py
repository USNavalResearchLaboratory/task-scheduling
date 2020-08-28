import time
from copy import deepcopy
from types import MethodType
import dill

import numpy as np
import matplotlib.pyplot as plt
import gym
from gym.spaces import Box, Space

from util.plot import plot_task_losses
from generators.scheduling_problems import Random as RandomProblem
from tree_search import TreeNode, TreeNodeShift


# Gym Spaces
class Sequence(Space):
    """Gym Space for index sequences."""

    def __init__(self, n):
        self.n = n      # sequence length
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
        self.elements = np.sort(np.array(list(elements)).flatten())   # ndarray representation of set
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
    """
    Base environment for task scheduling.

    Parameters
    ----------
    problem_gen : generators.scheduling_problems.Base
        Scheduling problem generation object.
    node_cls : TreeNode or callable
        Class for tree search node generation.
    features : ndarray, optional
        Structured numpy array of features with fields 'name', 'func', and 'lims'.
    sort_func : function or str, optional
        Method that returns a sorting value for re-indexing given a task index 'n'.
    masking : bool
        If True, features are zeroed out for scheduled tasks.

    """

    def __init__(self, problem_gen, node_cls=TreeNode, features=None, sort_func=None, masking=False):
        self.problem_gen = problem_gen
        self.solution = None

        self.n_tasks = self.problem_gen.n_tasks
        self.n_ch = self.problem_gen.n_ch

        # Set features and state bounds
        if features is not None:
            self.features = features
        else:
            self.features = self.problem_gen.task_gen.default_features
            # _task_param_names = self.problem_gen.task_gen.cls_task.param_names
            # self.features = np.array(list(zip(_task_param_names,
            #                                   [lambda task, name=_name: getattr(task, name)
            #                                    for _name in _task_param_names],     # note: late-binding closure
            #                                   self.problem_gen.task_gen.param_lims.values())),
            #                          dtype=[('name', '<U16'), ('func', object), ('lims', np.float, 2)])

        _low, _high = zip(*self.features['lims'])
        self._state_tasks_low = np.broadcast_to(_low, (self.n_tasks, len(_low)))
        self._state_tasks_high = np.broadcast_to(_high, (self.n_tasks, len(_high)))

        # Set sorting method
        if callable(sort_func):
            self.sort_func = MethodType(sort_func, self)
        elif type(sort_func) == str:
            def _sort_func(env, n):
                return getattr(env.tasks[n], sort_func)

            self.sort_func = MethodType(_sort_func, self)
        else:
            self.sort_func = None

        self.masking = masking

        self.reward_range = (-float('inf'), 0)
        self.loss_agg = None

        self.node_cls = node_cls
        self.node = None

    tasks = property(lambda self: self.node.tasks)
    ch_avail = property(lambda self: self.node.ch_avail)

    @property
    def sorted_index(self):
        """Indices for task re-ordering for environment state."""
        if callable(self.sort_func):
            return np.argsort([self.sort_func(n) for n in range(self.n_tasks)])
        else:
            return np.arange(self.n_tasks)

    @property
    def state_tasks(self):
        """State sub-array for task features."""
        state_tasks = np.array([task.feature_gen(*self.features['func']) for task in self.tasks])
        if self.masking:
            state_tasks[self.node.seq] = 0.     # zero out state rows for scheduled tasks

        return state_tasks[self.sorted_index]       # sort individual task states

    @property
    def state(self):
        """Complete state."""
        return self.state_tasks

    def _update_spaces(self):
        """Update observation and action spaces."""
        pass

    def reset(self, tasks=None, ch_avail=None, persist=False, solve=False, verbose=False):
        """
        Reset environment by re-initializing node object with random (or user-specified) tasks/channels.

        Parameters
        ----------
        tasks : Sequence of tasks.Generic, optional
            Optional task set for non-random reset.
        ch_avail : Sequence of float, optional
            Optional initial channel availabilities for non-random reset.
        persist : bool
            If True, keeps tasks and channels fixed during reset, regardless of other inputs.
        solve : bool
            Solves and stores the Branch & Bound optimal schedule.
        verbose : bool, optional
            Enables print-out progress information.

        """

        if not persist:
            if tasks is None or ch_avail is None:   # generate new scheduling problem
                if solve:
                    ((tasks, ch_avail), self.solution), = self.problem_gen(1, solve=solve, verbose=verbose)
                else:
                    (tasks, ch_avail), = self.problem_gen(1, solve=solve, verbose=verbose)
                    self.solution = None

            elif len(tasks) != self.n_tasks:
                raise ValueError(f"Input 'tasks' must be None or a list of {self.n_tasks} tasks")
            elif len(ch_avail) != self.n_ch:
                raise ValueError(f"Input 'ch_avail' must be None or an array of {self.n_ch} channel availabilities")

            self.node_cls._tasks_init = tasks
            self.node_cls._ch_avail_init = ch_avail

        self.node = self.node_cls()
        self.loss_agg = self.node.l_ex  # Loss can be non-zero due to time origin shift during node initialization

        self._update_spaces()

        return self.state

    def step(self, action):
        """
        Updates environment state based on task index input.

        Parameters
        ----------
        action : int or Sequence of int
            Complete index sequence.

        Returns
        -------
        observation : ndarray
        reward : float
            Negative loss achieved by the complete sequence.
        done : True
            Episode completes after one step.
        info : dict
            Auxiliary diagnostic information (helpful for debugging, and sometimes learning).

        """

        action = self.sorted_index[action]  # decode task index to original order
        self.node.seq_extend(action)  # updates sequence, loss, task parameters, etc.

        reward, self.loss_agg = self.loss_agg - self.node.l_ex, self.node.l_ex
        done = len(self.node.seq_rem) == 0      # sequence is complete

        self._update_spaces()

        return self.state, reward, done, {}

    def render(self, mode='human'):
        if mode == 'human':
            fig_env, ax_env = plt.subplots(num='Task Scheduling Env', clear=True)
            plot_task_losses(self.tasks, ax=ax_env)

    def close(self):
        plt.close('all')


class SeqTaskingEnv(BaseTaskingEnv):
    """Tasking environment with single action of a complete task index sequence."""

    def __init__(self, problem_gen, node_cls=TreeNode, features=None, sort_func=None, masking=False):
        super().__init__(problem_gen, node_cls, features, sort_func, masking)

        # gym.Env observation and action spaces
        self.observation_space = Box(self._state_tasks_low, self._state_tasks_high, dtype=np.float64)
        self.action_space = Sequence(self.n_tasks)

    # @property
    # def observation_space(self):
    #     """Gym space of valid observations."""
    #     return Box(self._state_tasks_low, self._state_tasks_high, dtype=np.float64)
    #
    # @property
    # def action_space(self):
    #     """Gym space of valid actions."""
    #     return Sequence(self.n_tasks)

    @staticmethod
    def infer_action_space(observation):
        """Determines the action Gym.Space from an observation."""
        return Sequence(len(observation))


class StepTaskingEnv(BaseTaskingEnv):
    """
    Tasking environment with actions of single task indices.

    Parameters
    ----------
    problem_gen : generators.scheduling_problems.Base
        Scheduling problem generation object.
    node_cls : TreeNode or callable
        Class for tree search node generation.
    features : ndarray, optional
        Structured numpy array of features with fields 'name', 'func', and 'lims'.
    sort_func : function or str, optional
        Method that returns a sorting value for re-indexing given a task index 'n'.
    seq_encoding : function or str, optional
        Method that returns a 1-D encoded sequence representation for a given task index 'n'. Assumes that the
        encoded array sums to one for scheduled tasks and to zero for unscheduled tasks.
    masking : bool
        If True, features are zeroed out for scheduled tasks.

    """

    def __init__(self, problem_gen, node_cls=TreeNode, features=None, sort_func=None, seq_encoding='one-hot',
                 masking=False):

        super().__init__(problem_gen, node_cls, features, sort_func, masking)

        # Set sequence encoder method
        if callable(seq_encoding):
            self.seq_encoding = MethodType(seq_encoding, self)

            env_copy = deepcopy(self)       # FIXME: hacked - find better way!
            env_copy.reset()
            self.len_seq_encode = env_copy.state_seq.shape[-1]
        elif type(seq_encoding) == str:     # simple string specification for supported encoders
            if seq_encoding == 'indicator':
                def _seq_encoding(env, n):
                    return [1] if n in env.node.seq else [0]

                self.len_seq_encode = 1
            elif seq_encoding == 'one-hot':
                def _seq_encoding(env, n):
                    out = np.zeros(env.n_tasks)
                    if n in env.node.seq:
                        out[env.node.seq.index(n)] = 1
                    return out

                self.len_seq_encode = self.n_tasks
            else:
                raise ValueError("Unsupported sequence encoder string.")

            self.seq_encoding = MethodType(_seq_encoding, self)
        else:
            raise TypeError("Sequence encoding input must be callable or str.")

        # gym.Env observation and action spaces
        _state_low = np.concatenate((np.zeros((self.n_tasks, self.len_seq_encode)), self._state_tasks_low), axis=1)
        _state_high = np.concatenate((np.ones((self.n_tasks, self.len_seq_encode)), self._state_tasks_high), axis=1)
        self.observation_space = Box(_state_low, _state_high, dtype=np.float64)
        self.action_space = DiscreteSet(set(range(self.n_tasks)))

    # @property
    # def observation_space(self):
    #     """Gym space of valid observations."""
    #     _state_low = np.concatenate((np.zeros(self.state_seq.shape), self._state_tasks_low), axis=1)
    #     _state_high = np.concatenate((np.ones(self.state_seq.shape), self._state_tasks_high), axis=1)
    #     return Box(_state_low, _state_high, dtype=np.float64)

    # @property
    # def action_space(self):
    #     """Gym space of valid actions."""
    #     seq_rem_sort = np.flatnonzero(np.isin(self.sorted_index, list(self.node.seq_rem)))
    #     return DiscreteSet(seq_rem_sort)

    def infer_action_space(self, observation):
        """Determines the action Gym.Space from an observation."""
        _state_seq = observation[:, :-len(self.features)]
        return DiscreteSet(np.flatnonzero(1 - _state_seq.sum(1)))

    def _update_spaces(self):
        """Update observation and action spaces."""
        seq_rem_sort = np.flatnonzero(np.isin(self.sorted_index, list(self.node.seq_rem)))
        self.action_space = DiscreteSet(seq_rem_sort)

    @property
    def state_seq(self):
        """State sub-array for encoded partial sequence."""
        state_seq = np.array([self.seq_encoding(n) for n in range(self.n_tasks)])
        return state_seq[self.sorted_index]  # sort individual sequence states

    @property
    def state(self):
        """Complete state."""
        return np.concatenate((self.state_seq, self.state_tasks), axis=1)

    def data_gen(self, n_batch, batch_size=1, weight_func=None, verbose=False):
        """
        Generate state-action data for learner training and evaluation.

        Parameters
        ----------
        n_batch : int
            Number of batches of state-action pair data to generate.
        batch_size : int
            Number of scheduling problems to make data from per yielded batch.
        weight_func : callable, optional
            Function mapping partial sequence length and number of tasks to a training weight.
        verbose : bool, optional
            Enables print-out progress information.

        Yields
        ------
        tuple of ndarray
            Observable predictor data, unobserved target data, and optional sample weights.

        """

        # TODO: generalize for other Env classes. TF loss func for full seq targets?

        for i_batch in range(n_batch):
            if verbose:
                print(f'Batch: {i_batch + 1}/{n_batch}', end='\n')

            x_set = np.empty((self.n_tasks * batch_size, *self.observation_space.shape))  # predictors
            y_set = np.empty(self.n_tasks * batch_size)  # targets
            w_set = np.empty(self.n_tasks * batch_size)  # weights

            for i_samp in range(batch_size):
                self.reset(solve=True, verbose=verbose)  # generates new scheduling problem

                # Optimal schedule
                t_ex, ch_ex = self.solution.t_ex, self.solution.ch_ex
                seq = np.argsort(t_ex)  # maps to optimal schedule (empirically confirmed...)
                # TODO: train using complete tree info, not just B&B solution?

                # Generate samples for each scheduling step of the optimal sequence
                for i, n in enumerate(seq):
                    i = i_samp * self.n_tasks + i
                    n = self.sorted_index.tolist().index(n)  # transform index using sorting function

                    x_set[i] = self.state.copy()
                    y_set[i] = n
                    if callable(weight_func):
                        w_set[i] = weight_func(i, self.n_tasks)

                    self.step(n)  # updates environment state

            if callable(weight_func):
                yield x_set, y_set, w_set
            else:
                yield x_set, y_set


# Agents
class RandomAgent:
    """The world's simplest agent!"""
    def __init__(self, infer_action_space):
        self.infer_action_space = infer_action_space

    def act(self, observation, reward, done):
        action_space = self.infer_action_space(observation)
        return action_space.sample()       # randomly selected action


# Learning
def train_agent(problem_gen, n_batch_train=1, n_batch_val=1, batch_size=1, env_cls=StepTaskingEnv, env_params=None,
                agent=None, save=False, save_dir=None):
    """
    Train a reinforcement learning agent.

    Parameters
    ----------
    problem_gen : generators.scheduling_problems.Base
        Scheduling problem generation object.
    n_batch_train : int
        Number of batches of state-action pair data to generate for agent training.
    n_batch_val : int
        Number of batches of state-action pair data to generate for agent validation.
    batch_size : int
        Number of scheduling problems to make data from per yielded batch.
    env_cls : class
        Gym environment class.
    env_params : dict, optional
        Parameters for environment initialization.
    agent : object
        Reinforcement learning agent.
    save : bool
        If True, the agent and environment are serialized.
    save_dir : str, optional
        String representation of sub-directory to save to.

    Returns
    -------
    function
        Wrapped agent. Takes tasks and channel availabilities and produces task execution times/channels.

    """

    if env_params is None:
        env_params = {}

    # Create environment
    env = env_cls(problem_gen, **env_params)

    if agent is None:
        agent = RandomAgent(env.infer_action_space)

    # Generate state-action data pairs, train
    # TODO

    # Save agent and environment
    if save:
        if save_dir is None:
            save_dir = 'temp/{}'.format(time.strftime('%Y-%m-%d_%H-%M-%S'))

        with open('agents/' + save_dir, 'wb') as file:
            dill.dump({'env': env, 'agent': agent}, file)    # save environment

    return wrap_agent(env, agent)


def load_agent(load_dir):
    """Loads agent and environment, returns wrapped scheduling function."""
    with open('agents/' + load_dir, 'rb') as file:
        pkl_dict = dill.load(file)
    return wrap_agent(**pkl_dict)


def wrap_agent(env, agent):
    """Generate scheduling function by running an agent on a single environment episode."""

    def scheduling_agent(tasks, ch_avail):
        observation, reward, done = env.reset(tasks, ch_avail), 0, False
        while not done:
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
            action = agent.act(observation, reward, done)
            observation, reward, done, info = env.step(action)

        runtime = time.perf_counter() - t_run
        if runtime >= max_runtime:
            raise RuntimeError(f"Algorithm timeout: {runtime} > {max_runtime}.")

        return env.node.t_ex, env.node.ch_ex

    return scheduling_agent


def main():

    problem_gen = RandomProblem.relu_drop_default(n_tasks=8, n_ch=2)

    features = np.array([('duration', lambda task: task.duration, problem_gen.task_gen.param_lims['duration']),
                         ('release time', lambda task: task.t_release,
                          (0., problem_gen.task_gen.param_lims['t_release'][1])),
                         ('slope', lambda task: task.slope, problem_gen.task_gen.param_lims['slope']),
                         ('drop time', lambda task: task.t_drop, (0., problem_gen.task_gen.param_lims['t_drop'][1])),
                         ('drop loss', lambda task: task.l_drop, (0., problem_gen.task_gen.param_lims['l_drop'][1])),
                         ('is available', lambda task: 1 if task.t_release == 0. else 0, (0, 1)),
                         ('is dropped', lambda task: 1 if task.l_drop == 0. else 0, (0, 1)),
                         ],
                        dtype=[('name', '<U16'), ('func', object), ('lims', np.float, 2)])
    # features = None

    # def seq_encoding(self, n):
    #     return [0] if n in self.node.seq else [1]

    def seq_encoding(self, n):
        out = np.zeros(self.n_tasks)
        if n in self.node.seq:
            out[self.node.seq.index(n)] = 1
        return out

    # seq_encoding = 'indicator'
    # seq_encoding = None

    def sort_func(self, n):
        if n in self.node.seq:
            return float('inf')
        else:
            return self.tasks[n].t_release
            # return 1 if self.tasks[n].l_drop == 0. else 0
            # return self.tasks[n].l_drop / self.tasks[n].t_drop

    # sort_func = 't_release'

    env_params = {'node_cls': TreeNodeShift,
                  'features': features,
                  'sort_func': sort_func,
                  'seq_encoding': seq_encoding,
                  'masking': False
                  }

    # env = SeqTaskingEnv(problem_gen, **env_params)
    env = StepTaskingEnv(problem_gen, **env_params)
    agent = RandomAgent(env.infer_action_space)

    # data_gen(env, n_gen=10)

    observation, reward, done = env.reset(), 0, False
    while not done:
        print(observation)
        print(env.sorted_index)
        print(env.node.seq)
        print(env.tasks)
        act = agent.act(observation, reward, done)
        observation, reward, done, info = env.step(act)
        print(reward)


if __name__ == '__main__':
    main()
