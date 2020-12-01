from copy import deepcopy
from types import MethodType
from math import factorial
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
import gym
from gym.spaces import Space, Box, Discrete
# from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines.gail import ExpertDataset

from task_scheduling.util.plot import plot_task_losses
from task_scheduling.util.generic import seq2num, num2seq
from task_scheduling.tree_search import TreeNode

np.set_printoptions(precision=2)


# Gym Spaces
class Permutation(Space):
    """Gym Space for index sequences."""

    def __init__(self, n):
        self.n = n      # sequence length
        super().__init__(shape=(self.n,), dtype=np.int)

    def sample(self):
        return self.np_random.permutation(self.n)

    def contains(self, x):
        return True if (np.sort(np.asarray(x, dtype=int)) == np.arange(self.n)).all() else False

    def __repr__(self):
        return f"Permutation({self.n})"

    def __eq__(self, other):
        if isinstance(other, Permutation):
            return self.n == other.n
        else:
            return NotImplemented

    def __len__(self):
        return factorial(self.n)


class DiscreteSet(Space):
    """Gym Space for discrete, non-integral elements."""

    def __init__(self, elements):
        self.elements = np.sort(np.array(list(elements)).flatten())   # ndarray representation of set
        super().__init__(shape=(), dtype=self.elements.dtype)

    def sample(self):
        return self.np_random.choice(self.elements)

    def contains(self, x):
        return True if x in self.elements else False

    def __repr__(self):
        return f"DiscreteSet({self.elements})"

    def __eq__(self, other):
        if isinstance(other, DiscreteSet):
            return self.elements == other.elements
        else:
            return NotImplemented

    def __len__(self):
        return self.elements.size


# Gym Environments
class BaseTasking(ABC, gym.Env):
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
    masking : bool, optional
        If True, features are zeroed out for scheduled tasks.

    """

    # FIXME: add normalization option for RL learners!? Or just use gym.Wrappers?

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

        _low, _high = zip(*self.features['lims'])
        self._state_tasks_low = np.broadcast_to(_low, (self.n_tasks, len(_low)))
        self._state_tasks_high = np.broadcast_to(_high, (self.n_tasks, len(_high)))

        # Set sorting method
        if callable(sort_func):
            self.sort_func = sort_func
        elif isinstance(sort_func, str):
            # attr_str = sort_func

            def _sort_func(task):
                return getattr(task, sort_func)

            self.sort_func = _sort_func
        else:
            self.sort_func = None

        self.masking = masking

        self.reward_range = (-float('inf'), 0)
        self.loss_agg = None

        self.node_cls = node_cls
        self.node = None

        self.steps_per_episode = None

        self.observation_space = None
        self.action_space = None

    tasks = property(lambda self: self.node.tasks)
    ch_avail = property(lambda self: self.node.ch_avail)

    def __repr__(self):
        if self.node is None:
            _status = 'Initialized'
        else:
            _status = f'{len(self.node.seq)}/{self.n_tasks} Tasks Scheduled'
        return f"{self.__class__.__name__}({_status})"

    @property
    def sorted_index(self):
        """Indices for task re-ordering for environment state."""
        if callable(self.sort_func):
            values = np.array([self.sort_func(task) for task in self.tasks])
            values[self.node.seq] = np.inf     # scheduled tasks to the end
            return np.argsort(values)
        else:
            return np.arange(self.n_tasks)

    @property
    def sorted_index_inv(self):
        _idx_list = self.sorted_index.tolist()
        return np.array([_idx_list.index(n) for n in range(self.n_tasks)])

    @property
    def state_tasks(self):
        """State sub-array for task features."""
        state_tasks = np.array([task.feature_gen(*self.features['func']) for task in self.tasks])
        if self.masking:
            state_tasks[self.node.seq] = 0.     # zero out state rows for scheduled tasks

        return state_tasks[self.sorted_index]       # sort individual task states

    @property
    @abstractmethod
    def state(self):
        """Complete state."""
        raise NotImplementedError

    @abstractmethod
    def infer_action_space(self, obs):
        """Determines the action Gym.Space from an observation."""
        raise NotImplementedError

    @abstractmethod
    def _update_spaces(self):
        """Update observation and action spaces."""
        raise NotImplementedError

    def reset(self, tasks=None, ch_avail=None, persist=False, solve=False, rng=None):
        """
        Reset environment by re-initializing node object with random (or user-specified) tasks/channels.

        Parameters
        ----------
        tasks : Iterable of tasks.Generic, optional
            Optional task set for non-random reset.
        ch_avail : Iterable of float, optional
            Optional initial channel availabilities for non-random reset.
        persist : bool
            If True, keeps tasks and channels fixed during reset, regardless of other inputs.
        solve : bool
            Solves for and stores the Branch & Bound optimal schedule.
        rng : int or RandomState or Generator, optional
            NumPy random number generator or seed. Instance RNG if None.

        Returns
        -------
        numpy.ndarray
            Observation.

        """

        if persist:
            tasks, ch_avail = self.tasks, self.ch_avail
        else:
            if tasks is None or ch_avail is None:   # generate new scheduling problem
                if solve:   # TODO: next()? Pass a generator, not a callable??
                    ((tasks, ch_avail), self.solution), = self.problem_gen(1, solve=solve, rng=rng)
                else:
                    (tasks, ch_avail), = self.problem_gen(1, solve=solve)
                    self.solution = None
            elif len(tasks) != self.n_tasks:
                raise ValueError(f"Input 'tasks' must be None or a list of {self.n_tasks} tasks")
            elif len(ch_avail) != self.n_ch:
                raise ValueError(f"Input 'ch_avail' must be None or an array of {self.n_ch} channel availabilities")

        self.node = self.node_cls(tasks, ch_avail)
        self.loss_agg = self.node.l_ex  # Loss can be non-zero due to time origin shift during node initialization

        self._update_spaces()

        return self.state

    def step(self, action):
        """
        Updates environment state based on task index input.

        Parameters
        ----------
        action : int or Iterable of int
            Complete index sequence.

        Returns
        -------
        ndarray
            Observation.
        float
            Reward (negative loss) achieved by the complete sequence.
        bool
            Indicates the end of the learning episode.
        dict
            Auxiliary diagnostic information (helpful for debugging, and sometimes learning).

        """

        action = self.sorted_index[action]  # decode task index to original order

        self.node.seq_extend(action)  # updates sequence, loss, task parameters, etc.

        reward, self.loss_agg = self.loss_agg - self.node.l_ex, self.node.l_ex
        done = len(self.node.seq_rem) == 0      # sequence is complete

        self._update_spaces()

        return self.state, reward, done, {}

        # ensure_valid = False      # TODO: formalize invalid action functionality?
        # action = self.sorted_index[action]  # decode task index to original order
        # if ensure_valid or action in self.node.seq_rem:
        #     self.node.seq_extend(action)  # updates sequence, loss, task parameters, etc.
        #
        #     reward, self.loss_agg = self.loss_agg - self.node.l_ex, self.node.l_ex
        #     done = len(self.node.seq_rem) == 0  # sequence is complete
        #
        #     self._update_spaces()
        #
        # else:
        #     reward = -100
        #     done = False
        #
        # return self.state, reward, done, {}

    def render(self, mode='human'):     # TODO: plot partial schedule instead?
        if mode == 'human':
            _, ax_env = plt.subplots(num='Task Scheduling Env', clear=True)
            plot_task_losses(self.tasks, ax=ax_env)

    def close(self):
        plt.close('all')

    @classmethod
    def from_problem_gen(cls, problem_gen, env_params=None):
        """Environment constructor from problem generators."""

        if env_params is None:
            return cls(problem_gen)
        else:
            return cls(problem_gen, **env_params)

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
            Function mapping environment object to a training weight.
        verbose : bool, optional
            Enables print-out progress information.

        Yields
        ------
        ndarray
            Predictor data.
        ndarray
            Target data.
        ndarray, optional
            Sample weights.

        """

        for i_batch in range(n_batch):
            if verbose:
                print(f'Batch: {i_batch + 1}/{n_batch}', end='\n')

            steps_total = batch_size * self.steps_per_episode

            x_set = np.empty((steps_total, *self.observation_space.shape), dtype=self.observation_space.dtype)
            y_set = np.empty((steps_total, *self.action_space.shape), dtype=self.action_space.dtype)
            w_set = np.empty(steps_total, dtype=np.float)

            for i_gen in range(batch_size):
                if verbose:
                    print(f'  Problem: {i_gen + 1}/{batch_size}', end='\r')

                self.reset(solve=True)  # generates new scheduling problem

                # Optimal schedule
                t_ex, ch_ex = self.solution.t_ex, self.solution.ch_ex
                seq = np.argsort(t_ex)  # maps to optimal schedule (empirically supported...)

                # TODO: train using complete tree info, not just B&B solution?

                # Generate samples for each scheduling step of the optimal sequence
                idx = i_gen * self.steps_per_episode + np.arange(self.steps_per_episode)
                x_set[idx], y_set[idx], w_set[idx] = self._gen_single(seq, weight_func)

            if callable(weight_func):
                yield x_set, y_set, w_set
            else:
                yield x_set, y_set

    @abstractmethod
    def _gen_single(self, seq, weight_func):
        """Generate lists of predictor/target/weight samples for a given optimal task index sequence."""
        raise NotImplementedError

    def data_gen_numpy(self, n_gen, weight_func=None, verbose=False):
        """Generate state-action data as NumPy arrays."""
        data, = self.data_gen(n_batch=1, batch_size=n_gen, weight_func=weight_func, verbose=verbose)
        return data     # TODO: save dataset to save on Env computation time?

    def data_gen_baselines(self, n_gen):
        steps_total = n_gen * self.steps_per_episode

        observations, actions = self.data_gen_numpy(n_gen)
        rewards = np.zeros(steps_total, dtype=np.float)
        episode_returns = np.zeros(n_gen, dtype=np.float)
        episode_starts = np.full(steps_total, False, dtype=np.bool)
        episode_starts[np.arange(0, steps_total, self.steps_per_episode)] = True

        numpy_dict = {
            'actions': actions,
            'obs': observations,
            'rewards': rewards,
            'episode_returns': episode_returns,
            'episode_starts': episode_starts
        }

        return numpy_dict
        # return ExpertDataset(traj_data=numpy_dict)

    def data_gen_baselines_adam(self, n_episodes=5):
        actions = []
        observations = []
        rewards = []
        episode_returns = np.zeros((n_episodes,))
        episode_starts = []

        ep_idx = 0
        while ep_idx < n_episodes:

            obs = self.reset(solve=True)  # generates new scheduling problem
            print(obs)

            t_ex, ch_ex = self.solution.t_ex, self.solution.ch_ex
            seq = np.argsort(t_ex)  # maps to optimal schedule (empirically supported...)

            for i in range(len(seq)):
                observations.append(obs.flatten())
                action = seq[i]
                actions.append(action)
                rewards.append(1)   # need to fix
                print(i)
                episode_starts.append(i == 0)

            ep_idx += 1

        observations = np.concatenate(observations).reshape((-1,) + self.observation_space.shape)  # this could be an issue
        actions = np.array(actions).reshape(-1, 1)
        rewards = np.array(rewards)
        episode_starts = np.array(episode_starts)

        numpy_dict = {
            'actions': actions,
            'obs': observations,
            'rewards': rewards,
            'episode_returns': episode_returns,
            'episode_starts': episode_starts
        }

        return numpy_dict
        # return ExpertDataset(traj_data=numpy_dict)


class SeqTasking(BaseTasking):
    """Tasking environment with single action of a complete task index sequence."""

    def __init__(self, problem_gen, node_cls=TreeNode, features=None, sort_func=None, masking=False,
                 action_type='seq'):
        super().__init__(problem_gen, node_cls, features, sort_func, masking)

        self.action_type = action_type      # 'seq' for sequences, 'int' for integers
        if self.action_type == 'seq':
            self._action_space_map = lambda n: Permutation(n)
        elif self.action_type == 'int':
            self._action_space_map = lambda n: Discrete(factorial(n))
        else:
            raise ValueError

        self.steps_per_episode = 1

        # gym.Env observation and action spaces
        self.observation_space = Box(self._state_tasks_low, self._state_tasks_high, dtype=np.float64)
        self.action_space = self._action_space_map(self.n_tasks)

    @property
    def state(self):
        """Complete state."""
        return self.state_tasks

    def infer_action_space(self, obs):
        """Determines the action Gym.Space from an observation."""
        return self._action_space_map(len(obs))

    def _update_spaces(self):
        """Update observation and action spaces."""
        pass

    def step(self, action):
        if self.action_type == 'seq':
            pass
        elif self.action_type == 'int':
            action = list(num2seq(action, self.n_tasks))        # decode integer to sequence
        else:
            raise ValueError

        return super().step(action)

    def _gen_single(self, seq, weight_func):
        """Generate lists of predictor/target/weight samples for a given optimal task index sequence."""
        seq_sort = self.sorted_index_inv[seq]

        x = self.state.copy()

        if self.action_type == 'seq':
            y = seq_sort
        elif self.action_type == 'int':
            y = seq2num(seq_sort)
        else:
            raise ValueError

        if callable(weight_func):
            w = weight_func(self)  # TODO: weighting based on loss value!? Use MethodType, or new call signature?
        else:
            w = 1.

        # self.step(action)
        super().step(seq)        # invoke super method to avoid unnecessary encode-decode process

        return np.array([x]), np.array([y]), np.array([w])


class StepTasking(BaseTasking):
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
    masking : bool, optional
        If True, features are zeroed out for scheduled tasks.
    action_type : str, optional
        If 'valid', action type is `DiscreteSet` of valid indices; if 'any', action space is `Discrete` and
        repeated actions are allowed (for experimental purposes only).
    seq_encoding : function or str, optional
        Method that returns a 1-D encoded sequence representation for a given task index 'n'. Assumes that the
        encoded array sums to one for scheduled tasks and to zero for unscheduled tasks.

    """

    def __init__(self, problem_gen, node_cls=TreeNode, features=None, sort_func=None, masking=False,
                 action_type='valid', seq_encoding='one-hot'):

        super().__init__(problem_gen, node_cls, features, sort_func, masking)

        # Action types
        if action_type == 'valid':
            self.do_valid_actions = True
        elif action_type == 'any':
            self.do_valid_actions = False
        else:
            raise ValueError("Action type must be 'valid' or 'any'.")

        # Set sequence encoder method
        if callable(seq_encoding):
            self.seq_encoding = MethodType(seq_encoding, self)

            env_copy = deepcopy(self)       # FIXME: hacked - find better way!
            env_copy.reset()
            self.len_seq_encode = len(env_copy.seq_encoding(0))
        elif isinstance(seq_encoding, str):     # simple string specification for supported encoders
            if seq_encoding == 'binary':
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
            raise TypeError("Permutation encoding input must be callable or str.")

        self.steps_per_episode = self.n_tasks

        # gym.Env observation and action spaces
        _state_low = np.concatenate((np.zeros((self.n_tasks, self.len_seq_encode)), self._state_tasks_low), axis=1)
        _state_high = np.concatenate((np.ones((self.n_tasks, self.len_seq_encode)), self._state_tasks_high), axis=1)
        self.observation_space = Box(_state_low, _state_high, dtype=np.float64)

        if self.do_valid_actions:
            self.action_space = DiscreteSet(set(range(self.n_tasks)))
        else:
            self.action_space = Discrete(self.n_tasks)

    @property
    def state(self):
        """Complete state."""
        state_seq = np.array([self.seq_encoding(n) for n in range(self.n_tasks)])[self.sorted_index]
        return np.concatenate((state_seq, self.state_tasks), axis=1)

    def infer_action_space(self, obs):
        """Determines the action Gym.Space from an observation."""
        if self.do_valid_actions:
            # _state_seq = obs[:, :-len(self.features)]
            _state_seq = obs[:, :self.len_seq_encode]
            return DiscreteSet(np.flatnonzero(1 - _state_seq.sum(1)))
        else:
            return Discrete(len(obs))

    def _update_spaces(self):
        """Update observation and action spaces."""
        if self.do_valid_actions:
            seq_rem_sort = self.sorted_index_inv[list(self.node.seq_rem)]
            self.action_space = DiscreteSet(seq_rem_sort)
        else:
            pass

    def _gen_single(self, seq, weight_func):
        """Generate lists of predictor/target/weight samples for a given optimal task index sequence."""

        x_set = np.empty((self.steps_per_episode, *self.observation_space.shape), dtype=self.observation_space.dtype)
        y_set = np.empty((self.steps_per_episode, *self.action_space.shape), dtype=self.action_space.dtype)
        w_set = np.ones(self.steps_per_episode, dtype=np.float)

        for idx, n in enumerate(seq):
            n = self.sorted_index_inv[n]

            x_set[idx] = self.state.copy()
            y_set[idx] = n
            if callable(weight_func):
                w_set[idx] = weight_func(self)

            self.step(n)  # updates environment state

        return x_set, y_set, w_set


# class DummyVecTaskingEnv(DummyVecEnv):
#     def reset(self, *args, **kwargs):
#         for env_idx in range(self.num_envs):
#             obs = self.envs[env_idx].reset(*args, **kwargs)
#             self._save_obs(env_idx, obs)
#         return self._obs_from_buf()
