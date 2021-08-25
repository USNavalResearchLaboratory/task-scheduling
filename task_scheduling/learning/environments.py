from abc import ABC, abstractmethod
from math import factorial
from types import MethodType
from operator import attrgetter

import matplotlib.pyplot as plt
import numpy as np
from gym import Env
from gym.spaces import Discrete, MultiDiscrete

from task_scheduling import tree_search
import task_scheduling.spaces as spaces_tasking
from task_scheduling.learning.features import param_features
from task_scheduling.util import plot_task_losses


# Gym Environments
class BaseTasking(Env, ABC):
    def __init__(self, problem_gen, features=None, sort_func=None, time_shift=False, masking=False):
        """Base environment for task scheduling.

        Parameters
        ----------
        problem_gen : generators.problems.Base
            Scheduling problem generation object.
        features : numpy.ndarray, optional
            Structured numpy array of features with fields 'name', 'func', and 'lims'.
        sort_func : function or str, optional
            Method that returns a sorting value for re-indexing given a task index 'n'.
        time_shift : bool, optional
            Enables task re-parameterization after sequence updates.
        masking : bool, optional
            If True, features are zeroed out for scheduled tasks.

        """
        self.problem_gen = problem_gen
        self.solution = None

        # Set features and state bounds
        if features is not None:
            self.features = features
        else:
            self.features = param_features(self.problem_gen, time_shift, masking)

        # Set sorting method
        if callable(sort_func):
            self.sort_func = sort_func
            self._sort_func_str = 'Custom'
        elif isinstance(sort_func, str):
            self.sort_func = attrgetter(sort_func)
            self._sort_func_str = sort_func
        else:
            self.sort_func = None
            self._sort_func_str = None

        self.time_shift = time_shift
        self.masking = masking

        self.reward_range = (-np.inf, 0)
        self.loss_agg = None

        self.node = None

        self.steps_per_episode = None

        # gym.Env observation and action spaces
        self._obs_space_features = spaces_tasking.stack(self.features['space'])
        self.observation_space = None
        self.action_space = None

    n_tasks = property(lambda self: self.problem_gen.n_tasks)
    n_ch = property(lambda self: self.problem_gen.n_ch)
    tasks = property(lambda self: self.node.tasks)
    ch_avail = property(lambda self: self.node.ch_avail)

    def __repr__(self):
        if self.node is None:
            _status = 'Initialized'
        else:
            _status = f'{len(self.node.seq)}/{self.n_tasks} Tasks Scheduled'
        return f"{self.__class__.__name__}({_status})"

    def _base_summary(self):
        cls_str = self.__class__.__name__
        # str_ = f"{cls_str}\n---\n"
        str_ = f"{cls_str}"
        str_ += f"\n- Features: {self.features['name'].tolist()}"
        str_ += f"\n- Sorting: {self._sort_func_str}"
        str_ += f"\n- Task shifting: {self.time_shift}"
        str_ += f"\n- Masking: {self.masking}"
        return str_

    def summary(self, file=None):
        print(self._base_summary(), file=file, end='\n\n')
        # if print_gen:
        #     self.problem_gen.summary(file)

    @property
    def sorted_index(self):
        """Indices for task re-ordering for environment state."""
        if callable(self.sort_func):
            values = np.array([self.sort_func(task) for task in self.tasks])
            values[self.node.seq] = np.inf  # scheduled tasks to the end
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

        state_tasks = np.array([func(self.tasks, self.ch_avail) for func in self.features['func']]).transpose()
        if self.masking:
            state_tasks[self.node.seq] = 0.  # zero out state rows for scheduled tasks

        return state_tasks[self.sorted_index]  # sort individual task states

    @property
    @abstractmethod
    def state(self):
        """Complete state."""
        raise NotImplementedError

    @abstractmethod
    def infer_action_space(self, obs):
        """Determines the action Gym.Space from an observation."""
        raise NotImplementedError

    def _update_spaces(self):
        """Update observation and action spaces."""
        pass

    def reset(self, tasks=None, ch_avail=None, persist=False, solve=False, rng=None):
        """
        Reset environment by re-initializing node object with random (or user-specified) tasks/channels.

        Parameters
        ----------
        tasks : Sequence of task_scheduling.tasks.Base, optional
            Optional task set for non-random reset.
        ch_avail : Sequence of float, optional
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
            if self.time_shift:
                raise NotImplementedError("Shift nodes cannot recover original tasks")
            else:
                tasks, ch_avail = self.tasks, self.ch_avail
        else:
            if tasks is None or ch_avail is None:  # generate new scheduling problem
                out = list(self.problem_gen(1, solve=solve, rng=rng))[0]
                if solve:
                    (tasks, ch_avail), self.solution = out
                else:
                    (tasks, ch_avail), self.solution = out, None

            elif len(tasks) != self.n_tasks:
                raise ValueError(f"Input 'tasks' must be None or a list of {self.n_tasks} tasks")
            elif len(ch_avail) != self.n_ch:
                raise ValueError(f"Input 'ch_avail' must be None or an array of {self.n_ch} channel availabilities")

        if self.time_shift:
            self.node = tree_search.ScheduleNodeShift(tasks, ch_avail)
        else:
            self.node = tree_search.ScheduleNode(tasks, ch_avail)
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
        done = len(self.node.seq_rem) == 0  # sequence is complete

        self._update_spaces()

        return self.state, reward, done, {}

    def render(self, mode='human'):  # TODO: improve or delete
        if mode == 'human':
            _, ax_env = plt.subplots(num='Task Scheduling Env', clear=True)
            plot_task_losses(self.tasks, ax=ax_env)

    def close(self):
        plt.close('all')

    @abstractmethod
    def _gen_single(self, seq, weight_func):
        """Generate lists of predictor/target/weight samples for a given optimal task index sequence."""
        raise NotImplementedError

    def data_gen(self, n_batch, batch_size=1, weight_func=None, verbose=0, rng=None):
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
        verbose : {0, 1, 2}, optional
            0: silent, 1: add batch info, 2: add problem info
        rng : int or RandomState or Generator, optional
            NumPy random number generator or seed. Instance RNG if None.

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
            # if verbose >= 1:
            #     print(f'Batch: {i_batch + 1}/{n_batch}', end='\n')

            steps_total = batch_size * self.steps_per_episode

            x_set = np.empty((steps_total, *self.observation_space.shape), dtype=self.observation_space.dtype)
            y_set = np.empty((steps_total, *self.action_space.shape), dtype=self.action_space.dtype)
            w_set = np.empty(steps_total, dtype=float)

            for i_gen in range(batch_size):
                # if verbose >= 2:
                #     print(f'  Problem: {i_gen + 1}/{batch_size}', end='\r')
                if verbose >= 1:
                    print(f'Problem: {batch_size*i_batch + i_gen + 1}/{n_batch * batch_size}', end='\r')

                self.reset(solve=True, rng=rng)  # generates new scheduling problem

                # Optimal schedule
                t_ex, ch_ex = self.solution.t_ex, self.solution.ch_ex
                seq = np.argsort(t_ex)  # maps to optimal schedule (empirically supported...)

                # Generate samples for each scheduling step of the optimal sequence
                idx = i_gen * self.steps_per_episode + np.arange(self.steps_per_episode)
                x_set[idx], y_set[idx], w_set[idx] = self._gen_single(seq, weight_func)

            if callable(weight_func):
                yield x_set, y_set, w_set
            else:
                yield x_set, y_set

    def data_gen_full(self, n_gen, weight_func=None, verbose=0):
        """Generate state-action data, return in single feature/class arrays."""
        data, = self.data_gen(n_batch=1, batch_size=n_gen, weight_func=weight_func, verbose=verbose)
        return data

    def data_gen_baselines(self, n_gen):
        steps_total = n_gen * self.steps_per_episode

        observations, actions = self.data_gen_full(n_gen)
        if observations.ndim == 1:
            observations.shape = (steps_total, 1)
        if actions.ndim == 1:
            actions.shape = (steps_total, 1)

        rewards = np.zeros(steps_total, dtype=float)
        episode_returns = np.zeros(n_gen, dtype=float)
        episode_starts = np.full(steps_total, False, dtype=bool)
        episode_starts[np.arange(0, steps_total, self.steps_per_episode)] = True

        numpy_dict = {
            'actions': actions,
            'obs': observations,
            'rewards': rewards,
            'episode_returns': episode_returns,
            'episode_starts': episode_starts
        }

        return numpy_dict  # used to instantiate ExpertDataset object via `traj_data` arg

    def mask_probability(self, p):  # TODO: deprecate?
        """Returns masked action probabilities."""
        return np.array(p)


def seq_to_num(seq, check_input=True):
    """
    Map an index sequence permutation to a non-negative integer.

    Parameters
    ----------
    seq : Sequence of int
        Elements are unique in range(len(seq)).
    check_input : bool
        Enables value checking of input sequence.

    Returns
    -------
    int
        Takes values in range(factorial(len(seq))).
    """

    length = len(seq)
    seq_rem = list(range(length))  # remaining elements
    if check_input and set(seq) != set(seq_rem):
        raise ValueError(f"Input must have unique elements in range({length}).")

    num = 0
    for i, n in enumerate(seq):
        k = seq_rem.index(n)  # position of index in remaining elements
        num += k * factorial(length - 1 - i)
        seq_rem.remove(n)

    return num


def num_to_seq(num, length, check_input=True):
    """
    Map a non-negative integer to an index sequence permutation.

    Parameters
    ----------
    num : int
        In range(factorial(length))
    length : int
        Length of the output sequence.
    check_input : bool
        Enables value checking of input number.

    Returns
    -------
    tuple
        Elements are unique in factorial(len(seq)).
    """

    if check_input and num not in range(factorial(length)):
        raise ValueError(f"Input 'num' must be in range(factorial({length})).")

    seq_rem = list(range(length))  # remaining elements
    seq = []
    while len(seq_rem) > 0:
        radix = factorial(len(seq_rem) - 1)
        i, num = num // radix, num % radix

        n = seq_rem.pop(i)
        seq.append(n)

    return tuple(seq)


class SeqTasking(BaseTasking):
    def __init__(self, problem_gen, features=None, sort_func=None, time_shift=False, masking=False, action_type='int'):
        """Tasking environment with single action of a complete task index sequence.

        Parameters
        ----------
        problem_gen : generators.problems.Base
            Scheduling problem generation object.
        features : numpy.ndarray, optional
            Structured numpy array of features with fields 'name', 'func', and 'lims'.
        sort_func : function or str, optional
            Method that returns a sorting value for re-indexing given a task index 'n'.
        time_shift : bool, optional
            Enables task re-parameterization after sequence updates.
        masking : bool, optional
            If True, features are zeroed out for scheduled tasks.
        action_type : {'seq', 'int'}, optional
            If 'seq', action type is index sequence `Permutation`; if 'int', action space is `Discrete` and
            index sequences are mapped to integers.

        """
        super().__init__(problem_gen, features, sort_func, time_shift, masking)

        self.action_type = action_type  # 'seq' for sequences, 'int' for integers
        if self.action_type == 'seq':
            self._action_space_map = lambda n: spaces_tasking.Permutation(n)
        elif self.action_type == 'int':
            self._action_space_map = lambda n: Discrete(factorial(n))
        else:
            raise ValueError

        self.steps_per_episode = 1

        # gym.Env observation and action spaces
        self.observation_space = spaces_tasking.broadcast_to(self._obs_space_features,
                                                             shape=(self.n_tasks, len(self.features)))
        self.action_space = self._action_space_map(self.n_tasks)

    def summary(self, file=None):
        # super().summary(file)
        str_ = self._base_summary()
        str_ += f"\n- Action type: {self.action_type}"
        print(str_, file=file, end='\n\n')

    @property
    def state(self):
        """Complete state."""
        return self.state_tasks

    def infer_action_space(self, obs):
        """Determines the action Gym.Space from an observation."""
        return self._action_space_map(len(obs))

    def step(self, action):
        if self.action_type == 'int':
            action = list(num_to_seq(action, self.n_tasks))  # decode integer to sequence

        return super().step(action)

    def _gen_single(self, seq, weight_func):
        """Generate lists of predictor/target/weight samples for a given optimal task index sequence."""
        seq_sort = self.sorted_index_inv[seq]

        x = self.state.copy()

        if self.action_type == 'seq':
            y = seq_sort
        elif self.action_type == 'int':
            y = seq_to_num(seq_sort)
        else:
            raise ValueError

        if callable(weight_func):
            w = weight_func(self)
        else:
            w = 1.

        super().step(seq)  # invoke super method to avoid unnecessary encode-decode process

        return np.array([x]), np.array([y]), np.array([w])


class StepTasking(BaseTasking):
    def __init__(self, problem_gen, features=None, sort_func=None, time_shift=False, masking=False, action_type='valid',
                 seq_encoding=None):
        """Tasking environment with actions of single task indices.

        Parameters
        ----------
        problem_gen : generators.problems.Base
            Scheduling problem generation object.
        features : numpy.ndarray, optional
            Structured numpy array of features with fields 'name', 'func', and 'lims'.
        sort_func : function or str, optional
            Method that returns a sorting value for re-indexing given a task index 'n'.
        time_shift : bool, optional
            Enables task re-parameterization after sequence updates.
        masking : bool, optional
            If True, features are zeroed out for scheduled tasks.
        action_type : {'valid', 'any'}, optional
            If 'valid', action space is `DiscreteMasked`; if 'any', action space is `Discrete` and
            repeated actions are allowed (for experimental purposes only).
        seq_encoding : function or str, optional
            Method that returns a 1-D encoded sequence representation for a given task index 'n'. Assumes that the
            encoded array sums to one for scheduled tasks and to zero for unscheduled tasks.

        """
        super().__init__(problem_gen, features, sort_func, time_shift, masking)

        # Action types
        self.action_type = action_type  # FIXME: deprecate once RL algorithms are successfully integrated
        if self.action_type == 'valid':
            self.do_valid_actions = True
        elif self.action_type == 'any':
            self.do_valid_actions = False
        else:
            raise ValueError("Action type must be 'valid' or 'any'.")

        # Set sequence encoder method
        if seq_encoding is None:
            self.seq_encoding = MethodType(lambda env, n: [], self)
            self.len_seq_encode = 0
        elif isinstance(seq_encoding, str):  # simple string specification for supported encoders
            if seq_encoding == 'binary':
                def _seq_encoding(env, n):
                    return [1] if n in env.node.seq else [0]

                self.len_seq_encode = 1
            elif seq_encoding == 'one-hot':
                def _seq_encoding(env, n):
                    out = np.zeros(env.n_tasks, dtype=int)
                    if n in env.node.seq:
                        out[env.node.seq.index(n)] = 1
                    return out

                self.len_seq_encode = self.n_tasks
            else:
                raise ValueError("Unsupported sequence encoder string.")

            self.seq_encoding = MethodType(_seq_encoding, self)

        elif callable(seq_encoding):
            raise NotImplementedError('Generic callables not yet supported.')
            # self.seq_encoding = MethodType(seq_encoding, self)
            #
            # env_copy = deepcopy(self)
            # env_copy.reset()
            # self.len_seq_encode = len(env_copy.seq_encoding(0))
        else:
            raise TypeError("Permutation encoding input must be callable or str.")

        self._seq_encode_str = seq_encoding

        self.steps_per_episode = self.n_tasks

        # gym.Env observation and action spaces
        obs_space_seq = MultiDiscrete(2 * np.ones(self.len_seq_encode))
        obs_space_concat = spaces_tasking.concatenate((obs_space_seq, self._obs_space_features))
        self.observation_space = spaces_tasking.broadcast_to(obs_space_concat,
                                                             shape=(self.n_tasks, *obs_space_concat.shape))

        if self.do_valid_actions:
            # self.action_space = spaces_tasking.DiscreteSet(range(self.n_tasks))
            self.action_space = spaces_tasking.DiscreteMasked(self.n_tasks)
        else:
            self.action_space = Discrete(self.n_tasks)

    def summary(self, file=None):
        # super().summary(file)
        str_ = self._base_summary()
        str_ += f"\n- Action type: {self.action_type}"
        str_ += f"\n- Sequence encoding: {self._seq_encode_str}"
        print(str_, file=file, end='\n\n')

    @property
    def state(self):
        """Complete state."""
        state_seq = np.array([self.seq_encoding(n) for n in self.sorted_index])
        return np.concatenate((state_seq, self.state_tasks), axis=1)

    def make_mask(self, obs):  # TODO: make method private?
        state_seq = obs[..., :self.len_seq_encode]
        return state_seq.sum(axis=-1)

    def infer_action_space(self, obs):
        """Determines the action Gym.Space from an observation."""
        obs = np.asarray(obs)
        if obs.ndim > 2:
            raise ValueError("Input must be a single observation.")

        if self.do_valid_actions:
            # state_seq = obs[..., :self.len_seq_encode]
            # # seq_rem_sort = np.flatnonzero(1 - state_seq.sum(1))
            # # return spaces_tasking.DiscreteSet(seq_rem_sort)
            #
            # mask = state_seq.sum(axis=-1).astype(bool)
            mask = self.make_mask(obs).astype(bool)
            return spaces_tasking.DiscreteMasked(self.n_tasks, mask)
        else:
            return Discrete(len(obs))

    def _update_spaces(self):
        """Update observation and action spaces."""
        if self.do_valid_actions:
            seq_rem_sort = self.sorted_index_inv[list(self.node.seq_rem)]
            # self.action_space = spaces_tasking.DiscreteSet(seq_rem_sort)
            self.action_space.mask = np.isin(np.arange(self.n_tasks), seq_rem_sort, invert=True)

    def _gen_single(self, seq, weight_func):
        """Generate lists of predictor/target/weight samples for a given optimal task index sequence."""

        x_set = np.empty((self.steps_per_episode, *self.observation_space.shape), dtype=self.observation_space.dtype)
        y_set = np.empty((self.steps_per_episode, *self.action_space.shape), dtype=self.action_space.dtype)
        w_set = np.ones(self.steps_per_episode, dtype=float)

        for idx, n in enumerate(seq):
            n = self.sorted_index_inv[n]

            x_set[idx] = self.state.copy()
            y_set[idx] = n
            if callable(weight_func):
                w_set[idx] = weight_func(self)

            self.step(n)  # updates environment state

        return x_set, y_set, w_set

    def mask_probability(self, p):  # TODO: deprecate?
        """Returns masked action probabilities based on unscheduled task indices."""

        if self.do_valid_actions:
            return np.ma.masked_array(p, self.action_space.mask)
        else:
            return super().mask_probability(p)
