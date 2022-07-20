"""Gym Environments."""

import pickle
from abc import ABC, abstractmethod
from math import factorial
from operator import attrgetter

import numpy as np
from gym import Env
from gym.spaces import Box, Dict, MultiDiscrete
from tqdm import trange

import task_scheduling.spaces as spaces_tasking
from task_scheduling.mdp.features import normalize as normalize_features
from task_scheduling.mdp.features import param_features
from task_scheduling.nodes import ScheduleNode, ScheduleNodeReform
from task_scheduling.util import plot_losses_and_schedule

# TODO: custom features combining release times and chan availabilities?


class Base(Env, ABC):
    """
    Base environment for task scheduling.

    Parameters
    ----------
    problem_gen : generators.problems.Base
        Scheduling problem generation object.
    features : numpy.ndarray, optional
        Structured numpy array of features with fields 'name', 'func', and 'space'.
    normalize : bool, optional
        Rescale task features to unit interval.
    sort_func : function or str, optional
        Method that returns a sorting value for re-indexing given a task index 'n'.
    reform : bool, optional
        Enables task re-parameterization after sequence updates.

    """

    def __init__(
        self,
        problem_gen,
        features=None,
        normalize=False,
        sort_func=None,
        reform=False,
    ):
        self._problem_gen = problem_gen

        # Get and modify problem space
        ch_avail_space, param_spaces = Base.get_problem_spaces(self.problem_gen, reform=reform)

        # Set features
        if features is not None:
            self.features = features
        else:
            self.features = param_features(param_spaces)

        if any(space.shape != () for space in self.features["space"]):
            raise NotImplementedError("Features must be scalar valued")

        self.normalize = normalize
        if self.normalize:
            self.features = normalize_features(self.features)

        # Set sorting method
        if callable(sort_func):
            self.sort_func = sort_func
            self._sort_func_str = "Custom"
        elif isinstance(sort_func, str):
            self.sort_func = attrgetter(sort_func)
            self._sort_func_str = sort_func
        else:
            self.sort_func = None
            self._sort_func_str = None

        if reform:
            self.node_cls = ScheduleNodeReform
        else:
            self.node_cls = ScheduleNode

        self.reward_range = (-np.inf, 0)
        self._loss_agg = 0.0

        self.node = None  # MDP state
        self._tasks_init = None
        self._ch_avail_init = None

        self._seq_opt = None

        # Observation and action space
        self._obs_space_ch = ch_avail_space

        self._obs_space_seq = MultiDiscrete(np.full(self.n_tasks, 2))

        _obs_space_features = spaces_tasking.stack(self.features["space"])
        self._obs_space_tasks = spaces_tasking.broadcast_to(
            _obs_space_features, shape=(self.n_tasks, len(self.features))
        )

        # note: `spaces` attribute is `OrderedDict` with sorted keys
        self.observation_space = Dict(
            ch_avail=self._obs_space_ch, seq=self._obs_space_seq, tasks=self._obs_space_tasks
        )

        self.action_space = None

    n_tasks = property(lambda self: self.problem_gen.n_tasks)
    n_ch = property(lambda self: self.problem_gen.n_ch)

    n_features = property(lambda self: len(self.features))

    tasks = property(lambda self: self.node.tasks)
    ch_avail = property(lambda self: self.node.ch_avail)

    def __str__(self):
        if self.node is None:
            _status = "Initialized"
        else:
            _status = f"{len(self.node.seq)}/{self.n_tasks}"
        return f"{self.__class__.__name__}({_status})"

    def summary(self):
        str_ = f"{self.__class__.__name__}"
        str_ += f"\n- Features: {self.features['name'].tolist()}"
        str_ += f"\n- Sort: {self._sort_func_str}"
        str_ += f"\n- Reform: {self.node_cls == ScheduleNodeReform}"
        return str_

    @property
    def problem_gen(self):
        return self._problem_gen

    @problem_gen.setter
    def problem_gen(self, value):
        if (
            self._problem_gen.task_gen != value.task_gen
            or self._problem_gen.ch_avail_gen != value.ch_avail_gen
        ):
            raise ValueError("New generator must match.")
        self._problem_gen = value

    @staticmethod
    def get_problem_spaces(problem_gen, reform=False):
        ch_avail_space = problem_gen.ch_avail_gen.space
        param_spaces = problem_gen.task_gen.param_spaces

        ch_avail_lim = spaces_tasking.get_space_lims(ch_avail_space)
        param_lims = {key: spaces_tasking.get_space_lims(val) for key, val in param_spaces.items()}
        if True:  # TODO: generalize? Move to `ScheduleNode`?
            low = ch_avail_lim[0]
            max_start = max(ch_avail_lim[1], param_lims["t_release"][1])
            high = max_start + problem_gen.n_tasks * param_lims["duration"][1]
            ch_avail_space = Box(low, high, shape=(problem_gen.n_ch,), dtype=float)

        if reform:
            param_lims = problem_gen.task_gen.cls_task.reform_param_lims(
                param_lims, ch_avail_lim, problem_gen.n_tasks
            )
            param_spaces = {
                key: Box(*val, shape=(), dtype=float) for key, val in param_lims.items()
            }
        return ch_avail_space, param_spaces

    @property
    def sorted_index(self):
        """Generate indices for re-ordering of observation rows."""
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

    def _obs_ch_avail(self):
        if self.normalize:
            low, high = self._obs_space_ch.low, self._obs_space_ch.high
            return (self.ch_avail - low) / (high - low)
        else:
            return self.ch_avail

    def _obs_seq(self):
        return np.array([1 if n in self.node.seq else 0 for n in self.sorted_index])

    def _obs_tasks(self):
        """Observation tensor for task features."""
        obs_tasks = np.array(
            [[func(task) for func in self.features["func"]] for task in self.tasks]
        )
        return obs_tasks[self.sorted_index]  # sort individual task observations

    def obs(self):
        """Complete observation."""
        # data = tuple(
        #     getattr(self, f"_obs_{key}")() for key in self.observation_space
        # )  # invoke `_obs_tasks`, etc.
        # dtype = [
        #     (key, space.dtype, space.shape)
        #     for key, space in self.observation_space.spaces.items()
        # ]
        # return np.array(data, dtype=dtype)

        # return OrderedDict(
        #     [(key, getattr(self, f"_obs_{key}")()) for key in self.observation_space]
        # )
        return {key: getattr(self, f"_obs_{key}")() for key in self.observation_space}

    @staticmethod
    @abstractmethod
    def infer_valid_mask(obs):
        """Create a binary valid action mask from an observation."""
        raise NotImplementedError

    # @abstractmethod
    # def infer_action_space(self, obs):
    #     """Determines the action `gym.Space` from an observation."""
    #     raise NotImplementedError

    def _update_spaces(self):
        """Update observation and action spaces."""
        pass

    def reset(self, tasks=None, ch_avail=None, solve=False, rng=None):
        """
        Reset environment by re-initializing node object with new tasks/channels.

        Parameters
        ----------
        tasks : Collection of task_scheduling.tasks.Base, optional
            Tasks for non-random reset.
        ch_avail : Collection of float, optional
            Initial channel availabilities for non-random reset.
        solve : bool
            Solves for and stores the Branch & Bound optimal schedule.
        rng : int or RandomState or Generator, optional
            NumPy random number generator or seed. Instance RNG if None.

        Returns
        -------
        numpy.ndarray
            Observation.

        """
        rng = self.problem_gen._get_rng(rng)
        if tasks is None or ch_avail is None:  # generate new scheduling problem
            out = list(self.problem_gen(1, solve=solve, rng=rng))[0]
            if solve:
                (tasks, ch_avail), (sch, *_) = out
                self._seq_opt = np.argsort(sch["t"])
                # optimal schedule (see `test_tree_nodes.test_argsort`)
            else:
                tasks, ch_avail = out
                self._seq_opt = None
        elif len(tasks) != self.n_tasks:
            raise ValueError(f"Input `tasks` must be None or a collection of {self.n_tasks} tasks")
        elif len(ch_avail) != self.n_ch:
            raise ValueError(
                f"Input `ch_avail` must be None or an array of {self.n_ch} channel availabilities"
            )

        # store problem before any in-place operations
        self._tasks_init, self._ch_avail_init = tasks, ch_avail

        self.node = self.node_cls(tasks, ch_avail)

        # loss can be non-zero due to problem reformation during node initialization
        self._loss_agg = self.node.loss

        self._update_spaces()

        return self.obs()

    def step(self, action):
        """
        Update environment state (node) based on task index input.

        Parameters
        ----------
        action : int or Collection of int
            Task indices.

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

        loss_step, self._loss_agg = self.node.loss - self._loss_agg, self.node.loss
        reward = -loss_step

        done = len(self.node.seq_rem) == 0  # sequence is complete

        self._update_spaces()

        return self.obs(), reward, done, {}

    def render(self, mode="human"):
        if mode != "human":
            raise NotImplementedError("Render `mode` must be 'human'")

        return plot_losses_and_schedule(
            self._tasks_init,
            self.node.sch,
            self.n_ch,
            loss=self._loss_agg,
            name=str(self),
            fig_kwargs=dict(
                # num=f"render_{id(self)}",
                # figsize=[12.8, 6.4],
                # gridspec_kw={"left": 0.05, "right": 0.7},
            ),
            legend=False,
        )

    def close(self):
        self.node = None

    def seed(self, seed=None):
        self.problem_gen.rng = seed

    @abstractmethod
    def opt_action(self):  # TODO: implement a optimal policy calling obs?
        """Optimal action based on current state."""
        raise NotImplementedError

    def opt_rollouts(self, n_gen, verbose=0, save_path=None, rng=None):
        """
        Generate observation-action data for learner training and evaluation.

        Parameters
        ----------
        n_gen : int, optional
            Number of scheduling problems to generate data from.
        verbose : {0, 1, 2}, optional
            0: silent, 1: add batch info, 2: add problem info
        save_path : os.PathLike or str, optional
            File path for saving data.
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
        collect_shape = (n_gen, self.n_tasks)
        if isinstance(self.observation_space, Dict):
            # x_set = OrderedDict([(key, np.empty((steps_total, *space.shape), dtype=space.dtype))
            #                      for key, space in self.observation_space.spaces.items()])
            o = {
                key: np.empty(collect_shape + space.shape, dtype=space.dtype)
                for key, space in self.observation_space.spaces.items()
            }
        else:
            o = np.empty(
                collect_shape + self.observation_space.shape, dtype=self.observation_space.dtype
            )
        a = np.empty(collect_shape + self.action_space.shape, dtype=self.action_space.dtype)
        r = np.empty(collect_shape, dtype=float)

        rng = self.problem_gen._get_rng(rng)
        for i_gen in trange(n_gen, desc="Creating tensors", disable=(verbose == 0)):
            obs = self.reset(solve=True, rng=rng)  # generates new scheduling problem

            for i_step in range(self.n_tasks):
                action = self.opt_action()

                for key in self.observation_space:
                    o[key][i_gen, i_step] = obs[key]
                a[i_gen, i_step] = action

                obs, reward, _done, _info = self.step(action)  # updates environment state
                r[i_gen, i_step] = reward

        if save_path is not None:
            with open(save_path, "wb") as f:
                pickle.dump({"obs": o, "act": a, "rew": r, "env_summary": self.summary()}, f)

        return o, a, r


class Index(Base):
    """
    Tasking environment with actions of single task indices.

    Parameters
    ----------
    problem_gen : generators.problems.Base
        Scheduling problem generation object.
    features : numpy.ndarray, optional
        Structured numpy array of features with fields 'name', 'func', and 'space'.
    normalize : bool, optional
        Rescale task features to unit interval.
    sort_func : function or str, optional
        Method that returns a sorting value for re-indexing given a task index 'n'.
    reform : bool, optional
        Enables task re-parameterization after sequence updates.

    """

    def __init__(
        self,
        problem_gen,
        features=None,
        normalize=False,
        sort_func=None,
        reform=False,
    ):
        super().__init__(problem_gen, features, normalize, sort_func, reform)
        self.action_space = spaces_tasking.DiscreteMasked(self.n_tasks)

    def _update_spaces(self):
        """Update observation and action spaces."""
        seq_rem_sort = self.sorted_index_inv[list(self.node.seq_rem)]
        self.action_space.mask = np.isin(np.arange(self.n_tasks), seq_rem_sort, invert=True)

    def opt_action(self):
        """Optimal action based on current state."""
        if self._seq_opt is None:
            raise ValueError(
                "Optimal action cannot be determined unless `reset` was called with `solve=True`."
            )

        n = self._seq_opt[len(self.node.seq)]  # next optimal task index
        return self.sorted_index_inv[n]  # encode task index to sorted action

    @staticmethod
    def infer_valid_mask(obs):
        """Create a binary valid action mask from an observation."""
        return obs["seq"]

    # def infer_action_space(self, obs):
    #     """Determines the action Gym.Space from an observation."""
    #     obs = np.asarray(obs)
    #     if obs.ndim > 3:
    #         raise ValueError("Input must be a single observation.")
    #
    #     mask = self.infer_valid_mask(obs).astype(bool)
    #     return spaces_tasking.DiscreteMasked(self.n_tasks, mask)


def seq_to_int(seq, check_input=True):
    """
    Map an index sequence permutation to a non-negative integer.

    Parameters
    ----------
    seq : Collection of int
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


def int_to_seq(num, length, check_input=True):
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
    tuple of int
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


# class Seq(Base):
#     def __init__(
#         self,
#         problem_gen,
#         features=None,
#         normalize=False,
#         sort_func=None,
#         reform=False,
#         action_type="int",
#     ):
#         """Tasking environment with single action of a complete task index sequence.

#         Parameters
#         ----------
#         problem_gen : generators.problems.Base
#             Scheduling problem generation object.
#         features : numpy.ndarray, optional
#             Structured numpy array of features with fields 'name', 'func', and 'space'.
#         normalize : bool, optional
#             Rescale task features to unit interval.
#         sort_func : function or str, optional
#             Method that returns a sorting value for re-indexing given a task index 'n'.
#         reform : bool, optional
#             Enables task re-parameterization after sequence updates.
#         action_type : {'seq', 'int'}, optional
#             If 'seq', action type is index sequence `Permutation`; if 'int', action space is
#               `Discrete` and
#             index sequences are mapped to integers.

#         """
#         super().__init__(problem_gen, features, normalize, sort_func, reform)

#         self.action_type = action_type  # 'seq' for sequences, 'int' for integers
#         if self.action_type == "int":
#             self._action_space_map = lambda n: Discrete(factorial(n))
#         elif self.action_type == "seq":
#             raise NotImplementedError("Deprecated.")
#             # self._action_space_map = lambda n: spaces_tasking.Permutation(n)
#         else:
#             raise ValueError

#         # Action space
#         self.steps_per_episode = 1
#         self.action_space = self._action_space_map(self.n_tasks)

#     def summary(self):
#         str_ = super().summary()
#         str_ += f"\n- Action type: {self.action_type}"
#         return str_

#     def step(self, action):
#         if self.action_type == "int":
#             action = list(int_to_seq(action, self.n_tasks))  # decode integer to sequence

#         return super().step(action)

#     def opt_action(self):
#         """Optimal action based on current state."""
#         seq_action = self.sorted_index_inv[self._seq_opt]  # encode sequence to sorted actions
#         if self.action_type == "int":
#             return seq_to_int(seq_action)
#         elif self.action_type == "seq":
#             return seq_action

#     @staticmethod
#     def infer_valid_mask(obs):
#         """Create a binary valid action mask from an observation."""
#         return np.ones(factorial(len(obs)))

#     # def infer_action_space(self, obs):
#     #     """Determines the action Gym.Space from an observation."""
#     #     return self._action_space_map(len(obs))
