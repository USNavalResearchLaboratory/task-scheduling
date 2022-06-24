"""
Generator objects for tasks.

Notes
-----
Assumes all tasks are instances of the same class. Heterogeneous task types will be supported in a
future version.

"""

from abc import ABC, abstractmethod
from collections import deque
from types import MethodType
from typing import Collection

import numpy as np
import pandas as pd
from gym import spaces

from task_scheduling import tasks as task_types
from task_scheduling.base import RandomGeneratorMixin
from task_scheduling.spaces import DiscreteSet


class Base(RandomGeneratorMixin, ABC):
    """
    Base class for generation of task objects.

    Parameters
    ----------
    cls_task : class
        Class for instantiating task objects.
    param_spaces : dict, optional
        Mapping of parameter name strings to gym.spaces.Space objects
    rng : int or RandomState or Generator, optional
        Random number generator seed or object.

    """

    def __init__(self, cls_task, param_spaces=None, rng=None):
        super().__init__(rng)
        self.cls_task = cls_task

        if param_spaces is None:
            self.param_spaces = {
                name: spaces.Box(-np.inf, np.inf, shape=(), dtype=float)
                for name in self.cls_task.param_names
            }
        else:
            self.param_spaces = param_spaces

    @abstractmethod
    def __call__(self, n_tasks, rng=None):
        """
        Generate tasks.

        Parameters
        ----------
        n_tasks : int
            Number of tasks.
        rng : int or RandomState or Generator, optional
            Random number generator seed or object.

        Returns
        -------
        Generator

        """
        raise NotImplementedError

    def summary(self):
        cls_str = self.__class__.__name__
        return f"{cls_str}\n---"


class BaseIID(Base, ABC):
    """Base class for generation of independently and identically distributed task objects."""

    def __call__(self, n_tasks, rng=None):
        """
        Randomly generate tasks.

        Parameters
        ----------
        n_tasks : int
            Number of tasks.
        rng : int or RandomState or Generator, optional
            Random number generator seed or object.

        Returns
        -------
        Generator

        """
        rng = self._get_rng(rng)
        for __ in range(n_tasks):
            yield self.cls_task(**self._param_gen(rng))

    @abstractmethod
    def _param_gen(self, rng):
        """Randomly generate task parameters."""
        raise NotImplementedError


class GenericIID(BaseIID):
    """
    Generic generator of independently and identically distributed random task objects.

    Parameters
    ----------
    cls_task : class
        Class for instantiating task objects.
    param_gen : callable
        Invoked with 'self' argument, for use as the '_param_gen' method.
    param_spaces : dict, optional
        Mapping of parameter name strings to gym.spaces.Space objects
    rng : int or RandomState or Generator, optional
        Random number generator seed or object.

    """

    def __init__(self, cls_task, param_gen, param_spaces=None, rng=None):
        super().__init__(cls_task, param_spaces, rng)
        self._param_gen_init = MethodType(param_gen, self)

    def _param_gen(self, rng):
        return self._param_gen_init(rng)

    @classmethod
    def linear_drop(cls, param_gen, param_spaces=None, rng=None):
        return cls(task_types.LinearDrop, param_gen, param_spaces, rng)


class ContinuousUniformIID(BaseIID):
    """
    Random generator of I.I.D. tasks with independently uniform continuous parameters.

    Parameters
    ----------
    cls_task : class
        Class for instantiating task objects.
    param_lims : dict of Collection
        Mapping of parameter name strings to 2-tuples of parameter limits.
    rng : int or RandomState or Generator, optional
        Random number generator seed or object.

    """

    def __init__(self, cls_task, param_lims, rng=None):
        param_spaces = {
            name: spaces.Box(*param_lims[name], shape=(), dtype=float)
            for name in cls_task.param_names
        }
        super().__init__(cls_task, param_spaces, rng)

        self.param_lims = param_lims

    def _param_gen(self, rng):
        """Randomly generate task parameters."""
        return {name: rng.uniform(*self.param_lims[name]) for name in self.cls_task.param_names}

    def __eq__(self, other):
        if isinstance(other, ContinuousUniformIID):
            return self.cls_task == other.cls_task and self.param_lims == other.param_lims
        else:
            return NotImplemented

    def summary(self):
        str_ = super().summary()
        str_ += f"\nTask class: {self.cls_task.__name__}"

        df = pd.DataFrame(
            {name: self.param_lims[name] for name in self.cls_task.param_names},
            index=pd.CategoricalIndex(["low", "high"]),
        )
        df_str = df.to_markdown(tablefmt="github", floatfmt=".3f")

        str_ += f"\n\n{df_str}"
        return str_

    @classmethod
    def linear(cls, duration_lim=(3, 6), t_release_lim=(0, 4), slope_lim=(0.5, 2), rng=None):
        """Construct `Linear` task objects."""
        param_lims = dict(duration=duration_lim, t_release=t_release_lim, slope=slope_lim)
        return cls(task_types.Linear, param_lims, rng)

    @classmethod
    def linear_drop(
        cls,
        duration_lim=(3, 6),
        t_release_lim=(0, 4),
        slope_lim=(0.5, 2),
        t_drop_lim=(6, 12),
        l_drop_lim=(35, 50),
        rng=None,
    ):
        """Construct `LinearDrop` task objects."""
        param_lims = dict(
            duration=duration_lim,
            t_release=t_release_lim,
            slope=slope_lim,
            t_drop=t_drop_lim,
            l_drop=l_drop_lim,
        )
        return cls(task_types.LinearDrop, param_lims, rng)

    @classmethod
    def linear_linear(
        cls,
        duration_lim=(3, 6),
        t_release_lim=(0, 4),
        slope_lim=(0.5, 2),
        t_drop_lim=(6, 12),
        l_drop_lim=(35, 50),
        slope_2_lim=(0.5, 2),
        rng=None,
    ):
        """Construct `LinearLinear` task objects."""
        param_lims = dict(
            duration=duration_lim,
            t_release=t_release_lim,
            slope=slope_lim,
            t_drop=t_drop_lim,
            l_drop=l_drop_lim,
            slope_2=slope_2_lim,
        )
        return cls(task_types.LinearLinear, param_lims, rng)


class DiscreteIID(BaseIID):
    """
    Random generator of I.I.D. tasks with independent discrete-valued parameters.

    Parameters
    ----------
    cls_task : class
        Class for instantiating task objects.
    param_probs: dict of str to dict
        Mapping of parameter name strings to dictionaries mapping values to probabilities.
    rng : int or RandomState or Generator, optional
        Random number generator seed or object.

    """

    def __init__(self, cls_task, param_probs, rng=None):
        param_spaces = {
            name: DiscreteSet(list(param_probs[name].keys())) for name in cls_task.param_names
        }
        super().__init__(cls_task, param_spaces, rng)

        self.param_probs = param_probs

    def _param_gen(self, rng):
        """Randomly generate task parameters."""
        return {
            name: rng.choice(
                list(self.param_probs[name].keys()), p=list(self.param_probs[name].values())
            )
            for name in self.cls_task.param_names
        }

    def __eq__(self, other):
        if isinstance(other, DiscreteIID):
            return self.cls_task == other.cls_task and self.param_probs == other.param_probs
        else:
            return NotImplemented

    def summary(self):
        str_ = super().summary()
        str_ += f"\nTask class: {self.cls_task.__name__}"
        for name in self.cls_task.param_names:
            # s = pd.Series(self.param_probs[name], name='Pr')
            # s = pd.DataFrame(self.param_probs[name], index=pd.CategoricalIndex(['Pr']))
            s = pd.DataFrame(
                {name: self.param_probs[name].keys(), "Pr": self.param_probs[name].values()}
            )
            str_ += f"\n\n{s.to_markdown(tablefmt='github', floatfmt='.3f', index=False)}"

        return str_

    @classmethod
    def linear_uniform(
        cls, duration_vals=(3, 6), t_release_vals=(0, 4), slope_vals=(0.5, 2), rng=None
    ):
        """Construct `Linear` task objects."""
        param_probs = {
            "duration": dict(zip(duration_vals, np.ones(len(duration_vals)) / len(duration_vals))),
            "t_release": dict(
                zip(t_release_vals, np.ones(len(t_release_vals)) / len(t_release_vals))
            ),
            "slope": dict(zip(slope_vals, np.ones(len(slope_vals)) / len(slope_vals))),
        }
        return cls(task_types.Linear, param_probs, rng)

    @classmethod
    def linear_drop_uniform(
        cls,
        duration_vals=(3, 6),
        t_release_vals=(0, 4),
        slope_vals=(0.5, 2),
        t_drop_vals=(6, 12),
        l_drop_vals=(35, 50),
        rng=None,
    ):
        """Construct `LinearDrop` task objects."""
        param_probs = {
            "duration": dict(zip(duration_vals, np.ones(len(duration_vals)) / len(duration_vals))),
            "t_release": dict(
                zip(t_release_vals, np.ones(len(t_release_vals)) / len(t_release_vals))
            ),
            "slope": dict(zip(slope_vals, np.ones(len(slope_vals)) / len(slope_vals))),
            "t_drop": dict(zip(t_drop_vals, np.ones(len(t_drop_vals)) / len(t_drop_vals))),
            "l_drop": dict(zip(l_drop_vals, np.ones(len(l_drop_vals)) / len(l_drop_vals))),
        }
        return cls(task_types.LinearDrop, param_probs, rng)


class Fixed(Base, ABC):
    """
    Permutation task generator.

    Parameters
    ----------
    tasks : Collection of task_scheduling.tasks.Base
        Tasks.
    param_spaces : dict, optional
        Mapping of parameter name strings to gym.spaces.Space objects
    rng : int or RandomState or Generator, optional
        Random number generator seed or object.

    """

    def __init__(self, tasks, param_spaces=None, rng=None):
        cls_task = tasks[0].__class__
        if not all(isinstance(task, cls_task) for task in tasks[1:]):
            raise TypeError("All tasks must be of the same type.")

        if param_spaces is None:
            param_spaces = {
                name: DiscreteSet([getattr(task, name) for task in tasks])
                for name in cls_task.param_names
            }

        super().__init__(cls_task, param_spaces, rng)

        self.tasks = list(tasks)

    @abstractmethod
    def __call__(self, n_tasks, rng=None):
        """
        Generate fixed tasks.

        Parameters
        ----------
        n_tasks : int
            Number of tasks.
        rng : int or RandomState or Generator, optional
            Random number generator seed or object.

        Returns
        -------
        Generator

        """
        raise NotImplementedError

    def __eq__(self, other):
        if isinstance(other, Fixed):
            return self.tasks == other.tasks
        else:
            return NotImplemented

    @classmethod
    def _task_gen_to_fixed(cls, n_tasks, task_gen, rng):
        tasks = list(task_gen(n_tasks, rng))
        return cls(tasks, task_gen.param_spaces, rng)

    @classmethod
    def continuous_linear_drop(cls, n_tasks, rng=None, **task_gen_kwargs):
        task_gen = ContinuousUniformIID.linear_drop(**task_gen_kwargs)
        return cls._task_gen_to_fixed(n_tasks, task_gen, rng)

    @classmethod
    def discrete_linear_drop(cls, n_tasks, rng=None, **task_gen_kwargs):
        task_gen = DiscreteIID.linear_drop_uniform(**task_gen_kwargs)
        return cls._task_gen_to_fixed(n_tasks, task_gen, rng)

    # @classmethod
    # def search_track(cls, n_tasks, p=None, t_release_lim=(0., .018), rng=None):
    #     task_gen = SearchTrackIID(p, t_release_lim)
    #     return cls._task_gen_to_fixed(n_tasks, task_gen, rng)


class Deterministic(Fixed):
    def __call__(self, n_tasks, rng=None):
        """Yield tasks in deterministic order."""
        if n_tasks != len(self.tasks):
            raise ValueError(f"Number of tasks must be {len(self.tasks)}.")

        for task in self.tasks:
            yield task


class Permutation(Fixed):
    def __call__(self, n_tasks, rng=None):
        """Yield tasks in a uniformly random order."""
        if n_tasks != len(self.tasks):
            raise ValueError(f"Number of tasks must be {len(self.tasks)}.")

        rng = self._get_rng(rng)
        for task in rng.permutation(self.tasks).tolist():
            yield task


class Dataset(Fixed):  # FIXME: inherit from `Base`??
    """
    Generator of tasks from a dataset.

    Parameters
    ----------
    tasks : Sequence of task_scheduling.tasks.Base
        Stored tasks to be yielded.
    shuffle : bool, optional
        Shuffle task during instantiation.
    repeat : bool, optional
        Allow tasks to be yielded more than once.
    param_spaces : dict, optional
        Mapping of parameter name strings to gym.spaces.Space objects
    rng : int or RandomState or Generator, optional
        Random number generator seed or object.

    """

    def __init__(self, tasks, shuffle=False, repeat=False, param_spaces=None, rng=None):
        super().__init__(tasks, param_spaces, rng)

        self.tasks = deque()
        self.add_tasks(tasks)

        if shuffle:
            self.shuffle()

        self.repeat = repeat

    def add_tasks(self, tasks):
        """Add tasks to the queue."""
        if isinstance(tasks, Collection):
            self.tasks.extendleft(tasks)
        else:
            self.tasks.appendleft(tasks)  # for single tasks

    def shuffle(self, rng=None):
        """Shuffle the task queue."""
        rng = self._get_rng(rng)
        self.tasks = deque(rng.permutation(self.tasks))

    def __call__(self, n_tasks, rng=None):
        """
        Yield tasks from the queue.

        Parameters
        ----------
        n_tasks : int
            Number of tasks.
        rng : int or RandomState or Generator, optional
            Random number generator seed or object.

        Returns
        -------
        Generator

        """
        for __ in range(n_tasks):
            if len(self.tasks) == 0:
                raise ValueError("Task generator data has been exhausted.")

            task = self.tasks.pop()
            if self.repeat:
                self.tasks.appendleft(task)

            yield task


# # Radar
# class SearchTrackIID(BaseIID):  # TODO: integrate or deprecate (and `search_track` methods)
#     """Search and Track tasks based on 2020 TSRS paper."""

#     targets = dict(
#         HS={"duration": 0.036, "t_revisit": 2.5},
#         AHS={"duration": 0.036, "t_revisit": 5.0},
#         AHS_short={"duration": 0.018, "t_revisit": 5.0},
#         Trk_hi={"duration": 0.018, "t_revisit": 1.0},
#         Trk_med={"duration": 0.018, "t_revisit": 2.0},
#         Trk_low={"duration": 0.018, "t_revisit": 4.0},
#     )

#     def __init__(self, p=None, t_release_lim=(0.0, 0.018), rng=None):
#         durations, t_revisits = map(
#             np.array, zip(*[target.values() for target in self.targets.values()])
#         )
#         param_spaces = {
#             "duration": DiscreteSet(durations),
#             "t_release": spaces.Box(*t_release_lim, shape=(), dtype=float),
#             "slope": DiscreteSet(1 / t_revisits),
#             "t_drop": DiscreteSet(t_revisits + 0.1),
#             "l_drop": DiscreteSet([300.0]),
#         }

#         super().__init__(task_types.LinearDrop, param_spaces, rng)

#         if p is None:
#             # n = np.array([28, 43, 49,  1,  1,  1])
#             # t_r = np.array([2.5, 5., 5., 1., 2., 4.])
#             # self.probs = np.array([0.36, 0.27, 0.31, 0.03, 0.02, 0.01])
# #             # proportionate to (# beams) / (revisit rate)
#             self.p = [0.36, 0.27, 0.31, 0.03, 0.02, 0.01]
#         else:
#             self.p = list(p)

#         self.t_release_lim = tuple(t_release_lim)

#     def _param_gen(self, rng):
#         """Randomly generate task parameters."""
#         duration, t_revisit = rng.choice(list(self.targets.values()), p=self.p).values()
#         params = {
#             "duration": duration,
#             "t_release": rng.uniform(*self.t_release_lim),
#             "slope": 1 / t_revisit,
#             "t_drop": t_revisit + 0.1,
#             "l_drop": 300.0,
#         }
#         return params

#     def __eq__(self, other):
#         if isinstance(other, SearchTrackIID):
#             return self.p == other.p and self.t_release_lim == other.t_release_lim
#         else:
#             return NotImplemented

#     def summary(self):
#         str_ = super().summary()
#         str_ += f"\nRelease time limits: {self.t_release_lim}"
#         df = pd.Series(dict(zip(self.targets.keys(), self.p)), name="Pr")
#         df_str = df.to_markdown(tablefmt="github", floatfmt=".3f")
#         str_ += f"\n\n{df_str}"
#         return str_


# def make_truncnorm(myclip_a, myclip_b, my_mean, my_std):
#     a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
#     return stats.truncnorm(a, b, loc=my_mean, scale=my_std)


# class Radar(BaseIID):
#     types_search = dict(
#         HS=dict(
#             pr=0.26,
#             t_release_rng=make_truncnorm(-5.9, -5.5, -5.7, 0.058).rvs,
#             t_release_space=spaces.Box(-5.9, -5.5, shape=(), dtype=float),
#             duration=0.036,
#             slope=0.17,
#             t_drop=5.98,
#             l_drop=300,
#         ),
#         AHS=dict(
#             pr=0.74,
#             t_release_rng=make_truncnorm(-11.8, -11.2, -11.5, 0.087).rvs,
#             t_release_space=spaces.Box(-11.8, -11.2, shape=(), dtype=float),
#             duration=0.036,
#             slope=0.085,
#             t_drop=11.86,
#             l_drop=300,
#         ),
#     )

#     types_track = dict(
#         HS=dict(
#             pr=0.269,
#             t_release_rng=make_truncnorm(-7.5, -6.8, -7.14, 0.092).rvs,
#             t_release_space=spaces.Box(-7.5, -6.8, shape=(), dtype=float),
#             duration=0.036,
#             slope=0.17,
#             t_drop=5.98,
#             l_drop=300,
#         ),
#         AHS=dict(
#             pr=0.696,
#             t_release_rng=make_truncnorm(-14.75, -13.75, -14.25, 0.132).rvs,
#             t_release_space=spaces.Box(-14.75, -13.75, shape=(), dtype=float),
#             duration=0.036,
#             slope=0.085,
#             t_drop=11.86,
#             l_drop=300,
#         ),
#         track_low=dict(
#             pr=0.012,
#             t_release_rng=lambda: -1.044,
#             t_release_space=DiscreteSet([-1.044]),
#             duration=0.036,
#             slope=1.0,
#             t_drop=1.1,
#             l_drop=500,
#         ),
#         track_high=dict(
#             pr=0.023,
#             t_release_rng=lambda: -0.53,
#             t_release_space=DiscreteSet([-0.53]),
#             duration=0.036,
#             slope=2.0,
#             t_drop=0.6,
#             l_drop=500,
#         ),
#     )

#     def __init__(self, mode, rng=None):
#         if mode == "search":
#             self.types = self.types_search
#         elif mode == "track":
#             self.types = self.types_track
#         else:
#             raise ValueError

#         param_spaces = {}
#         for name in task_types.LinearDrop.param_names:
#             if name == "t_release":
#                 # param_spaces[name] = spaces.Box(-np.inf, np.inf, shape=(), dtype=float)
#                 lows, highs = zip(
#                     *(get_space_lims(params["t_release_space"]) for params in self.types.values())
#                 )
#                 param_spaces[name] = spaces.Box(min(lows), max(highs), shape=(), dtype=float)
#             else:
#                 param_spaces[name] = DiscreteSet(
#                     np.unique([params[name] for params in self.types.values()])
#                 )

#         super().__init__(task_types.LinearDrop, param_spaces, rng)

#     @cached_property
#     def p(self):
#         return np.array([params["pr"] for params in self.types.values()])

#     def __call__(self, n_tasks, rng=None):
#         """Randomly generate tasks."""
#         rng = self._get_rng(rng)
#         for __ in range(n_tasks):
#             yield self.cls_task(**self._param_gen(rng))

#     def _param_gen(self, rng):
#         """Randomly generate task parameters."""
#         type_ = rng.choice(list(self.types.keys()), p=self.p)
#         params = self.types[type_].copy()
#         params["name"] = type_
#         params["t_release"] = params["t_release_rng"]()
#         del params["pr"], params["t_release_rng"], params["t_release_space"]
#         # params['t_release'] = rng.normal(params['t_release_mean'], params['t_release_std'])
#         # del params['t_release_mean']
#         # del params['t_release_std']

#         return params
