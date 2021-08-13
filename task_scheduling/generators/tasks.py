"""
Generator objects for tasks.

Notes
-----
Assumes all tasks are instances of the same class. Heterogeneous task types will be supported in a future version.

"""

from abc import ABC, abstractmethod
from collections import deque
from types import MethodType
from typing import Iterable

import numpy as np
import pandas as pd
from gym import spaces

from task_scheduling._core import RandomGeneratorMixin
from task_scheduling import tasks as task_types
from task_scheduling.learning.spaces import DiscreteSet


class Base(RandomGeneratorMixin, ABC):
    def __init__(self, cls_task, param_spaces=None, rng=None):
        """
        Base class for generation of task objects.

        Parameters
        ----------
        cls_task : class
            Class for instantiating task objects.
        param_spaces : dict, optional
            Maps parameter name strings to gym.spaces.Space objects
        rng : int or RandomState or Generator, optional
            Random number generator seed or object.

        """

        super().__init__(rng)
        self.cls_task = cls_task

        if param_spaces is None:
            self.param_spaces = {name: spaces.Box(-np.inf, np.inf, shape=(), dtype=float)
                                 for name in self.cls_task.param_names}
        else:
            self.param_spaces = param_spaces

    @abstractmethod
    def __call__(self, n_tasks, rng=None):
        """Yield tasks."""
        raise NotImplementedError

    def summary(self, file=None):
        cls_str = self.__class__.__name__
        print(f"{cls_str}\n---", file=file)


class BaseIID(Base, ABC):
    """Base class for generation of independently and identically distributed random task objects."""

    def __call__(self, n_tasks, rng=None):
        """Randomly generate tasks."""
        rng = self._get_rng(rng)
        for _ in range(n_tasks):
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
        Callable object with 'self' argument, for use as the '_param_gen' method.
    param_spaces : dict, optional
            Maps parameter name strings to gym.spaces.Space objects
    rng : int or RandomState or Generator, optional
        Random number generator seed or object.

    """

    def __init__(self, cls_task, param_gen, param_spaces=None, rng=None):
        super().__init__(cls_task, param_spaces, rng)
        self._param_gen_init = MethodType(param_gen, self)

    def _param_gen(self, rng):
        return self._param_gen_init(rng)

    @classmethod
    def relu_drop(cls, param_gen, param_spaces=None, rng=None):
        return cls(task_types.ReluDrop, param_gen, param_spaces, rng)


class ContinuousUniformIID(BaseIID):
    """Generates I.I.D. tasks with independently uniform continuous parameters."""

    def __init__(self, cls_task, param_lims, rng=None):
        param_spaces = {name: spaces.Box(*param_lims[name], shape=(), dtype=float)
                        for name in cls_task.param_names}
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

    def summary(self, file=None):
        super().summary(file)

        df = pd.DataFrame({name: self.param_lims[name] for name in self.cls_task.param_names},
                          index=pd.CategoricalIndex(['low', 'high']))
        df_str = df.to_markdown(tablefmt='github', floatfmt='.3f')

        str_ = f"Task class: {self.cls_task.__name__}\n\n{df_str}\n"
        print(str_, file=file)

    @classmethod
    def relu_drop(cls, duration_lim=(3, 6), t_release_lim=(0, 4), slope_lim=(0.5, 2),
                  t_drop_lim=(6, 12), l_drop_lim=(35, 50), rng=None):
        """Factory constructor for ReluDrop task objects."""

        param_lims = {'duration': duration_lim, 't_release': t_release_lim,
                      'slope': slope_lim, 't_drop': t_drop_lim, 'l_drop': l_drop_lim}

        return cls(task_types.ReluDrop, param_lims, rng)


class DiscreteIID(BaseIID):
    """
    Generates I.I.D. tasks with independently discrete parameters.

    Parameters
    ----------
    cls_task : class
        Class for instantiating task objects.
    param_probs: dict of str to dict
        Maps parameter name strings to dictionaries mapping values to probabilities.
    rng : int or RandomState or Generator, optional
        Random number generator seed or object.

    """

    def __init__(self, cls_task, param_probs, rng=None):
        param_spaces = {name: DiscreteSet(param_probs[name].keys()) for name in cls_task.param_names}
        super().__init__(cls_task, param_spaces, rng)

        self.param_probs = param_probs

    def _param_gen(self, rng):
        """Randomly generate task parameters."""
        return {name: rng.choice(list(self.param_probs[name].keys()), p=list(self.param_probs[name].values()))
                for name in self.cls_task.param_names}

    def __eq__(self, other):
        if isinstance(other, DiscreteIID):
            return self.cls_task == other.cls_task and self.param_probs == other.param_probs
        else:
            return NotImplemented

    def summary(self, file=None):
        super().summary(file)

        str_ = f"Task class: {self.cls_task.__name__}\n\n"
        for name in self.cls_task.param_names:
            # s = pd.Series(self.param_probs[name], name='Pr')
            # s = pd.DataFrame(self.param_probs[name], index=pd.CategoricalIndex(['Pr']))
            s = pd.DataFrame({name: self.param_probs[name].keys(), 'Pr': self.param_probs[name].values()})
            str_ += s.to_markdown(tablefmt='github', floatfmt='.3f', index=False) + "\n\n"

        print(str_, file=file, end='')

        # print(f"Task class: {self.cls_task.__name__}")
        # for name in self.cls_task.param_names:
        #     print(f"\n{name}:")
        #     # s = pd.Series(self.param_probs[name], name='Pr')
        #     s = pd.DataFrame(self.param_probs[name], index=pd.CategoricalIndex(['Pr']))
        #     print(s.to_markdown(tablefmt='github', floatfmt='.3f', index=False))

        # df = pd.DataFrame({name: self.param_lims[name] for name in self.cls_task.param_names},
        #                   index=pd.CategoricalIndex(['low', 'high']))
        # print(df.to_markdown(tablefmt='github', floatfmt='.3f'))

    @classmethod
    def relu_drop_uniform(cls, duration_vals=(3, 6), t_release_vals=(0, 4), slope_vals=(0.5, 2), t_drop_vals=(6, 12),
                          l_drop_vals=(35, 50), rng=None):
        """Factory constructor for ReluDrop task objects."""

        param_probs = {'duration': dict(zip(duration_vals, np.ones(len(duration_vals)) / len(duration_vals))),
                       't_release': dict(zip(t_release_vals, np.ones(len(t_release_vals)) / len(t_release_vals))),
                       'slope': dict(zip(slope_vals, np.ones(len(slope_vals)) / len(slope_vals))),
                       't_drop': dict(zip(t_drop_vals, np.ones(len(t_drop_vals)) / len(t_drop_vals))),
                       'l_drop': dict(zip(l_drop_vals, np.ones(len(l_drop_vals)) / len(l_drop_vals))),
                       }
        return cls(task_types.ReluDrop, param_probs, rng)


class SearchTrackIID(BaseIID):
    """Search and Track tasks based on 2020 TSRS paper."""

    targets = {'HS': {'duration': .036, 't_revisit': 2.5},
               'AHS': {'duration': .036, 't_revisit': 5.0},
               'AHS_short': {'duration': .018, 't_revisit': 5.0},
               'Trk_hi': {'duration': .018, 't_revisit': 1.0},
               'Trk_med': {'duration': .018, 't_revisit': 2.0},
               'Trk_low': {'duration': .018, 't_revisit': 4.0},
               }

    def __init__(self, probs=None, t_release_lim=(0., .018), rng=None):
        durations, t_revisits = map(np.array, zip(*[target.values() for target in self.targets.values()]))
        param_spaces = {'duration': DiscreteSet(durations),
                        't_release': spaces.Box(*t_release_lim, shape=(), dtype=float),
                        'slope': DiscreteSet(1 / t_revisits),
                        't_drop': DiscreteSet(t_revisits + 0.1),
                        'l_drop': DiscreteSet([300.])
                        }

        super().__init__(task_types.ReluDrop, param_spaces, rng)

        if probs is None:
            # n = np.array([28, 43, 49,  1,  1,  1])
            # t_r = np.array([2.5, 5., 5., 1., 2., 4.])
            # self.probs = np.array([0.36, 0.27, 0.31, 0.03, 0.02, 0.01])  # proportionate to (# beams) / (revisit rate)
            self.probs = [0.36, 0.27, 0.31, 0.03, 0.02, 0.01]
        else:
            self.probs = list(probs)

        self.t_release_lim = tuple(t_release_lim)

    def _param_gen(self, rng):
        """Randomly generate task parameters."""
        duration, t_revisit = rng.choice(list(self.targets.values()), p=self.probs).values()
        params = {'duration': duration,
                  't_release': rng.uniform(*self.t_release_lim),
                  'slope': 1 / t_revisit,
                  't_drop': t_revisit + 0.1,
                  'l_drop': 300.
                  }
        return params

    def __eq__(self, other):
        if isinstance(other, SearchTrackIID):
            return self.probs == other.probs and self.t_release_lim == other.t_release_lim
        else:
            return NotImplemented

    def summary(self, file=None):
        super().summary(file)
        str_ = f'Release time limits: {self.t_release_lim}'
        df = pd.Series(dict(zip(self.targets.keys(), self.probs)), name='Pr')
        df_str = df.to_markdown(tablefmt='github', floatfmt='.3f')
        print(f"{str_}\n\n{df_str}\n", file=file)


class Fixed(Base, ABC):
    def __init__(self, tasks, param_spaces=None, rng=None):
        """
        Permutation task generator.

        Parameters
        ----------
        tasks : Sequence of task_scheduling.tasks.Base
        param_spaces : dict, optional
            Maps parameter name strings to gym.spaces.Space objects
        rng : int or RandomState or Generator, optional
            Random number generator seed or object.
        """

        cls_task = tasks[0].__class__
        if not all(isinstance(task, cls_task) for task in tasks[1:]):
            raise TypeError("All tasks must be of the same type.")

        if param_spaces is None:
            param_spaces = {name: DiscreteSet([getattr(task, name) for task in tasks]) for name in cls_task.param_names}

        super().__init__(cls_task, param_spaces, rng)

        self.tasks = list(tasks)

    @abstractmethod
    def __call__(self, n_tasks, rng=None):
        """Yield tasks."""
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
    def continuous_relu_drop(cls, n_tasks, rng=None, **relu_lims):
        task_gen = ContinuousUniformIID.relu_drop(**relu_lims)
        return cls._task_gen_to_fixed(n_tasks, task_gen, rng)

    @classmethod
    def discrete_relu_drop(cls, n_tasks, rng=None, **relu_vals):
        task_gen = DiscreteIID.relu_drop_uniform(**relu_vals)
        return cls._task_gen_to_fixed(n_tasks, task_gen, rng)

    @classmethod
    def search_track(cls, n_tasks, probs=None, t_release_lim=(0., .018), rng=None):
        task_gen = SearchTrackIID(probs, t_release_lim)
        return cls._task_gen_to_fixed(n_tasks, task_gen, rng)


class Deterministic(Fixed):
    def __call__(self, n_tasks, rng=None):
        """Yields the tasks in deterministic order."""
        if n_tasks != len(self.tasks):
            raise ValueError(f"Number of tasks must be {len(self.tasks)}.")

        for task in self.tasks:
            yield task


class Permutation(Fixed):
    def __call__(self, n_tasks, rng=None):
        """Yields the tasks in a uniformly random order."""
        if n_tasks != len(self.tasks):
            raise ValueError(f"Number of tasks must be {len(self.tasks)}.")

        rng = self._get_rng(rng)
        for task in rng.permutation(self.tasks).tolist():
            yield task


class Dataset(Fixed):
    def __init__(self, tasks, shuffle=False, repeat=False, param_spaces=None, rng=None):
        super().__init__(tasks, param_spaces, rng)

        self.tasks = deque()
        self.add_tasks(tasks)

        if shuffle:
            self.shuffle()

        self.repeat = repeat

    def add_tasks(self, tasks):
        if isinstance(tasks, Iterable):
            self.tasks.extendleft(tasks)
        else:
            self.tasks.appendleft(tasks)  # for single tasks

    def shuffle(self, rng=None):
        rng = self._get_rng(rng)
        self.tasks = deque(rng.permutation(self.tasks))

    def __call__(self, n_tasks, rng=None):
        for __ in range(n_tasks):
            if len(self.tasks) == 0:
                raise ValueError("Task generator data has been exhausted.")

            task = self.tasks.pop()
            if self.repeat:
                self.tasks.appendleft(task)

            yield task
