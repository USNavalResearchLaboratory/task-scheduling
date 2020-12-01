"""Generator objects for tasks."""

from types import MethodType
from collections import deque
from abc import ABC, abstractmethod

import numpy as np

from task_scheduling.util.generic import RandomGeneratorMixin
from task_scheduling import tasks as task_types

np.set_printoptions(precision=2)


class Base(RandomGeneratorMixin, ABC):
    def __init__(self, cls_task, param_lims=None, rng=None):
        """
        Base class for generation of task objects.

        Parameters
        ----------
        cls_task : class
            Class for instantiating task objects.
        param_lims : dict, optional
            Maps parameter name strings to 2-tuples of parameter lower and upper bounds.
        rng : int or RandomState or Generator, optional
            Random number generator seed or object.

        """

        super().__init__(rng)
        self.cls_task = cls_task

        if param_lims is None:
            self.param_lims = {name: (-float('inf'), float('inf')) for name in self.cls_task.param_names}
        else:
            self.param_lims = param_lims

    @abstractmethod
    def __call__(self, n_tasks, rng=None):
        """Yield tasks."""
        raise NotImplementedError

    @property
    def default_features(self):
        """Returns a NumPy structured array of default features, the task parameters."""
        features = np.array(list(zip(self.cls_task.param_names,
                                     [lambda task, name=name_: getattr(task, name)
                                      for name_ in self.cls_task.param_names],  # note: late-binding closure
                                     self.param_lims.values())),
                            dtype=[('name', '<U16'), ('func', object), ('lims', np.float, 2)])

        return features


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
    param_lims : dict, optional
        Maps parameter name strings to 2-tuples of parameter lower and upper bounds.
    rng : int or RandomState or Generator, optional
        Random number generator seed or object.

    """

    def __init__(self, cls_task, param_gen, param_lims=None, rng=None):
        super().__init__(cls_task, param_lims, rng)
        self._param_gen_init = MethodType(param_gen, self)

    def _param_gen(self, rng):
        return self._param_gen_init(rng)

    @classmethod
    def relu_drop(cls, param_gen, param_lims=None, rng=None):
        return cls(task_types.ReluDrop, param_gen, param_lims, rng)


class ContinuousUniformIID(BaseIID):
    """Generates I.I.D. tasks with independently uniform continuous parameters."""

    def _param_gen(self, rng):
        """Randomly generate task parameters."""
        return {name: rng.uniform(*self.param_lims[name]) for name in self.cls_task.param_names}

    def __eq__(self, other):
        if isinstance(other, ContinuousUniformIID):
            return self.cls_task == other.cls_task and self.param_lims == other.param_lims
        else:
            return NotImplemented

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
        param_lims = {name: (min(param_probs[name].keys()), max(param_probs[name].keys()))
                      for name in self.cls_task.param_names}

        super().__init__(cls_task, param_lims, rng)

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

    @classmethod
    def uniform_relu_drop(cls, duration_vals, t_release_vals, slope_vals, t_drop_vals, l_drop_vals, rng=None):
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

    def __init__(self, probs=None, t_release_lim=(0., 0.), rng=None):
        self.targets = [{'duration': .036, 't_revisit': 2.5},
                        {'duration': .036, 't_revisit': 5.0},
                        {'duration': .018, 't_revisit': 5.0},
                        {'duration': .018, 't_revisit': 1.0},
                        {'duration': .018, 't_revisit': 2.0},
                        {'duration': .018, 't_revisit': 4.0},
                        ]

        durations, t_revisits = zip(*[target.values() for target in self.targets])
        param_lims = {'duration': (min(durations), max(durations)),
                      't_release': t_release_lim,
                      'slope': (1 / max(t_revisits), 1 / min(t_revisits)),
                      't_drop': (min(t_revisits) + 0.1, max(t_revisits) + 0.1),
                      'l_drop': (300., 300.)
                      }

        super().__init__(task_types.ReluDrop, param_lims, rng)

        if probs is None:
            self.probs = [.1, .2, .4, .1, .1, .1]
        else:
            self.probs = probs

    def _param_gen(self, rng):
        """Randomly generate task parameters."""
        duration, t_revisit = rng.choice(self.targets, p=self.probs).values()
        params = {'duration': duration,
                  't_release': rng.uniform(*self.param_lims['t_release']),
                  'slope': 1 / t_revisit,
                  't_drop': t_revisit + 0.1,
                  'l_drop': 300.
                  }
        return params


class Fixed(Base, ABC):
    def __init__(self, tasks, param_lims=None, rng=None):
        """
        Permutation task generator.

        Parameters
        ----------
        tasks : Iterable of tasks.Generic
        param_lims : dict, optional
            Maps parameter name strings to 2-tuples of parameter lower and upper bounds.
        rng : int or RandomState or Generator, optional
            Random number generator seed or object.
        """

        self.tasks = list(tasks)

        cls_task = self.tasks[0].__class__
        if not all(isinstance(task, cls_task) for task in self.tasks[1:]):
            raise TypeError("All tasks must be of the same type.")

        if param_lims is None:
            param_lims = {}
            for name in cls_task.param_names:
                values = [getattr(task, name) for task in tasks]
                param_lims[name] = (min(values), max(values))

        super().__init__(cls_task, param_lims, rng)

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
        param_lims = task_gen.param_lims
        return cls(tasks, param_lims, rng)

    @classmethod
    def relu_drop(cls, n_tasks, rng=None, **relu_lims):
        task_gen = ContinuousUniformIID.relu_drop(**relu_lims)
        return cls._task_gen_to_fixed(n_tasks, task_gen, rng)

    @classmethod
    def search_track(cls, n_tasks, probs=None, t_release_lim=(0., 0.), rng=None):
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


# FIXME: WIP!!!
class Queue(Base):
    def __init__(self, tasks, param_lims=None, rng=None):

        cls_task = tasks[0].__class__
        if not all(isinstance(task, cls_task) for task in tasks[1:]):
            raise TypeError("All tasks must be of the same type.")

        super().__init__(cls_task, param_lims, rng)
        self.tasks = deque()
        self.add_tasks(tasks)

    @property
    def n_tasks(self):
        return len(self.tasks)

    def __call__(self, n_tasks, rng=None):
        for __ in range(n_tasks):
            yield self.tasks.pop()

    def add_tasks(self, tasks):
        self.tasks.extendleft(tasks[::-1])

        # for task in tasks:        # TODO: move to task counting wrapper?
        #     try:
        #         task.count += 1
        #     except AttributeError:
        #         task.count = 0
        #
        #     self.tasks.append(task)

    def update(self, tasks, t_ex):
        for task, t_ex_i in zip(tasks, t_ex):
            task.t_release = t_ex_i + task.duration

        self.add_tasks(tasks)

        # TODO: calculate/return channel avails?



