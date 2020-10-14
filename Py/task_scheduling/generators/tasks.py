"""Generator objects for tasks."""

from types import MethodType
from abc import ABC, abstractmethod

import numpy as np

from util.generic import check_rng
from task_scheduling.tasks import ReluDrop as ReluDropTask

np.set_printoptions(precision=2)

# TODO: add TSRS search/track task generators


class Base(ABC):
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

        self.cls_task = cls_task
        self.rng = check_rng(rng)

        if param_lims is None:
            self.param_lims = {name: (-float('inf'), float('inf')) for name in self.cls_task.param_names}
        else:
            self.param_lims = param_lims

    @abstractmethod
    def __call__(self, n_tasks):
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


class BaseIID(Base):
    """Base class for generation of independently and identically distributed random task objects."""

    def __call__(self, n_tasks):
        """Randomly generate tasks."""
        for _ in range(n_tasks):
            yield self.cls_task(**self.param_gen())

    @abstractmethod
    def param_gen(self):
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
        Callable object with 'self' argument, for use as the 'param_gen' method.
    param_lims : dict, optional
        Maps parameter name strings to 2-tuples of parameter lower and upper bounds.
    rng : int or RandomState or Generator, optional
        Random number generator seed or object.

    """

    def __init__(self, cls_task, param_gen, param_lims=None, rng=None):
        super().__init__(cls_task, param_lims, rng)
        self._param_gen = MethodType(param_gen, self)

    def param_gen(self):
        return self._param_gen()

    @classmethod
    def relu_drop(cls, param_gen, param_lims=None, rng=None):
        return cls(ReluDropTask, param_gen, param_lims, rng)


class ContinuousUniformIID(BaseIID):
    """Generator of uniformly IID random task objects."""

    def param_gen(self):
        """Randomly generate task parameters."""
        return {name: self.rng.uniform(*self.param_lims[name]) for name in self.cls_task.param_names}

    @classmethod
    def relu_drop(cls, duration_lim=(3, 6), t_release_lim=(0, 4), slope_lim=(0.5, 2),
                  t_drop_lim=(6, 12), l_drop_lim=(35, 50), rng=None):
        """Factory constructor for ReluDrop task objects."""

        param_lims = {'duration': duration_lim, 't_release': t_release_lim,
                      'slope': slope_lim, 't_drop': t_drop_lim, 'l_drop': l_drop_lim}

        return cls(ReluDropTask, param_lims, rng)

    def __eq__(self, other):
        if isinstance(other, ContinuousUniformIID):
            return self.cls_task == other.cls_task and self.param_lims == other.param_lims
        else:
            return NotImplemented


class DiscreteIID(BaseIID):
    """
    Generator of discrete IID random task objects.

    Parameters
    ----------
    cls_task : class
        Class for instantiating task objects.
    param_probs: dict
        Maps parameter name strings to dictionaries mapping values to probabilities.
    rng : int or RandomState or Generator, optional
        Random number generator seed or object.

    """

    def __init__(self, cls_task, param_probs, rng=None):
        param_lims = {name: (min(param_probs[name].keys()), max(param_probs[name].keys()))
                      for name in cls_task.param_names}
        super().__init__(cls_task, param_lims, rng)

        self.param_probs = param_probs

    def param_gen(self):
        """Randomly generate task parameters."""
        return {name: self.rng.choice(self.param_probs[name].keys(), p=self.param_probs[name].values())
                for name in self.cls_task.param_names}

    @classmethod
    def relu_drop(cls, duration_prob, t_release_prob, slope_prob, t_drop_prob, l_drop_prob, rng=None):
        """Factory constructor for ReluDrop task objects."""

        param_probs = {'duration': duration_prob, 't_release': t_release_prob,
                       'slope': slope_prob, 't_drop': t_drop_prob, 'l_drop': l_drop_prob}

        return cls(ReluDropTask, param_probs, rng)

    def __eq__(self, other):
        if isinstance(other, DiscreteIID):
            return self.cls_task == other.cls_task and self.param_probs == other.param_probs
        else:
            return NotImplemented


class Deterministic(Base):
    def __init__(self, tasks, param_lims=None, rng=None):
        """
        Deterministic task generator.

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

        super().__init__(cls_task, param_lims, rng)

    def __call__(self, n_tasks):
        """Yields the deterministic tasks in order."""

        if n_tasks != len(self.tasks):
            raise ValueError(f"Number of tasks must be {len(self.tasks)}.")

        for task in self.tasks:
            yield task

    def __eq__(self, other):
        if isinstance(other, Deterministic):
            return self.tasks == other.tasks
        else:
            return NotImplemented

    @classmethod
    def relu_drop(cls, n_tasks, rng):
        task_gen = ContinuousUniformIID.relu_drop()

        tasks = list(task_gen(n_tasks))
        param_lims = task_gen.param_lims

        return cls(tasks, param_lims, rng)


class Permutation(Deterministic):
    """Generates fixed tasks in a uniformly random order."""

    def __call__(self, n_tasks):
        """Yields the deterministic tasks in a random order."""

        if n_tasks != len(self.tasks):
            raise ValueError(f"Number of tasks must be {len(self.tasks)}.")

        for task in self.rng.permutation(self.tasks).tolist():
            yield task


def main():
    a = ContinuousUniformIID.relu_drop(duration_lim=(3, 6), t_release_lim=(0, 4), slope_lim=(0.5, 2),
                                       t_drop_lim=(6, 12), l_drop_lim=(35, 50), rng=None)

    b = ContinuousUniformIID.relu_drop(duration_lim=(3, 6), t_release_lim=(0, 4), slope_lim=(0.5, 2),
                                       t_drop_lim=(6, 12), l_drop_lim=(35, 50), rng=None)

    assert a == b


if __name__ == '__main__':
    main()
