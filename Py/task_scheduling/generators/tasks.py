"""Generator objects for tasks."""

from types import MethodType
from abc import ABC, abstractmethod

import numpy as np

from util.generic import RandomGeneratorMixin
from task_scheduling.tasks import ReluDrop as ReluDropTask

np.set_printoptions(precision=2)

# TODO: add TSRS search/track task generators


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


class BaseIID(Base):
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
        return cls(ReluDropTask, param_gen, param_lims, rng)


class ContinuousUniformIID(BaseIID):
    """Generator of uniformly IID random task objects."""

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

        return cls(ReluDropTask, param_lims, rng)


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

    def _param_gen(self, rng):
        """Randomly generate task parameters."""
        return {name: rng.choice(self.param_probs[name].keys(), p=self.param_probs[name].values())
                for name in self.cls_task.param_names}

    def __eq__(self, other):
        if isinstance(other, DiscreteIID):
            return self.cls_task == other.cls_task and self.param_probs == other.param_probs
        else:
            return NotImplemented

    @classmethod
    def relu_drop(cls, duration_prob, t_release_prob, slope_prob, t_drop_prob, l_drop_prob, rng=None):
        """Factory constructor for ReluDrop task objects."""

        param_probs = {'duration': duration_prob, 't_release': t_release_prob,
                       'slope': slope_prob, 't_drop': t_drop_prob, 'l_drop': l_drop_prob}

        return cls(ReluDropTask, param_probs, rng)


class Permutation(Base):
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

        super().__init__(cls_task, param_lims, rng)

    def __call__(self, n_tasks, rng=None):
        """Yields the deterministic tasks in a random order."""
        if n_tasks != len(self.tasks):
            raise ValueError(f"Number of tasks must be {len(self.tasks)}.")

        rng = self._get_rng(rng)
        for task in rng.permutation(self.tasks).tolist():
            yield task

    def __eq__(self, other):
        if isinstance(other, Permutation):
            return self.tasks == other.tasks
        else:
            return NotImplemented

    @classmethod
    def relu_drop(cls, n_tasks, rng=None):
        task_gen = ContinuousUniformIID.relu_drop()

        tasks = list(task_gen(n_tasks))
        param_lims = task_gen.param_lims

        return cls(tasks, param_lims, rng)


class Deterministic(Permutation):
    """Generates fixed tasks in a deterministic order."""

    def __call__(self, n_tasks, rng=None):
        """Yields the deterministic tasks in order."""
        if n_tasks != len(self.tasks):
            raise ValueError(f"Number of tasks must be {len(self.tasks)}.")

        for task in self.tasks:
            yield task


def main():
    a = ContinuousUniformIID.relu_drop(duration_lim=(3, 6), t_release_lim=(0, 4), slope_lim=(0.5, 2),
                                       t_drop_lim=(6, 12), l_drop_lim=(35, 50), rng=None)

    b = ContinuousUniformIID.relu_drop(duration_lim=(3, 6), t_release_lim=(0, 4), slope_lim=(0.5, 2),
                                       t_drop_lim=(6, 12), l_drop_lim=(35, 50), rng=None)

    assert a == b


if __name__ == '__main__':
    main()
