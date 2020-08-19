"""Generator objects for tasks."""

from types import MethodType

import numpy as np
import matplotlib.pyplot as plt

from util.generic import check_rng
from task_scheduling.tasks import ReluDrop as ReluDropTask

np.set_printoptions(precision=2)
plt.style.use('seaborn')


class BaseIID:
    """
    Base class for generation of independently and identically distributed random task objects.

    Parameters
    ----------
    cls_task : class
        Class for instantiating task objects.
    param_lims : dict, optional
        Maps parameter name strings to 2-tuples of parameter lower and upper bounds.
    rng : int or RandomState or Generator, optional
        Random number generator seed or object.

    """

    def __init__(self, cls_task, param_lims=None, rng=None):
        self.cls_task = cls_task
        self.rng = check_rng(rng)

        if param_lims is None:
            self.param_lims = {name: (-float('inf'), float('inf')) for name in self.cls_task.param_names}
        else:
            self.param_lims = param_lims

    def __call__(self, n_tasks=1):
        """Randomly generate tasks."""
        for _ in range(n_tasks):
            yield self.cls_task(**self.param_gen())

    def param_gen(self):
        """Randomly generate task parameters."""
        raise NotImplementedError

    @property
    def param_repr_lim(self):
        """Low and high tuples bounding parametric task representations."""
        return zip(*self.param_lims.values())


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
    def relu_drop(cls, duration_lim, t_release_lim, slope_lim, t_drop_lim, l_drop_lim, rng=None):
        """Factory constructor for ReluDrop task objects."""
        param_lims = {'duration': duration_lim, 't_release': t_release_lim,
                      'slope': slope_lim, 't_drop': t_drop_lim, 'l_drop': l_drop_lim}

        return cls(ReluDropTask, param_lims, rng)

    def __eq__(self, other):
        conditions = [self.cls_task == other.cls_task,
                      self.param_lims == other.param_lims]
        return True if all(conditions) else False


class DiscreteIID(BaseIID):
    def __init__(self, cls_task, param_probs, rng=None):
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
        conditions = [self.cls_task == other.cls_task,
                      self.param_probs == other.param_probs]
        return True if all(conditions) else False


# class Deterministic:
#     def __init__(self, tasks):
#         self.tasks = list(tasks)
#
#     def __call__(self, n_tasks):
#         if n_tasks != len(self.tasks):
#             raise ValueError(f"Number of tasks must be {len(self.tasks)}.")
#         return self.tasks
#
#     def __eq__(self, other):
#         if not isinstance(other, Deterministic):
#             return False
#         else:
#             return True if self.tasks == other.tasks else False
#
#     @property
#     def param_repr_lim(self):
#         """Low and high tuples bounding parametric task representations."""
#         _type = type(self.tasks[0])
#         if all(type(task) == _type for task in self.tasks[1:]):     # all tasks have the same type
#             _array = np.array((list(task.params.values()) for task in self.tasks))
#             return _array.min(axis=1), _array.max(axis=1)
#         else:
#             raise TypeError("All tasks must be of the same type.")
#
#
# class Permutation(Deterministic):
#     def __init__(self, tasks, rng=None):
#         super().__init__(tasks)
#         self.rng = check_rng(rng)
#
#     def __call__(self, n_tasks):
#         if n_tasks != len(self.tasks):
#             raise ValueError(f"Number of tasks must be {len(self.tasks)}.")
#         return self.rng.permutation(self.tasks).tolist()


def main():
    a = ContinuousUniformIID.relu_drop(duration_lim=(3, 6), t_release_lim=(0, 4), slope_lim=(0.5, 2),
                                       t_drop_lim=(6, 12), l_drop_lim=(35, 50), rng=None)

    b = ContinuousUniformIID.relu_drop(duration_lim=(3, 6), t_release_lim=(0, 4), slope_lim=(0.5, 2),
                                       t_drop_lim=(6, 12), l_drop_lim=(35, 50), rng=None)

    assert a == b


if __name__ == '__main__':
    main()
