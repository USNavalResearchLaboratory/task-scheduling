"""Generator objects for tasks."""

import numpy as np
import matplotlib.pyplot as plt

from util.generic import check_rng
from task_scheduling.tasks import ReluDrop as ReluDropTask

np.set_printoptions(precision=2)
plt.style.use('seaborn')

# TODO: generalize, add docstrings


class BaseIID:
    """
    Generator of independently and identically distributed random task objects.

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

    def __eq__(self, other):
        conditions = [self.cls_task == other.cls_task,
                      self.param_lims == other.param_lims]
        return True if all(conditions) else False

    @property
    def param_repr_lim(self):
        """Low and high tuples bounding parametric task representations."""
        return zip(*self.param_lims.values())


class UniformIID(BaseIID):
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


# TODO: delete?
# class RandomIID:
#     def __init__(self, cls_task, param_gen, param_lims=None, rng=None):
#         self.cls_task = cls_task
#         self.param_gen = MethodType(param_gen, self)
#         self.rng = check_rng(rng)
#
#         if param_lims is None:
#             self.param_lims = {name: (-float('inf'), float('inf')) for name in self.cls_task.param_names}
#         else:
#             self.param_lims = param_lims
#
#     def __call__(self, n_tasks=1):
#         """Randomly generate tasks."""
#         for _ in range(n_tasks):
#             yield self.cls_task(**self.param_gen())
#
#     @property
#     def param_repr_lim(self):
#         """Low and high tuples bounding parametric task representations."""
#         return zip(*self.param_lims.values())
#
#
# class ReluDrop(RandomIID):
#     """
#     Generator of random ReluDrop objects.
#
#     Parameters
#     ----------
#     duration_lim : tuple of float or list of float
#         Lower and upper limits for random generation of tasks.ReluDrop.duration
#     t_release_lim : tuple of float or list of float
#         Lower and upper limits for random generation of tasks.ReluDrop.t_release
#     slope_lim : tuple of float or list of float
#         Lower and upper limits for random generation of tasks.ReluDrop.slope
#     t_drop_lim : tuple of float or list of float
#         Lower and upper limits for random generation of tasks.ReluDrop.t_drop
#     l_drop_lim : tuple of float or list of float
#         Lower and upper limits for random generation of tasks.ReluDrop.l_drop
#
#     """
#
#
#     def __init__(self, param_gen, param_lims=None, rng=None):
#         super().__init__(ReluDropTask, param_gen, param_lims, rng)
#
#         if param_lims is None:
#             self.param_lims = {name: (0., float('inf')) for name in self.cls_task.param_names}
#         else:
#             self.param_lims = param_lims
#
#     @classmethod
#     def iid_uniform(cls, duration_lim, t_release_lim, slope_lim, t_drop_lim, l_drop_lim, rng=None):
#         def _param_gen(self):
#             return {name: self.rng.uniform(*self.param_lims[name]) for name in self.cls_task.param_names}
#
#         param_lims = {'duration': duration_lim,
#                       't_release': t_release_lim,
#                       'slope': slope_lim,
#                       't_drop': t_drop_lim,
#                       'l_drop': l_drop_lim,
#                       }
#
#         return cls(_param_gen, param_lims, rng)
#
#     @classmethod
#     def iid_discrete(cls, duration_prob, t_release_prob, slope_prob, t_drop_prob, l_drop_prob, rng=None):
#         def _param_gen(self):
#             return {'duration': self.rng.choice(duration_prob.keys(), p=duration_prob.values()),
#                     't_release': self.rng.choice(t_release_prob.keys(), p=t_release_prob.values()),
#                     'slope': self.rng.choice(slope_prob.keys(), p=slope_prob.values()),
#                     't_drop': self.rng.choice(t_drop_prob.keys(), p=t_drop_prob.values()),
#                     'l_drop': self.rng.choice(l_drop_prob.keys(), p=l_drop_prob.values()),
#                     }
#
#         param_lims = {'duration': (min(duration_prob.keys()), max(duration_prob.keys())),
#                       't_release': (min(t_release_prob.keys()), max(t_release_prob.keys())),
#                       'slope': (min(slope_prob.keys()), max(slope_prob.keys())),
#                       't_drop': (min(t_drop_prob.keys()), max(t_drop_prob.keys())),
#                       'l_drop': (min(l_drop_prob.keys()), max(l_drop_prob.keys())),
#                       }
#
#         return cls(_param_gen, param_lims, rng)
#
#     def __eq__(self, other):
#         if not isinstance(other, ReluDrop):
#             return False
#
#         conditions = [self.param_gen.__code__ == other.param_gen.__code__,
#                       self.param_lims == other.param_lims]
#         return True if all(conditions) else False


# class ReluDrop(RandomIID):
#     """
#     Generator of random ReluDrop objects.
#
#     Parameters
#     ----------
#     duration_lim : tuple of float or list of float
#         Lower and upper limits for random generation of tasks.ReluDrop.duration
#     t_release_lim : tuple of float or list of float
#         Lower and upper limits for random generation of tasks.ReluDrop.t_release
#     slope_lim : tuple of float or list of float
#         Lower and upper limits for random generation of tasks.ReluDrop.slope
#     t_drop_lim : tuple of float or list of float
#         Lower and upper limits for random generation of tasks.ReluDrop.t_drop
#     l_drop_lim : tuple of float or list of float
#         Lower and upper limits for random generation of tasks.ReluDrop.l_drop
#
#     """
#
#
#     def __init__(self, duration_lim, t_release_lim, slope_lim, t_drop_lim, l_drop_lim, rng=None):
#         super().__init__(cls_task=ReluDropTask, rng=rng)
#         self.duration_lim = duration_lim
#         self.t_release_lim = t_release_lim
#         self.slope_lim = slope_lim
#         self.t_drop_lim = t_drop_lim
#         self.l_drop_lim = l_drop_lim
#
#     def __call__(self, n_tasks):
#         """Randomly generate a list of tasks."""
#         for _ in range(n_tasks):
#             yield self.cls_task(self.rng.uniform(*self.duration_lim),
#                                 self.rng.uniform(*self.t_release_lim),
#                                 self.rng.uniform(*self.slope_lim),
#                                 self.rng.uniform(*self.t_drop_lim),
#                                 self.rng.uniform(*self.l_drop_lim),
#                                 )
#
#     def __eq__(self, other):
#         if not isinstance(other, ReluDrop):
#             return False
#
#         conditions = [self.duration_lim == other.duration_lim,
#                       self.t_release_lim == other.t_release_lim,
#                       self.slope_lim == other.slope_lim,
#                       self.t_drop_lim == other.t_drop_lim,
#                       self.l_drop_lim == other.l_drop_lim]
#
#         return True if all(conditions) else False
#
#     @property
#     def param_repr_lim(self):
#         """Low and high tuples bounding parametric task representations."""
#         return zip(self.duration_lim, self.t_release_lim, self.slope_lim, self.t_drop_lim, self.l_drop_lim)


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
    a = UniformIID.relu_drop(duration_lim=(3, 6), t_release_lim=(0, 4), slope_lim=(0.5, 2),
                             t_drop_lim=(6, 12), l_drop_lim=(35, 50), rng=None)

    b = UniformIID.relu_drop(duration_lim=(3, 6), t_release_lim=(0, 4), slope_lim=(0.5, 2),
                             t_drop_lim=(6, 12), l_drop_lim=(35, 50), rng=None)

    assert a == b


if __name__ == '__main__':
    main()
