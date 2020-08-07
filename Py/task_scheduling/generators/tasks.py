"""Generator objects for tasks."""

from types import MethodType

import numpy as np
import matplotlib.pyplot as plt
from util.generic import check_rng

from task_scheduling.tasks import ReluDrop as ReluDropTask

np.set_printoptions(precision=2)
plt.style.use('seaborn')

# TODO: generalize, add docstrings


# Task generators
class Base:
    def __init__(self, rng=None):
        self.rng = check_rng(rng)

    def __call__(self, n_tasks):
        raise NotImplementedError

    @property
    def param_repr_lim(self):
        raise NotImplementedError


class Deterministic(Base):
    def __init__(self, tasks, rng=None):
        super().__init__(rng)
        self.tasks = tasks      # list of tasks

    def __call__(self, n_tasks):
        return self.tasks

    def __eq__(self, other):
        if not isinstance(other, Deterministic):
            return False

        return True if self.tasks == other.tasks else False

    @property
    def param_repr_lim(self):
        raise NotImplementedError


class PermuteOrder(Base):
    def __init__(self, tasks, rng=None):
        super().__init__(rng)
        self.tasks = tasks      # list of tasks

    def __call__(self, n_tasks):
        return self.rng.permutation(self.tasks)

    def __eq__(self, other):
        if not isinstance(other, Deterministic):
            return False

        return True if self.tasks == other.tasks else False

    @property
    def param_repr_lim(self):
        raise NotImplementedError


class ReluDrop(Base):
    """
    Generator of random ReluDrop objects.

    Parameters
    ----------
    duration_lim : tuple of float or list of float
        Lower and upper limits for random generation of tasks.ReluDrop.duration
    t_release_lim : tuple of float or list of float
        Lower and upper limits for random generation of tasks.ReluDrop.t_release
    slope_lim : tuple of float or list of float
        Lower and upper limits for random generation of tasks.ReluDrop.slope
    t_drop_lim : tuple of float or list of float
        Lower and upper limits for random generation of tasks.ReluDrop.t_drop
    l_drop_lim : tuple of float or list of float
        Lower and upper limits for random generation of tasks.ReluDrop.l_drop

    """

    # FIXME FIXME: generalize rng usage for non-uniform? or subclass to preserve parametric reprs?

    def __init__(self, param_gen, param_lims=None, rng=None):
        super().__init__(rng)
        self.param_gen = MethodType(param_gen, self)
        self.param_lims = param_lims

    @classmethod
    def iid_uniform(cls, duration_lim, t_release_lim, slope_lim, t_drop_lim, l_drop_lim, rng=None):
        def _param_gen(self):
            return {'duration': self.rng.uniform(*self.param_lims['duration']),
                    't_release': self.rng.uniform(*self.param_lims['t_release']),
                    'slope': self.rng.uniform(*self.param_lims['slope']),
                    't_drop': self.rng.uniform(*self.param_lims['t_drop']),
                    'l_drop': self.rng.uniform(*self.param_lims['l_drop']),
                    }

        param_lims = {'duration': duration_lim,
                      't_release': t_release_lim,
                      'slope': slope_lim,
                      't_drop': t_drop_lim,
                      'l_drop': l_drop_lim,
                      }

        return cls(_param_gen, param_lims, rng)

    @classmethod
    def iid_discrete(cls):
        pass

    @classmethod
    def search_and_track(cls):
        pass

    def __call__(self, n_tasks):
        """Randomly generate a list of tasks."""
        for _ in range(n_tasks):
            yield ReluDropTask(**self.param_gen())

    def __eq__(self, other):
        if not isinstance(other, ReluDrop):
            return False

        conditions = [self.param_gen.__code__ == other.param_gen.__code__, self.param_lims == other.param_lims]
        # conditions = [self.duration_lim == other.duration_lim,
        #               self.t_release_lim == other.t_release_lim,
        #               self.slope_lim == other.slope_lim,
        #               self.t_drop_lim == other.t_drop_lim,
        #               self.l_drop_lim == other.l_drop_lim]

        return True if all(conditions) else False

    @property
    def param_repr_lim(self):
        """Low and high tuples bounding parametric task representations."""

        # return zip(self.duration_lim, self.t_release_lim, self.slope_lim, self.t_drop_lim, self.l_drop_lim)
        return zip(*self.param_lims.values())   # TODO: move to Base?

# class ReluDrop(Base):
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
#     # TODO: generalize rng usage for non-uniform? or subclass to preserve parametric reprs?
#
#     def __init__(self, duration_lim, t_release_lim, slope_lim, t_drop_lim, l_drop_lim, rng=None):
#         super().__init__(rng)
#         self.duration_lim = duration_lim
#         self.t_release_lim = t_release_lim
#         self.slope_lim = slope_lim
#         self.t_drop_lim = t_drop_lim
#         self.l_drop_lim = l_drop_lim
#
#     def __call__(self, n_tasks):
#         """Randomly generate a list of tasks."""
#         for _ in range(n_tasks):
#             yield ReluDropTask(self.rng.uniform(*self.duration_lim),
#                                self.rng.uniform(*self.t_release_lim),
#                                self.rng.uniform(*self.slope_lim),
#                                self.rng.uniform(*self.t_drop_lim),
#                                self.rng.uniform(*self.l_drop_lim),
#                                )
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


def main():

    a = ReluDrop.iid_uniform(duration_lim=(3, 6), t_release_lim=(0, 4), slope_lim=(0.5, 2),
                             t_drop_lim=(6, 12), l_drop_lim=(35, 50), rng=None)

    b = ReluDrop.iid_uniform(duration_lim=(3, 6), t_release_lim=(0, 4), slope_lim=(0.5, 2),
                             t_drop_lim=(6, 12), l_drop_lim=(35, 50), rng=None)

    assert a == b


if __name__ == '__main__':
    main()
