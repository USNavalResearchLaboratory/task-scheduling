"""Generator objects for tasks, channel availabilities, and complete tasking problems with optimal solutions."""

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
        Limits for random generation of tasks.ReluDrop.duration
    t_release_lim : tuple of float or list of float
        Limits for random generation of tasks.ReluDrop.t_release
    slope_lim : tuple of float or list of float
        Limits for random generation of tasks.ReluDrop.slope
    t_drop_lim : tuple of float or list of float
        Limits for random generation of tasks.ReluDrop.t_drop
    l_drop_lim : tuple of float or list of float
        Limits for random generation of tasks.ReluDrop.l_drop

    """

    # TODO: generalize rng usage for non-uniform? or subclass to preserve parametric reprs?

    def __init__(self, duration_lim, t_release_lim, slope_lim, t_drop_lim, l_drop_lim, rng=None):
        super().__init__(rng)
        self.duration_lim = duration_lim
        self.t_release_lim = t_release_lim
        self.slope_lim = slope_lim
        self.t_drop_lim = t_drop_lim
        self.l_drop_lim = l_drop_lim

    def __call__(self, n_tasks):
        """Randomly generate a list of tasks."""
        for _ in range(n_tasks):
            yield ReluDropTask(self.rng.uniform(*self.duration_lim),
                               self.rng.uniform(*self.t_release_lim),
                               self.rng.uniform(*self.slope_lim),
                               self.rng.uniform(*self.t_drop_lim),
                               self.rng.uniform(*self.l_drop_lim),
                               )

    def __eq__(self, other):
        if not isinstance(other, ReluDrop):
            return False

        conditions = [self.duration_lim == other.duration_lim,
                      self.t_release_lim == other.t_release_lim,
                      self.slope_lim == other.slope_lim,
                      self.t_drop_lim == other.t_drop_lim,
                      self.l_drop_lim == other.l_drop_lim]

        return True if all(conditions) else False

    @property
    def param_repr_lim(self):
        """Low and high tuples bounding parametric task representations."""
        return zip(self.duration_lim, self.t_release_lim, self.slope_lim, self.t_drop_lim, self.l_drop_lim)
