from collections import namedtuple
from functools import wraps
from time import perf_counter

import numpy as np


SchedulingProblem = namedtuple('SchedulingProblem', ['tasks', 'ch_avail'])
SchedulingSolution = namedtuple('SchedulingSolution', ['t_ex', 'ch_ex', 'l_ex', 't_run'], defaults=(None, None))


class RandomGeneratorMixin:
    def __init__(self, rng=None):
        self.rng = rng

    @property
    def rng(self):
        return self._rng

    @rng.setter
    def rng(self, val):
        self._rng = self.make_rng(val)

    def _get_rng(self, rng=None):
        if rng is None:
            return self.rng
        else:
            return self.make_rng(rng)

    @staticmethod
    def make_rng(rng):
        """
        Return a random number generator.

        Parameters
        ----------
        rng : int or RandomState or Generator, optional
            Random number generator seed or object.

        Returns
        -------
        Generator

        """
        if rng is None:
            return np.random.default_rng()
        elif isinstance(rng, int):
            return np.random.default_rng(rng)
        elif isinstance(rng, np.random.Generator) or isinstance(rng, np.random.RandomState):
            return rng
        else:
            raise TypeError("Input must be None, int, or a valid NumPy random number generator.")


def check_schedule(tasks, t_ex, ch_ex, tol=1e-12):
    """
    Check schedule validity.

    Parameters
    ----------
    tasks : list of task_scheduling.tasks.Base
    t_ex : numpy.ndarray
        Task execution times.
    ch_ex : numpy.ndarray
        Task execution channels.
    tol : float, optional
        Time tolerance for validity conditions.

    Raises
    -------
    ValueError
        If tasks overlap in time.

    """

    # if np.isnan(t_ex).any():
    #     raise ValueError("All tasks must be scheduled.")

    for ch in np.unique(ch_ex):
        tasks_ch = np.array(tasks)[ch_ex == ch].tolist()
        t_ex_ch = t_ex[ch_ex == ch]
        for n_1 in range(len(tasks_ch)):
            if t_ex_ch[n_1] + tol < tasks_ch[n_1].t_release:
                raise ValueError("Tasks cannot be executed before their release time.")

            for n_2 in range(n_1 + 1, len(tasks_ch)):
                conditions = [t_ex_ch[n_1] + tol < t_ex_ch[n_2] + tasks_ch[n_2].duration,
                              t_ex_ch[n_2] + tol < t_ex_ch[n_1] + tasks_ch[n_1].duration]
                if all(conditions):
                    raise ValueError('Invalid Solution: Scheduling Conflict')


def evaluate_schedule(tasks, t_ex):
    """
    Evaluate scheduling loss.

    Parameters
    ----------
    tasks : Sequence of task_scheduling.tasks.Base
        Tasks
    t_ex : Sequence of float
        Task execution times.

    Returns
    -------
    float
        Total loss of scheduled tasks.

    """

    l_ex = 0.
    for task, t_ex in zip(tasks, t_ex):
        l_ex += task(t_ex)

    return l_ex


def eval_wrapper(scheduler):
    """Wraps a scheduler, creates a function that outputs runtime in addition to schedule."""

    @wraps(scheduler)
    def timed_scheduler(tasks, ch_avail):
        t_start = perf_counter()
        t_ex, ch_ex, *__ = scheduler(tasks, ch_avail)
        t_run = perf_counter() - t_start

        check_schedule(tasks, t_ex, ch_ex)
        l_ex = evaluate_schedule(tasks, t_ex)

        return SchedulingSolution(t_ex, ch_ex, l_ex, t_run)

    return timed_scheduler
