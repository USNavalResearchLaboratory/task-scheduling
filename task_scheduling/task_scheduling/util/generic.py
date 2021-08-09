from collections import namedtuple
from functools import wraps
from math import factorial
from time import perf_counter
from operator import attrgetter
from datetime import datetime

import numpy as np

NOW_STR = datetime.now().replace(microsecond=0).isoformat().replace(':', '_')

TIME_STR = datetime.now().isoformat(timespec='seconds').replace(':', '_')


SchedulingProblem = namedtuple('SchedulingProblem', ['tasks', 'ch_avail'])
SchedulingSolution = namedtuple('SchedulingSolution', ['t_ex', 'ch_ex', 't_run'], defaults=(None,))


# TODO: use for algorithms and wraps?


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


def timing_wrapper(scheduler):
    """Wraps a scheduler, creates a function that outputs runtime in addition to schedule."""

    @wraps(scheduler)
    def timed_scheduler(tasks, ch_avail):
        t_start = perf_counter()
        t_ex, ch_ex = scheduler(tasks, ch_avail)
        t_run = perf_counter() - t_start
        return t_ex, ch_ex, t_run

    return timed_scheduler


def runtime_wrapper(scheduler):
    @wraps(scheduler)
    def new_scheduler(tasks, ch_avail, runtimes):
        t_ex, ch_ex, t_run = timing_wrapper(scheduler)(tasks, ch_avail)
        for runtime in runtimes:
            if t_run < runtime:
                yield t_ex, ch_ex
            else:
                yield None
                # raise RuntimeError(f"Algorithm timeout: {t_run} > {runtime}.")

    return new_scheduler


def sort_wrapper(scheduler, sort_func):  # TODO: use for basic algorithms?
    if isinstance(sort_func, str):
        sort_func = attrgetter(sort_func)

    @wraps(scheduler)
    def sorted_scheduler(tasks, ch_avail):
        idx = list(np.argsort([sort_func(task) for task in tasks]))
        idx_inv = [idx.index(n) for n in range(len(tasks))]

        t_ex, ch_ex = scheduler([tasks[i] for i in idx], ch_avail)
        return t_ex[idx_inv], ch_ex[idx_inv]

    return sorted_scheduler


def seq2num(seq, check_input=True):
    """
    Map an index sequence permutation to a non-negative integer.

    Parameters
    ----------
    seq : Sequence of int
        Elements are unique in range(len(seq)).
    check_input : bool
        Enables value checking of input sequence.

    Returns
    -------
    int
        Takes values in range(factorial(len(seq))).
    """

    length = len(seq)
    seq_rem = list(range(length))  # remaining elements
    if check_input and set(seq) != set(seq_rem):
        raise ValueError(f"Input must have unique elements in range({length}).")

    num = 0
    for i, n in enumerate(seq):
        k = seq_rem.index(n)  # position of index in remaining elements
        num += k * factorial(length - 1 - i)
        seq_rem.remove(n)

    return num


def num2seq(num, length, check_input=True):
    """
    Map a non-negative integer to an index sequence permutation.

    Parameters
    ----------
    num : int
        In range(factorial(length))
    length : int
        Length of the output sequence.
    check_input : bool
        Enables value checking of input number.

    Returns
    -------
    tuple
        Elements are unique in factorial(len(seq)).
    """

    if check_input and num not in range(factorial(length)):
        raise ValueError(f"Input 'num' must be in range(factorial({length})).")

    seq_rem = list(range(length))  # remaining elements
    seq = []
    while len(seq_rem) > 0:
        radix = factorial(len(seq_rem) - 1)
        i, num = num // radix, num % radix

        n = seq_rem.pop(i)
        seq.append(n)

    return tuple(seq)

