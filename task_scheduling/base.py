"""Core package objects."""

from collections import namedtuple
from datetime import datetime

import numpy as np

SchedulingProblem = namedtuple("SchedulingProblem", ["tasks", "ch_avail"])
SchedulingSolution = namedtuple(
    "SchedulingSolution", ["sch", "loss", "t_run"], defaults=(None, None)
)


def get_now():
    return datetime.now().replace(microsecond=0).isoformat().replace(":", "_")


class RandomGeneratorMixin:
    """
    Mixin class providing a random number generating attribute and methods.

    Parameters
    ----------
    rng : int or RandomState or Generator, optional
        Random number generator seed or object.

    """

    def __init__(self, rng=None):
        self.rng = rng

    @property
    def rng(self):
        r"""The NumPy random number generator."""
        return self._rng

    @rng.setter
    def rng(self, value):
        self._rng = self.make_rng(value)

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
