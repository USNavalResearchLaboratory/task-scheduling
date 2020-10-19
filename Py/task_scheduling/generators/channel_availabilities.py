"""Generator objects for channel availabilities."""

from abc import ABC, abstractmethod

from util.generic import RandomGeneratorMixin


class BaseIID(RandomGeneratorMixin, ABC):
    @abstractmethod
    def __call__(self, n_ch, rng=None):
        raise NotImplementedError


class UniformIID(BaseIID):
    """
    Generator of uniformly random channel availabilities.

    Parameters
    ----------
    lim : tuple of float or list of float
        Lower and upper limits for uniform RNG

    """

    def __init__(self, lim=(0, 0), rng=None):
        super().__init__(rng)
        self.lim = lim

    def __call__(self, n_ch, rng=None):
        """Randomly generate a list of channel availabilities."""
        rng = self._get_rng(rng)
        for _ in range(n_ch):
            yield rng.uniform(*self.lim)

    def __eq__(self, other):
        if isinstance(other, UniformIID):
            return self.lim == other.lim
        else:
            return NotImplemented
