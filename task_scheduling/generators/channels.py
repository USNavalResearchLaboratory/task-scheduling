"""Generator objects for channel availabilities."""

from abc import ABC, abstractmethod

from gym.spaces import Box

from task_scheduling.base import RandomGeneratorMixin


class Base(RandomGeneratorMixin, ABC):
    """
    Base class for channel availability generators.

    Parameters
    ----------
    rng : int or RandomState or Generator, optional
        Random number generator seed or object.

    """
    def __init__(self, rng=None):
        super().__init__(rng)
        self.space = None

    @abstractmethod
    def __call__(self, n_ch, rng=None):
        """
        Generate channel availabilities.

        Parameters
        ----------
        n_ch : int
            Number of channels to generate.
        rng : int or RandomState or Generator, optional
            Random number generator seed or object.

        Returns
        -------
        Generator

        """
        raise NotImplementedError

    def summary(self):
        return f"Channel: {str(self)}"


class BaseIID(Base, ABC):
    def __call__(self, n_ch, rng=None):
        """
        Randomly generate channel availabilities.

        Parameters
        ----------
        n_ch : int
            Number of channels to generate.
        rng : int or RandomState or Generator, optional
            Random number generator seed or object.

        Returns
        -------
        Generator

        """
        rng = self._get_rng(rng)
        for _ in range(n_ch):
            yield self._gen_single(rng)

    @abstractmethod
    def _gen_single(self, rng):
        """Randomly generate task parameters."""
        raise NotImplementedError


class UniformIID(BaseIID):
    """
    Generator of random channel availabilities.

    Parameters
    ----------
    lims : Collection of float
        Lower and upper channel limits.
    rng : int or RandomState or Generator, optional
        NumPy random number generator or seed. Instance RNG if None.

    """
    def __init__(self, lims=(0., 0.), rng=None):
        super().__init__(rng)
        self.lims = tuple(lims)
        self.space = Box(*lims, shape=(), dtype=float)

    def _gen_single(self, rng):
        return rng.uniform(*self.lims)

    def __eq__(self, other):
        if isinstance(other, UniformIID):
            return self.lims == other.lims
        else:
            return NotImplemented

    def summary(self):
        return f"Channel: UniformIID{self.lims}"


class Deterministic(Base):
    def __init__(self, ch_avail):
        super().__init__()
        self.ch_avail = tuple(ch_avail)

    def __call__(self, n_ch, rng=None):
        """
        Generate deterministic channel availabilities.

        Parameters
        ----------
        n_ch : int
            Number of channels to generate.
        rng : int or RandomState or Generator, optional
            Random number generator seed or object.

        Returns
        -------
        Generator

        """
        if n_ch != len(self.ch_avail):
            raise ValueError(f"Number of channels must be {len(self.ch_avail)}.")

        for ch_avail_ in self.ch_avail:
            yield ch_avail_

    @classmethod
    def from_uniform(cls, n_ch, lims=(0., 0.), rng=None):
        ch_avail_gen = UniformIID(lims, rng=rng)
        return cls(tuple(ch_avail_gen(n_ch)))

    def summary(self):
        return f"Channel: Deterministic{self.ch_avail}"
