"""Generator objects for channel availabilities."""

from abc import ABC, abstractmethod

from task_scheduling.util.generic import RandomGeneratorMixin


class Base(RandomGeneratorMixin, ABC):
    """
    Generator of random channel availabilities.

    Parameters
    ----------
    lims : Iterable of float
        Lower and upper channel limits.

    """

    def __init__(self, lims=(0., 0.), rng=None):
        super().__init__(rng)
        self.lims = tuple(lims)

    @abstractmethod
    def __call__(self, n_ch, rng=None):
        raise NotImplementedError


class BaseIID(Base, ABC):
    def __call__(self, n_ch, rng=None):
        """Randomly generate tasks."""
        rng = self._get_rng(rng)
        for _ in range(n_ch):
            yield self._gen_single(rng)

    @abstractmethod
    def _gen_single(self, rng):
        """Randomly generate task parameters."""
        raise NotImplementedError


class UniformIID(BaseIID):
    def _gen_single(self, rng):
        return rng.uniform(*self.lims)

    def __eq__(self, other):
        if isinstance(other, UniformIID):
            return self.lims == other.lims
        else:
            return NotImplemented

    def summary(self, file=None):
        print(f"Channel: UniformIID{self.lims}", file=file)


class Deterministic(Base):
    def __init__(self, ch_avail):
        self.ch_avail = tuple(ch_avail)
        super().__init__(lims=(min(ch_avail), max(ch_avail)))

    def __call__(self, n_ch, rng=None):
        if n_ch != len(self.ch_avail):
            raise ValueError(f"Number of channels must be {len(self.ch_avail)}.")

        for ch_avail_ in self.ch_avail:
            yield ch_avail_

    @classmethod
    def from_uniform(cls, n_ch, lims=(0., 0.), rng=None):
        ch_avail_gen = UniformIID(lims, rng=rng)
        return cls(ch_avail=list(ch_avail_gen(n_ch)))
