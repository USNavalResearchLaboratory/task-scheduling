"""Generator objects for channel availabilities."""

from util.generic import check_rng


class Base:
    def __init__(self, rng=None):
        self.rng = check_rng(rng)

    def __call__(self, n_tasks):
        raise NotImplementedError

    @property
    def param_repr_lim(self):
        raise NotImplementedError


class Uniform(Base):
    """
    Generator of uniformly random channel availabilities.

    Parameters
    ----------
    lim : tuple of float or list of float
        Lower and upper limits for uniform RNG

    """

    def __init__(self, lim, rng=None):
        super().__init__(rng)
        self.lim = lim

    def __call__(self, n_tasks):
        """Randomly generate a list of channel availabilities."""
        for _ in range(n_tasks):
            yield self.rng.uniform(*self.lim)

    def __eq__(self, other):
        if not isinstance(other, Uniform):
            return False

        return True if self.lim == other.lim else False

    @property
    def param_repr_lim(self):
        """Low and high tuples bounding parametric task representations."""
        return self.lim