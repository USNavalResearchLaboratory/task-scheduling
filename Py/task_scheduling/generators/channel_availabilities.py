"""Generator objects for channel availabilities."""

from util.generic import check_rng


class BaseIID:
    def __init__(self, rng=None):
        self.rng = check_rng(rng)

    def __call__(self, n_ch):
        raise NotImplementedError


class UniformIID(BaseIID):
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

    def __call__(self, n_ch):
        """Randomly generate a list of channel availabilities."""
        for _ in range(n_ch):
            yield self.rng.uniform(*self.lim)

    def __eq__(self, other):
        if not isinstance(other, UniformIID):
            return False

        return True if self.lim == other.lim else False

