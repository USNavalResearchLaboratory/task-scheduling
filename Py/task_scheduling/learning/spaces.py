from math import factorial
from typing import Sequence

import numpy as np
from gym.spaces import Space, Discrete, MultiDiscrete, Box


# Utilities
def broadcast_to(space, shape):
    """Broadcast space to new shape."""

    if isinstance(space, Box):
        low, high = np.broadcast_to(space.low, shape), np.broadcast_to(space.high, shape)
        return Box(low, high, dtype=space.dtype)
    elif isinstance(space, MultiDiscrete):
        return MultiDiscrete(np.broadcast_to(space.nvec, shape))
    else:
        raise NotImplementedError("Only supported for Box and MultiDiscrete spaces.")


def get_space_lims(space):
    """Get minimum and maximum values of a space."""

    if isinstance(space, Box):
        return np.stack((space.low, space.high), axis=-1)
    elif isinstance(space, Discrete):
        return np.array([0, space.n - 1])
    elif isinstance(space, MultiDiscrete):
        return np.stack((np.zeros(space.shape), space.nvec - 1), axis=-1)
    elif isinstance(space, DiscreteSet):
        return space.elements[[0, -1]]
    else:
        raise NotImplementedError('Only supported for Box, Discrete, or DiscreteSet spaces.')


def stack(spaces, axis=0):
    """Join a sequence of spaces along a new axis, 'upcasting' to superset spaces when required."""

    if len(spaces) == 1:
        return spaces[0]

    if all(isinstance(space, (Discrete, MultiDiscrete)) for space in spaces):
        nvecs = [space.n if isinstance(space, Discrete) else space.nvec for space in spaces]
        return MultiDiscrete(np.stack(nvecs, axis=axis))
    else:
        if axis == -1:
            axis = -2
        lims = np.stack([get_space_lims(space) for space in spaces], axis=axis)
        return Box(lims[..., 0], lims[..., 1], dtype=float)


def concatenate(spaces, axis=0):
    """Join a sequence of spaces along an existing axis, 'upcasting' to superset spaces when required."""

    if len(spaces) == 1:
        return spaces[0]

    if all(isinstance(space, MultiDiscrete) for space in spaces):
        nvecs = [space.nvec for space in spaces]
        return MultiDiscrete(np.concatenate(nvecs, axis=axis))
    else:
        if axis == -1:
            axis = -2
        lims = np.concatenate([get_space_lims(space) for space in spaces], axis=axis)
        return Box(lims[..., 0], lims[..., 1], dtype=float)


# def as_box(space):
#     """Upcast space to a Box."""
#
#     if isinstance(space, Box):
#         return space
#     elif isinstance(space, MultiDiscrete):
#         return Box(np.zeros(space.shape), space.nvec - 1)
#     elif isinstance(space, Discrete):
#         return Box(0, space.n - 1, shape=())
#     elif isinstance(space, DiscreteSet):
#         return Box(*space.elements[[0, -1]], shape=())
#     else:
#         raise TypeError('Only supported for Box, Discrete, or DiscreteSet type inputs.')
#
#
# def as_multidiscrete(space):
#     if isinstance(space, MultiDiscrete):
#         return space
#     elif isinstance(space, Discrete):
#         return MultiDiscrete([space.n])
#     else:
#         raise TypeError


# Space classes
class Permutation(Space):
    """Gym Space for index sequences."""

    def __init__(self, n):
        self.n = n  # sequence length
        super().__init__(shape=(self.n,), dtype=int)

    def sample(self):
        return self.np_random.permutation(self.n)

    def contains(self, x):
        return True if (np.sort(np.asarray(x, dtype=int)) == np.arange(self.n)).all() else False

    def __repr__(self):
        return f"Permutation({self.n})"

    def __eq__(self, other):
        if isinstance(other, Permutation):
            return self.n == other.n
        else:
            return NotImplemented

    def __len__(self):
        return factorial(self.n)


class DiscreteSet(Space):
    def __init__(self, elements):
        """
        Gym Space for discrete, non-integral elements.

        Parameters
        ----------
        elements : Sequence

        """
        self.elements = np.unique(np.array(elements).flatten())  # sorted, flattened
        super().__init__(shape=(), dtype=self.elements.dtype)

    def sample(self):
        return self.np_random.choice(self.elements)

    def contains(self, x):
        return True if x in self.elements else False

    def __repr__(self):
        return f"DiscreteSet({self.elements})"

    def __eq__(self, other):
        if isinstance(other, DiscreteSet):
            return np.all(self.elements == other.elements)
        else:
            return NotImplemented

    def __len__(self):
        return self.elements.size


class DiscreteMasked(Discrete):
    def __init__(self, n, mask=np.ma.nomask):
        """A Discrete space with masked elements for sampling and membership testing."""

        super().__init__(n)
        self.mask = mask

        self._rng = np.random.default_rng()

    @property
    def n(self):
        return self._ma.size

    @n.setter
    def n(self, val):
        self._ma = np.ma.masked_array(range(int(val)))

    @property
    def mask(self):
        return self._ma.mask

    @mask.setter
    def mask(self, val):
        # self._ma.mask = np.ma.make_mask(val)  # TODO: use `ma.masked` instead?
        self._ma.mask = np.ma.nomask
        self._ma[np.array(val, dtype=bool)] = np.ma.masked

    @property
    def valid_entries(self):
        return self._ma.compressed()

    def sample(self):
        return self._rng.choice(self.valid_entries)

    def contains(self, x):
        return x in self.valid_entries

    def __repr__(self):
        return f"DiscreteMasked({self.n}, mask={self.mask})"

    def __eq__(self, other):
        return isinstance(other, DiscreteMasked) and self._ma == other._ma
