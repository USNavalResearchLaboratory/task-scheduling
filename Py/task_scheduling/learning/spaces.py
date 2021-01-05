from math import factorial

import numpy as np
from gym.spaces import Space, Discrete, MultiDiscrete, Box


# Utilities
def get_space_lims(space):
    """Get minimum and maximum values for a scalar-valued space."""

    if isinstance(space, Box):
        if space.shape == ():
            return space.low, space.high
        else:
            raise ValueError("Only supported for scalar-value spaces.")
    elif isinstance(space, Discrete):
        return 0, space.n - 1
    elif isinstance(space, DiscreteSet):
        return tuple(space.elements[[0, -1]])
    else:
        raise NotImplementedError('Only supported for Box, Discrete, or DiscreteSet spaces.')


def as_box(space):
    """Upcast space to a Box."""

    if isinstance(space, Box):
        return space
    elif isinstance(space, MultiDiscrete):
        return Box(np.zeros(space.shape), space.nvec - 1)
    elif isinstance(space, Discrete):
        return Box(0, space.n - 1, shape=())
    elif isinstance(space, DiscreteSet):
        return Box(*space.elements[[0, -1]], shape=())
    else:
        raise TypeError('Only supported for Box, Discrete, or DiscreteSet type inputs.')


def as_multidiscrete(space):
    if isinstance(space, MultiDiscrete):
        return space
    elif isinstance(space, Discrete):
        return MultiDiscrete([space.n])


def upcast_space(space, stype):
    pass


def broadcast_to(space, shape):
    """Broadcast space to new shape."""

    if isinstance(space, Box):
        low, high = np.broadcast_to(space.low, shape), np.broadcast_to(space.high, shape)
        return Box(low, high, dtype=space.dtype)
    elif isinstance(space, MultiDiscrete):
        return MultiDiscrete(np.broadcast_to(space.nvec, shape))
    # elif isinstance(space, Discrete):
    #     if shape == ():
    #         return space
    #     else:
    #         return MultiDiscrete(np.broadcast_to(space.n, shape))
    else:
        raise NotImplementedError("Only supported for Box and MultiDiscrete spaces.")


def concatenate(spaces, axis=0):
    shapes = [space.shape for space in spaces]


    if all([isinstance(space, Discrete) for space in spaces]) and axis == 0:
        # Combine into MultiDiscrete space
        return MultiDiscrete([space.n for space in spaces])
    elif all([isinstance(space, MultiDiscrete) for space in spaces]):
        # Combine into MultiDiscrete space
        return MultiDiscrete(np.concatenate([space.nvec for space in spaces], axis=axis))
    else:
        # Upcast each space and combine into multi-dimensional Box

        # boxes = [tasking_spaces.as_box(space) for space in self.features['space']]
        # low, high = zip(*[(box.low, box.high) for box in boxes])
        low, high = zip(*[get_space_lims(space) for space in spaces])
        return Box(low, high)


# Space classes
class Permutation(Space):
    """Gym Space for index sequences."""

    def __init__(self, n):
        self.n = n      # sequence length
        super().__init__(shape=(self.n,), dtype=np.int)

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
    """Gym Space for discrete, non-integral elements."""

    def __init__(self, elements):
        self.elements = np.unique(np.array(list(elements)).flatten())   # sorted, flattened
        super().__init__(shape=(), dtype=self.elements.dtype)

    def sample(self):
        return self.np_random.choice(self.elements)

    def contains(self, x):
        return True if x in self.elements else False

    def __repr__(self):
        return f"DiscreteSet({self.elements})"

    def __eq__(self, other):
        if isinstance(other, DiscreteSet):
            return self.elements == other.elements
        else:
            return NotImplemented

    def __len__(self):
        return self.elements.size
