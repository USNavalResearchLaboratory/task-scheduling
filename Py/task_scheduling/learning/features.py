import numpy as np
from gym.spaces import Discrete

from task_scheduling.learning.spaces import shift_space, DiscreteSet


def get_param(name):
    """Make a feature function to extract parameter attributes from tasks."""

    def func(tasks, ch_avail):
        return [getattr(task, name) for task in tasks]

    return func


def param_features(space_dict, shift_params=()):
    """Create array of parameter features from parameter spaces."""

    data = []
    for name, space in space_dict.items():
        if name in shift_params:
            space = shift_space(space)
        data.append((name, get_param(name), space))

    return np.array(data, dtype=[('name', '<U16'), ('func', object), ('space', object)])


def encode_param(name, space):
    """Make a feature function to encode a parameter value to the corresponding DiscreteSet index."""

    def func(tasks, ch_avail):
        return [np.flatnonzero(space.elements == getattr(task, name)).item() for task in tasks]

    return func


def encode_discrete_features(space_dict):
    """Create array of parameter features, encoding DiscreteSet-typed parameters to Discrete-type."""

    data = []
    for name, space in space_dict.items():
        if isinstance(space, DiscreteSet):  # use encoding feature func, change space to Discrete
            func = encode_param(name, space)
            space = Discrete(len(space))
        else:
            func = get_param(name)

        data.append((name, func, space))

    return np.array(data, dtype=[('name', '<U16'), ('func', object), ('space', object)])
