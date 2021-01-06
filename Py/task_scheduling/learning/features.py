import numpy as np

from gym.spaces import Discrete

from .spaces import shift_space


def get_param(name):
    """Make a feature function to extract parameter attributes from tasks."""

    def func(tasks, ch_avail):
        return [getattr(task, name) for task in tasks]

    return func


def param_features(space_dict, shift_keys=()):
    """Create array of parameter features from dictionary of parameter spaces."""

    data = []
    for name, space in space_dict.items():
        if name in shift_keys:
            space = shift_space(space)
        data.append((name, get_param(name), space))

    return np.array(data, dtype=[('name', '<U16'), ('func', object), ('space', object)])


# def encode_discrete(name, space):
#     if isinstance(space, Discrete):
#         pass
