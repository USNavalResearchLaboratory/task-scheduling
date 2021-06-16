import numpy as np
from gym.spaces import Discrete, MultiDiscrete, Box

from task_scheduling.learning.spaces import DiscreteSet
from task_scheduling.tasks import ReluDrop


# TODO: built feature library, create registry for easy access?


def _get_param(name):
    """Make a feature function to extract parameter attributes from tasks."""

    def func(tasks, ch_avail):
        return [getattr(task, name) for task in tasks]

    return func


def _shift_space(space):
    """Convert scalar space to Box with zero lower bound."""

    if space.shape != ():
        raise NotImplementedError("Only supported for scalar spaces.")

    if isinstance(space, Box):
        high = space.high
    elif isinstance(space, Discrete):
        high = space.n - 1
    elif isinstance(space, MultiDiscrete):
        high = space.nvec - 1
    elif isinstance(space, DiscreteSet):
        high = space.elements[-1]
    else:
        raise NotImplementedError

    return Box(0., high, shape=(), dtype=float)


def param_features(problem_gen, time_shift=False):
    """Create array of parameter features from parameter spaces."""

    if not time_shift:
        shift_params = ()
    elif problem_gen.task_gen.cls_task == ReluDrop:
        shift_params = ('t_release', 't_drop', 'l_drop')
    else:
        raise TypeError

    data = []
    for name, space in problem_gen.task_gen.param_spaces.items():
        if name in shift_params:
            space = _shift_space(space)
        data.append((name, _get_param(name), space))

    return np.array(data, dtype=[('name', '<U32'), ('func', object), ('space', object)])


def _encode_param(name, space):
    """Make a feature function to encode a parameter value to the corresponding DiscreteSet index."""

    def func(tasks, ch_avail):
        return [np.flatnonzero(space.elements == getattr(task, name)).item() for task in tasks]

    return func


def encode_discrete_features(problem_gen):
    """Create array of parameter features, encoding DiscreteSet-typed parameters to Discrete-type."""

    data = []
    for name, space in problem_gen.task_gen.param_spaces.items():
        if isinstance(space, DiscreteSet):  # use encoding feature func, change space to Discrete
            func = _encode_param(name, space)
            space = Discrete(len(space))
        else:
            func = _get_param(name)

        data.append((name, func, space))

    return np.array(data, dtype=[('name', '<U32'), ('func', object), ('space', object)])
