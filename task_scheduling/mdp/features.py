"""Observation feature extractors and utilities."""

from operator import attrgetter
from warnings import warn

import numpy as np
from gym.spaces import Box, Discrete, MultiDiscrete

from task_scheduling.spaces import DiscreteSet, get_space_lims
from task_scheduling.tasks import Shift

feature_dtype = [("name", "<U32"), ("func", object), ("space", object)]


# def _add_zero(space):
#     """Modify space to include zero as a possible value."""
#     if isinstance(space, Box):
#         space.low = np.array(0.0)
#         return space
#     elif isinstance(space, Discrete):
#         return space
#     elif isinstance(space, MultiDiscrete):
#         return space
#     elif isinstance(space, DiscreteSet):
#         space.add_elements(0)
#         return space
#     else:
#         raise NotImplementedError


def _as_box(space):
    """Convert scalar space to Box with zero lower bound."""
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

    low = np.zeros(space.shape)
    return Box(low, high, shape=space.shape, dtype=float)


def param_features(task_gen, time_shift=False):
    """
    Create array of parameter features from task parameter spaces.

    Parameters
    ----------
    task_gen : generators.tasks.Base
        Scheduling problem generation object.
    time_shift : bool, optional
        Enables modification of feature `space` to reflect shifted parameters.

    Returns
    -------
    ndarray
        Feature array with fields 'name', 'func', and 'space'.

    """
    if time_shift and issubclass(task_gen.cls_task, Shift):
        shift_params = task_gen.cls_task.shift_params
    else:
        shift_params = ()

    data = []
    for name, space in task_gen.param_spaces.items():
        if name in shift_params:
            space = _as_box(space)
        data.append((name, attrgetter(name), space))

    return np.array(data, dtype=feature_dtype)


def encode_discrete_features(problem_gen):
    """Create parameter features, encoding DiscreteSet-typed parameters to Discrete-type."""
    data = []
    for name, space in problem_gen.task_gen.param_spaces.items():
        if isinstance(space, DiscreteSet):  # use encoding feature func, change space to Discrete

            def func(task):
                return np.flatnonzero(space.elements == getattr(task, name)).item()

            space = Discrete(len(space))
        else:
            func = attrgetter(name)

        data.append((name, func, space))

    return np.array(data, dtype=feature_dtype)


def _make_norm_func(func, space):
    low, high = get_space_lims(space)
    if np.isinf([low, high]).any():
        warn("Cannot make a normalizing `func` due to unbounded `space`.")
        return func

    def norm_func(task):
        return (func(task) - low) / (high - low)

    return norm_func


def normalize(features):
    """Make normalized features."""
    data = []
    for name, func, space in features:
        func = _make_norm_func(func, space)
        space = Box(0, 1, shape=space.shape, dtype=float)
        data.append((name, func, space))
    return np.array(data, dtype=feature_dtype)
