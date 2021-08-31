import numpy as np
from gym.spaces import Discrete, MultiDiscrete, Box

from task_scheduling.tasks import Shift
from task_scheduling.spaces import DiscreteSet


def _get_param(name):
    """Make a feature function to extract parameter attributes from tasks."""

    def func(tasks, _ch_avail):
        return [getattr(task, name) for task in tasks]

    return func


def _add_zero(space):
    """Modify space to include zero as a possible value."""

    if isinstance(space, Box):
        space.low = 0
        return space
    elif isinstance(space, Discrete):
        return space
    elif isinstance(space, MultiDiscrete):
        return space
    elif isinstance(space, DiscreteSet):
        space.add_elements(0)
        return space
    else:
        raise NotImplementedError


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


def param_features(task_gen, time_shift=False, masking=False):
    """
    Create array of parameter features from task parameter spaces.

    Parameters
    ----------
    task_gen : generators.tasks.Base
        Scheduling problem generation object.
    time_shift : bool, optional
        Enables modification of feature `space` to reflect shifted parameters.
    masking : bool, optional
        Enables modification of feature `space` to account for masking.

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
        if masking:
            space = _add_zero(space)
        if name in shift_params:
            space = _as_box(space)
        data.append((name, _get_param(name), space))

    return np.array(data, dtype=[('name', '<U32'), ('func', object), ('space', object)])


def _encode_param(name, space):
    """Make a feature function to encode a parameter value to the corresponding DiscreteSet index."""

    def func(tasks, _ch_avail):
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
