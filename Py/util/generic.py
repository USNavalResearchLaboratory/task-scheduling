import copy
import numpy as np


def check_rng(rng):
    """
    Return a random number generator.

    Parameters
    ----------
    rng : None or int or Generator
        Random number generator seed or object.

    Returns
    -------
    Generator

    """
    if rng is None:
        return np.random.default_rng()
    elif type(rng) == int:
        return np.random.default_rng(rng)
    else:
        return rng      # TODO: type check? assumes valid rng


def algorithm_repr(alg):
    """
    Create algorithm string representations.

    Parameters
    ----------
    alg : functools.partial
        Algorithm as a partial function with keyword arguments.

    Returns
    -------
    str
        Compact string representation of the algorithm.

    """
    keys_del = ['ch_avail', 'verbose', 'rng']
    params = copy.deepcopy(alg.keywords)
    for key in keys_del:
        try:
            del params[key]
        except KeyError:
            pass
    if len(params) == 0:
        return alg.func.__name__
    else:
        p_str = ", ".join([f"{key}={str(val)}" for key, val in params.items()])
        return f"{alg.func.__name__}({p_str})"
