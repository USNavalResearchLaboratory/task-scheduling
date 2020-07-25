import copy
import time
import dill
import numpy as np


def check_rng(rng):
    """
    Return a random number generator.

    Parameters
    ----------
    rng : None or int or RandomState or Generator
        Random number generator seed or object.

    Returns
    -------
    Generator

    """
    if rng is None:
        return np.random.default_rng()
    elif type(rng) == int:
        return np.random.default_rng(rng)
    elif isinstance(rng, np.random.Generator) or isinstance(rng, np.random.RandomState):
        return rng
    else:
        raise TypeError("Input must be None, int, or a valid NumPy random number generator.")


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
    keys_del = ['verbose', 'rng']
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


# def save_scheduler(scheduler, file_str=None):     # TODO: delete?
#     """Save scheduling function via data persistence."""
#     if file_str is None:
#         file_str = 'temp/{}.pkl'.format(time.strftime('%Y-%m-%d_%H-%M-%S'))
#     with open('./schedulers/' + file_str, 'wb') as file:
#         dill.dump(scheduler, file)
#
#
# def load_scheduler(file_str):
#     """Load scheduling function via data persistence."""
#     with open('./schedulers/' + file_str, 'rb') as file:
#         return dill.load(file)
