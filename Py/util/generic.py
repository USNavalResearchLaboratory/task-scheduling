import copy
import time
import dill
import numpy as np
import scipy.stats as stats
from scipy.stats import rv_discrete, uniform

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


class Distribution:
    """
    Random Number Generator Object

    Parameters
    ----------
    feature_name: str, ex "duration"
    type: str - representing distribution type  ex: 'uniform', 'discrete'
    values: np-array of discrete values taken by distribution
    probs: tuple of probabilites (same length as values)
    lims: used to set (lower,upper) limits for uniform distribution

    output
    --------
    distro: random variable object.   Usage distro.rvs(size=10) --> produces length 10 vector of rvs
    distro.mean(), distro.var()  returns mean/vars of distribution
    """

    def __init__(self, feature_name: str, type: str, values=False, probs=False, lims=(0,1), distro = None, rng=None):
        self.feature_name = feature_name  # Feature Name
        self.type = type  # Distribution type
        self.values = values
        self.probs = probs
        self.lims = lims

        if type.lower() == 'uniform':
            lower_lim = lims(0)
            upper_lim = lims(1)
            loc = lower_lim
            scale = upper_lim - lower_lim
            distro = uniform(name=feature_name, loc=loc, scale=scale)
        elif type.lower() == 'discrete':
            xk = np.arange(len(values))
            distro = rv_discrete(name=feature_name, values=(xk, probs))

        self.distro = distro

    # def __call__(self):
    #
    #     if type.lower() == 'uniform':
    #         lower_lim = lims(0)
    #         upper_lim = lims(1)
    #         loc = lower_lim
    #         scale = upper_lim - lower_lim
    #         distro = uniform(name=feature_name, loc=loc, scale=scale)
    #     elif type.lower() == 'discrete':
    #         # values =
    #         distro = rv_discrete(name=feature_name, values=(values, probs))
    #
    #     return distro

    def gen_samps(self, size=None):

        if self.type.lower() == 'uniform':
            samps = self.distro.rvs(size=size)
        elif self.type.lower() == 'discrete':
            samps_idx = self.distro.rvs(size=size)
            samps = self.values[samps_idx]

        return samps








