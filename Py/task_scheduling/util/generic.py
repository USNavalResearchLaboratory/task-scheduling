from copy import deepcopy
from math import factorial
from numbers import Integral

import numpy as np
from scipy.stats import rv_discrete, uniform


def check_rng(rng):
    """
    Return a random number generator.

    Parameters
    ----------
    rng : int or RandomState or Generator, optional
        Random number generator seed or object.

    Returns
    -------
    Generator

    """
    if rng is None:
        return np.random.default_rng()
    elif isinstance(rng, (Integral, np.integer)):
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
    params = deepcopy(alg.keywords)
    for key in keys_del:
        try:
            del params[key]
        except KeyError:
            pass
    if len(params) == 0:
        return alg.func.__name__
    else:
        p_str = ", ".join(f"{key}={str(val)}" for key, val in params.items())
        return f"{alg.func.__name__}({p_str})"


def seq2num(seq, check_input=True):     # TODO: relate to Lehmer code? https://en.wikipedia.org/wiki/Lehmer_code
    """
    Map an index sequence permutation to a non-negative integer.

    Parameters
    ----------
    seq : Sequence
        Elements are unique in range(len(seq)).
    check_input : bool
        Enables value checking of input sequence.

    Returns
    -------
    int
        Takes values in range(factorial(len(seq))).
    """

    length = len(seq)
    seq_rem = list(range(length))     # remaining elements
    if check_input and set(seq) != set(seq_rem):
        raise ValueError(f"Input must be a Sequence with unique elements in range({length}).")

    num = 0
    for i, n in enumerate(seq):
        k = seq_rem.index(n)    # position of index in remaining elements
        num += k * factorial(length - 1 - i)
        seq_rem.remove(n)

    return num


def num2seq(num, length, check_input=True):
    """
    Map a non-negative integer to an index sequence permutation.

    Parameters
    ----------
    num : int
        In range(factorial(length))
    length : int
        Length of the output sequence.
    check_input : bool
        Enables value checking of input number.

    Returns
    -------
    tuple
        Elements are unique in factorial(len(seq)).
    """

    if check_input and num not in range(factorial(length)):
        raise ValueError(f"Input 'num' must be in range(factorial({length})).")

    seq_rem = list(range(length))     # remaining elements
    seq = []
    while len(seq_rem) > 0:
        radix = factorial(len(seq_rem) - 1)
        i, num = num // radix, num % radix

        n = seq_rem.pop(i)
        seq.append(n)

    return tuple(seq)


def main():
    length = 5
    for _ in range(100):
        seq = tuple(np.random.permutation(length))
        assert seq == num2seq(seq2num(seq), length)

        num = np.random.default_rng().integers(factorial(length))
        assert num == seq2num(num2seq(num, length))


if __name__ == '__main__':
    main()




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








