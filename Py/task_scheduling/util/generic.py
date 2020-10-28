from copy import deepcopy
from math import factorial
from numbers import Integral

import numpy as np


class RandomGeneratorMixin:
    def __init__(self, rng=None):
        self.rng = rng

    @property
    def rng(self):
        return self._rng

    @rng.setter
    def rng(self, val):
        self._rng = self.check_rng(val)

    def _get_rng(self, rng=None):
        if rng is None:
            return self._rng
        else:
            return self.check_rng(rng)

    @staticmethod
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


def seq2num(seq, check_input=True):
    """
    Map an index sequence permutation to a non-negative integer.

    Parameters
    ----------
    seq : Iterable
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
        raise ValueError(f"Input must have unique elements in range({length}).")

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


# def algorithm_repr(alg):
#     """
#     Create algorithm string representations.
#
#     Parameters
#     ----------
#     alg : functools.partial
#         Algorithm as a partial function with keyword arguments.
#
#     Returns
#     -------
#     str
#         Compact string representation of the algorithm.
#
#     """
#     keys_del = ['verbose', 'rng']
#     params = deepcopy(alg.keywords)
#     for key in keys_del:
#         try:
#             del params[key]
#         except KeyError:
#             pass
#     if len(params) == 0:
#         return alg.func.__name__
#     else:
#         p_str = ", ".join(f"{key}={str(val)}" for key, val in params.items())
#         return f"{alg.func.__name__}({p_str})"

if __name__ == '__main__':
    main()
