from math import factorial

import numpy as np

from task_scheduling.util.generic import num2seq, seq2num


def test_encode_decode():
    length = 5
    for i in range(100):
        print(f"{i}", end='\n')

        seq = tuple(np.random.permutation(length))
        assert seq == num2seq(seq2num(seq), length)

        num = np.random.default_rng().integers(factorial(length))
        assert num == seq2num(num2seq(num, length))


if __name__ == '__main__':
    test_encode_decode()
