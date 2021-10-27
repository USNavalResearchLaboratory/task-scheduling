from math import factorial

import numpy as np

from task_scheduling.learning.environments import int_to_seq, seq_to_int


def test_encode_decode():
    length = 5
    for i in range(10):
        print(f"{i}", end='\n')

        seq = tuple(np.random.permutation(length))
        assert seq == int_to_seq(seq_to_int(seq), length)

        num = np.random.default_rng().integers(factorial(length))
        assert num == seq_to_int(int_to_seq(num, length))


if __name__ == '__main__':
    test_encode_decode()
