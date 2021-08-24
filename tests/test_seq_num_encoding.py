from math import factorial

import numpy as np

from task_scheduling.learning.environments import num_to_seq, seq_to_num


def test_encode_decode():
    length = 5
    for i in range(100):
        print(f"{i}", end='\n')

        seq = tuple(np.random.permutation(length))
        assert seq == num_to_seq(seq_to_num(seq), length)

        num = np.random.default_rng().integers(factorial(length))
        assert num == seq_to_num(num_to_seq(num, length))


if __name__ == '__main__':
    test_encode_decode()
